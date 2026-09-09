(ns raster.compiler.passes.parallel.typed-effect-carry-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.walk :as walk]
            [raster.compiler.backend.jvm.segop-simd :as jvm]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.carried-effect-loop-fixture :as fixture]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.compiler.passes.parallel.typed-soac-fusion :as fusion]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(defn- rewrite-loop [program f]
  (walk/postwalk #(if (dialect/effect-loop-form? %) (f %) %) program))

(defn- reason [program]
  (try (dialect/validate! program) :accepted
       (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest canonical-carried-effects-project-without-source-reconstruction
  (doseq [trips [0 1 8]]
    (let [program (fixture/typed-program trips)
          equation (first (dialect/equations program))
          loop (-> equation dialect/operation-parts :lambda dialect/lambda-parts
                   :body-results first dialect/effect-parts)
          operation (first (lower/lower-typed-effect-map program :ze:0))
          carry (get-in operation [:scalar-region :effects 0 :loop :carry])
          execute (eval (list 'fn '[x words totals rows seed] (jvm/compile-effect-segmap operation)))
          totals (float-array 2) words (float-array (repeat 16 -77))]
      (is (= :accepted (reason program)))
      (is (= 'initial (get-in loop [:carry :init])))
      (is (= 'seed (:init carry)) "capture appears only in initializer but is still substituted")
      (is (= :float (:dtype carry)))
      (is (= 'double (:raster.type/tag (meta (:update carry)))))
      (is (nil? (:lambda operation)))
      (execute (float-array (range 16)) words totals 2 (float 0.25))
      (is (= (mapv (fn [row] (+ 0.25 (reduce + (take trips (drop (* row 8) (range 16)))))) [0 1])
             (vec totals)))
      (doseq [target [:opencl-portable :cuda :hip]]
        (is (= :kernel-body (get-in (fixture/artifact operation target) [:attributes :emission-route])))))))

(deftest carried-regions-survive-remapping-and-remain-an-effect-fusion-boundary
  (let [program (fixture/typed-program 8)
        remapped (dialect/remap-values program {'seed 'starting-value})
        operation (first (lower/lower-typed-effect-map remapped :ze:0))
        [fused stats] (fusion/fusion-fixpoint program)]
    (is (= 'starting-value (get-in operation [:scalar-region :effects 0 :loop :carry :init])))
    (is (= (dialect/equations program) (dialect/equations fused)))
    (is (zero? (:vertical stats)))
    (is (zero? (:horizontal stats)))
    (is (= :explicit-typed-algorithm (get-in (route/program-envelope program) [:attributes :host-control])))
    (is (= :carried-effect-realization
           (try ((ns-resolve 'raster.compiler.passes.parallel.typed-soac-route 'scalar-region)
                 (first (dialect/equations program)))
                :accepted
                (catch clojure.lang.ExceptionInfo e (:missing-rule (ex-data e))))))))

(deftest canonical-carries-reject-invalid-shapes-scope-and-hidden-writes
  (let [program (fixture/typed-program 1)
        variants
        [(rewrite-loop program (fn [[op attrs extent init lambda]] (list op (dissoc attrs :carry) extent init lambda)))
         (rewrite-loop program (fn [[op attrs extent _ lambda]] (list op attrs extent 'sum lambda)))
         (rewrite-loop program (fn [[op attrs _ init lambda]] (list op attrs 'acc init lambda)))
         (rewrite-loop program (fn [[op attrs extent init [_ _ region]]]
                                 (list op attrs extent init (list 'lambda '[k wrong] region))))
         (rewrite-loop program (fn [[op attrs extent init [lam params [region locals effects _]]]]
                                 (list op attrs extent init (list lam params (list region locals effects 'unknown)))))
         (rewrite-loop program (fn [[op attrs extent init [lam params [region locals effects _]]]]
                                 (list op attrs extent init
                                       (list lam params (list region locals effects '(aset packed 0 1.0))))))]]
    (doseq [variant variants] (is (not= :accepted (reason variant))))
    (is (= :typed-soac-effect-carry-write (reason (last variants))))))

(deftest carry-update-destination-reads-require-read-write-storage
  (let [program (rewrite-loop
                 (fixture/typed-program 1)
                 (fn [[op attrs extent init [lam params [region locals effects _]]]]
                   (list op attrs extent init
                         (list lam params (list region locals effects '(aget packed (+ (* i 8) k)))))))
        facts (assoc-in (dialect/facts program) [:equations :carry :attributes :result-storage 0 :access] :read-write)
        readable (dialect/make facts (dialect/equations program) [])
        operation (first (lower/lower-typed-effect-map readable :ze:0))]
    (is (= :typed-soac-effect-carry-read-access (reason program)))
    (is (contains? (:inputs operation) 'words))
    (is (= :kernel-body (get-in (fixture/artifact operation :opencl-portable) [:attributes :emission-route])))))

(deftest capture-collisions-fail-closed-before-execution
  (doseq [physical ['acc 'sum 'k]]
    (let [program (dialect/remap-values (fixture/typed-program 1) {'seed physical})
          operation (first (lower/lower-typed-effect-map program :ze:0))]
      (is (= :accepted (reason program)))
      (is (= :scheduled-effect-carry
             (try (jvm/compile-effect-segmap operation) :accepted
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest canonical-result-scope-follows-the-effect-spine
  (let [base (fixture/typed-program 1)
        reverse-effects (walk/postwalk
                         (fn [form]
                           (if (and (seq? form) (= 'effect-region (first form)) (= 3 (count form)))
                             (list 'effect-region (second form) (vec (reverse (nth form 2))))
                             form)) base)
        capture-update (rewrite-loop
                        base
                        (fn [[op attrs extent init [lam params [region locals effects _]]]]
                          (list op attrs extent init
                                (list lam params (list region locals effects
                                                       (with-meta '(+ acc row-count) {:raster.type/tag 'double}))))))
        operation (first (lower/lower-typed-effect-map capture-update :ze:0))
        totals (float-array 2)
        execute (eval (list 'fn '[x words totals rows seed] (jvm/compile-effect-segmap operation)))]
    (is (= :typed-soac-effect-loop (reason reverse-effects)))
    (is (= '(+ acc rows) (get-in operation [:scalar-region :effects 0 :loop :carry :update])))
    (execute (float-array (range 16)) (float-array 16) totals 2 (float 0.25))
    (is (= [2.25 2.25] (vec totals)))))
