(ns raster.compiler.backend.jvm.loop-carrier-test
  (:refer-clojure :exclude [aget aset])
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.arrays :as arrays :refer [aget aset alloc-like]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.backend.jvm.split :as split]
            [raster.compiler.passes.parallel.descriptors :as descriptors]
            [raster.compiler.pipeline :as pipeline]))

(deftm grouped-carry
  (All [T] [src :- (Array T) rows :- Long group :- Long slab :- Long] :- (Array T)
    (let [out (alloc-like src (* rows slab))]
      (par/map-void! o (clojure.core/* rows slab)
        (let [g (quot o slab) i (rem o slab) gg (clojure.core/* g group)]
          (aset out o
            (loop [r 1 acc (aget src (clojure.core/+ (clojure.core/* gg slab) i))]
              (if (< r group)
                (recur (inc r)
                       (n/+ acc (aget src (clojure.core/+ (clojure.core/* (clojure.core/+ gg r) slab) i))))
                acc)))))
      out)))

(deftm widening-carry [x :- Double count :- Long] :- Double
  (loop [i 0 acc 0]
    (if (< i count) (recur (inc i) (+ acc x)) acc)))

(deftm mapped-literal-carry
  (All [T] [src :- (Array T) count :- Long] :- (Array T)
    (let [out (alloc-like src 1)]
      (par/map! out t 1 nil
        (loop [i 0 acc 0.0]
          (if (< i count)
            (recur (inc i) (n/+ acc (n/* (aget src i) (aget src i))))
            acc)))
      out)))

(deftm mapped-cancelling-literal-carry
  (All [T] [src :- (Array T) count :- Long] :- (Array T)
    (let [out (alloc-like src 1)]
      (par/map! out t 1 nil
        (loop [i 0 acc 0.0]
          (if (< i count) (recur (inc i) (+ acc (aget src i))) acc)))
      out)))

(defmacro def-split-cancelling-carry [name seed & [length-param]]
  ;; Repeated retained maps cross the real method-size threshold. This tests
  ;; the helper-extraction optimizer, not only the small typed-call route.
  (let [length-param (or length-param 'cnt)
        stage `(par/map! ~'out ~'t 1 nil
                 (loop [~'i 0 ~'acc ~seed]
                   (if (< ~'i ~length-param)
                     (recur (inc ~'i) (n/+ ~'acc (aget ~'src ~'i))) ~'acc)))]
    `(deftm ~name
       (~'All [~'T] [~'src ~':- (~'Array ~'T) ~length-param ~':- ~'Long] ~':- (~'Array ~'T)
         (let [~'out (alloc-like ~'src 1)]
           ~stage ~stage ~stage ~stage ~'out)))))

(def-split-cancelling-carry split-contextual-carry 0.0)
(def-split-cancelling-carry split-explicit-double-carry (double 0.0))

(defmacro def-weighted-carry [name stages]
  (let [stage `(par/map! ~'out ~'t 1 nil
                 (loop [~'i 0 ~'acc 0.0]
                   (if (< ~'i ~'cnt)
                     (recur (inc ~'i)
                            (n/+ ~'acc (n/* ~'gain (double (aget ~'src ~'i)))))
                     ~'acc)))]
    `(deftm ~name
       [~'src ~':- (~'Array ~'float) ~'cnt ~':- ~'Long ~'gain ~':- ~'Double]
       ~':- (~'Array ~'float)
       (let [~'out (alloc-like ~'src 1)]
         ~@(repeat stages stage)
         ~'out))))

(def-weighted-carry unsplit-weighted-carry 1)
(def-weighted-carry split-weighted-carry 4)

(deftest helper-extraction-preserves-typed-mixed-precision-operands
  ;; The Double weighted term is converted at the retained Float addition,
  ;; not after a Double reduction. Both method layouts must produce zero.
  (let [input (float-array [1.0e8 1.0 -1.0e8])]
    (doseq [[v f] [[#'unsplit-weighted-carry unsplit-weighted-carry]
                   [#'split-weighted-carry split-weighted-carry]]]
      (let [compiled (pipeline/compile-aot v)]
        (doseq [gain [1.0 2.0]]
          (is (= [0.0] (vec (f input 3 gain))))
          (is (= [0.0] (vec (compiled input 3 gain))))))))
  (let [[_ bindings & body] (first (pipeline/get-walked-body #'split-weighted-carry :float))
        forms (concat (map second (partition 2 bindings)) body)]
    (is (> (reduce + (map split/estimate-form-size forms)) split/TARGET-SIZE))
    (let [extracted (#'bytecode/split-body-into-helpers
                      body bindings ['src 'cnt 'gain] ['floats 'long 'double]
                      (the-ns 'raster.compiler.backend.jvm.loop-carrier-test) 2)]
      (is (= 4 (count (:helpers extracted))))
      (is (every? #(some (fn [form]
                          (and (seq? form) (= '.invk (first form))
                               (= 'float (:raster.type/tag (meta form)))))
                        (tree-seq coll? seq (:walked-body %)))
                  (:helpers extracted))))))

(def ^:const helper-shadow-count 999)
(def-split-cancelling-carry split-core-shadow-carry 0.0 count)
(def-split-cancelling-carry split-constant-shadow-carry 0.0 helper-shadow-count)

(deftest extracted-helper-captures-lexical-inputs-before-global-resolution
  (doseq [f [split-core-shadow-carry split-constant-shadow-carry]]
    (is (= [0.0] (vec (f (float-array [1.0e8 1.0 -1.0e8]) 3))))
    (is (= [1.0] (vec (f (double-array [1.0e8 1.0 -1.0e8]) 3))))))

(deftest extracted-helper-distinguishes-parent-local-from-actual-constant
  (let [output (with-meta 'out {:raster.type/tag 'floats})
        length (with-meta 'count {:raster.type/tag 'long})
        bindings [output '(float-array cnt) length 'cnt]
        outer '(raster.par/map! out i count nil
                 (clojure.core/+ (clojure.core/aget src i) helper-shadow-count))
        inner '(raster.par/map! out i cnt nil
                 (let* [count 1] (clojure.core/aget src count)))
        capture (fn [body]
                  (set (:params (first (:helpers
                                        (#'bytecode/split-body-into-helpers
                                          [body] bindings ['src 'cnt] ['floats 'long]
                                          (the-ns 'raster.compiler.backend.jvm.loop-carrier-test) 2))))))]
    (is (= #{'src 'out 'count} (capture outer))
        "parent let-bound count is captured, but the actual numeric constant is not")
    (is (= #{'src 'out 'cnt} (capture inner))
        "the nested count binder is not captured as a parent dependency")))

(deftest extracted-initializers-see-only-prior-parent-bindings
  (let [temporary (with-meta 'tmp {:raster.type/tag 'floats})
        shadow (with-meta 'helper-shadow-count {:raster.type/tag 'long})
        early '(raster.par/map! out i cnt nil
                 (clojure.core/+ (clojure.core/aget src i) helper-shadow-count))
        body '(raster.par/map! out i cnt nil
                (clojure.core/+ (clojure.core/aget src i) helper-shadow-count))
        result (#'bytecode/split-body-into-helpers
                 [body] [temporary early shadow 3]
                 ['src 'out 'cnt] ['floats 'floats 'long]
                 (the-ns 'raster.compiler.backend.jvm.loop-carrier-test) 2)
        captures (mapv #(set (:params %)) (:helpers result))]
    (is (= #{'src 'out 'cnt} (first captures))
        "the earlier initializer sees the true global constant")
    (is (= #{'src 'out 'cnt 'helper-shadow-count} (second captures))
        "the body sees the now-bound local, not the global constant")))

(deftest helper-extraction-preserves-contextual-and-explicit-seed-widths
  (is (= [0.0] (vec (split-contextual-carry (float-array [1.0e8 1.0 -1.0e8]) 3))))
  (is (= [1.0] (vec (split-contextual-carry (double-array [1.0e8 1.0 -1.0e8]) 3))))
  (is (= [1.0] (vec (split-explicit-double-carry (float-array [1.0e8 1.0 -1.0e8]) 3))))
  ;; Check the actual extraction predicate rather than requiring first-call
  ;; compilation: this test also runs correctly in an already-warm REPL.
  (let [[_ bindings & body] (first (pipeline/get-walked-body #'split-contextual-carry :float))
        forms (concat (map second (partition 2 bindings)) body)]
    (is (some descriptors/par-form? forms))
    (is (> (reduce + (map split/estimate-form-size forms)) split/TARGET-SIZE))))

(deftest mapped-bare-floating-initializer-realizes-the-contextual-element-type
  (doseq [[dtype input expected] [[:float (float-array [1.0e8 1.0 -1.0e8]) 0.0]
                                [:double (double-array [1.0e8 1.0 -1.0e8]) 1.0]]]
    (is (= [expected] (vec (mapped-cancelling-literal-carry input 3))))
    (is (= [expected] (vec ((pipeline/compile-aot #'mapped-cancelling-literal-carry
                                                :dtype dtype) input 3)))))
  ;; Like an attention score, each Float product/add rounds independently.
  ;; A Double carry would retain nine unit terms and round to 100000008.
  (let [input (float-array (cons 10000.0 (repeat 9 1.0)))]
    (is (= [1.0e8] (vec (mapped-literal-carry input 10))))
    (is (= [1.0e8] (vec ((pipeline/compile-aot #'mapped-literal-carry :dtype :float) input 10))))))

(deftm coupled-carry [step :- Double] :- Double
  (loop [a (float 1.0e8) b (float 0.0) i 0]
    (if (< i 2) (recur (+ a step) (+ b a) (inc i)) b)))

(deftm case-loop-result [src :- (Array double) count :- Long tag :- Long] :- Double
  (case (int tag)
    0 (let [answer (loop [i 0 sum 0.0]
                     (if (< i count)
                       (recur (inc i) (+ sum (clojure.core/aget src i)))
                       sum))]
        answer)
    1 9.0
    (loop [i 0 sum 0.0]
      (if (< i count)
        (recur (inc i) (+ sum (clojure.core/aget src i)))
        sum))))

(deftest case-reconciles-boxed-loop-branches-with-its-primitive-merge
  (let [compiled (pipeline/compile-aot #'case-loop-result)]
    (doseq [[input expected] [[(double-array [1 2 3]) 6.0] [(double-array 0) 0.0]]
            tag [0 1 2]]
      (let [answer (if (= tag 1) 9.0 expected)]
        (is (== answer (case-loop-result input (count input) tag)))
        (is (== answer (compiled input (count input) tag)))))))

(deftest case-rejects-a-corrupt-boolean-result-prediction-before-verification
  (let [branch (with-meta '(let* [] 1.5) {:raster.type/tag 'boolean})
        body (list 'case* 0 0 0 branch {0 [0 branch]} :compact :int)
        reason (try
                 (bytecode/compile-specialized-class!
                   (str "raster.test.CorruptCase" (gensym))
                   [{:name 'corrupt :params [] :walked-body [body]
                     :source-ns *ns* :return-tag 'boolean :element-type 'double}])
                 nil
                 (catch Exception error
                   (loop [cause error]
                     (or (:reason (ex-data cause))
                         (when-let [nested (ex-cause cause)] (recur nested))))))]
    (is (= :case-result-type reason))))

(deftest mapped-float-carry-retains-per-add-rounding
  (let [input (float-array [1.0e8 4.0 1.0 2.0 -1.0e8 -1.0])
        ordinary (grouped-carry input 1 3 2)
        compiled (pipeline/compile-aot #'grouped-carry :dtype :float)]
    (is (= [0.0 5.0] (vec ordinary)))
    (is (= [0.0 5.0] (vec (compiled input 1 3 2))))))

(deftest mapped-double-carry-retains-double-arithmetic
  (let [input (double-array [1.0e8 1.0 -1.0e8])
        ordinary (grouped-carry input 1 3 1)
        compiled (pipeline/compile-aot #'grouped-carry :dtype :double)]
    (is (= [1.0] (vec ordinary)))
    (is (= [1.0] (vec (compiled input 1 3 1))))))

(deftest genuine-recurrence-widening-is-not-truncated
  (is (== 1.5 (widening-carry 0.5 3)))
  (is (== 1.5 ((pipeline/compile-aot #'widening-carry) 0.5 3))))

(deftest dependent-carry-widening-reaches-a-fixed-point
  (let [pairs [['a '(float 1.0e8)] ['b '(float 0.0)] ['i 0]]
        body '((if (< i 2) (recur (+ a step) (+ b a) (inc i)) b))]
    (is (= [:double :double :int]
           (#'bytecode/loop-recur-types-in-scope pairs body {'step {:type :double}})))))

(deftest coupled-carries-preserve-the-wider-value
  (is (== 200000001.0 (coupled-carry 1.0)))
  (is (== 200000001.0 ((pipeline/compile-aot #'coupled-carry) 1.0))))

(deftest branch-width-joins-do-not-forget-earlier-carry-widening
  (is (= [:double :double]
         (#'bytecode/loop-recur-types-in-scope
           [['a 0] ['b '(long 1)]]
           '((recur (if p b (long 0)) (+ b step)))
           {'step {:type :double} 'p {:type :bool}}))))

(deftest long-and-float-recurrences-use-the-common-numeric-width
  (is (= [:double]
         (#'bytecode/loop-recur-types-in-scope
           [['a '(float 0.0)]]
           '((if p (recur (float 1.0)) (recur (long 100000001)))) {'p {:type :bool}}))))

(deftest unknown-carry-shadows-an-outer-type
  (is (= [:ref]
         (#'bytecode/loop-recur-types-in-scope
           [['a '(opaque)]] '((recur a)) {'a {:type :float}}))))

(deftest unknown-nested-local-shadows-the-loop-carry
  (is (= [:double]
         (#'bytecode/loop-recur-types-in-scope
           [['a '(float 0.0)]]
           '((let* [a (opaque)] (recur (+ a (float 1.0))))) {}))))

(deftest recurrence-value-preserves-local-scope-and-tail-joins
  (is (= [:double]
         (#'bytecode/loop-recur-types-in-scope
           [['a '(float 0.0)]]
           '((recur (let* [a (opaque)] (+ a (float 1.0))))) {})))
  (is (= [:double]
         (#'bytecode/loop-recur-types-in-scope
           [['a '(float 0.0)]]
           '((recur (do (opaque) (if p (long 100000001) (float 1.0)))))
           {'p {:type :bool}}))))

(deftest recurrence-branches-and-initializers-preserve-nested-scope
  (doseq [expression
          '[(if p (let* [a (opaque)] (+ a (float 1.0))) (float 0.0))
            (let* [b (let* [a (opaque)] (+ a (float 1.0)))] b)
            (+ (float 1.0) (let* [a (opaque)] (+ a (float 1.0))))]]
    (is (= [:double]
           (#'bytecode/loop-recur-types-in-scope
             [['a '(float 0.0)]] (list (list 'recur expression))
             {'p {:type :bool}})))))

(deftest noncanonical-array-read-name-is-not-an-intrinsic
  (let [ns-name (gensym "raster.compiler.backend.jvm.noncanonical-read-probe-")
        probe (create-ns ns-name)]
    (try
      (intern probe 'aget (fn [_ _] -3.25))
      (binding [*ns* probe]
        (is (nil? (#'bytecode/infer-arg-stack-type '(aget src 0) {'src {:type :ref :hint 'floats}})))
        (is (= :double (#'bytecode/infer-arg-stack-type '(Math/abs (aget src 0))
                                                       {'src {:type :ref :hint 'floats}}))))
      ;; Caller/source namespace differences cannot turn a bare name into a
      ;; canonical read. Explicit lowered reads remain typed in either context.
      (binding [*ns* (the-ns 'clojure.core)]
        (is (nil? (#'bytecode/infer-arg-stack-type '(aget src 0) {'src {:type :ref :hint 'floats}})))
        (is (= :float (#'bytecode/infer-arg-stack-type '(clojure.core/aget src 0)
                                                       {'src {:type :ref :hint 'floats}}))))
      (is (nil? (#'bytecode/infer-arg-stack-type '(aget src 0)
                     {'aget {:type :ref} 'src {:type :ref :hint 'floats}})))
      (finally (remove-ns ns-name)))))
