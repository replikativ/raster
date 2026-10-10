(ns raster.compiler.compatibility-reduction-packet-test
  (:require [clojure.test :refer [deftest is]]
            [raster.par]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.segred-body :as segred-body]
            [raster.compiler.passes.parallel.segop-lower-pass :as slp]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(def options {:dtype :double :array-types {'a :double 'out :double 'indices :int 'values :double}
              :scalar-types {'n :long}})

(def stable '(raster.par/reduce acc 0.0 i n (+ acc (clojure.core/aget a i))))
(def compound '(raster.par/reduce acc 0.0 i (clojure.core/alength a)
                                (+ acc (clojure.core/aget a i))))

(defn- reject-legacy-reductions [f]
  (let [original soac/par-form->soac]
    (with-redefs [soac/par-form->soac
                  (fn [& arguments]
                    (let [expression (second arguments)]
                      (when (contains? '#{raster.par/reduce par/reduce} (first expression))
                        (throw (AssertionError. "typed reduction reached legacy semantic inference")))
                      (apply original arguments)))]
      (f))))

(deftest scalar-reductions-import-complete-typed-contracts
  (reject-legacy-reductions
   (fn []
     (doseq [[expression expected] [[stable 1] [compound 2]]]
       (let [result (slp/segop-lower-pass (list 'let* ['result expression] 'result) options)
             packet (:form result)
             equations (:equations packet)]
         (is (= expected (count equations)))
         (is (= 1 (get-in result [:stats :typed-soac-reused])))
         (is (= (dec expected) (get-in result [:stats :typed-scalar-equations])))
         (is (every? :algorithm equations))
         (is (every? :physical-algorithm equations))
         (is (= packet (program/validate! packet)))
         (doseq [equation equations]
           (is (= (:operands equation) (:inputs (dialect/facts (:algorithm equation)))))
           (is (= (:results equation) (dialect/outputs (:algorithm equation))))
           (is (= equation (program/equation-for-binding
                            packet (second (:site equation)) (:source equation))))))))))

(deftest body-prefix-stays-after-earlier-effects-and-failures
  (reject-legacy-reductions
   (fn []
     (let [source (list 'let* [] '(swap! events conj :before) compound)
           packet (:form (slp/segop-lower-pass source options))
           pairs (vec (partition 2 (second (:source packet))))
           primary (last (:equations packet))
           original (eval (list 'fn ['a 'events] source))
           transformed (eval (list 'fn ['a 'events] (:source packet)))]
       (is (= '(swap! events conj :before) (second (first pairs))))
       (is (= 2 (count (:equations packet))))
       (is (= :binding (first (:site primary))))
       (is (= [:body 1] (get-in primary [:provenance :compatibility-source-site])))
       (is (= primary (program/equation-for-binding packet (second (:site primary)) (:source primary))))
       (doseq [function [original transformed]]
         (let [events (atom [])]
           (is (= 6.0 (function (double-array [1.0 2.0 3.0]) events)))
           (is (= [:before] @events)))))
     (let [source (list 'let* [] '(throw failure) compound)
           transformed (:source (:form (slp/segop-lower-pass source options)))
           failure (ex-info "before extent" {})]
       (doseq [expression [source transformed]]
         (let [function (eval (list 'fn ['a 'failure] expression))]
           (is (identical? failure
                           (try (function (double-array [1.0]) failure)
                                nil (catch Throwable error error))))))))))

(deftest independently-generated-prefixes-do-not-collide
  (let [source (list 'let* ['rstr_extent_0 7 'left compound 'right compound] '(+ left right))
        packet (:form (slp/segop-lower-pass source options))
        prefixes (filter #(get-in % [:attributes :host-only]) (:equations packet))
        physical-ids (mapcat #(dialect/outputs (:physical-algorithm %)) prefixes)]
    (is (= 4 (count (:equations packet))))
    (is (= 2 (count (distinct physical-ids))))
    (is (not-any? #{'rstr_extent_0} physical-ids))
    (is (= 4 (count (distinct (map :id (:equations packet))))))
    (is (= packet (program/validate! packet)))
    (is (= 12.0 ((eval (list 'fn ['a] (:source packet))) (double-array [1.0 2.0 3.0]))))))

(deftest mixed-unsupported-source-does-not-demote-admitted-reduction
  (let [source (list 'let* ['unsupported '(raster.par/scatter! out indices values n)
                           'result compound] 'result)
        whole (route/attempt source :double (:array-types options)
                             {:scalar-types (:scalar-types options)})]
    (is (some? (:declined whole)))
    (reject-legacy-reductions
     (fn []
       (let [result (slp/segop-lower-pass source options)]
         (is (= 1 (get-in result [:stats :typed-soac-reused])))
         (is (= 2 (count (:equations (:form result)))))
         (is (seq (get-in result [:stats :segops-declined])))
         (is (= '(raster.par/scatter! out indices values n)
                (nth (second (:source (:form result))) 1))))))))

(deftest a-later-reduction-retains-the-earlier-result-carrier
  (let [source '(let* [left (raster.par/reduce acc (clojure.core/long 0) i n
                                              (clojure.core/unchecked-add acc (clojure.core/long i)))
                       right (raster.par/reduce acc 0.0 i n
                                               (+ acc (double left)))] right)
        result (slp/segop-lower-pass source options)
        packet (:form result)
        right (program/equation-for-binding packet 'right
                                           (get (into {} (map vec (partition 2 (second (:source packet))))) 'right))]
    (is (= 2 (get-in result [:stats :typed-soac-reused])))
    (is (= :long (get-in packet [:values [:binding 'left] :dtype])))
    (is (= :long (get-in (dialect/facts (:physical-algorithm right)) [:values 'left :dtype])))
    (is (some #{[:binding 'left]} (:operands right)))
    (let [original (eval (list 'fn ['n] source))
          transformed (eval (list 'fn ['n] (:source packet)))]
      (doseq [n [0 1 5]]
        (is (= (original n) (transformed n)))))))

(deftest integral-two-phase-reduction-retains-its-certified-wrapping-combine
  (let [source '(let* [result (raster.par/reduce acc (clojure.core/long 0) i n
                                               (clojure.core/unchecked-add
                                                acc (clojure.core/aget a i)))] result)
        original (eval (list 'fn ['a 'n] source))
        input (long-array [Long/MAX_VALUE 1 0 0])]
    (is (= Long/MIN_VALUE (original input 4)))
    (doseq [kernel-dtype [:double :float]]
      (let [packet (:form (slp/segop-lower-pass
                           source (assoc options :dtype kernel-dtype :array-types {'a :long})))
            equation (first (:equations packet))
            operations (:operations equation)
            terminal (last operations)
            terminal-plan (segred-body/scalar-plan terminal)
            transformed (eval (list 'fn ['a 'n] (:source packet)))]
        (is (= [:block-local :cross-block] (mapv :phase operations)))
        (is (= [:long :long] (mapv :dtype operations)))
        (is (= :long (get-in packet [:values [:binding 'result] :dtype])))
        (is (= 'clojure.core/unchecked-add (:combine terminal-plan)))
        (is (= 'clojure.core/unchecked-add
               (first (:lambda (segop/scalar-reduce-op terminal)))))
        (is (= :wrap (get-in terminal [:reduction :algebra :overflow])))
        (is (= Long/MIN_VALUE (transformed input 4)))))))

(deftest malformed-success-cannot-enter-legacy-reduction-fallback
  (reject-legacy-reductions
   (fn []
     (with-redefs [route/attempt (fn [& _] {:program {}})]
       (is (= :raster/bug
              (try (slp/segop-lower-pass (list 'let* ['result stable] 'result) options)
                   nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))))

(deftest malformed-success-source-is-a-compiler-bug
  (let [original route/attempt]
    (doseq [source ['not-a-let '(let* [dangling] result) '(let* [42 0] result)]]
      (with-redefs [route/attempt
                    (fn [& arguments]
                      (update (apply original arguments) :program assoc :source source))]
        (is (= :raster/bug
               (try (slp/segop-lower-pass (list 'let* ['result stable] 'result) options)
                    nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))))

(deftest typed-refusal-is-retained-even-when-legacy-can-lower
  (with-redefs [route/attempt (fn [& _] {:declined {:reason :test-typed-refusal}})]
    (let [result (slp/segop-lower-pass (list 'let* ['result stable] 'result) options)]
      (is (= 1 (get-in result [:stats :segops-lowered])))
      (is (some #(and (= :test-typed-refusal (:reason %)) (= :typed-admission (:stage %)))
                (:diagnostics (:form result)))))))

(deftest malformed-success-equation-source-cannot-recertify-a-different-binding
  (let [original route/attempt]
    (doseq [change [(fn [equation] (assoc equation :site [:binding 'missing]))
                    (fn [equation] (assoc equation :source '(different-source)))]]
      (with-redefs [route/attempt
                    (fn [& arguments]
                      (update-in (apply original arguments) [:program :equations]
                                 #(mapv change %)))]
        (is (= :raster/bug
               (try (slp/segop-lower-pass (list 'let* ['result stable] 'result) options)
                    nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))))

(deftest repeated-body-reductions-have-distinct-certified-binding-sites
  (let [source (list 'let* [] compound compound)
        packet (:form (slp/segop-lower-pass source options))
        reductions (remove #(get-in % [:attributes :host-only]) (:equations packet))
        compile (requiring-resolve 'raster.compiler.backend.gpu.opencl-pass/opencl-pass)
        emitted (compile packet :dtype :double :min-elements 0 :array-types {'a :double})]
    (is (= 2 (count reductions)))
    (is (= 2 (count (distinct (map :site reductions)))))
    (is (= [[:body 0] [:body 1]]
           (mapv #(get-in % [:provenance :compatibility-source-site]) reductions)))
    (doseq [equation reductions]
      (is (= equation (program/equation-for-binding packet (second (:site equation)) (:source equation)))))
    (is (= 6.0 ((eval (list 'fn ['a] (:source packet))) (double-array [1.0 2.0 3.0]))))
    (is (= 2 (get-in emitted [:stats :ze-reduces])))
    (is (= 2 (get-in emitted [:stats :segop-reused])))
    (is (nil? (get-in emitted [:stats :segop-relowered])))))

(deftest actual-opencl-consumer-reuses-compound-binding-and-body-packets
  (let [compile (requiring-resolve 'raster.compiler.backend.gpu.opencl-pass/opencl-pass)]
    (reject-legacy-reductions
     (fn []
       (doseq [source [(list 'let* ['result compound] 'result)
                       (list 'let* [] compound)]]
         (let [packet (:form (slp/segop-lower-pass source options))
               result (compile packet :dtype :double :min-elements 0
                               :array-types {'a :double})]
           (is (= 1 (get-in result [:stats :ze-reduces])))
           (is (= 1 (get-in result [:stats :segop-reused])))
           (is (nil? (get-in result [:stats :segop-relowered])))))))))
