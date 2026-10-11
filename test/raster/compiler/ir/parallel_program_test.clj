(ns raster.compiler.ir.parallel-program-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.soac-dialect :as soac-dialect]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]))

(def ^:private reduce-source
  '(raster.par/reduce acc 0.0 i n (clojure.core/+ acc (clojure.core/aget values i))))

(deftest physical-write-capacity-covers-every-aliased-result
  (let [value #(av/tensor {:dtype :float :shape [%]})
        extent #'equation-graph/required-write-extent]
    (is (= 100 (launch/resolve-expression {} (extent {} {} [(value 100) (value 10)]))))
    (is (= 100 (launch/resolve-expression {} (extent {} {} [(value 10) (value 100)]))))
    (is (= 100 (extent {} {} [(value 100) (value 100)])))
    (is (nil? (extent {} {} [(value 100) (value '(unknown-dimension out))])))
    (is (nil? (extent {} {} [nil (value 100)])))))

(defn- lowered-program []
  (:form (segop-lower/segop-lower-pass (list 'let* ['total reduce-source] 'total)
                                       {:target-device :cpu:0 :dtype :double
                                        :array-types {'values :double}})))

(defn- dataflow-equation
  [id operands results]
  (program/->ProgramEquation id [:test id] nil (vec operands) (vec results) nil [:test]
                             #{} {} {}))

(defn- dataflow-program
  [inputs equations outputs]
  (let [ids (set (concat inputs outputs (mapcat :operands equations) (mapcat :results equations)))
        value (av/tensor {:dtype :int :shape []})]
    (program/make {:dialect :test
                   :values (zipmap ids (repeat value))
                   :inputs inputs
                   :equations equations
                   :outputs outputs
                   :operation? #{:test}})))

(defn- reason-of
  [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo exception
      (:reason (ex-data exception)))))

(deftest segops-are-first-class-typed-equations
  (let [p (lowered-program)
        equation (first (:equations p))]
    (is (program/parallel-program? p))
    (is (= :segop (:dialect p)))
    (is (= [:binding 'total] (:site equation)))
    (is (= ['n 'values] (:inputs p)))
    (is (= [[:binding 'total]] (:outputs p)))
    (is (= [reduce-source] (mapv :source (:equations p))))
    (is (every? av/abstract-value? (vals (:values p))))
    (is (= ['n] (:shape (get (:values p) 'values))))
    (is (= [] (:shape (get (:values p) [:binding 'total]))))
    (is (= {'n :int 'values :double}
           (program/declared-value-types (assoc-in p [:values 'n :dtype] :int)
                                         ['n 'values])))
    (is (seq (:operations equation)))
    (is (soac-dialect/program-form? (:algorithm equation)))
    (is (= :analyzed-source
           (get-in (soac-dialect/facts (:algorithm equation)) [:provenance :front-end])))
    (is (= (:operands equation) (:inputs (soac-dialect/facts (:algorithm equation)))))
    (is (= (:results equation) (soac-dialect/outputs (:algorithm equation))))
    (is (= '[n] (get-in (soac-dialect/facts (:algorithm equation))
                        [:values 'values :shape])))
    (is (= :typed-soac (:algorithm-dialect (first (:operations equation)))))
    (testing "the source binder is ordinary Clojure data, not an IR side channel"
      (is (nil? (meta (first (second (:source p)))))))))

(deftest equation-consumption-is-invalidated-by-a-source-rewrite
  (let [p (lowered-program)
        retained-source (:source (first (:equations p)))]
    ;; The lowering retains semantic type evidence on the expression. An
    ;; unannotated spelling is not a certificate for that analyzed expression.
    (is (seq (program/operations-for-binding p 'total retained-source)))
    (is (nil? (program/operations-for-binding
               p 'total
               '(raster.par/reduce acc 1.0 i n
                                   (clojure.core/+ acc (clojure.core/aget values i))))))))

(deftest source-certification-retains-semantic-metadata-and-bits
  (let [p (lowered-program)
        original (first (:equations p))
        typed-source (with-meta reduce-source {:raster.type/tag 'double :line 1})
        variants
        [(with-meta typed-source {:raster.type/tag 'float :line 1})
         (with-meta (list* (first typed-source)
                           (with-meta (second typed-source) {:raster.type/tag 'float})
                           (nnext typed-source))
           (meta typed-source))]]
    (doseq [[site lookup] [[[:binding 'total] #(program/equation-for-binding % 'total %2)]
                          [[:body 0] program/equation-for-source]]]
      (let [equation (assoc original :site site :source typed-source)
            retained (assoc p :equations [equation])]
        (is (= equation (lookup retained typed-source)))
        (is (= equation (lookup retained
                                (with-meta typed-source
                                  {:raster.type/tag 'double :file "new.clj" :line 90
                                   :column 8 :end-line 91 :end-column 12}))))
        (doseq [changed variants]
          (is (= typed-source changed) "ordinary equality cannot certify these changes")
          (is (nil? (lookup retained changed))))
        (let [plus (last typed-source)
              changed (with-meta
                        (apply list (concat (butlast typed-source)
                                            [(with-meta plus {:raster.op/original 'raster.numeric/-})]))
                        (meta typed-source))]
          (is (= typed-source changed))
          (is (nil? (lookup retained changed))))
        (let [positive-zero (assoc equation :source (list 'identity 0.0))
              positive-program (assoc p :equations [positive-zero])]
          (is (= (list 'identity 0.0) (list 'identity -0.0)))
          (is (nil? (lookup positive-program (list 'identity -0.0)))))
        (is (= :semantic-fingerprint-unsupported
               (reason-of #(lookup retained
                                  (with-meta typed-source {:opaque (Object.)})))))))))

(deftest body-source-semantic-identity-still-refuses-ambiguity
  (let [p (lowered-program)
        original (assoc (first (:equations p)) :site [:body 0])
        second-site (assoc original :id 1 :site [:body 1]
                           :source (with-meta (:source original) {:line 44}))]
    (is (nil? (program/equation-for-source
                (assoc p :equations [original second-site]) reduce-source)))))

(deftest dialect-validation-is-a-full-conversion
  (let [p (lowered-program)]
    (try
      (program/validate! p (constantly false))
      (is false "an illegal remaining operation must fail validation")
      (catch clojure.lang.ExceptionInfo e
        (is (= :illegal-op-remains (:reason (ex-data e))))
        (is (= :segop (:target-dialect (ex-data e))))
        (is (= :none (:fallback (ex-data e))))))))

(deftest host-only-equations-require-an-explicit-empty-schedule-contract
  (let [p (lowered-program)
        equation (first (:equations p))
        host-only (assoc p :equations [(-> equation
                                           (assoc :operations [])
                                           (assoc-in [:attributes :host-only] true))])]
    (is (= host-only (program/validate! host-only (constantly false))))
    (testing "an empty schedule cannot arise accidentally"
      (try
        (program/validate! (assoc p :equations [(assoc equation :operations [])]))
        (is false "empty operations without :host-only must be rejected")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :parallel-program-operations (:reason (ex-data exception)))))))
    (testing "host-only equations cannot also claim scheduled operations"
      (try
        (program/validate! (assoc p :equations [(assoc-in equation
                                                          [:attributes :host-only] true)]))
        (is false "host-only plus scheduled operations is contradictory")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :parallel-program-host-only-operations
                 (:reason (ex-data exception)))))))))

(deftest source-shaped-bound-expressions-retain-the-complete-typed-packet
  (let [source '(raster.par/reduce acc 0.0 i (clojure.core/alength values)
                                   (+ acc (clojure.core/aget values i)))
        p (:form (segop-lower/segop-lower-pass
                  (list 'let* ['total source] 'total)
                  {:target-device :cpu:0 :dtype :double
                   :array-types {'values :double}}))
        prefix (first (:equations p))
        equation (second (:equations p))
        pairs (mapv vec (partition 2 (second (:source p))))]
    (is (= 2 (count (:equations p))))
    (is (true? (get-in prefix [:attributes :host-only])))
    (is (empty? (:operations prefix)))
    (is (every? :algorithm (:equations p)))
    (is (= :long (get-in p [:values (first (:results prefix)) :dtype])))
    (is (= (:source prefix) (second (first pairs))))
    (is (some (set (:results prefix)) (:operands equation)))
    (is (seq (:operations equation)))
    (is (= [:binding 'total] (:site equation)))
    (is (= p (program/validate! p)))))

(deftest parallel-program-enforces-ordered-logical-ssa
  (testing "external inputs and earlier results are the only available operands"
    (is (= :parallel-program-use-before-definition
           (reason-of #(dataflow-program
                        '[input]
                        [(dataflow-equation 'first '[later] '[middle])
                         (dataflow-equation 'second '[input] '[later])]
                        '[middle])))))
  (testing "results cannot clobber a program input or an earlier result"
    (is (= :parallel-program-result-redefinition
           (reason-of #(dataflow-program
                        '[input]
                        [(dataflow-equation 'clobber '[input] '[input])]
                        '[input]))))
    (is (= :parallel-program-result-redefinition
           (reason-of #(dataflow-program
                        '[input]
                        [(dataflow-equation 'first '[input] '[result])
                         (dataflow-equation 'second '[result] '[result])]
                        '[result])))))
  (testing "outputs must be available at the end of the program"
    (is (= :parallel-program-unavailable-output
           (reason-of #(dataflow-program
                        '[input]
                        [(dataflow-equation 'equation '[input] '[result])]
                        '[missing])))))
  (testing "pass-through outputs and unused inputs/results remain legal"
    (is (program/parallel-program?
         (dataflow-program
          '[input unused]
          [(dataflow-equation 'equation '[input] '[unused-result])]
          '[input])))))

(deftest external-input-inference-preserves-first-use-order
  (let [equations [(dataflow-equation 'first '[z a] '[x])
                   (dataflow-equation 'second '[x b a] '[y])]]
    (is (= '[z a b] (program/infer-inputs equations)))))

(deftest known-value-types-projects-only-declared-dtypes
  (testing "a compatibility program contributes the dtypes it declares and omits the rest"
    (let [p (lowered-program)
          p (-> p
                (assoc-in [:values 'scale] (av/tensor {:dtype :double :shape []}))
                ;; a descriptor-map dtype (e.g. a quant format) is not a keyword scalar dtype
                (assoc-in [:values 'unknown] (av/tensor {:dtype {:format :q8} :shape []})))]
      (is (= {'values :double 'scale :double}
             (program/known-value-types p '[values scale unknown missing])))
      (is (= :parallel-program-parameter-dtype
             (reason-of #(program/declared-value-types p '[values unknown])))
          "the strict projection still refuses an undeclared dtype"))))
