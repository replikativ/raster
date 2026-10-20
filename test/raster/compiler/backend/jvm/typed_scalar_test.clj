(ns raster.compiler.backend.jvm.typed-scalar-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.ir.soac-dialect :as soac]))

(defn- reason-of [thunk]
  (try (thunk) nil (catch clojure.lang.ExceptionInfo exception
                     (:reason (ex-data exception)))))

(deftest selected-and-lexical-calls-cannot-borrow-generic-scalar-semantics
  (let [selected (with-meta '(.invk not-a-live-receiver missing-a missing-b)
                   {:raster.op/original 'clojure.core/*})
        lexical (with-meta '(.invk local-receiver missing-a missing-b)
                  {:raster.op/lexical-callee true})]
    (doseq [expression [selected (list 'clojure.core/+ 1.0 selected)]]
      (is (= :typed-scalar-owned-source-required
             (reason-of #(scalar/evaluate-expression
                          'raster.compiler.backend.jvm.typed-scalar-test {} expression)))))
    ;; An unauthenticated lexical receiver has no purity evidence either. Reject
    ;; it at that earlier boundary; do not weaken purity merely to reach dispatch.
    (is (= :typed-scalar-effect
           (reason-of #(scalar/evaluate-expression
                        'raster.compiler.backend.jvm.typed-scalar-test {} lexical))))))

(deftest source-array-size-uses-the-actual-constructor-count-conversion
  (let [outcome (fn [f]
                  (try {:elements (f)}
                       (catch Throwable error {:error (class error)})))]
    ;; Every nonnegative converted count is at most three: these source oracles never
    ;; allocate a large host array, including the important 2^32 -> zero case.
    (doseq [count [0 1 3 -1 2147483648 4294967296 -4294967296 Long/MAX_VALUE
                  0.5 -0.5 -1.5 1/2 Double/NaN Double/NEGATIVE_INFINITY]]
      (is (= (outcome #(alength (double-array count)))
             (outcome #(:elements
                        (scalar/evaluate-array-size
                         'raster.compiler.backend.jvm.typed-scalar-test {'count count} 'count))))
          (str count)))
    (doseq [value [nil [1.0 2.0]]]
      (is (= :typed-scalar-array-size
             (reason-of #(scalar/evaluate-array-size
                          'raster.compiler.backend.jvm.typed-scalar-test {'count value} 'count)))
          "sequence overloads require their own retained realization contract")))
  (let [calls (atom 0)]
    (with-redefs [scalar/evaluate-expression (fn [& _] (swap! calls inc) 4294967296)]
      (is (= {:source-value 4294967296 :elements 0}
             (scalar/evaluate-array-size 'user {} 'extent)))
      (is (= 1 @calls)))))

(deftest short-circuit-calls-distinguish-falsy-values-from-exhaustion
  (doseq [expression ['(and) '(or)
                      '(and false 7) '(and nil 7)
                      '(or false 7) '(or nil 7)
                      '(and 7 false) '(and 7 nil)
                      '(or nil false) '(or false nil)
                      '(and 0 7) '(or 0 7)
                      '(and true 7) '(or false nil 7)]]
    (is (= (eval expression)
           (scalar/evaluate-expression
            'raster.compiler.backend.jvm.typed-scalar-test {} expression))
        (pr-str expression)))
  (testing "short-circuiting does not resolve a skipped SSA operand"
    (doseq [[expression expected] [['(and false missing) false]
                                  ['(and nil missing) nil]
                                  ['(or 7 missing) 7]]]
      (is (= expected (scalar/evaluate-expression
                       'raster.compiler.backend.jvm.typed-scalar-test {} expression)))))
  (testing "a required operand still fails rather than disappearing"
    (doseq [expression ['(and true missing) '(or false missing) '(or nil missing)]]
      (is (= :typed-scalar-unbound
             (reason-of #(scalar/evaluate-expression
                          'raster.compiler.backend.jvm.typed-scalar-test {} expression)))))))

(deftest typed-region-executes-retained-ssa-and-dtypes
  (let [lambda
        (soac/lambda-form
         '[x n]
         [(soac/local-value 'scaled :double '(clojure.core/* x 2.0))
          (soac/local-value 'counted :double '(double n))]
         '[(clojure.core/+ scaled counted)])]
    (is (= [{:type :double :value 5.0}]
           (scalar/evaluate-region
            'raster.compiler.backend.jvm.typed-scalar-test lambda
            [{:type :double :value 1.5} {:type :long :value 2}]
            [:double])))))

(deftest typed-region-resolves-java-static-semantics-without-a-function-table
  (is (= [{:type :double :value 3.0}]
         (scalar/evaluate-region
          'raster.compiler.backend.jvm.typed-scalar-test
          (soac/lambda-form '[x] '[(Math/sqrt x)])
          [{:type :double :value 9.0}]
          [:double]))))

(deftest source-executor-retains-explicit-region-checkpoints-and-failure-order
  ;; These dtypes are explicit region conversions, not inferred source carriers.
  ;; A source executor must not erase the Float local's rounding or exception.
  (let [initialize '(clojure.core/+ x 1.0)
        finish '(clojure.core/- rounded x)
        region (soac/lambda-form
                '[x] [(soac/local-value 'rounded :float initialize)] [finish])
        source (fn [x]
                 (let [rounded (float (+ (double x) 1.0))]
                   (- (double rounded) (double x))))
        calls (atom [])
        execute (fn [expression environment]
                  (swap! calls conj expression)
                  (cond
                    (identical? initialize expression) (+ (double (get environment 'x)) 1.0)
                    (identical? finish expression)
                    (do (is (instance? Float (get environment 'rounded)))
                        (- (double (get environment 'rounded)) (double (get environment 'x))))
                    :else (throw (ex-info "unexpected source fragment" {}))))]
    (doseq [x [16777216.0 2.0 -2.0]]
      (reset! calls [])
      (is (= [{:type :double :value (source x)}]
             (scalar/evaluate-region
              'raster.compiler.backend.jvm.typed-scalar-test region
              [{:type :double :value x}] [:double] execute)))
      (is (= [initialize finish] @calls)))
    (reset! calls [])
    (is (thrown? IllegalArgumentException (source Double/MAX_VALUE)))
    (is (thrown? IllegalArgumentException
                 (scalar/evaluate-region
                  'raster.compiler.backend.jvm.typed-scalar-test region
                  [{:type :double :value Double/MAX_VALUE}] [:double] execute)))
    (is (= [initialize] @calls)
        "failed local completion cannot run the suffix or retry the initializer")))

(deftest typed-region-fails-on-unbound-effectful-or-unrepresentable-values
  (testing "SSA closure is checked at execution as well as IR validation"
    (is (= :typed-scalar-unbound
           (reason-of #(scalar/evaluate-region
                        'raster.compiler.backend.jvm.typed-scalar-test
                        (soac/lambda-form '[] '[missing]) [] [:double])))))
  (testing "the reference backend never executes an unproven side effect"
    (is (= :typed-scalar-effect
           (reason-of #(scalar/evaluate-region
                        'raster.compiler.backend.jvm.typed-scalar-test
                        (soac/lambda-form '[] '[(println "no")]) [] [:double])))))
  (testing "FP16 requires an explicit representable host scalar contract"
    (is (= :typed-scalar-half
           (reason-of #(scalar/evaluate-region
                        'raster.compiler.backend.jvm.typed-scalar-test
                        (soac/lambda-form '[] '[1.0]) [] [:half]))))))

(deftest typed-region-consumes-only-canonical-closed-numeric-constants
  (doseq [[expression dtype expected]
          [['java.lang.Float/NEGATIVE_INFINITY :float Float/NEGATIVE_INFINITY]
           ['Double/POSITIVE_INFINITY :double Double/POSITIVE_INFINITY]
           ['java.lang.Integer/MAX_VALUE :int Integer/MAX_VALUE]
           ['Long/MAX_VALUE :long Long/MAX_VALUE]]]
    (is (= [{:type dtype :value expected}]
           (scalar/evaluate-region
            'raster.compiler.backend.jvm.typed-scalar-test
            (soac/lambda-form [] [expression]) [] [dtype]))))
  (testing "lexical SSA bindings take precedence over literal spellings"
    (is (= 7.0 (scalar/evaluate-expression
                'raster.compiler.backend.jvm.typed-scalar-test
                {'Float/NEGATIVE_INFINITY 7.0} 'Float/NEGATIVE_INFINITY))))
  (testing "arbitrary fields are not resolved or reflected"
    (is (= :typed-scalar-unbound
           (reason-of #(scalar/evaluate-expression
                        'raster.compiler.backend.jvm.typed-scalar-test
                        {} 'java.lang.System/out))))))
