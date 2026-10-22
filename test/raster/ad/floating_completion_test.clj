(ns raster.ad.floating-completion-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]
            [raster.compiler.core.value-completion :as completion]))

(def policy {:unchecked-math false})
(defn- typed [id tag] (with-meta id {:raster.type/tag tag}))
(defn- compile-program [params bindings result active]
  (#'reverse/make-runtime-value+grad-fn
   {:kind :generated-product-bindings :source-policy policy
    :bindings bindings :result result :active-inputs active}
   params {:generated-source-policy policy}))

(deftest floating-input-completion-uses-the-input-tangent-space
  (doseq [[input-tag result-tag input seed expected-class]
          [['double 'float 0.125 (float 2.0) Double]
           ['float 'double (float 0.125) 2.0 Float]
           ['double 'double 0.125 2.0 Double]
           ['float 'float (float 0.125) (float 2.0) Float]]]
    (let [x (typed 'x input-tag) result (typed 'result result-tag)
          carrier (if (= 'float result-tag) Float/TYPE Double/TYPE)
          term (completion/complete-value (completion/primitive-input-contract carrier policy) x)
          f (compile-program [x] [result term] result [x])
          [value pb] (f input)
          [gradient] (pb seed)]
      (is (= input value))
      (is (instance? expected-class gradient))
      (is (= seed gradient))
      (is (= gradient (first (pb seed)))))))

(deftest completion-fanout-projects-each-edge-before-accumulation
  (let [x (typed 'x 'float) a (typed 'a 'double) b (typed 'b 'double)
        result (typed 'result 'double)
        contract (completion/primitive-input-contract Double/TYPE policy)
        f (compile-program [x] [a (completion/complete-value contract x)
                                b (completion/complete-value contract x)
                                result '(raster.numeric/+ a b)] result [x])
        [value pb] (f (float 3.0))]
    (is (= 6.0 value))
    (doseq [seed [0.0 1.0 -2.0]]
      (let [[gradient] (pb seed)]
        (is (instance? Float gradient))
        (is (= (float (* 2.0 seed)) gradient))))))

(deftest completion-fanout-rounds-before-not-after-fanin
  (let [x (typed 'x 'float) a (typed 'a 'double) b (typed 'b 'double)
        result (typed 'result 'double)
        contract (completion/primitive-input-contract Double/TYPE policy)
        f (compile-program [x] [a (completion/complete-value contract x)
                                b (completion/complete-value contract x)
                                result '(raster.numeric/+ a b)] result [x])
        [_ pb] (f (float 1.0))
        ;; Two individually rounded subnormal contributions disappear; adding
        ;; them in Double before the Float conversion would produce MIN_VALUE.
        seed (* 0.49 (double Float/MIN_VALUE))]
    (is (pos? (float (+ seed seed))))
    (is (zero? (first (pb seed))))
    (is (instance? Float (first (pb seed))))))

(deftest completion-in-unselected-branch-materializes-public-typed-zero
  (let [x (typed 'x 'float) y (typed 'y 'double)
        result (typed 'result 'double) converted (typed 'converted 'double)
        positive (typed 'positive 'boolean)
        term (completion/complete-value
              (completion/primitive-input-contract Double/TYPE policy) x)
        branch (list 'if positive
                     (list 'let* [converted term] converted) y)
        f (compile-program [x y] [positive '(clojure.core/pos? y) result branch] result [x y])
        [_ selected] (f (float 2.0) 1.0)
        [_ unselected] (f (float 2.0) -1.0)]
    ;; Internal absence is materialized only at the public gradient boundary.
    (is (= [(float 1.0) 0.0] (selected 1.0)))
    (is (= [(float 0.0) 1.0] (unselected 1.0)))
    (is (= [(float 0.0) 0.0] (selected 0.0)))
    (is (= [(float 0.0) 0.0] (unselected 0.0)))
    (is (instance? Float (first (unselected 1.0))))))

(deftest primitive-completion-does-not-guess-missing-or-discrete-tangents
  (let [x (typed 'x 'double)]
    (doseq [[carrier result-tag] [[Integer/TYPE 'int] [Boolean/TYPE 'boolean]
                                  [Float/TYPE 'double] [Double/TYPE nil]]]
      (let [result (typed 'result result-tag)
            term (completion/complete-value (completion/primitive-input-contract carrier policy) x)]
        (is (= :ad-unclosed-completion-transpose
               (try (compile-program [x] [result term] result [x]) nil
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))))

(deftest primitive-completion-cannot-import-a-different-source-policy
  (let [x (typed 'x 'double) result (typed 'result 'float)
        term (completion/complete-value
              (completion/primitive-input-contract Float/TYPE {:unchecked-math true}) x)]
    (is (= :value-completion-policy-mismatch
           (try (compile-program [x] [result term] result [x]) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest bytecode-completion-cannot-import-a-clojure-source-owner
  (let [x (typed 'x 'double) result (typed 'result 'double)
        term (completion/complete-value
              (completion/bytecode-floating-input-contract
               Double/TYPE (completion/bytecode-operand-context {})) x)]
    (is (= :ad-unclosed-completion-transpose
           (try (compile-program [x] [result term] result [x])
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest integral-bytecode-completion-does-not-acquire-structural-ad
  (doseq [carrier [Integer/TYPE Long/TYPE]
          owner [nil {:backend :jvm-bytecode} {:source-policy policy}]
          active? [false true]]
    (let [x (typed 'x 'double) result (typed 'result 'long)
          term (completion/complete-value
                (completion/bytecode-integral-input-contract
                 carrier (completion/bytecode-operand-context {})) x)]
      (binding [reverse/*generated-product-context* owner]
        (doseq [run [#(#'reverse/ad-form-kind term active? result)
                     #(#'reverse/ad-record :completion result term {x active?})]]
          (is (= :ad-unclosed-completion-transpose
                 (try (run) :accepted
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))))
