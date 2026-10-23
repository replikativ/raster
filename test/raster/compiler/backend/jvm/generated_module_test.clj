(ns raster.compiler.backend.jvm.generated-module-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.ad.reverse :as reverse]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.backend.jvm.generated-body :as generated]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.scalar-conversion :as conversion]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.fixtures.distributed-training :as training]))

(defn- emit [parameters tags body]
  (let [interface (types/ensure-fn-interface! tags)
        emitted (core/emit-owned-bytecode!
                 bytecode/compile-typed-impl! "raster.compiled.GeneratedModuleTest"
                 parameters tags 'Object body
                 (the-ns 'raster.compiler.backend.jvm.generated-module-test)
                 (:iface-name interface) 'generated-module-test)]
    (assoc emitted :input-body body :interface-class (:iface-class interface))))

(deftest a-proposal-retains-every-context-and-is-not-admission
  (let [parameters '[x]
        context (completion/bytecode-operand-context {})
        input (completion/complete-value (completion/bytecode-floating-input-contract Float/TYPE context) 'x)
        boxed (completion/complete-value completion/object-input-contract 'converted)
        body [(list 'let* ['converted input 'boxed boxed] '[boxed nil])]
        physical (emit parameters '[double] body)
        proposal (generated/propose-value-source physical parameters)
        conversion (second (second (:source proposal)))
        obligations (:completion-obligations proposal)]
    (is (false? (:admitted? proposal)))
    (is (identical? (first body) (:original proposal)))
    (is (identical? physical (:physical proposal)))
    (is (not (completion/contains-completion? (:source proposal))))
    (is (conversion/canonical-form? conversion))
    (is (= {:source-dtype :double :target-dtype :float :rounding :nearest-even
            :overflow :ieee :source-op 'clojure.core/unchecked-float}
           (second conversion)))
    (is (= 2 (count obligations)))
    (is (every? #(= :pending (:status %)) obligations))
    (is (identical? input (:source (first obligations))))
    (is (identical? boxed (:source (second obligations))))
    (is (= '[boxed nil] (last (:source proposal))))
    (is (= Float/POSITIVE_INFINITY (first ((:implementation physical) Double/MAX_VALUE))))))

(deftest a-result-fact-is-not-the-inputs-builtin-number-proof
  (let [parameters '[x]
        context (completion/bytecode-operand-context {})
        input (completion/complete-value (completion/bytecode-floating-input-contract Double/TYPE context) 'x)
        body [(list 'let* ['converted input] '[converted])]
        physical (emit parameters '[Object] body)]
    (is (= [2.0] ((:implementation physical) (float 2.0))))
    (is (= :generated-body-unsupported
           (try (generated/propose-value-source physical parameters) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest proposals-require-every-original-completion-record
  (let [parameters '[x]
        term (completion/complete-value completion/object-input-contract 'x)
        body [(list 'let* ['value term] '[value])]
        physical (emit parameters '[double] body)
        missing (assoc-in physical [:emission :completion-contexts] {})]
    (is (= :unclosed-generated-module-completion
           (try (generated/propose-value-source missing parameters) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (generated/propose-value-source
                  (assoc physical :input-body (mapv identity body)) parameters)))))

(deftest root-spines-must-be-ssa-and-shadowed-locals-cannot-borrow-root-facts
  (doseq [bindings ['[value 1.0 value 2.0] '[x 1.0]]]
    (let [parameters '[x]
          body [(list 'let* bindings [(last (take-nth 2 bindings))])]
          physical (emit parameters '[double] body)]
      (is (thrown? clojure.lang.ExceptionInfo (generated/propose-value-source physical parameters)))))
  (let [parameters '[x other]
        term (completion/complete-value
              (completion/bytecode-floating-input-contract Double/TYPE (completion/bytecode-operand-context {})) 'x)
        body [(list 'let* ['result (list 'let* '[x other] term)] '[result])]
        physical (emit parameters '[double Object] body)]
    (is (= [2.0] ((:implementation physical) 1.0 (float 2.0))))
    (is (thrown? clojure.lang.ExceptionInfo (generated/propose-value-source physical parameters)))))

(deftest physical-entry-counts-cannot-be-truncated-into-a-proposal
  (let [parameters '[x] body ['(let* [value x] [value])]
        physical (emit parameters '[double] body)
        bridge (bytecode/retained-boxed-entry-bridge (:emission physical))]
    (with-redefs [bytecode/retained-boxed-entry-bridge
                  (constantly (assoc-in bridge [:boxed-entry :parameter-carriers] []))]
      (is (thrown? clojure.lang.ExceptionInfo (generated/propose-value-source physical parameters))))))

(deftest quoted-completion-shaped-data-is-not-a-module-operation
  (let [parameters '[x]
        term (completion/complete-value completion/object-input-contract 'x)
        body [(list 'let* ['value (list 'quote term)] '[value])]
        physical (emit parameters '[double] body)
        proposal (generated/propose-value-source physical parameters)]
    (is (empty? (:completion-obligations proposal)))
    (is (= (first body) (:source proposal)))
    (is (false? (:admitted? proposal)))))

(deftest the-real-training-body-produces-one-complete-provisional-spine
  (entry/call-with-definition-inputs
   (fn []
     (let [prepared (reverse/prepare-value+grad #'training/local-loss [0])
           _ (reverse/adopt-prepared-program-inputs! prepared)
           physical (reverse/prepared-bytecode-entry prepared)
           proposal (generated/propose-value-source physical (:params prepared))
           original (first (:input-body physical))
           source (:source proposal)
           obligations (:completion-obligations proposal)
           expected (filter completion/canonical-form? (tree-seq coll? seq original))]
       (is (false? (:admitted? proposal)))
       (is (= (count expected) (count obligations)))
       (is (pos? (count obligations)))
       (is (not (completion/contains-completion? source)))
       (is (= (take-nth 2 (second original)) (take-nth 2 (second source))))
       (is (= (last original) (last source)))
       (is (every? #(= :pending (:status %)) obligations))
       (is (identical? prepared (reverse/assert-prepared-program-source! prepared)))))))
