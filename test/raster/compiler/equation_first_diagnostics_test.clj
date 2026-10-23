(ns raster.compiler.equation-first-diagnostics-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.fixtures.distributed-training :as training]
            [raster.dl.loss :as loss]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.pipeline :as pipeline]))

(deftest conditional-reduction-diagnostic-neither-admits-nor-invents-a-physical-carrier
  (let [step '(clojure.core/+ acc (clojure.core/aget x i))
        reduce-form (with-meta (list 'raster.par/reduce 'acc (float 0.0) 'i 'count step)
                      {:raster.type/elem-type :float})
        quotient (with-meta '(clojure.core// sum 2.0)
                   {:raster.effect/preserve-evaluation true :raster.type/tag 'double})
        source (list 'let* ['sum reduce-form 'result quotient] 'result)
        report (binding [raster.compiler.core.method-entry/*emission-inputs*
                         (atom {:static [] :values []})]
                 (frontend/coverage-decline
                  source {:dtype :float :array-types {'x :float} :scalar-types {'count :long}
                          :source-ns (the-ns 'raster.compiler.equation-first-diagnostics-test)}))
        proof (get-in report [:reduction-value-correspondences 0 :value-correspondence])]
    (is (= :typed-soac-source-coverage (:reason report)))
    (is (= :conditional-fp-reduction-correspondence (:scope proof)))
    (is (= :float (:dtype proof)))
    (is (= {:conversion :checked-int :evaluation :original-position
            :domain-correspondence :unproved}
           (:source-count-obligation proof)))
    (is (not-any? #(contains? proof %) [:carrier :nonthrowing? :removable?]))
    (is (= :uncertified-host-scalar (get-in report [:bindings 0 :reason])))
    (is (contains? (get-in report [:bindings 0 :source-evaluation-flags]) :preserve-source-evaluation))
    (is (nil? (get-in report [:bindings 0 :normal-completion])))))

(deftest training-loss-keeps-its-float-checkpoint-before-the-double-quotient
  ;; Returning a Double does not mean that the sum was accumulated in Double.
  ;; These exact Float inputs distinguish the source specialization from an
  ;; accidentally widened accumulation without choosing a parallel tree order.
  (let [prediction (float-array [4096.0 1.0 1.0 1.0])
        target (float-array 4)
        result (loss/mse-loss prediction target 4)
        double-reference (/ (+ 16777216.0 1.0 1.0 1.0) 4.0)
        legal-float-results #{4194304.0 4194304.5 4194305.0}]
    (is (instance? Double result))
    (is (contains? legal-float-results result))
    (is (not= double-reference result))
    (is (Double/isNaN (double (loss/mse-loss (float-array 0) (float-array 0) 0)))
        "an empty reduction still reaches its IEEE Double zero-denominator quotient")))

(deftest unchanged-training-loss-reaches-portable-source-emission
  ;; This checks the public primal entry, not a replacement scalar formula.
  ;; Source emission is intentionally distinct from numerical/device acceptance.
  (let [target :ocl:training-loss-source
        compilation (equation-first/compile
                     #'training/local-loss
                     {:target target :dtype :float :schedule {:precision :f32-scalar}}
                     {:device-id target :device-type :gpu :backend :ocl
                      :subgroup-dialect :opencl-portable :max-workgroup-size 256})]
    (is (= target (:target compilation)))
    (is (= :float (:dtype compilation)))
    (is (some? (:semantic compilation)))
    (is (some? (:emitted compilation)))
    (is (seq (:kernels compilation)))))

(deftest coverage-reports-the-normalized-admission-refusal-without-reanalysis
  (let [attempt (atom nil)
        repeated-analyses (atom 0)
        refusal {:reason :typed-soac-source-coverage :bindings [{:reason :test-refusal}]}
        ;; A pure source-only target: no session, device allocation or vendor
        ;; compiler is necessary to reach semantic admission.
        target :ocl:normalized-coverage
        descriptor {:device-id target :device-type :gpu :backend :ocl
                    :subgroup-dialect :opencl-portable :max-workgroup-size 256}
        ;; Deliberately inject one refusal at the shared admission seam. This
        ;; checks diagnostic transport, not legality of this otherwise legal
        ;; primal. Conditional FP proof content is tested independently above;
        ;; generated gradients now stop earlier at the owned-import boundary.
        error (with-redefs [pipeline/normalized-semantic-program-attempt
                            (fn [form options & _]
                              (reset! attempt {:form form :options options})
                              {:declined refusal})
                           frontend/coverage-decline
                           (fn [& _]
                             (swap! repeated-analyses inc)
                             (throw (AssertionError. "admission refusal was reanalyzed")))]
                (try
                  (equation-first/compile #'training/local-loss
                                          {:target target :dtype :float
                                           :schedule {:precision :f32-scalar}}
                                          descriptor)
                  nil
                  (catch clojure.lang.ExceptionInfo error error)))]
    (is (= :equation-first-coverage (:reason (ex-data error))))
    (is (some? (:form @attempt)))
    (is (identical? refusal (:coverage (ex-data error)))
        "retain the exact refusal returned by the deciding admission attempt")
    (is (zero? @repeated-analyses)
        "do not replace actual admission evidence with a second source analysis")))
