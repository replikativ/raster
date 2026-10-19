(ns raster.compiler.passes.parallel.checked-map-stage-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.equation-artifact :as artifact]
            [raster.compiler.fixtures.mixed-storage :as storage]
            [raster.compiler.ir.emitted-parallel-equation :as emitted]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.stage-result :as result]
            [raster.compiler.passes.parallel.checked-map-stage :as checked]
            [raster.compiler.passes.parallel.resident-program-projection :as resident]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.hardware-fixture :as fixture]
            [raster.runtime.hardware :as hardware]))

(def targets [:ocl:checked-stage-test :cuda:checked-stage-test :hip:checked-stage-test])

(use-fixtures :each fixture/isolated
  (fn [f]
    (doseq [target targets]
      (hardware/register-target-device!
       target {:name "Checked stage source-only target"
               :capabilities {:compute-capability [8 0] :gfx-arch :gfx1100
                              :subgroup-sizes [32] :max-workgroup-size 1024
                              :total-eus 32 :shared-local-memory 65536}}))
    (f)))

(defn- compilation [target]
  (equation-first/compile #'storage/mixed-scale
                         (merge storage/policy {:target target :dtype :double})))

(defn- numerical-equation [compilation]
  (->> (get-in compilation [:emitted :equations])
       (remove #(get-in % [:attributes :host-only])) first :operations first))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest real-functional-map-retains-checked-semantics-and-private-observation
  (doseq [[target module] (map vector targets [:opencl-c :cuda-c :hip-cpp])]
    (let [compilation (compilation target)
          equation (numerical-equation compilation)
          report (emitted/validate-with-result-contracts equation)
          obligation (first (:stage-results report))
          graph (:graph equation)]
      (is (= 1 (count (:kernels compilation))))
      (is (= module (:target (first (:kernels compilation)))))
      (is (= checked/kind (:kind obligation)))
      (is (= :trap (get-in obligation [:source-conversion :overflow])))
      (is (= :conditional-on-success (:initialization obligation)))
      (is (= [:int :double] (mapv :dtype (:temporaries graph))))
      (is (empty? (:complete-write-domains report)))
      (is (result/requires-observation? compilation))
      (is (= (:stage-results report)
             (:stage-results (emitted/validate-with-result-contracts equation)))))))

(deftest physical-refinement-does-not-authorize-unobserved-consumers
  (let [compilation (compilation (first targets))
        equation (numerical-equation compilation)
        graph (:graph equation)
        kernel (first (:kernels compilation))
        calls (atom 0)
        arguments [2.0 (float-array [1.5 -2.0])]]
    (with-redefs [gpu/make-session (fn [& _] (swap! calls inc) (throw (AssertionError. "driver reached")))]
      (doseq [attempt [#(equation-first/lower compilation arguments)
                       #(link/instantiate-ordered-invocation! compilation arguments {})
                       #(resident/project (:emitted compilation) (first targets))
                       #(gpu/bind-kernel-graph! (atom nil) :checked graph {} {})
                       #(gpu/bind-kernel-call! (atom nil) :checked kernel [])
                       #(call/make kernel [] {:resolve-value (fn [_] (swap! calls inc))})
                       #(call/validate! (call/->KernelCall kernel [] nil))
                       #(#'artifact/validate-compilation! compilation nil)
                       #(link-plan/validate! {:instances [{:graph graph}]})]]
        (is (= :checked-stage-observation-required (reason attempt))))
      (is (zero? @calls)))))

(deftest checked-refinement-is-reconstructed-not-trusted-from-its-label
  (let [equation (numerical-equation (compilation (first targets)))
        source (equation-graph/make (:algorithm equation) (:body equation))
        witness (:refinement equation)]
    (doseq [kind [:ordinary-schedule :mixed-precision-contraction nil]]
      (is (= :checked-map-stage-refinement
             (reason #(emitted/validate-with-result-contracts
                       (assoc-in equation [:refinement :schedule :kind] kind))))))
    (is (= :checked-map-stage-refinement
           (reason #(emitted/validate-with-result-contracts
                     (assoc equation :refinement nil)))))
    ;; Neither a label nor removing the physical marker can erase the source
    ;; conversion's obligation at the independent emitted boundary.
    (is (= :checked-map-stage-refinement
           (reason #(emitted/validate-with-result-contracts
                     (-> equation
                         (assoc-in [:refinement :schedule :kind] :ordinary-schedule)
                         (update-in [:refinement :graph :attributes] dissoc :checked-stage-results)
                         (update-in [:graph :attributes] dissoc :checked-stage-results))))))
    (is (= :checked-map-stage-refinement
           (reason #(checked/validate-reconstruction!
                     (:algorithm equation) source
                     (assoc-in witness [:graph :attributes :checked-stage-results 0 :initialization]
                               :unconditional) {}))))
    (is (= :checked-map-stage-refinement
           (reason #(checked/validate-reconstruction!
                     (:algorithm equation) source
                     (assoc-in witness [:graph :temporaries 0 :elements] 1) {}))))))

(deftest malformed-or-partially-projected-obligations-still-deny
  (doseq [value [{:attributes {:checked-stage-result nil}}
                 {:attributes {:checked-stage-results []}}
                 {:attributes {:kernel-body {:attributes {:checked-stage-result nil}}}}
                 {:provenance {:scheduled-operation {:attributes {:checked-stage-result nil}}}}]]
    (is (result/requires-observation? value))
    (is (= :checked-stage-observation-required
           (reason #(result/refuse-unobserved! value :test)))))
  (is (false? (result/requires-observation? {:attributes {:ordinary true}}))))

(deftest checked-stage-trigger-does-not-capture-other-conversion-families
  (let [equation (numerical-equation (compilation (first targets)))
        source (equation-graph/make (:algorithm equation) (:body equation))
        path [:nodes 0 :operation :scalar-region :result]
        conversion (get-in source path)]
    (is (checked/candidate? (:algorithm equation) source))
    (doseq [attributes [{:source-dtype :long :target-dtype :int
                         :rounding :toward-zero :overflow :trap
                         :source-op 'clojure.core/int}
                        {:source-dtype :double :target-dtype :float
                         :rounding :nearest-even :overflow :ieee
                         :source-op 'clojure.core/unchecked-float}]]
      (is (not (checked/candidate?
                (:algorithm equation)
                (assoc-in source path (list (first conversion) attributes (nth conversion 2)))))))))
