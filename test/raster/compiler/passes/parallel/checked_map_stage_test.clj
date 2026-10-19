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
            [raster.runtime.hardware :as hardware]
            [raster.runtime.checked-stage-result :as observation]))

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

(deftest completed-status-protocol-is-bounded-and-empty-safe
  (let [equation (numerical-equation (compilation (first targets)))
        {:keys [extent status]} (first (:stage-results (emitted/validate-with-result-contracts equation)))]
    (is (symbol? extent))
    (doseq [length [0 1 16385]]
      (let [reads (atom [])]
        (is (nil? (observation/observe!
                   equation {extent length}
                   (fn [{:keys [elements] :as request}]
                     (swap! reads conj request)
                     (int-array elements)))))
        (is (= length (reduce + 0 (map :elements @reads))))
        (is (every? #(and (= status (:buffer %)) (= :int (:dtype %))
                          (<= 1 (:elements %) 16384)) @reads))
        (is (= (if (zero? length) [] [0])
               (vec (take 1 (map :offset @reads)))))
        (is (= (if (> length 16384) [0 16384] (if (zero? length) [] [0]))
               (mapv :offset @reads)))))
    (let [reads (atom 0)]
      (is (= :checked-stage-observation-protocol
             (reason #(observation/observe! equation {extent -1}
                                            (fn [_] (swap! reads inc))))))
      (is (zero? @reads)))))

(defn- thrown-description [f]
  (try (f) nil (catch Throwable error [(class error) (.getMessage error)])))

(deftest completed-failure-selects-first-source-lane-and-exact-diagnostic
  (let [equation (numerical-equation (compilation (first targets)))
        {:keys [extent status diagnostics]}
        (first (:stage-results (emitted/validate-with-result-contracts equation)))]
    (doseq [value [(Math/nextUp (double Float/MAX_VALUE))
                   (- (Math/nextUp (double Float/MAX_VALUE)))
                   Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY]]
      (let [reads (atom [])
            actual (thrown-description
                    #(observation/observe!
                      equation {extent 4}
                      (fn [request]
                        (swap! reads conj request)
                        (if (= status (:buffer request))
                          (int-array [0 1 0 1])
                          (double-array [value])))))
            expected (thrown-description #(float (double value)))]
        (is (some? expected))
        (is (= expected actual))
        (is (= [{:buffer status :dtype :int :offset 0 :elements 4}
                {:buffer diagnostics :dtype :double :offset 1 :elements 1}] @reads))))))

(deftest malformed-completed-status-protocol-never-credits-success
  (let [equation (numerical-equation (compilation (first targets)))
        {:keys [extent status]} (first (:stage-results (emitted/validate-with-result-contracts equation)))]
    (doseq [snapshot [(long-array [0]) (int-array 0) (int-array [2]) (int-array [-1])]]
      (is (= :checked-stage-observation-protocol
             (reason #(observation/observe! equation {extent 1} (constantly snapshot))))))
    (doseq [diagnostic [(float-array [Float/POSITIVE_INFINITY]) (double-array 0)
                        (double-array [0.0]) (double-array [Double/NaN])]]
      (is (= :checked-stage-observation-protocol
             (reason #(observation/observe!
                       equation {extent 1}
                       (fn [request]
                         (if (= status (:buffer request)) (int-array [1]) diagnostic)))))))
    (let [reads (atom 0)]
      (is (= :checked-map-stage-refinement
             (reason #(observation/observe!
                       (assoc-in equation [:refinement :schedule :kind] :ordinary)
                       {extent 1} (fn [_] (swap! reads inc))))))
      (is (zero? @reads)))))

(deftest first-failure-in-later-chunk-keeps-global-lane-order
  (let [equation (numerical-equation (compilation (first targets)))
        {:keys [extent status diagnostics]}
        (first (:stage-results (emitted/validate-with-result-contracts equation)))
        reads (atom [])
        failure (thrown-description
                 #(observation/observe!
                   equation {extent 40000}
                   (fn [{:keys [buffer offset elements] :as request}]
                     (swap! reads conj request)
                     (if (= buffer status)
                       (let [snapshot (int-array elements)]
                         (when (= offset 16384) (aset-int snapshot 7 1))
                         snapshot)
                       (double-array [Double/POSITIVE_INFINITY])))))]
    (is (= (thrown-description #(float Double/POSITIVE_INFINITY)) failure))
    (is (= [{:buffer status :dtype :int :offset 0 :elements 16384}
            {:buffer status :dtype :int :offset 16384 :elements 16384}
            {:buffer diagnostics :dtype :double :offset 16391 :elements 1}] @reads))))

(deftest binding-validation-scope-is-exact-thread-local-and-nonescaping
  (let [equation (numerical-equation (compilation (first targets)))
        graph (:graph equation)
        artifact (get-in graph [:nodes 0 :operation])
        escaped
        (result/call-with-checked-binding-validation
         equation {}
         (fn [_]
           (is (identical? artifact (result/refuse-unobserved! artifact :kernel-call)))
           (is (= :checked-stage-observation-required
                  (reason #(result/refuse-unobserved! (assoc artifact :unrelated true) :kernel-call))))
           (is (= :checked-stage-observation-required
                  (reason #(result/refuse-unobserved! graph :bind-kernel-graph))))
           (is (= :checked-stage-observation-required
                  @(future (reason #(result/refuse-unobserved! artifact :kernel-call)))))
           (bound-fn [] (result/refuse-unobserved! artifact :kernel-call))))]
    (is (= :checked-stage-observation-required (reason escaped)))
    (is (= :checked-stage-observation-required
           (reason #(result/refuse-unobserved! artifact :kernel-call))))
    (let [captured (atom nil)]
      (is (= :injected-scope-failure
             (reason #(result/call-with-checked-binding-validation
                       equation {}
                       (fn [_]
                         (reset! captured (bound-fn [] (result/refuse-unobserved! artifact :kernel-call)))
                         (throw (ex-info "scope failure" {:reason :injected-scope-failure})))))))
      (is (= :checked-stage-observation-required (reason @captured))))))

(defn- pending-entry-request [compilation equation]
  (let [graph (:graph equation)
        numerical (first (remove #(get-in % [:attributes :host-only])
                                 (get-in compilation [:emitted :equations])))
        input (:id (first (:inputs graph)))
        output (:id (first (:outputs graph)))
        target (:target compilation)]
    {:id :pending-checked-entry :target target
     :nodes [(link-plan/node {:id :input :role :input :dtype :float :shape [4] :device target})
             (link-plan/node {:id :output :role :output :dtype :float :shape [4] :device target})]
     :instances [(link-plan/graph-instance
                  {:id :checked :graph graph :bindings {input :input output :output}
                   :scalar-values (into {} (map (fn [{:keys [id dtype]}]
                                                 [id {:type dtype :value (case dtype :long 4 :double 2.0)}]))
                                        (:scalars graph))
                   :attributes {:emitted-equation-source
                                {:program (:emitted compilation) :equation-id (:id numerical)}}})]
     :outputs [:output]}))

(deftest checked-write-coverage-remains-conditional-in-an-ordered-entry
  (let [compilation (compilation (first targets))
        equation (numerical-equation compilation)
        request (pending-entry-request compilation equation)
        pending (link-plan/make-pending-observed-entry equation request)
        contract (:entry-contract pending)
        layout (:layout pending)
        facts (vec (#'link-plan/instance-access-facts layout))]
    (is (= #{:input} (:requires contract)))
    (is (= #{} (:full-produces contract)))
    (is (= #{:output} (:success-produces contract)))
    (is (= (:stage-results (emitted/validate-with-result-contracts equation))
           (:observations contract)))
    (is (= :link-unobserved-result-condition
           (reason #(#'link-plan/analyze-effects! layout facts))))
    (is (= #{:input :output}
           (:requires (#'link-plan/relative-entry-contract
                       (:nodes layout)
                       (conj facts {:facts [{:node :output :access :read}]
                                    :complete-writes #{}}))))
        "a following consumer cannot use a merely conditionally written view")
    (is (identical? pending (link-plan/validate-pending-observed-entry! pending)))
    (is (= :checked-stage-observation-required
           (reason #(link-plan/validate! (:layout pending)))))
    (is (= :checked-stage-observation-required
           (reason #(link/instantiate! (:layout pending)))))
    (is (= :link-pending-observation-contract
           (reason #(link-plan/validate-pending-observed-entry!
                     (assoc-in pending [:entry-contract :full-produces] #{:output})))))
    (is (= :link-pending-observation-contract
           (reason #(link-plan/validate-pending-observed-entry!
                     (assoc-in pending [:entry-contract :observations] [])))))
    (doseq [[bad-layout expected]
            [[(assoc-in layout [:nodes :output :source] (float-array 4)) :link-pending-observation-contract]
             [(assoc-in layout [:nodes :output :role] :state) :link-pending-observation-contract]
             [(assoc-in layout [:nodes :output :view :allocation :ownership] :borrowed)
              :link-value-abstract-ownership]]]
      (is (= expected (reason #(link-plan/make-pending-observed-entry equation bad-layout)))))
    (let [extent (:extent (first (:observations contract)))
          output (:output (first (:observations contract)))]
      (doseq [bad-request
              [(assoc-in request [:instances 0 :attributes :emitted-equation-source :program] {})
               (assoc-in request [:instances 0 :scalar-values extent :value] 5)
               (assoc-in request [:instances 0 :bindings output] :input)
               (assoc-in request [:instances 0 :attributes] {})]]
        (is (thrown? clojure.lang.ExceptionInfo
                     (link-plan/make-pending-observed-entry equation bad-request)))))))
