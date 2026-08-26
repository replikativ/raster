(ns raster.gpu.descriptor-fixture
  "Test-only execution fixture for raw resident descriptors.

   Production consumers use raster.gpu.compiled or compose LinkPlans directly. Compiler/device
   tests sometimes begin with an already-produced descriptor; this fixture certifies that
   descriptor through ResidentPlan and executes its LinkedExecutable without recreating the
   removed session program registry."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.resident-plan :as resident-plan]
            [raster.gpu.link :as link]))

(defrecord DescriptorExecutable [lowering executable descriptor roles outputs])

(defn instantiate!
  ([session descriptor arguments]
   (instantiate! session descriptor arguments {} {}))
  ([session descriptor arguments roles]
   (instantiate! session descriptor arguments roles {}))
  ([session descriptor arguments roles opts]
   (let [roles (merge (:array-roles descriptor) roles)
         pointers (link-plan/descriptor-pointer-symbols descriptor)
         outputs (vec (distinct
                       (concat (keep (fn [symbol]
                                       (when (= :output (get roles symbol)) symbol))
                                     (:array-params descriptor))
                               (when-let [result (:result-sym descriptor)]
                                 (when (contains? pointers result) [result])))))
         lowering (resident-plan/lower
                   {:id (random-uuid) :target (:device-id @session)
                    :descriptor descriptor :arguments arguments
                    :roles roles :outputs outputs})
         executable (link/instantiate! (:plan lowering)
                                       (merge {:session session} opts))]
     (->DescriptorExecutable lowering executable descriptor roles outputs))))

(defn- refresh-inputs!
  [{:keys [lowering executable descriptor roles]} arguments]
  (let [argument-map (zipmap (:all-params descriptor) arguments)
        bindings (get-in lowering [:certificate :bindings])]
    (doseq [symbol (:array-params descriptor)
            :when (= :input (get roles symbol :input))]
      (link/upload! executable (get bindings symbol) (get argument-map symbol)))))

(defn results
  [{:keys [lowering executable outputs]}]
  (let [bindings (get-in lowering [:certificate :bindings])]
    (into {} (map (fn [symbol]
                    [symbol (link/download executable (get bindings symbol))]))
          outputs)))

(defn download
  [{:keys [lowering executable]} symbol]
  (link/download executable (get-in lowering [:certificate :bindings symbol])))

(defn run!
  [program arguments]
  (refresh-inputs! program arguments)
  (link/run! (:executable program))
  (results program))

(defn profile!
  [program arguments]
  (refresh-inputs! program arguments)
  (assoc (link/profile! (:executable program)) :result (results program)))

(defn dispatch-arguments
  ([program arguments]
   (link/dispatch-arguments (:executable program) arguments))
  ([program step arguments]
   (link/dispatch-arguments (:executable program) step arguments)))

(defn close!
  [program]
  (link/close! (:executable program)))
