(ns native-lifetime
  "Shared construction/rollback ownership for low-level benchmark recordings."
  (:require [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.ze-runtime :as ze]))

(def ^:dynamic *retain-kernel!* nil)

(defn bind!
  "Retain the fresh kernel before argument/grid setup. Roll back setup failure."
  [module kernel-name bind]
  (let [kernel (ze/create-kernel-fresh module kernel-name)
        owner (::cleanup/owner kernel)]
    (when *retain-kernel!* (*retain-kernel!* kernel))
    (cleanup/build! owner #(bind (:handle kernel))
                    (when *retain-kernel!* (fn [_])))))

(defn record!
  "Reserve graph-before-kernel teardown before building even the first bound call.
   Failed prefixes and unknown recording acquisition remain reachable through the thrown owner."
  [build-calls options]
  (let [kernels (atom {}) recordings (atom {})
        retain! (fn [registry value] (swap! registry assoc (random-uuid) value))
        owner (cleanup/owner
               [{:id :recordings
                 :release #(cleanup/release-entries! recordings (vec @recordings)
                                                     (fn [graph] (cleanup/release! (::cleanup/owner graph))))}
                {:id :kernels :after #{:recordings}
                 :release #(cleanup/release-entries! kernels (vec @kernels) ze/destroy-kernel!)}])]
    (cleanup/build! owner
      #(binding [*retain-kernel!* (partial retain! kernels)]
         (let [calls (build-calls)
               graph (ze/record-graph! calls
                       (assoc options :adopt-cleanup!
                              (fn [cleanup] (retain! recordings {::cleanup/owner cleanup}))))]
           (retain! recordings graph)
           graph))
      nil)))
