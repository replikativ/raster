(ns raster.gpu.runtime-backend
  "The supported resident GPU runtime backends. Compiler targets are broader: CUDA
   and HIP can emit code without being registered here as executable sessions."
  (:require [clojure.string :as str]))

(def ^:private resident-backends
  {:ze {:namespace 'raster.gpu.ze-runtime
        :kernel-body-c-dialect :opencl-portable
        :memory-space :shared
        :coherence :host-coherent
        :owned-slice? false}
   :ocl {:namespace 'raster.gpu.ocl-runtime
         :kernel-body-c-dialect :opencl-portable
         :memory-space :device
         :coherence :explicit-transfer
         :owned-slice? true}})

(defn backend-type
  "Return the resident backend of a device ID, rejecting compile-only targets."
  [device-id]
  (let [s (when (keyword? device-id) (name device-id))
        backend (cond
                  (and s (str/starts-with? s "ze")) :ze
                  (and s (str/starts-with? s "ocl")) :ocl)]
    (if (and backend (contains? resident-backends backend))
      backend
      (throw (ex-info "No resident GPU runtime for device; use :ze:N or :ocl:N"
                      {:device-id device-id
                       :supported-backends (vec (sort (keys resident-backends)))
                       :compile-only? (boolean (and s (re-find #"^(cuda|hip):" s)))})))))

(defn descriptor
  "Return runtime facts used by the session and value layers."
  [device-id]
  (get resident-backends (backend-type device-id)))

(defn runtime-namespace
  [device-id]
  (:namespace (descriptor device-id)))
