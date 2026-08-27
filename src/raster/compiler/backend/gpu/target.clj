(ns raster.compiler.backend.gpu.target
  "Backend-emitter capability predicates.

   Hardware descriptors state what a device can execute. These predicates state which concrete
   source dialect the currently selected backend leaf can emit; schedule-neutral compiler passes
   must not use them to constrain otherwise-portable schedules."
  (:require [clojure.string :as str]))

(defn intel-opencl-subgroup-dialect?
  "True when the current OpenCL leaf may emit Intel subgroup attributes and builtins.

   `:subgroup-dialect` is the explicit cross-compile contract. Vendor and DPAS checks retain the
   detected/catalogued Intel path until runtime backend capabilities become a first-class facet."
  [desc]
  (let [vendor (some-> (:vendor desc) str str/lower-case)
        matrix-family (get-in desc [:matrix :family])
        backend (:backend desc)
        opencl-backend? (or (nil? backend)
                            (= :ze backend)
                            (= :ocl backend)
                            (= :opencl backend))]
    (boolean
     (and desc
          (= :gpu (:device-type desc))
          opencl-backend?
          (or (= :intel-opencl (:subgroup-dialect desc))
              (= :dpas matrix-family)
              (and vendor (str/includes? vendor "intel")))))))
