(ns raster.compiler.backend.gpu.layout-transform
  "Lower target-neutral affine layout-transform schedules to C-family kernel artifacts."
  (:require [raster.compiler.backend.gpu.kernel-body-opencl :as emitter]
            [raster.compiler.passes.parallel.layout-transform-schedule :as schedule]))

(defn cast-body
  "Build the target-neutral body for one dense representation conversion."
  [{:keys [id kernel-name input output source-dtype destination-dtype vector-width rounding overflow]
    :or {vector-width 1}}]
  (schedule/cast-body
   {:id (or id [:layout-transform kernel-name])
    :input input :output output
    :source-dtype source-dtype :destination-dtype destination-dtype
    :vector-width vector-width
    :rounding rounding :overflow overflow}))

(defn transpose-body
  "Build the target-neutral body for one dense row-major matrix transpose."
  [{:keys [id kernel-name input output element-dtype]}]
  (schedule/transpose-body
   {:id (or id [:layout-transform kernel-name])
    :input input :output output
    :element-dtype element-dtype}))

(defn emit-cast-kernel
  [{:keys [kernel-name input output source-dtype destination-dtype vector-width rounding overflow
           target-dialect]
    :or {vector-width 1 target-dialect :opencl-intel}}]
  (let [kernel-body (cast-body
                     {:kernel-name kernel-name :input input :output output
                      :source-dtype source-dtype :destination-dtype destination-dtype
                      :vector-width vector-width :rounding rounding :overflow overflow})]
    {:source (emitter/emit-scalar-kernel
              kernel-name kernel-body
              {:target-dialect target-dialect
               :parameter-names {input "in" output "out" :layout-elements "n"}})
     :kernel-body kernel-body}))

(defn emit-transpose-kernel
  [{:keys [kernel-name input output element-dtype target-dialect]
    :or {target-dialect :opencl-intel}}]
  (let [kernel-body (transpose-body
                     {:kernel-name kernel-name :input input :output output
                      :element-dtype element-dtype})]
    {:source (emitter/emit-scalar-kernel
              kernel-name kernel-body
              {:target-dialect target-dialect
               :parameter-names {input "in" output "out"
                                 :layout-rows "rows" :layout-cols "cols"}})
     :kernel-body kernel-body}))
