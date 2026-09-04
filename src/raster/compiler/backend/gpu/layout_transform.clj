(ns raster.compiler.backend.gpu.layout-transform
  "Lower target-neutral affine layout-transform schedules to C-family kernel artifacts."
  (:require [raster.compiler.backend.gpu.kernel-body-opencl :as emitter]
            [raster.compiler.passes.parallel.layout-transform-schedule :as schedule]))

(defn emit-cast-kernel
  [{:keys [kernel-name input output source-dtype destination-dtype vector-width rounding overflow
           target-dialect]
    :or {vector-width 1 target-dialect :opencl-intel}}]
  (let [kernel-body (schedule/cast-body
                     {:id [:layout-transform kernel-name]
                      :input input :output output
                      :source-dtype source-dtype :destination-dtype destination-dtype
                      :vector-width vector-width
                      :rounding rounding :overflow overflow})]
    {:source (emitter/emit-scalar-kernel
              kernel-name kernel-body
              {:target-dialect target-dialect
               :parameter-names {input "in" output "out" :layout-elements "n"}})
     :kernel-body kernel-body}))

(defn emit-transpose-kernel
  [{:keys [kernel-name input output element-dtype target-dialect]
    :or {target-dialect :opencl-intel}}]
  (let [kernel-body (schedule/transpose-body
                     {:id [:layout-transform kernel-name]
                      :input input :output output
                      :element-dtype element-dtype})]
    {:source (emitter/emit-scalar-kernel
              kernel-name kernel-body
              {:target-dialect target-dialect
               :parameter-names {input "in" output "out"
                                 :layout-rows "rows" :layout-cols "cols"}})
     :kernel-body kernel-body}))
