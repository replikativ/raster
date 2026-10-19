(ns raster.compiler.backend.gpu.checked-float-stage-test
  "Physical feasibility/semantics check. This does not authorize production
   checked-stage execution or unconditional result initialization."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.kernel-body-fixtures :as fixtures]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emit]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-body-abi :as body-abi]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as level-zero]))

(defn- values []
  (let [limit (double Float/MAX_VALUE)
        subnormal (double Float/MIN_VALUE)
        underflow-midpoint (* 0.5 subnormal)
        midpoint 1.0000000596046448]
    [0.0 -0.0 Double/MIN_VALUE
     (Math/nextDown midpoint) midpoint (Math/nextUp midpoint)
     subnormal (- subnormal)
     (Math/nextDown underflow-midpoint) underflow-midpoint (Math/nextUp underflow-midpoint)
     (- underflow-midpoint) (- (Math/nextUp underflow-midpoint))
     (Math/nextDown limit) limit (Math/nextUp limit)
     (Math/nextUp (- limit)) (- limit) (Math/nextDown (- limit))
     Double/MAX_VALUE (- Double/MAX_VALUE)
     Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY Double/NaN]))

(defn- source-outcome [value]
  (try {:status 0 :bits (Float/floatToRawIntBits (float (double value)))}
       (catch IllegalArgumentException _ {:status 1})))

(defn- compiled [target width]
  (let [body (fixtures/checked-float-stage-body width)
        dialect (case target :ze:0 :opencl-intel :opencl-portable)
        name "checked_float_stage_qualification"]
    (artifact/make
     {:kernel-name name :target :opencl-c
      :source (emit/emit-scalar-kernel name body {:target-dialect dialect
                                                 :parameter-names {'out "rstr_out"}})
      :abi (body-abi/project-contracts
            [(abi/slot 'x :input :double :c-name "rstr_x" :role :input)
             (abi/slot 'out :output :float :c-name "rstr_out" :role :result)
             (abi/slot 'status :output :int :c-name "rstr_status" :role :status)] body)
      :arguments '[x out status] :launch (:launch body)
      :effects {:kind :checked-stage-qualification}
      :provenance {:kernel-body body}})))

(deftest guard-is-expressible-in-every-c-family-dialect
  (doseq [dialect [:opencl-portable :opencl-intel :cuda :hip]]
    (let [source (emit/emit-scalar-kernel "checked_float_guard"
                                        (fixtures/checked-float-stage-body (count (values)))
                                        {:target-dialect dialect})]
      (is (string? source))
      (is (re-find #"if \(" source)))))

(defn- check-device [target]
  (let [xs (values) width (count xs) sentinel (float -555.0)]
    (gpu/with-gpu-session [session target]
      (gpu/alloc! session {:x [:double width (double-array xs)]
                           :out [:float width (float-array (repeat width sentinel))]
                           :status [:int width (int-array (repeat width -1))]})
      (let [handle (gpu/bind-kernel-call! session :checked-stage
                                        (compiled target width) [:x :out :status])]
        (try
          ;; All-failure followed by mixed and all-success catches stale failure
          ;; state. Failed lanes must not perform the conversion/output store.
          (doseq [input [(repeat width Double/MAX_VALUE) xs (repeat width 1.25) xs]]
            (gpu/upload! session :x (double-array input))
            (gpu/upload! session :out (float-array (repeat width sentinel)))
            (gpu/run-kernel-graph! session handle)
            (let [statuses (vec (gpu/download session :status))
                  output (vec (gpu/download session :out))]
              (doseq [[value status result] (map vector input statuses output)]
                (let [expected (source-outcome value)]
                  (is (= (:status expected) status) (str target " " value))
                  (is (= (if (= 1 status) (Float/floatToRawIntBits sentinel) (:bits expected))
                         (Float/floatToRawIntBits result)) (str target " " value))))))
          (finally (gpu/release-kernel-graph! session handle)))))))

(deftest guarded-conversion-and-replay-status-match-source-on-local-devices
  (if @level-zero/gpu-available?
    (check-device :ze:0)
    (level-zero/gpu-skip! "checked Float stage qualification on Level Zero"))
  (if @opencl/opencl-available?
    (check-device :ocl:0)
    (opencl/opencl-skip! "checked Float stage qualification on OpenCL")))
