(ns raster.gpu.paged-kv-append-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.paged-kv-append :as append]
            [raster.compiler.passes.parallel.paged-kv-append-route :as route]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as device-probe]))

(defn- half-bits
  [value]
  (Float/floatToFloat16 (float value)))

(defn- run-case
  [device-id]
  (let [batch-size 2
        key-width 4
        value-width 3
        page-size 2
        physical-pages 3
        physical-slots (* page-size physical-pages)
        key-rows (float-array [1.1 -2.2 3.3 -4.4, 5.5 -6.6 7.7 -8.8])
        value-rows (float-array [0.25 -0.5 0.75, 1.25 -1.5 1.75])
        slots (int-array [5 -1])
        sentinel (half-bits 42.0)
        key-pages (short-array (repeat (* physical-slots key-width) sentinel))
        value-pages (short-array (repeat (* physical-slots value-width) sentinel))
        problem (append/make
                 {:id [:device device-id]
                  :key-rows 'key-rows :value-rows 'value-rows
                  :slot-mapping 'slots :key-pages 'key-pages :value-pages 'value-pages
                  :batch-size batch-size :key-elements-per-token key-width
                  :value-elements-per-token value-width
                  :page-size page-size :physical-pages physical-pages})
        graph (:graph (route/route!
                       problem {:device-type :gpu :subgroup-size 16
                                :max-workgroup-size 256}))]
    (append/validate-slot-values! problem slots)
    (gpu/with-gpu-session [session device-id]
      (gpu/alloc! session
                  {:key-rows [:float (alength key-rows) key-rows]
                   :value-rows [:float (alength value-rows) value-rows]
                   :slots [:int (alength slots) slots]
                   :key-pages [:half (alength key-pages) key-pages :state]
                   :value-pages [:half (alength value-pages) value-pages :state]})
      (let [handle (gpu/bind-kernel-graph!
                    session :paged-kv-append graph
                    {'key-rows :key-rows 'value-rows :value-rows 'slots :slots
                     'key-pages :key-pages 'value-pages :value-pages}
                    {})]
        (try
          (gpu/run-kernel-graph! session handle)
          (let [actual-key ^shorts (gpu/download session :key-pages)
                actual-value ^shorts (gpu/download session :value-pages)]
            (doseq [slot (range physical-slots)
                    component (range key-width)]
              (let [index (+ (* slot key-width) component)
                    lane ({5 0} slot)
                    expected (if lane
                               (half-bits (aget key-rows (+ (* lane key-width) component)))
                               sentinel)]
                (is (= expected (aget actual-key index)))))
            (doseq [slot (range physical-slots)
                    component (range value-width)]
              (let [index (+ (* slot value-width) component)
                    lane ({5 0} slot)
                    expected (if lane
                               (half-bits (aget value-rows (+ (* lane value-width) component)))
                               sentinel)]
                (is (= expected (aget actual-value index))))))
          (finally
            (gpu/release-kernel-graph! session handle)))))))

(deftest level-zero-paged-kv-append-assigns-rte-halfs
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "paged K/V assignment on Level Zero")
    (run-case :ze:0)))

(deftest opencl-paged-kv-append-assigns-rte-halfs
  (if-not @device-probe/opencl-fp16-available?
    (device-probe/opencl-skip! "paged K/V assignment" :fp16)
    (run-case :ocl:0)))
