(ns raster.gpu.block-transfer-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.resident-plan :as resident-plan]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :as raster :refer [deftm]]
            [raster.dl.array-ops :as ops]
            [raster.dl.gpu-grad-parity :as gpu-probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.descriptor-fixture :as fixture]
            [raster.gpu.device-probe :as opencl-probe]))

(deftm block-roundtrip!
  [src :- (Array float), indices :- (Array int), paged :- (Array float),
   restored :- (Array float), nblocks :- Long, block-width :- Long,
   paged-blocks :- Long] :- Void
  (ops/scatter-blocks! src indices paged nblocks block-width paged-blocks)
  (ops/gather-blocks! paged indices restored nblocks block-width paged-blocks))

(deftm block-roundtrip-half!
  [src :- (Array short), indices :- (Array int), paged :- (Array short),
   restored :- (Array short), nblocks :- Long, block-width :- Long,
   paged-blocks :- Long] :- Void
  (ops/scatter-blocks! src indices paged nblocks block-width paged-blocks)
  (ops/gather-blocks! paged indices restored nblocks block-width paged-blocks))

(defn- storage-array
  [dtype values]
  (case dtype
    :float (float-array (map float values))
    :half (short-array (map #(Float/floatToFloat16 (float %)) values))))

(defn- run-roundtrip!
  [device-id dtype]
  (let [nblocks 4
        block-width 7
        paged-blocks 9
        source (storage-array dtype (map #(- (float %) 13.0)
                                         (range (* nblocks block-width))))
        indices (int-array [7 1 5 3])
        paged (storage-array dtype (repeat (* paged-blocks block-width) 0.0))
        restored (storage-array dtype (repeat (* nblocks block-width) 0.0))
        ;; FP16 uses a raw short[] carrier while scalar/index arithmetic remains FP32.
        semantic-var (raster/resolve-deftm-var
                      (if (= :half dtype) #'block-roundtrip-half! #'block-roundtrip!)
                      {:dtype :float})
        descriptor (pipeline/compile-gpu-program semantic-var device-id :dtype :float)
        session (gpu/make-session device-id)]
    (ops/validate-block-transfer! :scatter source indices paged
                                  nblocks block-width paged-blocks)
    (ops/validate-block-transfer! :gather paged indices restored
                                  nblocks block-width paged-blocks)
    (try
      (let [program (fixture/instantiate!
                     session descriptor
                     [source indices paged restored nblocks block-width paged-blocks])]
        (try
          (is (resident-plan/certified-plan? (:lowering program)))
          (is (= [:map-void :map-void] (mapv :convention (:steps descriptor))))
          (is (some #{dtype} (mapcat #(map :dtype (:abi %)) (:steps descriptor))))
          (is (= (vec source)
                 (vec (get (fixture/run!
                            program
                            [source indices paged restored nblocks block-width paged-blocks])
                           'restored))))
          (finally
            (fixture/close! program))))
      (finally
        (gpu/close-session! session)))))

(deftest level-zero-resident-block-transfer-roundtrip
  (if-not @gpu-probe/gpu-available?
    (gpu-probe/gpu-skip! "resident block scatter/gather on Level Zero")
    (do
      (run-roundtrip! :ze:0 :float)
      (run-roundtrip! :ze:0 :half))))

(deftest opencl-resident-block-transfer-roundtrip
  (if-not @opencl-probe/opencl-available?
    (opencl-probe/opencl-skip! "resident block scatter/gather on OpenCL")
    (do
      (run-roundtrip! :ocl:0 :float)
      (if @opencl-probe/opencl-fp16-available?
        (run-roundtrip! :ocl:0 :half)
        (opencl-probe/opencl-skip! "FP16 resident block scatter/gather on OpenCL" :fp16)))))
