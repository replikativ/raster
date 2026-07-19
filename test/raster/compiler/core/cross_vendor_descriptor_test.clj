(ns raster.compiler.core.cross-vendor-descriptor-test
  "Cross-vendor descriptor derivation (device-free): the matrix unit + register-file budget are
   derived per vendor from probed caps, so derive-gemm-tile sizes the tile against the RIGHT
   register file (Intel/NVIDIA/AMD differ ~4×). Uses register-target-device! with synthetic caps —
   no GPU needed."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.hardware :as hw]
            [raster.runtime.hardware :as rt]))

(deftest per-vendor-matrix-and-register-budget
  (rt/init!)
  (rt/register-target-device! :cuda:test
    {:type :cuda :name "synthetic-nvidia" :capabilities {:compute-capability [8 0] :warp-size 32 :total-eus 64}})
  (rt/register-target-device! :hip:test
    {:type :hip :name "synthetic-amd" :capabilities {:gfx-arch :gfx90a :total-eus 64}})
  (let [nv  (hw/descriptor-for :cuda:test)
        amd (hw/descriptor-for :hip:test)]
    (testing "NVIDIA: compute-capability ≥7 → WMMA :mma 16×16×16 warp32; register file 255×4"
      (is (= {:family :mma :m 16 :n 16 :k 16 :subgroup 32} (:matrix nv)))
      (is (= 1020 (:grf-bytes-per-lane nv))))
    (testing "AMD CDNA (gfx9): Matrix Cores :mfma 16×16×16 wavefront64; 256 VGPRs×4"
      (is (= {:family :mfma :m 16 :n 16 :k 16 :subgroup 64} (:matrix amd)))
      (is (= 1024 (:grf-bytes-per-lane amd))))
    (testing "the tile is sized against the vendor register file but CAPPED (not register-filling)"
      ;; both NVIDIA and AMD have ~4× Intel's register file; the cap keeps the warp tile at
      ;; 4 fragments (64), not the raw acc-cap (which would yield 80×80 / 128×128).
      (is (= 64 (:sg-m (hw/derive-gemm-tile nv))))
      (is (= 64 (:sg-m (hw/derive-gemm-tile amd))))
      (is (<= (* (:sg-m (hw/derive-gemm-tile nv)) (:sg-n (hw/derive-gemm-tile nv)))
              (quot (* (:grf-bytes-per-lane nv) 32) 4)) "still fits the register budget"))))
