(ns raster.compiler.core.gemm-tile-test
  "T2: the GEMM tile is DERIVED from the hardware descriptor's matrix unit + GRF budget, not
   hardcoded. Device-free — synthetic descriptors in, tile maps out."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.hardware :as hw]
            [raster.compiler.backend.gpu.opencl-codegen :as cg]))

(def arc-desc
  "Synthetic Arc 140V descriptor (the catalogue facts the tile derivation reads)."
  {:matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}
   :grf-bytes-per-lane 256 :subgroup-size 16})

(deftest arc-derives-the-hand-tuned-tile
  (testing "the Arc descriptor derives EXACTLY the T1 hand-tuned default (no magic numbers)"
    (is (= {:block-m 128 :block-n 128 :sg-m 32 :sg-n 32 :block-k 32
            :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}}
           (hw/derive-gemm-tile arc-desc)))))

(deftest tile-is-grf-bounded
  (testing "the per-subgroup accumulator tile fits the GRF budget (sg-m·sg-n/subgroup·4 ≤ grf/lane)"
    (doseq [desc [arc-desc
                  (assoc arc-desc :grf-bytes-per-lane 512)      ;; a bigger-GRF part
                  (assoc-in arc-desc [:matrix :subgroup] 32)]]  ;; wider subgroup
      (let [{:keys [sg-m sg-n matrix]} (hw/derive-gemm-tile desc)
            sg (:subgroup matrix)
            acc-bytes-per-lane (* (/ (* sg-m sg-n) sg) 4)]
        (is (<= acc-bytes-per-lane (:grf-bytes-per-lane desc))
            (str "acc " acc-bytes-per-lane "B/lane must fit GRF " (:grf-bytes-per-lane desc) "B/lane"))
        (is (zero? (mod sg-m (:m matrix))) "sg-m is a DPAS M_i multiple")
        (is (zero? (mod sg-n (:n matrix))) "sg-n is a DPAS N_i multiple")))))

(deftest a-bigger-grf-part-gets-a-bigger-tile
  (testing "doubling the GRF budget grows the accumulator tile (the derivation actually scales)"
    (let [small (hw/derive-gemm-tile arc-desc)
          big   (hw/derive-gemm-tile (assoc arc-desc :grf-bytes-per-lane 1024))]
      (is (> (* (:sg-m big) (:sg-n big)) (* (:sg-m small) (:sg-n small)))))))

(deftest derived-tile-feeds-the-generator
  (testing "the derived tile is a valid emit-gemm-tiled argument map (round-trips to a kernel)"
    (let [tile (hw/derive-gemm-tile arc-desc)
          src (apply cg/emit-gemm-tiled "gemm_derived" (mapcat identity tile))]
      (is (string? src))
      (is (.contains src "intel_sub_group_f16_f16_matrix_mad_k16"))
      (is (.contains src "acc31") "the 32×32/8×16 tile has 4×2 accumulators (acc00..acc31)"))))
