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
    (is (= {:block-m 128 :block-n 128 :sg-m 32 :sg-n 32 :block-k 32 :num-stages 3
            :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}}
           (hw/derive-gemm-tile arc-desc)))))

(deftest num-stages-is-slm-bounded
  (testing "the pipeline depth cap is SLM-capacity / K-panel bytes (T4)"
    (let [arc (assoc arc-desc :shared-local-memory 131072)   ;; Arc 140V: 128 KB SLM
          tile (hw/derive-gemm-tile arc)]
      ;; one stage = (128×32 + 32×128)×2 B = 16384 B; 131072/16384 = 8
      (is (= 8 (hw/max-stages-for arc tile)))
      (is (>= (hw/max-stages-for arc tile) 2) "always ≥ 2 (single/double buffer floor)"))
    (testing "a tiny-SLM device caps the pipeline shallower"
      (let [tiny (assoc arc-desc :shared-local-memory 16384)
            tile (hw/derive-gemm-tile tiny)]
        (is (< (hw/max-stages-for tiny tile) 8))))
    (testing "candidates never exceed the SLM bound"
      (doseq [t (hw/gemm-tile-candidates (assoc arc-desc :shared-local-memory 131072))]
        (is (<= (long (:num-stages t 3)) (hw/max-stages-for (assoc arc-desc :shared-local-memory 131072) t)))))))

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
          src (apply cg/emit-gemm-tiled "gemm_derived" :prefetch (:num-stages tile)
                     (mapcat identity (dissoc tile :num-stages)))]
      (is (string? src))
      (is (.contains src "intel_sub_group_f16_f16_matrix_mad_k16"))
      (is (.contains src "acc31") "the 32×32/8×16 tile has 4×2 accumulators (acc00..acc31)"))))
