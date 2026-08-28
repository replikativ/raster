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
      (is (= :warp (get-in nv [:execution :subgroup-kind])))
      (is (= #{32} (:subgroup-sizes nv)))
      (is (= 32 (:subgroup-size nv)))
      (is (= 1024 (:max-workgroup-size nv)))
      (is (= {:count 32 :word-bytes 4 :provenance :derived
              :family :cuda-shared-memory}
             (hw/shared-memory-bank-model nv)))
      (is (= 1020 (:grf-bytes-per-lane nv))))
    (testing "AMD CDNA (gfx9): Matrix Cores :mfma 16×16×16 wavefront64; 256 VGPRs×4"
      (is (= {:family :mfma :m 16 :n 16 :k 16 :subgroup 64} (:matrix amd)))
      (is (= :wavefront (get-in amd [:execution :subgroup-kind])))
      (is (= #{64} (:subgroup-sizes amd)))
      (is (= 64 (:subgroup-size amd)))
      (is (= 1024 (:max-workgroup-size amd)))
      (is (= {:count 32 :word-bytes 4 :provenance :derived :family :amd-lds}
             (hw/shared-memory-bank-model amd)))
      (is (= 1024 (:grf-bytes-per-lane amd))))
    (testing "the tile is sized against the vendor register file but CAPPED (not register-filling)"
      ;; both NVIDIA and AMD have ~4× Intel's register file; the cap keeps the warp tile at
      ;; 4 fragments (64), not the raw acc-cap (which would yield 80×80 / 128×128).
      (is (= 64 (:sg-m (hw/derive-gemm-tile nv))))
      (is (= 64 (:sg-m (hw/derive-gemm-tile amd))))
      (is (<= (* (:sg-m (hw/derive-gemm-tile nv)) (:sg-n (hw/derive-gemm-tile nv)))
              (quot (* (:grf-bytes-per-lane nv) 32) 4)) "still fits the register budget"))))

(deftest catalogue-targets-normalize-execution-hierarchy-and-provenance
  (rt/register-target-device! :cuda:a100 {:name "NVIDIA A100"})
  (rt/register-target-device! :ze:arc {:name "Intel(R) Arc(TM) Graphics"})
  (let [a100-device (rt/device :cuda:a100)
        a100 (hw/descriptor-for :cuda:a100)
        arc (hw/descriptor-for :ze:arc)
        tile (hw/derive-gemm-tile a100)
        tile-threads (* (quot (:block-m tile) (:sg-m tile))
                        (quot (:block-n tile) (:sg-n tile))
                        (:subgroup-size a100))]
    (testing "CUDA catalogue keys do not leak Level Zero defaults"
      (is (= :cuda (:backend a100)))
      (is (= #{32} (get-in a100 [:execution :subgroup-sizes])))
      (is (= 32 (get-in a100 [:execution :preferred-subgroup-size])))
      (is (= 1024 (get-in a100 [:execution :max-workgroup-size])))
      (is (<= tile-threads (:max-workgroup-size a100))))
    (testing "catalogue versus user provenance remains visible per capability"
      (is (= :catalogued (get-in a100-device [:source :warp-size])))
      (is (= :catalogued
             (get-in a100 [:execution :provenance :preferred-subgroup-size])))
      (is (= :catalogued
             (get-in a100 [:execution :provenance :max-workgroup-size]))))
    (testing "Intel retains several supported widths and a distinct preference"
      (is (= #{16 32} (get-in arc [:execution :subgroup-sizes])))
      (is (= 16 (get-in arc [:execution :preferred-subgroup-size])))
      (is (= {:count 16 :word-bytes 4 :provenance :derived :family :intel-slm
              :device-dependent? true}
             (hw/shared-memory-bank-model arc)))
      (is (= :subgroup (get-in arc [:execution :subgroup-kind]))))))

(deftest explicit-capabilities-override-catalogue-per-field
  (let [device (rt/register-target-device!
                :cuda:a100-user
                {:name "NVIDIA A100"
                 :capabilities {:max-threads-per-block 512}})
        desc (hw/descriptor-for :cuda:a100-user)]
    (is (= 32 (:subgroup-size desc)))
    (is (= 512 (:max-workgroup-size desc)))
    (is (= :catalogued (get-in device [:source :warp-size])))
    (is (= :user (get-in device [:source :max-threads-per-block])))
    (is (= :user (get-in desc [:execution :provenance :max-workgroup-size])))))

(deftest backend-native-capability-aliases-project-once
  (rt/register-target-device!
   :hip:rdna
   {:type :hip :name "synthetic-rdna"
    :capabilities {:gfx-arch :gfx1100
                   :wavefront-sizes [32 64]
                   :wavefront-size 32
                   :max-work-group-size 1024
                   :local-memory-bytes 65536}})
  (rt/register-target-device!
   :ocl:portable
   {:type :ocl :name "synthetic-opencl"
    :capabilities {:subgroup-sizes [8 16 32]
                   :simd-width 16
                   :max-work-group-size 512
                   :local-memory-bytes 32768}})
  (let [rdna (hw/descriptor-for :hip:rdna)
        opencl (hw/descriptor-for :ocl:portable)]
    (is (= #{32 64} (:subgroup-sizes rdna)))
    (is (= 32 (:subgroup-size rdna)))
    (is (= 1024 (:max-workgroup-size rdna)))
    (is (= 65536 (get-in rdna [:cache :slm])))
    (is (= #{8 16 32} (:subgroup-sizes opencl)))
    (is (= 16 (:subgroup-size opencl)))
    (is (= 512 (:max-workgroup-size opencl)))
    (is (= 32768 (get-in opencl [:cache :slm])))
    (is (nil? (hw/shared-memory-bank-model opencl))
        "generic OpenCL does not inherit Intel/NVIDIA/AMD bank topology")))

(deftest explicit-bank-topology-overrides-derived-vendor-defaults
  (rt/register-target-device!
   :cuda:explicit-banks
   {:type :cuda :name "synthetic-future-nvidia"
    :capabilities {:shared-memory-bank-count 64
                   :shared-memory-bank-word-bytes 8}})
  (let [desc (hw/descriptor-for :cuda:explicit-banks)]
    (is (= {:count 64 :word-bytes 8 :provenance :user}
           (hw/shared-memory-bank-model desc)))
    (is (= :user (get-in desc [:execution :provenance :shared-memory-banks])))))

(deftest authoritative-aliases-override-catalogue-spellings
  (let [device (rt/register-target-device!
                :ze:alias-priority
                {:name "Intel(R) Arc(TM) Graphics"
                 :capabilities {:max-work-group-size 512
                                :local-memory-bytes 32768}})
        desc (hw/descriptor-for :ze:alias-priority)]
    (is (= 512 (:max-workgroup-size desc)))
    (is (= 32768 (get-in desc [:cache :slm])))
    (is (= :user (get-in desc [:execution :provenance :max-workgroup-size])))
    (is (= :user (get-in desc [:execution :provenance :scratchpad-bytes])))
    (is (= :catalogued (get-in device [:source :max-workgroup-size])))
    (is (= :user (get-in device [:source :max-work-group-size])))))

(deftest unknown-opencl-target-keeps-noncollective-capabilities
  (rt/register-target-device!
   :ocl:no-subgroup
   {:name "portable-ocl"
    :capabilities {:max-work-group-size 256
                   :max-compute-units 12
                   :global-mem-bytes 1048576}})
  (let [desc (hw/descriptor-for :ocl:no-subgroup)]
    (is (= #{} (:subgroup-sizes desc)))
    (is (nil? (:subgroup-size desc)))
    (is (nil? (:vector-bits desc)))
    (is (= 256 (:max-workgroup-size desc)))
    (is (= 12 (:compute-units desc)))
    (is (= 1048576 (:global-memory-bytes desc)))))

(deftest matrix-scope-must-be-supported-by-the-execution-hierarchy
  (rt/register-target-device!
   :cuda:bad-matrix
   {:type :cuda :name "bad-matrix-nvidia"
    :capabilities {:compute-capability [8 0]
                   :warp-size 32
                   :matrix {:family :mma :m 16 :n 16 :k 16 :subgroup 64}}})
  (is (= :hardware-matrix-subgroup-unsupported
         (try
           (hw/descriptor-for :cuda:bad-matrix)
           (catch clojure.lang.ExceptionInfo e
             (:reason (ex-data e)))))))

(deftest inferred-matrix-scope-is-validated-after-capability-finalization
  (doseq [[device-id spec]
          [[:cuda:bad-inferred-mma
            {:type :cuda :name "bad-inferred-mma"
             :capabilities {:compute-capability [8 0]
                            :subgroup-sizes [64]}}]
           [:hip:bad-inferred-mfma
            {:type :hip :name "bad-inferred-mfma"
             :capabilities {:gfx-arch :gfx90a
                            :wavefront-sizes [32]
                            :wavefront-size 32}}]]]
    (rt/register-target-device! device-id spec)
    (is (= :hardware-matrix-subgroup-unsupported
           (try
             (hw/descriptor-for device-id)
             (catch clojure.lang.ExceptionInfo e
               (:reason (ex-data e))))))))

(deftest target-identity-and-provenance-are-authoritative
  (is (= :target-device-type-mismatch
         (try
           (rt/register-target-device! :cuda:type-conflict
                                       {:type :hip :name "wrong-family"})
           (catch clojure.lang.ExceptionInfo e
             (:reason (ex-data e))))))
  (let [device (rt/register-target-device!
                :cuda:source-override
                {:name "NVIDIA A100"
                 :source {:warp-size :measured}})]
    (is (= :cuda (:type device)))
    (is (= :catalogued (get-in device [:source :warp-size])))))

(deftest workgroup-derived-matrix-tile-respects-the-thread-limit
  (rt/register-target-device!
   :cuda:small-block
   {:type :cuda :name "small-block-nvidia"
    :capabilities {:compute-capability [8 0]
                   :warp-size 32
                   :max-threads-per-block 256}})
  (let [desc (hw/descriptor-for :cuda:small-block)
        tile (hw/derive-gemm-tile desc)
        threads (* (quot (:block-m tile) (:sg-m tile))
                   (quot (:block-n tile) (:sg-n tile))
                   (:subgroup-size desc))]
    (is (<= threads 256))))

(deftest pre-cdna-gfx9-does-not-acquire-mfma-by-prefix
  (rt/register-target-device!
   :hip:gfx900
   {:type :hip :name "Vega gfx900" :capabilities {:gfx-arch :gfx900}})
  (let [desc (hw/descriptor-for :hip:gfx900)]
    (is (nil? (:matrix desc)))
    (is (nil? (:subgroup-size desc)))))

(deftest equal-authority-alias-conflicts-fail-loudly
  (rt/register-target-device!
   :ocl:alias-conflict
   {:type :ocl :name "conflicting-opencl"
    :capabilities {:max-workgroup-size 256 :max-work-group-size 512
                   :subgroup-sizes [16] :simd-width 16}})
  (is (= :hardware-capability-alias-conflict
         (try
           (hw/descriptor-for :ocl:alias-conflict)
           (catch clojure.lang.ExceptionInfo e
             (:reason (ex-data e)))))))

(deftest contradictory-width-capabilities-fail-loudly
  (rt/register-target-device!
   :cuda:contradictory
   {:type :cuda :name "contradictory-nvidia"
    :capabilities {:compute-capability [8 0]
                   :subgroup-sizes [16]
                   :warp-size 32}})
  (is (= :hardware-preferred-subgroup-unsupported
         (try
           (hw/descriptor-for :cuda:contradictory)
           (catch clojure.lang.ExceptionInfo e
             (:reason (ex-data e)))))))
