(ns raster.gpu.one-tile-source-test
  "ONE tile source. Device-free.

   `derive-gemm-tile` used to reach nothing that shipped: the two emit front doors took
   `emit-gemm-tiled`'s own `:or` literals, the contraction door derived from an EMPTY descriptor (so
   Arc constants on every device), and three launch sites computed their grid from a hardcoded
   `/128.0` with a `*32` K-unroll. Five independent spellings of one number — and a kernel emitted
   with one tile but launched with geometry derived from another is a silent wrong-answer path the
   moment they diverge.

   The property these tests pin is that unifying them changed NOTHING on this device: generalizing
   the tile must not cost peak. `gemm-tile-for` on the default descriptor must reproduce
   `emit-gemm-tiled`'s literals exactly, and the split-k policy must produce the same schedule it
   produced from its own constants."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.hardware :as hw]))

(deftest default-tile-equals-the-emitters-own-literals
  (testing "gemm-tile-for on no descriptor == emit-gemm-tiled's :or defaults, so adopting the one
            source is a NO-OP on this device — de-hardcoding, not a behaviour change"
    (let [t (hw/gemm-tile-for nil)]
      (is (= 128 (:block-m t)))
      (is (= 128 (:block-n t)))
      (is (= 32 (:sg-m t)))
      (is (= 32 (:sg-n t)))
      (is (= 32 (:block-k t)))
      (is (= 16 (get-in t [:matrix :subgroup])))))
  (testing "…and an empty map is the same as nil (the contraction door used to pass `{}`)"
    (is (= (hw/gemm-tile-for nil) (hw/gemm-tile-for {})))))

(deftest a-different-part-gets-a-rescaled-tile
  (testing "the whole point of a derivation: a part with a smaller GRF budget and subgroup 8 gets a
            correctly rescaled tile instead of Arc's constants"
    (let [t (hw/gemm-tile-for {:matrix {:m 8 :n 8 :k 16 :subgroup 8}
                               :grf-bytes-per-lane 128})]
      (is (= 8 (get-in t [:matrix :subgroup])))
      (is (< (:sg-m t) 32) "per-subgroup tile shrinks with the register budget")
      (is (= 0 (mod (:block-m t) (:sg-m t))) "workgroup block stays divisible by the subgroup tile")
      (is (= 0 (mod (:sg-m t) (get-in t [:matrix :m])))
          "and the subgroup tile stays a whole number of matrix fragments"))))

(deftest emitted-kernel-and-launch-tile-must-agree
  ;; THE invariant. A kernel emitted with one tile and launched with geometry derived from another
  ;; leaves the tail of C unwritten — silent garbage. The previous version of this test
  ;; re-implemented `ceil(dim/block)` in the test body and asserted ceil(256/128)=2, touching no
  ;; production code; it passed while exactly that divergence was live. This parses the tile back
  ;; out of the emitter's own header line and compares it with the tile the geometry comes from.
  (let [emit (requiring-resolve 'raster.compiler.backend.gpu.opencl-codegen/emit-gemm-tiled)
        header-tile (fn [src]
                      (when-let [[_ bm bn sm sn bk]
                                 (re-find #"WG (\d+)x(\d+), SG (\d+)x(\d+), K (\d+)" src)]
                        {:block-m (parse-long bm) :block-n (parse-long bn)
                         :sg-m (parse-long sm) :sg-n (parse-long sn) :block-k (parse-long bk)}))
        keys' [:block-m :block-n :sg-m :sg-n :block-k]]
    (testing "the DEFAULT emit (no tile args) matches the default derived tile — so any door that
              emits without a tile may use the derived tile for its grid"
      (is (= (select-keys (hw/gemm-tile-for nil) keys')
             (header-tile (emit "probe" :c-dtype :half)))))
    (testing "and when a tile IS passed, the emitted header reflects THAT tile — which is what makes
              emitting and launching from one tile safe on a non-default part"
      (let [t (hw/gemm-tile-for {:matrix {:m 8 :n 8 :k 16 :subgroup 8} :grf-bytes-per-lane 128})
            src (apply emit "probe2" (concat [:c-dtype :half]
                                             (mapcat identity (select-keys t keys'))
                                             [:matrix (:matrix t)]))]
        (is (= (select-keys t keys') (header-tile src))
            "a rescaled tile must reach the kernel text, not just the grid")))
    (testing "the two differ — so a door that emits default-tiled but launches device-derived is
              the silent-garbage path this test exists to forbid"
      (is (not= (select-keys (hw/gemm-tile-for nil) keys')
                (select-keys (hw/gemm-tile-for {:matrix {:m 8 :n 8 :k 16 :subgroup 8}
                                                :grf-bytes-per-lane 128}) keys'))))))

(deftest split-k-policy-is-unchanged-on-this-device
  (testing "gemm-schedule now reads block-m/block-n/block-k instead of /128.0 and *32, and must
            produce the same schedule it did from its own constants"
    (let [sched (requiring-resolve 'raster.gpu.core/gemm-schedule)]
      (is (= [1 1024] (sched 128 128 1024 64)) "already fills the machine → no split")
      (is (= [8 1024] (sched 64 64 8192 64)) "small tile-count, long K → split-k")
      (testing "and every K-chunk is a multiple of the tile's K-unroll, by construction"
        (let [[_ kc] (sched 64 64 8192 64)]
          (is (zero? (mod (long kc) (long (:block-k (hw/gemm-tile-for nil)))))))))))
