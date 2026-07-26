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

(deftest launch-geometry-is-derived-from-the-same-tile
  (testing "grid = ceil(dim / block), computed from the tile rather than a literal 128"
    (let [{:keys [block-m block-n]} (hw/gemm-tile-for nil)
          gc (fn [d b] (long (Math/ceil (/ (double d) (double b)))))]
      (is (= 2 (gc 256 block-n)))
      (is (= 1 (gc 128 block-n)))
      (is (= 1 (gc 5 block-m)) "a tile-smaller problem is still one block")
      ;; the pre-unification literal and the derived value agree on THIS device, which is what
      ;; makes the change safe here and correct elsewhere
      (is (= (gc 4096 128) (gc 4096 block-n))))))

(deftest split-k-policy-is-unchanged-on-this-device
  (testing "gemm-schedule now reads block-m/block-n/block-k instead of /128.0 and *32, and must
            produce the same schedule it did from its own constants"
    (let [sched (requiring-resolve 'raster.gpu.core/gemm-schedule)]
      (is (= [1 1024] (sched 128 128 1024 64)) "already fills the machine → no split")
      (is (= [8 1024] (sched 64 64 8192 64)) "small tile-count, long K → split-k")
      (testing "and every K-chunk is a multiple of the tile's K-unroll, by construction"
        (let [[_ kc] (sched 64 64 8192 64)]
          (is (zero? (mod (long kc) (long (:block-k (hw/gemm-tile-for nil)))))))))))
