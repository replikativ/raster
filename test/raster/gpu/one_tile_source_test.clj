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
            [raster.compiler.backend.gpu.gemm :as gemm]
            [raster.compiler.backend.gpu.opencl-codegen :as opencl-codegen]
            [raster.compiler.core.hardware :as hw]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]))

(defn- split-schedule
  [m n k fill-workgroups]
  (let [tile (hw/gemm-tile-for nil)
        requested (launch/resolve-expression
                   {} (gemm/requested-splits
                       {:m m :n n :k k :tile tile :fill-workgroups fill-workgroups}))]
    (if (< requested 2)
      [1 k]
      (let [kc (launch/resolve-expression
                {} (launch/align-up (launch/ceil-div k requested) (:block-k tile)))]
        [(launch/resolve-expression {} (launch/ceil-div k kc)) kc]))))

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

(deftest resident-direct-and-autotune-sources-use-the-scheduled-body
  (let [emit-resident (do (require 'raster.gpu.ze-runtime)
                          (ns-resolve 'raster.gpu.ze-runtime 'emit-scheduled-gemm))
        tile (hw/gemm-tile-for nil)]
    (with-redefs [opencl-codegen/emit-gemm-tiled
                  (fn [& _]
                    (throw (ex-info "legacy template was called" {:reason :test/failure})))]
      (let [emitted (emit-resident "resident_body" :float tile nil)]
        (is (body/kernel-body? (:kernel-body emitted)))
        (is (= :float (:dtype (first (filter #(= :result (:role %))
                                             (get-in emitted [:kernel-body :parameters]))))))
        (is (= (get-in emitted [:kernel-body :launch :workgroup-size])
               (:workgroup-size emitted)))
        (is (re-find #"__global float\* restrict C" (:source emitted)))))))

(deftest resident-epilogues-use-typed-programs-and-body-owned-abi
  (let [ze (do (require 'raster.gpu.ze-runtime) (find-ns 'raster.gpu.ze-runtime))
        emit-resident (ns-resolve ze 'emit-scheduled-gemm)
        resident-program (ns-resolve ze 'resident-epilogue-program)
        tile (hw/gemm-tile-for nil)
        program {:acc 'acc
                 :expr '(raster.numeric/+ acc (aget bias j))
                 :operands [{:sym 'bias
                             :map (axis-map/of-axes [['j 'N]])
                             :dtype :half}]}
        descriptor (assoc program :bindings {'bias :resident-buffer})]
    (with-redefs [opencl-codegen/emit-gemm-tiled
                  (fn [& _]
                    (throw (ex-info "legacy template was called" {:reason :test/failure})))]
      (let [emitted (emit-resident "resident_typed_epilogue" :float tile program)]
        (is (= program (resident-program descriptor)))
        (is (= [['bias :input :half :epilogue]]
               (mapv (juxt :id :kind :dtype :role)
                     (filter #(= :epilogue (:role %))
                             (get-in emitted [:kernel-body :parameters])))))
        (is (re-find #"restrict bias" (:source emitted)))
        (is (re-find #"bias\[[^]]*col" (:source emitted)))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"unsupported fields"
                          (resident-program {:key :bias :fn identity :params "raw"
                                             :bindings {}})))))

(deftest resident-grid-z-sources-use-the-scheduled-body
  (let [ze (do (require 'raster.gpu.ze-runtime) (find-ns 'raster.gpu.ze-runtime))
        emit-split (ns-resolve ze 'emit-scheduled-split-k-gemm)
        emit-batched (ns-resolve ze 'emit-scheduled-batched-gemm)
        tile (hw/gemm-tile-for nil)]
    (with-redefs [opencl-codegen/emit-gemm-tiled
                  (fn [& _]
                    (throw (ex-info "legacy template was called" {:reason :test/failure})))]
      (let [split (emit-split "resident_split_body" tile)
            batched (emit-batched "resident_batched_body" tile)]
        (is (= [256 1 1] (:workgroup-size split)))
        (is (= 1 (count (get-in split [:kernel-body :views]))))
        (is (re-find #"int KC, int splits" (:source split)))
        (is (= [256 1 1] (:workgroup-size batched)))
        (is (= 3 (count (get-in batched [:kernel-body :views]))))
        (is (re-find #"int batch" (:source batched)))))))

(deftest split-k-policy-is-unchanged-on-this-device
  (testing "the emitted schedule expression reads block-m/block-n/block-k rather than independent
            literals and preserves the established policy"
    (let [sched split-schedule]
      (is (= [1 1024] (sched 128 128 1024 64)) "short K does not amortize a split")
      (is (= [8 1024] (sched 64 64 8192 64)) "small tile-count, long K → split-k")
      (testing "and every K-chunk is a multiple of the tile's K-unroll, by construction"
        (let [[_ kc] (sched 64 64 8192 64)]
          (is (zero? (mod (long kc) (long (:block-k (hw/gemm-tile-for nil)))))))))))
