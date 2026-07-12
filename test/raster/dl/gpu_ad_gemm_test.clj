(ns raster.dl.gpu-ad-gemm-test
  "GPU-AD validation: the matmuls an AD-transformed train step emits — the forward projection
   (linear-nb → dgemm-nt!, :nt) and its two backward gradients (linear-dx → dgemm!, :nn;
   linear-dW → dgemm-tn!, :tn) — each lower through compile-gpu-program to a RESIDENT XMX GEMM
   graph and match the CPU BLAS reference. This is the empirical end of the framework claim
   that AD-transformed IR flows through the GPU pipeline: the backward kernels ARE ordinary
   GEMMs, so recognizing them on the resident path (incl. the :tn weight-gradient variant) is
   what makes gradient computation run on device.

   The full mse∘linear-nb TRAIN STEP does NOT yet fully lower (its loss reduction + the
   daxpy-diff-into! elementwise gradient have no resident kernel in the composed AD path);
   compile-gpu-program returns nil for it (clean staging-fn fallback). That current boundary is
   asserted below so progress flips it — see .internal/ad_gpu_residency.md.

   Level-Zero / Intel-XMX only (the fp16 DPAS GEMM kernel). Skips cleanly without a GPU."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.pipeline :as pl]
            [raster.dl.array-ops :as ops]
            [raster.dl.attention :as attn]
            [raster.arrays :as ra]
            [raster.dl.nn :as nn]))

(def ^:private gpu-available?
  (delay (try (require 'raster.gpu.ze-runtime)
              (boolean (seq ((resolve 'raster.gpu.ze-runtime/query-devices))))
              (catch Throwable _ false))))

(defn- rnd [n seed]
  (let [a (float-array n) r (java.util.Random. seed)]
    (dotimes [i n] (aset a i (float (- (.nextDouble r) 0.5))))
    a))

(defn- rel-err [gpu cpu]
  (let [err (reduce max 0.0 (map #(Math/abs (double (- %1 %2))) (seq gpu) (seq cpu)))
        mag (reduce max 1e-9 (map #(Math/abs (double %)) cpu))]
    (/ err mag)))

(defn- run-resident [f-var args]
  (let [gpu (do (require 'raster.gpu.core) (find-ns 'raster.gpu.core))
        make-session (ns-resolve gpu 'make-session)
        bind-program! (ns-resolve gpu 'bind-program!)
        run-program! (ns-resolve gpu 'run-program!)
        close-session! (ns-resolve gpu 'close-session!)
        p (pl/compile-gpu-program f-var :ze:0 :dtype :float)
        s (make-session :ze:0)]
    (try
      (bind-program! s p args {})
      (let [r (run-program! s p args)]
        {:descriptor p :out (or (get r (:result-sym p)) (first (vals r)))})
      (finally (close-session! s)))))

(deftest gpu-ad-gemm-variants
  (if-not @gpu-available?
    (println "  [SKIP] gpu-ad-gemm: no Level Zero GPU")
    ;; XMX-friendly dims (multiples of 16); batch=16, in-f=32, out-f=16.
    (let [b 16 i 32 o 16
          ;; fp16 XMX GEMM: ~2.5e-4 relative error is the precision floor for these dims.
          tol 5e-3]
      (testing "forward linear-nb (:nt) on GPU matches CPU BLAS"
        (let [x (rnd (* b i) 1) W (rnd (* o i) 2)
              cpu (nn/linear-nb x W b i o)
              {:keys [descriptor out]} (run-resident #'nn/linear-nb [x W b i o])]
          (is (= [:nt] (mapv :variant (:steps descriptor)))
              "forward projection lowers to a single :nt resident GEMM step")
          (is (< (rel-err out cpu) tol)
              (str "forward relerr " (rel-err out cpu)))))
      (testing "backward linear-dx (:nn) on GPU matches CPU BLAS"
        (let [dy (rnd (* b o) 3) W (rnd (* o i) 4)
              cpu (nn/linear-dx dy W b i o)
              {:keys [descriptor out]} (run-resident #'nn/linear-dx [dy W b i o])]
          (is (= [:nn] (mapv :variant (:steps descriptor))))
          (is (< (rel-err out cpu) tol)
              (str "linear-dx relerr " (rel-err out cpu)))))
      (testing "backward linear-dW (:tn, weight gradient) on GPU matches CPU BLAS"
        (let [dy (rnd (* b o) 5) x (rnd (* b i) 6)
              cpu (nn/linear-dW dy x b i o)
              {:keys [descriptor out]} (run-resident #'nn/linear-dW [dy x b i o])]
          (is (= [:tn] (mapv :variant (:steps descriptor)))
              "weight-gradient lowers to a :tn resident GEMM step (transpose-A path)")
          (is (< (rel-err out cpu) tol)
              (str "linear-dW relerr " (rel-err out cpu))))))))

(deftest gpu-attention-layout-ops-resident
  ;; The four dense-attention LAYOUT ops (pack/unpack heads + broadcast/sum KV heads)
  ;; used inside the flat gqa-causal-mha are strided permutation/broadcast/segment-reduce
  ;; copies re-expressed over par/map-void!, so each lowers to a SINGLE resident :map-void
  ;; kernel on device (no raw loop / host-scalar fallback) and matches the CPU reference.
  (if-not @gpu-available?
    (println "  [SKIP] gpu-attention-layout: no Level Zero GPU")
    (let [tol 1e-5]
      (testing "pack-heads lowers to a resident :map-void kernel and matches CPU"
        (let [sl 6 nh 4 hd 8 x (rnd (* sl nh hd) 21)
              cpu (ops/pack-heads x sl nh hd)
              {:keys [descriptor out]} (run-resident #'ops/pack-heads [x sl nh hd])]
          (is (= [:map-void] (mapv :convention (:steps descriptor))))
          (is (< (rel-err out cpu) tol) (str "pack-heads relerr " (rel-err out cpu)))))
      (testing "unpack-heads lowers to a resident :map-void kernel and matches CPU"
        (let [sl 6 nh 4 hd 8 x (rnd (* sl nh hd) 22)
              cpu (ops/unpack-heads x sl nh hd)
              {:keys [descriptor out]} (run-resident #'ops/unpack-heads [x sl nh hd])]
          (is (= [:map-void] (mapv :convention (:steps descriptor))))
          (is (< (rel-err out cpu) tol) (str "unpack-heads relerr " (rel-err out cpu)))))
      (testing "broadcast-kv-heads (MQA fan-out) lowers resident and matches CPU"
        (let [nkv 2 grp 4 slab 48 src (rnd (* nkv slab) 23)
              cpu (ops/broadcast-kv-heads src nkv grp slab)
              {:keys [descriptor out]} (run-resident #'ops/broadcast-kv-heads [src nkv grp slab])]
          (is (= [:map-void] (mapv :convention (:steps descriptor))))
          (is (< (rel-err out cpu) tol) (str "broadcast-kv relerr " (rel-err out cpu)))))
      (testing "sum-kv-heads (fan-in segment reduce) lowers resident and matches CPU"
        (let [nkv 2 grp 4 slab 48 big (rnd (* nkv grp slab) 24)
              cpu (ops/sum-kv-heads big nkv grp slab)
              {:keys [descriptor out]} (run-resident #'ops/sum-kv-heads [big nkv grp slab])]
          (is (= [:map-void] (mapv :convention (:steps descriptor))))
          (is (< (rel-err out cpu) tol) (str "sum-kv relerr " (rel-err out cpu))))))))

;; ── Fused causal SDPA forward (dense-attention-on-GPU Phase 2+3) ─────────────────
;; batched-causal-sdpa's FORWARD is THREE resident par/map-void! kernels — scores
;; (one work item per (b,i,j)), the per-row softmax sweep, and the W·V accumulation
;; (one work item per OUTPUT ELEMENT (b,i,d)) — with NO GEMM and no host scalar-let.
;; (It was one work-item-per-(batch,query-row) kernel until S2a: that shape is only
;; b·seq = 256 work items wide at gemma dims and recomputed each QKᵀ dot three times;
;; the materialized-W shape is 65536 items wide and computes each dot once. The
;; materialized W is also what dQ/dK/dV read, so a fused value+grad program CSEs the
;; forward re-run's W with the backward's.) Its result equals the per-batch
;; GEMM+softmax reference, and gqa-causal-mha (pack→broadcast→fused-sdpa→unpack)
;; compiles to a FULLY resident program. Backward FD gates: decoder-ad-test +
;; backward-kernel-resident-test.

(defn- ref-batched-sdpa
  "Per-batch causal SDPA via the tested single-head causal-scaled-dot-product-attn —
   the pre-fusion GEMM+softmax semantics the fused kernel must match."
  [Q K V batch seq-len hd]
  (let [slab (* seq-len hd) out (float-array (* batch slab))]
    (dotimes [b batch]
      (let [boff (* b slab) Qb (float-array slab) Kb (float-array slab) Vb (float-array slab)]
        (dotimes [i slab]
          (aset Qb i (aget ^floats Q (+ boff i)))
          (aset Kb i (aget ^floats K (+ boff i)))
          (aset Vb i (aget ^floats V (+ boff i))))
        (let [ob (attn/causal-scaled-dot-product-attn Qb Kb Vb seq-len hd hd)]
          (dotimes [i slab] (aset out (+ boff i) (aget ^floats ob i))))))
    out))

(deftest fused-causal-sdpa-forward-equivalence
  ;; CPU-only: the fused single-kernel forward is numerically equivalent to the
  ;; GEMM+softmax reference (GEMM-vs-naive-dot tolerance for f32).
  (let [batch 4 seq-len 6 hd 8 n (* batch seq-len hd)
        Q (rnd n 71) K (rnd n 72) V (rnd n 73)
        fused (attn/batched-causal-sdpa Q K V batch seq-len hd)
        ref   (ref-batched-sdpa Q K V batch seq-len hd)]
    (is (< (rel-err fused ref) 1e-5)
        (str "fused batched-causal-sdpa vs GEMM reference relerr " (rel-err fused ref)))))

(deftest fused-causal-sdpa-resident
  ;; batched-causal-sdpa forward lowers to resident :map-void kernels ONLY (scores /
  ;; row-softmax / W·V accumulation — no GEMM, no host scalar-let) and matches CPU.
  (if-not @gpu-available?
    (println "  [SKIP] fused-causal-sdpa-resident: no Level Zero GPU")
    (let [batch 4 seq-len 6 hd 8 n (* batch seq-len hd)
          Q (rnd n 71) K (rnd n 72) V (rnd n 73)
          cpu (attn/batched-causal-sdpa Q K V batch seq-len hd)
          {:keys [descriptor out]} (run-resident #'attn/batched-causal-sdpa [Q K V batch seq-len hd])]
      (is (= [:map-void :map-void :map-void] (mapv :convention (:steps descriptor)))
          "causal SDPA lowers to the three resident :map-void kernels (no GEMM, no host scalar-let)")
      (is (< (rel-err out cpu) 1e-5) (str "fused-sdpa GPU relerr " (rel-err out cpu))))))

(deftest gqa-causal-mha-forward-fully-resident
  ;; THE Phase-2+3 milestone: the flat gqa-causal-mha forward (pack-heads →
  ;; broadcast-kv-heads → fused batched-causal-sdpa → unpack-heads) compiles to a
  ;; FULLY resident program — every step a :map-void kernel, no non-resident binding —
  ;; and matches the CPU forward on the Arc.
  (if-not @gpu-available?
    (println "  [SKIP] gqa-causal-mha-fully-resident: no Level Zero GPU")
    (let [seq-len 6 nq 4 nkv 1 hd 8
          Q (rnd (* seq-len nq hd) 81) K (rnd (* seq-len nkv hd) 82) V (rnd (* seq-len nkv hd) 83)
          cpu (attn/gqa-causal-mha Q K V seq-len nq nkv hd)
          p (pl/compile-gpu-program #'attn/gqa-causal-mha :ze:0 :dtype :float :on-non-resident :nil)]
      (is (some? p) "gqa-causal-mha forward compiles to a resident descriptor (no non-resident binding)")
      (when p
        (is (every? #(= :map-void %) (mapv :convention (:steps p)))
            "every gqa-causal-mha forward step is a resident :map-void kernel")
        (let [{:keys [out]} (run-resident #'attn/gqa-causal-mha [Q K V seq-len nq nkv hd])]
          (is (< (rel-err out cpu) 1e-5) (str "gqa-causal-mha GPU relerr " (rel-err out cpu))))))))

(deftest gpu-gemm-small-n-scalar-fallback
  ;; XMX GEMM's B-operand 2D-block VNNI read requires Intel 2D-block IO's 16-byte minimum
  ;; pitch (N*2 bytes at fp16) — output-column dims N<8 violated it and produced garbage
  ;; (relerr ~1). bind-program! now routes N<8 :gemm steps to a plain scalar f32 kernel
  ;; (no f16 convert/transpose). n=8 stays on the XMX path (16-byte pitch = the minimum)
  ;; as the boundary control.
  (if-not @gpu-available?
    (println "  [SKIP] gpu-gemm-small-n: no Level Zero GPU")
    (let [tol 5e-3]
      (doseq [n [2 4 6 8]]
        (testing (str "forward linear-nb (:nt) at out-features n=" n)
          (let [b 16 i 32 o n
                x (rnd (* b i) (+ 100 n)) W (rnd (* o i) (+ 200 n))
                cpu (nn/linear-nb x W b i o)
                {:keys [descriptor out]} (run-resident #'nn/linear-nb [x W b i o])]
            (is (= [:nt] (mapv :variant (:steps descriptor))))
            (is (< (rel-err out cpu) tol)
                (str ":nt n=" n " relerr " (rel-err out cpu)))))
        (testing (str "backward linear-dx (:nn) at in-features n=" n)
          (let [b 16 o 32 i n
                dy (rnd (* b o) (+ 300 n)) W (rnd (* o i) (+ 400 n))
                cpu (nn/linear-dx dy W b i o)
                {:keys [descriptor out]} (run-resident #'nn/linear-dx [dy W b i o])]
            (is (= [:nn] (mapv :variant (:steps descriptor))))
            (is (< (rel-err out cpu) tol)
                (str ":nn n=" n " relerr " (rel-err out cpu)))))))))

(deftest gpu-reduce-step-program-binds-and-replays
  ;; bind-program! (the resident whole-offload binder) wired only :map/:map-void/:gemm/
  ;; :scatter step kinds; a program containing a par/reduce whose result stays resident
  ;; (the #42 resident-reduce work: fuse-reduce-results gives it a device 1-elem out-buf)
  ;; threw at bind time even though bind-step! already handled :reduce. This pins the
  ;; wiring: sum(a) feeding an elementwise scale compiles to [:reduce :map] steps, binds,
  ;; and replays with the correct output. (An ESCAPING tail reduce — a loss returned as
  ;; the scalar result — still doesn't bind: that is the remaining #42 leftover, same as
  ;; bind-step!.)
  (if-not @gpu-available?
    (println "  [SKIP] gpu-reduce-step: no Level Zero GPU")
    (let [probe (eval '(raster.core/deftm gpu-reduce-consumer-probe
                         [a :- (Array float) out :- (Array float) n :- Long] :- Void
                         (let [s (raster.par/reduce
                                  acc 0.0 i n
                                  (raster.numeric/+ acc (raster.arrays/aget a i)))]
                           (raster.par/map-void!
                            j n
                            (raster.arrays/aset out j
                                                (raster.numeric/* (raster.arrays/aget a j) s))))))
          n 64
          a (rnd n 51)
          out (float-array n)
          s (double (reduce + 0.0 (map double (seq a))))
          cpu (float-array (map #(float (* (double %) s)) (seq a)))
          {:keys [descriptor] :as r} (run-resident probe [a out n])
          gpu (:out r)]
      (is (some #(= :reduce (:convention %)) (:steps descriptor))
          "resident-consumed par/reduce compiles to a :reduce step")
      (is (< (rel-err gpu cpu) 1e-4)
          (str "reduce→map-void replay relerr " (rel-err gpu cpu))))))

(deftest gpu-ad-full-train-step-fully-resident
  ;; This test does NOT need a GPU — it pins residency at the IR level. The full
  ;; mse∘linear-nb float train step (value+grad + SGD, returning the updated weights like
  ;; finetune's lora-train-step) is AD-transformed and EVERY piece — the matmuls (resident
  ;; GEMM steps), the fused loss-reduction chain, and the elementwise gradient/SGD kernels —
  ;; lowers to a resident program descriptor. This used to assert the OPPOSITE
  ;; (compile-gpu-program → nil) while the loss reduction had no resident kernel; the #42
  ;; par/reduce residency work flipped it. A regression back to nil means some step fell
  ;; off the resident path (staging-fn fallback = the old gap).
  ;;
  ;; NOTE the remaining #42 leftover: a train step whose RESULT is the loss scalar still
  ;; does NOT extract (the primal mse reduce ESCAPES to the host, so it can't fuse —
  ;; :scalar-let-reads-device-buffer). The resident train-step convention is therefore
  ;; "return the updated weights, monitor loss on host" (same as lora-train-step).
  (require 'raster.core 'raster.dl.loss 'raster.dl.optim 'raster.ad.reverse 'raster.arrays)
  (let [loss (eval '(raster.core/deftm gpu-ad-probe-loss
                      [W :- (Array float) x :- (Array float) tgt :- (Array float)
                       batch :- Long in-f :- Long out-f :- Long] :- Double
                      (let [pred (raster.dl.nn/linear-nb x W batch in-f out-f)]
                        (raster.dl.loss/mse-loss pred tgt (clojure.core/* batch out-f)))))
        train (eval '(raster.core/deftm gpu-ad-probe-train
                       [W :- (Array float) x :- (Array float) tgt :- (Array float)
                        batch :- Long in-f :- Long out-f :- Long lr :- Double] :- (Array float)
                       (let [vg ((raster.ad.reverse/value+grad #'gpu-ad-probe-loss)
                                 W x tgt batch in-f out-f)
                             dW (clojure.core/nth vg 1)]
                         (raster.dl.optim/sgd-step! W dW (raster.arrays/alength W) lr)
                         W)))]
    (testing "full AD train step compiles to a fully-resident program (pins residency)"
      ;; :on-non-resident :nil = probe the boundary without throwing (default is :throw now)
      (let [p (pl/compile-gpu-program train :ze:0 :dtype :float :on-non-resident :nil)]
        (is (some? p)
            "mse∘linear train step must extract fully resident (matmuls + fused loss + SGD)")
        (when p
          (is (seq (:steps p)) "resident descriptor carries kernel steps"))))))
