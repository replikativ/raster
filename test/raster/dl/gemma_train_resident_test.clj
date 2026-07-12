(ns raster.dl.gemma-train-resident-test
  "MILESTONE (gemma GPU trainability, step 1): a RESIDENT TRAIN-STEP for the full
   gemma LoRA decoder block — forward + reverse-AD backward + SGD adapter update in
   ONE resident GPU program, adapters updated on-device, loss decreasing over steps.

   The block here is a raster-side TWIN of finetune.gemma's gemma-block at tiny dims
   (same op graph: rms-norm sandwich, LoRA q/k/v/o + GeGLU gate/up/down, per-head
   qk-norm, rope, gqa-causal-mha, residuals), concrete (Array float) with the LoRA
   low-rank delta spelled inline (3 linear-nb + residual-add — the flattened
   lora-linear shape that keeps the body raster-templated-ops-only, see the B1 note
   in finetune gemma_f32_grad_test). The real-weights run is finetune-side (step 2).

   Train-step architecture (the M1 lora-train-step pattern, finetune lora_gpu.clj):
   value+grad is bound DIRECTLY to its own symbol (the AD transform only fires on a
   directly-bound call), the 14 LoRA adapter grads are nth'd out of the result
   vector, and raster.dl.optim/sgd-step! folds `p := p - lr*dp` in place per adapter
   — all ordinary deftm/par ops, so the whole fwd+bwd+update extracts as ONE
   resident program via compile-gpu-program. The primal loss (nth vg 0) is unused
   and DCE'd; loss is monitored host-side from the downloaded adapter state (the
   adapters are tiny). Adapters bind :state (resident, updated on-device, never
   re-uploaded); frozen weights / norms / data bind :constant.

   GATES:
     • the train step extracts FULLY RESIDENT under :gemm-precision :f32-scalar
       (exact-f32 grads — the training GEMM policy);
     • 25 on-device steps: loss decreases, adapters changed on-device;
     • a CPU-interpreted reference loop (same value+grad + sgd-step!, same lr/data)
       tracks the GPU loss trajectory within f32 tolerance."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.dl.nn :as nn]
            [raster.dl.attention :as attn]
            [raster.dl.loss :as loss]
            [raster.dl.optim :as optim]
            [raster.arrays :as ra]
            [raster.ad.reverse :as rev]
            [raster.compiler.pipeline :as pl]
            [raster.dl.gpu-grad-parity :as gp]))

;; ── the twin block (concrete float, LoRA delta inline) ──────────────────────────

;; Flattened LoRA linear: frozen base + trainable low-rank delta B·(A·x), spelled
;; as raster.dl.nn ops only (no nested parametric user deftm — the B1-safe shape).
(deftm lora-lin
  [x :- (Array float) W :- (Array float) A :- (Array float) B :- (Array float)
   rows :- Long in :- Long r :- Long out :- Long] :- (Array float)
  (let [ax   (raster.dl.nn/linear-nb x A rows in r)
        bax  (raster.dl.nn/linear-nb ax B rows r out)
        base (raster.dl.nn/linear-nb x W rows in out)]
    (raster.dl.nn/residual-add base bax (clojure.core/* rows out))))

;; One gemma decoder block (twin of finetune.gemma/gemma-block): x:[seq,d] → [seq,d].
(deftm gblk
  [x :- (Array float)
   input-ln :- (Array float) q-norm :- (Array float) k-norm :- (Array float)
   post-attn :- (Array float) pre-ffn :- (Array float) post-ffn :- (Array float)
   Wq :- (Array float) Aq :- (Array float) Bq :- (Array float)
   Wk :- (Array float) Ak :- (Array float) Bk :- (Array float)
   Wv :- (Array float) Av :- (Array float) Bv :- (Array float)
   Wo :- (Array float) Ao :- (Array float) Bo :- (Array float)
   Wg :- (Array float) Ag :- (Array float) Bg :- (Array float)
   Wu :- (Array float) Au :- (Array float) Bu :- (Array float)
   Wd :- (Array float) Ad :- (Array float) Bd :- (Array float)
   seq :- Long d :- Long nq :- Long nkv :- Long hd :- Long dff :- Long r :- Long
   eps :- Double theta :- Double] :- (Array float)
  (let [nqh (clojure.core/* nq hd) nkh (clojure.core/* nkv hd) n (clojure.core/* seq d)
        ;; qk-norm row counts pre-bound to locals: a source-level binding carries the
        ;; TC Long stamp, so rms-norm's inlined alloc size devirtualizes in the
        ;; FORWARD-only GPU program too (finetune.train/gblock's step-2 note).
        snq (clojure.core/* seq nq) snkv (clojure.core/* seq nkv)
        h  (nn/rms-norm x input-ln seq d eps 1.0)
        q  (raster.dl.gemma-train-resident-test/lora-lin h Wq Aq Bq seq d r nqh)
        k  (raster.dl.gemma-train-resident-test/lora-lin h Wk Ak Bk seq d r nkh)
        v  (raster.dl.gemma-train-resident-test/lora-lin h Wv Av Bv seq d r nkh)
        q  (nn/rms-norm q q-norm snq hd eps 1.0)
        k  (nn/rms-norm k k-norm snkv hd eps 1.0)
        q  (attn/rope q seq nq hd theta)
        k  (attn/rope k seq nkv hd theta)
        a  (attn/gqa-causal-mha q k v seq nq nkv hd)
        o  (raster.dl.gemma-train-resident-test/lora-lin a Wo Ao Bo seq nqh r d)
        o  (nn/rms-norm o post-attn seq d eps 1.0)
        x1 (nn/residual-add x o n)
        h2 (nn/rms-norm x1 pre-ffn seq d eps 1.0)
        g  (raster.dl.gemma-train-resident-test/lora-lin h2 Wg Ag Bg seq d r dff)
        g  (nn/gelu g (clojure.core/* seq dff))
        u  (raster.dl.gemma-train-resident-test/lora-lin h2 Wu Au Bu seq d r dff)
        hh (nn/hadamard g u (clojure.core/* seq dff))
        m  (raster.dl.gemma-train-resident-test/lora-lin hh Wd Ad Bd seq dff r d)
        m  (nn/rms-norm m post-ffn seq d eps 1.0)]
    (nn/residual-add x1 m n)))

;; mse loss over the block. Adapter-grad slots in value+grad's result vector
;; (arg j → vg j+1): Aq 8→9, Bq 9→10, Ak 11→12, Bk 12→13, Av 14→15, Bv 15→16,
;; Ao 17→18, Bo 18→19, Ag 20→21, Bg 21→22, Au 23→24, Bu 24→25, Ad 26→27, Bd 27→28.
(deftm gblk-loss
  [x :- (Array float)
   input-ln :- (Array float) q-norm :- (Array float) k-norm :- (Array float)
   post-attn :- (Array float) pre-ffn :- (Array float) post-ffn :- (Array float)
   Wq :- (Array float) Aq :- (Array float) Bq :- (Array float)
   Wk :- (Array float) Ak :- (Array float) Bk :- (Array float)
   Wv :- (Array float) Av :- (Array float) Bv :- (Array float)
   Wo :- (Array float) Ao :- (Array float) Bo :- (Array float)
   Wg :- (Array float) Ag :- (Array float) Bg :- (Array float)
   Wu :- (Array float) Au :- (Array float) Bu :- (Array float)
   Wd :- (Array float) Ad :- (Array float) Bd :- (Array float)
   tgt :- (Array float)
   seq :- Long d :- Long nq :- Long nkv :- Long hd :- Long dff :- Long r :- Long
   eps :- Double theta :- Double] :- Double
  (let [y (raster.dl.gemma-train-resident-test/gblk x
                                                    input-ln q-norm k-norm post-attn pre-ffn post-ffn
                                                    Wq Aq Bq Wk Ak Bk Wv Av Bv Wo Ao Bo Wg Ag Bg Wu Au Bu Wd Ad Bd
                                                    seq d nq nkv hd dff r eps theta)]
    (raster.dl.loss/mse-loss y tgt (clojure.core/* seq d))))

;; ── the RESIDENT TRAIN STEP ──────────────────────────────────────────────────────
;; value+grad (fwd+bwd of the whole block) + 14 in-place SGD updates, one program.
;; Returns Aq (an array result the resident extractor accepts); the loss primal is
;; dead here and DCE'd — loss is monitored host-side from the adapter state.
(deftm gblk-train-step
  [Aq :- (Array float) Bq :- (Array float) Ak :- (Array float) Bk :- (Array float)
   Av :- (Array float) Bv :- (Array float) Ao :- (Array float) Bo :- (Array float)
   Ag :- (Array float) Bg :- (Array float) Au :- (Array float) Bu :- (Array float)
   Ad :- (Array float) Bd :- (Array float)
   x :- (Array float)
   input-ln :- (Array float) q-norm :- (Array float) k-norm :- (Array float)
   post-attn :- (Array float) pre-ffn :- (Array float) post-ffn :- (Array float)
   Wq :- (Array float) Wk :- (Array float) Wv :- (Array float) Wo :- (Array float)
   Wg :- (Array float) Wu :- (Array float) Wd :- (Array float)
   tgt :- (Array float)
   seq :- Long d :- Long nq :- Long nkv :- Long hd :- Long dff :- Long r :- Long
   eps :- Double theta :- Double lr :- Double] :- (Array float)
  (let [vg ((raster.ad.reverse/value+grad
             (var raster.dl.gemma-train-resident-test/gblk-loss))
            x input-ln q-norm k-norm post-attn pre-ffn post-ffn
            Wq Aq Bq Wk Ak Bk Wv Av Bv Wo Ao Bo Wg Ag Bg Wu Au Bu Wd Ad Bd
            tgt seq d nq nkv hd dff r eps theta)
        dAq (clojure.core/nth vg 9)  dBq (clojure.core/nth vg 10)
        dAk (clojure.core/nth vg 12) dBk (clojure.core/nth vg 13)
        dAv (clojure.core/nth vg 15) dBv (clojure.core/nth vg 16)
        dAo (clojure.core/nth vg 18) dBo (clojure.core/nth vg 19)
        dAg (clojure.core/nth vg 21) dBg (clojure.core/nth vg 22)
        dAu (clojure.core/nth vg 24) dBu (clojure.core/nth vg 25)
        dAd (clojure.core/nth vg 27) dBd (clojure.core/nth vg 28)]
    (raster.dl.optim/sgd-step! Aq dAq (raster.arrays/alength Aq) lr)
    (raster.dl.optim/sgd-step! Bq dBq (raster.arrays/alength Bq) lr)
    (raster.dl.optim/sgd-step! Ak dAk (raster.arrays/alength Ak) lr)
    (raster.dl.optim/sgd-step! Bk dBk (raster.arrays/alength Bk) lr)
    (raster.dl.optim/sgd-step! Av dAv (raster.arrays/alength Av) lr)
    (raster.dl.optim/sgd-step! Bv dBv (raster.arrays/alength Bv) lr)
    (raster.dl.optim/sgd-step! Ao dAo (raster.arrays/alength Ao) lr)
    (raster.dl.optim/sgd-step! Bo dBo (raster.arrays/alength Bo) lr)
    (raster.dl.optim/sgd-step! Ag dAg (raster.arrays/alength Ag) lr)
    (raster.dl.optim/sgd-step! Bg dBg (raster.arrays/alength Bg) lr)
    (raster.dl.optim/sgd-step! Au dAu (raster.arrays/alength Au) lr)
    (raster.dl.optim/sgd-step! Bu dBu (raster.arrays/alength Bu) lr)
    (raster.dl.optim/sgd-step! Ad dAd (raster.arrays/alength Ad) lr)
    (raster.dl.optim/sgd-step! Bd dBd (raster.arrays/alength Bd) lr)
    Aq))

;; ── data / config ────────────────────────────────────────────────────────────────

(def CFG {:seq 4 :d 8 :nq 2 :nkv 1 :hd 4 :dff 16 :r 4 :eps 1.0e-6 :theta 10000.0})

(defn- fa
  "Deterministic gaussian float array."
  ^floats [n seed scale]
  (let [rng (java.util.Random. (long seed)) a (float-array n)]
    (dotimes [i n] (aset a i (float (* (double scale) (.nextGaussian rng))))) a))

(def ^:private adapter-syms
  '[Aq Bq Ak Bk Av Bv Ao Bo Ag Bg Au Bu Ad Bd])

(defn- init-state
  "All arrays for one training problem, keyed by param sym. Adapters random
   (scale 0.2 — nonzero so every adapter grad is live from step 0)."
  [{:keys [seq d nq nkv hd dff r]}]
  (let [nqh (* nq hd) nkh (* nkv hd) f (fn [n s] (fa n s 0.2))]
    {'x  (f (* seq d) 700)
     'input-ln (f d 1) 'q-norm (f hd 2) 'k-norm (f hd 3)
     'post-attn (f d 4) 'pre-ffn (f d 5) 'post-ffn (f d 6)
     'Wq (f (* nqh d) 7)  'Aq (f (* r d) 8)   'Bq (f (* nqh r) 9)
     'Wk (f (* nkh d) 10) 'Ak (f (* r d) 11)  'Bk (f (* nkh r) 12)
     'Wv (f (* nkh d) 13) 'Av (f (* r d) 14)  'Bv (f (* nkh r) 15)
     'Wo (f (* d nqh) 16) 'Ao (f (* r nqh) 17) 'Bo (f (* d r) 18)
     'Wg (f (* dff d) 19) 'Ag (f (* r d) 20)  'Bg (f (* dff r) 21)
     'Wu (f (* dff d) 22) 'Au (f (* r d) 23)  'Bu (f (* dff r) 24)
     'Wd (f (* d dff) 25) 'Ad (f (* r dff) 26) 'Bd (f (* d r) 27)
     'tgt (f (* seq d) 800)}))

(defn- loss-args
  "gblk-loss arg vector for a state map."
  [{:keys [seq d nq nkv hd dff r eps theta]} st]
  (into (mapv st '[x input-ln q-norm k-norm post-attn pre-ffn post-ffn
                   Wq Aq Bq Wk Ak Bk Wv Av Bv Wo Ao Bo Wg Ag Bg Wu Au Bu Wd Ad Bd
                   tgt])
        [seq d nq nkv hd dff r eps theta]))

(defn- train-args
  "gblk-train-step arg vector for a state map."
  [{:keys [seq d nq nkv hd dff r eps theta]} st lr]
  (into (mapv st (concat adapter-syms
                         '[x input-ln q-norm k-norm post-attn pre-ffn post-ffn
                           Wq Wk Wv Wo Wg Wu Wd tgt]))
        [seq d nq nkv hd dff r eps theta lr]))

(defn- host-loss ^double [cfg st]
  (double (apply gblk-loss (loss-args cfg st))))

(defn- clone-adapters [st]
  (reduce (fn [m s] (assoc m s (aclone ^floats (st s)))) st adapter-syms))

;; vg slot per adapter sym (arg idx + 1 in gblk-loss).
(def ^:private adapter-vg-slot
  (zipmap adapter-syms [9 10 12 13 15 16 18 19 21 22 24 25 27 28]))

(defn- cpu-train!
  "CPU-interpreted reference loop: n-steps of value+grad + sgd-step! in place on
   st's adapter arrays. Returns the loss trajectory [loss(state_0) … loss(state_n)]."
  [cfg st lr n-steps]
  (let [vg-fn (rev/value+grad #'gblk-loss)]
    (loop [k 0 losses []]
      (if (= k n-steps)
        (conj losses (host-loss cfg st))
        (let [vg (apply vg-fn (loss-args cfg st))]
          (doseq [s adapter-syms]
            (optim/sgd-step! (st s) (nth vg (adapter-vg-slot s))
                             (ra/alength ^floats (st s)) lr))
          (recur (inc k) (conj losses (double (nth vg 0)))))))))

;; ── the gate ─────────────────────────────────────────────────────────────────────

(deftest gemma-lora-block-resident-train-step
  (if-not @gp/gpu-available?
    (println "  [SKIP] gemma LoRA block resident train-step: no Level Zero GPU")
    (let [gpu (do (require 'raster.gpu.core) (find-ns 'raster.gpu.core))
          make-session (ns-resolve gpu 'make-session)
          bind-program! (ns-resolve gpu 'bind-program!)
          run-program! (ns-resolve gpu 'run-program!)
          close-session! (ns-resolve gpu 'close-session!)
          download (ns-resolve gpu 'download)
          cfg CFG
          lr 0.02
          n-steps 25
          st0 (init-state cfg)
          args (train-args cfg st0 lr)
          prog (pl/compile-gpu-program #'gblk-train-step :ze:0 :dtype :float
                                       :on-non-resident :nil
                                       :gemm-precision :f32-scalar)
          step-kinds (frequencies (mapv :convention (:steps prog)))]
      (testing "the fused fwd+bwd+SGD train step extracts FULLY RESIDENT"
        (is (some? prog)
            "compile-gpu-program returned nil ⇒ a step fell back to the host")
        (println "  [gemma train-step] resident steps:" (count (:steps prog))
                 "by kind:" step-kinds))
      (when prog
        (let [sess (make-session :ze:0)]
          (try
            (bind-program! sess prog args
                           (merge (zipmap adapter-syms (repeat :state))
                                  (zipmap '[x input-ln q-norm k-norm post-attn
                                            pre-ffn post-ffn Wq Wk Wv Wo Wg Wu Wd tgt]
                                          (repeat :constant))))
            (let [gpu-state (fn [] (reduce (fn [m s]
                                             (assoc m s (download sess (keyword (name s)))))
                                           st0 adapter-syms))
                  gpu-losses (loop [k 0 losses [(host-loss cfg st0)]]
                               (if (= k n-steps)
                                 losses
                                 (do (run-program! sess prog args)
                                     (recur (inc k)
                                            (conj losses (host-loss cfg (gpu-state)))))))
                  cpu-losses (cpu-train! cfg (clone-adapters st0) lr n-steps)
                  final-adapters (gpu-state)]
              (println "  [gemma train-step] GPU loss:"
                       (mapv #(format "%.6f" %) (take 4 gpu-losses)) "…"
                       (mapv #(format "%.6f" %) (take-last 3 gpu-losses)))
              (println "  [gemma train-step] CPU loss:"
                       (mapv #(format "%.6f" %) (take 4 cpu-losses)) "…"
                       (mapv #(format "%.6f" %) (take-last 3 cpu-losses)))
              (testing "loss decreases over 25 on-device steps"
                (is (< (peek gpu-losses) (* 0.7 (first gpu-losses)))
                    (str "final " (peek gpu-losses) " vs initial " (first gpu-losses)))
                ;; overall decreasing: allow small plateaus/bumps, not a rising tail
                (is (>= 3 (count (filter true? (map >= (rest gpu-losses) gpu-losses))))
                    "at most 3 non-decreasing steps"))
              (testing "adapters changed on-device"
                (doseq [s adapter-syms]
                  (is (not (java.util.Arrays/equals ^floats (final-adapters s)
                                                    ^floats (st0 s)))
                      (str s " must differ from init after training"))))
              (testing "CPU-interpreted loop tracks the GPU trajectory (f32 tolerance)"
                (doseq [[k g c] (map vector (range) gpu-losses cpu-losses)]
                  (is (< (/ (Math/abs (- g c)) (max 1e-9 (Math/abs c))) 1.0e-3)
                      (format "step %d GPU %.6f vs CPU %.6f" k g c)))))
            (finally (close-session! sess))))))))

;; ═════════════════════════════════════════════════════════════════════════════════
;; MIXED-PRECISION BACKWARD (S2a): the SAME resident train step (value+grad + SGD) under
;; :gemm-precision :f16-xmx — f16 GEMM inputs, f32 accumulate/output — must train the
;; adapters along the SAME loss trajectory as the exact :f32-scalar policy.
;;
;; This is the gate for running the VJP/backward program in mixed precision (the forward
;; was already validated). Measured on the real gemma-3-270m layer (seq 64, d 640, dff
;; 2048, r 16, ze:0, device-event timing): VJP kernel time 65.5 → 38.8 ms (1.69x), its
;; GEMM part 36.8 → 12.0 ms (3.07x); adapter-grad rel-err vs the f32 grads ~9e-4 with
;; cosine similarity 1.000 — f16 MANTISSA noise, not underflow (0.1-0.3% of grad entries
;; fall below the f16 min normal 6.1e-5; shrinking the seed cotangent 1000x — deep into
;; subnormal territory — only degrades rel-err to ~2.5e-3, and loss scaling recovers it
;; because the VJP is linear in the seed).
;;
;; Dims are bind-time scalars, so the SAME compiled descriptor serves both policies (and
;; both dim sets). CFG-MP is sized so that every :gemm step clears the XMX pitch gate
;; (n>=8 and k>=8) — at CFG's r=4/nkh=4 the gate would silently fall back to scalar and
;; the test would not exercise f16 at all. That is asserted, not assumed.
;; ═════════════════════════════════════════════════════════════════════════════════

(def CFG-MP
  "Same block, dims above the XMX pitch gate (r=8, nkh=8, nqh=16)."
  {:seq 8 :d 16 :nq 2 :nkv 1 :hd 8 :dff 32 :r 8 :eps 1.0e-6 :theta 10000.0})

(defn- gemm-dims
  "[{:variant :m :n :k} …] for every :gemm step of a descriptor at `args`."
  [prog args]
  (mapv (fn [s] {:variant (:variant s)
                 :m (long ((:m-fn s) args)) :n (long ((:n-fn s) args)) :k (long ((:k-fn s) args))})
        (filter #(= :gemm (:convention %)) (:steps prog))))

(defn- run-trajectory!
  "n-steps of the resident train step under `prog`'s GEMM policy in a fresh session.
   Returns the host loss trajectory [loss(state_0) … loss(state_n)]."
  [gpu prog cfg st0 lr n-steps]
  (let [make-session (ns-resolve gpu 'make-session)
        bind-program! (ns-resolve gpu 'bind-program!)
        run-program! (ns-resolve gpu 'run-program!)
        close-session! (ns-resolve gpu 'close-session!)
        download (ns-resolve gpu 'download)
        st (clone-adapters st0)
        args (train-args cfg st lr)
        sess (make-session :ze:0)]
    (try
      (let [h (bind-program! sess prog args
                             (merge (zipmap adapter-syms (repeat :state))
                                    (zipmap '[x input-ln q-norm k-norm post-attn
                                              pre-ffn post-ffn Wq Wk Wv Wo Wg Wu Wd tgt]
                                            (repeat :constant)))
                             {:key :train})]
        (loop [k 0 losses [(host-loss cfg st)]]
          (if (= k n-steps)
            losses
            (do (run-program! sess h args)
                (recur (inc k)
                       (conj losses
                             (host-loss cfg (reduce (fn [m s]
                                                      (assoc m s (download sess (keyword (name s)))))
                                                    st adapter-syms))))))))
      (finally (close-session! sess)))))

(deftest gemma-lora-mixed-precision-backward-trajectory
  (if-not @gp/gpu-available?
    (println "  [SKIP] gemma LoRA mixed-precision backward: no Level Zero GPU")
    (let [gpu (do (require 'raster.gpu.core) (find-ns 'raster.gpu.core))
          cfg CFG-MP
          lr 0.02
          n-steps 25
          st0 (init-state cfg)
          args (train-args cfg st0 lr)
          p32 (pl/compile-gpu-program #'gblk-train-step :ze:0 :dtype :float
                                      :on-non-resident :nil :gemm-precision :f32-scalar)
          ;; the descriptor is plain data: the policy is a bind-time re-tag, same program
          p16 (assoc p32 :gemm-precision :f16-xmx)]
      (is (some? p32) "train-step must extract fully resident")
      (when p32
        (let [dims (gemm-dims p32 args)]
          (testing "every backward GEMM clears the XMX pitch gate (so :f16-xmx really fires)"
            (println "  [mixed-precision bwd]" (count dims) "gemm steps, variants:"
                     (frequencies (map :variant dims)))
            (is (every? (fn [{:keys [n k]}] (and (>= n 8) (>= k 8))) dims)
                (str "gemm steps below the n>=8/k>=8 pitch gate would silently stay scalar: "
                     (pr-str (filter (fn [{:keys [n k]}] (or (< n 8) (< k 8))) dims)))))
          (let [l32 (run-trajectory! gpu p32 cfg st0 lr n-steps)
                l16 (run-trajectory! gpu p16 cfg st0 lr n-steps)]
            (println "  [mixed-precision bwd] f32-scalar loss:"
                     (mapv #(format "%.6f" %) (take 3 l32)) "…"
                     (mapv #(format "%.6f" %) (take-last 2 l32)))
            (println "  [mixed-precision bwd] f16-xmx    loss:"
                     (mapv #(format "%.6f" %) (take 3 l16)) "…"
                     (mapv #(format "%.6f" %) (take-last 2 l16)))
            (testing "the f16-xmx backward still trains (loss decreases on-device)"
              (is (< (peek l16) (* 0.7 (first l16)))
                  (str "final " (peek l16) " vs initial " (first l16))))
            (testing "the f16-xmx trajectory tracks the exact f32 trajectory step for step"
              ;; f16 GEMM inputs perturb each gradient by ~1e-3 relative; over 25 SGD steps
              ;; that stays a ~1e-3-level trajectory divergence (it does NOT compound into a
              ;; different optimization path at this lr).
              (doseq [[k a b] (map vector (range) l32 l16)]
                (is (< (/ (Math/abs (- a b)) (max 1.0e-9 (Math/abs a))) 5.0e-3)
                    (format "step %d: f32 %.6f vs f16 %.6f" k a b))))))))))

;; ═════════════════════════════════════════════════════════════════════════════════
;; DUAL-PROGRAM SHARED SESSION (task #9): the fwd program and the VJP train-step
;; program bound into ONE session over shared weight+adapter buffers. The fwd
;; program reads the adapters the train program updates on-device — NO per-step
;; adapter download/re-upload (the finetune 18-layer loop's refresh-adapters!
;; round-trip) and NO second copy of the frozen weights (the ~2x VRAM of the
;; two-session path).
;; ═════════════════════════════════════════════════════════════════════════════════

;; Forward-only program twin (finetune.train/gblock-fwd! shape): yo := gblk(x,…),
;; residual-add! with the constant zero buffer = the resident copy-to-output-param.
(deftm gblk-fwd!
  [x :- (Array float)
   input-ln :- (Array float) q-norm :- (Array float) k-norm :- (Array float)
   post-attn :- (Array float) pre-ffn :- (Array float) post-ffn :- (Array float)
   Wq :- (Array float) Aq :- (Array float) Bq :- (Array float)
   Wk :- (Array float) Ak :- (Array float) Bk :- (Array float)
   Wv :- (Array float) Av :- (Array float) Bv :- (Array float)
   Wo :- (Array float) Ao :- (Array float) Bo :- (Array float)
   Wg :- (Array float) Ag :- (Array float) Bg :- (Array float)
   Wu :- (Array float) Au :- (Array float) Bu :- (Array float)
   Wd :- (Array float) Ad :- (Array float) Bd :- (Array float)
   zero :- (Array float) yo :- (Array float)
   seq :- Long d :- Long nq :- Long nkv :- Long hd :- Long dff :- Long r :- Long
   eps :- Double theta :- Double] :- (Array float)
  (let [y (raster.dl.gemma-train-resident-test/gblk x
                                                    input-ln q-norm k-norm post-attn pre-ffn post-ffn
                                                    Wq Aq Bq Wk Ak Bk Wv Av Bv Wo Ao Bo Wg Ag Bg Wu Au Bu Wd Ad Bd
                                                    seq d nq nkv hd dff r eps theta)]
    (raster.dl.nn/residual-add! y zero yo (clojure.core/* seq d))
    yo))

(defn- fwd-args*
  "gblk-fwd! arg vector for a state map."
  [{:keys [seq d nq nkv hd dff r eps theta]} st zero yo]
  (-> (mapv st '[x input-ln q-norm k-norm post-attn pre-ffn post-ffn
                 Wq Aq Bq Wk Ak Bk Wv Av Bv Wo Ao Bo Wg Ag Bg Wu Au Bu Wd Ad Bd])
      (conj zero yo)
      (into [seq d nq nkv hd dff r eps theta])))

(def ^:private frozen-syms
  '[x input-ln q-norm k-norm post-attn pre-ffn post-ffn Wq Wk Wv Wo Wg Wu Wd])

(defn- session-bytes [sess]
  (reduce + 0 (map :byte-size (vals (:buffers @sess)))))

(deftest gemma-lora-dual-program-shared-session
  (if-not @gp/gpu-available?
    (println "  [SKIP] gemma dual-program shared session: no Level Zero GPU")
    (let [gpu (do (require 'raster.gpu.core) (find-ns 'raster.gpu.core))
          make-session (ns-resolve gpu 'make-session)
          bind-program! (ns-resolve gpu 'bind-program!)
          run-program! (ns-resolve gpu 'run-program!)
          close-session! (ns-resolve gpu 'close-session!)
          download (ns-resolve gpu 'download)
          cfg CFG
          lr 0.02
          n (* (:seq cfg) (:d cfg))
          st0 (init-state cfg)
          trn-prog (pl/compile-gpu-program #'gblk-train-step :ze:0 :dtype :float
                                           :on-non-resident :nil :gemm-precision :f32-scalar)
          fwd-prog (pl/compile-gpu-program #'gblk-fwd! :ze:0 :dtype :float
                                           :on-non-resident :nil :gemm-precision :f32-scalar)
          trn-roles (merge (zipmap adapter-syms (repeat :state))
                           (zipmap (conj frozen-syms 'tgt) (repeat :constant)))
          ;; the interleaved schedule: k train steps, fwd-only eval, k more, eval again
          k-steps 5
          run-schedule! (fn [train! eval!]
                          (vec (for [_ (range 2)]
                                 (do (dotimes [_ k-steps] (train!))
                                     (eval!)))))]
      (is (some? trn-prog) "train-step twin must extract fully resident")
      (is (some? fwd-prog) "fwd twin must extract fully resident")
      (when (and trn-prog fwd-prog)
        ;; ── baseline: the OLD two-session path with the explicit refresh round-trip ──
        (let [stB (clone-adapters st0)
              t-sess (make-session :ze:0)
              f-sess (make-session :ze:0)
              baseline
              (try
                (bind-program! t-sess trn-prog (train-args cfg stB lr) trn-roles)
                (bind-program! f-sess fwd-prog (fwd-args* cfg stB (float-array n) (float-array n))
                               (merge (zipmap frozen-syms (repeat :constant))
                                      {'zero :constant}
                                      (zipmap adapter-syms (repeat :input))))
                (let [refresh! (fn [] (doseq [s adapter-syms]
                                        (let [^floats host (stB s)
                                              ^floats dev (download t-sess (keyword (name s)))]
                                          (System/arraycopy dev 0 host 0 (alength host)))))
                      evals (run-schedule!
                             #(run-program! t-sess trn-prog (train-args cfg stB lr))
                             #(do (refresh!)
                                  (get (run-program! f-sess fwd-prog
                                                     (fwd-args* cfg stB (float-array 0) (float-array 0)))
                                       'yo)))]
                  {:evals evals
                   :adapters (do (refresh!) (reduce (fn [m s] (assoc m s (aclone ^floats (stB s)))) {} adapter-syms))
                   :bytes (+ (session-bytes t-sess) (session-bytes f-sess))})
                (finally (close-session! t-sess) (close-session! f-sess)))]
          ;; ── shared: BOTH programs in ONE session, weights + adapters shared ──
          (let [stS (clone-adapters st0)
                sess (make-session :ze:0)]
            (try
              (let [trn-h (bind-program! sess trn-prog (train-args cfg stS lr) trn-roles
                                         {:key :train})]
                (testing "collision without :reuse-buffers fails loud"
                  (is (thrown? clojure.lang.ExceptionInfo
                               (bind-program! sess fwd-prog
                                              (fwd-args* cfg stS (float-array n) (float-array n))
                                              {} {:key :fwd}))))
                (testing "re-binding an already-bound program key fails loud"
                  (is (thrown? clojure.lang.ExceptionInfo
                               (bind-program! sess trn-prog (train-args cfg stS lr) trn-roles
                                              {:key :train}))))
                (let [fwd-h (bind-program! sess fwd-prog
                                           (fwd-args* cfg stS (float-array n) (float-array n))
                                           (merge (zipmap frozen-syms (repeat :constant))
                                                  {'zero :constant}
                                                  ;; adapters :state — SHARED with the train
                                                  ;; program, never re-uploaded by run-program!
                                                  (zipmap adapter-syms (repeat :state)))
                                           {:key :fwd :reuse-buffers true})
                      evals (run-schedule!
                             #(run-program! sess trn-h (train-args cfg stS lr))
                             #(get (run-program! sess fwd-h
                                                 (fwd-args* cfg stS (float-array 0) (float-array 0)))
                                   'yo))
                      shared-adapters (reduce (fn [m s] (assoc m s (download sess (keyword (name s)))))
                                              {} adapter-syms)
                      shared-bytes (session-bytes sess)
                      max-abs-diff (fn [^floats a ^floats b]
                                     (reduce max 0.0 (map #(Math/abs (- (double %1) (double %2)))
                                                          (seq a) (seq b))))]
                  (testing "fwd sees the VJP-updated adapters with NO refresh round-trip"
                    (doseq [[i [ys yb]] (map-indexed vector (map vector evals (:evals baseline)))]
                      (is (< (max-abs-diff ys yb) 1.0e-5)
                          (format "eval %d: shared-session fwd output diverges from two-session baseline" i))))
                  (testing "on-device adapter state matches the two-session trajectory (f32 noise)"
                    (doseq [s adapter-syms]
                      (is (< (max-abs-diff (shared-adapters s) (get (:adapters baseline) s)) 1.0e-5)
                          (str s " trajectory diverged"))))
                  (testing "loss after interleaved training matches the baseline"
                    (let [lS (host-loss cfg (merge stS shared-adapters))
                          lB (host-loss cfg (merge stS (:adapters baseline)))]
                      (println (format "  [dual-program] loss shared %.6f vs two-session %.6f" lS lB))
                      (is (< (/ (Math/abs (- lS lB)) (max 1e-9 (Math/abs lB))) 1.0e-4))))
                  (testing "shared session VRAM drops by the duplicated shared-param bytes"
                    ;; the baseline holds the common params (x, frozen weights/norms, adapters)
                    ;; TWICE — once per session. Sharing must eliminate that second copy. (At
                    ;; these tiny dims scratch dominates; at real gemma dims the shared frozen
                    ;; weights ARE the ~2x — 1.7 GB → ~0.9 GB for the 18-layer stack.)
                    (let [shared-param-bytes (reduce + 0 (map #(* 4 (alength ^floats (stS %)))
                                                              (concat frozen-syms adapter-syms)))]
                      (println (format "  [dual-program] VRAM: shared %d KiB vs two-session %d KiB (%.2fx, %d KiB dedup'd)"
                                       (long (/ shared-bytes 1024)) (long (/ (:bytes baseline) 1024))
                                       (double (/ shared-bytes (:bytes baseline)))
                                       (long (/ shared-param-bytes 1024))))
                      (is (<= shared-bytes (- (:bytes baseline) shared-param-bytes))
                          "the second copy of every shared param must be gone")))))
              (finally (close-session! sess)))))))))
