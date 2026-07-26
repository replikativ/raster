(ns soac-contract-bench
  "Cadence-controlled COLD/WARM benchmark for the SOAC contraction ladder
   (par/contract → segmented SegRed → naive / block-tiled / register-tiled OpenCL),
   measured on the SAME rigorous method as `resident-gemm-cold-bench` and against the
   golden `emit-gemm-tiled` (the resident XMX GEMM) as the reference line.

   WHY THIS EXISTS
   ---------------
   The SOAC-unification thesis (see .internal/soac_unification_plan.md) is: express matmul
   AS a SOAC and let the GENERAL, schedule-driven emitter reproduce the peak kernel — no
   gemm-specific special case. The perf GATE for that claim is a like-for-like measurement
   of the SOAC ladder against the hand-written GEMM, under a harness that controls the two
   confounds the GEMM-tiling episode taught us to fear:

     1. CADENCE / CLOCK: one recorded graph = `launches` back-to-back kernels (barriers
        serialize them) so the GPU stays boosted across the whole sequence; configs are
        replayed ROUND-ROBIN each round so each is sampled under the same clock/thermal state.
     2. WARM-OPERAND: COLD = every launch reads a DISTINCT buffer triple (footprint >> L3,
        every read a first-touch DRAM read = a realistic per-step activation); WARM = one
        shared L3-resident triple, kept only as a control. Report COLD for any real claim.

   This is the salvaged methodology from `resident-gemm-cold-bench`, generalized from the
   single hard-wired GEMM to a SPEC-DRIVEN set of kernels so the whole ladder — and, once
   it lands, the DPAS-tensorized SOAC kernel — is measured next to the golden GEMM.

   DTYPE HONESTY
   -------------
   The golden GEMM is f16-in / f32-acc / f16-out — that IS its advantage (half the DRAM
   traffic + XMX DPAS). The portable SOAC ladder here is f64 (or f32). These are DIFFERENT
   points on the precision/perf curve, so cross-dtype rows are NOT apples-to-apples on
   memory traffic; the report labels each row's dtype. The apples-to-apples comparison that
   proves 'general emitter, no lost performance' is DPAS-SOAC(f16) vs golden(f16) — which
   arrives with the DPAS emitter (plan step 4). Until then this harness measures (a) the
   portable ladder's internal speedup (naive→block→regtiled, all one dtype) and (b) the
   ladder's absolute standing vs the golden reference line (dtype-labeled, not a fair FLOP
   race yet).

   MEASUREMENT REQUIRES A QUIET BOX. On a contended machine the numbers are meaningless
   (the whole point of the cadence control is a stable clock). Do NOT trust output from a
   box running other GPU/CPU load.

   USE
   ---
     (require 'soac-contract-bench :reload)
     ;; the portable ladder at one shape, all f64:
     (soac-contract-bench/compare-ladder {:name \"proj\" :m 1024 :k 640 :n 2048})
     ;; include the golden GEMM reference line (f16), dtype-labeled:
     (soac-contract-bench/compare-ladder {:name \"proj\" :m 1024 :k 640 :n 2048} :golden? true)"
  (:require [raster.gpu.ze-runtime :as ze]
            [raster.compiler.passes.parallel.contract-lower :as cl]
            [raster.compiler.backend.gpu.segop-opencl :as sco]
            [raster.compiler.backend.gpu.opencl-codegen :as cg]
            [raster.compiler.support.spirv-cache :as spv]
            [raster.runtime.hardware :as hw]
            [clojure.string :as str])
  (:import [java.lang.foreign MemorySegment ValueLayout]))

(def ^:private I32 ValueLayout/JAVA_INT)

(def ^:private device-hex
  (delay (get-in (hw/device :ze:0) [:capabilities :device-id-hex])))

;; ── stats (mirrors resident-gemm-cold-bench) ───────────────────────────────────────
(defn- median [xs]
  (let [v (vec (sort xs)) n (count v)]
    (if (odd? n) (nth v (quot n 2))
        (/ (+ (nth v (dec (quot n 2))) (nth v (quot n 2))) 2.0))))

(defn- pctl [xs p]
  (let [v (vec (sort xs)) n (count v)]
    (nth v (min (dec n) (int (* p n))))))

(defn- gflops [ms m n k] (/ (* 2.0 m k n) (* ms 1.0e6)))

;; ── the matmul contraction form (C[i,j] = Σ_l A[i,l]·B[l,j]) ─────────────────────────
(defn- matmul-form [m n k]
  (list 'raster.par/contract 'C [['i m] ['j n]] [['l k]]
        (list '* (list 'aget 'A (list '+ (list '* 'i k) 'l))
              (list 'aget 'B (list '+ (list '* 'l n) 'j)))))

;; ── buffer plumbing: one A[m×k]·B[k×n]→C[m×n] triple at a given dtype ────────────────
(def ^:private host-cache (atom {}))
(defn- host-doubles ^doubles [n]
  (or (get @host-cache n)
      (let [r (java.util.Random. (+ 1234 (long n)))
            a (double-array n)]
        (dotimes [i n] (aset a i (* 0.5 (.nextGaussian r))))
        (swap! host-cache assoc n a) a)))
(defn- host-floats ^floats [n]
  (let [d (host-doubles n) a (float-array n)]
    (dotimes [i n] (aset a i (float (aget d i)))) a))

(defn- alloc-triple
  "Allocate a distinct A[m×k]/B[k×n]/C[m×n] triple at `dtype` (:double | :float | :half)."
  [dtype m n k]
  (case dtype
    :half   {:a (ze/buffer-of-floats-as-half (host-floats (* m k)))
             :b (ze/buffer-of-floats-as-half (host-floats (* k n)))
             :c (ze/make-buffer (* m n) :half)}
    :float  {:a (ze/array->buffer! (ze/make-buffer (* m k) :float) (host-floats (* m k)))
             :b (ze/array->buffer! (ze/make-buffer (* k n) :float) (host-floats (* k n)))
             :c (ze/make-buffer (* m n) :float)}
    :double {:a (ze/array->buffer! (ze/make-buffer (* m k) :double) (host-doubles (* m k)))
             :b (ze/array->buffer! (ze/make-buffer (* k n) :double) (host-doubles (* k n)))
             :c (ze/make-buffer (* m n) :double)}))

(defn- free-triple [{:keys [a b c]}] (doseq [x [a b c]] (ze/free-buffer! x)))

(defn- set-gc! [^MemorySegment gc gx gy gz]
  (.set gc I32 0 (int gx)) (.set gc I32 4 (int gy)) (.set gc I32 8 (int gz)))

;; ── kernel specs ────────────────────────────────────────────────────────────────────
;; A spec knows how to (1) compile+cache its source for a shape, (2) bind a triple into a
;; bound 2D kernel with its group counts pre-baked. dims: [m n k]. Returns the bound map
;; (record-graph! reads :bound + :kernel-name; group-count nil ⇒ leave the pre-baked 2D grid).
(def ^:private mod-cache (atom {}))
(defn- compile-src! [cache-key kname src]
  (or (get @mod-cache cache-key)
      (let [spirv (spv/compile-opencl-to-spirv src :device @device-hex)
            module (ze/load-module! spirv)
            entry {:module module :kname kname}]
        (swap! mod-cache assoc cache-key entry)
        entry)))

(defn- bind! [module kname wg args gx gy gz]
  (let [kh (ze/create-kernel-fresh module kname)
        bnd (ze/bind-kernel-2d! kh wg args)]
    (set-gc! (:gc-seg bnd) gx gy gz)
    bnd))

(defn- naive-spec [dtype]
  {:label (str "naive-" (name dtype)) :dtype dtype
   :bind
   (fn [m n k triple]
     ;; literal-dim kernel (dims baked in source, matching block/regtiled): params A B out,
     ;; then the always-emitted trailing `int _nseg` (= number of segments = launch count).
     (let [sr (cl/contract-form->segred (matmul-form m n k) :dtype dtype)
           {:keys [kernel-name source]} (sco/generate-segmented-reduce-kernel sr 'C :dtype dtype)
           {:keys [module kname]} (compile-src! [:naive dtype m n k] kernel-name source)
           nseg (* m n) wg 256
           args [(:segment (:a triple)) (:segment (:b triple)) (:segment (:c triple))
                 {:type :int :value (int nseg)}]]
       (bind! module kname [wg 1] args (long (Math/ceil (/ (double nseg) wg))) 1 1)))})

(defn- block-spec [dtype tile]
  {:label (str "block" tile "-" (name dtype)) :dtype dtype
   :bind
   (fn [m n k triple]
     ;; literal-dim kernel (dims baked in source ⇒ recompile per shape): params A B out.
     (let [sr (cl/contract-form->segred (matmul-form m n k) :dtype dtype)
           {:keys [kernel-name source]} (sco/generate-regtiled-contraction-kernel sr 'C :dtype dtype :bm tile :bn tile :bk 16 :tm 1 :tn 1)
           {:keys [module kname]} (compile-src! [:block dtype tile m n k] kernel-name source)
           args [(:segment (:a triple)) (:segment (:b triple)) (:segment (:c triple))]]
       (bind! module kname [tile tile] args
              (long (Math/ceil (/ (double n) tile))) (long (Math/ceil (/ (double m) tile))) 1)))})

(defn- regtiled-spec [dtype]
  {:label (str "regtiled-" (name dtype)) :dtype dtype
   :bind
   (fn [m n k triple]
     (let [sr (cl/contract-form->segred (matmul-form m n k) :dtype dtype)
           {:keys [kernel-name source block workgroup]}
           (sco/generate-regtiled-contraction-kernel sr 'C :dtype dtype)
           [bm bn _bk] block
           [wgx wgy] workgroup
           {:keys [module kname]} (compile-src! [:regtiled dtype m n k] kernel-name source)
           args [(:segment (:a triple)) (:segment (:b triple)) (:segment (:c triple))]]
       (bind! module kname [wgx wgy] args
              (long (Math/ceil (/ (double n) bn))) (long (Math/ceil (/ (double m) bm))) 1)))})

(def ^:private golden-spec
  "The hand-written resident XMX GEMM (emit-gemm-tiled), f16 — the reference line."
  {:label "golden-gemm-f16" :dtype :half
   :bind
   (fn [m n k triple]
     (let [kname "gemm_nonsquare_bench"
           src (cg/emit-gemm-tiled kname :c-dtype :half)
           {:keys [module kname]} (compile-src! [:golden] kname src)
           args [(:segment (:a triple)) (:segment (:b triple)) (:segment (:c triple))
                 {:type :int :value (int m)} {:type :int :value (int n)} {:type :int :value (int k)}]]
       (bind! module kname [256 1] args
              (long (Math/ceil (/ (double n) 128.0))) (long (Math/ceil (/ (double m) 128.0))) 1)))})

(def ^:private dpas-spec
  "The DPAS/XMX-tensorized SOAC contraction — sourced from the segred via the legality gate,
   f16. This is the apples-to-apples proof that the SOAC path reproduces the golden GEMM:
   for a canonical matmul the gate accepts and emits the golden body, so a near-tie with
   golden-gemm-f16 is the 'general emitter, no lost performance' result. Requires N%8==0 and
   K%8==0 (the DPAS block-read pitch alignment the gate enforces)."
  {:label "dpas-soac-f16" :dtype :half
   :bind
   (fn [m n k triple]
     (let [sr (cl/contract-form->segred (matmul-form m n k) :dtype :half)
           {:keys [kernel-name source array-params tensorized reason]}
           (sco/generate-dpas-contraction-kernel sr 'C)
           _ (when-not tensorized
               (throw (ex-info (str "dpas-spec: gate rejected this shape (" reason
                                    ") — DPAS needs N%8==0 & K%8==0, canonical orientation")
                               {:m m :n n :k k :reason reason})))
           {:keys [module kname]} (compile-src! [:dpas] kernel-name source)
           ;; array-params = [row col] binding order; for the matmul row=A, col=B.
           bufs (mapv (fn [s] (:segment (get triple (keyword (str/lower-case (name s)))))) array-params)
           args (into bufs [(:segment (:c triple))
                            {:type :int :value (int m)} {:type :int :value (int n)} {:type :int :value (int k)}])]
       (bind! module kname [256 1] args
              (long (Math/ceil (/ (double n) 128.0))) (long (Math/ceil (/ (double m) 128.0))) 1)))})

;; ── cadence graph over a spec (mirrors resident-gemm-cold-bench/record-seq-graph) ─────
(defn- record-seq-graph
  "Record ONE graph of `launches` back-to-back launches of `spec` (barriers serialize).
   `triple-of` maps launch index → a buffer triple (distinct for cold, shared for warm)."
  [spec m n k launches triple-of]
  (ze/record-graph!
   (mapv (fn [i]
           {:bound ((:bind spec) m n k (triple-of i))
            :kernel-name (:label spec)
            :phase (:label spec)})
         (range launches))
   {:profile? true}))

(defn compare-ladder
  "Cadence-controlled, interleaved COLD-vs-WARM comparison of the SOAC contraction ladder
   on one contraction shape (C[m×n] = A[m×k]·B[k×n]). Every spec is replayed ROUND-ROBIN
   each round under the same clock/thermal state; COLD reads distinct DRAM triples, WARM one
   shared L3-resident triple (control). Reports median GFLOP/s (dtype-labeled — cross-dtype
   rows are not a fair FLOP race; see ns doc).

   shape:   {:name :m :k :n}
   opts:    :dtype (:double) — element type for the portable SOAC ladder (naive/block/regtiled)
            :golden? (false) — include the f16 golden GEMM reference line
            :dpas?   (false) — include the DPAS-tensorized SOAC kernel (f16; the apples-to-
                     apples comparand to golden; needs N%8==0 & K%8==0)
            :specs   — override the spec list entirely
            :launches (24) :reps (10) :warmup (6)"
  [{:keys [m n k name]} & {:keys [dtype golden? dpas? specs launches reps warmup]
                           :or {dtype :double golden? false dpas? false launches 24 reps 10 warmup 6}}]
  (let [specs (or specs
                  (cond-> [(naive-spec dtype) (block-spec dtype 16) (regtiled-spec dtype)]
                    dpas?   (conj dpas-spec)
                    golden? (conj golden-spec)))
        ;; per-spec buffers at its own dtype (cold triples distinct; warm shared)
        entries (mapv (fn [spec]
                        (let [dt (:dtype spec)
                              cold-triples (mapv (fn [_] (alloc-triple dt m n k)) (range launches))
                              warm-triple (alloc-triple dt m n k)]
                          {:label (:label spec) :dtype dt
                           :cold-triples cold-triples :warm-triple warm-triple
                           :cold (record-seq-graph spec m n k launches (fn [i] (nth cold-triples i)))
                           :warm (record-seq-graph spec m n k launches (constantly warm-triple))}))
                      specs)
        acc (atom (into {} (for [e entries] [(:label e) {:cold [] :warm []}])))]
    ;; interleaved warmup for thermal/clock steady state
    (dotimes [_ warmup]
      (doseq [e entries]
        (ze/replay-graph! (:cold e)) (ze/reset-graph-events! (:cold e))
        (ze/replay-graph! (:warm e)) (ze/reset-graph-events! (:warm e))))
    ;; interleaved timed rounds
    (dotimes [_ reps]
      (doseq [e entries]
        (ze/replay-graph! (:cold e))
        (let [ks (:kernels (ze/read-graph-timestamps! (:cold e)))]
          (swap! acc update-in [(:label e) :cold] into (map :ms ks)))
        (ze/replay-graph! (:warm e))
        (let [ks (:kernels (ze/read-graph-timestamps! (:warm e)))]
          (swap! acc update-in [(:label e) :warm] into (map :ms (drop 2 ks))))))
    (doseq [e entries]
      (ze/destroy-graph! (:cold e)) (ze/destroy-graph! (:warm e))
      (doseq [t (:cold-triples e)] (free-triple t)) (free-triple (:warm-triple e)))
    ;; report
    (println (format "\n===== SOAC CONTRACT BENCH  %s (M%d K%d N%d) reps=%d launches=%d ====="
                     name m k n reps launches))
    (println (format "%-18s | %6s | %8s %9s %9s | %8s %9s"
                     "config" "dtype" "cold-GF" "cMed-ms" "cP75-ms" "warm-GF" "wMed-ms"))
    (let [rows (for [e entries
                     :let [cs (get-in @acc [(:label e) :cold])
                           ws (get-in @acc [(:label e) :warm])
                           cmed (median cs) wmed (median ws)]]
                 {:label (:label e) :dtype (:dtype e)
                  :cold-gf (gflops cmed m n k) :cold-med cmed :cold-p75 (pctl cs 0.75)
                  :warm-gf (gflops wmed m n k) :warm-med wmed :n (count cs)})]
      (doseq [r rows]
        (println (format "%-18s | %6s | %8.1f %9.4f %9.4f | %8.1f %9.4f"
                         (:label r) (clojure.core/name (:dtype r))
                         (:cold-gf r) (:cold-med r) (:cold-p75 r)
                         (:warm-gf r) (:warm-med r))))
      (vec rows))))

;; Reference shapes (gemma FFN — same standard sweep as resident-gemm-cold-bench).
(def reference-shapes
  [{:name "proj      " :m 1024 :k 640  :n 2048}
   {:name "gate      " :m 1024 :k 640  :n 1024}
   {:name "down-proj " :m 1024 :k 2048 :n 640}])
