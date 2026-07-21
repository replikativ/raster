(ns raster.compiler.passes.parallel.contract-route
  "Routing brain for tensor contractions: a `(raster.par/contract …)` SOAC form → the
   hardware-optimal kernel choice, via the tensorize LEGALITY GATE.

   This is the pipeline INTEGRATION seam for the SOAC contraction ladder — the piece that
   makes the emitters load-bearing instead of proven-but-bypassed. `route-contraction`
   decides: if the DPAS/XMX gate accepts (canonical matmul, f16, pitch-aligned) → the peak
   tensorized kernel (byte-identical to the hand-wired GEMM front door); otherwise → the
   portable register-tiled kernel (any dtype, arbitrary dims). Same decision the walker/
   opencl-pass makes when it meets a contraction; kept in ONE place so the gate's hardware
   knowledge lives with the emitters, not scattered across passes.

   Returns a launch-ready descriptor: {:strategy :dpas|:regtiled, :kernel-name, :source,
   :array-params (binding order), :dtype, :wg [x y], :grid [gx gy], :scalar-args [{:type :int
   :value n}…], :dims [M N L]}."
  (:require [raster.compiler.passes.parallel.contract-lower :as cl]
            [raster.compiler.backend.gpu.segop-opencl :as sco]))

(defn par-contract-form?
  "Is `form` a (raster.par/contract out free-axes contract-axes body & opts) form?"
  [form]
  (and (seq? form) (= 'raster.par/contract (first form))))

(defn- ceil-div [a b] (long (Math/ceil (/ (double a) (double b)))))

(declare route-2free-1contract route-quant)

(defn route-contraction
  "Route a contraction form to the hardware-optimal kernel via the DPAS legality gate.
   dtype selects the element type of the intended kernel (:half tries DPAS; :byte/:int8 tries
   the int8 quant leaves — dp4a for the :nt operand layout, quant naive-widening for :nn;
   anything else, or a gate rejection, falls back to the register-tiled portable kernel).
   scheme = the quant decode descriptor {:scale :a-zp :b-zp} for int8 (default {:scale 1.0})."
  [contract-form & {:keys [dtype scheme] :or {dtype :half scheme {:scale 1.0}}}]
  (let [out-sym (second contract-form)
        free-axes (nth contract-form 2)
        contract-axes (nth contract-form 3)
        n-free (count free-axes)
        n-contract (count contract-axes)
        nseg (reduce * 1 (map second free-axes))]   ; product of free bounds (literal dims)
    (cond
      ;; int8 → the quant leaves (dp4a for :nt, quant naive-widening for :nn)
      (#{:byte :int8} dtype)
      (route-quant contract-form out-sym scheme n-free n-contract nseg)

      ;; 0 contract axes → outer product / broadcast → pure N-D SegMap (1-D launch)
      (zero? n-contract)
      (let [sm (cl/contract-form->segmap contract-form :dtype dtype)
            {:keys [kernel-name source array-params]} (sco/generate-segmap-nd-kernel sm out-sym :dtype dtype)]
        {:strategy :segmap
         :kernel-name kernel-name :source source :array-params array-params
         :dtype dtype :wg [256 1] :grid [(ceil-div nseg 256) 1]
         :scalar-args [{:type :int :value (int nseg)}] :dims [nseg]})

      ;; 2 free + 1 contract → the tensorize fast path (DPAS if legal, else regtiled)
      (and (= 2 n-free) (= 1 n-contract))
      (route-2free-1contract contract-form out-sym dtype)

      ;; everything else (n-free≠2 with 1 contract, OR n≥2 contract axes) → naive segmented
      ;; reduce. contract-form->segred flattens n≥2 contract axes into one innermost dim.
      :else
      (let [sr (cl/contract-form->segred contract-form :dtype dtype)
            {:keys [kernel-name source array-params]} (sco/generate-segmented-reduce-kernel sr out-sym :dtype dtype)]
        {:strategy :naive-segred
         :kernel-name kernel-name :source source :array-params array-params
         :dtype dtype :wg [256 1] :grid [(ceil-div nseg 256) 1]
         :scalar-args [{:type :int :value (int nseg)}] :dims [nseg]}))))

(defn- route-2free-1contract [contract-form out-sym dtype]
  (let [sr (cl/contract-form->segred contract-form :dtype dtype)
        dpas (sco/generate-dpas-contraction-kernel sr out-sym :dtype dtype)]
    (if (:tensorized dpas)
      (let [[M N _L] (:dims dpas)]
        {:strategy :dpas
         :kernel-name (:kernel-name dpas)
         :source (:source dpas)
         :array-params (:array-params dpas)          ; [row col] = [A-slot B-slot]
         :dtype :half
         :wg [256 1]
         :grid [(ceil-div N 128) (ceil-div M 128)]   ; [gc-n gc-m] (group-id0=N, id1=M)
         :scalar-args (mapv (fn [v] {:type :int :value (int v)}) (:dims dpas))  ; [m n k] params
         :dims (:dims dpas)})
      ;; gate rejected (dtype/orientation/pitch) → portable register-tiled kernel
      (let [rt (sco/generate-regtiled-contraction-kernel sr out-sym :dtype dtype)
            [bm bn _bk] (:block rt)
            [M N _L] (:dims rt)]
        {:strategy :regtiled
         :fallback-reason (:reason dpas)
         :kernel-name (:kernel-name rt)
         :source (:source rt)
         :array-params (:array-params rt)            ; sorted-by-name (dims baked in source)
         :dtype (:dtype rt)
         :wg (:workgroup rt)
         :grid [(ceil-div N bn) (ceil-div M bm)]
         :scalar-args []                             ; regtiled bakes dims → no scalar params
         :dims (:dims rt)}))))

(defn- quant-descriptor
  "A launch descriptor for an int8 quant leaf (dp4a or quant-naive). Both are 1-D kernels with
   signature (…arrays…, out(f32), float scale, int _nseg): int8 operands in, dequantized f32 out."
  [strategy k out-dtype scale nseg]
  {:strategy strategy
   :kernel-name (:kernel-name k) :source (:source k)
   :array-params (:array-params k)          ; [row col] binding order (dp4a) / sorted (quant)
   :dtype :byte :out-dtype out-dtype
   :scheme (:scheme k)
   :wg [256 1] :grid [(ceil-div nseg 256) 1]
   :scalar-args [{:type :float :value (float scale)} {:type :int :value (int nseg)}]
   :dims (:dims k)})

(defn- route-quant
  "Route an int8 (:byte) contraction. 2-free/1-contract: try the dp4a peak leaf (requires the
   :nt operand layout — B stored [N,K], K-contiguous, K%4==0); if that orientation isn't met,
   fall to the quant naive-widening kernel (:nn). The emitters ASSERT their own layout
   requirements, so a thrown AssertionError is the (clean) 'not this leaf' signal — dp4a and
   quant-naive are complementary (:nt vs :nn). Non-2-free / multi-contract int8 is deferred."
  [contract-form out-sym scheme n-free n-contract nseg]
  (let [scale (get scheme :scale 1.0)]
    (if (and (= 2 n-free) (= 1 n-contract))
      (let [sr (cl/contract-form->segred contract-form :dtype :byte)
            dp4a (try (sco/generate-dp4a-contraction-kernel sr out-sym :scheme scheme)
                      (catch AssertionError _ nil))]
        (if dp4a
          (quant-descriptor :dp4a dp4a :float scale nseg)          ; :nt → peak int8 leaf
          (let [q (sco/generate-quant-contraction-kernel sr out-sym :scheme scheme)]
            (quant-descriptor :quant-naive q :float scale nseg))))  ; :nn → naive widening
      (throw (ex-info "route-quant: int8 supported for 2 free + 1 contract axes (C1 first cut)"
                      {:n-free n-free :n-contract n-contract})))))
