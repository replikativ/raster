(ns raster.dl.gpt2-train
  "Trainable-shape GPT-2 forward+loss using HMap-typed weights.

  STATUS (updated 2026-06-24): the prior \"training is BLOCKED\" note is STALE.
  The full attention transformer (embeddings → layer-norm → causal-multi-head
  attention → residual → layer-norm → MLP/gelu → residual → final-LN → tied
  unembed → cross-entropy) trains end-to-end via `defmodel` +
  `raster.params/compile-train-step`. See test/raster/dl/gpt2_train_test.clj:
  `tiny-gpt2-loss` composes attention/LN/gelu over a single Params tree and
  `tiny-gpt2-compile-aot-train-step-converges` drives 10 fused Adam steps with
  strictly decreasing loss. Three confidence layers (lazy forward, lazy
  value+grad, fused compile-aot train step) all pass.

  This file keeps a hand-written inline-block FORWARD emitter as a lower-level
  demonstration; for trainable models prefer the `defmodel` + compile-train-step
  path shown in the test. (A residual goes through array-ops/array-add, which
  carries an explicit AD rrule; raw broadcast→reduce residuals also differentiate
  now — see raster.ad.reverse array-intermediate handling.)"
  (:require [raster.params :as rp]
            [raster.dl.gpt2 :as gpt2]
            [raster.dl.nn :as nn]
            [raster.dl.loss :as loss]
            [raster.dl.attention]
            [raster.dl.array-ops]))

;; ----------------------------------------------------------------------
;; HMap weights spec
;; ----------------------------------------------------------------------

(defn block-spec
  "HMap spec for one transformer block's weights (16 leaves)."
  []
  '(HMap :mandatory
         {:wq    (Param (Array double))   :bq    (Param (Array double))
          :wk    (Param (Array double))   :bk    (Param (Array double))
          :wv    (Param (Array double))   :bv    (Param (Array double))
          :wo    (Param (Array double))   :bo    (Param (Array double))
          :ln1-w (Param (Array double))   :ln1-b (Param (Array double))
          :ln2-w (Param (Array double))   :ln2-b (Param (Array double))
          :fc-w  (Param (Array double))   :fc-b  (Param (Array double))
          :proj-w (Param (Array double))  :proj-b (Param (Array double))}))

(defn weights-spec
  "Full GPT-2 weights tree spec for n-layer transformer.

  :unembed-b is a Frozen zero bias used as the unembedding-projection bias
  (GPT-2 ties wte as the unembedding weight and has no real bias). Carrying it
  as Frozen means the optimizer doesn't touch it but AD doesn't try to
  differentiate a literal zero array allocation inside the body."
  [n-layer]
  (let [bs (block-spec)]
    (list 'HMap :mandatory
          {:wte       '(Param (Array double))
           :wpe       '(Param (Array double))
           :ln-f-w    '(Param (Array double))
           :ln-f-b    '(Param (Array double))
           :unembed-b '(Frozen (Array double))
           :layers    (list 'HVec (vec (repeat n-layer bs)))})))

;; ----------------------------------------------------------------------
;; Convert raster.dl.gpt2/load-model output to tree shape
;; ----------------------------------------------------------------------

(defn model->tree
  "Convert a raster.dl.gpt2 model map (from load-model) into the tree shape
  expected by the gpt2-loss defmodel. Arrays in the result share storage with
  the model — mutations during training affect the original model."
  [model]
  (let [{:keys [layers wte wpe ln-f-w ln-f-b]} model
        n-layer (:n-layer model)]
    {:wte    wte    :wpe    wpe
     :ln-f-w ln-f-w :ln-f-b ln-f-b
     :layers (mapv layers (range n-layer))}))

(defn random-init-weights
  "Allocate a fresh random GPT-2 weights tree (Gaussian init, scaled by
  1/sqrt(d-model)). For training-from-scratch demos and tests."
  [{:keys [n-layer d-model n-head vocab-size max-position]}]
  (let [rng (java.util.Random. 42)
        d-inner (* 4 d-model)
        scale (/ 1.0 (Math/sqrt (double d-model)))
        rand-arr (fn [n]
                   (let [a (double-array n)]
                     (dotimes [i n] (aset a i (* scale (.nextGaussian rng))))
                     a))
        zeros (fn [n] (double-array n))
        ones  (fn [n] (let [a (double-array n)]
                        (java.util.Arrays/fill a 1.0) a))]
    {:wte       (rand-arr (* vocab-size d-model))
     :wpe       (rand-arr (* max-position d-model))
     :ln-f-w    (ones d-model)
     :ln-f-b    (zeros d-model)
     :unembed-b (zeros vocab-size)
     :layers    (vec (for [_ (range n-layer)]
                    {:wq (rand-arr (* d-model d-model))
                     :bq (zeros d-model)
                     :wk (rand-arr (* d-model d-model))
                     :bk (zeros d-model)
                     :wv (rand-arr (* d-model d-model))
                     :bv (zeros d-model)
                     :wo (rand-arr (* d-model d-model))
                     :bo (zeros d-model)
                     :ln1-w (ones d-model)
                     :ln1-b (zeros d-model)
                     :ln2-w (ones d-model)
                     :ln2-b (zeros d-model)
                     :fc-w  (rand-arr (* d-model d-inner))
                     :fc-b  (zeros d-inner)
                     :proj-w (rand-arr (* d-inner d-model))
                     :proj-b (zeros d-model)}))}))

;; ----------------------------------------------------------------------
;; Body generation: emit the layer chain
;; ----------------------------------------------------------------------

(defn- emit-layer-bindings
  "Per-layer let bindings expanding a pre-norm MLP-only block inline:
    pre-LN → linear → gelu → linear → residual.

  ATTENTION IS OMITTED in this demo: the standard multi-head causal-mha has
  manual head-split shuffles that don't differentiate, and even single-head
  attention composed of templated primitives (causal-scaled-dot-product-attn
  is now templated) hits a downstream compile-aot bug where AD-generated grad
  symbols don't resolve in the bytecode emit phase. Both gaps are real and
  documented; closing them is a focused follow-up.

  This MLP-only path validates that the rest of the GPT-2 training pipeline
  works: HMap weights, defmodel, value+grad through layer-norm/linear/gelu/
  array-add, fused compile-train-step, Adam updates."
  [n-layer d-model]
  (let [d-model (long d-model)
        n-inner (long (* 4 d-model))]
    (mapcat
     (fn [i]
       (let [layer-sym (symbol (str "layer-" i))
             h-in      (if (zero? i) 'h-init (symbol (str "h-" i)))
             h-ln      (symbol (str "h-" (inc i) "-ln"))
             h-fc      (symbol (str "h-" (inc i) "-fc"))
             h-fc-act  (symbol (str "h-" (inc i) "-fc-act"))
             h-proj    (symbol (str "h-" (inc i) "-proj"))
             h-out     (symbol (str "h-" (inc i)))]
         [layer-sym `(~'clojure.core/nth (:layers ~'w) ~i)
          h-ln     `(raster.dl.nn/layer-norm
                      ~h-in (:ln2-w ~layer-sym) (:ln2-b ~layer-sym)
                      ~'seq-len ~d-model 1e-5)
          h-fc     `(raster.dl.nn/linear
                      ~h-ln (:fc-w ~layer-sym) (:fc-b ~layer-sym)
                      ~'seq-len ~d-model ~n-inner)
          h-fc-act `(raster.dl.nn/gelu ~h-fc (clojure.core/* ~'seq-len ~n-inner))
          h-proj   `(raster.dl.nn/linear
                      ~h-fc-act (:proj-w ~layer-sym) (:proj-b ~layer-sym)
                      ~'seq-len ~n-inner ~d-model)
          h-out    `(raster.dl.array-ops/array-add
                      ~h-in ~h-proj (clojure.core/* ~'seq-len ~d-model))]))
     (range n-layer))))

(defn gpt2-loss-body
  "Generate the (single-head) GPT-2-style forward+cross-entropy-loss body."
  [n-layer d-model]
  (let [last-h  (symbol (str "h-" n-layer))
        d-model (long d-model)]
    `(~'let [~'h-init (raster.dl.gpt2/gpt2-embeddings
                        (:wte ~'w) (:wpe ~'w) ~'token-ids
                        ~'seq-len ~d-model)
             ~@(emit-layer-bindings n-layer d-model)
             ;; Final layer norm
             ~'h-final (raster.dl.nn/layer-norm
                         ~last-h (:ln-f-w ~'w) (:ln-f-b ~'w)
                         ~'seq-len ~d-model 1e-5)
             ;; Logits via tied wte. unembed-b is a Frozen zero bias in the spec.
             ~'logits (raster.dl.nn/linear
                        ~'h-final (:wte ~'w) (:unembed-b ~'w)
                        ~'seq-len ~d-model ~'vocab-size)
             ;; Cross-entropy: target is long[] of length seq-len
             ~'loss (raster.dl.loss/cross-entropy-loss
                      ~'logits ~'targets ~'seq-len ~'vocab-size)]
        ~'loss)))

;; ----------------------------------------------------------------------
;; Construct the defmodel for a given config
;; ----------------------------------------------------------------------

(defn make-gpt2-loss
  "Eval a deftm form for the given config and return its var. The var has
  structured-arg surface (takes a weights tree via the HMap arg type);
  compile-aot and value+grad pick up the same surface via tree metadata."
  [{:keys [n-layer d-model] :as _config}]
  (let [w-spec (weights-spec n-layer)
        body   (gpt2-loss-body n-layer d-model)
        sym    (symbol (str "gpt2-loss-" n-layer "L-" d-model "d"))
        form   `(raster.core/deftm ~sym
                  [~'w           :- ~w-spec
                   ~'token-ids   :- (~'Array ~'long)
                   ~'targets     :- (~'Array ~'long)
                   ~'seq-len     :- ~'Long
                   ~'vocab-size  :- ~'Long]
                  :- ~'Double
                  ~body)]
    (binding [*ns* (the-ns 'raster.dl.gpt2-train)]
      (eval form)
      (find-var (symbol "raster.dl.gpt2-train" (str sym))))))

(defn make-gpt2-train-step
  "Convenience: build the loss defmodel + compile a fused train step."
  [config]
  (let [loss-var (make-gpt2-loss config)]
    (assoc (rp/compile-train-step loss-var)
           :loss-var loss-var
           :config   config)))
