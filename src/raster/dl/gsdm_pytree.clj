(ns raster.dl.gsdm-pytree
  "GSDM forward+loss as a single defmodel using HMap-typed weights.

  Replaces the body-of-flat-args approach in raster.dl.gsdm (gen-gsdm-loss-body
  + compile-gsdm-train-fn) with an HMap-based defmodel. Compilation produces
  the same flat JVM method via raster.compiler.core.params-flatten — but the
  user works with structured weights and there is no name mangling, no manual
  param ordering, no parallel m/v arg lists.

  The block primitives (timestep-mlp, resnet-block, graph-attention-multihead,
  flat-embed, flat-unembed) keep their flat array-arg signatures from
  raster.dl.gsdm; the outer defmodel composes them with HMap path access."
  (:require [raster.params :as rp]
            [raster.dl.gsdm :as gsdm]
            [raster.dl.array-ops :as array-ops]))

;; ----------------------------------------------------------------------
;; HMap weight spec
;; ----------------------------------------------------------------------

(defn layer-spec
  "HMap spec for one transformer block's weights."
  []
  '(HMap :mandatory
         {:res-W1  (Param (Array double))  :res-b1  (Param (Array double))
          :res-W2  (Param (Array double))  :res-b2  (Param (Array double))
          :res-Wt  (Param (Array double))  :res-bt  (Param (Array double))
          :res-g1  (Param (Array double))  :res-bn1 (Param (Array double))
          :res-g2  (Param (Array double))  :res-bn2 (Param (Array double))
          :attn-Wq (Param (Array double))  :attn-bq (Param (Array double))
          :attn-Wk (Param (Array double))  :attn-bk (Param (Array double))
          :attn-Wv (Param (Array double))  :attn-bv (Param (Array double))
          :attn-g  (Param (Array double))  :attn-b  (Param (Array double))}))

(defn weights-spec
  "HMap spec for the full GSDM weights pytree at given n-layers."
  [n-layers]
  (let [ls (layer-spec)]
    (list 'HMap :mandatory
          {:temb-W1     '(Param (Array double))   :temb-b1     '(Param (Array double))
           :temb-W2     '(Param (Array double))   :temb-b2     '(Param (Array double))
           :embed-We    '(Param (Array double))   :embed-be    '(Param (Array double))
           :embed-space '(Param (Array double))   :embed-state '(Param (Array double))
           :unembed-Wu1 '(Param (Array double))   :unembed-bu1 '(Param (Array double))
           :unembed-Wu2 '(Param (Array double))   :unembed-bu2 '(Param (Array double))
           :layers      (list 'HVec (vec (repeat n-layers ls)))})))

;; ----------------------------------------------------------------------
;; Convert flat init-gsdm-weights map -> structured pytree value
;; ----------------------------------------------------------------------

(defn flat->pytree
  "Convert init-gsdm-weights output (flat keyword-keyed) into the pytree shape
  expected by gsdm-loss defmodel."
  [weights n-layers]
  (let [layer-keys [:res-W1 :res-b1 :res-W2 :res-b2 :res-Wt :res-bt
                    :res-g1 :res-bn1 :res-g2 :res-bn2
                    :attn-Wq :attn-bq :attn-Wk :attn-bk
                    :attn-Wv :attn-bv :attn-g :attn-b]]
    {:temb-W1     (:temb-W1 weights)     :temb-b1     (:temb-b1 weights)
     :temb-W2     (:temb-W2 weights)     :temb-b2     (:temb-b2 weights)
     :embed-We    (:embed-We weights)    :embed-be    (:embed-be weights)
     :embed-space (:embed-space weights) :embed-state (:embed-state weights)
     :unembed-Wu1 (:unembed-Wu1 weights) :unembed-bu1 (:unembed-bu1 weights)
     :unembed-Wu2 (:unembed-Wu2 weights) :unembed-bu2 (:unembed-bu2 weights)
     :layers (mapv (fn [i]
                     (into {} (map (fn [k]
                                     [k (get weights
                                             (keyword (str "layer-" i "-" (name k))))]))
                           layer-keys))
                   (range n-layers))}))

;; ----------------------------------------------------------------------
;; Body generation: emit the layer chain
;; ----------------------------------------------------------------------

(defn- emit-layer-bindings
  "For each layer index, emit let-binding pairs that:
   - alias the layer sub-pytree
   - run resnet-block
   - run graph-attention-multihead
  All accesses use HMap path access — pre-flatten resolves them to flat args."
  [n-layers]
  (mapcat
   (fn [i]
     (let [layer-sym (symbol (str "layer-" i))
           h-in      (if (zero? i) 'h-init (symbol (str "h-" i)))
           h-res     (symbol (str "h-" (inc i) "-res"))
           h-out     (symbol (str "h-" (inc i)))]
       [layer-sym `(~'clojure.core/nth (:layers ~'w) ~i)
        h-res `(raster.dl.gsdm/resnet-block
                 ~h-in ~'temb
                 (:res-W1 ~layer-sym) (:res-b1 ~layer-sym)
                 (:res-W2 ~layer-sym) (:res-b2 ~layer-sym)
                 (:res-Wt ~layer-sym) (:res-bt ~layer-sym)
                 (:res-g1 ~layer-sym) (:res-bn1 ~layer-sym)
                 (:res-g2 ~layer-sym) (:res-bn2 ~layer-sym)
                 ~'n-vars ~'emb-dim-val)
        h-out `(raster.dl.gsdm/graph-attention-multihead
                 ~h-res
                 (:attn-Wq ~layer-sym) (:attn-bq ~layer-sym)
                 (:attn-Wk ~layer-sym) (:attn-bk ~layer-sym)
                 (:attn-Wv ~layer-sym) (:attn-bv ~layer-sym)
                 (:attn-g  ~layer-sym) (:attn-b  ~layer-sym)
                 ~'src-edges ~'dst-edges
                 ~'n-vars ~'n-edges ~'emb-dim-val ~'n-heads-val)]))
   (range n-layers)))

(defn gsdm-loss-body
  "Generate the full GSDM loss body S-expression for given config.

  The body assumes outer scope binds: w, values, spaces, target, states,
  pos-emb, src-edges, dst-edges, t, n-vars, n-edges (the defmodel arg
  vector). It uses (:k w) / (nth (:layers w) i) accesses; the pre-flatten
  pass converts these to flat-arg references during compile."
  [n-layers emb-dim n-heads n-spaces]
  (let [last-h (symbol (str "h-" n-layers))]
    `(~'let [~'emb-dim-val ~(long emb-dim)
           ~'n-heads-val ~(long n-heads)
           ~'n-spaces-val ~(long n-spaces)
           ~'n-states-val ~(long gsdm/N-STATES)
           ~'h-init (raster.dl.gsdm/flat-embed
                      ~'values (:embed-space ~'w) ~'spaces
                      (:embed-state ~'w) ~'states ~'pos-emb
                      (:embed-We ~'w) (:embed-be ~'w)
                      ~'n-vars ~'emb-dim-val ~'n-spaces-val ~'n-states-val)
           ~'temb (raster.dl.gsdm/timestep-mlp
                    ~'t (:temb-W1 ~'w) (:temb-b1 ~'w)
                    (:temb-W2 ~'w) (:temb-b2 ~'w) ~'emb-dim-val)
           ~@(emit-layer-bindings n-layers)
           ~'pred (raster.dl.gsdm/flat-unembed
                    ~last-h
                    (:unembed-Wu1 ~'w) (:unembed-bu1 ~'w)
                    (:unembed-Wu2 ~'w) (:unembed-bu2 ~'w)
                    ~'n-vars ~'emb-dim-val)
           ~'loss (raster.dl.array-ops/masked-mse-loss
                    ~'pred ~'target ~'states ~'n-vars)]
       ~'loss)))

;; ----------------------------------------------------------------------
;; Construct the defmodel for a given config
;; ----------------------------------------------------------------------

(defn make-gsdm-loss
  "Eval a defmodel form for the given config and return its var.
  The var has structured-arg surface (takes a weights pytree); compile-aot
  and value+grad pick up the same surface via the params metadata."
  [{:keys [n-layers emb-dim n-heads n-spaces]
    :or {n-layers 2 emb-dim 8 n-heads 2 n-spaces 1}}]
  (let [w-spec (weights-spec n-layers)
        body (gsdm-loss-body n-layers emb-dim n-heads n-spaces)
        sym  (symbol (str "gsdm-loss-" n-layers "L-" emb-dim "d-" n-heads "h-" n-spaces "s"))
        form `(raster.params/defmodel ~sym
                [~'w :- (~'Params ~w-spec)
                 ~'values :- (~'Array ~'double)
                 ~'spaces :- (~'Array ~'long)
                 ~'target :- (~'Array ~'double)
                 ~'states :- (~'Array ~'long)
                 ~'pos-emb :- (~'Array ~'double)
                 ~'src-edges :- (~'Array ~'long)
                 ~'dst-edges :- (~'Array ~'long)
                 ~'t :- ~'Double
                 ~'n-vars :- ~'Long
                 ~'n-edges :- ~'Long]
                :- ~'Double
                ~body)]
    (binding [*ns* (the-ns 'raster.dl.gsdm-pytree)]
      (eval form))
    (find-var (symbol "raster.dl.gsdm-pytree" (str sym)))))
