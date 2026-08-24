(ns raster.compiler.ir.attention-ad
  "Semantic reverse-mode differentiation contract for AttentionProblem.

   Differentiation happens before route scheduling and target emission. Q/K/V are the only
   differentiable values. Packed row metadata, logical positions, visibility, physical routes,
   allocation identities and events are discrete. K/V cotangents are physical-cache-shaped
   routed sums, so shared pages accumulate contributions rather than being overwritten."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.attention :as attention]))

(defrecord AttentionVJP
           [id primal output-cotangent cotangents active-values
            softmax-state-mode saved-lse accumulator-dtype])

(def ^:private differentiable-values #{:query :key :value})
(def ^:private softmax-state-modes #{:recompute :saved-lse})

(defn attention-vjp?
  [x]
  (instance? AttentionVJP x))

(defn differentiable-buffer-ids
  "Logical differentiable roles and their primal buffer identities."
  [problem]
  (let [{:keys [query k-pages v-pages]} (attention/validate! problem)]
    {:query (:values query) :key k-pages :value v-pages}))

(defn nondifferentiable-buffer-ids
  "Discrete metadata that must never receive tangents. Visibility is static data and therefore
   has no buffer identity here; allocation/event state is intentionally absent from AttentionProblem."
  [problem]
  (let [{:keys [query route]} (attention/validate! problem)]
    {:query-row-offsets (:row-offsets query)
     :query-positions (:positions query)
     :route (attention/route-buffer-ids route)}))

(defn validate!
  "Validate an AttentionVJP independently of any target schedule. Activity is an explicit subset
   of Q/K/V; a detached inference cache therefore requests only `:query`, while training normally
   requests all three. Quantized storage declines until its decode/gradient rule is declared."
  [vjp]
  (when-not (attention-vjp? vjp)
    (throw (ex-info "attention VJP must be an AttentionVJP"
                    {:reason :attention-invalid-vjp :actual (type vjp)})))
  (let [{:keys [id primal output-cotangent cotangents active-values
                softmax-state-mode saved-lse accumulator-dtype]} vjp
        {:keys [k-format v-format] :as primal} (attention/validate! primal)
        active-values (set active-values)]
    (when (nil? id)
      (throw (ex-info "attention VJP requires a stable identity"
                      {:reason :attention-vjp-missing-id})))
    (when (or (empty? active-values)
              (not (set/subset? active-values differentiable-values)))
      (throw (ex-info "attention VJP activity must be a nonempty subset of Q/K/V"
                      {:reason :attention-invalid-vjp-activity
                       :active-values active-values
                       :supported differentiable-values})))
    (when-not (= active-values (set (keys cotangents)))
      (throw (ex-info "attention VJP cotangent outputs must exactly match active values"
                      {:reason :attention-vjp-cotangent-activity-mismatch
                       :active-values active-values :cotangent-roles (set (keys cotangents))})))
    (when (some nil? (cons output-cotangent (vals cotangents)))
      (throw (ex-info "attention VJP requires every active cotangent buffer identity"
                      {:reason :attention-vjp-missing-buffer
                       :output-cotangent output-cotangent :cotangents cotangents})))
    (when-not (contains? softmax-state-modes softmax-state-mode)
      (throw (ex-info "attention VJP has an unsupported softmax-state policy"
                      {:reason :attention-invalid-softmax-state-mode
                       :mode softmax-state-mode :supported softmax-state-modes})))
    (case softmax-state-mode
      :recompute
      (when (some? saved-lse)
        (throw (ex-info "recomputed attention VJP cannot bind saved LSE"
                        {:reason :attention-unexpected-saved-lse :saved-lse saved-lse})))

      :saved-lse
      (when (nil? saved-lse)
        (throw (ex-info "saved-LSE attention VJP requires its buffer identity"
                        {:reason :attention-missing-saved-lse}))))
    (when-not (and (dtype/known? accumulator-dtype)
                   (dtype/fp-dtype? accumulator-dtype))
      (throw (ex-info "attention VJP accumulator must be floating point"
                      {:reason :attention-invalid-vjp-accumulator
                       :accumulator-dtype accumulator-dtype})))
    (when (or (not= :none (:quantization k-format))
              (not= :none (:quantization v-format)))
      (throw (ex-info "quantized attention requires an explicit decode/gradient rule"
                      {:reason :attention-quantized-vjp-undeclared
                       :k-format k-format :v-format v-format})))
    (let [primal-ids (set (keys (attention/buffer-specs primal)))
          derivative-ids (vec (concat [output-cotangent] (vals cotangents)
                                      (when saved-lse [saved-lse])))]
      (when-not (= (count derivative-ids) (count (distinct derivative-ids)))
        (throw (ex-info "attention derivative buffer identities must be distinct"
                        {:reason :attention-duplicate-vjp-buffer-identity
                         :buffers derivative-ids})))
      (when-let [aliases (seq (filter primal-ids derivative-ids))]
        (throw (ex-info "attention derivative buffers cannot alias semantic primal identities"
                        {:reason :attention-vjp-primal-buffer-alias :buffers (vec aliases)}))))
    vjp))

(defn make
  "Construct a checked semantic attention VJP. `cotangents` is keyed by any active subset of
   `:query`, `:key`, and `:value`. `:recompute` is the default softmax checkpoint policy."
  [{:keys [id primal output-cotangent cotangents active-values
           softmax-state-mode saved-lse accumulator-dtype]
    :or {softmax-state-mode :recompute accumulator-dtype :float}}]
  (let [active-values (set (or active-values (keys cotangents)))
        vjp (->AttentionVJP id primal output-cotangent (or cotangents {}) active-values
                            softmax-state-mode saved-lse (dtype/canon accumulator-dtype))]
    (validate! vjp)))

(defn differentiation-contract
  "Machine-readable activity, checkpointing, and accumulation semantics used by AD legalization
   and later backward scheduling."
  [vjp]
  (let [{:keys [primal active-values softmax-state-mode saved-lse]} (validate! vjp)]
    {:differentiable (differentiable-buffer-ids primal)
     :active-values active-values
     :nondifferentiable (nondifferentiable-buffer-ids primal)
     :cotangent-accumulation
     (into {}
           (map (fn [role]
                  [role (if (= :query role) :write :routed-sum)]))
           active-values)
     :softmax-state {:mode softmax-state-mode :buffer saved-lse}
     :inference-cache-detached?
     (empty? (set/intersection active-values #{:key :value}))}))

(defn buffer-specs
  "Backward graph buffers keyed by compiler identity. Primal output is not read by the stable
   softmax VJP; the output cotangent and optional saved LSE are explicit inputs."
  [vjp]
  (let [{:keys [primal output-cotangent cotangents active-values saved-lse accumulator-dtype]}
        (validate! vjp)
        {:keys [query output q-dtype k-dtype v-dtype output-dtype]} primal
        primal-specs (dissoc (attention/buffer-specs primal) output)
        shapes (attention/layouts primal)
        elements (fn [shape] (reduce * 1 shape))
        role-specs {:query {:role :output :dtype q-dtype :shape (:q shapes)
                            :elements (elements (:q shapes))}
                    :key {:role :output :dtype k-dtype :shape (:k-pages shapes)
                          :elements (elements (:k-pages shapes))}
                    :value {:role :output :dtype v-dtype :shape (:v-pages shapes)
                            :elements (elements (:v-pages shapes))}}
        derivative-specs
        (into {output-cotangent {:role :input :dtype output-dtype :shape (:output shapes)
                                 :elements (elements (:output shapes))}}
              (map (fn [role] [(get cotangents role) (get role-specs role)]))
              active-values)]
    (cond-> (merge primal-specs derivative-specs)
      saved-lse
      (assoc saved-lse {:role :input :dtype accumulator-dtype
                        :shape [(:total-tokens query) (:q-heads primal)]
                        :elements (* (:total-tokens query) (:q-heads primal))}))))
