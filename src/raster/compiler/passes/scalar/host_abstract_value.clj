(ns raster.compiler.passes.scalar.host-abstract-value
  "Relational AbstractValue refinement for flat analyzed host bindings.

   This pass records shape equalities exposed by alength, clones, same-shaped allocation, and
   parallel maps. It does not execute host forms or infer numerical operators. Its first consumer
   is structured-loop functionalization: a mutating arraycopy becomes a loop carry only after this
   analysis proves a complete same-dtype state copy."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.par :as par]))

(defn- tensor
  [dtype shape]
  (av/tensor {:dtype (dtype/canon dtype) :shape shape :representation {:kind :plain}}))

(defn- normalize-value
  [value]
  (av/validate! value)
  (cond-> value
    (keyword? (:dtype value)) (update :dtype dtype/canon)))

(defn- buffer-value? [value]
  (and (= :tensor (:kind value)) (vector? (:shape value)) (seq (:shape value))))

(defn- buffer-declaration-compatible? [symbol value]
  (let [tag (types/sym-type-tag symbol)]
    (or (nil? tag)
        (and (keyword? (:dtype value))
             (= (dtype/canon (:dtype value)) (some-> tag dtype/dtype-for-array-tag dtype/canon))))))

(defn bound-buffer-alias-source
  "Return a preceding stable-SSA BUFFER source for a bare reference or exact
   Object-input completion of that reference, otherwise nil. Validated AVs,
   not dtype-to-Class guesses, establish buffer/storage compatibility. This
   analysis fact is not portable ownership or permission to erase evaluation."
  [expression values available expected-value]
  (let [parts (completion/object-input-parts expression)
        _ (when parts (completion/object-value-declaration expression))
        source (if parts (:operand parts) expression)
        value (when (and (symbol? source) (contains? available source)) (get values source))]
    (when value
      (av/validate! value)
      (when expected-value (av/validate! expected-value))
      (when (and (buffer-value? value)
                 (buffer-declaration-compatible? source value)
                 (buffer-declaration-compatible? expression value)
                 (or (nil? expected-value)
                     (and (buffer-value? expected-value)
                          (av/storage-contract-compatible? value expected-value))))
        source))))

(defn- scalar-cast-dtype
  [expression]
  (when (and (seq? expression)
             (descriptor/cast-op? (descriptor/semantic-op expression)))
    (dtype/dtype-for-scalar-tag
     (descriptor/cast-result-tag (descriptor/semantic-op expression)))))

(defn- operation-name
  [expression]
  (some-> expression descriptor/semantic-op name))

(defn- source-shaped-allocation
  [expression]
  (when (seq? expression)
    (let [operation (descriptor/semantic-op expression)
          arguments (descriptor/call-args expression)]
      (when (and (descriptor/alloc-op? operation)
                 (nil? (or (get descriptor/alloc-sym->array-tag operation)
                           (get descriptor/alloc-sym->array-tag
                                (some-> operation name symbol))))
                 (= 1 (count arguments)))
        (first arguments)))))

(defn- fresh-allocation-value
  [source-value shape]
  (av/tensor
   {:dtype (:dtype source-value)
    :shape shape
    :logical-layout (:logical-layout source-value)
    :representation (:representation source-value)}))

(defn- array-allocation
  [expression]
  (when (and (seq? expression)
             (descriptor/alloc-op? (descriptor/semantic-op expression)))
    (let [semantic-operation (descriptor/semantic-op expression)
          operation (operation-name expression)
          arguments (descriptor/call-args expression)
          array-tag (or (get descriptor/alloc-sym->array-tag semantic-operation)
                        (get descriptor/alloc-sym->array-tag
                             (some-> operation symbol)))
          allocation-dtype (dtype/dtype-for-array-tag array-tag)
          initialization (descriptor/allocation-initialization semantic-operation)]
      (cond
        (and allocation-dtype (= 1 (count arguments)))
        {:dtype allocation-dtype :extent (first arguments)}

        ;; Parametric allocators use the first argument only as a dtype exemplar. Their retained
        ;; AbstractValue and final length are sufficient for host planning; no value dependency on
        ;; the exemplar exists for :zero/:unspecified initialization.
        (and (contains? #{:zero :unspecified} initialization)
             (<= 2 (count arguments)))
        {:source (first arguments) :extent (last arguments)}

        :else nil))))

(defn- canonical-extent
  [equalities extent]
  (loop [extent extent seen #{}]
    (let [next (get equalities extent extent)]
      (if (or (= next extent) (contains? seen next))
        extent
        (recur next (conj seen extent))))))

(defn- canonical-shape
  [equalities shape]
  (mapv #(canonical-extent equalities %) shape))

(defn- value-shape
  [analysis value]
  (when value
    (canonical-shape (:equalities analysis) (:shape value))))

(defn- shape-extent
  [{:keys [values equalities]} expression]
  (let [expression (descriptor/unwrap-int-cast expression)]
    (if (and (seq? expression)
             (descriptor/alength-op? (descriptor/semantic-op expression))
             (= 1 (count (descriptor/call-args expression))))
      (let [array (first (descriptor/call-args expression))
            shape (:shape (get values array))]
        (when (= 1 (count shape))
          (canonical-extent equalities (first shape))))
      (canonical-extent equalities expression))))

(defn- refine-array-extent
  [state array extent]
  (if-let [value (get-in state [:values array])]
    (if (= 1 (count (:shape value)))
      (let [old (first (:shape value))]
        (-> state
            (assoc-in [:equalities old] extent)
            (assoc-in [:values array] (assoc value :shape [extent]))))
      state)
    state))

(defn- retain-extent-equality
  [state expression extent]
  (let [normalized (descriptor/unwrap-int-cast expression)]
    (cond-> state
      (not= normalized extent) (assoc-in [:equalities normalized] extent)
      (not= expression normalized) (assoc-in [:equalities expression] extent))))

(defn- infer-value
  [state symbol expression default-dtype bound-buffer-source]
  (let [values (:values state)
        retained-array-dtype
        (or (some-> symbol types/sym-type-tag dtype/dtype-for-array-tag)
            (some-> expression types/sym-type-tag dtype/dtype-for-array-tag))
        retained-scalar-dtype
        (or (some-> symbol types/sym-type-tag dtype/dtype-for-scalar-tag)
            (some-> expression types/sym-type-tag dtype/dtype-for-scalar-tag))
        cast-dtype (scalar-cast-dtype expression)
        inner (when (and cast-dtype (= 1 (count (descriptor/call-args expression))))
                (first (descriptor/call-args expression)))
        wrapped-alength-array
        (when (and inner (seq? inner)
                   (descriptor/alength-op? (descriptor/semantic-op inner))
                   (= 1 (count (descriptor/call-args inner))))
          (first (descriptor/call-args inner)))
        direct-alength?
        (and (seq? expression)
             (descriptor/alength-op? (descriptor/semantic-op expression)))
        alength-expression (cond wrapped-alength-array inner
                                 direct-alength? expression)
        alength-array (when alength-expression
                        (first (descriptor/call-args alength-expression)))
        clone-source (source-shaped-allocation expression)
        allocation (array-allocation expression)
        destination-write
        (cond
          (par/par-stencil-form? expression)
          (select-keys (par/extract-par-stencil-info expression)
                       [:out :bound :cast :elem-type])

          (par/par-map-form? expression)
          (let [{:keys [out bound cast elem-type offset]}
                (par/extract-par-map-info expression)]
            (when-not offset {:out out :bound bound :cast cast :elem-type elem-type})))]
    (cond
      ;; The opt-in source-capacity view requires an actual preceding buffer
      ;; exemplar. A retained result dtype cannot replace that dependency. The
      ;; ordinary relational analysis deliberately retains its historical rules.
      (and (:capacity-only? state)
           (or clone-source (:source allocation))
           (not (buffer-value? (get values (or clone-source (:source allocation))))))
      state

      alength-expression
      (let [state (assoc-in state [:values symbol]
                            (tensor (or cast-dtype
                                        (get-in values [symbol :dtype])
                                        retained-scalar-dtype
                                        (dtype/dtype-for-scalar-tag 'int)) []))]
        (refine-array-extent state alength-array symbol))

      clone-source
      (if-let [source-value (get values clone-source)]
        (assoc-in state [:values symbol]
                  (fresh-allocation-value
                   source-value
                   (canonical-shape (:equalities state) (:shape source-value))))
        state)

      allocation
      (let [extent (or (shape-extent state (:extent allocation))
                       (:extent allocation))
            allocation-dtype (or (:dtype allocation)
                                 (get-in values [(:source allocation) :dtype])
                                 retained-array-dtype)]
        (if allocation-dtype
          (-> state
              (retain-extent-equality (:extent allocation) extent)
              (assoc-in [:values symbol] (tensor allocation-dtype [extent])))
          state))

      (par/par-map-pure-form? expression)
      (let [{:keys [bound cast elem-type]} (par/extract-par-map-pure-info expression)
            cast-dtype (some-> cast descriptor/cast-result-tag
                               dtype/dtype-for-scalar-tag)
            elem-type (some-> elem-type dtype/canon)
            extent (or (shape-extent state bound) bound)]
        (when (and elem-type cast-dtype (not= elem-type cast-dtype))
          (throw (ex-info "parallel map element type disagrees with its explicit cast"
                          {:reason :host-abstract-value-pmap-dtype
                           :element-type elem-type :cast-dtype cast-dtype
                           :expression expression})))
        (-> state
            (retain-extent-equality bound extent)
            (assoc-in [:values symbol]
                      (tensor (or elem-type cast-dtype default-dtype) [extent]))))

      destination-write
      (let [{:keys [bound cast elem-type]} destination-write
            cast-dtype (some-> cast descriptor/cast-result-tag
                               dtype/dtype-for-scalar-tag)
            elem-type (some-> elem-type dtype/canon)
            extent (or (shape-extent state bound) bound)]
        (when (and elem-type cast-dtype (not= elem-type cast-dtype))
          (throw (ex-info "parallel write element type disagrees with its explicit cast"
                          {:reason :host-abstract-value-write-dtype
                           :element-type elem-type :cast-dtype cast-dtype
                           :expression expression})))
        (-> state
            (retain-extent-equality bound extent)
            (assoc-in [:values symbol]
                      (tensor (or elem-type cast-dtype default-dtype) [extent]))))

      (symbol? expression)
      (if-let [value (get values expression)]
        (if (and (contains? state :available) (buffer-value? value)
                 (not= expression bound-buffer-source))
          state
          (cond-> (assoc-in state [:values symbol] value)
            (and (= [] (:shape value))
                 (contains? #{:int :long} (dtype/canon (:dtype value))))
            (assoc-in [:equalities expression] symbol)))
        state)

      cast-dtype
      (assoc-in state [:values symbol] (tensor cast-dtype []))

      retained-scalar-dtype
      ;; The walker/TypedClojure result type is authoritative. We deliberately do not inspect the
      ;; operator here: this pass propagates retained contracts and relational shapes; it is not a
      ;; second scalar type-inference engine.
      (assoc-in state [:values symbol] (tensor retained-scalar-dtype []))

      :else state)))

(defn ^:no-doc begin-analysis
  "Initialize one ordered host analysis. The returned compiler-local work state
   is not portable alias/ownership evidence. Advance each emitted binding once."
  [source {:keys [values array-types scalar-types dtype lexical-inputs non-buffer-bindings] :as options
           :or {values {} array-types {} scalar-types {} dtype :double}}]
  (when-not (form/binding-form? source)
    (throw (ex-info "host AbstractValue analysis requires a flat binding form"
                    {:reason :host-abstract-value-source :source source})))
  (let [[_ bindings] source]
    (when-not (and (vector? bindings) (even? (count bindings)))
      (throw (ex-info "host AbstractValue analysis requires an even binding vector"
                      {:reason :host-abstract-value-bindings :bindings bindings}))))
  (let [seed-arrays (into {}
                          (map (fn [[id element-dtype]]
                                 [id (tensor element-dtype [(list 'extent id)])]))
                          array-types)
        seed-scalars (into {}
                           (map (fn [[id scalar-dtype]] [id (tensor scalar-dtype [])]))
                           scalar-types)
        values (into {} (map (fn [[id value]] [id (normalize-value value)])) values)
        [_ bindings] source
        pairs (partition 2 bindings)
        lexical? (contains? options :lexical-inputs)
        _ (when (and lexical? (not (and (or (sequential? lexical-inputs) (set? lexical-inputs))
                                      (every? symbol? lexical-inputs))))
            (throw (ex-info "host alias analysis requires explicit lexical input names"
                            {:reason :host-abstract-value-lexical-inputs})))
        _ (when (and (contains? options :non-buffer-bindings)
                     (not (and (or (vector? non-buffer-bindings) (set? non-buffer-bindings))
                               (every? symbol? non-buffer-bindings))))
            (throw (ex-info "host alias exclusions require certified binding names"
                            {:reason :host-abstract-value-non-buffer-bindings})))
        binders (map first pairs)
        ineligible (into (into (set (filter (set lexical-inputs) binders)) non-buffer-bindings)
                         (keep (fn [[id occurrences]] (when (< 1 occurrences) id)))
                         (frequencies binders))]
    {:dtype dtype :lexical? lexical? :ineligible ineligible
     :analysis (cond-> {:values (merge seed-arrays seed-scalars values)
                       :equalities {} :diagnostics []}
                 lexical? (assoc :available (set lexical-inputs) :aliases {}))}))

(defn ^:no-doc binding-buffer-alias-source
  "Query the existing ordered binding context without advancing it."
  [{:keys [analysis lexical? ineligible]} symbol expression]
  (when (and lexical? (symbol? symbol) (not (contains? ineligible symbol)))
    (when-let [source (bound-buffer-alias-source expression (:values analysis)
                                                (apply disj (:available analysis) ineligible)
                                                (get-in analysis [:values symbol]))]
      (when (buffer-declaration-compatible? symbol (get-in analysis [:values source])) source))))

(defn ^:no-doc advance-binding
  "Advance an existing ordered analysis, optionally excluding a certified
   non-buffer host result. Logical tensor facts remain intact."
  ([context symbol expression] (advance-binding context symbol expression false))
  ([{:keys [analysis dtype lexical? ineligible] :as context} symbol expression non-buffer?]
   (let [ineligible (cond-> ineligible non-buffer? (conj symbol))
         reference (if-let [parts (completion/object-input-parts expression)] (:operand parts) expression)
         refused-buffer-reference?
         (and (symbol? reference)
              (or (buffer-value? (get-in analysis [:values reference]))
                  (buffer-value? (get-in analysis [:values symbol]))))
         source (binding-buffer-alias-source (assoc context :ineligible ineligible) symbol expression)
         analysis (infer-value analysis symbol (or source expression) dtype source)]
     (assoc context :ineligible ineligible
            :analysis (cond-> analysis
                        source (assoc-in [:aliases symbol] source)
                        (and lexical? (or source (not refused-buffer-reference?)))
                        (update :available conj symbol))))))

(defn ^:no-doc enter-scalar-parameter
  "Enter a fresh, unknown scalar in a child lexical analysis context.

   This retains preceding relational facts, but establishes neither a value
   equality nor buffer ownership. The caller must alpha-rename the parameter:
   even an occurrence used only as a symbolic shape may not be shadowed."
  [{:keys [analysis lexical? ineligible] :as context} id scalar-dtype]
  (when-not (and (symbol? id)
                 (not (contains? ineligible id))
                 (not-any? #{id} (tree-seq coll? seq analysis)))
    (throw (ex-info "host analysis parameter requires a fresh lexical identity"
                    {:reason :host-abstract-value-parameter :parameter id})))
  (cond-> (assoc-in context [:analysis :values id] (tensor scalar-dtype []))
    lexical? (update-in [:analysis :available] conj id)))

(defn ^:no-doc finish-analysis [{:keys [analysis]}]
  (update analysis :values
          (fn [values]
            (into {} (map (fn [[id value]]
                            [id (update value :shape
                                        #(canonical-shape (:equalities analysis) %))])) values))))

(defn analyze
  "Analyze flat host bindings using authoritative AV seeds and retained shapes.
   Explicit :lexical-inputs admits stable-SSA aliases in source order;
   :non-buffer-bindings excludes certified non-buffer results. Only :aliases,
   not lexical availability or logical tensor shape alone, conveys safe aliases."
  [source options]
  (finish-analysis
   (reduce (fn [context [id expression]] (advance-binding context id expression))
           (begin-analysis source options) (partition 2 (second source)))))

(defn analyze-capacities
  "Analyze source capacities in binding order, retaining definition-time AVs.

   Unlike analyze, seed maps describe only explicit lexical inputs, not results
   of a later numerical analysis. :reference-aliases optionally supplies certified
   source-return facts {binding preceding-buffer}; these are analysis-only facts,
   not host type, purity, ownership, or permission to erase evaluation. The final
   :values/:equalities remain relational; :definition-values never receives later
   equality refinements and excludes certified non-buffer or unstable identities.
   Capacity transfers cannot consume logical-only tensor results; :aliases and
   :available likewise describe only the source-capacity view."
  [source {:keys [lexical-inputs reference-aliases] :as options}]
  (when-not (contains? options :lexical-inputs)
    (throw (ex-info "source capacity analysis requires explicit lexical inputs"
                    {:reason :host-abstract-value-lexical-inputs})))
  (when-not (or (not (contains? options :reference-aliases))
                (and (map? reference-aliases)
                     (every? (fn [[id reference]] (and (symbol? id) (symbol? reference)))
                             reference-aliases)))
    (throw (ex-info "source capacity reference aliases require symbol identities"
                    {:reason :host-abstract-value-reference-aliases})))
  ;; Let the existing initializer validate the source and lexical contract before
  ;; filtering. No future binder's seed may participate in any transfer rule.
  (let [validated (begin-analysis source options)
        inputs (set lexical-inputs)
        options (reduce (fn [options key]
                          (if (contains? options key)
                            (update options key select-keys inputs)
                            options))
                        options [:values :array-types :scalar-types])
        context (begin-analysis source options)
        pairs (partition 2 (second source))
        binders (set (map first pairs))
        reject! (fn [id reference]
                  (throw (ex-info "source capacity alias requires a preceding eligible buffer"
                                  {:reason :host-abstract-value-reference-alias
                                   :binding id :reference reference})))
        _ (doseq [[id reference] reference-aliases]
            (when (or (not (contains? binders id))
                      (contains? (:ineligible validated) id)
                      (contains? (:ineligible validated) reference))
              (reject! id reference)))
        initial-definitions (select-keys (get-in context [:analysis :values])
                                         (remove (:ineligible context) inputs))
        capacity-context (-> context
                             (assoc-in [:analysis :values] initial-definitions)
                             (assoc-in [:analysis :capacity-only?] true)
                             (update-in [:analysis :available]
                                        #(apply disj % (:ineligible context))))
        result
        (reduce
         (fn [{:keys [context logical-context definitions]} [id expression]]
           (let [reference (get reference-aliases id)
                 explicit-reference (if-let [parts (completion/object-input-parts expression)]
                                      (:operand parts) expression)
                 _ (when (and reference
                              (or (and (symbol? explicit-reference)
                                       (not= explicit-reference reference))
                                  (not= reference
                                        (binding-buffer-alias-source context id reference))
                                  (not (buffer-declaration-compatible?
                                        expression
                                        (get-in context [:analysis :values reference])))))
                     (reject! id reference))
                 ;; An unstable/excluded binder must not publish relational
                 ;; equalities either (for example n -> a repeated alength id).
                 next-context (if (contains? (:ineligible context) id)
                                context
                                (advance-binding context id (or reference expression)))
                 eligible? (and (contains? (get-in next-context [:analysis :available]) id)
                                (not (contains? (:ineligible next-context) id)))
                 ;; Earlier capacity entries were already checked at definition.
                 ;; Prune only this binding, not the entire growing value map.
                 next-context (cond-> next-context
                                (not eligible?) (update-in [:analysis :values] dissoc id)
                                (not eligible?) (update-in [:analysis :available] disj id))
                 value (when eligible?
                         (when-let [value (get-in next-context [:analysis :values id])]
                           (update value :shape
                                   #(canonical-shape
                                     (get-in next-context [:analysis :equalities]) %))))]
             {:context next-context
              :logical-context (advance-binding logical-context id (or reference expression))
              :definitions (cond-> definitions value (assoc id value))}))
         {:context capacity-context :logical-context context :definitions initial-definitions} pairs)]
    (assoc (finish-analysis (:logical-context result))
           :aliases (get-in result [:context :analysis :aliases])
           :available (get-in result [:context :analysis :available])
           :definition-values (:definitions result))))

(defn- zero-index?
  [expression]
  (= 0 (descriptor/unwrap-int-cast expression)))

(defn full-array-copy
  "Return a certified full-state copy descriptor, or nil.

   `expression` must be an arraycopy whose source/destination positions are zero. Both arrays must
   have equal rank-one dtype/shape contracts, and the copied length must equal that refined shape."
  [analysis expression]
  (when (and (seq? expression)
             (contains? #{'System/arraycopy 'java.lang.System/arraycopy}
                        (descriptor/semantic-op expression))
             (= 5 (count (descriptor/call-args expression))))
    (let [[source source-position destination destination-position length]
          (descriptor/call-args expression)
          source-value (get-in analysis [:values source])
          destination-value (get-in analysis [:values destination])
          copied-extent (shape-extent analysis length)
          source-shape (value-shape analysis source-value)
          destination-shape (value-shape analysis destination-value)]
      (when (and (symbol? source) (symbol? destination)
                 (zero-index? source-position) (zero-index? destination-position)
                 (= :tensor (:kind source-value) (:kind destination-value))
                 (av/storage-contract-compatible? source-value destination-value)
                 (= 1 (count source-shape) (count destination-shape))
                 (= source-shape destination-shape [copied-extent]))
        {:source source :destination destination :extent copied-extent
         :dtype (:dtype source-value)}))))

(defn full-array-write
  "Return a certified descriptor for one complete rank-one destination-writing SOAC.

   This is a storage proof, not an operation-name inference rule: the parallel dialect extractor
   supplies the destination and iteration extent, while retained AbstractValues prove that the
   operation covers the destination's complete logical storage contract. Offset maps deliberately
   remain outside this proof."
  [analysis result expression]
  (let [{:keys [out bound kind]}
        (cond
          (par/par-stencil-form? expression)
          (assoc (select-keys (par/extract-par-stencil-info expression) [:out :bound])
                 :kind :stencil)

          (par/par-map-form? expression)
          (let [{:keys [out bound offset]} (par/extract-par-map-info expression)]
            (when-not offset {:out out :bound bound :kind :map}))

          :else nil)
        destination-value (get-in analysis [:values out])
        result-value (get-in analysis [:values result])
        destination-shape (value-shape analysis destination-value)
        result-shape (value-shape analysis result-value)
        write-extent (shape-extent analysis bound)]
    (when (and (symbol? out) (symbol? result)
               (= :tensor (:kind destination-value) (:kind result-value))
               (av/storage-contract-compatible? destination-value result-value)
               (= 1 (count destination-shape) (count result-shape))
               (= destination-shape result-shape [write-extent]))
      {:destination out :extent write-extent :dtype (:dtype destination-value) :kind kind})))
