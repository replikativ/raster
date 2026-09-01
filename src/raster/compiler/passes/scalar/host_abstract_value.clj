(ns raster.compiler.passes.scalar.host-abstract-value
  "Relational AbstractValue refinement for flat analyzed host bindings.

   This pass records shape equalities exposed by alength, clones, same-shaped allocation, and
   parallel maps. It does not execute host forms or infer numerical operators. Its first consumer
   is structured-loop functionalization: a mutating arraycopy becomes a loop carry only after this
   analysis proves a complete same-dtype state copy."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
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

(defn- array-allocation
  [expression]
  (when (and (seq? expression)
             (descriptor/alloc-op? (descriptor/semantic-op expression)))
    (let [semantic-operation (descriptor/semantic-op expression)
          operation (operation-name expression)
          length (first (descriptor/call-args expression))
          array-tag (or (get descriptor/alloc-sym->array-tag semantic-operation)
                        (get descriptor/alloc-sym->array-tag
                             (some-> operation symbol)))
          allocation-dtype (dtype/dtype-for-array-tag array-tag)]
      (when (and allocation-dtype length)
        {:dtype allocation-dtype :extent length}))))

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

(defn- infer-value
  [state symbol expression default-dtype]
  (let [values (:values state)
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
        allocation (array-allocation expression)]
    (cond
      alength-expression
      (let [state (assoc-in state [:values symbol]
                            (tensor (or cast-dtype
                                        (dtype/dtype-for-scalar-tag 'int)) []))]
        (refine-array-extent state alength-array symbol))

      clone-source
      (if-let [source-value (get values clone-source)]
        (assoc-in state [:values symbol]
                  (assoc source-value :shape (canonical-shape (:equalities state)
                                                              (:shape source-value))))
        state)

      allocation
      (assoc-in state [:values symbol]
                (tensor (:dtype allocation)
                        [(or (shape-extent state (:extent allocation))
                             (:extent allocation))]))

      (par/par-map-pure-form? expression)
      (let [{:keys [bound cast elem-type]} (par/extract-par-map-pure-info expression)
            cast-dtype (some-> cast descriptor/cast-result-tag
                               dtype/dtype-for-scalar-tag)
            elem-type (some-> elem-type dtype/canon)]
        (when (and elem-type cast-dtype (not= elem-type cast-dtype))
          (throw (ex-info "parallel map element type disagrees with its explicit cast"
                          {:reason :host-abstract-value-pmap-dtype
                           :element-type elem-type :cast-dtype cast-dtype
                           :expression expression})))
        (assoc-in state [:values symbol]
                  (tensor (or elem-type cast-dtype default-dtype)
                          [(or (shape-extent state bound) bound)])))

      (symbol? expression)
      (if-let [value (get values expression)]
        (assoc-in state [:values symbol] value)
        state)

      cast-dtype
      (assoc-in state [:values symbol] (tensor cast-dtype []))

      :else state)))

(defn analyze
  "Analyze one flat let/let* source form and return value/shape equality facts.

   Caller-provided AbstractValues are authoritative seeds. `:array-types` and `:scalar-types`
   create conservative parameter seeds only when no value was supplied."
  [source {:keys [values array-types scalar-types dtype]
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
        pairs (partition 2 bindings)]
    (reduce (fn [state [symbol expression]]
              (infer-value state symbol expression dtype))
            {:values (merge seed-arrays seed-scalars values)
             :equalities {}
             :diagnostics []}
            pairs)))

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
