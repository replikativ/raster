(ns raster.compiler.ir.abstract-value
  "Backend-neutral logical value contracts.

   An AbstractValue describes program semantics: logical element type and shape, representation,
   placement and sharding constraints. It deliberately contains no BufferView or backend handle.
   Physical realization is a later concern; one logical value may become one dense buffer or an
   ordered tree of packed/scale/index buffers without teaching the allocator its numeric format."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.scalar-range :as scalar-range]))

(def value-kinds #{:tensor :record :opaque})

(defrecord AbstractValue
           [kind dtype shape logical-layout representation memory-space placement sharding
            ownership effects attributes])

(defn abstract-value?
  [value]
  (and value
       (= "raster.compiler.ir.abstract_value.AbstractValue" (.getName (class value)))))

(defn- dimension?
  [dimension]
  (or (and (integer? dimension) (not (neg? dimension)))
      (symbol? dimension)
      (seq? dimension)))

(defn validate!
  "Validate and return an AbstractValue. Tensor dimensions may remain symbolic; LinkPlan leaf
   BufferViews are the later realized, integer-shaped storage contract."
  [value]
  (when-not (abstract-value? value)
    (throw (ex-info "expected an AbstractValue"
                    {:reason :abstract-value-type :value value :actual (type value)})))
  (let [{:keys [kind dtype shape logical-layout representation memory-space placement sharding
                ownership effects attributes]}
        value]
    (when-not (contains? value-kinds kind)
      (throw (ex-info "abstract value has an invalid kind"
                      {:reason :abstract-value-kind :kind kind :allowed value-kinds})))
    (when (and (= :tensor kind) (nil? dtype))
      (throw (ex-info "a tensor abstract value requires a logical dtype"
                      {:reason :abstract-value-dtype :value value})))
    (when-not (or (nil? dtype) (keyword? dtype) (map? dtype))
      (throw (ex-info "abstract value dtype must be a keyword, descriptor map, or nil"
                      {:reason :abstract-value-dtype :dtype dtype})))
    (when-not (and (vector? shape) (every? dimension? shape))
      (throw (ex-info "abstract value shape must contain non-negative or symbolic dimensions"
                      {:reason :abstract-value-shape :shape shape})))
    (doseq [[field candidate] [[:logical-layout logical-layout]
                               [:representation representation]
                               [:placement placement]
                               [:sharding sharding]]]
      (when-not (or (nil? candidate) (map? candidate))
        (throw (ex-info "abstract value facet must be nil or a map"
                        {:reason :abstract-value-facet :field field :value candidate}))))
    (when-not (or (nil? memory-space) (keyword? memory-space))
      (throw (ex-info "abstract value memory space must be a keyword or nil"
                      {:reason :abstract-value-memory-space :memory-space memory-space})))
    (when-not (or (nil? ownership) (contains? #{:owned :borrowed :external} ownership))
      (throw (ex-info "abstract value has an invalid ownership contract"
                      {:reason :abstract-value-ownership :ownership ownership})))
    (when-not (set? effects)
      (throw (ex-info "abstract value effects must be an explicit set"
                      {:reason :abstract-value-effects :effects effects})))
    (when-not (map? attributes)
      (throw (ex-info "abstract value attributes must be a map"
                      {:reason :abstract-value-attributes :attributes attributes})))
    ;; This is a semantic entry/value precondition, not descriptive range
    ;; evidence. Consumers may use it only where the matching invocation
    ;; boundary enforces it. No domain is inferred from a name or shape use.
    (when (contains? attributes :scalar-domain)
      (let [domain (:scalar-domain attributes)
            scalar-type (when (and (keyword? dtype) (dtype/known? dtype))
                          (dtype/canon dtype))]
        (when-not (and (= :tensor kind) (= [] shape) (some? scalar-type)
                       (some? (scalar-range/for-dtype scalar-type))
                       (map? domain) (= #{:lower :upper} (set (keys domain)))
                       (integer? (:lower domain)) (integer? (:upper domain))
                       (<= (:lower domain) (:upper domain))
                       (scalar-range/contained-in-dtype? domain scalar-type))
          (throw (ex-info "scalar domain must be a closed integral interval within its scalar dtype"
                          {:reason :abstract-value-scalar-domain
                           :dtype dtype :shape shape :domain domain}))))))
  value)

(defn make
  [{:keys [kind dtype shape logical-layout representation memory-space placement sharding
           ownership effects attributes]
    :or {kind :tensor shape [] representation {:kind :plain} effects #{} attributes {}}}]
  (validate!
   (->AbstractValue kind dtype (vec shape) logical-layout representation memory-space placement
                    sharding ownership effects attributes)))

(defn tensor
  "Construct a tensor AbstractValue. `:representation` records numerical/storage semantics such
   as `{:kind :quantized :scheme :q4-k}` without prescribing its physical buffer leaves."
  [opts]
  (make (assoc opts :kind :tensor)))

(def ^:private storage-contract-facets
  [:kind :dtype :logical-layout :representation])

(defn storage-contract
  "Return the logical facets that must agree before two values can share raw element storage.

   Shape is excluded because relational analyses may prove symbolic dimensions equivalent;
   placement, ownership, and effects describe where/how a value is used rather than its element
   representation. Keyword dtypes are canonicalized through the compiler's single dtype table."
  [value]
  (validate! value)
  (cond-> (select-keys value storage-contract-facets)
    (keyword? (:dtype value)) (update :dtype dtype/canon)))

(defn storage-contract-compatible?
  "Whether two validated AbstractValues have the same logical raw-storage contract."
  [left right]
  (= (storage-contract left) (storage-contract right)))

(defn scalar-domain-implies?
  "Whether a source's declared domain entails a consumer's required domain.
   This proves only interval implication, not storage compatibility or ownership
   of an entry guard. An unrestricted producer cannot satisfy a narrower consumer."
  [source target]
  (validate! source)
  (validate! target)
  (let [offered (get-in source [:attributes :scalar-domain])
        required (get-in target [:attributes :scalar-domain])]
    (or (nil? required)
        (and (some? offered)
             (<= (:lower required) (:lower offered))
             (<= (:upper offered) (:upper required))))))

(defn- unknown-dimension?
  [dimension]
  (and (seq? dimension) (= 'unknown-dimension (first dimension)) (= 2 (count dimension))))

(defn merge-refinement
  "Refine facts for one SSA value by known shape dimensions and scalar-domain intersection.

   A concrete/static/symbolic dimension refines `(unknown-dimension id)`; two distinct known
   dimensions remain a contradiction. Other facets must agree exactly; disjoint domains conflict.
   This combines constraints, not evidence that an input guard or a producer established them."
  [left right]
  (let [left (validate! left)
        right (validate! right)]
    (when (and (= (-> left (assoc :shape []) (update :attributes dissoc :scalar-domain))
                   (-> right (assoc :shape []) (update :attributes dissoc :scalar-domain)))
               (= (count (:shape left)) (count (:shape right))))
      (let [shape (mapv (fn [left-dimension right-dimension]
                          (cond
                            (= left-dimension right-dimension) left-dimension
                            (unknown-dimension? left-dimension) right-dimension
                            (unknown-dimension? right-dimension) left-dimension
                            :else ::conflict))
                        (:shape left) (:shape right))
            a (get-in left [:attributes :scalar-domain])
            b (get-in right [:attributes :scalar-domain])
            domain (if (and a b)
                     {:lower (max (:lower a) (:lower b)) :upper (min (:upper a) (:upper b))}
                     (or a b))]
        (when (and (not (some #{::conflict} shape))
                   (or (nil? domain) (<= (:lower domain) (:upper domain))))
          (cond-> (assoc left :shape shape)
            domain (assoc-in [:attributes :scalar-domain] domain)))))))
