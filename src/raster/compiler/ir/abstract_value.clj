(ns raster.compiler.ir.abstract-value
  "Backend-neutral logical value contracts.

   An AbstractValue describes program semantics: logical element type and shape, representation,
   placement and sharding constraints. It deliberately contains no BufferView or backend handle.
   Physical realization is a later concern; one logical value may become one dense buffer or an
   ordered tree of packed/scale/index buffers without teaching the allocator its numeric format."
  (:require [raster.compiler.core.dtype :as dtype]))

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
                      {:reason :abstract-value-attributes :attributes attributes}))))
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

(defn- unknown-dimension?
  [dimension]
  (and (seq? dimension) (= 'unknown-dimension (first dimension)) (= 2 (count dimension))))

(defn merge-refinement
  "Join two facts for the same SSA value when they differ only by unknown shape dimensions.

   A concrete/static/symbolic dimension refines `(unknown-dimension id)`; two distinct known
   dimensions remain a contradiction. All non-shape AbstractValue facets must agree exactly, so
   this is a checked information refinement rather than dtype/layout/ownership inference. Returns
   nil when neither value refines the other."
  [left right]
  (let [left (validate! left)
        right (validate! right)]
    (when (and (= (assoc left :shape []) (assoc right :shape []))
               (= (count (:shape left)) (count (:shape right))))
      (let [shape (mapv (fn [left-dimension right-dimension]
                          (cond
                            (= left-dimension right-dimension) left-dimension
                            (unknown-dimension? left-dimension) right-dimension
                            (unknown-dimension? right-dimension) left-dimension
                            :else ::conflict))
                        (:shape left) (:shape right))]
        (when-not (some #{::conflict} shape)
          (assoc left :shape shape))))))
