(ns raster.compiler.ir.soac-dialect
  "Typed, functional S-expression view of Raster's SOAC middle end.

   The expression spine is deliberately small and pattern-friendly.  Stable value/equation
   identity and compiler facts live in the explicit program envelope, never in Clojure metadata.
   ParallelProgram equations retain this verified functional program as their semantic algorithm;
   scheduled SegOps remain a separate ordered field.

   Canonical forms:

     (soac-program facts
       [(= equation-id [result ...]
           (scalar {:dtypes [:long ...]} [capture ...]
             (lambda [capture-parameter ...]
               (region [(let-value local :dtype init-expr) ...]
                       [result-expr ...]))))
        (= equation-id [result ...]
           (map {:index i :extent n} [array ...] [capture ...]
             (lambda [element ... capture-parameter ...]
               (region [(let-value local :dtype init-expr) ...]
                       [result-expr ...]))))
        (= equation-id [result ...]
           (scatter {:index i :extent n
                     :conflict :unique-or-proof-carrying-reduction}
             [array ...] [capture ...]
             (lambda [element ... capture-parameter ...]
               (region [(let-value local :dtype init-expr) ...]
                       [(write destination-index predicate value) ...]))))
        (= equation-id [result ...]
           (effect-map {:index i :extent n :dtypes [:float ...]
                        :iteration-order :independent-or-sequential}
             [array ...] [capture ...] [destination ...]
             (lambda [element ... capture-parameter ... destination-parameter ...]
               (effect-region [(let-value local :dtype init-expr) ...]
                 [(effect destination-parameter conflict destination-index predicate value)
                  ...]))))
        (= equation-id [result ...]
           (reduce {:index i :extent n
                    :accumulators [acc ...] :identities [zero ...]
                    :dtypes [:float ...] :algebra [{} ...]}
             [array ...] [capture ...]
             (lambda [acc ... element ... capture-parameter ...]
               (region [(let-value local :dtype init-expr) ...]
                       [step-result ...]))))
        (= equation-id [result ...]
           (segmented-reduce {:segment-axes [[row rows] ...]
                              :index k :extent width
                              :accumulators [acc ...] :identities [zero ...]
                              :dtypes [:float ...] :algebra [{} ...]}
             [array ...] [capture ...]
             (lambda [acc ... element ... capture-parameter ...]
               (region [] [step-result ...]))))
        (= equation-id [result]
           (scan {:mode :inclusive :index i :extent n
                  :accumulators [acc] :identities [zero]
                  :dtypes [:float] :algebra [{}]}
             [array ...] [capture ...]
             (lambda [acc element ... capture-parameter ...]
               (region [(let-value local :dtype init-expr) ...]
                       [step-result]))))]
       [program-result ...])

   Scan mode may be `:inclusive` or `:exclusive`; it is an explicit result-layout property. Map
   lambdas return tuples, so horizontal fusion remains functional rather than encoding
   secondary results as hidden stores. Caller-owned map outputs use an equation-fact
   `:result-storage` vector aligned with those functional results; logical SSA identity therefore
   stays separate from physical write identity and host return semantics. Captures are explicit
   operands with explicit scalar lambda parameters, so stable value IDs never leak into lexical
   expression binding. Lambda regions have one typed, ordered local-SSA spine. A local initializer
   may reference parameters, the map index, and earlier locals; results may reference all locals.
   This represents shared scalar work once without smuggling type inference into an emitter.
   Ordered effect maps are the conservative boundary for imperative parallel work: each physical
   destination is explicit, every effect carries either an injectivity proof or a checked
   commutative-reduction contract, and source order is retained within a logical work item. They
   are not treated as pure maps or admitted to algebraic fusion without a later effect proof.
   Scan results use the same result-storage relation when materialized into caller-owned buffers."
  (:require [clojure.set :as set]
            [pattern.nanopass.dialect :as dialect
             :refer [def-dialect]]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.scan :as scan-ir]))

(defn value-id?
  "Stable logical value IDs. Vector IDs are used by existing SSA-like program envelopes."
  [value]
  (or (symbol? value) (keyword? value) (vector? value)))

(defn equation-id?
  [value]
  (or (integer? value) (value-id? value)))

(defn extent?
  [value]
  (or (value-id? value) (and (integer? value) (not (neg? value)))))

(defn scalar-literal?
  [value]
  (or (nil? value)
      (number? value)
      (boolean? value)
      (string? value)
      (keyword? value)))

(defn extent-shape
  "Canonical rank-one shape for an extent value ID.

   AbstractValue dimensions already accept symbolic S-expressions. Compound stable IDs therefore
   use an explicit `(value id)` dimension instead of being confused with shape structure."
  [extent]
  [(cond
     (integer? extent) extent
     (symbol? extent) extent
     :else (list 'value extent))])

(defn scan-result-shape
  "The functional result shape of a scan operation.

   Inclusive scan produces one value per input element. Raster's public exclusive scan includes
   the initial identity at element zero, so its result has `extent + 1` elements. The logical
   traversal extent remains unchanged and is what schedules use for work decomposition."
  [{:keys [mode extent]}]
  [(if (= :exclusive mode)
     (if (integer? extent) (inc extent) (list 'clojure.core/inc extent))
     (first (extent-shape extent)))])

(defn segmented-reduce-result-shape
  "Canonical tensor shape of a segmented reduction's ordered parallel axes."
  [attributes]
  (mapv (comp first extent-shape second) (:segment-axes attributes)))

(defn segmented-fold-map-result-shape
  "Canonical dense output shape of an ordered segmented fold-map."
  [attributes]
  (conj (segmented-reduce-result-shape attributes)
        (first (extent-shape (:extent attributes)))))

(defn map-attributes?
  [value]
  (and (map? value)
       (symbol? (:index value))
       (extent? (:extent value))
       (or (nil? (:attributes value)) (map? (:attributes value)))))

(def ^:private scatter-accumulator '%scatter-accumulator)
(def ^:private scatter-contribution '%scatter-contribution)

(defn reducing-scatter-conflict
  "Build the canonical proof-carrying conflict contract for a reducing scatter.

  The current atomic schedules accept commutative scalar monoids.  Keeping the certificate in the
  functional IR distinguishes an update contribution from a unique destination store and prevents
  an emitter from inventing reassociation legality from a function name."
  [operator dtype]
  (let [dtype (dtype/canon dtype)
        identity (descriptor/typed-reduce-identity operator dtype)
        algebra (scan-ir/certify
                 {:acc scatter-accumulator :init identity
                  :lambda (list operator scatter-accumulator scatter-contribution)}
                 dtype)]
    {:kind :reduce :operator (:combine algebra) :identity identity
     :dtype dtype :algebra algebra}))

(defn reducing-scatter-conflict?
  [value]
  (and (map? value)
       (= :reduce (:kind value))
       (contains? #{:int :float} (:dtype value))
       (try
         (= value (reducing-scatter-conflict (:operator value) (:dtype value)))
         (catch clojure.lang.ExceptionInfo _ false))))

(defn scatter-attributes?
  [value]
  (and (map-attributes? value)
       (or (= :unique (:conflict value))
           (reducing-scatter-conflict? (:conflict value)))))

(defn effect-conflict?
  "A race-freedom contract for one ordered effect destination.

   `:unique` certifies injective writes across logical work items. Reduction effects reuse the
   same checked commutative-monoid certificate as typed scatter. `:ordered` permits conflicts only
   when the enclosing effect map requires source-order iteration."
  [value]
  (or (contains? #{:unique :ordered} value) (reducing-scatter-conflict? value)))

(defn effect-map-attributes?
  [value]
  (and (map-attributes? value)
       (contains? #{:independent :sequential} (:iteration-order value))
       (vector? (:dtypes value))
       (seq (:dtypes value))
       (every? #(and (keyword? %) (dtype/known? %)
                     (= % (dtype/canon %)))
               (:dtypes value))))

(defn stencil-attributes?
  "Attributes for a boundary-aware neighborhood map.

   The first production schedule certifies a radius-one domain and Dirichlet boundaries.
   Neighborhood arrays remain explicit stable captures, so the scalar region sees whole tensors
   and retains its shifted indices rather than pretending they are pointwise elements."
  [value]
  (and (map-attributes? value)
       (= 1 (:radius value))
       (= :dirichlet (:boundary value))
       (vector? (:dtypes value))
       (= 1 (count (:dtypes value)))
       (let [component-dtype (first (:dtypes value))]
         (and (keyword? component-dtype)
              (contains? #{:float :double} component-dtype)
              (dtype/known? component-dtype)
              (= component-dtype (dtype/canon component-dtype))))))

(declare lambda-form result-transform?)

(defn reduce-attributes?
  [value]
  (and (map-attributes? value)
       (vector? (:accumulators value))
       (seq (:accumulators value))
       (every? symbol? (:accumulators value))
       (= (count (:accumulators value)) (count (distinct (:accumulators value))))
       (vector? (:identities value))
       (vector? (:dtypes value))
       (vector? (:algebra value))
       (= (count (:accumulators value))
          (count (:identities value))
          (count (:dtypes value))
          (count (:algebra value)))
       (every? keyword? (:dtypes value))
       (every? map? (:algebra value))
       (or (nil? (:result-transform value))
           (and (= 1 (count (:accumulators value)))
                (result-transform? (:result-transform value))))))

(defn result-transform?
  "A typed scalar transform applied once to a completed reduction result.

   This is semantic region data, not target epilogue source: every external value has an explicit
   operand/scalar role and dtype, and operand indexing is a structured axis map.  Schedules may
   place the region in a tile store, while untiled lowerings retain the same expression."
  [value]
  (let [operands (:operands value)
        scalars (:scalars value)
        ids (vec (concat (map :value operands) (map :value scalars)))
        parameters (when (and (seq? (:lambda value))
                              (= 'lambda (first (:lambda value))))
                     (second (:lambda value)))
        [_ _ region] (:lambda value)
        [_ locals results] region
        expected-parameters
        (vec (concat [(first parameters)]
                     (map :parameter operands) (map :parameter scalars)))
        axis-map? (fn [value]
                    (let [groups (:groups value)]
                      (and (map? value) (vector? groups) (seq groups)
                           (every? #(and (vector? %) (seq %)
                                         (every? (fn [pair]
                                                   (and (vector? pair) (= 2 (count pair))
                                                        (value-id? (first pair))
                                                        (extent? (second pair))))
                                                 %))
                                   groups))))]
    (and (map? value)
         (vector? parameters) (seq parameters) (every? symbol? parameters)
         (= parameters expected-parameters)
         (seq? region) (= 'region (first region))
         (= [] locals) (vector? results) (= 1 (count results))
         (vector? operands)
         (every? (fn [{:keys [value parameter dtype map]}]
                   (and (value-id? value) (symbol? parameter)
                        (keyword? dtype) (dtype/known? dtype) (= dtype (dtype/canon dtype))
                        (axis-map? map)))
                 operands)
         (vector? scalars)
         (every? (fn [{:keys [value parameter dtype]}]
                   (and (value-id? value) (symbol? parameter)
                        (keyword? dtype) (dtype/known? dtype) (= dtype (dtype/canon dtype))))
                 scalars)
         (= (count ids) (count (distinct ids)))
         (= (count parameters) (count (distinct parameters)))
         (keyword? (:result-dtype value))
         (dtype/known? (:result-dtype value))
         (= (:result-dtype value) (dtype/canon (:result-dtype value))))))

(defn make-result-transform
  "Close a post-reduction scalar expression over an alpha-stable typed boundary.

   Operand and scalar `:value` fields are program SSA IDs.  The returned lambda refers only to
   fresh lexical parameters, its accumulator, and the segmented-reduction axes.  Frontends and
   fusion rules use this one constructor so target projections never have to rediscover captures
   from an expression."
  [{:keys [accumulator expression operands scalars result-dtype]}]
  (let [operands (mapv (fn [ordinal {:keys [value dtype map]}]
                         {:value value
                          :parameter (symbol (str "%result-operand" ordinal))
                          :dtype dtype :map map})
                       (range) (vec operands))
        scalars (mapv (fn [ordinal {:keys [value dtype]}]
                        {:value value
                         :parameter (symbol (str "%result-scalar" ordinal))
                         :dtype dtype})
                      (range) (vec scalars))
        substitutions
        (into {}
              (concat (map (juxt :value :parameter) operands)
                      (map (juxt :value :parameter) scalars)))]
    {:operands operands
     :scalars scalars
     :result-dtype result-dtype
     :lambda (lambda-form
              (vec (concat [accumulator]
                           (map :parameter operands) (map :parameter scalars)))
              [(util/subst-syms substitutions expression)])}))

(defn segmented-reduce-attributes?
  "Attributes for a general segmented reduction. Segment axes are the ordered parallel result
   space; `:index/:extent` remain the innermost reduced axis shared with ordinary reduce."
  [value]
  (let [axes (:segment-axes value)
        indices (mapv first axes)]
    (and (reduce-attributes? (dissoc value :segment-axes))
         (vector? axes) (seq axes)
         (every? #(and (vector? %) (= 2 (count %))
                       (symbol? (first %)) (extent? (second %)))
                 axes)
         (= (count indices) (count (distinct indices)))
         (not (contains? (set indices) (:index value)))
         (or (nil? (:result-transform value))
             (and (= 1 (count (:accumulators value)))
                  (result-transform? (:result-transform value)))))))

(defn product-reduce-attributes?
  "Attributes for an associative segmented product reduction.

   `:result-components` aligns the equation's materialized result vector with component ordinals;
   omitted ordinals participate in the algebra but require no output storage. The element and
   combine regions remain explicit operation children rather than opaque attribute payloads."
  [value]
  (let [axes (:segment-axes value)
        component-ids (:component-ids value)
        component-count (count component-ids)
        result-components (:result-components value)
        indices (mapv first axes)]
    (and (map-attributes? value)
         (vector? axes) (seq axes)
         (every? #(and (vector? %) (= 2 (count %))
                       (symbol? (first %)) (extent? (second %)))
                 axes)
         (= (count indices) (count (distinct indices)))
         (not (contains? (set indices) (:index value)))
         (vector? component-ids) (pos? component-count)
         (= component-count (count (distinct component-ids)))
         (vector? (:accumulators value))
         (= component-count (count (:accumulators value)))
         (every? symbol? (:accumulators value))
         (= component-count (count (distinct (:accumulators value))))
         (vector? (:identities value))
         (vector? (:dtypes value))
         (= component-count (count (:identities value)) (count (:dtypes value)))
         (every? #(and (keyword? %) (dtype/known? %) (= % (dtype/canon %)))
                 (:dtypes value))
         (vector? result-components) (seq result-components)
         (= (count result-components) (count (distinct result-components)))
         (every? #(and (integer? %) (<= 0 %) (< % component-count)) result-components)
         (map? (:algebra value))
         (true? (get-in value [:algebra :associative?])))))

(defn fold-attributes?
  "Attributes of one loop-carried scalar fold.

   Ordered folds preserve the source recurrence.  Implementation-defined
   association is admitted only with the same checked monoid certificate used
   by parallel scans; this is the proof that permits target reassociation."
  [value]
  (and (map? value)
       (symbol? (:accumulator value))
       (or (nil? (:index value)) (symbol? (:index value)))
       (scalar-literal? (:identity value))
       (keyword? (:dtype value))
       (dtype/known? (:dtype value))
       (= (:dtype value) (dtype/canon (:dtype value)))
       (or (extent? (:extent value)) (seq? (:extent value)))
       (or (= :ordered (:association value))
           (and (= :implementation-defined (:association value))
                (scan-ir/associative-scan? (:algebra value))))))

(defn segmented-fold-map-attributes?
  "Attributes for independent segments containing dependent ordered folds and a final dense map."
  [value]
  (let [axes (:segment-axes value)
        indices (mapv first axes)]
    (and (map-attributes? value)
         (vector? axes) (seq axes)
         (every? #(and (vector? %) (= 2 (count %))
                       (symbol? (first %)) (extent? (second %)))
                 axes)
         (= (count indices) (count (distinct indices)))
         (not (contains? (set indices) (:index value)))
         (= :ordered (:association value))
         (vector? (:dtypes value)) (seq (:dtypes value))
         (every? #(and (keyword? %) (dtype/known? %) (= % (dtype/canon %)))
                 (:dtypes value)))))

(defn scan-attributes?
  "Attributes for a certified scan dialect operation. The explicit mode is load-bearing because
   inclusive and exclusive scans have different observable result layouts."
  [value]
  (and (contains? #{:inclusive :exclusive} (:mode value))
       (reduce-attributes? (dissoc value :mode))
       (every? scan-ir/associative-scan? (:algebra value))))

(defn scalar-attributes?
  [value]
  (and (map? value)
       (vector? (:dtypes value))
       (seq (:dtypes value))
       (every? keyword? (:dtypes value))
       (or (nil? (:attributes value)) (map? (:attributes value)))))

(defn equation-facts?
  [value]
  (and (map? value)
       (set? (:effects value))
       (map? (:aliases value))
       (map? (:provenance value))
       (map? (:attributes value))))

(defn program-facts?
  [value]
  (and (map? value)
       (map? (:values value))
       (every? value-id? (keys (:values value)))
       (every? av/abstract-value? (vals (:values value)))
       (vector? (:inputs value))
       (map? (:equations value))
       (every? equation-id? (keys (:equations value)))
       (every? equation-facts? (vals (:equations value)))
       (set? (:effects value))
       (vector? (:diagnostics value))
       (map? (:provenance value))
       (map? (:attributes value))))

(def-dialect TypedSOAC
  (terminals [id value-id?]
             [eid equation-id?]
             [sym symbol?]
             [lit scalar-literal?]
             [sa scalar-attributes?]
             [ma map-attributes?]
             [xa scatter-attributes?]
             [ema effect-map-attributes?]
             [ec effect-conflict?]
             [sta stencil-attributes?]
             [ra reduce-attributes?]
             [sra segmented-reduce-attributes?]
             [pra product-reduce-attributes?]
             [fa fold-attributes?]
             [sfma segmented-fold-map-attributes?]
             [ca scan-attributes?]
             [dt keyword?]
             [facts program-facts?])

  (Scalar [s :enforce]
          ?sym ?lit
          ?f
          (if ?s:test ?s:then ?s:else)
          (do (?:+ s))
          (let* [(?:* ?sym:binding ?s:init)] ?s:body)
          (write ?s:destination-index ?s:predicate ?s:value)
          [(?:* s)]
          (.invk ?sym:impl (?:* s:args))
          (& (?sym:f (?:* s:args)) (? _ seq?)))

  (Effect [e :enforce]
          (effect ?sym:destination ?ec:conflict
                  ?s:destination-index ?s:predicate ?s:value))

  (Local [d :enforce]
         (let-value ?sym:binding ?dt ?s:init))

  (Region [r :enforce]
          (region [(?:* d)] [(?:+ s)]))

  (EffectRegion [er :enforce]
                (effect-region [(?:* d)] [(?:+ e)]))

  (Lambda [l :enforce]
          (lambda [(?:* ?sym:parameter)] ?r))

  (EffectLambda [el :enforce]
                (lambda [(?:* ?sym:parameter)] ?er))

  (Fold [f :enforce]
        (fold ?fa ?l))

  (Operation [o :enforce]
             (scalar ?sa [(?:* ?id:capture)] ?l)
             (map ?ma [(?:* ?id:array)] [(?:* ?id:capture)] ?l)
             (scatter ?xa [(?:* ?id:array)] [(?:* ?id:capture)] ?l)
             (effect-map ?ema [(?:* ?id:array)] [(?:* ?id:capture)]
                         [(?:+ ?id:destination)] ?el)
             (stencil ?sta [(?:* ?id:array)] [(?:* ?id:capture)] ?l)
             (reduce ?ra [(?:* ?id:array)] [(?:* ?id:capture)] ?l)
             (segmented-reduce ?sra [(?:* ?id:array)] [(?:* ?id:capture)] ?l)
             (product-reduce ?pra [(?:* ?id:array)] [(?:* ?id:capture)]
                             ?l:element ?l:combine)
             (segmented-fold-map ?sfma [(?:* ?id:array)] [(?:* ?id:capture)]
                                 [(?:+ f)] ?l)
             (scan ?ca [(?:* ?id:array)] [(?:* ?id:capture)] ?l))

  (Equation [q :enforce]
            (= ?eid [(?:+ ?id:result)] ?o))

  (Program [p :enforce]
           (soac-program ?facts [(?:* q)] [(?:* ?id:output)]))

  (entry Program))

(defn program-form?
  [value]
  (and (seq? value) (= 'soac-program (first value)) (= 4 (count value))))

(defn facts [program] (second program))
(defn equations [program] (nth program 2))
(defn outputs [program] (nth program 3))

(defn operation-kind
  [equation]
  (first (nth equation 3)))

(defn operation-inputs
  "Ordered array and capture operands of an equation. Extent is a separate scalar operand."
  [equation]
  (let [operation (nth equation 3)]
    (if (= 'scalar (first operation))
      (vec (nth operation 2))
      (vec (concat (nth operation 2) (nth operation 3))))))

(declare operation-parts)

(defn operation-extent
  [equation]
  (let [operation (nth equation 3)]
    (when-not (= 'scalar (first operation))
      (:extent (second operation)))))

(defn operation-extents
  "All ordered iteration extents referenced by an operation. For segmented reductions this is
   the parallel segment space followed by the innermost reduction extent."
  [equation]
  (let [{:keys [kind attributes]} (operation-parts equation)]
    (if (contains? #{'segmented-reduce 'product-reduce 'segmented-fold-map} kind)
      (into (conj (mapv second (:segment-axes attributes)) (:extent attributes))
            (when (= 'segmented-fold-map kind)
              (map #(get-in % [:attributes :extent])
                   (:folds (operation-parts equation)))))
      (if-let [extent (:extent attributes)] [extent] []))))

(defn operation-parts
  "Normalize one operation into semantic fields shared by validation and lowering."
  [equation]
  (let [operation (nth equation 3)
        kind (first operation)]
    (cond
      (= 'scalar kind)
      (let [[_ attributes captures lambda] operation]
        {:kind kind :attributes attributes :arrays [] :captures captures :lambda lambda})

      (= 'product-reduce kind)
      (let [[_ attributes arrays captures element-lambda combine-lambda] operation]
        {:kind kind :attributes attributes :arrays arrays :captures captures
         :element-lambda element-lambda :combine-lambda combine-lambda})

      (= 'segmented-fold-map kind)
      (let [[_ attributes arrays captures folds map-lambda] operation]
        {:kind kind :attributes attributes :arrays arrays :captures captures
         :folds (mapv (fn [[_ fold-attributes fold-lambda]]
                        {:attributes fold-attributes :lambda fold-lambda})
                      folds)
         :map-lambda map-lambda})

      (= 'effect-map kind)
      (let [[_ attributes arrays captures destinations lambda] operation]
        {:kind kind :attributes attributes :arrays arrays :captures captures
         :destinations destinations :lambda lambda})

      :else
      (let [[_ attributes arrays captures lambda] operation]
        {:kind kind :attributes attributes :arrays arrays :captures captures :lambda lambda}))))

(defn write-form?
  "Whether a scatter-region result is one explicit conditional indexed write."
  [value]
  (and (seq? value) (= 'write (first value)) (= 4 (count value))))

(defn write-parts
  [value]
  (when (write-form? value)
    (let [[_ destination-index predicate written-value] value]
      {:destination-index destination-index :predicate predicate :value written-value})))

(defn effect-form?
  "Whether a typed ordered-effect region item has canonical syntax."
  [value]
  (and (seq? value) (= 'effect (first value)) (= 6 (count value))))

(defn effect-parts
  [value]
  (when (effect-form? value)
    (let [[_ destination conflict destination-index predicate written-value] value]
      {:destination destination :conflict conflict
       :destination-index destination-index :predicate predicate :value written-value})))

(defn local-value
  "Construct one explicitly typed scalar-region SSA definition."
  [id dtype init]
  (list 'let-value id dtype init))

(defn lambda-region
  "Construct the canonical scalar region used by every TypedSOAC lambda."
  [locals results]
  (list 'region (vec locals) (vec results)))

(defn effect-lambda-region
  "Construct a canonical ordered-effect region."
  [locals effects]
  (list 'effect-region (vec locals) (vec effects)))

(defn effect-lambda-form
  "Construct a canonical TypedSOAC ordered-effect lambda."
  ([parameters effects]
   (effect-lambda-form parameters [] effects))
  ([parameters locals effects]
   (list 'lambda (vec parameters) (effect-lambda-region locals effects))))

(defn lambda-form
  "Construct a canonical TypedSOAC lambda, with an optional ordered local-SSA spine."
  ([parameters results]
   (lambda-form parameters [] results))
  ([parameters locals results]
   (list 'lambda (vec parameters) (lambda-region locals results))))

(defn lambda-parts
  "Project a canonical lambda into parameters, typed locals, and ordered result expressions.

   Locals are returned as maps so compiler passes do not independently parse their S-expression
   spelling. The TypedSOAC form remains the sole serialized representation."
  [lambda]
  (let [[_ parameters [_ local-forms body-results]] lambda]
    {:parameters (vec parameters)
     :locals (mapv (fn [[_ id dtype init]]
                     {:id id :dtype dtype :init init})
                   local-forms)
     :body-results (vec body-results)}))

(defn scalar-fold-form?
  "True for a typed ordered fold embedded in a scalar region."
  [value]
  (and (seq? value) (= 'fold (first value)) (= 3 (count value))))

(defn scalar-fold-parts
  "Project a scalar-region fold without recovering semantics from host source."
  [value]
  (when (scalar-fold-form? value)
    (let [[_ attributes lambda] value]
      {:attributes attributes :lambda lambda})))

(defn- nested-scalar-folds
  [expressions]
  (->> expressions
       (mapcat #(tree-seq coll? seq %))
       (filter scalar-fold-form?)
       vec))

(defn emit-locals
  "Return local-value forms for normalized local maps."
  [locals]
  (mapv (fn [{:keys [id dtype init]}] (local-value id dtype init)) locals))

(defn result-storage
  "Ordered physical storage contracts for an equation's functional results.

   A map that writes caller-owned buffers still defines fresh functional values. Each entry maps
   one equation result to the physical destination used when that result is materialized. Absence
   means the equation owns fresh result storage. The vector is aligned with the equation results;
   it is deliberately not a destination-name registry reconstructed by a backend."
  [program-or-facts equation-id]
  (get-in (if (program-form? program-or-facts)
            (facts program-or-facts)
            program-or-facts)
          [:equations equation-id :attributes :result-storage]))

(defn physical-results
  "Resolve an equation's ordered logical results to their physical storage identities."
  [program-or-facts equation]
  (let [results (vec (nth equation 2))
        storage (result-storage program-or-facts (second equation))]
    (if (seq storage)
      (mapv :destination storage)
      results)))

(defn parameter-layout
  "Split a SOAC lambda's ordered parameters into semantic roles."
  [equation]
  (let [{:keys [kind attributes arrays captures destinations lambda element-lambda]}
        (operation-parts equation)
        parameters (:parameters (lambda-parts (or lambda element-lambda)))
        accumulator-count (if (contains? #{'reduce 'segmented-reduce 'scan} kind)
                            (count (:accumulators attributes)) 0)
        array-count (count arrays)
        accumulator-end accumulator-count
        element-end (+ accumulator-end array-count)
        capture-end (+ element-end (count captures))]
    (cond-> {:accumulators (subvec parameters 0 accumulator-end)
             :elements (subvec parameters accumulator-end element-end)
             :capture-parameters (subvec parameters element-end capture-end)}
      (seq destinations)
      (assoc :destination-parameters
             (subvec parameters capture-end (+ capture-end (count destinations)))))))

(defn- distinct-vector?
  [value]
  (and (vector? value) (= (count value) (count (distinct value)))))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :dialect :typed-soac))))

(def ^:private stencil-index-casts
  '#{byte short int long clojure.core/byte clojure.core/short clojure.core/int clojure.core/long})

(defn- unwrap-stencil-index-cast
  [expression]
  (loop [expression expression]
    (if (and (seq? expression) (= 2 (count expression))
             (contains? stencil-index-casts (first expression)))
      (recur (second expression))
      expression)))

(defn- stencil-index-offset
  "Return the constant offset of one admitted affine stencil index, or nil.

   The initial typed stencil contract deliberately admits only `i`, `inc i`, `dec i`, `i + c`,
   `c + i`, and `i - c`. `inc`/`dec` are the analyzed canonical forms of the same unit-affine
   indices. This keeps the radius proof semantic and target-independent; emitters never guess
   whether an arbitrary index expression remains inside the guarded neighborhood."
  [index expression]
  (let [expression (unwrap-stencil-index-cast expression)]
    (cond
    (= index expression) 0

    (and (seq? expression) (= 2 (count expression))
         (contains? '#{inc clojure.core/inc unchecked-inc-int
                       clojure.core/unchecked-inc-int}
                    (first expression))
         (= index (unwrap-stencil-index-cast (second expression))))
    1

    (and (seq? expression) (= 2 (count expression))
         (contains? '#{dec clojure.core/dec unchecked-dec-int
                       clojure.core/unchecked-dec-int}
                    (first expression))
         (= index (unwrap-stencil-index-cast (second expression))))
    -1

    (and (seq? expression) (= 3 (count expression))
         (contains? '#{+ clojure.core/+} (first expression)))
    (let [[_ left right] expression
          left (unwrap-stencil-index-cast left)
          right (unwrap-stencil-index-cast right)]
      (cond
        (and (= index left) (integer? right)) right
        (and (integer? left) (= index right)) left
        :else nil))

    (and (seq? expression) (= 3 (count expression))
         (contains? '#{- clojure.core/-} (first expression))
         (= index (unwrap-stencil-index-cast (second expression)))
         (integer? (unwrap-stencil-index-cast (nth expression 2))))
    (- (unwrap-stencil-index-cast (nth expression 2)))

    :else nil)))

(defn- validate-stencil-neighborhood!
  [equation-id attributes captures parameters body-results]
  (let [stable-ids (set (get-in attributes [:attributes :stable-array-captures]))
        capture-parameters (take-last (count captures) parameters)
        stable-parameters (->> (map vector captures capture-parameters)
                               (keep (fn [[capture parameter]]
                                       (when (contains? stable-ids capture) parameter)))
                               set)
        index (:index attributes)
        radius (:radius attributes)]
    (doseq [body body-results
            form (tree-seq coll? seq body)
            :when (descriptor/aget-call? form)]
      (let [array-parameter (descriptor/aget-array-sym form)
            _ (when-not (contains? stable-parameters array-parameter)
                (fail! :typed-soac-stencil-array-role
                       "every stencil array load must read an explicit stable tensor capture"
                       {:equation equation-id :array-parameter array-parameter
                        :stable-parameters stable-parameters :load form}))
            index-expression (descriptor/aget-index form)
            offset (stencil-index-offset index index-expression)]
        (when (or (nil? offset) (> (abs (long offset)) radius))
          (fail! :typed-soac-stencil-index
                 "stencil tensor loads require a constant affine index inside the declared radius"
                 {:equation equation-id :index index :index-expression index-expression
                  :radius radius :load form}))))))

(defn- validate-equation!
  [equation]
  (let [[_ equation-id results operation] equation
        {:keys [kind attributes arrays captures destinations lambda element-lambda combine-lambda
                folds map-lambda]}
        (operation-parts equation)
        product? (= 'product-reduce kind)
        fold-map? (= 'segmented-fold-map kind)
        {:keys [parameters locals body-results]}
        (lambda-parts (or lambda element-lambda map-lambda))
        component-count (when product? (count (:component-ids attributes)))
        result-count (count results)]
    (when-not (distinct-vector? results)
      (fail! :typed-soac-equation-results "SOAC equation results must be distinct"
             {:equation equation-id :results results}))
    (doseq [[field ids] [[:arrays arrays] [:captures captures]
                         [:destinations (or destinations [])]]]
      (when-not (distinct-vector? ids)
        (fail! :typed-soac-operands "SOAC operands must be ordered and distinct"
               {:equation equation-id :field field :ids ids})))
    (when (seq (set/intersection (set arrays) (set captures)))
      (fail! :typed-soac-operand-role "one value cannot be both an element input and a capture"
             {:equation equation-id :arrays arrays :captures captures}))
    (when (seq (set/intersection (set destinations)
                                 (set (concat arrays captures))))
      (fail! :typed-soac-operand-role
             "an effect destination cannot also be an element input or stable capture"
             {:equation equation-id :arrays arrays :captures captures
              :destinations destinations}))
    (let [stable-array-captures (get-in attributes [:attributes :stable-array-captures] [])]
      (when-not (and (distinct-vector? stable-array-captures)
                     (set/subset? (set stable-array-captures) (set captures)))
        (fail! :typed-soac-stable-array-captures
               "stable array capture roles must be an ordered subset of captures"
               {:equation equation-id :stable-array-captures stable-array-captures
                :captures captures})))
    (when-not (distinct-vector? parameters)
      (fail! :typed-soac-lambda-parameters "SOAC lambda parameters must be distinct"
             {:equation equation-id :parameters parameters}))
    (when-not (every? symbol? parameters)
      (fail! :typed-soac-lambda-parameters "SOAC lambda parameters must be symbols"
             {:equation equation-id :parameters parameters}))
    (let [local-ids (mapv :id locals)
          lexical-ids (vec (concat parameters local-ids))]
      (when-not (= (count lexical-ids) (count (distinct lexical-ids)))
        (fail! :typed-soac-region-binders
               "scalar-region parameters and local SSA definitions must be distinct"
               {:equation equation-id :parameters parameters :locals local-ids}))
      (when (and (contains? #{'map 'scatter 'effect-map 'stencil 'reduce 'segmented-reduce
                              'product-reduce 'segmented-fold-map 'scan} kind)
                 (some #{(:index attributes)} local-ids))
        (fail! :typed-soac-region-binders
               "a scalar-region local cannot shadow its operation index"
               {:equation equation-id :index (:index attributes) :locals local-ids})))
    (doseq [{:keys [dtype] :as local} locals]
      ;; Region locals are currently materialized as typed JVM scalar bindings before the common
      ;; CPU/GPU scalar emitters see them. Reuse the compiler's authoritative dtype facets here;
      ;; accepting :half or an unknown keyword would only fail later in target lowering.
      (when-not (and (dtype/known? dtype)
                     (= dtype (dtype/canon dtype))
                     (:scalar-tag (dtype/info dtype)))
        (fail! :typed-soac-local-dtype
               "scalar-region locals require a supported JVM scalar dtype"
               {:equation equation-id :local local :dtype dtype})))
    (when (and (seq locals)
               (not (contains? #{'map 'scatter 'effect-map
                                 'product-reduce 'segmented-fold-map} kind)))
      (fail! :typed-soac-region-operation
             "typed local SSA is not admitted in this operation's scalar region"
             {:equation equation-id :operation kind :locals locals}))
    (when-not (or (= 'effect-map kind)
                  (= (if product? component-count result-count) (count body-results)))
      (fail! :typed-soac-result-arity "SOAC result and lambda arity differ"
             {:equation equation-id :results result-count :components component-count
              :body-results (count body-results)}))
    (let [expected-parameter-count (+ (cond
                                        (contains? #{'reduce 'segmented-reduce 'scan} kind)
                                        (count (:accumulators attributes))
                                        fold-map? (count folds)
                                        :else 0)
                                      (count arrays)
                                      (count captures)
                                      (count destinations))]
      (when-not (= expected-parameter-count (count parameters))
        (fail! :typed-soac-lambda-arity
               "lambda parameters must cover accumulators, elements and captures in order"
               {:equation equation-id
                :expected expected-parameter-count
                :actual (count parameters)
                :arrays arrays
                :captures captures})))
    (let [initial-bound (cond-> (set parameters)
                          (contains? #{'map 'scatter 'effect-map 'stencil 'reduce 'segmented-reduce
                                      'product-reduce 'segmented-fold-map 'scan} kind)
                          (conj (:index attributes))
                          (contains? #{'segmented-reduce 'product-reduce
                                      'segmented-fold-map} kind)
                          (into (map first (:segment-axes attributes))))
          final-bound
          (reduce (fn [bound {:keys [id init] :as local}]
                    (let [unbound (util/free-syms init bound)]
                      (when (seq unbound)
                        (fail! :typed-soac-unbound-local
                               "local SSA initializers may reference only parameters and earlier locals"
                               {:equation equation-id :local local :unbound unbound}))
                      (conj bound id)))
                  initial-bound locals)]
      (doseq [fold (nested-scalar-folds
                    (concat (map :init locals) body-results))]
        (let [{:keys [attributes lambda]} (scalar-fold-parts fold)
              {fold-parameters :parameters fold-locals :locals
               fold-results :body-results} (lambda-parts lambda)
              expected [(:accumulator attributes) (:index attributes)]
              extent-unbound (util/free-syms (:extent attributes) final-bound)
              step-bound (into final-bound fold-parameters)
              step-unbound (when (= 1 (count fold-results))
                             (util/free-syms (first fold-results) step-bound))]
          (when-not (and (symbol? (:index attributes))
                         (= expected fold-parameters)
                         (empty? fold-locals)
                         (= 1 (count fold-results))
                         (empty? extent-unbound)
                         (empty? step-unbound)
                         (not (some write-form? fold-results)))
            (fail! :typed-soac-scalar-fold
                   "scalar folds require a closed ordered [accumulator index] region"
                   {:equation equation-id :fold fold :expected expected
                    :parameters fold-parameters :locals fold-locals
                    :results fold-results :extent-unbound extent-unbound
                    :step-unbound step-unbound}))))
      (doseq [expression (concat (map :init locals) body-results)
              form (tree-seq coll? seq expression)
              :when (and (seq? form) (symbol? (first form))
                         (= "raster.par" (namespace (first form))))]
        (fail! :typed-soac-opaque-parallel-scalar
               "parallel scalar-region work must use an explicit TypedSOAC term"
               {:equation equation-id :form form}))
      (doseq [body body-results
              expression (cond
                           (= 'scatter kind) (vals (write-parts body))
                           (= 'effect-map kind)
                           (let [{:keys [destination destination-index predicate value]}
                                 (effect-parts body)]
                             [destination destination-index predicate value])
                           :else [body])
              :let [unbound (util/free-syms expression final-bound)]
              :when (seq unbound)]
        (fail! :typed-soac-unbound-scalar
               "scalar-region results may reference only parameters and local SSA values"
               {:equation equation-id :unbound unbound :body body :expression expression})))
    (case kind
      scalar
      (when-not (= result-count (count (:dtypes attributes)))
        (fail! :typed-soac-scalar-results
               "scalar result arity must equal its declared dtype arity"
               {:equation equation-id :results results :dtypes (:dtypes attributes)}))

      map
      (when (some write-form? body-results)
        (fail! :typed-soac-map-write
               "functional map results cannot contain indexed writes"
               {:equation equation-id :results body-results}))

      scatter
      (when-not (every? write-form? body-results)
        (fail! :typed-soac-scatter-write
               "scatter results must be explicit conditional indexed writes"
               {:equation equation-id :results body-results}))

      effect-map
      (let [destination-count (count destinations)
            destination-parameters (vec (take-last destination-count parameters))
            destination-set (set destination-parameters)
            effects (mapv effect-parts body-results)
            by-destination (group-by :destination effects)
            iteration-order (:iteration-order attributes)]
        (when-not (= result-count destination-count (count (:dtypes attributes)))
          (fail! :typed-soac-effect-results
                 "effect-map results, physical destinations, and dtypes must align"
                 {:equation equation-id :results results :destinations destinations
                  :dtypes (:dtypes attributes)}))
        (when-not (every? some? effects)
          (fail! :typed-soac-effect-form
                 "effect-map regions contain only canonical ordered effects"
                 {:equation equation-id :effects body-results}))
        (when (and (= :independent iteration-order)
                   (some #(= :ordered (:conflict %)) effects))
          (fail! :typed-soac-effect-iteration-order
                 "conflicting effects require sequential logical iteration"
                 {:equation equation-id :iteration-order iteration-order
                  :effects body-results}))
        (when-not (= destination-set (set (keys by-destination)))
          (fail! :typed-soac-effect-destinations
                 "every declared effect destination must be referenced through its lambda parameter"
                 {:equation equation-id :destination-parameters destination-parameters
                  :used (vec (keys by-destination))}))
        (doseq [[ordinal destination-parameter dtype]
                (map vector (range) destination-parameters (:dtypes attributes))
                :let [contracts (set (map :conflict (get by-destination destination-parameter)))] ]
          (when-not (= 1 (count contracts))
            (fail! :typed-soac-effect-conflict
                   "one effect destination requires one uniform cross-work-item conflict contract"
                   {:equation equation-id :destination destination-parameter
                    :contracts contracts}))
          (let [contract (first contracts)]
            (when (and (reducing-scatter-conflict? contract)
                       (not= dtype (:dtype contract)))
              (fail! :typed-soac-effect-conflict-dtype
                     "a reduction effect contract must match its destination dtype"
                     {:equation equation-id :destination destination-parameter
                      :ordinal ordinal :dtype dtype :contract contract})))))

      stencil
      (do
        (when-not (and (= 1 result-count)
                       (= 1 (count body-results))
                       (empty? arrays)
                       (seq (get-in attributes [:attributes :stable-array-captures])))
          (fail! :typed-soac-stencil-boundary
                 "stencil requires one result, whole-tensor stable captures, and no pointwise operands"
                 {:equation equation-id :results results :arrays arrays
                  :stable-array-captures
                  (get-in attributes [:attributes :stable-array-captures])}))
        (validate-stencil-neighborhood! equation-id attributes captures parameters body-results)
        (when (some write-form? body-results)
          (fail! :typed-soac-stencil-write
                 "stencil interior regions must be pure scalar expressions"
                 {:equation equation-id :results body-results})))

      reduce
      (let [accumulators (:accumulators attributes)
            transform (:result-transform attributes)]
        (when-not (= accumulators (vec (take (count accumulators) parameters)))
          (fail! :typed-soac-reduce-accumulators
                 "reduce accumulator parameters must lead the lambda in declared order"
                 {:equation equation-id :accumulators accumulators :parameters parameters}))
        (when-not (= result-count (count accumulators))
          (fail! :typed-soac-reduce-results
                 "reduce result arity must equal accumulator arity"
                 {:equation equation-id :results results :accumulators accumulators}))
        (doseq [[accumulator identity dtype certificate result]
                (map vector accumulators (:identities attributes) (:dtypes attributes)
                     (:algebra attributes) body-results)]
          (let [derived (scan-ir/certify-reassociation
                         {:acc accumulator :init identity :lambda result} dtype)]
            (when-not (scan-ir/compatible-certificate? certificate derived)
              (fail! :typed-soac-reduction-certificate
                     "reduction algebra certificate disagrees with its scalar region"
                     {:equation equation-id :declared certificate :derived derived}))))
        (when transform
          (let [operand-ids (set (map :value (:operands transform)))
                scalar-ids (set (map :value (:scalars transform)))
                transform-ids (set/union operand-ids scalar-ids)
                {:keys [parameters body-results]} (lambda-parts (:lambda transform))
                unbound (util/free-syms (first body-results) (set parameters))]
            (when-not (set/subset? transform-ids (set captures))
              (fail! :typed-soac-result-transform-captures
                     "reduce result transforms require explicit capture values"
                     {:equation equation-id :transform-captures transform-ids
                      :captures captures}))
            ;; A full reduction has no remaining result axes with which to address a tensor.
            ;; Tensor epilogues belong to segmented reductions; this scalar boundary accepts only
            ;; uniform captures until an explicit scalar-load schedule exists.
            (when (seq operand-ids)
              (fail! :typed-soac-reduce-result-transform-operands
                     "full-reduction result transforms cannot address tensor operands"
                     {:equation equation-id :operands operand-ids}))
            (when (seq unbound)
              (fail! :typed-soac-result-transform-expression
                     "result-transform expressions may reference only their typed region boundary"
                     {:equation equation-id :unbound unbound :transform transform})))))

      segmented-reduce
      (let [accumulators (:accumulators attributes)
            transform (:result-transform attributes)]
        (when-not (= accumulators (vec (take (count accumulators) parameters)))
          (fail! :typed-soac-segmented-reduce-accumulators
                 "segmented-reduce accumulator parameters must lead the lambda in declared order"
                 {:equation equation-id :accumulators accumulators :parameters parameters}))
        (when-not (= result-count (count accumulators))
          (fail! :typed-soac-segmented-reduce-results
                 "segmented-reduce result arity must equal accumulator arity"
                 {:equation equation-id :results results :accumulators accumulators}))
        (doseq [[accumulator identity dtype certificate result]
                (map vector accumulators (:identities attributes) (:dtypes attributes)
                     (:algebra attributes) body-results)]
          (let [derived (scan-ir/certify-reassociation
                         {:acc accumulator :init identity :lambda result} dtype)]
            (when-not (scan-ir/compatible-certificate? certificate derived)
              (fail! :typed-soac-reduction-certificate
                     "segmented-reduction algebra certificate disagrees with its scalar region"
                     {:equation equation-id :declared certificate :derived derived}))))
        (when transform
          (let [operand-ids (set (map :value (:operands transform)))
                scalar-ids (set (map :value (:scalars transform)))
                transform-ids (set/union operand-ids scalar-ids)
                stable (set (get-in attributes [:attributes :stable-array-captures]))
                segment-indices (set (map first (:segment-axes attributes)))
                {:keys [parameters body-results]} (lambda-parts (:lambda transform))
                bound (set/union (set parameters) segment-indices)
                unbound (util/free-syms (first body-results) bound)
                map-axes (set (mapcat (comp axis-map/axes :map) (:operands transform)))]
            (when-not (set/subset? transform-ids (set captures))
              (fail! :typed-soac-result-transform-captures
                     "segmented-reduce result transforms require explicit capture values"
                     {:equation equation-id :transform-captures transform-ids
                      :captures captures}))
            (when-not (set/subset? operand-ids stable)
              (fail! :typed-soac-result-transform-operands
                     "result-transform tensor operands must be stable array captures"
                     {:equation equation-id :operands operand-ids :stable stable}))
            (when-not (set/subset? map-axes segment-indices)
              (fail! :typed-soac-result-transform-axis-map
                     "result-transform operand maps may reference only segment axes"
                     {:equation equation-id :axes map-axes
                      :segment-axes segment-indices}))
            (when (seq unbound)
              (fail! :typed-soac-result-transform-expression
                     "result-transform expressions may reference only their typed region boundary"
                     {:equation equation-id :unbound unbound :transform transform})))))

      product-reduce
      (let [result-components (:result-components attributes)
            {combine-parameters :parameters
             combine-locals :locals
             combine-results :body-results}
            (lambda-parts combine-lambda)]
        (when (some write-form? body-results)
          (fail! :typed-soac-product-reduce-element-write
                 "product-reduce element results must be pure scalar values"
                 {:equation equation-id :results body-results}))
        (when-not (= result-count (count result-components))
          (fail! :typed-soac-product-reduce-results
                 "product-reduce results must align with materialized component ordinals"
                 {:equation equation-id :results results
                  :result-components result-components}))
        (when-not (and (= (* 2 component-count) (count combine-parameters))
                       (every? symbol? combine-parameters)
                       (= (count combine-parameters) (count (distinct combine-parameters))))
          (fail! :typed-soac-product-reduce-combine-parameters
                 "product-reduce combine requires one distinct ordered left/right pair per component"
                 {:equation equation-id :components component-count
                  :parameters combine-parameters}))
        (let [combine-bound
              (loop [bound (set combine-parameters)
                     [local & remaining] combine-locals]
                (if-not local
                  bound
                  (let [{:keys [id dtype init]} local
                        unbound (util/free-syms init bound)]
                    (when-not (and (symbol? id) (keyword? dtype) (dtype/known? dtype)
                                   (= dtype (dtype/canon dtype))
                                   (:scalar-tag (dtype/info dtype)))
                      (fail! :typed-soac-local-dtype
                             "product-reduce combine locals require canonical scalar dtypes"
                             {:equation equation-id :local local}))
                    (when (or (contains? bound id) (seq unbound))
                      (fail! :typed-soac-product-reduce-combine-local
                             "combine locals may reference only component pairs and earlier locals"
                             {:equation equation-id :local local :unbound unbound}))
                    (recur (conj bound id) remaining))))]
          (when-not (= component-count (count combine-results))
            (fail! :typed-soac-product-reduce-combine-results
                   "product-reduce combine result arity must equal component arity"
                   {:equation equation-id :components component-count
                    :results combine-results}))
          (doseq [expression combine-results
                  :let [unbound (util/free-syms expression combine-bound)]
                  :when (seq unbound)]
            (fail! :typed-soac-product-reduce-combine-closure
                   "product-reduce combine must be a closed scalar binary operator"
                   {:equation equation-id :expression expression :unbound unbound}))))

      segmented-fold-map
      (let [fold-attributes (mapv :attributes folds)
            accumulators (mapv :accumulator fold-attributes)
            segment-indices (set (map first (:segment-axes attributes)))
            capture-count (count captures)
            map-parameters parameters
            capture-parameters (vec (take-last capture-count map-parameters))
            iteration-index (:index attributes)]
        (when-not (= (count results) (count (:dtypes attributes)))
          (fail! :typed-soac-segmented-fold-map-results
                 "segmented-fold-map result arity must equal its declared result dtype arity"
                 {:equation equation-id :results results :dtypes (:dtypes attributes)}))
        (when (seq arrays)
          (fail! :typed-soac-segmented-fold-map-array-role
                 "the initial ordered schedule uses whole-tensor stable captures, not pointwise arrays"
                 {:equation equation-id :arrays arrays}))
        (when-not (= (count accumulators) (count (distinct accumulators)))
          (fail! :typed-soac-segmented-fold-map-accumulators
                 "ordered fold accumulators must be distinct"
                 {:equation equation-id :accumulators accumulators}))
        (when-not (= accumulators
                     (subvec map-parameters 0 (count accumulators)))
          (fail! :typed-soac-segmented-fold-map-map-parameters
                 "the final map must receive completed folds, pointwise elements, and captures in order"
                 {:equation equation-id :parameters map-parameters
                  :accumulators accumulators :arrays arrays :captures captures}))
        (doseq [[ordinal {:keys [attributes lambda]}] (map-indexed vector folds)]
          (let [{fold-parameters :parameters fold-locals :locals
                 fold-results :body-results} (lambda-parts lambda)
                prior (subvec accumulators 0 ordinal)
                expected (vec (concat [(:accumulator attributes)] prior capture-parameters))
                bound (set/union (set fold-parameters) segment-indices
                                 #{iteration-index})
                final-bound
                (reduce (fn [bound {:keys [id init] :as local}]
                          (when-not (and (keyword? (:dtype local))
                                         (dtype/known? (:dtype local))
                                         (= (:dtype local) (dtype/canon (:dtype local)))
                                         (:scalar-tag (dtype/info (:dtype local))))
                            (fail! :typed-soac-local-dtype
                                   "ordered fold locals require canonical scalar dtypes"
                                   {:equation equation-id :fold ordinal :local local}))
                          (let [unbound (util/free-syms init bound)]
                            (when (or (contains? bound id) (seq unbound))
                              (fail! :typed-soac-segmented-fold-map-fold-local
                                     "ordered fold locals may reference only its lexical boundary and earlier locals"
                                     {:equation equation-id :fold ordinal :local local
                                      :unbound unbound}))
                            (conj bound id)))
                        bound fold-locals)]
            (when-not (= :ordered (:association attributes))
              (fail! :typed-soac-segmented-fold-map-association
                     "segmented fold-map folds must preserve declared order"
                     {:equation equation-id :fold ordinal
                      :association (:association attributes)}))
            (let [unbound-extent
                  (set/difference (util/free-syms (:extent attributes))
                                  (set/union segment-indices (set captures)))]
              (when (seq unbound-extent)
                (fail! :typed-soac-segmented-fold-map-fold-extent
                       "ordered fold extents may reference only segment axes and explicit captures"
                       {:equation equation-id :fold ordinal :extent (:extent attributes)
                        :unbound unbound-extent})))
            (when-not (= fold-parameters expected)
              (fail! :typed-soac-segmented-fold-map-fold-parameters
                     "each ordered fold must receive its accumulator, prior fold results, and captures"
                     {:equation equation-id :fold ordinal :parameters fold-parameters
                      :expected expected}))
            (when-not (= 1 (count fold-results))
              (fail! :typed-soac-segmented-fold-map-fold-result
                     "each ordered fold must yield exactly one next accumulator"
                     {:equation equation-id :fold ordinal :results fold-results}))
            (when (some write-form? fold-results)
              (fail! :typed-soac-segmented-fold-map-fold-write
                     "ordered folds are pure scalar regions"
                     {:equation equation-id :fold ordinal :results fold-results}))
            (let [unbound (binding [util/*shadowing-locals* (set accumulators)]
                            (util/free-syms (first fold-results) final-bound))]
              (when (seq unbound)
                (fail! :typed-soac-segmented-fold-map-fold-closure
                       "ordered fold steps may not reference future folds or implicit state"
                       {:equation equation-id :fold ordinal :unbound unbound})))))
        (when (some write-form? body-results)
          (fail! :typed-soac-segmented-fold-map-map-write
                 "the final fold-map region must yield pure scalar results"
                 {:equation equation-id :results body-results})))

      scan
      (let [accumulators (:accumulators attributes)]
        (when-not (= accumulators (vec (take (count accumulators) parameters)))
          (fail! :typed-soac-scan-accumulators
                 "scan accumulator parameters must lead the lambda in declared order"
                 {:equation equation-id :accumulators accumulators :parameters parameters}))
        (when-not (= result-count (count accumulators))
          (fail! :typed-soac-scan-results
                 "scan result arity must equal accumulator arity"
                 {:equation equation-id :results results :accumulators accumulators}))
        (doseq [[accumulator identity dtype certificate result]
                (map vector accumulators (:identities attributes) (:dtypes attributes)
                     (:algebra attributes) body-results)]
          (let [derived (scan-ir/certify {:acc accumulator :init identity :lambda result} dtype)]
            (when-not (= certificate derived)
              (fail! :typed-soac-scan-certificate
                     "scan algebra certificate disagrees with its scalar region"
                     {:equation equation-id :declared certificate :derived derived})))))

      (fail! :typed-soac-operation "unknown SOAC operation"
             {:equation equation-id :operation kind}))
    equation))

(defn- validate-equation-types!
  [values equation]
  (let [[_ equation-id results] equation
        {:keys [kind attributes arrays]} (operation-parts equation)
        extent (:extent attributes)
        stable-array-captures (get-in attributes [:attributes :stable-array-captures] [])]
    (doseq [id arrays]
      (let [value (get values id)]
        ;; Unknown IDs receive the more precise boundary diagnostic below.
        (when (and value
                   (not (and (= :tensor (:kind value)) (= (extent-shape extent) (:shape value)))))
          (fail! :typed-soac-array-type
                 "SOAC element operands must be rank-one tensors over the declared extent"
                 {:equation equation-id :id id :value value :extent extent}))))
    (doseq [id stable-array-captures]
      (let [value (get values id)]
        (when (and value
                   (not (and (= :tensor (:kind value)) (seq (:shape value)))))
          (when-not (= :resident-scalar-buffer (get-in value [:representation :kind]))
            (fail! :typed-soac-stable-array-type
                   "stable captures require tensor storage or an explicit resident scalar buffer"
                   {:equation equation-id :id id :value value})))))
    (case kind
      scalar
      (doseq [[id dtype] (map vector results (:dtypes attributes))]
        (let [value (get values id)]
          (when (and value
                     (not (and (= :tensor (:kind value)) (= [] (:shape value))
                               (= dtype (:dtype value)))))
            (fail! :typed-soac-scalar-result-type
                   "scalar results must be scalar tensors with their declared dtype"
                   {:equation equation-id :id id :value value :dtype dtype}))))

      map
      (doseq [id results]
        (let [value (get values id)]
          (when (and value
                     (not (and (= :tensor (:kind value))
                               (= (extent-shape extent) (:shape value)))))
            (fail! :typed-soac-map-result-type
                   "map results must be rank-one tensors over the declared extent"
                   {:equation equation-id :id id :value value :extent extent}))))

      scatter
      (doseq [id results]
        (let [value (get values id)]
          (when (and value (not= :tensor (:kind value)))
            (fail! :typed-soac-scatter-result-type
                   "scatter results require tensor storage contracts"
                   {:equation equation-id :id id :value value}))))

      effect-map
      (doseq [[id result-dtype] (map vector results (:dtypes attributes))]
        (let [value (get values id)]
          (when (and value
                     (not (and (= :tensor (:kind value))
                               (= result-dtype (:dtype value)))))
            (fail! :typed-soac-effect-result-type
                   "effect-map results require tensor storage with the declared dtype"
                   {:equation equation-id :id id :value value
                    :dtype result-dtype}))))

      stencil
      (let [result-dtype (first (:dtypes attributes))]
        (doseq [id stable-array-captures
                :let [value (get values id)]]
          (when (and value
                     (not (and (= :tensor (:kind value))
                               (= result-dtype (:dtype value)))))
            (fail! :typed-soac-stencil-input-type
                   "the first stencil schedule requires homogeneous tensor element dtypes"
                   {:equation equation-id :id id :value value :dtype result-dtype})))
        (doseq [id results]
          (let [value (get values id)]
            (when (and value
                       (not (and (= :tensor (:kind value))
                                 (= (extent-shape extent) (:shape value))
                                 (= result-dtype (:dtype value)))))
              (fail! :typed-soac-stencil-result-type
                     "stencil results must match the complete domain and declared dtype"
                     {:equation equation-id :id id :value value
                      :extent extent :dtype result-dtype})))))

      reduce
      (doseq [[id dtype] (map vector results (:dtypes attributes))]
        (let [value (get values id)]
          (when (and value
                     (not (and (= :tensor (:kind value)) (= [] (:shape value))
                               (= dtype (:dtype value)))))
            (fail! :typed-soac-reduce-result-type
                   "reduce results must be scalar tensors with their declared accumulator dtype"
                   {:equation equation-id :id id :value value :dtype dtype}))))

      segmented-reduce
      (let [result-shape (segmented-reduce-result-shape attributes)]
        (doseq [[id dtype] (map vector results (:dtypes attributes))]
          (let [value (get values id)]
            (when (and value
                       (not (and (= :tensor (:kind value)) (= result-shape (:shape value))
                                 (= dtype (:dtype value)))))
              (fail! :typed-soac-segmented-reduce-result-type
                     "segmented-reduce results must match the declared segment space and dtype"
                     {:equation equation-id :id id :value value
                      :segment-axes (:segment-axes attributes) :dtype dtype})))))

      product-reduce
      (let [result-shape (segmented-reduce-result-shape attributes)
            result-dtypes (mapv #((:dtypes attributes) %)
                                (:result-components attributes))]
        (doseq [[id dtype] (map vector results result-dtypes)]
          (let [value (get values id)]
            (when (and value
                       (not (and (= :tensor (:kind value)) (= result-shape (:shape value))
                                 (= dtype (:dtype value)))))
              (fail! :typed-soac-product-reduce-result-type
                     "product-reduce results must match their component dtype and segment space"
                     {:equation equation-id :id id :value value
                      :component-dtype dtype :segment-axes (:segment-axes attributes)})))))

      segmented-fold-map
      (let [result-shape (segmented-fold-map-result-shape attributes)]
        (doseq [[id result-dtype] (map vector results (:dtypes attributes))]
          (let [value (get values id)]
            (when (and value
                       (not (and (= :tensor (:kind value))
                                 (= result-shape (:shape value))
                                 (= result-dtype (:dtype value)))))
              (fail! :typed-soac-segmented-fold-map-result-type
                     "segmented-fold-map results must cover the complete segment/map space"
                     {:equation equation-id :id id :value value
                      :shape result-shape :dtype result-dtype})))))

      scan
      (doseq [[id dtype] (map vector results (:dtypes attributes))]
        (let [value (get values id)]
          (when (and value
                     (not (and (= :tensor (:kind value))
                               (= (scan-result-shape attributes) (:shape value))
                               (= dtype (:dtype value)))))
            (fail! :typed-soac-scan-result-type
                   "scan result shape must agree with its explicit inclusive/exclusive mode"
                   {:equation equation-id :id id :value value
                    :extent extent :mode (:mode attributes) :dtype dtype}))))

      nil)))

(defn- validate-result-storage!
  [program-facts equation]
  (let [[_ equation-id results] equation
        {:keys [kind]} (operation-parts equation)
        storage (result-storage program-facts equation-id)]
    (when storage
      (when-not (contains? #{'map 'scatter 'effect-map 'stencil 'segmented-reduce 'scan
                             'product-reduce 'segmented-fold-map} kind)
        (fail! :typed-soac-result-storage-operation
               "physical result storage is valid only for writing tensor operations"
               {:equation equation-id :operation kind :storage storage}))
      (when-not (and (vector? storage)
                     (= (count results) (count storage))
                     (every? #(and (map? %)
                                   (value-id? (:destination %))
                                   (contains? #{:write :read-write} (:access %))
                                   (contains? #{:buffer :effect} (:host-return %)))
                             storage))
        (fail! :typed-soac-result-storage
               "result storage must align every functional result with a typed destination contract"
               {:equation equation-id :results results :storage storage}))
      (let [destinations (mapv :destination storage)
            aliases (get-in program-facts [:equations equation-id :aliases])]
        (when (and (= 'effect-map kind)
                   (not= destinations (:destinations (operation-parts equation))))
          (fail! :typed-soac-effect-storage
                 "effect-map destinations must equal its ordered physical storage contract"
                 {:equation equation-id :operation-destinations
                  (:destinations (operation-parts equation))
                  :storage-destinations destinations}))
        (when (and (contains? #{'stencil 'segmented-fold-map 'scan} kind)
                   (seq (set/intersection
                         (set destinations)
                         (set (get-in (operation-parts equation)
                                     [:attributes :attributes :stable-array-captures])))))
          (fail! :typed-soac-stable-read-alias
                 "a stable tensor input must not alias this operation's output storage"
                 {:equation equation-id
                  :destinations destinations
                  :stable-array-captures
                  (get-in (operation-parts equation)
                          [:attributes :attributes :stable-array-captures])}))
        (when-not (= (count destinations) (count (distinct destinations)))
          (fail! :typed-soac-result-storage-alias
                 "one pointwise equation may write each physical destination only once"
                 {:equation equation-id :destinations destinations}))
        (doseq [[result destination] (map vector results destinations)]
          (when-not (= destination (get aliases result))
            (fail! :typed-soac-result-storage-alias
                   "every stored functional result must explicitly alias its physical destination"
                   {:equation equation-id :result result :destination destination
                    :aliases aliases})))
        (when-not (contains? (get-in program-facts [:equations equation-id :effects])
                             :memory/write)
          (fail! :typed-soac-result-storage-effect
                 "physical result storage requires an explicit memory-write effect"
                 {:equation equation-id :storage storage}))
        (doseq [[result destination] (map vector results destinations)
                :let [logical (get-in program-facts [:values result])
                      physical (get-in program-facts [:values destination])]]
          (when-not physical
            (fail! :typed-soac-result-storage-value
                   "a physical result destination requires an AbstractValue"
                   {:equation equation-id :result result :destination destination}))
          (when (and logical physical
                     (not= (dissoc logical :shape) (dissoc physical :shape)))
            (fail! :typed-soac-result-storage-type
                   "logical result and physical destination storage types disagree"
                   {:equation equation-id :result result :destination destination
                    :logical logical :physical physical})))))))

(defn validate!
  "Validate syntax, typed value references, ordered SSA definitions and explicit effects.
   Returns the input program unchanged."
  [program]
  (when-not (program-form? program)
    (fail! :typed-soac-program "expected a soac-program form" {:program program}))
  (when-not (dialect/valid? TypedSOAC program)
    (fail! :typed-soac-syntax "program does not conform to the TypedSOAC pattern dialect"
           {:details (dialect/validate TypedSOAC program)}))
  (let [program-facts (facts program)
        values (:values program-facts)
        program-inputs (:inputs program-facts)
        program-outputs (outputs program)
        equations (equations program)
        equation-ids (mapv second equations)
        equation-fact-ids (set (keys (:equations program-facts)))]
    (doseq [[id value] values]
      (try
        (av/validate! value)
        (catch clojure.lang.ExceptionInfo exception
          (fail! :typed-soac-value "invalid AbstractValue in typed SOAC program"
                 {:id id :cause (ex-data exception)}))))
    (when-not (distinct-vector? program-inputs)
      (fail! :typed-soac-inputs "program inputs must be an ordered distinct vector"
             {:inputs program-inputs}))
    (when-not (distinct-vector? program-outputs)
      (fail! :typed-soac-outputs "program outputs must be an ordered distinct vector"
             {:outputs program-outputs}))
    (when-not (= (count equation-ids) (count (distinct equation-ids)))
      (fail! :typed-soac-equation-ids "equation IDs must be unique"
             {:equations equation-ids}))
    (when-not (= equation-fact-ids (set equation-ids))
      (fail! :typed-soac-equation-facts
             "equation fact keys must exactly match the expression spine"
             {:facts equation-fact-ids :equations (set equation-ids)}))
    (doseq [equation equations]
      (validate-equation! equation)
      (validate-equation-types! values equation)
      (validate-result-storage! program-facts equation))
    (let [definitions (mapcat #(nth % 2) equations)
          definition-set (set definitions)
          references (set (mapcat (fn [equation]
                                    (into (operation-inputs equation)
                                          (filter value-id? (operation-extents equation))))
                                  equations))
          external (set/difference references definition-set)
          total-effects (reduce set/union #{}
                                (map :effects (vals (:equations program-facts))))]
      (when-not (= (count definitions) (count definition-set))
        (fail! :typed-soac-definitions "logical values may be defined by only one equation"
               {:definitions definitions}))
      (let [storage-destinations
            (set (mapcat (fn [equation]
                           (map :destination
                                (or (result-storage program-facts (second equation)) [])))
                         equations))]
        (doseq [id (set/union definition-set references storage-destinations
                              (set program-inputs) (set program-outputs))]
          (when-not (contains? values id)
            (fail! :typed-soac-unknown-value "SOAC program references an unknown value"
                   {:id id}))))
      (when-not (= external (set program-inputs))
        (fail! :typed-soac-input-boundary
               "program inputs must exactly name values used but not defined"
               {:declared (set program-inputs) :inferred external}))
      (loop [available (set program-inputs)
             [equation & remaining] equations]
        (when equation
          (let [equation-id (second equation)
                required (into (set (operation-inputs equation))
                               (filter value-id? (operation-extents equation)))
                missing (set/difference required available)]
            (when (seq missing)
              (fail! :typed-soac-use-before-definition
                     "equations may use only program inputs or earlier equation results"
                     {:equation equation-id :missing missing :available available}))
            (recur (into available (nth equation 2)) remaining))))
      (when-not (set/subset? (set program-outputs) definition-set)
        (fail! :typed-soac-output-boundary "program outputs must be equation results"
               {:outputs program-outputs :definitions definition-set}))
      (when-not (= total-effects (:effects program-facts))
        (fail! :typed-soac-effects "program effects must equal the union of equation effects"
               {:declared (:effects program-facts) :inferred total-effects}))))
  program)

(defn make
  "Construct and validate a typed SOAC program."
  [program-facts equation-forms program-outputs]
  (validate! (list 'soac-program program-facts (vec equation-forms) (vec program-outputs))))

(defn remap-values
  "Alpha-rename stable value IDs throughout a TypedSOAC program.

   Lambda parameters are lexical binders and are deliberately untouched. This is the bridge used
   when a typed algorithm enters ParallelProgram's SSA envelope: logical source binders become the
   envelope's authoritative value IDs without reparsing the scalar region or losing its facts."
  [program value-map]
  (let [program (validate! program)
        rename #(get value-map % %)
        rename-dimension
        (fn [dimension]
          (cond
            (contains? value-map dimension)
            (let [id (rename dimension)]
              (if (symbol? id) id (list 'value id)))

            (and (seq? dimension) (contains? #{'value 'unknown-dimension} (first dimension))
                 (= 2 (count dimension))
                 (contains? value-map (second dimension)))
            (list (first dimension) (rename (second dimension)))

            :else dimension))
        rename-value (fn [value] (update value :shape #(mapv rename-dimension %)))
        source-facts (facts program)
        values (reduce-kv (fn [result id value]
                            (let [id' (rename id)]
                              (when (contains? result id')
                                (fail! :typed-soac-remap-collision
                                       "value remapping collapses distinct TypedSOAC IDs"
                                       {:target id' :mapping value-map}))
                              (assoc result id' (rename-value value))))
                          {} (:values source-facts))
        equation-forms
        (mapv (fn [[equals equation-id results :as equation]]
                (let [{:keys [kind attributes arrays captures destinations lambda
                              element-lambda combine-lambda folds map-lambda]}
                      (operation-parts equation)
                      attributes (cond-> attributes
                                   (seq (get-in attributes
                                                [:attributes :stable-array-captures]))
                                   (update-in [:attributes :stable-array-captures]
                                              #(mapv rename %)))
                      attributes (if (contains? #{'segmented-reduce 'product-reduce
                                                  'segmented-fold-map} kind)
                                   (update attributes :segment-axes
                                           #(mapv (fn [[index extent]]
                                                    [index (if (value-id? extent)
                                                             (rename extent) extent)]) %))
                                   attributes)
                      attributes (cond-> attributes
                                   (:result-transform attributes)
                                   (update-in [:result-transform :operands]
                                              #(mapv (fn [operand]
                                                       (update operand :value rename)) %))
                                   (:result-transform attributes)
                                   (update-in [:result-transform :scalars]
                                              #(mapv (fn [scalar]
                                                       (update scalar :value rename)) %)))
                      attributes (if (= 'scalar kind)
                                   attributes
                                   (update attributes :extent
                                           #(if (value-id? %) (rename %) %)))
                      operation (case kind
                                  scalar
                                  (list kind attributes (mapv rename captures) lambda)

                                  product-reduce
                                  (list kind attributes (mapv rename arrays)
                                        (mapv rename captures) element-lambda combine-lambda)

                                  segmented-fold-map
                                  (list kind attributes (mapv rename arrays)
                                        (mapv rename captures)
                                        (mapv (fn [{:keys [attributes lambda]}]
                                                (list 'fold
                                                      (update attributes :extent
                                                              #(util/subst-syms value-map %))
                                                      lambda))
                                              folds)
                                        map-lambda)

                                  effect-map
                                  (list kind attributes (mapv rename arrays)
                                        (mapv rename captures) (mapv rename destinations)
                                        lambda)

                                  (list kind attributes (mapv rename arrays)
                                        (mapv rename captures) lambda))]
                  (list equals equation-id (mapv rename results) operation)))
              (equations program))
        rename-aliases
        (fn [aliases]
          (into {} (map (fn [[left right]] [(rename left) (rename right)])) aliases))
        equation-facts
        (into {}
              (map (fn [[id equation-facts]]
                     [id (-> equation-facts
                             (update :aliases rename-aliases)
                             (cond-> (seq (get-in equation-facts
                                                  [:attributes :result-storage]))
                               (update-in [:attributes :result-storage]
                                          #(mapv (fn [storage]
                                                   (update storage :destination rename))
                                                 %))))]))
              (:equations source-facts))
        facts' (assoc source-facts
                      :values values
                      :inputs (mapv rename (:inputs source-facts))
                      :equations equation-facts)]
    (make facts' equation-forms (mapv rename (outputs program)))))

(defn default-equation-facts
  ([] (default-equation-facts {}))
  ([provenance]
   {:effects #{} :aliases {} :provenance provenance :attributes {}}))

(defn default-program-facts
  [{:keys [values inputs equations effects diagnostics provenance attributes]
    :or {values {} inputs [] equations {} effects #{} diagnostics [] provenance {} attributes {}}}]
  {:values values
   :inputs (vec inputs)
   :equations equations
   :effects effects
   :diagnostics (vec diagnostics)
   :provenance provenance
   :attributes attributes})
