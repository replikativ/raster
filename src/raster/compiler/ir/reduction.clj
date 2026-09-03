(ns raster.compiler.ir.reduction
  "Canonical typed reduction operators.

   A scalar reduction is the one-component case.  Product reductions keep accumulator components,
   neutral values and step results in one ordered record so lowering cannot silently lose a value
   or guess its dtype.  The operator is semantic: segmentation and target schedules live in SOAC
   and SegOp layers, while physical scratch is introduced only during scheduling/materialization."
  (:require [clojure.walk :as walk]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.types :as types]
            [raster.compiler.ir.scan :as scan]))

(defrecord ReductionComponent
           [id accumulator neutral dtype result attributes])

(defrecord ReductionRegion
           [bindings results attributes])

(defrecord CombineRegion
           [parameters bindings results attributes])

(defrecord ProductReduction
           [components index element combine step algebra attributes])

(defrecord ReductionSchedule
           [strategy workgroup-size stages tuning-space numerical-mode attributes])

(defn component?
  [value]
  (and value
       (= "raster.compiler.ir.reduction.ReductionComponent" (.getName (class value)))))

(defn product-reduction?
  [value]
  (and value
       (= "raster.compiler.ir.reduction.ProductReduction" (.getName (class value)))))

(defn region?
  [value]
  (and value
       (= "raster.compiler.ir.reduction.ReductionRegion" (.getName (class value)))))

(defn combine-region?
  [value]
  (and value
       (= "raster.compiler.ir.reduction.CombineRegion" (.getName (class value)))))

(defn schedule?
  [value]
  (and value
       (= "raster.compiler.ir.reduction.ReductionSchedule" (.getName (class value)))))

(defn validate-schedule!
  [schedule]
  (when-not (schedule? schedule)
    (throw (ex-info "expected a ReductionSchedule"
                    {:reason :reduction-schedule-type :schedule schedule})))
  (let [{:keys [strategy workgroup-size stages tuning-space numerical-mode attributes]} schedule]
    (when-not (keyword? strategy)
      (throw (ex-info "reduction schedule strategy must be a keyword"
                      {:reason :reduction-schedule-strategy :strategy strategy})))
    (when-not (and (integer? workgroup-size) (pos? workgroup-size)
                   (zero? (bit-and workgroup-size (dec workgroup-size))))
      (throw (ex-info "reduction schedule workgroup size must be a positive power of two"
                      {:reason :reduction-schedule-workgroup :workgroup-size workgroup-size})))
    (when-not (and (vector? stages) (seq stages) (every? keyword? stages))
      (throw (ex-info "reduction schedule stages must be an ordered non-empty keyword vector"
                      {:reason :reduction-schedule-stages :stages stages})))
    (doseq [[field value] [[:tuning-space tuning-space]
                           [:numerical-mode numerical-mode]
                           [:attributes attributes]]]
      (when-not (map? value)
        (throw (ex-info "reduction schedule facets must be explicit maps"
                        {:reason :reduction-schedule-facet :field field :value value})))))
  schedule)

(defn schedule
  [{:keys [strategy workgroup-size stages tuning-space numerical-mode attributes]
    :or {tuning-space {} numerical-mode {} attributes {}}}]
  (validate-schedule!
   (->ReductionSchedule strategy workgroup-size (vec stages) tuning-space numerical-mode attributes)))

(defn- distinct-vector?
  [xs]
  (and (vector? xs) (= (count xs) (count (distinct xs)))))

(defn validate!
  "Validate and return a ProductReduction unchanged.

   `result` is either a logical output symbol or nil when the component participates in the
   reduction but is not externally materialized (argmax's winning value is the common example)."
  [reduction]
  (when-not (product-reduction? reduction)
    (throw (ex-info "expected a ProductReduction"
                    {:reason :product-reduction-type :value reduction})))
  (let [{:keys [components index element combine step algebra attributes]} reduction]
    (when-not (and (vector? components) (seq components) (every? component? components))
      (throw (ex-info "product reduction requires an ordered non-empty component vector"
                      {:reason :product-reduction-components :components components})))
    (doseq [{:keys [id accumulator dtype result attributes] :as component} components]
      (when (nil? id)
        (throw (ex-info "reduction component IDs cannot be nil"
                        {:reason :product-reduction-component-id :component component})))
      (when-not (symbol? accumulator)
        (throw (ex-info "reduction accumulators must be symbols"
                        {:reason :product-reduction-accumulator :component component})))
      (when-not (keyword? dtype)
        (throw (ex-info "reduction component dtype must be explicit"
                        {:reason :product-reduction-dtype :component component})))
      (when-not (or (nil? result) (symbol? result))
        (throw (ex-info "reduction component result must be a symbol or nil"
                        {:reason :product-reduction-result :component component})))
      (when-not (map? attributes)
        (throw (ex-info "reduction component attributes must be a map"
                        {:reason :product-reduction-component-attributes
                         :component component}))))
    (doseq [[field values] [[:ids (mapv :id components)]
                            [:accumulators (mapv :accumulator components)]]]
      (when-not (distinct-vector? values)
        (throw (ex-info "reduction component identities must be unique"
                        {:reason :product-reduction-duplicate :field field :values values}))))
    (when-not (symbol? index)
      (throw (ex-info "product reduction index must be a symbol"
                      {:reason :product-reduction-index :index index})))
    (when-not (or (and (region? step) (nil? element) (nil? combine))
                  (and (nil? step) (region? element) (combine-region? combine)))
      (throw (ex-info "reduction requires either one legacy fold region or element+combine regions"
                      {:reason :product-reduction-regions
                       :step step :element element :combine combine})))
    (doseq [[kind region] (remove (comp nil? second) [[:step step] [:element element]])]
      (let [{:keys [bindings results attributes]} region]
        (when-not (and (vector? bindings) (even? (count bindings)))
          (throw (ex-info "reduction region bindings must be a flat binding vector"
                          {:reason :product-reduction-region-bindings
                           :region kind :bindings bindings})))
        (let [bound-syms (vec (take-nth 2 bindings))]
          (when-not (and (every? symbol? bound-syms) (distinct-vector? bound-syms))
            (throw (ex-info "reduction region binding symbols must be unique"
                            {:reason :product-reduction-region-binders
                             :region kind :symbols bound-syms}))))
        (when-not (and (vector? results) (= (count components) (count results)))
          (throw (ex-info "product reduction step result arity must equal component arity"
                          {:reason :product-reduction-arity
                           :region kind :components (count components)
                           :region-results (count results)})))
        (when-not (map? attributes)
          (throw (ex-info "reduction region attributes must be a map"
                          {:reason :product-reduction-region-attributes
                           :region kind :attributes attributes})))))
    (when combine
      (let [{:keys [parameters bindings results attributes]} combine
            flat-params (vec (mapcat identity parameters))]
        (when-not (and (vector? parameters) (= (count components) (count parameters))
                       (every? #(and (vector? %) (= 2 (count %)) (every? symbol? %)) parameters)
                       (distinct-vector? flat-params))
          (throw (ex-info "combine parameters must be one distinct [left right] pair per component"
                          {:reason :product-reduction-combine-parameters
                           :parameters parameters :components (count components)})))
        (when-not (and (vector? bindings) (even? (count bindings)))
          (throw (ex-info "combine bindings must be a flat binding vector"
                          {:reason :product-reduction-combine-bindings :bindings bindings})))
        (when-not (and (vector? results) (= (count components) (count results)))
          (throw (ex-info "combine result arity must equal component arity"
                          {:reason :product-reduction-combine-arity
                           :components (count components) :results (count results)})))
        (when-not (map? attributes)
          (throw (ex-info "combine region attributes must be a map"
                          {:reason :product-reduction-combine-attributes
                           :attributes attributes})))))
    (when-not (map? algebra)
      (throw (ex-info "product reduction algebra facts must be an explicit map"
                      {:reason :product-reduction-algebra :algebra algebra})))
    (when-not (map? attributes)
      (throw (ex-info "product reduction attributes must be a map"
                      {:reason :product-reduction-attributes :attributes attributes}))))
  reduction)

(defn make
  [{:keys [components index step element-bindings element-results combine-parameters
           combine-bindings combine-results algebra attributes]
    :or {algebra {} attributes {}}}]
  (validate!
   (->ProductReduction
    (mapv (fn [component]
            (if (component? component)
              component
              (map->ReductionComponent (merge {:attributes {}} component))))
          components)
    index
    (when (some? element-results)
      (->ReductionRegion (vec element-bindings) (vec element-results) {}))
    (when (some? combine-results)
      (->CombineRegion (vec combine-parameters) (vec combine-bindings)
                       (vec combine-results) {}))
    step algebra attributes)))

(defn- retained-scalar-dtype
  "The walker-stamped scalar dtype of an expression, or nil when none is retained."
  [expression]
  (when (instance? clojure.lang.IObj expression)
    (some-> (types/sym-type-tag expression) dtype/dtype-for-scalar-tag)))

(defn- declare-element-conversion
  "Make the element's conversion into the accumulator's carrier explicit.

   The monoid is certified at the accumulator dtype, so an element computed at another
   floating-point precision (a double-typed scalar scaling a float load) must be converted before
   it enters the combine. The owner of the reduction declares that conversion here as an explicit
   cast the KernelBody lowering accepts; no schedule or emitter may invent a narrowing on its own."
  [step-result element dtype]
  (let [element-dtype (retained-scalar-dtype element)]
    (if (and element-dtype (dtype/fp-dtype? element-dtype) (dtype/fp-dtype? dtype)
             (not= element-dtype (dtype/canon dtype)))
      (let [cast-tag (dtype/scalar-tag-for-dtype (dtype/canon dtype))
            converted (with-meta (list (symbol "clojure.core" (name cast-tag)) element)
                        {:raster.type/tag cast-tag})]
        (walk/postwalk-replace {element converted} step-result))
      step-result)))

(defn scalar
  "Construct the canonical, proof-carrying representation of `raster.par/reduce`.

   Every scalar ProductReduction has parallel semantics, so its recurrence is certified here—the
   single constructor boundary—rather than rediscovered by individual schedules or emitters. An
   element whose retained precision differs from the accumulator's is wrapped in an explicit cast
   at this boundary."
  [{:keys [accumulator neutral dtype result index step-result algebra attributes]
    :or {algebra {} attributes {}}}]
  (let [certified (scan/certify-reassociation
                   {:acc accumulator :init neutral :lambda step-result} dtype)
        declared (declare-element-conversion step-result (:element certified) dtype)
        converted? (not (identical? declared step-result))
        step-result declared
        derived (if converted?
                  (scan/certify-reassociation
                   {:acc accumulator :init neutral :lambda step-result} dtype)
                  certified)
        algebra (if (empty? algebra) derived algebra)]
    (when-not (scan/compatible-certificate? algebra derived)
      (throw (ex-info "scalar reduction algebra disagrees with its recurrence"
                      {:reason :scalar-reduction-certificate-mismatch
                       :declared algebra :derived derived})))
    (make {:components [{:id :value
                         :accumulator accumulator
                         :neutral neutral
                         :dtype dtype
                         :result result}]
           :index index
           :step (->ReductionRegion [] [step-result] {})
           :algebra algebra
           :attributes attributes})))

(defn scalar?
  [reduction]
  (= 1 (count (:components (validate! reduction)))))

(defn accumulators [reduction] (mapv :accumulator (:components (validate! reduction))))
(defn neutrals [reduction] (mapv :neutral (:components (validate! reduction))))
(defn dtypes [reduction] (mapv :dtype (:components (validate! reduction))))
(defn results [reduction] (mapv :result (:components (validate! reduction))))
(defn fold-region [reduction] (:step (validate! reduction)))
(defn element-region [reduction] (:element (validate! reduction)))
(defn combine-region [reduction] (:combine (validate! reduction)))

(defn scalar-op
  "Project a one-component reduction into the legacy scalar emitter vocabulary.  This is a
   temporary target adapter, not a second semantic representation."
  [reduction]
  (when-not (scalar? reduction)
    (throw (ex-info "scalar emitter cannot consume a product reduction"
                    {:reason :product-reduction-needs-schedule
                     :components (count (:components reduction))})))
  (let [component (first (:components reduction))]
    {:acc (:accumulator component)
     :init (:neutral component)
     :lambda (first (:results (fold-region reduction)))}))
