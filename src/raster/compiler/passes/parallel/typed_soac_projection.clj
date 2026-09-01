(ns raster.compiler.passes.parallel.typed-soac-projection
  "Mechanical target projections from validated TypedSOAC equations.

   These functions do not recover semantics from source syntax.  They spell an already validated
   equation in the temporary surface vocabulary consumed by a target route while that route is
   migrated to accept the typed equation directly.  Keeping the projection here prevents the
   execution materializer and SegOp lowering from inventing subtly different forms."
  (:require [clojure.walk :as walk]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.contraction-facts :as contraction-facts]
            [raster.compiler.ir.soac-dialect :as dialect]))

(defn scalar-folds->source
  "Project explicit scalar Fold terms to Raster's interpreted host vocabulary."
  [expression]
  (walk/postwalk
   (fn [form]
     (if (dialect/scalar-fold-form? form)
       (let [{:keys [attributes lambda]} (dialect/scalar-fold-parts form)
             {:keys [parameters locals body-results]} (dialect/lambda-parts lambda)
             [accumulator index] parameters]
         (when (seq locals)
           (throw (ex-info "scalar fold projection does not admit local SSA yet"
                           {:reason :typed-soac-scalar-fold-projection :fold form})))
         (with-meta
           (list 'raster.par/reduce accumulator (:identity attributes)
                 index (:extent attributes) (first body-results))
           {:raster.type/elem-type (:dtype attributes)}))
       form))
   expression))

(defn- materialize-region
  [locals body]
  (if (seq locals)
    (list 'let*
          (vec
           (mapcat (fn [{:keys [id dtype init]}]
                     (let [tag (dtype/scalar-tag-for-dtype dtype)]
                       [(with-meta id {:raster.type/tag tag})
                        (list tag init)]))
                   locals))
          body)
    body))

(defn- flat-coordinate
  [segment-axes reduced-index reduced-extent]
  (reduce (fn [coordinate [index extent]]
            (list 'clojure.core/+ (list 'clojure.core/* coordinate extent) index))
          0
          (conj (vec segment-axes) [reduced-index reduced-extent])))

(defn- scalar-fold-parts
  [attributes expression]
  (let [accumulator (first (:accumulators attributes))
        arguments (vec (when (seq? expression) (descriptor/call-args expression)))]
    ;; `par/contract` spells the fold as (combine accumulator element).  Reversing a merely
    ;; associative, non-commutative fold here would be a semantic change, so refuse that shape.
    (when-not (and (seq? expression) (= 2 (count arguments))
                   (= accumulator (first arguments)))
      (throw (ex-info "segmented reduction has no ordered scalar contract projection"
                      {:reason :typed-soac-contract-projection
                       :accumulator accumulator :expression expression})))
    {:combine (descriptor/semantic-op expression)
     :element (second arguments)}))

(defn- result-transform-epilogue
  [transform]
  (when transform
    (let [{:keys [parameters body-results]} (dialect/lambda-parts (:lambda transform))
          accumulator (first parameters)
          substitutions
          (into {}
                (concat (map (juxt :parameter :value) (:operands transform))
                        (map (juxt :parameter :value) (:scalars transform))))]
      {:acc accumulator
       :expr (util/subst-syms substitutions (first body-results))
       :operands (mapv #(-> % (assoc :sym (:value %))
                            (dissoc :value :parameter))
                       (:operands transform))
       :scalars (mapv #(-> % (assoc :sym (:value %))
                           (dissoc :value :parameter))
                      (:scalars transform))
       :dtype (:result-dtype transform)})))

(defn segmented-reduce-contract-components
  "Project one validated scalar `segmented-reduce` equation to contraction semantic components.

   This is not a second IR: captures, physical storage, axes, algebra and the scalar region all
   come from the validated TypedSOAC program."
  [program equation]
  (let [program (dialect/validate! program)
        [_ equation-id results] equation
        {:keys [kind attributes arrays captures lambda]} (dialect/operation-parts equation)
        {:keys [locals body-results]} (dialect/lambda-parts lambda)
        {:keys [accumulators elements capture-parameters]} (dialect/parameter-layout equation)
        facts (dialect/facts program)
        storage (dialect/result-storage facts equation-id)
        physical-results (dialect/physical-results facts equation)
        _ (when-not (and (= 'segmented-reduce kind)
                         (= 1 (count results))
                         (= 1 (count accumulators))
                         (= 1 (count body-results))
                         (= 1 (count physical-results))
                         (= 1 (count storage)))
            (throw (ex-info "contract projection requires one scalar segmented reduction result"
                            {:reason :typed-soac-contract-projection
                             :equation equation-id :kind kind :results results
                             :physical-results physical-results :storage storage})))
        coordinate (flat-coordinate (:segment-axes attributes)
                                    (:index attributes) (:extent attributes))
        substitutions
        (into (zipmap capture-parameters captures)
              (map (fn [parameter array]
                     [parameter (list 'clojure.core/aget array coordinate)])
                   elements arrays))
        locals (mapv #(update % :init (fn [init] (util/subst-syms substitutions init))) locals)
        folded (materialize-region
                locals
                (util/subst-syms substitutions (first body-results)))
        {:keys [combine element]} (scalar-fold-parts attributes folded)
        contraction-dtype (first (:dtypes attributes))
        result-transform (result-transform-epilogue (:result-transform attributes))]
    {:out (first physical-results)
     :free-axes (:segment-axes attributes)
     :contract-axes [[(:index attributes) (:extent attributes)]]
     :body element
     :opts (cond-> (array-map :init (first (:identities attributes))
                              :combine combine
                              :algebra (first (:algebra attributes)))
             result-transform (assoc :epilogue result-transform))
     :dtype contraction-dtype
     :metadata {:raster.type/elem-type contraction-dtype}}))

(defn segmented-reduce-contract-form
  "Spell a typed scalar segmented reduction in the temporary host/leaf target vocabulary."
  [program equation]
  (contraction-facts/surface-form
   (segmented-reduce-contract-components program equation)))

(defn- source-bindings
  [locals substitutions]
  (vec
   (mapcat (fn [{:keys [id dtype init]}]
             (let [tag (dtype/scalar-tag-for-dtype dtype)]
               [(with-meta id {:raster.type/tag tag})
                (list tag (util/subst-syms substitutions init))]))
           locals)))

(defn product-reduce-form
  "Spell one validated product-reduce equation in Raster's interpreted host vocabulary."
  [program equation]
  (let [program (dialect/validate! program)
        [_ equation-id results] equation
        {:keys [kind attributes captures element-lambda combine-lambda]}
        (dialect/operation-parts equation)
        physical-results (dialect/physical-results program equation)
        component-count (count (:component-ids attributes))
        result-components (:result-components attributes)
        outputs (reduce (fn [outputs [component result]] (assoc outputs component result))
                        (vec (repeat component-count nil))
                        (map vector result-components physical-results))
        element (dialect/lambda-parts element-lambda)
        element-substitutions (zipmap (:parameters element) captures)
        combine (dialect/lambda-parts combine-lambda)
        _ (when-not (= 'product-reduce kind)
            (throw (ex-info "product projection requires a product-reduce equation"
                            {:reason :typed-soac-product-projection
                             :equation equation-id :kind kind :results results})))
        form
        (list 'raster.par/product-reduce!
              outputs
              (mapv vector (:accumulators attributes)
                    (:identities attributes) (:dtypes attributes))
              (:segment-axes attributes) (:index attributes) (:extent attributes)
              (source-bindings (:locals element) element-substitutions)
              (mapv #(util/subst-syms element-substitutions %) (:body-results element))
              (mapv vec (partition 2 (:parameters combine)))
              (source-bindings (:locals combine) {})
              (:body-results combine)
              (:algebra attributes))]
    (with-meta form {:raster.type/elem-type
                     ((:dtypes attributes) (first result-components))})))

(defn segmented-fold-map-form
  "Spell one validated ordered fold-map in Raster's interpreted host vocabulary."
  [program equation]
  (let [program (dialect/validate! program)
        [_ equation-id results] equation
        {:keys [kind attributes captures folds map-lambda]}
        (dialect/operation-parts equation)
        physical-results (dialect/physical-results program equation)
        map-region (dialect/lambda-parts map-lambda)
        accumulators (mapv #(get-in % [:attributes :accumulator]) folds)
        capture-parameters (vec (drop (count accumulators) (:parameters map-region)))
        substitutions (zipmap capture-parameters captures)
        _ (when-not (and (= 'segmented-fold-map kind)
                         (= (count results) (count physical-results)
                            (count (:body-results map-region))))
            (throw (ex-info "fold-map projection requires aligned logical and physical results"
                            {:reason :typed-soac-segmented-fold-map-projection
                             :equation equation-id :kind kind :results results
                             :physical-results physical-results})))
        source-folds
        (mapv (fn [{:keys [attributes lambda]}]
                (let [{:keys [body-results]} (dialect/lambda-parts lambda)]
                  [(:accumulator attributes) (:identity attributes) (:dtype attributes)
                   (:extent attributes)
                   (util/subst-syms substitutions (first body-results))]))
              folds)
        form (list 'raster.par/segmented-fold-map!
                   physical-results (:segment-axes attributes)
                   (:index attributes) (:extent attributes) source-folds
                   (mapv #(util/subst-syms substitutions %)
                         (:body-results map-region)))]
    (with-meta form {:raster.type/elem-type (first (:dtypes attributes))})))

(defn stencil-form
  "Spell one validated stencil equation in Raster's interpreted host vocabulary."
  [program equation]
  (let [program (dialect/validate! program)
        [_ equation-id results] equation
        {:keys [kind attributes arrays captures lambda]} (dialect/operation-parts equation)
        physical-results (dialect/physical-results program equation)
        {:keys [parameters locals body-results]} (dialect/lambda-parts lambda)
        substitutions (zipmap parameters captures)
        result-dtype (first (:dtypes attributes))
        cast (dtype/scalar-tag-for-dtype result-dtype)
        body (util/subst-syms substitutions (first body-results))
        body (if (and (seq? body) (= cast (first body)) (= 2 (count body)))
               (second body)
               body)
        stable (vec (get-in attributes [:attributes :stable-array-captures]))
        _ (when-not (and (= 'stencil kind) (empty? arrays) (empty? locals)
                         (= 1 (count results)) (= 1 (count physical-results)))
            (throw (ex-info "stencil projection requires one closed scalar result"
                            {:reason :typed-soac-stencil-projection
                             :equation equation-id :kind kind :results results
                             :physical-results physical-results})))
        form (list 'raster.par/stencil! (first physical-results) stable
                   (:radius attributes) (:boundary attributes) cast
                   (:index attributes) (:extent attributes) body)]
    (with-meta form {:raster.type/elem-type result-dtype})))
