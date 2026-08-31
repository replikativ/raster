(ns raster.compiler.passes.parallel.typed-soac-projection
  "Mechanical target projections from validated TypedSOAC equations.

   These functions do not recover semantics from source syntax.  They spell an already validated
   equation in the temporary surface vocabulary consumed by a target route while that route is
   migrated to accept the typed equation directly.  Keeping the projection here prevents the
   execution materializer and SegOp lowering from inventing subtly different forms."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.soac-dialect :as dialect]))

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

(defn segmented-reduce-contract-form
  "Project one validated, scalar `segmented-reduce` equation to `raster.par/contract`.

   This is a compatibility seam for contraction scheduling and host materialization, not a second
   semantic IR: captures, physical storage, axes, algebra and the scalar region all come from the
   validated TypedSOAC program."
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
        form (list 'raster.par/contract
                   (first physical-results)
                   (:segment-axes attributes)
                   [[(:index attributes) (:extent attributes)]]
                   element
                   :init (first (:identities attributes))
                   :combine combine
                   :algebra (first (:algebra attributes)))]
    (with-meta form {:raster.type/elem-type (first (:dtypes attributes))})))
