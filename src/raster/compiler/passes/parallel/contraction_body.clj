(ns raster.compiler.passes.parallel.contraction-body
  "Lower a verified scalar contraction schedule to target-neutral KernelBody.

   The portable baseline assigns one output segment to one work-item and represents the reduced
   axis as a typed loop-carried fold. It is deliberately simple but fully scheduled: launch
   geometry, segment decomposition, buffer shapes, loads, scalar SSA and the final store are IR.
   Target emitters only choose syntax."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.scalar-region-lower :as scalar-region-lower]
            [raster.compiler.passes.parallel.segred-body :as segred-body]))

(def ^:private index-operators
  {'+ :add, 'clojure.core/+ :add, 'raster.numeric/+ :add
   '- :sub, 'clojure.core/- :sub, 'raster.numeric/- :sub
   '* :mul, 'clojure.core/* :mul, 'raster.numeric/* :mul
   'quot :floor-div, 'clojure.core/quot :floor-div
   'rem :mod, 'clojure.core/rem :mod
   'mod :mod, 'clojure.core/mod :mod
   'min :min, 'clojure.core/min :min
   'max :max, 'clojure.core/max :max})

(def ^:private index-casts
  '#{int long clojure.core/int clojure.core/long})

(defn- decline!
  [rule message data]
  (throw (ex-info message (assoc data
                                 :reason :contraction-kernel-body-declined
                                 :missing-rule rule
                                 :fallback :verified-segmented-opencl))))

(defn declined?
  [exception]
  (or (= :contraction-kernel-body-declined (:reason (ex-data exception)))
      (scalar-region-lower/declined? exception)))

(defn lower-index
  "Translate verified source index arithmetic into KernelBody index expressions."
  [expression scope]
  (let [expression (descriptor/unwrap-int-cast expression)]
    (cond
      (integer? expression) expression

      (symbol? expression)
      (if (contains? scope expression)
        expression
        (decline! :unbound-index-symbol
                  "portable contraction index references an undeclared symbol"
                  {:expression expression :scope scope}))

      (and (seq? expression) (contains? index-casts (first expression))
           (= 2 (count expression)))
      (lower-index (second expression) scope)

      (seq? expression)
      (let [operator (get index-operators (descriptor/semantic-op expression))
            arguments (vec (descriptor/call-args expression))]
        (when-not (and operator (seq arguments)
                       (or (not= :sub operator) (= 2 (count arguments))))
          (decline! :index-expression
                    "portable contraction requires explicit integer affine/decomposition arithmetic"
                    {:expression expression :operator (descriptor/semantic-op expression)}))
        (apply body/expression operator (map #(lower-index % scope) arguments)))

      :else
      (decline! :index-expression
                "portable contraction index has an unsupported value"
                {:expression expression :type (type expression)}))))

(defn- product-expression
  [values]
  (case (count values)
    0 1
    1 (first values)
    (apply body/expression :mul values)))

(defn lower
  "Apply a portable one-work-item-per-segment schedule to a verified contraction SegRed."
  [contract-facts segred {:keys [workgroup-size array-types scalar-types]
                          :or {workgroup-size 256 array-types {} scalar-types {}}}]
  (when-not (facts/facts? contract-facts)
    (throw (ex-info "portable contraction lowering requires verified facts"
                    {:reason :raster/bug :facts contract-facts})))
  (let [space (:space segred)
        segment-dims (segop/seg-space-segment-dims space)
        reduced-dim (segop/seg-space-reduced-dim space)
        _ (when (empty? segment-dims)
            (decline! :no-segments
                      "portable contraction body requires at least one free axis"
                      {:segred-id (:id segred)}))
        dtype (dtype/canon (:dtype segred))
        result-transform (:epilogue contract-facts)
        result-region (scalar-region-lower/make-region result-transform)
        transform-operands (vec (:operands result-transform))
        transform-scalars (vec (:scalars result-transform))
        arrays (vec (sort-by name (:inputs segred)))
        scalars (vec (sort-by name (:scalars segred)))
        transform-only-operands
        (filterv #(not (contains? (set arrays) (:sym %))) transform-operands)
        transform-only-scalars
        (filterv #(not (contains? (set scalars) (:sym %))) transform-scalars)
        scalar-dtype (fn [id] (or (get scalar-types id)
                                  (get scalar-types (symbol (name id))) :int))
        array-dtype (fn [id] (dtype/canon
                              (or (get array-types id)
                                  (get array-types (symbol (name id))) dtype)))
        _ (when-not (and (dtype/known? dtype)
                         (integer? workgroup-size) (pos? workgroup-size)
                         (zero? (bit-and workgroup-size (dec workgroup-size)))
                         (every? #(= dtype (array-dtype %)) arrays)
                         (every? #(contains? #{:int :long} (dtype/canon (scalar-dtype %))) scalars))
            (decline! :storage-contract
                      "portable contraction requires uniform operand dtype and integral dimensions"
                      {:dtype dtype :arrays arrays :array-types array-types
                       :scalars scalars :scalar-types scalar-types
                       :workgroup-size workgroup-size}))
        operand-maps (into {}
                           (map (fn [array]
                                  [array (facts/operand-axis-map contract-facts array)]))
                           arrays)
        _ (when-let [missing (seq (keep (fn [[array amap]] (when-not amap array)) operand-maps))]
            (decline! :operand-layout
                      "portable contraction could not prove a dense physical map for every operand"
                      {:operands (vec missing)
                       :indices (mapv (juxt :sym :idx) (:operands contract-facts))}))
        reduced-index (:name reduced-dim)
        axis-symbols (set (concat (map :name segment-dims) [reduced-index]))
        index-scope (into axis-symbols scalars)
        segment-count-source (segop/seg-space-num-segments-expr space)
        segment-count (lower-index segment-count-source index-scope)
        reduced-bound (lower-index (:bound reduced-dim) index-scope)
        segment-index 'segment-index
        group-index 'segment-group
        local-index 'segment-lane
        active-mask :active-segment
        decomposition
        (mapv
         (fn [position {:keys [name bound]}]
           (let [following (subvec (vec segment-dims) (inc position))
                 divisor (product-expression
                          (mapv #(lower-index (:bound %) index-scope) following))
                 quotient (if (= 1 divisor)
                            segment-index
                            (body/expression :floor-div segment-index divisor))]
             (body/->IndexCompute
              name (body/expression :mod quotient (lower-index bound index-scope)))))
         (range) segment-dims)
        {:keys [operator identity element]} (segred-body/scalar-plan segred)
        _ (when-not (number? identity)
            (decline! :literal-identity
                      "portable contraction requires a typed numeric reduction identity"
                      {:identity identity :segred-id (:id segred)}))
        coordinate-lower #(lower-index % index-scope)
        {:keys [operations result]}
        (segred-body/lower-element-operations
         element {:index reduced-index :coordinate reduced-index :dtype dtype
                  :arrays (set arrays) :scalars (set scalars)
                  :coordinate-lower coordinate-lower
                  :load-predicate active-mask
                  :load-other (body/literal identity dtype)})
        accumulator 'segment-accumulator
        next-accumulator 'next-segment-accumulator
        reduction-result 'segment-result
        output (:out contract-facts)
        physical-extent
        (fn [array]
          (lower-index (axis-map/n-elements (get operand-maps array)) index-scope))
        transform-operand-parameters
        (mapv (fn [{:keys [sym dtype map]}]
                (let [dtype (dtype/canon dtype)
                      extent (lower-index (axis-map/n-elements map) index-scope)]
                  (body/->KernelParameter
                   sym :input dtype [extent] :global
                   (layout/row-major [extent] dtype) :epilogue)))
              transform-only-operands)
        transform-scalar-parameters
        (mapv (fn [{:keys [sym dtype]}]
                (body/->KernelParameter sym :scalar (dtype/canon dtype) [] nil nil :epilogue))
              transform-only-scalars)
        parameters
        (vec (concat
              (map (fn [array]
                     (let [extent (physical-extent array)]
                       (body/->KernelParameter
                        array :input dtype [extent] :global
                        (layout/row-major [extent] dtype) :operand)))
                   arrays)
              [(body/->KernelParameter output :output dtype [segment-count] :global
                                       (layout/row-major [segment-count] dtype) :result)]
              (map #(body/->KernelParameter % :scalar (scalar-dtype %) [] nil nil :parameter)
                   scalars)
              transform-operand-parameters
              transform-scalar-parameters
              [(body/->KernelParameter '_nseg :scalar :int [] nil nil :bound)]))]
    (let [parameter-map (into {} (map (juxt :id clojure.core/identity)) parameters)
          lowered-transform
          (when result-region
            (scalar-region-lower/lower
             result-region
             {:accumulator reduction-result
              :accumulator-dtype dtype
              :store-dtype dtype
              :parameters parameter-map
              :coordinate-lower coordinate-lower
              :predicate active-mask}))
          stored-result (or (:result lowered-transform) reduction-result)]
      {:kernel-body
       (body/make
        {:id [:contraction (:id segred) :portable-sequential]
         :parameters parameters
         :stable-reads (mapv body/stable-read
                             (distinct (concat arrays (map :sym transform-operands))))
         :indices (vec (concat
                        [(body/->IndexBinding group-index :group 0)
                         (body/->IndexBinding local-index :local 0)
                         (body/->IndexCompute
                          segment-index
                          (body/expression :add
                                           (body/expression :mul group-index workgroup-size)
                                           local-index))]
                        decomposition))
         :masks [(body/->Mask active-mask
                              [(body/predicate :lt segment-index '_nseg)])]
         :operations
         (vec
          (concat
           [(body/->ForLoop
             (body/value reduced-index :int)
             0 reduced-bound 1
             [(body/->LoopArg (body/value accumulator dtype)
                              (body/literal identity dtype))]
             (vec (concat
                   operations
                   [(body/->ScalarCompute
                     (body/value next-accumulator dtype)
                     (body/scalar-expression operator dtype [accumulator result]))
                    (body/->Yield [next-accumulator])]))
             [(body/value reduction-result dtype)]
             {})]
           (:operations lowered-transform)
           [(body/->ScalarStore output [segment-index] stored-result active-mask)]))
         :schedule {:strategy :sequential-segments
                    :workgroup-size workgroup-size
                    :reduction-operator operator}
         :launch (launch/spec
                  {:workgroup-size [workgroup-size]
                   :group-count [(launch/ceil-div segment-count workgroup-size)]})
         :provenance {:dialect :kernel-body :source-dialect :segcontract
                      :segop-id (:id segred)}
         :attributes {:kind :portable-contraction
                      :identity identity :operator operator
                      :segment-count segment-count
                      :reduced-bound reduced-bound
                      :axis-symbols (vec (concat (map :name segment-dims)
                                                 [reduced-index]))
                      :result-transform result-transform}})
       :arrays arrays
       :scalars scalars
       :output output
       :segment-count segment-count
       :workgroup-size workgroup-size})))
