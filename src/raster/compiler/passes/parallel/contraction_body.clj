(ns raster.compiler.passes.parallel.contraction-body
  "Lower a verified scalar contraction schedule to target-neutral KernelBody.

   The portable baseline assigns one output segment to one work-item and represents the reduced
   axis as a typed loop-carried fold. It is deliberately simple but fully scheduled: launch
   geometry, segment decomposition, buffer shapes, loads, scalar SSA and the final store are IR.
   Target emitters only choose syntax."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.index-expression :as index-expression]
            [raster.compiler.passes.parallel.scalar-region-lower :as scalar-region-lower]
            [raster.compiler.passes.parallel.segred-body :as segred-body]))

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
  (index-expression/lower expression scope decline!))

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
        core-operand-ids (set (map :sym (:operands contract-facts)))
        axes (concat (:free-axes contract-facts) (:contract-axes contract-facts))
        axis-indices (set (map first axes))
        core-symbols (reduce set/union #{}
                             (map util/free-syms
                                  (conj (mapv second axes) (:body contract-facts))))
        core-scalar-ids (set/difference core-symbols core-operand-ids axis-indices
                                        #{(:out contract-facts)})
        ;; The scheduled TypedSOAC operation exposes its complete physical boundary, including
        ;; result-transform captures. Core contraction loads and epilogue loads have different
        ;; layout/dtype rules, so partition those roles from the verified facts instead of
        ;; rebuilding a narrower SegRed from a generated host form.
        arrays (vec (sort-by name (set/intersection (:inputs segred) core-operand-ids)))
        scalars (vec (sort-by name (set/intersection (:scalars segred) core-scalar-ids)))
        output (:out contract-facts)
        ;; A result transform that reads the destination (`C := acc + beta·C`) reads the element
        ;; this work item stores: the destination is one read-write parameter, never a second
        ;; read-only view of the same storage.
        destination-read? (boolean (some #(= output (:sym %)) transform-operands))
        transform-only-operands
        (filterv #(not (or (contains? (set arrays) (:sym %)) (= output (:sym %))))
                 transform-operands)
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
        launch-segment-count (index-expression/to-launch-expression segment-count decline!)
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
        ;; Operand indices in the certified contraction body are already the authoritative
        ;; physical coordinate expressions.  `operand-maps` prove those expressions and provide
        ;; storage extents; they are not a wrapper around the expression handed back by
        ;; `lower-element-operations`.
        element-coordinate-lower #(lower-index % index-scope)
        epilogue-coordinate-lower #(lower-index (axis-map/index-expr %) index-scope)
        {:keys [operations result]}
        (segred-body/lower-element-operations
         element {:index reduced-index :coordinate reduced-index :dtype dtype
                  ;; The verified contraction region declares the product result dtype even when
                  ;; a synthetic surface fixture carries no walker metadata on its outer call.
                  ;; This fact applies only to the region result; nested scalar calls remain typed
                  ;; by their own retained source facts.
                  :declared-result-dtype dtype
                  :arrays (set arrays) :scalars (set scalars)
                  :coordinate-lower element-coordinate-lower
                  :load-predicate active-mask
                  :load-other (body/literal identity dtype)})
        accumulator 'segment-accumulator
        next-accumulator 'next-segment-accumulator
        reduction-result 'segment-result
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
              [(body/->KernelParameter output (if destination-read? :inout :output) dtype
                                       [segment-count] :global
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
              :coordinate-lower epilogue-coordinate-lower
              :predicate active-mask}))
          stored-result (or (:result lowered-transform) reduction-result)]
      {:kernel-body
       (body/make
        {:id [:contraction (:id segred) :portable-sequential]
         :parameters parameters
         :stable-reads (mapv body/stable-read
                             (distinct (remove #{output}
                                               (concat arrays (map :sym transform-operands)))))
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
                   :group-count [(launch/ceil-div launch-segment-count workgroup-size)]})
         :provenance {:dialect :kernel-body :source-dialect :segcontract
                      :segop-id (:id segred)}
         :attributes {:kind :portable-contraction
                      :identity identity :operator operator
                      :segment-count segment-count
                      :launch-segment-count launch-segment-count
                      :reduced-bound reduced-bound
                      :axis-symbols (vec (concat (map :name segment-dims)
                                                 [reduced-index]))
                      :result-transform result-transform}})
       :arrays arrays
       :scalars scalars
       :output output
       :segment-count segment-count
       :launch-segment-count launch-segment-count
       :workgroup-size workgroup-size})))
