(ns raster.compiler.passes.parallel.soac-lower
  "SOAC → SegOp lowering with hardware-aware decisions.

  Transforms high-level SOAC combinators into concrete GPU execution
  plans (SegOps) based on device capabilities:

  Map  → single SegMap with grid-stride loop
  Reduce → single-phase (small n) or two-phase (large n):
           Phase 1: block-local shared-memory tree reduction
           Phase 2: cross-block reduction of partial results
  Scan → single-phase (n ≤ block-size) or three-stage (large n):
         Stage 1: intra-block workgroup scan
         Stage 2: scan of block totals
         Stage 3: carry-in combination (SegMap)"
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-graph :as kernel-graph]
            [raster.compiler.ir.kernel-launch :as kernel-launch]
            [raster.compiler.ir.par :as par]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.ir.soac-dialect :as soac-dialect]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.execution-plan :as execution-plan]
            [raster.compiler.passes.parallel.scalar-region-lower :as scalar-region-lower]
            [raster.compiler.passes.parallel.segred-body :as segred-body]))

(declare lower-reduce)
(declare lower-map)
(declare lower-reduce-description)
(declare lower-map-description)
(declare lower-scan-description)
(declare scan-kernel-graph-description)
(declare phase-grid)

(defn- materialize-region-locals
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

(defn typed-map-program?
  "Whether a validated one-equation TypedSOAC program is a map accepted by SegMap lowering."
  [program]
  (and (soac-dialect/program-form? program)
       (= 1 (count (soac-dialect/equations program)))
       (= 'map (soac-dialect/operation-kind (first (soac-dialect/equations program))))))

(defn lower-typed-map
  "Lower one functional TypedSOAC map to SegMap from its explicit operands and lambda binders.

   A horizontally fused map has several logical results. SegMap keeps the first result in its
   structural output slot and spells the remaining stores explicitly in the scalar region; all
   results remain declared outputs, so the ABI and memory contracts do not infer writes from the
   generated expression."
  [program device-id & {:keys [dtype] :or {dtype :double}}]
  (let [program (soac-dialect/validate! program)]
    (when-not (typed-map-program? program)
      (throw (ex-info "typed SegMap lowering requires one map equation"
                      {:reason :typed-soac-map-subset :program program})))
    (let [equation (first (soac-dialect/equations program))
          [_ equation-id results operation] equation
          [_ attributes arrays captures lambda] operation
          {:keys [locals body-results]} (soac-dialect/lambda-parts lambda)
          {:keys [elements capture-parameters]} (soac-dialect/parameter-layout equation)
          _ (when-not (and (seq results)
                           (= (count results) (count body-results)))
              (throw (ex-info "typed SegMap requires one scalar body result per logical output"
                              {:reason :typed-soac-map-subset
                               :equation equation-id :results results})))
          index (:index attributes)
          substitutions
          (into (zipmap capture-parameters captures)
                (map (fn [parameter array]
                       [parameter (list 'clojure.core/aget array index)])
                     elements arrays))
          locals (mapv #(update % :init
                                (fn [init]
                                  (util/subst-syms substitutions init)))
                       locals)
          bodies (mapv #(util/subst-syms substitutions %)
                       body-results)
          result (first results)
          facts (soac-dialect/facts program)
          values (:values facts)
          physical-results (soac-dialect/physical-results facts equation)
          secondary-stores
          (mapv (fn [logical-result secondary-result secondary-body]
                  (let [secondary-dtype (:dtype (get values logical-result))
                        cast (dtype/scalar-tag-for-dtype secondary-dtype)]
                    (list 'clojure.core/aset secondary-result index
                          (list cast secondary-body))))
                (rest results) (rest physical-results) (rest bodies))
          scalar-result (if (seq secondary-stores)
                          (list* 'do (concat secondary-stores [(first bodies)]))
                          (first bodies))
          body (materialize-region-locals locals scalar-result)
          stable-array-captures (set (get-in attributes
                                             [:attributes :stable-array-captures]))
          scalar-captures (set (remove stable-array-captures captures))
          result-dtype (or (:dtype (get values result))
                           dtype :double)
          physical-result (first physical-results)
          description {:id equation-id
                       :bound (:extent attributes)
                       :idx index
                       :lambda body
                       :scalar-region {:locals locals :result scalar-result}
                       :inputs (into (set arrays) stable-array-captures)
                       :outputs (set physical-results)
                       :scalars scalar-captures
                       :elem-type result-dtype
                       :out-sym physical-result
                       :cast-fn nil}]
      (mapv #(assoc % :algorithm-dialect :typed-soac
                    :algorithm-equation equation-id)
            (lower-map-description description device-id :dtype result-dtype)))))

(defn lower-typed-stencil
  "Lower one validated boundary-aware TypedSOAC stencil to a scheduled SegStencil."
  [program device-id & {:keys [dtype] :or {dtype :double}}]
  (let [program (soac-dialect/validate! program)
        equation (first (soac-dialect/equations program))
        [_ equation-id results] equation
        {:keys [kind attributes arrays captures lambda]}
        (soac-dialect/operation-parts equation)
        {:keys [parameters locals body-results]} (soac-dialect/lambda-parts lambda)
        facts (soac-dialect/facts program)
        physical-results (soac-dialect/physical-results facts equation)
        stable (set (get-in attributes [:attributes :stable-array-captures]))
        result-dtype (or (first (:dtypes attributes)) dtype)
        cast-fn (dtype/scalar-tag-for-dtype result-dtype)
        substitutions (zipmap parameters captures)
        body (util/subst-syms substitutions (first body-results))
        body (if (and (seq? body) (= cast-fn (first body)) (= 2 (count body)))
               (second body)
               body)]
    (when-not (and (= 'stencil kind) (= 1 (count (soac-dialect/equations program)))
                   (= 1 (count results)) (= 1 (count physical-results))
                   (empty? arrays) (empty? locals))
      (throw (ex-info "typed stencil lowering requires one closed stencil equation"
                      {:reason :typed-soac-stencil-subset :equation equation-id
                       :kind kind :results results :physical-results physical-results})))
    (let [bound (:extent attributes)
          space (segop/make-seg-space (:index attributes) bound)
          level (segop/->SegLevel :thread :virtual)
          grid (phase-grid :map device-id bound result-dtype)]
      [(assoc (segop/->SegStencil
               equation-id space level body stable #{(first physical-results)}
               (set/difference (set captures) stable) grid result-dtype
               (first physical-results) (:radius attributes) (:boundary attributes) cast-fn
               :no-write-alias)
              :algorithm-dialect :typed-soac
              :algorithm-equation equation-id)])))

(defn lower-typed-segmented-fold-map
  "Lower one ordered segmented fold-map to a single semantic SegFoldMap.

   The operation remains target-neutral: segments are parallel, folds are explicitly ordered,
   and the final map covers the complete per-segment extent. A later schedule makes loop/control
   structure and launch geometry concrete in KernelBody."
  [program device-id & {:keys [dtype] :or {dtype :double}}]
  (let [program (soac-dialect/validate! program)
        equation (first (soac-dialect/equations program))
        [_ equation-id results] equation
        {:keys [kind attributes arrays captures folds map-lambda]}
        (soac-dialect/operation-parts equation)
        facts (soac-dialect/facts program)
        physical-results (soac-dialect/physical-results facts equation)
        stable (set (get-in attributes [:attributes :stable-array-captures]))
        accumulators (mapv #(get-in % [:attributes :accumulator]) folds)
        map-region (soac-dialect/lambda-parts map-lambda)
        capture-parameters (vec (drop (count accumulators) (:parameters map-region)))
        substitutions (zipmap capture-parameters captures)
        lowered-folds
        (mapv (fn [{:keys [attributes lambda]}]
                (let [{:keys [locals body-results]} (soac-dialect/lambda-parts lambda)]
                  (assoc attributes
                         :locals (mapv #(update % :init
                                                (fn [init]
                                                  (util/subst-syms substitutions init)))
                                       locals)
                         :step (util/subst-syms substitutions (first body-results)))))
              folds)
        map-results (mapv #(util/subst-syms substitutions %)
                          (:body-results map-region))
        result-dtypes (:dtypes attributes)]
    (when-not (and (= 1 (count (soac-dialect/equations program)))
                   (= 'segmented-fold-map kind) (empty? arrays)
                   (= (count results) (count physical-results) (count map-results)
                      (count result-dtypes)))
      (throw (ex-info "typed fold-map lowering requires one closed ordered equation"
                      {:reason :typed-soac-segmented-fold-map-subset
                       :equation equation-id :kind kind :results results
                       :physical-results physical-results})))
    (let [dims (conj (mapv (fn [[index bound]] {:name index :bound bound})
                           (:segment-axes attributes))
                     {:name (:index attributes) :bound (:extent attributes)})
          space (segop/make-seg-space-nd dims)
          segment-count (segop/seg-space-num-segments-expr space)
          result-dtype (or (first result-dtypes) dtype)
          grid (phase-grid :map device-id segment-count result-dtype)]
      [(assoc (segop/->SegFoldMap
               equation-id space (:index attributes) (:extent attributes)
               lowered-folds map-results stable (vec physical-results)
               (set/difference (set captures) stable) grid result-dtypes
               :no-write-alias)
              :algorithm-dialect :typed-soac
              :algorithm-equation equation-id)])))

(defn lower-typed-scatter
  "Lower one proof-carrying TypedSOAC scatter to an explicit-store SegMap.

   The SegMap has no implicit primary result store (`out-sym` is nil); its typed scalar region
   contains every conditional indexed write or certified atomic contribution. Physical
   destinations remain both inputs and outputs because an indexed update preserves unselected
   elements."
  [program device-id & {:keys [dtype] :or {dtype :double}}]
  (let [program (soac-dialect/validate! program)
        equation (first (soac-dialect/equations program))]
    (when-not (and (= 1 (count (soac-dialect/equations program)))
                   (= 'scatter (soac-dialect/operation-kind equation)))
      (throw (ex-info "typed scatter lowering requires one scatter equation"
                      {:reason :typed-soac-scatter-subset :program program})))
    (let [[_ equation-id results operation] equation
          [_ attributes arrays captures lambda] operation
          {:keys [locals body-results]} (soac-dialect/lambda-parts lambda)
          {:keys [elements capture-parameters]} (soac-dialect/parameter-layout equation)
          substitutions
          (into (zipmap capture-parameters captures)
                (map (fn [parameter array]
                       [parameter (list 'clojure.core/aget array (:index attributes))])
                     elements arrays))
          locals (mapv #(update % :init
                                (fn [init]
                                  (-> (util/subst-syms substitutions init)
                                      par/expand-par-forms)))
                       locals)
          writes (mapv #(some-> % soac-dialect/write-parts
                                (update-vals (fn [expression]
                                               (-> (util/subst-syms substitutions expression)
                                                   par/expand-par-forms))))
                       body-results)
          facts (soac-dialect/facts program)
          physical-results (soac-dialect/physical-results facts equation)
          _ (when-not (and (= (count results) (count writes) (count physical-results))
                           (every? some? writes))
              (throw (ex-info "typed scatter requires one write per physical result"
                              {:reason :typed-soac-scatter-subset :equation equation-id
                               :results results :writes writes
                               :physical-results physical-results})))
          conflict (:conflict attributes)
          reducing? (soac-dialect/reducing-scatter-conflict? conflict)
          result-dtype (or (:dtype (get-in facts [:values (first results)])) dtype :double)
          statements
          (mapv (fn [destination {:keys [destination-index predicate value]}]
                  (let [destination (if reducing?
                                      (with-meta destination
                                        {:raster.type/tag
                                         (dtype/scalar-tag-for-dtype result-dtype)
                                         :tag (dtype/scalar-tag-for-dtype result-dtype)})
                                      destination)
                        store (if reducing?
                                (list 'raster.par/atomic-add!
                                      destination destination-index value)
                                (list 'clojure.core/aset destination destination-index value))]
                    (if (contains? #{true 1} predicate) store (list 'if predicate store))))
                physical-results writes)
          scalar-result (list* 'do statements)
          body (materialize-region-locals locals scalar-result)
          stable (set (get-in attributes [:attributes :stable-array-captures]))
          scalar-captures (set (remove stable captures))
          description {:id equation-id
                       :bound (:extent attributes)
                       :idx (:index attributes)
                       :lambda body
                       :scalar-region {:locals locals :result scalar-result}
                       :inputs (into (set arrays) stable)
                       :outputs (set physical-results)
                       :scalars scalar-captures
                       :elem-type result-dtype
                       :out-sym nil
                       :cast-fn nil}]
      (mapv #(assoc % :algorithm-dialect :typed-soac
                    :algorithm-equation equation-id
                    :write-conflict (if reducing? :reduce :unique)
                    :conflict-contract conflict)
            (lower-map-description description device-id :dtype result-dtype)))))

(defn lower-typed-effect-map
  "Lower one validated ordered effect-map to the common portable SegMap schedule boundary.

   Effects remain structured data in `:scalar-region`; no source `map-void!` form is rebuilt.
   Physical destinations come from the checked result-storage contract and every destination keeps
   its unique/reduction conflict proof for KernelBody lowering."
  [program device-id & {:keys [dtype] :or {dtype :double}}]
  (let [program (soac-dialect/validate! program)
        equation (first (soac-dialect/equations program))]
    (when-not (and (= 1 (count (soac-dialect/equations program)))
                   (= 'effect-map (soac-dialect/operation-kind equation)))
      (throw (ex-info "typed effect-map lowering requires one effect-map equation"
                      {:reason :typed-soac-effect-map-subset :program program})))
    (let [[_ equation-id results _operation] equation
          {:keys [attributes arrays captures destinations lambda]}
          (soac-dialect/operation-parts equation)
          {:keys [locals body-results]} (soac-dialect/lambda-parts lambda)
          {:keys [elements capture-parameters destination-parameters]}
          (soac-dialect/parameter-layout equation)
          facts (soac-dialect/facts program)
          values (:values facts)
          physical-results (soac-dialect/physical-results facts equation)
          result-storage (soac-dialect/result-storage facts equation-id)
          read-write-destinations
          (into #{} (keep (fn [{:keys [destination access]}]
                            (when (= :read-write access) destination)))
                result-storage)
          destination-dtypes (zipmap physical-results (:dtypes attributes))
          _ (when-not (= destinations physical-results)
              (throw (ex-info "typed effect-map destination storage changed after validation"
                              {:reason :raster/bug :equation equation-id
                               :destinations destinations :physical physical-results})))
          substitutions
          (into (zipmap capture-parameters captures)
                (concat
                 (map (fn [parameter array]
                        [parameter (list 'clojure.core/aget array (:index attributes))])
                      elements arrays)
                 (map vector destination-parameters physical-results)))
          locals (mapv #(update % :init
                                (fn [init] (util/subst-syms substitutions init)))
                       locals)
          lower-effect
          (fn lower-effect [effect]
            (let [{:keys [loop destination conflict destination-index predicate value]
                   :as parts}
                  (soac-dialect/effect-parts effect)]
              (if loop
                (let [{:keys [locals body-results]} (soac-dialect/lambda-parts (:lambda parts))]
                  {:loop {:index (:index parts)
                          :lower (:lower parts)
                          :extent (util/subst-syms substitutions (:extent parts))
                          :locals (mapv #(update % :init
                                                 (fn [init] (util/subst-syms substitutions init)))
                                        locals)
                          :effects (mapv lower-effect body-results)}})
                {:destination (util/subst-syms substitutions destination)
                 :dtype (get destination-dtypes
                             (util/subst-syms substitutions destination))
                 :conflict conflict
                 :destination-index (util/subst-syms substitutions destination-index)
                 :predicate (util/subst-syms substitutions predicate)
                 :value (util/subst-syms substitutions value)})))
          effects (mapv lower-effect body-results)
          effect-leaves (soac-dialect/effect-leaves effects)
          stable (set (get-in attributes [:attributes :stable-array-captures]))
          scalar-captures (set (remove stable captures))
          write-conflicts
          (into {} (map (fn [destination]
                          [destination (:conflict
                                        (first (filter #(= destination (:destination %))
                                                       effect-leaves)))])
                        physical-results))
          result-dtype (or (:dtype (get values (first results))) dtype :double)
          iteration-order (:iteration-order attributes)
          description {:id equation-id
                       :bound (:extent attributes)
                       :idx (:index attributes)
                       :lambda nil
                       :scalar-region {:locals locals :effects effects
                                       :iteration-order iteration-order}
                       :inputs (into (into (set arrays) stable) read-write-destinations)
                       :outputs (set physical-results)
                       :scalars scalar-captures
                       :elem-type result-dtype
                       :out-sym nil
                       :cast-fn nil}]
      (mapv #(assoc % :algorithm-dialect :typed-soac
                    :algorithm-equation equation-id
                    :write-conflict :ordered-effects
                    :effect-iteration-order iteration-order
                    :write-conflicts write-conflicts)
            (lower-map-description description device-id :dtype result-dtype)))))

(defn typed-reduce-program?
  "Whether a validated one-equation TypedSOAC program is the scalar reduction vertical currently
   accepted by SegRed lowering."
  [program]
  (and (soac-dialect/program-form? program)
       (= 1 (count (soac-dialect/equations program)))
       (= 'reduce (soac-dialect/operation-kind (first (soac-dialect/equations program))))))

(defn lower-typed-reduce
  "Lower one functional TypedSOAC reduction to SegRed without inspecting its source spelling.

   TypedSOAC keeps element values and captures as lexical lambda parameters. SegRed's temporary
   scalar-region adapter still spells element reads as `aget`; this conversion is a mechanical
   projection from the validated parameter layout, not a second analysis of the host form."
  [program device-id & {:keys [dtype] :or {dtype :double}}]
  (let [program (soac-dialect/validate! program)]
    (when-not (typed-reduce-program? program)
      (throw (ex-info "typed SegRed lowering requires one reduce equation"
                      {:reason :typed-soac-reduce-subset :program program})))
    (let [equation (first (soac-dialect/equations program))
          [_ equation-id results operation] equation
          [_ attributes arrays captures lambda] operation
          {:keys [body-results]} (soac-dialect/lambda-parts lambda)
          {:keys [accumulators elements capture-parameters]}
          (soac-dialect/parameter-layout equation)
          _ (when-not (and (= 1 (count results))
                           (= 1 (count accumulators))
                           (= 1 (count body-results))
                           (symbol? (first results)))
              (throw (ex-info "initial typed SegRed vertical supports one symbolic result"
                              {:reason :typed-soac-reduce-subset
                               :equation equation-id :results results
                               :accumulators accumulators})))
          index (:index attributes)
          substitutions
          (into (zipmap capture-parameters captures)
                (map (fn [parameter array]
                       [parameter (list 'clojure.core/aget array index)])
                     elements arrays))
          step-result (util/subst-syms substitutions (first body-results))
          result (first results)
          accumulator (first accumulators)
          accumulator-dtype (or (first (:dtypes attributes)) dtype :double)
          stable-array-captures (set (get-in attributes
                                             [:attributes :stable-array-captures]))
          scalar-captures (set (remove stable-array-captures captures))
          result-region
          (scalar-region-lower/from-typed-result-transform (:result-transform attributes))
          operator (reduction/scalar
                    {:accumulator accumulator
                     :neutral (first (:identities attributes))
                     :dtype accumulator-dtype
                     :result result
                     :index index
                     :step-result step-result
                     :algebra (or (first (:algebra attributes)) {})
                     :attributes (cond-> {:source :typed-soac :equation equation-id}
                                   result-region (assoc :result-region result-region))})
          description {:id equation-id
                       :sym result
                       :reduction operator
                       :segment-axes []
                       :bound (:extent attributes)
                       :idx index
                       :inputs (into (set arrays) stable-array-captures)
                       :outputs (set results)
                       :scalars scalar-captures
                       :elem-type accumulator-dtype}]
      (mapv #(assoc % :algorithm-dialect :typed-soac
                    :algorithm-equation equation-id)
            (lower-reduce-description description device-id :dtype accumulator-dtype)))))

(defn typed-segmented-reduce-program?
  "Whether a validated one-equation TypedSOAC program is a general segmented reduction."
  [program]
  (and (soac-dialect/program-form? program)
       (= 1 (count (soac-dialect/equations program)))
       (= 'segmented-reduce
          (soac-dialect/operation-kind (first (soac-dialect/equations program))))))

(defn- flat-segment-coordinate
  [segment-axes reduced-index reduced-extent]
  (reduce (fn [coordinate [index extent]]
            (list 'clojure.core/+ (list 'clojure.core/* coordinate extent) index))
          0
          (conj (vec segment-axes) [reduced-index reduced-extent])))

(defn lower-typed-segmented-reduce
  "Lower one general TypedSOAC segmented reduction directly to SegRed.

   Segment axes are parallel result dimensions and the ordinary `:index/:extent` pair is the
   innermost reduced dimension. Stable tensor captures retain arbitrary index expressions, which
   is the general representation used by contractions; ordinary element operands denote dense
   row-major storage over the complete segment-plus-reduction space."
  [program device-id & {:keys [dtype] :or {dtype :double}}]
  (let [program (soac-dialect/validate! program)]
    (when-not (typed-segmented-reduce-program? program)
      (throw (ex-info "typed segmented reduction lowering requires one segmented-reduce equation"
                      {:reason :typed-soac-segmented-reduce-subset :program program})))
    (let [equation (first (soac-dialect/equations program))
          [_ equation-id results operation] equation
          [_ attributes arrays captures lambda] operation
          {:keys [body-results]} (soac-dialect/lambda-parts lambda)
          {:keys [accumulators elements capture-parameters]}
          (soac-dialect/parameter-layout equation)
          segment-axes (:segment-axes attributes)
          reduced-index (:index attributes)
          reduced-extent (:extent attributes)
          dense-coordinate (flat-segment-coordinate segment-axes reduced-index reduced-extent)
          substitutions
          (into (zipmap capture-parameters captures)
                (map (fn [parameter array]
                       [parameter (list 'clojure.core/aget array dense-coordinate)])
                     elements arrays))
          step-results (mapv #(util/subst-syms substitutions %) body-results)
          facts (soac-dialect/facts program)
          physical-results (soac-dialect/physical-results facts equation)
          components
          (mapv (fn [ordinal accumulator neutral component-dtype result]
                  {:id (keyword (str "component-" ordinal))
                   :accumulator accumulator :neutral neutral :dtype component-dtype
                   :result result})
                (range) accumulators (:identities attributes) (:dtypes attributes)
                physical-results)
          operator (reduction/make
                    {:components components
                     :index reduced-index
                     :step (reduction/->ReductionRegion [] step-results {})
                     :algebra {:components (:algebra attributes)}
                     :attributes {:source :typed-soac :equation equation-id
                                  :segmented true}})
          values (:values facts)
          stable-captures (set (get-in attributes [:attributes :stable-array-captures]))
          inputs (set/union (set arrays) stable-captures)
          axis-indices (set (concat (map first segment-axes) [reduced-index]))
          bound-symbols (reduce set/union #{}
                                (map util/free-syms
                                     (conj (mapv second segment-axes) reduced-extent)))
          body-symbols (reduce set/union #{} (map util/free-syms step-results))
          result-transform (:result-transform attributes)
          transform-scalars (set (map :value (:scalars result-transform)))
          scalars (set/difference (set/union bound-symbols body-symbols)
                                  inputs (set accumulators) axis-indices)
          scalars (set/union scalars transform-scalars)
          space (segop/make-seg-space-nd
                 (conj (mapv (fn [[index extent]] {:name index :bound extent}) segment-axes)
                       {:name reduced-index :bound reduced-extent}))
          output-dtype (or (first (:dtypes attributes)) dtype :double)
          contraction? (= :raster.par/contract
                          (get-in attributes [:attributes :source-operation]))
          planned-grid (when contraction?
                         (phase-grid :reduce device-id reduced-extent output-dtype))
          contraction-schedule
          (when contraction?
            (let [workgroup-size (:block-size planned-grid)
                  candidates (filterv #(<= % workgroup-size) [32 64 128 256 512 1024])]
              (reduction/schedule
               {:strategy :hardware-contraction-candidates
                :workgroup-size workgroup-size
                :stages [:segment-space :reduction :target-lowering]
                :tuning-space {:families [:matrix :register-tiled :portable]
                               :workgroup-size candidates}
                :numerical-mode (select-keys (first (get-in operator [:algebra :components]))
                                             [:order :reassociation :overflow])
                :attributes {:source-operation :raster.par/contract
                             :device device-id
                             :selection :target-lowering}})))
          _ (doseq [input inputs]
              (when-not (= :tensor (:kind (get values input)))
                (throw (ex-info "segmented reduction tensor input lacks an AbstractValue"
                                {:reason :typed-soac-segmented-reduce-input
                                 :equation equation-id :input input
                                 :value (get values input)}))))]
      [(segop/->SegRed equation-id space
                       (segop/->SegLevel :thread :virtual)
                       operator nil inputs (set physical-results) scalars planned-grid
                       (if contraction? :contraction :segmented)
                       contraction-schedule output-dtype)])))

(defn typed-product-reduce-program?
  "Whether a validated one-equation TypedSOAC program is a product reduction."
  [program]
  (and (soac-dialect/program-form? program)
       (= 1 (count (soac-dialect/equations program)))
       (= 'product-reduce
          (soac-dialect/operation-kind (first (soac-dialect/equations program))))))

(defn lower-typed-product-reduce
  "Mechanically project a typed two-region product reduction into the existing SegRed schedule."
  [program device-id & {:keys [dtype] :or {dtype :double}}]
  (let [program (soac-dialect/validate! program)]
    (when-not (typed-product-reduce-program? program)
      (throw (ex-info "typed product reduction lowering requires one product-reduce equation"
                      {:reason :typed-soac-product-reduce-subset :program program})))
    (let [equation (first (soac-dialect/equations program))
          [_ equation-id results] equation
          {:keys [attributes arrays captures element-lambda combine-lambda]}
          (soac-dialect/operation-parts equation)
          element (soac-dialect/lambda-parts element-lambda)
          combine (soac-dialect/lambda-parts combine-lambda)
          coordinate (flat-segment-coordinate (:segment-axes attributes)
                                              (:index attributes) (:extent attributes))
          array-count (count arrays)
          element-parameters (:parameters element)
          array-parameters (subvec element-parameters 0 array-count)
          capture-parameters (subvec element-parameters array-count)
          substitutions
          (into (zipmap capture-parameters captures)
                (map (fn [parameter array]
                       [parameter (list 'clojure.core/aget array coordinate)])
                     array-parameters arrays))
          element-bindings
          (vec (mapcat (fn [{:keys [id init]}]
                         [id (util/subst-syms substitutions init)])
                       (:locals element)))
          element-results (mapv #(util/subst-syms substitutions %)
                                (:body-results element))
          combine-bindings
          (vec (mapcat (fn [{:keys [id init]}] [id init]) (:locals combine)))
          facts (soac-dialect/facts program)
          physical-results (soac-dialect/physical-results facts equation)
          result-by-component (zipmap (:result-components attributes) physical-results)
          components
          (mapv (fn [ordinal component-id accumulator neutral component-dtype]
                  {:id component-id :accumulator accumulator :neutral neutral
                   :dtype component-dtype :result (get result-by-component ordinal)})
                (range) (:component-ids attributes) (:accumulators attributes)
                (:identities attributes) (:dtypes attributes))
          operator
          (reduction/make
           {:components components :index (:index attributes)
            :element-bindings element-bindings :element-results element-results
            :combine-parameters (mapv vec (partition 2 (:parameters combine)))
            :combine-bindings combine-bindings :combine-results (:body-results combine)
            :algebra (:algebra attributes)
            :attributes {:source :typed-soac :equation equation-id :segmented true}})
          stable (set (get-in attributes [:attributes :stable-array-captures]))
          inputs (into (set arrays) stable)
          scalars (set/difference (set captures) stable)
          output-dtype (or (first (:dtypes attributes)) dtype :double)
          description {:id equation-id
                       :sym (first results)
                       :reduction operator
                       :segment-axes (:segment-axes attributes)
                       :bound (:extent attributes)
                       :idx (:index attributes)
                       :inputs inputs
                       :outputs (set (keep :result components))
                       :scalars scalars
                       :elem-type output-dtype}]
      (mapv #(assoc % :algorithm-dialect :typed-soac
                    :algorithm-equation equation-id)
            (lower-reduce-description description device-id :dtype output-dtype)))))

(defn typed-scan-program?
  "Whether a validated one-equation TypedSOAC program is a certified scan."
  [program]
  (and (soac-dialect/program-form? program)
       (= 1 (count (soac-dialect/equations program)))
       (= 'scan (soac-dialect/operation-kind (first (soac-dialect/equations program))))))

(defn- typed-scan-description
  [program dtype]
  (let [equation (first (soac-dialect/equations program))
        [_ equation-id results operation] equation
        [_ attributes arrays captures lambda] operation
        {:keys [body-results]} (soac-dialect/lambda-parts lambda)
        {:keys [accumulators elements capture-parameters]}
        (soac-dialect/parameter-layout equation)
        facts (soac-dialect/facts program)
        storage (soac-dialect/result-storage facts equation-id)
        destination (:destination (first storage))
        _ (when-not (and (= 1 (count results)) (= 1 (count accumulators))
                         (= 1 (count body-results)) (= 1 (count storage)) destination)
            (throw (ex-info "typed scan requires one result, accumulator and destination"
                            {:reason :typed-soac-scan-subset :equation equation-id
                             :results results :accumulators accumulators
                             :result-storage storage})))
        index (:index attributes)
        substitutions
        (into (zipmap capture-parameters captures)
              (map (fn [parameter array]
                     [parameter (list 'clojure.core/aget array index)])
                   elements arrays))
        stable (set (get-in attributes [:attributes :stable-array-captures]))
        scalar-captures (set (remove stable captures))
        public-scalar-ids (cond-> scalar-captures
                            (or (symbol? (:extent attributes))
                                (keyword? (:extent attributes)))
                            (conj (:extent attributes)))
        ordered-public-scalars
        (vec (concat (filter public-scalar-ids (:inputs facts))
                     (sort-by pr-str (remove (set (:inputs facts)) public-scalar-ids))))
        scan-dtype (or (first (:dtypes attributes)) dtype :double)
        substitutions (assoc substitutions index index)
        algebra (update (first (:algebra attributes)) :element
                        #(util/subst-syms substitutions %))]
    {:id equation-id
     :sym destination
     :out destination
     :acc (first accumulators)
     :init (first (:identities attributes))
     :lambda (util/subst-syms substitutions (first body-results))
     :algebra algebra
     :mode (:mode attributes)
     :bound (:extent attributes)
     :idx index
     :inputs (into (set arrays) (disj stable destination))
     :outputs #{destination}
     :scalars scalar-captures
     :public-scalars
     (mapv (fn [id]
             (let [value (get (:values facts) id)]
               (when-not (and value (empty? (:shape value)))
                 (throw (ex-info "typed scan public scalar lacks a scalar AbstractValue"
                                 {:reason :typed-soac-scan-scalar :value id
                                  :abstract-value value})))
               (kernel-graph/scalar id (:dtype value))))
           ordered-public-scalars)
     :elem-type scan-dtype}))

(defn lower-typed-scan
  "Lower one certified TypedSOAC scan without reconstructing the legacy SOAC record IR.

   Returns both ordered SegOps and their KernelGraph because temporary storage and dependencies are
   properties of the selected schedule, not of the functional scan equation."
  [program device-id & {:keys [dtype array-types]
                        :or {dtype :double array-types {}}}]
  (let [program (soac-dialect/validate! program)]
    (when-not (typed-scan-program? program)
      (throw (ex-info "typed SegScan lowering requires one scan equation"
                      {:reason :typed-soac-scan-subset :program program})))
    (let [description (typed-scan-description program dtype)
          operations (mapv #(assoc % :algorithm-dialect :typed-soac
                                   :algorithm-equation (:id description))
                           (lower-scan-description description device-id
                                                   :dtype (:elem-type description)))]
      {:operations operations
       :kernel-graph (scan-kernel-graph-description description operations
                                                    {:array-types array-types})})))

;; ================================================================
;; Lowering helpers
;; ================================================================

(defn- soac-outputs*
  [soac]
  (or (soac/soac-outputs soac) (:outputs soac)))

(defn- phase-grid
  [segop-type device-id bound-expr dtype]
  (segop/compute-launch-params segop-type device-id bound-expr :dtype dtype))

(defn- scan-grid
  "A scan needs one workgroup per contiguous block. Unlike map/reduce it cannot cap the grid and
   recover coverage with a grid-stride loop because block prefixes are ordered dataflow."
  [device-id bound-expr dtype]
  (let [planned (phase-grid :scan device-id bound-expr dtype)
        block-size (:block-size planned)]
    (segop/->KernelGrid
     (kernel-launch/ceil-div bound-expr block-size)
     block-size
     (:shared-mem-bytes planned))))

(defn- single-block-grid
  [grid]
  (segop/->KernelGrid 1 (:block-size grid) (:shared-mem-bytes grid)))

(defn- floor-power-of-two
  [n]
  (loop [power 1]
    (if (<= (* 2 power) n)
      (recur (* 2 power))
      power)))

(defn- product-grid
  "Constrain a product reduction's workgroup by both the target's thread limit and its SLM
   budget. Each component has an independent local array, so charging only the primary dtype
   would make mixed products legal on paper while overcommitting local memory at emission."
  [device-id planned reduction]
  (let [descriptor (hardware/descriptor-for device-id)
        bytes-per-lane (reduce + (map (comp dtype/bytes-of :dtype) (:components reduction)))
        slm-budget (long (or (get-in descriptor [:cache :slm])
                             (:shared-memory-per-block descriptor)
                             65536))
        max-by-slm (max 1 (quot slm-budget bytes-per-lane))
        max-workgroup (long (:max-workgroup-size descriptor 1024))
        workgroup-size (floor-power-of-two
                        (max 1 (min (long (:block-size planned))
                                    max-workgroup
                                    max-by-slm)))]
    {:grid (segop/->KernelGrid (:num-blocks planned)
                               workgroup-size
                               (* workgroup-size bytes-per-lane))
     :slm-budget slm-budget
     :bytes-per-lane bytes-per-lane}))

(defn- reduction-info
  [soac]
  (:reduction soac))

(defn- scan-op-info
  [soac]
  {:acc (:acc soac) :init (:init soac)
   :lambda (:lambda soac) :out (:out soac)})

(defn- legacy-scan-description
  [node]
  (let [raw (scan-op-info node)]
    {:id (:id node)
     :sym (:sym node)
     :out (:out raw)
     :acc (:acc raw)
     :init (:init raw)
     :lambda (:lambda raw)
     :bound (:bound node)
     :idx (:idx node)
     :inputs (or (:inputs node) #{})
     :outputs (or (soac-outputs* node) #{})
     :scalars (or (:scalars node) #{})
     :elem-type (:elem-type node)
     :map-lambda nil}))

(defn- legacy-map-description
  [node]
  (let [outputs (or (soac-outputs* node) #{})]
    {:id (:id node)
     :bound (:bound node)
     :idx (soac/soac-idx node)
     :lambda (:lambda node)
     :scalar-region (:scalar-region node)
     :inputs (or (:inputs node) #{})
     :outputs outputs
     :scalars (or (:scalars node) #{})
     :elem-type (:elem-type node)
     ;; The compatibility node's :sym is the host binding result, not necessarily the physical
     ;; store destination. Preserve the one declared output when it is unambiguous; consumers must
     ;; never rediscover it by reparsing the source form.
     :out-sym (if (= 1 (count outputs)) (first outputs) (:sym node))
     :cast-fn (:cast-fn node)}))

(defn- legacy-reduce-description
  [node]
  {:id (:id node)
   :sym (:sym node)
   :reduction (reduction-info node)
   :segment-axes (or (:segment-axes node) [])
   :bound (:bound node)
   :idx (soac/soac-idx node)
   :inputs (or (:inputs node) #{})
   :outputs (or (soac-outputs* node) #{(:sym node)})
   :scalars (or (:scalars node) #{})
   :elem-type (:elem-type node)})

;; ================================================================
;; Map lowering
;; ================================================================

(defn lower-map
  "Lower a compatibility SoacMap to one SegMap with grid-stride virtualization."
  [soac device-id & {:keys [dtype] :or {dtype :double}}]
  (lower-map-description (legacy-map-description soac) device-id :dtype dtype))

(defn- lower-map-description
  [description device-id & {:keys [dtype] :or {dtype :double}}]
  (let [dtype (or (:elem-type description) dtype)
        bound (:bound description)
        idx (:idx description)
        space (segop/make-seg-space idx bound)
        level (segop/->SegLevel :thread :virtual)
        grid (phase-grid :map device-id bound dtype)
        out-sym (:out-sym description)
        cast-fn (:cast-fn description)]
    [(segop/->SegMap (:id description) space level
                     (:lambda description)
                     (:scalar-region description)
                     (:inputs description) (:outputs description)
                     (:scalars description) grid
                     dtype out-sym cast-fn)]))

;; ================================================================
;; Reduce lowering — single or two-phase
;; ================================================================

(defn lower-reduce
  "Lower a compatibility SoacReduce to SegRed SegOps.

  Decision criteria:
    - n ≤ block-size → single-phase (one block does everything)
    - n > block-size → two-phase:
        Phase 1 (:block-local): each block reduces its chunk via
                shared-memory tree reduction + warp shuffle.
                Output: per-block partial results array.
        Phase 2 (:cross-block): single block reduces partials.

  Returns a vector of SegRed records (1 or 2 elements)."
  [soac device-id & {:keys [dtype] :or {dtype :double}}]
  (lower-reduce-description (legacy-reduce-description soac) device-id :dtype dtype))

(defn- lower-reduce-description
  [description device-id & {:keys [dtype] :or {dtype :double}}]
  (let [reduction (:reduction description)
        product? (some? (:combine reduction))
        dtype (or (first (map :dtype (:components reduction))) (:elem-type description) dtype)
        bound (:bound description)
        idx (:idx description)
        map-lambda nil
        space (segop/make-seg-space-nd
               (conj (mapv (fn [[name axis-bound]] {:name name :bound axis-bound})
                           (:segment-axes description))
                     {:name idx :bound bound}))
        planned-grid (phase-grid :reduce device-id bound dtype)
        product-grid-info (when product? (product-grid device-id planned-grid reduction))
        grid-1 (or (:grid product-grid-info) planned-grid)
        product-schedule
        (when product?
          (let [workgroup-size (:block-size grid-1)
                candidates (filterv #(<= % workgroup-size) [32 64 128 256 512 1024])]
            (reduction/schedule
             {:strategy :segmented-workgroup-tree
              :workgroup-size workgroup-size
              :stages [:lane-fold :workgroup-tree :segment-store]
              :tuning-space {:workgroup-size candidates
                             :elements-per-lane [:runtime-stride]}
              :numerical-mode (select-keys (:algebra reduction)
                                           [:order :reassociation :overflow])
              :attributes {:scratch :workgroup-local
                           :component-dtypes (reduction/dtypes reduction)
                           :scratch-bytes-per-lane (:bytes-per-lane product-grid-info)
                           :slm-budget (:slm-budget product-grid-info)}})))
        execution (execution-plan/reduce-execution bound grid-1)]
    (if product?
      [(segop/->SegRed (:id description) space (segop/->SegLevel :block :virtual)
                       reduction map-lambda (:inputs description)
                       (set (filter symbol? (:outputs description))) (:scalars description)
                       grid-1 :product product-schedule dtype)]
      (case (:strategy execution)
        :single
        (let [grid (single-block-grid grid-1)
              reduction (assoc-in reduction [:attributes :physical-phase] :single)]
          [(segop/->SegRed (:id description)
                           space
                           (segop/->SegLevel :block :none)
                           reduction
                           map-lambda
                           (:inputs description)
                           (:outputs description)
                           (:scalars description)
                           grid
                           :single
                           (segred-body/scalar-workgroup-tree-schedule
                            reduction grid :single)
                           dtype)])

        :two-phase
        (let [level-1 (segop/->SegLevel :block :virtual)
              partials-sym (gensym "partials_")
              phase-1-reduction (-> reduction
                                    (update :attributes dissoc :result-region)
                                    (assoc-in [:attributes :physical-phase] :block-local))
              result-region (get-in reduction [:attributes :result-region])
              fold-symbols (reduce set/union #{}
                                   (map util/free-syms
                                        (cond-> (vec (get-in reduction [:step :results]))
                                          map-lambda (conj map-lambda))))
              fold-scalars (if result-region
                             (set/intersection (set (:scalars description)) fold-symbols)
                             (:scalars description))
              phase-1 (segop/->SegRed
                       (:id description) space level-1
                       phase-1-reduction map-lambda
                       (:inputs description)
                       #{partials-sym}
                       fold-scalars
                       grid-1 :block-local
                       (segred-body/scalar-workgroup-tree-schedule
                        phase-1-reduction grid-1 :block-local)
                       dtype)
              phase-2-idx (gensym "j_")
              phase-2-bound (segred-body/launch-group-count
                             (:num-blocks grid-1) bound (:block-size grid-1))
              phase-2-space (segop/make-seg-space phase-2-idx phase-2-bound)
              grid-2 (single-block-grid grid-1)
              level-2 (segop/->SegLevel :block :none)
              {:keys [operator identity accumulator]} (segred-body/scalar-plan phase-1)
              combine-op ({:+ 'clojure.core/+
                           :* 'clojure.core/*
                           :min 'clojure.core/min
                           :max 'clojure.core/max} operator)
              component (first (:components reduction))
              result-scalars (set (drop 1 (:parameters result-region)))
              phase-2-reduction
              (reduction/scalar
               {:accumulator accumulator
                :neutral identity
                :dtype (:dtype component)
                :result (:sym description)
                :index phase-2-idx
                :step-result (list combine-op accumulator
                                   (list 'clojure.core/aget partials-sym phase-2-idx))
                :algebra (:algebra reduction)
                :attributes (assoc (:attributes reduction)
                                   :physical-phase :cross-block)})
              phase-2 (segop/->SegRed
                       [:reduction-phase (:id description) :cross-block]
                       phase-2-space level-2
                       phase-2-reduction nil
                       #{partials-sym} #{(:sym description)}
                       result-scalars grid-2 :cross-block
                       (segred-body/scalar-workgroup-tree-schedule
                        phase-2-reduction grid-2 :cross-block)
                       dtype)]
          [phase-1 phase-2])))))

;; ================================================================
;; Scan lowering — single or three-stage
;; ================================================================

(defn lower-scan
  "Lower a compatibility SoacScan to SegScan SegOps.

  Decision criteria:
    - n ≤ block-size → single-phase intra-block workgroup scan
    - n > block-size → three-stage:
        Stage 1 (:intra-block): scan within each block,
                 last element = block total
        Stage 2 (:block-scan): ordered scan of block totals
        Stage 3 (:carry-in): combine carry-in with each element (SegMap)

  Returns a vector of SegScan/SegMap records."
  [soac device-id & {:keys [dtype] :or {dtype :double}}]
  (lower-scan-description (legacy-scan-description soac) device-id :dtype dtype))

(defn- lower-scan-description
  [description device-id & {:keys [dtype] :or {dtype :double}}]
  (let [dtype (or (:elem-type description) dtype)
        bound (:bound description)
        idx (:idx description)
        mode (or (:mode description) :inclusive)
        raw-scan-op (assoc (select-keys description [:acc :init :lambda :out]) :mode mode)
        map-lambda (:map-lambda description)
        _ (when map-lambda
            (throw (ex-info "fused scan/map needs an explicit scheduled scan epilogue"
                            {:reason :scan-fused-map-unimplemented
                             :scan-op raw-scan-op :map-lambda map-lambda})))
        scan-facts (or (:algebra description) (scan/certify raw-scan-op dtype))
        scan-op (assoc raw-scan-op :algebra scan-facts)
        space (segop/make-seg-space idx bound)
        grid-1 (scan-grid device-id bound dtype)
        execution (execution-plan/scan-execution bound grid-1)]
    (case (:strategy execution)
      :single
      [(segop/->SegScan (:id description)
                        space
                        (segop/->SegLevel :block :none)
                        scan-op
                        map-lambda
                        (:inputs description)
                        (:outputs description)
                        (:scalars description)
                        (single-block-grid grid-1)
                        :single
                        dtype)]

      :three-stage
      (let [level-1 (segop/->SegLevel :block :virtual)
            totals-sym (gensym "block_totals_")
            stage-1 (segop/->SegScan (:id description) space level-1
                                     scan-op map-lambda
                                     (:inputs description)
                                     (conj (:outputs description) totals-sym)
                                     (:scalars description)
                                     grid-1 :intra-block
                                     dtype)
            stage-2-idx (gensym "k_")
            stage-2-space (segop/make-seg-space stage-2-idx (:num-blocks grid-1))
            grid-2 (single-block-grid grid-1)
            level-2 (segop/->SegLevel :block :none)
            stage-2 (segop/->SegScan (+ (:id description) 2000)
                                     stage-2-space level-2
                                     scan-op nil
                                     #{totals-sym} #{totals-sym}
                                     #{} grid-2 :block-scan
                                     dtype)
            carry-idx (gensym "ci_")
            carry-space (segop/make-seg-space carry-idx bound)
            grid-3 (phase-grid :map device-id bound dtype)
            level-3 (segop/->SegLevel :thread :virtual)
            out-sym (or (:out scan-op) (first (:outputs description)))
            block-idx-expr (list 'clojure.core/quot carry-idx (:block-size grid-1))
            combine (:combine scan-facts)
            carry-lambda (list 'if (list '> block-idx-expr 0)
                               (list combine
                                     (list 'aget totals-sym
                                           (list 'clojure.core/- block-idx-expr 1))
                                     (list 'aget out-sym carry-idx))
                               (list 'aget out-sym carry-idx))
            stage-3 (segop/->SegMap (+ (:id description) 3000)
                                    carry-space level-3
                                    carry-lambda
                                    nil
                                    #{out-sym totals-sym}
                                    #{out-sym}
                                    #{} grid-3
                                    dtype out-sym nil)]
        [stage-1 stage-2 stage-3]))))

(defn scan-kernel-graph
  "Turn an already lowered scan into a verified scheduled graph.

   This consumes the exact SegOps returned by `lower-scan`; it must not lower a second time because
   the block-totals buffer identity is generated during decomposition."
  ([soac segops]
   (scan-kernel-graph soac segops {}))
  ([soac segops {:keys [array-types] :or {array-types {}}}]
   (scan-kernel-graph-description (legacy-scan-description soac) segops
                                  {:array-types array-types})))

(defn- scan-kernel-graph-description
  [description segops {:keys [array-types] :or {array-types {}}}]
  (let [external (set/union (or (:inputs description) #{})
                            (or (:outputs description) #{}))
        used (reduce set/union #{}
                     (map #(set/union (or (segop/segop-inputs %) #{})
                                      (or (segop/segop-outputs %) #{}))
                          segops))
        temporary-ids (set/difference used external)
        block-stage (some #(when (= :block-scan (:phase %)) %) segops)
        temporary-elements (when block-stage
                             (:bound (segop/seg-space-reduced-dim (:space block-stage))))
        dtype (or (:elem-type description) (:dtype (first segops)) :double)
        temporaries (into {}
                          (map (fn [id]
                                 [id {:dtype dtype :elements temporary-elements
                                      :memory-space :device}]))
                          temporary-ids)
        output-elements (if (= :exclusive (:mode description))
                          (kernel-launch/sum (:bound description) 1)
                          (:bound description))
        output-ids (set (:outputs description))
        buffer-specs (into {}
                           (map (fn [id]
                                  [id {:dtype (or (get array-types id)
                                                  (get array-types (symbol (name id)))
                                                  dtype)
                                       :elements (if (contains? output-ids id)
                                                   output-elements
                                                   (:bound description))}]))
                           external)]
    (kernel-graph/from-segops
     segops
     {:inputs (or (:inputs description) #{})
      :outputs (or (:outputs description) #{})
      :temporaries temporaries
      :scalars (:public-scalars description)
      :buffer-specs buffer-specs
      :dtype dtype
      :effects {:memory-order :dependency-ordered}
      :provenance {:dialect :segop :algorithm :scan :soac-id (:id description)}
      :attributes {:strategy (if (= 1 (count segops)) :single :three-stage)
                   :scan-mode (or (:mode description) :inclusive)
                   :scan-algebra (get-in (first segops) [:scan-op :algebra])}})))

(defn scan-soac?
  "True when a compatibility SOAC node selects scan decomposition."
  [node]
  (soac/soac-scan? node))

;; ================================================================
;; Unified lowering dispatch
;; ================================================================

(defn lower-soac
  "Lower one compatibility SOAC node to one or more SegOps.
  Returns a vector of SegOp records.

  Dispatches on SOAC type:
    SoacMap → [SegMap]
    SoacReduce → [SegRed SegRed] (two-phase)
    SoacScan → [SegScan SegScan SegMap] (three-stage)"
  [soac device-id & {:keys [dtype] :or {dtype :double}}]
  (let [dtype (or (:elem-type soac) dtype)]
    (cond
      (nil? soac)
      (throw (ex-info "Cannot lower nil: the preceding conversion produced no SOAC node"
                      {:reason :no-soac-node :target-dialect :segop}))

      ;; a contraction: record the facts for this target; the backend routes from them
      (soac/contract? soac)
      [(segop/->SegContract (:id soac) (:facts soac) dtype device-id)]

      (soac/soac-map? soac)
      (lower-map soac device-id :dtype dtype)

      (soac/soac-reduce? soac)
      (lower-reduce soac device-id :dtype dtype)

      (soac/soac-scan? soac)
      (lower-scan soac device-id :dtype dtype)

      :else
      (throw (ex-info "Unsupported SOAC type for lowering" {:soac soac})))))
