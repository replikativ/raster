(ns raster.compiler.passes.parallel.product-consumer-body
  "Candidate cooperative KernelBody for a product reduction followed by an ordered consumer.

   The product tree is parallel only along its declared reduced axis.  Its local suffix selects
   independent tree segments inside one workgroup; the ordered consumer fold remains sequential.
   Admission and intermediate-address correspondence come exclusively from
   product-consumer-region, so this lowering contains no quantization or operation-family rules."
  (:require [clojure.set :as set]
            [clojure.walk :as walk]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.extent-expression :as extent-expression]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.index-expression :as index-expression]
            [raster.compiler.passes.parallel.product-consumer-region :as region]
            [raster.compiler.passes.parallel.product-reduction-regions :as product-regions]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar]))

(defn- decline! [rule message data]
  (throw (ex-info message (assoc data :reason :product-consumer-body-declined
                                 :missing-rule rule :fallback :separate-kernels))))

(defn declined? [exception]
  (= :product-consumer-body-declined (:reason (ex-data exception))))

(defn- product-expression [values]
  (case (count values)
    0 1
    1 (first values)
    (with-meta (apply list 'clojure.core/* values)
      {:raster.type/tag 'long :tag 'long})))

(defn- graph-options [plan]
  (let [graph (get-in plan [:source :graph])
        scheduled-body (get-in plan [:source :body])
        buffers (into {} (map (juxt :id identity))
                      (concat (:inputs graph) (:outputs graph) (:temporaries graph)))
        array-types (into {} (map (juxt :id :dtype)) (vals buffers))
        scalar-types (into {} (map (juxt :id :dtype)) (:scalars graph))
        producer-requirements
        (product-regions/dense-read-requirements
         (:producer plan) {:array-types array-types :scalar-types scalar-types} decline!)
        requirements
        (merge-with
         (fn [left right]
           (if (extent-expression/equivalent? left right)
             left
             (decline! :shared-read-layout
                       "producer and consumer require incompatible dense capacities"
                       {:producer left :consumer right})))
         producer-requirements (:consumer-read-requirements plan))
        requirement-shape
        (fn [expression]
          (if (launch/expression? expression)
            expression
            (let [scope (set (util/free-syms expression))
                  lowered (index-expression/lower-typed expression scope scalar-types :long decline!)]
              (index-expression/to-launch-expression lowered decline!))))]
    {:buffers buffers
     :array-types array-types
     :array-shapes
     (into {}
           (map (fn [[id buffer]]
                  [id [(if-let [requirement (get requirements id)]
                         (requirement-shape requirement)
                         (:elements buffer))]]))
           buffers)
     :scalar-types scalar-types
     :values (:values scheduled-body)}))

(defn- ordered-fold-form? [value]
  (or (dialect/scalar-fold-form? value) (dialect/product-fold-form? value)))

(defn- ordered-fold-parts [value]
  (if (dialect/product-fold-form? value)
    (dialect/product-fold-parts value)
    (dialect/scalar-fold-parts value)))

(defn- fold-of [consumer]
  (->> (tree-seq coll? seq (:scalar-region consumer))
       (filter ordered-fold-form?) distinct first))

(defn- contains-fold? [value]
  (boolean (some ordered-fold-form? (tree-seq coll? seq value))))

(defn- scalar-operation-name [operation]
  (some-> operation class .getSimpleName))

(defn- slice-pure-scalar-operations
  "Retain the dependency slice for selected SSA results from a straight-line scalar region.

   Loads and computes are pure here: the source product element cannot contain stores. Unknown
   structured operations conservatively keep the complete region rather than guessing effects."
  [operations selected-results]
  (if-not (every? #(contains? #{"ScalarCompute" "ScalarLoad"}
                              (scalar-operation-name %))
                  operations)
    (vec operations)
    (let [{:keys [kept]}
          (reduce
           (fn [{:keys [needed kept] :as state} operation]
             (let [result-id (some-> operation :result :id)]
               (if (contains? needed result-id)
                 {:needed (into (disj needed result-id)
                                (disj (util/free-syms operation) result-id))
                  :kept (conj kept operation)}
                 state)))
           {:needed (set selected-results) :kept []}
           (reverse operations))]
      (vec (reverse kept)))))

(defn- lower-locals
  [lowerer locals substitutions environment]
  (reduce
   (fn [{:keys [operations substitutions environment]} {:keys [id dtype init]}]
     (let [init (util/subst-syms substitutions init)
           lowered ((:lower lowerer) init (dtype/canon dtype) environment)]
       {:operations (into operations (:operations lowered))
        :substitutions (assoc substitutions id (:result lowered))
        :environment (assoc environment (:result lowered) (:type lowered))}))
   {:operations [] :substitutions substitutions :environment environment}
   locals))

(defn- replace-intermediate-loads [plan reduced-width value]
  (let [replacement
        (into {}
              (map (fn [{:keys [load local-offset]}]
                     [load (descriptor/rewrite-aget-index load (* local-offset reduced-width))]))
              (:intermediate-loads plan))]
    (walk/postwalk #(get replacement % %) value)))

(defn- replace-intermediate-values [plan values value]
  (let [replacement
        (into {}
              (map (fn [{:keys [load local-offset component]}]
                     [load (or (get values [local-offset component])
                               (decline! :intermediate-local-offset
                                         "proved intermediate component/offset exceeds subgroup SSA values"
                                         {:local-offset local-offset :component component
                                          :values values}))]))
              (:intermediate-loads plan))]
    (walk/postwalk #(get replacement % %) value)))

(defn- local-offset-substitutions [axes offset]
  (loop [remaining (reverse (vec axes))
         value offset
         substitutions {}]
    (if-let [{:keys [name bound]} (first remaining)]
      (recur (next remaining) (quot value bound)
             (assoc substitutions name (mod value bound)))
      substitutions)))

(defn- replace-fold-components [fold carries value]
  (walk/postwalk
   (fn [form]
     (cond
       (and (dialect/scalar-fold-form? fold) (= fold form))
       (or (first carries)
           (decline! :consumer-component
                     "scalar ordered fold requires one lowered carry"
                     {:carries carries}))

       (and (dialect/product-component-form? form) (= fold (second form)))
       (let [ordinal (nth form 2)]
         (or (nth carries ordinal nil)
             (decline! :consumer-component
                       "consumer projection references a missing ordered-fold component"
                       {:ordinal ordinal :carries carries})))

       :else form))
   value))

(defn- barrier []
  (body/->WorkgroupBarrier :workgroup #{:workgroup} :acquire-release
                           (body/full-participation)))

(defn- positive-domain [bounds]
  (let [dynamic (vec (remove integer? bounds))
        predicates (mapv #(keyword (str "product-consumer-positive-" %)) (range (count dynamic)))
        comparisons
        (mapv (fn [id bound]
                (body/->ScalarCompute (body/value id :predicate)
                                     (body/scalar-expression :lt :predicate
                                                             [(body/literal 0 :long) bound])))
              predicates dynamic)]
    {:operations comparisons :predicates predicates}))

(defn- guard-positive-domain [predicates operations]
  (reduce (fn [nested predicate]
            [(body/->IfRegion predicate (conj (vec nested) (body/->Yield []))
                              [(body/->Yield [])] [])])
          (vec operations) (reverse predicates)))

(defn lower
  "Build the candidate body from a replayable product-consumer analysis plan."
  [plan]
  (when-not (= :product-ordered-consumer (:kind plan))
    (decline! :plan "product-consumer body requires an admitted region plan" {:plan plan}))
  (let [{:keys [producer consumer intermediates workgroup-size]} plan
        {:keys [prefix ordered local reduced]} (:axes plan)
        physical-schedule (or (:physical-schedule plan)
                              {:strategy :product-tree-ordered-consumer
                               :workgroup-size workgroup-size})
        strategy (:strategy physical-schedule)
        subgroup? (= :subgroup-product-ordered-consumer strategy)
        physical-workgroup-size (:workgroup-size physical-schedule)
        {:keys [buffers array-types array-shapes scalar-types]} (graph-options plan)
        components (get-in producer [:reduction :components])
        component-types (mapv (comp dtype/canon :dtype) components)
        _ (when-not (= (count intermediates) (count components))
            (decline! :producer-components
                      "private product storage must correspond to every component"
                      {:components components :intermediates intermediates}))
        reduced-width (:bound reduced)
        local-volume (reduce * 1 (map :bound local))
        _ (when-not (= workgroup-size (* local-volume reduced-width))
            (decline! :worker-decomposition
                      "cooperative width must equal local suffix volume times reduction width"
                      {:workgroup-size workgroup-size :local-volume local-volume
                       :reduced-width reduced-width}))
        external-inputs (vec (sort-by name
                                      (set/difference
                                       (set/union (:inputs producer) (:inputs consumer))
                                       (set intermediates))))
        outputs (vec (sort-by name (:outputs consumer)))
        _ (when-not (and (= 1 (count outputs)) (= (first outputs) (:out-sym consumer)))
            (decline! :consumer-output
                      "the initial fused body requires one ordinary map result"
                      {:outputs outputs :out-sym (:out-sym consumer)}))
        output (first outputs)
        scalars (vec (sort-by name (set/union (:scalars producer) (:scalars consumer)
                                             (set (keys scalar-types)))))
        parameter
        (fn [id kind role]
          (body/->KernelParameter id kind (array-types id) (array-shapes id) :global
                                  (layout/row-major (array-shapes id) (array-types id)) role))
        parameters (vec (concat (map #(parameter % :input :operand) external-inputs)
                                (map #(parameter % :output :result) outputs)
                                (map #(body/->KernelParameter % :scalar (scalar-types %) [] nil nil
                                                              :parameter)
                                     scalars)))
        launch-prefix-count (apply launch/product (mapv :bound prefix))
        active-domain (positive-domain (mapv :bound prefix))
        index-scope (set/union (set scalars) (set (map :name prefix))
                               #{(:name ordered) (:name reduced)})
        index-types (merge scalar-types
                           (into {} (map (fn [{:keys [name]}] [name :long]))
                                 (concat prefix [ordered reduced])))
        lower-index (fn [expression locals]
                      (index-expression/lower-typed expression
                                                    (set/union index-scope locals)
                                                    index-types :long decline!))
        producer-options {:array-types array-types :scalar-types scalar-types}
        workgroup-regions (when-not subgroup?
                            (product-regions/lower producer producer-options decline!))
        element (:element workgroup-regions)
        combine (:combine workgroup-regions)
        subgroup-regions
        (when subgroup?
          (mapv (fn [offset]
                  (product-regions/lower
                   producer
                   (assoc producer-options
                          :axis-substitutions (local-offset-substitutions local offset)
                          :id-prefix (str "product-subgroup-" offset))
                   decline!))
                (range local-volume)))
        group-index 'product-consumer-group
        lane-index 'product-consumer-lane
        lane-long (body/index-cast lane-index :long :exact)
        reduced-index (body/expression :mod lane-long (lower-index reduced-width #{}))
        local-linear (body/expression :floor-div lane-long (lower-index reduced-width #{}))
        prefix-computes
        (mapv (fn [position {:keys [name bound]}]
                (let [following (subvec (vec prefix) (inc position))
                      divisor (product-expression (mapv :bound following))
                      quotient (if (= 1 divisor) (body/index-cast group-index :long :exact)
                                   (body/expression :floor-div
                                                    (body/index-cast group-index :long :exact)
                                                    (lower-index divisor #{})))]
                  (body/->IndexCompute name
                                       (body/expression :mod quotient
                                                        (lower-index bound #{})))))
              (range) prefix)
        local-computes
        (mapv (fn [position {:keys [name bound]}]
                (let [following (subvec (vec local) (inc position))
                      divisor (reduce * 1 (map :bound following))
                      quotient (if (= 1 divisor) local-linear
                                   (body/expression :floor-div local-linear
                                                    (lower-index divisor #{})))]
                  (body/->IndexCompute name (body/expression :mod quotient
                                                             (lower-index bound #{})))))
              (range) local)
        reduced-compute (body/->IndexCompute (:name reduced) reduced-index)
        producer-values (some-> element :results vec)
        tree-strides (vec (take-while pos? (iterate #(quot % 2) (quot reduced-width 2))))
        tree-operations
        (when-not subgroup?
          (mapcat
           (fn [stride]
             (let [left (mapv #(symbol (str "product-consumer-left-" stride "-" %))
                              (range (count components)))
                   right (mapv #(symbol (str "product-consumer-right-" stride "-" %))
                               (range (count components)))
                   merged (combine left right)
                   predicate (keyword (str "product-consumer-tree-" stride))]
               [(body/->ScalarCompute (body/value predicate :predicate)
                                      (body/scalar-expression :lt :predicate
                                                              [(:name reduced)
                                                               (body/literal stride :long)]))
                (body/->IfRegion
                 predicate
                 (vec (concat
                       (mapcat (fn [left-id right-id component-type intermediate]
                                 [(body/->ScalarLoad (body/value left-id component-type)
                                                    intermediate [lane-index] nil nil :cached)
                                  (body/->ScalarLoad (body/value right-id component-type)
                                                    intermediate
                                                    [(body/expression :add lane-index stride)]
                                                    nil nil :cached)])
                               left right component-types intermediates)
                       (:operations merged)
                       (map (fn [intermediate result]
                              (body/->ScalarStore intermediate [lane-index] result nil))
                            intermediates (:results merged))
                       [(body/->Yield [])]))
                 [(body/->Yield [])] [])
                (barrier)]))
           tree-strides))
        reduced-active-id :product-consumer-reduced-active
        reduced-active-operation
        (when subgroup?
          (body/->ScalarCompute
           (body/value reduced-active-id :predicate)
           (body/scalar-expression :lt :predicate
                                   [lane-index (body/literal reduced-width :int)])))
        subgroup-component-pairs
        (when subgroup?
          (->> (:intermediate-loads plan)
               (map (juxt :local-offset :component))
               distinct
               sort
               vec))
        subgroup-values
        (when subgroup?
          (into {}
                (map (fn [[offset component :as pair]]
                       [pair (symbol (str "product-consumer-subgroup-partial-"
                                          offset "-" component))]))
                subgroup-component-pairs))
        subgroup-operations
        (when subgroup?
          (let [contracts (:subgroup-collectives plan)
                width (:subgroup-size physical-schedule)]
            (vec
             (mapcat
              (fn [offset {:keys [element]}]
                (let [component-indices
                      (->> subgroup-component-pairs
                           (keep (fn [[used-offset component]]
                                   (when (= offset used-offset) component)))
                           vec)]
                  (if (empty? component-indices)
                    []
                    (let [lane-values
                          (mapv #(symbol (str "product-consumer-subgroup-element-"
                                              offset "-" %))
                                component-indices)
                          result-ids (mapv #(get subgroup-values [offset %]) component-indices)
                          selected-results (mapv #(nth (:results element) %) component-indices)
                          selected-types (mapv #(nth component-types %) component-indices)
                          selected-contracts (mapv #(nth contracts %) component-indices)]
                      (concat
                       [(body/->IfRegion
                         reduced-active-id
                         (conj (slice-pure-scalar-operations (:operations element)
                                                            selected-results)
                               (body/->Yield selected-results))
                         [(body/->Yield
                           (mapv (fn [{:keys [neutral]} component-type]
                                   (body/literal neutral component-type))
                                 selected-contracts selected-types))]
                         (mapv body/value lane-values selected-types))]
                       (mapv (fn [result-id lane-value component-type
                                  {:keys [operator arithmetic association]}]
                               (body/->Collective
                                (body/value result-id component-type) :reduce :subgroup width lane-value
                                operator nil (body/full-participation) association arithmetic))
                             result-ids lane-values selected-types selected-contracts))))))
              (range local-volume) subgroup-regions))))
        fold (fold-of consumer)
        {fold-attributes :attributes fold-lambda :lambda} (ordered-fold-parts fold)
        {fold-parameters :parameters fold-locals :locals fold-results :body-results}
        (dialect/lambda-parts fold-lambda)
        carry-types (mapv dtype/canon
                          (or (:dtypes fold-attributes) [(:dtype fold-attributes)]))
        carries (mapv #(symbol (str "product-consumer-carry-" %)) (range (count carry-types)))
        updated-carries (mapv #(symbol (str "product-consumer-updated-" %))
                              (range (count carry-types)))
        loop-results (mapv #(symbol (str "product-consumer-result-" %)) (range (count carry-types)))
        raw-consumer-prefix-locals
        (vec (take-while #(not (contains-fold? (:init %)))
                         (get-in consumer [:scalar-region :locals])))
        consumer-prefix-count (count raw-consumer-prefix-locals)
        prefix-aliases (into {}
                             (keep identity
                                   (mapv (fn [consumer-digit producer-axis]
                                           (when (symbol? consumer-digit)
                                             [consumer-digit (:name producer-axis)]))
                                         (get-in plan [:axes :prefix-binding]) prefix)))
        consumer-prefix-locals (filterv #(not (contains? prefix-aliases (:id %)))
                                        raw-consumer-prefix-locals)
        consumer-arrays (into (set (:inputs consumer)) intermediates)
        consumer-index-types (merge scalar-types
                                    (into {} (map (fn [{:keys [name]}] [name :long])) prefix)
                                    (into {} (map (fn [{:keys [name]}] [name :long]))
                                          (get-in consumer [:space :dims]))
                                    {(:name ordered) :long})
        consumer-lower-index
        (fn [expression locals]
          (index-expression/lower-typed expression
                                        (set/union (set (keys consumer-index-types)) locals)
                                        consumer-index-types :long decline!))
        consumer-lowerer
        (scalar/make-lowerer {:arrays consumer-arrays :array-types array-types
                              :scalar-types consumer-index-types
                              :index-scope (set (keys consumer-index-types))
                              :lower-index consumer-lower-index
                              :source-region (:scalar-region consumer)
                              :id-prefix "product-consumer" :decline! decline!})
        base-environment (merge consumer-index-types
                                (into {} (map (fn [{:keys [name]}] [name :long])) prefix))
        prefix-state
        (reduce
         (fn [{:keys [operations substitutions environment locals]} {:keys [id dtype init]}]
           (let [type (dtype/canon dtype)
                 _ (when-not (contains? #{:int :long} type)
                     (decline! :consumer-prefix-local
                               "consumer prefix decomposition requires integral index locals"
                               {:local id :dtype dtype}))
                 expression (consumer-lower-index (util/subst-syms substitutions init) locals)]
             {:operations (conj operations (body/->IndexCompute id expression))
              :substitutions (assoc substitutions id id)
              :environment (assoc environment id type)
              :locals (conj locals id)}))
         {:operations [] :substitutions prefix-aliases :environment base-environment
          :locals (set (vals prefix-aliases))}
         consumer-prefix-locals)
        fold-substitutions (merge (:substitutions prefix-state)
                                  (zipmap (butlast fold-parameters) carries)
                                  {(last fold-parameters) (:name ordered)})
        rewritten-fold
        (let [region {:bindings (vec (mapcat (juxt :id :init) fold-locals))
                      :results fold-results}]
          (->> (if subgroup?
                 (replace-intermediate-values plan subgroup-values region)
                 (replace-intermediate-loads plan reduced-width region))
               (util/subst-syms fold-substitutions)))
        fold-binding-types (into {} (map (juxt :id (comp dtype/canon :dtype))) fold-locals)
        fold-environment (merge (:environment prefix-state)
                                (zipmap carries carry-types)
                                (when subgroup?
                                  (into {}
                                        (map (fn [[[offset component] id]]
                                               [id (nth component-types component)]))
                                        subgroup-values))
                                {(:name ordered) :long})
        fold-update ((:lower-region consumer-lowerer) rewritten-fold carry-types
                     fold-binding-types fold-environment)
        writer :product-consumer-writer
        writer-update
        (body/->IfRegion
         writer
         (conj (vec (:operations fold-update)) (body/->Yield (:results fold-update)))
         [(body/->Yield carries)]
         (mapv body/value updated-carries carry-types))
        identities
        (mapv (fn [identity type]
                (if-let [evidence (constant/value identity)]
                  (body/literal (:value evidence) type)
                  (decline! :consumer-identity
                            "ordered consumer identity requires checked literal evidence"
                            {:identity identity :dtype type})))
              (or (:identities fold-attributes) [(:identity fold-attributes)]) carry-types)
        ordered-loop
        (body/->ForLoop
         (body/value (:name ordered) :long)
         (consumer-lower-index (:lower fold-attributes 0) #{})
         (consumer-lower-index (:extent fold-attributes) #{}) 1
         (mapv #(body/->LoopArg (body/value %1 %2) %3) carries carry-types identities)
         (vec (concat
               (if subgroup?
                 subgroup-operations
                 (concat (:operations element)
                         (map (fn [intermediate producer-value]
                                (body/->ScalarStore intermediate [lane-index]
                                                    producer-value nil))
                              intermediates producer-values)
                         [(barrier)]
                         tree-operations))
               [(body/->ScalarCompute (body/value writer :predicate)
                                      (body/scalar-expression :eq :predicate
                                                              [lane-index (body/literal 0 :int)]))
                writer-update]
               (when-not subgroup? [(barrier)])
               [(body/->Yield updated-carries)]))
         (mapv body/value loop-results carry-types)
         {:association :ordered :source-order true
          :inner-reduction :implementation-defined
          :inner-schedule strategy})
        tail-locals (subvec (vec (get-in consumer [:scalar-region :locals]))
                            consumer-prefix-count)
        tail-locals (mapv #(update % :init (partial replace-fold-components fold loop-results))
                          tail-locals)
        tail-result (replace-fold-components fold loop-results
                                             (get-in consumer [:scalar-region :result]))
        tail-state (lower-locals consumer-lowerer tail-locals (:substitutions prefix-state)
                                 (merge (:environment prefix-state)
                                        (zipmap loop-results carry-types)))
        output-value ((:lower consumer-lowerer)
                      (util/subst-syms (:substitutions tail-state) tail-result)
                      (array-types output) (:environment tail-state))
        final-writer-id :product-consumer-final-writer
        final-writer
        (body/->IfRegion
         final-writer-id
         (vec (concat (:operations tail-state) (:operations output-value)
                      [(body/->ScalarStore output [group-index] (:result output-value) nil)
                       (body/->Yield [])]))
         [(body/->Yield [])] [])
        shared-bytes (if subgroup?
                       0
                       (* workgroup-size (reduce + (map dtype/bytes-of component-types))))
        kernel-body
        (body/make
         {:id [:product-ordered-consumer (:id producer) (:id consumer)]
          :parameters parameters
          :stable-reads (mapv body/stable-read external-inputs)
          :allocations (if subgroup?
                         []
                         (mapv (fn [intermediate component-type]
                                 (body/->WorkgroupAllocation
                                  intermediate component-type [workgroup-size]
                                  (layout/row-major [workgroup-size] component-type)
                                  (dtype/bytes-of component-type)))
                               intermediates component-types))
          :indices [(body/->IndexBinding group-index :group 0)
                    (body/->IndexBinding lane-index (if subgroup? :lane :local) 0)]
          :operations
          (vec
           (concat
            (:operations active-domain)
            (guard-positive-domain
             (:predicates active-domain)
             (concat prefix-computes (when-not subgroup? local-computes) [reduced-compute]
                     (when reduced-active-operation [reduced-active-operation])
                     (:operations prefix-state)
                     [ordered-loop
                      (body/->ScalarCompute
                       (body/value final-writer-id :predicate)
                       (body/scalar-expression :eq :predicate
                                               [lane-index (body/literal 0 :int)]))
                      final-writer]))))
          :schedule (cond-> {:strategy strategy
                             :workgroup-size physical-workgroup-size
                             :axis-partition (:axes plan)}
                      subgroup? (assoc :subgroup-size (:subgroup-size physical-schedule)
                                       :local-unrolling (:local-unrolling physical-schedule)
                                       :neutral-padding (:neutral-padding physical-schedule)))
          :launch (launch/spec
                   {:workgroup-size [physical-workgroup-size]
                    :group-count [(launch/maximum 1 launch-prefix-count)]
                    :shared-memory-bytes shared-bytes})
          :provenance {:dialect :kernel-body :source-dialect :segop
                       :source-operations [(:id producer) (:id consumer)]}
          :attributes {:kind :product-ordered-consumer
                       :physical-schedule physical-schedule
                       :inner-numerics (get-in plan [:numerics :inner])
                       :outer-numerics (get-in plan [:numerics :outer])}})]
    {:kernel-body kernel-body
     :arguments (mapv :id parameters)
     :plan plan}))

(defn from-program
  "Analyze two equations and build their candidate cooperative KernelBody."
  [parallel-program equations]
  (lower (region/analyze parallel-program equations)))
