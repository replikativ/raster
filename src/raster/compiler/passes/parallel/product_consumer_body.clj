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
        scalar-types (into {} (map (juxt :id :dtype)) (:scalars graph))
        requirement-shape
        (fn [expression]
          (let [scope (set (util/free-syms expression))
                lowered (index-expression/lower-typed expression scope scalar-types :long decline!)]
            (index-expression/to-launch-expression lowered decline!)))]
    {:buffers buffers
     :array-types (into {} (map (juxt :id :dtype)) (vals buffers))
     :array-shapes
     (into {}
           (map (fn [[id buffer]]
                  [id [(if-let [requirement (get (:consumer-read-requirements plan) id)]
                         (requirement-shape requirement)
                         (:elements buffer))]]))
           buffers)
     :scalar-types scalar-types
     :values (:values scheduled-body)}))

(defn- fold-of [consumer]
  (->> (tree-seq coll? seq (:scalar-region consumer))
       (filter dialect/product-fold-form?) distinct first))

(defn- contains-fold? [value]
  (boolean (some dialect/product-fold-form? (tree-seq coll? seq value))))

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

(defn- replace-fold-components [fold carries value]
  (walk/postwalk
   (fn [form]
     (if (and (dialect/product-component-form? form) (= fold (second form)))
       (let [ordinal (nth form 2)]
         (or (nth carries ordinal nil)
             (decline! :consumer-component
                       "consumer projection references a missing ordered-fold component"
                       {:ordinal ordinal :carries carries})))
       form))
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
  (let [{:keys [producer consumer intermediate workgroup-size]} plan
        {:keys [prefix ordered local reduced]} (:axes plan)
        {:keys [buffers array-types array-shapes scalar-types]} (graph-options plan)
        components (get-in producer [:reduction :components])
        _ (when-not (= 1 (count components))
            (decline! :producer-components
                      "the initial fused body requires one private product component"
                      {:components components}))
        component (first components)
        component-type (dtype/canon (:dtype component))
        reduced-width (:bound reduced)
        local-volume (reduce * 1 (map :bound local))
        _ (when-not (= workgroup-size (* local-volume reduced-width))
            (decline! :worker-decomposition
                      "cooperative width must equal local suffix volume times reduction width"
                      {:workgroup-size workgroup-size :local-volume local-volume
                       :reduced-width reduced-width}))
        external-inputs (vec (sort-by name
                                      (disj (set/union (:inputs producer) (:inputs consumer))
                                            intermediate)))
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
        {:keys [element combine]} (product-regions/lower producer producer-options decline!)
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
        producer-value (first (:results element))
        tree-strides (vec (take-while pos? (iterate #(quot % 2) (quot reduced-width 2))))
        tree-operations
        (mapcat
         (fn [stride]
           (let [left (symbol (str "product-consumer-left-" stride))
                 right (symbol (str "product-consumer-right-" stride))
                 merged (combine [left] [right])
                 predicate (keyword (str "product-consumer-tree-" stride))]
             [(body/->ScalarCompute (body/value predicate :predicate)
                                    (body/scalar-expression :lt :predicate
                                                            [(:name reduced)
                                                             (body/literal stride :long)]))
              (body/->IfRegion
               predicate
               (vec (concat
                     [(body/->ScalarLoad (body/value left component-type) intermediate [lane-index]
                                         nil nil :cached)
                      (body/->ScalarLoad (body/value right component-type) intermediate
                                         [(body/expression :add lane-index stride)] nil nil :cached)]
                     (:operations merged)
                     [(body/->ScalarStore intermediate [lane-index]
                                           (first (:results merged)) nil)
                      (body/->Yield [])]))
               [(body/->Yield [])] [])
              (barrier)]))
         tree-strides)
        fold (fold-of consumer)
        {fold-attributes :attributes fold-lambda :lambda} (dialect/product-fold-parts fold)
        {fold-parameters :parameters fold-locals :locals fold-results :body-results}
        (dialect/lambda-parts fold-lambda)
        carry-types (mapv dtype/canon (:dtypes fold-attributes))
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
        consumer-arrays (conj (set (:inputs consumer)) intermediate)
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
        (->> {:bindings (vec (mapcat (juxt :id :init) fold-locals)) :results fold-results}
             (replace-intermediate-loads plan reduced-width)
             (util/subst-syms fold-substitutions))
        fold-binding-types (into {} (map (juxt :id (comp dtype/canon :dtype))) fold-locals)
        fold-environment (merge (:environment prefix-state)
                                (zipmap carries carry-types)
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
              (:identities fold-attributes) carry-types)
        ordered-loop
        (body/->ForLoop
         (body/value (:name ordered) :long)
         (consumer-lower-index (:lower fold-attributes 0) #{})
         (consumer-lower-index (:extent fold-attributes) #{}) 1
         (mapv #(body/->LoopArg (body/value %1 %2) %3) carries carry-types identities)
         (vec (concat
               (:operations element)
               [(body/->ScalarStore intermediate [lane-index] producer-value nil)
                (barrier)]
               tree-operations
               [(body/->ScalarCompute (body/value writer :predicate)
                                      (body/scalar-expression :eq :predicate
                                                              [lane-index (body/literal 0 :int)]))
                writer-update
                (barrier)
                (body/->Yield updated-carries)]))
         (mapv body/value loop-results carry-types)
         {:association :ordered :source-order true
          :inner-reduction :implementation-defined})
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
        shared-bytes (* workgroup-size (dtype/bytes-of component-type))
        kernel-body
        (body/make
         {:id [:product-ordered-consumer (:id producer) (:id consumer)]
          :parameters parameters
          :stable-reads (mapv body/stable-read external-inputs)
          :allocations [(body/->WorkgroupAllocation
                         intermediate component-type [workgroup-size]
                         (layout/row-major [workgroup-size] component-type)
                         (dtype/bytes-of component-type))]
          :indices [(body/->IndexBinding group-index :group 0)
                    (body/->IndexBinding lane-index :local 0)]
          :operations
          (vec
           (concat
            (:operations active-domain)
            (guard-positive-domain
             (:predicates active-domain)
             (concat prefix-computes local-computes [reduced-compute]
                     (:operations prefix-state)
                     [ordered-loop
                      (body/->ScalarCompute
                       (body/value final-writer-id :predicate)
                       (body/scalar-expression :eq :predicate
                                               [lane-index (body/literal 0 :int)]))
                      final-writer]))))
          :schedule {:strategy :product-tree-ordered-consumer
                     :workgroup-size workgroup-size
                     :axis-partition (:axes plan)}
          :launch (launch/spec
                   {:workgroup-size [workgroup-size]
                    :group-count [(launch/maximum 1 launch-prefix-count)]
                    :shared-memory-bytes shared-bytes})
          :provenance {:dialect :kernel-body :source-dialect :segop
                       :source-operations [(:id producer) (:id consumer)]}
          :attributes {:kind :product-ordered-consumer
                       :inner-numerics (get-in plan [:numerics :inner])
                       :outer-numerics (get-in plan [:numerics :outer])}})]
    {:kernel-body kernel-body
     :arguments (mapv :id parameters)
     :plan plan}))

(defn from-program
  "Analyze two equations and build their candidate cooperative KernelBody."
  [parallel-program equations]
  (lower (region/analyze parallel-program equations)))
