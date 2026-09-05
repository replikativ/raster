(ns raster.compiler.passes.parallel.scheduled-equation-graph
  "Derive one target-neutral KernelGraph from a scheduled TypedSOAC program.

   This is the shared graph boundary for an ordinary equation and for one iteration of structured
   control. It consumes only the retained functional algorithm, ordered SegOps, and AbstractValue
   contracts; source spelling and operation-family names play no role."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.index-expression :as index-expression]))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :pass :scheduled-equation-graph))))

(defn- value-elements
  [derived-scalars value]
  (let [dimension-value (fn [dimension]
                          ;; AbstractValue uses `(value id)` to distinguish a compound stable
                          ;; value ID from shape syntax. KernelGraph owns explicit integer
                          ;; expressions, so remove only that marker at this physical boundary.
                          (if (and (seq? dimension)
                                   (= 'value (first dimension))
                                   (= 2 (count dimension)))
                            (second dimension)
                            dimension))
        shape (mapv dimension-value (:shape value))
        elements (cond
                   (empty? shape) 1
                   (= 1 (count shape)) (first shape)
                   :else (apply launch/product shape))]
    ;; Shapes retain stable SSA identities.  A preceding pure scalar equation may define one of
    ;; those identities as checked launch algebra (notably normalised `rstr_extent_n` values).
    ;; Rebind that compiler-owned definition here, where graph allocation becomes concrete, so a
    ;; KernelBody's source-derived storage certificate and the graph describe the same extent.
    (launch/rebind-expression elements derived-scalars)))

(defn- integral-scalar-value?
  [value]
  (and value
       (empty? (:shape value))
       (contains? #{:int :long} (some-> (:dtype value) dtype/canon))))

(defn- scalar-result-expression
  "Return one closed typed scalar result with lambda parameters replaced by its stable captures.

   This consumes the retained TypedSOAC scalar equation, rather than source spelling or an
   emitter-side symbol table.  Locals are expanded in their already-validated SSA order; a value
   that is not a pure launch expression is simply not a graph-storage definition."
  [equation]
  (let [algorithm (:algorithm equation)]
    (when (and (soac/program-form? algorithm)
               (= 1 (count (soac/equations algorithm))))
      (let [semantic-equation (first (soac/equations algorithm))
            {:keys [kind captures lambda]} (soac/operation-parts semantic-equation)
            results (nth semantic-equation 2)
            {:keys [parameters locals body-results]} (soac/lambda-parts lambda)]
        (when (and (= 'scalar kind)
                   (= results (:results equation))
                   (= (:operands equation) (:inputs (soac/facts algorithm)))
                   (= (:results equation) (soac/outputs algorithm))
                   (= 1 (count results))
                   (= 1 (count body-results))
                   (= (count captures) (count parameters)))
          (let [substitutions
                (reduce (fn [bindings {:keys [id init]}]
                          (assoc bindings id (util/subst-syms bindings init)))
                        (zipmap parameters captures)
                        locals)]
            [(first results)
             (util/subst-syms substitutions (first body-results))]))))))

(defn- launch-definition
  [values bindings equation]
  (when-let [[result expression] (scalar-result-expression equation)]
    (let [{:keys [captures]} (soac/operation-parts
                              (first (soac/equations (:algorithm equation))))]
      (when (and (integral-scalar-value? (get values result))
                 (every? #(integral-scalar-value? (get values %)) captures))
        (try
          (let [narrowing-cast?
                (boolean
                 (some (fn [form]
                         (and (seq? form) (= 2 (count form))
                              (contains? '#{int clojure.core/int} (first form))))
                       (tree-seq coll? seq expression)))
                _ (when narrowing-cast?
                    (throw (ex-info "launch/storage scalar projection cannot erase narrowing"
                                    {:reason :derived-scalar-narrowing
                                     :result result :expression expression})))
                decline! (fn [rule message data]
                           (throw (ex-info message
                                           (assoc data :reason rule
                                                       :pass :scheduled-equation-graph))))
                index (index-expression/lower-typed
                       expression (set captures)
                       #(get-in values [% :dtype]) (get-in values [result :dtype]) decline!)
                projected (launch/rebind-expression
                           (index-expression/to-launch-expression index decline!) bindings)
                projected-dtype
                (launch/typed-expression-dtype
                 projected #(some-> (get values %) :dtype))]
            (when-not (= projected-dtype (dtype/canon (get-in values [result :dtype])))
              (throw (ex-info "launch/storage scalar projection changes its retained width"
                              {:reason :derived-scalar-dtype
                               :result result :projected projected
                               :projected-dtype projected-dtype
                               :result-dtype (get-in values [result :dtype])})))
            [result projected])
          ;; Scalar equations also represent ordinary arithmetic.  Only the monotone, integral
          ;; subset accepted by KernelLaunch is an allocation definition; other scalar work stays
          ;; opaque at this boundary exactly as before.
          (catch clojure.lang.ExceptionInfo _ nil))))))

(defn derived-scalar-expressions
  "Derive replayable graph-storage expressions from ordered, already-validated scalar equations.

   Definitions are expanded in program order, so a later extent never remains an opaque alias of
   an earlier one.  This is deliberately a small projection from TypedSOAC scalar equations to
   KernelLaunch, not a source re-parser or a new scalar registry."
  [values equations]
  (reduce (fn [bindings equation]
            (if-let [[result expression] (launch-definition values bindings equation)]
              (assoc bindings result expression)
              bindings))
          {} equations))

(defn- typed-host-scalar-equation?
  [values equation algorithm]
  (when-let [[result _] (scalar-result-expression equation)]
    (let [semantic-equation (first (soac/equations algorithm))
          {:keys [attributes captures]} (soac/operation-parts semantic-equation)]
      (and (= algorithm (soac/validate! algorithm))
           (= 1 (count (:results equation)))
           (= (:dtypes attributes) [(get-in values [result :dtype])])
           (empty? (get-in values [result :shape]))
           (every? #(contains? values %) captures)))))

(defn- physical-outputs
  [algorithm]
  (let [facts (soac/facts algorithm)
        producers (into {}
                        (mapcat (fn [equation]
                                  (map vector (nth equation 2)
                                       (soac/physical-results facts equation))))
                        (soac/equations algorithm))]
    (mapv (fn [result]
            (or (get producers result)
                (fail! :scheduled-equation-result-storage
                       "a scheduled result has no physical storage identity"
                       {:result result})))
          (soac/outputs algorithm))))

(defn- external-inputs
  [operations]
  (:inputs
   (reduce (fn [{:keys [initialized] :as state} operation]
             (let [reads (segop/operation-inputs operation)
                   writes (segop/operation-outputs operation)]
               (-> state
                   (update :inputs set/union (set/difference reads initialized))
                   (update :initialized set/union writes))))
           {:inputs #{} :initialized #{}}
           operations)))

(defn- storage-spec
  [values derived-scalars id]
  (let [value (get values id)]
    (when-not value
      (fail! :scheduled-equation-storage-value
             "scheduled storage lacks an AbstractValue"
             {:value id}))
    {:dtype (:dtype value)
     :elements (value-elements derived-scalars value)
     :memory-space (or (:memory-space value) :device)}))

(defn- algorithm-boundary?
  [equation algorithm]
  (and (= algorithm (soac/validate! algorithm))
       (= (:operands equation) (:inputs (soac/facts algorithm)))
       (= (:results equation) (soac/outputs algorithm))))

(defn- ordered-distinct
  [values]
  (reduce (fn [result value]
            (if (some #(= value %) result) result (conj result value)))
          [] values))

(defn- public-scalars
  [scheduled operations buffer-specs]
  (let [operation-required (reduce set/union #{} (map segop/operation-scalars operations))
        ;; KernelLaunch intentionally permits a compound list identity such as `(extent input)`
        ;; as one resolver-owned leaf.  It is not a graph scalar and must not be destructured or
        ;; promoted to a fictitious ABI argument.  Expanded allocation algebra, on the other
        ;; hand, closes only over stable symbol/keyword scalar identities.
        storage-required
        (into #{}
              (filter #(or (symbol? %) (keyword? %)))
              (reduce set/union #{}
                      (map #(launch/expression-references (:elements %))
                           (vals buffer-specs))))
        required (set/union operation-required storage-required)
        ordered (ordered-distinct
                 (concat (filter required (:inputs scheduled))
                         (sort-by pr-str (remove (set (:inputs scheduled)) required))))
        values (:values scheduled)]
    (mapv (fn [id]
            (let [value (get values id)]
              (when-not value
                (fail! :scheduled-equation-scalar-value
                       "scheduled scalar lacks an AbstractValue"
                       {:value id}))
              (when-not (empty? (:shape value))
                (fail! :scheduled-equation-scalar-shape
                       "scheduled scalar dependency must have scalar shape"
                       {:value id :shape (:shape value)}))
              (when (and (contains? storage-required id)
                         (not (integral-scalar-value? value)))
                (fail! :scheduled-equation-storage-scalar
                       "graph storage algebra must close over a declared integral scalar"
                       {:value id :value-contract value}))
              (graph/scalar id (:dtype value))))
          ordered)))

(defn make
  "Build a verified graph from `algorithm` and its fully scheduled SegOp ParallelProgram.

   Optional facts describe the enclosing control/equation context without changing dataflow."
  ([algorithm scheduled] (make algorithm scheduled {}))
  ([algorithm scheduled {:keys [effects provenance attributes]
                         :or {provenance {} attributes {}}}]
   (let [algorithm (soac/validate! algorithm)
         scheduled (program/validate! scheduled segop/segop-node? algorithm-boundary?)
         _ (when-not (= :segop (:dialect scheduled))
             (fail! :scheduled-equation-dialect
                    "KernelGraph derivation requires a fully scheduled :segop program"
                    {:dialect (:dialect scheduled)}))
         host-prefix (vec (take-while #(true? (get-in % [:attributes :host-only]))
                                      (:equations scheduled)))
         numerical-equations (vec (drop (count host-prefix) (:equations scheduled)))
         _ (when-not (and (seq numerical-equations)
                          (every? #(typed-host-scalar-equation? (:values scheduled) % (:algorithm %))
                                  host-prefix))
             (fail! :scheduled-equation-prefix
                    "a graph body requires an earlier-only typed host-scalar prefix and numerical equations"
                    {:host-prefix (mapv :id host-prefix)
                     :numerical-equations (mapv :id numerical-equations)}))
         retained-equations (vec (mapcat (comp soac/equations :algorithm) numerical-equations))
         numerical-inputs (if (seq host-prefix)
                            (:operands (first numerical-equations))
                            (:inputs scheduled))
         _ (when-not (and (= (soac/equations algorithm) retained-equations)
                          (= numerical-inputs (:inputs (soac/facts algorithm)))
                          (= (soac/outputs algorithm) (:outputs scheduled)))
             (fail! :scheduled-equation-algorithm
                    "scheduled equations or program boundary differ from the retained algorithm"
                    {:algorithm-inputs (:inputs (soac/facts algorithm))
                     :scheduled-inputs numerical-inputs
                     :algorithm-outputs (soac/outputs algorithm)
                     :scheduled-outputs (:outputs scheduled)}))
         operations (vec (mapcat :operations numerical-equations))
         _ (when (empty? operations)
             (fail! :scheduled-equation-empty
                    "a KernelGraph requires at least one scheduled operation" {}))
         derived-scalars (derived-scalar-expressions (:values scheduled) host-prefix)
         inputs (external-inputs operations)
         outputs (set (physical-outputs algorithm))
         operation-values (reduce set/union #{}
                                  (map #(set/union (segop/operation-inputs %)
                                                   (segop/operation-outputs %))
                                       operations))
         temporary-ids (set/difference operation-values inputs outputs)
         values (:values scheduled)
         buffer-specs (into {}
                            (map (fn [id] [id (storage-spec values derived-scalars id)]))
                            (set/union inputs outputs temporary-ids))
         scalars (public-scalars scheduled operations buffer-specs)]
     (graph/from-segops
      operations
      {:inputs inputs
       :outputs outputs
       :temporaries (select-keys buffer-specs temporary-ids)
       :scalars scalars
       :buffer-specs buffer-specs
       :dtype (:dtype (first (vals buffer-specs)))
       :effects (or effects {:semantic (:effects (soac/facts algorithm))})
       :provenance (merge {:source-dialect :typed-soac
                           :algorithm-dialect :typed-soac
                           :schedule-dialect :segop}
                          provenance)
       ;; This map is derived solely from the validated leading host equations above.  It lets a
       ;; schedule validator normalize its source-level extent identities to the same public
       ;; launch algebra as the graph's fully expanded storage specs.  Callers cannot override it.
       :attributes (cond-> attributes
                     (seq derived-scalars)
                     (assoc :derived-storage-scalars derived-scalars))}))))
