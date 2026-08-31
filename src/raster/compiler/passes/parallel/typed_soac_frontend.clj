(ns raster.compiler.passes.parallel.typed-soac-frontend
  "Direct analyzed-source to TypedSOAC construction for the closed map/scalar/reduce/certified-scan subset.

   This is the production front door for TypedSOAC.  It consumes the retained source form and
   walker type metadata directly; it never constructs the older record graph.  Unsupported
   parallel forms return nil so the pipeline can use the explicitly separate compatibility route.
   Once an operation is accepted, all value, effect and provenance facts are made explicit before
   fusion or scheduling sees it."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.contraction-facts :as contraction-facts]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.par :as par]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.fusion-support :as fusion-support]
            [raster.compiler.passes.scalar.effects :as effects]))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :front-end :analyzed-source))))

(defn- extract-io
  [body index outputs & {:keys [accumulator]}]
  (let [inputs (par/collect-aget-arrays body)
        written (par/collect-aset-arrays body)
        excluded (set/union inputs (set outputs) written #{index}
                            (if accumulator #{accumulator} #{})
                            descriptor/aget-ops descriptor/aset-ops
                            #{'do 'let 'let* 'if 'double 'float 'int 'long})]
    {:inputs inputs
     :outputs (into (set outputs) written)
     :scalars (set/difference (util/free-syms body) excluded)}))

(defn- strip-index-cast
  [expression]
  (if (and (seq? expression)
           (contains? #{'long 'int 'clojure.core/long 'clojure.core/int}
                      (first expression))
           (= 2 (count expression)))
    (second expression)
    expression))

(def ^:private unique-index-ops
  '#{raster.par/unique-index unique-index})

(defn- unique-index-expression
  "Return the inner destination expression when `expression` carries Raster's explicit
   uniqueness contract. The marker may be a direct source call or a walker-devirtualized call;
   semantic-op/call-args are the only sanctioned way to look through the latter."
  [expression]
  (when (and (seq? expression)
             (contains? unique-index-ops (descriptor/semantic-op expression))
             (= 1 (count (descriptor/call-args expression))))
    (first (descriptor/call-args expression))))

(defn- retained-local-dtype
  [binding init]
  (let [init-tag (when (instance? clojure.lang.IObj init)
                   (or (:raster.type/tag (meta init)) (:tag (meta init))))]
    (or (dtype/dtype-for-scalar-tag (types/sym-type-tag binding))
        (dtype/dtype-for-scalar-tag init-tag))))

(defn- pointwise-region
  "Recognize an ordered, pure local-SSA spine ending exclusively in indexed stores.

   Local types come only from retained walker/TypedClojure facts. Nested local scopes and missing
   local types decline; guessing them in this source recognizer or a C emitter would make the
   region only nominally typed. Store legality is checked separately after local dependencies are
   expanded for analysis."
  [body index]
  (cond
    (and (seq? body) (form/let-head? (first body)))
    (let [[_ bindings & nested-body] body]
      (when (and (even? (count bindings))
                 (seq nested-body)
                 (not-any? util/effectful? (take-nth 2 (rest bindings))))
        (when-let [{nested-locals :locals stores :stores}
                   (pointwise-region (list* 'do nested-body) index)]
          ;; Flattening nested lexical scopes needs alpha-renaming across each scope. Keep this
          ;; first production slice to the ANF shape the walker emits: one local spine around the
          ;; store sequence.
          (when (empty? nested-locals)
            (let [pairs (vec (partition 2 bindings))
                  typed (mapv (fn [[binding init]]
                                [binding init (retained-local-dtype binding init)])
                              pairs)]
              (when (every? (comp some? #(nth % 2)) typed)
                (let [{:keys [locals substitutions]}
                      (reduce (fn [{:keys [locals substitutions]} [binding init local-dtype]]
                                (let [id (symbol (str "rstr_local_" (count locals)))]
                                  {:locals (conj locals
                                                 {:id id :dtype local-dtype
                                                  :init (util/subst-syms substitutions init)})
                                   :substitutions (assoc substitutions binding id)}))
                              {:locals [] :substitutions {}} typed)]
                  {:locals locals
                   :stores (mapv (fn [store]
                                   (reduce (fn [store field]
                                             (update store field
                                                     #(util/subst-syms substitutions %)))
                                           store [:index :predicate :value]))
                                 stores)})))))))

    (and (seq? body) (= 'do (first body)))
    (let [groups (mapv #(pointwise-region % index) (rest body))]
      (when (and (seq groups) (every? some? groups)
                 (every? (comp empty? :locals) groups))
        {:locals [] :stores (vec (mapcat :stores groups))}))

    (and (seq? body)
         (contains? #{'if 'clojure.core/if} (first body))
         (<= 3 (count body) 4))
    (let [[_ predicate then-expression else-expression] body
          then-region (pointwise-region then-expression index)
          else-region (when else-expression (pointwise-region else-expression index))]
      (when (and then-region
                 (empty? (:locals then-region))
                 (or (nil? else-expression)
                     (and else-region (empty? (:locals else-region))))
                 (or (nil? else-region)
                     (= (mapv (juxt :out :index) (:stores then-region))
                        (mapv (juxt :out :index) (:stores else-region)))))
        {:locals []
         :stores
         (mapv (fn [ordinal then-store]
                 (let [else-store (when else-region (nth (:stores else-region) ordinal))]
                   (if else-store
                     (assoc then-store
                            :value (list 'if predicate (:value then-store) (:value else-store))
                            :predicate (list 'if predicate
                                             (:predicate then-store)
                                             (:predicate else-store)))
                     (update then-store :predicate
                             #(if (contains? #{true 1} %)
                                predicate
                                (list 'if predicate % 0))))))
               (range) (:stores then-region))}))

    (descriptor/aset-call? body)
    (let [arguments (vec (descriptor/call-args body))]
      (when (= 3 (count arguments))
        (let [raw-index (nth arguments 1)
              unique-index (unique-index-expression raw-index)
              value (nth arguments 2)
              cast? (and (seq? value)
                         (contains? #{'float 'double 'int 'long
                                      'clojure.core/float 'clojure.core/double}
                                    (first value))
                         (= 2 (count value)))]
          {:locals []
           :stores [{:out (descriptor/aset-array-sym body)
                     :index (strip-index-cast (or unique-index raw-index))
                     :conflict (when unique-index :unique)
                     :predicate 1
                     :value (if cast? (second value) value)
                     :cast (when cast? (first value))}]})))

    :else nil))

(defn- expanded-local-expression
  [locals expression]
  (reduce (fn [body {:keys [id init]}]
            (util/subst-syms {id init} body))
          expression (reverse locals)))

(defn- independent-stores?
  [locals stores]
  (let [destinations (mapv :out stores)
        destination-set (set destinations)]
    (and (= (count destinations) (count destination-set))
         (every? (fn [{:keys [out index predicate value]}]
                   (empty? (disj (set/intersection destination-set
                                                   (par/collect-aget-arrays
                                                    (expanded-local-expression
                                                     locals (list 'do index predicate value))))
                                 out)))
                 stores))))

(defn- effect-result-id
  [equation-id ordinal]
  [:effect-map equation-id ordinal])

(defn- typed-result-transform
  "Close a source epilogue descriptor over lexical scalar-region parameters.

   Program value IDs remain on the boundary records; the expression names only lambda parameters
   and segment indices. This is what makes later ParallelProgram SSA remapping alpha-stable."
  [epilogue]
  (when epilogue
    (dialect/make-result-transform
     {:accumulator (:acc epilogue)
      :expression (:expr epilogue)
      :operands (mapv #(-> % (assoc :value (:sym %)) (dissoc :sym)) (:operands epilogue))
      :scalars (mapv #(-> % (assoc :value (:sym %)) (dissoc :sym)) (:scalars epilogue))
      :result-dtype (or (:dtype epilogue) :float)})))

(defn- effect-map-description
  [id symbol index extent {:keys [locals stores]} elem-type]
  (when (and (seq stores) (independent-stores? locals stores))
    (let [stores (mapv #(merge {:index index :predicate 1} %) stores)
          pointwise? (every? #(= index (:index %)) stores)
          stores (if pointwise?
                   (mapv (fn [{:keys [out predicate value] :as store}]
                           (assoc store :value
                                  (if (contains? #{true 1} predicate)
                                    value
                                    ;; A guarded dense write preserves its caller-owned
                                    ;; destination. Making that read explicit yields an ordinary
                                    ;; inout map instead of a hidden conditional effect.
                                    (list 'if predicate value
                                          (list 'clojure.core/aget out index)))))
                         stores)
                   stores)
          conflict (when (every? #(= :unique (:conflict %)) stores) :unique)
          destinations (mapv :out stores)
          values (mapv :value stores)
          write-indices (mapv :index stores)
          predicates (mapv :predicate stores)
          analysis-values (concat (map :init locals) write-indices predicates values)
          io (update (extract-io (list* 'do analysis-values) index destinations)
                     :scalars set/difference (set (map :id locals)))
          results (mapv #(effect-result-id id %) (range (count stores)))]
      (when (or pointwise? (= :unique conflict))
        (merge {:kind (if pointwise? :map :scatter)
              :id id :sym symbol :index index :extent extent
              :results results :locals locals :bodies values :casts (mapv :cast stores)
              :write-indices write-indices :predicates predicates
              :conflict (when-not pointwise? conflict)
              :effect-only? true :host-binding symbol :elem-type elem-type
              :result-storage
              (mapv (fn [destination]
                      {:destination destination
                       :access (if (or (not pointwise?)
                                       (contains? (:inputs io) destination))
                                 :read-write :write)
                       :host-return :effect})
                    destinations)}
               io)))))

(defn- operation-description
  [id symbol expression default-dtype]
  (cond
    (par/par-map-pure-form? expression)
    (let [{:keys [idx bound cast body elem-type]} (par/extract-par-map-pure-info expression)
          io (extract-io body idx [symbol])]
      (merge {:kind :map :id id :sym symbol :results [symbol]
              :index idx :extent bound :locals [] :casts [cast] :bodies [body]
              :pure? true :elem-type elem-type}
             io))

    (par/par-map-form? expression)
    (let [{:keys [out idx bound cast body elem-type offset]}
          (par/extract-par-map-info expression)
          io (extract-io body idx [out])]
      ;; Offset maps are not pointwise in the result coordinate and require an indexed/scatter
      ;; operation in the typed dialect. A binder with the same spelling as the caller-owned
      ;; destination also needs distinct value/view identity before it can be SSA. A pointwise read
      ;; of the destination is an explicit read/write operand; scheduling must retain it as one
      ;; physical inout value rather than manufacturing separate aliased input/output pointers.
      (when-not (or offset (= symbol out))
        (merge {:kind :map :id id :sym symbol :results [symbol]
                :index idx :extent bound :locals [] :casts [cast] :bodies [body]
                :result-storage [{:destination out
                                  :access (if (contains? (:inputs io) out) :read-write :write)
                                  :host-return :buffer}]
                :host-binding symbol
                :elem-type elem-type}
               io)))

    (par/par-map2-form? expression)
    (let [{:keys [out1 out2 idx bound cast body1 body2 elem-type]}
          (par/extract-par-map2-info expression)]
      (effect-map-description id symbol idx bound
                              {:locals []
                               :stores [{:out out1 :value body1 :cast cast}
                                        {:out out2 :value body2 :cast cast}]}
                              elem-type))

    (par/par-reduce-form? expression)
    (let [{:keys [acc init idx bound body elem-type]} (par/extract-par-reduce-info expression)
          io (extract-io body idx [symbol] :accumulator acc)]
      (merge {:kind :reduce :id id :sym symbol :index idx :extent bound
              :product (reduction/scalar
                        ;; The walker stamps contextual FP narrowing on the par form.  Without
                        ;; that retained fact, Clojure's scalar reduction semantics are double;
                        ;; the target's preferred array dtype is not permission to narrow it.
                        {:accumulator acc :neutral init :dtype (or elem-type :double)
                         :result symbol :index idx :step-result body
                         :attributes {:source :raster.par/reduce}})}
             io))

    (and (seq? expression) (= 'raster.par/contract (first expression)))
    (let [facts (contraction-facts/contraction-facts
                 expression :dtype (or (:raster.type/elem-type (meta expression))
                                       default-dtype :double))
          {:keys [free-axes contract-axes out opts]} facts
          epilogue (:epilogue facts)
          result-transform (typed-result-transform epilogue)
          epilogue-arrays (set (map :value (:operands result-transform)))
          epilogue-scalar-ids (set (map :value (:scalars result-transform)))]
      ;; The direct slice is scalar segmented-reduction algebra plus an optional closed typed
      ;; result transform. Staged quantization carries additional schedule/load contracts and must
      ;; not be admitted by dropping those facts.
      (when (and (seq free-axes) (seq contract-axes)
                 ;; Decode lambdas, declared physical maps, staged accumulators and output
                 ;; conversions are not scalar fold syntax. Keep them on the certified
                 ;; contraction route until TypedSOAC represents those facts explicitly.
                 (empty? (apply dissoc opts [:init :combine :algebra :epilogue]))
                 (or (nil? epilogue) (dialect/result-transform? result-transform))
                 (reduction/scalar? (:reduction facts)))
        (let [product (:reduction facts)
              component (first (:components product))
              step-result (first (:results (reduction/fold-region product)))
              arrays (set/union (set (map :sym (:operands facts))) epilogue-arrays)
              axis-symbols (set (concat (map first free-axes) [(:index product)]))
              bound-symbols (reduce set/union #{}
                                    (map util/free-syms
                                         (concat (map second free-axes)
                                                 (map second contract-axes))))
              scalars (set/difference
                       (set/union bound-symbols (util/free-syms step-result))
                       arrays axis-symbols epilogue-scalar-ids #{(:accumulator component) out})]
          {:kind :segmented-reduce :id id :sym symbol
           :segment-axes free-axes
           :reduce-index (:index product)
           :reduce-extent (second (:flat-contract-axis facts))
           :results [(effect-result-id id 0)]
           :product product :inputs arrays :outputs #{out}
           :scalars (set/union scalars epilogue-scalar-ids)
           :result-transform result-transform
           :effect-only? true :host-binding symbol
           :result-storage [{:destination out :access :write :host-return :effect}]})))

    (or (par/par-scan-form? expression)
        (par/par-scan-exclusive-form? expression))
    (let [mode (if (par/par-scan-exclusive-form? expression) :exclusive :inclusive)
          {:keys [out acc init idx bound cast body elem-type]}
          ((if (= :exclusive mode)
             par/extract-par-scan-exclusive-info
             par/extract-par-scan-info)
           expression)
          scan-dtype (or elem-type
                         (dtype/dtype-for-scalar-tag cast)
                         :double)]
      ;; A source binder with the same spelling as the caller-owned destination needs an explicit
      ;; value/view identity before it can be SSA. Keep that uncommon compatibility form outside
      ;; the typed route instead of pretending the destination is both an input and a definition.
      (when-not (= symbol out)
        (let [algebra (scan/certify {:acc acc :init init :lambda body :out out} scan-dtype)]
          (merge {:kind :scan :id id :sym symbol :index idx :extent bound :mode mode
                  :primary-out out :accumulator acc :identity init :dtype scan-dtype
                  :algebra algebra :body body}
                 (extract-io body idx [out] :accumulator acc)))))

    (par/par-map-void-form? expression)
    (let [{:keys [idx bound body elem-type]}
          (par/extract-par-map-void-info expression)]
      (when-let [region (pointwise-region body idx)]
        (effect-map-description id symbol idx bound region elem-type)))

    :else nil))

(defn- provably-pure-scalar?
  [expression]
  (or (effects/removable-expr? expression)
      (and (descriptor/alength-op? (descriptor/semantic-op expression))
           (= 1 (count (descriptor/call-args expression)))
           (symbol? (first (descriptor/call-args expression))))))

(defn- parallel-extent
  [expression]
  (cond
    (par/par-map-pure-form? expression) (nth expression 2)
    (par/par-map-form? expression) (nth expression 3)
    (par/par-map2-form? expression) (nth expression 4)
    (par/par-reduce-form? expression) (nth expression 4)
    (or (par/par-scan-form? expression) (par/par-scan-exclusive-form? expression))
    (nth expression 5)
    (par/par-map-void-form? expression) (nth expression 2)
    :else nil))

(defn- replace-parallel-extent
  [expression extent]
  (let [position (cond
                   (par/par-map-pure-form? expression) 2
                   (par/par-map-form? expression) 3
                   (par/par-map2-form? expression) 4
                   (par/par-reduce-form? expression) 4
                   (or (par/par-scan-form? expression)
                       (par/par-scan-exclusive-form? expression)) 5
                   (par/par-map-void-form? expression) 2)]
    (when position
      (with-meta (apply list (assoc (vec expression) position extent)) (meta expression)))))

(defn normalize-source
  "Give pure compound parallel extents stable scalar SSA identities before dialect construction.

   TypedSOAC operations name extents; executable host expressions never leak into schedule fields.
   This normalization inserts an ordinary typed scalar binding immediately before its operation,
   so the same expression remains visible to JVM materialization and runtime specialization."
  [source]
  (if (and (seq? source) (contains? #{'let 'let*} (first source)))
    (let [[head bindings & body] source
          pairs (vec (partition 2 bindings))
          normalized
          (mapcat
           (fn [ordinal [symbol expression]]
             (let [extent (parallel-extent expression)
                   canonical-extent (some-> extent descriptor/unwrap-int-cast)]
               (cond
                 (nil? extent)
                 [[symbol expression]]

                 ;; Integral casts are representation noise, not new semantic dimensions.
                 ;; Keeping their underlying value identity also prevents one array from
                 ;; acquiring incompatible [n] and [(long n)] shapes across operations.
                 (dialect/extent? canonical-extent)
                 [[symbol (replace-parallel-extent expression canonical-extent)]]

                 (provably-pure-scalar? canonical-extent)
                 (let [extent-id (with-meta
                                   (clojure.core/symbol (str "rstr_extent_" ordinal))
                                   {:tag 'long :raster.type/tag 'long})]
                   [[extent-id canonical-extent]
                    [symbol (replace-parallel-extent expression extent-id)]])

                 :else
                 [[symbol expression]])))
           (range) pairs)]
      (with-meta (list* head (vec (mapcat identity normalized)) body) (meta source)))
    source))

(defn- alength-array
  [expression]
  (when (and (seq? expression)
             (descriptor/alength-op? (descriptor/semantic-op expression))
             (= 1 (count (descriptor/call-args expression))))
    (first (descriptor/call-args expression))))

(defn- source-descriptions
  [pairs default-dtype]
  (mapv (fn [id [symbol expression]]
          (or (operation-description id symbol expression default-dtype)
              (if (par/par-form? expression)
                {:kind :unsupported :id id :sym symbol :expr expression}
                {:kind :scalar :id id :sym symbol :expr expression})))
        (range) pairs))

(defn- normalize-extents
  [descriptions]
  (:descriptions
   (reduce
    (fn [{:keys [extents scalar-representatives] :as state} description]
      (case (:kind description)
        :scalar
        (let [expression (:expr description)
              array (alength-array expression)
              representative (cond
                               (and array (contains? extents array)) (get extents array)
                               (symbol? expression)
                               (get scalar-representatives expression expression)
                               (integer? expression) expression
                               :else (:sym description))]
          (-> state
              (update :descriptions conj
                      (assoc description :expr (if (= representative (:sym description))
                                                 expression representative)))
              (assoc-in [:scalar-representatives (:sym description)] representative)))

        (:map :scatter :reduce :scan)
        (let [extent (descriptor/unwrap-int-cast (:extent description))
              array (alength-array extent)
              extent' (cond
                        (and array (contains? extents array)) (get extents array)
                        (symbol? extent) (get scalar-representatives extent extent)
                        :else extent)
              description' (assoc description :extent extent')]
          (cond-> (update state :descriptions conj description')
            (contains? #{:map :scatter :scan} (:kind description'))
            (assoc-in [:extents (:sym description')] extent')))

        (update state :descriptions conj description)))
    {:descriptions [] :extents {} :scalar-representatives {}}
    descriptions)))

(defn- generated-scaffolding?
  [description physical-outputs]
  (and (= :scalar (:kind description))
       (let [expression (:expr description)]
         (or (and (symbol? expression) (contains? physical-outputs expression))
             (and (contains? physical-outputs (:sym description))
                  (seq? expression)
                  (descriptor/alloc-op? (descriptor/semantic-op expression)))))))

(defn- physical-output-symbols
  [descriptions]
  (reduce set/union #{}
          (map #(if (contains? #{:map :scatter :reduce :segmented-reduce :scan} (:kind %))
                  (:outputs %) #{})
               descriptions)))

(defn- supported-descriptions?
  [descriptions]
  (let [physical-outputs (physical-output-symbols descriptions)]
    (every? (fn [description]
              (case (:kind description)
                :scalar (or (provably-pure-scalar? (:expr description))
                            (generated-scaffolding? description physical-outputs))
                :map (or (:pure? description)
                         (and (seq (:result-storage description))
                              (every? (comp symbol? :destination)
                                      (:result-storage description))))
                :scatter (and (seq (:result-storage description))
                              (= :unique (:conflict description))
                              (every? (comp symbol? :destination)
                                      (:result-storage description)))
                :reduce true
                :segmented-reduce (and (seq (:segment-axes description))
                                       (symbol? (get-in description [:result-storage 0 :destination])))
                :scan (symbol? (:primary-out description))
                false))
            descriptions)))

(defn- element-symbols [n] (mapv #(symbol (str "%element" %)) (range n)))
(defn- capture-symbols [n] (mapv #(symbol (str "%capture" %)) (range n)))

(defn- pointwise-input?
  [expressions array index]
  (let [reads (filter (fn [{:keys [sym]}]
                        (and (symbol? sym) (= (name array) (name sym))))
                      (mapcat descriptor/aget-reads expressions))]
    (and (seq reads)
         (every? #(= index (descriptor/unwrap-int-cast (:idx %))) reads))))

(defn- elementize
  [expressions arrays parameters index]
  (mapv (fn [expression]
          (reduce (fn [body [array parameter]]
                    (fusion-support/substitute-aget body array index index parameter))
                  expression (map vector arrays parameters)))
        expressions))

(defn- map-equation
  [description]
  (let [{:keys [id index extent locals casts bodies inputs results]} description
        expressions (mapv (fn [cast body] (if cast (list cast body) body)) casts bodies)
        all-expressions (into (mapv :init locals) expressions)
        [pointwise stable] ((juxt filter remove) #(pointwise-input? all-expressions % index) inputs)
        arrays (vec (sort-by pr-str pointwise))
        captures (vec (sort-by pr-str (distinct (concat stable (:scalars description)))))
        parameters (element-symbols (count arrays))
        capture-parameters (capture-symbols (count captures))
        substitutions (zipmap captures capture-parameters)
        local-forms
        (->> locals
             (mapv (fn [{:keys [id dtype init]}]
                     (dialect/local-value
                      id dtype
                      (util/subst-syms
                       substitutions
                       (first (elementize [init] arrays parameters index)))))))
        body-results (mapv #(util/subst-syms (zipmap captures capture-parameters) %)
                           (elementize expressions arrays parameters index))]
    (list '= id results
          (list 'map {:index index :extent extent
                      :attributes {:stable-array-captures (vec (sort-by pr-str stable))}}
                arrays captures
                (dialect/lambda-form (vec (concat parameters capture-parameters))
                                     local-forms body-results)))))

(defn- scatter-equation
  [description]
  (let [{:keys [id index extent locals casts bodies inputs results result-storage
                write-indices predicates conflict]} description
        values (mapv (fn [cast body] (if cast (list cast body) body)) casts bodies)
        destinations (mapv :destination result-storage)
        semantic-inputs (into (set inputs) destinations)
        all-expressions (vec (concat (map :init locals) write-indices predicates values))
        [pointwise stable]
        ((juxt filter remove) #(pointwise-input? all-expressions % index) semantic-inputs)
        arrays (vec (sort-by pr-str pointwise))
        captures (vec (sort-by pr-str (distinct (concat stable (:scalars description)))))
        parameters (element-symbols (count arrays))
        capture-parameters (capture-symbols (count captures))
        substitutions (zipmap captures capture-parameters)
        transform (fn [expression]
                    (util/subst-syms
                     substitutions
                     (first (elementize [expression] arrays parameters index))))
        local-forms (mapv (fn [{:keys [id dtype init]}]
                            (dialect/local-value id dtype (transform init)))
                          locals)
        writes (mapv (fn [destination-index predicate value]
                       (list 'write (transform destination-index)
                             (transform predicate) (transform value)))
                     write-indices predicates values)]
    (list '= id results
          (list 'scatter {:index index :extent extent :conflict conflict
                          :attributes {:stable-array-captures
                                       (vec (sort-by pr-str stable))}}
                arrays captures
                (dialect/lambda-form (vec (concat parameters capture-parameters))
                                     local-forms writes)))))

(defn- reduce-equation
  [{:keys [id extent inputs scalars product]}]
  (let [component (first (:components product))
        expressions (:results (reduction/fold-region product))
        index (:index product)
        [pointwise stable] ((juxt filter remove) #(pointwise-input? expressions % index) inputs)
        arrays (vec (sort-by pr-str pointwise))
        captures (vec (sort-by pr-str (distinct (concat stable scalars))))
        elements (element-symbols (count arrays))
        capture-parameters (capture-symbols (count captures))
        results (->> (elementize expressions arrays elements index)
                     (mapv #(util/subst-syms (zipmap captures capture-parameters) %)))]
    (list '= id (vec (filter some? (reduction/results product)))
          (list 'reduce {:index index :extent extent
                         :attributes {:stable-array-captures (vec (sort-by pr-str stable))}
                         :accumulators [(:accumulator component)]
                         :identities [(:neutral component)]
                         :dtypes [(:dtype component)]
                         :algebra [(:algebra product)]}
                arrays captures
                (dialect/lambda-form
                 (vec (concat [(:accumulator component)] elements capture-parameters))
                 results)))))

(defn- segmented-reduce-equation
  [{:keys [id segment-axes reduce-index reduce-extent inputs scalars product results
           result-transform]}]
  (let [component (first (:components product))
        stable (vec (sort-by pr-str inputs))
        captures (vec (sort-by pr-str (distinct (concat stable scalars))))
        capture-parameters (capture-symbols (count captures))
        step-results (mapv #(util/subst-syms (zipmap captures capture-parameters) %)
                           (:results (reduction/fold-region product)))]
    (list '= id results
          (list 'segmented-reduce
                {:segment-axes segment-axes
                 :index reduce-index :extent reduce-extent
                 :attributes {:stable-array-captures stable
                              :source-operation :raster.par/contract}
                 :accumulators [(:accumulator component)]
                 :identities [(:neutral component)]
                 :dtypes [(:dtype component)]
                 :algebra [(:algebra product)]
                 :result-transform result-transform}
                [] captures
                (dialect/lambda-form
                 (vec (concat [(:accumulator component)] capture-parameters))
                 step-results)))))

(defn- scan-equation
  [{:keys [id sym index extent mode inputs scalars primary-out accumulator identity dtype body]}]
  (let [[pointwise stable] ((juxt filter remove) #(pointwise-input? [body] % index) inputs)
        arrays (vec (sort-by pr-str pointwise))
        stable (conj (set stable) primary-out)
        captures (vec (sort-by pr-str (distinct (concat stable scalars))))
        elements (element-symbols (count arrays))
        capture-parameters (capture-symbols (count captures))
        result (util/subst-syms
                (zipmap captures capture-parameters)
                (first (elementize [body] arrays elements index)))
        algebra (scan/certify {:acc accumulator :init identity :lambda result
                               :out primary-out} dtype)]
    (list '= id [sym]
          (list 'scan {:mode mode :index index :extent extent
                       :attributes {:stable-array-captures (vec (sort-by pr-str stable))}
                       :accumulators [accumulator]
                       :identities [identity]
                       :dtypes [dtype]
                       :algebra [algebra]}
                arrays captures
                (dialect/lambda-form
                 (vec (concat [accumulator] elements capture-parameters))
                 [result])))))

(defn- scalar-dtype
  [{:keys [sym expr]} scalar-dtypes scalar-types]
  (let [expression-tag (when (instance? clojure.lang.IObj expr)
                         (or (:raster.type/tag (meta expr)) (:tag (meta expr))))]
    (or (get scalar-types sym)
        (when (symbol? expr) (get scalar-dtypes expr))
        (dtype/dtype-for-scalar-tag (types/sym-type-tag sym))
        (dtype/dtype-for-scalar-tag expression-tag))))

(defn- scalar-equation
  [description scalar-dtypes scalar-types]
  (let [{:keys [id sym expr]} description
        captures (vec (sort-by pr-str (util/free-syms expr)))
        parameters (capture-symbols (count captures))
        result-dtype (scalar-dtype description scalar-dtypes scalar-types)]
    (when-not result-dtype
      (fail! :unsupported-scalar-binding
             "typed scalar equations require a retained result dtype"
             {:binding id :symbol sym :expression expr}))
    (list '= id [sym]
          (list 'scalar {:dtypes [result-dtype]} captures
                (dialect/lambda-form
                 parameters
                 [(util/subst-syms (zipmap captures parameters) expr)])))))

(defn- terminal-results
  [descriptions body]
  (let [physical-outputs (physical-output-symbols descriptions)
        operations (filter #(contains? #{:map :scatter :reduce :segmented-reduce :scan} (:kind %)) descriptions)
        operation-definitions (set (mapcat #(case (:kind %)
                                              (:map :scatter) (:results %)
                                              :scan [(:sym %)]
                                              :segmented-reduce (:results %)
                                              (:outputs %))
                                           operations))
        terminal-operation-definitions
        (set (mapcat #(case (:kind %)
                        (:map :scatter) (if (:effect-only? %) [] (:results %))
                        :scan [(:sym %)]
                        :segmented-reduce (if (:effect-only? %) [] (:results %))
                        (:outputs %))
                     operations))
        scalar-definitions
        (set (keep #(when (and (= :scalar (:kind %))
                               (not (generated-scaffolding? % physical-outputs)))
                      (:sym %))
                   descriptions))
        all-definitions (set/union operation-definitions scalar-definitions)
        operation-uses (set (concat (mapcat #(concat (:inputs %) (:scalars %)) operations)
                                    (mapcat #(when (= :scalar (:kind %))
                                               (util/free-syms (:expr %)))
                                            descriptions)))
        body-uses (set (mapcat util/free-syms body))]
    (vec (sort-by pr-str
                  (set/union (set/difference terminal-operation-definitions operation-uses)
                             (set/intersection all-definitions body-uses))))))

(defn- selected-scalars
  [descriptions operation-equations outputs]
  (let [physical-outputs (physical-output-symbols descriptions)
        by-symbol (into {}
                        (keep #(when (and (= :scalar (:kind %))
                                          (not (generated-scaffolding? % physical-outputs)))
                                 [(:sym %) %]))
                        descriptions)
        roots (set (concat outputs
                           (mapcat (fn [equation]
                                     (into (dialect/operation-inputs equation)
                                           (filter dialect/value-id?
                                                   (dialect/operation-extents equation))))
                                   operation-equations)))]
    (loop [needed (set/intersection roots (set (keys by-symbol)))]
      (let [dependencies (set (mapcat #(util/free-syms (:expr (get by-symbol %))) needed))
            needed' (set/union needed (set/intersection dependencies (set (keys by-symbol))))]
        (if (= needed needed') needed (recur needed'))))))

(defn- tensor-value [dtype shape]
  (av/tensor {:dtype dtype :shape shape :representation {:kind :plain}}))

(defn- value-dtype [id default-dtype array-types]
  (or (get array-types id)
      (when (symbol? id) (get array-types (symbol (name id))))
      default-dtype :double))

(defn- declared-type
  [types id]
  (or (get types id)
      (when (symbol? id) (get types (clojure.core/symbol (name id))))))

(defn- equation-values
  [equation default-dtype array-types scalar-types known-values]
  (let [[_ _ results] equation
        {:keys [kind attributes arrays captures]} (dialect/operation-parts equation)
        extent (:extent attributes)
        dimension-ids (set (filter dialect/value-id? (dialect/operation-extents equation)))
        stable (set (get-in attributes [:attributes :stable-array-captures]))
        result-dtypes (case kind
                        scalar (:dtypes attributes)
                        reduce (:dtypes attributes)
                        segmented-reduce (:dtypes attributes)
                        scan (:dtypes attributes)
                        (map scatter) (map #(value-dtype % default-dtype array-types) results))]
    (merge
     (into {} (map (fn [id] [id (tensor-value :long [])])) dimension-ids)
     (into {} (map (fn [id] [id (tensor-value (value-dtype id default-dtype array-types)
                                              (dialect/extent-shape extent))]) arrays))
     (if (= 'scalar kind)
       (into {} (keep (fn [id]
                        (when-let [value (or (get known-values id)
                                             (when-let [declared (declared-type scalar-types id)]
                                               (tensor-value declared []))
                                             (when-let [declared (and (symbol? id)
                                                                      (dtype/dtype-for-scalar-tag
                                                                       (types/sym-type-tag id)))]
                                               (tensor-value declared [])))]
                          [id value]))) captures)
       (into {} (map (fn [id]
                       [id (or (when (contains? dimension-ids id)
                                 (tensor-value :long []))
                               (get known-values id)
                               (if (or (contains? stable id)
                                       (contains? array-types id)
                                       (and (symbol? id)
                                            (contains? array-types (symbol (name id)))))
                                 (tensor-value (value-dtype id default-dtype array-types)
                                               [(list 'unknown-dimension id)])
                                 (tensor-value (or (declared-type scalar-types id)
                                                   (value-dtype id default-dtype array-types))
                                               [])))])
                     captures)))
     (into {} (map (fn [id result-dtype]
                     [id (tensor-value (or result-dtype default-dtype :double)
                                       (case kind
                                         (scalar reduce) []
                                         segmented-reduce
                                         (dialect/segmented-reduce-result-shape attributes)
                                         scan (dialect/scan-result-shape attributes)
                                         scatter [(list 'unknown-dimension id)]
                                         (dialect/extent-shape extent)))])
                   results result-dtypes)))))

(defn- merge-value
  [values id contract]
  (if-let [prior (get values id)]
    (let [unknown-shape? (fn [value]
                           (= [(list 'unknown-dimension id)] (:shape value)))
          same-nonshape-contract? (= (dissoc prior :shape) (dissoc contract :shape))]
      (cond
        (= prior contract) values
        (and same-nonshape-contract? (unknown-shape? prior)) (assoc values id contract)
        (and same-nonshape-contract? (unknown-shape? contract)) values
        :else
        (fail! :source-value-conflict
               "source bindings imply incompatible AbstractValues for one logical value"
               {:id id :first prior :second contract})))
    (assoc values id contract)))

(defn form->program
  "Construct and validate TypedSOAC directly from a closed let form.

   Returns nil when any binding is outside the certified source subset. Type/effect/value
   contradictions throw ExceptionInfo because falling through after accepting them would hide a
   compiler correctness defect."
  [source {:keys [dtype array-types scalar-types values]
           :or {dtype :double array-types {} scalar-types {} values {}}}]
  (when (and (seq? source) (contains? #{'let 'let*} (first source)))
    (let [[_ bindings & body] source
          pairs (vec (partition 2 bindings))
          descriptions (normalize-extents (source-descriptions pairs dtype))]
      (when (and (even? (count bindings))
                 (seq descriptions)
                 (supported-descriptions? descriptions))
        (let [operation-descriptions
              (filterv #(contains? #{:map :scatter :reduce :segmented-reduce :scan} (:kind %)) descriptions)
              operation-equations (mapv #(case (:kind %) :map (map-equation %)
                                               :scatter (scatter-equation %)
                                               :reduce (reduce-equation %)
                                               :segmented-reduce (segmented-reduce-equation %)
                                               :scan (scan-equation %))
                                        operation-descriptions)
              outputs (terminal-results descriptions body)
              required-scalars (selected-scalars descriptions operation-equations outputs)
              {:keys [equations equation-descriptions]}
              (reduce (fn [{:keys [scalar-dtypes] :as state} description]
                        (case (:kind description)
                          :scalar
                          (if (contains? required-scalars (:sym description))
                            (let [equation (scalar-equation description scalar-dtypes scalar-types)
                                  result-dtype (first (:dtypes (second (nth equation 3))))]
                              (-> state
                                  (update :equations conj equation)
                                  (update :equation-descriptions conj description)
                                  (assoc-in [:scalar-dtypes (:sym description)] result-dtype)))
                            state)
                          (:map :scatter :reduce :segmented-reduce :scan)
                          (-> state
                              (update :equations conj
                                      (case (:kind description)
                                        :map (map-equation description)
                                        :scatter (scatter-equation description)
                                        :reduce (reduce-equation description)
                                        :segmented-reduce (segmented-reduce-equation description)
                                        :scan (scan-equation description)))
                              (update :equation-descriptions conj description))))
                      {:equations [] :equation-descriptions [] :scalar-dtypes {}}
                      descriptions)
              equation-info (mapv (fn [equation]
                                    {:results (nth equation 2)
                                     :inputs (dialect/operation-inputs equation)
                                     :extents (dialect/operation-extents equation)})
                                  equations)
              definitions (set (mapcat :results equation-info))
              references (set (mapcat #(into (:inputs %)
                                             (filter dialect/value-id? (:extents %)))
                                      equation-info))
              inputs (vec (sort-by pr-str (set/difference references definitions)))
              logical-result-types
              (into {}
                    (mapcat (fn [description]
                              (map (fn [result storage]
                                     [result (value-dtype (:destination storage)
                                                          dtype array-types)])
                                   (:results description)
                                   (:result-storage description))))
                    (filter :result-storage equation-descriptions))
              array-types' (merge array-types logical-result-types)
              destination-values
              (into {}
                    (mapcat (fn [description]
                              (map (fn [{:keys [destination]}]
                                     [destination
                                      (tensor-value
                                       (value-dtype destination dtype array-types)
                                       [(list 'unknown-dimension destination)])])
                                   (:result-storage description))))
                    equation-descriptions)
              inferred-values (reduce (fn [contracts equation]
                                        (reduce-kv merge-value contracts
                                                   (equation-values equation dtype array-types'
                                                                    scalar-types
                                                                    contracts)))
                                      destination-values equations)
              values (reduce-kv merge-value inferred-values values)
              equation-facts
              (into {}
                    (map (fn [description]
                           (let [destination (when (= :scan (:kind description))
                                               (:primary-out description))
                                 storage (:result-storage description)]
                             [(:id description)
                              (cond-> (dialect/default-equation-facts
                                       {:front-end :analyzed-source
                                        :source-binding-id (:id description)})
                                destination
                                (assoc :effects #{:memory/write}
                                       :aliases {(:sym description) destination}
                                       :attributes {:destination destination})
                                storage
                                (assoc :effects #{:memory/write}
                                       :aliases (into {}
                                                      (map (fn [result {:keys [destination]}]
                                                             [result destination])
                                                           (:results description) storage))
                                       :attributes {:result-storage storage
                                                    :host-binding
                                                    (:host-binding description)}))]))
                         equation-descriptions))
              total-effects (reduce set/union #{} (map :effects (vals equation-facts)))
              facts (dialect/default-program-facts
                     {:values values :inputs inputs :equations equation-facts
                      :effects total-effects
                      :provenance {:front-end :analyzed-source}
                      :attributes {:source-dialect :closed-clojure}})]
          (dialect/make facts equations outputs))))))
