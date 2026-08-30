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
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.par :as par]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.fusion-support :as fusion-support]
            [raster.compiler.passes.scalar.effects :as effects]))

(def ^:private array-constructor-heads
  '#{double-array float-array int-array long-array byte-array
     clojure.core/double-array clojure.core/float-array
     clojure.core/int-array clojure.core/long-array clojure.core/byte-array})

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

(defn- retained-local-dtype
  [binding init]
  (let [init-tag (when (instance? clojure.lang.IObj init)
                   (or (:raster.type/tag (meta init)) (:tag (meta init))))]
    (or (dtype/dtype-for-scalar-tag (types/sym-type-tag binding))
        (dtype/dtype-for-scalar-tag init-tag))))

(defn- pointwise-region
  "Recognize an ordered, pure local-SSA spine ending exclusively in pointwise stores.

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
                   :stores (mapv #(update % :value
                                          (fn [value]
                                            (util/subst-syms substitutions value)))
                                 stores)})))))))

    (and (seq? body) (= 'do (first body)))
    (let [groups (mapv #(pointwise-region % index) (rest body))]
      (when (and (seq groups) (every? some? groups)
                 (every? (comp empty? :locals) groups))
        {:locals [] :stores (vec (mapcat :stores groups))}))

    (descriptor/aset-call? body)
    (let [arguments (vec (descriptor/call-args body))]
      (when (and (= 3 (count arguments))
                 (= index (strip-index-cast (nth arguments 1))))
        (let [value (nth arguments 2)
              cast? (and (seq? value)
                         (contains? #{'float 'double 'int 'long
                                      'clojure.core/float 'clojure.core/double}
                                    (first value))
                         (= 2 (count value)))]
          {:locals []
           :stores [{:out (descriptor/aset-array-sym body)
                     :value (if cast? (second value) value)
                     :cast (when cast? (first value))}]})))

    :else nil))

(defn- expanded-local-expression
  [locals expression]
  (reduce (fn [body {:keys [id init]}]
            (util/subst-syms {id init} body))
          expression (reverse locals)))

(defn- independent-pointwise-stores?
  [locals stores]
  (let [destinations (mapv :out stores)
        destination-set (set destinations)]
    (and (= (count destinations) (count destination-set))
         (every? (fn [{:keys [out value]}]
                   (empty? (disj (set/intersection destination-set
                                                   (par/collect-aget-arrays
                                                    (expanded-local-expression locals value)))
                                 out)))
                 stores))))

(defn- effect-result-id
  [equation-id ordinal]
  [:effect-map equation-id ordinal])

(defn- effect-map-description
  [id symbol index extent {:keys [locals stores]} elem-type]
  (when (and (seq stores) (independent-pointwise-stores? locals stores))
    (let [destinations (mapv :out stores)
          values (mapv :value stores)
          analysis-values (concat (map :init locals) values)
          io (update (extract-io (list* 'do analysis-values) index destinations)
                     :scalars set/difference (set (map :id locals)))
          results (mapv #(effect-result-id id %) (range (count stores)))]
      (merge {:kind :map :id id :sym symbol :index index :extent extent
              :results results :locals locals :bodies values :casts (mapv :cast stores)
              :effect-only? true :host-binding symbol :elem-type elem-type
              :result-storage
              (mapv (fn [destination]
                      {:destination destination
                       :access (if (contains? (:inputs io) destination)
                                 :read-write :write)
                       :host-return :effect})
                    destinations)}
             io))))

(defn- operation-description
  [id symbol expression]
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
    (let [{:keys [idx bound body elem-type]} (par/extract-par-map-void-info expression)]
      (when-let [region (pointwise-region body idx)]
        (effect-map-description id symbol idx bound region elem-type)))

    :else nil))

(defn- provably-pure-scalar?
  [expression]
  (or (effects/removable-expr? expression)
      (and (descriptor/alength-op? (descriptor/semantic-op expression))
           (= 1 (count (descriptor/call-args expression)))
           (symbol? (first (descriptor/call-args expression))))))

(defn- alength-array
  [expression]
  (when (and (seq? expression)
             (descriptor/alength-op? (descriptor/semantic-op expression))
             (= 1 (count (descriptor/call-args expression))))
    (first (descriptor/call-args expression))))

(defn- source-descriptions
  [pairs]
  (mapv (fn [id [symbol expression]]
          (or (operation-description id symbol expression)
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

        (:map :reduce :scan)
        (let [extent (descriptor/unwrap-int-cast (:extent description))
              array (alength-array extent)
              extent' (cond
                        (and array (contains? extents array)) (get extents array)
                        (symbol? extent) (get scalar-representatives extent extent)
                        :else extent)
              description' (assoc description :extent extent')]
          (cond-> (update state :descriptions conj description')
            (contains? #{:map :scan} (:kind description'))
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
                  (contains? array-constructor-heads (first expression)))))))

(defn- supported-descriptions?
  [descriptions]
  (let [physical-outputs (reduce set/union #{}
                                 (map #(if (contains? #{:map :reduce :scan} (:kind %))
                                         (:outputs %) #{})
                                      descriptions))]
    (every? (fn [description]
              (case (:kind description)
                :scalar (or (provably-pure-scalar? (:expr description))
                            (generated-scaffolding? description physical-outputs))
                :map (or (:pure? description)
                         (and (seq (:result-storage description))
                              (every? (comp symbol? :destination)
                                      (:result-storage description))))
                :reduce true
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
  (let [operations (filter #(contains? #{:map :reduce :scan} (:kind %)) descriptions)
        operation-definitions (set (mapcat #(case (:kind %)
                                              :map (:results %)
                                              :scan [(:sym %)]
                                              (:outputs %))
                                           operations))
        terminal-operation-definitions
        (set (mapcat #(case (:kind %)
                        :map (if (:effect-only? %) [] (:results %))
                        :scan [(:sym %)]
                        (:outputs %))
                     operations))
        scalar-definitions (set (keep #(when (= :scalar (:kind %)) (:sym %)) descriptions))
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
  (let [by-symbol (into {} (keep #(when (= :scalar (:kind %)) [(:sym %) %])) descriptions)
        roots (set (concat outputs
                           (mapcat (fn [equation]
                                     (cond-> (dialect/operation-inputs equation)
                                       (dialect/value-id? (dialect/operation-extent equation))
                                       (conj (dialect/operation-extent equation))))
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

(defn- equation-values
  [equation default-dtype array-types known-values]
  (let [[_ _ results] equation
        {:keys [kind attributes arrays captures]} (dialect/operation-parts equation)
        extent (:extent attributes)
        stable (set (get-in attributes [:attributes :stable-array-captures]))
        result-dtypes (case kind
                        scalar (:dtypes attributes)
                        reduce (:dtypes attributes)
                        scan (:dtypes attributes)
                        map (map #(value-dtype % default-dtype array-types) results))]
    (merge
     (if (and extent (dialect/value-id? extent)) {extent (tensor-value :long [])} {})
     (into {} (map (fn [id] [id (tensor-value (value-dtype id default-dtype array-types)
                                              (dialect/extent-shape extent))]) arrays))
     (if (= 'scalar kind)
       (into {} (keep (fn [id]
                        (when-let [value (or (get known-values id)
                                             (when-let [declared (and (symbol? id)
                                                                      (dtype/dtype-for-scalar-tag
                                                                       (types/sym-type-tag id)))]
                                               (tensor-value declared [])))]
                          [id value]))) captures)
       (into {} (map (fn [id]
                       [id (or (get known-values id)
                               (if (or (contains? stable id)
                                       (contains? array-types id)
                                       (and (symbol? id)
                                            (contains? array-types (symbol (name id)))))
                                 (tensor-value (value-dtype id default-dtype array-types)
                                               [(list 'unknown-dimension id)])
                                 (tensor-value (value-dtype id default-dtype array-types) [])))])
                     captures)))
     (into {} (map (fn [id result-dtype]
                     [id (tensor-value (or result-dtype default-dtype :double)
                                       (case kind
                                         (scalar reduce) []
                                         scan (dialect/scan-result-shape attributes)
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
          descriptions (normalize-extents (source-descriptions pairs))]
      (when (and (even? (count bindings))
                 (seq descriptions)
                 (supported-descriptions? descriptions))
        (let [operation-descriptions (filterv #(contains? #{:map :reduce :scan} (:kind %)) descriptions)
              operation-equations (mapv #(case (:kind %) :map (map-equation %)
                                               :reduce (reduce-equation %)
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
                          (:map :reduce :scan)
                          (-> state
                              (update :equations conj
                                      (case (:kind description)
                                        :map (map-equation description)
                                        :reduce (reduce-equation description)
                                        :scan (scan-equation description)))
                              (update :equation-descriptions conj description))))
                      {:equations [] :equation-descriptions [] :scalar-dtypes {}}
                      descriptions)
              equation-info (mapv (fn [equation]
                                    {:results (nth equation 2)
                                     :inputs (dialect/operation-inputs equation)
                                     :extent (dialect/operation-extent equation)})
                                  equations)
              definitions (set (mapcat :results equation-info))
              references (set (mapcat #(cond-> (:inputs %)
                                         (and (:extent %) (dialect/value-id? (:extent %)))
                                         (conj (:extent %))) equation-info))
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
