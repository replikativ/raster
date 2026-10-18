(ns raster.compiler.passes.scalar.normal-return
  "Conservative normal-return presence and physical producer-carrier summaries.

   The lattice is :no-return < {:present :absent} < :unknown. A present result
   says nothing about ownership, aliasing, effects, cost or whether evaluation
   throws. Calls and initializers must remain at their original source positions.
   No source code is executed and no operation/function registry is introduced."
  (:require [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.support.mangled :as mangled]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.par :as par]
            [raster.compiler.passes.parallel.materialize :as materialize]))

(def ^:private unknown {:presence :unknown})

(defn- join [a b]
  (let [pa (:presence a) pb (:presence b)]
    (cond
      (= :no-return pa) b
      (= :no-return pb) a
      :else
      (cond-> {:presence (if (= pa pb) pa :unknown)}
        (and (:array-tag a) (= (:array-tag a) (:array-tag b)))
        (assoc :array-tag (:array-tag a))))))

(declare summary)

(defn- materialized-result [expression env visited]
  (let [key [:materialized-map expression]]
    (if (contains? visited key)
      unknown
      (let [result (try
                     (materialize/materialize-pass expression nil)
                     (catch clojure.lang.ExceptionInfo e
                       ;; Optional analysis may decline exactly the materializer's
                       ;; unsupported domain, never conceal an invariant failure.
                       (if (= :unsupported-pure-map-materialization (:reason (ex-data e)))
                         nil
                         (throw e))))]
        (if (and (contains? result :form) (not= expression (:form result)))
          (summary (:form result) env (conj visited key))
          unknown)))))

(defn- selected-result* [expression env visited]
  (let [{:keys [dispatch? semantic-op implementation-op arguments lexical-callee]}
        (descriptor/call-description expression)
        definition (when (and method-entry/*emission-inputs*
                              (not (or lexical-callee (contains? env implementation-op))))
                     (if dispatch?
                       (method-entry/observed-selected-definition :typed semantic-op implementation-op (count arguments))
                       (or
                        (when-let [operation (mangled/impl->op implementation-op)]
                          (method-entry/observed-selected-definition :direct operation implementation-op
                                                                    (count arguments)))
                        ;; Generic array selection uses actual producer carriers,
                        ;; never desired result/formal annotations or fake values.
                        (when (and (qualified-symbol? implementation-op)
                                   (find-ns (symbol (namespace implementation-op))))
                          (let [tags (mapv #(:array-tag (summary % env visited)) arguments)
                                signature (when (and (seq tags) (every? some? tags))
                                            (inference/registered-call-signature implementation-op tags))]
                            (when (and signature (= tags (:tags signature))
                                       (not (seq (:promotion-casts signature))))
                              (method-entry/observed-selected-definition
                               :direct implementation-op (:mangled-sym signature) (count arguments))))))))]
    (if definition
      (if-let [{:keys [source-var parameters source-body walked-body tags]}
               definition]
        (if (contains? visited source-var)
          unknown
          (let [;; Raw source can prove closed constructors/aliases too, but
                ;; never resolve its operation aliases in the caller's namespace.
                body (or (seq walked-body) (seq source-body))]
            ;; Transport actual producer facts, never presence from formal tags.
            ;; A changed input carrier cannot certify a callee's array result.
            (if (seq body)
              (summary (apply list 'do body)
                       (zipmap parameters
                               (map (fn [argument tag]
                                      (let [fact (summary argument env visited)]
                                        (if (= tag (:array-tag fact)) fact
                                            (dissoc fact :array-tag))))
                                    arguments tags))
                       (conj visited source-var))
              unknown)))
        unknown)
      unknown)))

(defn- selected-result [expression env visited]
  (or (method-entry/call-with-optional-definition-inputs
       #(let [result (selected-result* expression env visited)]
          (when (or (not= :unknown (:presence result)) (:array-tag result)) result)))
      unknown))

(defn- conditional-result [expression env visited]
  (let [[_ test then else] expression
        argument (when (and (seq? test) (= 2 (count test))
                             (= 'clojure.core/nil? (descriptor/semantic-op test))
                             (not (contains? env (first test))))
                   (second test))
        known (when argument (summary argument env visited))]
    (case (:presence known)
      :present (summary else env visited)
      :absent (summary then env visited)
      (let [local? (symbol? argument)
            then-env (if local? (assoc env argument {:presence :absent}) env)
            else-env (if local? (assoc env argument (assoc known :presence :present)) env)]
        (join (summary then then-env visited) (summary else else-env visited))))))

(defn- summary [expression env visited]
  (cond
    (nil? expression) {:presence :absent}
    (symbol? expression) (merge unknown (select-keys (get env expression) [:presence :array-tag]))
    (not (seq? expression)) {:presence :present}
    (contains? #{'quote 'clojure.core/quote} (first expression))
    {:presence (if (nil? (second expression)) :absent :present)}
    (= 'throw (first expression)) {:presence :no-return}
    (= 'do (first expression)) (summary (last (rest expression)) env visited)
    (form/binding-form? expression)
    (let [{:keys [sequential? scopes]} (form/scope-info expression)
          {:keys [binders inits body]} (first scopes)]
      (if (and sequential? (= 1 (count scopes)) (every? symbol? binders))
        (let [local (reduce (fn [local [id init]]
                              (assoc local id (summary init local visited)))
                            env (map vector binders inits))]
          (summary (last body) local visited))
        unknown))
    (= 'if (first expression))
    (conditional-result expression env visited)
    :else
    (let [operation (descriptor/semantic-op expression)]
      (cond
        ;; Exact qualified language constructors, not alloc-op?'s compatibility
        ;; lookup or buffer metadata which may only describe a call argument.
        (and (not= '.invk (first expression))
             (symbol? operation) (= "clojure.core" (namespace operation))
             (not (contains? env operation))
             (contains? descriptor/alloc-ops operation))
        (let [tag (get descriptor/alloc-sym->array-tag operation)]
          (cond-> {:presence :present}
            (contains? types/primitive-array-info tag) (assoc :array-tag tag)))
        ;; Use the existing pure-map materializer to expose its actual
        ;; constructor and returned local; do not duplicate SOAC lowering.
        (and (= 'raster.par/pmap operation) (par/par-map-pure-form? expression))
        (materialized-result expression env visited)
        ;; Only explicit destination identity contracts may forward a fact.
        (and (symbol? operation) (= operation (first expression))
             ;; form-info accepts surface aliases such as par/scan. Such a
             ;; spelling is not a closed semantic operation in raw source.
             (= "raster.par" (namespace operation))
             (some? (:return-alias-arg (form/form-info expression))))
        (let [index (:return-alias-arg (form/form-info expression))]
          (summary (nth (descriptor/call-args expression) index nil) env visited))
        :else (selected-result expression env visited)))))

(defn result-fact
  "Normal-return presence plus an independently derived primitive-array carrier.
   Lexical facts come from actual producers; formal/result annotations cannot
   establish either presence or a physical array carrier. Extra owner-local
   facts are ignored, and no call is executed or moved by this analysis.
   Selected definitions require an enclosing compilation-input owner; otherwise
   definition-dependent facts are unknown rather than silently losing provenance."
  ([expression] (result-fact expression {}))
  ([expression lexical-facts]
   (select-keys (summary expression lexical-facts #{}) [:presence :array-tag])))

(defn presence
  "Return :present, :absent, :unknown, or :no-return for a closed source region.
   Unknown is not absence. This function is an analysis, never an evaluator or
   permission to remove/move calls. Input values start unknown; only checked
   producers and lexical aliases establish presence."
  [expression]
  (:presence (result-fact expression)))
