(ns raster.compiler.passes.scalar.normal-return
  "Conservative normal-return presence summaries, independent of dtype and purity.

   The lattice is :no-return < {:present :absent} < :unknown. A present result
   says nothing about ownership, aliasing, effects, cost or whether evaluation
   throws. Calls and initializers must remain at their original source positions.
   No source code is executed and no operation/function registry is introduced."
  (:require [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.par :as par]
            [raster.compiler.passes.parallel.materialize :as materialize]))

(defn- join [a b]
  (cond (= :no-return a) b
        (= :no-return b) a
        (= a b) a
        :else :unknown))

(declare summary)

(defn- materialized-result [expression env visited]
  (let [key [:materialized-map expression]]
    (if (contains? visited key)
      :unknown
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
          :unknown)))))

(defn- selected-result [expression env visited]
  (let [{:keys [dispatch? semantic-op implementation-op arguments]}
        (descriptor/call-description expression)]
    (if (and dispatch? (not (contains? env implementation-op)))
      (if-let [{:keys [source-var parameters source-body walked-body]}
               (method-entry/selected-definition semantic-op implementation-op (count arguments))]
        (if (contains? visited source-var)
          :unknown
          (let [body (or (seq walked-body) (seq source-body))]
            ;; Formal inputs are deliberately unknown, even if their type tags
            ;; promise an array. A returned input may still be nullable.
            (if (seq body)
              (summary (apply list 'do body) (zipmap parameters (repeat :unknown))
                       (conj visited source-var))
              :unknown)))
        :unknown)
      :unknown)))

(defn- summary [expression env visited]
  (cond
    (nil? expression) :absent
    (symbol? expression) (get env expression :unknown)
    (not (seq? expression)) :present
    (contains? #{'quote 'clojure.core/quote} (first expression))
    (if (nil? (second expression)) :absent :present)
    (= 'throw (first expression)) :no-return
    (= 'do (first expression)) (summary (last (rest expression)) env visited)
    (form/binding-form? expression)
    (let [{:keys [sequential? scopes]} (form/scope-info expression)
          {:keys [binders inits body]} (first scopes)]
      (if (and sequential? (= 1 (count scopes)) (every? symbol? binders))
        (let [local (reduce (fn [local [id init]]
                              (assoc local id (summary init local visited)))
                            env (map vector binders inits))]
          (summary (last body) local visited))
        :unknown))
    (= 'if (first expression))
    (join (summary (nth expression 2 nil) env visited)
          (summary (nth expression 3 nil) env visited))
    :else
    (let [operation (descriptor/semantic-op expression)]
      (cond
        ;; Exact qualified language constructors, not alloc-op?'s compatibility
        ;; lookup or buffer metadata which may only describe a call argument.
        (and (not= '.invk (first expression))
             (symbol? operation) (= "clojure.core" (namespace operation))
             (not (contains? env operation))
             (contains? descriptor/alloc-ops operation)) :present
        ;; Use the existing pure-map materializer to expose its actual
        ;; constructor and returned local; do not duplicate SOAC lowering.
        (and (= 'raster.par/pmap operation) (par/par-map-pure-form? expression))
        (materialized-result expression env visited)
        ;; Only explicit destination identity contracts may forward a fact.
        (and operation (= operation (first expression))
             (some? (:return-alias-arg (form/form-info expression))))
        (let [index (:return-alias-arg (form/form-info expression))]
          (summary (nth (descriptor/call-args expression) index nil) env visited))
        :else (selected-result expression env visited)))))

(defn presence
  "Return :present, :absent, :unknown, or :no-return for a closed source region.
   Unknown is not absence. This function is an analysis, never an evaluator or
   permission to remove/move calls. Input values start unknown; only checked
   producers and lexical aliases establish presence."
  [expression]
  (summary expression {} #{}))
