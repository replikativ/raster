(ns raster.compiler.passes.scalar.numeric-regions
  "Checked projection of a closed source region onto a primitive value ABI.

   Object completion is discharged only when the ENTIRE region has numeric
   consumers. Identity, escaping products, heap stores, unknown calls and
   unresolved conversions refuse the whole projection. This initial vertical
   admits Double arithmetic; it grants neither reassociation nor fast math."
  (:require [raster.ad.accumulation :as accumulation]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-body]))

(defn project-closed-double-region
  "Return {:form ... :type :double} for a wholly closed Double value region,
   otherwise nil without publishing a partial rewrite. `primitive-ports` names
   actual physical ABI Classes, NOT logical annotations. The result boundary
   must also be an actual primitive double ABI. A collecting compilation owner
   must retain/validate all definition premises before publishing the result.

   Selected calls require an existing publisher-owned emitted entrypoint OR
   the frontend's same-receiver source contract and an independently checked
   source computation. These remain distinct proof kinds. This analysis never
   upgrades a receiver, evaluates a body or refreshes an old artifact.
   Primitive/checkcast completions remain unsupported, not erased.
   Every initializer remains evaluated once in source order, including unused
   numerical initializers. This is not an Object-returning JVM rewrite."
  [source primitive-ports result-carrier & [{:keys [on-decline]}]]
  (when (and method-entry/*emission-inputs* (identical? Double/TYPE result-carrier))
    (let [decline! #(throw (ex-info "Source region has no closed primitive representation"
                                   {:reason ::decline :expression %}))
          double-result (fn [expression] {:expression expression :type :double})
          relist (fn [original values] (with-meta (apply list values) (meta original)))
          observe-core!
          (fn [operation]
            (when (= "clojure.core" (namespace operation))
              (when-let [v (ns-resolve (the-ns 'clojure.core) (symbol (name operation)))]
                (method-entry/observe-definition-input! v (the-ns 'clojure.core) operation))))]
      (try
        (letfn [(check-hint! [id type]
                  (when-let [hint (inference/hint-tag id)]
                    (let [carrier (if (class? hint) hint (types/tag->check-class hint))]
                      (when-not (if (= :double type)
                                  (or (identical? Double carrier) (identical? Double/TYPE carrier))
                                  (or (identical? Boolean carrier) (identical? Boolean/TYPE carrier)))
                        (decline! id)))))

                (primitive-addition [expression arguments env]
                  (let [values (mapv #(visit % env) arguments)]
                    (when-not (every? #(= :double (:type %)) values) (decline! expression))
                    (when-let [addition
                               (accumulation/project-checked-addition
                                expression (mapv (constantly {:present? true :dtype :double}) values)
                                (fn [replacement _]
                                  (let [ports (mapv (fn [_] (gensym "sum_port__")) values)]
                                    (= :double
                                       (:type (scalar-body/checked-primitive-result
                                               (relist replacement (cons (first replacement) ports))
                                               (zipmap ports (repeat :double))))))))]
                      (visit (relist addition (cons (first addition) (map :expression values))) env))))

                (selected-call [expression arguments env]
                  (let [values (mapv #(visit % env) arguments)
                        proof (or (scalar-body/checked-selected-object-result expression)
                                  (scalar-body/checked-selected-source-double-result expression))]
                    (when-not (and proof (= :double (:type proof))
                                   (identical? Double (:object-carrier proof))
                                   (every? #(identical? Double/TYPE %) (:input-carriers proof))
                                   (every? #(= :double (:type %)) values))
                      (decline! expression))
                    ;; Original arguments still evaluate once in their original
                    ;; order. Substitution targets fresh ports, not caller names.
                    (let [ports (mapv (fn [_] (gensym "numeric_argument__")) values)
                          body (util/subst-syms (zipmap (:parameters proof) ports) (:body proof))
                          projected (visit body (into env (map #(vector % {:type :double}) ports)))]
                      (when-not (= :double (:type projected)) (decline! expression))
                      (double-result
                       (list 'let* (vec (mapcat vector ports (map :expression values)))
                             (:expression projected))))))

                (generic-call [expression operation arguments env]
                  (let [_ (when-not (method-entry/observe-generic-dispatch! operation) (decline! expression))
                        values (mapv #(visit % env) arguments)
                        signature (when (every? #(= :double (:type %)) values)
                                    (inference/registered-call-signature
                                     operation (vec (repeat (count values) 'double))))]
                    (when-not (and signature (qualified-symbol? (:mangled-sym signature))
                                   (= (vec (repeat (count values) 'double)) (:tags signature))
                                   (empty? (:promotion-casts signature)))
                      (decline! expression))
                    ;; Reuse the existing dispatch choice and actual selected
                    ;; producer proof; a declared result tag is not a recipe.
                    (let [mangled (:mangled-sym signature)
                          receiver (symbol (namespace mangled) (str (name mangled) "-impl"))
                          call (with-meta (list* '.invk receiver arguments)
                                 {:raster.op/original operation})]
                      (selected-call call arguments env))))

                (core-call [expression operation arguments env]
                  (let [values (mapv #(visit % env) arguments)
                        _ (when-not (every? #(= :double (:type %)) values) (decline! expression))
                        ports (mapv (fn [_] (gensym "scalar_port__")) values)
                        proof (scalar-body/checked-primitive-result
                               (relist expression (cons operation ports))
                               (zipmap ports (repeat :double)))]
                    ;; A scalar width proof is not a proof of Object equality
                    ;; semantics (notably NaN/reference shortcuts) or Clojure
                    ;; truthiness. Predicate-producing calls need their own
                    ;; source/primitive correspondence before admission here.
                    (when-not (= :double (:type proof)) (decline! expression))
                    (observe-core! operation)
                    {:expression (relist expression (cons operation (map :expression values)))
                     :type (:type proof)}))

                (visit [expression env]
                  (cond
                    (symbol? expression)
                    (if-let [fact (get env expression)]
                      (do (check-hint! expression (:type fact))
                          (assoc fact :expression expression))
                      (decline! expression))

                    (instance? Double expression) (double-result expression)
                    (boolean? expression) {:expression expression :type :predicate}

                    (completion/canonical-form? expression)
                    (let [{:keys [contract operand]} (completion/canonical-parts expression)]
                      (when-not (= completion/object-input-contract contract) (decline! expression))
                      (let [result (visit operand env)]
                        (when-not (= :double (:type result)) (decline! expression))
                        result))

                    (and (seq? expression) (contains? #{'let 'let*} (first expression)))
                    (let [[head bindings & body] expression]
                      (when-not (and (vector? bindings) (even? (count bindings))
                                     (every? symbol? (take-nth 2 bindings)) (= 1 (count body)))
                        (decline! expression))
                      (let [[local output]
                            (reduce (fn [[local output] [id init]]
                                      (let [result (visit init local)]
                                        (check-hint! id (:type result))
                                        [(assoc local id (dissoc result :expression))
                                         (conj output id (:expression result))]))
                                    [env []] (partition 2 bindings))
                            result (visit (first body) local)]
                        (assoc result :expression
                               (relist expression [head (with-meta output (meta bindings))
                                                   (:expression result)]))))

                    (and (seq? expression) (= 'if (first expression)) (= 4 (count expression)))
                    (let [condition (visit (second expression) env)]
                      (when-not (= :predicate (:type condition)) (decline! expression))
                      (if (boolean? (:expression condition))
                        (visit (nth expression (if (:expression condition) 2 3)) env)
                        (let [left (visit (nth expression 2) env) right (visit (nth expression 3) env)]
                          (when-not (= (:type left) (:type right)) (decline! expression))
                          (assoc left :expression
                                 (relist expression ['if (:expression condition)
                                                     (:expression left) (:expression right)])))))

                    (seq? expression)
                    (let [{:keys [dispatch? lexical-callee semantic-op implementation-op arguments]}
                          (descriptor/call-description expression)]
                      (when (or lexical-callee (contains? env implementation-op))
                        (decline! expression))
                      (cond
                        dispatch? (selected-call expression arguments env)
                        (and (= 'clojure.core/nil? semantic-op) (= 1 (count arguments))
                             (or (symbol? (first arguments)) (instance? Double (first arguments))))
                        (let [result (visit (first arguments) env)]
                          (when-not (= :double (:type result)) (decline! expression))
                          (observe-core! semantic-op)
                          {:expression false :type :predicate})
                        (accumulation/approved-call? expression)
                        (or (primitive-addition expression arguments env) (decline! expression))
                        (and (symbol? semantic-op)
                             (contains? #{"clojure.core" "java.lang.Math"}
                                        (namespace semantic-op)))
                        (core-call expression semantic-op arguments env)
                        (and (qualified-symbol? implementation-op)
                             (descriptor/scalar-op? implementation-op))
                        (generic-call expression implementation-op arguments env)
                        :else (decline! expression)))
                    :else (decline! expression)))]
          (let [ports (into {} (keep (fn [[id carrier]]
                                      (when (and (symbol? id) (identical? Double/TYPE carrier))
                                        [id {:type :double}]))) primitive-ports)
                result (visit source ports)]
            (when (= :double (:type result)) {:form (:expression result) :type :double})))
        (catch clojure.lang.ExceptionInfo failure
          (if (= ::decline (:reason (ex-data failure)))
            (do (when on-decline (on-decline (ex-data failure))) nil)
            (throw failure)))))))
