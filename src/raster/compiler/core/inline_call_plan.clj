(ns raster.compiler.core.inline-call-plan
  "Unpublished whole-call plans for ordinary sequential source expansion.
   Unknown physical input/body/exit facts retain the original call. No emitter
   is loaded and no source is prepared or evaluated by an optional query."
  (:require [raster.compiler.core.binding-completion :as binding]
            [raster.compiler.core.checked-inputs :as checked]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.source-policy :as policy]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.form :as form]))

(defn- backend [name]
  (some-> (find-ns 'raster.compiler.backend.jvm.bytecode) (ns-resolve name)))

(defn- leaf? [expression]
  (or (symbol? expression) (number? expression) (nil? expression)
      (string? expression) (keyword? expression) (boolean? expression) (char? expression)))

(defn source-context [options]
  (when (and entry/*emission-inputs* (:source-var options))
    (let [functions (into {} (map (fn [name] [name (backend name)]))
                          '[ordinary-evaluation-base source-evaluation-context binding-initializer-context resolved-parameter-locals
                            plan-form-carrier plan-resolved-invocation-inputs builtin-jvm-info binding-hint])]
      (when (every? val functions)
        (when-let [{:keys [parameters invocation-method source-entry] :as physical}
                   (entry/source-var-body-entry-carriers (:source-var options))]
          (when (and (= parameters (or (:public-parameters options) (:active-params options)))
                     (or (nil? (:source-ns options))
                         (identical? (:source-ns options) (:source-ns source-entry)))
                     (entry/source-entry-current? source-entry)
                     (policy/valid? (:source-policy source-entry)))
            (doseq [[name v] functions]
              (entry/observe-definition-input! v (:ns (meta v)) name))
            (when-let [locals ((functions 'resolved-parameter-locals) invocation-method parameters)]
              (let [base ((functions 'ordinary-evaluation-base) (:source-ns source-entry))
                    result ((functions 'builtin-jvm-info) (:return-tag source-entry))]
                (cond-> {:functions functions :physical physical :locals locals
                         :source-options options :policy (:source-policy source-entry)
                         :context ((functions 'binding-initializer-context) base)}
                  result (assoc :body-context ((functions 'source-evaluation-context)
                                               base :method (:stack-type result))))))))))))

(defn- binding-facts [_scope _id _expression raw _locals]
  ;; The actual shared binding producer stores the initializer's raw carrier;
  ;; hints can only narrow its reference Class with a CHECKCAST. Preserve the
  ;; broad Object upper bound, never inspect/initialize an optional hint Class.
  (:local-facts (binding/resolved-binding-completion-plan nil raw nil)))

(defn- raw-leaf [scope expression locals context]
  (when (leaf? expression)
    ((get-in scope [:functions 'plan-form-carrier]) expression locals context)))

(defn- invocation-occurrence-counts [source]
  ;; Syntax inventory only: the eventual inliner visits children outside this
  ;; planner's admitted spine too. An identity shared with ANY such child must
  ;; not borrow a plan from its recognized occurrence. Quoted data is opaque.
  (let [counts (java.util.IdentityHashMap.)]
    (letfn [(visit [node]
              (when-not (and (seq? node) (contains? '#{quote clojure.core/quote} (first node)))
                (when (and (seq? node) (= '.invk (first node)))
                  (.put counts node (inc (long (or (.get counts node) 0)))))
                (when (coll? node)
                  (doseq [child node] (visit child)))))]
      (visit source))
    counts))

(defn- substitution-inputs? [inputs]
  ;; A global typed invocation selects its runtime target AFTER completing
  ;; arguments. Number.xxxValue on an unknown reference can run user code
  ;; which changes that target; substituting the previously selected body
  ;; would freeze it too early. Keep those calls intact. This restriction is
  ;; for substitution only, not for the faithful bytecode input planner.
  (not-any? (fn [{:keys [raw-carrier parameter-class]}]
              (and (= :ref raw-carrier)
                   (.isPrimitive ^Class parameter-class)))
            inputs))

(declare stable-call-prefix?)

(defn call-plans [scope source]
  (let [plans (java.util.IdentityHashMap.)
        occurrences (when scope (invocation-occurrence-counts source))
        context-for (get-in scope [:functions 'source-evaluation-context])]
    (when scope
      (letfn [(call! [expression locals lexical context]
                (when (and context (seq? expression) (= '.invk (first expression)))
                  ;; One AST identity cannot stand for sites with potentially
                  ;; different lexical or evaluation contexts.
                  (when (= 1 (.get occurrences expression))
                      (when (every? leaf? (drop 2 expression))
                        (when-let [{:keys [plan arguments] :as call}
                                   (checked/typed-call-plan expression lexical #{} context)]
                          ;; The existing application constructor publishes one
                          ;; Object result. A primitive caller result requires a
                          ;; separate owned adaptation, never a return-tag guess.
                          (when (and (identical? Object (:interface-result-class plan))
                                     (nil? (:primitive-result-class plan)))
                            (when-let [inputs ((get-in scope [:functions 'plan-resolved-invocation-inputs])
                                              (:method plan) arguments locals context)]
                              (when (substitution-inputs? inputs)
                                (let [planned (assoc call :inputs inputs :caller-policy (:policy scope)
                                                     :caller-context context :candidate source
                                                     :source-options (:source-options scope))]
                                  (.put plans expression planned)
                                  ;; Method/result knowledge does not close the
                                  ;; callee's effects. It may change a later call's
                                  ;; target before that later invocation executes.
                                  (when (stable-call-prefix? scope planned) :ref))))))))))
              (initializer! [expression locals lexical context]
                (or (raw-leaf scope expression locals context)
                    (call! expression locals lexical context)))
              (body! [expressions locals lexical context]
                (loop [remaining (seq expressions)]
                  (if-not remaining
                    true
                    (let [tail? (nil? (next remaining))
                          site (if tail? :body-result :statement)]
                      (when (visit! (first remaining) locals lexical
                                    (context-for context site) tail?)
                        (recur (next remaining)))))))
              (visit! [expression locals lexical context result?]
                (let [head (when (seq? expression) (first expression))
                      sequential? (contains? '#{let let* do} head)
                      context (if (and context result? (:object-result? context))
                                (context-for context (if sequential? :object-region :object-call))
                                context)]
                  (cond
                    (contains? '#{let let*} head)
                    (let [bindings (second expression)]
                      (when (and (vector? bindings) (even? (count bindings)))
                        (loop [pairs (seq (partition 2 bindings)) locals locals lexical lexical]
                          (if-not pairs
                            (when context (body! (drop 2 expression) locals lexical context))
                            (let [[id initializer] (first pairs)
                                  initializer-context (if context (context-for context :initializer)
                                                          (:context scope))]
                              (when (and (symbol? id) (nil? (namespace id))
                                         (not (contains? lexical id)))
                                (when-let [raw (initializer! initializer locals lexical initializer-context)]
                                  (when-let [facts (binding-facts scope id initializer raw locals)]
                                    ;; A resolved Object-return call contributes
                                    ;; only its raw reference slot, never an array
                                    ;; Class, presence, alias or successful return.
                                    (recur (next pairs) (assoc locals id facts) (conj lexical id))))))))))

                    (= 'do head) (when context (body! (rest expression) locals lexical context))
                    (leaf? expression) (when context (raw-leaf scope expression locals context))
                    (= '.invk head) (call! expression locals lexical context)
                    ;; Unknown calls, quotes, branches, loops, nested operands
                    ;; and other scopes do not expose later sibling sites.
                    :else nil)))]
        (visit! source (:locals scope) (set (keys (:locals scope))) (:body-context scope) true)))
    plans))

(defn- body-plan [scope expression locals lexical context]
  (cond
    (form/binding-form? expression)
    (when (= 3 (count expression))
      (loop [pairs (seq (partition 2 (second expression))) locals locals lexical lexical]
        (if-not pairs
          (body-plan scope (nth expression 2) locals lexical context)
          (let [[id init] (first pairs)]
            (when (symbol? id)
              (let [initializer-context ((get-in scope [:functions 'binding-initializer-context]) context)
                    raw (body-plan scope init locals lexical initializer-context)
                    facts (when raw (binding-facts (assoc scope :context initializer-context) id init raw locals))]
                ;; An unknown result does not erase lexical shadowing or become
                ;; a global receiver. Its effects remain in the copied body.
                (recur (next pairs) (cond-> (dissoc locals id) facts (assoc id facts))
                       (conj lexical id))))))))

    (and (seq? expression) (= 'do (first expression)))
    ;; Earlier expressions are copied verbatim and still run under the saved
    ;; body context. Their result carriers cannot alter an independent tail.
    (body-plan scope (last (rest expression)) locals lexical context)

    (and (seq? expression) (= '.invk (first expression))
         (every? leaf? (drop 2 expression)))
    (when-let [{:keys [plan arguments]} (checked/typed-call-plan expression lexical #{} context)]
      (when ((get-in scope [:functions 'plan-resolved-invocation-inputs])
             (:method plan) arguments locals context)
        (let [carrier (or (:primitive-result-class plan) (:interface-result-class plan))
              builtin (get-in scope [:functions 'builtin-jvm-info])]
          ;; Inverse matching of an ACTUAL resolved Class against the shared
          ;; descriptor producer; no source tag or dtype predicts this exit.
          (some (fn [[tag cls]]
                  (when (identical? carrier cls) (:stack-type (builtin tag))))
                types/tag->java-class))))

    :else (raw-leaf scope expression locals context)))

(defn whole-call
  "Close input, body, compute-return, wrapper publication and caller-context edges.
   Builtin equal parameter bridges only; unsupported/unknown cases abstain."
  [scope planned info arguments body]
  (when (and scope planned (:original-frontend info)
             (= (count arguments) (count (:inputs planned)))
             (every? true? (map identical? arguments (map :operand (:inputs planned)))))
    (let [frontend (:original-frontend info)
          source (:source-var info)
          _ (entry/assert-frontend-source! frontend source)
          lineage (:source-entry frontend)
          physical (:physical planned)
          functions (:functions scope)
          builtin (functions 'builtin-jvm-info)
          params (mapv #(with-meta % nil) (:params info))
          parameter-info (mapv builtin (:tags lineage))
          result-info (builtin (:return-tag lineage))
          return-class (:class result-info)
          permitted-return? (or (contains? #{Float/TYPE Double/TYPE Long/TYPE Integer/TYPE Object} return-class)
                                (and return-class (.isArray ^Class return-class)))]
      (when (and (identical? source (get-in physical [:definition :source-var]))
                 (= (:policy scope) (:source-policy lineage))
                 (every? some? parameter-info) result-info permitted-return?
                 (= (mapv :class parameter-info) (:parameter-carriers physical))
                 (identical? Object (:result-carrier physical)))
        (let [context ((functions 'source-evaluation-context)
                       ((functions 'ordinary-evaluation-base) (:source-ns lineage))
                       :method (:stack-type result-info))
              locals ((functions 'resolved-parameter-locals) (:invocation-method physical) params)
              raw (policy/call-with-policy (:source-policy lineage)
                    #(body-plan scope body locals (set params) context))]
          (when (and (contains? #{:ref :float :double :long :int} raw)
                     ;; Actual reference-return roots use emit-object-value.
                     ;; Primitive bodies need that separate result-form plan.
                     (or (not= :ref (:stack-type result-info)) (= :ref raw)))
            {:inputs (:inputs planned) :body-context (completion/bytecode-operand-context context)
             :body-raw raw :return-class return-class :return-stack (:stack-type result-info)}))))))

(defn- closed-prefix-carrier
  "A deliberately small callback-free source domain, not an effect registry.
   Every evaluated prefix must close; global leaves, calls and nonempty lexical
   bindings (whose inferred hints may initialize Classes) abstain. No macro/source execution."
  [scope expression locals context]
  (cond
    (or (and (symbol? expression) (contains? locals expression))
        (and (number? expression) (types/numeric-literal-carrier expression))
        (nil? expression) (string? expression) (keyword? expression)
        (boolean? expression) (char? expression))
    (raw-leaf scope expression locals context)

    (and (seq? expression) (= 'do (first expression)))
    (let [forms (rest expression)]
      (when (seq forms)
        (loop [remaining forms]
          (when-let [raw (closed-prefix-carrier scope (first remaining) locals context)]
            (if (next remaining) (recur (next remaining)) raw)))))

    (and (seq? expression) (contains? '#{let let*} (first expression))
         (= [] (second expression))
         (= 3 (count expression)))
    (closed-prefix-carrier scope (nth expression 2) locals context)
    :else nil))

(defn- stable-call-prefix? [scope planned]
  ;; Re-select the actual original owner; a Method alone never authorizes
  ;; suffix planning. The ordinary whole-call proof also closes parameter and
  ;; return bridges. Unknown code may still inline as the FIRST call, but cannot
  ;; lend stability to later independently selected callees.
  (when-let [source (get-in planned [:physical :definition :source-var])]
    (when-let [frontend (entry/original-source-frontend source)]
      (entry/assert-frontend-source! frontend source)
      (let [lineage (:source-entry frontend)
            parameters (:params lineage)
            forms (:body frontend)
            body (if (= 1 (count forms)) (first forms) (list* 'do forms))
            locals ((get-in scope [:functions 'resolved-parameter-locals])
                    (get-in planned [:physical :invocation-method]) parameters)
            builtin (get-in scope [:functions 'builtin-jvm-info])
            result (builtin (:return-tag lineage))
            context (when result ((get-in scope [:functions 'source-evaluation-context])
                                  ((get-in scope [:functions 'ordinary-evaluation-base]) (:source-ns lineage))
                                  :method (:stack-type result)))]
        (when (and context locals (closed-prefix-carrier scope body locals context))
          (when-let [whole (whole-call scope planned
                                      {:source-var source :original-frontend frontend :params parameters}
                                      (:arguments planned) body)]
            (not (and (= :ref (:body-raw whole))
                      (.isPrimitive ^Class (:return-class whole))))))))))
