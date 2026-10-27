(ns raster.compiler.core.inline-call-plan
  "Unpublished whole-call plans for ordinary flat initializer expansion.
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
                          '[ordinary-evaluation-base binding-initializer-context resolved-parameter-locals
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
              {:functions functions :physical physical :locals locals
               :source-options options :policy (:source-policy source-entry)
               :context ((functions 'binding-initializer-context)
                         ((functions 'ordinary-evaluation-base) (:source-ns source-entry)))})))))))

(defn- binding-facts [_scope _id _expression raw _locals]
  ;; The actual shared binding producer stores the initializer's raw carrier;
  ;; hints can only narrow its reference Class with a CHECKCAST. Preserve the
  ;; broad Object upper bound, never inspect/initialize an optional hint Class.
  (:local-facts (binding/resolved-binding-completion-plan nil raw nil)))

(defn- raw-leaf [scope expression locals context]
  (when (leaf? expression)
    ((get-in scope [:functions 'plan-form-carrier]) expression locals context)))

(defn call-plans [scope source]
  (let [plans (java.util.IdentityHashMap.)]
    (when (and scope (form/binding-form? source) (vector? (second source)))
      (loop [pairs (seq (partition 2 (second source))) locals (:locals scope)]
        (when-let [[id initializer] (first pairs)]
          (when (and (symbol? id) (not (contains? locals id)))
            (if-let [raw (raw-leaf scope initializer locals (:context scope))]
              (when-let [facts (binding-facts scope id initializer raw locals)]
                (recur (next pairs) (assoc locals id facts)))
              ;; Initial domain ends at an unknown initializer. A direct call
              ;; can acquire a plan here, but its unobserved result never seeds
              ;; the rest of the prefix through logical annotations.
              (when (and (seq? initializer) (= '.invk (first initializer))
                         (every? leaf? (drop 2 initializer)))
                (when-let [{:keys [plan physical arguments] :as call}
                           (checked/typed-call-plan initializer (set (keys locals)) #{})]
                  (when-let [inputs ((get-in scope [:functions 'plan-resolved-invocation-inputs])
                                    (:method plan) arguments locals (:context scope))]
                    (.put plans initializer (assoc call :inputs inputs :caller-policy (:policy scope)
                                                   :candidate source :source-options (:source-options scope)))))))))))
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
        (let [context (assoc ((functions 'ordinary-evaluation-base) (:source-ns lineage))
                             :object-result? (= :ref (:stack-type result-info)))
              locals ((functions 'resolved-parameter-locals) (:invocation-method physical) params)
              raw (policy/call-with-policy (:source-policy lineage)
                    #(body-plan scope body locals (set params) context))]
          (when (and (contains? #{:ref :float :double :long :int} raw)
                     ;; Actual reference-return roots use emit-object-value.
                     ;; Primitive bodies need that separate result-form plan.
                     (or (not= :ref (:stack-type result-info)) (= :ref raw)))
            {:inputs (:inputs planned) :body-context (completion/bytecode-operand-context context)
             :body-raw raw :return-class return-class :return-stack (:stack-type result-info)}))))))
