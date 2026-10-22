(ns raster.compiler.core.value-completion
  "Explicit source evaluation-context completion, separate from numeric dtype.

   Object input completion preserves source boxing/reference identity. An
   explicit primitive input context preserves the original expected-parameter
   conversion under a retained source compilation policy. Neither is inferred
   from a scalar tag or permission to reorder calls or enable an AD rule."
  (:require [raster.compiler.core.util :as util]))

(def completion-symbol
  ;; Reserved IR syntax, deliberately not this namespace's constructor Var.
  ;; Unprojected terms must not run the constructor as if it were a value op.
  'raster.compiler.ir.value-completion/complete-value)

(def object-input-contract
  "An operand originally consumed in JVM Object context. Unknown primitive or
   checked-reference contexts must not be relabeled with this contract."
  {:source-context :jvm-object-input :target-carrier :jvm-object
   :policy :preserve-source-object})

(defn source-policy?
  "The retained Clojure source policy relevant to implicit primitive inputs.
   It is supplied by source compilation, never inferred from ambient state."
  [policy]
  (and (map? policy) (= #{:unchecked-math} (set (keys policy)))
       (contains? #{false true :warn-on-boxed} (:unchecked-math policy))))

(defn- primitive-carrier? [carrier]
  (and (class? carrier) (.isPrimitive ^Class carrier) (not= Void/TYPE carrier)))

(defn reference-check-contract
  "Retain a source CHECKCAST, including its exact Class identity.
   This is NOT a general reference-parameter conversion: source compilation may
   adapt functional interfaces instead of simply checking a reference. The
   producer must prove a CHECKCAST context, not infer one from a semantic tag.
   Nil passes the check; null receiver invocation still fails after arguments."
  [reference-class]
  (when-not (and (class? reference-class) (not (.isPrimitive ^Class reference-class)))
    (throw (ex-info "reference check completion requires a reference Class"
                    {:reason :unsupported-value-completion :carrier reference-class})))
  {:source-context :jvm-checkcast :target-carrier reference-class
   :policy :preserve-source-reference-check})

(defn primitive-input-contract
  "Retain an actual resolved parameter Class and its source compiler policy.
   This is not a tag-to-cast inference API. Call-site and policy correspondence
   remain the producer's obligations. No primitive wrapper identity is promised."
  [parameter-class source-policy]
  (when-not (and (primitive-carrier? parameter-class) (source-policy? source-policy))
    (throw (ex-info "primitive input completion requires a resolved carrier and source policy"
                    {:reason :unsupported-value-completion
                     :carrier parameter-class :source-policy source-policy})))
  {:source-context :jvm-primitive-input :target-carrier parameter-class
   :policy :preserve-source-input :source-policy source-policy})

(def bytecode-operand-context-keys
  "Bytecode result-position facts that affect an input's original evaluation.
   Retained only by a compilation owner, never inferred from scalar tags."
  [:object-result? :object-call-result? :preserve-source-call-results? :void-context])

(defn bytecode-operand-context
  "Capture the existing emitter's result-context facets at an actual input."
  [emitter-context]
  (into {} (map (fn [key] [key (boolean (get emitter-context key))])) bytecode-operand-context-keys))

(defn bytecode-operand-context?
  "Validate the complete saved context shape, not its producer or correspondence."
  [context]
  (and (map? context) (= (set bytecode-operand-context-keys) (set (keys context)))
       (every? boolean? (vals context))))

(defn bytecode-floating-input-contract
  "Retain a floating parameter conversion owned by Raster's bytecode emitter.
   This is NOT Clojure source input conversion: Number.xxxValue and JVM FP
   conversion instructions are interpreted by the original emitter authority.
   The producer still owes exact Method/lexical correspondence and the operand
   context captured from that actual emitter, not a chosen replacement context."
  [parameter-class operand-context]
  (when-not (and (contains? #{Float/TYPE Double/TYPE} parameter-class)
                 (bytecode-operand-context? operand-context))
    (throw (ex-info "bytecode floating completion requires an actual FP parameter Class"
                    {:reason :unsupported-value-completion :carrier parameter-class})))
  {:source-context :jvm-bytecode-floating-input :target-carrier parameter-class
   :policy :preserve-bytecode-input :operand-context operand-context})

(defn bytecode-integral-input-contract
  "Retain an observed Int/Long parameter's bytecode invocation completion.
   This is not a Clojure checked int/long source cast, nor permission to infer a
   parameter Class from a logical dtype. The producer owes the original resolved
   Method and operand context, just as for floating bytecode input completion."
  [parameter-class operand-context]
  (when-not (and (contains? #{Integer/TYPE Long/TYPE} parameter-class)
                 (bytecode-operand-context? operand-context))
    (throw (ex-info "bytecode integral completion requires an actual Int/Long parameter Class"
                    {:reason :unsupported-value-completion :carrier parameter-class})))
  {:source-context :jvm-bytecode-integral-input :target-carrier parameter-class
   :policy :preserve-bytecode-input :operand-context operand-context})

(defn canonical-contract? [contract]
  (and (map? contract)
       (or (= object-input-contract contract)
           (and (= #{:source-context :target-carrier :policy :operand-context} (set (keys contract)))
                (case (:source-context contract)
                  :jvm-bytecode-floating-input
                  (contains? #{Float/TYPE Double/TYPE} (:target-carrier contract))
                  :jvm-bytecode-integral-input
                  (contains? #{Integer/TYPE Long/TYPE} (:target-carrier contract))
                  false)
                (= :preserve-bytecode-input (:policy contract))
                (bytecode-operand-context? (:operand-context contract)))
           (and (= #{:source-context :target-carrier :policy} (set (keys contract)))
                (= :jvm-checkcast (:source-context contract))
                (= :preserve-source-reference-check (:policy contract))
                (class? (:target-carrier contract))
                (not (.isPrimitive ^Class (:target-carrier contract))))
           (and (= #{:source-context :target-carrier :policy :source-policy}
                   (set (keys contract)))
                (= :jvm-primitive-input (:source-context contract))
                (= :preserve-source-input (:policy contract))
                (primitive-carrier? (:target-carrier contract))
                (source-policy? (:source-policy contract))))))

(defn- value-meta [contract expression]
  ;; A new value term cannot inherit the original callee's identity/effects or
  ;; a JVM hint which changes physical compilation. Retain declarations only.
  (select-keys (meta expression)
               (cond-> [:line :column :file]
                 (= object-input-contract contract) (conj :raster.type/tag))))

(defn complete-value
  "Build a checked completion term. The contract must come from the original
   source evaluation context, not a semantic scalar tag. The operand is still
   evaluated once; quote, branch and loop regions remain in place."
  [contract operand]
  (when-not (canonical-contract? contract)
    (throw (ex-info "unsupported value completion contract"
                    {:reason :unsupported-value-completion :contract contract})))
  (with-meta (list completion-symbol contract operand) (value-meta contract operand)))

(defn canonical-form? [expression]
  (and (seq? expression) (= completion-symbol (first expression))
       (= 3 (count expression)) (canonical-contract? (second expression))))

(defn canonical-parts [expression]
  (when (canonical-form? expression)
    {:contract (second expression) :operand (nth expression 2)}))

(defn contains-completion?
  "Whether executable syntax contains a retained input/result context. Quoted
   data is opaque. Include malformed reserved terms so an optimization cannot
   silently erase the validation error they still owe.

   This is a conservative transformation barrier, not a representation proof:
   consumers may discharge one only with their own exact context evidence."
  [expression]
  (boolean
   (some #(and (seq? %) (= completion-symbol (first %)))
         (tree-seq #(and (coll? %)
                         (not (and (seq? %)
                                   (contains? '#{quote clojure.core/quote} (first %)))))
                   seq expression))))

(defn- project-primitive-input [contract operand compilation-policy]
  (when-not (source-policy? compilation-policy)
    (throw (ex-info "primitive completion requires an explicit source compilation policy"
                    {:reason :unclosed-value-completion-policy :contract contract})))
  (when-not (= compilation-policy (:source-policy contract))
    (throw (ex-info "primitive completion source policy differs from compilation policy"
                    {:reason :value-completion-policy-mismatch
                     :required (:source-policy contract) :supplied compilation-policy})))
  ;; :param-tags is a qualified-method feature of the pinned Clojure 1.12+
  ;; source dialect. Never let an unsupported compiler choose an overload from
  ;; the operand and silently lose the requested input context.
  (when-not (and (= 1 (:major *clojure-version*)) (<= 12 (:minor *clojure-version*)))
    (throw (ex-info "primitive input projection requires exact qualified-method parameter tags"
                    {:reason :unsupported-value-completion-source-dialect
                     :clojure-version *clojure-version*})))
  ;; Every primitive has an existing RT.box(T) overload. Exact parameter tags
  ;; make Clojure's own MethodExpr complete the ORIGINAL operand as T, preserving
  ;; primitive-vs-boxed and checked-vs-unchecked semantics without re-inference.
  (list (with-meta 'clojure.lang.RT/box {:param-tags [(:target-carrier contract)]}) operand))

(defn- reference-class-expression [contract reference-bindings]
  ;; Class literals are NOT an identity-preserving constant transport: Clojure
  ;; can reconstruct them by binary name in another loader. The compilation
  ;; owner must capture the actual Class in a private lexical value port.
  (let [port (get reference-bindings (:target-carrier contract))]
    (when-not (and (symbol? port) (nil? (namespace port)))
      (throw (ex-info "reference completion requires an owner-bound Class value"
                      {:reason :unclosed-reference-completion-class :contract contract})))
    (with-meta port {:tag 'java.lang.Class})))

(defn project-to-jvm-source
  "Project validated completion terms to JVM source, retaining other forms.

   Class.cast on bootstrap Object consumes the operand in Object context and
   returns that very object, including nil. The actual Class literal avoids
   lexical class-name lookup; the do receiver prevents host syntax interpreting
   Object as a static-method owner. No helper Var, overload guess or generated
   binder is involved. This is a source projection, not a Class.cast AD template.

   Quoted data is opaque. Malformed reserved terms fail rather than erase a
   conversion. Primitive inputs require explicit matching compilation policy;
   their exact RT.box(T) source context delegates conversion to Clojure rather
   than replacing it with a guessed core cast. The caller MUST compile the
   returned form under that same retained policy and physical lexical scope;
   this pure projection does not execute or police later compilation.
   CHECKCAST contexts additionally require an owner-local map from the actual
   Class to a fresh lexical Class value port. The owner must bind the actual
   Class there and preserve its scope; a Class literal can lose loader identity.
   This map is not persistent metadata or a global registry.
   Native/device consumers must discharge or refuse these source contexts."
  ([expression] (project-to-jvm-source expression nil))
  ([expression compilation-policy]
   (project-to-jvm-source expression compilation-policy {}))
  ([expression compilation-policy reference-bindings]
  (util/postwalk-preserving-meta
   (fn [form]
     (if (and (seq? form) (= completion-symbol (first form)))
       (if-let [{:keys [contract operand]} (canonical-parts form)]
         (with-meta (case (:source-context contract)
                      (:jvm-bytecode-floating-input :jvm-bytecode-integral-input)
                      (throw (ex-info "Bytecode-owned input completion requires its bytecode consumer"
                                      {:reason :unsupported-value-completion-source-context
                                       :contract contract}))
                      :jvm-object-input (list '. (list 'do Object) (list 'cast operand))
                      :jvm-checkcast (list '. (reference-class-expression contract reference-bindings)
                                          (list 'cast operand))
                      :jvm-primitive-input (project-primitive-input contract operand compilation-policy))
           (value-meta contract form))
         (throw (ex-info "invalid canonical value completion"
                         {:reason :invalid-value-completion :expression form})))
       form))
   expression)))

(defn verified-jvm-source-projection
  "Recompute the source projection; do not trust a different supplied form."
  ([expression] (project-to-jvm-source expression))
  ([expression supplied] (verified-jvm-source-projection expression supplied nil))
  ([expression supplied compilation-policy]
   (verified-jvm-source-projection expression supplied compilation-policy {}))
  ([expression supplied compilation-policy reference-bindings]
   (let [verified (project-to-jvm-source expression compilation-policy reference-bindings)]
     (when (and (some? supplied) (not= supplied verified))
       (throw (ex-info "supplied value completion projection differs"
                       {:reason :value-completion-projection-mismatch
                        :expression expression :supplied supplied :verified verified})))
     verified)))
