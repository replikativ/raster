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

(defn canonical-contract? [contract]
  (and (map? contract)
       (or (= object-input-contract contract)
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
   Native/device consumers must discharge or refuse these source contexts."
  ([expression] (project-to-jvm-source expression nil))
  ([expression compilation-policy]
  (util/postwalk-preserving-meta
   (fn [form]
     (if (and (seq? form) (= completion-symbol (first form)))
       (if-let [{:keys [contract operand]} (canonical-parts form)]
         (with-meta (if (= object-input-contract contract)
                      (list '. (list 'do Object) (list 'cast operand))
                      (project-primitive-input contract operand compilation-policy))
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
   (let [verified (project-to-jvm-source expression compilation-policy)]
     (when (and (some? supplied) (not= supplied verified))
       (throw (ex-info "supplied value completion projection differs"
                       {:reason :value-completion-projection-mismatch
                        :expression expression :supplied supplied :verified verified})))
     verified)))
