(ns raster.compiler.core.value-completion
  "Explicit source evaluation-context completion, separate from numeric dtype.

   The first supported contract is an Object input context (for example a
   literal product field). It preserves the exact source Object, including the
   source expression's boxing behavior. It is not an inferred type, numeric
   conversion, ownership/presence fact, AD rule, or permission to reorder calls."
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

(defn canonical-contract? [contract]
  (and (map? contract) (= object-input-contract contract)))

(defn- value-meta [expression]
  ;; A new value term cannot inherit the original callee's identity/effects or
  ;; a JVM hint which changes physical compilation. Retain declarations only.
  (select-keys (meta expression) [:raster.type/tag :line :column :file]))

(defn complete-value
  "Build a checked completion term. The contract must come from the original
   source evaluation context, not a semantic scalar tag. The operand is still
   evaluated once; quote, branch and loop regions remain in place."
  [contract operand]
  (when-not (canonical-contract? contract)
    (throw (ex-info "unsupported value completion contract"
                    {:reason :unsupported-value-completion :contract contract})))
  (with-meta (list completion-symbol contract operand) (value-meta operand)))

(defn canonical-form? [expression]
  (and (seq? expression) (= completion-symbol (first expression))
       (= 3 (count expression)) (canonical-contract? (second expression))))

(defn canonical-parts [expression]
  (when (canonical-form? expression)
    {:contract (second expression) :operand (nth expression 2)}))

(defn project-to-jvm-source
  "Project validated completion terms to JVM source, retaining other forms.

   Class.cast on bootstrap Object consumes the operand in Object context and
   returns that very object, including nil. The actual Class literal avoids
   lexical class-name lookup; the do receiver prevents host syntax interpreting
   Object as a static-method owner. No helper Var, overload guess or generated
   binder is involved. This is a source projection, not a Class.cast AD template.

   Quoted data is opaque. Malformed reserved terms fail rather than erase a
   conversion. Native/device consumers must retain this meaning, discharge its
   observability obligation, or refuse; they cannot strip it by scalar dtype."
  [expression]
  (util/postwalk-preserving-meta
   (fn [form]
     (if (and (seq? form) (= completion-symbol (first form)))
       (if-let [{:keys [operand]} (canonical-parts form)]
         (with-meta (list '. (list 'do Object) (list 'cast operand)) (value-meta form))
         (throw (ex-info "invalid canonical value completion"
                         {:reason :invalid-value-completion :expression form})))
       form))
   expression))

(defn verified-jvm-source-projection
  "Recompute the source projection; do not trust a different supplied form."
  ([expression] (project-to-jvm-source expression))
  ([expression supplied]
   (let [verified (project-to-jvm-source expression)]
     (when (and (some? supplied) (not= supplied verified))
       (throw (ex-info "supplied value completion projection differs"
                       {:reason :value-completion-projection-mismatch
                        :expression expression :supplied supplied :verified verified})))
     verified)))
