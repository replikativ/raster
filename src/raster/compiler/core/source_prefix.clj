(ns raster.compiler.core.source-prefix
  "Producer-local verification of an unchanged evaluated frontend prefix.

   This relates original lexical occurrences to constructed invocation steps;
   it grants no physical carrier, conversion, purity or execution permission.
   In particular a scalar equation's materialization cast is not original source."
  (:require [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.par :as par]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-source :as source]))

(deftype ^:private PrefixInput [frontend parameters scope pairs])
(deftype ^:private PrefixCorrespondence [input program entries])

(defn- parallel-source? [expression]
  (cond
    (and (seq? expression) (#{'quote 'clojure.core/quote} (first expression))) false
    (par/par-form? expression) true
    (coll? expression) (boolean (some parallel-source? expression))
    :else false))

(defn- flat-prefix [expression]
  (when (and (form/binding-form? expression)
             (contains? form/let-heads (first expression))
             (vector? (second expression)) (even? (count (second expression))))
    (let [pairs (mapv vec (partition 2 (second expression)))
          prefix (vec (take-while #(not (parallel-source? (second %))) pairs))]
      (when (and (seq prefix) (every? symbol? (map first pairs))
                 ;; A scalar-only function has no device-prefix boundary here.
                 (some parallel-source? (concat (map second pairs) (nnext expression))))
        prefix))))

(defn begin
  "Retain the exact authenticated frontend BEFORE source normalization.
   Parameters are the original declaration's ordered symbols. Unsupported roots
   abstain; forged or stale frontend ownership still throws. No emission occurs."
  [frontend parameters]
  (when frontend
    (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
    (when (and (= 1 (count (:body frontend))) (vector? parameters)
               (= parameters (get-in frontend [:source-entry :params]))
               (every? symbol? parameters) (= (count parameters) (count (set parameters))))
      (let [scope (first (:body frontend))
            pairs (flat-prefix scope)]
        (when (and pairs
                   (= (count (concat parameters (map first pairs)))
                      (count (set (concat parameters (map first pairs))))))
          ;; Shadowed declarations need a richer occurrence relation, not a
          ;; symbol-map guess that conflates two binders of the same spelling.
          (PrefixInput. frontend parameters scope pairs))))))

(defn- significant-metadata [value]
  (let [metadata (apply dissoc (or (meta value) {})
                        [:line :column :end-line :end-column :file])
        ;; Normalization may elaborate an already retained logical declaration
        ;; into :tag. Accept only that exact elaboration, never a changed width.
        ;; This is logical source correspondence, NOT a physical hint proof.
        logical (:raster.type/tag metadata)]
    (if (and logical (= logical (:tag metadata)))
      (dissoc metadata :tag)
      metadata)))

(defn- source-metadata-compatible? [original transformed declarations]
  (let [left (significant-metadata original)
        right (significant-metadata transformed)
        ;; The existing canonical core allocator owns its array result type.
        ;; Tag elaboration does not stand for a checked cast or a helper's
        ;; inferred result; no other missing source declaration is repaired.
        constructor-tag (when (and (seq? original) (symbol? (first original))
                                   (= "clojure.core" (namespace (first original))))
                          (get descriptor/alloc-sym->array-tag (first original)))]
    (or (= left right)
        ;; Scalar construction carries existing declaration facts onto lexical
        ;; references. Only the declaration visible BEFORE this initializer
        ;; may justify that elaboration; it proves no primitive JVM carrier.
        (and (symbol? original) (get declarations original)
             (not (contains? left :tag)) (not (contains? left :raster.type/tag))
             (= (get declarations original) (:raster.type/tag right))
             (= left (dissoc right :raster.type/tag)))
        (and constructor-tag (not (contains? left :tag))
             (= constructor-tag (:tag right)) (= left (dissoc right :tag))))))

(defn- same-expression? [original transformed renaming declarations]
  (and (source-metadata-compatible? original transformed declarations)
       (cond
         (symbol? original)
         (and (symbol? transformed) (= (get renaming original original) transformed))

         (and (seq? original) (#{'quote 'clojure.core/quote} (first original)))
         ;; Data is not a lexical expression. Refuse copied quoted products
         ;; until their reference identity has its own representation proof.
         (identical? original transformed)

         (seq? original)
         (and (seq? transformed)
              ;; The bounded relation has no nested binding/closure scopes.
              ;; Their capture-avoiding transport must be proved separately.
              (nil? (form/scope-info original))
              (nil? (form/scope-info transformed))
              (= (count original) (count transformed))
              (every? true? (map #(same-expression? %1 %2 renaming declarations) original transformed)))

         (instance? Double original)
         (and (instance? Double transformed)
              (= (Double/doubleToRawLongBits original) (Double/doubleToRawLongBits transformed)))

         (instance? Float original)
         (and (instance? Float transformed)
              (= (Float/floatToRawIntBits original) (Float/floatToRawIntBits transformed)))

         ;; No structural comparison of reference-valued products, executable
         ;; objects, Classes or arbitrary user values can manufacture origins.
         (or (nil? original) (string? original) (keyword? original)
             (boolean? original) (char? original) (integer? original))
         (and (= (class original) (class transformed)) (= original transformed))

         :else (identical? original transformed))))

(defn- prefix-renaming [original transformed parameters parameter-declarations]
  (when (= (count original) (count transformed))
    (let [free (:free
                (reduce (fn [{:keys [bound free]} [id expression]]
                          {:bound (conj bound id)
                           :free (into free (remove bound (util/free-syms expression)))})
                        {:bound (set parameters) :free #{}} original))]
      (loop [left (seq original) right (seq transformed)
             renaming (zipmap parameters parameters) bound (set parameters)
             declarations parameter-declarations]
        (if-let [[id expression] (first left)]
          (let [[result replacement] (first right)]
            (when (and (symbol? result) (nil? (namespace result))
                       (not (contains? bound result))
                       (or (= id result) (not (contains? free result)))
                       (= (significant-metadata id) (significant-metadata result))
                       (same-expression? expression replacement renaming declarations))
              (recur (next left) (next right) (assoc renaming id result) (conj bound result)
                     (if-let [tag (types/sym-type-tag id)]
                       (assoc declarations id tag) (dissoc declarations id)))))
          renaming)))))

(defn close
  "Verify the complete bounded transformation at the equation-first producer.
   Intermediate forms are checked in their actual phase order. The exact final
   program and step objects are retained, not found later by source equality.
   Aggregate projection, inserted/removed/reordered prefix work, nested scopes
   and changed source metadata prevent construction. Existing plan validation
   separately owns logical result-materialization conversions."
  [input normalized represented program]
  (when (and (instance? PrefixInput input)
             (flat-prefix (:source program))
             (nil? (get-in program [:attributes :invocation-plan :attributes :parameter-projection])))
    (let [^PrefixInput input input
          frontend (.-frontend input)
          parameters (.-parameters input)
          parameter-declarations (zipmap parameters (get-in frontend [:source-entry :tags]))
          original (.-pairs input)
          normalized-pairs (flat-prefix normalized)
          represented-pairs (flat-prefix represented)
          normalized-map (prefix-renaming original normalized-pairs parameters parameter-declarations)
          represented-map (prefix-renaming original represented-pairs parameters parameter-declarations)
          plan (get-in program [:attributes :invocation-plan])]
      (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
      (when (and normalized-map represented-map plan
                 (= parameters (mapv :symbol (:parameters plan))))
        (let [order (invocation/source-order plan program)
              steps (into {} (map (juxt :id identity)) (:steps plan))
              final-pairs (mapv vec (partition 2 (second (:source program))))
              declarations (reductions (fn [visible [id]]
                                         (if-let [tag (types/sym-type-tag id)]
                                           (assoc visible id tag) (dissoc visible id)))
                                       parameter-declarations represented-pairs)
              entries
              (mapv
               (fn [ordinal [original-id original-expression] [represented-id expression]]
                 (let [position (nth order ordinal nil)
                       [final-id final-expression] (nth final-pairs ordinal nil)
                       step (get steps (:invocation-step position))
                       equation (some #(when (= (:id %) (:equation position)) %) (:equations program))
                       computation (if (invocation/scalar-compute? step)
                                     (source/scalar-equation-expression equation)
                                     final-expression)]
                   (when (and (= ordinal (:position position))
                              (= represented-id final-id (:binding position))
                              (same-expression? expression computation {} (nth declarations ordinal))
                              ;; The existing plan validator owns the explicit
                              ;; source/result conversion, not this relation.
                              (or (not (invocation/scalar-compute? step))
                                  (and (= represented-id (:symbol step)) equation)))
                     {:ordinal ordinal :scope (.-scope input)
                      :binding original-id :expression original-expression
                      :step step :result-expression final-expression})))
               (range) original represented-pairs)]
          (when (every? some? entries)
            (PrefixCorrespondence. input program entries)))))))

(defn- owned-entries [correspondence frontend program]
  (when (and (instance? PrefixCorrespondence correspondence)
             (identical? program (.-program ^PrefixCorrespondence correspondence)))
    (let [^PrefixInput input (.-input ^PrefixCorrespondence correspondence)]
      (when (identical? frontend (.-frontend input))
        (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
        (.-entries ^PrefixCorrespondence correspondence)))))

(defn original-step
  "Query an exact constructed step under its exact original frontend/program.
   This proves correspondence only. It does not authorize interpreting source,
   manufacturing bytecode contexts or adapting an Object result to a primitive.
   The enclosing owner still owes complete namespace/definition history guards."
  [correspondence frontend program step]
  (when (invocation/scalar-compute? step)
    (let [matches (filterv #(identical? step (:step %))
                           (owned-entries correspondence frontend program))]
      (when (= 1 (count matches)) (first matches)))))

(defn prepare-selected-calls!
  "Explicit equation-first producer preparation, not an occurrence query.
   Prepare original direct registered invk initializers over primitive Long
   public parameters/literals, then their actual caller. Signature observations
   do not import an old emitted-module guard before renewal. Unsupported arguments
   and transformed/foreign prefix ownership abstain without compilation.
   Promotion/renewal invalidates the ORIGINAL owner and must escape to its whole
   compilation retry. This grants no source executor, result unboxing or purity."
  [correspondence frontend program prepare!]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when-let [entries (owned-entries correspondence frontend program)]
       (when entry/*emission-inputs*
         (let [caller (entry/assert-frontend-selection!
                       frontend (get-in frontend [:source-entry :source-var]))
               signature (entry/selected-parameter-carriers
                          (:operation caller) (:implementation-op caller)
                          (count (:parameters caller)))
               long-ports (into #{} (keep (fn [[id carrier]]
                                           (when (= Long/TYPE carrier) id)))
                                (map vector (:parameters signature) (:parameter-carriers signature)))
               callees
               (into []
                     (keep
                      (fn [{:keys [expression step]}]
                        (entry/call-with-optional-definition-inputs
                         (fn []
                           (let [{:keys [dispatch? semantic-op implementation-op arguments lexical-callee]}
                                 (descriptor/call-description expression)]
                             (when (and signature (invocation/scalar-compute? step)
                                        dispatch? (not lexical-callee)
                                        (every? #(or (instance? Long %)
                                                     (and (symbol? %) (contains? long-ports %)))
                                                arguments))
                               (when-let [selected (entry/selected-parameter-carriers
                                                    semantic-op implementation-op (count arguments))]
                                 (let [receiver (get-in selected [:definition :receiver-var])]
                                   (when (and (every? #(= Long/TYPE %) (:parameter-carriers selected))
                                              (= Object (:result-carrier selected))
                                              (not (identical? receiver (:receiver-var caller))))
                                     receiver)))))))))
                     entries)]
           (when (and (:receiver-var caller) (seq callees))
             ;; All selections are authenticated before the first callback.
             ;; The service keeps this order and owns compilation separately.
             (entry/prepare-owned-definition-inputs! prepare! (conj callees (:receiver-var caller)))
             true)))))))
