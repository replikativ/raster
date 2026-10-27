(ns raster.compiler.core.source-declaration
  "Original local declarations, separate from inferred type and physical completion evidence."
  (:require [raster.compiler.core.types :as types]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.form :as form]))

(def ^:private declaration-key :raster.source/binding-declaration)
(def ^:private absent-declaration {:present? false :hint nil})

(defn declaration [binder]
  (get (meta binder) declaration-key))

(defn reference-declared? [binder]
  (let [{:keys [present? hint]} (declaration binder)]
    (and present? (some? hint)
         (not (contains? types/primitive-info hint))
         (not= Object hint)
         (not (and (class? hint) (.isPrimitive ^Class hint))))))

(defn- capture-binder [binder]
  (cond
    (symbol? binder)
    ;; This operation receives original source, never transformed IR. Caller-supplied/stale
    ;; markers cannot suppress the actual declaration. Rewalk transports markers without capture.
    (vary-meta binder assoc declaration-key
               (if (contains? (meta binder) :tag)
                 {:present? true :hint (:tag (meta binder))}
                 absent-declaration))

    ;; Source destructuring keeps metadata on its eventual local symbols. No name table is
    ;; constructed: two equal names in different scopes retain their own occurrences.
    (coll? binder) (util/postwalk-preserving-meta #(if (symbol? %) (capture-binder %) %) binder)
    :else binder))

(defn- transform-binders [source transform]
  (cond
    ;; The operand has already been walked in its owned source context. Parent
    ;; macro expansion must neither rebuild it nor invent declarations inside it.
    (completion/canonical-form? source) source
    (record? source) source
    (and (seq? source) (contains? '#{quote clojure.core/quote var} (first source))) source
    (seq? source)
    (if-let [{:keys [scopes outer rebuild]} (form/scope-info source)]
      (let [locals? (contains? #{:binding :scope} (:kind (form/form-info source)))
            scopes (mapv (fn [scope]
                           (cond-> (-> scope
                                       (update :inits #(mapv (fn [x] (transform-binders x transform)) %))
                                       (update :body #(mapv (fn [x] (transform-binders x transform)) %)))
                             locals? (update :binders #(mapv transform %))
                             (:aux scope) (update :aux #(mapv (fn [x] (transform-binders x transform)) %)))) scopes)]
        (rebuild scopes (mapv #(transform-binders % transform) outer)))
      (util/remake-if-changed source (mapv #(transform-binders % transform) source)))
    (vector? source) (util/remake-if-changed source (mapv #(transform-binders % transform) source))
    (map? source) (with-meta (into (empty source) (map (fn [[k v]] [(transform-binders k transform) (transform-binders v transform)])) source)
                           (meta source))
    (set? source) (with-meta (into (empty source) (map #(transform-binders % transform)) source) (meta source))
    :else source))

(defn capture-form
  "Capture ONLY at an original source producer, before walking/inference. Never use on rewalked
   IR to infer declarations from generated :tag metadata. Retained raw-source re-entry still has
   its original hints; supplied capture markers are always overwritten. Quoted data and
   structural type syntax remain opaque."
  [source]
  (transform-binders source capture-binder))

(defn capture-expansion
  "Capture declarations introduced by an actual macro expansion, before its fresh locals are
   typed. Exact incoming binder occurrences are transported unchanged: their :tag may already
   be inferred. No name-based comparison, global registry, or rewalk recapture is involved."
  [input expanded]
  (let [incoming (java.util.IdentityHashMap.)
        each-symbol (fn [f binder]
                      (if (symbol? binder) (f binder)
                          (util/postwalk-preserving-meta #(if (symbol? %) (f %) %) binder)))]
    (transform-binders input #(each-symbol (fn [binder] (.put incoming binder true) binder) %))
    (transform-binders expanded
                       #(each-symbol (fn [binder]
                                       (if (.containsKey incoming binder) binder
                                           (capture-binder binder))) %))))

(defn capture-body [body]
  (mapv capture-form body))

(defn retain-hint
  "Restore an original declaration after updating inference metadata. Absence is not a new hint.
   This records no CHECKCAST execution and grants no initializer Class or presence proof."
  [binder]
  (let [{:keys [present? hint]} (declaration binder)]
    (cond-> binder present? (vary-meta assoc :tag hint))))

(defn array-declaration-violation
  "Return the first scoped array declaration incompatibility, or nil.
   :native checks selected value -> declaration; :simd checks declared JVM carrier -> vector
   operand. These are opposite assignability directions. Neither is a normal-return proof."
  [source mode]
  (when-not (contains? #{:native :simd} mode)
    (throw (IllegalArgumentException. "Unknown array declaration constraint")))
  (letfn [(binding-violation [binder]
            (let [{:keys [present? hint]} (declaration binder)
                  selected (:raster.type/tag (meta binder))
                  selected-class (get types/primitive-array-info selected)
                  declared-class (cond (class? hint) hint
                                       (symbol? hint) (types/builtin-array-hint-class hint))]
              (when (and present? (some? hint) selected-class
                         (or (nil? declared-class)
                             (not (case mode
                                    :native (.isAssignableFrom ^Class declared-class selected-class)
                                    :simd (.isAssignableFrom ^Class selected-class declared-class)))))
                {:reason (if declared-class :source-array-declaration-conflict
                                            :source-array-declaration-unsupported)
                 :binding binder :declared hint :selected selected :constraint mode})))
          (visit [source]
            (when (and (coll? source) (not (record? source))
                       (not (completion/canonical-form? source))
                       (not (and (seq? source) (contains? '#{quote clojure.core/quote var} (first source)))))
              (if-let [{:keys [scopes outer]} (form/scope-info source)]
                (or (some (fn [{:keys [binders inits body aux]}]
                            (or (some binding-violation binders)
                                (some visit (concat inits body aux)))) scopes)
                    (some visit outer))
                (some visit source))))]
    (visit source)))

(defn validate-native-array-declarations!
  "Fail closed on incompatible or unresolved array declarations; grant no Class/return proof."
  [source]
  (when-let [violation (array-declaration-violation source :native)]
    (throw (ex-info "native array representation has an unsupported or incompatible source declaration"
                    violation)))
  source)
