(ns raster.compiler.core.dispatch
  "Dispatch engine for the raster typed multiple dispatch system.

  Contains:
  - MethodEntry record and dispatch table management
  - Subtype registry
  - Auto-specialization (JIT-style lazy specialization for Object[] args)
  - Generic function management
  - register-method! — the core registration API

  Uses :raster.core/ qualified keywords for metadata to maintain
  compatibility with bytecode.clj and tooling/inspect.clj."
  (:require [clojure.string :as str]
            [clojure.core.typed :as t]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.method-entry]
            [raster.compiler.core.specialize :as specialize]))

;; ================================================================
;; Method Entry & Dispatch Table
;; ================================================================

(defn- tag-specificity
  "Higher = more specific type. Object = 0."
  [tag]
  (let [cls (types/tag->check-class tag)
        base (if (= cls Object) 0 (inc (count (supers cls))))]
    (if (types/compound-tag? tag)
      (+ base (count (types/compound-tag-params tag)))
      base)))

(defn make-method-entry
  "Create a MethodEntry from tags and the specialized function."
  ([tags target-fn]
   (make-method-entry tags target-fn nil nil nil nil nil))
  ([tags target-fn typed-impl typed-iface]
   (make-method-entry tags target-fn typed-impl typed-iface nil nil nil))
  ([tags target-fn typed-impl typed-iface typed-target-fn]
   (make-method-entry tags target-fn typed-impl typed-iface typed-target-fn nil nil))
  ([tags target-fn typed-impl typed-iface typed-target-fn warning-meta]
   (make-method-entry tags target-fn typed-impl typed-iface typed-target-fn warning-meta nil))
  ([tags target-fn typed-impl typed-iface typed-target-fn warning-meta mangled-ns]
   {:tags tags
    :arity (count tags)
    :check-classes (mapv types/tag->check-class tags)
    :target-fn target-fn
    :specificity (reduce + (map tag-specificity tags))
    :typed-impl typed-impl
    :typed-iface typed-iface
    :typed-target-fn typed-target-fn
    :warning-meta warning-meta
    :has-compound-tags (boolean (some types/compound-tag? tags))
    :mangled-ns mangled-ns}))

;; ================================================================
;; Subtype Registry
;; ================================================================

(defonce subtype-registry (atom {}))

(defn register-subtype!
  "Register an external class as a subtype of an abstract type."
  [child-class parent-abstract]
  (swap! subtype-registry update child-class (fnil conj #{}) parent-abstract))

(def numeric-widening
  "JVM numeric widening conversions. Canonical copy in types.clj."
  types/numeric-widening)

(defn check-type
  "Check if arg matches expected-cls, considering instanceof, widening,
  and subtype registry."
  [^Class expected-cls arg]
  (cond
    (nil? arg) (not (.isPrimitive expected-cls))
    (instance? expected-cls arg) true
    :else
    (or (when-let [wider (get numeric-widening (class arg))]
          (contains? wider expected-cls))
        (and (contains? @types/fn-interface-by-name (symbol (.getName expected-cls)))
             (instance? clojure.lang.IFn arg))
        (some (fn [e]
                (let [^Class registered-child (key e)]
                  (and (instance? registered-child arg)
                       (some #(.isAssignableFrom expected-cls ^Class %) (val e)))))
              @subtype-registry))))

(defn assignable-from?
  "Like .isAssignableFrom but also considers the subtype registry."
  [^Class parent ^Class child]
  (or (.isAssignableFrom parent child)
      (some (fn [e]
              (let [^Class registered-child (key e)]
                (and (.isAssignableFrom registered-child child)
                     (some #(.isAssignableFrom parent ^Class %) (val e)))))
            @subtype-registry)))

(defn- shape-matches?
  "Check if arg's runtime shape matches compound tag parameters."
  [tag arg]
  (if-let [params (types/compound-tag-params tag)]
    (and (satisfies? types/Shaped arg)
         (= (vec params) (types/shape arg)))
    true))

(defn- entry-matches?
  "Does this method entry match the given argument types?"
  [entry args]
  (let [classes (:check-classes entry)
        n (:arity entry)]
    (when (= n (count args))
      (if (:has-compound-tags entry)
        (let [tags (:tags entry)]
          (loop [i 0]
            (if (>= i n)
              true
              (let [arg (nth args i)]
                (if (and (check-type (nth classes i) arg)
                         (shape-matches? (nth tags i) arg))
                  (recur (inc i))
                  false)))))
        (loop [i 0]
          (if (>= i n)
            true
            (if (check-type (nth classes i) (nth args i))
              (recur (inc i))
              false)))))))

(defn- invoke-entry
  "Invoke a matched method entry with the given args."
  [entry a0 a1 a2 a3 args n]
  (let [f (or (:typed-target-fn entry) (:target-fn entry))]
    (case n
      0 (f)
      1 (f a0)
      2 (f a0 a1)
      3 (f a0 a1 a2)
      4 (f a0 a1 a2 a3)
      5 (f a0 a1 a2 a3 (nth args 4))
      6 (f a0 a1 a2 a3 (nth args 4) (nth args 5))
      7 (f a0 a1 a2 a3 (nth args 4) (nth args 5) (nth args 6))
      8 (f a0 a1 a2 a3 (nth args 4) (nth args 5) (nth args 6) (nth args 7))
      (apply f args))))

(defn- format-type-name
  "Format a class name for display."
  [^Class cls]
  (if (nil? cls) "nil"
      (let [n (.getName cls)]
        (case n
          "java.lang.Long" "Long"
          "java.lang.Double" "Double"
          "java.lang.Float" "Float"
          "java.lang.Integer" "Integer"
          "java.lang.String" "String"
          "java.lang.Boolean" "Boolean"
          "java.lang.Object" "Object"
          "[D" "(Array double)"
          "[F" "(Array float)"
          "[J" "(Array long)"
          "[I" "(Array int)"
          "[Z" "(Array boolean)"
          "[Ljava.lang.Object;" "(Array Object)"
          (let [simple (.getSimpleName cls)]
            (if (str/blank? simple) n simple))))))

(defn- method-matches-count
  [entry arg-types]
  (let [checks (:check-classes entry)
        n (min (count arg-types) (count checks))]
    (loop [i 0, matches 0]
      (if (>= i n)
        matches
        (let [expected (nth checks i)
              actual (nth arg-types i)]
          (recur (inc i)
                 (if (and actual expected
                          (or (.isAssignableFrom ^Class expected actual)
                              (some (fn [e]
                                      (let [^Class rc (key e)]
                                        (and (.isAssignableFrom rc actual)
                                             (some #(.isAssignableFrom ^Class expected ^Class %) (val e)))))
                                    @subtype-registry)))
                   (inc matches)
                   matches)))))))

(defn- throw-no-method [fn-name args dispatch-table]
  (let [arg-types (mapv class args)
        arity (count args)
        all-methods (when dispatch-table
                      (for [[a ms] @dispatch-table, m ms] [a m]))
        same-arity (when dispatch-table (get @dispatch-table arity))
        closest (when same-arity
                  (->> same-arity
                       (map (fn [m] {:entry m
                                     :matches (method-matches-count m arg-types)
                                     :tags (:tags m)}))
                       (sort-by :matches >)
                       (take 3)))
        sb (StringBuilder.)
        _ (.append sb (str "No matching method for `" fn-name
                           "` with argument types ["
                           (str/join ", " (map format-type-name arg-types))
                           "].\n"))
        _ (when (seq all-methods)
            (.append sb "\nAvailable methods:\n")
            (doseq [[a m] (sort-by first all-methods)]
              (.append sb (str "  (" fn-name " "
                               (str/join " " (map name (:tags m)))
                               ")  [arity=" a ", specificity=" (:specificity m) "]\n"))))
        _ (when (and (seq closest) (pos? (:matches (first closest))))
            (.append sb "\nClosest match:\n")
            (let [{:keys [entry matches]} (first closest)
                  checks (:check-classes entry)]
              (.append sb (str "  (" fn-name " "
                               (str/join " " (map name (:tags entry))) ")\n"))
              (dotimes [i arity]
                (let [expected (nth checks i)
                      actual (nth arg-types i)
                      ok? (and actual expected
                               (.isAssignableFrom ^Class expected actual))]
                  (.append sb (str "    Arg " (inc i) ": "
                                   (format-type-name actual)
                                   (if ok? " OK"
                                       (str " -> needs " (format-type-name expected)))
                                   "\n"))))))]
    (throw (ex-info (str sb)
                    {:name fn-name :types arg-types :args args
                     :available-methods (when same-arity (mapv :tags same-arity))}))))

(def ^:dynamic *trace-dispatch*
  "When true, dispatch calls print tracing info."
  false)

;; ================================================================
;; Auto-specialization
;; ================================================================

(defonce ^:private specialization-cache (atom {}))

(def ^:dynamic *auto-specialize*
  "When true (default), deftm dispatch auto-specializes methods with Object[]."
  true)

(defn- get-specialization-cache [fn-key]
  (or (get @specialization-cache fn-key)
      (do (swap! specialization-cache
                 (fn [m] (if (contains? m fn-key) m (assoc m fn-key (atom {})))))
          (get @specialization-cache fn-key))))

(defn clear-specialization-cache! [fn-key]
  (when-let [cache-atom (get @specialization-cache fn-key)]
    (reset! cache-atom {})))

(defn- has-objects-tag? [entry]
  (some #{'objects} (:tags entry)))

(def ^:private object-array-class (class (object-array 0)))

(defn- detect-element-type
  [entry a0 a1 a2 a3 args n]
  (let [tags (:tags entry)]
    (loop [i 0]
      (when (< i (count tags))
        (if (= 'objects (nth tags i))
          ;; Only a genuine Object[] arg carries a per-element class to auto-
          ;; specialize on (value classes like Dual). A primitive array
          ;; (float[]/double[]) reaching an 'objects slot has a fixed element
          ;; type handled by the parametric path — skip it here rather than CCE
          ;; on the forced Object[] cast.
          (let [arg (case (int i) 0 a0 1 a1 2 a2 3 a3 (nth args i))]
            (when (instance? object-array-class arg)
              (let [^objects arr arg]
                (when (pos? (alength arr))
                  (let [elem (aget arr 0)]
                    (when (and elem (not (.isArray (class elem))))
                      (class elem)))))))
          (recur (inc i)))))))

(defn- try-auto-specialize!
  "Attempt async specialization. Resolves specialize-fn! dynamically."
  [dispatch-var elem-class cache-atom]
  (let [cached (get @cache-atom elem-class)]
    (cond
      (and cached (:fn cached)) (:fn cached)
      (and cached (:pending? cached)) nil
      :else
      (let [prev (get (swap! cache-atom
                             (fn [m]
                               (if (contains? m elem-class) m
                                   (assoc m elem-class {:fn nil :pending? true}))))
                      elem-class)]
        (when (and (:pending? prev) (nil? (:fn prev)))
          (let [elem-sym (symbol (.getName ^Class elem-class))
                var-ns (the-ns (:ns (meta dispatch-var)))
                var-name (:name (meta dispatch-var))
                elem-short (let [s (name elem-sym)]
                             (symbol (last (str/split s #"\."))))]
            (future
              (try
                (let [specialize! specialize/specialize-fn!]
                  (when specialize!
                    (binding [*ns* var-ns]
                      (let [spec-sym (specialize! dispatch-var {:element-type elem-short :ns-obj var-ns})
                            spec-var (ns-resolve var-ns spec-sym)
                            spec-fn (when spec-var (deref spec-var))]
                        (swap! cache-atom assoc elem-class {:fn spec-fn :pending? false})))))
                (catch Exception e
                  (swap! cache-atom assoc elem-class {:fn nil :pending? false})
                  (binding [*out* *err*]
                    (println (str "Auto-specialization failed for " var-name
                                  " [" elem-sym "]: " (.getMessage e)))))))))
        nil))))

(declare try-parametric-dispatch)

(defn- dispatch-arity
  [fn-name methods a0 a1 a2 a3 args n dispatch-table dispatch-var]
  (loop [ms (seq methods)]
    (if-not ms
      ;; No concrete match — try parametric specialization before throwing
      (let [full-args (case n 0 [] 1 [a0] 2 [a0 a1] 3 [a0 a1 a2]
                            4 [a0 a1 a2 a3] (vec args))
            qname (when dispatch-var
                    (let [m (meta dispatch-var)]
                      (symbol (str (ns-name (:ns m))) (str (:name m)))))]
        (if-let [boxed (when qname
                         (try-parametric-dispatch qname full-args))]
          (::result boxed)
          (throw-no-method fn-name full-args dispatch-table)))
      (let [m (first ms)]
        (if (case n
              1 (and (check-type (nth (:check-classes m) 0) a0)
                     (or (not (:has-compound-tags m))
                         (shape-matches? (nth (:tags m) 0) a0)))
              2 (and (check-type (nth (:check-classes m) 0) a0)
                     (check-type (nth (:check-classes m) 1) a1)
                     (or (not (:has-compound-tags m))
                         (and (shape-matches? (nth (:tags m) 0) a0)
                              (shape-matches? (nth (:tags m) 1) a1))))
              3 (and (check-type (nth (:check-classes m) 0) a0)
                     (check-type (nth (:check-classes m) 1) a1)
                     (check-type (nth (:check-classes m) 2) a2)
                     (or (not (:has-compound-tags m))
                         (and (shape-matches? (nth (:tags m) 0) a0)
                              (shape-matches? (nth (:tags m) 1) a1)
                              (shape-matches? (nth (:tags m) 2) a2))))
              (entry-matches? m (case n 0 [] 4 [a0 a1 a2 a3] args)))
          (do
            (when *trace-dispatch*
              (let [arg-types (case n 0 [] 1 [(type a0)] 2 [(type a0) (type a1)]
                                    3 [(type a0) (type a1) (type a2)]
                                    (mapv type (case n 4 [a0 a1 a2 a3] args)))]
                (println (str "DISPATCH " fn-name " "
                              (mapv format-type-name arg-types)
                              " -> " (str/join " " (map name (:tags m)))
                              " [specificity=" (:specificity m) "]"))))
            (cond
              ;; Auto-specialize for array element types
              (and *auto-specialize* dispatch-var (has-objects-tag? m))
              (if-let [elem-class (detect-element-type m a0 a1 a2 a3 args n)]
                (let [fn-key (let [vm (meta dispatch-var)]
                               (symbol (str (ns-name (:ns vm))) (str (:name vm))))
                      cache-atom (get-specialization-cache fn-key)
                      cached (get @cache-atom elem-class)]
                  (if (and cached (:fn cached))
                    (let [f (:fn cached)]
                      (case n 0 (f) 1 (f a0) 2 (f a0 a1) 3 (f a0 a1 a2)
                            4 (f a0 a1 a2 a3) (apply f args)))
                    (do (try-auto-specialize! dispatch-var elem-class cache-atom)
                        (invoke-entry m a0 a1 a2 a3 args n))))
                (invoke-entry m a0 a1 a2 a3 args n))

              ;; Try parametric dispatch before generic Object catch-all.
              ;; When the only match is [Object, Object, ...], a parametric template
              ;; (e.g., [Dual, Dual] with All [T]) may provide a more specific method.
              (and dispatch-var (every? #(= 'Object %) (:tags m)))
              (let [full-args (case n 0 [] 1 [a0] 2 [a0 a1] 3 [a0 a1 a2]
                                    4 [a0 a1 a2 a3] (vec args))
                    qname (let [mv (meta dispatch-var)]
                            (symbol (str (ns-name (:ns mv))) (str (:name mv))))]
                (if-let [boxed (try-parametric-dispatch qname full-args)]
                  (::result boxed)
                  (invoke-entry m a0 a1 a2 a3 args n)))

              :else
              (invoke-entry m a0 a1 a2 a3 args n)))
          (recur (next ms)))))))

;; ================================================================
;; Generic Function Management
;; ================================================================

(defonce dispatch-tables (atom {}))

(defn- gf-key [ns-sym fn-name]
  (symbol (str ns-sym) (str fn-name)))

(defn- get-or-create-table! [k]
  (or (get @dispatch-tables k)
      (do (swap! dispatch-tables
                 (fn [m] (if (contains? m k) m (assoc m k (atom {})))))
          (get @dispatch-tables k))))

(defn- strictly-more-specific?
  [a b]
  (let [ca (:check-classes a) cb (:check-classes b) n (:arity a)]
    (loop [i 0 all-assignable? true any-narrower? false]
      (if (>= i n)
        (and all-assignable? any-narrower?)
        (let [ai (nth ca i) bi (nth cb i)
              assignable (assignable-from? bi ai)
              narrower (and assignable (not= ai bi))]
          (recur (inc i)
                 (and all-assignable? assignable)
                 (or any-narrower? narrower)))))))

(defn- types-overlap? [^Class a ^Class b]
  (or (assignable-from? a b)
      (assignable-from? b a)
      (and (.isInterface a) (.isInterface b))))

(defn methods-ambiguous?
  [a b]
  (let [ca (:check-classes a) cb (:check-classes b) n (:arity a)]
    (when (= n (:arity b))
      (and (every? true? (map types-overlap? ca cb))
           (not (strictly-more-specific? a b))
           (not (strictly-more-specific? b a))))))

(defn- ifn-interface-tag?
  "True if tag names an auto-generated IFn__ typed dispatch interface.
   These are internal — ambiguity with user types is expected and harmless."
  [tag]
  (and (symbol? tag)
       (let [n (name tag)]
         (or (.startsWith n "IFn__")
             (.contains n ".IFn__")))))

(defn- check-ambiguities! [fn-name methods new-entry]
  (doseq [existing methods]
    (when (and (not= (:tags existing) (:tags new-entry))
               (methods-ambiguous? existing new-entry)
               ;; Suppress for auto-generated IFn__ interfaces
               (not (some ifn-interface-tag? (:tags existing)))
               (not (some ifn-interface-tag? (:tags new-entry)))
               (not (types/warning-suppressed? (:warning-meta existing) :ambiguous-methods))
               (not (types/warning-suppressed? (:warning-meta new-entry) :ambiguous-methods)))
      (types/emit-warning! (:warning-meta new-entry) :ambiguous-methods
                           (str "WARNING: Ambiguous methods for `" fn-name "`:\n"
                                "  " (:tags existing) " [specificity=" (:specificity existing) "]\n"
                                "  " (:tags new-entry) " [specificity=" (:specificity new-entry) "]\n"
                                "  Define a more specific method to resolve.")))))

(defn- add-method!
  ([fn-name table-atom tags target-fn]
   (add-method! fn-name table-atom tags target-fn nil nil nil nil nil))
  ([fn-name table-atom tags target-fn typed-impl typed-iface]
   (add-method! fn-name table-atom tags target-fn typed-impl typed-iface nil nil nil))
  ([fn-name table-atom tags target-fn typed-impl typed-iface typed-target-fn]
   (add-method! fn-name table-atom tags target-fn typed-impl typed-iface typed-target-fn nil nil))
  ([fn-name table-atom tags target-fn typed-impl typed-iface typed-target-fn warning-meta]
   (add-method! fn-name table-atom tags target-fn typed-impl typed-iface typed-target-fn warning-meta nil))
  ([fn-name table-atom tags target-fn typed-impl typed-iface typed-target-fn warning-meta mangled-ns]
   (let [entry (make-method-entry tags target-fn typed-impl typed-iface typed-target-fn warning-meta mangled-ns)
         arity (count tags)]
     (swap! table-atom update arity
            (fn [methods]
              (let [without (vec (remove #(= (:tags %) tags) (or methods [])))
                    updated (vec (sort (fn [a b]
                                         (let [sa (:specificity a) sb (:specificity b)]
                                           (cond
                                             (> sa sb) -1
                                             (< sa sb) 1
                                             (strictly-more-specific? a b) -1
                                             (strictly-more-specific? b a) 1
                                             :else 0)))
                                       (conj without entry)))]
                (check-ambiguities! fn-name without entry)
                updated))))))

(defn make-dispatch-fn
  "Create a dispatch function that reads from the dispatch table atom."
  ([fn-name table-atom reducible?]
   (make-dispatch-fn fn-name table-atom reducible? nil))
  ([fn-name table-atom reducible? target-ns]
   (let [dispatch-var-ref (delay (when target-ns (ns-resolve target-ns fn-name)))
         dispatch
         (fn raster-dispatch
           ([]
            (let [t @table-atom]
              (if-let [ms (get t 0)]
                (invoke-entry (first ms) nil nil nil nil nil 0)
                (throw-no-method fn-name [] table-atom))))
           ([a0]
            (let [ms (get @table-atom 1)]
              (if ms
                (dispatch-arity fn-name ms a0 nil nil nil nil 1 table-atom @dispatch-var-ref)
                (if-let [boxed (when @dispatch-var-ref
                                 (try-parametric-dispatch
                                  (let [m (meta @dispatch-var-ref)]
                                    (symbol (str (ns-name (:ns m))) (str (:name m))))
                                  [a0]))]
                  (::result boxed)
                  (throw-no-method fn-name [a0] table-atom)))))
           ([a0 a1]
            (let [ms (get @table-atom 2)]
              (if ms
                (dispatch-arity fn-name ms a0 a1 nil nil nil 2 table-atom @dispatch-var-ref)
                (if-let [boxed (when @dispatch-var-ref
                                 (try-parametric-dispatch
                                  (let [m (meta @dispatch-var-ref)]
                                    (symbol (str (ns-name (:ns m))) (str (:name m))))
                                  [a0 a1]))]
                  (::result boxed)
                  (throw-no-method fn-name [a0 a1] table-atom)))))
           ([a0 a1 a2]
            (let [t @table-atom]
              (if-let [ms (get t 3)]
                (dispatch-arity fn-name ms a0 a1 a2 nil nil 3 table-atom @dispatch-var-ref)
                (if (and reducible? (get t 2))
                  (raster-dispatch (raster-dispatch a0 a1) a2)
                  (if-let [boxed (when @dispatch-var-ref
                                   (try-parametric-dispatch
                                    (let [m (meta @dispatch-var-ref)]
                                      (symbol (str (ns-name (:ns m))) (str (:name m))))
                                    [a0 a1 a2]))]
                    (::result boxed)
                    (throw-no-method fn-name [a0 a1 a2] table-atom))))))
           ([a0 a1 a2 a3]
            (let [t @table-atom]
              (if-let [ms (get t 4)]
                (dispatch-arity fn-name ms a0 a1 a2 a3 [a0 a1 a2 a3] 4 table-atom @dispatch-var-ref)
                (if (and reducible? (get t 2))
                  (raster-dispatch (raster-dispatch (raster-dispatch a0 a1) a2) a3)
                  (if-let [boxed (when @dispatch-var-ref
                                   (try-parametric-dispatch
                                    (let [m (meta @dispatch-var-ref)]
                                      (symbol (str (ns-name (:ns m))) (str (:name m))))
                                    [a0 a1 a2 a3]))]
                    (::result boxed)
                    (throw-no-method fn-name [a0 a1 a2 a3] table-atom))))))
           ([a0 a1 a2 a3 & more]
            (let [args (into [a0 a1 a2 a3] more)
                  n (count args)
                  t @table-atom]
              (if-let [ms (get t n)]
                (dispatch-arity fn-name ms a0 a1 a2 a3 args n table-atom @dispatch-var-ref)
                (if (and reducible? (get t 2))
                  (reduce raster-dispatch args)
                  (if-let [boxed (when @dispatch-var-ref
                                   (try-parametric-dispatch
                                    (let [m (meta @dispatch-var-ref)]
                                      (symbol (str (ns-name (:ns m))) (str (:name m))))
                                    args))]
                    (::result boxed)
                    (throw-no-method fn-name args table-atom)))))))]
     dispatch)))

;; ================================================================
;; TC annotation helpers
;; ================================================================

(def primitive-tc-ann
  "Static map for primitive and core types → TC type forms."
  {'long 'Long, 'double 'Double, 'int 'Integer, 'float 'Float,
   'boolean 'Boolean, 'byte 'Byte, 'short 'Short, 'char 'Character,
   'String 'String, 'Number 'Number, 'Object 'clojure.core.typed/Any,
   'doubles '(Array double), 'longs '(Array long), 'ints '(Array int),
   'floats '(Array float), 'bytes '(Array byte), 'shorts '(Array short),
   'chars '(Array char), 'booleans '(Array boolean), 'objects '(Array Object)})

(defn- tag->tc-ann
  "Resolve a dispatch tag to a TC type form.
   Primitives use the static map. IFn__ interface names are converted to
   TC arrow function types. Reference types resolve via tag-class-registry."
  [tag]
  (or (get primitive-tc-ann tag)
      ;; IFn__ interface names → TC arrow function types
      (when (and (symbol? tag)
                 (let [s (str tag)] (.startsWith s "raster.fn.IFn__")))
        (when-let [entry (get @types/fn-interface-by-name tag)]
          (let [param-tags (:param-tags entry)
                param-types (mapv #(get primitive-tc-ann % 'clojure.core.typed/Any) param-tags)]
            (vec (concat param-types [:-> 'clojure.core.typed/Any])))))
      (when-let [^Class cls (get @types/tag-class-registry tag)]
        (symbol (.getName cls)))))

(defn- build-tc-ann [table-atom ret-tags]
  (let [table @table-atom
        sigs (for [[_arity methods] table
                   {:keys [tags]} methods
                   :let [known-params? (every? tag->tc-ann tags)]
                   :when known-params?]
               (let [param-types (mapv tag->tc-ann tags)
                     ret-type (get ret-tags tags 'clojure.core.typed/Any)]
                 (vec (concat param-types [:-> ret-type]))))]
    (when (seq sigs)
      (list* 'typed.clojure/IFn sigs))))

(defn- emit-tc-ann! [target-ns-obj simple-name table-atom]
  (try
    (let [ann-fn t/-ann]
      (let [ns-sym (ns-name target-ns-obj)
            qsym (symbol (str ns-sym) (str simple-name))
            ret-tags (reduce
                      (fn [m [_arity methods]]
                        (reduce
                         (fn [m {:keys [tags mangled-ns]}]
                           (let [mangled-sym (types/mangle simple-name tags)
                                 ;; Resolve in the defining namespace — mangled-ns is
                                 ;; where the impl var was created by deftm
                                 impl-ns (if mangled-ns (the-ns mangled-ns) target-ns-obj)
                                 v (ns-resolve impl-ns mangled-sym)
                                 ret (when v (or (:raster.core/return-tag (meta v))
                                                 (:tag (meta v))))]
                             (if ret (assoc m tags (or (tag->tc-ann ret) ret)) m)))
                         m methods))
                      {} @table-atom)
            type-form (build-tc-ann table-atom ret-tags)]
        (when type-form
          (ann-fn [ns-sym qsym type-form {} nil nil]))))
    (catch Exception e
      (binding [*out* *err*]
        (println (str "WARNING: TC annotation failed for " simple-name ": " (.getMessage e)))))))

;; ================================================================
;; ^:no-inline opacity — the SINGLE source of truth
;; ================================================================

(defn no-inline?
  "True if a deftm var (or the symbol naming it) is ^:no-inline — directly, or via its
   `_m_` dispatch parent (devirtualized impl/method names carry the mangle, e.g.
   `foo_m_double-impl` → parent `foo`). The one predicate every pass consults for inline
   opacity: the scalar inliner and the JVM bytecode backend call THIS rather than each
   re-parsing the `_m_` convention. Accepts a Var or a (qualified) symbol."
  [v-or-sym]
  (let [[the-ns base-name direct-var]
        (cond
          (var? v-or-sym)
          [(:ns (meta v-or-sym)) (str (:name (meta v-or-sym))) v-or-sym]
          (qualified-symbol? v-or-sym)
          [(some-> (namespace v-or-sym) symbol find-ns) (name v-or-sym)
           (try (resolve v-or-sym) (catch Exception _ nil))]
          :else [nil nil nil])]
    (boolean
     (or (some-> direct-var meta :no-inline)
         (when (and the-ns base-name)
           (when-let [idx (str/index-of base-name "_m_")]
             (when-let [parent (ns-resolve the-ns (symbol (subs base-name 0 idx)))]
               (:no-inline (meta parent)))))))))

;; ================================================================
;; register-method!
;; ================================================================

(defn register-method!
  "Register a typed method and update the dispatch function."
  ([specialized-fn fn-name tags ns-obj]
   (register-method! specialized-fn fn-name tags ns-obj nil nil nil nil))
  ([specialized-fn fn-name tags ns-obj typed-impl typed-iface]
   (register-method! specialized-fn fn-name tags ns-obj typed-impl typed-iface nil nil))
  ([specialized-fn fn-name tags ns-obj typed-impl typed-iface typed-target-fn]
   (register-method! specialized-fn fn-name tags ns-obj typed-impl typed-iface typed-target-fn nil))
  ([specialized-fn fn-name tags ns-obj typed-impl typed-iface typed-target-fn warning-meta]
   (let [[target-ns-obj simple-name]
         (if (qualified-symbol? fn-name)
           [(the-ns (symbol (namespace fn-name))) (symbol (name fn-name))]
           [ns-obj fn-name])
         k (gf-key (ns-name target-ns-obj) simple-name)
         table-atom (get-or-create-table! k)]
     (add-method! simple-name table-atom tags specialized-fn typed-impl typed-iface typed-target-fn warning-meta
                  (ns-name ns-obj))
     ;; Register tag→Class mappings globally
     (doseq [tag tags]
       (let [cls (binding [*ns* ns-obj] (types/tag->check-class tag))]
         (when (and (symbol? tag) (not= cls Object))
           (swap! types/tag-class-registry assoc tag cls))))
     (let [existing-var (ns-resolve target-ns-obj simple-name)
           reducible? (when existing-var (:raster.core/reducible (meta existing-var)))
           dfn (make-dispatch-fn simple-name table-atom reducible? target-ns-obj)
           ;; Wrap in typed dispatch class (IFn + all IFn__ interfaces)
           dispatch-obj (try
                          (require 'raster.compiler.core.typed-dispatch)
                          (let [make-td (resolve 'raster.compiler.core.typed-dispatch/make-typed-dispatch)]
                            (make-td simple-name table-atom dfn))
                          (catch Exception e
                            (when (System/getProperty "raster.debug")
                              (println "WARN: typed dispatch failed for" simple-name ":" (.getMessage e)))
                            dfn))
           v (intern target-ns-obj simple-name dispatch-obj)
           ;; Set :arglists without primitive type hints. Clojure's compiler
           ;; emits IFn$ checkcast when arglists have ^long/^double hints,
           ;; but the dispatch class only implements our IFn__ interfaces.
           ;; Boxed invoke is the correct slow path; compiled code uses .invk.
           arglists (vec (for [[_arity methods] @table-atom
                               {:keys [tags]} methods]
                           (mapv #(symbol (str "arg" %)) (range (count tags)))))]
       (alter-meta! v assoc
                    :raster.core/generic-function true
                    :raster.core/dispatch-table table-atom
                    :raster.core/reducible reducible?
                    ;; Propagate ^:no-inline explicitly. For unqualified names `intern`
                    ;; copies it off `simple-name`'s meta, but for QUALIFIED/extension
                    ;; names simple-name = (symbol (name fn-name)) strips the meta — so
                    ;; without this the flag would silently drop and the op be inlinable.
                    :no-inline (:no-inline (meta fn-name))
                    :arglists (seq arglists))
       (clear-specialization-cache! k)
       (emit-tc-ann! target-ns-obj simple-name table-atom)
       v))))

(defn set-reducible!
  "Mark a generic function as reducible (associative)."
  [fn-var]
  (alter-meta! fn-var assoc :raster.core/reducible true)
  (let [table-atom (:raster.core/dispatch-table (meta fn-var))
        fn-name (symbol (name (:name (meta fn-var))))
        target-ns (the-ns (:ns (meta fn-var)))
        dfn (make-dispatch-fn fn-name table-atom true target-ns)
        dispatch-obj (try
                       (let [make-td (resolve 'raster.compiler.core.typed-dispatch/make-typed-dispatch)]
                         (if make-td (make-td fn-name table-atom dfn) dfn))
                       (catch Exception _ dfn))]
    (alter-var-root fn-var (constantly dispatch-obj)))
  fn-var)

;; ================================================================
;; Generic function helpers
;; ================================================================

(defn ensure-generic!
  "Ensure a generic function var exists. Creates if needed."
  [target-ns simple-name]
  (let [k (gf-key (ns-name target-ns) simple-name)]
    (get-or-create-table! k)
    (when-not (ns-resolve target-ns simple-name)
      (let [table-atom (get-or-create-table! k)
            dfn (make-dispatch-fn simple-name table-atom false target-ns)
            v (intern target-ns simple-name dfn)]
        (alter-meta! v assoc
                     :raster.core/generic-function true
                     :raster.core/dispatch-table table-atom)
        v))))

;; ================================================================
;; Parametric Dispatch (Julia-style where {T} specialization)
;; ================================================================

;; {qualified-fn-sym -> [{:type-vars [T], :annotations [...], :body ..., :ns ...}]}
(defonce parametric-registry (atom {}))

;; Maps parametric value type base names to their type variable lists.
;; e.g. {Dual [T], SparseVector [T]}
;; Used by unify-type-var to extract T from Dual__Sym matching Dual annotation.
(defonce parametric-value-type-vars (atom {}))

(defn register-parametric-value-type!
  "Register a parametric value type's type variables for dispatch unification.
   Called by defvalue (All [T]) at macro expansion time."
  [base-name type-vars]
  (swap! parametric-value-type-vars assoc base-name (vec type-vars)))

(def ^:private array-element-types
  "Map from JVM array class to element type tag."
  {(Class/forName "[D") 'double
   (Class/forName "[F") 'float
   (Class/forName "[J") 'long
   (Class/forName "[I") 'int
   (Class/forName "[S") 'short
   (Class/forName "[B") 'byte})

(def ^:private element->array-tag
  {'double 'doubles 'float 'floats 'long 'longs 'int 'ints
   'short 'shorts 'byte 'bytes})

(def ^:private element->scalar-tag
  {'double 'Double 'float 'Float 'long 'Long 'int 'Int})

(defn- unify-type-var
  "Try to unify annotation pattern with concrete arg type.
   (Array T) matched against double[] → {T double}.
   T matched against Double → {T Double}.
   Returns binding map or nil on failure."
  [annotation arg-class type-vars]
  (cond
    ;; (Array T) — extract element type from array class or array-like registry
    (and (sequential? annotation) (= 'Array (first annotation))
         (symbol? (second annotation))
         (contains? type-vars (second annotation)))
    (let [elem (or (get array-element-types arg-class)
                   ;; Array-like types: bind T to the simple class name.
                   ;; The class is imported into the source namespace by
                   ;; try-parametric-dispatch before eval, so the short name resolves.
                   ;; Using the short name matches dispatch table entries (e.g. Multivector).
                   (when-not (.isArray arg-class)
                     (let [simple-tag (symbol (.getSimpleName ^Class arg-class))]
                       (when (contains? @types/array-like-registry simple-tag)
                         simple-tag))))]
      (when elem
        {(second annotation) elem}))

    ;; Bare type variable: T matched against a class
    (and (symbol? annotation) (contains? type-vars annotation))
    (let [tag (cond
                (= arg-class Double) 'double
                (= arg-class Double/TYPE) 'double
                (= arg-class Float) 'float
                (= arg-class Float/TYPE) 'float
                (= arg-class Long) 'long
                (= arg-class Long/TYPE) 'long
                (= arg-class Integer) 'int
                (= arg-class Integer/TYPE) 'int
                :else (symbol (.getName ^Class arg-class)))]
      {annotation tag})

    ;; Parametric value type: Dual annotation matched against Dual__Sym arg → {T Sym}
    ;; When a concrete annotation names a parametric value type and the arg is a
    ;; specialization of that type, extract the type suffix and bind it to the
    ;; matching type variable from the deftm's (All [T]) declaration.
    (and (symbol? annotation)
         (not (contains? type-vars annotation)))
    (let [arg-name (.getSimpleName ^Class arg-class)
          ann-name (name annotation)]
      (when (and (.contains ^String arg-name "__")
                 (.startsWith ^String arg-name (str ann-name "__")))
        (let [suffix-sym (symbol (subs arg-name (+ (count ann-name) 2)))]
          (when-let [pvt-vars (get @parametric-value-type-vars annotation)]
            ;; Find the first type variable that's also in the deftm's type-vars
            (when-let [matching-tv (first (filter type-vars (set pvt-vars)))]
              {matching-tv suffix-sym})))))

    ;; Concrete type (no variable) — just check match
    :else nil))

(defn- unify-parametric
  "Try to unify all parametric annotations against concrete argument types.
   Returns unified type-var bindings {T elem-type} or nil on failure."
  [annotations args type-vars-set]
  (when (= (count annotations) (count args))
    (loop [bindings {}
           anns (seq annotations)
           args (seq args)]
      (if-not anns
        bindings
        (let [ann (first anns)
              arg (first args)
              arg-class (class arg)]
          (if-let [new-bindings (unify-type-var ann arg-class type-vars-set)]
            ;; Check consistency: if T was already bound, must agree
            (let [consistent? (every? (fn [[k v]]
                                        (or (not (contains? bindings k))
                                            (= (get bindings k) v)))
                                      new-bindings)]
              (if consistent?
                (recur (merge bindings new-bindings) (next anns) (next args))
                nil))  ;; Inconsistent binding
            ;; No type variable to bind — check if annotation contains
            ;; unresolved type vars (e.g. (Fn [(Array T)] Double)).
            ;; If so, skip validation — we'll verify after all vars are bound.
            (let [has-type-var? (some type-vars-set (flatten (if (sequential? ann) ann [ann])))
                  expected-class (when-not has-type-var?
                                   (try (types/tag->check-class (types/annotation->tag ann nil))
                                        (catch Exception _ Object)))]
              (if (or has-type-var?
                      (check-type (or expected-class Object) arg)
                      ;; Check if arg is a parametric specialization of the expected type.
                      ;; e.g. RK4Cache__Multivector matches annotation RK4Cache
                      ;; (Julia: RK4Cache{Multivector} <: RK4Cache)
                      (when (and (symbol? ann) expected-class)
                        (let [arg-name (.getSimpleName (class arg))
                              ann-name (name ann)]
                          (.startsWith ^String arg-name (str ann-name "__")))))
                (recur bindings (next anns) (next args))
                nil))))))))

(defn substitute-type-var-raw
  "Substitute type variable T with its raw bound value (element type, not scalar type).
   Used for recursive substitution inside type constructors.
   Collapses (Array X) → X when X is a registered array-like type."
  [annotation bindings]
  (cond
    ;; Bare T → raw element type (double, float, long, etc.)
    (and (symbol? annotation) (contains? bindings annotation))
    (get bindings annotation)

    ;; Vector: preserve structure
    (vector? annotation)
    (let [substituted (mapv #(substitute-type-var-raw % bindings) annotation)]
      (if (= substituted annotation) annotation substituted))

    ;; List: recurse, then collapse (Array ArrayLikeType) → ArrayLikeType
    (sequential? annotation)
    (let [substituted (map #(substitute-type-var-raw % bindings) annotation)]
      (if (= (seq substituted) (seq annotation))
        annotation
        (let [result (apply list substituted)]
          ;; Collapse: (Array Multivector) → Multivector for array-like types
          (if (and (= 'Array (first result))
                   (= 2 (count result))
                   (symbol? (second result))
                   (let [s (second result) simple (when (symbol? s) (symbol (last (clojure.string/split (str s) #"\."))))] (or (contains? @types/array-like-registry s) (contains? @types/array-like-registry simple))))
            (second result)
            result))))

    :else annotation))

(defn substitute-type-var
  "Substitute type variable T with its bound value in a TOP-LEVEL annotation.
   Bare T at top level → scalar TC type (Double, Float, etc.).
   Inside type constructors, uses raw element type (double, float, etc.)."
  [annotation bindings]
  (cond
    ;; Bare T at top level → scalar TC type (Double, Float)
    (and (symbol? annotation) (contains? bindings annotation))
    (get element->scalar-tag (get bindings annotation) annotation)

    ;; Compound: recurse with RAW substitution to keep element types
    (vector? annotation)
    (let [substituted (mapv #(substitute-type-var-raw % bindings) annotation)]
      (if (= substituted annotation) annotation substituted))

    (sequential? annotation)
    (let [substituted (map #(substitute-type-var-raw % bindings) annotation)]
      (if (= (seq substituted) (seq annotation))
        annotation
        (let [result (apply list substituted)]
          ;; Collapse: (Array Multivector) → Multivector for array-like types
          (if (and (= 'Array (first result))
                   (= 2 (count result))
                   (symbol? (second result))
                   (let [s (second result) simple (when (symbol? s) (symbol (last (clojure.string/split (str s) #"\."))))] (or (contains? @types/array-like-registry s) (contains? @types/array-like-registry simple))))
            (second result)
            result))))

    :else annotation))

(defn register-parametric!
  "Register a parametric method template.
   type-vars: [T] — the type variables
   annotations: [(Array T) (Array T)] — with type vars
   ret-annotation: T or (Array T) — return type with vars
   body: the deftm source body (with T references)
   params: [a b] — parameter names
   fn-name: qualified symbol
   source-ns: namespace"
  [fn-name type-vars annotations ret-annotation params body source-ns]
  (let [new-entry {:type-vars (set type-vars)
                   :type-var-list (vec type-vars)
                   :annotations (vec annotations)
                   :ret-annotation ret-annotation
                   :params (vec params)
                   :body body
                   :source-ns source-ns}
        anns-key (vec annotations)]
    ;; Replace existing entry with matching annotations (REPL reload safety),
    ;; or append if no match exists.
    (swap! parametric-registry update fn-name
           (fn [entries]
             (let [entries (or entries [])
                   idx (some (fn [[i e]] (when (= (:annotations e) anns-key) i))
                             (map-indexed vector entries))]
               (if idx
                 (assoc entries idx new-entry)
                 (conj entries new-entry)))))))

;; Callback for parametric specialization. Set by core.clj during loading.
;; Breaks the dispatch↔core cycle: dispatch defines the protocol,
;; core provides the implementation.
(defonce ^:private parametric-specializer
  (atom (fn [_fn-name _template _bindings _args]
          (throw (ex-info "Parametric specializer not initialized (core.clj not loaded)" {})))))

(defn set-parametric-specializer!
  "Set the callback for parametric specialization. Called by core.clj at load time."
  [f]
  (reset! parametric-specializer f))

;; Registration-only callback: generate + register the concrete specialization
;; WITHOUT invoking it. Used at compile time, where invoking the fn (as
;; try-parametric-dispatch does) is both unnecessary and unsafe — the trigger
;; args are degenerate placeholders (zero-length arrays, 0 scalars), so a body
;; that divides by a scalar param (e.g. head-dim = hidden/num-heads) would crash
;; even though the bytecode compiled fine.
(defonce ^:private parametric-register
  (atom (fn [_fn-name _template _bindings _args]
          (throw (ex-info "Parametric register not initialized (core.clj not loaded)" {})))))

(defn set-parametric-register!
  "Set the registration-only callback (register, do not invoke). Called by core.clj."
  [f]
  (reset! parametric-register f))

(defn- try-parametric-with
  "Shared core of parametric dispatch/registration: unify the first matching
  template against `args` and hand off to `callback` (which either invokes or
  just registers). Returns {::result r} (r = the callback's result) when a
  template matched, or nil when no template matches. The box distinguishes a
  matched method that legitimately returned nil — a Void deftm like
  residual-add! — from no-match; an unboxed nil here made every (All [T]) Void
  deftm throw no-matching-method on its FIRST eager dispatch for a fresh
  element type (after already running its side effect)."
  [callback fn-name args]
  (let [;; Resolve namespace aliases: nn/linear → raster.dl.nn/linear
        resolved-name (if-let [v (resolve fn-name)]
                        (symbol (str (:ns (meta v))) (str (:name (meta v))))
                        fn-name)]
    (when-let [templates (or (get @parametric-registry resolved-name)
                             (get @parametric-registry fn-name))]
      (some (fn [template]
              (when-let [bindings (unify-parametric
                                   (:annotations template) args
                                   (:type-vars template))]
                {::result (callback resolved-name template bindings args)}))
            templates))))

(defn try-parametric-dispatch
  "Try to dispatch via parametric specialization.
   If a parametric template matches, delegates to the specializer callback
   (provided by core.clj) to generate and register a concrete method,
   then INVOKES it. Returns {::result r} (r may be nil for Void methods)
   when a template matched, nil if no match — unwrap with ::result."
  [fn-name args]
  (try-parametric-with @parametric-specializer fn-name args))

(defn try-parametric-register!
  "Like try-parametric-dispatch, but only REGISTERS the concrete specialization
  (does not invoke it). This is the compile-time trigger: it makes the concrete
  mangled method available for devirtualization without running the fn on the
  degenerate trigger args. Returns a truthy registration result or nil if no
  template matches (throws if a template matches but compilation fails)."
  [fn-name args]
  (when-let [boxed (try-parametric-with @parametric-register fn-name args)]
    (or (::result boxed) true)))
