(ns raster.compiler.core.specialize
  "Element-type specialization for deftm functions.

  Takes a deftm generic function and produces a devirtualized,
  bytecode-compiled version specialized for a specific element type."
  (:require [clojure.string :as str]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.inference :as inf]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.walker :as walker]))

(def ^:dynamic *specializing*
  "Set of [fn-var element-type] pairs currently being specialized.
  Used to detect and break circular deftm dependencies."
  #{})

;; ================================================================
;; Helpers
;; ================================================================

(defn- specializable-var?
  "Check if a var can be specialized."
  [v]
  (let [m (meta v)]
    (:raster.core/generic-function m)))

(defn- deftm-resolve-source
  "Resolve source body and hinted params from a deftm dispatch var.
  Finds the method entry whose params contain 'objects' tags (the generic version),
  then reads the mangled var's ::deftm-source-body.
  Returns {:source-params [...], :source-body [...], :mangled-var Var} or nil."
  [dispatch-var]
  (let [m (meta dispatch-var)
        table-atom (:raster.core/dispatch-table m)
        fn-name (:name m)]
    (when table-atom
      (let [table @table-atom
            ;; Find the method entry with the most 'objects' tags (the generic template)
            all-entries (mapcat val table)
            best-entry (first (sort-by
                               (fn [e] (- (count (filter #{'objects} (:tags e)))))
                               all-entries))]
        (when best-entry
          (let [tags (:tags best-entry)
                mangled-sym (types/mangle fn-name tags)
                mangled-var (ns-resolve (the-ns (:ns m)) mangled-sym)]
            (when mangled-var
              (let [mm (meta mangled-var)
                    deftm-params (:raster.core/deftm-params mm)
                    deftm-tags (:raster.core/deftm-tags mm)
                    source-body (:raster.core/deftm-source-body mm)
                    ;; Reconstruct hinted params: add ^tag metadata to each param symbol
                    hinted-params (mapv (fn [p tag]
                                          (if (and tag (not= tag 'Object))
                                            (vary-meta p assoc :tag tag)
                                            p))
                                        deftm-params deftm-tags)]
                {:source-params hinted-params
                 :source-body source-body
                 :mangled-var mangled-var}))))))))

;; ================================================================
;; Main specialization entry point
;; ================================================================

(defn specialize-fn!
  "Specialize a deftm function for a given element type.
  Walks the source body with element-env to devirtualize dispatch calls,
  then compiles to a JVM static method via bytecode generation.

  Returns the symbol of the specialized function.
  Caches the result in ::specialized-versions metadata on the var.

  opts:
    :element-type — the element type tag (symbol, e.g. 'Complex)
    :fn-bindings — {param-sym -> var-sym} for higher-order function params
    :ns-obj — namespace to intern the specialized fn (default: var's ns)
    :param-hints — {param-sym -> tag-sym} for bytecode-only primitive hints
                   (e.g. {t double, dt double}). These don't affect dispatch."
  [fn-var {:keys [element-type fn-bindings ns-obj param-hints]}]
  (let [m (meta fn-var)
        is-deftm? (:raster.core/generic-function m)
        _ (when-not is-deftm?
            (throw (ex-info (str "Cannot specialize: " (:name m) " is not a deftm function")
                            {:var fn-var})))
        ;; Check cache first
        existing (get (:raster.core/specialized-versions m) element-type)]
    (if existing
      existing
      (if (contains? *specializing* [fn-var element-type])
        nil ;; cycle detected — skip, dependency will use unspecialized version
        (binding [*specializing* (conj *specializing* [fn-var element-type])]
          (let [fn-name (:name m)
                walk-body walker/walk-body
                deftm-source (deftm-resolve-source fn-var)
                source-params (:source-params deftm-source)
            ;; Apply bytecode-only param hints (e.g. {t double, dt double})
                source-params (if param-hints
                                (mapv (fn [p]
                                        (if-let [hint (get param-hints p)]
                                          (vary-meta p assoc :tag hint)
                                          p))
                                      source-params)
                                source-params)
                source-body (:source-body deftm-source)
                target-ns (or ns-obj (the-ns (:ns m)))
            ;; Fully qualify element-type for internal resolution.
                fq-element-type (let [resolved (or (ns-resolve target-ns element-type)
                                                   (resolve element-type))]
                                  (if (class? resolved)
                                    (symbol (.getName ^Class resolved))
                                    element-type))
            ;; Build unified type-env from param hints
                type-env (reduce (fn [te p]
                                   (let [tag (:tag (meta p))]
                                     (cond-> te
                                       (and tag (not= tag 'Object))
                                       (assoc p {:tag tag})
                                       (= tag 'objects)
                                       (update p assoc :element fq-element-type))))
                                 {} (remove #{'&} source-params))
                fn-bindings (or fn-bindings {})
            ;; Add fn-binding return type info to type-env as fn-info :ret-tag
                type-env (reduce (fn [te [param-sym target-sym]]
                                   (let [target-var (resolve target-sym)
                                         m (when target-var (meta target-var))
                                         ret (when m
                                               (or (:raster.core/return-tag m)
                                                   (when (:raster.core/generic-function m)
                                                     (when-let [src (deftm-resolve-source target-var)]
                                                       (:raster.core/return-tag (meta (:mangled-var src)))))))]
                                     (if ret
                                       (update te param-sym (fnil assoc {}) :fn-info {:ret-tag ret})
                                       te)))
                                 type-env fn-bindings)
                source-ns (or (the-ns (:ns m)) target-ns)
            ;; TC binding tags for the specialized body (concrete types now known)
                tc-binding-tags
                (let [params (mapv #(with-meta % nil) (remove #{'&} source-params))
                      tags (mapv #(or (:tag (meta %)) 'Object) (remove #{'&} source-params))
                      annotations (mapv (fn [t] (if (= t 'Object) nil t)) tags)]
                  (inf/safe-tc-binding-tags '<specialize> params annotations source-body source-ns))
            ;; Walk the body with unified type-env + TC binding tags.
            ;; element-dtype: the concrete specialization tags are known here —
            ;; same effective-dtype rule as all other monomorphized walk seams,
            ;; so bare 0.0 accumulator inits narrow to the element dtype.
                element-dtype (dtype/infer-dtype-from-tags
                               (mapv #(or (:tag (meta %)) 'Object)
                                     (remove #{'&} source-params)))
                walked-body (binding [*ns* source-ns]
                              (mapv #(walk-body % (cond-> {:type-env type-env :source-ns source-ns}
                                                    (seq tc-binding-tags)
                                                    (assoc :tc-binding-tags tc-binding-tags)
                                                    (#{:float :double} element-dtype)
                                                    (assoc :element-dtype element-dtype)))
                                    source-body))
            ;; Apply fn-bindings: replace calls to bound params with specialized targets
                walked-body (if (seq fn-bindings)
                              (let [replace-fn-calls
                                    (fn replace-fn-calls [form]
                                      (cond
                                    ;; fn-bound symbol as call head → specialize & replace
                                        (and (seq? form) (symbol? (first form))
                                             (get fn-bindings (first form)))
                                        (let [target-sym (get fn-bindings (first form))
                                              target-var (resolve target-sym)
                                              target-spec (when (and target-var
                                                                     (specializable-var? target-var))
                                                            (specialize-fn! target-var
                                                                            {:element-type element-type
                                                                             :ns-obj target-ns}))]
                                          (cons (or target-spec target-sym)
                                                (map replace-fn-calls (rest form))))
                                    ;; fn-bound symbol in non-head position
                                        (and (symbol? form) (get fn-bindings form))
                                        (let [target-sym (get fn-bindings form)
                                              target-var (resolve target-sym)]
                                          (if target-var
                                            (symbol (str (.name (.ns ^clojure.lang.Var target-var)))
                                                    (str (.sym ^clojure.lang.Var target-var)))
                                            target-sym))
                                        (seq? form)
                                        (apply list (map replace-fn-calls form))
                                        (vector? form)
                                        (mapv replace-fn-calls form)
                                        :else form))]
                                (mapv replace-fn-calls walked-body))
                              walked-body)
            ;; Generate specialized function name
                safe-fn-name (let [n (clojure.core/name fn-name)]
                               (if (every? #(contains? types/op-name-map %) n)
                                 (str "_" (str/join "" (map types/op-name-map n)) "_")
                                 n))
                specialized-sym (symbol (str safe-fn-name "__" element-type))]
        ;; Recursively specialize any deftm calls found in the walked body
            (let [dependency-calls (atom #{})
                  _ (binding [*ns* source-ns]
                      (letfn [(find-dependency-calls [form]
                                (cond
                                  (and (seq? form) (symbol? (first form)))
                                  (do
                                    (when-let [v (resolve (first form))]
                                      (when (specializable-var? v)
                                        (swap! dependency-calls conj (first form))))
                                    (doseq [f (rest form)] (find-dependency-calls f)))
                                  (seq? form) (doseq [f form] (find-dependency-calls f))
                                  (vector? form) (doseq [f form] (find-dependency-calls f))
                                  :else nil))]
                        (doseq [b source-body] (find-dependency-calls b))))
              ;; Pre-specialize all deftm dependencies
                  _ (binding [*ns* source-ns]
                      (doseq [dep-sym @dependency-calls]
                        (when-let [dep-var (resolve dep-sym)]
                          (when (and (specializable-var? dep-var)
                                     (not (get (:raster.core/specialized-versions (meta dep-var)) element-type)))
                            (specialize-fn! dep-var {:element-type element-type :ns-obj target-ns})))))
              ;; Build mapping from deftm mangled names → specialized names
                  deftm-to-specialized
                  (binding [*ns* source-ns]
                    (into {} (keep (fn [dep-sym]
                                     (when-let [dep-var (resolve dep-sym)]
                                       (when (:raster.core/generic-function (meta dep-var))
                                         (when-let [spec-sym (get (:raster.core/specialized-versions (meta dep-var)) element-type)]
                                           (when-let [src (deftm-resolve-source dep-var)]
                                             (let [mangled-var (:mangled-var src)
                                                   mangled-sym (symbol (str (ns-name (:ns (meta mangled-var))))
                                                                       (str (:name (meta mangled-var))))
                                                   qualified-spec (symbol (str (ns-name target-ns))
                                                                          (str spec-sym))]
                                               [mangled-sym qualified-spec]))))))
                                   @dependency-calls)))
              ;; Re-walk with all dependencies now specialized (TC tags available)
                  walked-body (binding [*ns* source-ns]
                                (mapv #(walk-body % (cond-> {:type-env type-env :source-ns source-ns}
                                                      (seq tc-binding-tags)
                                                      (assoc :tc-binding-tags tc-binding-tags)))
                                      source-body))
              ;; Replace deftm mangled names with specialized names
                  walked-body (if (seq deftm-to-specialized)
                                (let [replace-deftm
                                      (fn replace-deftm [form]
                                        (cond
                                          (and (seq? form) (symbol? (first form))
                                               (get deftm-to-specialized (first form)))
                                          (cons (get deftm-to-specialized (first form))
                                                (map replace-deftm (rest form)))
                                          (seq? form)
                                          (apply list (map replace-deftm form))
                                          (vector? form)
                                          (mapv replace-deftm form)
                                          :else form))]
                                  (mapv replace-deftm walked-body))
                                walked-body)
              ;; Re-apply fn-bindings after re-walk
                  walked-body (if (seq fn-bindings)
                                (binding [*ns* source-ns]
                                  (let [replace-fn-calls
                                        (fn replace-fn-calls [form]
                                          (cond
                                        ;; fn-bound symbol as call head
                                            (and (seq? form) (symbol? (first form))
                                                 (get fn-bindings (first form)))
                                            (let [target-sym (get fn-bindings (first form))
                                                  target-var (resolve target-sym)
                                                  target-spec (when (and target-var
                                                                         (specializable-var? target-var))
                                                                (specialize-fn! target-var
                                                                                {:element-type element-type
                                                                                 :ns-obj target-ns}))]
                                              (cons (or target-spec target-sym)
                                                    (mapv replace-fn-calls (rest form))))
                                        ;; fn-bound symbol in non-head position
                                            (and (symbol? form) (get fn-bindings form))
                                            (let [target-sym (get fn-bindings form)
                                                  target-var (resolve target-sym)]
                                              (if target-var
                                                (symbol (str (.name (.ns ^clojure.lang.Var target-var)))
                                                        (str (.sym ^clojure.lang.Var target-var)))
                                                target-sym))
                                            (seq? form)
                                            (apply list (mapv replace-fn-calls form))
                                            (vector? form)
                                            (mapv replace-fn-calls form)
                                            :else form))]
                                    (mapv replace-fn-calls walked-body)))
                                walked-body)]
          ;; Compile to JVM static method via bytecode generation
              (do (require 'raster.compiler.backend.jvm.bytecode)
                  (let [compile-fn! (resolve 'raster.compiler.backend.jvm.bytecode/compile-specialized-class!)
                        class-name (str (ns-name target-ns) "." specialized-sym "__BC")
                        bc-result (compile-fn! class-name
                                               [{:name     specialized-sym
                                                 :params   source-params
                                                 :walked-body (vec walked-body)
                                                 :element-type fq-element-type
                                                 :source-ns (the-ns (:ns m))}])
                        bc-method (get (:methods bc-result) specialized-sym)]
                ;; Wrap the static method in an IFn
                    (let [param-syms (filterv #(or (symbol? %) (vector? %)) source-params)
                          n (count param-syms)
                          wrapper (case n
                                    0 (fn [] (.invoke bc-method nil (object-array 0)))
                                    1 (fn [a] (.invoke bc-method nil (object-array [a])))
                                    2 (fn [a b] (.invoke bc-method nil (object-array [a b])))
                                    3 (fn [a b c] (.invoke bc-method nil (object-array [a b c])))
                                    4 (fn [a b c d] (.invoke bc-method nil (object-array [a b c d])))
                                    (fn [& args] (.invoke bc-method nil (object-array args))))]
                      (intern target-ns specialized-sym wrapper))
                ;; Store bytecode metadata for direct invocation from other compiled methods
                    (alter-meta! (ns-resolve target-ns specialized-sym)
                                 assoc :raster.core/bytecode-class (:class bc-result)
                                 :raster.core/bytecode-method bc-method
                                 :raster.core/bytecode-bytes (:bytes bc-result)
                                 :raster.core/pipeline-fn-spec {:name specialized-sym
                                                                :params source-params
                                                                :walked-body (vec walked-body)
                                                                :element-type fq-element-type
                                                                :return-tag (:raster.core/return-tag
                                                                             (meta (:mangled-var deftm-source)))
                                                                :source-ns (the-ns (:ns m))})))
          ;; Cache in the source var's metadata
              (alter-meta! fn-var update :raster.core/specialized-versions
                           (fnil assoc {}) element-type specialized-sym)
              specialized-sym)))))))
