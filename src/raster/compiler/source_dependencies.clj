(ns raster.compiler.source-dependencies
  "Conservative transitive source evidence for persistent compiler-template lookup.

   Retained deftm bodies are resolved in their declaring namespace and fingerprinted recursively.
   Calls owned by Raster or Clojure are covered by the packaged compiler-build manifest. Opaque
   application Vars, unresolved calls and dynamic local calls become explicit blockers; they are
   never approximated with printing or object identity."
  (:require [clojure.string :as str]
            [raster.compiler.core.types :as types]
            [raster.compiler.ir.semantic-fingerprint :as semantic-fingerprint]
            [raster.core :as rcore]))

(def schema-version 1)

(defn- qualified-var-symbol [v]
  (let [{:keys [ns name]} (meta v)]
    (when (and ns name) (symbol (str (ns-name ns)) (str name)))))

(defn- source-namespace [v]
  (let [metadata (meta v)
        source (:raster.core/deftm-source-ns metadata)]
    (or (when source (find-ns source)) (:ns metadata))))

(defn- retained-source [v]
  (let [metadata (meta v)]
    (when-let [body (:raster.core/deftm-source-body metadata)]
      {:symbol (qualified-var-symbol v)
       :params (:raster.core/deftm-params metadata)
       :tags (:raster.core/deftm-tags metadata)
       :return-tag (:raster.core/return-tag metadata)
       :source-body body
       :source-fingerprint (semantic-fingerprint/fingerprint body)})))

(defn- owned-namespace? [build-owned-namespaces v]
  (when-let [namespace (some-> v meta :ns ns-name str)]
    (or (= namespace "clojure")
        (str/starts-with? namespace "clojure.")
        (contains? build-owned-namespaces (symbol namespace)))))

(defn- call-heads
  "Return syntactic call heads without descending into quoted data. `.invk` retains its explicit
   implementation Var as a dependency candidate. This is discovery only; resolution below decides
   whether a candidate is certified, build-owned, or a blocker."
  [forms]
  (let [calls (volatile! [])]
    (letfn [(visit [form]
              (cond
                (seq? form)
                (let [head (first form)]
                  (when-not (= 'quote head)
                    (cond
                      (and (= '.invk head) (symbol? (second form)))
                      (vswap! calls conj (second form))

                      (symbol? head)
                      (vswap! calls conj head))
                    (doseq [child (rest form)] (visit child))))

                (map? form) (doseq [[key value] form] (visit key) (visit value))
                (coll? form) (doseq [child form] (visit child))
                :else nil))]
      (doseq [form forms] (visit form))
      @calls)))

(defn- resolve-call [source-ns symbol]
  (when (and source-ns (symbol? symbol) (not (special-symbol? symbol)))
    (ns-resolve source-ns symbol)))

(defn- dispatch-candidates
  "Return every registered backing method for a generic deftm Var. Raw source lacks enough typed
   call-site information to select an overload soundly (the compilation dtype does not determine
   integer, scalar, array and AD arguments), so dependency evidence deliberately over-approximates.
   Explicit `.invk` and already-mangled Vars remain singletons."
  [v]
  (if (:raster.core/deftm-source-body (meta v))
    [v]
    (if-let [table (:raster.core/dispatch-table (meta v))]
      (let [generic-name (:name (meta v))
            generic-ns (:ns (meta v))]
        (->> @table
             vals
             (mapcat identity)
             (keep (fn [{:keys [tags mangled-ns]}]
                     (let [target-ns (if mangled-ns (find-ns mangled-ns) generic-ns)]
                       (when target-ns
                         (ns-resolve target-ns (types/mangle generic-name tags))))))
             distinct
             vec))
      [v])))

(defn- jvm-interop-call
  "Canonicalize syntactic JVM member calls covered by the compiler-build and Java identities."
  [build-owned-namespaces source-ns call]
  (cond
    (str/starts-with? (name call) ".") call

    (namespace call)
    (or (when-let [^Class imported (get (ns-imports source-ns) (symbol (namespace call)))]
          (symbol (.getName imported) (name call)))
        (let [class-name (namespace call)
              separator (.lastIndexOf ^String class-name ".")
              package (when (pos? separator) (subs class-name 0 separator))]
          (when (or (some #(str/starts-with? class-name %)
                          ["java." "javax." "jdk." "clojure.lang."])
                    (and package (contains? build-owned-namespaces (symbol package))))
            call)))

    :else nil))

(defn manifest
  "Derive transitive retained-source evidence for `root-var` at `dtype`.

   The result is pure data with `:complete?`, ordered `:dependencies`, structured `:blockers`, and
   a canonical `:fingerprint`. Completeness means every application-level syntactic call reachable
   through retained deftm bodies was resolved and fingerprinted; compiler/build-owned calls are
   represented by `:covered-build-calls` and rely on the separate compiler-build fingerprint."
  ([root-var dtype] (manifest root-var dtype {}))
  ([root-var dtype {:keys [build-owned-namespaces]
                    :or {build-owned-namespaces #{}}}]
   (let [build-owned-namespaces (set build-owned-namespaces)
         root (or (try
                    (rcore/resolve-deftm-var root-var {:dtype dtype :ambiguity :throw})
                    (catch clojure.lang.ExceptionInfo _ nil))
                  root-var)
         pending (volatile! [root])
         seen (volatile! #{})
         dependencies (volatile! {})
         covered (volatile! #{})
         blockers (volatile! #{})]
     (loop []
       (when-let [candidate (peek @pending)]
         (vswap! pending pop)
         (let [resolved candidate
               symbol (qualified-var-symbol resolved)]
           (when-not (contains? @seen symbol)
             (vswap! seen conj symbol)
             (if-let [{:keys [source-body] :as source} (retained-source resolved)]
               (do
                 (vswap! dependencies assoc symbol (dissoc source :source-body))
                 (let [source-ns (source-namespace resolved)
                       locals (set (:params source))]
                   (doseq [call (distinct (call-heads source-body))]
                     (cond
                       (contains? locals call)
                       (vswap! blockers conj {:kind :dynamic-local-call
                                              :owner symbol :call call})

                       :else
                       (if-let [called (resolve-call source-ns call)]
                         (if (owned-namespace? build-owned-namespaces called)
                          ;; Raster/Clojure implementation changes are already part of the exact
                          ;; packaged compiler build. Do not expand their dispatch tables into
                          ;; irrelevant overloads merely to rediscover the same build identity.
                           (vswap! covered conj (qualified-var-symbol called))
                           (let [candidates (dispatch-candidates called)]
                             (if (seq candidates)
                               (doseq [called candidates]
                                 (if (:raster.core/deftm-source-body (meta called))
                                   (vswap! pending conj called)
                                   (vswap! blockers conj {:kind :opaque-application-var
                                                          :owner symbol
                                                          :call call
                                                          :resolved (qualified-var-symbol called)})))
                               (vswap! blockers conj {:kind :empty-dispatch
                                                      :owner symbol :call call}))))
                         (if-let [interop (jvm-interop-call build-owned-namespaces source-ns call)]
                           (vswap! covered conj interop)
                           (when-not (special-symbol? call)
                             (vswap! blockers conj {:kind :unresolved-call
                                                    :owner symbol :call call}))))))))
               (vswap! blockers conj {:kind :missing-retained-source :var symbol}))))
         (recur)))
     (let [evidence {:schema-version schema-version
                     :root (qualified-var-symbol root)
                     :dtype dtype
                     :dependencies (->> @dependencies vals
                                        (sort-by (comp str :symbol)) vec)
                     :covered-build-calls (vec (sort-by str @covered))
                     :blockers (vec (sort-by semantic-fingerprint/fingerprint @blockers))}]
       (assoc evidence
              :complete? (empty? (:blockers evidence))
              :fingerprint (semantic-fingerprint/fingerprint evidence))))))
