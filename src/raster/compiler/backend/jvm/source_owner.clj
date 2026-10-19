(ns raster.compiler.backend.jvm.source-owner
  "Compile an owner-local JVM source projection with actual private captures.

   This is a source compiler boundary, not another evaluator or executable
   registry. The producer supplies the compilation policy, resolved Methods,
   lexical ports and dependency guard. It must retain runtime/residual guards
   and must not export these private captures as ordinary portable source."
  (:require [clojure.set :as set]
            [raster.compiler.core.ordered-call :as ordered-call]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.form :as form])
  (:import [java.lang.reflect Method]))

(defn assert-portable-source!
  "Refuse source extraction that would discard an unprojected private owner.
   Prepared programs and callable metadata use separate, explicit owner slots."
  [program]
  (when (:raster.core/anonymous-function program)
    (let [retained (:ns program)
          live (some-> (:raster.core/deftm-source-ns program) find-ns)]
      (when-not (and (instance? clojure.lang.Namespace retained)
                     (identical? retained live))
        (throw (ex-info "Anonymous source namespace has been removed or replaced"
                        {:reason :stale-anonymous-source-namespace
                         :source-ns (:raster.core/deftm-source-ns program)}))))
    (let [policy (:raster.core/source-policy program)]
      (when-not (completion/source-policy? policy)
        (throw (ex-info "Anonymous source has no retained compilation policy"
                        {:reason :unclosed-anonymous-source-policy})))
      (when-not (false? (:unchecked-math policy))
        (throw (ex-info "Anonymous unchecked arithmetic needs source-owned lowering"
                        {:reason :unsupported-anonymous-source-policy :source-policy policy})))))
  (when (or (contains? program :source-owner)
            (contains? program :raster.compiler/source-owner)
            (seq (:raster.core/lexical-captures program)))
    (throw (ex-info "Private JVM source ownership requires a capture-aware projection"
                    {:reason :unprojected-jvm-source-owner})))
  program)

(defn call-with-source-context
  "Reuse an anonymous producer's actual namespace/policy, never the caller's.
   Named and separately owned entries retain their existing context contracts."
  [metadata compile!]
  (if (:raster.core/anonymous-function metadata)
    (do (assert-portable-source! metadata)
        (binding [*ns* (:ns metadata)
                  *unchecked-math* (get-in metadata [:raster.core/source-policy :unchecked-math])]
          (let [compiled (compile!)]
            (assert-portable-source! metadata)
            compiled)))
    (compile!)))

(defn- bound-identifiers [source]
  (let [identifiers (volatile! #{})]
    ;; Binder knowledge remains exclusively in scope-info, also for closures,
    ;; branches and recurrence scopes. Quoted data is opaque to this walk.
    (util/postwalk-preserving-meta
     (fn [expression]
       (when-let [{:keys [scopes]} (form/scope-info expression)]
         (doseq [scope scopes parameter (:binders scope) :when (symbol? parameter)]
           (vswap! identifiers conj parameter)))
       expression)
     source)
    @identifiers))

(defn compile-function
  "Compile one function under its owner's actual source context.

   function-form is the existing owner's parameter/arity-aware function builder.
   Captures are [{:port unqualified-symbol :value actual-object}], including
   actual Classes used by canonical CHECKCAST completion terms. No captured
   executable is installed in a namespace or reconstructed by its class name.

   The guard fences compilation before and after eval. This function does not
   establish call-site correspondence or provide a runtime/pullback lifetime
   wrapper; those remain with the producer that owns the captured executable."
  [params source function-form
   {:keys [source-policy source-ns captures reference-bindings methods guard]}]
  (when-not (and (vector? params) (every? symbol? params)
                 (= (count params) (count (distinct params)))
                 (ifn? function-form) (completion/source-policy? source-policy)
                 (instance? clojure.lang.Namespace source-ns)
                 (vector? captures) (every? #(= #{:port :value} (set (keys %))) captures)
                 (every? #(and (symbol? (:port %)) (nil? (namespace (:port %)))) captures)
                 (map? reference-bindings) (vector? methods)
                 (every? #(instance? Method %) methods) (or (nil? guard) (ifn? guard)))
    (throw (ex-info "JVM source owner requires a closed compilation context"
                    {:reason :unclosed-jvm-source-owner})))
  (let [ports (mapv :port captures)
        bound (into (set params) (bound-identifiers source))
        conflict (set/intersection bound (set ports))]
    (when (or (not= (count ports) (count (distinct ports))) (seq conflict))
      (throw (ex-info "private source captures must not shadow an owner binder"
                      {:reason :source-owner-capture-scope :conflicts conflict})))
    (doseq [[carrier port] reference-bindings]
      (when-not (and (class? carrier)
                     (some #(and (= port (:port %)) (identical? carrier (:value %))) captures))
        (throw (ex-info "reference completion requires its actual captured Class"
                        {:reason :source-owner-reference-port :port port}))))
    (let [body (completion/project-to-jvm-source source source-policy reference-bindings)
          used (into (into bound ports) (util/free-syms-flat [params body]))
          arguments (loop [id (gensym "owner_captures__")]
                      (if (contains? used id) (recur (gensym "owner_captures__")) id))
          bindings (vec (mapcat (fn [port index]
                                 [port (list 'clojure.core/nth arguments index)]) ports (range)))
          factory (list 'fn ['& arguments]
                        (list 'let* bindings (function-form params body)))]
      (binding [*ns* source-ns *unchecked-math* (:unchecked-math source-policy)]
        (when guard (guard))
        (doseq [method methods] (ordered-call/verify-source-method! method))
        (let [execute (apply (eval factory) (mapv :value captures))]
          (when guard (guard))
          execute)))))
