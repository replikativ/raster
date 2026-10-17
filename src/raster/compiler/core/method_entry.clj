(ns raster.compiler.core.method-entry
  "MethodEntry — dispatch table entry as a plain map.

  Fields:
    :tags             — vector of type tags [doubles double ...]
    :arity            — argument count
    :check-classes    — vector of Class objects for runtime type checking
    :target-fn        — the backing function
    :specificity      — dispatch priority score
    :typed-impl       — deftype singleton for .invk, or nil
    :typed-iface      — typed interface symbol, or nil
    :typed-target-fn  — wrapper fn for typed dispatch, or nil
    :warning-meta     — metadata for warning suppression
    :has-compound-tags — true if any tag is compound
    :mangled-ns       — namespace symbol where mangled defn lives

  Previously a defrecord, now a plain map to avoid classloader conflicts.
  Maps use value equality across classloaders, eliminating ClassCastException
  when compiled (eval'd) code calls back into dispatch."
  (:require [raster.compiler.core.types :as types]))

(defn- retained-global-var [callee]
  (when (and (symbol? callee) (namespace callee))
    (when-let [source-ns (find-ns (symbol (namespace callee)))]
      (.findInternedVar ^clojure.lang.Namespace source-ns (symbol (name callee))))))

(defn selected-definition
  "Project the live registered definition of an already selected typed call.
   This is correspondence evidence, not a purity, type-inference or result
   presence rule. No method is selected from guessed operand types or invoked.
   Both the receiver and its delegated source root must still match registration.
   Unresolved/generic calls and legacy mutable-Var entries have no projection."
  [operation implementation-op argument-count]
  (let [generic (retained-global-var operation)
        receiver (retained-global-var implementation-op)
        table (:raster.core/dispatch-table (meta generic))]
    (when (and generic receiver table)
      (some
       (fn [{:keys [tags mangled-ns typed-impl target-fn] :as entry}]
         (when (and (some? typed-impl) (not (var? typed-impl))
                    (identical? typed-impl @receiver))
           (let [source-name (types/mangle (symbol (name operation)) tags)
                 source (when mangled-ns (retained-global-var (symbol (str mangled-ns) (str source-name))))
                 source-meta (meta source)
                 params (:raster.core/deftm-params source-meta)]
             (when (and source
                        (= implementation-op (symbol (str mangled-ns) (str source-name "-impl")))
                        (identical? target-fn @source)
                        (= tags (:raster.core/deftm-tags source-meta))
                        (= argument-count (count params) (count tags)))
               {:entry entry :operation operation :implementation-op implementation-op
                :source-var source :receiver-var receiver :parameters params :tags tags
                :return-tag (:raster.core/return-tag source-meta)
                :source-body (:raster.core/deftm-source-body source-meta)
                :walked-body (:raster.core/deftm-walked-body source-meta)}))))
       (get @table argument-count)))))

(def ^:dynamic *compilation-classloader*
  "When bound, ALL defineClass calls during BC compilation use this DCL.
  Prevents class identity mismatches from nested eval/require."
  nil)

(def ^:dynamic *bytecode-compiler*
  "When bound, compile-typed-impl! function for lazy JIT bytecode upgrade.
  Set by bytecode.clj at load time. Used by core.clj's do-bytecode-upgrade!."
  nil)
