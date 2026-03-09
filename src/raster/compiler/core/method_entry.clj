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
  when compiled (eval'd) code calls back into dispatch.")

(def ^:dynamic *compilation-classloader*
  "When bound, ALL defineClass calls during BC compilation use this DCL.
  Prevents class identity mismatches from nested eval/require."
  nil)

(def ^:dynamic *bytecode-compiler*
  "When bound, compile-typed-impl! function for lazy JIT bytecode upgrade.
  Set by bytecode.clj at load time. Used by core.clj's do-bytecode-upgrade!."
  nil)
