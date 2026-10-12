(ns raster.compiler.ir.static-validation
  "Bounded request-local reuse of successful structural IR validation.

   No proof survives a request. Runtime arguments, aliases, storage capacities, native resources
   and dispatch applicability are deliberately outside this context. Validator function identity
   is part of the key: replacing a validator cannot inherit another implementation's results.
   As with compilation, validator definitions and their dependencies must stay stable while a
   request executes; this is not a cross-redefinition or cross-request proof cache."
  (:import [java.util IdentityHashMap]))

(def ^:dynamic ^:private *context* nil)
(def ^:private max-entries 1024)

(defn with-validation-scope
  "Invoke with a fresh thread-confined structural validation context, discarded on every exit."
  [invoke]
  (binding [*context* {:thread (Thread/currentThread)
                      :validators (IdentityHashMap.) :entries (volatile! 0)}]
    (invoke)))

(defn validate
  "Run validator on value, reusing only an exact successful pair in the current request.
   A propagated dynamic binding on another thread never grants access to the owning map."
  [validator value]
  (if (and *context* (identical? (:thread *context*) (Thread/currentThread)))
    (let [^IdentityHashMap validators (:validators *context*)
          ^IdentityHashMap cache (.get validators validator)]
      (if (and cache (.containsKey cache value))
        (.get cache value)
        (let [checked (validator value)]
          ;; Cache only success. Overflow simply performs fresh checks, never evicts authority
          ;; into a process-global cache or retains additional compiler objects.
          (when (< @(:entries *context*) max-entries)
            (let [^IdentityHashMap cache (or (.get validators validator) (IdentityHashMap.))]
              (.put cache value checked)
              (.put validators validator cache)
              (vswap! (:entries *context*) inc)))
          checked)))
    (validator value)))
