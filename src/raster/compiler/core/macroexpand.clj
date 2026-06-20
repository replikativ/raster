(ns raster.compiler.core.macroexpand
  "Metadata-preserving macroexpansion, shared by code-gen backends.

   The walker is a non-macroexpanding source rewriter (it keeps `let`/`loop`/`case`/
   `or`/`and` etc. as high-level forms so passes can pattern-match them). Each
   backend macroexpands at the LAST moment, before emission, turning those into the
   ~12 special forms (`let*`/`loop*`/`case*`/`if`). `clojure.walk/macroexpand-all`
   would strip metadata (the :raster.type/tag, :raster.op/original the backends
   rely on), so we re-implement it preserving meta."
  (:refer-clojure :exclude []))

(defn walk-preserving-meta
  "Like clojure.walk/walk but preserves metadata on forms."
  [inner outer form]
  (let [m (meta form)
        result (cond
                 (list? form)   (outer (apply list (map inner form)))
                 (seq? form)    (outer (doall (map inner form)))
                 (vector? form) (outer (mapv inner form))
                 (map? form)    (outer (into (empty form)
                                             (map (fn [[k v]] [(inner k) (inner v)]) form)))
                 (set? form)    (outer (into (empty form) (map inner form)))
                 :else          (outer form))]
    (if (and m (instance? clojure.lang.IObj result))
      (with-meta result (merge (meta result) m))
      result)))

(defn macroexpand-all-preserving
  "Like clojure.walk/macroexpand-all but preserves metadata on forms.

   `skip-head?` (optional) is a predicate on a form's head symbol; when it returns
   true that form is NOT macroexpanded at its top level (but its children still
   are). Backends use it to keep macros they lower specially — e.g. the wasm
   backend keeps `dotimes` so its SIMD vectorizer can match the counted-loop
   shape instead of the generic `loop*` expansion."
  ([form] (macroexpand-all-preserving form (constantly false)))
  ([form skip-head?]
   (let [expanded (if (and (seq? form)
                           (not (and (symbol? (first form)) (skip-head? (first form)))))
                    (macroexpand form)
                    form)
         m (meta form)
         result (walk-preserving-meta #(macroexpand-all-preserving % skip-head?)
                                      identity expanded)]
     (if (and m (instance? clojure.lang.IObj result))
       (with-meta result (merge (meta result) m))
       result))))

(defn resugar-interop
  "Re-sugar canonical `(. obj method args...)` interop forms back into the
   `(.method obj args...)` reader-sugar form, preserving metadata.

   `macroexpand` canonicalizes the walker's devirtualized `(.invk impl a b)`
   dispatch nodes into `(. impl invk a b)`. The JVM bytecode backend matches the
   canonical `.` form directly, but the wasm backend (which has no general JVM
   interop) matches the `.method` sugar. Run this after macroexpand-all-preserving
   on the wasm path to restore the form the emitter expects."
  [form]
  (cond
    ;; flat form: (. obj method args...)
    (and (seq? form) (= '. (first form)) (>= (count form) 3) (symbol? (nth form 2)))
    (let [[_ obj method & args] form
          obj*  (resugar-interop obj)
          args* (map resugar-interop args)]
      (with-meta (apply list (symbol (str "." (name method))) obj* args*)
        (meta form)))

    ;; nested form: (. obj (method args...))
    (and (seq? form) (= '. (first form)) (= 3 (count form))
         (seq? (nth form 2)) (symbol? (first (nth form 2))))
    (let [[_ obj call] form
          method (first call)
          obj*  (resugar-interop obj)
          args* (map resugar-interop (rest call))]
      (with-meta (apply list (symbol (str "." (name method))) obj* args*)
        (meta form)))

    (seq? form)   (with-meta (apply list (map resugar-interop form)) (meta form))
    (vector? form) (with-meta (mapv resugar-interop form) (meta form))
    (map? form)   (with-meta (into (empty form)
                                   (map (fn [[k v]] [(resugar-interop k) (resugar-interop v)]) form))
                    (meta form))
    :else form))
