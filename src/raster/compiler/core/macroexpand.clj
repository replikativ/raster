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
   ;; `quote` is opaque: its body is code-as-data, not code. The walker classifies
   ;; (quote ...) as a :leaf and never descends; we must match that contract here, or
   ;; macroexpand would rewrite macros/classes sitting in head position INSIDE quoted
   ;; data (e.g. '(let [a 1] a) -> '(let* [a 1] a)), silently corrupting it.
   (if (and (seq? form) (= 'quote (first form)))
     form
     (let [expanded (if (and (seq? form)
                             (not (and (symbol? (first form)) (skip-head? (first form)))))
                      (macroexpand form)
                      form)
           m (meta form)
           result (walk-preserving-meta #(macroexpand-all-preserving % skip-head?)
                                        identity expanded)]
       (if (and m (instance? clojure.lang.IObj result))
         (with-meta result (merge (meta result) m))
         result)))))

(defn core-skip-head?
  "Heads PRESERVED by macroexpand-core (not expanded at top level; children still
  expanded): interop forms (`.method`/`.-field`/`.invk`, head starts with `.`),
  the `dotimes` counted-loop primitive (the SIMD/loop-lift matchers key on it),
  `ftm` closures, and every `raster.par/*` SOAC primitive — these last are MACROS
  that would otherwise expand to scalar loops, destroying the structured form the
  fusion/segop/SIMD/GPU passes consume. Everything else in the macro zoo
  (when/and/or/cond/-> and let/loop/case/fn) expands to the closed core."
  [h]
  (and (symbol? h)
       (or (.startsWith (name h) ".")
           (= 'dotimes h)
           (= 'ftm h)
           ;; raster.par or raster.par.* — NOT raster.params (startsWith would
           ;; wrongly match the params ns and preserve its defmodel macro).
           (let [ns (namespace h)]
             (and ns (or (= ns "raster.par") (.startsWith ^String ns "raster.par.")))))))

(defn macroexpand-core
  "Macroexpand the open-ended macro zoo to a CLOSED CORE — `let*`/`loop*`/`fn*`/
  `if`/`do`/`case*`/`try`/`recur` + special forms — while PRESERVING the SOAC
  primitives (`raster.par/*`), `dotimes`, `ftm`, and interop. Run right after the
  walker so every downstream pass and the AD see a closed, scope-tractable IR
  (the finite binding-form set the scope foundation depends on). MUST run with
  `*ns*` bound to the source ns so user macros resolve."
  [form]
  (macroexpand-all-preserving form core-skip-head?))

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
