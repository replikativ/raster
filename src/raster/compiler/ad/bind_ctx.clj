(ns raster.compiler.ad.bind-ctx
  "Binding context for flat code generation.

  Templates and AD passes thread a BindCtx to accumulate let* bindings.
  This ensures all allocations appear as top-level bindings (visible to
  fusion) rather than nested inside let blocks.

  Usage:
    (let [ctx (make-ctx gensym-fn)
          [ctx sym] (genlet ctx \"buf\" '(nc/raw x))]
      (emit-let* ctx sym))
    ;; => (let* [buf__1 (nc/raw x)] buf__1)")

(defn make-ctx
  "Create a fresh binding context.
  gensym-fn: (fn [prefix] -> unique-symbol)"
  [gensym-fn]
  {:bindings [] :gensym-fn gensym-fn})

(defn genlet
  "Add a binding to the context. Returns [updated-ctx, sym].
  prefix: string prefix for gensym'd name.
  expr: the expression to bind."
  [ctx prefix expr]
  (let [sym ((:gensym-fn ctx) prefix)]
    [(update ctx :bindings into [sym expr]) sym]))

(defn emit-let*
  "Convert BindCtx + body into a let* S-expression.
  If no bindings, returns body directly."
  [ctx body]
  (if (empty? (:bindings ctx))
    body
    (list 'let* (vec (:bindings ctx)) body)))
