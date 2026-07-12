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

(defn tag-capable
  "Canonicalize a caller-supplied gensym-fn into the two-arity BindCtx
  contract:
    (f prefix)      — mint a fresh symbol (unchanged behavior)
    (f prefix tag)  — mint a fresh symbol stamped {:raster.type/tag tag}
                      when `tag` is non-nil; nil tag = plain mint, so
                      callers can pass Π's nil (no tangent space / unknown
                      primal tag) without branching.
  Always mints through the wrapped fn's single arity and stamps in the
  wrapper, so any provider (plain gensym closures, reverse.clj's
  ad-gensym, jvp-gensym) yields identical tagging semantics. Idempotent
  under re-wrapping."
  [gensym-fn]
  (fn mint
    ([prefix] (gensym-fn prefix))
    ([prefix tag]
     (let [sym (gensym-fn prefix)]
       (if (some? tag)
         (vary-meta sym assoc :raster.type/tag tag)
         sym)))))

(defn make-ctx
  "Create a fresh binding context.
  gensym-fn: (fn [prefix] -> unique-symbol); wrapped via `tag-capable`
  so (gensym-fn prefix tag) is always available to template grads-fns."
  [gensym-fn]
  {:bindings [] :gensym-fn (tag-capable gensym-fn)})

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
