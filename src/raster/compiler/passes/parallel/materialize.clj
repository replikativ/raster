(ns raster.compiler.passes.parallel.materialize
  "Late pass: materialize pure par/pmap forms into alloc + par/map!.

  Pure par/pmap forms (raster.par/pmap idx bound cast body) are value-producing
  with no output buffer. This pass converts them to:
    count = (clojure.core/int bound)
    out = (alloc count)
    sym = (raster.par/map! out private-idx count cast (let [idx (long private-idx)] body))

  The runtime macro and JVM emitter use this same recipe. Bound evaluation,
  allocation and the source binding's final completion retain their order.

  This runs after SOAC fusion so that fusion operates on pure forms.
  Downstream passes (segop-lower, backend, mem-merge) expect par/map!."
  (:require [raster.compiler.ir.par :as par]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.form :as form]))

(declare materialize-let)

(defn- materialize-pure-map
  "Use the source pmap's one checked count, allocation and store recipe."
  [sym form]
  (let [cast (:cast (par/extract-par-map-pure-info form))]
    (when-not (or (nil? cast) (descriptor/cast-op? cast))
      (throw (ex-info "materialize requires a supported primitive map cast"
                      {:reason :unsupported-pure-map-materialization :sym sym :cast cast})))
    (:bindings (par/pure-map-expansion sym form))))

(defn- materialize-expr
  "Recursively materialize pure par/map forms in an expression.
  For non-let-bound pure par/maps, wraps in let*."
  [expr]
  (cond
    (and (seq? expr) (contains? #{'quote 'clojure.core/quote} (first expr))) expr

    (par/par-map-pure-form? expr)
    (let [result-sym (gensym "result__")
          bindings (materialize-pure-map result-sym expr)]
      (list 'let* (vec (mapcat identity bindings)) result-sym))

    (and (seq? expr) (= 'let* (first expr)))
    (materialize-let expr)

    (seq? expr)
    (with-meta (apply list (map materialize-expr expr)) (meta expr))

    :else expr))

(defn- materialize-let
  "Materialize pure par/map forms in a let* body.
  Splices alloc bindings before the par/map! binding."
  [form]
  (let [[_ bindings & body] form
        pairs (partition 2 bindings)
        new-pairs (mapcat
                   (fn [[sym init]]
                     (if (par/par-map-pure-form? init)
                       (materialize-pure-map sym init)
                       [[sym (materialize-expr init)]]))
                   pairs)
        new-bindings (vec (mapcat identity new-pairs))
        new-body (map materialize-expr body)]
    ;; Preserve the original form's metadata (e.g. ^DoubleVector type tags the
    ;; bytecoder relies on for interop overload selection). Rebuilding via list*
    ;; without this drops the tag, which silently miscompiles a fused vector
    ;; subexpression passed as a method argument (wrong primitive overload).
    (with-meta (list* 'let* new-bindings new-body) (meta form))))

(defn materialize-pass
  "Top-level pass: convert all pure par/map forms to alloc + par/map!.
  Returns {:form :stats}."
  [form _opts]
  (if (form/binding-form? form)
    (let [result (materialize-let form)
          ;; Count how many par/map forms were materialized
          count-pure (atom 0)]
      ;; Count original pure forms
      (letfn [(count-forms [f]
                (when (and (seq? f)
                           (not (contains? #{'quote 'clojure.core/quote} (first f))))
                  (when (par/par-map-pure-form? f)
                    (swap! count-pure inc))
                  (doseq [child (rest f)]
                    (count-forms child))))]
        (count-forms form))
      {:form result :stats {:materialized @count-pure}})
    {:form (materialize-expr form) :stats {:materialized 0}}))
