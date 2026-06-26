(ns raster.compiler.passes.parallel.materialize
  "Late pass: materialize pure par/pmap forms into alloc + par/map!.

  Pure par/pmap forms (raster.par/pmap idx bound cast body) are value-producing
  with no output buffer. This pass converts them to:
    out = (alloc bound)
    sym = (raster.par/map! out idx bound cast body)

  This runs after SOAC fusion so that fusion operates on pure forms.
  Downstream passes (segop-lower, backend, mem-merge) expect par/map!."
  (:require [raster.compiler.ir.par :as par]
            [raster.compiler.ir.form :as form]))

(declare materialize-let)

(def ^:private elem-type->alloc-fn
  {:float  'clojure.core/float-array
   :double 'clojure.core/double-array
   :long   'clojure.core/long-array
   :int    'clojure.core/int-array})

(def ^:private elem-type->array-tag
  {:float  'floats
   :double 'doubles
   :long   'longs
   :int    'ints})

(defn- materialize-pure-map
  "Convert a pure par/map form to an [alloc-binding map!-binding] pair.
  Returns [[out-sym alloc-expr] [result-sym map!-expr]] where result-sym
  is the original let-binding symbol."
  [sym form]
  (let [{:keys [idx bound cast body elem-type]} (par/extract-par-map-pure-info form)
        ;; Derive elem-type from cast-fn if metadata doesn't carry it
        elem-type (or elem-type
                      (case cast
                        (float clojure.core/float) :float
                        (long clojure.core/long) :long
                        (int clojure.core/int) :int
                        (double clojure.core/double) :double
                        (throw (ex-info (str "materialize: cannot determine element type for par/map with cast `"
                                             cast "`. Ensure walker stamps :elem-type metadata.")
                                        {:sym sym :cast cast}))))
        alloc-fn (or (get elem-type->alloc-fn elem-type)
                     (throw (ex-info (str "materialize: no alloc-fn for elem-type " elem-type)
                                     {:elem-type elem-type})))
        array-tag (or (get elem-type->array-tag elem-type)
                      (throw (ex-info (str "materialize: no array-tag for elem-type " elem-type)
                                      {:elem-type elem-type})))
        out-sym (with-meta (gensym "out__")
                  {:tag array-tag
                   :raster.type/tag array-tag
                   :raster.buffer/hoistable true})
        map!-form (with-meta
                    (list 'raster.par/map! out-sym idx bound cast body)
                    (meta form))]
    [[out-sym (list alloc-fn bound)]
     [sym map!-form]]))

(defn- materialize-expr
  "Recursively materialize pure par/map forms in an expression.
  For non-let-bound pure par/maps, wraps in let*."
  [expr]
  (cond
    (par/par-map-pure-form? expr)
    (let [result-sym (gensym "result__")
          [[out-sym alloc-expr] [_ map!-form]] (materialize-pure-map result-sym expr)]
      (list 'let* [out-sym alloc-expr
                   result-sym map!-form]
            result-sym))

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
                       (let [[[out-sym alloc-expr] [sym' map!-form]]
                             (materialize-pure-map sym init)]
                         [[out-sym alloc-expr] [sym' map!-form]])
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
                (when (seq? f)
                  (when (par/par-map-pure-form? f)
                    (swap! count-pure inc))
                  (doseq [child (rest f)]
                    (count-forms child))))]
        (count-forms form))
      {:form result :stats {:materialized @count-pure}})
    {:form (materialize-expr form) :stats {:materialized 0}}))
