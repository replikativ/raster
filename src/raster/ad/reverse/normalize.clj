(ns raster.ad.reverse.normalize
  "ANF normalization for reverse-mode AD.

   Flattens walked S-expressions so that:
   - All call arguments are trivial (symbols or literals)
   - If-expression arms retain their lexical evaluation regions
   - Body is reduced to a single symbol

   This is required before the reverse pass can track operations."
  (:require [raster.compiler.ir.form :as form]))

(defn trivial-expr?
  "True if expr is simple enough to not need its own binding."
  [x]
  (or (symbol? x) (number? x) (nil? x) (boolean? x) (string? x) (keyword? x)))

(defn anf-normalize-expr
  "Flatten nested sub-expressions in a call into separate let bindings.
  Returns [extra-bindings-vec, normalized-expr] where all call arguments
  are trivial (symbols or literals).
  extra-bindings is a flat vector [sym1 init1 sym2 init2 ...].
  gensym-fn: (fn [prefix] -> unique-symbol)"
  [expr gensym-fn]
  (cond
    ;; Trivial expressions need no lifting
    (trivial-expr? expr) [[] expr]

    ;; Function call: lift non-trivial arguments
    (seq? expr)
    (let [head (first expr)
          extras (atom [])]
      (cond
        ;; An if is a region boundary, not a call with eagerly evaluated arguments.
        ;; Normalize and snapshot the predicate only; each AD mode transforms the selected
        ;; arm locally. Purity does not license speculation (sqrt/checked reads are partial).
        (= 'if head)
        (let [[_ test then else] expr
              [te tn] (anf-normalize-expr test gensym-fn)
              decision (if (trivial-expr? tn) tn
                           (with-meta (gensym-fn "br") (meta test)))]
          [(cond-> (vec te) (not (trivial-expr? tn)) (into [decision tn]))
           (with-meta (list 'if decision then else) (meta expr))])

        ;; Do expressions: flatten side-effect forms, return last
        (= 'do head)
        (let [sub-exprs (rest expr)]
          (doseq [se (butlast sub-exprs)]
            (let [[sub-extras normalized] (anf-normalize-expr se gensym-fn)
                  s (gensym-fn "_side")]
              (swap! extras into (concat sub-extras [s normalized]))))
          (let [last-expr (last sub-exprs)
                [le ln] (anf-normalize-expr last-expr gensym-fn)]
            [(vec (concat @extras le)) ln]))

        ;; Scope-introducing forms (let*/loop*/dotimes/fn*/ftm AND every par/*
        ;; SOAC) pass through unchanged — their bodies bind their own locals (loop
        ;; index, accumulator), so lifting a sub-expr OUT of the scope would let a
        ;; scoped var escape. Use the unified scope classifier, not a hardcoded set,
        ;; so par forms are covered (and any new binder is automatically handled).
        (or (form/binding-form? expr) (form/introduces-scope? expr))
        [[] expr]

        ;; Regular call: lift non-trivial args
        :else
        (let [args (rest expr)
              norm-args
              (mapv (fn [arg]
                      (if (trivial-expr? arg)
                        arg
                        (let [[sub-extras normalized] (anf-normalize-expr arg gensym-fn)
                              s (gensym-fn "anf")]
                          (swap! extras into (concat sub-extras [s normalized]))
                          s)))
                    args)]
          [@extras (with-meta (cons head norm-args) (meta expr))])))

    :else [[] expr]))

(defn anf-normalize-bindings
  "ANF-normalize all bindings: lift nested sub-expressions into flat bindings.
  Input: flat bindings vector [s1 e1 s2 e2 ...].
  Output: flat bindings vector with nested calls flattened."
  [bindings gensym-fn]
  (let [result (atom [])]
    (doseq [[sym init] (partition 2 bindings)]
      (let [;; Shared hoisting can retain call identity on its binding rather than the
            ;; reconstructed initializer. Project that retained identity, never decode an impl.
            init (if (and (seq? init) (= '.invk (first init))
                          (nil? (:raster.op/original (meta init)))
                          (:raster.op/original (meta sym)))
                   (vary-meta init assoc :raster.op/original
                              (:raster.op/original (meta sym)))
                   init)
            [extras normalized] (anf-normalize-expr init gensym-fn)]
        (swap! result into extras)
        (swap! result conj sym normalized)))
    @result))

(defn normalize-for-ad
  "Normalize a let form for uniform AD processing.
  Input: flat bindings vector [s1 e1 s2 e2 ...], body-exprs list, gensym-fn.
  Output: [flat-bindings-vector, body-symbol]

  Guarantees:
  - Body is a single symbol
  - All call arguments are trivial (symbols or literals) — ANF
  - Conditional arms remain lexical regions, never eager sibling bindings
  - Nontrivial branch conditions are snapshotted before the selected arm"
  [bindings body-exprs gensym-fn]
  (let [anf-bindings (anf-normalize-bindings bindings gensym-fn)
        body-expr (if (= 1 (count body-exprs))
                    (first body-exprs)
                    (cons 'do body-exprs))
        [extras normalized] (anf-normalize-expr body-expr gensym-fn)
        body-sym (if (symbol? normalized) normalized
                     (with-meta (gensym-fn "body") (meta body-expr)))]
    [(cond-> (into (vec anf-bindings) extras)
       (not (symbol? normalized)) (into [body-sym normalized]))
     body-sym]))
