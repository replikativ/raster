(ns raster.ad.reverse.normalize
  "ANF normalization for reverse-mode AD.

   Flattens walked S-expressions so that:
   - All call arguments are trivial (symbols or literals)
   - If-expression branches are lifted to separate bindings
   - Body is reduced to a single symbol

   This is required before the reverse pass can track operations.")

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
        ;; If expressions: normalize recursively in each position
        (= 'if head)
        (let [[_ test then else] expr
              [te tn] (anf-normalize-expr test gensym-fn)
              [the thn] (anf-normalize-expr then gensym-fn)
              [ee en] (anf-normalize-expr (or else nil) gensym-fn)]
          [(vec (concat @extras te the ee))
           (list 'if tn thn en)])

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

        ;; Let/let*/loop/loop*/dotimes: structural forms, pass through unchanged
        (#{'let 'let* 'loop 'loop* 'dotimes} head)
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
          [@extras (cons head norm-args)])))

    :else [[] expr]))

(defn anf-normalize-bindings
  "ANF-normalize all bindings: lift nested sub-expressions into flat bindings.
  Input: flat bindings vector [s1 e1 s2 e2 ...].
  Output: flat bindings vector with nested calls flattened."
  [bindings gensym-fn]
  (let [result (atom [])]
    (doseq [[sym init] (partition 2 bindings)]
      (let [[extras normalized] (anf-normalize-expr init gensym-fn)]
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
  - If-expression branches in bindings are lifted to separate bindings
  - Branch conditions stored as bindings"
  [bindings body-exprs gensym-fn]
  (let [;; === Step 0: ANF normalize — flatten nested calls ===
        anf-bindings (anf-normalize-bindings bindings gensym-fn)
        [body-extras body-exprs]
        (let [body-expr (if (= 1 (count body-exprs))
                          (first body-exprs)
                          (cons 'do body-exprs))
              [extras norm-body] (anf-normalize-expr body-expr gensym-fn)]
          [extras (list norm-body)])
        bindings (vec (concat anf-bindings body-extras))

        body-expr (if (= 1 (count body-exprs))
                    (first body-exprs)
                    (cons 'do body-exprs))
        result (atom []) ;; accumulates [sym init sym init ...]

        ;; Process existing bindings, lifting if-branches
        _ (doseq [[sym init] (partition 2 bindings)]
            (if (and (seq? init) (= 'if (first init)))
              ;; If-init: lift branches and condition
              (let [[_ test then else] init
                    ;; Lift non-trivial then branch
                    [then-sym] (if (trivial-expr? then)
                                 [then]
                                 (let [s (gensym-fn "then_v")]
                                   (swap! result conj s then)
                                   [s]))
                    ;; Lift non-trivial else branch
                    [else-sym] (if (or (nil? else) (trivial-expr? else))
                                 [(or else nil)]
                                 (let [s (gensym-fn "else_v")]
                                   (swap! result conj s else)
                                   [s]))
                    ;; Store branch condition
                    branch-sym (gensym-fn "br")]
                (swap! result conj
                       branch-sym test
                       sym (list 'if branch-sym then-sym else-sym)))
              ;; Normal binding
              (swap! result conj sym init)))

        ;; Normalize body
        [extra body-sym]
        (cond
          ;; Already a symbol
          (symbol? body-expr)
          [[] body-expr]

          ;; If-body: lift branches, condition, and result
          (and (seq? body-expr) (= 'if (first body-expr)))
          (let [[_ test then else] body-expr
                extra (atom [])
                then-sym (if (trivial-expr? then)
                           then
                           (let [s (gensym-fn "then_v")]
                             (swap! extra conj s then)
                             s))
                else-sym (if (or (nil? else) (trivial-expr? else))
                           (or else nil)
                           (let [s (gensym-fn "else_v")]
                             (swap! extra conj s else)
                             s))
                branch-sym (gensym-fn "br")
                result-sym (gensym-fn "if_r")]
            [(vec (concat @extra [branch-sym test
                                  result-sym (list 'if branch-sym then-sym else-sym)]))
             result-sym])

          ;; Other expression: bind it
          :else
          (let [s (gensym-fn "body")]
            [[s body-expr] s]))]

    [(vec (concat @result extra)) body-sym]))
