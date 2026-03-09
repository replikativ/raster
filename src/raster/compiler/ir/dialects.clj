;; {:clj-kondo/config {:skip-files ["src/raster/compiler/ir/dialects.clj"]}}
(ns raster.compiler.ir.dialects
  "Formal dialect definitions for each compiler pipeline stage.

  Each dialect defines the grammar of S-expressions valid at that point
  in the pipeline, using the pattern library's nanopass system.

  Dialects (IR stages):
    Walked → Lowered → ADTransformed → Flattened → Simplified → Rewalked

  Usage:
    (require '[raster.compiler.ir.dialects :as d])
    (d/valid-walked? form)      ;; quick predicate
    (d/validate-walked form)    ;; detailed error on failure

  The validate-* functions return :ok on success, or a map with
  :fail and :details keys describing the structural violation."
  (:require [pattern.nanopass.dialect :as dialect
             :refer [def-dialect def-derived valid? validate]]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.par :as par]
            [clojure.set :as set]))

;; ================================================================
;; Base dialect: Walked S-expressions (output of raster.compiler.core.walker/walk-body)
;; ================================================================

(def-dialect Walked
  (terminals [sym symbol?]
             [num number?]
             [kw keyword?]
             [str string?]
             [bool boolean?])
  ;; nil is handled as a literal pattern below

  (Expr [e]
    ;; Terminals
        ?num ?sym ?kw ?str ?bool
        nil

    ;; Let binding (the main structural form)
        (let* [(?:* ?sym ?e)] ?e:body)
        (let  [(?:* ?sym ?e)] ?e:body)

    ;; Control flow
        (if ?e:test ?e:then ?e:else)
        (do (?:+ e))

    ;; Loops
        (loop [(?:* ?sym ?e)] ?e:body)
        (loop* [(?:* ?sym ?e)] ?e:body)
        (recur (?:+ e))
        (dotimes [?sym ?e:bound] ?e:body)

    ;; Devirtualized method call
        (.invk ?sym (?:* e))

    ;; Array ops (may appear as qualified or unqualified)
        (aget ?e:arr ?e:idx)
        (aset ?e:arr ?e:idx ?e:val)
        (alength ?e:arr)
        (clojure.core/aget ?e:arr ?e:idx)
        (clojure.core/aset ?e:arr ?e:idx ?e:val)
        (clojure.core/alength ?e:arr)

    ;; Array constructors
        (double-array ?e)
        (float-array ?e)
        (long-array ?e)
        (int-array ?e)
        (clojure.core/double-array ?e)
        (clojure.core/float-array ?e)
        (clojure.core/long-array ?e)
        (clojure.core/int-array ?e)

    ;; Parallel forms
        (broadcast ?e:fn ?e:n)
        (reduce! ?e:fn ?e:init ?e:coll)

    ;; Java interop
        (new ?sym (?:* e))
        (. ?e:obj ?sym (?:* e:args))
        (throw ?e)

    ;; Primitive casts
        (double ?e)
        (float ?e)
        (long ?e)
        (int ?e)

    ;; Misc
        (quote ?e)

    ;; Catch-all: any function call (qualified symbols, etc.)
    ;; This must be last so specific patterns match first
        (& (?e:f (?:* e:args)) (? _ seq?)))

  (entry Expr))

;; ================================================================
;; After lower (pre-AD inlining): same structure, semantically narrower
;; ================================================================

(def-derived Lowered Walked
  (entry Expr))

;; ================================================================
;; AD output: canonical let* with [primal, pullback-fn] body
;; ================================================================
;;
;; The AD pass canonicalizes its output before dialect validation, so the
;; compiler sees a single shape here:
;;   (let* [...] [primal (fn* [dy] ...)])

(def-dialect ADTransformed
  (terminals [sym symbol?]
             [num number?]
             [kw keyword?]
             [str string?]
             [bool boolean?])

  (Expr [e]
    ;; All Walked forms are valid in AD context too
        ?num ?sym ?kw ?str ?bool nil
        (let* [(?:* ?sym ?e)] ?e:body)
        (if ?e:test ?e:then ?e:else)
        (do (?:+ e))
        (loop [(?:* ?sym ?e)] ?e:body)
        (loop* [(?:* ?sym ?e)] ?e:body)
        (recur (?:+ e))
        (dotimes [?sym ?e:bound] ?e:body)
        (.invk ?sym (?:* e))
        (aget ?e:arr ?e:idx)
        (aset ?e:arr ?e:idx ?e:val)
        (alength ?e:arr)
        (double-array ?e) (float-array ?e) (long-array ?e) (int-array ?e)
        (new ?sym (?:* e))
        (. ?e:obj ?sym (?:* e:args))
        (throw ?e)
        (double ?e) (float ?e) (long ?e) (int ?e)
        (quote ?e)

    ;; AD-specific: pullback closure
        (fn* [?sym] ?e:body)

    ;; AD-specific: vector results
        [(?:+ e)]

    ;; AD-specific: nth for unpacking
        (nth ?e ?num)

    ;; Qualified raster.numeric ops (from forward pass qualification)
        (raster.numeric/+ (?:* e))
        (raster.numeric/- (?:* e))
        (raster.numeric/* (?:* e))
        (raster.numeric// (?:* e))
        (raster.math/sin ?e) (raster.math/cos ?e)
        (raster.math/exp ?e) (raster.math/log ?e)
        (raster.math/tan ?e)
        (raster.numeric/sqrt ?e) (raster.numeric/pow ?e ?e)
        (raster.numeric/abs ?e)

    ;; Catch-all for any other call
        (& (?e:f (?:* e:args)) (? _ seq?)))

  (entry Expr))

;; ================================================================
;; Flattened: single let* form, no closures
;; ================================================================

(def-derived Flattened Walked
  (Expr [e]
    ;; Remove parallel forms (expanded before AD)
        - (broadcast ?e:fn ?e:n)
        - (reduce! ?e:fn ?e:init ?e:coll)

    ;; Add AD-qualified ops that survive flattening
        + (raster.numeric/+ (?:* e))
        + (raster.numeric/- (?:* e))
        + (raster.numeric/* (?:* e))
        + (raster.numeric// (?:* e))
        + (raster.math/sin ?e) + (raster.math/cos ?e)
        + (raster.math/exp ?e) + (raster.math/log ?e)
        + (raster.math/tan ?e)
        + (raster.numeric/sqrt ?e) + (raster.numeric/pow ?e ?e)
        + (raster.numeric/abs ?e)

    ;; SGD updates
        + (raster.numeric/axpy! ?e ?e ?e)
        + (raster.numeric/axpy! ?e ?e ?e ?e)

    ;; Vector results for gradient output
        + [(?:+ e)]))

;; ================================================================
;; After CSE: structurally identical to Flattened (duplicate bindings removed)
;; ================================================================

(def-derived CSEEliminated Flattened
  (entry Expr))

;; ================================================================
;; After scalar normalization: structurally identical (call shapes canonicalized)
;; ================================================================

(def-derived Normalized CSEEliminated
  (entry Expr))

;; ================================================================
;; After algebraic simplification: structurally identical (expressions rewritten)
;; ================================================================

(def-derived Simplified Normalized
  (entry Expr))

;; ================================================================
;; After PE + re-walk: devirtualized calls
;; ================================================================

(def-derived Rewalked Simplified
  (entry Expr))

;; ================================================================
;; Pragmatic validation functions
;; ================================================================
;;
;; The dialect definitions above serve as formal grammar specifications.
;; The functions below provide pragmatic validation that works well
;; with our walked S-expressions. We use a combination of dialect
;; validation (where the pattern library handles it well) and
;; structural checks (for edge cases like AD output shape).

(defn valid-walked?
  "Check if form is a valid walked S-expression."
  [form]
  (or (valid? Walked form)
      ;; Fallback: at minimum check structural shape
      (form/binding-form? form)))

(defn validate-walked
  "Validate a walked form. Returns :ok or error details."
  [form]
  (let [result (validate Walked form)]
    (if (= result (pattern.types/->Ok))
      :ok
      {:fail :walked
       :details result
       :form-head (when (seq? form) (first form))})))

(defn valid-ad-transformed?
  "Check if form is canonical AD output.
  The AD contract at this stage is a let* whose body is
  [primal (fn* [dy] ...)]."
  [form]
  (and (seq? form)
       (= 'let* (first form))
       (vector? (second form))
       (let [body (last (drop 2 form))]
         (and (vector? body)
              (= 2 (count body))
              (symbol? (first body))
              (let [pb (second body)]
                (and (seq? pb) (= 'fn* (first pb))))))))

(defn validate-ad-transformed
  "Validate AD-transformed output. Returns :ok or error details."
  [form]
  (if (valid-ad-transformed? form)
    :ok
    (let [details (cond
                    (not (seq? form))
                    {:expected "canonical let* AD form"
                     :actual-type (type form)}

                    (not= 'let* (first form))
                    {:expected "let* head"
                     :actual (first form)}

                    (not (vector? (second form)))
                    {:expected "vector bindings"
                     :actual-type (type (second form))}

                    :else
                    (let [body (last (drop 2 form))]
                      {:expected "let* body should be [primal (fn* [dy] ...)]"
                       :actual-body body
                       :body-type (type body)}))]
      {:fail :ad-transformed
       :details details})))

(defn valid-flattened?
  "Check if form is a valid flattened form (single let*, no closures)."
  [form]
  (and (seq? form)
       (= 'let* (first form))
       (vector? (second form))
       ;; Body should be a symbol (result) or vector (gradient output)
       (let [body (last (drop 2 form))]
         (or (symbol? body) (vector? body)))))

(defn validate-flattened
  "Validate flattened form. Returns :ok or error details."
  [form]
  (if (valid-flattened? form)
    :ok
    {:fail :flattened
     :details (cond
                (not (seq? form))
                {:expected "seq (let* ...)" :actual-type (type form)}

                (not= 'let* (first form))
                {:expected "let* head" :actual (first form)}

                (not (vector? (second form)))
                {:expected "vector bindings" :actual-type (type (second form))}

                :else
                (let [body (last (drop 2 form))]
                  {:expected "symbol or vector body"
                   :actual-type (type body)
                   :actual body}))}))

(defn valid-let*?
  "Check if form is a valid let* form with proper bindings structure.
   Validates iteratively to handle forms with hundreds of bindings."
  [form]
  (and (seq? form)
       (= 'let* (first form))
       (vector? (second form))
       (even? (count (second form)))
       (every? symbol? (take-nth 2 (second form)))))

(defn validate-let*
  "Validate that form is a let* form with proper structure."
  [form]
  (if (valid-let*? form)
    :ok
    {:fail :let*
     :details {:expected "let* form"
               :actual-type (type form)
               :form-head (when (seq? form) (first form))}}))

;; ================================================================
;; Use-before-def validation for let* binding ordering
;; ================================================================

(defn- collect-free-syms
  "Collect free (unbound) symbols from an S-expression.
  Handles let*, loop*, fn*, if, do, quote, try/catch, .invk, dotimes."
  [expr bound]
  (cond
    (symbol? expr) (if (or (contains? bound expr)
                           (namespace expr))    ;; qualified syms are resolved externally
                     #{}
                     #{expr})
    (not (sequential? expr)) #{}
    :else
    (let [head (first expr)]
      (cond
        (#{'let 'let*} head)
        (let [pairs (partition 2 (second expr))
              [init-fvs new-bound]
              (reduce (fn [[fvs b] [sym init]]
                        [(set/union fvs (collect-free-syms init b)) (conj b sym)])
                      [#{} bound] pairs)]
          (reduce set/union init-fvs
                  (map #(collect-free-syms % new-bound) (nnext expr))))

        ;; loop/loop*: init expressions are sequential (like let*), but ALL
        ;; loop bindings are in scope for the body (and recur).
        (#{'loop 'loop*} head)
        (let [pairs (partition 2 (second expr))
              ;; Init expressions: sequential scoping (like let*)
              [init-fvs new-bound]
              (reduce (fn [[fvs b] [sym init]]
                        [(set/union fvs (collect-free-syms init b)) (conj b sym)])
                      [#{} bound] pairs)]
          ;; Body: all loop vars in scope
          (reduce set/union init-fvs
                  (map #(collect-free-syms % new-bound) (nnext expr))))

        (#{'fn 'fn*} head)
        (let [arities (if (vector? (second expr)) (list (rest expr)) (rest expr))]
          (reduce set/union #{}
                  (map (fn [arity]
                         (let [params (first arity)
                               param-set (if (vector? params) (set params) #{})]
                           (reduce set/union #{}
                                   (map #(collect-free-syms % (set/union bound param-set))
                                        (rest arity)))))
                       arities)))

        (= head 'if) (let [[_ t th el] expr]
                       (set/union (collect-free-syms t bound)
                                  (collect-free-syms th bound)
                                  (if el (collect-free-syms el bound) #{})))
        (= head 'do) (reduce set/union #{} (map #(collect-free-syms % bound) (rest expr)))
        (= head 'quote) #{}
        (= head 'dotimes)
        (let [[_ [sym bound-expr] & body] expr]
          (set/union (collect-free-syms bound-expr bound)
                     (reduce set/union #{}
                             (map #(collect-free-syms % (conj bound sym)) body))))
        (#{'new 'throw 'recur} head)
        (reduce set/union #{} (map #(collect-free-syms % bound) (rest expr)))
        (= head '.invk) (reduce set/union #{} (map #(collect-free-syms % bound) (rest expr)))
        (= head '.) (let [[_ target _method & args] expr]
                      (set/union (collect-free-syms target bound)
                                 (reduce set/union #{} (map #(collect-free-syms % bound) args))))
        (= head 'try)
        (reduce set/union #{}
                (map (fn [c]
                       (if (and (sequential? c) (= 'catch (first c)))
                         (let [[_ _cls sym & body] c]
                           (reduce set/union #{} (map #(collect-free-syms % (conj bound sym)) body)))
                         (collect-free-syms c bound)))
                     (rest expr)))
        ;; par forms — use par-scope-info for scope-aware handling
        (and (symbol? head)
             (some? (namespace head))
             (.startsWith ^String (namespace head) "raster.par"))
        (if-let [{:keys [scoped-syms inner-exprs outer-exprs]} (par/par-scope-info expr)]
          (let [scoped-set (set (filter symbol? scoped-syms))
                body-bound (set/union bound scoped-set)
                outer-frees (reduce set/union #{}
                                    (map #(collect-free-syms % bound) outer-exprs))
                inner-frees (reduce set/union #{}
                                    (map #(collect-free-syms % body-bound) inner-exprs))]
            (set/union outer-frees inner-frees))
          ;; No scope info (scatter!, rng-fill!, etc.) — recurse into all args
          (reduce set/union #{} (map #(collect-free-syms % bound) (rest expr))))

        :else (reduce set/union #{} (map #(collect-free-syms % bound) expr))))))

(defn find-forward-refs
  "Find forward references in a let* form: bindings whose init expressions
  reference symbols defined later in the same let*. Returns nil if none found,
  or a vector of {:binding sym :references [forward-ref-sym ...] :positions [i j]}."
  [form]
  (when (and (seq? form) (= 'let* (first form)))
    (let [pairs (vec (partition 2 (second form)))
          ;; Map each binding symbol to its position
          sym-positions (into {} (map-indexed (fn [i [sym _]] [sym i]) pairs))
          ;; Track cumulative bound symbols
          violations
          (loop [i 0, bound #{}, result []]
            (if (>= i (count pairs))
              result
              (let [[sym init] (nth pairs i)
                    ;; Collect free syms in init (respecting inner scopes)
                    free (collect-free-syms init bound)
                    ;; Find any that are defined later in this let*
                    forward (filter (fn [s]
                                      (let [pos (get sym-positions s)]
                                        (and pos (> pos i))))
                                    free)]
                (recur (inc i)
                       (conj bound sym)
                       (if (seq forward)
                         (conj result {:binding sym
                                       :binding-pos i
                                       :references (vec forward)
                                       :ref-positions (mapv #(get sym-positions %) forward)})
                         result)))))]
      (when (seq violations)
        violations))))

(defn valid-let*-ordering?
  "Check if a let* form has valid binding ordering (no forward references)
  at the top level. Does NOT recurse into nested let* — each let* is
  validated independently when it appears as a binding init."
  [form]
  (nil? (find-forward-refs form)))

(defn validate-let*-ordering
  "Validate binding ordering in a let* form. Returns :ok or error details."
  [form]
  (if-let [violations (find-forward-refs form)]
    {:fail :forward-reference
     :details {:message "Binding references symbol defined later in same let*"
               :violations violations}}
    :ok))

(defn valid-let*-ordered?
  "Valid let* structure AND no forward references in bindings."
  [form]
  (and (valid-let*? form)
       (valid-let*-ordering? form)))

(defn validate-let*-ordered
  "Validate both let* structure and binding ordering."
  [form]
  (let [struct-result (validate-let* form)]
    (if (not= :ok struct-result)
      struct-result
      (validate-let*-ordering form))))

(def ^:private raw-scalar-call-heads
  #{'+ '- '* '/
    'clojure.core/+ 'clojure.core/- 'clojure.core/* 'clojure.core//
    'raster.numeric/+ 'raster.numeric/- 'raster.numeric/* 'raster.numeric//
    'raster.numeric/sqrt 'raster.numeric/pow
    'Math/sin 'Math/cos 'Math/exp 'Math/log 'Math/sqrt 'Math/pow
    'Math/abs 'Math/tan 'Math/asin 'Math/acos 'Math/atan 'Math/atan2
    'Math/min 'Math/max 'Math/fma 'Math/signum 'Math/floor 'Math/ceil})

(defn- raw-scalar-call-head?
  [head]
  (contains? raw-scalar-call-heads head))

(defn- find-raw-scalar-call
  "Return the first raw scalar call that should have been devirtualized in a
  rewalked form, or nil if none are found."
  [form]
  (cond
    (not (coll? form)) nil
    (seq? form)
    (let [head (first form)]
      (cond
        (= '.invk head) (some find-raw-scalar-call (rest form))
        (raw-scalar-call-head? head) form
        :else (some find-raw-scalar-call form)))
    (vector? form) (some find-raw-scalar-call form)
    (map? form) (some find-raw-scalar-call (mapcat identity form))
    (set? form) (some find-raw-scalar-call form)
    :else nil))

(defn valid-rewalked?
  "Check if form matches the expected post-rewalk contract.
  Non-effect bindings and the body should no longer contain raw scalar calls
  that the walker is expected to devirtualize."
  [form]
  (and (valid-let*? form)
       (let [pairs (partition 2 (second form))
             effect-binding? (fn [[sym _]]
                               (and (symbol? sym)
                                    (or (:raster.effect/effectful (meta sym))
                                        (.startsWith ^String (name sym) "_"))))
             walkable-exprs (concat (map second (remove effect-binding? pairs))
                                    [(last (drop 2 form))])]
         (not-any? find-raw-scalar-call walkable-exprs))))

(defn validate-rewalked
  "Validate the post-rewalk contract. Returns :ok or error details."
  [form]
  (if (valid-rewalked? form)
    :ok
    (let [pairs (when (valid-let*? form) (partition 2 (second form)))
          effect-binding? (fn [[sym _]]
                            (and (symbol? sym)
                                 (or (:raster.effect/effectful (meta sym))
                                     (.startsWith ^String (name sym) "_"))))
          walkable-exprs (when pairs
                           (concat (map second (remove effect-binding? pairs))
                                   [(last (drop 2 form))]))
          offending (some find-raw-scalar-call walkable-exprs)]
      {:fail :pe-rewalked
       :details (cond
                  (not (valid-let*? form))
                  {:expected "let* form"
                   :actual-type (type form)
                   :form-head (when (seq? form) (first form))}

                  offending
                  {:expected "devirtualized scalar calls (.invk or non-scalar direct calls)"
                   :offending offending}

                  :else
                  {:expected "valid rewalked form"})})))

;; ================================================================
;; Dialect registry for pipeline integration
;; ================================================================

(def dialect-checkers
  "Map from dialect keyword to [valid? validate] function pairs.
  Used by run-passes to validate IR at pass boundaries."
  {:walked           [valid-walked?       validate-walked]
   :lowered          [valid-walked?      validate-walked]
   :ad-transformed   [valid-ad-transformed? validate-ad-transformed]
   :flattened        [valid-flattened?    validate-flattened]
   :cse-eliminated   [valid-let*?         validate-let*]
   :normalized       [valid-let*?         validate-let*]
   :simplified       [valid-let*?         validate-let*]
   :pe-rewalked      [valid-rewalked?     validate-rewalked]
   :peepholed        [valid-let*?         validate-let*]
   :expanded         [valid-let*?         validate-let*]
   :fixpointed       [valid-let*?         validate-let*]
   ;; Late-stage passes: valid let* + binding ordering check
   :dce-cleaned      [valid-let*-ordered? validate-let*-ordered]
   :buffer-fused     [valid-let*-ordered? validate-let*-ordered]
   :late-cleaned     [valid-let*-ordered? validate-let*-ordered]
   :loop-lifted      [valid-let*-ordered? validate-let*-ordered]
   :write-read-fused [valid-let*-ordered? validate-let*-ordered]
   :par-fused        [valid-let*-ordered? validate-let*-ordered]
   :soac-fused       [valid-let*-ordered? validate-let*-ordered]
   :materialized     [valid-let*-ordered? validate-let*-ordered]
   :segop-lowered    [valid-let*-ordered? validate-let*-ordered]
   :compound-detected [valid-let*-ordered? validate-let*-ordered]
   :gpu-planned      [valid-let*-ordered? validate-let*-ordered]
   :dtype-remapped   [valid-let*-ordered? validate-let*-ordered]
   :backend-applied  [valid-let*-ordered? validate-let*-ordered]
   :alength-resolved [valid-let*-ordered? validate-let*-ordered]
   :mem-merged       [valid-let*-ordered? validate-let*-ordered]})
