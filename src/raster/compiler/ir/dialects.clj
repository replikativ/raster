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
            [raster.compiler.core.util :as util]
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

    ;; Let binding (the main structural form). CLOSED CORE: only let* — the
    ;; macro `let` is eliminated by walk-body's macroexpand-core and rejected by
    ;; invariant I4 (raster.compiler.ir.invariants/non-closed-core).
        (let* [(?:* ?sym ?e)] ?e:body)

    ;; Control flow
        (if ?e:test ?e:then ?e:else)
        (do (?:+ e))

    ;; Loops — closed core: only loop* (macro `loop` rejected by I4)
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

    ;; Parallel SOAC primitives (raster.par/*) and other calls are accepted by
    ;; the catch-all below; their internal scope structure is checked via
    ;; ir/par.par-scope-info, not enumerated here. (The former `broadcast` and
    ;; `reduce!` patterns were stale: broadcast is a typed macro the walker
    ;; lowers to raster.par/map at walk time, and reduce! is not an IR form.)

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
    ;; (broadcast/reduce! were already removed from the Walked parent.)
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

;; Free-variable collection for forward-ref validation uses the unified,
;; scope-aware `util/free-syms` (single source of binder knowledge — handles
;; every closed-core binder incl. letfn*/reify*, treats quote opaque). The
;; former private `collect-free-syms` was a divergent duplicate with different
;; leaf semantics (qualified-only exclusion); it could not change forward-ref
;; results on valid IR — binding names are gensym locals, never core vars.

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
                    free (util/free-syms init bound)
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
