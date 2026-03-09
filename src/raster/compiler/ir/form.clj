(ns raster.compiler.ir.form
  "Unified form classification for the compiler IR.

   Every compiler pass that needs to know about form structure should use
   this module instead of ad-hoc `contains?` checks. This is the single
   source of truth for:
   - What introduces a scope (has scoped variables that can't be lifted)?
   - What is liftable (can its bindings be merged into the parent let*)?
   - What is a call that might be inlinable?
   - What is a binding form vs a control flow form?

   When adding a new form type to the compiler, update `form-info` here.
   All passes will automatically handle it correctly.")

(defn form-info
  "Classify a compiler IR form and return its properties.

   Returns a map:
     :kind             — keyword identifying the form type
     :introduces-scope? — true if the form introduces scoped variables
                         that must not be lifted past the form boundary
     :liftable?        — true if nested let bindings can be lifted out
     :head             — the head symbol (for calls)

   Form kinds:
     :leaf     — literal, symbol, or non-seq value
     :binding  — let/let* (introduces bindings, liftable)
     :scope    — dotimes/loop (introduces scoped vars, NOT liftable)
     :branch   — if/when/case (conditional, NOT liftable — arms may not execute)
     :lambda   — fn/fn* (closure, NOT liftable)
     :par      — raster.par/* (parallel primitive with scoped loop var)
     :invk     — .invk call (typed dispatch, liftable)
     :call     — regular function call (liftable)
     :do       — do block (sequential effects)
     :special  — try/catch/finally/recur/throw (NOT liftable)"
  [form]
  (if-not (seq? form)
    {:kind :leaf :introduces-scope? false :liftable? true}
    (let [head (first form)]
      (cond
        ;; Binding forms — liftable, no scope
        (contains? #{'let 'let*} head)
        {:kind :binding :introduces-scope? false :liftable? true :head head}

        ;; Scope-introducing iteration — NOT liftable
        (contains? #{'dotimes 'loop 'loop*} head)
        {:kind :scope :introduces-scope? true :liftable? false :head head}

        ;; Conditionals — NOT liftable (arms may not execute)
        (contains? #{'if 'when 'case*} head)
        {:kind :branch :introduces-scope? false :liftable? false :head head}

        ;; Closures — introduce scope, NOT liftable
        (contains? #{'fn 'fn* 'ftm} head)
        {:kind :lambda :introduces-scope? true :liftable? false :head head}

        ;; Special forms — NOT liftable
        (contains? #{'try 'catch 'finally 'recur 'throw 'new} head)
        {:kind :special :introduces-scope? false :liftable? false :head head}

        ;; Parallel primitives — have scoped loop variables, NOT liftable
        ;; :return-type-arg — 0-based arg index whose type = form return type
        ;;   map!/scan!: arg 0 (target buffer)
        ;;   reduce:     arg 1 (init accumulator value)
        (and (symbol? head)
             (some? (namespace head))
             (.startsWith ^String (namespace head) "raster.par"))
        (let [n (name head)
              mutating? (.endsWith n "!")]
          {:kind :par :introduces-scope? true :liftable? false :head head
           :return-type-arg (if mutating? 0 1)})

        ;; do block — sequential, liftable (effects + result)
        (= 'do head)
        {:kind :do :introduces-scope? false :liftable? true :head head}

        ;; .invk typed call
        (= '.invk head)
        {:kind :invk :introduces-scope? false :liftable? true :head head}

        ;; var reference
        (= 'var head)
        {:kind :var-ref :introduces-scope? false :liftable? true :head head}

        ;; Regular function call
        :else
        {:kind :call :introduces-scope? false :liftable? true :head head}))))

;; ================================================================
;; Convenience predicates
;; ================================================================

(defn introduces-scope?
  "True if the form introduces scoped variables that must not be
   lifted past the form boundary (dotimes, loop, fn, par/map!, etc.)."
  [form]
  (:introduces-scope? (form-info form)))

(defn liftable?
  "True if the form's content can be lifted into the parent scope
   (let bindings, do effects, function call arguments)."
  [form]
  (:liftable? (form-info form)))

(defn binding-form?
  "True if the form is a let/let* binding form."
  [form]
  (= :binding (:kind (form-info form))))

(defn scope-form?
  "True if the form introduces a new scope (dotimes, loop, fn, par)."
  [form]
  (contains? #{:scope :lambda :par} (:kind (form-info form))))

(defn call-form?
  "True if the form is a function call (.invk or regular)."
  [form]
  (contains? #{:invk :call} (:kind (form-info form))))

(defn effective-op
  "Extract the effective operator symbol from any call expression.
   For (.invk impl-sym args...) returns impl-sym (the devirtualized op).
   For (op args...) returns op (the direct call target).
   Returns nil for non-seq or non-call forms.

   Use this instead of bare `(first expr)` when checking operator identity
   with descriptor predicates (alloc-op?, alength-op?, etc.), since the
   walker devirtualizes calls into .invk forms where the operator is at
   position 2, not position 1."
  [expr]
  (when (seq? expr)
    (let [head (first expr)]
      (if (= '.invk head)
        (second expr)
        head))))

(defn effective-args
  "Extract the effective argument list from any call expression.
   For (.invk impl-sym args...) returns args after the impl-sym.
   For (op args...) returns args after the op.
   Returns nil for non-seq forms."
  [expr]
  (when (seq? expr)
    (if (= '.invk (first expr))
      (drop 2 expr)
      (rest expr))))

(def known-form-heads
  "Set of symbols that are recognized form heads (not variable references).
   Used by qualify-body-symbols and free-var analysis to exclude form
   keywords from namespace qualification / variable collection.

   MUST be a superset of the heads recognized by form-info above.
   Also includes Clojure special forms that form-info treats as :call
   but that should never be namespace-qualified (quote, set!, letfn*, etc.)."
  #{'let 'let* 'loop 'loop* 'dotimes 'if 'when 'case*
    'fn 'fn* 'ftm 'try 'catch 'finally 'recur 'throw 'new
    'do '.invk 'var 'quote 'set! 'letfn* 'reify* '. 'def
    'import* 'clojure.core/import*})
