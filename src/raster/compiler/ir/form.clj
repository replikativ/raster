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
   All passes will automatically handle it correctly."
  (:require [raster.compiler.ir.par :as par]))

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
;; scope-info — authoritative binder/scope decomposition
;; ================================================================

(defn- rl
  "Rebuild a list form preserving the original form's metadata."
  [form & children]
  (with-meta (apply list children) (meta form)))

(defn- lambda-strip
  "Split an ftm/fn* arity tail (everything after the param vector) into
   [type-prefix source-bodies walked-body]:
     type-prefix   = leading `:- <ret>` type annotation (opaque — a TYPE, never a
                     value, so never scanned or renamed).
     source-bodies = the raw vecs from each `:raster.walker/source-body <vec>`
                     marker. These are deliberately-unqualified raw AD payload —
                     NOT scanned by free-syms (they leak bogus frees), but they
                     DO share the arity's lexical scope, so subst/alpha MUST
                     rename them consistently with the walked body (exposed as the
                     scope's `:aux`). An ftm may be walked more than once, nesting
                     multiple markers — hence a vector of vecs.
     walked-body   = the rest (fully-qualified real code).
   Plain fn* arities have empty type-prefix and source-bodies."
  [after]
  (loop [a after, tprefix [], sbodies []]
    (cond
      (= :- (first a))
      (recur (drop 2 a) (into tprefix (take 2 a)) sbodies)
      (= :raster.walker/source-body (first a))
      (recur (drop 2 a) tprefix (conj sbodies (second a)))
      :else [tprefix sbodies (vec a)])))

(defn- param-binders
  "The binding symbols of a param vector — every symbol except the `&` varargs
   marker (which is syntax, not a binding)."
  [params]
  (vec (remove #(= '& %) (filter symbol? params))))

(defn- reapply-binders
  "Rebuild a param vector from `orig-params`, replacing each binding symbol with
   the next of `new-binders` (in order) while keeping `&` and any non-symbol in
   place. Inverse of `param-binders` for rebuild — lets subst/alpha rename params
   consistently with their body references."
  [orig-params new-binders]
  (loop [ps (seq orig-params), bs (seq new-binders), out []]
    (if-not ps
      out
      (let [p (first ps)]
        (if (and (symbol? p) (not= '& p))
          (recur (next ps) (next bs) (conj out (first bs)))
          (recur (next ps) bs (conj out p)))))))

(defn- lambda-arities
  "Parse a fn/fn*/ftm form into [name arities] where name is the optional
   self-reference symbol (or nil) and arities is a seq of raw arity lists
   `([params] tail...)`."
  [form]
  (let [named? (symbol? (second form))
        nm     (when named? (second form))
        tail   (if named? (nnext form) (rest form))
        arities (if (vector? (first tail)) [tail] tail)]
    [nm arities]))

(defn scope-info
  "Authoritative binder/scope decomposition for a closed-core IR form.

   Returns nil for non-binding forms (the caller recurses generically into all
   children). Otherwise returns:

     {:scopes [{:binders [sym ...]   ;; symbols this scope introduces
                :inits   [expr ...]  ;; init exprs paired w/ binders (empty for
                                     ;;   fn*/dotimes/catch/par-idx — no init)
                :body    [expr ...]  ;; exprs in which :binders are visible
                :aux     [expr ...]} ;; (optional, ftm only) raw source-body payload
                                     ;;   that SHARES the scope and so must be RENAMED
                                     ;;   by subst/alpha — but is INVISIBLE to free-syms
                                     ;;   (free-syms reads only binders/inits/body)
               ...]
      :sequential? bool   ;; within a scope, init_k sees binders_0..k-1 (let*/loop*)
      :rec?        bool   ;; binders visible to their OWN inits (letfn* recursion)
      :outer       [expr ...]  ;; exprs evaluated in the ENCLOSING scope
      :rebuild     (fn [scopes' outer'] -> form)}

   `:rebuild` reconstructs the form from transformed scope maps (same shape and
   arity as `:scopes`) and a transformed `:outer` vector. Structural non-value
   bits (fn* name, ftm `:- ret`/source-body prefix, catch ex-type, reify*
   interfaces/method-names, par layout) are carried in the rebuild closure and
   reattached verbatim — they are never exposed for scanning/substitution.

   This is THE single source of binder knowledge: free-syms, capture-avoiding
   subst-syms, and alpha-convert are all generic over it (see core/util.clj)."
  [form]
  (when (seq? form)
    (let [head (first form)
          kind (:kind (form-info form))]
      (case kind
        ;; let/let* — sequential bindings, body sees all
        :binding
        (let [[_ bindings & body] form
              pairs (partition 2 bindings)]
          {:sequential? true
           :scopes [{:binders (mapv first pairs) :inits (mapv second pairs) :body (vec body)}]
           :outer []
           :rebuild (fn [[{:keys [binders inits body]}] _]
                      (apply rl form head (vec (interleave binders inits)) body))})

        ;; dotimes / loop / loop*
        :scope
        (if (= 'dotimes head)
          (let [[_ [idx bound-expr] & body] form]
            {:sequential? false
             :scopes [{:binders [idx] :inits [] :body (vec body)}]
             :outer [bound-expr]
             :rebuild (fn [[{:keys [binders body]}] [bound']]
                        (apply rl form head [(first binders) bound'] body))})
          (let [[_ bindings & body] form
                pairs (partition 2 bindings)]
            {:sequential? true
             :scopes [{:binders (mapv first pairs) :inits (mapv second pairs) :body (vec body)}]
             :outer []
             :rebuild (fn [[{:keys [binders inits body]}] _]
                        (apply rl form head (vec (interleave binders inits)) body))}))

        ;; fn / fn* / ftm — one independent scope per arity (params local to it)
        :lambda
        (let [[nm arities] (lambda-arities form)
              ;; original single-arity-unwrapped shape: (fn* name? [params] body...)
              unwrapped? (vector? (first (if nm (nnext form) (rest form))))
              parsed (mapv (fn [arity]
                             (let [params (first arity)
                                   [tprefix sbodies wbody] (lambda-strip (rest arity))]
                               {:params params :tprefix tprefix :sbodies sbodies :body wbody}))
                           arities)
              ;; reattach an arity tail: type-prefix (opaque) + each source-body
              ;; marker (rebuilt from the RENAMED :aux) + the transformed body.
              arity-tail (fn [tprefix aux body]
                           (concat tprefix
                                   (mapcat (fn [v] [:raster.walker/source-body v]) aux)
                                   body))]
          {:sequential? false
           :scopes (mapv (fn [{:keys [params body sbodies]}]
                           ;; :aux = source-body vecs — renamed by subst/alpha (they
                           ;; share the arity scope) but invisible to free-syms.
                           {:binders (param-binders params) :inits [] :body body :aux sbodies})
                         parsed)
           :outer []
           :rebuild (fn [scopes' _]
                      ;; params reconstructed from the (possibly renamed) scope
                      ;; binders; type-prefix verbatim; source-bodies from renamed :aux
                      (if (and unwrapped? (= 1 (count parsed)))
                        ;; preserve the original unwrapped single-arity shape
                        (let [{:keys [params tprefix]} (first parsed)
                              {:keys [binders body aux]} (first scopes')
                              params' (reapply-binders params binders)]
                          (apply rl form head (concat (when nm [nm]) [params']
                                                      (arity-tail tprefix aux body))))
                        (let [arities' (map (fn [{:keys [params tprefix]} {:keys [binders body aux]}]
                                              (apply list (reapply-binders params binders)
                                                     (arity-tail tprefix aux body)))
                                            parsed scopes')]
                          (apply rl form head (concat (when nm [nm]) arities')))))})

        ;; letfn* — mutual recursion: binders visible to their own inits
        ;; (matched as :call by form-info, so dispatch on head here)
        :call
        (cond
          (= 'letfn* head)
          (let [[_ bindings & body] form
                pairs (partition 2 bindings)]
            {:sequential? false :rec? true
             :scopes [{:binders (mapv first pairs) :inits (mapv second pairs) :body (vec body)}]
             :outer []
             :rebuild (fn [[{:keys [binders inits body]}] _]
                        (apply rl form head (vec (interleave binders inits)) body))})

          ;; reify* — (reify* [interfaces] (mname [this & args] body...) ...)
          ;; each method binds this+args within its body; interfaces & method
          ;; names are structural (carried in the closure).
          (= 'reify* head)
          (let [[_ interfaces & methods] form
                parsed (mapv (fn [m] {:mname (first m) :params (second m) :body (vec (nnext m))})
                             methods)]
            {:sequential? false
             :scopes (mapv (fn [{:keys [params body]}]
                             {:binders (param-binders params) :inits [] :body body})
                           parsed)
             :outer []
             :rebuild (fn [scopes' _]
                        (let [methods' (map (fn [{:keys [mname params]} {:keys [binders body]}]
                                              (apply list mname (reapply-binders params binders) body))
                                            parsed scopes')]
                          (apply rl form head interfaces methods')))})

          :else nil)

        ;; catch — binds the exception symbol to the catch body only.
        ;; (try itself binds nothing → nil → caller recurses into its children,
        ;;  visiting each catch node here.)
        :special
        (when (= 'catch head)
          (let [[_ ex-type ex-sym & body] form]
            {:sequential? false
             :scopes [{:binders [ex-sym] :inits [] :body (vec body)}]
             :outer []
             :rebuild (fn [[{:keys [binders body]}] _]
                        (apply rl form head ex-type (first binders) body))}))

        ;; par/* — delegate to par-scope-info (authoritative + co-located rebuild)
        :par
        (when-let [{:keys [scoped-syms inner-exprs outer-exprs rebuild]} (par/par-scope-info form)]
          {:sequential? false
           :scopes [{:binders (vec scoped-syms) :inits [] :body (vec inner-exprs)}]
           :outer (vec outer-exprs)
           :rebuild (fn [[{:keys [binders body]}] outer']
                      (rebuild (vec binders) (vec body) (vec outer')))})

        ;; everything else introduces no binders
        nil))))

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

(defn tail-symbol
  "The symbol an expression ultimately evaluates to via its tail position — follows
   let*/loop*/do/if (else-first) to the value leaf. nil if the tail isn't a symbol.
   Used by copy-propagation to recognise a binding that aliases another value even
   through an effectful let*/loop* (e.g. r = (let* [..writes b..] b))."
  [expr]
  (cond
    (symbol? expr) expr
    (not (seq? expr)) nil
    (contains? #{'let* 'let} (first expr)) (tail-symbol (last expr))
    (contains? #{'loop* 'loop} (first expr)) (tail-symbol (nth expr 2 nil)) ; loop body
    (= 'do (first expr)) (tail-symbol (last expr))
    (= 'if (first expr)) (or (tail-symbol (nth expr 3 nil))   ; base/else first
                             (tail-symbol (nth expr 2 nil)))
    :else nil))

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
