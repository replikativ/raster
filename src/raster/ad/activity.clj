(ns raster.ad.activity
  "Activity analysis for reverse-mode AD on TypedExpr IR.

  Determines which IR nodes are 'active' (carry derivative information)
  vs 'constant' (no derivative flows through them). This is critical
  for performance: inactive nodes need no adjoint computation.

  Inspired by Tapenade's activity analysis:
  - Forward sweep: propagate activity from active seeds through data flow
  - Backward sweep: filter to useful-active (result actually depends on it)

  Uses IR side-tables: stores results in *activity-table*.

  Usage:
    (with-fresh-tables
      (let [ir (sexp->ir body)
            active-params #{x-sym y-sym}]
        (analyze-activity! ir active-params)
        ;; Now (get-activity node) returns :active or :const
        ))"
  (:require [raster.compiler.ir.core :as ir])
  (:import [raster.compiler.ir.core TLiteral TLocal TLet TIf TDo TCall TInvk
            TFn TRecur TThrow TTry TNew TDot TQuote TVar
            TVector TMap TSet TCase]))

;; ================================================================
;; Activity side-table
;; ================================================================

(def ^:dynamic *activity-table*
  "Activity analysis results: {node-id -> :active | :const}"
  (atom {}))

(defn get-activity
  "Look up the activity of an IR node."
  [node]
  (get @*activity-table* (:id node) :const))

(defn active?
  "True if the IR node is active (carries derivative info)."
  [node]
  (= :active (get-activity node)))

(defn- mark-active!
  "Mark an IR node as active."
  [node]
  (swap! *activity-table* assoc (:id node) :active)
  node)

(defn- mark-const!
  "Mark an IR node as constant."
  [node]
  (swap! *activity-table* assoc (:id node) :const)
  node)

;; ================================================================
;; Forward activity propagation
;; ================================================================
;; Propagate from active seeds (parameters) through data flow.
;; A node is active if any of its inputs are active.

(declare propagate-forward!)

(defn- propagate-forward-bindings!
  "Propagate activity through let/loop bindings.
  Updates sym-env with activity of each binding."
  [bindings sym-env]
  (reduce (fn [env [sym init-ir]]
            (propagate-forward! init-ir env)
            (assoc env sym (get-activity init-ir)))
          sym-env bindings))

(defn propagate-forward!
  "Forward propagation: mark nodes as active if any input is active.
  sym-env: {symbol -> :active | :const} for symbols in scope."
  [node sym-env]
  (cond
    ;; Literals are always constant
    (instance? TLiteral node)
    (mark-const! node)

    ;; Locals: active if the symbol is active in env
    (instance? TLocal node)
    (if (= :active (get sym-env (:sym node)))
      (mark-active! node)
      (mark-const! node))

    ;; Let/loop: propagate through bindings, then body
    (instance? TLet node)
    (let [env' (propagate-forward-bindings! (:bindings node) sym-env)
          _ (doseq [b (:body node)]
              (propagate-forward! b env'))
          ;; The let node is active if its body (last form) is active
          last-body (last (:body node))]
      (if (and last-body (active? last-body))
        (mark-active! node)
        (mark-const! node)))

    ;; If: active if either branch is active
    (instance? TIf node)
    (do
      (propagate-forward! (:test node) sym-env)
      (propagate-forward! (:then node) sym-env)
      (when (:else node)
        (propagate-forward! (:else node) sym-env))
      (if (or (active? (:then node))
              (and (:else node) (active? (:else node))))
        (mark-active! node)
        (mark-const! node)))

    ;; Do: active if last form is active
    (instance? TDo node)
    (do
      (doseq [f (:forms node)]
        (propagate-forward! f sym-env))
      (if (and (seq (:forms node)) (active? (last (:forms node))))
        (mark-active! node)
        (mark-const! node)))

    ;; Call: active if any arg is active (conservative)
    (instance? TCall node)
    (do
      (doseq [a (:args node)]
        (propagate-forward! a sym-env))
      (if (some active? (:args node))
        (mark-active! node)
        (mark-const! node)))

    ;; Invk: active if any arg is active
    (instance? TInvk node)
    (do
      (doseq [a (:args node)]
        (propagate-forward! a sym-env))
      (if (some active? (:args node))
        (mark-active! node)
        (mark-const! node)))

    ;; Fn: conservative — don't propagate into closures
    (instance? TFn node)
    (mark-const! node)

    ;; Recur: active if any arg is active
    (instance? TRecur node)
    (do
      (doseq [a (:args node)]
        (propagate-forward! a sym-env))
      (if (some active? (:args node))
        (mark-active! node)
        (mark-const! node)))

    ;; Throw: not active (control flow, not value)
    (instance? TThrow node)
    (do
      (propagate-forward! (:expr node) sym-env)
      (mark-const! node))

    ;; New: active if any constructor arg is active
    (instance? TNew node)
    (do
      (doseq [a (:args node)]
        (propagate-forward! a sym-env))
      (if (some active? (:args node))
        (mark-active! node)
        (mark-const! node)))

    ;; Dot: active if target or any arg is active
    (instance? TDot node)
    (do
      (propagate-forward! (:target node) sym-env)
      (doseq [a (:args node)]
        (propagate-forward! a sym-env))
      (if (or (active? (:target node)) (some active? (:args node)))
        (mark-active! node)
        (mark-const! node)))

    ;; Vector: active if any element is active
    (instance? TVector node)
    (do
      (doseq [item (:items node)]
        (propagate-forward! item sym-env))
      (if (some active? (:items node))
        (mark-active! node)
        (mark-const! node)))

    ;; Map: active if any key or value is active
    (instance? TMap node)
    (do
      (doseq [[k v] (:entries node)]
        (propagate-forward! k sym-env)
        (propagate-forward! v sym-env))
      (if (some (fn [[k v]] (or (active? k) (active? v))) (:entries node))
        (mark-active! node)
        (mark-const! node)))

    ;; Set: active if any element is active
    (instance? TSet node)
    (do
      (doseq [item (:items node)]
        (propagate-forward! item sym-env))
      (if (some active? (:items node))
        (mark-active! node)
        (mark-const! node)))

    ;; Quote, Var, Try, Case: constant
    :else
    (mark-const! node)))

;; ================================================================
;; Backward useful-activity propagation
;; ================================================================
;; An active node is useful-active only if its value flows to
;; the output. This prunes nodes that are technically active
;; (depend on active inputs) but whose result is unused.

(declare propagate-backward!)

(defn- propagate-backward-bindings!
  "Backward propagation through let bindings.
  needed-syms: set of symbols whose adjoints are needed."
  [bindings body needed-syms]
  ;; Walk bindings in reverse order
  (loop [pairs (reverse bindings)
         needed needed-syms]
    (if-not (seq pairs)
      needed
      (let [[sym init-ir] (first pairs)]
        (if (contains? needed sym)
          ;; This binding is needed — propagate into init expr
          (let [init-needed (propagate-backward! init-ir)]
            (recur (rest pairs) (into (disj needed sym) init-needed)))
          ;; Not needed — mark init as const regardless of forward activity
          (do
            (mark-const! init-ir)
            (recur (rest pairs) needed)))))))

(defn propagate-backward!
  "Backward propagation: returns set of symbols needed by this node.
  Also marks nodes as const if they're not useful-active."
  [node]
  (cond
    (instance? TLiteral node)
    #{}

    (instance? TLocal node)
    (if (active? node)
      #{(:sym node)}
      #{})

    (instance? TLet node)
    (if-not (active? node)
      #{}
      (let [;; Body needs determine which bindings matter
            body-needed (reduce into #{} (map propagate-backward! (:body node)))
            ;; Propagate backward through bindings
            binding-needed (propagate-backward-bindings!
                            (:bindings node) (:body node) body-needed)]
        binding-needed))

    (instance? TIf node)
    (if-not (active? node)
      #{}
      (let [then-needed (propagate-backward! (:then node))
            else-needed (if (:else node)
                          (propagate-backward! (:else node))
                          #{})
            test-needed (propagate-backward! (:test node))]
        (into test-needed (into then-needed else-needed))))

    (instance? TDo node)
    (if-not (active? node)
      #{}
      (reduce into #{} (map propagate-backward! (:forms node))))

    (instance? TCall node)
    (if-not (active? node)
      #{}
      (reduce into #{} (map propagate-backward! (:args node))))

    (instance? TInvk node)
    (if-not (active? node)
      #{}
      (reduce into #{} (map propagate-backward! (:args node))))

    (instance? TRecur node)
    (if-not (active? node)
      #{}
      (reduce into #{} (map propagate-backward! (:args node))))

    (instance? TNew node)
    (if-not (active? node)
      #{}
      (reduce into #{} (map propagate-backward! (:args node))))

    (instance? TDot node)
    (if-not (active? node)
      #{}
      (into (propagate-backward! (:target node))
            (reduce into #{} (map propagate-backward! (:args node)))))

    (instance? TVector node)
    (if-not (active? node)
      #{}
      (reduce into #{} (map propagate-backward! (:items node))))

    (instance? TMap node)
    (if-not (active? node)
      #{}
      (reduce into #{} (mapcat (fn [[k v]]
                                 [(propagate-backward! k) (propagate-backward! v)])
                               (:entries node))))

    (instance? TSet node)
    (if-not (active? node)
      #{}
      (reduce into #{} (map propagate-backward! (:items node))))

    :else #{}))

;; ================================================================
;; High-level API
;; ================================================================

(defn analyze-activity!
  "Run activity analysis on an IR tree.
  active-params: set of parameter symbols that are active (carry derivatives).

  Populates *activity-table* with :active/:const for each node.
  Returns the set of symbols that are useful-active at the top level."
  [ir active-params]
  (let [sym-env (into {} (map (fn [s] [s :active]) active-params))]
    ;; Forward: propagate activity from seeds
    (propagate-forward! ir sym-env)
    ;; Backward: prune to useful-active
    (propagate-backward! ir)))

(defn active-bindings
  "Given a TLet node after activity analysis, return only the bindings
  where the init expression is active."
  [let-node]
  (filter (fn [[_sym init-ir]] (active? init-ir))
          (:bindings let-node)))
