(ns raster.compiler.ir.core
  "Typed expression IR for compiler passes.

  Provides a structured representation of walked S-expressions
  with guaranteed round-trip: (= (ir->sexp (sexp->ir form)) form)

  Records wrap the same information as walked S-expressions but
  enable pattern matching, analysis via side-tables, and structured
  transformation.

  Side-tables (atoms) store analysis results keyed by node ID:
  - *type-table*: {id -> type-tag}
  - *purity-table*: {id -> :pure/:local-mut/:global-mut/:io}
  - *const-table*: {id -> constant-value}

  Usage:
    (let [ir (sexp->ir walked-form)]
      ;; analyze, transform
      (ir->sexp ir))  ;; => original walked-form")

;; ================================================================
;; Side-tables for analysis results
;; ================================================================

(def ^:dynamic *type-table*
  "Type analysis results: {node-id -> type-tag}"
  (atom {}))

(def ^:dynamic *purity-table*
  "Purity analysis results: {node-id -> :pure/:local-mut/:global-mut/:io}"
  (atom {}))

(def ^:dynamic *const-table*
  "Constant value analysis: {node-id -> value}"
  (atom {}))

(def ^:dynamic *id-counter*
  "Counter for generating unique node IDs."
  (atom 0))

(defn- next-id []
  (swap! *id-counter* inc))

(defmacro with-fresh-tables
  "Execute body with fresh analysis side-tables and reset ID counter."
  [& body]
  `(binding [*type-table* (atom {})
             *purity-table* (atom {})
             *const-table* (atom {})
             *id-counter* (atom 0)]
     ~@body))

;; ================================================================
;; IR node types
;; ================================================================

(defrecord TLiteral [id value])
(defrecord TLocal   [id sym meta-map])
(defrecord TLet     [id let-sym bindings body])     ;; bindings: [[sym init-ir] ...]
(defrecord TIf      [id test then else])
(defrecord TDo      [id forms])
(defrecord TCall    [id head args meta-map])         ;; head: symbol or IR node
(defrecord TInvk    [id impl args])                  ;; (.invk impl arg1 arg2 ...)
(defrecord TFn      [id fname arities])              ;; arities: [[[params] body-ir...] ...]
(defrecord TRecur   [id args])
(defrecord TThrow   [id expr])
(defrecord TTry     [id body catches finally-body])
(defrecord TNew     [id class-sym args])
(defrecord TDot     [id target method args])         ;; (.method target args...)
(defrecord TQuote   [id form])
(defrecord TVar     [id sym])
(defrecord TVector  [id items])
(defrecord TMap     [id entries])                    ;; entries: [[k-ir v-ir] ...]
(defrecord TSet     [id items])
(defrecord TCase    [id expr clauses default])

;; ================================================================
;; S-expression -> IR conversion
;; ================================================================

(declare sexp->ir)

(defn- sexp->ir-bindings
  "Convert let bindings: [sym1 init1 sym2 init2 ...] -> [[sym1 init1-ir] ...]"
  [bindings]
  (mapv (fn [[sym init]]
          [sym (sexp->ir init)])
        (partition 2 bindings)))

(defn sexp->ir
  "Convert a walked S-expression to IR.
  Preserves all metadata and structural information."
  [form]
  (cond
    ;; Nil literal
    (nil? form)
    (->TLiteral (next-id) nil)

    ;; Numeric/string/keyword/boolean literal
    (or (number? form) (string? form) (keyword? form)
        (boolean? form) (char? form))
    (->TLiteral (next-id) form)

    ;; Symbol (local variable reference)
    (symbol? form)
    (->TLocal (next-id) form (meta form))

    ;; List (special form or call)
    (seq? form)
    (let [head (first form)]
      (cond
        ;; let/let*/loop/loop* — forms with paired binding vectors
        (contains? #{'let 'let* 'loop 'loop*} head)
        (let [bindings (sexp->ir-bindings (second form))
              body (mapv sexp->ir (nnext form))]
          (->TLet (next-id) head bindings body))

        ;; if
        (= head 'if)
        (let [[_ test then else] form]
          (->TIf (next-id) (sexp->ir test) (sexp->ir then)
                 (when else (sexp->ir else))))

        ;; do
        (= head 'do)
        (->TDo (next-id) (mapv sexp->ir (rest form)))

        ;; fn/fn*
        (and (symbol? head) (#{'fn 'fn*} head))
        (let [[_ & rest-args] form
              [fname arities] (if (symbol? (first rest-args))
                                [(first rest-args) (rest rest-args)]
                                [nil rest-args])
              arities (if (vector? (first arities))
                        (list arities)
                        arities)
              ir-arities (mapv (fn [arity]
                                 (let [params (first arity)
                                       body (mapv sexp->ir (rest arity))]
                                   [params body]))
                               arities)]
          (->TFn (next-id) fname ir-arities))

        ;; recur
        (= head 'recur)
        (->TRecur (next-id) (mapv sexp->ir (rest form)))

        ;; throw
        (= head 'throw)
        (->TThrow (next-id) (sexp->ir (second form)))

        ;; try
        (= head 'try)
        ;; Store the raw form — try/catch is complex enough to preserve as-is
        (->TTry (next-id) form nil nil)

        ;; new
        (= head 'new)
        (->TNew (next-id) (second form) (mapv sexp->ir (nnext form)))

        ;; quote
        (= head 'quote)
        (->TQuote (next-id) (second form))

        ;; var
        (= head 'var)
        (->TVar (next-id) (second form))

        ;; case*
        (= head 'case*)
        (->TCase (next-id) form nil nil)  ;; preserve raw form

        ;; set!
        ;; Keep as a generic call
        (= head 'set!)
        (->TCall (next-id) head (mapv sexp->ir (rest form)) (meta form))

        ;; .invk (devirtualized typed call)
        (= head '.invk)
        (->TInvk (next-id) (second form) (mapv sexp->ir (nnext form)))

        ;; Dot form (.method target args...)
        (and (symbol? head) (.startsWith (name head) ".") (not= head '.invk))
        (->TDot (next-id) (sexp->ir (second form)) head (mapv sexp->ir (nnext form)))

        ;; General function call
        :else
        (->TCall (next-id) head (mapv sexp->ir (rest form)) (meta form))))

    ;; Vector
    (vector? form)
    (->TVector (next-id) (mapv sexp->ir form))

    ;; Map
    (map? form)
    (->TMap (next-id) (mapv (fn [[k v]] [(sexp->ir k) (sexp->ir v)]) form))

    ;; Set
    (set? form)
    (->TSet (next-id) (mapv sexp->ir (seq form)))

    ;; Fallback — wrap as literal
    :else
    (->TLiteral (next-id) form)))

;; ================================================================
;; IR -> S-expression conversion
;; ================================================================

(defmulti ir->sexp
  "Convert IR back to walked S-expression.
  Guarantees: (= (ir->sexp (sexp->ir form)) form)"
  type)

(defmethod ir->sexp TLiteral [^TLiteral node]
  (:value node))

(defmethod ir->sexp TLocal [^TLocal node]
  (let [sym (:sym node)]
    (if-let [m (:meta-map node)]
      (with-meta sym m)
      sym)))

(defmethod ir->sexp TLet [^TLet node]
  (let [bindings (vec (mapcat (fn [[sym init]]
                                [sym (ir->sexp init)])
                              (:bindings node)))
        body (map ir->sexp (:body node))]
    (apply list (:let-sym node) bindings body)))

(defmethod ir->sexp TIf [^TIf node]
  (if (:else node)
    (list 'if (ir->sexp (:test node)) (ir->sexp (:then node)) (ir->sexp (:else node)))
    (list 'if (ir->sexp (:test node)) (ir->sexp (:then node)))))

(defmethod ir->sexp TDo [^TDo node]
  (apply list 'do (map ir->sexp (:forms node))))

(defmethod ir->sexp TCall [^TCall node]
  (let [args (map ir->sexp (:args node))]
    (apply list (:head node) args)))

(defmethod ir->sexp TInvk [^TInvk node]
  (let [args (map ir->sexp (:args node))]
    (apply list '.invk (:impl node) args)))

(defmethod ir->sexp TFn [^TFn node]
  (let [arities (map (fn [[params body]]
                       (cons params (map ir->sexp body)))
                     (:arities node))]
    (if (:fname node)
      (if (= 1 (count arities))
        (list* 'fn (:fname node) arities)
        (list* 'fn (:fname node) arities))
      (if (= 1 (count arities))
        (list* 'fn arities)
        (list* 'fn arities)))))

(defmethod ir->sexp TRecur [^TRecur node]
  (apply list 'recur (map ir->sexp (:args node))))

(defmethod ir->sexp TThrow [^TThrow node]
  (list 'throw (ir->sexp (:expr node))))

(defmethod ir->sexp TTry [^TTry node]
  ;; Returns preserved raw form
  (:body node))

(defmethod ir->sexp TNew [^TNew node]
  (apply list 'new (:class-sym node) (map ir->sexp (:args node))))

(defmethod ir->sexp TQuote [^TQuote node]
  (list 'quote (:form node)))

(defmethod ir->sexp TVar [^TVar node]
  (list 'var (:sym node)))

(defmethod ir->sexp TCase [^TCase node]
  (:expr node))  ;; preserved raw form

(defmethod ir->sexp TDot [^TDot node]
  (apply list (:method node) (ir->sexp (:target node))
         (map ir->sexp (:args node))))

(defmethod ir->sexp TVector [^TVector node]
  (mapv ir->sexp (:items node)))

(defmethod ir->sexp TMap [^TMap node]
  (into {} (map (fn [[k v]] [(ir->sexp k) (ir->sexp v)])) (:entries node)))

(defmethod ir->sexp TSet [^TSet node]
  (into #{} (map ir->sexp) (:items node)))

;; ================================================================
;; Analysis helpers
;; ================================================================

(defn set-type!
  "Record a type for an IR node in the type side-table."
  [node tag]
  (swap! *type-table* assoc (:id node) tag)
  node)

(defn get-type
  "Look up the type of an IR node."
  [node]
  (get @*type-table* (:id node)))

(defn set-purity!
  "Record purity for an IR node."
  [node level]
  (swap! *purity-table* assoc (:id node) level)
  node)

(defn get-purity
  "Look up the purity of an IR node."
  [node]
  (get @*purity-table* (:id node)))

(defn set-const!
  "Record that an IR node is a known constant."
  [node value]
  (swap! *const-table* assoc (:id node) value)
  node)

(defn get-const
  "Look up the constant value of an IR node, if any."
  [node]
  (get @*const-table* (:id node)))

;; ================================================================
;; IR traversal utilities
;; ================================================================

(defn walk-ir
  "Walk an IR tree, applying f to each node bottom-up.
  f receives the node with children already transformed."
  [f node]
  (let [walked
        (cond
          (instance? TLiteral node) node
          (instance? TLocal node) node
          (instance? TQuote node) node
          (instance? TVar node) node

          (instance? TLet node)
          (let [bindings (mapv (fn [[sym init]] [sym (walk-ir f init)])
                               (:bindings node))
                body (mapv #(walk-ir f %) (:body node))]
            (assoc node :bindings bindings :body body))

          (instance? TIf node)
          (assoc node
                 :test (walk-ir f (:test node))
                 :then (walk-ir f (:then node))
                 :else (when (:else node) (walk-ir f (:else node))))

          (instance? TDo node)
          (assoc node :forms (mapv #(walk-ir f %) (:forms node)))

          (instance? TCall node)
          (assoc node :args (mapv #(walk-ir f %) (:args node)))

          (instance? TInvk node)
          (assoc node :args (mapv #(walk-ir f %) (:args node)))

          (instance? TRecur node)
          (assoc node :args (mapv #(walk-ir f %) (:args node)))

          (instance? TThrow node)
          (assoc node :expr (walk-ir f (:expr node)))

          (instance? TNew node)
          (assoc node :args (mapv #(walk-ir f %) (:args node)))

          (instance? TDot node)
          (assoc node
                 :target (walk-ir f (:target node))
                 :args (mapv #(walk-ir f %) (:args node)))

          (instance? TFn node)
          (assoc node :arities
                 (mapv (fn [[params body]]
                         [params (mapv #(walk-ir f %) body)])
                       (:arities node)))

          (instance? TVector node)
          (assoc node :items (mapv #(walk-ir f %) (:items node)))

          (instance? TMap node)
          (assoc node :entries
                 (mapv (fn [[k v]] [(walk-ir f k) (walk-ir f v)])
                       (:entries node)))

          (instance? TSet node)
          (assoc node :items (mapv #(walk-ir f %) (:items node)))

          ;; TTry, TCase — preserved raw forms, don't walk
          :else node)]
    (f walked)))

(defn collect-locals
  "Collect all free variable symbols referenced in an IR tree."
  [node]
  (cond
    (instance? TLocal node) #{(:sym node)}
    (instance? TLiteral node) #{}
    (instance? TQuote node) #{}
    (instance? TVar node) #{}

    (instance? TLet node)
    (let [bound (set (map first (:bindings node)))
          init-vars (reduce into #{} (map #(collect-locals (second %)) (:bindings node)))
          body-vars (reduce into #{} (map collect-locals (:body node)))]
      (into init-vars (remove bound body-vars)))

    (instance? TIf node)
    (into (into (collect-locals (:test node))
                (collect-locals (:then node)))
          (when (:else node) (collect-locals (:else node))))

    (instance? TDo node)
    (reduce into #{} (map collect-locals (:forms node)))

    (instance? TCall node)
    (reduce into #{} (map collect-locals (:args node)))

    (instance? TInvk node)
    (reduce into #{} (map collect-locals (:args node)))

    (instance? TFn node)
    (reduce (fn [vars [params body]]
              (let [param-set (set (remove #{'&} params))
                    body-vars (reduce into #{} (map collect-locals body))]
                (into vars (remove param-set body-vars))))
            #{} (:arities node))

    (instance? TRecur node)
    (reduce into #{} (map collect-locals (:args node)))

    (instance? TThrow node)
    (collect-locals (:expr node))

    (instance? TNew node)
    (reduce into #{} (map collect-locals (:args node)))

    (instance? TDot node)
    (into (collect-locals (:target node))
          (reduce into #{} (map collect-locals (:args node))))

    (instance? TVector node)
    (reduce into #{} (map collect-locals (:items node)))

    (instance? TMap node)
    (reduce into #{} (mapcat (fn [[k v]] [(collect-locals k) (collect-locals v)])
                             (:entries node)))

    (instance? TSet node)
    (reduce into #{} (map collect-locals (:items node)))

    :else #{}))
