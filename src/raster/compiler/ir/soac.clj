(ns raster.compiler.ir.soac
  "SOAC (Second-Order Array Combinators) IR for fusion.

  Lifts flat let* bindings containing raster.par/* forms into typed
  SOAC nodes that a dependency graph can reason about. Provides
  round-trip conversion: par S-expr ↔ SOAC record ↔ par S-expr.

  SOAC types mirror Futhark's combinator classification:
    SoacMap     — element-wise parallel map
    SoacReduce  — parallel fold with associative operator
    SoacScan    — parallel prefix scan (inclusive)
    SoacStencil — neighborhood stencil (fixed radius)
    ScalarBinding — non-parallel scalar/allocation binding"
  (:require [raster.compiler.ir.par :as par]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [clojure.set :as set]
            [clojure.walk :as walk]))

;; ================================================================
;; SOAC records
;; ================================================================

(defrecord SoacMap
           [id          ;; int — unique binding index in let*
            sym         ;; symbol — LHS of (let* [sym expr ...])
            idx         ;; symbol — loop index variable
            bound       ;; expr — iteration count
            cast-fn     ;; symbol or nil — cast applied to body result
            lambda      ;; expr — kernel body S-expression
            inputs      ;; #{sym} — array symbols read via aget in lambda
            outputs     ;; #{sym} — output array symbols written
            scalars])   ;; #{sym} — free symbols in lambda that are not arrays or idx

(defrecord SoacReduce
           [id          ;; int
            sym         ;; symbol
            acc         ;; symbol — accumulator variable
            init        ;; expr — initial accumulator value
            idx         ;; symbol — loop index variable
            bound       ;; expr — iteration count
            lambda      ;; expr — reduction body
            inputs      ;; #{sym} — array symbols read
            output      ;; symbol — result symbol (= sym)
            scalars])   ;; #{sym} — free scalars

(defrecord SoacScan
           [id          ;; int
            sym         ;; symbol
            out         ;; symbol — output array
            acc         ;; symbol — accumulator variable
            init        ;; expr — initial accumulator value
            idx         ;; symbol — loop index variable
            bound       ;; expr — iteration count
            cast-fn     ;; symbol or nil
            lambda      ;; expr — scan body
            inputs      ;; #{sym} — array symbols read
            outputs     ;; #{sym} — output array symbols
            scalars])   ;; #{sym} — free scalars

(defrecord SoacStencil
           [id          ;; int
            sym         ;; symbol
            in-arrays   ;; vector — input array symbols
            radius      ;; int — stencil radius
            boundary    ;; keyword — boundary handling (:dirichlet etc.)
            cast-fn     ;; symbol or nil
            idx         ;; symbol — loop index variable
            bound       ;; expr — iteration count
            lambda      ;; expr — stencil body
            inputs      ;; #{sym} — array symbols read
            outputs     ;; #{sym} — output array symbols
            scalars])   ;; #{sym} — free scalars

(defrecord ScalarBinding
           [id          ;; int
            sym         ;; symbol
            expr])      ;; the RHS expression

;; ================================================================
;; Input/output extraction helpers
;; ================================================================

(defn- collect-aget-arrays
  "Collect array symbols referenced via aget in body."
  [body]
  (par/collect-aget-arrays body))

(defn- extract-io
  "Extract inputs, outputs, and scalars from a SOAC lambda.
  inputs: array syms read via aget
  outputs: the output array sym(s) PLUS any array the lambda writes via aset
  scalars: free syms minus inputs, outputs, idx, acc, and operators

  INVARIANT: an array symbol written inside the lambda is an array OUTPUT of
  the SOAC — never a scalar. The structural out slot alone is not enough: a
  horizontally-fused multi-output map carries its secondary outputs only as
  side-effect asets in the lambda body (soac->par-form flattens the fused node
  to a single-out par/map!), so the write-side collection is what keeps them
  classified as array params when the par form is re-parsed."
  [lambda idx-sym out-syms & {:keys [acc-sym]}]
  (let [inputs (collect-aget-arrays lambda)
        written (par/collect-aset-arrays lambda)
        all-free (util/free-syms lambda)
        exclude (set/union inputs
                           (set out-syms)
                           written
                           #{idx-sym}
                           (if acc-sym #{acc-sym} #{})
                           ;; Exclude common operators/literals that appear as symbols
                           (set/union descriptor/aget-ops descriptor/aset-ops
                                      #{'do 'let 'let* 'if 'double 'float 'int 'long}))
        scalars (set/difference all-free exclude)]
    {:inputs inputs
     :outputs (into (set out-syms) written)
     :scalars scalars}))

;; ================================================================
;; par S-expression → SOAC conversion
;; ================================================================

(defn- strip-index-cast
  "Unwrap a (long i)/(int i) cast around a bare index symbol; else return as-is."
  [e]
  (if (and (seq? e) (contains? #{'long 'int 'clojure.core/long 'clojure.core/int} (first e))
           (= 2 (count e)))
    (second e)
    e))

(defn- inline-let-bindings
  "Beta-reduce let* bindings into `expr`: substitute each bound symbol with its
   definition. Walker-produced bindings are SSA (each symbol bound once, no
   shadowing), so iterative substitution in reverse order is sound — a later
   binding referencing an earlier one is resolved when the earlier one is
   inlined. Duplicates loads (e.g. g=gate[i] used several times), which the C
   compiler CSEs; the payoff is a single flat expression that the existing
   aget-substitution fusion + expression emitter handle directly."
  [binds expr]
  (reduce (fn [acc [s e]] (walk/postwalk-replace {s e} acc))
          expr (reverse (partition 2 binds))))

(defn- single-aset-void
  "If a par/map-void! body is a SINGLE in-place write `(aset OUT idx VAL)` whose
   index is exactly the loop index (a 1:1 elementwise write — NOT an offset/scatter
   write like cache[base+i]), return {:out OUT :value VAL :cast cf}. Otherwise nil.

   Unwraps a `do` with one statement and any let* scaffolding the walker emits
   around the write (e.g. `(let* [g (aget gate i)] (aset tmp i …g…))`), inlining
   those bindings into VAL so the returned :value is one flat expression — the
   same shape a pure map's lambda has, which is what lets the SAME vertical-fusion
   path inline it downstream. Recognizes bare `aset` and walker-devirtualized
   `(.invk aset-impl …)` via the op-descriptor matchers."
  [body idx-sym]
  (let [stmt (if (and (seq? body) (= 'do (first body)) (= 2 (count body)))
               (second body) body)]
    (cond
      ;; let*-wrapped write: recurse into the let body, then inline the bindings
      (and (seq? stmt) (contains? #{'let* 'let} (first stmt)))
      (let [[_ binds & lbody] stmt]
        (when-let [inner (single-aset-void (last lbody) idx-sym)]
          (update inner :value #(inline-let-bindings binds %))))

      (descriptor/aset-call? stmt)
      (let [cargs (vec (descriptor/call-args stmt))]   ;; (arr idx-expr val)
        (when (and (= 3 (count cargs))
                   (= idx-sym (strip-index-cast (nth cargs 1))))
          (let [out-arr (descriptor/aset-array-sym stmt)
                val (nth cargs 2)
                [cast-fn value]
                (if (and (seq? val)
                         (contains? #{'float 'double 'int 'long
                                      'clojure.core/float 'clojure.core/double} (first val))
                         (= 2 (count val)))
                  [(first val) (second val)]
                  [nil val])]
            {:out out-arr :value value :cast cast-fn})))

      :else nil)))

(defn par-form->soac
  "Convert a raster.par/* S-expression to a SOAC record.
  Returns a SoacMap, SoacReduce, SoacScan, SoacStencil, or nil."
  [sym expr id]
  (cond
    ;; Pure par/map (no output buffer) — must check before par-map-form?
    (par/par-map-pure-form? expr)
    (let [info (par/extract-par-map-pure-info expr)
          {:keys [inputs outputs scalars]}
          (extract-io (:body info) (:idx info) [sym])
          elem-type (or (:elem-type info)
                        (:raster.type/elem-type (meta expr)))]
      (cond-> (->SoacMap id sym (:idx info) (:bound info)
                         (:cast info) (:body info)
                         inputs #{sym} scalars)
        elem-type (assoc :elem-type elem-type)
        true (assoc :pure? true)))

    ;; Imperative par/map! (with output buffer)
    (par/par-map-form? expr)
    (let [info (par/extract-par-map-info expr)
          {:keys [inputs outputs scalars]}
          (extract-io (:body info) (:idx info) [(:out info)])
          elem-type (or (:elem-type info)
                        (:raster.type/elem-type (meta expr)))]
      (cond-> (->SoacMap id sym (:idx info) (:bound info)
                         (:cast info) (:body info)
                         inputs outputs scalars)
        elem-type (assoc :elem-type elem-type)))

    (par/par-reduce-form? expr)
    (let [info (par/extract-par-reduce-info expr)
          {:keys [inputs scalars]}
          (extract-io (:body info) (:idx info) [sym]
                      :acc-sym (:acc info))
          elem-type (or (:elem-type info)
                        (:raster.type/elem-type (meta expr)))]
      (cond-> (->SoacReduce id sym (:acc info) (:init info)
                            (:idx info) (:bound info)
                            (:body info) inputs sym scalars)
        elem-type (assoc :elem-type elem-type)))

    ;; Scan: (raster.par/scan out acc init idx bound cast body)
    (and (seq? expr) (= 'raster.par/scan (first expr)))
    (let [[_ out-sym acc-sym init-expr idx-sym bound-expr cast-fn body-expr] expr
          {:keys [inputs outputs scalars]}
          (extract-io body-expr idx-sym [out-sym] :acc-sym acc-sym)
          elem-type (:raster.type/elem-type (meta expr))]
      (cond-> (->SoacScan id sym out-sym acc-sym init-expr idx-sym bound-expr
                          cast-fn body-expr inputs outputs scalars)
        elem-type (assoc :elem-type elem-type)))

    ;; Stencil: (raster.par/stencil! out [in-arrays] radius boundary cast idx bound body)
    (par/par-stencil-form? expr)
    (let [info (par/extract-par-stencil-info expr)
          {:keys [inputs outputs scalars]}
          (extract-io (:body info) (:idx info) [(:out info)])
          elem-type (or (:elem-type info)
                        (:raster.type/elem-type (meta expr)))]
      (cond-> (->SoacStencil id sym (:in-arrays info) (:radius info)
                             (:boundary info) (:cast info)
                             (:idx info) (:bound info)
                             (:body info) inputs outputs scalars)
        elem-type (assoc :elem-type elem-type)))

    ;; Imperative par/map-void! with a SINGLE 1:1 in-place write — model as a
    ;; non-pure (in-place) SoacMap whose OUTPUT is the written array and whose
    ;; LAMBDA is the written value. This makes it indistinguishable from a pure
    ;; map for the SOAC fuser: a downstream map reading (aget OUT i) fuses by
    ;; substituting this value (the existing vertical-fusion path), eliminating
    ;; the intermediate buffer. :void? round-trips it back to map-void! in
    ;; soac->par-form, so codegen stays on the (correct) legacy void generator
    ;; whether or not it fused. Multi-aset / offset-scatter / reduction void
    ;; bodies return nil here → ScalarBinding → unchanged legacy path.
    (par/par-map-void-form? expr)
    (let [info (par/extract-par-map-void-info expr)
          idx (:idx info)]
      (when-let [{:keys [out value cast]} (single-aset-void (:body info) idx)]
        (let [{:keys [inputs scalars]} (extract-io value idx [out])
              elem-type (or (:elem-type info)
                            (:raster.type/elem-type (meta expr)))]
          (cond-> (->SoacMap id sym idx (:bound info) cast value
                             inputs #{out} scalars)
            elem-type (assoc :elem-type elem-type)
            true (assoc :void? true :primary-out out)))))

    :else nil))

;; ================================================================
;; SOAC → par S-expression conversion
;; ================================================================

(defn- stamp-elem-type
  "Attach :raster.type/elem-type metadata from a SOAC's :elem-type to a form."
  [form soac]
  (if-let [et (:elem-type soac)]
    (vary-meta form assoc :raster.type/elem-type et)
    form))

(defn soac->par-form
  "Convert a SOAC record back to a raster.par/* S-expression."
  [soac]
  (stamp-elem-type
   (condp instance? soac
     SoacMap
     (cond
       ;; In-place single-aset void map: reconstruct the imperative write so codegen
       ;; stays on the legacy void generator (handles normalize + per-array dtype +
       ;; written-arrays). The lambda is the produced value; re-wrap it in the aset.
       (:void? soac)
       (let [out (or (:primary-out soac) (first (:outputs soac)))
             v (if (:cast-fn soac) (list (:cast-fn soac) (:lambda soac)) (:lambda soac))]
         (list 'raster.par/map-void! (:idx soac) (:bound soac)
               (list 'raster.arrays/aset out (:idx soac) v)))

       (:pure? soac)
       (list 'raster.par/pmap (:idx soac) (:bound soac) (:cast-fn soac) (:lambda soac))

       :else
       (list 'raster.par/map! (or (:primary-out soac) (first (:outputs soac)))
             (:idx soac) (:bound soac) (:cast-fn soac) (:lambda soac)))

     SoacReduce
     (list 'raster.par/reduce (:acc soac) (:init soac)
           (:idx soac) (:bound soac) (:lambda soac))

     SoacScan
     (list 'raster.par/scan (:out soac) (:acc soac) (:init soac)
           (:idx soac) (:bound soac) (:cast-fn soac) (:lambda soac))

     SoacStencil
     (list 'raster.par/stencil! (first (:outputs soac))
           (:in-arrays soac) (:radius soac) (:boundary soac)
           (:cast-fn soac) (:idx soac) (:bound soac) (:lambda soac))

     ;; Not a SOAC
     (throw (ex-info "Cannot convert non-SOAC to par form" {:node soac})))
   soac))

;; ================================================================
;; Bulk conversion: let* bindings ↔ SOAC/Scalar nodes
;; ================================================================

(defn let-bindings->nodes
  "Convert a sequence of [sym expr] binding pairs to SOAC/Scalar nodes.
  Each pair gets a sequential id for ordering."
  [pairs]
  (vec
   (map-indexed
    (fn [id [sym expr]]
      (or (par-form->soac sym expr id)
          (->ScalarBinding id sym expr)))
    pairs)))

(declare node-all-free-syms node-produced-syms)

(defn- node-full-free-syms
  "Get ALL free symbols of a node by computing free-syms on its full expression.
  More robust than node-all-free-syms after fusion modifies lambda bodies."
  [node]
  (if (instance? ScalarBinding node)
    (util/free-syms (:expr node))
    ;; For SOACs: free-syms of the full par expression + output buffers
    ;; (outputs must be allocated before the SOAC runs)
    (let [par-form (soac->par-form node)]
      (set/union (util/free-syms par-form)
                 (or (:outputs node) #{})))))

(defn- topological-sort-nodes
  "Sort nodes respecting data dependencies: a node appears after all nodes
  that produce symbols it references. Falls back to :id order for nodes
  with no dependency relationship (preserves original evaluation order)."
  [nodes-seq]
  (let [;; Build producer map: each sym → producing node.
        ;; Use only :sym (the LHS binding) per node, preserving the FIRST
        ;; producer (lowest :id) when multiple nodes produce the same sym.
        ;; This ensures allocation ScalarBindings win over SOACs that
        ;; also claim the output buffer.
        producer-of (reduce (fn [m node]
                              (let [sym (:sym node)]
                                (if (contains? m sym)
                                  m  ;; keep earliest producer
                                  (assoc m sym node))))
                            {} (sort-by :id nodes-seq))
        ;; Build dependency graph using full expression free-syms.
        ;; This is robust against fusion modifying lambda bodies (e.g.
        ;; horizontal fusion adding aset calls for secondary outputs).
        node-deps (into {}
                        (map (fn [node]
                               (let [free (node-full-free-syms node)
                                     dep-nodes (set (keep producer-of free))]
                                 [node (disj dep-nodes node)])))
                        nodes-seq)
        ;; Kahn's algorithm with :id tiebreaking for stability
        ;; Start with nodes that have no internal dependencies
        ready (into (sorted-set-by (fn [a b] (compare (:id a) (:id b))))
                    (filter #(empty? (get node-deps %)) nodes-seq))
        remaining-deps (atom node-deps)]
    (loop [ready ready, result []]
      (if (empty? ready)
        (let [leftover (remove (set result) nodes-seq)]
          (if (empty? leftover)
            result
            ;; Cycle or unresolved — append remaining by :id (shouldn't happen)
            (into result (sort-by :id leftover))))
        (let [node (first ready)
              ready' (disj ready node)
              result' (conj result node)
              ;; Remove node from all dependency sets; add newly-ready nodes
              produced (node-produced-syms node)
              newly-ready
              (reduce (fn [acc other]
                        (let [deps (get @remaining-deps other #{})]
                          (when (contains? deps node)
                            (swap! remaining-deps update other disj node))
                          (let [new-deps (disj deps node)]
                            (if (and (seq deps) (empty? new-deps)
                                     (not (contains? (set result') other)))
                              (conj acc other)
                              acc))))
                      #{}
                      nodes-seq)]
          (recur (into ready' newly-ready) result'))))))

(defn nodes->let-bindings
  "Convert SOAC/Scalar nodes back to [sym expr] binding pairs.
  Uses topological sort to respect data dependencies — a binding appears
  after all bindings it references. Falls back to :id order for nodes
  with no dependency relationship."
  [nodes]
  (let [nodes-seq (vals (if (map? nodes) nodes
                            (into {} (map (fn [n] [(:id n) n]) nodes))))
        sorted (topological-sort-nodes nodes-seq)]
    (mapv (fn [node]
            (if (instance? ScalarBinding node)
              [(:sym node) (:expr node)]
              [(:sym node) (soac->par-form node)]))
          sorted)))

;; ================================================================
;; Predicates and accessors
;; ================================================================

(defn soac?
  "Check if a node is a SOAC (not a ScalarBinding)."
  [node]
  (not (instance? ScalarBinding node)))

(defn soac-inputs
  "Get the set of input array symbols for a SOAC node."
  [node]
  (when (soac? node) (:inputs node)))

(defn soac-outputs
  "Get the set of output symbols for a SOAC node."
  [node]
  (cond
    (instance? SoacMap node)     (:outputs node)
    (instance? SoacReduce node)  #{(:output node)}
    (instance? SoacScan node)    (:outputs node)
    (instance? SoacStencil node) (:outputs node)
    :else nil))

(defn soac-bound
  "Get the iteration bound expression for a SOAC node."
  [node]
  (when (soac? node) (:bound node)))

(defn soac-idx
  "Get the index variable for a SOAC node."
  [node]
  (when (soac? node) (:idx node)))

(defn node-all-free-syms
  "Get all free symbols referenced by a node (inputs + outputs + scalars + bound + init).
  Outputs are included for imperative SOACs because they write to pre-allocated
  output buffers that must be defined before the SOAC runs. Pure par/map SOACs
  don't have external output buffers — their output IS their binding symbol."
  [node]
  (if (instance? ScalarBinding node)
    (util/free-syms (:expr node))
    (set/union (or (:inputs node) #{})
               (if (:pure? node) #{} (or (:outputs node) #{}))
               (or (:scalars node) #{})
               (util/free-syms (:bound node))
               (if (:init node) (util/free-syms (:init node)) #{}))))

(defn- expr-written-arrays
  "Arrays written via aset (bare or devirtualized) anywhere in `expr`. Used so a
   ScalarBinding wrapping a side-effecting void form (a multi-aset par/map-void!
   like rms-norm/attention/quant-act that doesn't reduce to a single-aset SoacMap)
   still declares the buffers it produces — otherwise consumers of those buffers
   get no dependency edge and the scheduler / horizontal-fuser can incorrectly
   reorder or parallelize across the write (silent miscompile in composed layers)."
  [expr]
  (let [w (volatile! #{})]
    (walk/postwalk
     (fn [f]
       (when (descriptor/aset-call? f)
         (when-let [a (descriptor/aset-array-sym f)] (vswap! w conj a)))
       f)
     expr)
    @w))

(defn node-produced-syms
  "Get the set of symbols produced (defined) by a node. For a ScalarBinding this
   is its binding sym PLUS any arrays its expression writes in place (so in-place
   void writes participate in dependency analysis)."
  [node]
  (if (instance? ScalarBinding node)
    (conj (expr-written-arrays (:expr node)) (:sym node))
    (set/union #{(:sym node)} (or (soac-outputs node) #{}))))

;; ================================================================
;; Screma: unified SOAC node (Futhark's key abstraction)
;; ================================================================

(defrecord Screma
           [id          ;; int
            sym         ;; symbol
            idx         ;; symbol — loop index variable
            bound       ;; expr — iteration count
            inputs      ;; #{sym} — array symbols read
            outputs     ;; #{sym} — output array symbols
            scalars     ;; #{sym} — free scalars
            cast-fn     ;; symbol or nil
            scans       ;; [{:acc sym :init expr :lambda expr :out sym} ...]
            reduces     ;; [{:acc sym :init expr :lambda expr} ...]
            map-lambda  ;; body expr for map, or nil
            ])

(defn soac->screma
  "Convert a specific SOAC record to a unified Screma.
  Classification:
    SoacMap    → Screma with map-lambda only
    SoacReduce → Screma with one reduce, no map-lambda
    SoacScan   → Screma with one scan, no map-lambda"
  [soac]
  (let [elem-type (:elem-type soac)
        screma
        (condp instance? soac
          SoacMap
          (->Screma (:id soac) (:sym soac) (:idx soac) (:bound soac)
                    (:inputs soac) (:outputs soac) (:scalars soac)
                    (:cast-fn soac) [] [] (:lambda soac))

          SoacReduce
          (->Screma (:id soac) (:sym soac) (:idx soac) (:bound soac)
                    (:inputs soac) #{(:output soac)} (:scalars soac)
                    nil [] [{:acc (:acc soac) :init (:init soac)
                             :lambda (:lambda soac)}]
                    nil)

          SoacScan
          (->Screma (:id soac) (:sym soac) (:idx soac) (:bound soac)
                    (:inputs soac) (:outputs soac) (:scalars soac)
                    (:cast-fn soac)
                    [{:acc (:acc soac) :init (:init soac)
                      :lambda (:lambda soac) :out (:out soac)}]
                    [] nil)

          ;; Pass through other types
          soac)]
    (if elem-type (assoc screma :elem-type elem-type) screma)))

(defn screma->par-form
  "Convert a Screma back to a raster.par/* S-expression.
  Dispatches based on which fields are populated."
  [screma]
  (let [et (:elem-type screma)
        form
        (cond
          ;; Pure map: no scans or reduces
          (and (empty? (:scans screma)) (empty? (:reduces screma)) (:map-lambda screma))
          (list 'raster.par/map! (first (:outputs screma))
                (:idx screma) (:bound screma) (:cast-fn screma) (:map-lambda screma))

          ;; Pure reduce: one reduce, no map-lambda
          (and (empty? (:scans screma)) (= 1 (count (:reduces screma))) (nil? (:map-lambda screma)))
          (let [r (first (:reduces screma))]
            (list 'raster.par/reduce (:acc r) (:init r)
                  (:idx screma) (:bound screma) (:lambda r)))

          ;; Pure scan: one scan, no map-lambda
          (and (= 1 (count (:scans screma))) (empty? (:reduces screma)) (nil? (:map-lambda screma)))
          (let [s (first (:scans screma))]
            (list 'raster.par/scan (:out s) (:acc s) (:init s)
                  (:idx screma) (:bound screma) (:cast-fn screma) (:lambda s)))

          ;; Map+Reduce: map feeds reduce
          (and (empty? (:scans screma)) (= 1 (count (:reduces screma))) (:map-lambda screma))
          ;; Emit as reduce with map body inlined
          (let [r (first (:reduces screma))]
            (list 'raster.par/reduce (:acc r) (:init r)
                  (:idx screma) (:bound screma) (:lambda r)))

          :else
          (throw (ex-info "Cannot convert complex Screma to single par form"
                          {:screma screma})))]
    (if et (vary-meta form assoc :raster.type/elem-type et) form)))

(defn- substitute-aget-sym
  "Replace (aget target-sym idx) with replacement-expr in body,
  adjusting index variable from src-idx to dst-idx."
  [body target-sym src-idx dst-idx replacement-expr]
  (walk/postwalk
   (fn [f]
     (if (and (seq? f)
              (descriptor/aget-op? (first f))
              (>= (count f) 3)
              (symbol? (second f))
              (= (name target-sym) (name (second f))))
       (walk/postwalk
        (fn [g] (if (= g src-idx) dst-idx g))
        replacement-expr)
       f))
   body))

(defn screma-compose
  "Vertically compose producer Screma into consumer Screma.
  The intermediate-sym is the array produced by producer and consumed by consumer.

  Composition rules:
    Map→Map:     merge map-lambdas (producer body inlined into consumer)
    Map→Reduce:  producer map-lambda feeds consumer reduce lambda
    Map→Scan:    producer map-lambda feeds consumer scan lambda
    Scan→Map:    scan output feeds map — creates combined Screma"
  [producer consumer intermediate-sym]
  (let [prod-idx (:idx producer)
        cons-idx (:idx consumer)
        ;; The producer's map-lambda provides the body to inline
        prod-body (or (:map-lambda producer)
                      ;; For scan producers, inline the scan body
                      (when (seq (:scans producer))
                        (:lambda (first (:scans producer)))))
        ;; Substitute in consumer's relevant lambda
        substitute (fn [lambda]
                     (substitute-aget-sym lambda intermediate-sym
                                          prod-idx cons-idx prod-body))
        new-inputs (set/union (disj (:inputs consumer) intermediate-sym)
                              (:inputs producer))
        new-scalars (set/union (disj (:scalars consumer) intermediate-sym)
                               (:scalars producer))]
    (cond
      ;; Map→Map
      (and (:map-lambda producer) (:map-lambda consumer))
      (assoc consumer
             :map-lambda (substitute (:map-lambda consumer))
             :inputs new-inputs
             :scalars new-scalars)

      ;; Map→Reduce
      (and (:map-lambda producer) (seq (:reduces consumer)))
      (assoc consumer
             :reduces (mapv (fn [r] (update r :lambda substitute)) (:reduces consumer))
             :inputs new-inputs
             :scalars new-scalars)

      ;; Map→Scan
      (and (:map-lambda producer) (seq (:scans consumer)))
      (assoc consumer
             :scans (mapv (fn [s] (update s :lambda substitute)) (:scans consumer))
             :inputs new-inputs
             :scalars new-scalars)

      ;; Scan→Map
      (and (seq (:scans producer)) (:map-lambda consumer))
      (assoc consumer
             :scans (:scans producer)
             :map-lambda (substitute (:map-lambda consumer))
             :inputs new-inputs
             :scalars new-scalars)

      :else
      (throw (ex-info "Unsupported Screma composition"
                      {:producer producer :consumer consumer})))))
