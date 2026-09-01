(ns raster.compiler.ir.soac
  "SOAC (Second-Order Array Combinators) IR for fusion.

  Lifts flat let* bindings containing raster.par/* forms into typed
  SOAC nodes that a dependency graph can reason about. Provides
  round-trip conversion: par S-expr ↔ SOAC record ↔ par S-expr.

  SOAC types mirror Futhark's combinator classification:
    SoacMap     — element-wise parallel map
    SoacReduce  — parallel fold with associative operator
    SoacScan    — parallel prefix scan (inclusive)
    ScalarBinding — non-parallel scalar/allocation binding"
  (:require [raster.compiler.ir.par :as par]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.form :as form]
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
            reduction   ;; ProductReduction — scalar reduce is one component
            segment-axes ;; ordered [[idx bound] ...], empty for a scalar/full reduction
            bound       ;; expr — iteration count
            inputs      ;; #{sym} — array symbols read
            outputs     ;; ordered logical result symbols (nil components are not materialized)
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

(defrecord ScalarBinding
           [id          ;; int
            sym         ;; symbol
            expr])      ;; the RHS expression

;; ================================================================
(defrecord SoacContract
           [id           ;; int
            sym          ;; symbol (binding) — the contraction's output
            facts])      ;; contraction-facts — THE sole semantic payload (ir/contraction_facts)
;; A contraction as a first-class SOAC node. `facts` is the whole payload: free/contract axes,
;; operands with VERIFIED axis-maps, :decode, :combine/:init, dtype pair, :stages, :epilogue,
;; :roles, :dims. Nothing is copied out of it onto the record (north-star §10: one registry);
;; graph/scheduling projections are DERIVED (`soac-inputs`/`soac-outputs` below). `Dot` is the
;; representation this replaces — it had one constructor (a unit test), was excluded from lowering,
;; reconstruction and fusion, and could not carry most of the facts.

(defn- record-kind?
  "Recognize a SOAC record without pinning the caller to one DynamicClassLoader instance.

   Typed Clojure may analyze and re-evaluate this namespace in a child DynamicClassLoader. A
   record produced after that re-evaluation has the same public record type but is not `instance?`
   the class literal captured earlier by a compiler pass. Dispatching through this namespace's Var
   and comparing the stable generated class name keeps the IR boundary valid across that reload."
  [record-class node]
  (and node (= record-class (.getName (class node)))))

(defn soac-map? [node] (record-kind? "raster.compiler.ir.soac.SoacMap" node))
(defn soac-reduce? [node] (record-kind? "raster.compiler.ir.soac.SoacReduce" node))
(defn soac-scan? [node] (record-kind? "raster.compiler.ir.soac.SoacScan" node))
(defn scalar-binding? [node] (record-kind? "raster.compiler.ir.soac.ScalarBinding" node))
(defn screma? [node] (record-kind? "raster.compiler.ir.soac.Screma" node))
(defn contract? [node] (record-kind? "raster.compiler.ir.soac.SoacContract" node))

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

(defn- single-aset-void
  "If a par/map-void! body is a SINGLE in-place write `(aset OUT idx VAL)` whose
   index is exactly the loop index (a 1:1 elementwise write — NOT an offset/scatter
   write like cache[base+i]), return {:out OUT :value VAL :cast cf}. Otherwise nil.

   Unwraps a `do` with one statement and any let* scaffolding the walker emits
   around the write (e.g. `(let* [g (aget gate i)] (aset tmp i …g…))`), inlining
   those bindings into VAL so the returned :value is one flat expression — the
   same shape a pure map's lambda has, which is what lets the SAME vertical-fusion
   path inline it downstream. Recognizes bare `aset` and walker-devirtualized
   `(.invk aset-impl …)` via the op-descriptor matchers.

   The let* body must be a SINGLE form. A multi-statement let* body
   `(let* [v …] (aset a i v) (aset b i …))` is NOT a single write: modelling it as
   a SoacMap over `b` DROPPED the store to `a` outright (soac->par-form rebuilds the
   body from the lambda alone), and the store vanished from the emitted kernel —
   a silent miscompile, not a missed fusion. Anything with more than one statement
   returns nil ⇒ ScalarBinding ⇒ the legacy void path, which emits the body as
   written."
  [body idx-sym]
  (let [stmt (if (and (seq? body) (= 'do (first body)) (= 2 (count body)))
               (second body) body)]
    (cond
      ;; let*-wrapped write: recurse into the SOLE let body form, then inline the bindings
      (and (seq? stmt) (form/let-head? (first stmt)))
      (let [[_ binds & lbody] stmt]
        (when (and (= 1 (count lbody))
                   ;; PURITY GATE. This inlining duplicates each binding into the lambda, so an
                   ;; effectful init would have its effect duplicated or reordered. The old
                   ;; `postwalk-replace` copy had no gate and rested on an unstated `walker bindings
                   ;; are SSA` precondition. Refusing here is free: nil ⇒ ScalarBinding ⇒ the legacy
                   ;; void path, which emits the body as written.
                   (not-any? util/effectful? (take-nth 2 (rest binds))))
          (when-let [inner (single-aset-void (first lbody) idx-sym)]
            ;; capture-avoiding: an init that rebinds an earlier name internally, or a name that
            ;; also appears in a binder/array position, is not corrupted the way postwalk-replace
            ;; corrupted it.
            (update inner :value #(util/subst-syms (util/binding-env binds) %)))))

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
  Returns a SoacMap, SoacReduce, SoacScan, SoacContract, or nil.

  Stencil source intentionally returns nil: its only compiler algorithm is the validated
  TypedSOAC stencil equation. The compatibility graph must not reconstruct a second stencil IR."
  [sym expr id & {:keys [dtype]}]
  (cond
    ;; par/contract → SoacContract. Was DECLINED here (:no-lowering-rule) — the one par form that
    ;; was not a SOAC node, so it bypassed the segop boundary through a backend side branch. The
    ;; facts are derived ONCE, here; every consumer downstream reads them instead of re-parsing.
    (and (seq? expr) (= 'raster.par/contract (first expr)))
    (->SoacContract id sym ((requiring-resolve 'raster.compiler.ir.contraction-facts/contraction-facts)
                            expr :dtype (or dtype :double)))

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
      (cond-> (->SoacReduce id sym
                            (reduction/scalar
                             {:accumulator (:acc info)
                              :neutral (:init info)
                              :dtype (or elem-type dtype :double)
                              :result sym
                              :index (:idx info)
                              :step-result (:body info)
                              :attributes {:source :raster.par/reduce}})
                            [] (:bound info) inputs [sym] scalars)
        elem-type (assoc :elem-type elem-type)))

    (par/par-product-reduce-form? expr)
    (let [{:keys [outputs components segment-axes idx bound
                  element-bindings element-results combine-parameters
                  combine-bindings combine-results algebra]}
          (par/extract-par-product-reduce-info expr)
          component-records
          (mapv (fn [[component-id [acc neutral dtype] result]]
                  {:id component-id :accumulator acc :neutral neutral :dtype dtype :result result})
                (map vector (range) components outputs))
          region-form (list 'let* element-bindings (vec element-results))
          combine-form (list 'let* combine-bindings (vec combine-results))
          accumulators (set (map first components))
          segment-indices (set (map first segment-axes))
          inputs (collect-aget-arrays region-form)
          output-set (set (filter symbol? outputs))
          excluded (set/union inputs output-set accumulators segment-indices #{idx}
                              (set (take-nth 2 element-bindings))
                              (set (mapcat identity combine-parameters))
                              (set (take-nth 2 combine-bindings))
                              descriptor/aget-ops descriptor/aset-ops
                              #{'do 'let 'let* 'if 'double 'float 'int 'long})
          scalar-uses (set/union (util/free-syms region-form) (util/free-syms combine-form)
                                 (reduce set/union #{} (map (comp util/free-syms second) components)))
          scalars (set/difference scalar-uses excluded)
          operator (reduction/make
                    {:components component-records
                     :index idx
                     :element-bindings element-bindings
                     :element-results element-results
                     :combine-parameters combine-parameters
                     :combine-bindings combine-bindings
                     :combine-results combine-results
                     :algebra algebra
                     :attributes {:source :raster.par/product-reduce!}})]
      (->SoacReduce id sym operator (vec segment-axes) bound inputs (vec outputs) scalars))

    ;; Scan: (raster.par/scan out acc init idx bound cast body)
    (and (seq? expr) (= 'raster.par/scan (first expr)))
    (let [[_ out-sym acc-sym init-expr idx-sym bound-expr cast-fn body-expr] expr
          {:keys [inputs outputs scalars]}
          (extract-io body-expr idx-sym [out-sym] :acc-sym acc-sym)
          elem-type (:raster.type/elem-type (meta expr))]
      (cond-> (->SoacScan id sym out-sym acc-sym init-expr idx-sym bound-expr
                          cast-fn body-expr inputs outputs scalars)
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
     ;; a contraction reconstructs to its ORIGINAL surface form — the facts keep it for exactly
     ;; this dialect boundary (the CPU path round-trips SOAC nodes back to par forms after
     ;; fusion; before contract was a node it was never round-tripped, so no arm existed)
     SoacContract
     (:form (:facts soac))

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
     (if (= :raster.par/product-reduce! (get-in soac [:reduction :attributes :source]))
       (list 'raster.par/product-reduce!
             (reduction/results (:reduction soac))
             (mapv (fn [component]
                     [(:accumulator component) (:neutral component) (:dtype component)])
                   (:components (:reduction soac)))
             (:segment-axes soac)
             (:index (:reduction soac)) (:bound soac)
             (:bindings (reduction/element-region (:reduction soac)))
             (:results (reduction/element-region (:reduction soac)))
             (:parameters (reduction/combine-region (:reduction soac)))
             (:bindings (reduction/combine-region (:reduction soac)))
             (:results (reduction/combine-region (:reduction soac)))
             (:algebra (:reduction soac)))
       (let [{:keys [acc init lambda]} (reduction/scalar-op (:reduction soac))]
         (list 'raster.par/reduce acc init
               (:index (:reduction soac)) (:bound soac) lambda)))

     SoacScan
     (list 'raster.par/scan (:out soac) (:acc soac) (:init soac)
           (:idx soac) (:bound soac) (:cast-fn soac) (:lambda soac))

     ;; Not a SOAC
     (throw (ex-info "Cannot convert non-SOAC to par form" {:node soac})))
   soac))

;; ================================================================
;; Bulk conversion: let* bindings ↔ SOAC/Scalar nodes
;; ================================================================

(defn let-bindings->nodes
  "Convert a sequence of [sym expr] binding pairs to SOAC/Scalar nodes.
  Each pair gets a sequential id for ordering.

  This compatibility adapter must not turn a legal sequential `par/reduce` fallback into an
  invalid parallel tree. If the canonical reduction constructor declines reassociation (for
  example a finite max sentinel rather than the true typed identity), retain the complete source
  expression as a ScalarBinding. Direct typed routing remains strict and reports its structured
  decline separately."
  [pairs]
  (vec
   (map-indexed
    (fn [id [sym expr]]
      (let [converted
            (try
              (par-form->soac sym expr id)
              (catch clojure.lang.ExceptionInfo exception
                (if (and (par/par-reduce-form? expr)
                         (some-> exception ex-data :reason name
                                 (.startsWith "reduction-")))
                  nil
                  (throw exception))))]
        (or converted (->ScalarBinding id sym expr))))
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
  "Check if a node is a generic compatibility SOAC (map/reduce/scan) — NOT a ScalarBinding, and NOT a
   SoacContract (a contraction has its own lowering; generic map/reduce lowering must skip it)."
  [node]
  (and (not (scalar-binding? node))
       ;; a contraction has its own lowering (SegContract) and must NOT enter the generic
       ;; map/reduce/scan lowering or lambda fusion, whose consumers assume idx/lambda/1-D bound
       (not (contract? node))))

(defn soac-inputs
  "Get the set of input array symbols for a SOAC node."
  [node]
  (if (contract? node)
    (set (map :sym (:operands (:facts node))))
    (when (soac? node) (:inputs node))))

(defn soac-outputs
  "Get the set of output symbols for a SOAC node."
  [node]
  (if (contract? node)
    #{(:out (:facts node))}
    (cond
      (soac-map? node)     (:outputs node)
      (soac-reduce? node)  (set (filter symbol? (:outputs node)))
      (soac-scan? node)    (:outputs node)
      :else nil)))

(defn soac-bound
  "Get the iteration bound expression for a SOAC node."
  [node]
  (when (soac? node) (:bound node)))

(defn soac-idx
  "Get the index variable for a SOAC node."
  [node]
  (when (soac? node)
    (if (soac-reduce? node) (:index (:reduction node)) (:idx node))))

(defn node-all-free-syms
  "Get all free symbols referenced by a node (inputs + outputs + scalars + bound + init).
  Outputs are included for imperative SOACs because they write to pre-allocated
  output buffers that must be defined before the SOAC runs. Pure par/map SOACs
  don't have external output buffers — their output IS their binding symbol."
  [node]
  (if (instance? ScalarBinding node)
    (util/free-syms (:expr node))
    (set/union (or (:inputs node) #{})
               (if (:pure? node) #{} (set (filter symbol? (or (:outputs node) #{}))))
               (or (:scalars node) #{})
               (util/free-syms (:bound node))
               (if (soac-reduce? node)
                 (reduce set/union #{} (map util/free-syms (reduction/neutrals (:reduction node))))
                 (if (:init node) (util/free-syms (:init node)) #{})))))

(defn- expr-written-arrays
  "Arrays written via aset (bare or devirtualized) anywhere in `expr`. Used so a
   ScalarBinding wrapping a side-effecting void form (a multi-aset par/map-void!
   like rms-norm/attention/quant-act that doesn't reduce to a single-aset SoacMap)
   still declares the buffers it produces — otherwise consumers of those buffers
   get no dependency edge and the scheduler / horizontal-fuser can incorrectly
   reorder or parallelize across the write (silent miscompile in composed layers).

   KNOWN GAP (silently-ignored-information family, deferred): this sees ONLY aset
   writes. An op whose output buffer is declared via the op-descriptor :buffer-write
   registry but that writes with NO literal aset in this expr (a BLAS/GEMM .invk that
   writes C in place, a runtime scatter) contributes NO producer edge here, so the
   SOAC topological sort could reorder a consumer ahead of it. The registry DCE now
   consults IS the right source of truth, but wiring it in here is a non-trivial change
   (this runs pre-lowering where those ops are still symbolic deftm calls, not the
   devirtualized .invk the registry keys on). Fixing it belongs with the descriptor
   VALIDATOR (S4). Until then a genuinely reorderable buffer-write-only op reaching the
   SOAC scheduler is the outstanding risk — documented, not yet guarded."
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
          (if (seq (:segment-axes soac))
            soac
            (->Screma (:id soac) (:sym soac) (:index (:reduction soac)) (:bound soac)
                      (:inputs soac) (set (filter symbol? (:outputs soac))) (:scalars soac)
                      nil [] [(:reduction soac)]
                      nil))

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
          (let [{:keys [acc init lambda]} (reduction/scalar-op (first (:reduces screma)))]
            (list 'raster.par/reduce acc init
                  (:idx screma) (:bound screma) lambda))

          ;; Pure scan: one scan, no map-lambda
          (and (= 1 (count (:scans screma))) (empty? (:reduces screma)) (nil? (:map-lambda screma)))
          (let [s (first (:scans screma))]
            (list 'raster.par/scan (:out s) (:acc s) (:init s)
                  (:idx screma) (:bound screma) (:cast-fn screma) (:lambda s)))

          ;; Map+Reduce: a reduce WITH a still-present map-lambda. screma-compose's Map→Reduce
          ;; inlines the producer map body INTO the reduce lambda and leaves :map-lambda nil, so
          ;; a screma reaching here with both set is an UN-fused map — emitting only the reduce
          ;; lambda (as this arm used to) would SILENTLY DROP the map computation. Reject loudly.
          (and (empty? (:scans screma)) (= 1 (count (:reduces screma))) (:map-lambda screma))
          (throw (ex-info (str "screma->par-form: Map+Reduce screma still carries a :map-lambda"
                               " — the map body was not inlined into the reduce lambda; emitting"
                               " the reduce alone would drop the map computation")
                          {:screma screma}))

          :else
          (throw (ex-info "Cannot convert complex Screma to single par form"
                          {:screma screma})))]
    (if et (vary-meta form assoc :raster.type/elem-type et) form)))

(defn- substitute-aget-sym
  "Replace (aget target-sym idx) with replacement-expr in body,
  adjusting index variable from src-idx to dst-idx.

  SAME-POSITION ONLY (layout-soundness guard): vertical Screma fusion inlines the producer's body
  (which computes the intermediate element at the CONSUMER's iteration index dst-idx) in place of
  the consumer's read. That is correct iff the consumer reads the intermediate at its OWN iteration
  index — i.e. `(aget target-sym dst-idx)`. A consumer that reads at any other index (a gather /
  transpose / neighbour / offset read) is NOT elementwise-fusible without a layout convert, and
  inlining the producer body there would silently compute the wrong element. Rather than mis-fuse
  (the former index-insensitive behaviour), we FAIL LOUD — the layout-inference pass is where such a
  read gets a `convert_layout` instead. This never fires for the same-position case that is all that
  reaches fusion today; it closes the documented hazard for the non-same-position future."
  [body target-sym src-idx dst-idx replacement-expr]
  (walk/postwalk
   (fn [f]
     (if (and (seq? f)
              (descriptor/aget-op? (first f))
              (>= (count f) 3)
              (symbol? (second f))
              (= (name target-sym) (name (second f))))
       (let [read-idx (nth f 2)]
         (when-not (or (= read-idx dst-idx) (= read-idx src-idx))
           (throw (ex-info (str "Screma fusion: non-same-position read of intermediate '" target-sym
                                "' at index " (pr-str read-idx) " ≠ iteration index " (pr-str dst-idx)
                                " — gather/transpose/offset reads are not elementwise-fusible without a"
                                " layout convert (layout-inference pass); refusing to mis-fuse.")
                           {:target target-sym :read-idx read-idx :dst-idx dst-idx :src-idx src-idx})))
         (walk/postwalk
          (fn [g] (if (= g src-idx) dst-idx g))
          replacement-expr))
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
             :reduces (mapv (fn [r] (update-in r [:step :results] #(mapv substitute %)))
                            (:reduces consumer))
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
