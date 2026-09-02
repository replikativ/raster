(ns raster.compiler.ir.soac
  "Compatibility SOAC descriptions for direct backend doors.

  Parses individual raster.par/* forms into records consumed by compatibility SegOp lowering.
  Functional programs and fusion live exclusively in `ir.soac-dialect` (TypedSOAC).

  SOAC types mirror Futhark's combinator classification:
    SoacMap     — element-wise parallel map
    SoacReduce  — parallel fold with associative operator
    SoacScan    — parallel prefix scan (inclusive)"
  (:require [raster.compiler.ir.par :as par]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.form :as form]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [clojure.set :as set]))

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
  the compatibility operation—never a scalar. The structural out slot alone
  is not enough for map bodies that carry secondary side-effect stores."
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
   same shape as a pure map's lambda. Recognizes bare `aset` and walker-devirtualized
   `(.invk aset-impl …)` via the op-descriptor matchers.

   The let* body must be a SINGLE form. A multi-statement let* body
   `(let* [v …] (aset a i v) (aset b i …))` is NOT a single write: modelling it as
   a SoacMap over `b` would drop the store to `a` in compatibility lowering. Anything
   with more than one statement returns nil so the complete source stays on the
   specialized void path."
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
                   ;; are SSA` precondition. Refusing here preserves the complete source for the
                   ;; specialized void path.
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
          _ (when (:offset info)
              (throw (ex-info "offset map requires an explicit indexed-store operation"
                              {:reason :offset-map-requires-indexed-store
                               :operation 'raster.par/map!
                               :offset (:offset info)
                               :target-dialect :typed-soac-scatter
                               :fallback :scalar-expansion})))
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
    ;; LAMBDA is the written value. This is the narrow compatibility shape that
    ;; generic SegMap lowering can preserve. Multi-aset, offset-scatter, and reduction
    ;; bodies return nil and stay on their specialized source-level paths.
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
;; Predicates and accessors
;; ================================================================

(defn soac-inputs
  "Get the set of input array symbols for a SOAC node."
  [node]
  (if (contract? node)
    (set (map :sym (:operands (:facts node))))
    (:inputs node)))

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

(defn soac-idx
  "Get the index variable for a SOAC node."
  [node]
  (if (soac-reduce? node) (:index (:reduction node)) (:idx node)))

(defn node-all-free-syms
  "Get all free symbols referenced by a node (inputs + outputs + scalars + bound + init).
  Outputs are included for imperative SOACs because they write to pre-allocated
  output buffers that must be defined before the SOAC runs. Pure par/map SOACs
  don't have external output buffers — their output IS their binding symbol."
  [node]
  (set/union (or (:inputs node) #{})
             (if (:pure? node) #{} (set (filter symbol? (or (:outputs node) #{}))))
             (or (:scalars node) #{})
             (util/free-syms (:bound node))
             (if (soac-reduce? node)
               (reduce set/union #{} (map util/free-syms (reduction/neutrals (:reduction node))))
               (if (:init node) (util/free-syms (:init node)) #{}))))
