(ns raster.compiler.passes.parallel.segop-lower-pass
  "Pipeline pass: lower par forms to SegOp records.

   Walks let* bindings and converts raster.par/* forms to SegOp IR via the SOAC intermediate.
   The result is a first-class ParallelProgram whose typed equations own the SegOps.  Binding
   metadata is not an IR transport.

   This decouples hardware-aware execution planning from backend codegen:
   - Lowering decides phase decomposition, launch params, accumulator count
   - Backend translates SegOp to target code (SIMD, OpenCL, scalar)"
  (:require [raster.compiler.core.util :as util]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.types :as types]
            [raster.compiler.ir.par :as par]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.ir.soac-dialect :as soac-dialect]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]
            [raster.compiler.passes.parallel.segred-body :as segred-body]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.ir.form :as form]
            [clojure.set :as set]))

(def ^:private id-counter (atom 0))

(defn- diagnostic
  "The structured record north-star §3.5 asks for in place of a warning: WHICH operation, in which
   binding, for which target dialect and device, what rule was missing, and what happens instead."
  [sym form stage ^Exception e device-id dtype]
  {:op (when (seq? form) (first form))
   :sym sym
   :stage stage                     ; :soac (par form → SOAC) or :segop (SOAC → SegOp)
   :target-dialect :segop
   :device (or device-id :cpu:0)
   :dtype (or dtype :double)
   :reason (or (:reason (ex-data e)) :no-lowering-rule)
   :message (.getMessage e)
   ;; a fallback is a stated outcome, not the absence of one
   :fallback :backend-relowers-or-uses-specialized-codegen})

(defn- lower-attempt
  "Lower a par form to SegOp records via the SOAC intermediate.

   Returns `{:segops [...]}` on success, `{:declined <diagnostic>}` when the form IS a parallel
   primitive but no lowering rule applies, and nil when the form is simply not a par form — the
   common, correctly-silent case (most bindings are ordinary values).

   This used to `println` a WARNING to stderr and return nil for BOTH of the last two cases. That
   conflation is the defect: `nil` meant \"nothing to do here\" and \"a parallel form the middle end
   cannot represent\" at once, so a real coverage gap looked exactly like an ordinary binding and the
   only trace was a line on stderr that no pass, stat, or diagnostic could see. north-star §3.5 names
   this precise code: SegOp lowering \"may no longer warn and return nil\".

   The lowering ATTEMPT is unchanged — still tried for any seq, so nothing that lowered before stops
   lowering now. What is new is that a failure on a recognized par form is reported as data.

   NB success/failure is carried in an explicit `{:ok …}`/`{:err …}` rather than a truthy value: a
   SOAC node is a RECORD, and records satisfy `map?` and are always truthy, so a compact
   `or`/`if-let` version silently misread every successful lowering as a decline marker."
  [sym form device-id target-descriptor dtype array-types scalar-types]
  (when (seq? form)
    (let [par? (par/par-form? form)
          decline (fn [stage e] (util/rethrow-compiler-invariant! e)
                    (when par? {:declined (diagnostic sym form stage e device-id dtype)}))
          ;; capture value-or-exception in one call: the alternative (catch a sentinel, then call
          ;; again to get the exception) re-runs a side-effecting conversion
          ;; Only an intentional, structured conversion refusal may become a decline. A raw
          ;; NullPointerException/ClassCastException/etc. is an implementation bug and must escape;
          ;; treating it as "no lowering rule" would silently select a different backend path.
          attempt (fn [f] (try {:ok (f)} (catch clojure.lang.ExceptionInfo e {:err e})))
          soac (attempt #(soac/par-form->soac sym form (swap! id-counter inc) :dtype (or dtype :double)))]
      (cond
        (:err soac) (decline :soac (:err soac))

        (nil? (:ok soac))
        (when par? {:declined (diagnostic sym form :soac
                                          (ex-info "par-form->soac produced no SOAC node" {})
                                          device-id dtype)})

        :else
        (let [legacy-node (:ok soac)
              segops (attempt #(soac-lower/lower-soac legacy-node (or device-id :cpu:0)
                                                     :dtype (or dtype :double)))]
          (cond
            (:err segops) (decline :segop (:err segops))
            (seq (:ok segops)) (cond-> {:soac legacy-node :segops (:ok segops)}
                                 (soac-lower/scan-soac? legacy-node)
                                 (assoc :kernel-graph
                                        (soac-lower/scan-kernel-graph
                                         legacy-node (:ok segops) {:array-types array-types})))
            :else (when par? {:declined (diagnostic sym form :segop
                                                    (ex-info "lower-soac produced no SegOps" {})
                                                    device-id dtype)})))))))

(defn- unknown-vector-shape [] ['?])

(defn- result-shape
  [node]
  (cond
    (soac/soac-reduce? node) []
    ;; An imperative map/scan binding aliases a caller-provided output buffer. Its iteration domain
    ;; is not proof of the buffer's physical/logical extent (padded rows and strided views are common).
    :else (unknown-vector-shape)))

(defn- declared-scalar-type!
  [id provided kernel-dtype]
  (let [source (some-> (types/sym-type-tag id) dtype/dtype-for-scalar-tag)
        source (when source
                 (if (and kernel-dtype (dtype/fp-dtype? source)
                          (dtype/fp-dtype? kernel-dtype))
                   (dtype/canon kernel-dtype) source))
        provided (some-> provided dtype/canon)]
    (when (and source provided (not= source provided))
      (throw (ex-info "compatibility scalar type contradicts its retained source declaration"
                      {:reason :parallel-program-source-type-conflict
                       :value id :source-dtype source :provided-dtype provided})))
    (or provided source)))

(defn- value-contract
  [id node dtype array-types scalar-types result?]
  (let [array-ids (set/union (or (soac/soac-inputs node) #{})
                             (or (soac/soac-outputs node) #{}))
        shape (cond
                result? (result-shape node)
                ;; A scalar reduction's indexed reads are certified over its reduction extent.
                ;; Retain that minimum physical capacity instead of the compatibility `?` shape;
                ;; graph staging can then prove the input is large enough without guessing.
                (and (contains? array-ids id) (soac/soac-reduce? node)) [(:bound node)]
                (contains? array-ids id) (unknown-vector-shape)
                :else [])
        type-id (if (and result? (soac/contract? node)) (get-in node [:facts :out]) id)
        declared-types (if (contains? array-ids type-id) array-types scalar-types)
        provided (or (get declared-types type-id)
                     (when (symbol? type-id) (get declared-types (symbol (name type-id)))))
        value-dtype (or (if (contains? array-ids type-id)
                         provided
                         (declared-scalar-type! type-id provided dtype))
                        ;; Compatibility scheduling must retain integral source widths before
                        ;; re-entry into TypedSOAC. Floating specialization keeps its existing
                        ;; precision policy; a kernel's element dtype is not an integer type.
                        (:elem-type node)
                        dtype
                        :double)]
    (av/tensor {:dtype value-dtype
                :shape shape
                :representation {:kind :plain}
                :effects #{}})))

(defn- equation
  [equation-id site sym source {:keys [soac algorithm segops kernel-graph]}
   dtype array-types scalar-types]
  (let [;; A scalar reduction is value-producing even in body position. Give that expression the
        ;; same internal SSA envelope as a source binding: whether an algorithm can be scheduled
        ;; must not depend on surface syntax. Effect-only body operations still have no result.
        value-producing? (or (= :binding (first site))
                             (and (soac/soac-reduce? soac)
                                  (empty? (:segment-axes soac))
                                  (reduction/scalar? (:reduction soac))))
        algorithm (when value-producing? algorithm)
        operands (-> (soac/node-all-free-syms soac) (disj sym) (->> (sort-by str) vec))
        result-ids (if value-producing? [sym] [])
        effects (cond-> #{:memory/read}
                  (seq (soac/soac-outputs soac)) (conj :memory/write))
        operand-values (into {}
                             (map (fn [id]
                                    [id (value-contract id soac dtype array-types scalar-types false)]))
                             operands)
        result-values (into {}
                            (map (fn [id]
                                   [id (value-contract id soac dtype array-types scalar-types true)]))
                            result-ids)
        eq (program/->ProgramEquation
            equation-id site source operands result-ids algorithm (vec segops) effects
            {:source-dialect :soac :target-dialect :segop :soac-id (:id soac)}
            (cond-> {:device (:device-id (first segops))}
              kernel-graph (assoc :kernel-graph kernel-graph)))]
    {:equation eq :operand-values operand-values :result-values result-values}))

(defn- merge-values
  "Merge independently inferred value contracts and reject inconsistent views of one value ID."
  [left right]
  (reduce-kv
   (fn [values id contract]
     (if-let [prior (get values id)]
       (if (= prior contract)
         values
         (throw (ex-info "parallel equations inferred incompatible contracts for one value"
                         {:reason :parallel-program-value-conflict
                          :id id :first prior :second contract})))
       (assoc values id contract)))
   left right))

(defn- ensure-use-compatible!
  [id definition inferred-use context]
  (let [facets [:kind :dtype :representation]
        defined (select-keys definition facets)
        inferred (select-keys inferred-use facets)]
    (when-not (= defined inferred)
      (throw (ex-info "parallel equation use is incompatible with its defining value"
                      (merge {:reason :parallel-program-use-type-conflict
                              :id id :definition defined :use inferred}
                             context)))))
  definition)

(defn- declare-scheduled-temporaries
  "Declare physical SSA storage introduced by a multi-phase schedule.

   Both typed whole-program lowering and compatibility discovery use this one rule. A backend must
   never learn a partial buffer's dtype or extent from its generated name."
  [values equations]
  (reduce
   (fn [values equation]
     (let [values
           (reduce (fn [values temporary]
                     (let [id (:id temporary)]
                       (if (contains? values id)
                         values
                         (assoc values id
                                (av/tensor
                                 {:dtype (:dtype temporary)
                                  :shape (soac-dialect/extent-shape (:elements temporary))
                                  :representation {:kind :plain}
                                  :memory-space (:memory-space temporary)})))))
                   values
                   (get-in equation [:attributes :kernel-graph :temporaries]))]
       (reduce
        (fn [values operation]
          (if (and (segop/seg-red? operation)
                   (= :block-local (:phase operation)))
            (let [grid (:grid operation)
                  reduced-bound (-> operation :space segop/seg-space-reduced-dim :bound)
                  partial-extent
                  (segred-body/launch-group-count
                   (:num-blocks grid) reduced-bound (:block-size grid))]
              (reduce (fn [values id]
                        (if (contains? values id)
                          values
                          (assoc values id
                                 (av/tensor
                                  {:dtype (:dtype operation)
                                   :shape (soac-dialect/extent-shape partial-extent)
                                   :representation {:kind :plain}
                                   :memory-space :device}))))
                      values (:outputs operation)))
            values))
        values (:operations equation))))
   values equations))

(defn- build-program
  [source lowered-equations declined device-id dtype]
  (let [{:keys [equations values]}
        (reduce
         (fn [{:keys [environment] :as state}
              {:keys [equation operand-values result-values physical-values]}]
           (let [source-results (:results equation)
                 operands (mapv #(get environment % %) (:operands equation))
                 ;; Source binders and pre-existing buffers may share a spelling in imperative IR.
                 ;; Give equation results their own SSA-like IDs so a scalar binding can never
                 ;; collide with an array operand of the same name.
                 ;; A body-position result is already an internal compiler identity and has no
                 ;; user binding whose successive definitions need disambiguation. Keeping that
                 ;; symbol also matches the typed reduction component's physical result contract.
                 results (if (= :body (first (:site equation)))
                           source-results
                           (mapv (fn [source-id] [:binding source-id]) source-results))
                 values-with-operands
                 (reduce (fn [values [source-id value-id]]
                           ;; A mapped operand already has the defining equation's authoritative
                           ;; contract. Only infer contracts for external program inputs here.
                           (if-let [definition (get values value-id)]
                             (do (ensure-use-compatible! value-id definition
                                                         (get operand-values source-id)
                                                         {:source-id source-id
                                                          :equation-id (:id equation)
                                                          :equation-site (:site equation)
                                                          :equation-source (:source equation)})
                                 values)
                             (merge-values values {value-id (get operand-values source-id)})))
                         (merge-values (:values state) (or physical-values {}))
                         (map vector (:operands equation) operands))
                 values-with-results
                 (reduce (fn [values [source-id value-id]]
                           (merge-values values {value-id (get result-values source-id)}))
                         values-with-operands
                         (map vector source-results results))
                 value-remap (into (zipmap source-results results)
                                   (map vector (:operands equation) operands))
                 algorithm' (when-let [algorithm (:algorithm equation)]
                              (soac-dialect/remap-values algorithm value-remap))
                 equation' (-> equation
                               (assoc :operands operands :results results :algorithm algorithm')
                               (update :attributes assoc :source-results source-results))]
             (-> state
                 (assoc :values values-with-results)
                 (update :equations conj equation')
                 (update :environment into (map vector source-results results)))))
         {:environment {} :values {} :equations []}
         lowered-equations)
        values (declare-scheduled-temporaries values equations)
        result-ids (set (mapcat :results equations))
        operand-ids (set (mapcat :operands equations))
        inputs (->> (set/difference operand-ids result-ids) (sort-by str) vec)
        outputs (->> equations (mapcat :results) distinct vec)
        effects (reduce set/union #{} (map :effects equations))]
    (program/make
     {:dialect :segop
      :source source
      :values values
      :inputs inputs
      :equations equations
      :outputs outputs
      :effects effects
      :diagnostics declined
      :provenance {:pass :segop-lower :device (or device-id :cpu:0) :dtype (or dtype :double)}
      :attributes {:host-control :source-expression}
      :operation? segop/segop-node?
      :algorithm? (fn [equation algorithm]
                    (and (soac-dialect/program-form? algorithm)
                         (= algorithm (soac-dialect/validate! algorithm))
                         (= (:operands equation) (:inputs (soac-dialect/facts algorithm)))
                         (= (:results equation) (soac-dialect/outputs algorithm))))})))

(defn- equation-physical-results
  [algorithm]
  (let [facts (soac-dialect/facts algorithm)
        producers
        (into {}
              (mapcat (fn [equation]
                        (map vector (nth equation 2)
                             (soac-dialect/physical-results facts equation))))
              (soac-dialect/equations algorithm))]
    (mapv producers (soac-dialect/outputs algorithm))))

(declare segop-lower-pass)

(defn- singleton-packet
  "Admit, freshen and schedule the complete typed singleton, including host SSA."
  [result source opts used kind]
  (let [packet-stage (if (= kind 'reduce) :typed-reduction-packet :typed-map-packet)
        source-stage (if (= kind 'reduce) :typed-reduction-source :typed-map-source)]
    (let [attempt (typed-route/attempt
                   (list 'let* [result source] result)
                   (or (:dtype opts) :double) (:array-types opts)
                   {:scalar-types (:scalar-types opts)
                    :resident-reductions? (true? (:resident-reductions? opts))})]
      (if (:declined attempt)
        {:declined (:declined attempt)}
        (let [packet (:program attempt)
              _ (when-not (and (program/parallel-program? packet)
                               (= :typed-soac (:dialect packet))
                               (seq (:equations packet)))
                  (throw (ex-info (str "typed " (name kind) " admission requires a complete packet")
                                  {:reason :raster/bug :stage packet-stage
                                   :source source})))
              _ (program/validate! packet)
              source-form (:source packet)
              _ (when-not (and (form/binding-form? source-form)
                               (vector? (second source-form))
                               (even? (count (second source-form)))
                               (seq (nnext source-form))
                               (every? symbol? (take-nth 2 (second source-form))))
                  (throw (ex-info (str "typed " (name kind) " packet requires its ordered source realization")
                                  {:reason :raster/bug :stage source-stage
                                   :source source})))
              source-pairs (mapv vec (partition 2 (second source-form)))
              source-bindings (into {} source-pairs)
              _ (doseq [equation (:equations packet)]
                  (let [[site-kind binding] (:site equation)]
                    (when-not (and (= :binding site-kind)
                                   (contains? source-bindings binding)
                                   (= (:source equation) (get source-bindings binding)))
                      (throw (ex-info (str "typed " (name kind) " equation lacks its exact source witness")
                                      {:reason :raster/bug :stage source-stage
                                       :equation (:id equation) :site (:site equation)
                                       :source source})))))
              locals (disj (set (concat (map first source-pairs)
                                       (mapcat :results (:equations packet)))) result)
              fresh (fn [id]
                      (loop [candidate (with-meta (gensym (str (name id) "_")) (meta id))]
                        (if (contains? @used candidate) (recur (gensym (str (name id) "_")))
                            (do (swap! used conj candidate) candidate))))
              rename (into {} (map #(vector % (fresh %))) locals)
              rename-id #(get rename % %)
              pairs (mapv (fn [[id expression]]
                            ;; The realization owns source hints (in particular the absence of
                            ;; primitive :tag on let binders), not the value contract's metadata.
                            [(with-meta (rename-id id) (meta id))
                             (util/subst-syms rename expression)])
                          source-pairs)
              body (mapv #(util/subst-syms rename %) (nnext source-form))
              bindings (into {} pairs)
              equations
              (mapv (fn [equation]
                      (let [algorithm (soac-dialect/remap-values (:algorithm equation) rename)
                            site (update (:site equation) 1 rename-id)]
                        (-> equation
                            (assoc :site site :source (get bindings (second site))
                                   :operands (mapv rename-id (:operands equation))
                                   :results (mapv rename-id (:results equation))
                                   :algorithm algorithm
                                   :operations (vec (soac-dialect/equations algorithm)))
                            (update :attributes #(util/subst-syms rename %)))))
                    (:equations packet))
              values (reduce merge-values {} (map #(-> % :algorithm soac-dialect/facts :values) equations))
              packet (assoc packet :source (apply util/remake source-form 'let*
                                                 (vec (mapcat identity pairs)) body)
                                   :values values :inputs (mapv rename-id (:inputs packet))
                                   :outputs (mapv rename-id (:outputs packet)) :equations equations)
              scheduled (:form (segop-lower-pass packet opts))
              primary (some #(when (and (= [result] (:results %))
                                        (= kind (soac-dialect/operation-kind
                                                    (first (soac-dialect/equations (:algorithm %)))))) %)
                            (:equations scheduled))]
          (when-not (and primary
                         (= kind (soac-dialect/operation-kind
                                     (first (soac-dialect/equations (:algorithm primary)))))
                         (= [result] (:results primary)))
            (throw (ex-info (str "typed " (name kind) " packet lost its result equation")
                            {:reason :raster/bug :stage packet-stage :source source})))
          {:program scheduled :pairs pairs :body body :primary primary})))))

(defn- reduction-packet [result source opts used]
  (when (par/par-reduce-form? source)
    (singleton-packet result source opts used 'reduce)))

(defn- selected-operation-packet [result source opts used]
  (if (par/par-map-pure-form? source)
    (singleton-packet result source opts used 'map)
    (reduction-packet result source opts used)))

(defn- packet-equations
  [packet primary-site ids]
  (let [scheduled (:program packet)
        values (:values scheduled)
        produced (set (mapcat :results (:equations scheduled)))]
    (mapv (fn [equation]
            {:equation (-> equation
                           (assoc :id (swap! ids inc))
                           (update :provenance assoc :compatibility-source-site primary-site))
             :operand-values (select-keys values (:operands equation))
             :result-values (select-keys values (:results equation))
             :physical-values (select-keys values (set/intersection produced (set (:results equation))))})
          (:equations scheduled))))

(defn segop-lower-pass
  "Pipeline pass: convert par forms in let* bindings to SegOp records.

   Walks the form's let* bindings. For each par/map!, par/reduce, par/scan! binding, converts to a
   typed ProgramEquation containing ordered SegOps. The returned `:form` is a ParallelProgram;
   its `:source` is the undecorated host expression consumed around those equations.

   Scan decomposition additionally records one verified KernelGraph with its intermediate buffers
   and dependencies. Returns both `:segops-lowered` and `:kernel-graphs-lowered` stats.

   Options from pipeline opts:
     :target-device — device for launch param computation
     :target-descriptor — optional frozen descriptor for that compilation
     :dtype — element type (:double or :float)"
  [form opts]
  (if (and (program/parallel-program? form) (= :typed-soac (:dialect form)))
    ;; Every declared value of a TypedSOAC program is a local of the compiled form, even one
    ;; spelled like a `clojure.core` name (a parameter called `seq`): free-symbol analysis in
    ;; the lowerings must read it as a scalar operand, never as the core function.
    (binding [util/*shadowing-locals* (set (keys (:values form)))]
    (let [device-id (or (:target-device opts) :cpu:0)
          target-descriptor (:target-descriptor opts)
          dtype (or (:dtype opts) :double)
          ;; The TypedSOAC program remains purely functional, while `:result-storage` gives every
          ;; equation result its physical identity. Scheduling later equations must consume that
          ;; identity—not reintroduce the logical result as a second external graph buffer.
          equations
          (:equations
           (reduce
            (fn [{:keys [physical] :as state} equation]
              (let [algorithm (:algorithm equation)
                    scheduling-algorithm (soac-dialect/remap-values algorithm physical)
                    kind (soac-dialect/operation-kind
                          (first (soac-dialect/equations scheduling-algorithm)))
                    lowered (case kind
                              scalar {:operations []}
                              contract {:operations (soac-lower/lower-typed-contract
                                                      scheduling-algorithm device-id)}
                              map {:operations (soac-lower/lower-typed-map
                                                scheduling-algorithm device-id :dtype dtype
                                                :target-descriptor target-descriptor)}
                              scatter {:operations (soac-lower/lower-typed-scatter
                                                    scheduling-algorithm device-id :dtype dtype
                                                    :target-descriptor target-descriptor)}
                              effect-map
                              {:operations (soac-lower/lower-typed-effect-map
                                            scheduling-algorithm device-id :dtype dtype
                                            :target-descriptor target-descriptor)}
                              stencil {:operations (soac-lower/lower-typed-stencil
                                                    scheduling-algorithm device-id :dtype dtype
                                                    :target-descriptor target-descriptor)}
                              reduce {:operations (soac-lower/lower-typed-reduce
                                                   scheduling-algorithm device-id :dtype dtype
                                                   :target-descriptor target-descriptor)}
                              segmented-reduce
                              {:operations (soac-lower/lower-typed-segmented-reduce
                                            scheduling-algorithm device-id :dtype dtype
                                            :target-descriptor target-descriptor)}
                              product-reduce
                              {:operations (soac-lower/lower-typed-product-reduce
                                            scheduling-algorithm device-id :dtype dtype
                                            :target-descriptor target-descriptor)}
                              segmented-fold-map
                              {:operations (soac-lower/lower-typed-segmented-fold-map
                                            scheduling-algorithm device-id :dtype dtype
                                            :target-descriptor target-descriptor)}
                              scan (soac-lower/lower-typed-scan
                                    scheduling-algorithm device-id :dtype dtype
                                    :array-types (:array-types opts)
                                    :target-descriptor target-descriptor))
                    operations (:operations lowered)
                    scheduled-equation
                    (-> equation
                        (assoc :operations operations
                               ;; The algorithm in physical value names: a logical result that
                               ;; aliases an earlier destination is spelled as that destination
                               ;; here, exactly as the scheduled SegOps name it.
                               :physical-algorithm scheduling-algorithm)
                        (update :provenance assoc :target-dialect :segop)
                        (update :attributes assoc :device device-id)
                        (cond-> (:kernel-graph lowered)
                          (update :attributes assoc :kernel-graph (:kernel-graph lowered)))
                        (cond-> (= 'scalar kind)
                          (update :attributes assoc :host-only true)))
                    result-storage (zipmap (:results equation)
                                           (equation-physical-results algorithm))]
                (-> state
                    (update :equations conj scheduled-equation)
                    (update :physical merge result-storage))))
            {:physical {} :equations []}
            (:equations form)))
          ;; A multi-phase schedule introduces physical SSA values that do not exist in the
          ;; functional algorithm. They still require explicit contracts; an emitter must never
          ;; infer their dtype or extent from a generated name.
          scheduled-values (declare-scheduled-temporaries (:values form) equations)
          ;; Scheduled operations are not exempt from SSA validation merely because they are
          ;; records nested inside an equation. Every physical operand/result—including generated
          ;; partial arrays and aliased destinations—must have an AbstractValue contract.
          _ (doseq [equation equations
                    operation (:operations equation)
                    id (set/union (segop/operation-inputs operation)
                                  (segop/operation-outputs operation)
                                  (segop/operation-scalars operation))]
              (when-not (contains? scheduled-values id)
                (throw (ex-info "scheduled SegOp references an undeclared value"
                                {:reason :segop-unknown-scheduled-value
                                 :equation (:id equation)
                                 :operation (:id operation)
                                 :value id}))))
          ;; A typed equation's alias facts define its physical output boundary. This is
          ;; particularly load-bearing for map-void: the semantic result aliases the resident
          ;; destination, and the scheduled store must name that destination explicitly.
          _ (doseq [equation equations
                    :when (seq (:operations equation))]
              (let [algorithm (:algorithm equation)
                    algorithm-equation (first (soac-dialect/equations algorithm))
                    algorithm-id (second algorithm-equation)
                    aliases (get-in (soac-dialect/facts algorithm)
                                    [:equations algorithm-id :aliases])
                    expected (set (map #(get aliases % %) (:results equation)))
                    scheduled-outputs (apply set/union #{}
                                             (map segop/operation-outputs
                                                  (:operations equation)))]
                (when-not (set/subset? expected scheduled-outputs)
                  (throw (ex-info "scheduled SegOp does not realize the typed output boundary"
                                  {:reason :segop-output-boundary
                                   :equation (:id equation)
                                   :expected expected
                                   :scheduled scheduled-outputs})))))
          parallel-equation-count (count (remove #(get-in % [:attributes :host-only]) equations))
          scalar-equation-count (- (count equations) parallel-equation-count)
          kernel-graph-count (count (filter #(get-in % [:attributes :kernel-graph]) equations))
          lowered (assoc form
                         :dialect :segop
                         :values scheduled-values
                         :equations equations
                         :provenance (assoc (:provenance form)
                                            :pass :segop-lower :source-dialect :typed-soac
                                            :device device-id :dtype dtype))]
      {:form (program/validate!
              lowered segop/segop-node?
              (fn [equation algorithm]
                (and (= algorithm (soac-dialect/validate! algorithm))
                     (= (:operands equation) (:inputs (soac-dialect/facts algorithm)))
                     (= (:results equation) (soac-dialect/outputs algorithm)))))
       :stats {:segops-lowered parallel-equation-count
               :kernel-graphs-lowered kernel-graph-count
               :typed-soac-reused parallel-equation-count
               :typed-scalar-equations scalar-equation-count}}))
    (if-not (form/binding-form? form)
      {:form (build-program form [] [] (:target-device opts) (:dtype opts))
       :stats {:segops-lowered 0 :kernel-graphs-lowered 0}}
      (let [original-source form
            original-binding-count (quot (count (second original-source)) 2)
            original-body-count (count (nnext original-source))
            form (->> form
                      util/normalize-let-body
                      util/uniquify-rebindings)
            [let-sym bindings-vec & body-exprs] form
            pairs (partition 2 bindings-vec)
            device-id (:target-device opts)
            target-descriptor (:target-descriptor opts)
            dtype (:dtype opts)
            ;; Retain flat source binder declarations for every later use, including uses
            ;; whose symbol occurrence no longer carries the binder's metadata.
            scalar-types (reduce (fn [known [id _]]
                                   (if-let [declared (declared-scalar-type! id (get known id) dtype)]
                                     (assoc known id declared)
                                     known))
                                 (:scalar-types opts) pairs)
            lowered (atom 0)
            graphs-lowered (atom 0)
          ;; Every par form the middle end could NOT represent, as data. Previously these went to
          ;; stderr as `WARNING: …` and vanished — invisible to stats, to explain-pipeline, and to
          ;; anyone diagnosing why a kernel took the legacy path.
            declined (atom [])
            attempt (fn [sym init current-options]
                      (let [r (lower-attempt sym init device-id target-descriptor dtype
                                             (:array-types current-options) (:scalar-types current-options))]
                        (when-let [d (:declined r)] (swap! declined conj d))
                        (when (:segops r) r)))
            used (atom (set (filter symbol? (tree-seq coll? seq form))))
            ids (atom -1)
            typed-count (atom 0)
            scalar-count (atom 0)
            process
            (fn [{:keys [options] :as state} [site sym expression original-site]]
              (let [packet (selected-operation-packet sym expression options used)]
                (if (:program packet)
                  (let [rows (packet-equations packet original-site ids)
                        packet-pairs (:pairs packet)
                        body? (= :body (first site))
                        packet-body (:body packet)]
                    (swap! lowered inc)
                    (swap! typed-count inc)
                    (swap! scalar-count + (count (filter #(get-in % [:equation :attributes :host-only]) rows)))
                    (-> state
                        (update :pairs into packet-pairs)
                        (cond-> body? (update :body into packet-body))
                        (update :equations into rows)
                        ;; A later singleton consumes the earlier map's retained element storage,
                        ;; not the compilation's default precision. Only the owned map result is
                        ;; newly introduced here; do not replace caller input declarations.
                        (cond-> (par/par-map-pure-form? expression)
                          (assoc-in [:options :array-types sym]
                                    (get-in packet [:program :values sym :dtype])))
                        (update-in [:options :scalar-types] merge
                                   (into {} (keep (fn [[id value]]
                                                    (when (= [] (:shape value))
                                                      [id (:dtype value)])))
                                         (:values (:program packet))))))
                  (let [_ (when-let [refusal (:declined packet)]
                            (swap! declined conj (assoc refusal :stage :typed-admission
                                                      :sym sym :source expression)))
                        current-scalar-types (:scalar-types options)
                        lowered-values (attempt sym expression options)
                        row (when lowered-values
                              (swap! lowered inc)
                              (when (:kernel-graph lowered-values) (swap! graphs-lowered inc))
                              (equation (swap! ids inc) site sym expression lowered-values
                                        dtype (:array-types options) current-scalar-types))]
                    (-> state
                        (cond-> (= :binding (first site)) (update :pairs conj [sym expression])
                                (= :body (first site)) (update :body conj expression)
                                row (update :equations conj row)))))))
            state (reduce process
                          {:options (assoc opts :scalar-types scalar-types)
                           :pairs [] :body [] :equations []}
                          (concat (map-indexed
                                   (fn [idx [sym expression]]
                                     [[:binding sym] sym expression
                                      (if (< idx original-binding-count)
                                        [:binding sym]
                                        [:body (- idx original-binding-count)])]) pairs)
                                  (map-indexed (fn [idx expression]
                                                 [[:body idx] (gensym "body_parallel_") expression
                                                  [:body (+ (max 0 (dec original-body-count)) idx)]])
                                               body-exprs)))
            source (with-meta (list* let-sym (vec (mapcat identity (:pairs state))) (:body state))
                     (meta form))
            result (build-program source (:equations state) @declined device-id dtype)]
        {:form (update result :provenance assoc :original-source original-source)
         :stats (cond-> {:segops-lowered @lowered
                         :kernel-graphs-lowered @graphs-lowered
                         :typed-soac-reused @typed-count
                         :typed-scalar-equations @scalar-count}
                  (seq @declined) (assoc :segops-declined @declined))}))))

(defn- schedule-direct-program
  [result-id source opts]
  (let [;; Imperative map callers commonly pass the destination as their nominal result ID. Give
        ;; the host result its own value identity so the typed frontend can represent the declared
        ;; result-to-storage alias without conflating it with the caller-owned buffer.
        physical-map-output (when (par/par-map-form? source)
                              (:out (par/extract-par-map-info source)))
        host-result (if (= result-id physical-map-output)
                      (gensym "direct_parallel_result_")
                      result-id)
        host-source (list 'let* [host-result source] host-result)
        typed-result
        (typed-route/attempt
         host-source (or (:dtype opts) :double) (:array-types opts)
         {:scalar-types (:scalar-types opts)
          :resident-reductions? (true? (:resident-reductions? opts))})
        {scheduled :form stats :stats}
        (segop-lower-pass (or (:program typed-result) host-source) opts)
        ;; Compound extents may introduce a preceding host-scalar equation. Select the one
        ;; scheduled parallel equation rather than assuming it is first in program order.
        equation (some #(when (seq (:operations %)) %) (:equations scheduled))]
    {:program scheduled
     :equation equation
     :operations (:operations equation)
     :algorithm (:algorithm equation)
     :diagnostics (:diagnostics scheduled)
     :declined (:declined typed-result)
     :stats (merge (:stats typed-result) stats)}))

(defn schedule-single-operation
  "Schedule one closed direct-backend operation through the shared typed boundary.

   This projection is only valid when the operation has no preceding host equations. Callers whose
   source normalization introduces scalar SSA (for example a compound extent) must consume
   `schedule-single-program`; silently discarding that prefix would create an unbound kernel ABI."
  [result-id source opts]
  (let [scheduled (schedule-direct-program result-id source opts)
        host-equations (filterv #(true? (get-in % [:attributes :host-only]))
                                (get-in scheduled [:program :equations]))]
    (when (seq host-equations)
      (throw (ex-info "direct operation requires its complete scheduled program"
                      {:reason :direct-operation-requires-program
                       :source source
                       :host-equations (mapv :id host-equations)})))
    scheduled))

(defn schedule-single-program
  "Run one direct-backend source form through the complete typed program boundary.

   This entry preserves host scalar equations introduced by normalization and reconstructs their
   host bindings around the scheduled parallel equation. Backends use it whenever a direct source
   spelling may expand to a genuine mini-program."
  [result-id source opts]
  (schedule-direct-program result-id source opts))

(defn schedule-source-program
  "Schedule a complete direct-backend binding form through the shared typed boundary.

   This non-cyclic middle-end entry is for backends that receive source without the ordinary
   pipeline having run. Supported equations retain TypedSOAC algorithms and cross-equation scalar
   dependencies; a structured source decline remains an explicit compatibility ParallelProgram."
  [source opts]
  (let [typed-result
        (typed-route/attempt
         source (or (:dtype opts) :double) (:array-types opts)
         {:resident-reductions? (true? (:resident-reductions? opts))
          :scalar-types (:scalar-types opts)
          :values (:values opts)
          :abstract-machine (:abstract-machine opts)})
        {scheduled :form stats :stats}
        (segop-lower-pass (or (:program typed-result) source) opts)]
    {:program scheduled
     :declined (:declined typed-result)
     :stats (merge (:stats typed-result) stats)}))
