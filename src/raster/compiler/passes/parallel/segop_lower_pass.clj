(ns raster.compiler.passes.parallel.segop-lower-pass
  "Pipeline pass: lower par forms to SegOp records.

   Walks let* bindings and converts raster.par/* forms to SegOp IR via the SOAC intermediate.
   The result is a first-class ParallelProgram whose typed equations own the SegOps.  Binding
   metadata is not an IR transport.

   This decouples hardware-aware execution planning from backend codegen:
   - Lowering decides phase decomposition, launch params, accumulator count
   - Backend translates SegOp to target code (SIMD, OpenCL, scalar)"
  (:require [raster.compiler.ir.par :as par]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.ir.soac-dialect :as soac-dialect]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.soac-dialect-adapter :as soac-adapter]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]
            [raster.compiler.passes.parallel.segred-body :as segred-body]
            [raster.compiler.ir.form :as form]
            [clojure.set :as set]))

(def ^:private id-counter (atom 0))

(def ^:private fatal-reasons
  "A violated invariant is not a missing lowering rule. Recording one as a conversion decline would
   let the pipeline continue on the legacy path with the bug intact — the loud-to-silent trade this
   whole change exists to prevent."
  #{:raster/fatal :raster/bug})

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
  [sym form device-id dtype array-types]
  (when (seq? form)
    (let [par? (par/par-form? form)
          decline (fn [stage e] (when (contains? fatal-reasons (:reason (ex-data e))) (throw e))
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
              typed? (and (soac/soac-reduce? legacy-node)
                          (empty? (:segment-axes legacy-node))
                          (reduction/scalar? (:reduction legacy-node))
                          ;; TypedSOAC deliberately names extents with stable values (or static
                          ;; literals). Older source-shaped bounds such as `(alength x)` remain on
                          ;; the compatibility route until bound expressions have their own SSA
                          ;; equation; do not smuggle executable host forms into the typed dialect.
                          (soac-dialect/extent? (:bound legacy-node)))
              algorithm (when typed?
                          (soac-adapter/legacy-nodes->program
                           [legacy-node] {:outputs (:outputs legacy-node)
                                          :dtype (or dtype :double)
                                          :array-types array-types}))
              segops (attempt #(if algorithm
                                 (soac-lower/lower-typed-reduce
                                  algorithm (or device-id :cpu:0) :dtype (or dtype :double))
                                 (soac-lower/lower-soac legacy-node (or device-id :cpu:0)
                                                        :dtype (or dtype :double))))]
          (cond
            (:err segops) (decline :segop (:err segops))
            (seq (:ok segops)) (cond-> {:soac legacy-node :algorithm algorithm :segops (:ok segops)}
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
    (and (soac/screma? node) (seq (:reduces node)) (empty? (:scans node))) []
    ;; An imperative map/scan binding aliases a caller-provided output buffer. Its iteration domain
    ;; is not proof of the buffer's physical/logical extent (padded rows and strided views are common).
    :else (unknown-vector-shape)))

(defn- value-contract
  [id node dtype array-types result?]
  (let [array-ids (set/union (or (soac/soac-inputs node) #{})
                             (or (soac/soac-outputs node) #{}))
        shape (cond
                result? (result-shape node)
                (contains? array-ids id) (unknown-vector-shape)
                :else [])
        value-dtype (or (get array-types id)
                        (when (symbol? id) (get array-types (symbol (name id))))
                        (:elem-type node)
                        dtype
                        :double)]
    (av/tensor {:dtype value-dtype
                :shape shape
                :representation {:kind :plain}
                :effects #{}})))

(defn- equation
  [equation-id site sym source {:keys [soac algorithm segops kernel-graph]} dtype array-types]
  (let [;; Body-position equations currently have no ParallelProgram result ID; retain the typed
        ;; algorithm only where its result has an authoritative envelope identity.
        algorithm (when (= :binding (first site)) algorithm)
        operands (-> (soac/node-all-free-syms soac) (disj sym) (->> (sort-by str) vec))
        result-ids (if (= :binding (first site)) [sym] [])
        effects (cond-> #{:memory/read}
                  (seq (soac/soac-outputs soac)) (conj :memory/write))
        operand-values (into {}
                             (map (fn [id]
                                    [id (value-contract id soac dtype array-types false)]))
                             operands)
        result-values (into {}
                            (map (fn [id]
                                   [id (value-contract id soac dtype array-types true)]))
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
  [id definition inferred-use]
  (let [facets [:kind :dtype :representation]
        defined (select-keys definition facets)
        inferred (select-keys inferred-use facets)]
    (when-not (= defined inferred)
      (throw (ex-info "parallel equation use is incompatible with its defining value"
                      {:reason :parallel-program-use-type-conflict
                       :id id :definition defined :use inferred}))))
  definition)

(defn- build-program
  [source lowered-equations declined device-id dtype]
  (let [{:keys [equations values]}
        (reduce
         (fn [{:keys [environment] :as state}
              {:keys [equation operand-values result-values]}]
           (let [source-results (:results equation)
                 operands (mapv #(get environment % %) (:operands equation))
                 ;; Source binders and pre-existing buffers may share a spelling in imperative IR.
                 ;; Give equation results their own SSA-like IDs so a scalar binding can never
                 ;; collide with an array operand of the same name.
                 results (mapv (fn [source-id] [:binding source-id]) source-results)
                 values-with-operands
                 (reduce (fn [values [source-id value-id]]
                           ;; A mapped operand already has the defining equation's authoritative
                           ;; contract. Only infer contracts for external program inputs here.
                           (if-let [definition (get values value-id)]
                             (do (ensure-use-compatible! value-id definition
                                                         (get operand-values source-id))
                                 values)
                             (merge-values values {value-id (get operand-values source-id)})))
                         (:values state)
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

(defn segop-lower-pass
  "Pipeline pass: convert par forms in let* bindings to SegOp records.

   Walks the form's let* bindings. For each par/map!, par/reduce, par/scan! binding, converts to a
   typed ProgramEquation containing ordered SegOps. The returned `:form` is a ParallelProgram;
   its `:source` is the undecorated host expression consumed around those equations.

   Scan decomposition additionally records one verified KernelGraph with its intermediate buffers
   and dependencies. Returns both `:segops-lowered` and `:kernel-graphs-lowered` stats.

   Options from pipeline opts:
     :target-device — device for launch param computation
     :dtype — element type (:double or :float)"
  [form opts]
  (if (and (program/parallel-program? form) (= :typed-soac (:dialect form)))
    (let [device-id (or (:target-device opts) :cpu:0)
          dtype (or (:dtype opts) :double)
          equations
          (mapv
           (fn [equation]
             (let [algorithm (:algorithm equation)
                   kind (soac-dialect/operation-kind
                         (first (soac-dialect/equations algorithm)))
                   lowered (case kind
                             scalar {:operations []}
                             map {:operations (soac-lower/lower-typed-map
                                               algorithm device-id :dtype dtype)}
                             scatter {:operations (soac-lower/lower-typed-scatter
                                                   algorithm device-id :dtype dtype)}
                             stencil {:operations (soac-lower/lower-typed-stencil
                                                   algorithm device-id :dtype dtype)}
                             reduce {:operations (soac-lower/lower-typed-reduce
                                                  algorithm device-id :dtype dtype)}
                             segmented-reduce
                             {:operations (soac-lower/lower-typed-segmented-reduce
                                           algorithm device-id :dtype dtype)}
                             product-reduce
                             {:operations (soac-lower/lower-typed-product-reduce
                                           algorithm device-id :dtype dtype)}
                             segmented-fold-map
                             {:operations (soac-lower/lower-typed-segmented-fold-map
                                           algorithm device-id :dtype dtype)}
                             scan (soac-lower/lower-typed-scan
                                   algorithm device-id :dtype dtype
                                   :array-types (:array-types opts)))
                   operations (:operations lowered)]
               (-> equation
                   (assoc :operations operations)
                   (update :provenance assoc :target-dialect :segop)
                   (update :attributes assoc :device device-id)
                   (cond-> (:kernel-graph lowered)
                     (update :attributes assoc :kernel-graph (:kernel-graph lowered)))
                   (cond-> (= 'scalar kind)
                     (update :attributes assoc :host-only true)))))
           (:equations form))
          ;; A multi-phase schedule introduces physical SSA values that do not exist in the
          ;; functional algorithm. They still require explicit contracts; an emitter must never
          ;; infer their dtype or extent from a generated name.
          scheduled-values
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
                                          :shape (soac-dialect/extent-shape
                                                  (:elements temporary))
                                          :representation {:kind :plain}
                                          :memory-space (:memory-space temporary)})))))
                           values
                           (get-in equation [:attributes :kernel-graph :temporaries]))]
               (reduce
                (fn [values operation]
                  (if (and (instance? raster.compiler.ir.segop.SegRed operation)
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
           (:values form) equations)
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
               :typed-scalar-equations scalar-equation-count}})
    (if-not (form/binding-form? form)
      {:form (build-program form [] [] (:target-device opts) (:dtype opts))
       :stats {:segops-lowered 0 :kernel-graphs-lowered 0}}
      (let [[let-sym bindings-vec & body-exprs] form
            pairs (partition 2 bindings-vec)
            device-id (:target-device opts)
            dtype (:dtype opts)
            array-types (:array-types opts)
            lowered (atom 0)
            graphs-lowered (atom 0)
          ;; Every par form the middle end could NOT represent, as data. Previously these went to
          ;; stderr as `WARNING: …` and vanished — invisible to stats, to explain-pipeline, and to
          ;; anyone diagnosing why a kernel took the legacy path.
            declined (atom [])
            attempt (fn [sym init]
                      (let [r (lower-attempt sym init device-id dtype array-types)]
                        (when-let [d (:declined r)] (swap! declined conj d))
                        (when (:segops r) r)))
            binding-equations
            (keep-indexed
             (fn [idx [sym init]]
               (when-let [lowered-values (attempt sym init)]
                 (swap! lowered inc)
                 (when (:kernel-graph lowered-values) (swap! graphs-lowered inc))
                 (equation idx [:binding sym] sym init lowered-values dtype array-types)))
             pairs)
          ;; Also check body expressions for par forms
            body-equations
            (keep-indexed
             (fn [idx expr]
               (let [tmp-sym (symbol (str "body_parallel_" idx))]
                 (when-let [lowered-values (attempt tmp-sym expr)]
                   (swap! lowered inc)
                   (when (:kernel-graph lowered-values) (swap! graphs-lowered inc))
                   (equation (+ (count pairs) idx) [:body idx] tmp-sym expr
                             lowered-values dtype array-types))))
             body-exprs)
            equations (vec (concat binding-equations body-equations))]
        {:form (build-program (list* let-sym bindings-vec body-exprs)
                              equations @declined device-id dtype)
         :stats (cond-> {:segops-lowered @lowered
                         :kernel-graphs-lowered @graphs-lowered}
                  (seq @declined) (assoc :segops-declined @declined))}))))
