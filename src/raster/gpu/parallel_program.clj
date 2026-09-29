(ns raster.gpu.parallel-program
  "Stage-once execution of a checked emitted parallel program call.

   Every numerical graph is bound before the first launch. Structured control reuses prepared
   graphs for equal carry-buffer variants; all buffers remain resident and suffix equations
   consume the exact carry bindings selected by the call IR."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.structured-loop-call :as loop-call]))

(declare release-prepared!)

(defrecord PreparedParallelProgram [call plan handles binding-order run! release! closed?]
  java.io.Closeable
  (close [this] (release-prepared! this)))

(defrecord PreparedParallelSequence [instances closed?]
  java.io.Closeable
  (close [this] (release-prepared! this)))

(defrecord PreparedKernelGraph [graph buffers handle run! release! closed?]
  java.io.Closeable
  (close [this] (release-prepared! this)))

(defn prepared-kernel-graph? [value]
  (instance? PreparedKernelGraph value))

(defn prepared-sequence? [value]
  (instance? PreparedParallelSequence value))

(defn prepared-parallel-program?
  [value]
  (and value (= "raster.gpu.parallel_program.PreparedParallelProgram"
                (.getName (class value)))))

(defn- ensure-prepared!
  [prepared operation]
  (when-not (prepared-parallel-program? prepared)
    (throw (ex-info (str operation " requires a PreparedParallelProgram")
                    {:operation operation :actual (type prepared)})))
  (when @(:closed? prepared)
    (throw (ex-info "prepared parallel program is closed"
                    {:reason :parallel-program-closed :operation operation})))
  prepared)

(defn- loop-staging-plan
  [step execution-id step-index]
  (when (and (get-in step [:scalars :iteration]) (> (:trip-count step) 1))
    (throw (ex-info
            "stage-once execution cannot freeze a changing loop induction scalar"
            {:reason :parallel-program-dynamic-loop-binding
             :step-index step-index :trip-count (:trip-count step)
             :fallback :bounded-iteration-execution})))
  ;; StructuredLoopCall has either an in-place carry or one initial buffer plus a two-buffer
  ;; parity rotation. Consequently iteration zero and the two parities after it are the complete
  ;; binding state space. Inspecting more iterations would only allocate O(trip-count) host data
  ;; before the first launch—the exact failure bounded replay exists to avoid.
  (reduce
   (fn [{:keys [bindings entries] :as state} iteration]
     (let [{:keys [buffers scalar-values]} (loop-call/iteration-binding step iteration)
           binding [buffers scalar-values]]
       (if (contains? bindings binding)
         state
         (let [variant (count bindings)
               key [:parallel-program execution-id step-index :variant variant]]
           (when (>= variant 3)
             (throw (ex-info
                     "structured loop produced more than the bounded carry rotation variants"
                     {:reason :parallel-program-unbounded-loop-binding
                      :step-index step-index :iteration iteration
                      :variants (inc variant)})))
           {:bindings (assoc bindings binding key)
            :entries (conj entries
                           {:key key :graph (:graph step)
                            :buffers buffers :scalar-values scalar-values})}))))
   {:bindings {} :entries []}
   (range (min 3 (:trip-count step)))))

(defn- preparation-plan
  [call execution-id]
  (let [program-scalars (:scalar-values call)
        plan
        (reduce
         (fn [{:keys [entries step-keys] :as plan} [step-index step]]
           (cond
             (program-call/evaluated-host-equation? step)
             plan

             (program-call/emitted-equation-call? step)
             (let [key [:parallel-program execution-id step-index]]
               {:entries (conj entries
                               {:key key :graph (:graph step)
                                :buffers (:buffers step)
                                :scalar-values (:scalar-values step)})
                :step-keys (assoc step-keys step-index key)})

             (loop-call/structured-loop-call? step)
             (let [{:keys [bindings] loop-entries :entries}
                   (loop-staging-plan step execution-id step-index)]
               {:entries (into entries loop-entries)
                :step-keys (assoc step-keys step-index bindings)})))
         {:entries [] :step-keys {}}
         (map-indexed vector (:steps call)))]
    ;; Kernel-local maps deliberately contain only ABI scalars. Program-wide shape values still
    ;; participate in graph buffer extents, including intermediate tensors consumed by a later
    ;; equation, so retain them while staging. A local target-width cast remains authoritative.
    (update plan :entries
            (fn [entries]
              (mapv #(update % :scalar-values
                             (fn [local] (merge program-scalars local)))
                    entries)))))

(defn staging-plan
  "Return the bounded set of distinct graph bindings to prepare without contacting a driver.

   The initial preserved carry may add one prologue variant to the two parity variants. Both the
   returned host data and the eventual driver bindings are therefore constant rather than
   proportional to trip count; replay order is streamed separately by `run-with!`."
  [call execution-id]
  (let [call (program-call/validate! call)]
    (:entries (preparation-plan call execution-id))))

(defn prepare-with!
  "Bind every distinct graph/carry variant once and return a reusable prepared program.

   Preparation is transactional: a failed binding releases all earlier handles in reverse order.
   `run-prepared!` streams loop replay from the bounded binding table, so preparation remains
   constant in the loop trip count. Calls with logical result views additionally require the
   executor's `:buffer-view` resolver from a buffer token to its checked live BufferView; exact
   logical extent and prefix aliasing are checked before the first bind."
  [call {:keys [bind! run! release! buffer-view] :as executor}]
  (let [call (program-call/validate! call)]
    (doseq [step (:steps call)
            [result physical] (:result-views step)]
      (when-not (ifn? buffer-view)
        (throw (ex-info "result views require checked runtime buffer-view resolution"
                        {:reason :parallel-program-result-view-resolver :result result})))
      (let [view (bview/validate-view! (buffer-view (get (:buffers call) result)))
            base (bview/validate-view! (buffer-view (get (:buffers call) physical)))
            expected (get-in call [:program :values result])
            expected-elements (reduce *' 1 (map #(graph-call/resolve-integer (:scalar-values call) %)
                                                (:shape expected)))]
        (when-not (and (= (:dtype expected) (:dtype view))
                       (= expected-elements (reduce *' 1 (:shape view))))
          (throw (ex-info "runtime result view differs from its logical tensor contract"
                          {:reason :parallel-program-result-view-shape :result result
                           :expected expected :actual view})))
        (when-not (bview/prefix-view? base view)
          (throw (ex-info "runtime result view is not a prefix of its physical destination"
                          {:reason :parallel-program-result-view :result result :physical physical})))))
    (doseq [[operation function] [[:bind! bind!] [:run! run!] [:release! release!]]]
      (when-not (ifn? function)
        (throw (ex-info "parallel program executor requires callable operations"
                        {:reason :parallel-program-executor
                         :operation operation :executor executor}))))
    (let [handles (volatile! {})
          binding-order (volatile! [])
          plan (preparation-plan call (random-uuid))]
      (try
        (doseq [{:keys [key graph buffers scalar-values]} (:entries plan)]
          (let [handle (bind! key graph buffers scalar-values)]
            (vswap! handles assoc key handle)
            (vswap! binding-order conj key)))
        (->PreparedParallelProgram call plan @handles @binding-order run! release! (atom false))
        (catch Throwable error
          (doseq [key (rseq @binding-order)]
            (try (release! (get @handles key)) (catch Throwable _)))
          (throw error))))))

(defn prepare-sequence-with!
  "Prepare ordered emitted programs and direct graphs over one shared resident binding.
   A later binding failure releases all earlier programs before their storage is freed."
  [instances executor]
  (when-not (and (vector? instances) (seq instances)
                 (every? #(and (contains? % :id) (contains? % :call)
                               (contains? #{nil :program :graph} (:kind %))) instances)
                 (= (count instances) (count (distinct (map :id instances)))))
    (throw (ex-info "prepared sequence requires unique ordered instance calls"
                    {:reason :parallel-program-sequence-instances})))
  (let [prepared (volatile! [])]
    (try
      (doseq [{:keys [id call kind]} instances]
        (vswap! prepared conj {:id id :program
                               (if (= :graph kind)
                                 (let [{:keys [graph bindings scalar-values]} call
                                       key [:parallel-program (random-uuid) id]
                                       handle ((:bind! executor) key graph bindings scalar-values)]
                                   (->PreparedKernelGraph graph bindings handle
                                                          (:run! executor) (:release! executor)
                                                          (atom false)))
                                 (prepare-with! call executor))}))
      (->PreparedParallelSequence @prepared (atom false))
      (catch Throwable error
        (doseq [{:keys [program]} (rseq @prepared)]
          (try (release-prepared! program) (catch Throwable _)))
        (throw error)))))

(defn- visit-handles!
  [prepared operation visit!]
  (ensure-prepared! prepared operation)
  (when-not (ifn? visit!)
    (throw (ex-info "prepared program handle visitor must be callable"
                    {:reason :parallel-program-handle-visitor
                     :operation operation :actual (type visit!)})))
  (let [{:keys [call plan handles]} prepared
        results (volatile! (transient []))
        visit-key! (fn [key]
                     (vswap! results conj! (visit! (get handles key))))]
    (doseq [[step-index step] (map-indexed vector (:steps call))]
      (cond
        (program-call/evaluated-host-equation? step)
        nil

        (program-call/emitted-equation-call? step)
        (visit-key! (get-in plan [:step-keys step-index]))

        (loop-call/structured-loop-call? step)
        (doseq [iteration (range (:trip-count step))]
          (let [{:keys [buffers scalar-values]}
                (loop-call/iteration-binding step iteration)
                key (get-in plan [:step-keys step-index [buffers scalar-values]])]
            (when-not key
              (throw (ex-info
                      "structured loop escaped its certified bounded carry rotation"
                      {:reason :parallel-program-unbounded-loop-binding
                       :step-index step-index :iteration iteration})))
            (visit-key! key)))))
    (persistent! @results)))

(defn run-prepared!
  "Replay a prepared program and return its resident output bindings.
   A sequence keys each component's outputs by its stable LinkPlan instance identity."
  [prepared]
  (cond
    (prepared-kernel-graph? prepared)
    (do
      (when @(:closed? prepared)
        (throw (ex-info "prepared graph is closed"
                        {:reason :parallel-program-closed :operation :run-prepared!})))
      ((:run! prepared) (:handle prepared))
      (:buffers prepared))

    (prepared-sequence? prepared)
    (do
      (when @(:closed? prepared)
        (throw (ex-info "prepared sequence is closed"
                        {:reason :parallel-program-closed :operation :run-prepared!})))
      (reduce (fn [outputs {:keys [id program]}]
                (assoc outputs id (run-prepared! program)))
              {} (:instances prepared)))

    :else
    (do
      (visit-handles! prepared :run-prepared! (:run! prepared))
      (:outputs (:call prepared)))))

(defn execution-order
  "Compose selected graph orders for a straight-line prepared program, retaining source step
   indices (including skipped host equations). Structured control deliberately declines rather
   than expanding trip counts or confusing one-time preparation with repeated execution."
  [prepared graph-order]
  (cond
    (prepared-kernel-graph? prepared)
    (do
      (when @(:closed? prepared)
        (throw (ex-info "prepared graph is closed"
                        {:reason :parallel-program-closed :operation :execution-order})))
      (graph-order (:handle prepared)))

    (prepared-sequence? prepared)
    (do
      (when @(:closed? prepared)
        (throw (ex-info "prepared sequence is closed"
                        {:reason :parallel-program-closed :operation :execution-order})))
      (reduce (fn [order {:keys [id program]}]
                (let [selected (execution-order program graph-order)
                      annotate #(mapv (fn [entry]
                                        (assoc-in entry [:source :instance] id)) %)]
                  (-> order
                      (update :record-time-prologue into
                              (annotate (:record-time-prologue selected)))
                      (update :per-replay into (annotate (:per-replay selected))))))
              {:record-time-prologue [] :per-replay [] :completion :unproven}
              (:instances prepared)))

    :else
    (do
      (ensure-prepared! prepared :execution-order)
      (program-call/execution-order
       (:call prepared)
       (fn [step-index _]
         (let [key (get-in prepared [:plan :step-keys step-index])]
           (graph-order (get (:handles prepared) key))))))))

(defn execution-info
  "Describe each distinct prepared graph binding once, in binding order. Structured-loop carry
   variants stay bounded; this is neither expanded replay order nor measured execution evidence."
  [prepared graph-info]
  (cond
    (prepared-kernel-graph? prepared)
    (do
      (when @(:closed? prepared)
        (throw (ex-info "prepared graph is closed"
                        {:reason :parallel-program-closed :operation :execution-info})))
      [{:phase (:handle prepared) :executable (graph-info (:handle prepared))}])

    (prepared-sequence? prepared)
    (do
      (when @(:closed? prepared)
        (throw (ex-info "prepared sequence is closed"
                        {:reason :parallel-program-closed :operation :execution-info})))
      (vec (mapcat (fn [{:keys [id program]}]
                     (map #(assoc % :instance id)
                          (execution-info program graph-info)))
                   (:instances prepared))))

    :else
    (do
      (ensure-prepared! prepared :execution-info)
      (when-not (ifn? graph-info)
        (throw (ex-info "prepared program execution reporting requires a graph observer"
                        {:reason :parallel-program-execution-info-observer})))
      (mapv (fn [key]
              {:phase key :executable (graph-info (get (:handles prepared) key))})
            (:binding-order prepared)))))

(defn profile-prepared!
  "Replay a prepared program in exact program order through `profile-handle!` and aggregate its
   device-event intervals. Every handle callback must consume/reset one completed profiling replay;
   no host duration is substituted for device time. Repeated loop handles remain repeated samples
   in the aggregate because they are distinct launches in the program schedule."
  [prepared profile-handle!]
  (let [started (System/nanoTime)
        profiles (cond
                   (prepared-kernel-graph? prepared)
                   [(do
                      (when @(:closed? prepared)
                        (throw (ex-info "prepared graph is closed"
                                        {:reason :parallel-program-closed
                                         :operation :profile-prepared!})))
                      (profile-handle! (:handle prepared)))]

                   (prepared-sequence? prepared)
                   (do
                     (when @(:closed? prepared)
                       (throw (ex-info "prepared sequence is closed"
                                       {:reason :parallel-program-closed
                                        :operation :profile-prepared!})))
                     (mapv (fn [{:keys [id program]}]
                             (update (profile-prepared! program profile-handle!) :profile
                                     #(mapv (fn [event] (assoc event :instance id)) %)))
                           (:instances prepared)))
                   :else
                   (visit-handles! prepared :profile-prepared! profile-handle!))
        finished (System/nanoTime)
        finite-nonnegative? #(and (number? %)
                                  (Double/isFinite (double %))
                                  (not (neg? (double %))))]
    (doseq [[index profile] (map-indexed vector profiles)
            field [:kernel-total-ms :device-wall-ms]]
      (when-not (finite-nonnegative? (get profile field))
        (throw (ex-info "prepared program profiling requires finite device-event spans"
                        {:reason :parallel-program-profile-span
                         :graph-index index :field field :value (get profile field)}))))
    {:profile (vec (mapcat :profile profiles))
     :kernel-total-ms (reduce + 0.0 (map :kernel-total-ms profiles))
     :device-wall-ms (reduce + 0.0 (map :device-wall-ms profiles))
     :host-wall-ms (/ (- finished started) 1.0e6)
     :program-graph-count (reduce + 0 (map #(or (:program-graph-count %) 1) profiles))
     :timing-scope (if (= 1 (reduce + 0 (map #(or (:program-graph-count %) 1)
                                           profiles)))
                     :single-graph-span :sum-of-graph-events)}))

(defn release-prepared!
  "Release every prepared graph in reverse binding order. Idempotent."
  [prepared]
  (when-not (or (prepared-parallel-program? prepared) (prepared-sequence? prepared)
                (prepared-kernel-graph? prepared))
    (throw (ex-info "release-prepared! requires a prepared parallel program"
                    {:actual (type prepared)})))
  (when (compare-and-set! (:closed? prepared) false true)
    (let [failure (volatile! nil)]
      (cond
        (prepared-kernel-graph? prepared)
        (try ((:release! prepared) (:handle prepared))
             (catch Throwable error (vreset! failure error)))

        (prepared-sequence? prepared)
        (doseq [{:keys [program]} (rseq (:instances prepared))]
          (try (release-prepared! program)
               (catch Throwable error
                 (when-not @failure (vreset! failure error)))))

        :else
        (doseq [key (rseq (:binding-order prepared))]
          (try
            ((:release! prepared) (get (:handles prepared) key))
            (catch Throwable error
              (when-not @failure (vreset! failure error))))))
      (when-let [error @failure] (throw error))))
  nil)

(defn run-with!
  "Bind the complete call, then execute it through an injected graph executor.

  `executor` contains `:bind!`, `:run!`, and `:release!`. A staging failure releases all prior
   handles without launching; an execution failure releases the entire staged program."
  [call executor]
  (let [prepared (prepare-with! call executor)]
    (try
      (run-prepared! prepared)
      (finally
        (release-prepared! prepared)))))

(defn run!
  "Run a prepared emitted parallel program in one GPU session."
  [session call]
  (let [bind-graph! (requiring-resolve 'raster.gpu.core/bind-kernel-graph!)
        run-graph! (requiring-resolve 'raster.gpu.core/run-kernel-graph!)
        release-graph! (requiring-resolve 'raster.gpu.core/release-kernel-graph!)]
    (run-with!
     call
     {:bind! (fn [key graph buffers scalars]
               (bind-graph! session key graph buffers scalars))
      :run! (fn [handle] (run-graph! session handle))
      :release! (fn [handle] (release-graph! session handle))})))
