(ns raster.gpu.parallel-program
  "Stage-once execution of a checked emitted parallel program call.

   Every numerical graph is bound before the first launch. Structured control reuses prepared
   graphs for equal carry-buffer variants; all buffers remain resident and suffix equations
   consume the exact carry bindings selected by the call IR."
  (:refer-clojure :exclude [run!])
  (:require [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.compiler.ir.structured-loop-call :as loop-call]))

(declare release-prepared! straight-line-handles run-prepared! execution-order execution-info
         profile-prepared!)

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

(def ^:private prepared-request-token (Object.))

(defn- prepared-request-seal [reference request]
  (fn [candidate]
    (when (identical? candidate (.get ^java.lang.ref.WeakReference @reference))
      [prepared-request-token request])))

(def ^:private prepared-request-seal-class
  (class (prepared-request-seal nil nil)))

(defn- seal-prepared-request [prepared caller-options]
  (let [policy (numerics/validate-scalar-math-policy! (:scalar-math caller-options))
        request (when (some? caller-options) {:scalar-math policy})
        reference (volatile! nil)
        sealed (assoc prepared ::request-seal (prepared-request-seal reference request))]
    (vreset! reference (java.lang.ref.WeakReference. sealed))
    sealed))

(defn- prepared-request! [prepared]
  (let [seal (::request-seal prepared)
        result (when (.isInstance ^Class prepared-request-seal-class seal) (seal prepared))]
    (when-not (identical? prepared-request-token (first result))
      (throw (ex-info "prepared program lost its exact request owner"
                      {:reason :parallel-program-request-owner})))
    (second result)))

(defn- attach-prepared-owner [prepared owner]
  (assoc prepared ::cleanup/owner owner ::active-uses (volatile! 0)))

(defn- own-prepared
  "Completed prepared values retain one cleanup plan across every close attempt."
  ([prepared]
   (let [resources
         (cond
           (prepared-kernel-graph? prepared)
           [{:id :graph :release #((:release! prepared) (:handle prepared))}]

           (prepared-sequence? prepared)
           (mapv (fn [{:keys [id program]}]
                   {:id [:instance id] :release #(release-prepared! program)})
                 (rseq (:instances prepared)))

           :else
           (mapv (fn [key]
                   {:id [:graph key] :release #((:release! prepared) (get (:handles prepared) key))})
                 (rseq (:binding-order prepared))))]
     (own-prepared prepared (cleanup/owner resources))))
  ([prepared owner]
   (let [owned (attach-prepared-owner prepared owner)]
     (if (prepared-parallel-program? owned)
       (seal-prepared-request owned nil)
       owned))))

(defn- with-live-prepared
  [prepared operation use!]
  (when-not (or (prepared-parallel-program? prepared) (prepared-sequence? prepared)
                (prepared-kernel-graph? prepared))
    (throw (ex-info "Operation requires a prepared parallel program"
                    {:operation operation :actual (type prepared)})))
  (locking (:closed? prepared)
    (when (prepared-parallel-program? prepared) (prepared-request! prepared))
    (when @(:closed? prepared)
      (throw (ex-info "Prepared parallel program is closed"
                      {:reason :parallel-program-closed :operation operation})))
    (if-let [owner (::cleanup/owner prepared)]
      (cleanup/assert-live! owner)
      (throw (ex-info "Prepared program has lost its cleanup owner"
                      {:reason :missing-cleanup-owner})))
    (when-not (::active-uses prepared)
      (throw (ex-info "Prepared program has lost its use-scope state"
                      {:reason :parallel-program-use-state-missing})))
    (vswap! (::active-uses prepared) inc)
    (try (program-call/without-validation-context use!)
         (finally (vswap! (::active-uses prepared) dec)))))

(defn straight-line-call?
  "Whether an emitted call has one statically ordered graph sequence. Host equations are
   already evaluated by the call; structured loops require the separate bounded runner."
  [call]
  (every? #(or (program-call/evaluated-host-equation? %)
               (program-call/emitted-equation-call? %))
          (:steps call)))

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
  [step execution-id step-index caller-options]
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
     (let [{:keys [buffers scalar-values]}
           (if (nil? caller-options)
             (loop-call/iteration-binding step iteration)
             (loop-call/iteration-binding step iteration caller-options))
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
  [call execution-id caller-options]
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
                   (loop-staging-plan step execution-id step-index caller-options)]
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

(defn- staging-plan-for-request
  "Return the bounded set of distinct graph bindings to prepare without contacting a driver.

   The initial preserved carry may add one prologue variant to the two parity variants. Both the
   returned host data and the eventual driver bindings are therefore constant rather than
   proportional to trip count; replay order is streamed separately by `run-with!`."
  [call execution-id caller-options]
  (let [call (if (nil? caller-options)
               (program-call/validate! call)
               (program-call/validate! call caller-options))]
    (:entries (preparation-plan call execution-id caller-options))))

(defn staging-plan
  "Return bounded graph bindings under independent caller math intent, without driver work."
  ([call execution-id] (staging-plan-for-request call execution-id nil))
  ([call execution-id caller-options] (staging-plan-for-request call execution-id caller-options)))

(defn- validate-executor! [executor]
  (doseq [operation [:bind! :run! :release!]]
    (when-not (ifn? (get executor operation))
      (throw (ex-info "parallel program executor requires callable operations"
                      {:reason :parallel-program-executor
                       :operation operation :executor executor}))))
  (when (and (:adopt-cleanup! executor) (not (fn? (:adopt-cleanup! executor))))
    (throw (ex-info "parallel program cleanup adoption requires a function"
                    {:reason :parallel-program-executor :operation :adopt-cleanup!})))
  executor)

(defn- prepare-graph-with! [id call executor]
  (let [{:keys [graph bindings scalar-values]} call
        key [:parallel-program (random-uuid) id]
        handle (volatile! {})
        owner (cleanup/owner [{:id :graph
                               :release #(when (contains? @handle :value)
                                           ((:release! executor) (:value @handle)))}])]
    (cleanup/build!
     owner
     (fn []
       (vreset! handle {:value ((:bind! executor) key graph bindings scalar-values)})
       (own-prepared (->PreparedKernelGraph graph bindings (:value @handle)
                                            (:run! executor) (:release! executor) (atom false))
                     owner))
     (:adopt-cleanup! executor))))

(defn- prepare-with-request!
  "Bind every distinct graph/carry variant once and return a reusable prepared program.

   A failed binding attempts earlier handle releases in reverse order. Failed cleanup remains
   owned: executor :adopt-cleanup! receives the owner, or the exception retains ::cleanup/unresolved.
   The executor must retain any native acquisition that throws before returning a handle.
   `run-prepared!` streams loop replay from the bounded binding table, so preparation remains
   constant in the loop trip count. Calls with logical result views additionally require the
   executor's `:buffer-view` resolver from a buffer token to its checked live BufferView; exact
   logical extent and prefix aliasing are checked before the first bind."
  [call {:keys [bind! run! release! buffer-view] :as executor} caller-options]
  (let [call (if (nil? caller-options)
               (program-call/validate! call)
               (program-call/validate! call caller-options))]
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
    (validate-executor! executor)
    (let [handles (volatile! {})
          binding-order (volatile! [])
          plan (preparation-plan call (random-uuid) caller-options)
          owner (cleanup/owner
                 (mapv (fn [{:keys [key]}]
                         {:id [:graph key]
                          :release #(when (contains? @handles key)
                                      (release! (get @handles key)))})
                       (rseq (:entries plan))))]
      (seal-prepared-request
       (cleanup/build!
       owner
       (fn []
         (doseq [{:keys [key graph buffers scalar-values]} (:entries plan)]
           (let [handle (bind! key graph buffers scalar-values)]
             (vswap! handles assoc key handle)
             (vswap! binding-order conj key)))
         (attach-prepared-owner
          (->PreparedParallelProgram call plan @handles @binding-order run! release! (atom false))
          owner))
        (:adopt-cleanup! executor))
       caller-options))))

(defn prepare-with!
  "Prepare bounded graph bindings and retain validated math intent with their exact owner.
   Executor callbacks receive neither compiler proof scopes nor policy authority."
  ([call executor]
   (program-call/without-validation-context #(prepare-with-request! call executor nil)))
  ([call executor caller-options]
   (program-call/without-validation-context #(prepare-with-request! call executor caller-options))))

(defn- prepare-sequence-with-request!
  "Prepare ordered emitted programs and direct graphs over one shared resident binding.
   A later binding failure attempts earlier program cleanup before their storage may be freed.
   Unresolved cleanup is adopted through the executor or retained on the thrown exception."
  [instances executor caller-options]
  (numerics/validate-scalar-math-policy! (:scalar-math caller-options))
  (when-not (and (vector? instances) (seq instances)
                 (every? #(and (contains? % :id) (contains? % :call)
                               (contains? #{nil :program :graph} (:kind %))) instances)
                 (= (count instances) (count (distinct (map :id instances)))))
    (throw (ex-info "prepared sequence requires unique ordered instance calls"
                    {:reason :parallel-program-sequence-instances})))
  (validate-executor! executor)
  (let [prepared (volatile! [])
        owner (cleanup/owner
               (mapv (fn [{:keys [id]}]
                       {:id [:instance id]
                        :release #(when-let [program (some (fn [instance]
                                                             (when (= id (:id instance))
                                                               (:program instance))) @prepared)]
                                    (release-prepared! program))})
                     (rseq instances)))]
    (cleanup/build!
     owner
     (fn []
       (doseq [{:keys [id call kind]} instances]
         (vswap! prepared conj {:id id :program
                                (if (= :graph kind)
                                  (prepare-graph-with! id call executor)
                                  (if (nil? caller-options)
                                    (prepare-with! call executor)
                                    (prepare-with! call executor caller-options)))}))
       (own-prepared (->PreparedParallelSequence @prepared (atom false)) owner))
     (:adopt-cleanup! executor))))

(defn prepare-sequence-with!
  "Prepare ordered instances under one independent caller math request."
  ([instances executor]
   (program-call/without-validation-context #(prepare-sequence-with-request! instances executor nil)))
  ([instances executor caller-options]
   (program-call/without-validation-context
    #(prepare-sequence-with-request! instances executor caller-options))))

(defn- visit-handles!
  [prepared operation visit!]
  (ensure-prepared! prepared operation)
  (when-not (ifn? visit!)
    (throw (ex-info "prepared program handle visitor must be callable"
                    {:reason :parallel-program-handle-visitor
                     :operation operation :actual (type visit!)})))
  (let [caller-options (prepared-request! prepared)
        {:keys [call plan handles]} prepared
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
                (if (nil? caller-options)
                  (loop-call/iteration-binding step iteration)
                  (loop-call/iteration-binding step iteration caller-options))
                key (get-in plan [:step-keys step-index [buffers scalar-values]])]
            (when-not key
              (throw (ex-info
                      "structured loop escaped its certified bounded carry rotation"
                      {:reason :parallel-program-unbounded-loop-binding
                       :step-index step-index :iteration iteration})))
            (visit-key! key)))))
    (persistent! @results)))

(defn- straight-line-handles-unlocked
  "Return bound handles in exact source order without expanding structured control. Each entry
   carries its program step, and a prepared sequence also carries its LinkPlan instance id.
   Used only when a caller will record one command graph from the existing bound kernels."
  [prepared]
  (cond
    (prepared-kernel-graph? prepared)
    (do
      (when @(:closed? prepared)
        (throw (ex-info "prepared graph is closed"
                        {:reason :parallel-program-closed :operation :straight-line-handles})))
      [{:step nil :handle (:handle prepared)}])

    (prepared-sequence? prepared)
    (do
      (when @(:closed? prepared)
        (throw (ex-info "prepared sequence is closed"
                        {:reason :parallel-program-closed :operation :straight-line-handles})))
      (vec (mapcat (fn [{:keys [id program]}]
                     (map #(assoc % :instance id) (straight-line-handles program)))
                   (:instances prepared))))

    :else
    (do
      (ensure-prepared! prepared :straight-line-handles)
      (when-not (straight-line-call? (:call prepared))
        (throw (ex-info "structured program has no static command-graph replay order"
                        {:reason :parallel-program-dynamic-recording-order})))
      (vec (keep-indexed
            (fn [step-index step]
              (when (program-call/emitted-equation-call? step)
                {:step step-index
                 :handle (get (:handles prepared)
                              (get-in prepared [:plan :step-keys step-index]))}))
            (get-in prepared [:call :steps]))))))

(defn- run-prepared-unlocked!
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
      (select-keys (:buffers prepared) (map :id (:outputs (:graph prepared)))))

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

(defn- execution-order-unlocked
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
      (let [caller-options (prepared-request! prepared)
            observer (fn [step-index _]
                       (let [key (get-in prepared [:plan :step-keys step-index])]
                         (graph-order (get (:handles prepared) key))))]
        (if (nil? caller-options)
          (program-call/execution-order (:call prepared) observer)
          (program-call/execution-order (:call prepared) observer caller-options))))))

(defn- execution-info-unlocked
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

(defn- profile-prepared-unlocked!
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

(defn straight-line-handles
  "Return borrowed bound handles in source order while the prepared owner is live.
   The returned handles do not pin its lifetime after this call: keep the prepared owner alive
   through their use, or use the run/profile callback-under-lock APIs."
  [prepared]
  (with-live-prepared prepared :straight-line-handles #(straight-line-handles-unlocked prepared)))

(defn run-prepared!
  "Replay the prepared program; release cannot race replay or occur reentrantly within it."
  [prepared]
  (with-live-prepared prepared :run-prepared! #(run-prepared-unlocked! prepared)))

(defn execution-order
  "Observe selected graph order under the prepared lifetime lock."
  [prepared graph-order]
  (with-live-prepared prepared :execution-order #(execution-order-unlocked prepared graph-order)))

(defn execution-info
  "Describe distinct selected bindings under the prepared lifetime lock. Any handles in the
   returned report remain borrowed; this observation does not extend the owner's lifetime."
  [prepared graph-info]
  (with-live-prepared prepared :execution-info #(execution-info-unlocked prepared graph-info)))

(defn profile-prepared!
  "Replay and aggregate device-event profiles under the prepared lifetime lock."
  [prepared profile-handle!]
  (with-live-prepared prepared :profile-prepared! #(profile-prepared-unlocked! prepared profile-handle!)))

(defn release-prepared!
  "Release graphs in reverse binding order. Failed ownership stays visible on later calls;
   successful releases are never repeated and uncertain native outcomes are not retried."
  [prepared]
  (when-not (or (prepared-parallel-program? prepared) (prepared-sequence? prepared)
                (prepared-kernel-graph? prepared))
    (throw (ex-info "release-prepared! requires a prepared parallel program"
                    {:actual (type prepared)})))
  (locking (:closed? prepared)
    (when (prepared-parallel-program? prepared) (prepared-request! prepared))
    (when-not (::cleanup/owner prepared)
      (throw (ex-info "Prepared program has lost its cleanup owner"
                      {:reason :missing-cleanup-owner})))
    (when-not (::active-uses prepared)
      (throw (ex-info "Prepared program has lost its use-scope state"
                      {:reason :parallel-program-use-state-missing})))
    (when (and (::active-uses prepared) (pos? @(::active-uses prepared)))
      (throw (ex-info "Cannot release a prepared program from its active use callback"
                      {:reason :parallel-program-in-use})))
    (when-not @(:closed? prepared) (reset! (:closed? prepared) true))
    (program-call/without-validation-context #(cleanup/release! (::cleanup/owner prepared))))
  nil)

(defn- run-with-request!
  "Bind the complete call, then execute it through an injected graph executor.

  `executor` contains `:bind!`, `:run!`, and `:release!`. A staging failure releases all prior
   handles without launching; an execution failure releases the entire staged program."
  [call executor caller-options]
  (let [prepared (if (nil? caller-options)
                   (prepare-with! call executor)
                   (prepare-with! call executor caller-options))]
    (try
      (run-prepared! prepared)
      (finally
        (release-prepared! prepared)))))

(defn run-with!
  "Prepare, replay, and release a program under independent caller math intent."
  ([call executor] (run-with-request! call executor nil))
  ([call executor caller-options] (run-with-request! call executor caller-options)))

(defn- run-session-with-request!
  [session call caller-options]
  (let [bind-graph! (requiring-resolve 'raster.gpu.core/bind-kernel-graph!)
        run-graph! (requiring-resolve 'raster.gpu.core/run-kernel-graph!)
        release-graph! (requiring-resolve 'raster.gpu.core/release-kernel-graph!)]
    (run-with-request!
     call
     {:bind! (fn [key graph buffers scalars]
               (bind-graph! session key graph buffers scalars))
      :run! (fn [handle] (run-graph! session handle))
      :release! (fn [handle] (release-graph! session handle))}
     caller-options)))

(defn run!
  "Run an emitted parallel program in one GPU session under independent caller math intent."
  ([session call] (run-session-with-request! session call nil))
  ([session call caller-options] (run-session-with-request! session call caller-options)))
