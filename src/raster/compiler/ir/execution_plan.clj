(ns raster.compiler.ir.execution-plan
  "Backend-neutral queue and event scheduling for executable operations.

   LogicalQueue and LogicalEvent are compiler values, never driver handles. An ExecutionPlan
   lowers data dependencies to explicit wait/signal edges while leaving the runtime responsible
   for mapping queue classes and event identities to OpenCL, Level Zero, MPI, or another executor."
  (:require [clojure.set :as set]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-graph-call :as kgcall]))

(def queue-classes #{:compute :transfer :collective :host})
(def queue-orderings #{:in-order :out-of-order})

(defrecord LogicalQueue [id class ordinal ordering])
(defrecord LogicalEvent [id])
(defrecord ScheduledOperation [id operation queue waits completion])
(defrecord ExecutionPlan [queues inputs operations outputs])

(defn logical-queue? [x]
  (and x (= "raster.compiler.ir.execution_plan.LogicalQueue" (.getName (class x)))))

(defn logical-event? [x]
  (and x (= "raster.compiler.ir.execution_plan.LogicalEvent" (.getName (class x)))))

(defn scheduled-operation? [x]
  (and x (= "raster.compiler.ir.execution_plan.ScheduledOperation" (.getName (class x)))))

(defn execution-plan? [x]
  (and x (= "raster.compiler.ir.execution_plan.ExecutionPlan" (.getName (class x)))))

(defn compute-queue
  "The default target-neutral in-order compute queue."
  []
  (->LogicalQueue :compute-0 :compute 0 :in-order))

(defn validate!
  "Validate and return an ExecutionPlan.

   Every wait must name an input event or an event completed by an earlier operation. Output
   events must be available after the full plan. This makes the event DAG explicit without
   admitting native handles or assuming a particular backend queue implementation."
  [plan]
  (when-not (execution-plan? plan)
    (throw (ex-info "execution plan must be an ExecutionPlan value"
                    {:plan plan :actual (type plan)})))
  (let [{:keys [queues inputs operations outputs]} plan]
    (when-not (vector? queues)
      (throw (ex-info "execution plan queues must be an ordered vector" {:queues queues})))
    (when-not (vector? inputs)
      (throw (ex-info "execution plan input events must be an ordered vector" {:inputs inputs})))
    (when-not (vector? operations)
      (throw (ex-info "execution plan operations must be an ordered vector"
                      {:operations operations})))
    (when-not (vector? outputs)
      (throw (ex-info "execution plan output events must be an ordered vector" {:outputs outputs})))
    (doseq [queue queues]
      (when-not (logical-queue? queue)
        (throw (ex-info "execution plan contains a non-logical queue" {:queue queue})))
      (when (nil? (:id queue))
        (throw (ex-info "logical queue requires an identity" {:queue queue})))
      (when-not (contains? queue-classes (:class queue))
        (throw (ex-info "logical queue has an unsupported class" {:queue queue})))
      (when-not (and (integer? (:ordinal queue)) (not (neg? (:ordinal queue))))
        (throw (ex-info "logical queue ordinal must be a non-negative integer" {:queue queue})))
      (when-not (contains? queue-orderings (:ordering queue))
        (throw (ex-info "logical queue has an unsupported ordering" {:queue queue}))))
    (let [queue-ids (mapv :id queues)
          input-ids (mapv :id inputs)]
      (when-not (= (count queue-ids) (count (set queue-ids)))
        (throw (ex-info "logical queue identities must be unique" {:ids queue-ids})))
      (doseq [event inputs]
        (when-not (logical-event? event)
          (throw (ex-info "execution plan contains a non-logical input event" {:event event})))
        (when (nil? (:id event))
          (throw (ex-info "logical event requires an identity" {:event event}))))
      (when-not (= (count input-ids) (count (set input-ids)))
        (throw (ex-info "execution plan input event identities must be unique"
                        {:ids input-ids})))
      (loop [remaining operations
             operation-ids #{}
             available-events (set input-ids)]
        (if-let [operation (first remaining)]
          (do
            (when-not (scheduled-operation? operation)
              (throw (ex-info "execution plan contains a non-scheduled operation"
                              {:operation operation})))
            (when (or (nil? (:id operation)) (contains? operation-ids (:id operation)))
              (throw (ex-info "scheduled operation identity must be non-nil and unique"
                              {:id (:id operation)})))
            (when (nil? (:operation operation))
              (throw (ex-info "scheduled operation requires an executable operation"
                              {:id (:id operation)})))
            (when-not (contains? (set queues) (:queue operation))
              (throw (ex-info "scheduled operation references an undeclared queue"
                              {:id (:id operation) :queue (:queue operation)})))
            (when-not (vector? (:waits operation))
              (throw (ex-info "scheduled operation waits must be an ordered vector"
                              {:id (:id operation) :waits (:waits operation)})))
            (doseq [event (:waits operation)]
              (when-not (logical-event? event)
                (throw (ex-info "scheduled operation contains a non-logical wait"
                                {:id (:id operation) :event event}))))
            (let [wait-ids (mapv :id (:waits operation))
                  completion (:completion operation)]
              (when-not (= (count wait-ids) (count (set wait-ids)))
                (throw (ex-info "scheduled operation waits must be unique"
                                {:id (:id operation) :waits wait-ids})))
              (when-not (set/subset? (set wait-ids) available-events)
                (throw (ex-info "scheduled operation wait must name an available event"
                                {:id (:id operation) :waits wait-ids
                                 :available available-events})))
              (when-not (logical-event? completion)
                (throw (ex-info "scheduled operation requires a logical completion event"
                                {:id (:id operation) :completion completion})))
              (when (or (nil? (:id completion))
                        (contains? available-events (:id completion)))
                (throw (ex-info "scheduled operation completion identity must be non-nil and unique"
                                {:id (:id operation) :completion (:id completion)})))
              (recur (next remaining)
                     (conj operation-ids (:id operation))
                     (conj available-events (:id completion)))))
          (let [output-ids (mapv :id outputs)]
            (doseq [event outputs]
              (when-not (logical-event? event)
                (throw (ex-info "execution plan contains a non-logical output event"
                                {:event event}))))
            (when-not (= (count output-ids) (count (set output-ids)))
              (throw (ex-info "execution plan output event identities must be unique"
                              {:ids output-ids})))
            (when-not (set/subset? (set output-ids) available-events)
              (throw (ex-info "execution plan output must name an available event"
                              {:outputs output-ids :available available-events})))))))
    plan))

(defn from-kernel-graph-call
  "Lower a KernelGraphCall dependency DAG to one target-neutral compute queue and explicit
   logical wait/completion events. Backends may conservatively serialize this plan while they
   lack multiple queues, but cannot lose its dependency information."
  [graph-call]
  (let [graph-call (kgcall/validate! graph-call)
        queue (compute-queue)
        event-by-node (into {} (map (fn [{:keys [id]}]
                                      [id (->LogicalEvent [:operation-complete id])])
                                    (:nodes graph-call)))
        operations (mapv (fn [{:keys [id call dependencies]}]
                           (kcall/validate! call)
                           (->ScheduledOperation id call queue
                                                 (mapv event-by-node dependencies)
                                                 (get event-by-node id)))
                         (:nodes graph-call))
        depended-on (set (mapcat :dependencies (:nodes graph-call)))
        outputs (mapv event-by-node
                      (keep (fn [{:keys [id]}]
                              (when-not (contains? depended-on id) id))
                            (:nodes graph-call)))]
    (validate! (->ExecutionPlan [queue] [] operations outputs))))
