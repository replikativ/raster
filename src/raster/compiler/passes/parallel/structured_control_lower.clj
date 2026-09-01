(ns raster.compiler.passes.parallel.structured-control-lower
  "Certified scheduling of typed sequential control.

   The loop remains a sequential fixpoint while its closed TypedSOAC body takes the ordinary
   TypedSOAC -> ParallelProgram -> SegOp -> KernelGraph vertical.  This pass deliberately does not
   emit a persistent kernel or reconstruct a source-shaped compound program."
  (:require [clojure.set :as set]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]))

(defrecord ScheduledStructuredLoop [algorithm body graph strategy effects provenance attributes])

(defn scheduled-loop?
  [value]
  (and value
       (= "raster.compiler.passes.parallel.structured_control_lower.ScheduledStructuredLoop"
          (.getName (class value)))))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :pass :structured-control-lower))))

(defn- value-elements
  [value]
  (let [shape (:shape value)]
    (cond
      (empty? shape) 1
      (= 1 (count shape)) (first shape)
      :else (apply launch/product shape))))

(defn- physical-body-outputs
  [algorithm]
  (let [facts (soac/facts algorithm)
        producer (into {}
                       (mapcat (fn [equation]
                                 (map vector (nth equation 2)
                                      (soac/physical-results facts equation))))
                       (soac/equations algorithm))]
    (mapv (fn [result]
            (or (get producer result)
                (fail! :structured-loop-result-storage
                       "structured loop body result has no physical storage identity"
                       {:result result})))
          (soac/outputs algorithm))))

(defn- external-inputs
  "Find physical values read before any body operation initializes them."
  [operations]
  (:inputs
   (reduce (fn [{:keys [initialized] :as state} operation]
             (let [reads (segop/operation-inputs operation)
                   writes (segop/operation-outputs operation)]
               (-> state
                   (update :inputs set/union (set/difference reads initialized))
                   (update :initialized set/union writes))))
           {:inputs #{} :initialized #{}}
           operations)))

(defn- storage-spec
  [values id]
  (let [value (get values id)]
    (when-not value
      (fail! :structured-loop-storage-value
             "scheduled structured loop storage lacks an AbstractValue"
             {:value id}))
    {:dtype (:dtype value)
     :elements (value-elements value)
     :memory-space (or (:memory-space value) :device)}))

(defn- body-graph
  [algorithm scheduled]
  (let [operations (vec (mapcat :operations (:equations scheduled)))
        _ (when (empty? operations)
            (fail! :structured-loop-empty-schedule
                   "structured loop body requires at least one scheduled parallel operation" {}))
        _ (when (some #(get-in % [:attributes :host-only]) (:equations scheduled))
            (fail! :structured-loop-host-scalar
                   "host scalar equations inside structured control require explicit loop scalar lowering"
                   {}))
        inputs (external-inputs operations)
        outputs (set (physical-body-outputs algorithm))
        operation-values (reduce set/union #{}
                                 (map #(set/union (segop/operation-inputs %)
                                                  (segop/operation-outputs %))
                                      operations))
        temporary-ids (set/difference operation-values inputs outputs)
        values (:values scheduled)
        buffer-specs (into {}
                           (map (fn [id] [id (storage-spec values id)]))
                           (set/union inputs outputs temporary-ids))
        temporaries (select-keys buffer-specs temporary-ids)]
    (graph/from-segops
     operations
     {:inputs inputs
      :outputs outputs
      :temporaries temporaries
      :buffer-specs buffer-specs
      :dtype (:dtype (first (vals buffer-specs)))
      :effects {:semantic (:effects (soac/facts algorithm))}
      :provenance {:source-dialect :typed-structured-control
                   :algorithm-dialect :typed-soac
                   :schedule-dialect :segop}
      :attributes {:control :sequential-fixpoint
                   :execution :host-repetition}})))

(defn validate!
  [scheduled-loop]
  (when-not (scheduled-loop? scheduled-loop)
    (fail! :structured-loop-schedule-type "expected a ScheduledStructuredLoop"
           {:actual (type scheduled-loop)}))
  (let [{:keys [algorithm body graph strategy effects provenance attributes]} scheduled-loop
        algorithm (control/validate! algorithm)
        body (parallel-program/validate! body segop/segop-node?)
        graph (graph/validate! graph)
        typed-body (control/body algorithm)
        retained-equations (vec (mapcat (comp soac/equations :algorithm) (:equations body)))]
    (when-not (= :segop (:dialect body))
      (fail! :structured-loop-body-dialect "structured loop body must be fully scheduled SegOp"
             {:dialect (:dialect body)}))
    (when-not (and (= (soac/equations typed-body) retained-equations)
                   (= (:inputs (soac/facts typed-body)) (:inputs body))
                   (= (soac/outputs typed-body) (:outputs body)))
      (fail! :structured-loop-algorithm
             "structured loop body boundary or equations changed during scheduling"
             {:algorithm-inputs (:inputs (soac/facts typed-body))
              :scheduled-inputs (:inputs body)
              :algorithm-outputs (soac/outputs typed-body)
              :scheduled-outputs (:outputs body)}))
    (when-not (= {:kind :host-repetition :association :sequential} strategy)
      (fail! :structured-loop-strategy "first structured loop schedule must be host repetition"
             {:strategy strategy}))
    (when-not (= effects (:effects (control/facts algorithm)))
      (fail! :structured-loop-effects "scheduled loop effects differ from its algorithm"
             {:scheduled effects :algorithm (:effects (control/facts algorithm))}))
    (doseq [[field value] [[:provenance provenance] [:attributes attributes]]]
      (when-not (map? value)
        (fail! :structured-loop-description "scheduled loop descriptions must be maps"
               {:field field :value value})))
    (when-not (= (vec (mapcat :operations (:equations body)))
                 (mapv :operation (:nodes graph)))
      (fail! :structured-loop-graph "KernelGraph operations differ from the scheduled body" {}))
    scheduled-loop))

(defn schedule
  "Schedule a typed structured loop without changing its sequential association.

   The returned KernelGraph describes one iteration. A runtime host-repetition binder can replay
   it with the loop's ordered invariant/carry bindings; persistent device execution is a distinct
   future schedule, not an emitter fallback."
  [algorithm opts]
  (let [algorithm (control/validate! algorithm)
        envelope (typed-route/program-envelope (control/body algorithm))
        scheduled (:form (segop-lower/segop-lower-pass envelope opts))
        graph (body-graph (control/body algorithm) scheduled)]
    (validate!
     (->ScheduledStructuredLoop
      algorithm scheduled graph
      {:kind :host-repetition :association :sequential}
      (:effects (control/facts algorithm))
      {:source-dialect :typed-structured-control :pass :structured-control-lower}
      {:body-dialect :typed-soac :schedule-dialect :segop :graph-dialect :kernel-graph}))))
