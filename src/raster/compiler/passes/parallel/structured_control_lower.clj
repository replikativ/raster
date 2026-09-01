(ns raster.compiler.passes.parallel.structured-control-lower
  "Certified scheduling of typed sequential control.

   The loop remains a sequential fixpoint while its closed TypedSOAC body takes the ordinary
   TypedSOAC -> ParallelProgram -> SegOp -> KernelGraph vertical.  This pass deliberately does not
   emit a persistent kernel or reconstruct a source-shaped compound program."
  (:require [clojure.set :as set]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-control-schedule :as control-schedule]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]))

(def scheduled-loop? control-schedule/scheduled-loop?)

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

(def validate! control-schedule/validate!)

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
    (control-schedule/make
     {:algorithm algorithm
      :body scheduled
      :graph graph
      :strategy {:kind :host-repetition :association :sequential}
      :effects (:effects (control/facts algorithm))
      :provenance {:source-dialect :typed-structured-control
                   :pass :structured-control-lower}
      :attributes {:body-dialect :typed-soac
                   :schedule-dialect :segop
                   :graph-dialect :kernel-graph}})))
