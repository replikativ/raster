(ns raster.compiler.passes.parallel.resident-program-projection
  "Project a source-independent emitted ParallelProgram back onto the resident marker ABI.

   This is deliberately a projection, not another scheduler: graphs, ABI order, effects and
   launch geometry are already certified by equation-first emission.  The resulting let form is
   consumed by `extract-gpu-program`, whose generic executable convention builds the ordinary
   resident descriptor used by LinkPlan and existing callers."
  (:require [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.soac-dialect :as soac]))

(defn- fail! [reason message data]
  (throw (ex-info message (assoc data :reason reason
                                 :pass :resident-program-projection))))

(defn- numerical-operations [program]
  (into []
        (comp (remove #(true? (get-in % [:attributes :host-only])))
              (mapcat :operations))
        (:equations program)))

(defn- operation-graph [operation]
  (when-not (emitted-equation/emitted-equation? operation)
    (fail! :resident-projection-operation
           "resident projection currently requires straight-line emitted equations"
           {:operation operation :actual (type operation)}))
  (:graph (emitted-equation/validate! operation)))

(defn- graph-strategy [operation graph]
  (or (executable/strategy graph)
      (get-in operation [:refinement :schedule :strategy])
      (get-in graph [:attributes :scheduled-kernel-body :body :schedule :strategy])
      :scheduled-graph))

(defn- graph-dispatch [ordinal operation]
  (let [graph0 (operation-graph operation)
        strategy (graph-strategy operation graph0)
        graph (-> graph0
                  (assoc-in [:attributes :strategy] strategy)
                  executable/validate!)
        id (str "resident-emitted-graph-"
                ordinal "-" (Integer/toUnsignedString (hash graph) 16))]
    (dispatch/make
     {:id id
      :alternatives [graph]
      :default-strategy strategy
      :selector {:kind :fixed-strategy :strategy strategy}
      :provenance {:pass :resident-program-projection
                   :source-dialect :emitted-parallel-program}
      :attributes {:operation-family :emitted-parallel-equation}})))

(defn- required-prefix-symbols [plan graphs]
  (let [steps-by-symbol (into {} (map (juxt :symbol identity)) (:steps plan))]
    (loop [pending (vec (mapcat :arguments graphs))
           required #{}]
      (if-let [symbol (peek pending)]
        (if (contains? required symbol)
          (recur (pop pending) required)
          (let [step (get steps-by-symbol symbol)]
            (recur (into (pop pending) (map :symbol (:operands step)))
                   (cond-> required step (conj symbol)))))
        required))))

(def ^:private array-constructor
  {:byte 'byte-array :half 'short-array :short 'short-array :int 'int-array
   :long 'long-array :float 'float-array :double 'double-array})

(defn- extent-expression [shape]
  (case (count shape)
    0 1
    1 (first shape)
    (list* 'clojure.core/* shape)))

(defn- scalar-expression [step]
  (let [{:keys [body-results]} (soac/lambda-parts (:region step))]
    (when-not (= 1 (count body-results))
      (fail! :resident-projection-scalar
             "resident scalar prefix step must have one result"
             {:step (:id step) :results body-results}))
    (first body-results)))

(defn- step-expression [step]
  (cond
    (invocation/scalar-compute? step)
    (scalar-expression step)

    (invocation/shape-projection? step)
    (let [source (some-> step :operands first :symbol)]
      (when-not (= 0 (:axis step))
        (fail! :resident-projection-shape
               "flat resident arrays can project only their physical axis zero"
               {:step (:id step) :axis (:axis step)}))
      (list 'clojure.core/alength source))

    (invocation/buffer-allocation? step)
    (let [dtype (get-in step [:value :dtype])
          constructor (array-constructor dtype)]
      (when-not constructor
        (fail! :resident-projection-allocation
               "resident prefix allocation has no JVM primitive storage constructor"
               {:step (:id step) :dtype dtype}))
      (list constructor (extent-expression (get-in step [:value :shape]))))

    (invocation/value-alias? step)
    (some-> step :operands first :symbol)

    :else
    (fail! :resident-projection-prefix
           "resident projection does not implement this invocation prefix step"
           {:step (:id step) :actual (type step)})))

(defn project
  "Return the emitted graphs, fixed dispatches, and resident-extractable marker form.

   Prefix allocation is retained only when an emitted graph still names that value.  A graph
   refinement can therefore eliminate a semantic temporary without leaving dead resident scratch."
  [parallel-program device-id]
  (let [program (emitted-program/validate! parallel-program)
        plan (some-> program :attributes :invocation-plan invocation/validate!)]
    (when-not plan
      (fail! :resident-projection-invocation
             "resident emitted-program projection requires its typed invocation plan" {}))
    (let [operations (numerical-operations program)
          dispatches (mapv graph-dispatch (range) operations)
          graphs (mapv dispatch/default-alternative dispatches)
          required (required-prefix-symbols plan graphs)
          prefix-steps (filterv #(contains? required (:symbol %)) (:steps plan))
          prefix-bindings (mapcat (fn [step] [(:symbol step) (step-expression step)]) prefix-steps)
          marker-bindings
          (mapcat
           (fn [ordinal graph graph-dispatch]
             (let [result-slots (keep-indexed
                                 (fn [index slot]
                                   (when (= :result (:role slot)) index))
                                 (:abi graph))
                   policy (if (= 1 (count result-slots)) :single :none)]
               [(symbol (str "resident_emitted_step_" ordinal))
                (list 'raster.compiler.pipeline/invoke-scheduled-executable!
                      device-id (:id graph-dispatch) (vec (:arguments graph)) policy)]))
           (range) graphs dispatches)
          result (case (count (:outputs program))
                   0 nil
                   1 (first (:outputs program))
                   (vec (:outputs program)))]
      {:form (list 'let* (vec (concat prefix-bindings marker-bindings)) result)
       :dispatches dispatches
       :kernels (vec (mapcat executable/artifacts graphs))
       :graphs graphs})))
