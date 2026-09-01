(ns raster.compiler.backend.gpu.parallel-program-opencl
  "Equation-first OpenCL emission for a scheduled mixed ParallelProgram.

   This pass never inspects retained source or equation sites. Structured loops emit their exact
   one-iteration graph; ordinary TypedSOAC equations derive and emit the same checked graph. Host
   scalar equations remain explicit host-only steps for the whole-program executor."
  (:require [raster.compiler.backend.gpu.segop-opencl :as opencl]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-control-schedule :as schedule]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.structured-control-route :as structured-route]))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :pass :parallel-program-opencl))))

(defn- scalar-types
  [values operations]
  (into {}
        (map (fn [id]
               (let [value (get values id)]
                 (when-not value
                   (fail! :opencl-parallel-scalar-value
                          "scheduled scalar lacks an AbstractValue"
                          {:value id}))
                 [id (:dtype value)])))
        (distinct (mapcat segop/operation-scalars operations))))

(defn- scheduled-body
  [parallel-program equation]
  (let [algorithm (:algorithm equation)]
    (program/make
     {:dialect :segop
      :source nil
      :values (:values parallel-program)
      :inputs (:operands equation)
      :equations [equation]
      :outputs (:results equation)
      :effects (:effects equation)
      :diagnostics []
      :provenance {:source-dialect :typed-soac
                   :pass :parallel-program-opencl}
      :attributes {:host-control :explicit-typed-algorithm}
      :operation? segop/segop-node?
      :algorithm? (fn [candidate retained]
                    (and (= retained (soac/validate! retained))
                         (= (:operands candidate) (:inputs (soac/facts retained)))
                         (= (:results candidate) (soac/outputs retained))))})))

(defn- emit-graph
  [scheduled-graph values operations opts]
  (let [types (scalar-types values operations)]
    (opencl/generate-kernel-graph
     scheduled-graph
     :scalar-types (merge (:scalar-types opts) types))))

(defn- emit-equation
  [parallel-program equation opts]
  (let [algorithm (:algorithm equation)]
    (cond
      (control/loop-program? algorithm)
      (let [scheduled (schedule/validate! (first (:operations equation)))
            body (:body scheduled)
            operations (vec (mapcat :operations (:equations body)))
            emitted (emit-graph (:graph scheduled) (:values body) operations opts)]
        (assoc equation :operations
               [(emitted-loop/make
                 scheduled emitted
                 {:provenance {:target-dialect :opencl-c
                               :pass :parallel-program-opencl}})]))

      (and (soac/program-form? algorithm)
           (true? (get-in equation [:attributes :host-only])))
      equation

      (soac/program-form? algorithm)
      (let [body (scheduled-body parallel-program equation)
            scheduled-graph (equation-graph/make algorithm body)
            emitted (emit-graph scheduled-graph (:values body) (:operations equation) opts)]
        (assoc equation :operations
               [(emitted-equation/make
                 algorithm body emitted
                 {:provenance {:target-dialect :opencl-c
                               :pass :parallel-program-opencl}})]))

      :else
      (fail! :opencl-parallel-algorithm
             "scheduled equation has no supported retained algorithm"
             {:equation (:id equation) :algorithm algorithm}))))

(defn- emitted-operation?
  [operation]
  (or (emitted-loop/emitted-loop? operation)
      (emitted-equation/emitted-equation? operation)))

(defn- emitted-boundary?
  [equation algorithm]
  (cond
    (control/loop-program? algorithm)
    (and (= 1 (count (:operations equation)))
         (let [operation (first (:operations equation))]
           (and (emitted-loop/emitted-loop? operation)
                (= algorithm (:algorithm (:schedule (emitted-loop/validate! operation)))))))

    (soac/program-form? algorithm)
    (if (true? (get-in equation [:attributes :host-only]))
      (empty? (:operations equation))
      (and (= 1 (count (:operations equation)))
           (let [operation (first (:operations equation))]
             (and (emitted-equation/emitted-equation? operation)
                  (= algorithm (:algorithm (emitted-equation/validate! operation)))))))

    :else false))

(defn validate-program!
  [parallel-program]
  (when-not (= :opencl-parallel (:dialect parallel-program))
    (fail! :opencl-parallel-dialect
           "equation-first OpenCL emission requires :opencl-parallel"
           {:dialect (:dialect parallel-program)}))
  (program/validate! parallel-program emitted-operation? emitted-boundary?))

(defn emit-program
  "Emit every numerical equation directly and return its checked target program plus artifacts."
  ([parallel-program] (emit-program parallel-program {}))
  ([parallel-program opts]
   (let [parallel-program (structured-route/validate-scheduled-program! parallel-program)
         equations (mapv #(emit-equation parallel-program % opts)
                         (:equations parallel-program))
         emitted-program
         (validate-program!
          (program/make
           {:dialect :opencl-parallel
            :source (:source parallel-program)
            :values (:values parallel-program)
            :inputs (:inputs parallel-program)
            :equations equations
            :outputs (:outputs parallel-program)
            :effects (:effects parallel-program)
            :diagnostics (:diagnostics parallel-program)
            :provenance (assoc (:provenance parallel-program)
                               :target-dialect :opencl-parallel)
            :attributes (:attributes parallel-program)
            :operation? emitted-operation?
            :algorithm? emitted-boundary?}))
         graphs (keep (fn [equation]
                        (when-let [operation (first (:operations equation))]
                          (cond
                            (emitted-loop/emitted-loop? operation) (:graph operation)
                            (emitted-equation/emitted-equation? operation) (:graph operation))))
                      equations)
         kernels (vec (mapcat #(map :operation (:nodes (graph/validate! %)))
                              graphs))]
     {:program emitted-program
      :kernels kernels
      :stats {:structured-loops-emitted
              (count (filter (comp emitted-loop/emitted-loop? first :operations) equations))
              :typed-equations-emitted
              (count (filter (comp emitted-equation/emitted-equation? first :operations)
                             equations))
              :host-scalar-equations
              (count (filter #(get-in % [:attributes :host-only]) equations))}})))
