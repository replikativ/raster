(ns raster.compiler.backend.gpu.parallel-program-c-family
  "Equation-first C-family emission for a scheduled mixed ParallelProgram.

   This pass never inspects retained source or equation sites. Structured loops emit their exact
   one-iteration graph; ordinary TypedSOAC equations derive and emit the same checked graph. Host
   scalar equations remain explicit host-only steps. OpenCL, CUDA, and HIP differ only at the
   KernelBody source dialect boundary."
  (:require [raster.compiler.backend.gpu.kernel-body-c-dialect :as c-dialect]
            [raster.compiler.backend.gpu.segop-opencl :as segop-emission]
            [raster.compiler.ir.contraction-facts :as contraction-facts]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.compiler.ir.kernel-artifact :as kernel-artifact]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-control-schedule :as schedule]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.structured-control-route :as structured-route]
            [raster.compiler.passes.parallel.typed-soac-projection :as typed-projection]))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :pass :parallel-program-c-family))))

(defn- scalar-types
  [values operations]
  (into {}
        (map (fn [id]
               (let [value (get values id)]
                 (when-not value
                   (fail! :c-family-parallel-scalar-value
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
                   :pass :parallel-program-c-family}
      :attributes {:host-control :explicit-typed-algorithm}
      :operation? segop/segop-node?
      :algorithm? (fn [candidate retained]
                    (and (= retained (soac/validate! retained))
                         (= (:operands candidate) (:inputs (soac/facts retained)))
                         (= (:results candidate) (soac/outputs retained))))})))

(defn- emit-graph
  [scheduled-graph values operations opts]
  (let [types (scalar-types values operations)]
    (segop-emission/generate-kernel-graph
     scheduled-graph
     :scalar-types (merge (:scalar-types opts) types)
     :array-types (:array-types opts)
     :target-dialect (get opts :target-dialect :opencl-intel)
     :target-device (:target-device opts)
     :contraction-facts (:contraction-facts opts))))

(defn- contraction-facts-by-operation
  "Project typed contraction facts once at the algorithm/schedule boundary.

   The target emitter receives verified facts keyed by the immutable SegRed identity; it never
   reparses retained Clojure source or guesses that an arbitrary segmented reduction is GEMM."
  [algorithm operations]
  (let [equations (into {} (map (juxt second identity)) (soac/equations algorithm))]
    (into {}
          (keep (fn [operation]
                  (when (= :contraction (:phase operation))
                    (let [equation (or (get equations (:id operation))
                                       (fail! :c-family-contraction-equation
                                              "scheduled contraction lacks its typed equation"
                                              {:operation (:id operation)}))]
                      [(:id operation)
                       (contraction-facts/from-components
                        (typed-projection/segmented-reduce-contract-components
                         algorithm equation))]))))
          operations)))

(defn- target-program-dialect
  [target-dialect]
  (case (c-dialect/target (c-dialect/resolve! target-dialect))
    :opencl-c :opencl-parallel
    :cuda-c :cuda-parallel
    :hip-cpp :hip-parallel))

(defn- kernel-attribute
  [kernel key]
  (or (get kernel key) (get-in kernel [:attributes key])))

(defn- kernel-body-decline-key
  [kernel]
  (when-let [decline (kernel-attribute kernel :kernel-body-decline)]
    [(:reason decline) (:missing-rule decline) (:fallback decline)]))

(defn- emit-equation
  [parallel-program equation opts]
  (let [algorithm (:algorithm equation)
        target-dialect (get opts :target-dialect :opencl-intel)
        target-module (c-dialect/target (c-dialect/resolve! target-dialect))
        provenance {:target-dialect target-dialect
                    :target-module target-module
                    :pass :parallel-program-c-family}]
    (cond
      (control/loop-program? algorithm)
      (let [scheduled (schedule/validate! (first (:operations equation)))
            body (:body scheduled)
            operations (vec (mapcat :operations (:equations body)))
            emitted (emit-graph (:graph scheduled) (:values body) operations opts)]
        (assoc equation :operations
               [(emitted-loop/make scheduled emitted {:provenance provenance})]))

      (and (soac/program-form? algorithm)
           (true? (get-in equation [:attributes :host-only])))
      equation

      (soac/program-form? algorithm)
      (let [body (scheduled-body parallel-program equation)
            ;; Multi-phase schedules retain family-independent decisions (algebra, phase
            ;; decomposition, tuning choice) on their certified graph. Rebuilding the physical
            ;; dataflow remains generic, but those schedule facts must survive to target lowering.
            graph-contract (get-in equation [:attributes :kernel-graph])
            scheduled-graph (equation-graph/make
                             algorithm body
                             (cond-> {}
                               graph-contract
                               (assoc :provenance (:provenance graph-contract)
                                      :attributes (:attributes graph-contract))))
            operations (:operations equation)
            contraction-facts (contraction-facts-by-operation algorithm operations)
            emitted (emit-graph scheduled-graph (:values body) operations
                                (cond-> opts
                                  (seq contraction-facts)
                                  (assoc :contraction-facts contraction-facts)))]
        (assoc equation :operations
               [(emitted-equation/make algorithm body emitted {:provenance provenance})]))

      :else
      (fail! :c-family-parallel-algorithm
             "scheduled equation has no supported retained algorithm"
             {:equation (:id equation) :algorithm algorithm}))))

(defn validate-program!
  [parallel-program]
  (emitted-program/validate! parallel-program))

(defn emit-program
  "Emit every numerical equation directly to `:target-dialect`.

   Supported dialects are `:opencl-intel`, `:opencl-portable`, `:cuda`, and `:hip`."
  ([parallel-program]
   (emit-program parallel-program {}))
  ([parallel-program opts]
   (let [parallel-program (structured-route/validate-scheduled-program! parallel-program)
         target-dialect (get opts :target-dialect :opencl-intel)
         target-module (c-dialect/target (c-dialect/resolve! target-dialect))
         program-dialect (target-program-dialect target-dialect)
         equations (mapv #(emit-equation parallel-program % opts)
                         (:equations parallel-program))
         emitted-program
         (validate-program!
          (program/make
           {:dialect program-dialect
            :source (:source parallel-program)
            :values (:values parallel-program)
            :inputs (:inputs parallel-program)
            :equations equations
            :outputs (:outputs parallel-program)
            :effects (:effects parallel-program)
            :diagnostics (:diagnostics parallel-program)
            :provenance (assoc (:provenance parallel-program)
                               :target-dialect target-dialect
                               :target-module target-module)
            :attributes (:attributes parallel-program)
            :operation? emitted-program/emitted-operation?
            :algorithm? emitted-program/emitted-boundary?}))
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
              (count (filter #(get-in % [:attributes :host-only]) equations))
              :emission-routes (frequencies (map kernel-artifact/emission-route kernels))
              :kernel-body-declines
              (frequencies (keep kernel-body-decline-key kernels))}})))
