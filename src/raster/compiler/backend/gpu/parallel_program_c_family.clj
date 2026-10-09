(ns raster.compiler.backend.gpu.parallel-program-c-family
  "Equation-first C-family emission for a scheduled mixed ParallelProgram.

   This pass never inspects retained source or equation sites. Structured loops emit their exact
   one-iteration graph; ordinary TypedSOAC equations derive and emit the same checked graph. Host
   scalar equations remain explicit host-only steps. OpenCL, CUDA, and HIP differ only at the
   KernelBody source dialect boundary."
  (:require [raster.compiler.backend.gpu.kernel-body-c-dialect :as c-dialect]
            [raster.compiler.backend.gpu.kernel-body-target :as body-target]
            [raster.compiler.backend.gpu.gemm :as matrix-emission]
            [raster.compiler.backend.gpu.segop-opencl :as segop-emission]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.compiler.ir.kernel-artifact :as kernel-artifact]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-control-schedule :as schedule]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.contraction-schedule :as contraction-schedule]
            [raster.compiler.passes.parallel.mixed-matrix-candidate :as mixed-candidate]
            [raster.compiler.passes.parallel.product-consumer-region :as product-consumer-region]
            [raster.compiler.passes.parallel.product-consumer-route :as product-consumer-route]
            [raster.compiler.passes.parallel.structured-control-route :as structured-route]
            [raster.compiler.passes.parallel.typed-contraction-context :as contraction-context]))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :pass :parallel-program-c-family))))

(defn- make-emitted-equation
  [algorithm body emitted metadata opts]
  (if (contains? opts :scalar-math)
    (emitted-equation/make algorithm body emitted metadata (select-keys opts [:scalar-math]))
    (emitted-equation/make algorithm body emitted metadata)))

(defn- contraction-write-domains
  [reference opts]
  (if (contains? opts :scalar-math)
    (emitted-equation/contraction-write-domains reference (select-keys opts [:scalar-math]))
    (emitted-equation/contraction-write-domains reference)))

(defn- emit-graph
  [scheduled-graph opts]
  ;; Graph construction already projects and checks the logical scalar interface. Do not
  ;; reconstruct a second interface by inspecting operation families at the target boundary.
  (let [scheduled-graph (graph/validate! scheduled-graph)
        types (into {} (map (juxt :id :dtype)) (:scalars scheduled-graph))]
    (segop-emission/generate-kernel-graph
     scheduled-graph
     (assoc (select-keys opts [:array-types :target-device :target-descriptor :schedule :scalar-math
                              :contraction-facts :scheduled-equation-algorithm
                              :scheduled-equation-body :scheduled-bodies])
            :scalar-types (merge (:scalar-types opts) types)
            :target-dialect (get opts :target-dialect :opencl-intel)))))

(defn- contraction-facts-by-operation
  "Project typed contraction facts once at the algorithm/schedule boundary.

   The target emitter receives verified facts keyed by the immutable SegRed identity; it never
   reparses retained Clojure source or guesses that an arbitrary segmented reduction is GEMM."
  [algorithm operations]
  (into {}
        (keep (fn [operation]
                (when (= :contraction (:phase operation))
                  [(:id operation)
                   (:facts (contraction-context/validate! algorithm operation))])))
        operations))

(defn emit-register-contraction-alternative
  "Emit a legal register candidate from one already certified plain FP32 equation.

   Unsupported equations return an explicit admission decline. No source analysis, source
   recognition, or second semantic program is involved; target emission retains the same spine."
  [reference {:keys [target-descriptor schedule target-dialect] :as opts}]
  (if-not (contraction-write-domains reference opts)
    {:ok false :reason :not-single-plain-fp32-contraction}
    (let [algorithm (:algorithm reference)
          source (equation-graph/make algorithm (:body reference))
          node (first (:nodes source))
          facts (:facts (contraction-context/validate! algorithm (:operation node)))
          planned (contraction-schedule/plan-register-tiled-for-node
                   node source facts target-descriptor
                   (assoc opts :precision (:precision schedule)
                               :multiply-add (get-in schedule [:typed-contraction :multiply-add]
                                                     :decomposed)))]
      (if-not (:ok planned)
        planned
        (let [original (get-in reference [:graph :nodes 0 :operation])
              artifact (body-target/emit-artifact
                        (str (:kernel-name original) "_register_tiled")
                        (:scheduled planned) target-dialect
                        (select-keys opts [:target-descriptor :scalar-math]))
              candidate (make-emitted-equation
                         algorithm (:body reference)
                         (-> (:graph reference)
                             (assoc-in [:nodes 0 :operation] artifact)
                             (assoc-in [:attributes :strategy] :register-tiled))
                         {:provenance (:provenance reference)} opts)]
          {:ok true :candidate candidate})))))

(defn emit-mixed-contraction-alternative
  "Emit one explicitly requested full-K mixed candidate from a certified public equation.

   This does not select or authorize approximation. Dispatch admission must consent to the
   independently reconstructed numerical model and retain the exact portable fallback."
  [reference {:keys [target-descriptor target-dialect schedule]
              :or {target-dialect :opencl-intel} :as opts}]
  (cond
    (not= :opencl-intel (:id (c-dialect/resolve! target-dialect)))
    {:ok false :reason :mixed-matrix-target-dialect}
    (not (contraction-write-domains reference opts))
    {:ok false :reason :not-single-plain-fp32-contraction}
    :else
    (let [algorithm (:algorithm reference)
          source (equation-graph/make algorithm (:body reference))
          planned (mixed-candidate/plan
                   algorithm source target-descriptor
                   (merge (select-keys schedule [:precision])
                          (select-keys (:typed-contraction schedule) [:tile :input-fusion])
                          (select-keys opts [:scalar-math])))]
      (if-not (:ok planned)
        planned
        (let [emitted (matrix-emission/emit-scheduled-stage-graph
                       (:graph planned)
                       (merge {:target-dialect target-dialect
                        :prefix (str (get-in reference [:graph :nodes 0 :operation :kernel-name])
                                     "_mixed")
                        :refinement (:refinement planned)}
                              (select-keys opts [:scalar-math :target-descriptor])))
              candidate (make-emitted-equation
                         algorithm (:body reference) emitted
                         {:refinement (:refinement planned)
                          :provenance (:provenance reference)} opts)
              report (if (contains? opts :scalar-math)
                       (emitted-equation/validate-with-result-contracts candidate (select-keys opts [:scalar-math]))
                       (emitted-equation/validate-with-result-contracts candidate))]
          (if-not (seq (:complete-write-domains report))
            {:ok false :reason :mixed-matrix-complete-write}
            {:ok true :candidate candidate :numerical-model (:numerical-model report)
             :target-schedule (:target-schedule planned)}))))))

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
            emitted (emit-graph
                     (:graph scheduled)
                     (assoc opts
                            :scheduled-equation-algorithm
                            (control/body (:algorithm scheduled))
                            :scheduled-equation-body body))]
        (assoc equation :operations
               [(if (contains? opts :scalar-math)
                  (emitted-loop/make scheduled emitted {:provenance provenance} (select-keys opts [:scalar-math]))
                  (emitted-loop/make scheduled emitted {:provenance provenance}))]))

      (and (soac/program-form? algorithm)
           (true? (get-in equation [:attributes :host-only])))
      equation

      (soac/program-form? algorithm)
      (let [{:keys [body graph]} (equation-graph/make-for-equation
                                  parallel-program equation)
            ;; Multi-phase schedules retain family-independent decisions (algebra, phase
            ;; decomposition, tuning choice) on their certified graph. Rebuilding the physical
            ;; dataflow remains generic, but those schedule facts must survive to target lowering.
            operations (:operations equation)
            contraction-facts (contraction-facts-by-operation algorithm operations)
            emitted (emit-graph graph
                                (cond-> (assoc opts
                                               :scheduled-equation-algorithm algorithm
                                               :scheduled-equation-body body)
                                  (seq contraction-facts)
                                  (assoc :contraction-facts contraction-facts)))]
        (assoc equation :operations
               [(make-emitted-equation algorithm body emitted {:provenance provenance} opts)]))

      (swr/plan? algorithm)
      (let [{:keys [body graph]} (equation-graph/make-for-plan-equation
                                  parallel-program equation)
            node (first (:nodes graph))
            certificate (first (:operations equation))
            emitted (emit-graph graph (assoc opts :scheduled-bodies {(:id node) certificate}))]
        (assoc equation :operations
               [(make-emitted-equation algorithm body emitted {:provenance provenance} opts)]))

      :else
      (fail! :c-family-parallel-algorithm
             "scheduled equation has no supported retained algorithm"
             {:equation (:id equation) :algorithm algorithm}))))

(defn product-consumer-plans
  "Admit non-overlapping consecutive numerical pairs through the generic region proof.

   Declines are ordinary: their equations continue through independent emission. Any unexpected
   analysis failure remains a compiler error rather than silently selecting another route."
  [parallel-program]
  (let [numerical (filterv (comp seq :operations) (:equations parallel-program))]
    (mapv #(assoc %2 :region-ordinal %1)
          (range)
          (loop [remaining numerical plans []]
            (if (< (count remaining) 2)
              plans
              (let [pair (subvec remaining 0 2)
                    result (if (every? (comp soac/program-form? :algorithm) pair)
                             (try
                               {:plan (product-consumer-region/analyze parallel-program pair)}
                               (catch clojure.lang.ExceptionInfo exception
                                 (if (product-consumer-region/declined? exception)
                                   {:declined true}
                                   (throw exception))))
                             {:declined true})]
                (if-let [plan (:plan result)]
                  (recur (subvec remaining 2) (conj plans plan))
                  (recur (subvec remaining 1) plans))))))))

(defn- emit-product-consumer
  [plan opts]
  (let [target-dialect (get opts :target-dialect :opencl-intel)
        target-module (c-dialect/target (c-dialect/resolve! target-dialect))
        target-device (:target-device opts)
        target-description (or (:target-descriptor opts)
                               (cond
                                 (map? target-device) target-device
                                 target-device (hardware/descriptor-for target-device)
                                 :else nil))
        routed (if (contains? opts :scalar-math)
                 (product-consumer-route/schedule plan target-description (select-keys opts [:scalar-math]))
                 (product-consumer-route/schedule plan target-description))
        kernel-name (str "rstr_product_consumer_" (:region-ordinal plan))
        {:keys [emitted refinement]} (product-consumer-route/emit
                                      kernel-name routed target-dialect
                                      (select-keys opts [:target-descriptor :scalar-math]))
        algorithm (get-in plan [:source :algorithm])
        body (get-in plan [:source :body])
        facts (soac/facts algorithm)
        source-graph (get-in routed [:refinement :source])
        retained-operands (set (concat (map :id (:inputs source-graph))
                                       (map :id (:scalars source-graph))))
        operands (filterv retained-operands (:inputs facts))
        operation (make-emitted-equation
                   algorithm body emitted
                   {:refinement refinement
                    :provenance {:target-dialect target-dialect
                                 :target-module target-module
                                 :pass :parallel-program-c-family
                                 :schedule :product-ordered-consumer}} opts)]
    (program/->ProgramEquation
     [:product-ordered-consumer (:equations plan)]
     [:equation-region (:equations plan)] nil
     operands (soac/outputs algorithm) algorithm [operation]
     (:effects facts)
     {:source-equations (:equations plan)
      :pass :parallel-program-c-family}
     {:emitted-source-equations (:equations plan)
      :schedule :product-ordered-consumer})))

(defn- emit-equations
  [parallel-program opts]
  (let [plans (product-consumer-plans parallel-program)
        by-consumer (into {} (map (fn [plan] [(peek (:equations plan)) plan])) plans)
        producers (set (map (comp first :equations) plans))]
    (mapv (fn [equation]
            (if (contains? by-consumer (:id equation))
              (emit-product-consumer (get by-consumer (:id equation)) opts)
              (emit-equation parallel-program equation opts)))
          (remove #(contains? producers (:id %)) (:equations parallel-program)))))

(defn validate-program!
  ([parallel-program] (emitted-program/validate! parallel-program))
  ([parallel-program caller-options] (emitted-program/validate! parallel-program caller-options)))

(defn- validate-program-for-request
  [parallel-program opts]
  (if (contains? opts :scalar-math)
    (validate-program! parallel-program (select-keys opts [:scalar-math]))
    (validate-program! parallel-program)))

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
         equations (emit-equations parallel-program opts)
         region-count (count (filter #(= :product-ordered-consumer
                                         (get-in % [:attributes :schedule])) equations))
         program-inputs (if (pos? region-count)
                          (program/infer-inputs equations)
                          (:inputs parallel-program))
         emitted-program
         (validate-program-for-request
          (program/make
           {:dialect program-dialect
            :source (:source parallel-program)
            :values (:values parallel-program)
            :inputs program-inputs
            :equations equations
            :outputs (:outputs parallel-program)
            :effects (:effects parallel-program)
            :diagnostics (:diagnostics parallel-program)
            :provenance (assoc (:provenance parallel-program)
                               :target-dialect target-dialect
                               :target-module target-module)
            :attributes (:attributes parallel-program)
            :operation? emitted-program/emitted-operation?
            :algorithm? (if (contains? opts :scalar-math)
                          #(emitted-program/emitted-boundary? %1 %2 (select-keys opts [:scalar-math]))
                          emitted-program/emitted-boundary?)})
          opts)
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
      :stats (cond->
              {:structured-loops-emitted
               (count (filter (comp emitted-loop/emitted-loop? first :operations) equations))
               :typed-equations-emitted
               (count (filter (comp emitted-equation/emitted-equation? first :operations)
                              equations))
               :host-scalar-equations
               (count (filter #(get-in % [:attributes :host-only]) equations))
               :emission-routes (frequencies (map kernel-artifact/emission-route kernels))
               :kernel-body-declines
               (frequencies (keep kernel-body-decline-key kernels))}
               (pos? region-count)
               (assoc :product-consumer-regions-emitted region-count))})))
