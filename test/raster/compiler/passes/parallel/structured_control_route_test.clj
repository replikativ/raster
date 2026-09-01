(ns raster.compiler.passes.parallel.structured-control-route-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.walk :as walk]
            [raster.compiler.backend.gpu.parallel-program-opencl :as program-opencl]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.dialects :as dialects]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-control-schedule :as schedule]
            [raster.compiler.ir.structured-loop-call :as loop-call]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.structured-control-frontend :as frontend]
            [raster.compiler.passes.parallel.structured-control-route :as route]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.pipeline :as pipeline]
            [raster.gpu.parallel-program :as program-runtime]
            [raster.ode.pde :as pde]))

(defn- evaluate-test-scalar-expression [expression operands]
  (cond
    (number? expression) expression
    (symbol? expression)
    (if-let [value (get operands expression)]
      (:value value)
      (throw (ex-info "test scalar references an unknown operand" {:expression expression})))
    (seq? expression)
    (let [operation (name (first expression))
          arguments (mapv #(evaluate-test-scalar-expression % operands) (rest expression))]
      (case operation
        "int" (int (first arguments))
        "long" (long (first arguments))
        "float" (float (first arguments))
        "double" (double (first arguments))
        "*" (reduce * arguments)
        "/" (reduce / arguments)
        (throw (ex-info "test scalar operation is unsupported" {:expression expression}))))
    :else (throw (ex-info "test scalar expression is unsupported" {:expression expression}))))

(defn- evaluate-test-scalar [step operands]
  (let [expression (-> step :region soac/lambda-parts :body-results first)]
    {:type (get-in step [:value :dtype])
     :value (evaluate-test-scalar-expression expression operands)}))

(defn- loop-decomposition
  []
  (let [extent (av/tensor {:dtype :int :shape []})
        index (av/tensor {:dtype :long :shape []})
        tensor (av/tensor {:dtype :float :shape '[n]})
        inner (av/tensor {:dtype :float :shape '[n-in]})
        equation (list '= 'advance '[u-next]
                       (list 'map {:index 'i :extent 'n-in}
                             '[u-in] '[]
                             (soac/lambda-form '[value] '[(+ value 1.0)])))
        body (soac/make
              (soac/default-program-facts
               {:values {'iteration index 'n-in extent 'u-in inner 'u-next inner}
                :inputs '[n-in u-in]
                :equations {'advance (soac/default-equation-facts)}})
              [equation] '[u-next])
        loop (control/make
              {:id 'time-loop :effects #{} :provenance {:source :test}
               :attributes {:association :sequential}}
              '[iteration steps]
              [{:outer 'n :parameter 'n-in}]
              [{:initial 'u :parameter 'u-in :result 'u-next :output 'u-final}]
              body
              {'steps index 'n extent 'u tensor 'u-final tensor})]
    {:loop loop :loop-binding 'time-loop :source '(let* [] nil)}))

(deftest structured-control-uses-the-common-program-equation-spine
  (let [typed (route/program-envelope (loop-decomposition))
        scheduled (route/schedule-program typed {:target-device :cpu:0 :dtype :float})
        typed-equation (first (:equations typed))
        scheduled-equation (first (:equations scheduled))]
    (is (program/parallel-program? typed))
    (is (= :typed-parallel (:dialect typed)))
    (is (= '[steps n u] (:inputs typed)))
    (is (= '[u-final] (:outputs typed)))
    (is (= (:algorithm typed-equation) (:algorithm scheduled-equation)))
    (is (= :scheduled-parallel (:dialect scheduled)))
    (is (schedule/scheduled-loop? (first (:operations scheduled-equation))))
    (testing "the scheduled equation retains the certified one-iteration graph"
      (is (= :kernel-graph
             (get-in scheduled-equation [:attributes :graph-dialect])))
      (is (= '[u-in]
             (mapv :id (get-in scheduled-equation [:operations 0 :graph :inputs])))))))

(deftest repeated-outer-operands-do-not-break-program-ssa
  (let [{:keys [loop] :as decomposition} (loop-decomposition)
        loop (control/make
              (control/facts loop)
              (assoc (control/loop-index loop) 1 'n)
              (control/invariants loop)
              (control/carried loop)
              (control/body loop)
              (control/outer-values loop))
        typed (route/program-envelope (assoc decomposition :loop loop))]
    (is (= '[n u] (:inputs typed)))
    (is (= '[n u] (:operands (first (:equations typed)))))))

(defn- mixed-source
  []
  '(let* [n (int (clojure.core/alength u0))
          u (clojure.core/aclone u0)
          scratch (clojure.core/double-array n)
          time-loop
          (dotimes [step steps]
            (raster.par/map! scratch i n double
                             (+ (clojure.core/aget u i) 1.0))
            (let* [next (raster.par/pmap j n double
                                         (+ (clojure.core/aget u j)
                                            (clojure.core/aget scratch j)
                                            (* 0.0 step)))]
                  (java.lang.System/arraycopy next 0 u 0 n)))
          after (raster.par/pmap k n double
                                 (* 2.0 (clojure.core/aget u k)))]
         after))

(defn- mixed-source-without-induction
  []
  (walk/postwalk
   #(if (= '(* 0.0 step) %) 0.0 %)
   (mixed-source)))

(defn- source-with-suffix
  [suffix-bindings result]
  (let [[_ bindings] (mixed-source)]
    (list 'let* (vec (concat (take 8 bindings) suffix-bindings)) result)))

(defn- reason-of
  [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo exception
      (:reason (ex-data exception)))))

(deftest loop-output-feeds-an-ordinary-typed-suffix
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        decomposition (frontend/form->structured-loop (mixed-source) options)
        typed (route/program-envelope decomposition options)
        scheduled (route/schedule-program typed {:target-device :cpu:0 :dtype :double})
        [loop-equation suffix-equation] (:equations typed)
        [scheduled-loop scheduled-suffix] (:equations scheduled)
        loop-output (first (:results loop-equation))]
    (is (= 2 (count (:equations typed))))
    (is (= [loop-output 'n] (:operands suffix-equation)))
    (is (= '[after] (:outputs typed)))
    (is (= '[steps n u] (:inputs typed)))
    (is (= true (get-in typed [:attributes :mixed-algorithms])))
    (is (schedule/scheduled-loop? (first (:operations scheduled-loop))))
    (is (segop/segop-node? (first (:operations scheduled-suffix))))
    (is (not-any? #{'u} (:operands suffix-equation)))))

(deftest mixed-routing-never-drops-an-unsupported-suffix
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        source (source-with-suffix
                '[unsupported (java.lang.System/gc)]
                'unsupported)
        decomposition (frontend/form->structured-loop source options)]
    (is (= :structured-control-unsupported-suffix
           (reason-of #(route/program-envelope decomposition options))))))

(deftest suffix-shadowing-is-rejected-instead-of-breaking-program-ssa
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        source (source-with-suffix
                '[u (raster.par/pmap k n double
                                     (* 2.0 (clojure.core/aget u k)))]
                'u)
        decomposition (frontend/form->structured-loop source options)]
    (is (= :parallel-program-result-redefinition
           (reason-of #(route/program-envelope decomposition options))))))

(deftest algorithm-kind-and-operation-kind-must-remain-paired
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        decomposition (frontend/form->structured-loop (mixed-source) options)
        typed (route/program-envelope decomposition options)
        suffix-operation (get-in typed [:equations 1 :operations 0])
        malformed (assoc-in typed [:equations 0 :operations] [suffix-operation])]
    (is (= :parallel-program-algorithm
           (reason-of #(route/schedule-program malformed
                                               {:target-device :cpu:0 :dtype :double}))))))

(deftest public-scheduler-selects-the-structured-semantic-program
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :target-device :ocl:0
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        attempted (route/attempt (mixed-source) options)
        scheduled (pipeline/schedule-parallel-form (mixed-source) options)]
    (is (= :typed-structured-control (get-in attempted [:stats :route])))
    (is (dialects/valid-source-or-typed-soac? (:program attempted)))
    (is (= :scheduled-parallel (:dialect (:form scheduled))))
    (is (dialects/valid-scheduled-program? (:form scheduled)))
    (is (= :typed-structured-control (get-in scheduled [:stats :source-dialect])))
    (is (= 1 (get-in scheduled [:stats :structured-loops-scheduled])))
    (is (= 2 (count (:equations (:form scheduled)))))))

(deftest structured-attempt-exposes-a-coverage-decline
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :target-device :ocl:0
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        source (source-with-suffix '[unsupported (java.lang.System/gc)] 'unsupported)
        attempted (route/attempt source options)]
    (is (nil? (:program attempted)))
    (is (= :structured-control-unsupported-suffix
           (get-in attempted [:declined :reason])))))

(deftest cpu-public-scheduling-does-not-enter-the-gpu-structured-route
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :target-device :cpu:0
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        scheduled (pipeline/schedule-parallel-form (mixed-source) options)]
    (is (not= :scheduled-parallel (:dialect (:form scheduled))))
    (is (not= :typed-structured-control (get-in scheduled [:stats :source-dialect])))))

(deftest ordinary-suffix-equations-use-the-same-kernel-graph-builder
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :target-device :ocl:0
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        typed (:program (route/attempt (mixed-source) options))
        algorithm (get-in typed [:equations 1 :algorithm])
        scheduled (:form (segop-lower/segop-lower-pass
                          (typed-route/program-envelope algorithm) options))
        kernel-graph (equation-graph/make
                      algorithm scheduled
                      {:provenance {:source-dialect :mixed-suffix}})]
    (is (= kernel-graph (graph/validate! kernel-graph)))
    (is (= (vec (mapcat :operations (:equations scheduled)))
           (mapv :operation (:nodes kernel-graph))))
    (is (= :mixed-suffix (get-in kernel-graph [:provenance :source-dialect])))
    (is (nil? (:source scheduled)))))

(deftest equation-first-opencl-emits-loop-and-suffix-without-reading-source
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :target-device :ocl:0
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        scheduled (:form (pipeline/schedule-parallel-form (mixed-source) options))
        opaque-source (assoc scheduled :source ::must-not-be-inspected)
        {:keys [program kernels stats]} (program-opencl/emit-program opaque-source options)
        [loop-equation suffix-equation] (:equations program)]
    (is (= :opencl-parallel (:dialect program)))
    (is (= ::must-not-be-inspected (:source program)))
    (is (emitted-loop/emitted-loop? (first (:operations loop-equation))))
    (is (emitted-equation/emitted-equation? (first (:operations suffix-equation))))
    (is (every? artifact/kernel-artifact? kernels))
    (is (= {:structured-loops-emitted 1
            :typed-equations-emitted 1
            :host-scalar-equations 0}
           stats))
    (is (= program (program-opencl/validate-program! program)))
    (testing "emitted algorithm kinds cannot be exchanged between equations"
      (let [suffix-operation (get-in suffix-equation [:operations 0])]
        (is (= :parallel-program-algorithm
               (reason-of #(program-opencl/validate-program!
                            (assoc-in program [:equations 0 :operations]
                                      [suffix-operation])))))))))

(deftest real-rk4-pde-reaches-the-equation-first-opencl-vertical
  (let [options {:dtype :double :target-device :ze:debug
                 :source-ns (the-ns 'raster.ode.pde)
                 :public-parameters '[u0 target alpha inv-dx2 dt nsteps]
                 :array-types {'u0 :double 'target :double}
                 :scalar-types {'alpha :double 'inv-dx2 :double
                                'dt :double 'nsteps :long}}
        walked (pipeline/get-walked-body #'pde/heat-loss-rk4 :double)
        source (if (= 1 (count walked)) (first walked) (list* 'do walked))
        semantic (pipeline/run-passes
                  source pipeline/gpu-resident-pre-soa-passes options)
        scheduled (route/schedule-program semantic options)
        emitted (program-opencl/emit-program scheduled options)
        emitted-program (:program emitted)
        invocation-plan (get-in semantic [:attributes :invocation-plan])
        materialized
        (materialization/materialize
         invocation-plan
         [(double-array 8) (double-array 8) 0.1 1.0 0.01 3]
         evaluate-test-scalar)
        loop-operation (get-in scheduled [:equations 0 :operations 0])
        loop-equation (first (:equations emitted-program))
        trip-count-id (second (control/loop-index (:algorithm loop-equation)))
        loop-output (-> loop-equation :algorithm control/carried first :output)
        device-results
        (set (mapcat :results
                     (remove #(true? (get-in % [:attributes :host-only]))
                             (:equations emitted-program))))
        buffer-ids
        (into device-results
              (filter #(seq (:shape (get (:values emitted-program) %)))
                      (:inputs emitted-program)))
        buffers (zipmap buffer-ids (map #(keyword (str "rk4-buffer-" (hash %))) buffer-ids))
        scalar-values
        (into {}
              (keep (fn [id]
                      (when-not (contains? buffer-ids id)
                        (let [dtype (get-in emitted-program [:values id :dtype])]
                          [id {:type dtype
                               :value (case dtype
                                        :int (if (= id trip-count-id) 2 64)
                                        :long 2
                                        :float (float 0.25)
                                        :double 0.25)}]))))
              (:inputs emitted-program))
        call
        (program-call/make
         emitted-program buffers scalar-values {loop-output :rk4-carry-scratch}
         (fn [equation {:keys [operands]}]
           (let [operand-id (first (:operands equation))
                 divisor (double (get-in operands [operand-id :value]))]
             (into {}
                   (map (fn [id]
                          [id {:type (get-in emitted-program [:values id :dtype])
                               :value (/ 1.0 divisor)}]))
                   (:results equation)))))]
    (is (= :typed-parallel (:dialect semantic)))
    (is (invocation/invocation-plan? invocation-plan))
    (is (= '[u0 target alpha inv-dx2 dt nsteps]
           (mapv :symbol (:parameters invocation-plan))))
    (is (= (:inputs semantic) (mapv :program-value (:bindings invocation-plan))))
    (is (= 1 (count (filter invocation/shape-projection? (:steps invocation-plan)))))
    (is (= 1 (count (filter invocation/buffer-clone? (:steps invocation-plan)))))
    (is (= 4 (count (filter invocation/buffer-allocation? (:steps invocation-plan)))))
    (is (= 4 (count (filter invocation/value-alias? (:steps invocation-plan)))))
    (is (= 3 (count (filter invocation/scalar-compute? (:steps invocation-plan)))))
    (is (every? #(not (contains? % :expression)) (:steps invocation-plan)))
    (is (= (set (:inputs semantic))
           (set (concat (keys (:program-buffers materialized))
                        (keys (:program-scalars materialized))))))
    (is (= 0 (get-in materialized [:attributes :driver-allocations])))
    (let [alias (first (filter invocation/value-alias? (:steps invocation-plan)))]
      (is (identical? (get-in materialized [:values (:id alias)])
                      (get-in materialized [:values (:source alias)]))))
    (let [public-nsteps (:id (first (filter #(= 'nsteps (:symbol %))
                                            (:parameters invocation-plan))))
          narrowed (first (filter #(some (fn [operand]
                                           (= public-nsteps (:value operand)))
                                         (:operands %))
                                  (:steps invocation-plan)))
          bound-nsteps (some (fn [{:keys [invocation-value]}]
                               (when (= (:id narrowed) invocation-value)
                                 invocation-value))
                             (:bindings invocation-plan))]
      (is (invocation/scalar-compute? narrowed))
      (is (not= public-nsteps (:id narrowed)))
      (is (= public-nsteps (-> narrowed :operands first :value)))
      (is (= (:id narrowed) bound-nsteps)))
    (is (= :scheduled-parallel (:dialect scheduled)))
    (is (= 3 (count (:equations scheduled)))
        "the generic fixpoint, host inv-n, and reduction with a final scalar epilogue represent the RK4 loss")
    (is (= 8 (count (get-in loop-operation [:graph :nodes])))
        "the loop body is the ordinary stencil/map schedule, not an RK4 primitive")
    (is (= :opencl-parallel (get-in emitted [:program :dialect])))
    (is (= 10 (count (:kernels emitted))))
    (is (= {:structured-loops-emitted 1
            :typed-equations-emitted 1
            :host-scalar-equations 1}
           (:stats emitted)))
    (let [[partial final] (take-last 2 (:kernels emitted))
          inv-n? #(and (symbol? %) (.startsWith (name %) "inv-n_"))]
      (is (= [:kernel-body :kernel-body]
             (mapv #(get-in % [:attributes :emission-route]) [partial final])))
      (is (not-any? inv-n? (:arguments partial))
          "the pre-combine phase must not apply or even bind the completed-result transform")
      (is (some inv-n? (:arguments final))
          "only the final reduction phase consumes the scalar epilogue input"))
    (is (= ["StructuredLoopCall" "EvaluatedHostEquation" "EmittedEquationCall"]
           (mapv #(-> % class .getSimpleName) (:steps call)))
        "the real emitted workload must cross the checked runtime-call boundary")
    (is (= (set (:outputs emitted-program)) (set (keys (:outputs call)))))
    (let [sources (program-call/buffer-identities call)
          mapping (zipmap sources (mapv #(vector :rk4-storage %)
                                        (range (count sources))))
          remapped (program-call/map-buffers call mapping)]
      (is (every? (set (vals mapping)) (vals (:outputs remapped)))
          "the rank-zero GPU reduction result remains resident storage during remapping"))))

(defn- prepared-mixed-call
  ([trip-count]
   (prepared-mixed-call trip-count (mixed-source-without-induction)))
  ([trip-count source]
   (let [initial (av/tensor {:dtype :double :shape ['extent]
                             :representation {:kind :plain}})
         options {:dtype :double
                  :target-device :ocl:0
                  :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                  :scalar-types {'steps :long}}
         scheduled (:form (pipeline/schedule-parallel-form source options))
         emitted (:program (program-opencl/emit-program
                            (assoc scheduled :source ::must-not-be-inspected) options))
         loop-equation (first (:equations emitted))
         carried (first (control/carried (:algorithm loop-equation)))
         initial (:initial carried)
         loop-output (:output carried)
         result (first (:outputs emitted))
         scalar (fn [id value]
                  {:type (get-in emitted [:values id :dtype]) :value value})]
     {:call
      (program-call/make
       emitted
       {initial :initial loop-output :loop-output result :suffix-output}
       {'steps (scalar 'steps trip-count)
        'n (scalar 'n 64)}
       (if (> trip-count 1) {loop-output :carry-scratch} {})
       nil)
      :loop-output loop-output
      :result result})))

(deftest emitted-program-stages-once-before-running-resident-equations
  (let [{:keys [call loop-output result]} (prepared-mixed-call 3)
        events (atom [])
        outputs
        (program-runtime/run-with!
         call
         {:bind! (fn [key _graph buffers scalars]
                   (let [handle {:key key :buffers buffers :scalars scalars}]
                     (swap! events conj [:bind handle])
                     handle))
          :run! (fn [handle] (swap! events conj [:run handle]))
          :release! (fn [handle] (swap! events conj [:release handle]))})
        bindings (mapv (comp :buffers second) (filter #(= :bind (first %)) @events))]
    (is (= {result :suffix-output} outputs))
    (is (= 4 (count bindings)))
    (is (= [:bind :bind :bind :bind :run :run :run :run
            :release :release :release :release]
           (mapv first @events))
        "all target graphs are staged before the first launch")
    (is (= :loop-output (get (last bindings) loop-output)))
    (is (= :suffix-output (get (last bindings) result)))
    (is (= :stage-once-host-repetition (get-in call [:attributes :execution])))
    (is (false? (get-in call [:attributes :source-inspected])))))

(deftest emitted-program-buffer-remapping-is-total-and-alias-stable
  (let [{:keys [call]} (prepared-mixed-call 3)
        sources (vec (distinct (concat (vals (:buffers call))
                                       (vals (:loop-scratch call)))))
        mapping (zipmap sources (mapv #(vector :link-value %) (range (count sources))))
        remapped (program-call/map-buffers call mapping)
        loop-step (first (:steps remapped))]
    (is (= (set (vals mapping))
           (set (concat (vals (:buffers remapped))
                        (vals (:loop-scratch remapped))))))
    (is (= (:scalar-values call) (:scalar-values remapped)))
    (is (= (:program call) (:program remapped)))
    (doseq [iteration (range 3)]
      (is (every? (set (vals mapping))
                  (vals (:buffers (loop-call/iteration-binding loop-step iteration))))))
    (is (= :emitted-program-buffer-remap-missing
           (:reason
            (ex-data
             (try (program-call/map-buffers call {})
                  (catch clojure.lang.ExceptionInfo exception exception))))))
    (is (= :emitted-program-buffer-remap-collision
           (:reason
            (ex-data
             (try (program-call/map-buffers call (constantly :same-storage))
                  (catch clojure.lang.ExceptionInfo exception exception))))))))

(deftest emitted-program-call-is-a-native-link-plan-instance
  (let [{:keys [call]} (prepared-mixed-call 3)
        sources (program-call/buffer-identities call)
        mapping (zipmap sources (mapv #(vector :program-value %)
                                      (range (count sources))))
        remapped (program-call/map-buffers call mapping)
        source-contracts
        (reduce (fn [contracts [compiler-value source]]
                  (assoc contracts source (get-in call [:program :values compiler-value])))
                {} (program-call/buffer-bindings call))
        nodes
        (mapv (fn [[source value-id]]
                (let [abstract (get source-contracts source)]
                  (link/node {:id value-id :dtype (:dtype abstract) :shape [64]
                              :device :ocl:0 :role :state})))
              mapping)
        values
        (mapv (fn [[source value-id]]
                (link/value {:id value-id :abstract (get source-contracts source)
                             :leaves [{:name :value :node value-id}]}))
              mapping)
        instance (link/program-instance {:id :rk4-call :call remapped})
        output-id (-> remapped :outputs vals first)
        plan (link/make {:id :linked-rk4 :target :ocl:0
                         :nodes nodes :values values
                         :instances [instance] :outputs [output-id]})]
    (is (link/program-link-instance? instance))
    (is (link/link-plan? plan))
    (is (= #{:state} (set (vals (link/instance-roles plan instance)))))
    (is (every? #(= 1 (count (:leaves %))) (vals (:values plan))))
    (is (false? (get-in remapped [:attributes :source-inspected])))
    (is (= :program-link-graph-range
           (reason-of
            #(link/make
              {:id :undersized-rk4 :target :ocl:0
               :nodes (mapv (fn [node]
                              (link/node {:id (:id node) :dtype (get-in node [:view :dtype])
                                          :shape [1] :device :ocl:0 :role :state}))
                            nodes)
               :values values :instances [instance] :outputs [output-id]}))))))

(deftest scheduled-loop-consumers-read-physical-result-storage
  (let [graph (-> (prepared-mixed-call 1) :call :steps first :graph)
        [producer consumer] (:nodes graph)]
    (is (= '[rstr_loop_carry_4] (mapv :id (:inputs graph))))
    (is (= '[scratch] (mapv :id (:temporaries graph))))
    (is (some #(and (= 'scratch (:buffer %)) (= :read (:access %))) (:uses consumer)))
    (is (= [(:id producer)] (:dependencies consumer)))
    (is (not-any? #(= 'rstr_loop_value_0 (:buffer %)) (:uses consumer)))))

(deftest structured-loop-staging-is-bounded-by-carry-rotation-not-trip-count
  (let [{:keys [call]} (prepared-mixed-call 9)
        events (atom [])]
    (program-runtime/run-with!
     call
     {:bind! (fn [key _graph buffers _scalars]
               (let [handle {:key key :buffers buffers}]
                 (swap! events conj [:bind handle])
                 handle))
      :run! #(swap! events conj [:run %])
      :release! #(swap! events conj [:release %])})
    (let [bindings (filter #(= :bind (first %)) @events)
          runs (filter #(= :run (first %)) @events)
          releases (filter #(= :release (first %)) @events)]
      (is (= 4 (count bindings))
          "one preserved-input prologue, two carry parities, and one suffix are prepared")
      (is (= 10 (count runs)) "nine loop launches replay three handles before the suffix")
      (is (= 4 (count releases)))
      (is (= 4 (count (distinct (map (comp :key second) bindings))))))))

(deftest structured-loop-preparation-stays-constant-for-huge-trip-counts
  (let [call (:call (prepared-mixed-call 1000000000))]
    (is (= 4 (count (program-runtime/staging-plan call :execution)))
        "one billion iterations still prepare three carry variants and one suffix")
    (let [events (atom [])
          launches (atom 0)]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"stop streamed replay"
           (program-runtime/run-with!
            call
            {:bind! (fn [key & _]
                      (swap! events conj [:bind key])
                      key)
             :run! (fn [handle]
                     (swap! events conj [:run handle])
                     (when (= 4 (swap! launches inc))
                       (throw (ex-info "stop streamed replay" {}))))
             :release! (fn [handle]
                         (swap! events conj [:release handle]))})))
      (is (= 4 (count (filter #(= :bind (first %)) @events))))
      (is (= 4 (count (filter #(= :run (first %)) @events))))
      (is (= 4 (count (filter #(= :release (first %)) @events)))))))

(deftest stage-once-declines-a-changing-induction-scalar
  (let [call (:call (prepared-mixed-call 3 (mixed-source)))]
    (try
      (program-runtime/staging-plan call :execution)
      (is false "an iteration-varying ABI value cannot be frozen into a prepared graph")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :parallel-program-dynamic-loop-binding
               (:reason (ex-data exception))))
        (is (= :bounded-iteration-execution (:fallback (ex-data exception))))))))

(deftest zero-trip-program-feeds-the-initial-carry-to-its-suffix
  (let [{:keys [call loop-output result]} (prepared-mixed-call 0)
        plan (program-runtime/staging-plan call :execution)]
    (is (= 1 (count plan)))
    (is (= :initial (get-in plan [0 :buffers loop-output])))
    (is (= :suffix-output (get-in plan [0 :buffers result])))))

(deftest whole-program-staging-is-transactional
  (let [call (:call (prepared-mixed-call 3))
        events (atom [])
        attempts (atom 0)]
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"staging failed"
         (program-runtime/run-with!
          call
          {:bind! (fn [key & _]
                    (if (= 3 (swap! attempts inc))
                      (throw (ex-info "staging failed" {}))
                      (let [handle [:handle key]]
                        (swap! events conj [:bind handle])
                        handle)))
           :run! (fn [handle] (swap! events conj [:run handle]))
           :release! (fn [handle] (swap! events conj [:release handle]))})))
    (is (= [:bind :bind :release :release] (mapv first @events)))
    (is (empty? (filter #(= :run (first %)) @events)))
    (is (= (reverse (map second (take 2 @events)))
           (map second (drop 2 @events))))))

(deftest effect-free-host-scalars-are-evaluated-before-device-staging
  (let [initial (av/tensor {:dtype :double :shape ['extent]
                            :representation {:kind :plain}})
        options {:dtype :double
                 :target-device :ocl:0
                 :values {'u0 initial 'steps (av/tensor {:dtype :long :shape []})}
                 :scalar-types {'steps :long}}
        scheduled (:form (pipeline/schedule-parallel-form
                          (mixed-source-without-induction) options))
        base (:program (program-opencl/emit-program
                        (assoc scheduled :source ::must-not-be-inspected) options))
        loop-equation (first (:equations base))
        answer-value (av/tensor {:dtype :long :shape []})
        host-algorithm
        (soac/make
         (soac/default-program-facts
          {:values {'steps (get-in base [:values 'steps]) 'answer answer-value}
           :inputs '[steps]
           :equations {'host-scalar (soac/default-equation-facts)}})
         [(list '= 'host-scalar '[answer]
                (list 'scalar {:dtypes [:long]} '[steps]
                      (soac/lambda-form '[value] '[(unchecked-inc value)])))]
         '[answer])
        host-equation
        (program/->ProgramEquation
         [:host-scalar] [:test :host-scalar] nil '[steps] '[answer]
         host-algorithm [] #{} {:source :test} {:host-only true})
        emitted (-> base
                    (assoc :values (assoc (:values base) 'answer answer-value))
                    (assoc :equations [loop-equation host-equation])
                    (assoc :outputs '[answer]))
        carried (first (control/carried (:algorithm loop-equation)))
        initial-id (:initial carried)
        loop-output (:output carried)
        result 'answer
        scalar (fn [id value]
                 {:type (get-in emitted [:values id :dtype]) :value value})
        evaluations (atom [])
        call
        (program-call/make
         emitted
         {initial-id :initial loop-output :loop-output}
         {'steps (scalar 'steps 2) 'n (scalar 'n 64)}
         {loop-output :carry-scratch}
         (fn [equation {:keys [operands]}]
           (swap! evaluations conj [(:id equation) operands])
           {result (scalar result (inc (get-in operands ['steps :value])))}))]
    (is (true? (get-in host-equation [:attributes :host-only])))
    (is (= 1 (count @evaluations)))
    (is (= 3 (get-in call [:outputs result :value])))
    (is (= 2 (count (program-runtime/staging-plan call :execution))))
    (is (false? (get-in call [:attributes :source-inspected])))))
