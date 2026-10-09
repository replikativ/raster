(ns raster.gpu.compiled-program-composition-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.arrays :as arrays]
            [raster.core :refer [deftm]]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.dl.nn :as nn]
            [raster.dl.loss :as loss]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.pipeline :as pipeline]
            [raster.hardware-fixture :as hardware-fixture]
            [raster.runtime.hardware :as hardware]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as link]
            [raster.gpu.value :as value]
            [raster.par :as par]))

(deftm twice!
  [input :- (Array float) result :- (Array float) n :- Long] :- Void
  (par/map-void! i n
                 (arrays/aset result i (* 2.0 (arrays/aget input i)))))

(deftm update-state!
  [state :- (Array float) gradient :- (Array float) lr :- Double n :- Long] :- Void
  (par/map-void! i n
    (arrays/aset state i (- (arrays/aget state i) (* lr (arrays/aget gradient i))))))

(deftm linear-objective
  [weights :- (Array float) inputs :- (Array float) targets :- (Array float)
   rows :- Long width :- Long] :- Double
  (loss/mse-loss (nn/linear-nb inputs weights rows width 1) targets rows))

(deftest public-objective-cannot-expose-a-parallel-intermediate
  (if-not @opencl/opencl-available?
    (opencl/opencl-skip! "public scalar objective source-result coverage")
    (let [arguments [(float-array [0.25 -0.5])
                     (float-array [1 2 3 4 5 6])
                     (float-array [0.125 0.25 0.5]) 3 2]
          expected (apply linear-objective arguments)]
      (is (number? expected) "the independent JVM computation returns the declared scalar loss")
      (doseq [inline? [false true]]
        ;; Until the complete loss is lowered, decline before device allocation rather
        ;; than promoting its valid contraction island to the public scalar result.
        (with-redefs [gpu/alloc! (fn [& _] (throw (AssertionError. "unexpected allocation")))
                      link/instantiate! (fn [& _] (throw (AssertionError. "unexpected instantiation")))]
          (try
            (compiled/lower #'linear-objective arguments
                            {:compiler :equation-first :target :ocl:0
                             :dtype :float :inline? inline?})
            (is false "the loss continuation is not yet represented by typed equations")
            (catch clojure.lang.ExceptionInfo error
              (is (= :structured-control-source-result (:reason (ex-data error))))
              (is (= [:return] (:site (ex-data error)))))))))))

(defn- mutable-case [target]
  (let [state (float-array [2.0 4.0 6.0 8.0])
        options {:compiler :equation-first :target target :dtype :float}
        forward (compiled/lower #'twice! [state (float-array 4) 4]
                                (assoc options :outputs '[result]))
        update (compiled/lower #'update-state! [state (float-array 4) 0.125 4]
                               (assoc options :donate '[state]))]
    {:id :shared-training-state
     :components [{:id :forward :program forward} {:id :update :program update}]
     :connections [{:from [:forward :result] :to [:update :gradient]}]
     :mutable-shares [{:owner [:update :state] :borrowers [[:forward :input]]
                       :output [:update :state']}]
     :outputs [{:key :gradient :from [:forward :result]}
               {:key :weights' :from [:update :state']}]}))

(defn- run-mutable-case [target]
  (let [prepared (compiled/compose (mutable-case target))
        artifact (compiled/instantiate! prepared)]
    (try
      (is (not-any? #(= [:forward :input] (:key %)) (:in-tree prepared)))
      (is (= {[:update :state] :weights'} (:donated prepared)))
      (loop [iteration 0 state [2.0 4.0 6.0 8.0] previous nil]
        (when (< iteration 3)
          (let [expected-gradient (mapv #(* 2.0 %) state)
                expected-state (mapv #(* 0.75 %) state)
                outputs (artifact (if previous {[:update :state] (:weights' previous)} {}))]
            (when previous
              (is (not (value/live? (:weights' previous))))
              (is (not (value/live? (:gradient previous)))))
            (is (= expected-gradient (vec (value/->host (:gradient outputs)))))
            (is (= expected-state (vec (value/->host (:weights' outputs)))))
            (recur (inc iteration) expected-state outputs))))
      (let [lease (compiled/invoke-leased artifact {})
            outputs @lease
            expected (mapv #(* % 0.75 0.75 0.75 0.75) [2.0 4.0 6.0 8.0])]
        (try
          (is (= expected (vec (value/->host (:weights' outputs)))))
          (is (= :link-output-lease-active
                 (try (artifact {}) nil
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
          (is (= expected (vec (value/->host (:weights' outputs))))
              "lease rejection precedes state mutation")
          (finally (.close ^java.io.Closeable lease)))
        (is (not (value/live? (:weights' outputs)))))
      (finally (compiled/close! artifact)))))

(deftest mutable-composition-preflights-semantic-ownership-without-allocation
  (hardware-fixture/isolated
   (fn []
     (let [target :ze:mutable-preflight]
       (hardware/register-target-device!
        target {:name "Synthetic mutable composition target"
                :capabilities {:subgroup-sizes [16] :total-eus 32
                               :max-workgroup-size 1024 :shared-local-memory 65536}})
       (let [request (mutable-case target)
             state (:default (first (filter #(= :input (:key %))
                                            (get-in request [:components 0 :program :in-tree]))))
             prepare-reader (fn [values options]
                              (compiled/lower #'twice! [values (float-array 4) 4]
                                              (merge {:compiler :equation-first :target target
                                                      :dtype :float :outputs '[result]} options)))
             reason (fn [request]
                      (try (compiled/compose request) nil
                           (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))]
         (with-redefs [link/instantiate! (fn [& _] (throw (AssertionError. "unexpected allocation")))
                       gpu/alloc! (fn [& _] (throw (AssertionError. "unexpected allocation")))]
           (is (= :compiled-composition-donation (reason (assoc request :mutable-shares []))))
           (is (= :compiled-composition-donation
                  (reason (update request :mutable-shares conj (first (:mutable-shares request))))))
           (is (= :compiled-composition-mutable-owner
                  (reason (assoc-in request [:mutable-shares 0 :borrowers] [[:update :state]]))))
           (is (= :compiled-composition-mutable-owner
                  (reason (assoc-in request [:mutable-shares 0 :output] [:forward :result]))))
           (is (= :compiled-composition-mutable-owner
                  (reason (assoc request :outputs [(first (:outputs request))]))))
           (is (= :compiled-composition-mutable-owner
                  (reason (assoc-in request [:components 0 :program]
                                    (prepare-reader state {:constants '[input]})))))
           (is (= :link-composition-share-source
                  (reason (assoc-in request [:components 0 :program]
                                    (prepare-reader (aclone ^floats state) {})))))
           (is (= :link-composition-mutable-escape
                  (reason (-> request
                              (assoc-in [:components 0 :program]
                                        (prepare-reader state {:outputs '[result input]}))
                              (update :outputs conj {:key :old-state :from [:forward :input]})))))
           (let [prepared (compiled/compose request)]
             (is (compiled/prepared? prepared))
             (doseq [[input output] (:donated prepared)]
               (is (= 1 (count (filter #(= input (:key %)) (:in-tree prepared)))))
               (is (= 1 (count (filter #(= output (:key %)) (:out-tree prepared)))))))))))))

(deftest mutable-forward-update-composition-replays-current-state
  (if @opencl/opencl-available?
    (run-mutable-case :ocl:0)
    (opencl/opencl-skip! "mutable forward/update composition"))
  (if @gp/gpu-available?
    (run-mutable-case :ze:0)
    (gp/gpu-skip! "mutable forward/update composition on Level Zero")))

(defn- compose-case [target]
  (let [prepare #(compiled/lower #'twice! [(float-array 4) (float-array 4) 4]
                                 {:compiler :equation-first :target target
                                  :dtype :float :outputs '[result]
                                  :on-non-resident :throw})
        first-program (prepare)
        second-program (prepare)
        prepared (compiled/compose
                  {:id :two-equation-programs
                   :components [{:id :first :program first-program}
                                {:id :second :program second-program}]
                   :connections [{:from [:first :result] :to [:second :input]}]
                   :outputs [{:key :result :from [:second :result]}]})]
    {:prepared prepared :first first-program :second second-program}))

(defn- run-case [target]
  (let [{:keys [prepared first second]} (compose-case target)
        plan (compiled/plan prepared)
        mapping (get-in prepared [:lowering :certificate :node-mapping])
        intermediate (get mapping [:first (get-in first [:out-tree 0 :node])])
        artifact (compiled/instantiate! prepared)]
    (try
      (is (= 2 (count (:instances plan))))
      (is (= intermediate
             (get mapping [:second (get-in second [:in-tree 0 :node])])))
      (is (contains? (:nodes plan) intermediate))
      (is (nil? (get-in plan [:nodes intermediate :source]))
          "the intermediate is not uploaded from or copied through the host")
      (is (= (set (map :id (:instances plan)))
             (set (map #(get-in % [:source :instance])
                       (:per-replay (link/execution-order (:executable artifact)))))))
      (is (some? (get-in artifact [:executable :graph-key]))
          "straight-line emitted-only composition records one command graph")
      (is (every? (comp nil? :runtime-graph)
                  (vals (:kernel-graphs @(:session (:executable artifact)))))
          "the enclosing replay does not record each emitted subgraph again")
      (doseq [input [[1.0 2.0 3.0 4.0] [5.0 6.0 7.0 8.0]]]
        (let [output (artifact {[:first :input] (float-array input)})]
          (is (= (mapv #(* 4.0 %) input)
                 (vec (value/->host (:result output)))))))
      (finally (compiled/close! artifact)))))

(deftest opencl-two-equation-programs-share-a-resident-intermediate
  (if @opencl/opencl-available?
    (run-case :ocl:0)
    (opencl/opencl-skip! "two composed equation programs")))

(deftest level-zero-two-equation-programs-share-a-resident-intermediate
  (if @gp/gpu-available?
    (run-case :ze:0)
    (gp/gpu-skip! "two composed equation programs on Level Zero")))

(defn- run-mixed-graph-case [target]
  (let [{:keys [prepared first second]} (compose-case target)
        original (compiled/plan prepared)
        second-instance (clojure.core/second (:instances original))
        call (:call second-instance)
        step (clojure.core/first (:steps call))
        graph (:graph step)
        external (set (map :id (concat (:inputs graph) (:outputs graph))))
        graph-instance (link-plan/graph-instance
                        {:id (:id second-instance) :graph graph
                         :bindings (select-keys (:buffers step) external)
                         :scalar-values (merge (:scalar-values call) (:scalar-values step))})
        plan (link-plan/make
              (assoc original :instances [(clojure.core/first (:instances original))
                                          graph-instance]))
        mapping (get-in prepared [:lowering :certificate :node-mapping])
        input (get mapping [:first (get-in first [:in-tree 0 :node])])
        output (get mapping [:second (get-in second [:out-tree 0 :node])])
        intermediate (get mapping [:first (get-in first [:out-tree 0 :node])])
        executable (link/instantiate! plan)]
    (try
      (is (nil? (get-in plan [:nodes intermediate :source])))
      (is (= [false true]
             (mapv link-plan/graph-link-instance? (:instances plan))))
      (is (= (mapv :id (:instances plan))
             (mapv #(get-in % [:source :instance])
                   (:per-replay (link/execution-order executable)))))
      (doseq [values [[1.0 2.0 3.0 4.0] [5.0 6.0 7.0 8.0]]]
        (link/upload! executable input (float-array values))
        (link/run! executable)
        (is (= (mapv #(* 4.0 %) values)
               (vec (link/download executable output)))))
      (finally (link/close! executable)))))

(deftest opencl-program-and-direct-graph-share-a-resident-intermediate
  (if @opencl/opencl-available?
    (run-mixed-graph-case :ocl:0)
    (opencl/opencl-skip! "program and direct graph composition")))

(deftest level-zero-program-and-direct-graph-share-a-resident-intermediate
  (if @gp/gpu-available?
    (run-mixed-graph-case :ze:0)
    (gp/gpu-skip! "program and direct graph composition on Level Zero")))

(defn- run-descriptor-and-graph-case [target]
  (let [{:keys [prepared first second]} (compose-case target)
        original (compiled/plan prepared)
        first-instance (clojure.core/first (:instances original))
        second-instance (clojure.core/second (:instances original))
        descriptor (pipeline/compile-gpu-program #'twice! target
                                                 :dtype :float :on-non-resident :nil)
        pointers (link-plan/descriptor-pointer-symbols descriptor)
        descriptor-instance
        (link-plan/instance
         {:id (:id first-instance) :descriptor descriptor
          :bindings (select-keys (get-in first-instance [:call :buffers]) pointers)
          :scalars {'n 4}})
        step (clojure.core/first (get-in second-instance [:call :steps]))
        graph (:graph step)
        external (set (map :id (concat (:inputs graph) (:outputs graph))))
        graph-instance
        (link-plan/graph-instance
         {:id (:id second-instance) :graph graph
          :bindings (select-keys (:buffers step) external)
          :scalar-values (merge (get-in second-instance [:call :scalar-values])
                                (:scalar-values step))})
        plan (link-plan/make
              (assoc original :instances [descriptor-instance graph-instance]))
        mapping (get-in prepared [:lowering :certificate :node-mapping])
        input (get mapping [:first (get-in first [:in-tree 0 :node])])
        output (get mapping [:second (get-in second [:out-tree 0 :node])])
        executable (link/instantiate! plan {:profile? true})]
    (try
      (is (some? (:graph-key executable)) "mixed LinkPlan records one replay graph")
      (is (some? (:prepared-program executable)))
      (is (= 2 (count (link/execution-info executable))))
      (is (= [{:instance (:id descriptor-instance) :step 0}
              {:instance (:id graph-instance)}]
             (mapv :source (:per-replay (link/execution-order executable)))))
      (doseq [values [[1.0 2.0 3.0 4.0] [5.0 6.0 7.0 8.0]]]
        (link/upload! executable input (float-array values))
        (link/run! executable)
        (is (= (mapv #(* 4.0 %) values)
               (vec (link/download executable output)))))
      (is (= (mapv :source (:per-replay (link/execution-order executable)))
             (mapv :source (:profile (link/profile! executable)))))
      (finally (link/close! executable)))))

(deftest opencl-descriptor-and-direct-graph-one-linked-replay
  (if @opencl/opencl-available?
    (run-descriptor-and-graph-case :ocl:0)
    (opencl/opencl-skip! "descriptor and direct graph linked replay")))

(deftest level-zero-descriptor-and-direct-graph-one-linked-replay
  (if @gp/gpu-available?
    (run-descriptor-and-graph-case :ze:0)
    (gp/gpu-skip! "descriptor and direct graph linked replay on Level Zero")))

(deftest failed-second-bind-cleans-an-attached-session
  (if-not @opencl/opencl-available?
    (opencl/opencl-skip! "composed program transactional binding")
    (let [prepared (:prepared (compose-case :ocl:0))
          original gpu/bind-kernel-graph!
          binds (atom 0)]
      (let [session (gpu/make-session :ocl:0)]
        (try
          (let [before (select-keys @session [:buffers :kernel-graphs])]
            (with-redefs [gpu/bind-kernel-graph!
                          (fn [& arguments]
                            (if (= 2 (swap! binds inc))
                              (throw (ex-info "second bind failed" {:reason :second-bind}))
                              (apply original arguments)))]
              (is (= :second-bind
                     (try (compiled/instantiate! prepared {:session session})
                          (catch clojure.lang.ExceptionInfo error
                            (:reason (ex-data error)))))))
            (is (= 2 @binds))
            (is (= before (select-keys @session [:buffers :kernel-graphs]))))
          (finally (gpu/close-session! session)))))))
