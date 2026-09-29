(ns raster.gpu.compiled-program-composition-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.arrays :as arrays]
            [raster.core :refer [deftm]]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.compiler.ir.link-plan :as link-plan]
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
