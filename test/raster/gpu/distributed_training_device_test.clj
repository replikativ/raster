(ns raster.gpu.distributed-training-device-test
  "Actual AD → collective → SGD compiler fixture using checked LinkPlan rebinding,
   not a released trainer API. Co-location is not fabric evidence."
  (:require [clojure.test :refer [deftest is]]
            [clojure.set :as set]
            [raster.core :refer [deftm broadcast]]
            [raster.numeric :as numeric]
            [raster.ad.reverse :as reverse]
            [raster.dl.nn :as nn]
            [raster.dl.loss :as loss]
            [raster.dl.optim :as optim]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.distributed-refinement-test :as fixture]
            [raster.compiler.ir.link-composition :as composition]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.passes.parallel.collective-combine :as arithmetic]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.distributed :as runtime]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]))

(deftm local-loss
  [theta :- (Array float) x :- (Array float) y :- (Array float)
   rows :- Long width :- Long] :- Double
  (loss/mse-loss (nn/linear-nb x theta rows width 1) y rows))

(deftm local-gradient-sum
  [theta :- (Array float) x :- (Array float) y :- (Array float)
   rows :- Long width :- Long] :- (Array float)
  (let [vg ((reverse/value+grad (var raster.gpu.distributed-training-device-test/local-loss) :wrt [0])
            theta x y rows width)
        gradient (clojure.core/nth vg 1)]
    ;; Each local loss is a mean. Weight its gradient by its actual batch count before all-reduce.
    (broadcast [gradient] (numeric/* gradient (float rows)))))

(defn- batches [row-counts]
  (mapv (fn [rows worker]
          {:rows rows
           :x (float-array (for [i (range rows) j (range 17)]
                             (* 0.125 (- (mod (+ j (* i 3) worker) 7) 3))))
           :y (float-array (map #(* 0.25 (- % worker)) (range rows)))})
        row-counts (range)))

(defn- initial-theta []
  (float-array (map #(* 0.03125 (- % 8)) (range 17))))

(defn- place-lowering
  ([id f args device] (place-lowering id f args device :program))
  ([id f args device component]
  (let [prepared (compiled/lower f args {:compiler :equation-first :target device :dtype :float
                                        :schedule {:precision :f32-scalar}})
        lowering (:lowering prepared)]
    (:plan (composition/compose
            {:id id :components [{:id component :lowering lowering}]
             :outputs (mapv #(vector component %) (link/output-value-ids (:plan lowering)))})))))

(defn- value-node [plan value-id]
  (get-in plan [:nodes (get-in plan [:values value-id :leaves 0 :node])]))

(defn- initialized-value [plan source]
  (let [matches (for [[id _] (:values plan)
                     :when (identical? source (:source (value-node plan id)))] id)]
    (assert (= 1 (count matches)) "fixture input must have one exact initialized boundary")
    (first matches)))

(defn- training-request
  ([device row-counts] (training-request device row-counts {}))
  ([device row-counts {:keys [scope theta-by-worker parameter-views predecessors]}]
  (let [n (count row-counts)
        scoped #(if scope [scope %] %)
        [refinement topology original-inputs costs] (fixture/projection-inputs n scope)
        workers (get-in refinement [:group :devices])
        inputs (update-vals original-inputs
                            #(if-let [previous (get predecessors (:device %))]
                               (assoc % :dependencies [previous]) %))
        theta-by-worker (or theta-by-worker (zipmap workers (repeatedly n initial-theta)))
        facts (distributed/refinement-facts refinement)
        options {:target-device device :target-descriptor (hardware/descriptor-for device)}
        data (batches row-counts)
        lr (float (/ 0.125 (reduce + row-counts)))
        producers (into {} (map (fn [worker {:keys [rows x y]}]
                                  (let [theta (get theta-by-worker worker)
                                        local (place-lowering (scoped [worker :gradient]) #'local-gradient-sum
                                                              [theta x y rows 17] device
                                                              (scoped :program))
                                        parameter (initialized-value local theta)
                                        node (value-node local parameter)
                                        local (if-let [view (get parameter-views worker)]
                                                (-> local
                                                    (assoc-in [:nodes (:id node) :view]
                                                              (assoc view :id (get-in node [:view :id])))
                                                    link/validate!) local)]
                                    [worker local])) workers data))
        producer-parameters (into {} (for [[worker local] producers]
                                      [worker (initialized-value local (get theta-by-worker worker))]))
        producer-outputs (update-vals producers #(first (link/output-value-ids %)))
        nodes (into {} (for [[ssa {:keys [device]}] (:values facts)]
                        [ssa (if-let [worker (get (:inputs refinement) ssa)]
                               (value-node (get producers worker) (get producer-outputs worker))
                               (link/node {:id ssa :device (:target-device options)
                                           :dtype :float :shape [17] :role :input}))]))
        combines (into {} (for [{:keys [id kind left right]} (:nodes refinement) :when (= :combine kind)]
                            [id (arithmetic/bind-local
                                 (get-in refinement [:operation :reduction]) 17 options
                                 {:id id :nodes {:left (get nodes left) :right (get nodes right)
                                                 :result (get nodes id)}})]))
        consumers
        (into {} (for [worker workers
                       :let [gradient (float-array 17)
                             theta (if scope (get theta-by-worker worker) (initial-theta))
                             local (place-lowering (scoped [worker :update]) #'optim/sgd-step!
                                                   [theta gradient 17 lr] device (scoped :program))
                             parameter (initialized-value local theta)
                             parameter-node (value-node local parameter)
                             parameter-view (:view (value-node (get producers worker)
                                                               (get producer-parameters worker)))
                             local (if scope
                                     (-> local
                                         (assoc-in [:nodes (:id parameter-node) :view]
                                                   (assoc parameter-view :id (get-in parameter-node [:view :id])))
                                         link/validate!) local)
                             input (initialized-value local gradient)
                             node (value-node local input)
                             canonical (get nodes (get (:outputs refinement) worker))
                             ;; Rebind ordinary, unsealed LinkPlan storage, then validate it again.
                             ;; The copy/combine writes this exact allocation before SGD consumes it.
                             local (-> local
                                       (assoc-in [:nodes (:id node) :source] nil)
                                       (assoc-in [:nodes (:id node) :view]
                                                 (assoc (:view canonical) :id (get-in node [:view :id])))
                                       link/validate!)]]
                   [worker {:plan local :gradient input :parameter parameter}]))
        locals (merge (into {} (map (fn [[worker local]] [(scoped [worker :produce]) local])) producers)
                      (update-vals combines :link-plan)
                      (into {} (map (fn [[worker {:keys [plan]}]] [(scoped [worker :update]) plan])) consumers))
        node-to-ssa (into {} (map (fn [[ssa node]] [(:id node) ssa])) nodes)
        bindings
        (into {} (for [[step local] locals]
                   [step (into {} (for [[local-id _] (:values local)
                                       :let [owner (some (fn [worker]
                                                           (when (contains? #{(scoped [worker :produce])
                                                                              (scoped [worker :update])} step)
                                                             worker)) workers)]]
                                    [local-id
                                     (or (when (and scope owner
                                                    (= local-id (if (= step (scoped [owner :produce]))
                                                                  (get producer-parameters owner)
                                                                  (get-in consumers [owner :parameter]))))
                                           [owner :parameters])
                                         (when (= step (scoped [owner :produce]))
                                           (when (= local-id (get producer-outputs owner))
                                             (some (fn [[ssa worker]] (when (= worker owner) ssa))
                                                   (:inputs refinement))))
                                         (when (= step (scoped [owner :update]))
                                           (when (= local-id (get-in consumers [owner :gradient]))
                                             (get-in refinement [:outputs owner])))
                                         (get node-to-ssa (:id (value-node local local-id)))
                                         [step local-id])]))]))
        projected (distributed/project-refinement refinement topology inputs costs)
        storage (into {} (for [[ssa _] (:values facts)]
                           [ssa (cond
                                  (contains? inputs ssa)
                                  {:step (:id (get inputs ssa))
                                   :local-value (get producer-outputs (get-in refinement [:inputs ssa]))}
                                  (contains? combines ssa) {:step ssa :local-value ssa}
                                  :else
                                  (or (some (fn [{:keys [id left right]}]
                                              (when (or (= ssa left) (= ssa right))
                                                {:step id :local-value (:id (get nodes ssa))}))
                                            (filter #(= :combine (:kind %)) (:nodes refinement)))
                                      (some (fn [[worker output]]
                                              (when (= ssa output)
                                                {:step (scoped [worker :update])
                                                 :local-value (get-in consumers [worker :gradient])}))
                                            (:outputs refinement))))]))
        updates (mapv (fn [worker completion]
                        (distributed/compute-step {:id (scoped [worker :update]) :device worker :duration-ns 1
                                                   :dependencies [completion]}))
                      workers (:completions projected))
        step-map (into {} (map (juxt :id identity))
                       (concat (vals inputs) (:steps projected) updates))
        additional
        (into {} (for [[step local] locals [local-id value] (:values local)
                       :let [global-id (get-in bindings [step local-id])]
                       :when (not (contains? (:values projected) global-id))]
                   [global-id {:abstract (assoc (:abstract value) :sharding
                                               {:kind :replicated :devices [(:device (get step-map step))]})
                               :worker (:device (get step-map step))}]))]
    {:id :differentiated-all-reduce-sgd :mesh (distributed/mesh [{:name :workers :size n}] workers)
      :topology topology :refinement refinement :input-producers inputs :combine-costs costs
      :storage storage :combines (update-vals combines :emitted)
      :values (update-vals additional :abstract)
      :shards (into {} (for [[id {:keys [abstract worker]}] additional]
                        [id [(distributed/shard {:id id :value id :device worker
                                                :offsets (vec (repeat (count (:shape abstract)) 0))
                                                :shape (:shape abstract) :ownership :replica})]]))
      :device-plans
      (into {} (for [worker workers :let [owned (filter #(= worker (:device (get step-map (key %)))) locals)]]
                 [worker {:target device
                          :entries (into {} (map (fn [[id local]] [id {:link-plan local}])) owned)
                          :steps (into {} (for [[id _] owned]
                                            [id {:entry id :bindings
                                                 (update-vals (get bindings id) #(hash-map :value % :shard %))}]))}]))
      :steps updates :outputs (mapv :id updates)})))

(defn- training-plan [device row-counts]
  (distributed/refinement-plan (training-request device row-counts)))

(defn- multi-step-training-plan [device row-counts iterations]
  (let [workers (mapv #(keyword (str "worker-" %)) (range (count row-counts)))
        theta-by-worker (zipmap workers (repeatedly (count workers) initial-theta))
        requests
        (:requests
         (reduce
          (fn [{:keys [requests parameter-views predecessors]} epoch]
            (let [scope [:epoch epoch]
                  request (training-request device row-counts
                                            {:scope scope :theta-by-worker theta-by-worker
                                             :parameter-views parameter-views :predecessors predecessors})
                  views (into {} (for [worker workers
                                       :let [local (get-in request [:device-plans worker :entries
                                                                   [scope [worker :produce]] :link-plan])
                                             parameter (initialized-value local (get theta-by-worker worker))]]
                                   [worker (:view (value-node local parameter))]))]
              {:requests (conj requests request) :parameter-views views
               :predecessors (zipmap workers (:outputs request))}))
          {:requests [] :parameter-views {} :predecessors {}} (range iterations)))
        context (assoc (select-keys (last requests) [:mesh :topology :outputs])
                       :id :finite-differentiated-training)]
    (distributed/compose-refinement-plans context requests)))

(defn- analytic-gradient-sum [{:keys [rows x y]}]
  (let [theta (initial-theta)
        predictions (mapv (fn [row]
                            (reduce + (for [j (range 17)]
                                        (* (double (aget theta j))
                                           (double (aget x (+ (* row 17) j)))))))
                          (range rows))]
    (mapv (fn [j]
            (reduce + (for [row (range rows)]
                        (* 2.0 (- (nth predictions row) (double (aget y row)))
                           (double (aget x (+ (* row 17) j)))))))
          (range 17))))

(defn- max-error [expected actual]
  (reduce max 0.0 (map #(Math/abs (- (double %1) (double %2))) expected actual)))

(defn- expected-update [row-counts]
  (let [contributions (mapv analytic-gradient-sum (batches row-counts))
        ;; Round each contribution's storage and every declared left-associated FP32 tree node.
        sum (apply mapv (fn [& xs] (reduce #(float (+ (double %1) (double %2))) (map float xs)))
                   contributions)
        lr (float (/ 0.125 (reduce + row-counts)))]
    (mapv (fn [parameter gradient]
            (float (- (double parameter) (double (float (* (double lr) (double gradient)))))))
          (initial-theta) sum)))

(defn- rounded-gradient-sum
  "Independent scalar oracle with explicit FP32 product/accumulator/storage boundaries."
  [theta {:keys [rows x y]}]
  (let [multiply (fn [a b] (float (* (double a) (double b))))
        add (fn [a b] (float (+ (double a) (double b))))
        predictions (mapv (fn [row]
                            (reduce add (float 0)
                                    (map #(multiply (aget ^floats theta %)
                                                    (aget ^floats x (+ (* row 17) %)))
                                         (range 17)))) (range rows))
        cotangents (mapv (fn [prediction target]
                          (float (/ (* 2.0 (- (double prediction) (double target))) rows)))
                        predictions y)]
    (mapv (fn [j]
            (multiply (float rows)
                      (reduce add (float 0)
                              (map #(multiply (nth cotangents %)
                                              (aget ^floats x (+ (* % 17) j)))
                                   (range rows)))))
          (range 17))))

(defn- expected-updates [row-counts iterations]
  (let [data (batches row-counts)
        lr (float (/ 0.125 (reduce + row-counts)))]
    (vec (rest
          (reductions
           (fn [theta _]
             (let [contributions (mapv #(rounded-gradient-sum theta %) data)
                   gradient (apply mapv (fn [& xs]
                                         (reduce #(float (+ (double %1) (double %2))) xs))
                                   contributions)]
               (float-array
                (map (fn [parameter grad]
                       (float (- (double parameter)
                                 (double (float (* (double lr) (double grad))))))) theta gradient))))
           (initial-theta) (range iterations))))))

(deftest repeated-local-ad-matches-rounded-independent-oracle
  (doseq [row-counts [[1 3] [1 2 4]]
          theta (cons (initial-theta) (expected-updates row-counts 2))
          {:keys [rows x y] :as batch} (batches row-counts)]
    (is (< (max-error (rounded-gradient-sum theta batch)
                     (local-gradient-sum theta x y rows 17)) 1.0e-7))))

(deftest finite-training-composition-retains-parameters-and-orders-updates
  (let [device :ocl:analytic
        descriptor {:device-id device :device-type :gpu :backend :ocl
                    :subgroup-dialect :opencl-portable :max-workgroup-size 256}]
    (with-redefs [hardware/descriptor-for (constantly descriptor)
                  gpu/make-session (fn [& _] (throw (ex-info "assembly opened a session" {})))
                  gpu/alloc! (fn [& _] (throw (ex-info "assembly allocated device storage" {})))]
      (doseq [row-counts [[1 3] [1 2 4]]]
        (let [plan (multi-step-training-plan device row-counts 3)
              workers (get-in plan [:mesh :devices])
              parameter-allocations
              (into #{} (for [worker workers
                              :let [step [[:epoch 0] [worker :produce]]
                                    local (get-in plan [:device-plans worker :entries step :link-plan])]
                              [id binding] (get-in plan [:device-plans worker :steps step :bindings])
                              :when (= [worker :parameters] (:value binding))]
                          (get-in (value-node local id) [:view :allocation :id])))
              scratch-by-epoch
              (mapv (fn [epoch]
                      (set/difference
                       (into #{} (for [[_ worker-plan] (:device-plans plan)
                                       [[scope _] entry] (:entries worker-plan)
                                       :when (= [:epoch epoch] scope)
                                       [_ node] (get-in entry [:link-plan :nodes])
                                       :let [id (get-in node [:view :allocation :id])]
                                       :when id] id))
                       parameter-allocations)) (range 3))]
          (is (= 3 (count (:refinements plan))))
          (is (= (mapv #(vector [:epoch 2] [% :update]) workers) (:outputs plan)))
          (is (= (distributed/certify plan) (distributed/verify! (distributed/certify plan))))
          (is (= (count workers) (count parameter-allocations)))
          (is (every? seq scratch-by-epoch))
          (doseq [left (range 3) right (range (inc left) 3)]
            (is (empty? (set/intersection (nth scratch-by-epoch left) (nth scratch-by-epoch right)))
                "private scratch and immutable contribution storage are fresh across epochs"))
          (is (thrown? clojure.lang.ExceptionInfo
                       (distributed/check-readiness
                        (update plan :steps
                                #(mapv (fn [step]
                                         (if (= [[:epoch 1] [(first workers) :produce]] (:id step))
                                           (assoc step :dependencies []) step)) %))))
              "an epoch cannot read parameters unordered with respect to the preceding update")
          (doseq [worker workers]
            (let [parameters (for [epoch (range 3) kind [:produce :update]
                                   :let [step [[:epoch epoch] [worker kind]]
                                         local (get-in plan [:device-plans worker :entries step :link-plan])]
                                   [id binding] (get-in plan [:device-plans worker :steps step :bindings])
                                   :when (= [worker :parameters] (:value binding))]
                               (value-node local id))]
              (is (= 6 (count parameters)))
              (is (apply = (map #(dissoc (:view %) :id) parameters)))
              (is (every? #(identical? (:source (first parameters)) (:source %)) parameters)))
            (doseq [epoch [1 2]]
              (is (= [[[:epoch (dec epoch)] [worker :update]]]
                     (:dependencies (first (filter #(= [[:epoch epoch] [worker :produce]] (:id %))
                                                   (:steps plan)))))))))))))

(defn- check-multi-step-training! [device]
  (doseq [row-counts [[1 3] [1 2 4]]]
    (let [iterations 3
          plan (multi-step-training-plan device row-counts iterations)
          expected (last (expected-updates row-counts iterations))]
      (with-open [executable (runtime/instantiate! plan {:transport :resident-copy
                                                        :device-capacities {device 1048576}})]
        (runtime/run! executable)
        (let [session (get (:sessions executable) device)
              actual (mapv (fn [[_ values]]
                             (is (= 1 (count values)))
                             (let [result (float-array 17)]
                               (gpu/download-range! session (first (vals values)) result {:elements 17})
                               result)) (runtime/output-values executable))]
          (is (= (count row-counts) (count actual)))
          (is (> (max-error (first (expected-updates row-counts 1)) expected) 1.0e-3)
              "the oracle distinguishes three updates from a stale single-step replay")
          (doseq [result actual]
            (is (< (max-error expected result) 1.0e-7)))
          (doseq [result (rest actual)]
            (is (java.util.Arrays/equals ^floats (first actual) ^floats result)))
          (is (thrown? clojure.lang.ExceptionInfo (runtime/run! executable))
              "finite unrolling does not authorize replay with stale startup evidence"))))))

(deftest finite-multi-step-ad-all-reduce-sgd-on-local-devices
  (if @opencl/opencl-available?
    (check-multi-step-training! :ocl:0)
    (opencl/opencl-skip! "finite multi-step distributed training"))
  (if @ze/gpu-available?
    (check-multi-step-training! :ze:0)
    (ze/gpu-skip! "finite multi-step distributed training")))

(deftest unequal-batch-local-ad-producers-match-independent-analytic-gradients
  (doseq [rows [[1 3] [1 2 4]]
          {:keys [x y rows] :as batch} (batches rows)]
    (let [actual (local-gradient-sum (initial-theta) x y rows 17)]
      (is (= 17 (alength actual)))
      (is (< (max-error (analytic-gradient-sum batch) actual) 1.0e-7)))))

(deftest differentiated-training-assembly-certifies-without-device-resources
  (let [device :ocl:analytic
        descriptor {:device-id device :device-type :gpu :backend :ocl
                    :subgroup-dialect :opencl-portable :max-workgroup-size 256}]
    (with-redefs [hardware/descriptor-for (fn [target]
                                         (assert (= device target))
                                         descriptor)
                  gpu/make-session (fn [& _] (throw (ex-info "hardware-free assembly opened a session" {})))
                  gpu/alloc! (fn [& _] (throw (ex-info "hardware-free assembly allocated device storage" {})))
                  compiled/instantiate! (fn [& _] (throw (ex-info "hardware-free assembly realized a program" {})))]
      (doseq [row-counts [[1 3] [1 2 4]]]
        (let [n (count row-counts)
              plan (training-plan device row-counts)
              certified (distributed/certify plan)]
          (is (= certified (distributed/verify! certified)))
          (is (= n (count (filter #(= :produce (second (:id %))) (:steps plan)))))
          (is (= n (count (filter #(= :update (second (:id %))) (:steps plan)))))
          (is (= (* 2 (dec n)) (count (filter #(= :transfer (:kind %)) (:steps plan)))))
          (is (= (dec n) (count (filter #(= :combined (second (:id %))) (:steps plan)))))
          (is (= (mapv #(vector % :update) (get-in plan [:mesh :devices])) (:outputs plan)))
          (doseq [[worker ssa] (get-in plan [:refinements :all-reduce :refinement :outputs])
                  :let [local (get-in plan [:device-plans worker :entries [worker :update] :link-plan])
                        matches (for [[value binding]
                                      (get-in plan [:device-plans worker :steps [worker :update] :bindings])
                                      :when (= ssa (:value binding))] value)]]
            (is (= 1 (count matches)))
            (is (nil? (:source (value-node local (first matches))))
                "host gradient initialization cannot stand in for the actual collective producer")))))))

(defn- check-training! [device]
  (doseq [row-counts [[1 3] [1 2 4]]]
    (let [plan (training-plan device row-counts)
          expected (expected-update row-counts)]
      (distributed/verify! (distributed/certify plan))
      (is (= (count row-counts) (count (:outputs plan))))
      (is (> (max-error (initial-theta) expected) 1.0e-3)
          "fixture must distinguish an actual update from unchanged parameter replicas")
      (with-open [executable (runtime/instantiate! plan {:transport :resident-copy
                                                        :device-capacities {device 1048576}})]
        (runtime/run! executable)
        (let [outputs (runtime/output-values executable)
              session (get (:sessions executable) device)
              actual
              (mapv (fn [[_ values]]
                      (is (= 1 (count values)))
                      (let [result (float-array 17)]
                        (gpu/download-range! session (first (vals values)) result {:elements 17})
                        result)) outputs)]
          (is (= (count row-counts) (count actual)))
          (doseq [result actual]
            (is (< (max-error expected result) 1.0e-7)
                "all participants update from the globally sample-weighted gradient"))
          (doseq [result (rest actual)]
            (is (java.util.Arrays/equals ^floats (first actual) ^floats result)
                "all replicas consume the same reduced gradient and initial parameters")))))))

(defn- check-distributed-scalar-loss! [device]
  (let [{:keys [x y rows]} (first (batches [1]))
        args [(initial-theta) x y rows 17]
        expected (apply local-loss args)
        local (place-lowering :scalar-objective #'local-loss args device)
        output (first (link/output-value-ids local))
        accesses (link/value-accesses local)
        boundaries (into #{output}
                         (keep (fn [[id _]]
                                 (let [node (value-node local id)]
                                   (when (and (contains? accesses id)
                                              (or (contains? #{:input :state} (:role node))
                                                  (and (= :constant (:role node)) (nil? (:source node)))))
                                     id))))
                         (:values local))
        globals (into {} (map (fn [id]
                               [id (assoc (get-in local [:values id :abstract])
                                          :shape (if (= id output) []
                                                     (get-in local [:values id :abstract :shape]))
                                          :sharding {:kind :replicated :devices [:worker]})])) boundaries)
        plan (distributed/plan
              {:id :scalar-loss :mesh (distributed/mesh [{:name :workers :size 1}] [:worker])
               :topology (distributed/topology [(distributed/device {:id :worker
                                                                      :memory-capacity-bytes 1048576})] [])
               :values globals
               :shards (into {} (map (fn [[id value]]
                                      [id [(distributed/shard {:id id :value id :device :worker
                                                              :shape (:shape value)
                                                              :offsets (vec (repeat (count (:shape value)) 0))
                                                              :ownership :replica})]])) globals)
               :device-plans {:worker {:target device :entries {:loss {:link-plan local}}
                                      :steps {:loss {:entry :loss
                                                     :bindings (into {} (map (fn [id]
                                                                              [id (if (= id output)
                                                                                    {:local-shape []
                                                                                     :placements [{:kind :owned :value id
                                                                                                   :shard id :local-offsets []}]}
                                                                                    {:value id :shard id})])) boundaries)}}}}
               :steps [(distributed/compute-step {:id :loss :device :worker :duration-ns 1})]
               :outputs [:loss]})]
    (distributed/verify! (distributed/certify plan))
    (is (= [] (get-in (distributed/compute-bindings plan)
                      [:bindings :loss :values output :domain :shape])))
    (with-open [executable (runtime/instantiate! plan {:device-capacities {device 1048576}})]
      (runtime/run! executable)
      (let [actual (double-array 1)
            result (get-in (runtime/output-values executable) [:loss output])]
        (gpu/download-range! (get (:sessions executable) device) result actual {:elements 1})
        (is (= [expected] (vec actual))
            "a rank-zero distributed output is the complete public loss, not an intermediate array")))))

(deftest scalar-loss-as-an-explicit-rank-zero-distributed-output
  (if @opencl/opencl-available?
    (check-distributed-scalar-loss! :ocl:0)
    (opencl/opencl-skip! "rank-zero distributed scalar loss"))
  (if @ze/gpu-available?
    (check-distributed-scalar-loss! :ze:0)
    (ze/gpu-skip! "rank-zero distributed scalar loss")))

(deftest actual-ad-all-reduce-sgd-on-colocated-opencl-workers
  (if @opencl/opencl-available?
    (check-training! :ocl:0)
    (opencl/opencl-skip! "actual AD/all-reduce/SGD with unequal local batches")))

(deftest actual-ad-all-reduce-sgd-on-colocated-level-zero-workers
  (if @ze/gpu-available?
    (check-training! :ze:0)
    (ze/gpu-skip! "actual AD/all-reduce/SGD with unequal local batches")))
