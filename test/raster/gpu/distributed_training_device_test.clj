(ns raster.gpu.distributed-training-device-test
  "Actual AD → collective → SGD compiler fixture using checked LinkPlan rebinding,
   not a released trainer API. Co-location is not fabric evidence."
  (:require [clojure.test :refer [deftest is]]
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
        ;; The AD tuple's nth projection currently needs its declared primitive-array tag
        ;; retained explicitly before the JVM broadcast path; see compiler-consolidation.md.
        ^floats gradient (clojure.core/nth vg 1)]
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

(defn- place-lowering [id f args device]
  (let [prepared (compiled/lower f args {:compiler :equation-first :target device :dtype :float
                                        :schedule {:precision :f32-scalar}})
        lowering (:lowering prepared)]
    (:plan (composition/compose
            {:id id :components [{:id :program :lowering lowering}]
             :outputs (mapv #(vector :program %) (link/output-value-ids (:plan lowering)))}))))

(defn- value-node [plan value-id]
  (get-in plan [:nodes (get-in plan [:values value-id :leaves 0 :node])]))

(defn- initialized-value [plan source]
  (let [matches (for [[id _] (:values plan)
                     :when (identical? source (:source (value-node plan id)))] id)]
    (assert (= 1 (count matches)) "fixture input must have one exact initialized boundary")
    (first matches)))

(defn- training-plan [device row-counts]
  (let [n (count row-counts)
        [refinement topology inputs costs] (fixture/projection-inputs n)
        workers (get-in refinement [:group :devices])
        facts (distributed/refinement-facts refinement)
        options {:target-device device :target-descriptor (hardware/descriptor-for device)}
        data (batches row-counts)
        lr (float (/ 0.125 (reduce + row-counts)))
        producers (into {} (map (fn [worker {:keys [rows x y]}]
                                  [worker (place-lowering [worker :gradient] #'local-gradient-sum
                                                         [(initial-theta) x y rows 17] device)]) workers data))
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
                             local (place-lowering [worker :update] #'optim/sgd-step!
                                                   [(initial-theta) gradient 17 lr] device)
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
                   [worker {:plan local :gradient input}]))
        locals (merge (into {} (map (fn [[worker local]] [[worker :produce] local])) producers)
                      (update-vals combines :link-plan)
                      (into {} (map (fn [[worker {:keys [plan]}]] [[worker :update] plan])) consumers))
        node-to-ssa (into {} (map (fn [[ssa node]] [(:id node) ssa])) nodes)
        bindings
        (into {} (for [[step local] locals]
                   [step (into {} (for [[local-id _] (:values local)]
                                    [local-id
                                     (or (when (= :produce (second step))
                                           (when (= local-id (get producer-outputs (first step)))
                                             [(first step) :input]))
                                         (when (= :update (second step))
                                           (when (= local-id (get-in consumers [(first step) :gradient]))
                                             (get-in refinement [:outputs (first step)])))
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
                                                {:step [worker :update]
                                                 :local-value (get-in consumers [worker :gradient])}))
                                            (:outputs refinement))))]))
        updates (mapv (fn [worker completion]
                        (distributed/compute-step {:id [worker :update] :device worker :duration-ns 1
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
    (distributed/refinement-plan
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

(deftest actual-ad-all-reduce-sgd-on-colocated-opencl-workers
  (if @opencl/opencl-available?
    (check-training! :ocl:0)
    (opencl/opencl-skip! "actual AD/all-reduce/SGD with unequal local batches")))

(deftest actual-ad-all-reduce-sgd-on-colocated-level-zero-workers
  (if @ze/gpu-available?
    (check-training! :ze:0)
    (ze/gpu-skip! "actual AD/all-reduce/SGD with unequal local batches")))
