(ns raster.gpu.test-lifecycle
  "Ownership-complete host fixtures. Generic record helpers do not validate plans or acquire native
   state; the completed-output helper explicitly uses production plan and boundary validation. Tests
   still mock their corresponding native execution/resource release operations."
  (:require [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.resource-cleanup :as cleanup]))

(defn native-buffer
  "Fault-double constructor with the production ownership protocol. acquire must return a map;
   uncertain acquisition is retained rather than guessed to have created nothing."
  [acquire release {:keys [retain-owner! adopt-cleanup!]}]
  (let [slot (cleanup/acquisition-slot)
        owner-ref (volatile! nil)
        owner (cleanup/owner [{:id :memory
                               :release #(cleanup/release-native!
                                          slot (fn [value] (release (assoc value ::cleanup/owner @owner-ref))))}])]
    (vreset! owner-ref owner)
    (cleanup/build! owner
                    #(do (when retain-owner! (retain-owner! owner))
                         (cleanup/acquire-native! slot acquire))
                    (or adopt-cleanup! retain-owner!))))

(defn linked-executable [fields]
  ((ns-resolve 'raster.gpu.link 'own-linked-executable)
   (link/map->LinkedExecutable
    (merge {:closed? (atom false) :lifetime-lock (Object.)
            :execution-state (atom {:value-epoch 0})}
           fields))))

(def ^:private output-kernel
  (artifact/make
   {:kernel-name "test_lifecycle_output"
    :source (str "__kernel void test_lifecycle_output(__global const float* input, "
                 "__global float* output, long n) { "
                 "long i = get_global_id(0); if (i < n) output[i] = input[i]; }")
    :abi [(kabi/slot 'x :input :float :c-name "input")
          (kabi/slot 'x :output :float :c-name "output")
          (kabi/slot 'n :scalar :long)]
    :arguments '[x x n]
    :launch (launch/spec {:workgroup-size [64]
                          :group-count [(launch/ceil-div 'n 64)]})
    :effects {:kind :map :reads '[x] :writes '[x]}
    :attributes {:strategy :test-lifecycle}}))

(def ^:private output-descriptor
  {:dtype :float
   :all-params '[x n]
   :array-params '[x]
   :scalar-params '[n]
   :array-roles {'x :state}
   :allocs []
   :steps [{:phase :output :kernel-name "test_lifecycle_output" :convention :map
            :artifact output-kernel
            :argument-specs [{:kind :input :sym 'x}
                             {:kind :output :sym 'x}
                             {:kind :scalar :type :long
                              :value-fn (fn [args] (long (nth args 1)))}]}]
   :result-sym 'x})

(defn linked-output-executable
  "Build an ownership-complete, initially unexecuted float-output fixture.

   Each output is {:id :shape, optional :source/:ownership/:buffer}. The plan and retained effect
   evidence use the production constructors; session allocations and ResidentBufferViews match that
   exact plan. This grants no completion: tests must mock only native replay and call link/run! before
   requesting a completed boundary or output lease."
  [{:keys [id target session-id outputs owns-session? session-fields fields]
    :or {id :test-output-plan target :ocl:test session-id :test-output-session
         owns-session? true session-fields {} fields {}}}]
  (when-not (and (vector? outputs) (seq outputs))
    (throw (ex-info "linked output fixture requires ordered outputs" {:outputs outputs})))
  (let [nodes
        (mapv (fn [{:keys [id shape source ownership]
                    :or {ownership :owned}}]
                (link-plan/node {:id id :dtype :float :shape (vec shape) :role :state
                                 :source source :device target :memory-space :device
                                 :ownership ownership :allocation-id id}))
              outputs)
        instances
        (mapv (fn [node]
                (link-plan/instance
                 {:id [:test-output (:id node)] :descriptor output-descriptor
                  :bindings {'x (:id node)}
                  :scalars {'n (reduce * 1 (get-in node [:view :shape]))}}))
              nodes)
        {plan :plan effect-evidence :effect-evidence}
        (link-plan/make-with-effect-evidence
         {:id id :target target :nodes nodes :instances instances :outputs (mapv :id nodes)})
        buffers
        (into {}
              (map (fn [node output]
                     (let [view (:view node)]
                       [(:id node)
                        (or (:buffer output)
                            {:dtype :float
                             :n-elements (reduce * 1 (:shape view))
                             :byte-size (get-in view [:allocation :byte-size])})]))
                   nodes outputs))
        allocations (into {} (map (fn [node] [(:id node) (get-in node [:view :allocation])])) nodes)
        session (atom (-> {:device-id target :session-id session-id :closed? false
                           :buffers buffers :allocations allocations :events {}}
                          (merge session-fields)
                          (assoc :buffers buffers :allocations allocations)))
        node-views (into {} (map (fn [node]
                                   [(:id node) (gpu/->ResidentBufferView
                                                session-id (:id node) (:view node))])) nodes)
        pending (set (get-in effect-evidence [:initialization :requires]))]
    (linked-executable
     (merge fields
            {:plan plan :session session :owns-session? owns-session? :graph-key ::output-graph
             :node-views node-views :pending-inputs (atom pending) :tainted-inputs (atom #{})
             :execution-state (atom {:value-epoch 0 :effect-evidence effect-evidence})
             :completed-replays (atom 0) :output-leases (atom 0)
             :output-ready? (atom false)}))))

(defn completed-evidence
  "Seal hardware-free completed evidence only for an actually completed linked boundary.
   The real output lease and current epoch are retained; no readiness/replay atom is stamped here."
  ([executable data] (completed-evidence executable data {}))
  ([executable data values]
   (locking (:lifetime-lock executable)
     (let [lease (link/output-lease! executable)]
       (try
         ;; The common lifetime lock and acquired lease make this exactly the leased completion.
         (let [{:keys [epoch]} (link/completed-boundary executable)
               _ (require 'raster.gpu.compiled)
               construct (ns-resolve 'raster.gpu.compiled 'map->CompletedEvidence)
               seal (ns-resolve 'raster.gpu.compiled 'seal-artifact)]
           (seal (construct {:data data :lease lease :executable executable :epoch epoch
                             :values values :released? (atom false)})))
         (catch Throwable error
           (.close ^java.io.Closeable lease)
           (throw error)))))))

(defn distributed-executable
  "Synthetic original runtime owner for lifecycle doubles, not compiler/completion evidence.
   Tests must still supply states and mock release operations; record copies remain unissued."
  [fields]
  (let [constructor (requiring-resolve 'raster.gpu.distributed/map->DistributedExecutable)
        seal (ns-resolve 'raster.gpu.distributed 'seal-runtime-value)]
    (seal (constructor fields))))
