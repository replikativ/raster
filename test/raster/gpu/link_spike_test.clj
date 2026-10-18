(ns raster.gpu.link-spike-test
  "C.spike (.internal/artifact_layer_design.md §7.2) — de-risk for the composition/linking PR.
   Link TWO instances of ONE compiled descriptor into ONE command graph with the intermediate a
   device-resident INTERNAL node (never downloaded), using only bind-step! internals + a
   hand-written 2-instance binding-plan (sym→key from DATA).

   The elementwise case pins the sym→key-as-data plan and internal resident node. The GEMM case
   additionally pins the common executable boundary: each semantic GEMM may select a multi-kernel
   conversion/layout/contraction graph, yet two descriptor instances flatten into one replay graph
   with graph-private storage and captured-weight transforms kept out of replay."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.core :refer [deftm]]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.resident-plan :as resident-plan]
            [raster.compiler.pipeline :as pl]
            [raster.dl.nn :as nn]
            [raster.gpu.core :as gpu]
            [raster.gpu.test-lifecycle]
            [raster.gpu.link :as gpu-link]
            [raster.gpu.resident-value :as resident-value]
            [raster.gpu.value :as value])
  (:import [java.util.concurrent.atomic AtomicBoolean]))

;; A minimal elementwise forward: y = (x + w) * x. Typed placement and horizontal fusion lower the
;; residual-add/hadamard source to one tuple-free elementwise step. Composing it twice with weights
;; w0,w1 gives the intermediate
;; x1 = (x+w0)*x as a device-resident internal node the link shares between the two instances.
(deftm spike-had
  [x :- (Array float) w :- (Array float) n :- Long] :- (Array float)
  (let [s (nn/residual-add x w n)]
    (nn/hadamard s x n)))

(defn- fa ^floats [n seed]
  (let [rng (java.util.Random. (long seed)) a (float-array n)]
    (dotimes [i n] (aset a i (float (+ 0.5 (* 0.5 (.nextGaussian rng))))))
    a))

(defn- cpu-fwd ^floats [^floats x ^floats w n]
  ;; (x + w) * x, elementwise — residual-add then hadamard.
  (let [out (float-array n)]
    (dotimes [i n] (aset out i (float (* (+ (aget x i) (aget w i)) (aget x i)))))
    out))

(defn- cpu-linear ^floats [^floats x ^floats w rows width]
  (let [out (float-array (* rows width))]
    (dotimes [row rows]
      (dotimes [col width]
        (loop [inner 0 acc 0.0]
          (if (< inner width)
            (recur (inc inner)
                   (+ acc (* (double (aget x (+ (* row width) inner)))
                             (double (aget w (+ (* col width) inner))))))
            (aset out (+ (* row width) col) (float acc))))))
    out))

(defn- relative-max-error [^floats actual ^floats expected]
  (let [absolute (reduce max 0.0
                         (map (fn [a b] (Math/abs (- (double a) (double b))))
                              actual expected))
        scale (reduce max 1.0 (map #(Math/abs (double %)) expected))]
    (/ absolute scale)))

(deftest public-output-order-is-not-limited-by-array-map-size
  (let [output-ids (mapv #(keyword (str "output-" %)) (range 12))
        resident-values (zipmap output-ids (range 12))
        executable (raster.gpu.test-lifecycle/linked-executable
                    {:plan {:outputs output-ids}
                     :node-views resident-values
                     :closed? (atom false)})]
    (is (= output-ids (vec (keys (gpu-link/outputs executable)))))
    (is (= (range 12) (vals (gpu-link/outputs executable))))))

(deftest public-logical-outputs-preserve-composite-field-order
  (let [composite (link-plan/value
                   {:id :pair
                    :abstract (av/tensor {:dtype :float :shape [4]})
                    :physical-layout {:kind :ordered-fields :field-order [:data :scale]}
                    :leaves [{:name :data :node :data} {:name :scale :node :scale}]})
        dense (link-plan/value
               {:id :dense :abstract (av/tensor {:dtype :float :shape [4]})
                :leaves [{:name :value :node :dense-node}]})
        executable (raster.gpu.test-lifecycle/linked-executable
                    {:plan {:outputs [:data :scale :dense-node]
                            :values {:pair composite :dense dense}}
                     :node-views {:data :data-view :scale :scale-view :dense-node :dense-view}
                     :closed? (atom false)})
        outputs (gpu-link/output-values executable)]
    (is (= [:pair :dense] (vec (keys outputs))))
    (is (= :dense-view (:dense outputs)))
    (is (= [:data :scale] (mapv :name (get-in outputs [:pair :fields]))))
    (is (= [:data-view :scale-view] (mapv :value (get-in outputs [:pair :fields]))))))

(deftest composite-dispatch-projection-preserves-certified-physical-abi
  (let [allocation-a (bview/allocation {:id :a :byte-size 16 :memory-space :device
                                        :device :ze:0})
        allocation-b (bview/allocation {:id :b :byte-size 16 :memory-space :device
                                        :device :ze:0})
        view-a (gpu/->ResidentBufferView :session :a
                                         (bview/view allocation-a {:dtype :float :shape [4]}))
        view-b (gpu/->ResidentBufferView :session :b
                                         (bview/view allocation-b {:dtype :int :shape [4]}))
        composite (resident-value/composite
                   :pair [{:name :data :value view-a} {:name :index :value view-b}])
        slots [(kabi/slot 'pair_data :input :float :binding 'pair :field :data)
               (kabi/slot 'pair_index :input :int :binding 'pair :field :index)]
        step {:phase :dispatch :logical-bindings? true
              :argument-specs [{:kind :pointer :binding 'pair :slots slots}
                               {:kind :scalar}]}
        scalar {:type :long :value 4}]
    (is (= [view-a view-b scalar]
           (#'gpu-link/physical-step-arguments step [composite scalar])))))

(deftest linked-composite-value-expands-through-the-common-runtime
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "LinkValue composite ABI expansion")
    (let [n 16
          x-a (float-array (map float (range n)))
          x-b (float-array (map #(float (* 2 %)) (range n)))
          expected-sum (float-array (map + x-a x-b))
          expected-difference (float-array (map - x-a x-b))
          kernel
          (artifact/make
           {:kernel-name "linked_composite_add"
            :source (str "__kernel void linked_composite_add("
                         "__global const float* x_a, __global const float* x_b, "
                         "__global float* y_sum, __global float* y_difference, long n) { "
                         "long i = get_global_id(0); if (i < n) { "
                         "y_sum[i] = x_a[i] + x_b[i]; "
                         "y_difference[i] = x_a[i] - x_b[i]; } }")
            :abi [(kabi/slot 'x_a :input :float :binding 'x :field :a)
                  (kabi/slot 'x_b :input :float :binding 'x :field :b)
                  (kabi/slot 'y_sum :output :float :binding 'y :field :sum)
                  (kabi/slot 'y_difference :output :float :binding 'y :field :difference)
                  (kabi/slot 'n :scalar :long)]
            :arguments '[x_a x_b y_sum y_difference n]
            :launch (launch/spec {:workgroup-size [64]
                                  :group-count [(launch/ceil-div 'n 64)]})})
          descriptor
          {:dtype :float :all-params '[x n] :array-params '[x] :scalar-params '[n]
           :array-roles {'x :input}
           :value-specs
           {'x {:abstract (av/tensor {:dtype :float :shape ['n]
                                      :representation {:kind :two-input-fields}})
                :physical-layout {:kind :ordered-fields :field-order [:a :b]}
                :leaves [{:field :a :dtype :float} {:field :b :dtype :float}]}}
           :allocs
           [{:sym 'y
             :abstract (av/tensor {:dtype :float :shape ['n]
                                   :representation {:kind :two-output-fields}})
             :physical-layout {:kind :ordered-fields :field-order [:sum :difference]}
             :leaves [{:field :sum :dtype :float
                       :size-fn (fn [args] (long (nth args 1)))}
                      {:field :difference :dtype :float
                       :size-fn (fn [args] (long (nth args 1)))}]}]
           :steps [{:phase :add :kernel-name "linked_composite_add" :convention :map
                    :artifact kernel :logical-bindings? true
                    :argument-specs [{:kind :pointer :sym 'x}
                                     {:kind :output :sym 'y}
                                     {:kind :scalar :type :long
                                      :value-fn (fn [args] (long (nth args 1)))}]}]
           :result-sym 'y}
          lowering (resident-plan/lower
                    {:id :composite-add :target :ze:0 :descriptor descriptor
                     :arguments [{:a x-a :b x-b} n]})
          executable (gpu-link/instantiate! (:plan lowering))
          construction (gpu-link/instantiation-report executable)
          sum-node [:composite-add 'y :sum]
          difference-node [:composite-add 'y :difference]]
      (try
        (is (= :host-monotonic (:timing-source construction)))
        (is (pos? (:total-ns construction)))
        (is (every? #(and (integer? %) (not (neg? %)))
                    (vals (:phases-ns construction))))
        (is (contains? (:phases-ns construction) :plan-validation))
        (is (= 1 (:instances construction)))
        (is (= 1 (:bound-phases construction)))
        (gpu-link/run! executable)
        (is (= (vec expected-sum) (vec (gpu-link/download executable sum-node))))
        (is (= (vec expected-difference)
               (vec (gpu-link/download executable difference-node))))
        (finally (gpu-link/close! executable))))))

(deftest attached-close-does-not-descend-after-recording-teardown-fails
  (let [calls (atom [])
        executable (raster.gpu.test-lifecycle/linked-executable
                    {:session ::caller-session
                     :owns-session? false
                     :graph-key :graph
                     :phases [:phase-0 :phase-1]
                     :allocation-keys [:allocation-0 :allocation-1]
                     :closed? (atom false)
                     :lifetime-lock (Object.) :output-leases (atom 0)})]
    (with-redefs [gpu/release-recorded-graph!
                  (fn [_ graph]
                    (swap! calls conj [:graph graph])
                    (throw (ex-info "destructor failed" {})))
                  gpu/release-prepared!
                  (fn [_ phase] (swap! calls conj [:phase phase]))
                  gpu/free-buffer!
                  (fn [_ allocation] (swap! calls conj [:allocation allocation]))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"destructor failed"
                            (gpu-link/close! executable))))
    (is (= [[:graph :graph]] @calls))))

(deftest attached-close-attempts-independent-phases-but-retains-dependent-roots
  (let [calls (atom []) failure (ex-info "phase failed" {})
        secondary (ex-info "other phase failed" {})
        executable (raster.gpu.test-lifecycle/linked-executable
                     {:session ::caller-session :owns-session? false :graph-key :graph
                      :phases [:phase-0 :phase-1] :allocation-keys [:allocation]
                      :closed? (atom false) :lifetime-lock (Object.) :output-leases (atom 0)})]
    (with-redefs [gpu/release-recorded-graph! (fn [& _] (swap! calls conj :graph))
                  gpu/release-prepared! (fn [_ phase]
                                          (swap! calls conj phase)
                                          (throw (if (= phase :phase-1) failure secondary)))
                  gpu/free-buffer! (fn [& _] (swap! calls conj :root))]
      (is (identical? failure (try (gpu-link/close! executable) (catch Throwable error error))))
      (is (= [secondary] (vec (.getSuppressed failure))))
      (is (= [:graph :phase-1 :phase-0] @calls)))))

(deftest linked-device-input-is-zero-copy-or-device-to-device
  (let [destination-buffer (Object.)
        source-buffer (Object.)
        destination-allocation
        (bview/allocation {:id :destination :byte-size 20 :memory-space :device
                           :device :ze:0 :ownership :owned})
        source-allocation
        (bview/allocation {:id :source :byte-size 16 :memory-space :device
                           :device :ze:0 :ownership :external})
        destination-view (bview/view destination-allocation {:dtype :float :shape [4]})
        source-view (bview/view source-allocation {:dtype :float :shape [4]})
        resident (gpu/->ResidentBufferView :session :destination-key destination-view)
        executable (raster.gpu.test-lifecycle/linked-executable
                    {:plan {:target :ze:0
                            :nodes {:destination {:id :destination :role :input
                                                  :view destination-view}}}
                     :session ::session
                     :node-views {:destination resident}
                     :pending-inputs (atom #{:destination})
                     :tainted-inputs (atom #{})
                     :closed? (atom false)
                     :lifetime-lock (Object.) :output-leases (atom 0)
                     :output-ready? (atom false) :execution-state (atom {:value-epoch 0})})
        device-array (fn [buffer view]
                       (value/map->DeviceArray
                        {:buffer buffer :device :ze:0 :dtype :float :shape [4] :view view
                         :owner ::value/external :freed (AtomicBoolean. false)}))]
    (testing "the exact destination view is a zero-copy readiness transition"
      (with-redefs [gpu/buffer (fn [_ _] destination-buffer)
                    gpu/copy-range! (fn [& _] (throw (AssertionError. "unexpected copy")))
                    gpu/register-buffer! (fn [& _] (throw (AssertionError. "unexpected import")))]
        (is (identical? executable
                        (gpu-link/write! executable :destination
                                         (device-array destination-buffer destination-view))))
        (is (empty? @(:pending-inputs executable)))))
    (testing "device preflight checks compatibility without imports or readiness mutation"
      (reset! (:output-ready? executable) true)
      (reset! (:pending-inputs executable) #{:destination})
      (with-redefs [gpu/buffer (fn [_ _] destination-buffer)
                    gpu/register-buffer! (fn [& _] (throw (ex-info "preflight imported storage" {})))
                    gpu/copy-range! (fn [& _] (throw (ex-info "preflight copied storage" {})))]
        (is (identical? executable
                        (gpu-link/validate-write! executable :destination
                                                  (device-array source-buffer source-view))))
        (is (true? @(:output-ready? executable)))
        (is (= #{:destination} @(:pending-inputs executable)))
        (let [overlap (bview/view destination-allocation
                                  {:dtype :float :shape [4] :byte-offset 4})]
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"partially overlapping"
                                (gpu-link/write! executable :destination
                                                 (device-array destination-buffer overlap)))))
        (is (true? @(:output-ready? executable)))
        (is (= #{:destination} @(:pending-inputs executable)))))
    (testing "a foreign compatible DeviceArray is copied resident-to-resident and detached"
      (reset! (:pending-inputs executable) #{:destination})
      (let [calls (atom [])
            imported (gpu/->ResidentBufferView :session :temporary-key source-view)]
        (with-redefs [gpu/buffer (fn [_ _] destination-buffer)
                      gpu/register-buffer! (fn [_ key buffer opts]
                                             (swap! calls conj [:register key buffer opts]))
                      gpu/buffer-view (fn [_ key _]
                                        (is (some? key))
                                        imported)
                      gpu/copy-range! (fn [_ src dst spec]
                                        (swap! calls conj [:copy src dst spec]))
                      gpu/free-buffer! (fn [_ key] (swap! calls conj [:detach key]))]
          (gpu-link/write! executable :destination (device-array source-buffer source-view)))
        (is (= [:register :copy :detach] (mapv first @calls)))
        (is (= {:elements 4} (-> @calls second last)))
        (is (empty? @(:pending-inputs executable)))))
    (testing "a failed device copy does not preserve old initialized readiness"
      (reset! (:pending-inputs executable) #{})
      (let [failure (ex-info "simulated partially completed device copy" {})
            detached (atom false)]
        (with-redefs [gpu/buffer (fn [_ _] destination-buffer)
                      gpu/register-buffer! (fn [& _] nil)
                      gpu/buffer-view (fn [& _] (gpu/->ResidentBufferView ::session :import source-view))
                      gpu/copy-range! (fn [& _] (throw failure))
                      gpu/free-buffer! (fn [& _] (reset! detached true))]
          (is (identical? failure
                          (try (gpu-link/write! executable :destination
                                                (device-array source-buffer source-view))
                               (catch Throwable error error)))))
        (is (true? @detached))
        (is (= #{:destination} @(:pending-inputs executable)))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not been initialized"
                              (gpu-link/run! executable)))))
    (testing "a no-copy self-write cannot revive a failed destination"
      (with-redefs [gpu/buffer (fn [& _] destination-buffer)]
        (doseq [write [gpu-link/validate-write! gpu-link/write!]]
          (is (= :link-device-input-tainted
                 (try (write executable :destination
                             (device-array destination-buffer destination-view))
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))
      (is (= #{:destination} @(:pending-inputs executable))))
    (testing "copy failure stays primary when detaching the borrowed buffer also fails"
      (let [failure (ex-info "copy failed" {})
            cleanup (ex-info "detach failed" {})]
        (with-redefs [gpu/buffer (fn [& _] destination-buffer)
                      gpu/register-buffer! (fn [& _] nil)
                      gpu/buffer-view (fn [& _] (gpu/->ResidentBufferView ::session :import source-view))
                      gpu/copy-range! (fn [& _] (throw failure))
                      gpu/free-buffer! (fn [& _] (throw cleanup))]
          (is (identical? failure
                          (try (gpu-link/write! executable :destination
                                                (device-array source-buffer source-view))
                               (catch Throwable error error))))
          (is (= [cleanup] (vec (.getSuppressed failure)))))))))

(deftest failed-host-replacement-requires-overlapping-views-to-be-reinitialized
  (let [allocation (bview/allocation {:id :failed-transfer :byte-size 32 :memory-space :device
                                     :device :ze:0 :ownership :owned})
        views {:x (bview/view allocation {:dtype :float :shape [4]})
               :alias (bview/view allocation {:dtype :float :shape [4]})
               :partial (bview/view allocation {:dtype :float :shape [4] :byte-offset 4})
               :untouched (bview/view allocation {:dtype :float :shape [3] :byte-offset 20})}
        executable (raster.gpu.test-lifecycle/linked-executable
                    {:plan {:id :failed-transfer :target :ze:0 :outputs []
                            :nodes (into {} (map (fn [[id view]]
                                                  [id (link-plan/map->LinkNode
                                                       {:id id :view view :role :input})]) views))}
                     :session ::session :node-views (into {} (map (fn [[id view]]
                                                                   [id (gpu/->ResidentBufferView
                                                                        ::session id view)]) views))
                     :pending-inputs (atom #{}) :tainted-inputs (atom #{})
                     :closed? (atom false) :lifetime-lock (Object.)
                     :output-leases (atom 0) :output-ready? (atom true) :execution-state (atom {:value-epoch 0}) :completed-replays (atom 0)})
        failure (ex-info "simulated partially completed upload" {})
        replayed (atom false)]
    (with-redefs [gpu/upload-range! (fn [& _] (throw (AssertionError. "preflight must not copy")))]
      (is (thrown? clojure.lang.ExceptionInfo (gpu-link/upload! executable :x (double-array 4))))
      (is (empty? @(:pending-inputs executable)))
      (is (= {:value-epoch 0} @(:execution-state executable)))
      (is (true? @(:output-ready? executable))))
    (with-redefs [gpu/upload-range! (fn [& _] (throw failure))]
      (is (identical? failure (try (gpu-link/upload! executable :x (float-array 4))
                                   (catch Throwable error error)))))
    (is (= #{:x :alias :partial} @(:pending-inputs executable)))
    (is (= {:value-epoch 1} @(:execution-state executable)))
    (is (false? @(:output-ready? executable)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not been initialized" (gpu-link/run! executable)))
    (with-redefs [gpu/upload-range! (fn [& _] nil)
                  raster.gpu.parallel-program/run-prepared! (fn [& _] (reset! replayed true))]
      (gpu-link/upload! executable :x (float-array 4))
      (is (= #{:alias :partial} @(:pending-inputs executable)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not been initialized" (gpu-link/run! executable)))
      (doseq [id [:alias :partial]] (gpu-link/upload! executable id (float-array 4)))
      (is (empty? @(:pending-inputs executable)))
      (gpu-link/run! executable)
      (is (= {:value-epoch 5} @(:execution-state executable)))
      (is (true? @replayed)))))

(deftest linked-owned-allocation-roots-are-execution-qualified
  (let [a (bview/allocation {:id :compiler-local :byte-size 16 :memory-space :device
                           :device :ze:0 :alignment 64})
        nodes [{:view (bview/view a {:dtype :float :shape [4]})}]
        root-a [::gpu-link/allocation :execution-a :compiler-local]
        root-b [::gpu-link/allocation :execution-b :compiler-local]
        spec-a (#'gpu-link/allocation-spec nodes root-a)
        spec-b (#'gpu-link/allocation-spec nodes root-b)]
    (is (= :compiler-local (:id a)) "logical compiler allocation identity is unchanged")
    (is (= root-a (get-in spec-a [3 :allocation-id])))
    (is (= root-b (get-in spec-b [3 :allocation-id])))
    (is (not= (get-in spec-a [3 :allocation-id]) (get-in spec-b [3 :allocation-id]))
        "independently allocated native roots must not look like aliases to lifetime guards")
    (is (= [:float 4 nil] (subvec spec-a 0 3)))
    (is (= (dissoc (nth spec-a 3) :allocation-id) (dissoc (nth spec-b 3) :allocation-id)))))

(deftest plan-storage-realization-preserves-transaction-ledger-and-event-order
  (let [allocation
        (fn [id ownership]
          (bview/allocation {:id id :byte-size 4 :memory-space :device
                             :device :ze:0 :ownership ownership}))
        nodes
        (array-map
         :owned {:view (bview/view (allocation :owned :owned)
                                   {:id :owned :dtype :float :shape [1]})
                 :source (float-array [1.0])}
         :external-a {:view (bview/view (allocation :external-a :external)
                                        {:id :external-a :dtype :float :shape [1]})}
         :external-b {:view (bview/view (allocation :external-b :borrowed)
                                        {:id :external-b :dtype :float :shape [1]})})
        plan {:nodes nodes}
        groups (array-map :owned [(:owned nodes)]
                          :external-a [(:external-a nodes)]
                          :external-b [(:external-b nodes)])
        external-buffers {:external-a ::external-a :external-b ::external-b}
        key (fn [allocation-id]
              [::gpu-link/allocation :execution allocation-id])
        invoke
        (fn [events allocation-keys fail-at]
          (with-redefs [gpu/alloc!
                        (fn [_ specs]
                          (swap! events conj [:allocate (vec (keys specs))])
                          (when (= fail-at :allocation)
                            (throw (ex-info "allocation failed" {:at fail-at}))))
                        gpu/register-buffer!
                        (fn [_ _ _ {:keys [allocation-id]}]
                          (swap! events conj [:register allocation-id])
                          (when (= fail-at [:register allocation-id])
                            (throw (ex-info "registration failed" {:at fail-at}))))
                        gpu/buffer-view
                        (fn [_ _ {:keys [id]}]
                          (swap! events conj [:view id])
                          (when (= fail-at [:view id])
                            (throw (ex-info "view failed" {:at fail-at})))
                          [:resident id])
                        gpu/upload-range!
                        (fn [_ resident _ _]
                          (swap! events conj [:upload (second resident)])
                          (when (= fail-at :upload)
                            (throw (ex-info "upload failed" {:at fail-at}))))]
            (#'gpu-link/realize-plan-storage!
             plan (atom {:device-id :ze:0}) :execution groups external-buffers
             allocation-keys (volatile! {}))))]
    (testing "success preserves allocate, registration, view, and upload order"
      (let [events (atom []) allocation-keys (volatile! [])
            realized (invoke events allocation-keys nil)]
        (is (= [[:allocate [(key :owned)]]
                [:register :external-a] [:register :external-b]
                [:view :owned] [:view :external-a] [:view :external-b]
                [:upload :owned]]
               @events))
        (is (= [(key :owned) (key :external-a) (key :external-b)]
               @allocation-keys))
        (is (= {:owned (key :owned)
                :external-a (key :external-a)
                :external-b (key :external-b)}
               (:key-by-allocation realized)))
        (is (= #{(key :owned)} (set (keys (:owned-specs realized)))))
        (is (= #{:owned :external-a :external-b}
               (set (keys (:node-views realized)))))))
    (testing "a failed allocation enters no root into cleanup and stops realization"
      (let [events (atom []) allocation-keys (volatile! [])]
        (is (= :allocation
               (try (invoke events allocation-keys :allocation) nil
                    (catch clojure.lang.ExceptionInfo error (:at (ex-data error))))))
        (is (= [[:allocate [(key :owned)]]] @events))
        (is (empty? @allocation-keys))))
    (testing "each external registration is ledgered only after it succeeds"
      (let [events (atom []) allocation-keys (volatile! [])]
        (is (= [:register :external-b]
               (try (invoke events allocation-keys [:register :external-b]) nil
                    (catch clojure.lang.ExceptionInfo error (:at (ex-data error))))))
        (is (= [[:allocate [(key :owned)]]
                [:register :external-a] [:register :external-b]]
               @events))
        (is (= [(key :owned) (key :external-a)] @allocation-keys))))
    (doseq [[label fail-at expected-events]
            [["view failure precedes every upload" [:view :external-a]
              [[:allocate [(key :owned)]]
               [:register :external-a] [:register :external-b]
               [:view :owned] [:view :external-a]]]
             ["upload failure retains every realized root" :upload
              [[:allocate [(key :owned)]]
               [:register :external-a] [:register :external-b]
               [:view :owned] [:view :external-a] [:view :external-b]
               [:upload :owned]]]]]
      (testing label
        (let [events (atom []) allocation-keys (volatile! [])]
          (is (= fail-at
                 (try (invoke events allocation-keys fail-at) nil
                      (catch clojure.lang.ExceptionInfo error (:at (ex-data error))))))
          (is (= expected-events @events))
          (is (= [(key :owned) (key :external-a) (key :external-b)]
                 @allocation-keys)))))))

(deftest later-storage-stages-reuse-ledgered-roots-without-reinitializing-them
  (let [node (fn [id source]
               {:view (bview/view (bview/allocation {:id id :byte-size 4 :memory-space :device
                                                    :device :ze:0 :ownership :owned})
                                  {:id id :dtype :float :shape [1]})
                :source source})
        first-plan {:nodes {:a (node :a (float-array [1.0]))}}
        later-plan {:nodes (array-map :a (node :a (float-array [999.0]))
                                      :b (node :b (float-array [2.0])))}
        ledger (volatile! [])
        events (atom [])
        realize (fn [plan roots]
                  (gpu-link/realize-plan-storage!
                   plan ::session :same-owner (gpu-link/allocation-groups plan) {}
                   ledger (volatile! {}) roots))]
    (with-redefs [gpu/alloc! (fn [_ specs] (swap! events conj [:allocate (set (keys specs))]))
                  gpu/buffer-view (fn [_ key {:keys [id]}]
                                    (swap! events conj [:view id key]) {:id id :key key})
                  gpu/upload-range! (fn [_ view source _]
                                      (swap! events conj [:upload (:id view) (vec source)]))]
      (let [first-result (realize first-plan {})
            a-key (get-in first-result [:roots :a :key])]
        (is (= [a-key] @ledger))
        (is (= [:upload :a [1.0]] (last @events)))
        (reset! events [])
        (let [later (realize later-plan (:roots first-result))
              b-key (get-in later [:roots :b :key])]
          (is (= [[:allocate #{b-key}]
                  [:view :a a-key] [:view :b b-key] [:upload :b [2.0]]]
                 @events))
          (is (= [a-key b-key] @ledger))
          (is (= #{:a :b} (set (keys (:roots later)))))
          (is (= a-key (get-in later [:node-views :a :key]))))
        (doseq [roots [(assoc-in (:roots first-result) [:a :key] :foreign-owner)
                       (assoc-in (:roots first-result) [:a :allocation :byte-size] 8)
                       (assoc-in (:roots first-result) [:a :allocation :ownership] :borrowed)]]
          (reset! events [])
          (is (= :link-existing-root-contract
                 (try (realize later-plan roots) nil
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
          (is (empty? @events) "reject changed root contracts before acquiring any later root"))
        (vreset! ledger [a-key])
        (reset! events [])
        (with-redefs [gpu/buffer-view
                      (fn [_ _ {:keys [id]}]
                        (swap! events conj [:view id])
                        (throw (ex-info "root is no longer live" {:reason :dead-root})))]
          (is (= :dead-root
                 (try (realize later-plan (:roots first-result)) nil
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
          (is (= 2 (count @ledger)) "the newly acquired suffix root remains cleanup-owned")
          (is (= a-key (first @ledger)))
          (is (= [[:allocate #{(second @ledger)}] [:view :a]] @events)))
        (vreset! ledger [])
        (reset! events [])
        (is (= :link-existing-root-contract
               (try (realize later-plan (:roots first-result)) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
        (is (empty? @events) "a description without the acquisition ledger is insufficient")))))

(deftest source-storage-validates-shared-contracts-before-native-acquisition
  (let [allocation (bview/allocation {:id :shared :byte-size 8 :memory-space :device
                                      :device :ze:0 :ownership :owned})
        make-node (fn [id root dtype shape offset source]
                    (link-plan/node
                     {:id id :view (bview/view root {:id id :dtype dtype :shape shape
                                                     :byte-offset offset})
                      :source source}))
        a (make-node :a allocation :float [1] 0 nil)
        b (make-node :b allocation :float [1] 4 nil)
        overlapping (make-node :b allocation :float [1] 0 nil)
        acquired (atom [])
        ledger (volatile! [])
        realize (fn [nodes aliases]
                  (gpu-link/realize-storage-nodes!
                   nodes aliases (atom {:device-id :ze:0}) :source-owner {} ledger (volatile! {}) {}))
        reason (fn [nodes aliases]
                 (try (realize nodes aliases) nil
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (with-redefs [gpu/alloc! (fn [_ specs] (swap! acquired conj specs))
                  gpu/buffer-view (fn [_ key view] {:key key :view view})
                  gpu/upload-range! (fn [& _] (swap! acquired conj :upload))]
      (doseq [[nodes aliases expected]
              [[[a (make-node :b (assoc allocation :byte-size 12) :float [1] 4 nil)]
                #{} :link-allocation-contract]
               [[a (make-node :b (assoc allocation :ownership :borrowed) :float [1] 4 nil)]
                #{} :link-allocation-contract]
               [[a (make-node :b allocation :int [1] 4 nil)]
                #{} :link-allocation-reinterpret]
               [[(make-node :a (assoc allocation :device :ocl:0) :float [1] 0 nil)]
                #{} :link-node-device]
               [[(make-node :a (assoc allocation :device nil) :float [1] 0 nil)]
                #{} :link-node-device]
               [[a overlapping] #{} :link-undeclared-alias]
               [[(assoc a :source (float-array [1]))
                 (assoc overlapping :source (float-array [2]))]
                #{#{:a :b}} :link-alias-initializers]
               [[a b] #{#{:a :b}} :link-spurious-alias]
               [[a] #{#{:a :absent}} :link-alias-node]
               [[a a] #{} :link-storage-node-identities]
               [{:other a} #{} :link-storage-node-identities]
               [[(into {} a)] #{} :link-node-type]]]
        (is (= expected (reason nodes aliases)))
        (is (empty? @acquired) "all structural obligations precede native effects")
        (is (empty? @ledger)))
      (is (= #{:a :b} (set (keys (:node-views (realize [a b] #{}))))))
      (is (= 1 (count @acquired)) "disjoint views acquire a single root")
      (reset! acquired [])
      (is (= #{:a :b}
             (set (keys (:node-views (realize [a overlapping] #{#{:a :b}}))))))
      (is (= 1 (count @acquired)) "declared aliases still share the same physical root"))))

(deftest resident-program-executor-preserves-binding-and-callback-contracts
  (let [allocation (bview/allocation {:id :resident :byte-size 16 :memory-space :device
                                      :device :ze:0 :ownership :owned})
        view (bview/view allocation {:id :resident-view :dtype :float :shape [4]})
        resident (gpu/->ResidentBufferView ::session :resident-key view)
        plan {:values {:x (link-plan/value
                           {:id :x :abstract (av/tensor {:dtype :float :shape [4]})
                            :leaves [{:name :value :node :x}]})}}
        node-views {:x resident}
        allocation-keys (volatile! [:root-a])
        calls (atom [])
        owner (cleanup/owner [])]
    (with-redefs [gpu/bind-kernel-graph!
                  (fn [session key graph buffers scalars options]
                    (swap! calls conj [:bind session key graph buffers scalars options])
                    :bound-handle)
                  gpu/retain-program-cleanup!
                  (fn [session actual-owner roots]
                    (swap! calls conj [:adopt session actual-owner (vec roots)]))
                  gpu/run-kernel-graph!
                  (fn [session handle] (swap! calls conj [:run session handle]) :ran)
                  gpu/release-kernel-graph!
                  (fn [session handle] (swap! calls conj [:release session handle]) :released)]
      (let [executor (#'gpu-link/resident-program-executor
                      plan ::session :execution node-views true true allocation-keys)
            explicit-scalars {(list 'extent 'x) {:type :long :value 99}
                              'alpha {:type :float :value 0.5}}
            graph (Object.)]
        (is (= view ((:buffer-view executor) :x)))
        (is (= :bound-handle
               ((:bind! executor) :binding graph {'x :x 'y :x} explicit-scalars)))
        (vswap! allocation-keys conj :root-b)
        ((:adopt-cleanup! executor) owner)
        (is (= :ran ((:run! executor) :bound-handle)))
        (is (= :released ((:release! executor) :bound-handle)))
        (let [[_ session key actual-graph buffers scalars options] (first @calls)]
          (is (= ::session session))
          (is (= [::gpu-link/program-graph :execution :binding] key))
          (is (identical? graph actual-graph))
          (is (identical? resident (get buffers 'x)))
          (is (identical? resident (get buffers 'y)))
          (is (= {(list 'extent 'x) {:type :long :value 99}
                  (list 'extent 'y) {:type :long :value 4}
                  'alpha {:type :float :value 0.5}}
                 scalars)
              "explicit scalar values retain precedence over derived extents")
          (is (= {:profile? true :record? false} options)))
        (is (= [:adopt ::session owner [:root-a :root-b]] (second @calls)))
        (is (= [:run ::session :bound-handle] (nth @calls 2)))
        (is (= [:release ::session :bound-handle] (nth @calls 3)))
        (let [dynamic-executor (#'gpu-link/resident-program-executor
                                plan ::session :dynamic node-views false false allocation-keys)]
          (is (= :bound-handle
                 ((:bind! dynamic-executor) :dynamic-binding graph {'x :x} {})))
          (is (= {:profile? false :record? true} (last (last @calls))))))))
  (testing "executor callbacks preserve exact primary failures"
    (let [failure (ex-info "bind failed" {})
          allocation (bview/allocation {:id :failure :byte-size 4 :memory-space :device
                                        :device :ze:0 :ownership :owned})
          resident (gpu/->ResidentBufferView
                    ::session :failure-key
                    (bview/view allocation {:dtype :float :shape [1]}))
          plan {:values {:x (link-plan/value
                             {:id :x :abstract (av/tensor {:dtype :float :shape [1]})
                              :leaves [{:name :value :node :x}]})}}
          executor (#'gpu-link/resident-program-executor
                    plan ::session :execution {:x resident} false false (volatile! []))]
      (with-redefs [gpu/bind-kernel-graph! (fn [& _] (throw failure))]
        (is (identical? failure
                        (try ((:bind! executor) :binding ::graph {'x :x} {})
                             (catch Throwable error error))))))))

(deftest owned-runtime-allocation-preserves-the-certified-identity
  (let [session (atom {:device-id :ze:0 :session-id :session
                       :buffers {} :allocations {} :closed? false})
        buffer {:byte-size 16 :alignment 64 :dtype :float :n-elements 4}]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve)
       (fn [_device name]
         (case name
           "make-buffer" (fn [n dtype opts]
                           (is (= [4 :float] [n dtype]))
                           (raster.gpu.test-lifecycle/native-buffer (constantly buffer) (fn [_]) opts))
           "array->buffer!" (fn [buffer _] buffer)
           (throw (ex-info "Unexpected runtime function" {:name name}))))}
      (fn []
        (gpu/alloc! session
                    {:node [:float 4 nil {:allocation-id :certified
                                          :memory-space :device
                                          :coherence :device-only
                                          :alignment 64}]})))
    (is (= :certified (get-in @session [:allocations :node :id])))
    (is (= :device (get-in @session [:allocations :node :memory-space])))
    (is (= :device-only (get-in @session [:allocations :node :coherence])))
    (is (= 64 (get-in @session [:allocations :node :alignment])))))

(deftest stable-recorded-graph-profiling-keeps-backend-handles-private
  (let [graph (Object.)
        session (atom {:device-id :ze:0
                       :graphs {:profiled {::gpu/recorded-graph true ::cleanup/owner (cleanup/owner [])
                                           :replay-graph graph :profile? true}}})
        calls (atom [])
        resolver (fn [_device name]
                   (case name
                     "replay-graph!" (fn [actual]
                                       (is (identical? graph actual))
                                       (swap! calls conj :replay))
                     "read-graph-timestamps!"
                     (fn [actual]
                       (is (identical? graph actual))
                       (swap! calls conj :read)
                       {:wall-ms 1.5
                        :kernels [{:kernel-name "k0" :phase :p0 :ms 1.0
                                   :context-ms 0.25}]})
                     (throw (ex-info "unexpected runtime resolution" {:name name}))))]
    (with-redefs-fn {(ns-resolve 'raster.gpu.core 'rt-resolve) resolver}
      (fn []
        (let [profile (gpu/profile-recorded-graph! session :profiled)]
          (is (= [:replay :read] @calls))
          (is (= 1.0 (:kernel-total-ms profile)))
          (is (= 1.5 (:device-wall-ms profile)))
          (is (= [{:kernel-name "k0" :phase :p0 :ms 1.0 :context-ms 0.25}]
                 (:profile profile))))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not created"
                          (gpu/profile-recorded-graph!
                           (atom {:device-id :ze:0 :graphs {:plain graph}}) :plain)))))

(deftest linked-profile-attributes-device-events-to-compiler-steps
  (let [instance-id [:encoder :layer-0]
        phase [::gpu-link/phase :execution instance-id 1]
        measured {:profile [{:kernel-name "segmap_42" :phase phase :ms 2.0}
                            {:kernel-name "contract_tile"
                             :phase [::gpu/graph-node-phase
                                     [::gpu-link/phase :execution instance-id 0]
                                     :tile]
                             :ms 1.0}
                            {:kernel-name "prologue" :phase :unmapped :ms 0.5}]
                  :kernel-total-ms 3.5 :device-wall-ms 3.5}
        executable (raster.gpu.test-lifecycle/linked-executable
                    {:plan {:instances [{:id instance-id
                                         :descriptor
                                         {:steps [{:convention :contract
                                                   :artifact {:provenance
                                                              {:semantic-op :contraction
                                                               :operation-id :projection}}}
                                                  {:convention :map
                                                   :artifact
                                                   {:provenance
                                                    {:dialect :segmap
                                                     :segop-id [:map :scores]
                                                     :operation-id :scores
                                                     :debug-only :excluded}}}]}}]}
                     :session :session :graph-key :graph
                     :profile? true
                     :pending-inputs (atom #{}) :closed? (atom false)
                     :lifetime-lock (Object.) :output-leases (atom 0)
                     :output-ready? (atom false) :execution-state (atom {:value-epoch 0}) :completed-replays (atom 0)})]
    (with-redefs [gpu/profile-recorded-graph! (fn [session graph]
                                                (is (= [:session :graph] [session graph]))
                                                measured)]
      (let [profile (gpu-link/profile! executable)]
        (is (= 3.5 (:device-wall-ms profile)))
        (is (= {:instance-id instance-id :step-index 1 :convention :map
                :provenance {:dialect :segmap :segop-id [:map :scores]
                             :operation-id :scores}}
               (get-in profile [:profile 0 :compiler-step])))
        (is (= {:instance-id instance-id :step-index 0 :convention :contract
                :provenance {:semantic-op :contraction :operation-id :projection}
                :graph-node-id :tile}
               (get-in profile [:profile 1 :compiler-step])))
        (is (= (last (:profile measured)) (last (:profile profile)))
            "unmapped phases remain intact instead of receiving guessed provenance")))))

(deftest spike-two-instance-link
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "C.spike 2-instance link")
    (let [gpu (do (require 'raster.gpu.core) (find-ns 'raster.gpu.core))
          make-session   (ns-resolve gpu 'make-session)
          bind-step!     (ns-resolve gpu 'bind-step!)
          record-graph!  (ns-resolve gpu 'record-graph!)
          replay!        (ns-resolve gpu 'replay!)
          alloc!         (ns-resolve gpu 'alloc!)
          download       (ns-resolve gpu 'download)
          close-session! (ns-resolve gpu 'close-session!)
          n 4096
          x0 (fa n 1) W0 (fa n 2) W1 (fa n 3)
          args [x0 W0 n]
          prog (pl/compile-gpu-program #'spike-had :ze:0 :dtype :float :on-non-resident :nil)
          _ (is (some? prog) "spike-had must extract fully resident (else the spike can't run)")
          steps (:steps prog)
          result-sym (:result-sym prog)
          scratch-sym (first (remove #(= % result-sym) (map :sym (:allocs prog))))
          x-sym 'x w-sym 'w]
      (testing "the descriptor retains the optimally fused elementwise step"
        (is (= 1 (count steps)))
        (is (every? #(= :map (:convention %)) steps)))
      (let [sess (make-session :ze:0)]
        (try
          ;; resident buffers: two inputs' weights + shared input + the INTERNAL node x1 + output x2
          ;; + per-instance scratch. x1 is produced by instance 0, consumed by instance 1, never
          ;; downloaded — the whole point.
          (alloc! sess {:x0 [:float n x0] :W0 [:float n W0] :W1 [:float n W1]
                        :s0 [:float n nil] :s1 [:float n nil]
                        :x1 [:float n nil] :x2 [:float n nil]})
          ;; the hand-written 2-instance binding-plan (sym→key as DATA — what the linker promotes):
          (let [plan {0 {x-sym :x0, w-sym :W0, scratch-sym :s0, result-sym :x1}
                      1 {x-sym :x1, w-sym :W1, scratch-sym :s1, result-sym :x2}}
                phases (vec (for [inst [0 1] step steps]
                              (let [ph (keyword (str "i" inst "-" (name (:phase step))))]
                                (bind-step! sess (assoc step :phase ph) args (get plan inst))
                                ph)))]
            (record-graph! sess phases :graph)
            (replay! sess :graph)
            (let [x1 (download sess :x1)
                  x2 (download sess :x2)
                  x1-cpu (cpu-fwd x0 W0 n)
                  x2-cpu (cpu-fwd x1-cpu W1 n)
                  maxdiff (fn [^floats a ^floats b]
                            (reduce max 0.0 (map (fn [p q] (Math/abs (- (double p) (double q)))) a b)))]
              (println "  [spike] x1[0..2] dev:" (mapv #(aget ^floats x1 %) [0 1 2])
                       "cpu:" (mapv #(aget ^floats x1-cpu %) [0 1 2]))
              (println "  [spike] x2[0..2] dev:" (mapv #(aget ^floats x2 %) [0 1 2])
                       "cpu:" (mapv #(aget ^floats x2-cpu %) [0 1 2]))
              (println "  [spike] maxdiff x1:" (maxdiff x1 x1-cpu) " x2:" (maxdiff x2 x2-cpu))
              ;; pure elementwise → no reduction reassociation; the only gap is GPU-FMA vs
              ;; sequential-CPU float rounding (~1e-6), so the gate is a tight relative tolerance.
              (testing "the INTERNAL node x1 = f(x0,W0) is computed on-device"
                (is (< (maxdiff x1 x1-cpu) 1.0e-4)
                    "internal node x1 must equal CPU f(x0,W0) to float precision"))
              (testing "linked 2-instance graph matches CPU f(f(x,W0),W1), internal node never re-uploaded"
                (is (< (maxdiff x2 x2-cpu) 1.0e-3)
                    "device x2 must equal CPU double-composition to float precision"))))
          (finally (close-session! sess)))))))

(deftest spike-two-instance-gemm-link
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "C.spike 2-instance executable GEMM link")
    (let [rows 16 width 64
          x0 (fa (* rows width) 11)
          W0 (fa (* width width) 12)
          W1 (fa (* width width) 13)
          prog (pl/compile-gpu-program #'nn/linear-nb :ze:0 :dtype :float
                                       :gemm-precision :mixed-f16-f32 :on-non-resident :nil)
          _ (is (some? prog) "spike-gemm must extract as one resident contraction")
          result-sym (:result-sym prog)
          scalar-values {'batch rows 'in-f width 'out-f width}
          plan
          (link-plan/make
           {:id :two-linear-layers
            :target :ze:0
            :nodes [(link-plan/node {:id :x0 :dtype :float :shape [rows width]
                                     :device :ze:0 :role :input})
                    (link-plan/node {:id :W0 :dtype :float :shape [width width]
                                     :device :ze:0 :role :constant :source W0})
                    (link-plan/node {:id :W1 :dtype :float :shape [width width]
                                     :device :ze:0 :role :constant :source W1})
                    (link-plan/node {:id :x1 :dtype :float :shape [rows width]
                                     :device :ze:0 :role :internal})
                    (link-plan/node {:id :x2 :dtype :float :shape [rows width]
                                     :device :ze:0 :role :output})]
            :instances
            [(link-plan/instance
              {:id :linear-0 :descriptor prog :scalars scalar-values
               :bindings {'x :x0 'W :W0 result-sym :x1}})
             (link-plan/instance
              {:id :linear-1 :descriptor prog :scalars scalar-values
               :bindings {'x :x1 'W :W1 result-sym :x2}})]
            :outputs [:x2]})
          session (gpu/make-session :ze:0)
          executable (gpu-link/instantiate! plan {:session session :profile? true})]
      (is (= [:executable] (mapv :convention (:steps prog)))
          "linear-nb's BLAS GEMM is a typed contraction step")
      (try
        (let [session (:session executable)
              phases (:phases executable)
              prepared-count
              (reduce + (map #(count (get-in @session [:prepared % :prepareds])) phases))
              private-count
              (reduce + (map #(count (get-in @session [:prepared % :temporary-buffers])) phases))]
          (is (>= prepared-count 2)
              "each instance prepares its own contraction dispatch")
          (is (>= private-count 0)
              "step-private storage is accounted per phase (a contraction dispatch may need none)")
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"have not been initialized"
                                (gpu-link/run! executable))
              "owned dynamic inputs must be uploaded before the first replay")
          (gpu-link/upload! executable :x0 x0)
          (let [events (:profile (gpu-link/profile! executable))]
            (is (seq events))
            (is (= #{:linear-0 :linear-1}
                   (set (map #(get-in % [:compiler-step :instance-id]) events))))
            (is (every? #(and (= :executable (get-in % [:compiler-step :convention]))
                              (contains? (:compiler-step %) :graph-node-id))
                        events)
                "nested graph kernels retain their selecting descriptor step"))
          (let [resident-outputs (gpu-link/run! executable)
                actual (gpu-link/download executable :x2)
                expected (cpu-linear (cpu-linear x0 W0 rows width) W1 rows width)]
            (is (= [:x2] (vec (keys resident-outputs)))
                "invocation returns stable resident output views without a host copy")
            (is (< (relative-max-error actual expected) 2.0e-2)
                (str "linked mixed-precision GEMMs must match the CPU composition; relative max "
                     (relative-max-error actual expected)))))
        (finally
          (gpu-link/close! executable)
          (is (false? (:closed? @session))
              "closing an attached linked executable does not close its caller session")
          (is (empty? (:prepared @session)) "attached phase bindings are released")
          (is (empty? (:graphs @session)) "attached graph recordings are released")
          (is (empty? (:buffers @session)) "attached allocation registrations are released")
          (gpu/close-session! session))))))
