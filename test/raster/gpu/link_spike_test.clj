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
            [raster.gpu.link :as gpu-link]
            [raster.gpu.resident-value :as resident-value]
            [raster.gpu.value :as value])
  (:import [java.util.concurrent.atomic AtomicBoolean]))

;; A minimal elementwise forward: y = (x + w) * x  (residual-add then hadamard — the proven-
;; resident dt-two-step shape). Composing it twice with weights w0,w1 gives the intermediate
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
        executable (gpu-link/map->LinkedExecutable
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
        executable (gpu-link/map->LinkedExecutable
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
          sum-node [:composite-add 'y :sum]
          difference-node [:composite-add 'y :difference]]
      (try
        (gpu-link/run! executable)
        (is (= (vec expected-sum) (vec (gpu-link/download executable sum-node))))
        (is (= (vec expected-difference)
               (vec (gpu-link/download executable difference-node))))
        (finally (gpu-link/close! executable))))))

(deftest attached-close-attempts-the-entire-reverse-order-teardown
  (let [calls (atom [])
        executable (gpu-link/map->LinkedExecutable
                    {:session ::caller-session
                     :owns-session? false
                     :graph-key :graph
                     :phases [:phase-0 :phase-1]
                     :allocation-keys [:allocation-0 :allocation-1]
                     :closed? (atom false)})]
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
    (is (= [[:graph :graph]
            [:phase :phase-1] [:phase :phase-0]
            [:allocation :allocation-1] [:allocation :allocation-0]]
           @calls))
    (is (nil? (gpu-link/close! executable)) "close remains idempotent after a destructor failure")))

(deftest linked-device-input-is-zero-copy-or-device-to-device
  (let [destination-buffer (Object.)
        source-buffer (Object.)
        destination-allocation
        (bview/allocation {:id :destination :byte-size 16 :memory-space :device
                           :device :ze:0 :ownership :owned})
        source-allocation
        (bview/allocation {:id :source :byte-size 16 :memory-space :device
                           :device :ze:0 :ownership :external})
        destination-view (bview/view destination-allocation {:dtype :float :shape [4]})
        source-view (bview/view source-allocation {:dtype :float :shape [4]})
        resident (gpu/->ResidentBufferView :session :destination-key destination-view)
        executable (gpu-link/map->LinkedExecutable
                    {:plan {:target :ze:0
                            :nodes {:destination {:id :destination :role :input
                                                  :view destination-view}}}
                     :session ::session
                     :node-views {:destination resident}
                     :pending-inputs (atom #{:destination})
                     :closed? (atom false)})
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
        (is (empty? @(:pending-inputs executable)))))))

(deftest owned-runtime-allocation-preserves-the-certified-identity
  (let [session (atom {:device-id :ze:0 :session-id :session
                       :buffers {} :allocations {} :closed? false})
        buffer {:byte-size 16 :alignment 64 :dtype :float :n-elements 4}]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'alloc-buffers-transactional)
       (fn [specs _device]
         (is (= {:node [:float 4 nil {:allocation-id :certified
                                      :memory-space :device
                                      :coherence :device-only
                                      :alignment 64}]}
                specs))
         {:node buffer})}
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
                       :graphs {:profiled {::gpu/recorded-graph true
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
      (testing "the descriptor is a clean 2-step elementwise chain"
        (is (= 2 (count steps)))
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
                                       :gemm-precision :f16-xmx :on-non-resident :nil)
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
          executable (gpu-link/instantiate! plan {:session session})]
      (is (= [:gemm] (mapv :convention (:steps prog))))
      (try
        (let [session (:session executable)
              phases (:phases executable)
              prepared-count
              (reduce + (map #(count (get-in @session [:prepared % :prepareds])) phases))
              private-count
              (reduce + (map #(count (get-in @session [:prepared % :temporary-buffers])) phases))]
          (is (> prepared-count 2)
              "each semantic GEMM selects and flattens a multi-kernel schedule")
          (is (pos? private-count) "conversion/layout storage stays step-private")
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"have not been initialized"
                                (gpu-link/run! executable))
              "owned dynamic inputs must be uploaded before the first replay")
          (gpu-link/upload! executable :x0 x0)
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
