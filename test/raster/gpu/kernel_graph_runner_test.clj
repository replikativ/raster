(ns raster.gpu.kernel-graph-runner-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.gpu.core :as gpu]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.runtime.hardware :as hardware]))

(defn- probe-artifact
  []
  (artifact/make
   {:kernel-name "bound_call_probe"
    :source (str "__kernel void bound_call_probe(__global const float* x, "
                 "__global float* out, int n) {}")
    :abi [(kabi/slot 'x :input :float :role :operand)
          (kabi/slot 'out :output :float :role :result)
          (kabi/slot 'n :scalar :int :role :bound)]
    :arguments '[x out n]
    :launch (launch/spec {:workgroup-size [64]
                          :group-count [(launch/ceil-div 'n 64)]})
    :effects {:kind :elementwise-map :reads ['x] :writes ['out]}}))

(defn- emitted-graph []
  (let [node (soac/par-form->soac
              'scan-result
              '(raster.par/scan out acc 0.0 i n float (+ acc (aget values i)))
              92)
        operations (lower/lower-scan node nil :dtype :float)]
    (emit/generate-scan-kernel-graph
     (lower/scan-kernel-graph
      node operations {:array-types {'values :float 'out :float}}))))

(deftest integrated-opencl-allocation-still-uploads-through-cl-mem
  (let [source (float-array [1.0 2.0])
        uploaded (atom [])
        mock-buffer (Object.)
        resolver (fn [_device-id name]
                   (case name
                     "make-buffer" (fn [_n _dtype] mock-buffer)
                     "array->buffer!" (fn [buffer array]
                                        (swap! uploaded conj [buffer array])
                                        buffer)
                     "buffer-as-float-buffer"
                     (fn [_] (throw (ex-info "OpenCL staging is not coherent cl_mem" {})))
                     "buffer-as-int-buffer"
                     (fn [_] (throw (ex-info "OpenCL staging is not coherent cl_mem" {})))))]
    (with-redefs-fn
      {#'hardware/memory-topology (constantly {:model :unified :integrated? true})
       (ns-resolve 'raster.gpu.core 'rt-resolve) resolver}
      (fn []
        (let [allocated ((ns-resolve 'raster.gpu.core 'alloc-buffers-internal)
                         {:input [:float 2 source]} :ocl:0)]
          (is (identical? mock-buffer (:input allocated)))
          (is (= [[mock-buffer source]] @uploaded)))))))

(deftest session-runner-owns-only-graph-temporaries-and-bound-driver-objects
  (let [graph (emitted-graph)
        values-buffer {:id :values-buffer :dtype :float :n-elements 1025 :byte-size 4100}
        output-buffer {:id :output-buffer :dtype :float :n-elements 1025 :byte-size 4100}
        allocation (fn [id]
                     (bview/allocation {:id id :byte-size 4100 :memory-space :shared
                                        :device :ze:0 :coherence :host-coherent
                                        :ownership :owned}))
        registered (atom [])
        bound (atom [])
        recorded (atom [])
        submitted (atom [])
        awaited (atom [])
        released-events (atom [])
        destroyed-graphs (atom [])
        destroyed-prepareds (atom [])
        freed (atom [])
        sess (atom {:device-id :ze:0
                    :session-id :test-session
                    :buffers {:values values-buffer :out output-buffer}
                    :allocations {:values (allocation :values-allocation)
                                  :out (allocation :out-allocation)}
                    :kernel-graphs {}
                    :events {}
                    :closed? false})
        resolver
        (fn [_device-id name]
          (case name
            "make-buffer" (fn [n dtype] {:temporary true :elements n :dtype dtype})
            "array->buffer!" (fn [buffer _] buffer)
            "buffer-as-float-buffer" identity
            "buffer-as-int-buffer" identity
            "free-buffer!" #(swap! freed conj %)
            "register-kernel!" (fn [kernel-name artifact]
                                 (swap! registered conj [kernel-name artifact]))
            "bind-kernel-call" (fn [call & _]
                                 (let [prepared {:mock-call call}]
                                   (swap! bound conj prepared)
                                   prepared))
            "record-graph!" (fn [prepareds opts]
                              (let [recording {:prepareds prepareds :opts opts}]
                                (swap! recorded conj recording)
                                recording))
            "submit-graph!" (fn [graph]
                              (let [event {:submitted graph}]
                                (swap! submitted conj event)
                                event))
            "await-event!" #(swap! awaited conj %)
            "event-complete?" (constantly true)
            "release-event!" #(swap! released-events conj %)
            "reset-graph-events!" (fn [_])
            (throw (ex-info "unexpected mocked runtime function" {:name name}))))
        soft-resolver
        (fn [_device-id name]
          (case name
            "destroy-graph!" #(swap! destroyed-graphs conj %)
            "destroy-prepared!" #(swap! destroyed-prepareds conj %)
            nil))]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) resolver
       (ns-resolve 'raster.gpu.core 'rt-resolve-soft) soft-resolver}
      (fn []
        (let [handle (gpu/bind-kernel-graph!
                      sess :prefix graph {'values :values 'out :out}
                      {'n {:type :int :value 1025}
                       'enclosing-count {:type :long :value 1025}
                       '(extent values) {:type :long :value 1025}} {:profile? true})
              event (gpu/submit-kernel-graph! sess handle)
              _ (is (gpu/gpu-event? event))
              _ (is (gpu/event-complete? sess event))
              _ (is (empty? @released-events)
                    "a positive status query is not a host wait and must not consume the event")
              _ (is (thrown-with-msg? clojure.lang.ExceptionInfo #"in-flight submission"
                                      (gpu/submit-kernel-graph! sess handle)))
              async-outputs (gpu/await-event! sess event)
              _ (is (gpu/event-complete? sess event))
              _ (gpu/release-event! sess event)
              outputs (gpu/run-kernel-graph! sess handle)
              temporary (first (vals (get-in @sess [:kernel-graphs :prefix
                                                    :temporary-buffers])))]
          (is (gpu/kernel-graph-handle? handle))
          (let [info (gpu/kernel-graph-execution-info sess handle)]
            (is (= :kernel-graph (:kind info)))
            (is (= :fixed (:selection info)))
            (is (= [] (:admission info)))
            (is (= (mapv #(get-in % [:operation :kernel-name]) (:nodes graph))
                   (:entry-points info))))
          (is (= {'n {:type :int :value 1025}}
                 (get-in @sess [:kernel-graphs :prefix :graph-call :scalar-values]))
              "enclosing shape bookkeeping does not widen the executable scalar ABI")
          (is (= 3 (count @registered)))
          (is (= 3 (count @bound)))
          (is (= 1 (count @recorded)))
          (is (= {:barriers? true :profile? true} (get-in @recorded [0 :opts])))
          (is (= 2 (count @submitted)))
          (is (= @submitted @awaited @released-events))
          (is (empty? (:events @sess)))
          (is (identical? output-buffer (get async-outputs 'out)))
          (is (identical? output-buffer (get outputs 'out)))
          (gpu/submit-kernel-graph! sess handle)
          (gpu/release-kernel-graph! sess handle)
          (is (= 3 (count @submitted)))
          (is (= @submitted @awaited @released-events)
              "graph release waits and releases an in-flight submission")
          (is (empty? (:events @sess)))
          (is (empty? (:kernel-graphs @sess)))
          (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not bound"
                               (gpu/kernel-graph-execution-info sess handle)))
          (is (= 1 (count @destroyed-graphs)))
          (is (= 3 (count @destroyed-prepareds)))
          (is (= [temporary] @freed))
          (is (not-any? #(or (identical? values-buffer %)
                             (identical? output-buffer %))
                        @freed)))))))

(deftest one-kernel-call-binds-validates-and-measures-through-the-graph-runtime
  (let [input-buffer {:dtype :float :n-elements 128 :byte-size 512}
        output-buffer {:dtype :float :n-elements 128 :byte-size 512}
        allocation (fn [id]
                     (bview/allocation {:id id :byte-size 512 :memory-space :shared
                                        :device :ze:0 :coherence :host-coherent
                                        :ownership :owned}))
        session (atom {:device-id :ze:0 :session-id :call-session
                       :buffers {:x input-buffer :out output-buffer}
                       :allocations {:x (allocation :x) :out (allocation :out)}
                       :kernel-graphs {} :events {} :closed? false})
        registered (atom [])
        resets (atom 0)
        replays (atom 0)
        destroyed (atom [])
        resolver
        (fn [_ name]
          (case name
            "register-kernel!" (fn [kernel-name emitted]
                                 (swap! registered conj [kernel-name emitted]))
            "bind-kernel-call" (fn [call & _] {:kernel-call call})
            "record-graph!" (fn [prepared opts]
                              {:prepared prepared :profile? (:profile? opts)})
            "submit-graph!" (fn [graph] {:graph graph})
            "await-event!" identity
            "event-complete?" (constantly true)
            "release-event!" identity
            "reset-graph-events!" (fn [_] (swap! resets inc))
            "replay-graph!" (fn [_] (swap! replays inc))
            "read-graph-timestamps!" (fn [_] {:wall-ms 0.001})
            (throw (ex-info "unexpected mocked runtime function" {:name name}))))
        soft-resolver
        (fn [_ name]
          (case name
            "destroy-graph!" #(swap! destroyed conj [:graph %])
            "destroy-prepared!" #(swap! destroyed conj [:prepared %])
            nil))]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) resolver
       (ns-resolve 'raster.gpu.core 'rt-resolve-soft) soft-resolver}
      (fn []
        (let [emitted (probe-artifact)
              handle (gpu/bind-kernel-call!
                      session :candidate emitted
                      [:x :out {:type :int :value 128}]
                      {:profile? true})
              outputs (gpu/run-kernel-graph! session handle)
              measured (gpu/measure-bound-kernel-graph!
                        session handle :warmup-iterations 1 :budget-ms 1
                        :min-samples 3 :max-samples 3)]
          (is (= [["bound_call_probe" emitted]] @registered))
          (is (= #{:x :out}
                 (get-in @session [:kernel-graphs :candidate
                                   :resident-footprint :buffer-keys])))
          (is (= #{:x :out}
                 (set (map :key (vals (get-in @session
                                              [:kernel-graphs :candidate :resident-views]))))))
          (is (= {:kind :kernel-artifact :strategy nil :precision nil
                  :entry-points ["bound_call_probe"] :selection :fixed :admission []}
                 (gpu/kernel-graph-execution-info session handle)))
          (is (identical? output-buffer (get outputs 'out)))
          (is (= 1 @resets) "validation replay resets discarded profiling events")
          (is (= :device-event (:timing-source measured)))
          (is (= 1000.0 (:min-ns measured)))
          (doseq [[field value reason] [[:session-id :foreign :foreign-graph-handle]
                                        [:generation :retired :stale-graph-handle]]]
            (is (= reason
                   (try (gpu/release-kernel-graph! session (assoc handle field value))
                        (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
          (is (empty? @destroyed) "foreign/stale handles do not destroy the live generation")
          (is (= (+ 1 5 3) @replays))
          (gpu/release-kernel-graph! session handle)
          (is (= 2 (count @destroyed))))))))

(deftest opencl-sub-buffer-handles-follow-the-graph-lifetime
  (let [n 1025
        n-bytes (* 4 n)
        output-offset 4224
        graph (emitted-graph)
        root-buffer {:root true :dtype :float
                     :n-elements (quot (+ output-offset n-bytes) 4)
                     :byte-size (+ output-offset n-bytes)}
        allocation (bview/allocation
                    {:id :shared-allocation :byte-size (:byte-size root-buffer)
                     :memory-space :device :device :ocl:0 :coherence :explicit-transfer
                     :ownership :owned})
        sess (atom {:device-id :ocl:0 :session-id :test-session
                    :buffers {:storage root-buffer}
                    :allocations {:storage allocation}
                    :kernel-graphs {} :events {} :closed? false})
        input (gpu/buffer-view sess :storage {:shape [n]})
        output (gpu/buffer-view sess :storage {:byte-offset output-offset :shape [n]})
        slices (atom [])
        freed (atom [])
        fail-bind? (atom false)
        resolver
        (fn [_device-id name]
          (case name
            "make-buffer" (fn [elements dtype]
                            {:temporary true :n-elements elements
                             :byte-size (* elements 4) :dtype dtype})
            "array->buffer!" (fn [buffer _] buffer)
            "buffer-as-float-buffer" identity
            "buffer-as-int-buffer" identity
            "slice-buffer" (fn [buffer byte-offset byte-length dtype]
                             (let [slice {:sub-buffer true :parent buffer
                                          :byte-offset byte-offset :byte-size byte-length
                                          :n-elements (quot byte-length 4) :dtype dtype}]
                               (swap! slices conj slice)
                               slice))
            "free-buffer!" #(swap! freed conj %)
            "register-kernel!" (fn [& _])
            "bind-kernel-call" (fn [call & _]
                                 (when @fail-bind?
                                   (throw (ex-info "mock bind failure" {})))
                                 {:mock-call call})
            "record-graph!" (fn [prepareds opts] {:prepareds prepareds :opts opts})
            (throw (ex-info "unexpected mocked runtime function" {:name name}))))
        soft-resolver (fn [_device-id name]
                        (case name
                          "destroy-graph!" (fn [_])
                          "destroy-prepared!" (fn [_])
                          nil))]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) resolver
       (ns-resolve 'raster.gpu.core 'rt-resolve-soft) soft-resolver}
      (fn []
        (let [handle (gpu/bind-kernel-graph!
                      sess :view-scan graph {'values input 'out output}
                      {'n {:type :int :value n}})
              [input-sub-buffer output-sub-buffer] @slices
              temporary (first (vals (get-in @sess [:kernel-graphs :view-scan
                                                    :temporary-buffers])))]
          (is (= [[0 n-bytes] [output-offset n-bytes]]
                 (mapv (juxt :byte-offset :byte-size) @slices))
              "both proper subranges retain their exact physical extent")
          (is (identical? output-sub-buffer
                          (get-in @sess [:kernel-graphs :view-scan :outputs 'out])))
          (gpu/release-kernel-graph! sess handle)
          (is (= #{input-sub-buffer output-sub-buffer temporary} (set @freed)))
          (is (not-any? #(identical? root-buffer %) @freed)
              "the session-owned root allocation survives graph release"))
        (reset! slices [])
        (reset! freed [])
        (reset! fail-bind? true)
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"mock bind failure"
                              (gpu/bind-kernel-graph!
                               sess :failed-view-scan graph {'values input 'out output}
                               {'n {:type :int :value n}})))
        (is (= 2 (count @slices)))
        (is (= 3 (count @freed))
            "failed binding releases both created sub-buffers and its temporary")
        (is (= 2 (count (filter :sub-buffer @freed))))
        (is (some :temporary @freed))
        (is (empty? (:kernel-graphs @sess)))))))

(deftest failed-construction-adopts-ownership-even-with-suppression-disabled
  (let [buffer {:dtype :float :n-elements 128 :byte-size 512}
        allocation (bview/allocation {:id :root :byte-size 512 :memory-space :shared
                                      :device :ze:0 :coherence :host-coherent :ownership :owned})
        sess (atom {:device-id :ze:0 :session-id :rollback :closed? false :events {}
                    :buffers {:x buffer :out buffer} :allocations {:x allocation :out allocation}
                    :kernel-graphs {} :prepared {} :graphs {}})
        primary (proxy [RuntimeException] ["binding failed" nil false false])
        destruction (ex-info "native outcome unknown" {})
        destroyed (atom 0) freed (atom [])
        resolver (fn [_ name]
                   (case name
                     "register-kernel!" (fn [& _])
                     "record-graph!" (fn [& _] (throw (AssertionError. "unreachable after failed bind")))
                     "bind-kernel-call"
                     (fn [_ {:keys [adopt-cleanup!]}]
                       (cleanup/construct! :kernel (constantly :native)
                                           (fn [_] (swap! destroyed inc) (throw destruction))
                                           (fn [_ _] (throw primary)) adopt-cleanup!))
                     "free-buffer!" #(swap! freed conj %)
                     (throw (ex-info "unexpected runtime call" {:name name}))))]
    (with-redefs-fn {(ns-resolve 'raster.gpu.core 'rt-resolve) resolver}
      (fn []
        (is (identical? primary
                        (try (gpu/bind-kernel-call! sess :failed (probe-artifact)
                                                   [:x :out {:type :int :value 128}])
                             (catch Throwable e e))))
        (is (= 1 @destroyed))
        (is (= 1 (count (:kernel-graphs @sess))))
        (is (= #{:x :out} (-> @sess :kernel-graphs vals first :resident-footprint :buffer-keys)))
        (is (thrown? clojure.lang.ExceptionInfo (gpu/free-buffer! sess :x)))
        (dotimes [_ 2]
          (is (identical? destruction (try (gpu/close-session! sess) (catch Throwable e e)))))
        (is (= 1 @destroyed) "unknown destruction outcome is never retried")
        (is (empty? @freed))
        (is (true? (:closed? @sess)))
        (is (= :releasing (:lifecycle @sess)))
        (is (thrown? clojure.lang.ExceptionInfo (gpu/upload! sess :x (float-array 128))))
        (is (thrown? clojure.lang.ExceptionInfo
                     (gpu/bind-kernel-call! sess :new (probe-artifact) [:x :out {:type :int :value 128}])))
        (is (= #{:x :out} (set (keys (:buffers @sess)))))))))

(deftest partial-session-close-retires-successes-before-a-later-failure
  (let [calls (atom []) fault (ex-info "kernel outcome unknown" {})
        resolver (fn [_ name]
                   (case name
                     "destroy-graph!" (fn [graph] (swap! calls conj [:graph graph]))
                     "destroy-prepared!" (fn [prepared]
                                           (swap! calls conj [:kernel prepared])
                                           (when (= :bad prepared) (throw fault)))
                     (throw (ex-info "unexpected destructor" {:name name}))))]
    (with-redefs-fn {(ns-resolve 'raster.gpu.core 'rt-resolve-soft) resolver}
      (fn []
        (doseq [order [[:good :bad] [:bad :good]]]
        (reset! calls [])
        (let [own (ns-resolve 'raster.gpu.core 'own-kernel-graph-entry)
              good (own :ze:0 {:runtime-graph :good :prepareds [:good]})
              bad (own :ze:0 {:runtime-graph :bad :prepareds [:bad]})
              sess (atom {:device-id :ze:0 :closed? false :events {} :graphs {} :prepared {}
                          :kernel-graphs (into (array-map) (map (fn [id] [id (get {:good good :bad bad} id)]) order))})]
          (dotimes [_ 2]
            (is (identical? fault (try (gpu/close-session! sess) (catch Throwable e e)))))
          (is (= (vec (mapcat (fn [id] [[:graph id] [:kernel id]]) order)) @calls))
          (is (= [:bad] (vec (keys (:kernel-graphs @sess)))))
          (is (true? (:closed? @sess)))
          (is (= :releasing (:lifecycle @sess)))))))))

(deftest close-waits-for-a-modern-profile-before-destroying-its-native-recording
  (let [started (promise) proceed (promise) closing (promise) calls (atom [])
        sess (atom {:device-id :ze:0 :session-id :profile-race :closed? false :events {}
                    :graphs {} :prepared {} :buffers {} :allocations {} :kernel-graphs {}})
        resolver (fn [_ name]
                   (case name
                     "destroy-graph!" (fn [_] (swap! calls conj :destroy))
                     "free-buffer!" (fn [_])
                     "close-kernel-arena!" (fn [_])
                     (throw (ex-info "unexpected runtime call" {:name name}))))]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) resolver
       (ns-resolve 'raster.gpu.core 'rt-resolve-soft) resolver
       (ns-resolve 'raster.gpu.core 'profile-runtime-graph!)
       (fn [_ _]
         (swap! calls conj [:profile-lock (Thread/holdsLock sess)])
         (deliver started true)
         (when (= :timeout (deref proceed 5000 :timeout))
           (throw (ex-info "test profile timed out" {})))
         (swap! calls conj :profile-complete)
         :profile)}
      (fn []
        (let [own (ns-resolve 'raster.gpu.core 'own-kernel-graph-entry)
              entry (own :ze:0 {:runtime-graph :recording :prepareds [] :profile? true})
              handle (gpu/->KernelGraphHandle :profile :profile-race (:generation entry))
              _ (swap! sess assoc-in [:kernel-graphs :profile] entry)
              profile (future (gpu/profile-bound-kernel-graph! sess handle))]
          (try
            (is (= true (deref started 5000 :timeout)))
            (let [close (future (deliver closing true) (gpu/close-session! sess))]
              (is (= true (deref closing 5000 :timeout)))
              (is (= :blocked (deref close 50 :blocked)))
              (is (= [[:profile-lock true]] @calls))
              (deliver proceed true)
              (is (= :profile (deref profile 5000 :timeout)))
              (is (not= :timeout (deref close 5000 :timeout))))
            (is (= [[:profile-lock true] :profile-complete :destroy] @calls))
            (is (= :closed (:lifecycle @sess)))
            (finally (deliver proceed true))))))))
