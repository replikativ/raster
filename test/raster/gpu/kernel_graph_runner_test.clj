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
        values-buffer {:dtype :float :n-elements 1025 :byte-size 4100}
        output-buffer {:dtype :float :n-elements 1025 :byte-size 4100}
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
            "bind-kernel-call" (fn [call]
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
                      {'n {:type :int :value 1025}})
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
          (is (= 3 (count @registered)))
          (is (= 3 (count @bound)))
          (is (= 1 (count @recorded)))
          (is (= {:barriers? true} (get-in @recorded [0 :opts])))
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
            "bind-kernel-call" (fn [call] {:kernel-call call})
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
          (is (identical? output-buffer (get outputs 'out)))
          (is (= 1 @resets) "validation replay resets discarded profiling events")
          (is (= :device-event (:timing-source measured)))
          (is (= 1000.0 (:min-ns measured)))
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
            "bind-kernel-call" (fn [call]
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
              sub-buffer (first @slices)
              temporary (first (vals (get-in @sess [:kernel-graphs :view-scan
                                                    :temporary-buffers])))]
          (is (= 1 (count @slices)) "only the nonzero view needs a cl_mem sub-buffer")
          (is (identical? sub-buffer (get-in @sess [:kernel-graphs :view-scan
                                                    :outputs 'out])))
          (gpu/release-kernel-graph! sess handle)
          (is (= #{sub-buffer temporary} (set @freed)))
          (is (not-any? #(identical? root-buffer %) @freed)
              "the session-owned root allocation survives graph release"))
        (reset! slices [])
        (reset! freed [])
        (reset! fail-bind? true)
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"mock bind failure"
                              (gpu/bind-kernel-graph!
                               sess :failed-view-scan graph {'values input 'out output}
                               {'n {:type :int :value n}})))
        (is (= 1 (count @slices)))
        (is (= 2 (count @freed))
            "failed binding releases both its created sub-buffer and temporary")
        (is (some :sub-buffer @freed))
        (is (some :temporary @freed))
        (is (empty? (:kernel-graphs @sess)))))))
