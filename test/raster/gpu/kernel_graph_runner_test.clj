(ns raster.gpu.kernel-graph-runner-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.gpu.core :as gpu]
            [raster.runtime.hardware :as hardware]))

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
        values-buffer (Object.)
        output-buffer (Object.)
        registered (atom [])
        bound (atom [])
        recorded (atom [])
        replayed (atom [])
        destroyed-graphs (atom [])
        destroyed-prepareds (atom [])
        freed (atom [])
        sess (atom {:device-id :ze:0
                    :buffers {:values values-buffer :out output-buffer}
                    :kernel-graphs {}
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
            "replay-graph!" #(swap! replayed conj %)
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
              outputs (gpu/run-kernel-graph! sess handle)
              temporary (first (vals (get-in @sess [:kernel-graphs :prefix
                                                    :temporary-buffers])))]
          (is (gpu/kernel-graph-handle? handle))
          (is (= 3 (count @registered)))
          (is (= 3 (count @bound)))
          (is (= 1 (count @recorded)))
          (is (= {:barriers? true} (get-in @recorded [0 :opts])))
          (is (= 1 (count @replayed)))
          (is (identical? output-buffer (get outputs 'out)))
          (gpu/release-kernel-graph! sess handle)
          (is (empty? (:kernel-graphs @sess)))
          (is (= 1 (count @destroyed-graphs)))
          (is (= 3 (count @destroyed-prepareds)))
          (is (= [temporary] @freed))
          (is (not-any? #(or (identical? values-buffer %)
                             (identical? output-buffer %))
                        @freed)))))))
