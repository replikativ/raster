(ns raster.gpu.kernel-graph-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.arrays :as arrays]
            [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.core :refer [deftm]]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as device-probe]
            [raster.par :as par]))

(deftm double-values!
  [values :- (Array float), scratch :- (Array float), n :- Long] :- Void
  (par/map-void! i n
                 (arrays/aset scratch i (* 2.0 (arrays/aget values i)))))

(defn- emitted-graph []
  (let [node (soac/par-form->soac
              'scan-result
              '(raster.par/scan out acc 0.0 i n float (+ acc (aget values i)))
              103)
        operations (lower/lower-scan node nil :dtype :float)]
    (emit/generate-scan-kernel-graph
     (lower/scan-kernel-graph
      node operations {:array-types {'values :float 'out :float}}))))

(defn- run-inclusive-scan
  [device-id]
  (let [n 1025
        graph (emitted-graph)
        input (float-array n 1.0)]
    (gpu/with-gpu-session [sess device-id]
      (gpu/alloc! sess {:values [:float n input]
                        :out [:float n nil]})
      (let [handle (gpu/bind-kernel-graph!
                    sess :inclusive-scan graph {'values :values 'out :out}
                    {'n {:type :int :value n}})]
        (try
          (let [event (gpu/submit-kernel-graph! sess handle)]
            (try
              (is (boolean? (gpu/event-complete? sess event)))
              (gpu/await-event! sess event)
              (is (gpu/event-complete? sess event))
              (gpu/download sess :out)
              (finally
                (gpu/release-event! sess event))))
          (finally
            (gpu/release-kernel-graph! sess handle)))))))

(defn- assert-prefix!
  [device-id]
  (let [result (run-inclusive-scan device-id)]
    (is (= 1025.0 (double (aget ^floats result 1024))))
    (is (every? true?
                (map-indexed (fn [i value] (= (double (inc i)) (double value))) result)))))

(deftest level-zero-kernel-graph-inclusive-scan
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "Level Zero KernelGraph inclusive scan")
    (assert-prefix! :ze:0)))

(deftest level-zero-kernel-graph-binds-disjoint-views-of-one-allocation
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "Level Zero KernelGraph BufferView aliases")
    (let [n 1025
          graph (emitted-graph)
          storage (float-array (* 2 n))]
      (dotimes [i n] (aset storage i 1.0))
      (gpu/with-gpu-session [sess :ze:0]
        (gpu/alloc! sess {:storage [:float (* 2 n) storage]})
        (let [input (gpu/buffer-view sess :storage {:shape [n]})
              output (gpu/buffer-view sess :storage {:byte-offset (* 4 n) :shape [n]})
              handle (gpu/bind-kernel-graph!
                      sess :view-scan graph {'values input 'out output}
                      {'n {:type :int :value n}})]
          (try
            (gpu/run-kernel-graph! sess handle)
            (let [result (float-array n)]
              (gpu/download-range! sess output result {:elements n})
              (is (= 1025.0 (double (aget result 1024))))
              (is (every? true?
                          (map-indexed (fn [i value] (= (double (inc i)) (double value)))
                                       result))))
            (finally
              (gpu/release-kernel-graph! sess handle))))))))

(deftest opencl-kernel-graph-inclusive-scan
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "OpenCL KernelGraph inclusive scan")
    (assert-prefix! :ocl:0)))

(defn- assert-mixed-recording!
  [device-id]
  (let [n 1025]
    (gpu/with-gpu-session [sess device-id]
      (gpu/compile! sess :double #'double-values!)
      (gpu/alloc! sess {:values [:float n nil]
                        :scratch [:float n nil]
                        :out [:float n nil]})
      (gpu/prepare! sess :double {"values" :values "scratch" :scratch} [] n)
      (let [handle (gpu/bind-kernel-graph!
                    sess :scan (emitted-graph) {'values :scratch 'out :out}
                    {'n {:type :int :value n}})]
        (try
          (gpu/record-bound-sequence!
           sess [{:kind :phase :phase :double}
                 {:kind :graph :handle handle}]
           :mixed)
          (try
            (is (= 4 (count (get-in (gpu/graph-execution-order sess :mixed)
                                    [:per-replay]))))
            (doseq [factor [1.0 3.0]]
              (gpu/upload! sess :values (float-array n factor))
              (gpu/replay! sess :mixed)
              (let [result (gpu/download sess :out)]
                (is (= (* 2.0 factor n) (double (aget ^floats result (dec n)))))))
            (finally (gpu/release-recorded-graph! sess :mixed)))
          (finally (gpu/release-kernel-graph! sess handle)
                   (gpu/release-prepared! sess :double)))))))

(deftest opencl-bound-phase-and-emitted-graph-one-replay
  (if @device-probe/opencl-available?
    (assert-mixed-recording! :ocl:0)
    (device-probe/opencl-skip! "mixed bound replay")))

(deftest level-zero-bound-phase-and-emitted-graph-one-replay
  (if @gp/gpu-available?
    (assert-mixed-recording! :ze:0)
    (gp/gpu-skip! "mixed bound replay on Level Zero")))

(deftest opencl-kernel-graph-binds-aligned-disjoint-views-of-one-allocation
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "OpenCL KernelGraph BufferView aliases")
    (let [n 1025
          n-bytes (* 4 n)
          alignment ((resolve 'raster.gpu.ocl-runtime/buffer-offset-alignment))
          output-offset (* alignment (quot (+ n-bytes (dec alignment)) alignment))
          graph (emitted-graph)
          storage (float-array (quot (+ output-offset n-bytes 4) 4))]
      (is (zero? (mod output-offset alignment)))
      (dotimes [i n] (aset storage i 1.0))
      (gpu/with-gpu-session [sess :ocl:0]
        (gpu/alloc! sess {:storage [:float (alength storage) storage]})
        (is (= alignment (get-in @sess [:allocations :storage :alignment]))
            "the canonical allocation exposes the queried OpenCL alignment")
        (let [input (gpu/buffer-view sess :storage {:shape [n]})
              output (gpu/buffer-view sess :storage
                                      {:byte-offset output-offset :shape [n]})
              handle (gpu/bind-kernel-graph!
                      sess :opencl-view-scan graph {'values input 'out output}
                      {'n {:type :int :value n}})]
          (try
            (gpu/run-kernel-graph! sess handle)
            (let [result (float-array n)]
              (gpu/download-range! sess output result {:elements n})
              (is (= 1025.0 (double (aget result 1024))))
              (is (every? true?
                          (map-indexed (fn [i value] (= (double (inc i)) (double value)))
                                       result))))
            (finally
              (gpu/release-kernel-graph! sess handle)))
          (when (> alignment 4)
            (let [misaligned-output (gpu/buffer-view
                                     sess :storage
                                     {:byte-offset (+ output-offset 4) :shape [n]})]
              (is (thrown-with-msg?
                   clojure.lang.ExceptionInfo #"alignment requirement"
                   (gpu/bind-kernel-graph!
                    sess :misaligned-view-scan graph
                    {'values input 'out misaligned-output}
                    {'n {:type :int :value n}})))
              (is (empty? (:kernel-graphs @sess))
                  "misaligned binding leaves no graph-owned resources"))))))))
