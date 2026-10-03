(ns raster.gpu.storage-representation-device-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.storage-representation :as probe]
            [raster.compiler.core.dtype :as dtype]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze])
  (:import [java.lang.foreign MemorySegment]))

(deftest optional-skips-require-explicit-live-absence
  (with-redefs [ze/gpu-available? (delay true)]
    (doseq [[capability facts] [[:fp16? {}] [:fp16? {:fp16? nil}]
                               [:fp16? {:fp16? true}] [:unknown {:unknown false}]]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (ze/gpu-capability-skip! "invalid synthetic skip" capability facts)))))
  (with-redefs [ze/gpu-available? (delay false)]
    (is (thrown? clojure.lang.ExceptionInfo
                 (ze/gpu-capability-skip! "no live device" :fp64? {:fp64? false})))))

(defn- check-storage! [target dt]
  (let [session (gpu/make-session target)
        output (byte-array (* 2 (dtype/bytes-of dt)))]
    (try
      (gpu/alloc! session {:probe [dt 2 nil]})
      (let [handle (gpu/bind-kernel-call! session :probe
                                         (probe/emit-artifact dt :opencl-portable) [:probe])]
        (try
          (gpu/run-kernel-graph! session handle)
          (gpu/download-range! session :probe (MemorySegment/ofArray output) {:elements 2})
          (let [observation (mapv #(bit-and 255 %) output)
                order (probe/classify-bytes dt observation)]
            (is (contains? #{:little-endian :big-endian :order-invariant} order))
            (is (= observation (probe/expected-bytes dt
                                 (if (= :order-invariant order) :little-endian order)))))
          (finally (gpu/release-kernel-graph! session handle))))
      (gpu/free-buffer! session :probe)
      (is (empty? (:kernel-graphs @session)))
      (is (empty? (:events @session)))
      (is (empty? (:buffers @session)))
      (finally (gpu/close-session! session)))))

(deftest typed-storage-probes-on-opencl
  (if @opencl/opencl-available?
    (doseq [dt (keys dtype/dtype-info)]
      (testing (str dt)
        (let [capability (case dt :half :fp16 :double :fp64 nil)]
          (if (or (nil? capability)
                  (opencl/capability-supported? (:device @opencl/opencl-status) capability))
            (check-storage! :ocl:0 dt)
            (opencl/opencl-skip! (str "storage representation " dt) capability)))))
    (opencl/opencl-skip! "typed storage representation probes")))

(deftest typed-storage-probes-on-level-zero
  (if @ze/gpu-available?
    (let [caps ((requiring-resolve 'raster.gpu.ze-runtime/module-capabilities))]
      (doseq [dt (keys dtype/dtype-info)]
        (testing (str dt)
          (if (case dt :half (:fp16? caps) :double (:fp64? caps) true)
            (check-storage! :ze:0 dt)
            (ze/gpu-capability-skip! (str "storage representation " dt)
                                     (case dt :half :fp16? :double :fp64?) caps)))))
    (ze/gpu-skip! "typed storage representation probes")))
