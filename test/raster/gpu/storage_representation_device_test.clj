(ns raster.gpu.storage-representation-device-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.storage-representation :as probe]
            [raster.compiler.core.dtype :as dtype]
            [raster.core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.par :as par]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.value :as value]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze])
  (:import [java.lang.foreign MemorySegment]))

(deftm double-fold-float-storage
  [x :- (Array float)] :- (Array float)
  (let [out (arrays/zeros-like x 1)]
    (par/map! out i 1 nil
      (float
       (loop [j 0 acc (double 0.0)]
         (if (< j 3)
           (recur (inc j) (+ acc (double (aget x j))))
           acc))))))

(defn- check-declared-local-storage! [target]
  (let [input (float-array [16777216.0 1.0 -16777216.0])
        prepared (compiled/lower #'double-fold-float-storage [input]
                                 {:compiler :equation-first :target target :dtype :double
                                  :preserve-declared-array-storage? true})
        executable (compiled/instantiate! prepared)]
    (try
      (doseq [middle [1.0 3.0]]
        (aset input 1 (float middle))
        (let [outputs (compiled/invoke-compiled executable {:x input})
              actual (value/->host (:result outputs))]
          (is (= (class input) (class actual)))
          (is (= (vec (double-fold-float-storage input)) (vec actual)))
          (is (= [(float middle)] (vec actual))
              "the Double carry must not round each step to the Float destination type")))
      (finally (compiled/close! executable)))))

(deftest declared-local-float-storage-with-double-fold-on-opencl
  (if @opencl/opencl-fp64-available?
    (check-declared-local-storage! :ocl:0)
    (opencl/opencl-skip! "declared local Float storage with Double fold" :fp64)))

(deftest declared-local-float-storage-with-double-fold-on-level-zero
  (if @ze/gpu-available?
    (let [caps ((requiring-resolve 'raster.gpu.ze-runtime/module-capabilities))]
      (if (:fp64? caps)
        (check-declared-local-storage! :ze:0)
        (ze/gpu-capability-skip! "declared local Float storage with Double fold" :fp64? caps)))
    (ze/gpu-skip! "declared local Float storage with Double fold")))

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
