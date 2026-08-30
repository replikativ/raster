(ns raster.gpu.ocl-session-test
  "Linked resident execution on OpenCL: compile → certify → instantiate → replay → download,
  on whatever OpenCL device is available. With
  RASTER_OCL_DEVICE_TYPE=cpu this runs on POCL/Intel-CPU — the no-GPU
  vendor-portability oracle (CUDA/HIP Phase A). Skips cleanly without OpenCL.

  Portability lesson encoded here: kernels must not read from :output-only
  buffers (uninitialized device memory — Intel's driver zeroes allocations,
  POCL's malloc does not)."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.par-opencl :as par-opencl]
            [raster.compiler.pipeline :as pl]
            [raster.arrays :as ra]
            [raster.core :refer [deftm]]
            [raster.dl.array-ops :as ops]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as device-probe]
            [raster.gpu.descriptor-fixture :as fixture]))

(deftest opencl-max-work-group-size-selector
  (require 'raster.gpu.ocl-runtime)
  (is (= 0x1004
         (var-get
          (ns-resolve 'raster.gpu.ocl-runtime 'CL_DEVICE_MAX_WORK_GROUP_SIZE)))
      "CL_DEVICE_MAX_WORK_GROUP_SIZE must not regress to the dimensions selector 0x1003"))

(deftm ocl-session-ax! (All [T] [x :- (Array T) y :- (Array T) a :- T n :- Long] :- Void
                            (raster.par/map-void! i n (ra/aset y i (* a (ra/aget x i))))))

(deftm ocl-session-map
  [x :- (Array float) y :- (Array float) a :- Float n :- Long] :- (Array float)
  (raster.par/map! y i n float (* a (ra/aget x i))))

(deftm ocl-session-contract
  [A :- (Array float) B :- (Array float)] :- (Array float)
  (let [C (ra/alloc-like A 64)]
    (raster.par/contract C [[i 8] [j 8]] [[l 8]]
                         (clojure.core/*
                          (ra/aget A (clojure.core/+ (clojure.core/* i 8) l))
                          (ra/aget B (clojure.core/+ (clojure.core/* l 8) j))))
    C))

(deftm ocl-session-reduce
  [x :- (Array float) out :- (Array float) scale :- Double n :- Long] :- Void
  (let [s (raster.par/reduce acc 0.0 i n
                             (clojure.core/+ acc (clojure.core/* scale (ra/aget x i))))]
    (raster.par/map-void! j n
                          (ra/aset out j (clojure.core/* (ra/aget x j) s)))))

(deftm ocl-session-scan
  [x :- (Array float) out :- (Array float) n :- Long] :- (Array float)
  (raster.par/scan out acc 0.0 i n float
                   (clojure.core/+ acc (ra/aget x i))))

(deftest ocl-map-void-mixed-storage-abi-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "mixed-storage map-void")
    (let [n 64
          q (byte-array (map #(byte (- (mod % 31) 15)) (range n)))
          out (int-array n)
          kernel (par-opencl/generate-par-map-void-kernel
                  '(raster.par/map-void! i n
                                         (aset out i (+ (int (aget q i)) limit)))
                  :dtype :float
                  :array-types {'out :int 'q :byte}
                  :scalar-types {'limit :int})
          register! (resolve 'raster.gpu.ocl-runtime/register-kernel!)
          invoke! (resolve 'raster.gpu.ocl-runtime/invoke-registered-map-void-kernel)]
      (register! (:kernel-name kernel) kernel)
      (invoke! (:kernel-name kernel) [out q] [{:type :int :value 7}] n)
      (is (every? true? (map-indexed (fn [i v] (= v (+ 7 (aget q i)))) out))))))

(deftest ocl-resident-session-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident session")
    (let [p (pl/compile-gpu-program #'ocl-session-ax! :ocl:0 :dtype :float)
          n 4096
          x (float-array (map float (range n)))
          y (float-array n)
          s (gpu/make-session :ocl:0)]
      (try
        (let [program (fixture/instantiate! s p [x y (float 2.0) n]
                                            {'x :input 'y :output})]
          (testing "replayed program computes correctly"
            (let [r (fixture/run! program [x y (float 2.0) n])
                  ^floats yg (get r 'y)]
              (is (every? (fn [i] (< (Math/abs (- (aget yg (int i)) (* 2.0 i))) 1e-3))
                          (range n)))))
          (testing "replay is stable across repeated runs"
            (let [r2 (fixture/run! program [x y (float 2.0) n])
                  ^floats yg (get r2 'y)]
              (is (< (Math/abs (- (aget yg 100) 200.0)) 1e-3)))))
        (finally (gpu/close-session! s))))))

(deftest ocl-resident-typed-scan-runs-through-the-compiled-graph-step
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident typed scan graph")
    (let [n 1025
          descriptor (pl/compile-gpu-program #'ocl-session-scan :ocl:0 :dtype :float)
          x (float-array (repeat n 1.0))
          out (float-array n)
          session (gpu/make-session :ocl:0)]
      (try
        (is (= [:executable] (mapv :convention (:steps descriptor))))
        (let [program (fixture/instantiate! session descriptor [x out n]
                                            {'x :input 'out :output})
              result (fixture/run! program [x out n])
              ^floats scanned (get result 'out)]
          (is (= (float n) (aget scanned (dec n))))
          (is (every? (fn [i] (= (float (inc i)) (aget scanned i)))
                      [0 1 255 256 511 512 1024])))
        (finally (gpu/close-session! session))))))

(deftest ocl-staged-typed-scan-runs-the-same-compiled-graph
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "staged typed scan graph")
    (let [n 1025
          execute (pl/compile-aot #'ocl-session-scan
                                  :target-device :ocl:0 :dtype :float)
          x (float-array (repeat n 1.0))
          out (float-array n)
          result (execute x out n)]
      (is (identical? out result))
      (is (= (float n) (aget out (dec n))))
      (is (every? (fn [i] (= (float (inc i)) (aget out i)))
                  [0 1 255 256 511 512 1024])))))

(deftest ocl-resident-indexed-row-reduction-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "indexed row reduction")
    (let [nrows 3
          width 513
          values (float-array (* nrows width) (float -10.0))
          indices (int-array nrows)
          descriptor (pl/compile-gpu-program #'ops/argmax-rows! :ocl:0 :dtype :float)
          s (gpu/make-session :ocl:0)]
      (aset values 3 (float 9.0))
      (aset values 256 (float 9.0))
      (aset values 512 (float 9.0))
      (aset values (+ width 255) Float/NaN)
      (aset values (+ width 257) Float/NaN)
      (aset values (+ width 300) (float 100.0))
      (aset values (+ (* 2 width) 411) (float 8.0))
      (try
        (let [program (fixture/instantiate! s descriptor [values indices nrows width])
              result (get (fixture/run! program [values indices nrows width]) 'indices)]
          (is (= [3 255 411] (vec result))))
        (finally (gpu/close-session! s))))))

(deftest ocl-resident-row-gather-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "row gather")
    (let [nrows 3
          width 4
          table (float-array (map float (range 32)))
          indices (int-array [5 1 5])
          rows (float-array (* nrows width))
          descriptor (pl/compile-gpu-program #'ops/gather-rows! :ocl:0 :dtype :float)
          s (gpu/make-session :ocl:0)]
      (try
        (let [program (fixture/instantiate! s descriptor [table indices rows nrows width])
              result (get (fixture/run! program [table indices rows nrows width]) 'out)]
          (is (= [20.0 21.0 22.0 23.0
                  4.0 5.0 6.0 7.0
                  20.0 21.0 22.0 23.0]
                 (vec result))))
        (finally (gpu/close-session! s))))))

(deftest ocl-resident-segmap-artifact-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident SegMap artifact")
    (let [descriptor (pl/compile-gpu-program #'ocl-session-map :ocl:0 :dtype :float)
          n 4096
          x (float-array (map float (range n)))
          y (float-array n)
          s (gpu/make-session :ocl:0)]
      (try
        (let [program (fixture/instantiate! s descriptor [x y (float 2.0) n])
              result (get (fixture/run! program [x y (float 2.0) n]) 'y)]
          (is (= [:map] (mapv :convention (:steps descriptor))))
          (is (every? (fn [i] (< (Math/abs (- (aget ^floats result (int i)) (* 2.0 i)))
                                 1e-3))
                      (range n))))
        (finally (gpu/close-session! s))))))

(deftest ocl-resident-contraction-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident contraction")
    (let [descriptor (pl/compile-gpu-program #'ocl-session-contract :ocl:0 :dtype :float)
          A (float-array (map float (range 64)))
          B (float-array (repeat 64 (float 1.0)))
          s (gpu/make-session :ocl:0)]
      (try
        (let [program (fixture/instantiate! s descriptor [A B])
              result (get (fixture/run! program [A B]) 'C)]
          (is (= [:contract] (mapv :convention (:steps descriptor))))
          (is (= (vec (mapcat #(repeat 8 (float %))
                              (map (fn [row] (reduce + (range (* row 8) (* (inc row) 8))))
                                   (range 8))))
                 (vec result))))
        (finally (gpu/close-session! s))))))

(deftest ocl-resident-reduction-ordered-abi-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident reduction")
    (let [descriptor (pl/compile-gpu-program #'ocl-session-reduce :ocl:0 :dtype :float)
          n 1024
          scale 0.75
          x (float-array (map #(float (/ (inc %) 1024.0)) (range n)))
          out (float-array n)
          expected-sum (reduce + 0.0 (map #(* scale (double %)) (seq x)))
          s (gpu/make-session :ocl:0)]
      (try
        (let [program (fixture/instantiate! s descriptor [x out scale n])
              result (get (fixture/run! program [x out scale n]) 'out)
              red-step (first (filter #(= :reduce (:convention %)) (:steps descriptor)))]
          (is (= [:input :output :scalar :scalar]
                 (mapv :kind (:argument-specs red-step))))
          (is (< (Math/abs (- (double (aget ^floats result 511))
                              (* (double (aget x 511)) expected-sum)))
                 1e-3)))
        (finally (gpu/close-session! s))))))
