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

(deftm ocl-session-inout
  [state :- (Array float) a :- Float n :- Long] :- (Array float)
  (raster.par/map! state i n float (* a (ra/aget state i))))

(deftm ocl-session-map2
  [x :- (Array float) y :- (Array float)
   a :- (Array float) b :- (Array float) n :- Long] :- Void
  (raster.par/map2! a b i n float
                    (+ (ra/aget x i) 1.0)
                    (* (ra/aget y i) 2.0)))

(deftm ocl-session-effect-mixed
  [x :- (Array float) q :- (Array byte)
   y :- (Array float) labels :- (Array int) n :- Long] :- Void
  (raster.par/map-void!
   i n
   (do (ra/aset y i (float (* (ra/aget x i) 2.0)))
       (ra/aset labels i (int (+ (int (ra/aget q i)) 7))))))

(deftm ocl-session-effect-shared-local
  [x :- (Array float) a :- (Array float) b :- (Array float) n :- Long] :- Void
  (raster.par/map-void!
   i n
   (let [shifted (float (+ (ra/aget x i) 1.0))
         squared (float (* shifted shifted))]
     (ra/aset a i shifted)
     (ra/aset b i squared))))

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

(deftm ocl-session-exclusive-scan
  [x :- (Array float) out :- (Array float) n :- Long] :- (Array float)
  (raster.par/scan-exclusive out acc 0.0 i n float
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

(deftest ocl-resident-typed-effect-tuple-map-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident typed effect tuple map")
    (let [descriptor (pl/compile-gpu-program #'ocl-session-map2 :ocl:0 :dtype :float)
          n 1024
          x (float-array (map float (range n)))
          y (float-array (map #(float (+ 10 %)) (range n)))
          a (float-array n)
          b (float-array n)
          session (gpu/make-session :ocl:0)]
      (try
        (is (= [:map-void] (mapv :convention (:steps descriptor)))
            "host control retains the nil-returning effect convention")
        (is (= :segmap (get-in descriptor [:steps 0 :artifact :provenance :dialect]))
            "kernel generation consumes the scheduled TypedSOAC SegMap")
        (is (= [:input :input :output :output :scalar]
               (mapv :kind (get-in descriptor [:steps 0 :abi]))))
        (let [program (fixture/instantiate! session descriptor [x y a b n]
                                            {'x :input 'y :input
                                             'a :output 'b :output})
              result (fixture/run! program [x y a b n])
              ^floats actual-a (get result 'a)
              ^floats actual-b (get result 'b)]
          (is (every? (fn [i] (= (float (inc i)) (aget actual-a i)))
                      [0 1 255 256 1023]))
          (is (every? (fn [i] (= (float (* 2.0 (+ 10 i))) (aget actual-b i)))
                      [0 1 255 256 1023])))
        (finally (gpu/close-session! session))))))

(deftest ocl-resident-typed-effect-tuple-map-preserves-mixed-storage
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident typed mixed effect tuple map")
    (let [descriptor (pl/compile-gpu-program #'ocl-session-effect-mixed
                                             :ocl:0 :dtype :float)
          n 257
          x (float-array (map #(float (/ % 8.0)) (range n)))
          q (byte-array (map #(byte (- (mod % 17) 8)) (range n)))
          y (float-array n)
          labels (int-array n)
          session (gpu/make-session :ocl:0)]
      (try
        (is (= :segmap (get-in descriptor [:steps 0 :artifact :provenance :dialect])))
        (is (= [:byte :float :int :float :int]
               (mapv :dtype (get-in descriptor [:steps 0 :abi]))))
        (let [program (fixture/instantiate! session descriptor [x q y labels n]
                                            {'x :input 'q :input
                                             'y :output 'labels :output})
              result (fixture/run! program [x q y labels n])
              ^floats actual-y (get result 'y)
              ^ints actual-labels (get result 'labels)]
          (is (= (float (* 2.0 (aget x 256))) (aget actual-y 256)))
          (is (= (+ 7 (int (aget q 256))) (aget actual-labels 256))))
        (finally (gpu/close-session! session))))))

(deftest ocl-resident-typed-region-ssa-shares-tuple-work
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident typed scalar-region SSA")
    (let [descriptor (pl/compile-gpu-program #'ocl-session-effect-shared-local
                                             :ocl:0 :dtype :float)
          source (get-in descriptor [:steps 0 :artifact :source])
          n 513
          x (float-array (map float (range n)))
          a (float-array n)
          b (float-array n)
          session (gpu/make-session :ocl:0)]
      (try
        (is (= :segmap (get-in descriptor [:steps 0 :artifact :provenance :dialect])))
        (is (= 1 (count (re-seq #"x\[idx\]" source)))
            "the live kernel computes its shared tuple producer once")
        (let [program (fixture/instantiate! session descriptor [x a b n]
                                            {'x :input 'a :output 'b :output})
              result (fixture/run! program [x a b n])
              ^floats actual-a (get result 'a)
              ^floats actual-b (get result 'b)]
          (is (= 513.0 (double (aget actual-a 512))))
          (is (= (* 513.0 513.0) (double (aget actual-b 512)))))
        (finally (gpu/close-session! session))))))

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

(deftest ocl-resident-and-staged-exclusive-scan-use-the-same-typed-graph
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident and staged typed exclusive scan graph")
    (let [n 1025
          x (float-array (repeat n 1.0))
          descriptor (pl/compile-gpu-program #'ocl-session-exclusive-scan
                                             :ocl:0 :dtype :float)
          session (gpu/make-session :ocl:0)]
      (try
        (is (= :exclusive
               (get-in descriptor [:steps 0 :artifact :attributes :scan-mode])))
        (let [out (float-array (inc n))
              program (fixture/instantiate! session descriptor [x out n]
                                            {'x :input 'out :output})
              ^floats scanned (get (fixture/run! program [x out n]) 'out)]
          (is (every? (fn [i] (= (float i) (aget scanned i)))
                      [0 1 255 256 257 512 1024 1025])))
        (let [execute (pl/compile-aot #'ocl-session-exclusive-scan
                                      :target-device :ocl:0 :dtype :float)
              out (float-array (inc n))
              result (execute x out n)]
          (is (identical? out result))
          (is (every? (fn [i] (= (float i) (aget out i)))
                      [0 1 255 256 257 512 1024 1025])))
        (finally (gpu/close-session! session))))))

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

(deftest ocl-resident-typed-inout-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "resident typed inout SegMap")
    (let [descriptor (pl/compile-gpu-program #'ocl-session-inout :ocl:0 :dtype :float)
          n 4096
          state (float-array (map #(float (inc %)) (range n)))
          s (gpu/make-session :ocl:0)]
      (try
        (is (= [:inout :scalar :scalar]
               (mapv :kind (:argument-specs (first (:steps descriptor))))))
        (let [program (fixture/instantiate! s descriptor [state (float 2.0) n]
                                            {'state :state})]
          (fixture/run! program [state (float 2.0) n])
          (let [^floats once (fixture/download program 'state)]
            (is (= (float (* 2.0 513.0)) (aget once 512))))
          (fixture/run! program [state (float 2.0) n])
          (let [^floats twice (fixture/download program 'state)]
            (is (= (float (* 4.0 513.0)) (aget twice 512))
                "resident state is read and written through one ABI pointer on replay")))
        (finally (gpu/close-session! s))))))

(deftest ocl-staged-typed-inout-roundtrip
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "staged typed inout SegMap")
    (let [n 1024
          execute (pl/compile-aot #'ocl-session-inout
                                  :target-device :ocl:0 :dtype :float)
          state (float-array (map #(float (inc %)) (range n)))
          result (execute state (float 3.0) n)]
      (is (identical? state result))
      (is (= (float (* 3.0 513.0)) (aget state 512))))))

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
