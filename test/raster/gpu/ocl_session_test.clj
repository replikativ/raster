(ns raster.gpu.ocl-session-test
  "Resident-session layer on the OpenCL backend: compile → bind-program! →
  record → replay → download, on whatever OpenCL device is available. With
  RASTER_OCL_DEVICE_TYPE=cpu this runs on POCL/Intel-CPU — the no-GPU
  vendor-portability oracle (CUDA/HIP Phase A). Skips cleanly without OpenCL.

  Portability lesson encoded here: kernels must not read from :output-only
  buffers (uninitialized device memory — Intel's driver zeroes allocations,
  POCL's malloc does not)."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.par-opencl :as par-opencl]
            [raster.compiler.pipeline :as pl]
            [raster.arrays :as ra]
            [raster.core :refer [deftm]]))

(def ^:private ocl-available?
  (delay (try (require 'raster.gpu.ocl-runtime)
              ((resolve 'raster.gpu.ocl-runtime/init!))
              true
              (catch Throwable _ false))))

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

(deftest ocl-map-void-mixed-storage-abi-roundtrip
  (if-not @ocl-available?
    (println "SKIP ocl mixed-storage map-void (no OpenCL device)")
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
  (if-not @ocl-available?
    (println "SKIP ocl-session test (no OpenCL device)")
    (let [gpu (do (require 'raster.gpu.core) (find-ns 'raster.gpu.core))
          make-session (ns-resolve gpu 'make-session)
          bind-program! (ns-resolve gpu 'bind-program!)
          run-program! (ns-resolve gpu 'run-program!)
          close-session! (ns-resolve gpu 'close-session!)
          p (pl/compile-gpu-program #'ocl-session-ax! :ocl:0 :dtype :float)
          n 4096
          x (float-array (map float (range n)))
          y (float-array n)
          s (make-session :ocl:0)]
      (try
        (bind-program! s p [x y (float 2.0) n] {'x :input 'y :output})
        (testing "replayed program computes correctly"
          (let [r (run-program! s p [x y (float 2.0) n])
                ^floats yg (get r 'y)]
            (is (every? (fn [i] (< (Math/abs (- (aget yg (int i)) (* 2.0 i))) 1e-3))
                        (range n)))))
        (testing "replay is stable across repeated runs"
          (let [r2 (run-program! s p [x y (float 2.0) n])
                ^floats yg (get r2 'y)]
            (is (< (Math/abs (- (aget yg 100) 200.0)) 1e-3))))
        (finally (close-session! s))))))

(deftest ocl-resident-segmap-artifact-roundtrip
  (if-not @ocl-available?
    (println "SKIP ocl resident SegMap artifact (no OpenCL device)")
    (let [gpu (do (require 'raster.gpu.core) (find-ns 'raster.gpu.core))
          make-session (ns-resolve gpu 'make-session)
          bind-program! (ns-resolve gpu 'bind-program!)
          run-program! (ns-resolve gpu 'run-program!)
          close-session! (ns-resolve gpu 'close-session!)
          descriptor (pl/compile-gpu-program #'ocl-session-map :ocl:0 :dtype :float)
          n 4096
          x (float-array (map float (range n)))
          y (float-array n)
          s (make-session :ocl:0)]
      (try
        (let [handle (bind-program! s descriptor [x y (float 2.0) n])
              result (get (run-program! s handle [x y (float 2.0) n]) 'y)]
          (is (= [:map] (mapv :convention (:steps descriptor))))
          (is (every? (fn [i] (< (Math/abs (- (aget ^floats result (int i)) (* 2.0 i)))
                                  1e-3))
                      (range n))))
        (finally (close-session! s))))))

(deftest ocl-resident-contraction-roundtrip
  (if-not @ocl-available?
    (println "SKIP ocl resident contraction (no OpenCL device)")
    (let [gpu (do (require 'raster.gpu.core) (find-ns 'raster.gpu.core))
          make-session (ns-resolve gpu 'make-session)
          bind-program! (ns-resolve gpu 'bind-program!)
          run-program! (ns-resolve gpu 'run-program!)
          close-session! (ns-resolve gpu 'close-session!)
          descriptor (pl/compile-gpu-program #'ocl-session-contract :ocl:0 :dtype :float)
          A (float-array (map float (range 64)))
          B (float-array (repeat 64 (float 1.0)))
          s (make-session :ocl:0)]
      (try
        (let [handle (bind-program! s descriptor [A B])
              result (get (run-program! s handle [A B]) 'C)]
          (is (= [:contract] (mapv :convention (:steps descriptor))))
          (is (= (vec (mapcat #(repeat 8 (float %))
                              (map (fn [row] (reduce + (range (* row 8) (* (inc row) 8))))
                                   (range 8))))
                 (vec result))))
        (finally (close-session! s))))))

(deftest ocl-resident-reduction-ordered-abi-roundtrip
  (if-not @ocl-available?
    (println "SKIP ocl resident reduction (no OpenCL device)")
    (let [gpu (do (require 'raster.gpu.core) (find-ns 'raster.gpu.core))
          make-session (ns-resolve gpu 'make-session)
          bind-program! (ns-resolve gpu 'bind-program!)
          run-program! (ns-resolve gpu 'run-program!)
          close-session! (ns-resolve gpu 'close-session!)
          descriptor (pl/compile-gpu-program #'ocl-session-reduce :ocl:0 :dtype :float)
          n 1024
          scale 0.75
          x (float-array (map #(float (/ (inc %) 1024.0)) (range n)))
          out (float-array n)
          expected-sum (reduce + 0.0 (map #(* scale (double %)) (seq x)))
          s (make-session :ocl:0)]
      (try
        (let [handle (bind-program! s descriptor [x out scale n])
              result (get (run-program! s handle [x out scale n]) 'out)
              red-step (first (filter #(= :reduce (:convention %)) (:steps descriptor)))]
          (is (= [:input :output :scalar :scalar]
                 (mapv :kind (:argument-specs red-step))))
          (is (< (Math/abs (- (double (aget ^floats result 511))
                              (* (double (aget x 511)) expected-sum)))
                 1e-3)))
        (finally (close-session! s))))))
