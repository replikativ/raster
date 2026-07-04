(ns raster.gpu.ocl-session-test
  "Resident-session layer on the OpenCL backend: compile → bind-program! →
  record → replay → download, on whatever OpenCL device is available. With
  RASTER_OCL_DEVICE_TYPE=cpu this runs on POCL/Intel-CPU — the no-GPU
  vendor-portability oracle (CUDA/HIP Phase A). Skips cleanly without OpenCL.

  Portability lesson encoded here: kernels must not read from :output-only
  buffers (uninitialized device memory — Intel's driver zeroes allocations,
  POCL's malloc does not)."
  (:require [clojure.test :refer [deftest is testing]]
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
