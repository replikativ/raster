(ns raster.compiler.passes.parallel.gemm-epilogue-device-test
  "An accumulating BLAS GEMM (`beta ≠ 0`) is a typed contraction whose result transform reads
   the destination element it overwrites. The deftm below keeps its BLAS spelling; the typed
   route turns it into one resident `:contract` step whose destination is a single `:inout`
   ABI slot, and the same source runs bit-comparably on the JVM (host expansion of the same
   result transform), on OpenCL and on Level Zero."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :refer [deftm]]
            [raster.dl.gpu-grad-parity :as gpu-probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.descriptor-fixture :as fixture]
            [raster.gpu.device-probe :as device-probe]
            [raster.linalg.blas :as blas]
            [raster.numeric :as n]))

(deftm accumulate-gemm! [A :- (Array float), B :- (Array float), C :- (Array float),
                         m :- Long, k :- Long, n :- Long] :- (Array float)
  (let [_ (blas/dgemm! A B C m k n (n/oftype A 1.0) (n/oftype A 1.0))]
    C))

(def ^:private m 3)
(def ^:private k 4)
(def ^:private n 5)

(defn- a-matrix [] (float-array (map float (range (* m k)))))
(defn- b-matrix [] (float-array (map #(float (* 0.5 %)) (range (* k n)))))
(defn- c-initial [] (float-array (map #(float (* 100 %)) (range (* m n)))))

(defn- reference
  "C0 + A·B in float, the value every backend must reproduce."
  []
  (let [A (a-matrix) B (b-matrix) C0 (c-initial) C (aclone C0)]
    (dotimes [i m]
      (dotimes [j n]
        (aset C (+ (* i n) j)
              (float (+ (aget C0 (+ (* i n) j))
                        (reduce + (for [l (range k)]
                                    (* (aget A (+ (* i k) l)) (aget B (+ (* l n) j))))))))))
    (vec C)))

(deftest the-jvm-accumulates-into-the-destination
  (let [C (c-initial)]
    (accumulate-gemm! (a-matrix) (b-matrix) C m k n)
    (is (= (reference) (vec C)))))

(deftest the-resident-program-is-one-contract-step-with-a-read-write-destination
  (doseq [target [:ocl:0 :ze:0]]
    (let [descriptor (pipeline/compile-gpu-program #'accumulate-gemm! target :dtype :float)
          step (first (:steps descriptor))]
      (testing (str target)
        (is (= [:contract] (mapv :convention (:steps descriptor))))
        (is (= '[[A :input] [B :input] [C :inout]]
               (vec (for [slot (get-in step [:artifact :abi])
                          :when (not= :scalar (:kind slot))]
                      [(:name slot) (:kind slot)]))))
        (is (empty? (:allocs descriptor)))
        (is (re-find #"__global float\* C" (get-in step [:artifact :source]))
            "the destination pointer is writable: no const qualifier")))))

(defn- run-on-device
  [target]
  (let [descriptor (pipeline/compile-gpu-program #'accumulate-gemm! target :dtype :float)
        session (gpu/make-session target)
        C (c-initial)
        arguments [(a-matrix) (b-matrix) C m k n]]
    (try
      (let [program (fixture/instantiate! session descriptor arguments
                                          {'A :input 'B :input 'C :output})]
        (vec (get (fixture/run! program arguments) 'C)))
      (finally (gpu/close-session! session)))))

(deftest opencl-accumulates-into-the-destination
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "accumulating GEMM on OpenCL")
    (is (= (reference) (run-on-device :ocl:0)))))

(deftest level-zero-accumulates-into-the-destination
  (if-not @gpu-probe/gpu-available?
    (gpu-probe/gpu-skip! "accumulating GEMM on Level Zero")
    (is (= (reference) (run-on-device :ze:0)))))
