(ns raster.compiler.passes.parallel.gemm-epilogue-device-test
  "An accumulating BLAS GEMM (`beta ≠ 0`) is a typed contraction whose result transform reads
   the destination element it overwrites. The deftm below keeps its BLAS spelling; the typed
   route turns it into one typed executable dispatch whose destination is a single `:inout`
   ABI slot, and the same source runs bit-comparably on the JVM (host expansion of the same
   result transform), on OpenCL and on Level Zero."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-executable :as kexec]
            [raster.core :refer [deftm]]
            [raster.dl.gpu-grad-parity :as gpu-probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.descriptor-fixture :as fixture]
            [raster.gpu.device-probe :as device-probe]
            [raster.linalg.blas :as blas]
            [raster.dl.nn :as nn]
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

(defn- default-executable
  [step]
  (kdispatch/default-alternative (:dispatch step)))

(defn- default-artifact
  [step]
  (first (kexec/artifacts (default-executable step))))

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

(deftest the-resident-program-is-one-typed-executable-with-a-read-write-destination
  (doseq [target [:ocl:0 :ze:0]]
    (let [descriptor (pipeline/compile-gpu-program #'accumulate-gemm! target :dtype :float)
          step (first (:steps descriptor))
          executable (default-executable step)
          artifact (default-artifact step)]
      (testing (str target)
        (is (= [:executable] (mapv :convention (:steps descriptor))))
        (is (= '[[A :input] [B :input] [C :inout]]
               (vec (for [slot (kexec/abi executable)
                          :when (not= :scalar (:kind slot))]
                      [(:name slot) (:kind slot)]))))
        (is (empty? (:allocs descriptor)))
        (is (re-find #"__global float\* C" (:source artifact))
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

;; `linear!` prefills its output with the bias, one row copy at a time, then accumulates
;; `x·Wᵀ` into it with `beta = 1`. Both halves are now typed: the row copies are a store loop
;; the lifter turns into one map, and the GEMM is a contraction whose result transform reads
;; the destination.
(def ^:private batch 3)
(def ^:private in-f 4)
(def ^:private out-f 5)

(defn- linear-arguments
  []
  [(float-array (map #(float (* 0.25 %)) (range (* batch in-f))))
   (float-array (map #(float (- (* 0.5 %) 3.0)) (range (* out-f in-f))))
   (float-array (map #(float (* 10 %)) (range out-f)))
   (float-array (* batch out-f))
   batch in-f out-f])

(defn- linear-reference
  []
  (let [[x W b] (linear-arguments)]
    (vec (for [i (range batch) j (range out-f)]
           (float (+ (aget b j)
                     (reduce + (for [l (range in-f)]
                                 (* (aget x (+ (* i in-f) l)) (aget W (+ (* j in-f) l)))))))))))

(deftest the-linear-layer-is-a-resident-map-and-typed-executable
  (doseq [target [:ocl:0 :ze:0]]
    (let [descriptor (pipeline/compile-gpu-program #'nn/linear! target :dtype :float)]
      (testing (str target)
        (is (= [:map :executable] (mapv :convention (:steps descriptor))))
        (is (= '[[W :input] [x :input] [y :inout]]
               (vec (for [slot (kexec/abi
                                (default-executable (second (:steps descriptor))))
                          :when (not= :scalar (:kind slot))]
                      [(:name slot) (:kind slot)]))))))))

(defn- run-linear-on-device
  [target]
  (let [descriptor (pipeline/compile-gpu-program #'nn/linear! target :dtype :float)
        session (gpu/make-session target)
        arguments (linear-arguments)]
    (try
      (let [program (fixture/instantiate! session descriptor arguments
                                          {'x :input 'W :input 'b :input 'y :output})]
        (vec (get (fixture/run! program arguments) 'y)))
      (finally (gpu/close-session! session)))))

(deftest the-linear-layer-matches-on-every-backend
  (let [[x W b y & dims] (linear-arguments)]
    (apply nn/linear! x W b y dims)
    (is (= (linear-reference) (vec y)) "JVM"))
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "linear! on OpenCL")
    (is (= (linear-reference) (run-linear-on-device :ocl:0)) "OpenCL"))
  (if-not @gpu-probe/gpu-available?
    (gpu-probe/gpu-skip! "linear! on Level Zero")
    (is (= (linear-reference) (run-linear-on-device :ze:0)) "Level Zero")))
