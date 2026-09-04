(ns raster.compiler.passes.parallel.typed-contraction-matrix-device-test
  "The optimized matrix route is exercised from the typed contraction equation through the
   backend-neutral KernelExecutable binder.  This is the device guard that permits the old
   Level Zero GEMM-specific binding surface to disappear without losing its numerical oracle."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.passes.parallel.contract-route :as contract-route]
            [raster.compiler.pipeline :as pipeline]
            [raster.dl.gpu-grad-parity :as gpu-probe]
            [raster.gpu.core :as gpu]))

(def ^:private source
  '(let* [step (raster.par/contract C [[i m] [j n]] [[l k]]
                                      (* (clojure.core/aget A (+ (* i k) l))
                                         (clojure.core/aget B (+ (* l n) j))))]
     step))

(defn- typed-dispatch
  [device-id]
  (let [{:keys [form]}
        (pipeline/schedule-parallel-form
         source {:target-device device-id
                 :dtype :float
                 :array-types {'A :float 'B :float 'C :float}
                 :scalar-types {'m :int 'n :int 'k :int}})
        equation (first (:equations form))]
    (contract-route/route-typed-contraction-dispatch
     (:algorithm equation) (first (:operations equation))
     :dtype :float
     :desc (hardware/descriptor-for device-id)
     :precision :mixed-f16-f32)))

(defn- input-array
  [n seed]
  (let [result (float-array n)
        random (java.util.Random. (long seed))]
    (dotimes [index n]
      (aset result index (float (* 0.05 (.nextGaussian random)))))
    result))

(defn- f16
  ^double [value]
  (double (Float/float16ToFloat (Float/floatToFloat16 (float value)))))

(defn- reference
  [^floats a ^floats b m n k]
  (let [result (float-array (* m n))]
    (dotimes [i m]
      (dotimes [j n]
        (aset result (+ (* i n) j)
              (float
               (loop [l 0
                      sum 0.0]
                 (if (< l k)
                   (recur (inc l)
                          (+ sum (* (f16 (aget a (+ (* i k) l)))
                                    (f16 (aget b (+ (* l n) j))))))
                   sum))))))
    result))

(defn- relative-l1
  [^floats actual ^floats expected]
  (loop [index 0
         difference 0.0
         scale 0.0]
    (if (< index (alength actual))
      (recur (inc index)
             (+ difference
                (Math/abs (- (double (aget actual index))
                             (double (aget expected index)))))
             (+ scale (Math/abs (double (aget expected index)))))
      (/ difference (max scale 1.0e-30)))))

(defn- run-contraction
  [device-id scheduled m n k]
  (let [a (input-array (* m k) 17)
        b (input-array (* k n) 29)
        runtime-arguments
        [:a :b :c
         {:type :int :value m}
         {:type :int :value n}
         {:type :int :value k}]
        selected (dispatch/select-alternative scheduled runtime-arguments)]
    (gpu/with-gpu-session [session device-id]
      (gpu/alloc! session {:a [:float (* m k) a]
                           :b [:float (* k n) b]
                           :c [:float (* m n) nil]})
      (let [handle (gpu/bind-kernel-executable!
                    session [:typed-contraction m n k] selected runtime-arguments)]
        (try
          (gpu/run-kernel-graph! session handle)
          {:strategy (executable/strategy selected)
           :actual (gpu/download session :c)
           :expected (reference a b m n k)}
          (finally
            (gpu/release-kernel-graph! session handle)))))))

(deftest ordinary-typed-contraction-executes-the-matrix-schedule
  (if-not @gpu-probe/gpu-available?
    (gpu-probe/gpu-skip! "typed contraction matrix KernelExecutable")
    (let [scheduled (typed-dispatch :ze:0)]
      (testing "an aligned dynamic contraction selects and executes the direct matrix graph"
        (let [{:keys [strategy actual expected]}
              (run-contraction :ze:0 scheduled 16 16 16)]
          (is (= :xmx-direct strategy))
          (is (< (relative-l1 actual expected) 1.0e-3))))
      (testing "a low-output-occupancy contraction executes the graph-private split-K schedule"
        (let [{:keys [strategy actual expected]}
              (run-contraction :ze:0 scheduled 13 16 8192)]
          (is (= :xmx-split-k strategy))
          (is (< (relative-l1 actual expected) 1.0e-3)))))))
