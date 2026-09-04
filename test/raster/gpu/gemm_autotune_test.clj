(ns raster.gpu.gemm-autotune-test
  "Device-event measurement of finite compiler-emitted matrix schedule spaces.

   Split factor and tile are schedule facts over an ordinary typed contraction. The tests execute
   those alternatives through the common KernelExecutable binder; the backend runtime knows
   neither GEMM semantics nor candidate-specific ABIs."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.gemm :as gemm]
            [raster.compiler.backend.gpu.typed-matrix-device-support :as support]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.dl.gpu-grad-parity :as gpu-probe]
            [raster.gpu.core :as gpu]))

(defn- bench-split-ms
  "Measure one complete typed contraction graph, including layout conversion and final combine."
  [session scheduled m n k splits]
  (let [strategy (if (= 1 splits) :xmx-direct (gemm/split-factor-strategy splits))
        graph (dispatch/alternative scheduled strategy)
        output [:split-output splits]
        _ (gpu/alloc! session {output [:float (* m n) nil]})
        handle (gpu/bind-kernel-executable!
                session [:split-candidate splits] graph
                [:split-a :split-b output
                 {:type :int :value m}
                 {:type :int :value n}
                 {:type :int :value k}]
                {:profile? true})]
    (try
      (gpu/run-kernel-graph! session handle)
      (/ (:median-ns
          (gpu/measure-bound-kernel-graph!
           session handle
           :warmup-iterations 3 :budget-ms 1000
           :min-samples 11 :max-samples 11))
         1.0e6)
      (finally
        (gpu/release-kernel-graph! session handle)))))

(deftest autotune-finds-splitk-optimum
  (if-not @gpu-probe/gpu-available?
    (gpu-probe/gpu-skip! "typed contraction autotune: split-K optimum")
    (let [m 16 n 512 k 131072
          candidates [1 2 4 8 16 32]
          measured (atom 0)
          random (java.util.Random. 3)
          make-input (fn [size]
                       (let [result (float-array size)]
                         (dotimes [index size]
                           (aset result index (float (.nextGaussian random))))
                         result))
          scheduled (support/dense-dispatch :ze:0 :split-factors (vec (rest candidates)))
          gflops (fn [ms] (/ (* 2.0 m k n) (* ms 1.0e6)))]
      (gpu/with-gpu-session [session :ze:0]
        (gpu/alloc! session {:split-a [:float (* m k) (make-input (* m k))]
                             :split-b [:float (* k n) (make-input (* k n))]})
        (let [results (mapv (fn [splits]
                              (swap! measured inc)
                              {:splits splits
                               :ms (bench-split-ms session scheduled m n k splits)})
                            candidates)
              baseline-ms (:ms (first results))
              winner (apply min-key :ms results)
              tuned (:splits winner)
              tuned-ms (:ms winner)
              speedup (/ baseline-ms tuned-ms)]
          (println (format (str "\n=== typed contraction split-K autotune (m=%d n=%d k=%d) ===\n"
                                "  baseline splits=1 : %.2f GFLOPS\n"
                                "  autotuned splits=%d: %.2f GFLOPS  (%.1fx, %d graphs measured)\n")
                           m n k (gflops baseline-ms) tuned (gflops tuned-ms) speedup @measured))
          (testing "finite compiler schedule measurement finds a large occupancy win"
            (is (> tuned 1) "the measured search selects a non-trivial split factor")
            (is (> speedup 2.0)
                (str "split-K must beat the complete direct graph by >2× (got "
                     (format "%.1fx" speedup) ")"))
            (is (= (count candidates) @measured)
                "every compiler-emitted schedule is measured once")))))))

(defn- run-tiled-gemm
  "Measure the target matrix artifact selected from an ordinary typed contraction."
  [session m n k tile key]
  (let [scheduled (support/dense-dispatch :ze:0 :tile tile)
        artifact (support/matrix-artifact scheduled :xmx-direct)
        output [:tile-output key]
        _ (gpu/alloc! session {output [:float (* m n) nil]})
        handle (gpu/bind-kernel-executable!
                session [:tile-candidate key] artifact
                [:tile-a :tile-b output
                 {:type :int :value m}
                 {:type :int :value n}
                 {:type :int :value k}]
                {:profile? true})]
    (try
      (gpu/run-kernel-graph! session handle)
      (let [measurement (gpu/measure-bound-kernel-graph!
                         session handle
                         :warmup-iterations 2 :budget-ms 1000
                         :min-samples 7 :max-samples 7)]
        [(vec (gpu/download session output)) (/ (:median-ns measurement) 1.0e6)])
      (finally
        (gpu/release-kernel-graph! session handle)))))

(deftest autotune-tile-axis-all-candidates-correct
  (if-not @gpu-probe/gpu-available?
    (gpu-probe/gpu-skip! "typed contraction autotune: tile candidates")
    (let [descriptor (hardware/descriptor-for :ze:0)
          candidates (hardware/gemm-tile-candidates descriptor)
          default-tile (hardware/derive-gemm-tile descriptor)
          m 256 n 256 k 512
          a (support/input-array (* m k) 4)
          b (support/input-array (* k n) 5)]
      (gpu/with-gpu-session [session :ze:0]
        (gpu/alloc! session {:tile-a [:half (* m k) (support/half-array a)]
                             :tile-b [:half (* k n) (support/half-array b)]})
        (testing "the finite schedule space is non-trivial and contains the analytic default"
          (is (> (count candidates) 1))
          (is (some #(= % default-tile) candidates)))
        (let [reference (first (run-tiled-gemm session m n k default-tile :reference))
              results (mapv (fn [index tile]
                              (let [[result ms] (run-tiled-gemm session m n k tile index)]
                                {:tile tile :bit-identical? (= reference result) :ms ms}))
                            (range) candidates)]
          (doseq [{:keys [tile bit-identical?]} results]
            (is bit-identical?
                (str "tile " (select-keys tile [:block-m :block-n :block-k])
                     " preserves the contraction result")))
          (let [winner (:tile (apply min-key :ms results))]
            (println (format "\n=== typed contraction tile autotune (m=%d n=%d k=%d) ==="
                             m n k))
            (doseq [{:keys [tile ms]} (sort-by :ms results)]
              (println (format "  %-22s %.3f ms%s"
                               (str (:block-m tile) "x" (:block-n tile)
                                    " k" (:block-k tile))
                               ms (if (= tile winner) "  <- winner" ""))))
            (is (some #(= % winner) candidates))
            (is (= reference (first (run-tiled-gemm session m n k winner :winner))))))))))
