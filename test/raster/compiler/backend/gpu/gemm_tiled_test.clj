(ns raster.compiler.backend.gpu.gemm-tiled-test
  "Device guards for compiler-derived matrix schedules.

   The tests start from ordinary TypedSOAC contractions, isolate a scheduled matrix artifact only
   when measuring target-kernel properties, and use the common KernelExecutable binder.  The
   Level Zero runtime neither recognizes GEMM nor reconstructs its ABI or launch geometry."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.typed-matrix-device-support :as support]
            [raster.compiler.core.hardware :as hardware]
            [raster.dl.gpu-grad-parity :as gpu-probe]
            [raster.gpu.core :as gpu]))

(defn- rms
  [actual expected]
  (Math/sqrt
   (/ (reduce + (map (fn [x y]
                       (let [difference (- (double x) (double y))]
                         (* difference difference)))
                     actual expected))
      (double (count actual)))))

(defn- run-tile
  [session tile ^floats a ^floats b m n k key]
  (let [scheduled (support/dense-dispatch :ze:0 :tile tile)
        artifact (support/matrix-artifact scheduled :xmx-direct)]
    (gpu/alloc! session {[:a key] [:half (* m k) (support/half-array a)]
                         [:b key] [:half (* k n) (support/half-array b)]
                         [:c key] [:float (* m n) nil]})
    (let [handle (gpu/bind-kernel-executable!
                  session [:matrix key] artifact
                  [[:a key] [:b key] [:c key]
                   {:type :int :value m}
                   {:type :int :value n}
                   {:type :int :value k}])]
      (try
        (gpu/run-kernel-graph! session handle)
        (gpu/download session [:c key])
        (finally
          (gpu/release-kernel-graph! session handle))))))

(deftest tiled-gemm-matches-cpu-and-is-tile-invariant
  (if-not @gpu-probe/gpu-available?
    (gpu-probe/gpu-skip! "typed matrix schedule: CPU correctness + cross-tile invariance")
    (let [descriptor (hardware/descriptor-for :ze:0)
          default-tile (hardware/gemm-tile-for descriptor)
          tile-64 (assoc default-tile :block-m 64 :block-n 64)
          tile-k64 (assoc default-tile :block-k 64)
          m 64 n 128 k 256
          a (support/input-array (* m k) 7)
          b (support/input-array (* k n) 11)
          expected (support/reference a b m n k)]
      (gpu/with-gpu-session [session :ze:0]
        (let [default-result (run-tile session default-tile a b m n k :default)
              tile-64-result (run-tile session tile-64 a b m n k :tile-64)
              tile-k64-result (run-tile session tile-k64 a b m n k :tile-k64)]
          (testing "the compiler-derived default matrix artifact matches an independent CPU oracle"
            (is (< (rms default-result expected) 1.0e-2))
            (is (some #(> (Math/abs (double %)) 0.05) default-result)))
          (testing "schedule tile changes preserve the contraction result bit-for-bit"
            (is (= (seq default-result) (seq tile-64-result)))
            (is (= (seq default-result) (seq tile-k64-result)))))))))

(deftest typed-result-transform-is-part-of-the-matrix-artifact
  (if-not @gpu-probe/gpu-available?
    (gpu-probe/gpu-skip! "typed matrix result transform")
    (let [m 64 n 128 k 256 scale 0.5
          a (support/input-array (* m k) 3)
          b (support/input-array (* k n) 5)
          bias (support/input-array n 13)
          base (support/reference a b m n k)
          expected (float-array (* m n))
          scheduled (support/epilogue-dispatch :ze:0)
          artifact (support/matrix-artifact scheduled :xmx-direct)]
      (dotimes [index (* m n)]
        (aset expected index
              (float (* scale (+ (double (aget base index))
                                 (double (aget bias (mod index n))))))))
      (gpu/with-gpu-session [session :ze:0]
        (gpu/alloc! session {:a [:half (* m k) (support/half-array a)]
                             :b [:half (* k n) (support/half-array b)]
                             :bias [:float n bias]
                             :c [:float (* m n) nil]})
        (let [handle (gpu/bind-kernel-executable!
                      session :typed-matrix-result-transform artifact
                      [:a :b :c
                       {:type :int :value m}
                       {:type :int :value n}
                       {:type :int :value k}
                       :bias
                       {:type :float :value scale}])]
          (try
            (gpu/run-kernel-graph! session handle)
            (is (< (support/relative-l1 (gpu/download session :c) expected) 1.0e-3))
            (finally
              (gpu/release-kernel-graph! session handle))))))))
