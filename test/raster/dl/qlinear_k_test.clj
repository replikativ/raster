(ns raster.dl.qlinear-k-test
  "The composable K-quant GEMV deftms compile to C (compile-aot :target :c) and match the
   dequant-matmul reference — proving the registry's formats reach a working C kernel via
   the SAME composable path the legacy Q4_0 uses (and that the GPU/OpenCL path will reuse).
   Single-call correctness; no spin-pool, no Valhalla."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.qlinear-k :as qk]
            [raster.compiler.backend.cpu.quant :as q]
            [raster.compiler.pipeline :as pipeline]))

(defn- clang-available? []
  (try
    (let [cc (or (System/getenv "RASTER_CC") "clang")
          p (-> (ProcessBuilder. ^java.util.List [cc "--version"])
                (.redirectErrorStream true) (.start))]
      (.waitFor p) (zero? (.exitValue p)))
    (catch Exception _ false)))

(defn- gen [n seed]
  (let [a (float-array n) r (java.util.Random. seed)]
    (dotimes [i n] (aset a i (float (- (.nextDouble r) 0.5)))) a))

(defn- ref-matmul [^floats W-dq ^floats x-dq out in]
  (let [y (float-array out)]
    (dotimes [o out]
      (aset y o (float (areduce x-dq k s 0.0
                                (+ s (* (aget W-dq (+ (* (long o) (long in)) k)) (aget x-dq k)))))))
    y))

(defn- dequant-act [{:keys [xq xs]} in]
  (let [d (float-array in) dact (aget ^floats xs 0)]    ; in = one super-block
    (dotimes [k in] (aset d k (float (* dact (aget ^bytes xq k))))) d))

(deftest q4k-composable-c
 (when (clang-available?)
  (testing "composable Q4_K GEMV → C matches dequant-matmul"
    (let [out 8 in 256
          W (gen (* out in) 1) x (gen in 2)
          {:keys [wq da db aq bq] :as ew} (q/quantize-weight-q4k W q/q4-K)
          {:keys [xq xs bsums] :as ea} (q/quantize-act-q8k x in q/q4-K)
          cfn (pipeline/compile-aot #'qk/qmatmul-q4k-composable! :target :c)
          y (float-array out)]
      (cfn xq xs bsums wq da db aq bq y in out 0 out)
      (let [yref (ref-matmul (q/dequant-q4k ew q/q4-K (* out in)) (dequant-act ea in) out in)]
        (dotimes [o out]
          (is (< (Math/abs (- (aget y o) (aget yref o))) 1e-2)
              (str "Q4_K row " o ": C " (aget y o) " vs ref " (aget yref o)))))))))

(deftest q6k-composable-c
 (when (clang-available?)
  (testing "composable Q6_K GEMV → C matches dequant-matmul"
    (let [out 8 in 256
          W (gen (* out in) 3) x (gen in 4)
          {:keys [wq sc ds] :as ew} (q/quantize-weight-q6k W q/q6-K)
          {:keys [xq xs bsums] :as ea} (q/quantize-act-q8k x in q/q6-K)
          cfn (pipeline/compile-aot #'qk/qmatmul-q6k-composable! :target :c)
          y (float-array out)]
      (cfn xq xs bsums wq sc ds y in out 0 out)
      (let [yref (ref-matmul (q/dequant-q6k ew q/q6-K (* out in)) (dequant-act ea in) out in)]
        (dotimes [o out]
          (is (< (Math/abs (- (aget y o) (aget yref o))) 1e-2)
              (str "Q6_K row " o ": C " (aget y o) " vs ref " (aget yref o)))))))))
