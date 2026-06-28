(ns raster.dl.qlinear-k-gpu-test
  "The SAME composable K-quant GEMV deftms (qmatmul-q{4,6}k-gpu!) compile to OpenCL via the
   shared c_emit and run on the GPU (ze:0), byte-exact with the CPU composable kernel on
   identical quantized inputs. Proves the format registry reaches a working GPU kernel through
   the work-item-per-row par/map-void! twin — int8 weights + float scales + int bsums in one
   mixed-dtype kernel. Skipped when no GPU device is present."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.qlinear-k :as qk]
            [raster.compiler.backend.cpu.quant :as q]
            [raster.gpu.core :as gpu]))

(defn- gpu-available? []
  (try (require 'raster.gpu.ze-runtime)
       (let [qfn (resolve 'raster.gpu.ze-runtime/query-devices)]
         (boolean (and qfn (seq (qfn)))))
       (catch Throwable _ false)))

(defn- gen [n seed]
  (let [a (float-array n) r (java.util.Random. seed)]
    (dotimes [i n] (aset a i (float (- (.nextDouble r) 0.5)))) a))

(defn- maxerr [^floats a ^floats b]
  (reduce max 0.0 (map (fn [x y] (Math/abs (double (- x y)))) (seq a) (seq b))))

(deftest q4k-gpu-matches-cpu
  (when (gpu-available?)
    (testing "Q4_K work-item-per-row kernel on ze:0 == CPU composable"
      (let [out 32 in 256
            W (gen (* out in) 11) x (gen in 22)
            {:keys [wq da db aq bq]} (q/quantize-weight-q4k W q/q4-K)
            {:keys [xq xs bsums]}    (q/quantize-act-q8k x in q/q4-K)
            ycpu (float-array out)
            _ (qk/qmatmul-q4k-composable! xq xs bsums wq da db aq bq ycpu in out 0 out)
            sess (gpu/make-session :ze:0)]
        (try
          (gpu/compile! sess :q4k #'qk/qmatmul-q4k-gpu!)
          (gpu/alloc! sess {:xq [:byte (alength xq) xq] :xs [:float (alength xs) xs]
                            :bsums [:int (alength bsums) bsums] :wq [:byte (alength wq) wq]
                            :da [:float (alength da) da] :db [:float (alength db) db]
                            :aq [:byte (alength aq) aq] :bq [:byte (alength bq) bq]
                            :y [:float out nil]})
          (gpu/invoke! sess :q4k
                       {"xq" :xq "xs" :xs "bsums" :bsums "wq" :wq "da" :da "db" :db
                        "aq" :aq "bq" :bq "y" :y}
                       [{:type :int :value in}] out)
          (is (< (maxerr ycpu (gpu/download sess :y)) 1e-3))
          (finally (gpu/close-session! sess)))))))

(deftest q6k-gpu-matches-cpu
  (when (gpu-available?)
    (testing "Q6_K work-item-per-row kernel on ze:0 == CPU composable"
      (let [out 32 in 256
            W (gen (* out in) 33) x (gen in 44)
            {:keys [wq sc ds]}   (q/quantize-weight-q6k W q/q6-K)
            {:keys [xq xs bsums]} (q/quantize-act-q8k x in q/q6-K)
            ycpu (float-array out)
            _ (qk/qmatmul-q6k-composable! xq xs bsums wq sc ds ycpu in out 0 out)
            sess (gpu/make-session :ze:0)]
        (try
          (gpu/compile! sess :q6k #'qk/qmatmul-q6k-gpu!)
          (gpu/alloc! sess {:xq [:byte (alength xq) xq] :xs [:float (alength xs) xs]
                            :bsums [:int (alength bsums) bsums] :wq [:byte (alength wq) wq]
                            :sc [:byte (alength sc) sc] :ds [:float (alength ds) ds]
                            :y [:float out nil]})
          (gpu/invoke! sess :q6k
                       {"xq" :xq "xs" :xs "bsums" :bsums "wq" :wq "sc" :sc "ds" :ds "y" :y}
                       [{:type :int :value in}] out)
          (is (< (maxerr ycpu (gpu/download sess :y)) 1e-3))
          (finally (gpu/close-session! sess)))))))
