(ns raster.quant.kernels-k-test
  "The composable K-quant GEMV deftms compile to C (compile-aot :target :c) and match the
   dequant-matmul reference — proving the registry's formats reach a working C kernel via
   the SAME composable path the legacy Q4_0 uses (and that the GPU/OpenCL path will reuse).
   Single-call correctness; no spin-pool, no Valhalla."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.quant.kernels-k :as qk]
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

(defn- bytes->ints-le [^bytes values]
  (let [result (int-array (quot (alength values) 4))
        buffer (.order (java.nio.ByteBuffer/wrap values) java.nio.ByteOrder/LITTLE_ENDIAN)]
    (dotimes [i (alength result)]
      (aset result i (.getInt buffer (* i 4))))
    result))

(defn- float-row [^floats values row width]
  (let [result (float-array width)]
    (System/arraycopy values (* row width) result 0 width)
    result))

(defn- pad-rows [^floats values nrows width padded-width]
  (let [result (float-array (* nrows padded-width))]
    (dotimes [row nrows]
      (System/arraycopy values (* row width) result (* row padded-width) width))
    result))

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

(deftest q8k-activation-quantization-has-an-ordinary-row-axis
  (let [nrows 3 in 512 nsb (quot in 256) nsub (quot in 32)
        x (gen (* nrows in) 71)
        xp (int-array (* nrows (quot in 4)))
        xs (float-array (* nrows nsb))
        bsums (int-array (* nrows nsub))
        submax (float-array (* nrows nsub))]
    (qk/quant-act-q8k-rows-gpu! x xp xs bsums submax in nrows)
    (dotimes [row nrows]
      (let [{expected-xq :xq expected-xs :xs expected-bsums :bsums}
            (q/quantize-act-q8k (float-row x row in) in q/q4-K)
            expected-xp (bytes->ints-le expected-xq)]
        (is (= (vec expected-xp)
               (subvec (vec xp) (* row (quot in 4)) (* (inc row) (quot in 4))))
            (str "packed activation row " row))
        (is (= (vec expected-xs)
               (subvec (vec xs) (* row nsb) (* (inc row) nsb)))
            (str "activation scales row " row))
        (is (= (vec expected-bsums)
               (subvec (vec bsums) (* row nsub) (* (inc row) nsub)))
            (str "activation block sums row " row))))))

(deftest q8k-padded-row-quantization-does-not-materialize-layout-padding
  (let [nrows 3 width 640 padded-in 768
        nsub (quot padded-in 32) nsb (quot padded-in 256)
        dense (gen (* nrows width) 73)
        padded (pad-rows dense nrows width padded-in)
        expected-xp (int-array (* nrows (quot padded-in 4)))
        expected-xs (float-array (* nrows nsb))
        expected-bsums (int-array (* nrows nsub))
        expected-submax (float-array (* nrows nsub))
        actual-xp (int-array (alength expected-xp))
        actual-xs (float-array (alength expected-xs))
        actual-bsums (int-array (alength expected-bsums))
        actual-submax (float-array (alength expected-submax))]
    (qk/quant-act-q8k-rows-gpu! padded expected-xp expected-xs expected-bsums expected-submax
                                padded-in nrows)
    (qk/quant-act-q8k-padded-rows-gpu! dense actual-xp actual-xs actual-bsums actual-submax
                                       width padded-in nrows)
    (is (= (vec expected-xp) (vec actual-xp)) "packed words equal explicit zero padding")
    (is (= (vec expected-xs) (vec actual-xs)) "super-block scales equal explicit zero padding")
    (is (= (vec expected-bsums) (vec actual-bsums)) "sub-block sums equal explicit zero padding")
    (is (= (vec expected-submax) (vec actual-submax)) "reduction scratch preserves row boundaries")
    (let [equal-xp (int-array (alength expected-xp))
          equal-xs (float-array (alength expected-xs))
          equal-bsums (int-array (alength expected-bsums))
          equal-submax (float-array (alength expected-submax))]
      (qk/quant-act-q8k-padded-rows-gpu! padded equal-xp equal-xs equal-bsums equal-submax
                                         padded-in padded-in nrows)
      (is (= (vec expected-xp) (vec equal-xp)) "equal-width adapter agrees with branch-free packing")
      (is (= (vec expected-xs) (vec equal-xs)) "equal-width adapter agrees on scales")
      (is (= (vec expected-bsums) (vec equal-bsums)) "equal-width adapter agrees on sums"))))

(deftest q4k-projection-shares-weights-across-activation-rows
  (let [nrows 3 in 512 out 11
        x (gen (* nrows in) 81)
        W (gen (* out in) 82)
        {:keys [wq da db aq bq]} (q/quantize-weight-q4k W q/q4-K)
        wp (bytes->ints-le wq)
        xp (int-array (* nrows (quot in 4)))
        xs (float-array (* nrows (quot in 256)))
        bsums (int-array (* nrows (quot in 32)))
        submax (float-array (* nrows (quot in 32)))
        actual (float-array (* nrows out))]
    (qk/quant-act-q8k-rows-gpu! x xp xs bsums submax in nrows)
    (qk/qmatmul-q4k-dp4a-rows! xp xs bsums wp da db aq bq actual in out nrows)
    (dotimes [row nrows]
      (let [{:keys [xq xs bsums]} (q/quantize-act-q8k (float-row x row in) in q/q4-K)
            expected (float-array out)]
        (qk/qmatmul-q4k-composable! xq xs bsums wq da db aq bq expected in out 0 out)
        (dotimes [o out]
          (is (< (Math/abs (- (aget actual (+ (* row out) o)) (aget expected o))) 1e-3)
              (str "Q4_K row " row ", output " o)))))))

(deftest row-capable-q4k-path-lowers-through-the-shared-gpu-pipeline
  (let [quant-kernels (:kernels (pipeline/show-pipeline #'qk/quant-act-q8k-rows-gpu!
                                                        :target-device :ze:0 :dtype :float))
        padded-kernels (:kernels (pipeline/show-pipeline #'qk/quant-act-q8k-padded-rows-gpu!
                                                         :target-device :ze:0 :dtype :float))
        projection-kernels (:kernels (pipeline/show-pipeline #'qk/qmatmul-q4k-dp4a-rows!
                                                             :target-device :ze:0 :dtype :float))]
    (is (= 2 (count quant-kernels)) "Q8_K remains an ordered two-phase reduction")
    (is (= '[[[submax :float] [x :float] [_n_bound :int]]
             [[bsums :int] [submax :float] [x :float] [xp :int]
              [xs :float] [in :int] [_n_bound :int]]]
           (mapv #(mapv (juxt :name :dtype) (:abi %)) quant-kernels)))
    (is (= '[[[submax :float] [x :float] [padded-in :int] [width :int] [_n_bound :int]]
             [[bsums :int] [submax :float] [x :float] [xp :int] [xs :float]
              [padded-in :int] [width :int] [_n_bound :int]]]
           (mapv #(mapv (juxt :name :dtype) (:abi %)) padded-kernels))
        "the adapter changes layout indexing without changing the two-phase Q8_K representation")
    (is (every? #(re-find #"col < .*width" (:source %)) padded-kernels)
        "both phases guard dense-row reads and synthesize the padding region")
    (is (= 1 (count projection-kernels)))
    (is (= '[[aq :byte] [bq :byte] [bsums :int] [da :float] [db :float]
             [wp :int] [xp :int] [xs :float] [y :float]
             [in :int] [out :int] [_n_bound :int]]
           (mapv (juxt :name :dtype) (:abi (first projection-kernels)))))
    (is (re-find #"rstr_dp4a" (:source (first projection-kernels)))
        "Q4_K row projection reaches the target-neutral integer-dot intrinsic")
    (is (not (re-find #"\\bdouble\\b" (:source (first projection-kernels))))
        "the float projection does not accidentally promote accumulation to FP64")))
