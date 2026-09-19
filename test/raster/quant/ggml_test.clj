(ns raster.quant.ggml-test
  "Raster's ggml formats against llama.cpp's own output, byte for byte.

  The fixtures in test/resources/ggml_oracle come from dev/ggml_oracle, which
  calls ggml_quantize_chunk and to_float in a pinned llama.cpp build."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [raster.quant.ggml :as ggml]))

(def ^:private manifest
  (delay (edn/read-string (slurp (io/resource "resources/ggml_oracle/manifest.edn")))))

(defn- fixture-bytes ^bytes [file]
  (with-open [in (io/input-stream (io/resource (str "resources/ggml_oracle/" file)))]
    (.readAllBytes in)))

(defn- first-difference
  "Index of the first differing byte, or nil."
  [^bytes a ^bytes b]
  (if-not (= (alength a) (alength b))
    :length
    (first (filter #(not= (aget a (int %)) (aget b (int %))) (range (alength a))))))

(defn- float-bits [^floats xs]
  (mapv #(Float/floatToRawIntBits %) xs))

(deftest scalar-semantics-match-c
  (testing "roundf rounds halves away from zero"
    (is (= [3 -3 2 -2 0 0] (mapv ggml/roundf [2.5 -2.5 2.4999 -2.4999 0.4 -0.4]))))
  (testing "nearest_int rounds halves to even"
    (is (= [2 -2 4 0 0] (mapv ggml/nearest-int [2.5 -2.5 3.5 0.5 -0.5]))))
  (testing "FP16 conversion rounds to nearest even and keeps subnormals"
    (is (= 0x3C00 (ggml/fp32->fp16-bits 1.0)))
    (is (= 0x0001 (ggml/fp32->fp16-bits 5.9604645E-8)))
    (is (= 0x7C00 (ggml/fp32->fp16-bits 1.0e6)))))

(defn- check-format [format]
  (doseq [{:keys [id n-per-row nrows input formats]} (:quantize @manifest)
          :let [{:keys [blocks dequant]} (get formats format)]
          :when blocks]
    (let [x (ggml/read-f32 (fixture-bytes input))
          expected (fixture-bytes blocks)
          actual (ggml/quantize format x n-per-row nrows)]
      (testing (str (name format) " " (name id))
        (is (nil? (first-difference expected actual))
            (str "first differing byte " (first-difference expected actual)))
        (is (= (float-bits (ggml/read-f32 (fixture-bytes dequant)))
               (float-bits (ggml/dequantize format expected (* n-per-row nrows))))
            "dequantizing llama.cpp's blocks reproduces its to_float bit for bit")))))

(deftest q8-0-matches-llama-cpp (check-format :q8_0))

(deftest q5-0-matches-llama-cpp (check-format :q5_0))

(deftest q4-K-matches-llama-cpp (check-format :q4_K))

(deftest q6-K-matches-llama-cpp (check-format :q6_K))

(deftest activations-match-the-reference-quantizers
  ;; q8_0 and q5_0 weights take q8_0 activations; q4_K and q6_K take q8_K.
  ;; Raster follows the platform-independent reference. For q8_0 the AVX2
  ;; backend rounds ties to even instead of away from zero, which the tie
  ;; fixtures record; for q8_K both round to even.
  (doseq [{:keys [id n activations activation-format activation-ref-blocks activation-blocks]}
          (:dot @manifest)]
    (let [x (ggml/read-f32 (fixture-bytes activations))
          actual (ggml/quantize activation-format x n 1)]
      (testing id
        (is (nil? (first-difference (fixture-bytes activation-ref-blocks) actual)))
        (when (and (.endsWith ^String id "-ties") (= :q8_0 activation-format))
          (is (some? (first-difference (fixture-bytes activation-blocks) actual))
              "the AVX2 backend differs from the reference on exact ties"))))))

(deftest generic-dot-products-match-llama-cpp
  ;; The scalar generic vec_dot over reference activations, bit for bit. The
  ;; manifest records which float steps this llama.cpp build contracted into
  ;; FMA; the reference reproduces exactly that, and without contraction it
  ;; is the ISO C evaluation of the same source.
  (let [contraction (:contraction @manifest)]
    (doseq [{:keys [id n weights activations activation-format generic-result-bits]
             weight-format :format}
            (:dot @manifest)]
      (let [w (ggml/quantize weight-format (ggml/read-f32 (fixture-bytes weights)) n 1)
            x (ggml/quantize activation-format (ggml/read-f32 (fixture-bytes activations)) n 1)
            s (ggml/vec-dot weight-format w x n
                            {:contract? (= :fma (get contraction weight-format))})]
        (testing id
          (is (= generic-result-bits (format "%08x" (Float/floatToRawIntBits (float s))))
              (str "got " s)))))))
