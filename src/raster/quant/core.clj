(ns raster.quant.core
  "Quantized tensors — the compositional, extensible quantization surface.

  Lives OUTSIDE dl/ on purpose: quantization is a numeric/tensor concern (a matrix
  type + a specialized matmul), like linalg — not deep-learning-specific. A
  `QMatrix` wraps a format's packed weight arrays + its descriptor; `matmul`
  dispatches on the format to its registered GEMV, which reuses the shared int8-MAC
  primitive (raster.quant.qlinear-composable/wi8-dot…) and lowers per backend (maddubs
  on CPU-C, dp4a on GPU, i32x4.dot on wasm).

  EXTENDING (add a new format, e.g. Q5_K) — three things, no compiler internals and
  no per-backend code:
    1. a FORMAT DESCRIPTOR (data): block structure {super-block, subblock, bits, …}
    2. a GEMV deftm that unpacks + scale-decodes + calls the shared int8-MAC primitive
    3. (register-quant! :q5_K {:descriptor … :quantize … :apply …})
  then `(matmul (qmatrix :q5_K W rows cols) x)` just works, on every backend the
  base kernels support. (After the #28 skeleton-factoring, step 2 shrinks to a
  descriptor + a small unpack fn.)"
  (:require [raster.compiler.backend.cpu.quant :as q]
            [raster.quant.qlinear-k :as qk]))

(defrecord QMatrix [format rows cols arrays])

(defonce ^{:doc "format-key → {:descriptor data, :quantize (fn [W rows cols descriptor]→arrays),
             :apply (fn [qmatrix x-floats]→y-floats)}"}
  quant-registry (atom {}))

(defn register-quant!
  "Register a quantization format. `spec` is
     {:descriptor <data map>
      :quantize   (fn [^floats W rows cols descriptor] → {packed weight arrays})
      :apply      (fn [QMatrix ^floats x] → ^floats y)}   ; quantizes x, runs the GEMV
   The :apply wrapper owns the format-specific activation-quant + arg marshaling, so
   `matmul` stays uniform across formats."
  [fmt-key spec]
  (swap! quant-registry assoc fmt-key spec))

(defn registered-formats [] (set (keys @quant-registry)))

(defn qmatrix
  "Quantize a row-major float weight matrix [rows×cols] into a QMatrix{fmt-key}."
  [fmt-key ^floats W rows cols]
  (let [{:keys [descriptor quantize]} (or (get @quant-registry fmt-key)
                                          (throw (ex-info (str "unknown quant format " fmt-key)
                                                          {:known (registered-formats)})))]
    (->QMatrix fmt-key rows cols (quantize W rows cols descriptor))))

(defn matmul
  "y = QMatrix · x  — dispatches on the QMatrix's format to its registered GEMV."
  ^floats [qm ^floats x]
  ((:apply (get @quant-registry (:format qm))) qm x))

;; ── built-in formats (reuse the composable kernels; the int8-MAC is the shared seam) ──

(register-quant! :q4_K
  {:descriptor q/q4-K
   :quantize (fn [W _rows _cols descriptor] (q/quantize-weight-q4k W descriptor))
   :apply (fn [qm x]
            (let [{:keys [wq da db aq bq]} (:arrays qm)
                  in (long (:cols qm)) out (long (:rows qm))
                  {:keys [xq xs bsums]} (q/quantize-act-q8k x in q/q4-K)
                  y (float-array out)]
              (qk/qmatmul-q4k-composable! xq xs bsums wq da db aq bq y in out 0 out)
              y))})

(register-quant! :q6_K
  {:descriptor q/q6-K
   :quantize (fn [W _rows _cols descriptor] (q/quantize-weight-q6k W descriptor))
   :apply (fn [qm x]
            (let [{:keys [wq sc ds]} (:arrays qm)
                  in (long (:cols qm)) out (long (:rows qm))
                  {:keys [xq xs bsums]} (q/quantize-act-q8k x in q/q6-K)
                  y (float-array out)]
              (qk/qmatmul-q6k-composable! xq xs bsums wq sc ds y in out 0 out)
              y))})
