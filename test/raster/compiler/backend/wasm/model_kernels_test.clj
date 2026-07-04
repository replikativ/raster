(ns raster.compiler.backend.wasm.model-kernels-test
  "Differential execution tests for the model-kernel wasm dialect: real
   transformer kernels (raster.dl.nn / attention / quant) compiled to wasm at
   :dtype :float, executed via Chicory, compared against the SAME deftm run on
   the JVM. This is the correctness oracle behind the compile+validate smoke —
   wasm validation proves type-soundness, these prove value-equality.

   Also pins the regressions found while building the dialect:
   - stale form tags after :float specialization (walker re-derive fix)
   - .invk aget/aset must emit inline via elems (outlined callees monomorphize
     under the CALLER's dtype and e.g. f32-load an int array)
   - par/dp4a scalar lowering semantics (packed signed bytes, little-endian)"
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.pipeline :as pl]
            [raster.core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.dl.nn :as nn])
  (:import [com.dylibso.chicory.wasm Parser]
           [com.dylibso.chicory.runtime Instance]))

(defn- instantiate ^Instance [^bytes wasm]
  (-> (Instance/builder (Parser/parse wasm)) (.build)))

(defn- compile-f32 [v nm]
  (pl/compile-wasm v :name nm :dtype :float))

(defn- write-f32s! [mem base ^floats xs]
  (dotimes [i (alength xs)] (.writeF32 mem (+ base (* 4 i)) (aget xs i))))

(defn- read-f32s ^floats [mem base n]
  (let [out (float-array n)]
    (dotimes [i n] (aset out i (.readFloat mem (+ base (* 4 i)))))
    out))

(defn- max-abs-err ^double [^floats a ^floats b]
  (reduce (fn [mx i] (max (double mx)
                          (Math/abs (- (double (aget a i)) (double (aget b i))))))
          0.0 (range (alength a))))

(defn- randf ^floats [n seed]
  (let [r (java.util.Random. (long seed)) out (float-array n)]
    (dotimes [i n] (aset out i (float (- (.nextDouble r) 0.5))))
    out))

;; Hand-rolled Clojure references — an oracle EXTERNAL to raster dispatch,
;; mirroring the deftm bodies' f32 store discipline (float cast per store).

(defn- ref-residual-add ^floats [^floats a ^floats b n]
  (let [out (float-array n)]
    (dotimes [i n] (aset out i (float (+ (aget a i) (aget b i)))))
    out))

(defn- ref-gelu-mul ^floats [^floats gate ^floats up n]
  (let [out (float-array n)]
    (dotimes [i n]
      (let [g (double (aget gate i))]
        (aset out i (float (* (* 0.5 g (+ 1.0 (Math/tanh (* 0.7978845608028654
                                                            (+ g (* 0.044715 g g g))))))
                              (aget up i))))))
    out))

(defn- ref-rms-norm ^floats [^floats x ^floats w rows features eps gain]
  (let [out (float-array (* rows features))]
    (dotimes [r rows]
      (let [off (* r features)
            ms (/ (reduce (fn [s i] (let [v (double (aget x (+ off i)))] (+ s (* v v))))
                          0.0 (range features))
                  (double features))
            inv (/ 1.0 (Math/sqrt (+ ms (double eps))))]
        (dotimes [i features]
          (aset out (+ off i)
                (float (* (aget x (+ off i)) inv (+ (double gain) (aget w i))))))))
    out))

(deftest residual-add-differential
  (testing "residual-add! wasm f32 == JVM deftm"
    (let [n 257                                   ; non-power-of-2 on purpose
          a (randf n 1) b (randf n 2)
          jvm (ref-residual-add a b n)
          m (compile-f32 #'nn/residual-add! "residual_add")
          inst (instantiate (byte-array (:bytes m)))
          mem (.memory inst)
          [pa pb po] [0 (* 4 n) (* 8 n)]]
      (write-f32s! mem pa a) (write-f32s! mem pb b)
      (.apply (.export inst "residual_add") (long-array [pa pb po n]))
      (is (zero? (max-abs-err jvm (read-f32s mem po n)))
          "elementwise add must be bit-identical"))))

(deftest gelu-mul-differential
  (testing "gelu-mul! wasm f32 == JVM deftm (tanh-approx transcendental path)"
    (let [n 128
          gate (randf n 3) up (randf n 4)
          jvm (ref-gelu-mul gate up n)
          m (compile-f32 #'nn/gelu-mul! "gelu_mul")
          inst (instantiate (byte-array (:bytes m)))
          mem (.memory inst)
          [pg pu po] [0 (* 4 n) (* 8 n)]]
      (write-f32s! mem pg gate) (write-f32s! mem pu up)
      (.apply (.export inst "gelu_mul") (long-array [pg pu po n]))
      ;; wasm tanh is the emitter's f64 polynomial; JVM uses Math/tanh — allow
      ;; polynomial-approximation error, not structural error
      (let [err (max-abs-err jvm (read-f32s mem po n))]
        (is (< err 1e-5) (str "gelu max-err=" err))))))

(deftest rms-norm-differential
  (testing "rms-norm! wasm f32 == JVM deftm (reduce loop + rsqrt over rows)"
    (let [rows 4 features 64 n (* rows features)
          x (randf n 5) w (randf features 6)
          eps 1e-6 gain 0.0
          jvm (ref-rms-norm x w rows features eps gain)
          m (compile-f32 #'nn/rms-norm! "rms_norm")
          inst (instantiate (byte-array (:bytes m)))
          mem (.memory inst)
          [px pw po] [0 (* 4 n) (* 4 (+ n features))]]
      (write-f32s! mem px x) (write-f32s! mem pw w)
      (.apply (.export inst "rms_norm")
              (long-array [px pw po rows features
                           (Double/doubleToRawLongBits eps)
                           (Double/doubleToRawLongBits gain)]))
      (let [err (max-abs-err jvm (read-f32s mem po n))]
        (is (< err 1e-5) (str "rms-norm max-err=" err))))))

;; --- dp4a semantics: packed signed bytes, little-endian, 32-bit accumulate ---

(deftm dp4a-rows-k! [xp :- (Array int) yo :- (Array int) n :- Long] :- Void
  (raster.par/map-void! i n
    (ra/aset yo i
             (loop [k 0 d 0]
               (if (< k 2)
                 (recur (inc k) (raster.par/dp4a (ra/aget xp (+ (* i 2) k))
                                                 (ra/aget xp (+ (* i 2) k))
                                                 d))
                 d)))))

(deftest dp4a-scalar-lowering-differential
  (testing "par/dp4a wasm scalar lowering == JVM reference (self dot = Σ byte²)"
    (let [n 3
          ;; rows of 2 ints, each packing 4 signed bytes little-endian
          packs (int-array [(unchecked-int 0x01020304) (unchecked-int 0xFF7F8001)
                            (unchecked-int 0x00000000) (unchecked-int 0x7F7F7F7F)
                            (unchecked-int 0xDEADBEEF) (unchecked-int 0x11223344)])
          jvm (int-array n)
          _ (dp4a-rows-k! packs jvm n)
          m (pl/compile-wasm #'dp4a-rows-k! :name "dp4a_rows" :dtype :float)
          inst (instantiate (byte-array (:bytes m)))
          mem (.memory inst)
          [pp py] [0 (* 4 (alength packs))]]
      (dotimes [i (alength packs)] (.writeI32 mem (+ pp (* 4 i)) (aget packs i)))
      (.apply (.export inst "dp4a_rows") (long-array [pp py n]))
      (dotimes [i n]
        (is (= (aget jvm i) (.readInt mem (+ py (* 4 i))))
            (str "dp4a row " i " diverges from JVM reference"))))))
