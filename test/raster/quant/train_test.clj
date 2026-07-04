(ns raster.quant.train-test
  "AD through a FROZEN quantized weight — the LoRA/QLoRA base-layer primitive.
   Pins that qlinear-q8's pullback dx = dequant(W)^T·dy is the exact gradient of
   the forward y = dequant(W)@x, in BOTH the runtime and compile-aot AD paths, and
   that a LoRA adapter composed on the frozen base trains."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.quant.train :as qt]
            [raster.core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.dl.nn :as nn]
            [raster.ad.reverse :as rev]
            [raster.compiler.pipeline :as pl]))

(defn- fa ^floats [n seed]
  (let [r (java.util.Random. (long seed)) a (float-array n)]
    (dotimes [i n] (aset a i (float (* 0.3 (.nextGaussian r))))) a))

(deftest q8-codec-roundtrip
  (testing "dequant(quantize(W)) recovers W to Q8 precision"
    (let [out 4 in 64 W (fa (* out in) 1)
          {:keys [codes scales]} (qt/q8-quantize W out in)
          Wr (qt/dequant-q8-dense codes scales in out)
          err (reduce max (map #(Math/abs (- (aget W %) (aget Wr %))) (range (* out in))))]
      (is (< err 0.01) (str "Q8 round-trip err=" err)))))

(deftest qlinear-forward-matches-dense
  (testing "the dequant-fused forward equals dense dequant(W)@x"
    (let [out 4 in 64 m 2 W (fa (* out in) 1) X (fa (* m in) 2)
          {:keys [codes scales]} (qt/q8-quantize W out in)
          Wd (qt/dequant-q8-dense codes scales in out)
          y  (qt/qlinear-q8 X codes scales m in out)
          ref (let [r (float-array (* m out))]
                (dotimes [p m] (dotimes [o out]
                  (aset r (+ (* p out) o)
                        (float (reduce + (map #(* (aget X (+ (* p in) %)) (aget Wd (+ (* o in) %)))
                                              (range in)))))))
                r)
          err (reduce max (map #(Math/abs (- (aget ^floats y %) (aget ^floats ref %)))
                               (range (* m out))))]
      (is (< err 1e-5) (str "forward vs dense err=" err)))))

(deftm qt-loss [x :- (Array float) codes :- (Array byte) scales :- (Array float)
                tgt :- (Array float) m :- Long in :- Long out :- Long] :- Double
  (let [y (raster.quant.train/qlinear-q8 x codes scales m in out)
        n* (clojure.core/* (long m) (long out))]
    (loop [i 0 s 0.0]
      (if (clojure.core/< i n*)
        (let [e (- (ra/aget y i) (ra/aget tgt i))]
          (recur (clojure.core/inc i) (+ s (* 0.5 (* e e)))))
        s))))

(deftest dx-pullback-numerically-correct
  (testing "dL/dx through the frozen quantized weight matches finite differences"
    (let [out 4 in 64 m 2
          W (fa (* out in) 1) x (fa (* m in) 2) tgt (fa (* m out) 3)
          {:keys [codes scales]} (qt/q8-quantize W out in)
          gx (second ((rev/value+grad #'qt-loss) x codes scales tgt m in out))
          eps 1e-3
          num (fn [idx]
                (let [x0 (aget ^floats x idx)
                      f (fn [xv] (aset ^floats x idx (float xv))
                          (qt-loss x codes scales tgt m in out))
                      fp (f (+ x0 eps)) fm (f (- x0 eps))]
                  (aset ^floats x idx (float x0))
                  (/ (- fp fm) (* 2.0 eps))))]
      (doseq [idx [3 17 40 63 64 100]]
        (is (< (Math/abs (- (double (aget ^floats gx idx)) (num idx))) 2e-3)
            (str "dx[" idx "] analytic=" (aget ^floats gx idx) " numeric=" (num idx)))))))

(deftm qt-step [x :- (Array float) codes :- (Array byte) scales :- (Array float)
                tgt :- (Array float) m :- Long in :- Long out :- Long] :- Double
  (double (first ((raster.ad.reverse/value+grad #'raster.quant.train-test/qt-loss)
                  x codes scales tgt m in out))))

(deftest compiled-ad-matches-runtime
  (testing "compile-aot AD through qlinear equals the interpreted value"
    (let [out 4 in 64 m 2
          W (fa (* out in) 1) x (fa (* m in) 2) tgt (fa (* m out) 3)
          {:keys [codes scales]} (qt/q8-quantize W out in)
          f (pl/compile-aot #'qt-step)
          interp (qt-loss x codes scales tgt m in out)
          comp   (f x codes scales tgt m in out)]
      (is (< (Math/abs (- interp comp)) 1e-4)
          (str "interp=" interp " compiled=" comp)))))

(deftm lora-q8-loss [x :- (Array float) codes :- (Array byte) scales :- (Array float)
                     A :- (Array float) B :- (Array float) zr :- (Array float) zb :- (Array float)
                     tgt :- (Array float) m :- Long in :- Long out :- Long r :- Long] :- Double
  (let [base (raster.quant.train/qlinear-q8 x codes scales m in out)
        ax   (raster.dl.nn/linear x A zr m in r)
        bax  (raster.dl.nn/linear ax B zb m r out)
        n*   (clojure.core/* (long m) (long out))]
    (loop [i 0 s 0.0]
      (if (clojure.core/< i n*)
        (let [y (+ (ra/aget base i) (* 2.0 (ra/aget bax i)))
              e (- y (ra/aget tgt i))]
          (recur (clojure.core/inc i) (+ s (* 0.5 (* e e)))))
        (/ s (double n*))))))

(deftest lora-adapter-trains-on-frozen-base
  (testing "SGD on A,B over a frozen quantized base drops the loss"
    (let [out 8 in 64 m 4 r 4
          W (fa (* out in) 1) x (fa (* m in) 2) tgt (fa (* m out) 9)
          {:keys [codes scales]} (qt/q8-quantize W out in)
          A (fa (* r in) 5) B (float-array (* out r))
          zr (float-array r) zb (float-array out)
          vg (rev/value+grad #'lora-q8-loss)
          lr 0.05
          l0 (first (vg x codes scales A B zr zb tgt m in out r))]
      (dotimes [_ 300]
        (let [res (vg x codes scales A B zr zb tgt m in out r)
              gA (nth res 4) gB (nth res 5)]
          (dotimes [i (* r in)]  (aset ^floats A i (float (- (aget ^floats A i) (* lr (aget ^floats gA i))))))
          (dotimes [i (* out r)] (aset ^floats B i (float (- (aget ^floats B i) (* lr (aget ^floats gB i))))))))
      (let [lN (first (vg x codes scales A B zr zb tgt m in out r))]
        (is (< lN (* 0.25 l0)) (str "loss0=" l0 " lossN=" lN))))))
