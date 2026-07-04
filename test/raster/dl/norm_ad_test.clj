(ns raster.dl.norm-ad-test
  "AD rules for the modern-decoder op variants that Gemma/Qwen fine-tuning needs:
   rms-norm (dx + dweight, with the Gemma gain-offset) and bias-free linear-nb.
   Each analytic gradient is checked against central finite differences."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.nn :as nn]
            [raster.core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.ad.reverse :as rev]))

(defn- fa ^floats [n seed]
  (let [r (java.util.Random. (long seed)) a (float-array n)]
    (dotimes [i n] (aset a i (float (.nextGaussian r)))) a))

(defn- fd [f ^floats arr idx eps]
  (let [a0 (aget arr idx)
        fp (do (aset arr idx (float (+ a0 eps))) (f))
        fm (do (aset arr idx (float (- a0 eps))) (f))]
    (aset arr idx (float a0))
    (/ (- fp fm) (* 2.0 eps))))

(deftm rms-loss [x :- (Array float) w :- (Array float) tgt :- (Array float)
                 rows :- Long feat :- Long eps :- Double go :- Double] :- Double
  (let [y (raster.dl.nn/rms-norm x w rows feat eps go)
        n* (clojure.core/* rows feat)]
    (loop [i 0 s 0.0]
      (if (clojure.core/< i n*)
        (let [e (- (ra/aget y i) (ra/aget tgt i))]
          (recur (clojure.core/inc i) (+ s (* 0.5 (* e e)))))
        s))))

(deftest rms-norm-grad
  (testing "rms-norm dx and dweight match finite differences (Gemma gain-offset 1.0)"
    (let [rows 3 feat 16 go 1.0 eps 1e-6
          x (fa (* rows feat) 1) w (fa feat 2) tgt (fa (* rows feat) 3)
          res ((rev/value+grad #'rms-loss) x w tgt rows feat eps go)
          gx (nth res 1) gw (nth res 2)
          f  (fn [] (rms-loss x w tgt rows feat eps go))]
      (doseq [i [0 5 17 40 47]]
        (is (< (Math/abs (- (double (aget ^floats gx i)) (fd f x i 1e-3))) 3e-3)
            (str "dx[" i "]")))
      (doseq [i [0 7 15]]
        (is (< (Math/abs (- (double (aget ^floats gw i)) (fd f w i 1e-3))) 3e-3)
            (str "dweight[" i "]"))))))

(deftest rms-norm-grad-llama-offset
  (testing "rms-norm also correct with gain-offset 0.0 (Llama/Qwen plain weight)"
    (let [rows 2 feat 32 go 0.0 eps 1e-6
          x (fa (* rows feat) 4) w (fa feat 5) tgt (fa (* rows feat) 6)
          res ((rev/value+grad #'rms-loss) x w tgt rows feat eps go)
          gx (nth res 1)
          f  (fn [] (rms-loss x w tgt rows feat eps go))]
      (doseq [i [0 33 63]]
        (is (< (Math/abs (- (double (aget ^floats gx i)) (fd f x i 1e-3))) 3e-3)
            (str "dx[" i "]"))))))

(deftm lnb-loss [x :- (Array float) W :- (Array float) tgt :- (Array float)
                 batch :- Long in :- Long out :- Long] :- Double
  (let [y (raster.dl.nn/linear-nb x W batch in out)
        n* (clojure.core/* batch out)]
    (loop [i 0 s 0.0]
      (if (clojure.core/< i n*)
        (let [e (- (ra/aget y i) (ra/aget tgt i))]
          (recur (clojure.core/inc i) (+ s (* 0.5 (* e e)))))
        s))))

(deftest linear-nb-grad
  (testing "bias-free linear dx and dW match finite differences"
    (let [batch 3 in 8 out 5
          x (fa (* batch in) 1) W (fa (* out in) 2) tgt (fa (* batch out) 3)
          res ((rev/value+grad #'lnb-loss) x W tgt batch in out)
          gx (nth res 1) gW (nth res 2)
          f (fn [] (lnb-loss x W tgt batch in out))]
      (doseq [i [0 3 20 23]]
        (is (< (Math/abs (- (double (aget ^floats gx i)) (fd f x i 1e-3))) 2e-3)
            (str "dx[" i "]")))
      (doseq [i [0 12 39]]
        (is (< (Math/abs (- (double (aget ^floats gW i)) (fd f W i 1e-3))) 2e-3)
            (str "dW[" i "]"))))))
