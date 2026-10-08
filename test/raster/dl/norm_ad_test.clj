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

;; ── rms-norm-chunked: the GPU (two-stage) SCHEDULE of the same op ───────────────
;; Same math, chunk-parallel decomposition (raster.dl.nn's schedule note). The gates
;; are (a) it agrees with the row-parallel rms-norm to reassociation tolerance for
;; every chunk count — including the awkward ones (chunks ∤ features, chunks=1,
;; chunks > features, rows=1) — and (b) its rrule's gradients match finite differences.

(defn- da ^doubles [n seed]
  (let [r (java.util.Random. (long seed)) a (double-array n)]
    (dotimes [i n] (aset a i (.nextGaussian r))) a))

(defn- maxrel [a b]
  (let [n (ra/alength a)]
    (loop [i 0 m 0.0]
      (if (< i n)
        (let [x (double (ra/aget a i)) y (double (ra/aget b i))]
          (recur (inc i) (max m (/ (Math/abs (- x y)) (max 1.0e-9 (Math/abs y))))))
        m))))

(defn- rms-dx-reference
  "Independent Double arithmetic, with a cancellation-aware backward-error scale.
   dx subtracts two terms; their sum of magnitudes conditions that subtraction."
  [dy x w rows features eps gain]
  (let [values (double-array (* rows features))
        scales (double-array (* rows features))]
    (dotimes [row rows]
      (let [offset (* row features)
            [sq dot] (loop [i 0 sq 0.0 dot 0.0]
                       (if (= i features)
                         [sq dot]
                         (let [j (+ offset i) xi (double (ra/aget x j))
                               gi (+ gain (double (ra/aget w i)))]
                           (recur (inc i) (+ sq (* xi xi))
                                  (+ dot (* gi xi (double (ra/aget dy j))))))))
            inv (/ 1.0 (Math/sqrt (+ (/ sq features) eps)))
            factor (/ (* inv inv inv dot) features)]
        (dotimes [i features]
          (let [j (+ offset i)
                a (* inv (+ gain (double (ra/aget w i))) (double (ra/aget dy j)))
                b (* factor (double (ra/aget x j)))]
            (aset values j (- a b))
            (aset scales j (+ (Math/abs a) (Math/abs b)))))))
    {:values values :scales scales}))

(defn- maxscaled [actual expected scales]
  (reduce max 0.0
          (map (fn [a b scale]
                 (let [error (Math/abs (- (double a) (double b)))
                       scale (double scale)]
                   (cond
                     (or (not (Double/isFinite scale)) (neg? scale)) Double/POSITIVE_INFINITY
                     (zero? scale) (if (zero? error) 0.0 Double/POSITIVE_INFINITY)
                     :else (/ error scale))))
               actual expected scales)))

(deftest rms-norm-chunked-matches-row-parallel
  (testing "f64: chunked forward + backward-dx agree with the row-parallel schedule"
    (doseq [[rows feat chunks] [[8 64 8] [8 64 1] [8 64 7] [8 64 64] [8 64 128]
                                [1 640 32] [64 640 32] [16 32 13]]]
      (let [go 1.0 eps 1e-6
            x (da (* rows feat) 11) w (da feat 12) dy (da (* rows feat) 13)
            y0 (nn/rms-norm x w rows feat eps go)
            y1 (nn/rms-norm-chunked x w rows feat chunks eps go)
            d0 (nn/rms-norm-backward-dx dy x w rows feat eps go)
            d1 (nn/rms-norm-chunked-backward-dx dy x w rows feat chunks eps go)
            {:keys [values scales]} (rms-dx-reference dy x w rows feat eps go)
            tag (str "rows=" rows " feat=" feat " chunks=" chunks)]
        ;; Forward reassociation uses relative error; backward subtractive
        ;; cancellation uses the independent reference's two term magnitudes.
        (is (< (maxrel y1 y0) 1e-12) (str "fwd " tag))
        (is (< (maxscaled d1 d0 scales) 1e-9) (str "bwd-dx " tag))
        (is (< (maxscaled d1 values scales) 1e-9) (str "chunked reference " tag))
        (is (< (maxscaled d0 values scales) 1e-9) (str "rowwise reference " tag)))))
  (testing "f32: same, at f32 tolerance"
    (let [rows 64 feat 640 chunks 32 go 1.0 eps 1e-6
          x (fa (* rows feat) 21) w (fa feat 22) dy (fa (* rows feat) 23)]
      (is (< (maxrel (nn/rms-norm-chunked x w rows feat chunks eps go)
                     (nn/rms-norm x w rows feat eps go)) 1e-4) "fwd f32")
      (let [chunked (nn/rms-norm-chunked-backward-dx dy x w rows feat chunks eps go)
            rowwise (nn/rms-norm-backward-dx dy x w rows feat eps go)
            {:keys [values scales]} (rms-dx-reference dy x w rows feat eps go)]
        ;; A coordinate's dx may nearly cancel. Dividing by that result measures
        ;; conditioning, not reduction accuracy. Keep the same bound, using an
        ;; independent mathematical reference's two term magnitudes as the scale.
        (is (< (maxscaled chunked rowwise scales) 1e-3) "bwd-dx f32 reassociation")
        (is (< (maxscaled chunked values scales) 1e-3) "chunked vs independent dx")
        (is (< (maxscaled rowwise values scales) 1e-3) "rowwise vs independent dx")))))

(deftm rms-chunked-loss [x :- (Array float) w :- (Array float) tgt :- (Array float)
                         rows :- Long feat :- Long chunks :- Long
                         eps :- Double go :- Double] :- Double
  (let [y (raster.dl.nn/rms-norm-chunked x w rows feat chunks eps go)
        n* (clojure.core/* rows feat)]
    (loop [i 0 s 0.0]
      (if (clojure.core/< i n*)
        (let [e (- (ra/aget y i) (ra/aget tgt i))]
          (recur (clojure.core/inc i) (+ s (* 0.5 (* e e)))))
        s))))

(deftest rms-norm-chunked-grad
  (testing "the chunked rrule's dx and dweight match finite differences"
    (let [rows 3 feat 16 chunks 4 go 1.0 eps 1e-6
          x (fa (* rows feat) 1) w (fa feat 2) tgt (fa (* rows feat) 3)
          res ((rev/value+grad #'rms-chunked-loss) x w tgt rows feat chunks eps go)
          gx (nth res 1) gw (nth res 2)
          f  (fn [] (rms-chunked-loss x w tgt rows feat chunks eps go))]
      (doseq [i [0 5 17 40 47]]
        (is (< (Math/abs (- (double (aget ^floats gx i)) (fd f x i 1e-3))) 3e-3)
            (str "dx[" i "]")))
      (doseq [i [0 7 15]]
        (is (< (Math/abs (- (double (aget ^floats gw i)) (fd f w i 1e-3))) 3e-3)
            (str "dweight[" i "]"))))))

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
