(ns raster.sci.signal-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.sci.signal :refer [hann hamming blackman kaiser
                                       correlate convolve lfilter firwin sosfilt]]))

(def ^:const EPS 1e-6)

(defn- approx=
  ([^double a ^double b] (< (Math/abs (- a b)) EPS))
  ([^double a ^double b ^double tol] (< (Math/abs (- a b)) tol)))

;; ================================================================
;; Window functions: Parseval energy
;; ================================================================

(deftest hann-parseval-test
  (testing "Hann window: sum(w^2)/N ≈ 0.375"
    (let [n 256
          w (hann n)
          energy (loop [i 0 s 0.0]
                   (if (>= i n) (/ s (double n))
                       (recur (inc i) (+ s (* (aget w i) (aget w i))))))]
      (is (approx= 0.375 energy 2e-3)
          (str "Hann Parseval energy should be ~0.375, got " energy)))))

;; ================================================================
;; Window symmetry
;; ================================================================

(deftest window-symmetry-test
  (testing "Hann window symmetry: w[i] = w[N-1-i]"
    (let [n 64
          w (hann n)]
      (dotimes [i (/ n 2)]
        (is (approx= (aget w i) (aget w (- n 1 i)))
            (str "Hann asymmetry at i=" i)))))
  (testing "Hamming window symmetry"
    (let [n 64
          w (hamming n)]
      (dotimes [i (/ n 2)]
        (is (approx= (aget w i) (aget w (- n 1 i)))
            (str "Hamming asymmetry at i=" i)))))
  (testing "Blackman window symmetry"
    (let [n 64
          w (blackman n)]
      (dotimes [i (/ n 2)]
        (is (approx= (aget w i) (aget w (- n 1 i)))
            (str "Blackman asymmetry at i=" i)))))
  (testing "Kaiser window symmetry"
    (let [n 64
          w (kaiser n 8.6)]
      (dotimes [i (/ n 2)]
        (is (approx= (aget w i) (aget w (- n 1 i)))
            (str "Kaiser asymmetry at i=" i))))))

;; ================================================================
;; Convolution
;; ================================================================

(deftest convolve-impulse-test
  (testing "convolve [1 0 0] with [1 1] = [1 1 0 0]"
    (let [x (double-array [1.0 0.0 0.0])
          h (double-array [1.0 1.0])
          y (convolve x h 3 2)]
      ;; Output length = 3 + 2 - 1 = 4
      (is (= 4 (alength y)))
      (is (approx= 1.0 (aget y 0)))
      (is (approx= 1.0 (aget y 1)))
      (is (approx= 0.0 (aget y 2) 1e-10))
      (is (approx= 0.0 (aget y 3) 1e-10)))))

(deftest convolve-known-test
  (testing "convolve [1 2 3] with [1 1] = [1 3 5 3]"
    (let [y (convolve (double-array [1.0 2.0 3.0]) (double-array [1.0 1.0]) 3 2)]
      (is (approx= 1.0 (aget y 0)))
      (is (approx= 3.0 (aget y 1)))
      (is (approx= 5.0 (aget y 2)))
      (is (approx= 3.0 (aget y 3))))))

;; ================================================================
;; lfilter: pass-through
;; ================================================================

(deftest lfilter-passthrough-test
  (testing "lfilter with b=[1], a=[1] is identity"
    (let [x (double-array [1.0 2.0 3.0 4.0 5.0])
          b (double-array [1.0])
          a (double-array [1.0])
          y (lfilter b a x 1 1 5)]
      (dotimes [i 5]
        (is (approx= (aget x i) (aget y i))
            (str "Pass-through mismatch at index " i))))))

(deftest lfilter-first-order-test
  (testing "lfilter with first-order IIR: y[n] = x[n] + 0.5*y[n-1]"
    ;; b = [1], a = [1, -0.5]
    ;; x = [1 0 0 0], y = [1 0.5 0.25 0.125]
    (let [x (double-array [1.0 0.0 0.0 0.0])
          b (double-array [1.0])
          a (double-array [1.0 -0.5])
          y (lfilter b a x 1 2 4)]
      (is (approx= 1.0 (aget y 0)))
      (is (approx= 0.5 (aget y 1)))
      (is (approx= 0.25 (aget y 2)))
      (is (approx= 0.125 (aget y 3))))))

;; ================================================================
;; Correlate: autocorrelation peak at center
;; ================================================================

(deftest correlate-autocorrelation-test
  (testing "autocorrelation peak is at center"
    (let [x (double-array [1.0 2.0 3.0 2.0 1.0])
          n 5
          y (correlate x x n n)
          ;; Output length = 2*n - 1 = 9
          center (dec n) ;; index 4 is the zero-lag
          peak (aget y center)]
      ;; Zero-lag autocorrelation = sum(x_i^2) = 1+4+9+4+1 = 19
      (is (approx= 19.0 peak))
      ;; All other lags should be <= peak
      (dotimes [i (alength y)]
        (is (<= (aget y i) (+ peak 1e-10))
            (str "Autocorrelation at lag " i " exceeds peak"))))))

;; ================================================================
;; firwin: FIR filter design
;; ================================================================

(deftest firwin-unit-dc-test
  (testing "firwin coefficients sum to 1 (unit DC gain)"
    (let [h (firwin 51 1000.0 8000.0)
          sum (loop [i 0 s 0.0]
                (if (>= i (alength h)) s
                    (recur (inc i) (+ s (aget h i)))))]
      (is (approx= 1.0 sum 1e-4)
          (str "DC gain should be 1.0, got " sum)))))

;; ================================================================
;; sosfilt: second-order sections
;; ================================================================

(deftest sosfilt-passthrough-test
  (testing "sosfilt with identity section (b=[1 0 0], a=[1 0 0])"
    (let [;; One section: [b0 b1 b2 a0 a1 a2] = [1 0 0 1 0 0]
          sos (double-array [1.0 0.0 0.0 1.0 0.0 0.0])
          x (double-array [1.0 2.0 3.0 4.0])
          y (sosfilt sos x 1 4)]
      (dotimes [i 4]
        (is (approx= (aget x i) (aget y i))
            (str "Pass-through mismatch at index " i))))))
