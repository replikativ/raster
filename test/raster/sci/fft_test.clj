(ns raster.sci.fft-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.sci.fft :refer [fft ifft rfft fftfreq magnitude fft-complex
                                    dct idct dst idst]]))

(def ^:const EPS 1e-6)

(defn- approx=
  ([^double a ^double b] (< (Math/abs (- a b)) EPS))
  ([^double a ^double b ^double tol] (< (Math/abs (- a b)) tol)))

(defn- array-approx=
  "Check if all elements of two arrays are approximately equal."
  [^doubles a ^doubles b ^double tol]
  (and (= (alength a) (alength b))
       (every? true? (map #(< (Math/abs (- %1 %2)) tol) a b))))

;; ================================================================
;; Basic FFT tests
;; ================================================================

(deftest fft-dc-test
  (testing "FFT of constant signal"
    (let [[^doubles re ^doubles im] (fft (double-array [1.0 1.0 1.0 1.0]))]
      ;; DC component = sum = 4
      (is (approx= 4.0 (aget re 0)))
      ;; All other components zero
      (is (approx= 0.0 (aget re 1) 1e-10))
      (is (approx= 0.0 (aget re 2) 1e-10))
      (is (approx= 0.0 (aget re 3) 1e-10))
      ;; Imaginary all zero
      (is (approx= 0.0 (aget im 0) 1e-10)))))

(deftest fft-impulse-test
  (testing "FFT of impulse [1 0 0 0]"
    (let [[^doubles re ^doubles im] (fft (double-array [1.0 0.0 0.0 0.0]))]
      ;; All frequency components equal 1
      (dotimes [i 4]
        (is (approx= 1.0 (aget re i) 1e-10))
        (is (approx= 0.0 (aget im i) 1e-10))))))

(deftest fft-sinusoid-test
  (testing "FFT detects pure sinusoid at correct frequency"
    (let [n 64
          ;; Pure sinusoid at frequency k=4: cos(2*pi*4*t/N)
          x (double-array (map #(Math/cos (* 2.0 Math/PI 4.0 (/ (double %) n)))
                               (range n)))
          [^doubles re ^doubles im] (fft x)
          ^doubles mag (magnitude re im)]
      ;; Peak at bin 4 (and mirror at bin 60=N-4)
      (is (> (aget mag 4) 30.0)
          (str "Peak at bin 4 should be large, got " (aget mag 4)))
      (is (> (aget mag (- n 4)) 30.0)
          (str "Mirror peak at bin " (- n 4)))
      ;; Other bins should be near zero
      (is (< (aget mag 1) 1e-10)
          (str "Non-peak bin should be ~0, got " (aget mag 1))))))

;; ================================================================
;; Inverse FFT
;; ================================================================

(deftest ifft-roundtrip-test
  (testing "FFT -> IFFT roundtrip"
    (let [x (double-array [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0])
          [re im] (fft x)
          [^doubles re2 ^doubles im2] (ifft re im)]
      (dotimes [i (alength x)]
        (is (approx= (aget x i) (aget re2 i) 1e-10)
            (str "Roundtrip mismatch at index " i))
        (is (approx= 0.0 (aget im2 i) 1e-10)
            (str "Imaginary part should be 0 at index " i))))))

;; ================================================================
;; Parseval's theorem
;; ================================================================

(deftest parseval-test
  (testing "Parseval's theorem: sum(|x|^2) = (1/N) * sum(|X|^2)"
    (let [x (double-array [1.0 -2.0 3.0 -4.0 5.0 -6.0 7.0 -8.0])
          n (alength x)
          time-energy (loop [i 0 s 0.0]
                        (if (>= i n) s
                            (recur (inc i) (+ s (* (aget x i) (aget x i))))))
          [^doubles re ^doubles im] (fft x)
          freq-energy (loop [i 0 s 0.0]
                        (if (>= i n) s
                            (recur (inc i) (+ s (* (aget re i) (aget re i))
                                              (* (aget im i) (aget im i))))))]
      (is (approx= time-energy (/ freq-energy (double n)) 1e-8)
          (str "Parseval: time=" time-energy " freq/N=" (/ freq-energy (double n)))))))

;; ================================================================
;; Real FFT
;; ================================================================

(deftest rfft-test
  (testing "rfft returns N/2+1 coefficients"
    (let [[^doubles re ^doubles im] (rfft (double-array [1.0 2.0 3.0 4.0]))]
      (is (= 3 (alength re)))  ;; N/2+1 = 3
      (is (= 3 (alength im)))
      ;; DC component
      (is (approx= 10.0 (aget re 0)))  ;; sum = 10
      ;; Nyquist
      (is (approx= -2.0 (aget re 2))))))

;; ================================================================
;; Frequency bins
;; ================================================================

(deftest fftfreq-test
  (testing "fftfreq produces correct frequencies"
    (let [freqs (fftfreq 8 1.0)]
      ;; [0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]
      (is (approx= 0.0 (aget freqs 0)))
      (is (approx= 0.125 (aget freqs 1)))
      (is (approx= 0.25 (aget freqs 2)))
      (is (approx= -0.5 (aget freqs 4)))
      (is (approx= -0.125 (aget freqs 7))))))

;; ================================================================
;; Zero padding
;; ================================================================

(deftest fft-zero-pad-test
  (testing "FFT handles non-power-of-2 input via zero padding"
    (let [x (double-array [1.0 2.0 3.0])  ;; length 3, padded to 4
          [^doubles re ^doubles im] (fft x)]
      ;; Should have length 4
      (is (= 4 (alength re)))
      ;; DC = 1+2+3+0 = 6
      (is (approx= 6.0 (aget re 0))))))

;; ================================================================
;; DCT (Discrete Cosine Transform)
;; ================================================================

(deftest dct-constant-signal-test
  (testing "DCT of constant signal: only DC component nonzero"
    (let [n 8
          x (double-array (repeat n 1.0))
          X (dct x n)]
      ;; DC (k=0): 2 * N * 1.0 = 2*8 = 16
      (is (> (Math/abs (aget X 0)) 1.0)
          "DC component should be nonzero")
      ;; All other components should be near zero
      (doseq [k (range 1 n)]
        (is (approx= 0.0 (aget X k) 1e-10)
            (str "Non-DC component k=" k " should be ~0, got " (aget X k)))))))

(deftest dct-idct-roundtrip-test
  (testing "IDCT(DCT(x)) ≈ x"
    (let [n 8
          x (double-array [1.0 -2.0 3.0 -4.0 5.0 -6.0 7.0 -8.0])
          X (dct x n)
          x2 (idct X n)]
      (dotimes [i n]
        (is (approx= (aget x i) (aget x2 i) 1e-10)
            (str "DCT roundtrip mismatch at index " i))))))

;; ================================================================
;; DST (Discrete Sine Transform)
;; ================================================================

(deftest dst-sinusoidal-test
  (testing "DST of pure sine mode"
    (let [n 7
          ;; x[i] = sin(pi*(i+1)*2/(n+1)) — second mode (k=1)
          x (double-array (map #(Math/sin (* Math/PI (inc %) 2.0 (/ 1.0 (inc n))))
                               (range n)))
          X (dst x n)]
      ;; The k=1 component (index 1) should be dominant
      (is (> (Math/abs (aget X 1)) 1.0)
          (str "Second mode should be large, got " (aget X 1))))))

(deftest dst-idst-roundtrip-test
  (testing "IDST(DST(x)) ≈ x"
    (let [n 7
          x (double-array [1.0 2.0 3.0 4.0 5.0 6.0 7.0])
          X (dst x n)
          x2 (idst X n)]
      (dotimes [i n]
        (is (approx= (aget x i) (aget x2 i) 1e-10)
            (str "DST roundtrip mismatch at index " i))))))
