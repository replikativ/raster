(ns raster.sci.signal
  "Signal processing functions: window functions, spectral analysis,
  correlation, convolution, and digital filters.

  Usage:
    (require '[raster.sci.signal :refer [hann hamming blackman kaiser
                                     welch correlate convolve
                                     lfilter firwin sosfilt]])
    (hann 256)                          ;=> double[256] Hann window
    (welch x nx nperseg noverlap win)   ;=> Object[freqs, psd]
    (correlate x y nx ny)               ;=> cross-correlation array
    (lfilter b a x nb na nx)            ;=> IIR-filtered signal"
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.core :refer [deftm ftm]]
            [raster.arrays :refer [aget aset alength aclone acopy!]]
            [raster.par :as par]
            [raster.math :as m]
            [raster.numeric :as n]
            [raster.sci.fft :as fft]))

;; ================================================================
;; Window functions
;; ================================================================

(deftm hann [n :- Long] :- (Array double)
  (let [n-1 (double (dec n))
        two-pi (n/* 2.0 n/pi)]
    (par/map [i n]
             (n/* 0.5 (n/- 1.0 (m/cos (n// (n/* two-pi (double i)) n-1)))))))

(deftm hamming [n :- Long] :- (Array double)
  (let [n-1 (double (dec n))
        two-pi (n/* 2.0 n/pi)]
    (par/map [i n]
             (n/- 0.54 (n/* 0.46 (m/cos (n// (n/* two-pi (double i)) n-1)))))))

(deftm blackman [n :- Long] :- (Array double)
  (let [n-1 (double (dec n))
        two-pi (n/* 2.0 n/pi)
        four-pi (n/* 4.0 n/pi)]
    (par/map [i n]
             (let [x (double i)]
               (n/+ (n/- 0.42 (n/* 0.5 (m/cos (n// (n/* two-pi x) n-1))))
                    (n/* 0.08 (m/cos (n// (n/* four-pi x) n-1))))))))

(deftm besseli0-kaiser
  "Modified Bessel function I0(x) via series expansion for Kaiser window."
  [x :- Double] :- Double
  (let [half-x (n/* 0.5 x)]
    (loop [k 1 term 1.0 sum 1.0]
      (if (or (> k 25) (< (n/abs term) (n/* 1e-16 (n/abs sum))))
        sum
        (let [t (n// half-x (double k))
              new-term (n/* (n/* term t) t)]
          (recur (inc k) new-term (n/+ sum new-term)))))))

(deftm kaiser [n :- Long beta :- Double] :- (Array double)
  (let [n-1 (double (dec n))
        inv-i0-beta (n// 1.0 (besseli0-kaiser beta))]
    (par/map [i n]
             (let [ratio (n/- (n/* 2.0 (n// (double i) n-1)) 1.0)
                   arg (n/* beta (n/sqrt (n/max 0.0 (n/- 1.0 (n/* ratio ratio)))))]
               (n/* (besseli0-kaiser arg) inv-i0-beta)))))

;; ================================================================
;; Spectral analysis — Welch method
;; ================================================================

(deftm welch
  "Welch's method for power spectral density estimation.
  x       — input signal (length nx)
  nx      — length of x
  nperseg — samples per segment (must be power of 2)
  noverlap— overlap between segments
  window  — window function (length nperseg)
  Returns (object-array [freqs psd]) where freqs has nperseg/2+1 elements."
  [x :- (Array double) nx :- Long nperseg :- Long
   noverlap :- Long window :- (Array double)] :- (Array Object)
  (let [step (- nperseg noverlap)
        n-freqs (inc (bit-shift-right nperseg 1))
        ;; Compute window power for normalization
        win-power (loop [i 0 s 0.0]
                    (if (>= i nperseg)
                      s
                      (let [wi (aget window i)]
                        (recur (inc i) (n/+ s (n/* wi wi))))))
        psd (double-array n-freqs)
        seg (double-array nperseg)]
    ;; Average periodograms over segments
    (let [n-segs (loop [start 0 cnt 0]
                   (if (> (+ start nperseg) nx)
                     cnt
                     (do
                       ;; Extract and window segment
                       (dotimes [j nperseg]
                         (aset seg j (n/* (aget x (+ start j)) (aget window j))))
                       ;; FFT
                       (let [[re im] (fft/fft seg)]
                         ;; Accumulate one-sided periodogram
                         (dotimes [k n-freqs]
                           (let [rk (aget re k)
                                 ik (aget im k)
                                 power (n/+ (n/* rk rk) (n/* ik ik))]
                             (aset psd k (n/+ (aget psd k) power)))))
                       (recur (+ start step) (inc cnt)))))]
      ;; Normalize: PSD = sum / (n-segs * win-power)
      (let [norm (n// 1.0 (n/* (double n-segs) win-power))]
        (dotimes [k n-freqs]
          (aset psd k (n/* (aget psd k) norm)))
        ;; Double interior bins (one-sided spectrum)
        (loop [k 1]
          (when (< k (dec n-freqs))
            (aset psd k (n/* (aget psd k) 2.0))
            (recur (inc k)))))
      ;; Frequency axis: k / nperseg for k = 0..n-freqs-1 (normalized)
      (let [freqs (double-array n-freqs)
            inv-n (n// 1.0 (double nperseg))]
        (dotimes [k n-freqs]
          (aset freqs k (n/* (double k) inv-n)))
        (object-array [freqs psd])))))

;; ================================================================
;; Cross-correlation via FFT
;; ================================================================

(deftm correlate
  "Cross-correlation of x and y via FFT.
  Returns array of length nx+ny-1.
  correlate[k] = sum_i x[i] * y[i - k + (ny-1)]"
  [x :- (Array double) y :- (Array double) nx :- Long ny :- Long] :- (Array double)
  (let [out-len (+ nx ny -1)
        n (fft/next-power-of-2 (long out-len))
        ;; Zero-pad x into length n
        xp (double-array n)
        _ (acopy! x 0 xp 0 nx)
        ;; Zero-pad y (reversed) into length n for correlation
        ;; correlation(x,y) = ifft(fft(x) * conj(fft(y)))
        yp (double-array n)
        _ (acopy! y 0 yp 0 ny)
        ;; FFT both
        [xr xi] (fft/fft xp)
        [yr yi] (fft/fft yp)
        ;; Multiply X * conj(Y): (a+bi)(c-di) = (ac+bd) + (bc-ad)i
        prod-re (double-array n)
        prod-im (double-array n)]
    (dotimes [i n]
      (let [a (aget xr i)
            b (aget xi i)
            c (aget yr i)
            d (aget yi i)]
        (aset prod-re i (n/+ (n/* a c) (n/* b d)))
        (aset prod-im i (n/- (n/* b c) (n/* a d)))))
    ;; IFFT
    (let [[res-re _] (fft/ifft prod-re prod-im)
          result (double-array out-len)]
      ;; Rearrange: correlation output needs circular shift
      ;; The IFFT gives circular correlation; we need the linear part.
      ;; For correlation, result[k] for k=0..out-len-1 maps to
      ;; ifft index (k - (ny-1)) mod n
      (dotimes [k out-len]
        (let [idx (mod (+ (- k (dec ny)) n) n)]
          (aset result k (aget res-re idx))))
      result)))

;; ================================================================
;; Convolution via FFT
;; ================================================================

(deftm convolve
  "Convolution of x and y via FFT.
  Returns array of length nx+ny-1."
  [x :- (Array double) y :- (Array double) nx :- Long ny :- Long] :- (Array double)
  (let [out-len (+ nx ny -1)
        n (fft/next-power-of-2 (long out-len))
        ;; Zero-pad both to length n
        xp (double-array n)
        _ (acopy! x 0 xp 0 nx)
        yp (double-array n)
        _ (acopy! y 0 yp 0 ny)
        ;; FFT both
        [xr xi] (fft/fft xp)
        [yr yi] (fft/fft yp)
        ;; Pointwise multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        prod-re (double-array n)
        prod-im (double-array n)]
    (dotimes [i n]
      (let [a (aget xr i)
            b (aget xi i)
            c (aget yr i)
            d (aget yi i)]
        (aset prod-re i (n/- (n/* a c) (n/* b d)))
        (aset prod-im i (n/+ (n/* a d) (n/* b c)))))
    ;; IFFT and take first out-len samples
    (let [[res-re _] (fft/ifft prod-re prod-im)
          result (double-array out-len)]
      (acopy! res-re 0 result 0 out-len)
      result)))

;; ================================================================
;; IIR filter — Direct Form II Transposed
;; ================================================================

(deftm lfilter (All [T]
                    [b :- (Array T) a :- (Array T) x :- (Array T)
                     nb :- Long na :- Long nx :- Long] :- (Array T)
                    (let [;; Normalize by a[0]
                          a0 (aget a 0)
                          nz (max nb na)
        ;; Normalized coefficients — use double for precision
                          bn (double-array nz)
                          an (double-array nz)
                          _ (dotimes [i nb]
                              (aset bn i (n// (aget b i) a0)))
                          _ (dotimes [i na]
                              (aset an i (n// (aget a i) a0)))
        ;; State (delay line), length nz-1 (or 1 if nz<=1)
                          nstate (max 1 (dec nz))
                          z (double-array nstate)
        ;; Output — same type as input
                          y (raster.numeric/similar x)]
                      (dotimes [n nx]
                        (let [xn (aget x n)
            ;; Output: y[n] = b[0]*x[n] + z[0]
                              yn (n/+ (n/* (aget bn 0) xn) (aget z 0))]
                          (aset y n yn)
        ;; Update delay line: z[i] = b[i+1]*x[n] - a[i+1]*y[n] + z[i+1]
                          (loop [i 0]
                            (when (< i (dec nstate))
                              (let [bi1 (aget bn (inc i))
                                    ai1 (aget an (inc i))]
                                (aset z i (n/+ (n/- (n/* bi1 xn) (n/* ai1 yn)) (aget z (inc i)))))
                              (recur (inc i))))
        ;; Last state element (no z[i+1] term)
                          (when (> nstate 0)
                            (let [last-i (dec nstate)
                                  bi1 (if (< (inc last-i) nz) (aget bn (inc last-i)) 0.0)
                                  ai1 (if (< (inc last-i) nz) (aget an (inc last-i)) 0.0)]
                              (aset z last-i (n/- (n/* bi1 xn) (n/* ai1 yn)))))))
                      y)))

;; ================================================================
;; FIR filter design — windowed sinc
;; ================================================================

(deftm firwin
  "FIR lowpass filter design using windowed sinc method.
  numtaps — number of filter taps (should be odd for type I)
  cutoff  — cutoff frequency in Hz
  fs      — sampling frequency in Hz
  Returns filter coefficients (length numtaps) with Hamming window applied."
  [numtaps :- Long cutoff :- Double fs :- Double] :- (Array double)
  (let [h (double-array numtaps)
        fc (n// cutoff (n/* 0.5 fs))  ;; normalized cutoff [0, 1]
        m (double (dec numtaps))
        half-m (n/* 0.5 m)
        win (hamming numtaps)]
    ;; Ideal sinc lowpass
    (dotimes [i numtaps]
      (let [n-shifted (n/- (double i) half-m)]
        (if (< (n/abs n-shifted) 1e-15)
          (aset h i fc)
          (aset h i (n// (m/sin (n/* (n/* n/pi fc) n-shifted))
                         (n/* n/pi n-shifted))))))
    ;; Apply window
    (dotimes [i numtaps]
      (aset h i (n/* (aget h i) (aget win i))))
    ;; Normalize to unit DC gain
    (let [sum (loop [i 0 s 0.0]
                (if (>= i numtaps)
                  s
                  (recur (inc i) (n/+ s (aget h i)))))
          inv-sum (n// 1.0 sum)]
      (dotimes [i numtaps]
        (aset h i (n/* (aget h i) inv-sum))))
    h))

;; ================================================================
;; Second-order sections (biquad cascade) filter
;; ================================================================

(deftm sosfilt
  "Second-order sections (biquad cascade) filter.
  sos        — flat array of [n-sections * 6] coefficients,
               each section: [b0, b1, b2, a0, a1, a2]
  x          — input signal (length nx)
  n-sections — number of biquad sections
  nx         — length of input signal
  Returns filtered signal of length nx."
  (All [T] [sos :- (Array T) x :- (Array T)
            n-sections :- Long nx :- Long] :- (Array T)
       (let [;; Work with two buffers to avoid extra allocation
             buf-in (raster.numeric/similar x)
             buf-out (raster.numeric/similar x)
             _ (acopy! x 0 buf-in 0 nx)]
         (loop [s 0]
           (when (< s n-sections)
             (let [base (* s 6)
                   b0 (aget sos base)
                   b1 (aget sos (+ base 1))
                   b2 (aget sos (+ base 2))
                   a0 (aget sos (+ base 3))
                   a1 (aget sos (+ base 4))
                   a2 (aget sos (+ base 5))
              ;; Normalize by a0
                   inv-a0 (n// 1.0 a0)
                   b0n (n/* b0 inv-a0)
                   b1n (n/* b1 inv-a0)
                   b2n (n/* b2 inv-a0)
                   a1n (n/* a1 inv-a0)
                   a2n (n/* a2 inv-a0)]
          ;; Direct Form II Transposed for this section
               (loop [n 0 z1 0.0 z2 0.0]
                 (if (>= n nx)
                   nil
                   (let [xn (aget buf-in n)
                         yn (n/+ (n/* b0n xn) z1)
                         new-z1 (n/+ (n/+ (n/* b1n xn) (n/- (n/* a1n yn))) z2)
                         new-z2 (n/- (n/* b2n xn) (n/* a2n yn))]
                     (aset buf-out n yn)
                     (recur (inc n) new-z1 new-z2))))
          ;; Copy output to input for next section
               (when (< (inc s) n-sections)
                 (acopy! buf-out 0 buf-in 0 nx)))
             (recur (inc s))))
         buf-out)))
