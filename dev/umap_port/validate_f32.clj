;; Step 2b validation: parametric optimize-layout! — f64 regression gate +
;; f32 run vs umap's REAL numba f32 output (emb_umap).
(ns umap-validate-f32
  (:require [clojure.edn :as edn]
            [raster.umap :as u]))

(def gold (edn/read-string (slurp "/tmp/umap_gold/gold.edn")))

(defn read-npy-f64
  "Minimal .npy reader for little-endian float64/float32 C-order arrays -> double[]."
  ^doubles [path]
  (let [bs (java.nio.file.Files/readAllBytes (java.nio.file.Path/of path (make-array String 0)))
        bb (doto (java.nio.ByteBuffer/wrap bs) (.order java.nio.ByteOrder/LITTLE_ENDIAN))
        ;; header: \x93NUMPY ver(2) hlen(2) then hlen bytes of dict
        hlen (do (.position bb 8) (bit-and (int (.getShort bb)) 0xFFFF))
        hdr  (let [a (byte-array hlen)] (.get bb a) (String. a))
        f32? (boolean (re-find #"<f4" hdr))
        data-off (+ 10 hlen)
        elt (if f32? 4 8)
        n (quot (- (alength bs) data-off) elt)
        out (double-array n)]
    (.position bb data-off)
    (dotimes [i n]
      (aset out i (double (if f32? (.getFloat bb) (.getDouble bb)))))
    out))

(defn init-states ^longs [init-flat dim n rng-state]
  (let [st (long-array (* n 3)) [r0 r1 r2] rng-state]
    (dotimes [v n]
      (let [bits (Double/doubleToLongBits (double (nth init-flat (* v dim))))]
        (aset st (+ (* v 3) 0) (+ (long r0) bits))
        (aset st (+ (* v 3) 1) (+ (long r1) bits))
        (aset st (+ (* v 3) 2) (+ (long r2) bits))))
    st))

(defn max-abs-diff-d [^doubles a ref-vec]
  (areduce a i m 0.0 (max m (Math/abs (- (aget a i) (double (nth ref-vec i)))))))

(defn max-abs-diff-f [^floats a ^doubles ref]
  (areduce a i m 0.0 (max m (Math/abs (- (double (aget a i)) (aget ref i))))))

;; ---- f64 regression: attractive-only gate (must still be ~1e-12) ----
(defn run-f64 [{:keys [n dim neg-rate gamma init-alpha a b head tail eps init rng-state]}
               epochs attractive-only?]
  (let [emb (double-array init) hd (int-array head) tl (int-array tail)
        epsA (double-array eps)
        epnA (if attractive-only? (double-array (count eps) 1.0e18)
                 (double-array (map #(/ (double %) (double neg-rate)) eps)))
        eonsA (double-array eps) eonsnA (double-array (alength epnA))
        _ (System/arraycopy epnA 0 eonsnA 0 (alength epnA))
        states (init-states init dim n rng-state)]
    (u/optimize-layout! emb hd tl epsA epnA eonsA eonsnA states
                        (double a) (double b) (double gamma) (double init-alpha)
                        (long n) (long dim) (long epochs))
    emb))

;; ---- f32: mirror umap exactly (float[] embedding, f64 scalars) ----
(defn run-f32 [{:keys [n dim neg-rate gamma init-alpha a b head tail eps init rng-state]}
               epochs attractive-only?]
  (let [emb (float-array (map float init)) hd (int-array head) tl (int-array tail)
        epsA (double-array eps)
        epnA (if attractive-only? (double-array (count eps) 1.0e18)
                 (double-array (map #(/ (double %) (double neg-rate)) eps)))
        eonsA (double-array eps) eonsnA (double-array (alength epnA))
        _ (System/arraycopy epnA 0 eonsnA 0 (alength epnA))
        states (init-states init dim n rng-state)]
    (u/optimize-layout! emb hd tl epsA epnA eonsA eonsnA states
                        (double a) (double b) (double gamma) (double init-alpha)
                        (long n) (long dim) (long epochs))
    emb))

(def ref-attr (vec (:ref-attr gold)))
(def ref-attr5 (vec (:ref-attr-5 gold)))

;; (1) f64 regression: our f64 kernel still matches the f64 reference to ~1e-7.
(println "=== f64 regression (attractive-only) ===")
(println "attr   5:" (max-abs-diff-d (run-f64 gold 5 true) ref-attr5))
(println "attr 200:" (max-abs-diff-d (run-f64 gold 200 true) ref-attr))

;; (2) f32 attractive-only: f32 rounding of the deterministic ref — must be ~1e-5.
(println "=== f32 vs ref_attr (attractive-only, f32 rounding) ===")
(println "attr   5:" (max-abs-diff-f (run-f32 gold 5 true) (double-array ref-attr5)))
(println "attr 200:" (max-abs-diff-f (run-f32 gold 200 true) (double-array ref-attr)))

;; (3) f32 vs REAL umap (numba f32) at short horizon, WITH negative sampling.
;; 1 epoch is bit-exact (proves RNG seeding + neg-sample gradient match numba
;; exactly). Divergence at later epochs is numba fastmath=True reassociation
;; amplified by chaotic SGD — coordinate matching of the full run is not a gate.
;; Run gen_short.py first to produce emb_umap_{1,3,10}.npy.
(println "=== f32 vs umap (real numba f32, with neg sampling) ===")
(doseq [ep [1 3 10]]
  (let [g (read-npy-f64 (str "/tmp/umap_gold/emb_umap_" ep ".npy"))
        e (run-f32 (assoc gold :n-epochs ep) ep false)]
    (println (format "ep %2d  max|raster_f32 - umap_f32|: %.6g  %s"
                     ep (max-abs-diff-f e g)
                     (cond (== ep 1) "(expect 0.0 — bit-exact)"
                           (== ep 3) "(expect ~1e-4 — fastmath ULP)"
                           :else      "(expect O(1) — chaos)")))))
