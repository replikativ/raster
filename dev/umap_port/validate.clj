;; Validation harness for the raster UMAP port. load-file this in a raster REPL
;; after running dev/umap_port/gen.py (writes /tmp/umap_gold/*.edn).
(ns umap-validate
  (:require [clojure.edn :as edn]
            [raster.umap :as u]
            [raster.umap.knn :as knn]))

(defn max-abs-diff [^doubles a ref-vec]
  (let [n (alength a)]
    (loop [i 0 m 0.0]
      (if (< i n)
        (recur (inc i) (max m (Math/abs (- (aget a i) (double (nth ref-vec i)))))) m))))

;; ---------- kNN ----------
(def kg (edn/read-string (slurp "/tmp/umap_gold/knn_gold.edn")))
(let [n (long (:n kg)) dim (long (:dim-in kg)) k (long (:k kg))
      X (double-array (:X kg))
      oi (int-array (* n k)) od (double-array (* n k))
      gold-idx (vec (:idx kg)) gold-dst (vec (:dst kg))]
  (knn/knn-brute! X n dim k oi od)
  ;; distances must match to ~1e-6; neighbor SETS must match (tie order may differ)
  (let [dmax (loop [i 0 m 0.0] (if (< i (* n k))
                                 (recur (inc i) (max m (Math/abs (- (aget od i) (double (nth gold-dst i)))))) m))
        set-ok (count (filter (fn [i]
                                (= (set (for [t (range k)] (aget oi (+ (* i k) t))))
                                   (set (for [t (range k)] (nth gold-idx (+ (* i k) t))))))
                              (range n)))]
    (println "=== kNN brute-force vs sklearn exact ===")
    (println "max|dist diff| :" dmax)
    (println "rows with identical neighbor set :" set-ok "/" n)))

;; ---------- layout ----------
(def gold (edn/read-string (slurp "/tmp/umap_gold/gold.edn")))
(defn init-states ^longs [init-flat dim n rng-state]
  (let [st (long-array (* n 3)) [r0 r1 r2] rng-state]
    (dotimes [v n]
      (let [bits (Double/doubleToLongBits (double (nth init-flat (* v dim))))]
        (aset st (+ (* v 3) 0) (+ (long r0) bits))
        (aset st (+ (* v 3) 1) (+ (long r1) bits))
        (aset st (+ (* v 3) 2) (+ (long r2) bits))))
    st))
(defn run [{:keys [n dim neg-rate gamma init-alpha a b head tail eps init rng-state]} epochs attractive-only?]
  (let [emb (double-array init) hd (int-array head) tl (int-array tail)
        epsA (double-array eps)
        epnA (if attractive-only? (double-array (count eps) 1.0e18)
                 (double-array (map #(/ (double %) (double neg-rate)) eps)))
        eonsA (double-array eps)
        eonsnA (double-array (alength epnA))
        _ (System/arraycopy epnA 0 eonsnA 0 (alength epnA))
        states (init-states init dim n rng-state)]
    (u/optimize-layout! emb hd tl epsA epnA eonsA eonsnA states
                        (double a) (double b) (double gamma) (double init-alpha)
                        (long n) (long dim) (long epochs))
    emb))
(println "=== layout: deterministic attractive-only gate ===")
(println "max|raster-ref| attr  5 epochs:" (max-abs-diff (run gold 5 true)   (:ref-attr-5 gold)))
(println "max|raster-ref| attr 200 epochs:" (max-abs-diff (run gold 200 true) (:ref-attr gold)))
(spit "/tmp/umap_gold/raster_full.edn" (pr-str (vec (run gold (:n-epochs gold) false))))
(println "wrote raster_full.edn")
