(ns raster.compiler.fixtures.lora-training
  "Plain low-rank training witness, following finetune-rstr's numerical program.
   The oracle uses coordinate loops, not Raster linear algebra, loss or AD."
  (:require [raster.core :refer [deftm]]
            [raster.dl.nn :as nn]
            [raster.dl.loss :as loss]
            [raster.dl.optim :as optim]
            [raster.arrays :as arrays]
            [raster.ad.reverse :as reverse]))

(deftm delta
  (All [T] [x :- (Array T) A :- (Array T) B :- (Array T)
            rows :- Long in :- Long r :- Long out :- Long] :- (Array T)
       (let [ax (nn/linear-nb x A rows in r)]
         (nn/linear-nb ax B rows r out))))

(deftm objective
  [x :- (Array float) A :- (Array float) B :- (Array float) tgt :- (Array float)
   rows :- Long in :- Long r :- Long out :- Long] :- Double
  (loss/mse-loss (delta x A B rows in r out) tgt (clojure.core/* rows out)))

(deftm train-step
  [A :- (Array float) B :- (Array float) x :- (Array float) tgt :- (Array float)
   rows :- Long in :- Long r :- Long out :- Long lr :- Double] :- (Array float)
  (let [vg ((reverse/value+grad (var raster.compiler.fixtures.lora-training/objective))
            x A B tgt rows in r out)
        dA (clojure.core/nth vg 2)
        dB (clojure.core/nth vg 3)]
    (optim/sgd-step! A dA (arrays/alength A) (float lr))
    (optim/sgd-step! B dB (arrays/alength B) (float lr))
    A))

(defn inputs
  "Fresh arrays each time. Both adapters are nonzero, so both gradients are exercised."
  ([] (inputs 0))
  ([batch]
   {:A (float-array (map #(* 0.0625 (- (mod % 7) 3)) (range 32)))
    :B (float-array (map #(* 0.0625 (- (mod % 5) 2)) (range 32)))
    :x (float-array (map #(* 0.125 (- (mod (+ % batch) 7) 3)) (range 16)))
    :tgt (float-array (map #(* 0.125 (- (mod (+ % (* 2 batch)) 5) 2)) (range 16)))
    :rows 2 :in 8 :r 4 :out 8 :lr 0.01}))

(defn arguments [{:keys [A B x tgt rows in r out lr]}]
  [A B x tgt rows in r out lr])

(defn oracle
  "Real-arithmetic MSE and analytic derivatives for y[b,o]=sum_k B[o,k]sum_j A[k,j]x[b,j].
   Double arithmetic is an independent reference, not the emitted FP32 rounding contract."
  [{:keys [A B x tgt rows in r out]}]
  (let [hidden (double-array (* rows r))
        residual (double-array (* rows out))
        da (double-array (* r in))
        db (double-array (* out r))]
    (dotimes [b rows]
      (dotimes [k r]
        (aset hidden (+ (* b r) k)
              (reduce + 0.0 (for [j (range in)]
                              (* (double (nth A (+ (* k in) j)))
                                 (double (nth x (+ (* b in) j))))))))
      (dotimes [o out]
        (aset residual (+ (* b out) o)
              (- (reduce + 0.0 (for [k (range r)]
                                 (* (double (nth B (+ (* o r) k)))
                                    (aget hidden (+ (* b r) k)))))
                 (double (nth tgt (+ (* b out) o)))))))
    (dotimes [b rows]
      (dotimes [o out]
        (let [d (* (/ 2.0 (* rows out)) (aget residual (+ (* b out) o)))]
          (dotimes [k r]
            (let [bi (+ (* o r) k)]
              (aset db bi (+ (aget db bi) (* d (aget hidden (+ (* b r) k)))))
              (dotimes [j in]
                (let [ai (+ (* k in) j)]
                  (aset da ai (+ (aget da ai)
                                 (* d (double (nth B bi))
                                    (double (nth x (+ (* b in) j)))))))))))))
    {:loss (/ (reduce + 0.0 (map #(* % %) residual)) (* rows out)) :dA da :dB db}))

(defn update-oracle [{:keys [A B lr] :as input}]
  (let [{:keys [dA dB]} (oracle input)]
    (assoc input
           :A (float-array (map #(- (double %1) (* (double (float lr)) %2)) A dA))
           :B (float-array (map #(- (double %1) (* (double (float lr)) %2)) B dB)))))

(defn max-error [a b]
  (assert (= (count a) (count b)))
  (reduce max 0.0 (map #(Math/abs (- (double %1) (double %2))) a b)))
