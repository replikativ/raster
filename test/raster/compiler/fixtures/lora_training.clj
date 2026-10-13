(ns raster.compiler.fixtures.lora-training
  "Plain low-rank training witness, following finetune-rstr's numerical program.
   The oracle uses coordinate loops, not Raster linear algebra, loss or AD."
  (:require [raster.core :refer [deftm]]
            [raster.dl.nn :as nn]
            [raster.dl.loss :as loss]
            [raster.dl.optim :as optim]
            [raster.arrays :as arrays]
            [raster.numeric :as n]
            [raster.par :as par]
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

(deftm joint-clipped-lora-adam-step
  "One deliberately bounded public acceptance step: reverse AD, one joint clipping norm, two
   Adam updates at the caller-supplied time, and the pre-update scalar loss. Multi-step time
   transport is intentionally not implied by this fixture."
  [A :- (Array float) B :- (Array float)
   mA :- (Array float) vA :- (Array float)
   mB :- (Array float) vB :- (Array float)
   x :- (Array float) tgt :- (Array float)
   rows :- Long in :- Long r :- Long out :- Long
   max-grad-norm :- Double lr :- Double
   beta1 :- Double beta2 :- Double eps :- Double adam-t :- Long] :- Double
  (let [vg ((reverse/value+grad (var raster.compiler.fixtures.lora-training/objective))
            x A B tgt rows in r out)
        dA (clojure.core/nth vg 2)
        dB (clojure.core/nth vg 3)
        na (arrays/alength A)
        nb (arrays/alength B)
        norm-a (par/reduce sum (float 0.0) i na
                           (let [g (arrays/aget dA i)] (n/+ sum (n/* g g))))
        norm-b (par/reduce sum (float 0.0) i nb
                           (let [g (arrays/aget dB i)] (n/+ sum (n/* g g))))
        norm (n/sqrt (n/+ (double norm-a) (double norm-b)))
        scale (float (if (> norm max-grad-norm) (n// max-grad-norm norm) 1.0))]
    (par/map-void! i na (arrays/aset dA i (n/* (arrays/aget dA i) scale)))
    (par/map-void! i nb (arrays/aset dB i (n/* (arrays/aget dB i) scale)))
    (optim/adam-step! A dA mA vA na (float lr) (float beta1) (float beta2)
                      (float eps) adam-t)
    (optim/adam-step! B dB mB vB nb (float lr) (float beta1) (float beta2)
                      (float eps) adam-t)
    (double (clojure.core/nth vg 0))))

(deftm joint-clipped-lora-adam-buffered-step
  "The same training step, with the ordinary resident clock input changed between calls."
  [A :- (Array float) B :- (Array float)
   mA :- (Array float) vA :- (Array float)
   mB :- (Array float) vB :- (Array float)
   x :- (Array float) tgt :- (Array float)
   rows :- Long in :- Long r :- Long out :- Long
   max-grad-norm :- Double lr :- Double
   beta1 :- Double beta2 :- Double eps :- Double time-buffer :- (Array long)] :- Double
  (joint-clipped-lora-adam-step
   A B mA vA mB vB x tgt rows in r out max-grad-norm lr beta1 beta2 eps
   (arrays/aget time-buffer 0)))

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

(defn adam-inputs
  "Fresh one-step Adam state. Time is fixed at one; callers must not treat replay as a
   multi-step Adam qualification."
  []
  (let [input (inputs)]
    (assoc input
           :mA (float-array (count (:A input))) :vA (float-array (count (:A input)))
           :mB (float-array (count (:B input))) :vB (float-array (count (:B input)))
           :max-grad-norm 1.0e-4 :beta1 0.9 :beta2 0.999 :eps 1.0e-8
           :adam-t (long 1))))

(defn adam-arguments
  [{:keys [A B mA vA mB vB x tgt rows in r out max-grad-norm lr
           beta1 beta2 eps adam-t]}]
  [A B mA vA mB vB x tgt rows in r out max-grad-norm lr beta1 beta2 eps adam-t])

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

(defn joint-clipped-adam-oracle
  "Independent real-arithmetic recurrence over the coordinate derivatives above. The final
   arrays are rounded to their declared Float storage; no Raster AD, clipping, or optimizer code
   participates."
  [{:keys [A B mA vA mB vB max-grad-norm lr beta1 beta2 eps adam-t] :as input}]
  (let [{:keys [loss dA dB]} (oracle input)
        norm (Math/sqrt (+ (reduce + 0.0 (map #(* % %) dA))
                           (reduce + 0.0 (map #(* % %) dB))))
        scale (if (zero? norm) 1.0 (min 1.0 (/ max-grad-norm norm)))
        bc1 (- 1.0 (Math/pow beta1 adam-t))
        bc2 (- 1.0 (Math/pow beta2 adam-t))
        update
        (fn [parameters gradients previous-m previous-v]
          (let [moments (mapv (fn [gradient old-m old-v]
                                (let [g (* scale gradient)]
                                  [(+ (* beta1 (double old-m)) (* (- 1.0 beta1) g))
                                   (+ (* beta2 (double old-v)) (* (- 1.0 beta2) g g))]))
                              gradients previous-m previous-v)]
            {:parameters
             (float-array
              (map (fn [parameter [m v]]
                     (- (double parameter)
                        (* lr (/ (/ m bc1) (+ (Math/sqrt (/ v bc2)) eps)))))
                   parameters moments))
             :m (float-array (map first moments))
             :v (float-array (map second moments))}))
        a (update A dA mA vA)
        b (update B dB mB vB)]
    {:loss loss :gradient-norm norm :scale scale
     :A (:parameters a) :mA (:m a) :vA (:v a)
     :B (:parameters b) :mB (:m b) :vB (:v b)}))

(defn max-error [a b]
  (assert (= (count a) (count b)))
  (reduce max 0.0 (map #(Math/abs (- (double %1) (double %2))) a b)))
