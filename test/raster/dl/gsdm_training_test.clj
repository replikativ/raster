(ns raster.dl.gsdm-training-test
  "GSDM trained on a linear-Gaussian model against its exact posterior.
  ^:statistical: minutes of training, excluded from -M:test; run with
  -M:statistical."
  (:require [clojure.test :refer [deftest is]]
            [raster.dl.diffusion :as diff]
            [raster.dl.gsdm :as gsdm]))

;; z ~ N(0, 1), x_i ~ N(z, 0.5²), i = 1..3: z | x ~ N(Σx/0.25/13, 1/13)
(def ^:private posterior-sd (Math/sqrt (/ 1.0 13.0)))

(defn- posterior-samples [weights cfg betas xs n]
  (let [{:keys [src-edges dst-edges n-edges]} (gsdm/fully-connected-edges 4)
        spaces (long-array 4)
        states (long-array [gsdm/LATENT gsdm/OBSERVED gsdm/OBSERVED gsdm/OBSERVED])
        obs (double-array (cons 0.0 xs))
        rng (java.util.Random. 9)]
    (vec (repeatedly n #(aget ^doubles (gsdm/gsdm-sample weights cfg spaces states obs src-edges dst-edges
                                                         4 n-edges betas :rng rng)
                              0)))))

(deftest ^:statistical posterior-of-a-linear-gaussian
  ;; At this CPU budget (6000 steps of 32, emb 32, 2 layers) the trained
  ;; model is within about 0.2–0.6 posterior sds and 0.6–0.9 of the sd; the
  ;; bounds catch gross errors (an untrained or mis-conditioned model lands
  ;; several sds off with a fraction of the sd), not fine calibration.
  (let [cfg (gsdm/make-gsdm-config :n-layers 2 :emb-dim 32 :n-heads 2 :n-spaces 1)
        betas (diff/cosine-beta-schedule 100)
        {:keys [ema-weights]} (gsdm/train-gsdm (gsdm/make-linear-gaussian-problem 3)
                                               {:config cfg :batch 32 :steps 6000 :betas betas :seed 4
                                                :mask-mix {:posterior 1.0}})]
    (doseq [xs [[0.5 1.0 0.2] [-1.2 -0.6 -1.5]]]
      (let [zs (posterior-samples ema-weights cfg betas xs 400)
            exact (/ (/ (reduce + xs) 0.25) 13.0)
            m (/ (reduce + zs) (count zs))
            s (Math/sqrt (/ (reduce + (map #(let [e (- % m)] (* e e)) zs)) (dec (count zs))))]
        (is (< (Math/abs (/ (- m exact) posterior-sd)) 1.0) (str xs " mean " m " vs " exact))
        (is (< 0.6 (/ s posterior-sd) 1.4) (str xs " sd " s " vs " posterior-sd))))))
