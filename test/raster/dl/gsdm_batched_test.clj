(ns raster.dl.gsdm-batched-test
  "The batched GSDM (B examples as one disjoint-union graph) against the
  single-example model, and the training batches it is fed."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.diffusion :as diff]
            [raster.dl.gsdm :as gsdm]
            [raster.params :as rp]))

(def ^:private emb-dim 8)
(def ^:private n-per 4)
(def ^:private n-layers 2)

(def ^:private cfg (gsdm/make-gsdm-config :n-layers n-layers :emb-dim emb-dim :n-heads 2 :n-spaces 1))

;; three examples of four variables with different graphs, states and timesteps
(def ^:private examples
  (let [rng (java.util.Random. 3)
        full (let [{:keys [src-edges dst-edges]} (gsdm/fully-connected-edges n-per)]
               [(vec src-edges) (vec dst-edges)])]
    (mapv (fn [[s d] st t]
            {:x0 (vec (repeatedly n-per #(.nextGaussian rng)))
             :xt (vec (repeatedly n-per #(.nextGaussian rng)))
             :states st :src s :dst d :t t})
          [[[0 1 1 2 2 3] [1 0 2 1 3 2]] [[0 1 0 2 0 3] [1 0 2 0 3 0]] full]
          [[0 1 1 0] [0 0 1 3] [1 0 0 0]]
          [10.0 50.0 90.0])))

(defn- single-loss [loss w {:keys [x0 xt states src dst t]}]
  (let [spaces (long-array n-per)]
    (loss w (double-array xt) spaces (double-array x0) (long-array states)
          (gsdm/compute-position-embeddings spaces n-per emb-dim)
          (long-array src) (long-array dst) t n-per (count src))))

(defn- batch-args [exs]
  (let [b (count exs)
        {:keys [node-example to-cf to-nm]} (gsdm/batch-layout b n-per emb-dim)
        offsets (map #(* % n-per) (range b))]
    [(double-array (mapcat :xt exs)) (long-array (* b n-per)) (double-array (mapcat :x0 exs))
     (long-array (mapcat :states exs))
     (double-array (mapcat (fn [_] (gsdm/compute-position-embeddings (long-array n-per) n-per emb-dim)) exs))
     (long-array (mapcat (fn [e o] (map #(+ % o) (:src e))) exs offsets))
     (long-array (mapcat (fn [e o] (map #(+ % o) (:dst e))) exs offsets))
     (gsdm/embed-timesteps (map :t exs) emb-dim) node-example to-cf to-nm
     b n-per (reduce + (map (comp count :src) exs))]))

(deftest the-batch-is-its-examples
  ;; the batched loss is the latent-count-weighted mean of the per-example
  ;; losses: per-example GroupNorm and timesteps, no edges across examples
  (let [w (gsdm/flat->tree (gsdm/init-gsdm-weights cfg) n-layers)
        single (gsdm/make-gsdm-loss {:n-layers n-layers :emb-dim emb-dim :n-heads 2 :n-spaces 1})
        batched (gsdm/make-gsdm-batched-loss {:n-layers n-layers :emb-dim emb-dim :n-heads 2 :n-spaces 1})
        losses (mapv #(single-loss single w %) examples)
        counts (mapv #(count (filter zero? (:states %))) examples)
        expected (/ (reduce + (map * losses counts)) (reduce + counts))]
    (is (< (Math/abs (- expected (apply batched w (batch-args examples)))) 1e-12))
    (testing "one example is the single-example model"
      (is (< (Math/abs (- (first losses) (apply batched w (batch-args [(first examples)])))) 1e-12)))))

(deftest batch-layout-permutations-are-inverse
  (let [{:keys [to-cf to-nm node-example]} (gsdm/batch-layout 3 4 8)]
    (is (= (range (* 3 4 8)) (map #(aget ^longs to-nm (aget ^longs to-cf %)) (range (* 3 4 8)))))
    (is (= [0 0 0 0 1 1 1 1 2 2 2 2] (vec node-example)))
    ;; channel-first [b c i] reads node-major [(b·n + i)·d + c]
    (is (= (+ (* (+ (* 1 4) 2) 8) 5) (aget ^longs to-cf (+ (* 1 8 4) (* 5 4) 2))))))

(deftest condition-masks
  (let [rng (java.util.Random. 1)
        default (long-array [gsdm/LATENT gsdm/OBSERVED gsdm/MARGINALIZED gsdm/INTERVENED])]
    (is (= [0 1 2 3] (vec (gsdm/condition-states default {:posterior 1.0} rng))))
    (is (= [0 0 2 0] (vec (gsdm/condition-states default {:joint 1.0} rng))))
    (is (= [1 0 2 0] (vec (gsdm/condition-states default {:likelihood 1.0} rng))))
    (testing "random masks keep marginalized variables and observe about half"
      (let [draws (repeatedly 2000 #(vec (gsdm/condition-states default {:random 1.0} rng)))]
        (is (every? #(= 2 (nth % 2)) draws))
        (is (< 0.45 (/ (count (filter #(= 1 (first %)) draws)) 2000.0) 0.55))))))

(deftest training-batches
  (let [problem (gsdm/make-linear-gaussian-problem 3)
        ac (diff/compute-alphas-cumprod (diff/cosine-beta-schedule 100))
        [xt spaces target states pos-emb src dst t-emb node-example _ _ b n-per n-edges]
        (gsdm/training-batch problem 5 ac {:posterior 1.0} cfg (java.util.Random. 2))]
    (is (= [5 4 (* 5 16)] [b n-per n-edges]))
    (is (= [20 20 20 20 (* 20 emb-dim) 80 80 (* 5 emb-dim) 20]
           (mapv alength [xt spaces target states pos-emb src dst t-emb node-example])))
    (testing "clamped variables keep x0 as input and target, latent ones are noised"
      (doseq [i (range 20)]
        (if (gsdm/clamped? (aget ^longs states i))
          (is (== (aget ^doubles target i) (aget ^doubles xt i)))
          (is (not= (aget ^doubles target i) (aget ^doubles xt i))))))
    (testing "edges stay within their example"
      (is (every? true? (map #(= (quot %1 4) (quot %2 4)) src dst)))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"one size"
                        (gsdm/training-batch (let [a (gsdm/make-linear-gaussian-problem 2)
                                                   c (gsdm/make-linear-gaussian-problem 3)
                                                   k (atom 0)]
                                               (fn [rng] (if (even? (swap! k inc)) (a rng) (c rng))))
                                             2 (diff/compute-alphas-cumprod (diff/cosine-beta-schedule 10))
                                             gsdm/default-mask-mix cfg (java.util.Random. 1)))))

(deftest preconditioning
  (testing "at unit data scale: input x_t, prediction √a·x_t + √(1−a)·F"
    (let [{:keys [c-in c-skip c-out]} (gsdm/preconditioning cfg 0.36)]
      (is (< (Math/abs (- c-in 1.0)) 1e-12))
      (is (< (Math/abs (- c-skip 0.6)) 1e-12))
      (is (< (Math/abs (- c-out 0.8)) 1e-12))))
  (testing "plain x0 prediction"
    (is (= {:c-in 1.0 :c-skip 0.0 :c-out 1.0} (gsdm/preconditioning (assoc cfg :parametrization :x0) 0.36)))))

(deftest preconditioned-targets
  ;; one noise level and known x0: c-skip·x_t + c-out·target recovers x0
  (let [base (gsdm/make-linear-gaussian-problem 3)
        problem (fn [rng] (assoc (base rng) :values (double-array [1.0 2.0 3.0 4.0])))
        ac (double-array [0.64])]
    (doseq [config [(assoc cfg :sigma-data 2.0) (assoc cfg :parametrization :x0)]]
      (let [{:keys [c-in c-skip c-out]} (gsdm/preconditioning config 0.64)
            [input _ target states] (gsdm/training-batch problem 3 ac {:joint 1.0} config (java.util.Random. 7))]
        (doseq [i (range 12)]
          (let [x0 (double (inc (mod i 4)))
                xt (/ (aget ^doubles input i) c-in)]
            (is (not (gsdm/clamped? (aget ^longs states i))))
            (is (< (Math/abs (- x0 (+ (* c-skip xt) (* c-out (aget ^doubles target i))))) 1e-12))))))))

(deftest ^:compiled the-compiled-batch-is-its-examples
  (let [w (gsdm/flat->tree (gsdm/init-gsdm-weights cfg) n-layers)
        batched (gsdm/make-gsdm-batched-loss {:n-layers n-layers :emb-dim emb-dim :n-heads 2 :n-spaces 1})
        fast (rp/compile-aot batched)]
    (is (< (Math/abs (- (apply batched w (batch-args examples)) (apply fast w (batch-args examples)))) 1e-12))))

(deftest marginalized-training-batches
  ;; each example marginalizes its own 2 of 4 variables: all keep 2 nodes
  (let [problem (gsdm/make-linear-gaussian-problem 3)
        ac (diff/compute-alphas-cumprod (diff/cosine-beta-schedule 100))
        [xt _ _ states _ src dst _ node-example _ _ b n-per n-edges]
        (gsdm/training-batch problem 6 ac gsdm/default-mask-mix cfg (java.util.Random. 4) 2)]
    (is (= [6 2] [b n-per]))
    (is (= 12 (alength ^doubles xt) (alength ^longs states) (alength ^longs node-example)))
    (is (not-any? #(= gsdm/MARGINALIZED %) (vec states)))
    ;; the fully connected graph on the 2 kept variables, self-loops included
    (is (= (* 6 4) n-edges))
    (is (every? true? (map #(= (quot %1 2) (quot %2 2)) src dst)))))

(deftest given-marginalized-variables-are-always-removed
  ;; a problem whose default states already marginalize variable 2
  (let [base (gsdm/make-linear-gaussian-problem 3)
        problem (fn [rng] (let [ex (base rng) st (aclone ^longs (:states ex))]
                            (aset st 2 (long gsdm/MARGINALIZED))
                            (assoc ex :states st)))
        ac (diff/compute-alphas-cumprod (diff/cosine-beta-schedule 100))
        n-per-of (fn [k] (nth (gsdm/training-batch problem 4 ac {:posterior 1.0} cfg (java.util.Random. 5) k) 12))]
    (is (= 3 (n-per-of 0)))
    (is (= 2 (n-per-of 1)))))

(deftest ema-update
  (let [e (double-array [1.0 2.0]) w (double-array [3.0 6.0])]
    (gsdm/ema-update! e w 0.75 0.25 2)
    (is (= [1.5 3.0] (vec e)))))

(deftest marginalized-variables-keep-their-identity
  ;; variable 0 marginalized: the kept variables 1..3 carry the positions
  ;; they have in the full graph, not 0..2
  (let [base (gsdm/make-linear-gaussian-problem 3)
        problem (fn [rng] (let [ex (base rng) st (aclone ^longs (:states ex))]
                            (aset st 0 (long gsdm/MARGINALIZED))
                            (assoc ex :states st)))
        ac (diff/compute-alphas-cumprod (diff/cosine-beta-schedule 10))
        pos-emb (nth (gsdm/training-batch problem 1 ac {:posterior 1.0} cfg (java.util.Random. 6) 0) 4)
        full (gsdm/compute-position-embeddings (long-array 4) 4 emb-dim)]
    (is (= (vec (java.util.Arrays/copyOfRange ^doubles full emb-dim (* 4 emb-dim))) (vec pos-emb)))))
