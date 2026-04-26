(ns raster.dl.gsdm-pytree-test
  "Parity gate: the new HMap-based gsdm-loss (raster.dl.gsdm-pytree) produces
  bit-identical loss values to the existing code-gen path
  (raster.dl.gsdm/compile-gsdm-train-fn). Once this passes, deleting
  gen-gsdm-loss-body / weight-param-order / weights-map->args / etc. is a
  mechanical follow-up."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.gsdm :as gsdm]
            [raster.dl.gsdm-pytree :as gp]
            [raster.params :as rp]))

(defn- fixture-data [n-vars n-edges emb-dim]
  (let [rng (java.util.Random. 7)]
    {:values (let [a (double-array n-vars)]
               (dotimes [i n-vars] (aset a i (.nextGaussian rng))) a)
     :spaces (long-array (repeat n-vars 0))
     :target (let [a (double-array n-vars)]
               (dotimes [i n-vars] (aset a i (.nextGaussian rng))) a)
     :states (long-array (repeat n-vars gsdm/LATENT))
     :pos-emb (let [a (double-array (* n-vars emb-dim))]
                (dotimes [i (alength a)] (aset a i (.nextGaussian rng))) a)
     :src-edges (long-array (repeat n-edges 0))
     :dst-edges (long-array (repeat n-edges 1))
     :t 0.5}))

(deftest ^:compiled gsdm-pytree-fused-train-step-converges
  (testing "Fused pytree train step (compile-train-step) reduces loss over steps"
    (let [n-layers 2 emb-dim 8 n-heads 2 n-spaces 1
          n-vars 4 n-edges 6
          cfg (gsdm/make-gsdm-config :emb-dim emb-dim :n-heads n-heads
                                      :n-layers n-layers :n-spaces n-spaces)
          weights (gsdm/init-gsdm-weights cfg)
          gsdm-loss-var (gp/make-gsdm-loss {:n-layers n-layers :emb-dim emb-dim
                                            :n-heads n-heads :n-spaces n-spaces})
          {:keys [train-fn init-state]} (rp/compile-train-step gsdm-loss-var)
          w-pytree (gp/flat->pytree weights n-layers)
          state    (init-state w-pytree)
          {:keys [values spaces target states pos-emb src-edges dst-edges t]}
          (fixture-data n-vars n-edges emb-dim)
          losses (vec (for [step (range 1 6)]
                        (train-fn w-pytree (:m state) (:v state)
                                  values spaces target states pos-emb src-edges dst-edges
                                  t n-vars n-edges 1.0 0.05 0.9 0.999 1e-8 step)))]
      (is (every? Double/isFinite losses))
      (is (< (last losses) (first losses)) "loss decreases over 5 steps"))))

(deftest ^:compiled gsdm-pytree-matches-existing-codegen
  (testing "Forward loss is bit-identical between defmodel path and existing code-gen path"
    (let [n-layers 2 emb-dim 8 n-heads 2 n-spaces 1
          n-vars 4 n-edges 6
          cfg (gsdm/make-gsdm-config :emb-dim emb-dim :n-heads n-heads
                                      :n-layers n-layers :n-spaces n-spaces)
          weights (gsdm/init-gsdm-weights cfg)
          {:keys [values spaces target states pos-emb src-edges dst-edges t]}
          (fixture-data n-vars n-edges emb-dim)

          ;; --- New path: defmodel + compile-aot ---
          gsdm-loss-var (gp/make-gsdm-loss {:n-layers n-layers :emb-dim emb-dim
                                            :n-heads n-heads :n-spaces n-spaces})
          fast (rp/compile-aot gsdm-loss-var)
          w-pytree (gp/flat->pytree weights n-layers)
          loss-pytree (fast w-pytree values spaces target states
                            pos-emb src-edges dst-edges t n-vars n-edges)

          ;; --- Existing path: compile-gsdm-train-fn ---
          existing (gsdm/compile-gsdm-train-fn :n-layers n-layers :emb-dim emb-dim
                                                :n-heads n-heads :n-spaces n-spaces)
          w-args (gsdm/weights-map->args weights n-layers)
          {:keys [m-arrays v-arrays]} (gsdm/init-adam-state weights n-layers)
          loss-existing (double
                          (apply (:train-fn existing)
                                 (concat w-args
                                         [values spaces target states pos-emb
                                          src-edges dst-edges
                                          (double t) (double n-vars) (double n-edges)]
                                         m-arrays v-arrays
                                         [0.0 0.9 0.999 1e-8 1.0 1])))]
      (is (Double/isFinite loss-pytree))
      (is (= loss-existing loss-pytree)
          "Bit-identical loss between paths (deterministic init, same data)"))))
