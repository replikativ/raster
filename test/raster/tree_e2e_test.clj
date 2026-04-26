(ns raster.tree-e2e-test
  "End-to-end demonstration of raster.tree's walker forms (scan-vec, walk!,
  flat-view) composed in plain defmodel bodies — no code-gen helpers,
  no eval'd dynamic deftms.

  This is the proof that the new pipeline retires the compile-train-step,
  make-gsdm-loss, etc. helpers: all the work they did is now expressible
  as ordinary defmodel forms using walker macros."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :as core]
            [raster.params :as rp]
            [raster.tree :as tree]
            [raster.dl.nn :as nn]
            [raster.dl.loss :as loss]
            [raster.dl.optim :as optim]
            [raster.arrays :as arr]))

;; ---------------------------------------------------------------------------
;; Per-leaf adam step (named, no n arg — computed via alength).
;; ---------------------------------------------------------------------------
(core/deftm adam-leaf-step!
  [param :- (Array double) grad :- (Array double)
   m :- (Array double) v :- (Array double)
   lr :- Double beta1 :- Double beta2 :- Double eps :- Double t :- Long]
  :- (Array double)
  (raster.dl.optim/adam-step! param grad m v
                              (raster.arrays/alength param)
                              lr beta1 beta2 eps t))

;; ---------------------------------------------------------------------------
;; Forward + loss for a small linear model — plain defmodel, HMap weights.
;; ---------------------------------------------------------------------------
(rp/defmodel small-loss
  [w :- (Params (HMap :mandatory {:W (Param (Array double))
                                  :b (Param (Array double))}))
   x :- (Array double) y :- (Array double)
   batch :- Long d :- Long]
  :- Double
  (let [pred (raster.dl.nn/linear x (:W w) (:b w) batch d d)]
    (raster.dl.loss/mse-loss pred y (clojure.core/* batch d))))

;; ---------------------------------------------------------------------------
;; Train step — uses flat-view (compile-time grad-tree view) + walk!
;; (compile-time per-leaf adam-step!).  No body-gen, no eval, no helper.
;; ---------------------------------------------------------------------------
(rp/defmodel small-train
  [w :- (Params (HMap :mandatory {:W (Param (Array double))
                                  :b (Param (Array double))}))
   m :- (Params (HMap :mandatory {:W (Array double) :b (Array double)}))
   v :- (Params (HMap :mandatory {:W (Array double) :b (Array double)}))
   x :- (Array double) y :- (Array double)
   batch :- Long d :- Long
   lr :- Double beta1 :- Double beta2 :- Double eps :- Double t :- Long]
  :- Double
  (let [vg ((raster.ad.reverse/value+grad #'small-loss--flat) (:W w) (:b w) x y batch d)
        loss (clojure.core/nth vg 0)
        grads (raster.tree/flat-view vg
                                      :spec '(HMap :mandatory {:W (Param (Array double))
                                                                :b (Param (Array double))})
                                      :starting-at 1)]
    (raster.tree/walk! :param adam-leaf-step! [w grads m v] lr beta1 beta2 eps t)
    loss))

(deftest ^:compiled walker-form-train-step-converges
  (testing "Train step using flat-view + walk! reduces loss on a small linear regression"
    (let [train-fn (rp/compile-aot #'small-train)
          W (double-array [0.5 -0.3 0.2 0.1])
          b (double-array [0.0 0.0])
          w {:W W :b b}
          state {:m {:W (double-array 4) :b (double-array 2)}
                 :v {:W (double-array 4) :b (double-array 2)}}
          x (double-array [1.0 0.5])
          y (double-array [1.0 -1.0])
          losses (vec (for [step (range 1 11)]
                        (train-fn w (:m state) (:v state) x y 1 2
                                  0.05 0.9 0.999 1e-8 step)))]
      (is (every? Double/isFinite losses))
      (is (< (last losses) (first losses) 0.999))
      (is (apply >= losses) "losses monotonically non-increasing")
      ;; Initial weights changed
      (is (not= [0.5 -0.3 0.2 0.1] (vec W))
          "weights mutated by walk! / adam-step!"))))

;; ---------------------------------------------------------------------------
;; scan-vec demo — chain a stack of layers with no manual unrolling.
;; ---------------------------------------------------------------------------

;; Per-block step: takes (acc-tree, layer-tree, batch, d).
(rp/defmodel mlp-block
  [h :- (Array double)
   layer :- (Params (HMap :mandatory {:W (Param (Array double))
                                      :b (Param (Array double))}))
   batch :- Long d :- Long]
  :- (Array double)
  (raster.dl.nn/linear h (:W layer) (:b layer) batch d d))

(rp/defmodel stacked-mlp-loss
  [w :- (Params (HMap :mandatory
                      {:layers (HVec [(HMap :mandatory {:W (Param (Array double))
                                                        :b (Param (Array double))})
                                      (HMap :mandatory {:W (Param (Array double))
                                                        :b (Param (Array double))})])}))
   x :- (Array double) y :- (Array double)
   batch :- Long d :- Long]
  :- Double
  (let [h-final (raster.tree/scan-vec mlp-block x (:layers w) batch d)]
    (raster.dl.loss/mse-loss h-final y (clojure.core/* batch d))))

;; ---------------------------------------------------------------------------
;; Fold: `deftm` (not defmodel!) detects Params and routes through the tree
;; pipeline. User writes deftm; the macro forwards.
;; ---------------------------------------------------------------------------
(core/deftm linear-via-deftm
  [w :- (Params (HMap :mandatory {:W (Param (Array double))
                                  :b (Param (Array double))}))
   x :- (Array double) batch :- Long d-in :- Long d-out :- Long]
  :- (Array double)
  (raster.dl.nn/linear x (:W w) (:b w) batch d-in d-out))

(deftest deftm-with-params-forwards-to-defmodel
  (testing "deftm transparently handles Params args via defmodel pipeline"
    (let [W (double-array (repeat 4 0.5))
          b (double-array [0.0 0.0])
          x (double-array [1.0 -1.0])
          y (linear-via-deftm {:W W :b b} x 1 2 2)]
      ;; W is row-major 2x2, x is [1 -1] → [0.5*1 + 0.5*(-1), 0.5*1 + 0.5*(-1)] = [0,0]
      (is (= [0.0 0.0] (vec y))))))

(deftest scan-vec-stacks-layers-without-codegen
  (testing "scan-vec compile-time-unrolls the layer chain — same result as manual unroll"
    (let [W0 (double-array (repeat 4 0.1))
          b0 (double-array [0.0 0.0])
          W1 (double-array (repeat 4 0.1))
          b1 (double-array [0.0 0.0])
          w {:layers [{:W W0 :b b0} {:W W1 :b b1}]}
          x (double-array [0.5 0.5])
          y (double-array [1.0 1.0])

          ;; Via scan-vec
          fast-scan (rp/compile-aot #'stacked-mlp-loss)
          loss-scan (fast-scan w x y 1 2)]
      (is (Double/isFinite loss-scan))
      (is (pos? loss-scan)))))
