(ns raster.params-train-step-precision-test
  (:require [clojure.test :refer [deftest is]]
            [raster.params :as params]
            [raster.numeric :as n]))

(params/defmodel mixed-precision-loss
  [w :- (Params (HMap :mandatory {:single (Param (Array float))
                                 :double (Param (Array double))
                                 :frozen (Frozen (Array float))}))]
  :- Double
  (let [x (double (aget (:single w) 0))
        y (aget (:double w) 0)]
    (n/+ (n/* x x) (n/* y y))))

(deftest generated-optimizer-casts-scalars-to-each-parameter-leaf
  (doseq [optimizer [:sgd :adam]]
    (let [weights {:single (float-array [2.0]) :double (double-array [3.0])
                   :frozen (float-array [7.0])}
          step (params/compile-train-step #'mixed-precision-loss {:optimizer optimizer})
          state ((:init-state step) weights)
          loss (case optimizer
                 :sgd ((:train-fn step) weights 100.0 0.125)
                 :adam ((:train-fn step) weights (:m state) (:v state)
                        100.0 0.125 0.9 0.999 1.0e-8 (long 1)))]
      (is (= 13.0 loss))
      (is (= [7.0] (vec (:frozen weights))))
      (is (= (class (float-array 0)) (class (:single weights))))
      (is (= (class (double-array 0)) (class (:double weights))))
      (case optimizer
        :sgd (do (is (= [1.5] (vec (:single weights))))
                 (is (= [2.25] (vec (:double weights)))))
        :adam (do (is (< (Math/abs (- 1.875 (double (aget ^floats (:single weights) 0)))) 1.0e-6))
                  (is (< (Math/abs (- 2.875 (aget ^doubles (:double weights) 0))) 1.0e-8))
                  (doseq [[role single-value double-value] [[:m 0.4 0.6] [:v 0.016 0.036]]]
                    (let [single (get-in state [role :single])
                          double-values (get-in state [role :double])]
                      (is (= (class (float-array 0)) (class single)))
                      (is (= (class (double-array 0)) (class double-values)))
                      (is (< (Math/abs (- single-value (double (aget ^floats single 0)))) 1.0e-6))
                      (is (< (Math/abs (- double-value (aget ^doubles double-values 0))) 1.0e-12)))))))))
