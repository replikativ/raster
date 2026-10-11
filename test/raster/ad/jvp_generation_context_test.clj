(ns raster.ad.jvp-generation-context-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.ad.reverse :as rev]
            [raster.ad.jvp :as jvp]))

(deftm explicit-core-carry [h0 :- Double, steps :- Long] :- Double
  (loop [i 0 acc h0]
    (if (< i steps)
      (recur (inc i)
             (if (n/== i 0)
               (clojure.core/double (n/+ acc 0.0))
               (let [z (n/- acc 16777216.0)] (n/* z z))))
      acc)))

(deftest public-jvp-core-cast-nonlinear-loop-retains-primal-and-tangent
  (let [directional (jvp/jvp #'explicit-core-carry)
        x 16777216.5 h 0.0625]
    (doseq [[steps primal derivative] [[0 x 1.0] [1 x 1.0] [2 0.25 1.0]]]
      (is (= primal (explicit-core-carry x steps)))
      (is (= derivative (/ (- (explicit-core-carry (+ x h) steps)
                              (explicit-core-carry (- x h) steps)) (* 2.0 h))))
      (is (= [primal derivative] (directional x steps 1.0)))
      (is (= [primal (* 2.0 derivative)] (directional x steps 2.0))))))

(deftest nested-public-jvp-construction-inherits-the-monotonic-generation-context
  (rev/call-with-shared-ad-gensym
   (fn []
     (let [counter-var (ns-resolve 'raster.ad.reverse '*gensym-counter*)
           counter @counter-var
           fresh (ns-resolve 'raster.ad.reverse 'ad-gensym)
           before (fresh "context_probe")
           first-jvp (jvp/jvp #'explicit-core-carry)
           middle (fresh "context_probe")
           second-jvp (rev/call-with-shared-ad-gensym
                       #(jvp/jvp #'explicit-core-carry))
           after (fresh "context_probe")]
       (is (identical? counter @counter-var))
       (is (= 3 (count (set [before middle after]))))
       ;; Both independently generated programs remain executable after the
       ;; nested construction; no temporary from one phase overwrote another.
       (is (= [0.25 1.0] (first-jvp 16777216.5 2 1.0)))
       (is (= [0.25 2.0] (second-jvp 16777216.5 2 2.0)))))))
