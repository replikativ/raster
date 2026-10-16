(ns raster.ad.source-backward-demand-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as reverse]))

(deftm unused-singular-map [x :- Double out :- (Array double)] :- Double
  (let [ignored (par/map! out i 1 nil (n/sqrt x))]
    (n/+ x 1.0)))

(deftm used-result-map [x :- Double out :- (Array double)] :- Double
  (let [written (par/map! out i 1 nil (n/* x x))]
    (aget written 0)))

(deftm used-buffer-map [x :- Double out :- (Array double)] :- Double
  (let [ignored (par/map! out i 1 nil (n/* x x))]
    (aget out 0)))

(deftest unused-array-writer-retains-primal-not-singular-pullback
  (let [gradient (reverse/value+grad #'unused-singular-map :wrt [0])
        out (double-array [17.0])]
    (is (= [1.0 1.0 nil] (gradient 0.0 out)))
    (is (= [0.0] (vec out)))
    (is (= [5.0 1.0 nil] (gradient 4.0 out)))
    (is (= [2.0] (vec out)))))

(deftest cotangents-through-result-and-declared-buffer-both-demand-backward
  (doseq [source [#'used-result-map #'used-buffer-map]]
    (let [out (double-array [17.0])
          gradient (reverse/value+grad source :wrt [0])]
      (is (= [9.0 6.0 nil] (gradient 3.0 out)))
      (is (= [9.0] (vec out))))))

(deftest declared-output-demand-is-not-numeric-nonzero-testing
  (let [sym (with-meta 'value {:raster.type/tag 'double})
        record {:type :probe :sym sym :written-arrs '[first-out second-out]
                :replays '#{input}}]
    (doseq [[seeds expected]
            [[{} false] [{'first-out []} false] [{sym []} false]
             [{sym [nil]} true] [{'second-out [0.0]} true]
             [{'first-out ['dy]} true] [{sym ['dy]} true]]]
      (with-bindings {#'reverse/*gensym-counter* (atom 0)}
        (let [calls (atom 0)
              result (with-redefs-fn
                       {#'reverse/emit-backward
                        (fn [_ _ _ context] (swap! calls inc) context)}
                       #(#'reverse/reverse-pass [record] {sym true} seeds))]
          (is (= (if expected 1 0) @calls))
          (is (= (if expected '#{input} #{}) (:pullback-replays result))))))))
