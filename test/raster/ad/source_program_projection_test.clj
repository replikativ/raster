(ns raster.ad.source-program-projection-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]))

(deftest output-projections-share-forward-data-not-adjoint-accumulation
  (let [typed #(with-meta % {:raster.type/tag 'double})
        x (typed 'x) a (typed 'a) b (typed 'b)]
    (with-bindings {#'reverse/*gensym-counter* (atom 0)}
      (let [forward (#'reverse/forward-pass
                     [a (list 'raster.numeric/* x x)
                      b (list 'raster.numeric/+ a x)] [x])
            projections
            (with-redefs-fn
              {#'reverse/forward-pass
               (fn [& _] (throw (ex-info "projection reran forward normalization" {})))}
              #(mapv (fn [output]
                       (#'reverse/reverse-program-projection forward [x] {output [1.0]}))
                     [a b]))]
        (doseq [[projection expected] (map vector projections [[6.0] [7.0]])]
          (is (= expected
                 (eval (list 'let*
                             (vec (concat [x 3.0] (:fwd-bindings forward)
                                          (get-in projection [:rev-ctx :bindings])))
                             (:param-adj-syms projection)))))
          (is (= #{} (:pullback-replays projection))))))))
