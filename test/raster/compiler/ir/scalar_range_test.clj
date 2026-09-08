(ns raster.compiler.ir.scalar-range-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.scalar-range :as ranges]))

(deftest typed-index-ranges-cover-every-small-domain-value
  (doseq [width [1 2 7 12] divisor [1 2 3 5]
          op [:floor-div :mod]]
    (let [expression (body/expression op 'i (body/index-cast divisor :long :exact))
          proof (ranges/typed-index-range expression {'i :long}
                                          {'i {:lower 0 :upper (dec width)}})
          values (map #((if (= op :mod) rem quot) % divisor) (range width))]
      (is (some? proof))
      (is (every? #(<= (:lower proof) % (:upper proof)) values)))))

(deftest typed-index-proof-declines-unsafe-or-unknown-intermediates
  (let [prove #(ranges/typed-index-range % {'i :int} {'i {:lower 0 :upper 3}})]
    (is (nil? (prove (body/expression :add 'i Integer/MAX_VALUE))))
    (is (nil? (prove (body/expression :sub
                                    (body/expression :add 'i Integer/MAX_VALUE)
                                    Integer/MAX_VALUE))))
    (is (nil? (prove (body/expression :floor-div 'i 0))))
    (is (nil? (prove (body/expression :mod 'i -1))))
    (is (nil? (prove 'unknown)))
    (is (nil? (prove (body/index-cast 'i :long :wrap))))
    (is (nil? (prove (body/expression :add Integer/MAX_VALUE 1 -1))))
    (is (nil? (prove (body/expression :mul Integer/MAX_VALUE 2 0))))
    (is (nil? (ranges/typed-index-range (body/index-cast 'j :int :exact)
                                        {'j :long} {'j {:lower 0 :upper 3}})))
    (is (nil? (prove (reduce (fn [x _] (body/expression :add x 0)) 'i (range 130)))))))
