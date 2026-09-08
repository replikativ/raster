(ns raster.compiler.ir.scalar-range-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.scalar-range :as ranges]))

(deftest accumulation-proof-includes-intermediate-prefixes
  (let [term (ranges/arithmetic :* (repeat 2 (ranges/for-dtype :byte)))
        prove #(ranges/accumulation-prefixes (ranges/literal 0 :long) term %)]
    (is (= {:lower -16256 :upper 16384} term))
    (is (ranges/contained-in-dtype? (prove 131068) :int))
    (is (not (ranges/contained-in-dtype? (prove 131072) :int)))
    (is (= 2147483648 (:upper (prove 131072))))
    (is (= (*' Long/MAX_VALUE 16384) (:upper (prove Long/MAX_VALUE))))
    (doseq [n [nil 'n -1 1.5]] (is (nil? (prove n)))))
  (is (= {:lower -5 :upper 10}
         (ranges/accumulation-prefixes {:lower 10 :upper 10}
                                      {:lower -3 :upper -2} 5))))

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
