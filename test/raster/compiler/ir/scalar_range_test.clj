(ns raster.compiler.ir.scalar-range-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.scalar-range :as ranges]))

(deftest counted-loop-proof-includes-the-terminal-increment
  (doseq [[lower upper step terminal]
          [[0 8 1 8] [0 8 3 9] [8 8 3 8] [9 8 3 9] [-5 4 3 4]
           [(dec Integer/MAX_VALUE) Integer/MAX_VALUE 1 Integer/MAX_VALUE]
           [(dec Integer/MAX_VALUE) Integer/MAX_VALUE 2 (inc (long Integer/MAX_VALUE))]
           [Long/MIN_VALUE Long/MAX_VALUE 1 Long/MAX_VALUE]
           [(dec Long/MAX_VALUE) Long/MAX_VALUE 2 (inc (bigint Long/MAX_VALUE))]]]
    (is (= {:lower lower :upper terminal} (ranges/counted-loop-index-range lower upper step))))
  (doseq [[lower upper step] [[0 'n 1] ['n 4 1] [0 4 'step] [0 4 0] [0 4 -1] [0 4 1.0]]]
    (is (nil? (ranges/counted-loop-index-range lower upper step))))
  (doseq [lower (range -4 5) upper (range -4 5) step [1 2 3]]
    (let [indices (vec (take-while #(< % upper) (iterate #(+ % step) lower)))
          terminal (if (seq indices) (+ (peek indices) step) lower)]
      (is (= {:lower lower :upper terminal} (ranges/counted-loop-index-range lower upper step))))))

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

(deftest additive-loop-carry-entry-excludes-the-final-backedge
  (is (= {:entry {:lower 4 :upper 10} :result {:lower 4 :upper 13}}
         (ranges/additive-loop-ranges {:lower 4 :upper 4} {:lower 2 :upper 3} 3)))
  (is (= {:entry nil :result {:lower 17 :upper 17}}
         (ranges/additive-loop-ranges {:lower 17 :upper 17} nil 0)))
  (doseq [trips [-1 nil 'n 1.0]]
    (is (nil? (ranges/additive-loop-ranges {:lower 0 :upper 0} {:lower 1 :upper 1} trips))))
  (is (nil? (ranges/additive-loop-ranges {:lower 0 :upper 0} nil 1)))
  (doseq [initial [-3 0 7] term [-4 -1 0 2 5] n (range 1 7)]
    (let [{:keys [entry result]} (ranges/additive-loop-ranges
                                {:lower initial :upper initial} {:lower term :upper term} n)
          prefixes (map #(+ initial (* term %)) (range (inc n)))]
      (is (every? #(<= (:lower entry) % (:upper entry)) (butlast prefixes)))
      (is (every? #(<= (:lower result) % (:upper result)) prefixes))))
  (is (= (bigint "18446744073709551615")
         (ranges/counted-loop-trips Long/MIN_VALUE Long/MAX_VALUE 1))))

(deftest quotient-proof-requires-a-positive-constant-divisor
  (doseq [lo (range -9 10) hi (range lo 10) divisor [1 2 7]]
    (let [proof (ranges/quotient [{:lower lo :upper hi}
                                 {:lower divisor :upper divisor}])]
      (is (every? #(<= (:lower proof) (quot % divisor) (:upper proof))
                  (range lo (inc hi))))))
  (doseq [divisor [nil {:lower 0 :upper 0} {:lower -1 :upper -1}
                   {:lower -1 :upper 1} {:lower 1 :upper 2}]]
    (is (nil? (ranges/quotient [(ranges/for-dtype :long) divisor])))))

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
