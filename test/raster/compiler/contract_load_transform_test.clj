(ns raster.compiler.contract-load-transform-test
  (:require [clojure.test :refer [deftest is]]
            [raster.par]
            [raster.compiler.ir.contraction-facts :as facts]))

(deftest host-contract-applies-decode-before-accumulation
  (let [a (double-array [3.0]) out (double-array 1) shift 1.0]
    (raster.par/contract out [[i 1]] [[k 1]] (aget a k)
                         :decode {a (- x shift)})
    (is (= [2.0] (vec out)))))

(deftest host-stages-and-result-transform-compose-with-decoded-loads
  (let [a (double-array [3.0 4.0 5.0 6.0]) b (double-array [2.0 3.0 4.0 5.0])
        out (double-array 1) scale 0.5 bias 3.0]
    (raster.par/contract out [[i 1]] [[blk 2] [t 2]]
                         (* (aget a (+ (* blk 2) t)) (aget b (+ (* blk 2) t)))
                         :decode {a (- x 1.0) b (+ x 2.0)}
                         :stages [{:axis blk :extent 2 :dtype :double :init 0.0 :lift (* inner scale)}
                                  {:axis t :extent 2 :dtype :double :init 0.0}]
                         :epilogue {:acc value :expr (+ value bias) :dtype :double})
    (is (= [44.0] (vec out)))))

(deftest load-transform-substitution-preserves-lexical-scope
  (let [a (double-array [3.0 100.0]) out (double-array 1)]
    (raster.par/contract out [[i 1]] [[k 1]] (aget a k)
                         :decode {a (let [k 1] (+ x k))})
    (is (= [4.0] (vec out)) "decode-local k cannot capture the raw load's index"))
  (let [a (double-array [3.0]) out (double-array 1)]
    (raster.par/contract out [[i 1]] [[k 1]] (aget a k)
                         :decode {a (let [x 7.0] x)})
    (is (= [7.0] (vec out)) "a local x shadows the raw-load binder")))

(deftest summand-locals-cannot-capture-decode-captures-or-array-identity
  (let [a (double-array [3.0]) out (double-array 1) shift 1.0]
    (raster.par/contract out [[i 1]] [[k 1]]
                         (let [shift 100.0] (aget a k))
                         :decode {a (- x shift)})
    (is (= [2.0] (vec out)) "decode shift belongs to its declaration scope"))
  (let [a (double-array [3.0]) out (double-array 1)]
    (raster.par/contract out [[i 1]] [[k 1]]
                         (let [a (double-array [100.0])] (aget a k))
                         :decode {a (- x 1.0)})
    (is (= [100.0] (vec out)) "a shadowing local array does not inherit the outer decode")))

(deftest load-transform-rewriting-preserves-metadata-and-quoted-data
  (let [tagged (with-meta '(+ (aget a k) 1.0) {:raster.type/tag 'double})
        result (facts/apply-load-transforms tagged {'a '(- x shift)})]
    (is (= 'double (:raster.type/tag (meta result))))
    (is (= '(+ (- (aget a k) shift) 1.0) result))
    (is (= '(quote (aget a k))
           (facts/apply-load-transforms '(quote (aget a k)) {'a '(- x shift)})))))
