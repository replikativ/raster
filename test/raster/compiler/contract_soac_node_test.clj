(ns raster.compiler.contract-soac-node-test
  "PAR/CONTRACT IS A FIRST-CLASS SOAC NODE. Device-free.

   It was the one par form that was NOT a node: `par-form->soac` returned nil, `segop-lower`
   DECLINED it (`:no-lowering-rule`), and `opencl-pass` compiled it through a side branch that
   re-parsed the surface form — calling `contraction-facts` three separate times. Increment 1's
   gate (north-star §8) names 'one contraction' travelling the same recorded path as map/reduce.

   The node carries `contraction-facts` as its SOLE payload (§10: no second registry). Everything
   downstream reads the facts; nothing re-derives them. `Dot` — one constructor and a unit test —
   is the dead representation this replaces."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.segop-lower-pass :as slp]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.backend.gpu.opencl-pass :as op]
            [raster.compiler.pipeline :as pl]
            [raster.linalg.contract]))

(def ^:private cform
  '(raster.par/contract C [[i 128] [j 128]] [[l 128]]
                        (raster.numeric/* (clojure.core/aget A (clojure.core/+ (clojure.core/* i 128) l))
                                          (clojure.core/aget B (clojure.core/+ (clojure.core/* l 128) j)))))
(def ^:private bound (list 'let* ['c cform] 'c))

(deftest a-contraction-is-a-soac-node-carrying-its-facts
  (let [n (soac/par-form->soac 'c cform 1 :dtype :double)]
    (is (soac/contract? n))
    (testing "facts are the payload, derived once"
      (is (= '[[i 128] [j 128]] (:free-axes (:facts n))))
      (is (= '[[l 128]] (:contract-axes (:facts n))))
      (is (= '[A B] (mapv :sym (:operands (:facts n)))))
      (is (reduction/product-reduction? (:reduction (:facts n)))))
    (testing "compatibility projections derive from the facts"
      (is (= '#{A B} (soac/soac-inputs n)))
      (is (= '#{C} (soac/soac-outputs n))))))

(deftest segop-lower-records-a-segcontract-instead-of-declining
  (let [r (slp/segop-lower-pass bound {:target-device :ze:0 :dtype :double})
        p (:form r)
        source (second (second bound))
        so (program/operations-for-binding p 'c source)]
    (is (= 1 (:segops-lowered (:stats r))))
    (is (empty? (:segops-declined (:stats r))) "was {:reason :no-lowering-rule}")
    (is (instance? raster.compiler.ir.segop.SegContract (first so)))
    (is (= :double (:dtype (first so))))
    (is (= :ze:0 (:device-id (first so))))))

(deftest opencl-pass-consumes-the-segcontract
  (let [run (fn [f] (op/opencl-pass f :device-id :ze:0 :dtype :double :min-elements 0))
        norm #(str/replace (str %) #"_\d{3,}" "_N")]
    (testing "consumed after segop-lower — reused, not re-lowered — routing unchanged"
      (let [r (run (:form (slp/segop-lower-pass bound {:target-device :ze:0 :dtype :double})))]
        (is (= 1 (:segop-reused (:stats r))))
        (is (nil? (:segop-relowered (:stats r))))
        (is (= :regtiled (kart/attribute (first (:kernels r)) :strategy)))))
    (testing "without segop-lower (door C): re-lowered and COUNTED"
      (is (= 1 (:segop-relowered (:stats (run bound))))))
    (testing "it is a refactor: identical kernel source both ways"
      (is (= (norm (:source (first (:kernels (run (:form (slp/segop-lower-pass bound {:target-device :ze:0 :dtype :double})))))))
             (norm (:source (first (:kernels (run bound))))))))))

(deftest the-cpu-path-still-compiles-and-is-numerically-right
  (testing "the production deftm through compile-aot on the JVM, against a scalar oracle"
    (let [f (pl/compile-aot #'raster.linalg.contract/contract-mm)
          A (double-array (range 16)) B (double-array (range 16))
          ref (double-array (for [i (range 4) j (range 4)]
                              (reduce + (for [l (range 4)] (* (aget A (+ (* i 4) l)) (aget B (+ (* l 4) j)))))))]
      (is (java.util.Arrays/equals ^doubles (f A B 4 4 4) ^doubles ref)))))
