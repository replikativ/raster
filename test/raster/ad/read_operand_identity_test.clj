(ns raster.ad.read-operand-identity-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.ad.reverse :as reverse]
            [raster.compiler.core.op-descriptor :as operation]))

(defn- analyze [body sequential?]
  (reverse/call-with-shared-ad-gensym
   #((var reverse/analyze-par-map-body) body 'i sequential?)))

(defn- retained-read [array index]
  (with-meta (list '.invk 'read-implementation array index)
    {:raster.op/original 'raster.arrays/aget :raster.type/tag 'double}))

(deftest active-read-operands-survive-retained-dispatch
  (let [array (with-meta 'samples {:raster.type/tag 'doubles})]
    (doseq [read [(list 'aget array 'i)
                  (list 'clojure.core/aget array 'i)
                  (list 'raster.arrays/aget array 'i)
                  (retained-read array 'i)]]
      (testing (pr-str read)
        (is (operation/aget-call? read))
        (is ((var reverse/gradient-bearing-array-read?) read))
        (doseq [body [(list '+ 'acc read)
                      (list 'let* ['loaded read] (list '+ 'acc 'loaded))]]
          (let [reads (:agets (analyze body true))]
            (is (= [['samples 'i]] (mapv (juxt :arr :idx) reads)))
            (is (= (meta array) (meta (:arr (first reads)))))))))))

(deftest read-identity-does-not-relax-rank-activity-or-scope
  (let [array (with-meta 'samples {:raster.type/tag 'doubles})
        read (retained-read array 0)
        lexical (vary-meta read assoc :raster.op/lexical-callee true)]
    (doseq [unsupported [lexical
                         '(.invk read-implementation samples i)
                         '(foreign/aget samples i)
                         '(aget samples i j)
                         (with-meta '(.invk read-implementation samples i j)
                           {:raster.op/original 'raster.arrays/aget})]]
      (is (not ((var reverse/gradient-bearing-array-read?) unsupported))))
    (with-bindings {(var reverse/*constant-gradient-arrays*) #{'samples}}
      (is (not ((var reverse/gradient-bearing-array-read?) read)))
      (is (empty? (:agets (analyze (list '+ 'acc read) true)))))
    (is (empty? (:agets (analyze (list '+ 'acc read) false)))
        "a parallel map does not gain literal-index scatter admission")
    (is (= [['samples 0]]
           (mapv (juxt :arr :idx) (:agets (analyze (list '+ 'acc read) true)))))
    (doseq [body [(list '+ 'acc (list 'if 'condition read 0.0))
                  (list 'let* ['samples 'other] (list '+ 'acc read))]]
      (is (empty? (:agets (analyze body true)))
          "conditional and shadowed reads cannot move before their scope"))))

(deftest active-read-dtype-uses-the-array-not-the-dispatch-receiver
  (doseq [[tag expected] [['doubles true] ['floats false]]]
    (let [array (with-meta 'samples {:raster.type/tag tag})
          direct (list 'clojure.core/aget array 'i)
          retained (retained-read array 'i)]
      (is (= expected ((var reverse/active-reads-match-dtype?) :double direct)))
      (is (= expected ((var reverse/active-reads-match-dtype?) :double retained))))))
