(ns raster.compiler.core.value-product-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.value-product :as product]))

(deftest literal-fields-are-not-type-or-ownership-inference
  (let [borrowed (double-array [2.0])
        fields (with-meta [nil false borrowed '[nested]] {:source/annotation :kept})]
    (is (identical? fields (product/literal-fields fields)))
    (is (identical? borrowed (nth (product/literal-fields fields) 2)))
    (is (= {:source/annotation :kept} (meta (product/literal-fields fields)))))
  (is (= [] (product/literal-fields [])))
  (is (nil? (product/literal-fields '(opaque-product))))
  (is (nil? (product/literal-fields nil))))

(deftest constant-projection-requires-canonical-closed-index-evidence
  (doseq [[expression index] [['(clojure.core/nth p 0) 0]
                              ['(clojure.core/nth p (clojure.core/long 1)) 1]]]
    (is (= {:source 'p :index index} (product/constant-projection expression 2))))
  (doseq [expression ['(nth p 0) '(fake/nth p 0) '(clojure.core/nth p -1)
                      '(clojure.core/nth p 2) '(clojure.core/nth p i)
                      '(clojure.core/nth p 0 :default)
                      '(clojure.core/nth p (opaque-index))
                      '(clojure.core/nth p (clojure.core/double 0))
                      (with-meta '(clojure.core/nth p 0) {:raster.op/lexical-callee true})
                      (with-meta '(clojure.core/nth p i) {:raster.type/tag 'double})
                      (with-meta '(.invk arbitrary p 0) {:raster.op/original 'clojure.core/nth})]]
    (is (nil? (product/constant-projection expression 2)) (pr-str expression)))
  (is (nil? (product/constant-projection '(clojure.core/nth p 0) 0))))

(deftest lexical-index-evidence-is-frozen-and-never-inferred-from-a-type
  (let [known {'zero 0 'one 1 'unknown 'zero 'typed {:raster.type/tag 'long}}
        lookup #(get known %)]
    (is (= 0 (product/index-value 'zero lookup)))
    (is (= 1 (product/index-value '(clojure.core/int one) lookup)))
    (is (= {:source 'p :index 1}
           (product/constant-projection '(clojure.core/nth p one) 2 lookup)))
    (doseq [expression ['unknown 'typed 'missing '(opaque zero)
                        '(do (side-effect) 0) '(clojure.core/int 2147483648)
                        '(clojure.core/long (clojure.core/double 0))
                        (with-meta '(clojure.core/long 0) {:raster.op/lexical-callee true})]]
      (is (nil? (product/index-value expression lookup))))
    (doseq [expression ['(clojure.core/nth p one :default) '(clojure.core/nth p 2)
                        (with-meta '(clojure.core/nth p zero)
                          {:raster.op/lexical-callee true})]]
      (is (nil? (product/constant-projection expression 2 lookup))))))
