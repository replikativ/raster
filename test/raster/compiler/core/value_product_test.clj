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
