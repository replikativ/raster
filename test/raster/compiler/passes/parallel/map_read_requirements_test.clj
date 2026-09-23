(ns raster.compiler.passes.parallel.map-read-requirements-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.index-algebra :as algebra]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.map-read-requirements :as requirements]))

(defn- packed-head-map
  [coordinate]
  (segop/map->SegMap
   {:id :packed-heads
    :space (segop/make-seg-space 'idx 'total)
    :inputs #{'input} :outputs #{'output} :scalars '#{rows heads width total}
    :dtype :float :out-sym 'output
    :scalar-region
    {:locals
     '[{:id column :dtype :long :init (rem idx width)}
       {:id q :dtype :long :init (quot idx width)}
       {:id head :dtype :long :init (quot q rows)}
       {:id row :dtype :long :init (rem q rows)}]
     :result (list 'aget 'input coordinate)}}))

(deftest symbolic-packing-read-retains-its-exact-runtime-capacity
  (let [coordinate '(+ (* row (* heads width)) (+ (* head width) column))
        result (requirements/symbolic-read-requirements
                (packed-head-map coordinate)
                {:scalar-definitions {'total (launch/product 'rows 'heads 'width)}})]
    (is (= {:const 1 :factors '[heads rows width]}
           (algebra/monomial (get result 'input))))))

(deftest symbolic-read-proof-declines-padding-translation-and-indirection
  (doseq [coordinate ['(+ base (* row (* heads width)) (+ (* head width) column))
                      '(+ (* row stride) column)
                      '(aget indices idx)]]
    (is (nil? (requirements/symbolic-read-requirements
               (packed-head-map coordinate)
               {:scalar-definitions {'total (launch/product 'rows 'heads 'width)}})))))
