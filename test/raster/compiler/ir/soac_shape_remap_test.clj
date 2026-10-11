(ns raster.compiler.ir.soac-shape-remap-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.soac-dialect :as dialect]))

(defn- shape-program [dimension]
  (dialect/make
   (dialect/default-program-facts
    {:values {'x (av/tensor {:dtype :double :shape [dimension]})
              'y (av/tensor {:dtype :long :shape []})
              'n (av/tensor {:dtype :long :shape []})}
     :inputs '[n x] :equations {0 (dialect/default-equation-facts)}})
   [(list '= 0 '[y]
          ;; A scalar capture may retain arbitrary backing shape; a pointwise map would
          ;; independently require x's shape to equal its traversal extent.
          (list 'scalar {:dtypes [:long]} '[n x]
                (dialect/lambda-form '[bound array] '[(clojure.core/alength array)])))] '[y]))

(defn- input-shape [program]
  (:shape (get-in (dialect/facts program) [:values 'x])))

(deftest compound-dimensions-use-expression-form-ssa-identities
  (let [dimension (with-meta '(clojure.core/inc n) {:raster.type/tag 'long})
        original (shape-program dimension)
        vectorized (dialect/remap-values original {'n [:extent 0]})
        restored (dialect/remap-values vectorized {[:extent 0] 'n})]
    (is (= '[(clojure.core/inc (value [:extent 0]))] (input-shape vectorized)))
    (is (= (meta dimension) (meta (first (input-shape vectorized)))))
    (is (= '[(clojure.core/inc n)] (input-shape restored)))
    (is (= :double (get-in (dialect/facts restored) [:values 'x :dtype])))
    (is (= (input-shape original) (input-shape restored)))
    (is (= [(first (input-shape vectorized))]
           (dialect/scan-result-shape {:mode :exclusive :extent [:extent 0]})))))

(deftest nested-dimensions-preserve-opaque-ids-and-single-step-renaming
  (let [original (shape-program '(clojure.core/* (clojure.core/inc count)
                                                 (extent n)))
        renamed (dialect/remap-values original {'count 'width 'width 'other 'n [:extent 0]})]
    (is (= '[(clojure.core/* (clojure.core/inc width) (extent [:extent 0]))]
           (input-shape renamed)))
    (is (= '[(clojure.core/* (clojure.core/inc count) (extent n))]
           (input-shape (dialect/remap-values renamed {'width 'count [:extent 0] 'n}))))))

(deftest exclusive-result-shapes-share-the-extent-representation
  (is (= [4] (dialect/scan-result-shape {:mode :exclusive :extent 3})))
  (is (= '[(clojure.core/inc n)]
         (dialect/scan-result-shape {:mode :exclusive :extent 'n})))
  (is (= '[(clojure.core/inc (value [:binding n]))]
         (dialect/scan-result-shape {:mode :exclusive :extent [:binding 'n]})))
  (is (= '[(value [:binding n])]
         (dialect/scan-result-shape {:mode :inclusive :extent [:binding 'n]}))))
