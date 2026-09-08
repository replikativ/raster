(ns raster.compiler.core.intel-block-io-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.intel-block-io :as block-io]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.kernel-precondition :as precondition]))

(deftest surface-boundaries-are-independent-of-allocation
  (doseq [dims [[1 32 32] [16777216 8388608 8388608] [17 40 48]]]
    (is (nil? (apply block-io/static-failure dims))))
  (doseq [dims [[0 32 32] [16777217 32 32] [1 16 32] [1 32 16]
                [1 8388616 32] [1 32 8388624] [1 33 32] [1 32 40]]]
    (is (some? (apply block-io/static-failure dims))))
  (is (nil? (block-io/static-failure 'm 'n 'k))))

(deftest selector-and-binding-share-the-same-conditions
  (let [conditions (block-io/matrix-preconditions 'm 'n 'k [(launch/product 'm 'k)])
        cases (block-io/fallback-cases conditions :portable)]
    (doseq [values [{'m 1 'n 32 'k 32} {'m 3 'n 40 'k 48}
                    {'m 4 'n 40 'k 48} {'m 4 'n 16 'k 32}]]
      (let [binding-ok? (try (precondition/check! conditions values)
                             (catch clojure.lang.ExceptionInfo _ false))
            fallback? (boolean
                       (some (fn [{:keys [expression op value]}]
                               (precondition/compare-value?
                                op (launch/resolve-expression values expression) value)) cases))]
        (is (= binding-ok? (not fallback?)))))))
