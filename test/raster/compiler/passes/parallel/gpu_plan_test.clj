(ns raster.compiler.passes.parallel.gpu-plan-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.gpu-plan :as gpu-plan]
            [raster.compiler.passes.parallel.patterns :as patterns]))

(deftest detect-do-wrapped-transpose-pattern
  (testing "Shared pattern matcher detects do-wrapped transpose loops"
    (let [form '(do "transpose"
                    (dotimes [i rows]
                      (dotimes [j cols]
                        (clojure.core/aset out
                                           (+ (* j rows) i)
                                           (clojure.core/aget src (+ (* i cols) j)))))
                    out)
          match (patterns/match-do-wrapped-transpose form)]
      (is (= {:src 'src
              :out 'out
              :out-buf 'out
              :rows 'rows
              :cols 'cols}
             match)))))

(deftest rewrite-do-wrapped-transpose-to-gpu-kernel
  (testing "gpu-plan rewrites transpose loops to registered transpose kernels"
    (let [form '(let* [xt (do "transpose"
                              (dotimes [i rows]
                                (dotimes [j cols]
                                  (clojure.core/aset out
                                                     (+ (* j rows) i)
                                                     (clojure.core/aget src (+ (* i cols) j)))))
                              out)]
                      xt)
          {:keys [form kernels stats]} (gpu-plan/gpu-plan-pass form :dtype :float)
          bindings (second form)
          rewritten-expr (second bindings)]
      (is (= 1 (:transpose-rewrites stats)))
      (is (= 1 (count kernels)))
      (is (= :transpose (:type (first kernels))))
      (is (seq? rewritten-expr))
      (is (= 'raster.gpu.ze-runtime/invoke-registered-transpose!
             (first rewritten-expr)))
      (is (= 'src (nth rewritten-expr 2)))
      (is (= 'out (nth rewritten-expr 3)))
      (is (= 'rows (nth rewritten-expr 4)))
      (is (= 'cols (nth rewritten-expr 5))))))