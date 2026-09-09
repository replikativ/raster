(ns raster.compiler.passes.parallel.nonpacked-integral-stage-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.staged-contraction-fixtures :as packed]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.fixtures.staged-contracts :as fixtures]
            [raster.compiler.passes.parallel.staged-scalar-body :as staged]
            [raster.gpu.device-probe :as probe]
            [raster.gpu.link :as link]))

(deftest nonpacked-inner-folds-require-verified-prefix-bounds
  (doseq [width [1 3 5]]
    (is (some? (staged/analyze! (packed/packed-facts 2 3 2 width)))))
  (let [failure (try (staged/analyze! (packed/packed-facts 1 1 1 131072)) nil
                     (catch clojure.lang.ExceptionInfo e (ex-data e)))]
    (is (= :kernel-body-proof (:missing-rule failure)))))

(deftest public-decoded-byte-fold-matches-integer-reference
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "nonpacked decoded byte staged reduction")
    (doseq [[function decode] [[#'fixtures/decoded-byte-three-wide! #(- (long %) 7)]
                              [#'fixtures/widened-byte-three-wide! long]]]
     (let [a (byte-array [-128 127 0 11 -13 5 17 -128 3 127 -1 7])
          b (byte-array [127 -128 9 0 11 -4 -17 127 1 -128 7 3])
          scale (float 0.13)
          expected (vec (for [i (range 2)]
                          (reduce (fn [sum blk]
                                    (let [dot (reduce + 0
                                                      (for [t (range 3)
                                                            :let [k (+ (* i 6) (* blk 3) t)]]
                                                        (* (decode (aget a k))
                                                           (long (aget b k)))))]
                                      (float (+ sum (float (* (float dot) scale))))))
                                  (float 0) (range 2))))
          out (float-array (repeat 2 Float/NaN))
          compiled (equation-first/compile function
                                          {:target :ocl:0 :dtype :float})
          plan (equation-first/lower compiled [a b out scale])
          output-id (some (fn [[id node]] (when (identical? out (:source node)) id)) (:nodes plan))
          executable (link/instantiate! plan)]
      (try
        (is (= :none (get-in compiled [:stats :fallback])))
        (link/run! executable)
        (is (= expected (vec (link/download executable output-id))))
        (finally (link/close! executable)))))))
