(ns raster.compiler.passes.parallel.staged-result-conversion-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.fixtures.staged-contracts :as fixtures]
            [raster.gpu.device-probe :as probe]
            [raster.gpu.link :as link]))

(deftest output-conversion-does-not-widen-the-accumulation
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "staged accumulator/output conversion")
    (doseq [[function values expected alternative]
            [[#'fixtures/floating-stages-double-output! [16777216.0 1.0 -16777216.0]
              0.0 1.0]
             [#'fixtures/double-stages-float-identity-output! [16777216.0 1.0 0.0]
              16777216.0 16777217.0]]]
     (let [a (double-array values)
          out (double-array [Double/NaN])
          compiled (equation-first/compile function
                                           {:target :ocl:0 :dtype :double})
          plan (equation-first/lower compiled [a out])
          output-id (some (fn [[id node]] (when (identical? out (:source node)) id)) (:nodes plan))
          executable (link/instantiate! plan)]
      (try
        (is (not= expected alternative))
        (is (= alternative (reduce + a)) "dropping Float rounding would give a different answer")
        (is (= :none (get-in compiled [:stats :fallback])))
        (link/run! executable)
        (is (= [expected] (vec (link/download executable output-id))))
        (finally (link/close! executable)))))))
