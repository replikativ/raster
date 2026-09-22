(ns raster.compiler.passes.parallel.abm-scan-route-test
  "Production-route coverage for the integer scans used by the firms workload."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.abm.firms.phases :as phases]
            [raster.compiler.ir.kernel-graph :as kernel-graph]
            [raster.compiler.pipeline :as pipeline]))

(deftest firms-integer-scans-compile-as-resident-kernel-graphs
  (doseq [phase-var [#'phases/csr-scan-par! #'phases/startup-scan-par!]]
    (testing (str (:name (meta phase-var)))
      (let [descriptor (pipeline/compile-gpu-program
                        phase-var :ze:debug :dtype :float :on-non-resident :nil)]
        (is (some? descriptor))
        (is (= 1 (count (:steps descriptor))))
        (is (kernel-graph/kernel-graph? (get-in descriptor [:steps 0 :artifact])))))))
