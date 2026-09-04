(ns raster.compiler.coverage-corpus-test
  "Typed-route coverage ratchet over the deftm corpus.

   Runs wherever an OpenCL device is present (the CI CPU OpenCL gate, or a local GPU). The
   committed baseline lists every corpus var's route and declines; this test fails when a var
   leaves the typed route or starts failing, and prints the new declines so the regression is
   named at the PR that introduced it. Refresh the baseline with
   `scripts/update-coverage-baseline.sh` after an intended change."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.coverage :as coverage]
            [raster.gpu.device-probe :as device-probe]))

(deftest corpus-does-not-leave-the-typed-route
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "typed-route coverage corpus")
    (let [baseline (coverage/read-baseline coverage/default-baseline-path)
          report (coverage/corpus-report {:target-device :ocl:0 :dtype :float})
          violations (coverage/ratchet-violations baseline report)]
      (println "  [coverage] summary:" (pr-str (:summary report)))
      (testing "every var that took the typed route still does, and no var started failing"
        (is (empty? violations)
            (with-out-str
              (doseq [v violations]
                (println "  coverage regression:" (pr-str v))))))
      (testing "the corpus compiled for a real device"
        (is (pos? (:total (:summary report))))))))
