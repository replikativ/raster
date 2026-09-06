(ns raster.compiler.coverage-corpus-test
  "Typed-route coverage ratchet over the deftm corpus.

   Runs wherever an OpenCL device is present (the CI CPU OpenCL gate, or a local GPU). The
   committed baseline lists every corpus var's route and declines; this test fails when a var
   leaves the typed route or starts failing, and prints the new declines so the regression is
   named at the PR that introduced it. Refresh the baseline with
   `scripts/update-coverage-baseline.sh` after an intended change."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.coverage :as coverage]
            [raster.compiler.pipeline :as pipeline]
            [raster.gpu.device-probe :as device-probe]))

(deftest corpus-does-not-leave-the-typed-route
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "typed-route coverage corpus")
    (let [baseline (coverage/read-baseline coverage/default-baseline-path)
          report (coverage/corpus-report {:target-device :ocl:0 :dtype :float})
          violations (coverage/ratchet-violations baseline report)]
      (println "  [coverage] summary:" (pr-str (:summary report)))
      (println "  [coverage] emitted artifacts:" (pr-str (:emission-summary report)))
      (testing "every var that took the typed route still does, and no var started failing"
        (is (empty? violations)
            (with-out-str
              (doseq [v violations]
                (println "  coverage regression:" (pr-str v))))))
      (testing "the corpus compiled for a real device"
        (is (pos? (:total (:summary report))))))))

(deftest frontend-coverage-retains-independent-emission-evidence
  (let [calls (atom 0)
        compiled {:backend :opencl
                  :soac-fused-stats {:route :typed-soac :typed-validated true}
                  :kernels [{:target :opencl-c
                             :attributes {:emission-route :verified-segmap-opencl
                                          :kernel-body-decline {:reason :unsupported-loop
                                                                :missing-rule :ordered-loop}}}]}]
    (with-redefs [pipeline/show-pipeline (fn [& _] (swap! calls inc) compiled)]
      (let [row (coverage/report-var #'coverage/report-var {:target-device :ocl:0})]
        (is (= 1 @calls) "emission diagnostics must not trigger another compilation")
        (is (= :typed-soac (:route row)))
        (is (= {:verified-segmap-opencl 1} (get-in row [:emission :routes])))
        (is (= :unsupported-loop (get-in row [:emission :declines 0 :reason])))
        (is (= 1 (:emission-declines row)))))))

(deftest emitted-artifact-summary-does-not-change-the-portable-ratchet
  (let [rows [{:var 'a :route :typed-soac :typed-validated true :declines []
               :emission-declines 0 :emission {:routes {:kernel-body 2} :declines []}}
              {:var 'b :route :typed-soac :typed-validated true :declines []
               :emission-declines 1 :emission {:routes {:verified-segmap-opencl 1}
                                             :declines [{:reason :unsupported-loop}]}}
              {:var 'c :route :error :error :unsupported}]]
    (with-redefs [coverage/corpus-vars (fn [_] rows)
                  coverage/report-var (fn [row _] row)]
      (let [report (coverage/corpus-report [] {:target-device :ocl:0})
            baseline (coverage/baseline-facts report)]
        (is (= {:artifact-routes {:kernel-body 2 :verified-segmap-opencl 1}
                :programs-with-declines 1} (:emission-summary report)))
        (is (= {:total 3 :typed-soac 2 :error 1} (:summary report)))
        (is (not (contains? baseline :emission-summary)))
        (is (every? #(not-any? (set (keys %)) [:emission :emission-declines]) (:vars baseline)))
        (is (empty? (coverage/ratchet-violations baseline report)))))))
