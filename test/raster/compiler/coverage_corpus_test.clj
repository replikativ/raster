(ns raster.compiler.coverage-corpus-test
  "Typed-route coverage ratchet over the deftm corpus.

   Runs wherever an OpenCL device is present (the CI CPU OpenCL gate, or a local GPU). The
   committed baseline lists every corpus var's route and declines; this test fails when a var
   leaves the typed route or starts failing, and prints the new declines so the regression is
   named at the PR that introduced it. Refresh the baseline with
   `scripts/update-coverage-baseline.sh` after an intended change."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.coverage :as coverage]
            [raster.dl.attention :as attention]
            [raster.dl.nn :as dl-nn]
            [raster.compiler.pipeline :as pipeline]
            [raster.nn :as nn]
            [raster.ode.pde :as pde]
            [raster.gpu.device-probe :as device-probe]))

(deftest corpus-does-not-leave-the-typed-route
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "typed-route coverage corpus")
    (let [baseline (coverage/read-baseline coverage/default-baseline-path)
          report (coverage/corpus-report {:target-device :ocl:0 :dtype :float})
          violations (coverage/ratchet-violations baseline report)
          report-path (coverage/write-report! coverage/default-current-report-path report)]
      (println "  [coverage] summary:" (pr-str (:summary report)))
      (println "  [coverage] emitted artifacts:" (pr-str (:emission-summary report)))
      (println "  [coverage] full report:" report-path)
      (doseq [row (coverage/residual-rows report)]
        (println "  [coverage] residual:"
                 (pr-str (select-keys row [:var :route :error :declines]))))
      (testing "every var that took the typed route still does, and no var started failing"
        (is (empty? violations)
            (with-out-str
              (doseq [v violations]
                (println "  coverage regression:" (pr-str v))))))
      (testing "the corpus compiled for a real device"
        (is (pos? (:total (:summary report)))))
      (testing "migrated graph and strided-norm workloads remain entirely KernelBody-emitted"
        (let [rows (into {} (map (juxt :var identity)) (:vars report))]
          (doseq [[workload expected-kernels]
                  {'raster.dl.attention/graph-attention 5
                   'raster.dl.nn/rms-norm-chunked 3
                   'raster.dl.nn/rms-norm-chunked-backward-dx 3
                   'raster.dl.nn/generate-dropout-mask-seeded 1}]
            (is (= {:kernel-body expected-kernels}
                   (get-in rows [workload :emission :routes]))
                (str workload " regained a compatibility emitter"))))))))

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

(deftest coverage-uses-the-retained-dtype-of-an-unambiguous-deftm
  (is (= :double (#'coverage/effective-corpus-dtype
                  #'attention/scaled-dot-product-attn-jvp :float))
      "an unambiguous fixed-double deftm is measured at its real dtype")
  (is (= :float (#'coverage/effective-corpus-dtype
                 #'attention/scaled-dot-product-attn :float))
      "an overloaded parametric deftm still uses the requested corpus specialization"))

(deftest explicit-host-orchestration-is-not-probed-as-device-coverage
  (with-redefs [pipeline/show-pipeline
                (fn [& _] (throw (AssertionError. "host-only source reached GPU compilation")))]
    (doseq [v [#'nn/xavier-init! #'nn/kaiming-init!
               #'dl-nn/xavier-init #'dl-nn/he-init
               #'pde/solve-fixed-step]]
      (is (= {:route :host-only :host-contract :explicit}
             (select-keys (coverage/report-var v {:target-device :ocl:0})
                          [:route :host-contract]))))))

(deftest gpu-front-door-rejects-explicit-host-orchestration
  (is (= :gpu-compiler-host-only
         (:reason (ex-data (try
                             (pipeline/compile-gpu-program
                              #'pde/solve-fixed-step :ze:debug :dtype :double)
                             (catch clojure.lang.ExceptionInfo error error)))))))

(deftest unique-scatter-retains-independent-effect-ratchet-evidence
  (let [algorithm (list 'soac-program {}
                        [(list '= 0 ['result]
                               (list 'scatter {:index 'i :extent 'n :conflict :unique}
                                     [] [] '(lambda [] (region [] []))))]
                        ['result])
        compiled {:soac-fused {:dialect :typed-soac
                               :equations [{:algorithm algorithm}]}}]
    (is (= {:independent 1} (#'coverage/effect-order-facts compiled))
        "refining an independent effect-map to a proved unique scatter is not serialization")))

(deftest effect-order-ratchet-compares-only-established-parallelism
  (let [row {:var 'workload :route :typed-soac :typed-validated true :declines []}
        report (fn [effect-orders]
                 {:vars [(cond-> row effect-orders (assoc :effect-orders effect-orders))]})]
    (is (empty? (coverage/ratchet-violations {:vars [row]}
                                             (report {:sequential 1})))
        "newly admitted ordered control has no prior parallel schedule to regress")
    (is (= :effect-map-serialized
           (:violation
            (first (coverage/ratchet-violations
                    {:vars [(assoc row :effect-orders {:independent 1})]}
                    (report {:sequential 1})))))
        "an established independent effect remains a protected performance fact")))

(deftest host-only-is-honest-without-erasing-established-device-coverage
  (let [host-row {:var 'orchestrator :route :host-only :host-contract :explicit}
        typed-row {:var 'orchestrator :route :typed-soac :typed-validated true}]
    (is (empty? (coverage/ratchet-violations {:vars [host-row]} {:vars [host-row]})))
    (is (= :route-downgraded
           (:violation (first (coverage/ratchet-violations
                               {:vars [typed-row]} {:vars [host-row]})))))))

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

(deftest residual-rows-name-only-compatibility-and-errors
  (let [rows [{:var 'typed :route :typed-soac}
              {:var 'host :route :host-only}
              {:var 'compatible :route :compatibility :declines [{:reason :legacy}]}
              {:var 'scalar :route :scalar}
              {:var 'broken :route :error :error :unsupported}]]
    (is (= ['compatible 'broken]
           (mapv :var (coverage/residual-rows {:vars rows}))))))
