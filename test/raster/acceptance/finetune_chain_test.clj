(ns raster.acceptance.finetune-chain-test
  (:require [clojure.java.io :as io]
            [clojure.java.shell :as shell]
            [clojure.test :refer [deftest is]]
            [raster.acceptance.finetune-chain :as acceptance]
            [raster.ad.reverse :as reverse]
            [raster.dl.loss :as loss]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.value :as value]))

(deftest source-check-compares-the-pinned-file-before-evaluation
  (let [file (java.io.File/createTempFile "raster-external-source-" ".clj")
        text "(ns example)\n(def x 1)\n"
        calls (atom [])]
    (try
      (with-redefs [io/file (fn [& _] file)
                    shell/sh (fn [& args]
                               (swap! calls conj args)
                               {:exit 0 :out text :err ""})
                    clojure.core/slurp (fn [& _] text)]
        (is (= text (#'acceptance/checked-source "checkout with spaces"
                    {:path "src/example.clj"})))
        (is (= [["git" "--no-replace-objects" "-C" "checkout with spaces" "show"
                 (str acceptance/source-revision ":src/example.clj")]] @calls)))
      (with-redefs [io/file (fn [& _] file)
                    shell/sh (fn [& _] {:exit 0 :out text :err ""})
                    clojure.core/slurp (fn [& _] "modified")]
        (is (= :external-training-source-drift
               (try (#'acceptance/checked-source "checkout" {:path "src/example.clj"}) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
      (with-redefs [shell/sh (fn [& _] {:exit 1 :out "" :err "missing object"})]
        (is (= :external-training-source
               (try (#'acceptance/checked-source "checkout" {:path "src/example.clj"}) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
      (finally (.delete file)))))

(deftest declaration-selection-retains-exact-forms-and-rejects-incomplete-sets
  (let [text "(ns example) (defn kept [x] x) (defn retired [] :old-adapter)"
        spec {:namespace 'example :definitions '#{kept}}]
    (is (= '[(ns example) (defn kept [x] x)] (#'acceptance/selected-forms text spec)))
    (doseq [bad ["(ns wrong) (defn kept [x] x)"
                 "(ns example) (defn retired [] :old)"
                 "(ns example) (defn kept [x] x) (defn kept [x] x)"]]
      (is (= :external-training-declarations
             (try (#'acceptance/selected-forms bad spec) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest declaration-reader-does-not-evaluate-reader-forms
  (is (thrown? RuntimeException
               (#'acceptance/selected-forms
                "(ns example) #=(throw (Exception. \"reader executed\")) (def kept 1)"
                {:namespace 'example :definitions '#{kept}}))))

(deftest complete-source-preflight-precedes-any-evaluation
  (let [evaluations (atom [])
        train-spec (first @#'acceptance/source-specs)]
    (with-redefs-fn {#'acceptance/checked-source
                    (fn [_ spec]
                      (if (= (:path train-spec) (:path spec))
                        "first file"
                        (throw (ex-info "second source drift" {:reason :source-drift}))))
                    #'acceptance/selected-forms (fn [_ _] '[(ns example)])
                    #'clojure.core/eval (fn [form] (swap! evaluations conj form))}
      #(is (= :source-drift
              (try (acceptance/load-sources! "checkout") nil
                   (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (is (empty? @evaluations))))

(deftest oracle-comparison-rejects-truncated-arrays
  (let [called? (atom false)
        oracle {'worst-rel (fn [& _] (reset! called? true) 0.0)}]
    (is (= :external-training-shape
           (try (#'acceptance/checked-error oracle (float-array 2) (float-array 1)) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (is (false? @called?))
    (is (zero? (#'acceptance/checked-error oracle (float-array 2) (float-array 2))))))

(deftest existing-external-namespace-is-not-overwritten
  (let [evaluations (atom [])]
    (with-redefs-fn {#'acceptance/checked-source (fn [& _] "checked")
                    #'acceptance/selected-forms (fn [& _] '[(ns example)])
                    #'clojure.core/find-ns (fn [_] :already-loaded)
                    #'clojure.core/eval (fn [form] (swap! evaluations conj form))}
      #(is (= :external-training-namespace
              (try (acceptance/load-sources! "checkout") nil
                   (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    (is (empty? @evaluations))))

(deftest runner-requires-explicit-source-and-unique-supported-targets
  (doseq [options [{} {:source-root "checkout" :targets []}
                   {:source-root "checkout" :targets [:ocl:0 :ocl:0]}
                   {:source-root "checkout" :targets [:cuda:0]}]]
    (is (= :external-training-options
           (try (acceptance/run! options) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))

(deftest explicit-cases-reject-invalid-dimensions-before-preparation
  (let [valid {:cfg {:bs 1 :seq 2 :d 3} :weights [{} {}] :adapters [{} {}]
               :input (float-array 6) :target (float-array 6) :replay-count 1}]
    (doseq [bad [(assoc valid :replay-count 0)
                 (assoc-in valid [:cfg :bs] 2)
                 (assoc-in valid [:cfg :seq] nil)
                 (assoc valid :cfg {:bs 1 :seq -2 :d -3})
                 (assoc valid :cfg {:bs 1 :seq Long/MAX_VALUE :d Long/MAX_VALUE})
                 (assoc valid :weights '({} {}))
                 (assoc valid :adapters [nil nil])
                 (assoc valid :weights [{}])
                 (assoc valid :target (float-array 5))]]
      (is (= :external-training-case
             (try (acceptance/run-loaded! {} :ocl:0 bad) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))

(deftest mismatch-diagnostics-retain-error-magnitude-without-changing-admission
  (let [summary (#'acceptance/difference-summary (float-array [1 4]) (float-array [1 2]))]
    (is (= 2.0 (:max-absolute summary)))
    (is (= 2.0 (:reference-max summary)))
    (is (= 2.0 (:error-l2 summary)))
    (is (= (Math/sqrt 5.0) (:reference-l2 summary)))))

(deftest difference-diagnostics-retain-coordinates-and-do-not-hide-nonfinite-values
  (let [summary (#'acceptance/difference-summary [3.0 -2.0 6.0] [1.0 0.0 5.0])]
    (is (= {:index 0 :actual 3.0 :expected 1.0 :absolute-error 2.0}
           (:worst-finite-absolute-coordinate summary)))
    (is (zero? (:nonfinite-coordinate-count summary))))
  (let [summary (#'acceptance/difference-summary [Double/NaN 3.0 Double/POSITIVE_INFINITY]
                                                 [0.0 1.0 Double/POSITIVE_INFINITY])]
    (is (= 2 (:nonfinite-coordinate-count summary)))
    (is (= 1 (get-in summary [:worst-finite-absolute-coordinate :index])))
    (is (Double/isNaN (:max-absolute summary))))
  (is (nil? (:worst-finite-absolute-coordinate (#'acceptance/difference-summary [] []))))
  (is (= :external-training-shape
         (try (#'acceptance/difference-summary [1.0] []) nil
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))

(deftest bound-schedules-survive-parity-failure-without-leaking-the-artifact
  (let [keys (mapv #(keyword (str "adapter" %)) (range 14))
        gradients (zipmap keys (repeat (float-array [1])))
        model {'adapter-keys (atom keys) 'layer-theta (fn [& _] {})}
        oracle {'layer-arrs (fn [& _] []) 'ref-loss2 (fn [& _] 0.0)
                'adapter-pos (atom (zipmap keys (range)))
                'recovered-grads (fn [& _] gradients) 'worst-rel (fn [& _] 0.0)}
        case {:cfg {:bs 1 :seq 1 :d 1} :weights [{} {}]
              :adapters [gradients gradients] :input (float-array [0])
              :target (float-array [0]) :replay-count 1}
        outputs (into {:prediction (float-array [0]) :dx0 (float-array [1])}
                      (for [layer [0 1] key keys] [[layer key] (gradients key)]))
        live (fn [_] outputs)
        closed (atom [])
        inspected (atom [])
        info [{:instance :matrix
               :executable {:selection :fixed :entry-points ["actual_leaf"]}}]
        expected (into [0.0 (float-array [1])] (repeat 54 (float-array [1])))
        predicted (atom 0.0)]
    (with-redefs-fn
      {#'acceptance/prepare-chain (fn [& _] :prepared)
       #'compiled/instantiate! (fn [_] live)
       #'compiled/plan (fn [_] {:instances []})
       #'compiled/execution-info (fn [artifact] (swap! inspected conj artifact) info)
       #'compiled/close! (fn [artifact] (swap! closed conj artifact))
       #'reverse/value+grad (fn [_] (fn [& _] expected))
       #'value/->host identity
       #'loss/mse-loss (fn [& _] @predicted)}
      (fn []
        (let [result (acceptance/run-loaded! {:train model :oracle oracle} :ocl:0 case)]
          (is (= info (:bound-schedules result)))
          (is (= 1 (count (:replays result)))))
        (reset! predicted 1.0)
        (let [data (try (acceptance/run-loaded! {:train model :oracle oracle} :ocl:0 case)
                        nil (catch clojure.lang.ExceptionInfo e (ex-data e)))]
          (is (= :external-training-parity (:reason data)))
          (is (= info (:bound-schedules data)))
          (is (= 1.0 (:loss-error data)))
          (is (= 28 (count (:adapter-errors data)))))))
    (is (= [live live] @inspected))
    (is (= [live live] @closed))
    (reset! closed [])
    (with-redefs-fn
      {#'acceptance/prepare-chain (fn [& _] :prepared)
       #'compiled/instantiate! (fn [_] live)
       #'compiled/execution-info (fn [_] (throw (ex-info "inspection failed" {:reason :inspection})))
       #'compiled/close! (fn [artifact] (swap! closed conj artifact))}
      #(is (= :inspection
              (try (acceptance/run-loaded! {:train model :oracle oracle} :ocl:0 case)
                   nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (is (= [live] @closed))))
