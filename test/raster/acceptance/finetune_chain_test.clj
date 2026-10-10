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

(deftest coordinate-diagnostics-are-total-and-do-not-truncate-or-hide-nonfinite-inputs
  (let [summary (#'acceptance/difference-summary [3.0 -2.0 6.0] [1.0 0.0 5.0])]
    (is (= 3 (:length summary)))
    (is (= 2.0 (:max-absolute-error-of-finite-inputs summary)))
    (is (= {:index 0 :actual 3.0 :expected 1.0 :absolute-error 2.0}
           (:worst-finite-input-coordinate summary)))
    (is (zero? (:nonfinite-coordinate-count summary))))
  (let [summary (#'acceptance/difference-summary
                 [Double/NaN 3.0 Double/POSITIVE_INFINITY]
                 [0.0 1.0 Double/POSITIVE_INFINITY])]
    (is (= 2 (:nonfinite-coordinate-count summary)))
    (is (= 0 (get-in summary [:first-nonfinite-coordinate :index])))
    (is (Double/isNaN (get-in summary [:first-nonfinite-coordinate :actual])))
    (is (= 1 (get-in summary [:worst-finite-input-coordinate :index]))))
  (let [summary (#'acceptance/difference-summary [Double/MAX_VALUE] [(- Double/MAX_VALUE)])]
    (is (= Double/POSITIVE_INFINITY (:max-absolute-error-of-finite-inputs summary)))
    (is (= Double/POSITIVE_INFINITY
           (get-in summary [:worst-finite-input-coordinate :absolute-error])))
    (is (zero? (:nonfinite-coordinate-count summary))))
  (let [summary (#'acceptance/difference-summary [] [])]
    (is (zero? (:length summary)))
    (is (nil? (:worst-finite-input-coordinate summary)))
    (is (nil? (:first-nonfinite-coordinate summary))))
  (is (= :external-training-shape
         (try (#'acceptance/difference-summary [1.0] []) nil
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))

(deftest two-update-runner-retains-diagnostics-without-changing-its-default-case
  (let [keys (mapv #(keyword (str "adapter" %)) (range 14))
        gradients (zipmap keys (repeat (float-array [1])))
        seeds (atom [])
        model {'adapter-keys (atom keys) 'layer-theta (fn [& _] {})
               'rand-weights (fn [_ seed] (swap! seeds conj [:weight seed]) {})
               'init-adapters (fn [_ seed scale]
                                (swap! seeds conj [:adapter seed scale]) gradients)}
        relative-error (atom 0.0)
        oracle {'chain-cfg (atom {:bs 1 :seq 1 :d 1})
                'fvec (fn [n seed scale]
                        (swap! seeds conj [:vector n seed scale]) (float-array n))
                'layer-arrs (fn [& _] []) 'ref-loss2 (fn [& _] 0.0)
                'adapter-pos (atom (zipmap keys (range)))
                'recovered-grads (fn [& _] gradients)
                'worst-rel (fn [actual expected]
                             (if (identical? actual (gradients (keys 0)))
                               @relative-error 0.0))}
        outputs (into {:prediction (float-array [0]) :dx0 (float-array [1])}
                      (for [layer [0 1] key keys] [[layer key] (gradients key)]))
        calls (atom 0)
        live (fn [_] (swap! calls inc) outputs)
        closed (atom []) inspected (atom [])
        info [{:instance :matrix :executable {:entry-points ["bound_leaf"]}}]
        expected (into [0.0 (float-array [1])] (repeat 54 (float-array [1])))
        predicted (atom 0.0)
        loaded {:train model :oracle oracle}
        replacements {#'acceptance/prepare-chain (fn [& _] :prepared)
                      #'compiled/instantiate! (fn [_] live)
                      #'compiled/plan (fn [_] {:instances []})
                      #'compiled/execution-info (fn [artifact]
                                                 (swap! inspected conj artifact) info)
                      #'compiled/close! (fn [artifact] (swap! closed conj artifact))
                      #'reverse/value+grad (fn [_] (fn [& _] expected))
                      #'value/->host identity #'value/live? (constantly false)
                      #'loss/mse-loss (fn [& _] @predicted)}]
    (with-redefs-fn replacements
      (fn []
        (let [result (acceptance/run-loaded! loaded :ocl:0)]
          (is (= info (:bound-schedules result)))
          (is (= 2 (count (:replays result))))
          (is (= 2 @calls))
          (is (= [[:weight 11] [:weight 12] [:adapter 21 0.02] [:adapter 22 0.02]
                  [:vector 1 31 0.5] [:vector 1 32 0.5]] @seeds)))
        (reset! predicted 1.0)
        (let [data (try (acceptance/run-loaded! loaded :ocl:0) nil
                        (catch clojure.lang.ExceptionInfo e (ex-data e)))]
          (is (= :external-training-parity (:reason data)))
          (is (= info (:bound-schedules data)))
          (is (= 1.0 (:predicted-loss data)))
          (is (zero? (:reference-loss data)))
          (is (= 28 (count (:adapter-errors data))))
          (is (= 28 (count (:adapter-diagnostics data))))
          (is (= 1 (get-in data [:dx-diagnostics :length]))))
        (reset! predicted 0.0)
        (reset! relative-error 0.02)
        (let [data (try (acceptance/run-loaded! loaded :ocl:0) nil
                        (catch clojure.lang.ExceptionInfo e (ex-data e)))]
          (is (= :external-training-parity (:reason data)))
          (is (= 0.02 (first (:adapter-errors data))))
          (is (= 0.02 (get-in data [:adapter-diagnostics 0 :oracle-relative-error])))
          (is (= 28 (count (:adapter-errors data)))))))
    (is (= [live live live] @inspected))
    (is (= [live live live] @closed))
    (reset! closed [])
    (let [before @calls]
      (with-redefs-fn
        (assoc replacements #'compiled/execution-info
               (fn [_] (throw (ex-info "inspection failed" {:reason :inspection}))))
        #(is (= :inspection
                (try (acceptance/run-loaded! loaded :ocl:0) nil
                     (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
      (is (= before @calls))
      (is (= [live] @closed)))))
