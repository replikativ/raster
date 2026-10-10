(ns raster.acceptance.finetune-chain-test
  (:require [clojure.java.io :as io]
            [clojure.java.shell :as shell]
            [clojure.test :refer [deftest is]]
            [raster.acceptance.finetune-chain :as acceptance]))

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
