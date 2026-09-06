(ns raster.compiler.core.auto-specialization-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.specialize :as specialize])
  (:import [java.util.concurrent CountDownLatch TimeUnit]))

(defn subject [x] x)
(defn compiled-subject [x] [:compiled x])

(deftest admission-limit-preserves-the-generic-dispatch-result
  (let [entry (dispatch/make-method-entry ['objects] (fn [xs] [:generic (vec xs)]))
        cache (atom {})]
    (binding [dispatch/*auto-specialization-limit* 0]
      (with-redefs-fn {#'dispatch/get-specialization-cache (fn [_] cache)
                      #'clojure.core/future-call (fn [_] (throw (ex-info "unexpected compile" {})))}
        (fn []
          (is (= [:generic ["value"]]
                 (#'dispatch/dispatch-arity "subject" [entry]
                  (object-array ["value"]) nil nil nil nil 1 (atom {1 [entry]}) #'subject)))
          (is (empty? @cache)))))))

(deftest automatic-specialization-admission-is-bounded-and-single-flight
  (let [cache (atom {}) tasks (atom []) calls (atom 0)]
    (binding [dispatch/*auto-specialization-limit* 2]
      (with-redefs [clojure.core/future-call (fn [task] (swap! tasks conj task) nil)
                    specialize/specialize-fn! (fn [& _]
                                               (swap! calls inc)
                                               'compiled-subject)]
        (doseq [cls [String String Long Long Double]]
          (is (nil? (#'dispatch/try-auto-specialize! #'subject cls cache))))
        (is (= 2 (count @tasks)))
        (is (= #{String Long} (set (keys @cache))))
        (run! (fn [task] (task)) @tasks)
        (is (= 2 @calls))
        (is (= [:compiled :value]
               ((#'dispatch/try-auto-specialize! #'subject String cache) :value)))
        (is (nil? (#'dispatch/try-auto-specialize! #'subject Double cache)))
        (is (= 2 (count @tasks)))))))

(deftest rejected-and-failed-attempts-cannot-grow-the-cache
  (let [cache (atom {}) tasks (atom [])]
    (binding [dispatch/*auto-specialization-limit* 0]
      (is (nil? (#'dispatch/claim-specialization! cache String)))
      (is (empty? @cache)))
    (binding [dispatch/*auto-specialization-limit* 1
              *err* (java.io.StringWriter.)]
      (with-redefs [clojure.core/future-call (fn [task] (swap! tasks conj task) nil)
                    specialize/specialize-fn! (fn [& _] (throw (ex-info "fixture failure" {})))]
        (#'dispatch/try-auto-specialize! #'subject String cache)
        ((first @tasks))
        (is (= {:fn nil :pending? false} (get @cache String)))
        (#'dispatch/try-auto-specialize! #'subject String cache)
        (#'dispatch/try-auto-specialize! #'subject Long cache)
        (is (= 1 (count @tasks)))
        (is (= 1 (count @cache)))))
    (doseq [limit [-1 1.5 nil "2"]]
      (binding [dispatch/*auto-specialization-limit* limit]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"nonnegative integer"
                             (#'dispatch/claim-specialization! (atom {}) String)))))))

(deftest clearing-a-cache-revokes-an-old-completion
  (let [cache (atom {})
        old (#'dispatch/claim-specialization! cache String)]
    (reset! cache {})
    (let [current (#'dispatch/claim-specialization! cache String)]
      (#'dispatch/complete-specialization! cache String old :stale)
      (is (identical? current (get-in @cache [String :claim])))
      (#'dispatch/complete-specialization! cache String current :current)
      (is (= {:fn :current :pending? false} (get @cache String))))
    (reset! cache {})
    (#'dispatch/complete-specialization! cache String old :stale)
    (is (empty? @cache))))

(deftest concurrent-admission-has-exactly-one-owner
  ;; Only admission is raced; no compiler, namespace mutation or device work occurs here.
  (let [cache (atom {}) start (CountDownLatch. 1)
        workers (mapv (fn [_]
                        (future
                          (when (.await start 2 TimeUnit/SECONDS)
                            (#'dispatch/claim-specialization! cache String))))
                      (range 4))]
    (try
      (.countDown start)
      (let [claims (mapv #(deref % 2000 ::timeout) workers)]
        (is (not-any? #{::timeout} claims))
        (is (= 1 (count (remove nil? claims))))
        (is (= 1 (count @cache))))
      (finally (run! future-cancel workers)))))
