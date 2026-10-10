(ns raster.gpu.donation-token-test
  "Single-use transfer tests with fake storage; no native allocation or device."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.gpu.value :as value])
  (:import [java.lang.ref Cleaner$Cleanable]
           [java.util.concurrent.atomic AtomicBoolean]))

(defn- input []
  (value/wrap-external {:n-elements 8 :dtype :float} :ze:0 :float [8]))

(deftest only-original-issued-token-can-transfer
  (let [token (value/consume! (input))
        calls (atom 0)
        replacements [(assoc token :owner (:owner token) :copy true)
                      (assoc token :claimed (AtomicBoolean. false))
                      (value/map->DonatedBuffer (into {} token))
                      (assoc token :claim-state (fn [& _] (swap! calls inc)
                                                 (AtomicBoolean. false)))]]
    (is (value/donated-buffer? token))
    (doseq [copy replacements]
      (is (not (value/donated-buffer? copy)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"token returned by consume"
                           (value/donate-output copy :float [8]))))
    (is (zero? @calls) "unissued state closures are never invoked")
    (is (value/live? (value/donate-output token :float [8])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"already claimed"
                         (value/donate-output token :float [8])))))

(deftest claim-state-cannot-be-reset-through-record-data
  (let [token (value/consume! (input))
        output (value/donate-output token :float [8])]
    (is (nil? (:claimed token)))
    (is (nil? ((:claim-state token) (Object.))))
    (is (value/live? output))
    (is (thrown? clojure.lang.ExceptionInfo
                 (value/donate-output (assoc token :claimed (AtomicBoolean. false)) :float [8])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"already claimed"
                         (value/donate-output token :float [8])))))

(deftest failed-output-validation-does-not-claim
  (let [token (value/consume! (input))]
    (is (thrown? clojure.lang.ExceptionInfo (value/donate-output token :double [8])))
    (is (thrown? clojure.lang.ExceptionInfo (value/donate-output token :float [9])))
    (is (value/live? (value/donate-output token :float [4])))))

(deftest concurrent-claims-produce-exactly-one-output
  (let [token (value/consume! (input))
        start (promise)
        attempts (doall (repeatedly 2 #(future @start
                                              (try (value/donate-output token :float [8])
                                                   (catch clojure.lang.ExceptionInfo e e)))))]
    (deliver start true)
    (let [results (mapv #(deref % 5000 ::timeout) attempts)]
      (is (= 1 (count (filter value/device-array? results))))
      (is (= 1 (count (filter #(instance? clojure.lang.ExceptionInfo %) results)))))))

(deftest transfer-invalidates-the-entire-alias-chain
  (let [source (input)
        alias (value/alias-of source {:shape [4]})
        nested (value/alias-of alias {:shape [2]})]
    (is (value/live? nested))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"cannot transfer"
                         (value/consume! nested)))
    (let [token (value/consume! source)]
      (is (not (value/live? alias)))
      (is (not (value/live? nested)))
      (is (value/live? (value/donate-output token :float [8]))))))

(deftest cleaner-and-claim-share-private-exactly-once-state
  (let [arm-var (ns-resolve 'raster.gpu.value 'arm-cleaner!)
        runtime-var (ns-resolve 'raster.gpu.value 'rt-fn)
        original (var-get arm-var)
        registrations (atom [])
        frees (atom [])]
    (with-redefs-fn
      {arm-var (fn [holder state buffer device owned?]
                 (let [handle (original holder state buffer device owned?)]
                   (swap! registrations conj {:holder holder :handle handle})
                   handle))
       runtime-var (fn [_ operation]
                     (is (= "free-buffer!" operation))
                     #(swap! frees conj %))}
      (fn []
        (doseq [claim? [false true]]
          (reset! registrations [])
          (reset! frees [])
          (let [buffer {:n-elements 8 :dtype :float}
                source (value/wrap-owned buffer :ze:0 :float [8])
                token (value/consume! source)
                token-registration (second @registrations)
                output (when claim? (value/donate-output token :float [8]))]
            (is (identical? token (:holder token-registration))
                "Cleaner tracks the final sealed token, not its pre-seal record")
            (.clean ^Cleaner$Cleanable (:handle (first @registrations)))
            (is (empty? @frees) "consumed source cannot reclaim transferred storage")
            (.clean ^Cleaner$Cleanable (:handle token-registration))
            (is (= (if claim? [] [buffer]) @frees))
            (when output
              (is (value/live? output))
              (.clean ^Cleaner$Cleanable (:handle (last @registrations))))
            (.clean ^Cleaner$Cleanable (:handle token-registration))
            (is (= [buffer] @frees) "one reclaim across source, token and output")
            (is (thrown-with-msg? clojure.lang.ExceptionInfo #"already claimed"
                                 (value/donate-output token :float [8])))))))))
