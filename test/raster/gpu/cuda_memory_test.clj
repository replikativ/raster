(ns raster.gpu.cuda-memory-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.cuda-driver :as driver]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.cuda-prepared-execution :as execution]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-backend :as backend])
  (:import [java.lang.foreign Arena MemorySegment ValueLayout]))

(defn- error-of [f] (try (f) nil (catch Throwable e e)))

(defn- with-runtime [f]
  (with-open [arena (Arena/ofShared)]
   (let [calls (atom []) address (atom 4096)
        native (fn [operation args]
                 (swap! calls conj [operation args])
                 (case operation
                   :initialize {:context (MemorySegment/ofAddress 1) :stream (MemorySegment/ofAddress 2)
                                :device 0 :compute-capability [8 6]}
                   :allocate (swap! address + 4096)
                   :host-allocate (.allocate arena (long (first args)))
                   nil))]
    (binding [driver/*invoke* native]
      (with-redefs [runtime/state (atom {})]
        (f calls native))))))

(deftest manufactured-byte-plans-cannot-exceed-an-exact-slice
  (with-runtime
    (fn [calls _]
      (let [parent (runtime/make-buffer 16 :float)
            slice (runtime/slice-buffer parent 0 8 :float)
            plan {:buf-off 8 :host-off 0 :n-bytes 4
                  :host-seg (MemorySegment/ofArray (float-array [7.0]))}]
        (try
          (let [before @calls error (error-of #(runtime/execute-range! slice plan :upload))]
            (is (= :cuda-range-capacity (:reason (ex-data error))))
            (is (= before @calls) "reject before synchronization or native transfer"))
          (is (= :missing-cleanup-owner (:reason (ex-data (error-of #(runtime/free-buffer! slice))))))
          (finally (runtime/free-buffer! parent)))))))

(deftest synchronous-copy-keeps-memory-live-after-synchronization
  (with-runtime
    (fn [calls _]
      (let [buffer (runtime/make-buffer 4 :float)
            entered (promise) proceed (promise) original execution/synchronize!]
        (try
          (with-redefs [execution/synchronize!
                        (fn [& args]
                          (apply original args)
                          (deliver entered true)
                          (when (= :timeout (deref proceed 5000 :timeout))
                            (throw (ex-info "test barrier timed out" {}))))]
            (let [worker (future (runtime/upload-range! buffer (float-array [1 2 3 4]) {:elements 4}))]
              (try
                (is (= true (deref entered 5000 :timeout)))
                (let [error (error-of #(runtime/free-buffer! buffer))]
                  (is (= :owner-in-use (:reason (ex-data error))))
                  (is (= :live (:phase (cleanup/status (::cleanup/owner buffer))))))
                (is (not-any? #(= :free (first %)) @calls))
                (finally (deliver proceed true)))
              (is (identical? buffer (deref worker 5000 :timeout)))))
          (finally (runtime/free-buffer! buffer)))
        (let [ops (mapv first @calls)]
          (is (< (.indexOf ops :upload) (.indexOf ops :free))))))))

(deftest range-plan-validation-covers-alignment-direction-and-host-permissions
  (with-runtime
    (fn [calls _]
      (let [buffer (runtime/make-buffer 4 :float)
            plan {:buf-off 0 :host-off 0 :n-bytes 4
                  :host-seg (MemorySegment/ofArray (float-array [1 2 3 4]))}]
        (try
          (doseq [[p direction reason]
                  [[(assoc plan :buf-off 1) :upload :cuda-transfer-plan]
                   [(assoc plan :n-bytes -4) :upload :cuda-transfer-plan]
                   [(assoc plan :host-off Long/MAX_VALUE) :upload :cuda-transfer-plan]
                   [plan :invalid :cuda-transfer-plan]
                   [(assoc plan :host-seg (.asReadOnly ^MemorySegment (:host-seg plan))) :download
                    :cuda-transfer-host-segment]]]
            (let [before @calls error (error-of #(runtime/execute-range! buffer p direction))]
              (is (= reason (:reason (ex-data error))))
              (is (= before @calls))))
          (finally (runtime/free-buffer! buffer)))))))

(deftest resident-memory-foundation-does-not-enable-cuda-sessions
  (is (true? (:compile-only? (ex-data (error-of #(backend/backend-type :cuda:0)))))))

(deftest nonzero-ordinal-cannot-produce-ordinal-zero-memory-facts
  (with-runtime
    (fn [calls _]
      (is (= :cuda-runtime-device-unsupported
             (:reason (ex-data (error-of #(runtime/initialize! {:device-ordinal 1}))))))
      (is (empty? @calls)))))

(deftest retirement-winning-admission-does-not-invert-owner-and-root-locks
  (with-runtime
    (fn [calls native]
      (let [buffer (runtime/make-buffer 4 :float) original cleanup/retain-use!
            reserving (promise) reserve (promise) freeing (promise) finish-free (promise)]
        (binding [driver/*invoke* (fn [operation args]
                                   (when (= :free operation)
                                     (deliver freeing true)
                                     (deref finish-free 5000 :timeout))
                                   (native operation args))]
          (with-redefs [cleanup/retain-use!
                        (fn [owner]
                          ;; Fail before launching the free worker if a regression
                          ;; holds root here; this avoids creating a test deadlock.
                          (when (Thread/holdsLock runtime/state)
                            (deliver reserving :wrong-lock-order)
                            (throw (ex-info "root held before owner admission" {})))
                          (deliver reserving true)
                          (deref reserve 5000 :timeout)
                          (original owner))]
            (let [transfer (future (error-of #(runtime/upload-range! buffer (float-array 4) {:elements 4})))]
              (try
                (is (= true (deref reserving 5000 :timeout)))
                (when (= true (deref reserving 0 :timeout))
                  (let [retirement (future (runtime/free-buffer! buffer))]
                    (try
                      (is (= true (deref freeing 5000 :timeout)))
                      (is (= :available (locking runtime/state :available)))
                      (finally (deliver reserve true) (deliver finish-free true)))
                    (is (nil? (deref retirement 5000 :timeout)))
                    (is (= :owner-releasing (:reason (ex-data (deref transfer 5000 nil)))))
                    (is (not-any? #(= :upload (first %)) @calls))))
                (finally (deliver reserve true) (deliver finish-free true))))))))))

(deftest offset-copies-stage-exact-values-and-publish-only-the-requested-host-range
  (with-runtime
    (fn [calls native]
      (let [parent (runtime/make-buffer 16 :float)
            slice (runtime/slice-buffer parent 4 8 :float)
            uploads (atom [])]
        (try
          (binding [driver/*invoke*
                    (fn [operation args]
                      (let [result (native operation args)]
                      (case operation
                        :upload (let [[pointer staging _] args]
                                  (swap! uploads conj [pointer (vec (.toArray ^MemorySegment staging ValueLayout/JAVA_FLOAT))]))
                        :download (MemorySegment/copy
                                   (MemorySegment/ofArray (float-array [7 8])) (long 0)
                                   (second args) (long 0) (long (nth args 2)))
                        result)))]
            (runtime/upload-range! slice (float-array [9 1 2 9]) {:src-element 1 :dst-element 0 :elements 2})
            (is (= [[(+ (:device-pointer parent) 4) [1.0 2.0]]] @uploads))
            (let [host (float-array 4)]
              (is (identical? host (runtime/download-range! slice host {:src-element 0 :dst-element 1 :elements 2})))
              (is (= [0.0 7.0 8.0 0.0] (vec host)))))
          (is (= 2 (count (filter #(= :host-free (first %)) @calls))))
          (finally (runtime/free-buffer! parent)))))))

(deftest uncertain-copy-completion-retains-staging-and-memory-use-without-native-retry
  (with-runtime
    (fn [calls native]
      (let [buffer (runtime/make-buffer 4 :float)
            copy-error (ex-info "unknown upload outcome" {})
            wait-error (ex-info "unknown default-stream completion" {})]
        (binding [driver/*invoke*
                  (fn [operation args]
                    (let [result (native operation args)]
                    (cond (= :upload operation) (throw copy-error)
                          (and (= :stream-sync operation)
                               (zero? (.address ^MemorySegment (first args)))) (throw wait-error)
                          :else result)))]
          (is (identical? copy-error
                          (error-of #(runtime/upload-range! buffer (float-array [1 2 3 4]) {:elements 4}))))
          (is (= [wait-error] (vec (.getSuppressed copy-error))))
          (is (= :owner-in-use (:reason (ex-data (error-of #(runtime/free-buffer! buffer))))))
          (is (not-any? #(contains? #{:free :host-free} (first %)) @calls))
          (let [before @calls ledger (:prepared-execution @runtime/state)]
            (is (= 1 (count @(:debts ledger))))
            (is (identical? wait-error
                            (error-of #(execution/synchronize! ledger (:stream @runtime/state) @#'runtime/native!))))
            (is (= before @calls) "unknown completion cannot be retried through another route")))))))

(deftest failed-copy-with-known-rollback-completion-releases-memory-use
  (with-runtime
    (fn [calls native]
      (let [buffer (runtime/make-buffer 4 :float) primary (ex-info "copy failed" {})]
        (binding [driver/*invoke* (fn [operation args]
                                   (let [result (native operation args)]
                                     (if (= :upload operation) (throw primary) result)))]
          (is (identical? primary
                          (error-of #(runtime/upload-range! buffer (float-array 4) {:elements 4}))))
          (is (some #(and (= :stream-sync (first %))
                          (zero? (.address ^MemorySegment (first (second %))))) @calls))
          (runtime/free-buffer! buffer)
          (is (= 1 (count (filter #(= :host-free (first %)) @calls))))
          (is (= 1 (count (filter #(= :free (first %)) @calls)))))))))
