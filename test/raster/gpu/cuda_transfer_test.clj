(ns raster.gpu.cuda-transfer-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.cuda-driver :as driver]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.compiler.ir.transfer-capabilities :as capabilities]
            [raster.gpu.runtime-root :as root])
  (:import [java.lang.foreign Arena MemorySegment]))

(defn- with-device [f]
  (with-open [arena (Arena/ofShared)]
    (let [operations (atom []) memory (atom {}) pinned (atom {}) queued (atom [])
          ready (atom false) fault (atom nil) next-event (atom 20)
          finish (fn [] (doseq [copy @queued] (copy)) (reset! queued []) (reset! ready true))
          fake (fn [op args]
                 (swap! operations conj [op args])
                 (when (and (get @fault op) (not (:after? (get @fault op)))) (throw (:error (get @fault op))))
                 (let [result
                       (case op
                         :initialize {:context (MemorySegment/ofAddress 1) :stream (MemorySegment/ofAddress 2)
                                      :device 0 :compute-capability [8 6]}
                         :allocate (let [segment (.allocate arena (long (first args)) (long 256))]
                                     (swap! memory assoc (.address segment) segment) (.address segment))
                         :free (swap! memory dissoc (first args))
                         :host-allocate (let [segment (.allocate arena (long (first args)))]
                                          (swap! pinned assoc (.address segment) segment) segment)
                         :host-free (swap! pinned dissoc (.address (first args)))
                         :event-create (MemorySegment/ofAddress (swap! next-event inc))
                         :event-query @ready
                         (:event-sync :stream-sync) (finish)
                         :event-elapsed 10.0
                         :upload-async (let [[pointer host bytes _] args]
                                         (reset! ready false)
                                         (swap! queued conj #(MemorySegment/copy host (long 0)
                                                                                 (get @memory pointer) (long 0) (long bytes))))
                         :download-async (let [[pointer host bytes _] args]
                                           (reset! ready false)
                                           (swap! queued conj #(MemorySegment/copy (get @memory pointer) (long 0)
                                                                                   host (long 0) (long bytes))))
                         :download (let [[pointer host bytes] args]
                                     (MemorySegment/copy (get @memory pointer) (long 0) host (long 0) (long bytes)))
                         nil)]
                   (when (:after? (get @fault op)) (throw (:error (get @fault op))))
                   result))]
      (binding [driver/*invoke* fake]
        (with-redefs [runtime/state (atom {})]
          (f {:operations operations :pinned pinned :queued queued :finish finish :fault fault}))))))

(deftest asynchronous-upload-snapshots-input-and-download-poll-does-not-publish
  (with-device
    (fn [{:keys [operations pinned finish]}]
      (let [buffer (runtime/make-buffer 4 :float) input (float-array [1 2 3 4])
            output (float-array [-7 -7 -7 -7])
            token (try (runtime/submit-range-batch! [[buffer (runtime/plan-range buffer input {:elements 4} :upload)]] :upload)
                       (catch clojure.lang.ExceptionInfo _ nil))]
        (try
          (is (some? token) "CUDA must submit a native asynchronous upload over private pinned staging")
          (when token
            (is (not-any? #(contains? #{:stream-sync :event-sync} (first %)) @operations))
            (is (= 1 (count @pinned)))
            (aset input 0 (float 99))
            (runtime/await-event! token) (runtime/release-event! token)
            (is (empty? @pinned))
            (let [download (runtime/submit-range-batch!
                            [[buffer (runtime/plan-range buffer output {:elements 4} :download)]] :download)]
              (finish)
              (is (runtime/event-complete? download))
              (is (= [-7.0 -7.0 -7.0 -7.0] (vec output)) "completed polling does not publish downloads")
              (runtime/await-event! download)
              (is (= [1.0 2.0 3.0 4.0] (vec output)) "await publishes the immutable upload snapshot")
              (runtime/release-event! download)
              (is (empty? @pinned))))
          (finally (when token (runtime/release-event! token)) (runtime/free-buffer! buffer)))))))

(defn- error-data [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (ex-data e))))
(defn- plan [buffer host direction] (runtime/plan-range buffer host {:elements (:n-elements buffer)} direction))

(deftest asynchronous-host-return-does-not-advertise-physical-overlap-or-peers
  (let [capabilities (runtime/transfer-capabilities)]
    (is (identical? capabilities (capabilities/validate! capabilities)))
    (is (= :device-event (:submission capabilities)))
    (is (= :pinned (:staging-memory capabilities)))
    (is (= true (:async-h2d? capabilities) (:async-d2h? capabilities)))
    (is (false? (:independent-physical-queue? capabilities)))
    (is (true? (:physically-serialized? capabilities)))
    (is (false? (:peer-transfer? capabilities)))))

(deftest entire-batch-is-revalidated-before-staging-or-device-contact
  (with-device
    (fn [{:keys [operations]}]
      (let [buffer (runtime/make-buffer 4 :float) input (float-array 4)
            valid (plan buffer input :upload) before (count @operations)]
        (try
          (is (= :cuda-range-capacity
                 (:reason (error-data #(runtime/submit-range-batch!
                                        [[buffer valid] [buffer (assoc valid :n-bytes 20)]] :upload)))))
          (is (= before (count @operations)))
          (finally (runtime/free-buffer! buffer)))))))

(deftest active-retirement-refuses-before-poisoning-the-live-owner
  (with-device
    (fn [{:keys [operations]}]
      (let [buffer (runtime/make-buffer 4 :float) host (float-array [1 2 3 4])
            a (runtime/submit-range-batch! [[buffer (plan buffer host :upload)]] :upload)]
        (is (= :owner-in-use (:reason (error-data #(cleanup/release! (::cleanup/owner buffer))))))
        (is (= :live (:phase (cleanup/status (::cleanup/owner buffer)))))
        (let [b (runtime/submit-range-batch! [[buffer (plan buffer host :upload)]] :upload)]
          (runtime/release-event! a)
          (is (= :owner-in-use (:reason (error-data #(runtime/free-buffer! buffer)))))
          (runtime/release-event! b))
        (runtime/free-buffer! buffer)
        (is (= 1 (count (filter #(= :free (first %)) @operations))))))))

(deftest polling-retains-its-event-against-concurrent-retirement
  (with-device
    (fn [{:keys [operations]}]
      (let [buffer (runtime/make-buffer 4 :float) host (float-array 4)
            token (runtime/submit-range-batch! [[buffer (plan buffer host :upload)]] :upload)
            entered (promise) proceed (promise) native driver/*invoke*]
        (try
          (binding [driver/*invoke* (fn [operation args]
                                      (when (= :event-query operation)
                                        (deliver entered true)
                                        (deref proceed 5000 :timeout))
                                      (native operation args))]
            (let [worker (future (runtime/event-complete? token))]
              (try
                (is (= true (deref entered 5000 :timeout)))
                (is (= :owner-in-use (:reason (error-data #(runtime/release-event! token)))))
                (is (not-any? #(= :event-destroy (first %)) @operations))
                (finally (deliver proceed true)))
              (is (false? (deref worker 5000 :timeout)))))
          (finally (runtime/release-event! token) (runtime/free-buffer! buffer)))))))

(deftest zero-extent-completes-without-native-staging-or-events
  (with-device
    (fn [{:keys [operations pinned]}]
      (let [buffer (runtime/make-buffer 0 :half) destination (short-array 0)
            before (count @operations)
            token (runtime/submit-range-batch! [[buffer (plan buffer destination :download)]] :download)]
        (try
          (is (runtime/event-complete? token))
          (is (= {:timing-source :host-monotonic :elapsed-ns 0 :bytes 0 :commands 0
                  :direction :download :asynchronous? false} (runtime/await-event! token)))
          (runtime/release-event! token)
          (is (= before (count @operations)))
          (is (empty? @pinned))
          (finally (runtime/free-buffer! buffer)))))))

(deftest release-without-await-publishes-and-retires-private-staging-once
  (with-device
    (fn [{:keys [operations pinned]}]
      (let [buffer (runtime/make-buffer 4 :half) input (short-array [15360 -17408 0 -32768])
            output (short-array [7 7 7 7])
            upload (runtime/submit-range-batch! [[buffer (plan buffer input :upload)]] :upload)]
        (try
          (runtime/release-event! upload)
          (let [download (runtime/submit-range-batch! [[buffer (plan buffer output :download)]] :download)]
            (is (= [7 7 7 7] (vec output)))
            (runtime/release-event! download)
            (is (= (vec input) (vec output)))
            (is (empty? (cleanup/pending (::cleanup/owner download))))
            (runtime/release-event! download)
            (is (= 2 (count (filter #(= :host-free (first %)) @operations)))))
          (is (empty? @pinned))
          (finally (runtime/free-buffer! buffer)))))))

(deftest query-failure-and-retryable-wait-failure-do-not-release-staging-or-root
  (with-device
    (fn [{:keys [operations pinned fault]}]
      (let [buffer (runtime/make-buffer 4 :float) host (float-array [1 2 3 4])
            token (runtime/submit-range-batch! [[buffer (plan buffer host :upload)]] :upload)
            query-error (ex-info "query outcome unavailable" {:reason :injected-query})
            wait-error (ex-info "wait retry permitted" {:cleanup-retry-safe? true})]
        (try
          (reset! fault {:event-query {:error query-error}})
          (is (identical? query-error (try (runtime/event-complete? token) nil (catch Throwable e e))))
          (is (= 1 (count @pinned)))
          (is (= 2 (root/lease-count runtime/state)))
          (reset! fault {:event-sync {:error wait-error}})
          (is (identical? wait-error (try (runtime/release-event! token) nil (catch Throwable e e))))
          (is (seq (cleanup/pending (::cleanup/owner token))))
          (is (not-any? #(= :host-free (first %)) @operations))
          (is (= 2 (root/lease-count runtime/state)))
          (reset! fault nil)
          (runtime/release-event! token)
          (is (empty? @pinned))
          (is (= 1 (root/lease-count runtime/state)))
          (finally (runtime/free-buffer! buffer)))))))

(deftest unknown-enqueue-and-failed-drain-retain-exact-debt-without-retry
  (with-device
    (fn [{:keys [operations pinned fault]}]
      (let [buffer (runtime/make-buffer 4 :float) host (float-array [1 2 3 4])
            enqueue-error (ex-info "enqueue outcome unknown" {}) drain-error (ex-info "drain outcome unknown" {})
            retained (atom [])]
        (reset! fault {:upload-async {:error enqueue-error :after? true} :stream-sync {:error drain-error}})
        (is (identical? enqueue-error
                        (try (runtime/submit-range-batch! [[buffer (plan buffer host :upload)]] :upload
                                                          {:adopt-cleanup! #(swap! retained conj %)})
                             nil (catch Throwable e e))))
        (is (some #(identical? drain-error %) (.getSuppressed enqueue-error)))
        (is (= 1 (count @retained)))
        (is (= 1 (count @pinned)))
        (is (not-any? #(= :host-free (first %)) @operations))
        (reset! fault nil)
        (is (identical? drain-error (try (cleanup/release! (first @retained)) nil (catch Throwable e e))))
        (is (= 1 (count (filter #(= :stream-sync (first %)) @operations))))
        (is (= :owner-in-use (:reason (error-data #(runtime/free-buffer! buffer)))))))))

(deftest retryable-pinned-free-keeps-buffer-pins-and-retries-only-the-failed-resource
  (with-device
    (fn [{:keys [operations pinned fault]}]
      (let [buffer (runtime/make-buffer 4 :float) host (float-array [1 2 3 4])
            token (runtime/submit-range-batch! [[buffer (plan buffer host :upload)]] :upload)
            failure (ex-info "pinned free failed without destruction" {:cleanup-retry-safe? true})]
        (try
          (reset! fault {:host-free {:error failure}})
          (is (identical? failure (try (runtime/release-event! token) nil (catch Throwable e e))))
          (is (= 1 (count @pinned)))
          (is (= :owner-in-use (:reason (error-data #(runtime/free-buffer! buffer)))))
          (reset! fault nil)
          (runtime/release-event! token)
          (is (empty? @pinned))
          (is (= 2 (count (filter #(= :event-destroy (first %)) @operations))))
          (is (= 2 (count (filter #(= :host-free (first %)) @operations))))
          (finally (runtime/free-buffer! buffer)))))))

(deftest release-without-await-reports-host-copy-failure-once-after-safe-retirement
  (with-device
    (fn [{:keys [pinned]}]
      (let [buffer (runtime/make-buffer 4 :float) host (float-array [1 2 3 4])
            upload (runtime/submit-range-batch! [[buffer (plan buffer host :upload)]] :upload)
            destination-arena (Arena/ofShared) destination (.allocate destination-arena (long 16))]
        (runtime/release-event! upload)
        (let [token (runtime/submit-range-batch! [[buffer (plan buffer destination :download)]] :download)]
          (.close destination-arena)
          (is (instance? IllegalStateException (try (runtime/release-event! token) nil (catch Throwable e e))))
          (is (empty? @pinned))
          (is (empty? (cleanup/pending (::cleanup/owner token))))
          (is (nil? (try (runtime/release-event! token) nil (catch Throwable e e))))
          (is (= 1 (root/lease-count runtime/state)))
          (when (empty? @pinned) (runtime/free-buffer! buffer)))))))

(deftest publication-error-remains-primary-while-destructive-failure-retains-only-outstanding-debt
  (with-device
    (fn [{:keys [operations pinned fault]}]
      (let [buffer (runtime/make-buffer 4 :float) host (float-array [1 2 3 4])
            upload (runtime/submit-range-batch! [[buffer (plan buffer host :upload)]] :upload)
            destination-arena (Arena/ofShared) destination (.allocate destination-arena (long 16))
            destruction-error (ex-info "pinned free not performed" {:cleanup-retry-safe? true})]
        (runtime/release-event! upload)
        (let [token (runtime/submit-range-batch! [[buffer (plan buffer destination :download)]] :download)]
          (.close destination-arena)
          (reset! fault {:host-free {:error destruction-error}})
          (let [operation-error (try (runtime/release-event! token) nil (catch Throwable e e))]
            (is (instance? IllegalStateException operation-error))
            (is (some #(identical? destruction-error %) (.getSuppressed operation-error)))
            (is (= 1 (count @pinned)))
            (is (= 2 (root/lease-count runtime/state)))
            (is (= :owner-in-use (:reason (error-data #(runtime/free-buffer! buffer)))))
            (let [destroyed (count (filter #(= :event-destroy (first %)) @operations))]
              (reset! fault nil)
              (runtime/release-event! token)
              (is (empty? @pinned))
              (is (= destroyed (count (filter #(= :event-destroy (first %)) @operations))))
              (is (empty? (cleanup/pending (::cleanup/owner token))))
              (is (= 1 (root/lease-count runtime/state)))))
          (runtime/free-buffer! buffer))))))

(deftest unknown-pinned-acquisition-retains-root-and-buffer-without-inventing-a-handle
  (with-device
    (fn [{:keys [operations pinned fault]}]
      (let [buffer (runtime/make-buffer 4 :float) host (float-array 4)
            failure (ex-info "allocation readback outcome unknown" {}) retained (atom [])]
        (reset! fault {:host-allocate {:error failure :after? true}})
        (is (identical? failure
                        (try (runtime/submit-range-batch! [[buffer (plan buffer host :upload)]] :upload
                                                          {:adopt-cleanup! #(swap! retained conj %)})
                             nil (catch Throwable error error))))
        (is (= 1 (count @retained)))
        (is (= 1 (count @pinned)))
        (is (= 2 (root/lease-count runtime/state)))
        (reset! fault nil)
        (is (identical? failure (try (cleanup/release! (first @retained)) nil (catch Throwable error error))))
        (is (not-any? #(contains? #{:host-free :upload-async :event-create} (first %)) @operations))
        (is (= :owner-in-use (:reason (error-data #(runtime/free-buffer! buffer)))))))))

(deftest indeterminate-event-destruction-retains-staging-and-never-repeats-the-native-call
  (with-device
    (fn [{:keys [operations pinned fault]}]
      (let [buffer (runtime/make-buffer 4 :float) host (float-array 4)
            token (runtime/submit-range-batch! [[buffer (plan buffer host :upload)]] :upload)
            failure (ex-info "event destruction outcome unknown" {})]
        (reset! fault {:event-destroy {:error failure :after? true}})
        (is (identical? failure (try (runtime/release-event! token) nil (catch Throwable error error))))
        (is (= 1 (count @pinned)))
        (is (= 2 (root/lease-count runtime/state)))
        (let [before (count @operations)]
          (reset! fault nil)
          (is (identical? failure (try (runtime/release-event! token) nil (catch Throwable error error))))
          (is (= before (count @operations))))
        (is (not-any? #(= :host-free (first %)) @operations))
        (is (= :owner-in-use (:reason (error-data #(runtime/free-buffer! buffer)))))))))
