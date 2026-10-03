(ns raster.gpu.opencl-transfer-cleanup-test
  "Hardware-free faults through the production asynchronous transfer ownership path."
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root]
            [raster.gpu.core :as gpu]
            [raster.compiler.ir.buffer-view :as bview])
  (:import [java.lang AutoCloseable]
           [java.lang.foreign MemorySegment ValueLayout]))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))

(defn- with-transfer [options test!]
  (let [v #(ns-resolve 'raster.gpu.ocl-runtime %)
        state (atom {:initialized? false})
        queue (MemorySegment/ofAddress 10)
        calls (atom []) staging (atom []) index (atom -1)
        primary (ex-info "submission failed" {})
        secondary (ex-info "cleanup outcome unknown" {})
        source (MemorySegment/ofArray (float-array [1 2]))
        entries (vec (repeat 2 [{:cl-mem (MemorySegment/ofAddress 100)}
                                {:buf-off 0 :host-off 0 :n-bytes 8 :host-seg source}]))]
    (root/initialize! state [] (fn [_] {:transfer-queue queue}))
    (with-redefs-fn
      (merge {(v 'state) state (v 'assert-buffer-live!) identity
              (v 'cl-call!)
              (fn [label _ args]
                (swap! calls conj label)
                (when-let [callback (:on-native options)] (callback label args))
                (case label
                  "clEnqueueWriteBuffer"
                  (let [i (swap! index inc)]
                    (swap! staging conj (nth args 5))
                    (when (= i (:fail-enqueue options)) (throw primary))
                    (.set ^MemorySegment (last args) ValueLayout/ADDRESS 0
                          (if (= i (:null-event options)) MemorySegment/NULL
                              (MemorySegment/ofAddress (long (+ 200 i))))))
                  "clFlush" (when (:fail-flush? options) (throw primary))
                  "clFinish" (when (:fail-drain? options) (throw secondary))
                  "clReleaseEvent" (when (and (:fail-release? options)
                                              (= 200 (.address ^MemorySegment (first args))))
                                     (throw secondary))
                  "clWaitForEvents" (when (:fail-wait? options) (throw primary))
                  "clGetEventProfilingInfo" (when (:fail-profile? options) (throw primary))
                  nil))}
             (into {} (map (fn [name] [(v name) (delay :fake)])
                           '[h-clEnqueueWriteBuffer h-clFlush h-clFinish h-clReleaseEvent
                             h-clWaitForEvents h-clGetEventProfilingInfo h-clGetEventInfo])))
      #(test! {:state state :calls calls :staging staging :entries entries
               :primary primary :secondary secondary}))))

(deftest successful-transfer-release-drains-events-before-staging-and-root
  (with-transfer {}
    (fn [{:keys [state calls staging entries]}]
      (let [token (ocl/submit-range-batch! entries :upload)]
        (is (= 1 (root/lease-count state)))
        (is (every? #(.isAlive (.scope ^MemorySegment %)) @staging))
        (ocl/release-event! token)
        (is (= ["clFinish" "clReleaseEvent" "clReleaseEvent"] (subvec @calls 3)))
        (is (every? #(not (.isAlive (.scope ^MemorySegment %))) @staging))
        (is (zero? (root/lease-count state)))
        (let [before @calls]
          (ocl/release-event! token)
          (is (= before @calls)))))))

(deftest known-post-enqueue-failure-retires-all-transfer-resources
  (with-transfer {:fail-flush? true}
    (fn [{:keys [state staging entries primary]}]
      (is (identical? primary (error-of #(ocl/submit-range-batch! entries :upload))))
      (is (every? #(not (.isAlive (.scope ^MemorySegment %))) @staging))
      (is (zero? (root/lease-count state))))))

(deftest uncertain-transfer-outcomes-retain-staging-and-root-without-native-retry
  (doseq [options [{:fail-enqueue 0} {:null-event 1}
                   {:fail-flush? true :fail-drain? true}
                   {:fail-flush? true :fail-release? true}]]
    (with-transfer options
      (fn [{:keys [state calls staging entries]}]
        (let [error (error-of #(ocl/submit-range-batch! entries :upload))
              owner (::cleanup/unresolved (ex-data error))
              before @calls]
          (is (some? owner))
          (is (= 1 (root/lease-count state)))
          (is (every? #(.isAlive (.scope ^MemorySegment %)) @staging))
          (is (some #{:runtime-root-lease} (cleanup/pending owner)))
          (is (some? (error-of #(cleanup/release! owner))))
          (is (= before @calls)))))))

(deftest transfer-token-cannot-drop-its-canonical-owner
  (with-transfer {}
    (fn [{:keys [state entries]}]
      (let [token (ocl/submit-range-batch! entries :upload)]
        (try
          (doseq [operation [ocl/await-event! ocl/event-complete? ocl/release-event!]]
            (is (= :missing-cleanup-owner
                   (:reason (ex-data (error-of #(operation (dissoc token ::cleanup/owner))))))))
          (is (= 1 (root/lease-count state)))
          (finally (ocl/release-event! token)))))))

(defn- with-session-transfer [test!]
  (let [buffer {:dtype :float :n-elements 2 :byte-size 8
                :cl-mem (MemorySegment/ofAddress 100)}
        allocation (bview/allocation {:id :out :byte-size 8 :memory-space :device
                                      :device :ocl:0 :coherence :host-coherent :ownership :owned})
        session (atom {:device-id :ocl:0 :session-id :transfer-fault :closed? false
                       :buffers {:out buffer :alias buffer}
                       :allocations {:out allocation :alias (assoc allocation :ownership :borrowed)}
                       :graphs {} :kernel-graphs {} :prepared {} :events {}})]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve)
       (fn [_ name]
         (case name
           "plan-range" (fn [_ host _ _] {:buf-off 0 :host-off 0 :n-bytes 8
                                          :host-seg (MemorySegment/ofArray ^floats host)})
           "submit-range-batch!" ocl/submit-range-batch!
           "await-event!" ocl/await-event!
           "release-event!" ocl/release-event!
           "event-complete?" ocl/event-complete?
           (throw (ex-info "Unexpected contact after transfer debt" {:name name}))))}
      #(test! session))))

(deftest failed-public-submission-keeps-resident-footprint-but-not-caller-host-leases
  (doseq [unknown? [false true]]
    (with-transfer (if unknown? {:fail-enqueue 0} {:fail-flush? true})
      (fn [{:keys [state staging primary]}]
        (with-session-transfer
          (fn [session]
            (let [closed (atom 0)
                  host-lease (reify AutoCloseable (close [_] (swap! closed inc)))]
              (is (identical? primary
                              (error-of #(gpu/submit-upload-ranges-retained!
                                          session [[:out (float-array [1 2]) {:elements 2}]]
                                          [host-lease]))))
              (is (zero? @closed))
              (is (= (if unknown? 1 0) (root/lease-count state)))
              (if unknown?
                (do
                  (is (= 1 (count (:events @session))))
                  (is (= #{:out} (:buffer-keys (first (vals (:events @session))))))
                  (doseq [key [:out :alias]]
                    (is (= :buffer-pending-transfer
                           (:reason (ex-data (error-of #(gpu/free-buffer! session key)))))))
                  (is (identical? primary (error-of #(gpu/close-session! session))))
                  (is (some? (gpu/buffer session :out)))
                  (is (every? #(.isAlive (.scope ^MemorySegment %)) @staging)))
                (is (empty? (:events @session))))
              (.close host-lease)
              (is (= 1 @closed)))))))))

(deftest session-publication-watch-rejection-precedes-any-transfer-contact
  (with-transfer {}
    (fn [{:keys [state calls]}]
      (with-session-transfer
        (fn [session]
          (add-watch session :reject-event
                     (fn [_ _ _ next-state]
                       (when (seq (:events next-state))
                         (swap! session assoc :events {}))))
          (is (= :transfer-event-publication-lost
                 (:reason (ex-data (error-of #(gpu/submit-upload-ranges!
                                               session [[:out (float-array [1 2]) {:elements 2}]]))))))
          (is (empty? @calls))
          (is (empty? (:events @session)))
          (is (zero? (root/lease-count state))))))))

(deftest successful-public-completion-closes-host-lease-once-after-backend-cleanup
  (with-transfer {}
    (fn [{:keys [state]}]
      (with-session-transfer
        (fn [session]
          (let [closed (atom 0)
                host-lease (reify AutoCloseable
                             (close [_]
                               (is (zero? (root/lease-count state)))
                               (swap! closed inc)))
                event (gpu/submit-upload-ranges-retained!
                       session [[:out (float-array [1 2]) {:elements 2}]] [host-lease])]
            (is (zero? @closed))
            (gpu/await-event! session event)
            (gpu/await-event! session event)
            (gpu/release-event! session event)
            (is (= 1 @closed))
            (is (empty? (:events @session)))))))))

(deftest await-operation-errors-do-not-prevent-backend-or-host-retirement
  (doseq [fail-release? [false true] failure [:wait :profile]]
    (with-transfer {(if (= failure :wait) :fail-wait? :fail-profile?) true
                    :fail-release? fail-release?}
      (fn [{:keys [state calls primary secondary]}]
        (with-session-transfer
          (fn [session]
            (let [closed (atom 0)
                  lease (reify AutoCloseable (close [_] (swap! closed inc)))
                  event (gpu/submit-upload-ranges-retained!
                         session [[:out (float-array [1 2]) {:elements 2}]] [lease])]
              (is (identical? primary (error-of #(gpu/await-event! session event))))
              (is (= (if fail-release? 1 0) (root/lease-count state)))
              (is (= (if fail-release? 0 1) @closed))
              (is (identical? primary (error-of #(gpu/event-measurement session event))))
              (when fail-release?
                (is (some #(identical? secondary %) (.getSuppressed primary))))
              (let [before @calls]
                (is (identical? primary (error-of #(gpu/release-event! session event))))
                (is (= before @calls) "no repeated await or indeterminate native release"))
              (is (= fail-release? (boolean (seq (:events @session))))))))))))

(deftest event-consumption-cannot-reenter-session-root-lifetime-operations
  (doseq [[operation native-label] [[gpu/event-complete? "clGetEventInfo"]
                                    [gpu/await-event! "clWaitForEvents"]
                                    [gpu/release-event! "clReleaseEvent"]]
          uncaught? [false true]]
    (let [session-slot (volatile! nil) declined (atom []) active? (atom true)]
      (with-transfer
        {:on-native
         (fn [label _]
           (when (and @active? (= label native-label))
             (let [session @session-slot
                   errors (mapv #(error-of %) [#(gpu/close-session! session)
                                               #(gpu/free-buffer! session :out)])]
               (swap! declined into errors)
               (when uncaught? (throw (first errors))))))}
        (fn [{:keys [state]}]
          (with-session-transfer
            (fn [session]
              (vreset! session-slot session)
              (let [event (gpu/submit-upload-ranges!
                           session [[:out (float-array [1 2]) {:elements 2}]])
                    error (error-of #(operation session event))]
                (is (= [:reentrant-root-lifecycle :reentrant-root-lifecycle]
                       (mapv #(-> % ex-data :reason) @declined)))
                (is (= uncaught? (some? error)))
                (is (false? (:closed? @session)))
                (is (some? (gpu/buffer session :out)))
                (reset! active? false)
                (when (seq (:events @session))
                  (error-of #(gpu/release-event! session event)))
                (is (= (if (and uncaught? (= operation gpu/release-event!)) 1 0)
                       (root/lease-count state)))))))))))

(deftest cleanup-publication-watch-cannot-reenter-session-root-lifetime-operations
  (doseq [operation [gpu/await-event! gpu/release-event!]
          uncaught? [false true]]
    (with-transfer {}
      (fn [{:keys [state calls]}]
        (with-session-transfer
          (fn [session]
            (let [event (gpu/submit-upload-ranges!
                         session [[:out (float-array [1 2]) {:elements 2}]])
                  owner (::cleanup/owner (get-in @session [:events (:id event)]))
                  declined (atom [])
                  before @calls]
              (add-watch (:state owner) :lifecycle-reentry
                         (fn [_ _ previous next-state]
                           (when (and (not= :releasing (:phase previous))
                                      (= :releasing (:phase next-state)))
                             (let [errors (mapv error-of [#(gpu/close-session! session)
                                                          #(gpu/free-buffer! session :out)])]
                               (swap! declined into errors)
                               (when uncaught? (throw (first errors)))))))
              (let [error (error-of #(operation session event))]
                (is (= [:reentrant-root-lifecycle :reentrant-root-lifecycle]
                       (mapv #(-> % ex-data :reason) @declined)))
                (is (= uncaught? (some? error)))
                (is (false? (:closed? @session)))
                (is (some? (gpu/buffer session :out)))
                (remove-watch (:state owner) :lifecycle-reentry)
                (if uncaught?
                  (do
                    (is (= before @calls) "publication failed before backend cleanup contact")
                    (is (= 1 (root/lease-count state)))
                    (is (identical? owner (::cleanup/owner
                                           (get-in @session [:events (:id event)]))))
                    (is (seq (cleanup/pending owner))))
                  (do
                    (when (seq (:events @session)) (gpu/release-event! session event))
                    (is (zero? (root/lease-count state)))))))))))))

(deftest session-teardown-retires-transfer-after-known-await-operation-failure
  (doseq [failure [:wait :profile]]
    (with-transfer {(if (= failure :wait) :fail-wait? :fail-profile?) true}
      (fn [{:keys [state calls]}]
        (with-session-transfer
          (fn [session]
            (let [closed (atom 0)
                  lease (reify AutoCloseable (close [_] (swap! closed inc)))
                  resolver (ns-resolve 'raster.gpu.core 'rt-resolve)
                  original @resolver]
              ;; The buffer is a hardware-free fixture; its empty canonical owner retires
              ;; without native contact. The event still uses the production transfer owner.
              (let [owner (cleanup/owner [])]
                (swap! session #(-> %
                                    (assoc :buffer-owners {:out owner})
                                    (assoc-in [:buffers :out ::cleanup/owner] owner)
                                    (assoc-in [:buffers :alias ::cleanup/lifetime-owner] owner))))
              (gpu/submit-upload-ranges-retained!
               session [[:out (float-array [1 2]) {:elements 2}]] [lease])
              (with-redefs-fn
                {resolver (fn [device name]
                            (if (= name "close-kernel-arena!") (fn [_] nil)
                                (original device name)))}
                #(do
                   (is (nil? (error-of (fn [] (gpu/close-session! session)))))
                   (is (= :closed (:lifecycle @session)))
                   (is (empty? (:events @session)))
                   (is (zero? (root/lease-count state)))
                   (is (= 1 @closed))
                   (let [before @calls]
                     (gpu/close-session! session)
                     (is (= before @calls))))))))))))
