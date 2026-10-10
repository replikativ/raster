(ns raster.gpu.opencl-recording-cleanup-test
  "Hardware-free faults through the production recording/submission ownership paths."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.storage-representation :as probe]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.gpu.core :as gpu]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root])
  (:import [java.lang.foreign MemorySegment ValueLayout]
           [java.lang.invoke MethodHandles]))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))

(defn- mocked-recording [options f]
  (let [v #(ns-resolve 'raster.gpu.ocl-runtime %)
        queue (MemorySegment/ofAddress 100)
        queue-result (if (:null-queue? options) MemorySegment/NULL queue)
        queue-create (MethodHandles/dropArguments
                      (MethodHandles/constant MemorySegment queue-result) 0
                      (into-array Class [MemorySegment MemorySegment Long/TYPE MemorySegment]))
        calls (atom []) index (atom -1)
        state (atom {:initialized? false})
        _ (root/initialize! state [] (fn [_] {:queue queue :context MemorySegment/NULL
                                              :device MemorySegment/NULL}))]
    (with-redefs-fn
      {(v 'ensure-init!) (fn [] nil)
       (v 'state) state
       (v 'h-clCreateCommandQueue) (delay (if-let [error (:create-failure options)]
                                            (throw error) queue-create))
       (v 'h-clFinish) (delay :fake)
       (v 'h-clFlush) (delay :fake)
       (v 'h-clWaitForEvents) (delay :fake)
       (v 'h-clReleaseCommandQueue) (delay :fake)
       (v 'h-clReleaseEvent) (delay :fake)
       (v 'h-clGetEventProfilingInfo) (delay :fake)
       (v 'cl-call!) (fn [label _ args]
                       (swap! calls conj [label (first args)])
                       (when-let [on-native (:on-native options)] (on-native label args))
                       (when-let [error (get (:failures options) label)] (throw error))
                       (when (= label "clReleaseEvent")
                         (when-let [error (get (:event-release-failures options)
                                               (.address ^MemorySegment (first args)))]
                           (throw error)))
                       (when (= label "clGetEventProfilingInfo")
                         (.set ^MemorySegment (nth args 3) ValueLayout/JAVA_LONG 0 (long 100))))
       (v 'enqueue-bound!)
       (fn [_ _ event-out _]
         (let [i (swap! index inc)]
           (swap! calls conj [:enqueue i])
           (when-let [error (get (:enqueue-failures options) i)] (throw error))
           (when-not (.equals MemorySegment/NULL event-out)
             (.set ^MemorySegment event-out ValueLayout/ADDRESS 0
                   (if (= (:null-event-index options) i) MemorySegment/NULL
                       (MemorySegment/ofAddress (long (+ 200 i))))))))}
      #(f calls))))

(defn- recording [profile?]
  (ocl/record-graph! [{:bound {:wg 32} :group-count 1 :kernel-name "a"}
                      {:bound {:wg 32} :group-count 1 :kernel-name "b"}]
                     {:profile? profile?}))

(deftest public-graph-retirement-keeps-await-progress-after-native-release-failure
  (doseq [retry-safe? [true false]]
    (let [failure (ex-info "event release failed" {:cleanup-retry-safe? retry-safe?})
          release-count (atom 0)
          closed (atom 0)]
      (mocked-recording
       {:on-native (fn [label _]
                     (when (and (= "clReleaseEvent" label)
                                (= 1 (swap! release-count inc)))
                       (throw failure)))}
       (fn [calls]
         (let [graph (recording false)
               session (atom {:device-id :ocl:0 :session-id :retirement-progress :closed? false
                              :kernel-graphs {:graph {:generation 1 :runtime-graph graph
                                                      :outputs {'out :resident}
                                                      :execution-plan {:queues [{:class :compute}]}
                                                      :resident-footprint {:allocation-ids #{:out}
                                                                           :resident-buffers []}}}
                              :events {}})
               handle (gpu/->KernelGraphHandle :graph :retirement-progress 1)]
           (with-redefs-fn
             {(ns-resolve 'raster.gpu.core 'rt-resolve)
              (fn [_ name]
                (case name
                  "submit-graph!" ocl/submit-graph!
                  "await-event!" ocl/await-event!
                  "release-event!" ocl/release-event!
                  "event-complete?" ocl/event-complete?
                  (throw (ex-info "unexpected runtime operation" {:name name}))))}
             (fn []
               (let [event (gpu/submit-kernel-graph! session handle)
                     entry-path [:events (:id event)]
                     lease (reify java.lang.AutoCloseable (close [_] (swap! closed inc)))]
                 ;; Exercise retained-resource ordering on the same common event path.
                 (swap! session assoc-in (conj entry-path :retained-resources) [lease])
                 (is (identical? failure (error-of #(gpu/await-event! session event))))
                 (is (= :awaited (:status (get-in @session entry-path))))
                 (is (zero? @closed))
                 (is (true? (gpu/event-complete? session event)))
                 (is (re-find #"in-flight"
                              (.getMessage ^Throwable
                                           (error-of #(gpu/submit-kernel-graph! session handle)))))
                 (if retry-safe?
                   (do
                     (is (nil? (gpu/release-event! session event)))
                     (is (= 2 @release-count))
                     (is (= 1 @closed))
                     (is (empty? (:events @session)))
                     (is (nil? @(:submission-state graph)))
                     (ocl/destroy-graph! graph))
                   (let [before @calls]
                     (dotimes [_ 2]
                       (is (identical? failure (error-of #(gpu/release-event! session event)))))
                     (is (= before @calls) "indeterminate native release is never retried")
                     (is (= 1 @release-count))
                     (is (zero? @closed))
                     (is (= :awaited (:status (get-in @session entry-path))))))
               (is (= 1 (count (filter #(= "clWaitForEvents" (first %)) @calls)))))))))))))

(deftest session-close-retirement-retry-does-not-repeat-await-or-descend-early
  (doseq [retry-safe? [true false]]
    (let [failure (ex-info "retirement failed" {:cleanup-retry-safe? retry-safe?})
          releases (atom 0) retired (atom [])]
      (mocked-recording
       {:on-native (fn [label _]
                     (when (and (= "clReleaseEvent" label) (= 1 (swap! releases inc)))
                       (throw failure)))}
       (fn [calls]
         (let [graph (recording false)
               graph-owner (cleanup/owner [{:id :graph :release #(do (ocl/destroy-graph! graph)
                                                                    (swap! retired conj :graph))}])
               session (atom {:device-id :ocl:0 :session-id :close-progress :closed? false
                              :kernel-graphs {:graph {::cleanup/owner graph-owner :generation 1
                                                      :runtime-graph graph :outputs {}
                                                      :execution-plan {:queues [{:class :compute}]}
                                                      :resident-footprint {:allocation-ids #{:out}
                                                                           :resident-buffers []}}}
                              :events {}})
               handle (gpu/->KernelGraphHandle :graph :close-progress 1)]
           (with-redefs-fn
             {(ns-resolve 'raster.gpu.core 'rt-resolve)
              (fn [_ name]
                (case name
                  "submit-graph!" ocl/submit-graph!
                  "await-event!" ocl/await-event!
                  "release-event!" ocl/release-event!
                  "close-kernel-arena!" (fn [_] (swap! retired conj :arena))
                  (throw (ex-info "unexpected teardown operation" {:name name}))))}
             (fn []
               (let [event (gpu/submit-kernel-graph! session handle)
                     lease (reify java.lang.AutoCloseable (close [_] (swap! retired conj :lease)))]
                 (swap! session assoc-in [:events (:id event) :retained-resources] [lease])
                 (is (identical? failure (error-of #(gpu/close-session! session))))
                 (is (= :releasing (:lifecycle @session)))
                 (is (= :awaited (get-in @session [:events (:id event) :status])))
                 (is (empty? @retired))
                 (if retry-safe?
                   (do (is (nil? (error-of #(gpu/close-session! session))))
                       (is (= :closed (:lifecycle @session)))
                       (is (= [:lease :graph :arena] @retired))
                       (is (= 2 @releases)))
                   (let [before @calls]
                     (is (identical? failure (error-of #(gpu/close-session! session))))
                     (is (= before @calls))
                     (is (= 1 @releases))
                     (is (empty? @retired))))
                 (is (= 1 (count (filter #(= "clWaitForEvents" (first %)) @calls)))))))))))))

(deftest recording-root-pin-outlives-submission-and-retires-with-graph
  (doseq [profile? [false true]]
    (mocked-recording {}
                      (fn [_]
                        (let [state @(ns-resolve 'raster.gpu.ocl-runtime 'state)
                              graph (recording profile?)]
                          (is (= 1 (root/lease-count state)))
                          (let [token (ocl/submit-graph! graph)]
                            (ocl/await-event! token)
                            (ocl/release-event! token)
                            (is (= 1 (root/lease-count state))))
                          (ocl/destroy-graph! graph)
                          (is (zero? (root/lease-count state))))))))

(deftest unknown-recording-drain-keeps-the-root-pinned
  (mocked-recording {:failures {"clFinish" (ex-info "drain outcome unknown" {})}}
                    (fn [_]
                      (let [state @(ns-resolve 'raster.gpu.ocl-runtime 'state)
                            graph (recording false)]
                        (ocl/submit-graph! graph)
                        (is (some? (error-of #(ocl/destroy-graph! graph))))
                        (is (= [:submission :runtime-root-lease] (cleanup/pending (::cleanup/owner graph))))
                        (is (= 1 (root/lease-count state)))))))

(deftest partial-enqueue-is-drained-before-independent-event-release
  (doseq [profile? [false true] drain-fails? [false true]]
    (let [primary (proxy [RuntimeException] ["enqueue failed" nil false false])
          drain (ex-info "queue completion unknown" {})]
      (mocked-recording {:enqueue-failures {1 primary}
                         :failures (when drain-fails? {"clFinish" drain})}
        (fn [calls]
          (let [graph (recording profile?)]
            (is (identical? primary (error-of #(ocl/submit-graph! graph))))
            (is (= [[:enqueue 0] [:enqueue 1] "clFinish"]
                   (mapv (fn [[label value]] (if (= label :enqueue) [label value] label))
                         (take 3 @calls))))
            (is (some? @(:submission-state graph)))
            (let [submission @(:submission-state graph)
                  arena (:resource @(:arena-slot submission))]
              (is (.isAlive (.scope arena)))
              (is (= (if drain-fails? (if profile? [:drain [:event 0] [:event 1] :arena]
                                         [:drain [:event 0] :arena])
                        (if profile? [[:event 1] :arena] [[:event 0] :arena]))
                     (cleanup/pending (::cleanup/owner submission))))
              (let [before @calls]
                (is (some? (error-of #(ocl/destroy-graph! graph))))
                (is (= before @calls)))
              (is (not-any? #(= "clReleaseCommandQueue" (first %)) @calls))
              ;; The unresolved native handle is synthetic. Close this test-only arena after
              ;; proving production retains it; no claim that unknown native events were freed.
              (.close arena))))))))

(deftest flush-failure-drains-and-releases-known-events-before-arena
  (doseq [profile? [false true]]
    (let [primary (ex-info "flush failed" {})]
      (mocked-recording {:failures {"clFlush" primary}}
        (fn [calls]
          (let [graph (recording profile?)]
            (is (identical? primary (error-of #(ocl/submit-graph! graph))))
            (is (nil? @(:submission-state graph)))
            (is (= ["clFlush" "clFinish"] (mapv first (take 2 (drop 2 @calls)))))
            (is (= (if profile? 2 1) (count (filter #(= "clReleaseEvent" (first %)) @calls))))
            (ocl/destroy-graph! graph)))))))

(deftest completed-replay-releases-exact-once-and-rejects-stale-token-use
  (doseq [profile? [false true]]
    (mocked-recording {}
      (fn [calls]
        (let [graph (recording profile?) token (ocl/submit-graph! graph)
              arena (:arena token)]
          (ocl/await-event! token)
          (ocl/release-event! token)
          (when profile? (ocl/reset-graph-events! graph))
          (is (nil? @(:submission-state graph)))
          (is (not (.isAlive (.scope arena))))
          (is (not-any? #(= "clFinish" (first %)) @calls))
          (is (= (if profile? 2 1) (count (filter #(= "clReleaseEvent" (first %)) @calls))))
          (let [before @calls]
            (ocl/release-event! token)
            (is (= before @calls)))
          (is (= :owner-releasing (:reason (ex-data (error-of #(ocl/await-event! token))))))
          (ocl/destroy-graph! graph)
          (let [before @calls]
            (ocl/destroy-graph! graph)
            (is (= before @calls))
            (doseq [operation [ocl/submit-graph! ocl/reset-graph-events! ocl/read-graph-timestamps!]]
              (is (= :owner-releasing (:reason (ex-data (error-of #(operation graph)))))))
            (is (= before @calls))))))))

(deftest missing-owner-graph-tokens-never-fall-back-to-legacy-release
  (doseq [profile? [false true]]
    (mocked-recording {}
      (fn [calls]
        (let [graph (recording profile?) token (ocl/submit-graph! graph)
              broken (dissoc token ::cleanup/owner) before @calls]
          (doseq [operation [ocl/await-event! ocl/event-complete? ocl/release-event!]]
            (is (= :missing-cleanup-owner (:reason (ex-data (error-of #(operation broken)))))))
          (is (= before @calls))
          (is (.isAlive (.scope (:arena token))))
          (is (some? @(:submission-state graph)))
          (ocl/await-event! token)
          (ocl/release-event! token)
          (ocl/destroy-graph! graph)))))
  (is (nil? (ocl/await-event! {:complete? true})))
  (is (true? (ocl/event-complete? {:complete? true})))
  (is (nil? (ocl/release-event! {:complete? true}))))

(deftest failed-event-release-keeps-arena-and-queue-owned
  (let [failure (ex-info "event release unknown" {})]
    (mocked-recording {:failures {"clReleaseEvent" failure}}
      (fn [calls]
        (let [graph (recording true) token (ocl/submit-graph! graph)
              arena (:arena token)]
          (ocl/await-event! token)
          (is (identical? failure (error-of #(ocl/reset-graph-events! graph))))
          (is (= 2 (count (filter #(= "clReleaseEvent" (first %)) @calls))))
          (is (.isAlive (.scope arena)))
          (let [before @calls]
            (is (identical? failure (error-of #(ocl/destroy-graph! graph))))
            (is (= before @calls)))
          (is (not-any? #(= "clReleaseCommandQueue" (first %)) @calls))
          (.close arena))))))

(deftest timestamp-read-rejects-partially-released-submission
  (let [failure (ex-info "second event release unknown" {})]
    (mocked-recording {:event-release-failures {201 failure}}
      (fn [calls]
        (let [graph (recording true) token (ocl/submit-graph! graph)]
          (ocl/await-event! token)
          (is (identical? failure (error-of #(ocl/reset-graph-events! graph))))
          (is (= [[:event 1] :arena]
                 (cleanup/pending (::cleanup/owner @(:submission-state graph)))))
          (let [before @calls]
            (is (= :owner-releasing (:reason (ex-data (error-of #(ocl/read-graph-timestamps! graph))))))
            (is (= before @calls) "do not query an event released during earlier cleanup"))
          (.close (:arena token)))))))

(deftest timestamp-error-remains-primary-when-event-cleanup-also-fails
  (let [primary (ex-info "timestamp read failed" {})
        secondary (ex-info "event release unknown" {})]
    (mocked-recording {:failures {"clGetEventProfilingInfo" primary "clReleaseEvent" secondary}}
      (fn [_]
        (let [graph (recording true) token (ocl/submit-graph! graph)]
          (ocl/await-event! token)
          (is (identical? primary (error-of #(ocl/read-graph-timestamps! graph))))
          (is (some #(identical? secondary %) (.getSuppressed primary)))
          (is (.isAlive (.scope (:arena token))))
          (.close (:arena token)))))))

(deftest queue-acquisition-errors-retain-an-explicit-owner
  (doseq [options [{:null-queue? true} {:create-failure (ex-info "create outcome unknown" {})}]]
    (mocked-recording options
                      (fn [calls]
                        (let [adopted (atom nil)
                              result (error-of #(ocl/record-graph! [] {:profile? true :adopt-cleanup! (fn [owner] (reset! adopted owner))}))]
                          (is (some? result))
                          (is (= [:queue :runtime-root-lease] (cleanup/pending @adopted)))
                          (is (= 1 (root/lease-count @(ns-resolve 'raster.gpu.ocl-runtime 'state))))
                          (is (empty? @calls) "do not pass NULL/unknown handles to a native destructor"))))))

(deftest common-session-cannot-free-kernels-or-roots-after-unknown-submit-drain
  (let [primary (ex-info "flush failed" {}) drain (ex-info "completion unknown" {})
        buffer {:dtype :float :n-elements 2 :byte-size 8}
        allocation (bview/allocation {:id :out :byte-size 8 :memory-space :shared
                                      :device :ocl:0 :coherence :host-coherent :ownership :owned})
        sess (atom {:device-id :ocl:0 :session-id :submit-failure :closed? false :events {}
                    :buffers {:out buffer} :allocations {:out allocation}
                    :kernel-graphs {} :prepared {} :graphs {}})
        destroyed (atom [])
        resolver (fn [_ name]
                   (case name
                     "register-kernel!" (fn [& _])
                     "bind-kernel-call" (fn [& _] {:bound {:wg 1} :group-count 1})
                     "record-graph!" ocl/record-graph!
                     "submit-graph!" ocl/submit-graph!
                     "destroy-graph!" ocl/destroy-graph!
                     "destroy-prepared!" (fn [_] (swap! destroyed conj :kernel))
                     "free-buffer!" (fn [_] (swap! destroyed conj :root))
                     (throw (ex-info "unexpected runtime call" {:name name}))))
        soft (fn [device name]
               (when (contains? #{"destroy-graph!" "destroy-prepared!"} name) (resolver device name)))]
    (mocked-recording {:failures {"clFlush" primary "clFinish" drain}}
      (fn [calls]
        (with-redefs-fn {(ns-resolve 'raster.gpu.core 'rt-resolve) resolver
                        (ns-resolve 'raster.gpu.core 'rt-resolve-soft) soft}
          (fn []
            (let [handle (gpu/bind-kernel-call! sess :fault (probe/emit-artifact :float :opencl-portable)
                                                [:out] {:profile? true})]
              (is (identical? primary (error-of #(gpu/submit-kernel-graph! sess handle))))
              (is (empty? (:events @sess)) "submission failed before a public event existed")
              (is (some? (error-of #(gpu/free-buffer! sess :out))))
              (let [before @calls]
                (dotimes [_ 2]
                  (is (identical? drain (error-of #(gpu/close-session! sess)))))
                (is (= before @calls)))
              (is (empty? @destroyed))
              (is (= #{:out} (set (keys (:buffers @sess)))))
              (let [submission-state (get-in @sess [:kernel-graphs :fault :runtime-graph :submission-state])
                    arena (:resource @(:arena-slot @submission-state))]
                (is (.isAlive (.scope arena)))
                (.close arena)))))))))
