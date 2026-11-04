(ns raster.gpu.cuda-session-contract-test
  "Deferred public session contracts from the 063e CUDA contribution, exercised
  against the current shared session/runtime owners. Only native transport and
  scoped test admission are substituted; global CUDA registration stays disabled.
  Run serially in an isolated JVM, never alongside other session tests."
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.cuda-driver :as driver]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root]
            [raster.gpu.runtime-backend :as backend]
            [raster.gpu.core :as gpu])
  (:import [java.lang.foreign Arena MemorySegment]))

(defn- with-cuda-session-admission [f]
  ;; Test-local dispatch only. Session allocation, event settlement and cleanup
  ;; remain real production code; every other device retains its original policy.
  (let [original-type backend/backend-type
        original-descriptor backend/descriptor
        descriptor {:namespace 'raster.gpu.cuda-runtime
                    :kernel-body-c-dialect :cuda
                    :memory-space :device :coherence :explicit-transfer
                    :owned-slice? true}]
    (with-redefs [backend/backend-type
                  (fn [device] (if (= :cuda:0 device) :cuda (original-type device)))
                  backend/descriptor
                  (fn [device] (if (= :cuda:0 device) descriptor
                                  (original-descriptor device)))]
      (f))))

(defn- with-device [f]
  (with-cuda-session-admission
    (fn []
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
            (with-redefs [runtime/state (atom {}) runtime/kernel-registry (atom {})]
              (f {:operations operations :pinned pinned :queued queued :finish finish :fault fault}))))))))

(defn- compile-only? [device]
  (try (backend/backend-type device) false
       (catch clojure.lang.ExceptionInfo error
         (true? (:compile-only? (ex-data error))))))

(deftest scoped-admission-preserves-global-and-other-device-policy
  (let [descriptors (into {} (for [device [:ze:0 :ocl:0]]
                              [device (backend/descriptor device)]))]
    (is (compile-only? :cuda:0))
    (with-cuda-session-admission
      (fn []
        (is (= :cuda (backend/backend-type :cuda:0)))
        (is (= 'raster.gpu.cuda-runtime (backend/runtime-namespace :cuda:0)))
        (is (compile-only? :cuda:1))
        (is (compile-only? :hip:0))
        (doseq [[device descriptor] descriptors]
          (is (= descriptor (backend/descriptor device))))))
    (is (compile-only? :cuda:0))))


(deftest public-session-close-awaits-download-and-frees-transfer-staging
  (with-device
    (fn [{:keys [pinned]}]
      (let [session (gpu/make-session :cuda:0) input (float-array [1 2 3 4]) output (float-array [-7 -7 -7 -7])]
        (gpu/alloc! session {:x [:float 4 nil]})
        (gpu/submit-upload-ranges! session [[:x input {:elements 4}]])
        (gpu/submit-download-ranges! session [[:x output {:elements 4}]])
        (gpu/close-session! session)
        (is (= [1.0 2.0 3.0 4.0] (vec output)))
        (is (empty? @pinned))
        (is (= :closed (:lifecycle @session)))
        (is (= 0 (root/lease-count runtime/state)))))))

(deftest host-publication-error-preserves-operation-failure-but-retires-native-resources
  (with-device
    (fn [{:keys [pinned]}]
      (let [session (gpu/make-session :cuda:0) host (float-array [1 2 3 4])
            destination-arena (Arena/ofShared) destination (.allocate destination-arena (long 16))]
        (gpu/alloc! session {:x [:float 4 nil]})
        (gpu/release-event! session (gpu/submit-upload-ranges! session [[:x host {:elements 4}]]))
        (let [event (gpu/submit-download-ranges! session [[:x destination {:elements 4}]])]
          (.close destination-arena)
          (let [failure (try (gpu/await-event! session event) nil (catch Throwable e e))]
            (is (instance? IllegalStateException failure))
            (is (empty? @pinned) "completed pinned staging is retired despite the host-copy operation failure")
            (is (= 1 (root/lease-count runtime/state)))
            (is (empty? (cleanup/pending (::cleanup/owner (get-in @session [:events (:id event)])))))
            (is (nil? (try (gpu/close-session! session) nil (catch Throwable e e))))
            (is (= :closed (:lifecycle @session)))
            (is (= 0 (root/lease-count runtime/state)))))))))

(deftest public-await-retains-unknown-completion-with-one-native-attempt
  (with-device
    (fn [{:keys [operations pinned fault]}]
      (let [session (gpu/make-session :cuda:0) output (float-array [-7 -7 -7 -7])
            failure (ex-info "completion outcome unknown" {})]
        (gpu/alloc! session {:x [:float 4 nil]})
        (let [event (gpu/submit-download-ranges! session [[:x output {:elements 4}]])]
          (reset! fault {:event-sync {:error failure}})
          (is (identical? failure (try (gpu/await-event! session event) nil (catch Throwable error error))))
          (is (= 1 (count (filter #(= :event-sync (first %)) @operations))))
          (is (= [-7.0 -7.0 -7.0 -7.0] (vec output)))
          (is (= 1 (count @pinned)))
          (is (= 2 (root/lease-count runtime/state)))
          (is (seq (cleanup/pending (::cleanup/owner (get-in @session [:events (:id event)])))))
          (reset! fault nil)
          (let [before (count @operations)]
            (is (identical? failure (try (gpu/release-event! session event) nil (catch Throwable error error))))
            (is (identical? failure (try (gpu/close-session! session) nil (catch Throwable error error))))
            (is (= before (count @operations))))
          (is (= [-7.0 -7.0 -7.0 -7.0] (vec output)))
          (is (= 1 (count @pinned)))
          (is (= 2 (root/lease-count runtime/state))))))))

(deftest public-await-explicitly-retry-safe-completion-can-retire-after-retry
  (with-device
    (fn [{:keys [operations pinned fault]}]
      (let [session (gpu/make-session :cuda:0) output (float-array [-7 -7 -7 -7])
            failure (ex-info "completion retry allowed" {:cleanup-retry-safe? true})]
        (gpu/alloc! session {:x [:float 4 nil]})
        (let [event (gpu/submit-download-ranges! session [[:x output {:elements 4}]])]
          (reset! fault {:event-sync {:error failure}})
          (is (identical? failure (try (gpu/await-event! session event) nil (catch Throwable error error))))
          (is (= [-7.0 -7.0 -7.0 -7.0] (vec output)))
          (is (= 1 (count @pinned)))
          (let [attempts (count (filter #(= :event-sync (first %)) @operations))]
            (reset! fault nil)
            (is (identical? failure (try (gpu/release-event! session event) nil (catch Throwable error error))))
            (is (= (inc attempts) (count (filter #(= :event-sync (first %)) @operations)))))
          (is (= [0.0 0.0 0.0 0.0] (vec output)))
          (is (empty? @pinned))
          (is (= 1 (root/lease-count runtime/state)))
          (gpu/close-session! session)
          (is (= 0 (root/lease-count runtime/state))))))))
