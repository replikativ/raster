(ns raster.gpu.graph-submission-entry-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.core :as gpu]
            [raster.gpu.resource-cleanup :as cleanup]))

(defn- fixture []
  (let [entry {:generation 1 :runtime-graph :recorded :outputs {'out :resident}
               :profile? true :execution-plan {:queues [{:class :compute}]}
               :resident-footprint {:allocation-ids #{:output} :resident-buffers []}}
        session (atom {:device-id :ze:0 :session-id :entry-test :closed? false
                       :kernel-graphs {:graph entry} :events {}})]
    {:session session :handle (gpu/->KernelGraphHandle :graph :entry-test 1)}))

(defn- error-of [f]
  (try (f) nil (catch Throwable error error)))

(defn- with-runtime [f]
  (let [calls (atom [])
        resolutions (atom 0)
        resolve-entry (ns-resolve 'raster.gpu.core 'resolve-kernel-graph-entry)
        original @resolve-entry]
    (with-redefs-fn
      {resolve-entry (fn [& args] (swap! resolutions inc) (apply original args))
       (ns-resolve 'raster.gpu.core 'rt-resolve)
       (fn [_ name]
         (fn [& args]
           (swap! calls conj name)
           (case name
             "submit-graph!" :backend-event
             "event-complete?" true
             nil)))}
      #(f calls resolutions))))

(deftest synchronous-and-asynchronous-submission-admit-once-per-request
  (with-runtime
    (fn [calls resolutions]
      (let [{:keys [session handle]} (fixture)]
        (is (= {'out :resident} (gpu/run-kernel-graph! session handle)))
        (is (= 1 @resolutions))
        (is (= ["submit-graph!" "await-event!" "release-event!" "reset-graph-events!"] @calls))
        (is (empty? (:events @session)))
        (let [event (gpu/submit-kernel-graph! session handle)]
          (is (= 2 @resolutions))
          (gpu/release-event! session event))))))

(deftest foreign-stale-and-closed-handles-reject-before-backend-contact
  (doseq [operation [gpu/run-kernel-graph! gpu/submit-kernel-graph!]
          [change expected] [[#(assoc % :session-id :foreign) :foreign-graph-handle]
                             [#(assoc % :generation 0) :stale-graph-handle]]]
    (with-runtime
      (fn [calls _]
        (let [{:keys [session handle]} (fixture)]
          (is (= expected (:reason (ex-data (error-of #(operation session (change handle)))))))
          (is (empty? @calls))))))
  (with-runtime
    (fn [calls _]
      (let [{:keys [session handle]} (fixture)]
        (swap! session assoc :closed? true)
        (is (= :session-releasing (:reason (ex-data (error-of #(gpu/run-kernel-graph! session handle))))))
        (is (empty? @calls))))))

(deftest observed-completion-does-not-permit-replay-before-await
  (with-runtime
    (fn [calls _]
      (let [{:keys [session handle]} (fixture)
            event (gpu/submit-kernel-graph! session handle)]
        (is (true? (gpu/event-complete? session event)))
        (is (re-find #"in-flight" (.getMessage ^Throwable (error-of #(gpu/run-kernel-graph! session handle)))))
        (is (= 1 (count (filter #{"submit-graph!"} @calls))))
        (gpu/await-event! session event)
        (gpu/release-event! session event)
        (is (= {'out :resident} (gpu/run-kernel-graph! session handle)))))))

(deftest synchronous-failures-preserve-identity-and-event-drain
  (doseq [failure-phase ["submit-graph!" "await-event!" "reset-graph-events!"]]
    (let [{:keys [session handle]} (fixture)
          failure (ex-info "injected runtime failure" {:phase failure-phase})
          failed? (atom false)
          calls (atom [])]
      (with-redefs-fn
        {(ns-resolve 'raster.gpu.core 'rt-resolve)
         (fn [_ name]
           (fn [& _]
             (swap! calls conj name)
             (when (and (= name failure-phase) (compare-and-set! failed? false true))
               (throw failure))
             (when (= name "submit-graph!") :backend-event)))}
        (fn []
          (is (identical? failure (error-of #(gpu/run-kernel-graph! session handle))))
          (is (empty? (:events @session)))
          (is (= (if (= failure-phase "submit-graph!") 0 1)
                 (count (filter #{"release-event!"} @calls))))
          (is (= {'out :resident} (gpu/run-kernel-graph! session handle))))))))

(deftest retired-owner-cannot-submit-through-either-boundary
  (doseq [operation [gpu/run-kernel-graph! gpu/submit-kernel-graph!]]
    (with-runtime
      (fn [calls _]
        (let [{:keys [session handle]} (fixture)
              owner (cleanup/owner [])]
          (swap! session assoc-in [:kernel-graphs :graph ::cleanup/owner] owner)
          (cleanup/release! owner)
          (is (some? (error-of #(operation session handle))))
          (is (empty? @calls)))))))

(deftest synchronous-replay-retains-transfer-footprint-admission
  (doseq [[footprint transfer-allocation expected]
          [[nil :output :graph-resident-footprint-missing]
           [{:allocation-ids #{:output} :resident-buffers []} :output :graph-pending-transfer]
           [{:allocation-ids #{:output} :resident-buffers []} :other nil]]]
    (with-runtime
      (fn [calls _]
        (let [{:keys [session handle]} (fixture)]
          (swap! session assoc-in [:kernel-graphs :graph :resident-footprint] footprint)
          (swap! session assoc-in [:events :transfer]
                 {:kind :transfer :status :pending :allocation-ids #{transfer-allocation}
                  :resident-buffers []})
          (if expected
            (do (is (= expected (:reason (ex-data (error-of #(gpu/run-kernel-graph! session handle))))))
                (is (empty? @calls)))
            (is (= {'out :resident} (gpu/run-kernel-graph! session handle)))))))))
