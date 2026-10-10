(ns raster.gpu.graph-submission-entry-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.core :as gpu]
            [raster.gpu.invocation-observation :as observation]
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

(deftest replay-details-do-not-double-count-or-add-backend-operations
  (doseq [profile? [false true]]
    (with-runtime
      (fn [calls resolutions]
        (let [{:keys [session handle]} (fixture)
              _ (swap! session assoc-in [:kernel-graphs :graph :profile?] profile?)
              {:keys [outputs report]}
              (observation/observe
               #(observation/phase :replay-host (gpu/run-kernel-graph! session handle)))]
          (is (= {'out :resident} outputs))
          (is (= 1 @resolutions))
          (is (= (cond-> ["submit-graph!" "await-event!" "release-event!"]
                   profile? (conj "reset-graph-events!")) @calls))
          (is (empty? (:events @session)))
          (is (= (cond-> #{:graph-resolution :submission :await :event-release}
                   profile? (conj :profile-reset)) (set (keys (:replay-detail-ns report)))))
          (is (every? #(<= 0 %) (vals (:replay-detail-ns report))))
          (is (<= (reduce + 0 (vals (:replay-detail-ns report)))
                  (get-in report [:phases-ns :replay-host])))
          (is (= (:total-ns report)
                 (+ (:unpartitioned-ns report) (reduce + 0 (vals (:phases-ns report))))))
          (is (nil? observation/*collector*)))))))

(deftest recorded-replay-reports-its-opaque-synchronous-boundary
  (with-runtime
    (fn [calls _]
      (let [{:keys [session]} (fixture)
            _ (swap! session assoc :graphs {:recorded :runtime-graph})
            {:keys [report]}
            (observation/observe
             #(observation/phase :replay-host (gpu/replay! session :recorded)))]
        (is (= ["replay-graph!"] @calls))
        (is (= #{:graph-resolution :synchronous-backend-replay}
               (set (keys (:replay-detail-ns report)))))
        (is (<= (reduce + 0 (vals (:replay-detail-ns report)))
                (get-in report [:phases-ns :replay-host])))
        (is (= (:total-ns report)
               (+ (:unpartitioned-ns report) (reduce + 0 (vals (:phases-ns report))))))
        (gpu/replay! session :recorded)
        (is (= ["replay-graph!" "replay-graph!"] @calls))))))

(deftest recorded-replay-observation-preserves-backend-exception
  (doseq [observed? [false true]]
    (let [{:keys [session]} (fixture)
          _ (swap! session assoc :graphs {:recorded :runtime-graph})
          failure (ex-info "recorded replay failed" {})
          calls (atom 0)
          collector (when observed? (volatile! {:phases-ns {} :replay-detail-ns {}}))]
      (with-redefs-fn
        {(ns-resolve 'raster.gpu.core 'rt-resolve)
         (fn [_ name]
           (is (= "replay-graph!" name))
           (fn [_] (swap! calls inc) (throw failure)))}
        #(binding [observation/*collector* collector]
           (is (identical? failure (error-of (fn [] (gpu/replay! session :recorded)))))))
      (is (= 1 @calls))
      (when observed?
        (is (= #{:graph-resolution :synchronous-backend-replay}
               (set (keys (:replay-detail-ns @collector)))))))))

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
  (doseq [failure-phase ["submit-graph!" "await-event!" "reset-graph-events!"]
          observed? [false true]]
    (let [{:keys [session handle]} (fixture)
          failure (ex-info "injected runtime failure" {:phase failure-phase})
          failed? (atom false)
          calls (atom [])
          collector (when observed? (volatile! {:phases-ns {} :replay-detail-ns {}}))]
      (with-redefs-fn
        {(ns-resolve 'raster.gpu.core 'rt-resolve)
         (fn [_ name]
           (fn [& _]
             (swap! calls conj name)
             (when (and (= name failure-phase) (compare-and-set! failed? false true))
               (throw failure))
             (when (= name "submit-graph!") :backend-event)))}
        (fn []
          (is (identical? failure
                          (binding [observation/*collector* collector]
                            (error-of #(gpu/run-kernel-graph! session handle)))))
          (when observed?
            (is (contains? (:replay-detail-ns @collector) :submission))
            (is (every? #(<= 0 %) (vals (:replay-detail-ns @collector)))))
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
