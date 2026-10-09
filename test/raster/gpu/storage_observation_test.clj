(ns raster.gpu.storage-observation-test
  "Raw observations reuse the existing backend fault fixture, not another runtime double."
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.storage-evidence-test :as fixture]
            [raster.gpu.storage-representation :as storage]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest raw-observation-is-data-not-owner-evidence
  (let [c (#'fixture/owner)]
    (#'fixture/simulated c {}
      (fn [trace session]
        (let [observed (storage/observe! session :f32)]
          (is (= :float (:dtype observed)))
          (is (= :little-endian (:byte-order observed)))
          (is (= (:session-id @session) (:session-id observed)))
          (is (string? (:device-fingerprint observed)))
          (is (string? (:probe-fingerprint observed)))
          (is (not (compiled/representation-evidence? observed)))
          (is (nil? (:program-fingerprint observed)))
          (is (= [:allocate :bind :run :download :classify :release-graph :free] @trace))
          (is (empty? (:buffers @session)))
          (is (empty? (:kernel-graphs @session)))
          (is (zero? (:value-epoch @(:execution-state (:executable c)))))
          (is (true? @(:output-ready? (:executable c)))))))))

(deftest admission-precedes-owner-mutation-wrapper
  (doseq [scenario [:unsupported :events]]
    (let [c (#'fixture/owner) called? (atom false)]
      (#'fixture/simulated c {}
        (fn [trace session]
          (when (= :events scenario) (swap! session assoc :events {:pending :unit}))
          (is (= (if (= :events scenario) :storage-representation-unready
                       :storage-representation-unsupported)
                 (reason #(storage/observe!
                           session (if (= :unsupported scenario) :half :float)
                           (fn [observe] (reset! called? true) (observe))))))
          (is (false? @called?))
          (is (empty? @trace)))))))

(deftest identity-drift-rejects-observation-after-cleanup
  (doseq [scenario [:session :device]]
    (let [c (#'fixture/owner)]
      (#'fixture/simulated c {}
        (fn [trace session]
          (let [run! gpu/run-kernel-graph!
                device-info gpu/execution-device-info
                changed? (atom false)]
            (with-redefs [gpu/run-kernel-graph!
                          (fn [& args]
                            (reset! changed? true)
                            (when (= :session scenario)
                              (swap! session assoc :session-id (random-uuid)))
                            (apply run! args))
                          gpu/execution-device-info
                          (fn [sess]
                            (cond-> (device-info sess)
                              (and @changed? (= :device scenario))
                              (assoc-in [:driver :version] 8)))]
              (is (= :storage-representation-device-changed
                     (reason #(storage/observe! session :float)))))
            (is (= [:release-graph :free] (vec (take-last 2 @trace))))
            (is (empty? (:buffers @session)))
            (is (empty? (:kernel-graphs @session)))))))))

(deftest owner-thunk-is-synchronous-one-shot-and-cannot-escape
  (let [c (#'fixture/owner)]
    (#'fixture/simulated c {}
      (fn [trace session]
        (let [retained (atom nil)]
          (is (= :storage-representation-wrapper
                 (reason #(storage/observe! session :float nil))))
          (is (= :storage-representation-not-observed
                 (reason #(storage/observe! session :float
                           (fn [observe] (reset! retained observe) :no-measurement)))))
          (is (= :storage-representation-scope (reason #(@retained))))
          (is (empty? @trace))
          (is (= :little-endian
                 (:byte-order
                  (storage/observe!
                   session :float
                   (fn [observe]
                     (reset! retained observe)
                     (is (= :storage-representation-scope
                            (deref (future (reason observe)) 5000 :timeout)))
                     (let [result (observe)]
                       (is (= :storage-representation-replayed (reason observe)))
                       result))))))
          (is (= :storage-representation-scope (reason #(@retained))))
          (is (= 1 (count (filter #{:allocate} @trace))))
          (is (empty? (:buffers @session)))
          (is (empty? (:kernel-graphs @session))))))))

(deftest wrapper-cannot-submit-async-work-before-invoking-the-probe
  (let [c (#'fixture/owner)]
    (#'fixture/simulated c {}
      (fn [trace session]
        (is (= :storage-representation-unready
               (reason #(storage/observe!
                         session :float
                         (fn [observe]
                           (swap! session assoc :events {:pending :unit})
                           (observe))))))
        (is (empty? @trace))))))
