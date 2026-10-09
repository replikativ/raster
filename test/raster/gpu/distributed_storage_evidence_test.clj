(ns raster.gpu.distributed-storage-evidence-test
  "Explicit synthetic issuer/backend fault fixtures, not authenticated compiler executions."
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.distributed :as runtime]
            [raster.gpu.storage-evidence-test :as fixture]))

(defn- owner [session state budget]
  (#'runtime/seal-runtime-value
   (runtime/map->DistributedExecutable
    {:plan {:id :unit :outputs []} :sessions {:unit session} :state (atom state)
     :allocation-budgets {:unit budget}})))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- error-of [f] (try (f) nil (catch Throwable e e)))

(deftest distributed-facts-are-original-owner-bound-and-distinct-from-local-evidence
  (let [c (#'fixture/owner)]
    (#'fixture/simulated c {}
      (fn [trace session]
        (let [executable (owner session :complete {:capacity-bytes 64 :resident-bytes 16})
              fact (runtime/measure-storage-representation! executable :unit :f32)]
          (is (runtime/original-executable? executable))
          (is (runtime/representation-evidence? fact))
          (is (identical? executable (:executable fact)))
          (is (identical? session (:session fact)))
          (is (= (:session-id @session) (:session-id fact)))
          (is (= :float (:dtype @fact)))
          (is (= :little-endian (:byte-order @fact)))
          (is (= :raster.distributed/resident-representation-v1 (:kind @fact)))
          (is (= :unit (:target @fact)))
          (is (not (compiled/representation-evidence? fact)))
          (is (not (runtime/representation-evidence? (assoc fact :data @fact))))
          (is (not (runtime/representation-evidence?
                    (runtime/map->ResidentRepresentationEvidence (into {} fact)))))
          (is (= :distributed-representation-owner (reason #(deref (assoc fact :data @fact)))))
          (is (= :distributed-representation-owner
                 (reason #(deref (#'compiled/seal-artifact fact)))))
          (is (= :complete @(:state executable)))
          (is (= :distributed-runtime-state
                 (reason #(runtime/storage-representation-description executable :unit :float fact))))
          (runtime/with-output-values!
           executable
           (fn [_]
             (is (= @fact (runtime/storage-representation-description executable :unit :float fact)))
             (doseq [[target dtype f] [[:unit :int fact] [:foreign :float fact]
                                      [:unit :float @fact] [:unit :float (assoc fact :data @fact)]]]
               (is (= :distributed-representation-mismatch
                      (reason #(runtime/storage-representation-description executable target dtype f)))))))
          (is (= [:allocate :bind :run :download :classify :release-graph :free] @trace))
          (is (empty? (:buffers @session)))
          (is (empty? (:kernel-graphs @session))))))))

(deftest admission-declines-never-run-the-probe-or-consume-completed-outputs
  (doseq [scenario [:copy :unissued :ready :failed :closed :target :budget :events :dtype]]
    (let [c (#'fixture/owner)]
      (#'fixture/simulated c {}
        (fn [trace session]
          (let [state (if (contains? #{:ready :failed :closed} scenario) scenario :complete)
                budget {:capacity-bytes (if (= :budget scenario) 23 64) :resident-bytes 16}
                original (owner session state budget)
                executable (case scenario
                             :copy (assoc original :plan (:plan original))
                             :unissued (runtime/map->DistributedExecutable (into {} original))
                             original)]
            (when (= :events scenario) (swap! session assoc :events {:pending :unit}))
            (is (= (case scenario
                     (:copy :unissued) :distributed-runtime-owner
                     (:ready :failed :closed) :distributed-runtime-state
                     :target :distributed-representation-target
                     :budget :distributed-representation-budget
                     :events :storage-representation-unready
                     :dtype :storage-representation-unsupported)
                   (reason #(runtime/measure-storage-representation!
                             executable (if (= :target scenario) :foreign :unit)
                             (if (= :dtype scenario) :half :float)))))
            (is (= state @(:state original)))
            (is (empty? @trace))))))))

(deftest probe-failure-poisons-owner-only-when-the-exclusive-scope-ends
  (doseq [stage [:allocate :bind :run :download :classify :release-graph :free]]
    (let [c (#'fixture/owner) failure (ex-info "probe failed" {:stage stage})]
      (#'fixture/simulated c {stage failure}
        (fn [_ session]
          (let [executable (owner session :complete {:capacity-bytes 64 :resident-bytes 16})]
            (is (identical? failure
                            (error-of #(runtime/measure-storage-representation! executable :unit :float))))
            (is (= :failed @(:state executable)))
            (is (= :distributed-runtime-state
                   (reason #(runtime/with-output-values! executable identity))))
            (is (empty? (:buffers @session)))
            (is (empty? (:kernel-graphs @session)))))))))

(deftest facts-cannot-cross-owner-or-live-session-identity-boundaries
  (let [c (#'fixture/owner)]
    (#'fixture/simulated c {}
      (fn [_ session]
        (let [budget {:capacity-bytes 64 :resident-bytes 16}
              executable (owner session :complete budget)
              foreign (owner session :complete budget)
              fact (runtime/measure-storage-representation! executable :unit :float)
              validate #(runtime/storage-representation-description executable :unit :float fact)
              original @session]
          (runtime/with-output-values!
           foreign
           (fn [_]
             (is (= :distributed-representation-mismatch
                    (reason #(runtime/storage-representation-description foreign :unit :float fact))))))
          (doseq [alter [#(assoc % :session-id (random-uuid))
                         #(assoc % :events {:pending :unit})
                         #(assoc % :closed? true)]]
            (runtime/with-output-values!
             executable
             (fn [_]
               (swap! session alter)
               (try (is (= :distributed-representation-mismatch (reason validate)))
                    (finally (reset! session original))))))
          (runtime/with-output-values!
           executable
           (fn [_]
             (with-redefs [raster.gpu.core/execution-device-info (constantly {:driver {:version 8}})]
               (is (= :distributed-representation-mismatch (reason validate))))))
          (is (= :complete @(:state executable))))))))
