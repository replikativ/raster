(ns raster.gpu.distributed-output-scope-test
  "Hardware-free lifecycle checks; these do not fabricate compiler completion evidence."
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.distributed :as distributed]))

(defn- owner [state]
  (distributed/map->DistributedExecutable
   {:plan {:outputs []} :sessions {} :state (atom state)}))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest only-completed-owners-enter-the-read-scope
  (doseq [state [:ready :running :failed :closed]]
    (let [executable (owner state) called? (atom false)]
      (is (= :distributed-runtime-state
             (reason #(distributed/with-output-values! executable
                        (fn [_] (reset! called? true))))))
      (is (false? @called?))
      (is (= state @(:state executable))))))

(deftest callback-close-and-nested-scope-cannot-invalidate-reads
  (let [executable (owner :complete)]
    (is (= :copied
           (distributed/with-output-values!
            executable
            (fn [values]
              (is (= {} values))
              (is (= :reading-outputs @(:state executable)))
              (is (= :distributed-runtime-output-scope-active
                     (reason #(distributed/close! executable))))
              (is (= :distributed-runtime-state
                     (reason #(distributed/run! executable))))
              (is (= :distributed-runtime-state
                     (reason #(distributed/with-output-values! executable identity))))
              :copied))))
    (is (= :complete @(:state executable)))
    (distributed/close! executable)
    (is (= :closed @(:state executable)))
    (is (nil? (distributed/close! executable)))))

(deftest callback-failure-releases-the-read-scope
  (let [executable (owner :complete)
        failure (ex-info "capture failed" {:reason :capture-failed})]
    (is (identical? failure
                    (try (distributed/with-output-values! executable (fn [_] (throw failure)))
                         (catch Throwable e e))))
    (is (= :complete @(:state executable)))
    (is (= {} (distributed/with-output-values! executable identity)))
    (distributed/close! executable)
    (is (= :closed @(:state executable)))))

(deftest provider-worker-close-is-refused-without-monitor-deadlock
  (let [executable (owner :complete)
        worker (atom nil)]
    (distributed/with-output-values!
     executable
     (fn [_]
       (reset! worker (future (reason #(distributed/close! executable))))
       (is (= :distributed-runtime-output-scope-active
              (deref @worker 5000 :timeout)))
       (is (= :reading-outputs @(:state executable)))
       (is (= :distributed-runtime-state
              (deref (future (reason #(distributed/run! executable))) 5000 :timeout)))))
    (is (= :complete @(:state executable)))
    (distributed/close! executable)
    (is (= :closed @(:state executable)))))
