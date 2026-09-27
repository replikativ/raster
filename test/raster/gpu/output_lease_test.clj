(ns raster.gpu.output-lease-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.link-plan :as plan]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.value :as value]))

(defn- executable
  ([] (executable :owned true))
  ([ownership owns-session?]
   (link/map->LinkedExecutable
    {:plan {:id :resident-result :outputs [:out]
            :nodes {:out {:view {:allocation {:ownership ownership}}}}}
     :session (atom {}) :owns-session? owns-session? :graph-key :graph
     :pending-inputs (atom #{}) :closed? (atom false)
     :lifetime-lock (Object.) :completed-replays (atom 0)
     :output-leases (atom 0) :output-ready? (atom false)})))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest output-lease-pins-a-completed-replay-until-every-holder-releases
  (let [executable (executable)
        replays (atom 0)
        closes (atom 0)]
    (with-redefs [gpu/replay! (fn [& _] (swap! replays inc))
                  gpu/close-session! (fn [& _] (swap! closes inc))
                  gpu/upload-range! (fn [& _] nil)
                  plan/validate-node-source! (fn [& _] nil)
                  link/node-view (fn [_ _] :resident-view)
                  link/outputs (fn [_] {:out :resident-view})
                  link/output-values (fn [_] {:semantic-out :resident-view})]
      (is (= :link-output-lease-before-replay
             (reason #(link/output-lease! executable))))
      (is (= {:out :resident-view} (link/run! executable)))
      (let [first-lease (link/output-lease! executable)
            second-lease (link/output-lease! executable)]
        (is (= {:outputs {:out :resident-view}
                :values {:semantic-out :resident-view}
                :replay 1}
               @first-lease))
        (doseq [operation [#(link/run! executable)
                           #(link/upload! executable :out (double-array 1))
                           #(link/write! executable :out (double-array 1))
                           #(link/close! executable)]]
          (is (= :link-output-lease-active (reason operation))))
        (is (= 1 @replays))
        (is (zero? @closes))
        (.close ^java.io.Closeable first-lease)
        (.close ^java.io.Closeable first-lease)
        (is (= :link-output-lease-released (reason #(deref first-lease))))
        (is (= :link-output-lease-active (reason #(link/run! executable))))
        (.close ^java.io.Closeable second-lease)
        (link/upload! executable :out (double-array 1))
        (is (= :link-output-lease-before-replay
               (reason #(link/output-lease! executable))))
        (with-open [fresh (link/run-and-lease! executable)]
          (is (= {:out :resident-view} (:outputs @fresh)))
          (is (= 2 (:replay @fresh))))
        (is (= 2 @replays))
        (link/close! executable)
        (is (= 1 @closes))
        (is (= :link-output-lease-released (reason #(deref second-lease))))))))

(deftest output-lease-requires-owned-storage-and-session
  (let [replays (atom 0)]
    (with-redefs [gpu/replay! (fn [& _] (swap! replays inc))
                  link/outputs (fn [_] {:out :resident-view})
                  link/output-values (fn [_] {:semantic-out :resident-view})]
      (doseq [executable [(executable :borrowed true)
                          (executable :external true)
                          (executable :owned false)]]
        (reset! (:completed-replays executable) 1)
        (reset! (:output-ready? executable) true)
        (is (= :link-output-lease-ownership
               (reason #(link/output-lease! executable))))
        (is (= :link-output-lease-ownership
               (reason #(link/run-and-lease! executable)))))
      (is (zero? @replays)))))

(deftest failed-replay-cannot-create-an-output-lease
  (let [executable (executable)]
    (with-redefs [gpu/replay! (fn [& _] (throw (ex-info "failed" {})))]
      (is (thrown? clojure.lang.ExceptionInfo (link/run! executable)))
      (is (zero? @(:completed-replays executable)))
      (is (= :link-output-lease-before-replay
             (reason #(link/output-lease! executable))))
      (reset! (:completed-replays executable) 1)
      (reset! (:output-ready? executable) true)
      (is (thrown? clojure.lang.ExceptionInfo (link/run! executable)))
      (is (false? @(:output-ready? executable)))
      (is (= :link-output-lease-before-replay
             (reason #(link/output-lease! executable)))))))

(deftest lease-acquisition-waits-for-complete-synchronous-replay
  (let [executable (executable)
        entered (promise)
        release (promise)]
    (with-redefs [gpu/replay! (fn [& _] (deliver entered true) @release)
                  link/outputs (fn [_] {:out :resident-view})
                  link/output-values (fn [_] {:semantic-out :resident-view})]
      (let [running (future (link/run! executable))]
        (try
          (is (= true (deref entered 5000 ::timeout)))
          (let [leasing-entered (promise)
                leasing (future (deliver leasing-entered true)
                                (link/output-lease! executable))]
            (is (= true (deref leasing-entered 5000 ::timeout)))
            (is (= ::timeout (deref leasing 100 ::timeout)))
            (deliver release true)
            (is (= {:out :resident-view} (deref running 5000 ::timeout)))
            (let [lease (deref leasing 5000 ::timeout)]
              (is (= 1 (:replay @lease)))
              (.close ^java.io.Closeable lease)))
          (finally (deliver release true)))))))

(deftest compiled-wrapper-does-not-invalidate-a-leased-output-before-declining
  (let [executable (executable)
        releases (atom 0)
        donations (atom 0)
        closes (atom 0)
        replays (atom 0)
        compiled (compiled/map->Compiled
                  {:executable executable
                   :in-tree [{:key :state :role :state :node :state}]
                   :out-tree [] :donated {:state :out}
                   :target :ocl:0 :live-outputs (atom [:old-output])})]
    (with-redefs [gpu/replay! (fn [& _] (swap! replays inc))
                  gpu/close-session! (fn [& _] (swap! closes inc))
                  value/free! (fn [_] (swap! releases inc))
                  value/consume! (fn [_] (swap! donations inc))
                  link/outputs (fn [_] {:out :resident-view})
                  link/output-values (fn [_] {:semantic-out :resident-view})]
      (link/run! executable)
      (with-open [lease (link/output-lease! executable)]
        (is (= :link-output-lease-active
               (reason #(compiled/invoke-compiled compiled {:state :donated-value}))))
        (is (= :link-output-lease-active
               (reason #(compiled/close! compiled))))
        (is (= 1 (:replay @lease)))
        (is (= [:old-output] @(:live-outputs compiled)))
        (is (zero? @releases))
        (is (zero? @donations))
        (is (zero? @closes))
        (is (= 1 @replays)))
      (compiled/close! compiled)
      (is (= 1 @releases))
      (is (= 1 @closes)))))
