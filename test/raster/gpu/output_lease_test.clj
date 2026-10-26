(ns raster.gpu.output-lease-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.link-plan :as plan]
            [raster.compiler.ir.link-plan-test :as plan-fixture]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.test-lifecycle]
            [raster.gpu.link :as link]
            [raster.gpu.value :as value]))

(defn- executable
  ([] (executable :owned true))
  ([ownership owns-session?]
   ;; Pure canonical geometry/effect proof, with replay/release/output projection
   ;; mocked by the lifecycle tests below. This is not a native execution fixture.
   (let [original (#'plan-fixture/valid-plan)
         nodes (assoc-in (:nodes original) [:out :view :allocation :ownership] ownership)
         {:keys [plan effect-evidence]}
         (plan/make-with-effect-evidence {:id :resident-result :target (:target original)
                                          :nodes nodes :instances (:instances original) :outputs [:out]})
         descriptor (get-in nodes [:out :view])
         buffer {:dtype :float :n-elements 16 :byte-size 64}
         session (atom {:session-id :lease-fixture :closed? false
                        :buffers {:out buffer} :allocations {:out (:allocation descriptor)}})]
   (raster.gpu.test-lifecycle/linked-executable
    {:plan plan :node-views {:out (gpu/->ResidentBufferView :lease-fixture :out descriptor)}
     :session session :owns-session? owns-session? :graph-key :graph
     :pending-inputs (atom #{}) :tainted-inputs (atom #{}) :closed? (atom false)
     :lifetime-lock (Object.) :execution-state (atom {:value-epoch 0 :effect-evidence effect-evidence})
     :completed-replays (atom 0) :output-leases (atom 0) :output-ready? (atom false)}))))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest absent-resident-boundary-is-not-an-empty-successful-output
  (let [executable (assoc (executable) :plan nil)]
    (doseq [access [#(link/node-view executable :out)
                    #(link/value-view executable :out)
                    #(link/outputs executable)
                    #(link/output-values executable)]]
      (is (= :link-runtime-boundary-unavailable (reason access))))
    (is (zero? @(:completed-replays executable)))
    (is (false? @(:output-ready? executable)))
    (is (zero? @(:output-leases executable)))))

(deftest completed-boundary-requires-current-effect-owner-and-physical-geometry
  (let [executable (executable)
        original @(:session executable)
        evidence (:effect-evidence @(:execution-state executable))]
    (is (= :link-completed-boundary-unavailable (reason #(link/completed-boundary executable))))
    (with-redefs [gpu/replay! (fn [& _]) link/outputs (fn [_] {:out :resident-view})]
      (link/run! executable))
    (let [boundary (link/completed-boundary executable)]
      (is (identical? (:plan executable) (:plan boundary)))
      (is (identical? evidence (:effect-evidence boundary)))
      (is (= [1 1] [(:epoch boundary) (:replay boundary)])))
    (swap! (:execution-state executable) assoc :effect-evidence (with-meta evidence (meta evidence)))
    (is (= :link-completed-boundary-effects (reason #(link/completed-boundary executable))))
    (swap! (:execution-state executable) assoc :effect-evidence evidence)
    (swap! (:session executable) assoc-in [:buffers :out :byte-size] 0)
    (is (= :link-completed-boundary-geometry (reason #(link/completed-boundary executable))))
    (reset! (:session executable) original)
    (#'link/begin-mutation! executable)
    (is (= :link-completed-boundary-unavailable (reason #(link/completed-boundary executable))))))

(deftest ordered-capability-and-replay-count-cannot-fabricate-a-completed-boundary
  ;; Lifecycle double deliberately supplies no source or physical plan. Capability
  ;; preflight must not manufacture either merely to enter the invocation callback.
  (let [executable (assoc (executable) :plan nil)
        calls (atom 0)]
    (swap! (:execution-state executable) assoc :ordered {:leaseable-output? true})
    (is (= :link-runtime-boundary-unavailable
           (reason #(link/execute-and-lease!
                     executable :fake-completion
                     (fn [] (swap! calls inc)
                       (swap! (:completed-replays executable) inc)
                       (reset! (:output-ready? executable) true))))))
    (is (= 1 @calls))
    (is (zero? @(:output-leases executable)))))

(deftest completed-output-specializes-only-an-unspecified-device-to-the-plan-target
  (let [original (executable)
        old-plan (:plan original)
        {:keys [plan effect-evidence]}
        (plan/make-with-effect-evidence
         {:id :unspecified-device :target (:target old-plan)
          :nodes (update-vals (:nodes old-plan) #(assoc-in % [:view :allocation :device] nil))
          :instances (:instances old-plan) :outputs (:outputs old-plan)})
        executable (assoc original :plan plan)]
    (swap! (:execution-state executable) assoc :effect-evidence effect-evidence)
    (with-redefs [gpu/replay! (fn [& _]) link/outputs (fn [_] {:out :resident-view})]
      (link/run! executable))
    (is (identical? plan (:plan (link/completed-boundary executable))))
    (let [allocation (assoc (get-in @(:session executable) [:allocations :out]) :device :ocl:0)
          resident (assoc-in (get-in executable [:node-views :out]) [:view :allocation] allocation)
          changed (assoc-in executable [:node-views :out] resident)]
      (swap! (:session executable) assoc-in [:allocations :out] allocation)
      (is (= :link-completed-boundary-geometry (reason #(link/completed-boundary changed)))))))

(deftest output-lease-deref-refuses-an-invalidated-runtime-epoch
  (let [executable (executable)]
    (with-redefs [gpu/replay! (fn [& _])
                  link/outputs (fn [_] {:out :resident-view})
                  link/output-values (fn [_] {:out :resident-view})]
      (link/run! executable)
      (let [lease (link/output-lease! executable)]
        (try
          ;; Simulate illicit out-of-band invalidation; public mutations are
          ;; independently checked above/below to refuse while a lease is live.
          (#'link/begin-mutation! executable)
          (is (= :link-output-lease-stale (reason #(deref lease))))
          (finally (.close ^java.io.Closeable lease)))
        (is (= :link-output-lease-released (reason #(deref lease))))))))

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
  (let [executable (executable)
        failure (ex-info "failed" {})
        replays (atom 0)
        closes (atom 0)]
    (with-redefs [gpu/replay! (fn [& _] (swap! replays inc) (throw failure))
                  gpu/close-session! (fn [& _] (swap! closes inc))]
      (is (identical? failure (try (link/run! executable)
                                  (catch Throwable error error))))
      (is (zero? @(:completed-replays executable)))
      (is (false? @(:output-ready? executable)))
      (is (= 1 (:value-epoch @(:execution-state executable))))
      (doseq [operation [#(link/run! executable)
                         #(link/upload! executable :out (double-array 1))
                         #(link/profile! executable)
                         #(link/measure! executable)
                         #(link/output-lease! executable)]]
        (let [error (try (operation) (catch Throwable error error))]
          (is (= :link-execution-poisoned (:reason (ex-data error))))
          (is (identical? failure (.getCause ^Throwable error)))))
      (is (= 1 @replays))
      (link/close! executable)
      (link/close! executable)
      (is (= 1 @closes)))))

(deftest exclusive-offline-mutation-invalidates-without-publishing-a-plan-result
  (let [executable (executable)
        calls (atom 0)]
    (with-redefs [gpu/replay! (fn [& _] nil)
                  link/outputs (fn [_] {:out :resident-view})
                  link/output-values (fn [_] {:out :resident-view})]
      (link/run! executable)
      (with-open [lease (link/output-lease! executable)]
        (is (= :link-output-lease-active
               (reason #(link/with-exclusive-mutation! executable :tune
                                                      (fn [] (swap! calls inc))))))
        (is (= 1 (:value-epoch @(:execution-state executable))))
        (is (true? @(:output-ready? executable))))
      (is (= :selected
             (link/with-exclusive-mutation!
              executable :tune
              #(do (swap! calls inc)
                   (link/run! executable)
                   (is (= :link-mutation-scope-active
                          (reason (fn [] (link/output-lease! executable)))))
                   :selected))))
      (is (= 1 @calls))
      (is (= 4 (:value-epoch @(:execution-state executable))))
      (is (= 2 @(:completed-replays executable)))
      (is (false? @(:output-ready? executable)))
      (is (= :link-output-lease-before-replay (reason #(link/output-lease! executable)))))))

(deftest failed-exclusive-mutation-poisons-with-the-original-cause
  (let [executable (executable)
        failure (ex-info "candidate validation failed after writes" {})]
    (is (identical? failure
                    (try (link/with-exclusive-mutation! executable :tune #(throw failure))
                         (catch Throwable error error))))
    (is (false? @(:output-ready? executable)))
    (is (nil? (:exclusive-mutation? @(:execution-state executable))))
    (is (= 1 (:value-epoch @(:execution-state executable))))
    (is (= :link-execution-poisoned (reason #(link/run! executable))))))

(deftest caught-nested-replay-failure-retains-the-first-poison
  (let [executable (executable)
        failure (ex-info "inner replay failed" {})
        closes (atom 0)]
    (with-redefs [gpu/replay! (fn [& _] (throw failure))
                  gpu/close-session! (fn [& _] (swap! closes inc))]
      (let [error (try (link/with-exclusive-mutation!
                       executable :tune #(try (link/run! executable) (catch Throwable _)))
                      (catch Throwable error error))]
        (is (= :link-execution-poisoned (:reason (ex-data error))))
        (is (identical? failure (.getCause ^Throwable error)))
        (is (identical? failure (:failure @(:execution-state executable))))
        (is (nil? (:exclusive-mutation? @(:execution-state executable))))
        (let [later (try (link/run! executable) (catch Throwable error error))]
          (is (identical? failure (.getCause ^Throwable later))))
        (link/close! executable)
        (link/close! executable)
        (is (= 1 @closes))))))

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

(deftest compiled-invoke-lease-pins-device-values-and-preflights-ownership
  (let [executable (executable)
        compiled (compiled/map->Compiled
                  {:executable executable :in-tree []
                   :out-tree [{:key :out :node :out}] :donated {}
                   :target :ocl:0 :live-outputs (atom nil)})
        replays (atom 0)
        frees (atom 0)]
    (with-redefs-fn
      {#'gpu/replay! (fn [& _] (swap! replays inc))
       #'link/outputs (fn [_] {:out :resident-view})
       #'link/output-values (fn [_] {:out :resident-view})
       #'value/free! (fn [_] (swap! frees inc))
       #'compiled/project-node (fn [& _] :device-result)}
      (fn []
        (with-open [lease (compiled/invoke-leased compiled {})]
          (is (= {:out :device-result} @lease))
          (is (= 1 @replays))
          (is (= :link-output-lease-active
                 (reason #(compiled/invoke-compiled compiled {}))))
          (is (zero? @frees)))
        (is (= 1 @frees) "closing the compiled lease invalidates its external wrapper")
        (is (= :compiled-output-lease-released
               (let [lease (compiled/invoke-leased compiled {})]
                 (.close ^java.io.Closeable lease)
                 (reason #(deref lease)))))
        (is (= 2 @replays)))))
  (let [executable (executable :borrowed true)
        compiled (compiled/map->Compiled
                  {:executable executable :in-tree []
                   :out-tree [{:key :out :node :out}] :donated {}
                   :live-outputs (atom nil)})
        calls (atom 0)]
    (with-redefs [gpu/replay! (fn [& _] (swap! calls inc))]
      (is (= :link-output-lease-ownership
             (reason #(compiled/invoke-leased compiled {}))))
      (is (zero? @calls))))
  (let [executable (executable)
        compiled (compiled/map->Compiled
                  {:executable executable :in-tree []
                   :out-tree [{:key :leak :node :internal}] :donated {}
                   :live-outputs (atom nil)})
        calls (atom 0)]
    (with-redefs [gpu/replay! (fn [& _] (swap! calls inc))]
      (is (= :compiled-output-lease-boundary
             (reason #(compiled/invoke-leased compiled {}))))
      (is (zero? @calls)))))

(deftest compiled-projection-and-lease-acquisition-are-atomic-against-replay
  (let [executable (executable)
        compiled (compiled/map->Compiled
                  {:executable executable :in-tree []
                   :out-tree [{:key :out :node :out}] :donated {}
                   :target :ocl:0 :live-outputs (atom nil)})
        entered (promise)
        release (promise)]
    (with-redefs-fn
      {#'gpu/replay! (fn [& _] nil)
       #'link/outputs (fn [_] {:out :resident-view})
       #'link/output-values (fn [_] {:out :resident-view})
       #'value/free! (fn [_] nil)
       #'compiled/project-node (fn [& _] (deliver entered true) @release :device-result)}
      (fn []
        (let [invocation (future (compiled/invoke-leased compiled {}))]
          (try
            (is (= true (deref entered 5000 ::timeout)))
            (let [racing (future (reason #(link/run! executable)))]
              (is (= ::timeout (deref racing 100 ::timeout)))
              (deliver release true)
              (with-open [lease (deref invocation 5000 ::timeout)]
                (is (= {:out :device-result} @lease))
                (is (= :link-output-lease-active (deref racing 5000 ::timeout)))))
            (finally (deliver release true))))))))

(deftest leased-execution-requires-exactly-one-replay
  (let [executable (executable)]
    (is (= :link-leased-invocation-replay-count
           (reason #(link/execute-and-lease! executable :test (fn [] :no-replay)))))
    (is (zero? @(:output-leases executable)))))
