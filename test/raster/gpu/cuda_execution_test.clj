(ns raster.gpu.cuda-execution-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.gpu.cuda-execution :as execution]
            [raster.gpu.cuda-prepared-execution :as pending]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root]))

(defn- fixture []
  (let [state (atom {}) context (Object.) calls (atom []) failure (atom nil)
        native! (fn [op args]
                  (swap! calls conj [op args])
                  (when (= op @failure) (throw (ex-info "mock native failure" {:op op})))
                  (case op :event-create (Object.) :event-query true :event-elapsed 1000 nil))]
    (root/initialize! state [] (fn [_] {:context context :stream :stream
                                        :prepared-execution (pending/ledger state)}))
    {:state state :calls calls :failure failure :native! native!
     :prepared (fn [name]
                 (cleanup/build!
                  (cleanup/owner [{:id :prepared :release (fn [] (swap! calls conj [:prepared-free name]))}])
                  (fn [] {:kernel-name name :function name :geometry {} :parameters []
                          :pending (pending/binding-state)
                          :raster.gpu.cuda-runtime/context context}) nil))}))

(deftest failed-standalone-launch-cannot-bypass-admission-through-replay
  (let [{:keys [state native! prepared calls failure]} (fixture)
        p (prepared "p") graph (execution/record-graph! state [p] {})
        ledger (:prepared-execution @state)]
    (reset! failure :launch)
    (is (thrown? clojure.lang.ExceptionInfo
                 (pending/submit! ledger (:pending p) (:stream @state) native!
                                  (fn [] nil) :launch [] true)))
    (reset! failure nil)
    (let [before @calls]
      (is (thrown? clojure.lang.ExceptionInfo (execution/submit-graph! state graph native!)))
      (is (= before @calls)))
    (is (nil? @(:submission-state graph)))
    (pending/synchronize! ledger (:stream @state) native!)
    (let [token (execution/submit-graph! state graph native!)]
      (execution/release-event! token))
    (execution/destroy-graph! graph)
    (cleanup/release! (::cleanup/owner p))
    (is (zero? (root/lease-count state)))))

(deftest ordered-replay-retains-exact-prepared-owners
  (let [{:keys [state native! calls prepared]} (fixture)
        a (prepared "a") b (prepared "b")
        graph (execution/record-graph! state [a b a] {})]
    (is (= :owner-in-use
           (try (cleanup/release! (::cleanup/owner a)) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (let [token (execution/submit-graph! state graph native!)]
      (is (= ["a" "b" "a"] (mapv (comp first second) (filter #(= :launch (first %)) @calls))))
      (is (execution/event-complete? token native!))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"previous submission"
                            (execution/submit-graph! state graph native!)))
      (execution/await-event! token native!)
      (execution/release-event! token)
      (is (nil? @(:submission-state graph)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"no longer current"
                            (execution/release-event! token))))
    (execution/destroy-graph! graph)
    (cleanup/release! (::cleanup/owner a))
    (cleanup/release! (::cleanup/owner b))
    (is (zero? (root/lease-count state)))
    (is (= 2 (count (filter #(= :prepared-free (first %)) @calls))))))

(deftest profiles-consume-events-after-device-completion
  (let [{:keys [state native! calls prepared]} (fixture)
        p (prepared "p") graph (execution/record-graph! state [p] {:profile? true})
        token (execution/submit-graph! state graph native!)]
    (execution/release-event! token)
    (is (some? @(:submission-state graph)))
    (let [result (execution/read-graph-timestamps! graph native!)]
      (is (= :device-event (:timing-source result)))
      (is (= 0.001 (:wall-ms result)))
      (is (= "p" (get-in result [:kernels 0 :kernel-name]))))
    (is (nil? @(:submission-state graph)))
    (is (= 2 (count (filter #(= :event-destroy (first %)) @calls))))
    (execution/destroy-graph! graph)
    (cleanup/release! (::cleanup/owner p))
    (is (zero? (root/lease-count state)))))

(deftest unknown-drain-keeps-submission-and-prepared-leases
  (let [{:keys [state native! failure calls prepared]} (fixture)
        p (prepared "p") graph (execution/record-graph! state [p] {})]
    (reset! failure :launch)
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"mock native failure"
                          (execution/submit-graph! state graph native!)))
    ;; A launch failure can be successfully drained; its failed construction retires.
    (is (nil? @(:submission-state graph)))
    (reset! failure nil)
    (let [token (execution/submit-graph! state graph native!)
          destroyed (count (filter #(= :event-destroy (first %)) @calls))]
      (reset! failure :stream-sync)
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"mock native failure"
                            (execution/await-event! token native!)))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"mock native failure"
                            (execution/destroy-graph! graph)))
      (is (some? @(:submission-state graph)))
      (is (= destroyed (count (filter #(= :event-destroy (first %)) @calls))))
      (is (= :owner-in-use
             (try (cleanup/release! (::cleanup/owner p)) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (let [n (count (filter #(= :stream-sync (first %)) @calls))]
        (is (thrown? clojure.lang.ExceptionInfo (execution/destroy-graph! graph)))
        (is (= n (count (filter #(= :stream-sync (first %)) @calls))))))))

(deftest generation-and-empty-replay-controls
  (let [{:keys [state native! prepared]} (fixture)
        p (prepared "p") bad (assoc p :raster.gpu.cuda-runtime/context (Object.))]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"another context"
                          (execution/record-graph! state [bad] {})))
    ;; Failed recording released its admitted dependency reservation.
    (cleanup/release! (::cleanup/owner p))
    (let [graph (execution/record-graph! state [] {})
          token (execution/submit-graph! state graph native!)]
      (is (execution/event-complete? token native!))
      (execution/await-event! token native!)
      (execution/release-event! token)
      (execution/destroy-graph! graph))
    (is (zero? (root/lease-count state)))))

(deftest reentrant-query-cannot-retire-its-recording
  (let [{:keys [state native! prepared]} (fixture)
        p (prepared "p") graph (execution/record-graph! state [p] {})
        token (execution/submit-graph! state graph native!)
        attempted (atom nil)
        guarded (fn [op args]
                  (when (= op :event-query)
                    (reset! attempted
                            (try (execution/destroy-graph! graph) :destroyed
                                 (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
                  (native! op args))]
    (is (execution/event-complete? token guarded))
    (is (= :owner-in-use @attempted))
    (is (= :live (:phase (cleanup/status (::cleanup/owner graph)))))
    (execution/release-event! token)
    (execution/destroy-graph! graph)
    (cleanup/release! (::cleanup/owner p))
    (is (zero? (root/lease-count state)))))

(deftest stale-context-does-not-acquire-native-events
  (let [{:keys [state native! prepared calls]} (fixture)
        p (prepared "p") graph (execution/record-graph! state [p] {}) context (:context @state)]
    (swap! state assoc :context (Object.))
    (is (= :runtime-generation-mismatch
           (try (execution/submit-graph! state graph native!) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (not-any? #(= :event-create (first %)) @calls))
    (is (nil? @(:submission-state graph)))
    ;; Restore the deliberately corrupted mock projection, not a native device reset.
    (swap! state assoc :context context)
    (execution/destroy-graph! graph)
    (cleanup/release! (::cleanup/owner p))
    (is (zero? (root/lease-count state)))))

(deftest concurrent-query-keeps-recording-and-events-live
  (let [{:keys [state native! prepared calls]} (fixture)
        p (prepared "p") graph (execution/record-graph! state [p] {})
        token (execution/submit-graph! state graph native!)
        entered (promise) proceed (promise)
        guarded (fn [op args]
                  (when (= op :event-query)
                    (deliver entered true)
                    (deref proceed 5000 :timeout))
                  (native! op args))
        worker (future (execution/event-complete? token guarded))]
    (try
      (is (= true (deref entered 5000 :timeout)))
      (is (= :owner-in-use
             (try (execution/destroy-graph! graph) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (= :live (:phase (cleanup/status (::cleanup/owner graph)))))
      (is (not-any? #(= :event-destroy (first %)) @calls))
      (finally (deliver proceed true)))
    (is (= true (deref worker 5000 :timeout)))
    (execution/release-event! token)
    (execution/destroy-graph! graph)
    (cleanup/release! (::cleanup/owner p))
    (is (zero? (root/lease-count state)))))

(deftest profile-retirement-cannot-destroy-a-newer-submission
  (let [{:keys [state native! prepared calls]} (fixture)
        p (prepared "p") graph (execution/record-graph! state [p] {:profile? true})
        old-token (execution/submit-graph! state graph native!)
        original @#'execution/retire! next-token (atom nil)]
    (with-redefs-fn
      {#'execution/retire!
       (fn
         ([s slot] (original s slot nil))
         ([s slot expected]
          ;; Deterministically interleave reset + replay at the profile-to-retire gap.
          (original s slot nil)
          (reset! next-token (execution/submit-graph! state graph native!))
          (original s slot expected)))}
      #(is (= :submission-not-current
              (try (execution/read-graph-timestamps! graph native!) nil
                   (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (is (not (identical? (::cleanup/owner old-token) (::cleanup/owner @next-token))))
    (is (identical? @next-token @(:submission-state graph)))
    (is (= 2 (count (filter #(= :event-destroy (first %)) @calls))))
    (execution/reset-graph-events! graph)
    (is (= 4 (count (filter #(= :event-destroy (first %)) @calls))))
    (execution/destroy-graph! graph)
    (cleanup/release! (::cleanup/owner p))
    (is (zero? (root/lease-count state)))))

(deftest independent-event-retirement-failure-retains-exact-dependent-leases
  (let [{:keys [state native! prepared calls]} (fixture)
        p (prepared "p") fail? (atom true)
        guarded (fn [op args]
                  (native! op args)
                  (when (and (= op :event-destroy) (compare-and-set! fail? true false))
                    (throw (ex-info "retryable event retirement" {:cleanup-retry-safe? true}))))
        graph (execution/record-graph! state [p] {:profile? true})]
    (execution/submit-graph! state graph guarded)
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"retryable event"
                          (execution/destroy-graph! graph)))
    (is (= 2 (count (filter #(= :event-destroy (first %)) @calls))))
    (is (= :owner-in-use
           (try (cleanup/release! (::cleanup/owner p)) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (execution/destroy-graph! graph)
    (is (= 3 (count (filter #(= :event-destroy (first %)) @calls))))
    (cleanup/release! (::cleanup/owner p))
    (is (zero? (root/lease-count state)))))
