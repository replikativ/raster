(ns raster.gpu.private-executor-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.link-plan :as plan]
            [raster.gpu.link :as link]))

(defn- fixture [f]
  (let [nodes (into {} (map (fn [[id role]]
                             [id (plan/node {:id id :dtype :double :shape [2] :role role})]))
                    [[:x :input] [:state :state] [:constant :constant] [:out :output] [:tmp :internal]])
        state (atom {:bindings 0 :uploads [] :runs 0 :closes 0})
        pending (atom #{})
        failure (atom nil)
        check! (fn [phase]
                 (when (= phase @failure)
                   (throw (ex-info "injected runtime failure" {:reason phase}))))]
    (with-redefs-fn
      {#'link/prepare-private-reuse!
       (fn [p]
         (swap! state update :bindings inc)
         (reset! pending (:pending p #{}))
         {:executable {:plan p :pending-inputs pending} :memory {:allocations-saved 1}})
       #'link/upload! (fn [_ node source]
                        (check! :upload)
                        (swap! pending disj node)
                        (swap! state update :uploads conj [node (vec source)]))
       #'link/run! (fn [_] (check! :run) (swap! state update :runs inc))
       #'link/download (fn [_ _] (check! :download) (double-array [(:runs @state) 7]))
       #'link/close! (fn [_] (swap! state update :closes inc) (check! :close))}
      #(f {:plan {:nodes nodes :outputs [:out]} :state state :failure failure}))))

(deftest private-executor-retains-one-binding-and-detaches-each-result
  (fixture
   (fn [{:keys [plan state]}]
     (let [execute (link/private-executor! plan)
           first-result (execute {:x (double-array [1 2])})
           second-result (execute {:state (double-array [3 4])})]
       (is (not (map? execute)))
       (is (not (link/linked-executable? execute)))
       (is (= #{:outputs :memory} (set (keys first-result))))
       (is (= {:allocations-saved 1} (:memory second-result)))
       (is (= [1.0 7.0] (vec (get-in first-result [:outputs :out]))))
       (is (= [2.0 7.0] (vec (get-in second-result [:outputs :out]))))
       (.close ^java.io.Closeable execute)
       (.close ^java.io.Closeable execute)
       (is (= {:bindings 1 :runs 2 :closes 1
               :uploads [[:x [1.0 2.0]] [:state [3.0 4.0]]]} @state))
       (is (= [1.0 7.0] (vec (get-in first-result [:outputs :out]))))
       (is (= :link-private-scope-closed
              (try (execute) (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))))

(deftest private-executor-validates-the-whole-batch-before-mutating
  (fixture
   (fn [{:keys [plan state]}]
     (with-open [execute (link/private-executor! plan)]
       (doseq [updates [nil [] {:x nil} {:x false} {:x (float-array 2)} {:x (double-array 1)}
                        (array-map :x (double-array 2) :tmp (double-array 2))
                        {:constant (double-array 2)} {:out (double-array 2)} {:missing (double-array 2)}]]
         (is (thrown? clojure.lang.ExceptionInfo (execute updates)))
         (is (empty? (:uploads @state)))
         (is (zero? (:closes @state))))
       (is (= [1.0 7.0] (vec (get-in (execute) [:outputs :out]))))))))

(deftest incomplete-initial-inputs-reject-before-upload-and-can-be-retried
  (fixture
   (fn [{:keys [plan state]}]
     (with-open [execute (link/private-executor! (assoc plan :pending #{:x :state}))]
       (is (= :link-pending-inputs
              (try (execute {:x (double-array 2)})
                   (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
       (is (empty? (:uploads @state)))
       (is (zero? (:closes @state)))
       (is (map? (execute {:x (double-array 2) :state (double-array 2)})))
       (is (map? (execute)))))))

(deftest private-executor-rejects-overlapping-input-updates
  (fixture
   (fn [{:keys [plan state]}]
     (let [plan (assoc-in plan [:nodes :state :view] (get-in plan [:nodes :x :view]))]
       (with-open [execute (link/private-executor! plan)]
         (is (= :link-private-input-overlap
                (try (execute {:x (double-array 2) :state (double-array 2)})
                     (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
         (is (empty? (:uploads @state)))
         (is (= 1 (:runs (do (execute) @state)))))))))

(deftest private-runtime-failure-closes-the-scope-and-preserves-the-error
  (doseq [phase [:upload :run :download]]
    (fixture
     (fn [{:keys [plan state failure]}]
       (let [execute (link/private-executor! plan)]
         (reset! failure phase)
         (is (= phase (try (execute {:x (double-array 2)})
                          (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
         (is (= 1 (:closes @state)))
         (is (= :link-private-scope-closed
                (try (execute) (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
         (.close ^java.io.Closeable execute)
         (is (= 1 (:closes @state))))))))

(deftest cleanup-failure-does-not-replace-the-runtime-failure
  (fixture
   (fn [{:keys [plan failure]}]
     (let [execute (link/private-executor! plan)]
       (reset! failure :run)
       (with-redefs [link/close! (fn [_] (throw (ex-info "cleanup failure" {:reason :cleanup})))]
         (let [error (try (execute) (catch clojure.lang.ExceptionInfo e e))]
           (is (= :run (:reason (ex-data error))))
           (is (= [:cleanup] (mapv #(-> % ex-data :reason) (.getSuppressed error))))))
       (.close ^java.io.Closeable execute)))))

(deftest close-is-serialized-after-the-active-replay
  (fixture
   (fn [{:keys [plan state]}]
     (let [entered (promise) release (promise) closing (promise)
           execute (link/private-executor! plan)]
       (with-redefs [link/run! (fn [_] (deliver entered true) @release (swap! state update :runs inc))]
         (let [replay (future (execute))]
           (try
             (is (= true (deref entered 5000 ::timeout)))
             (let [close-task (future (deliver closing true) (.close ^java.io.Closeable execute))]
               (is (= true (deref closing 5000 ::timeout)))
               (is (zero? (:closes @state)))
               (deliver release true)
               (is (map? (deref replay 5000 ::timeout)))
               (is (nil? (deref close-task 5000 ::timeout)))
               (is (= 1 (:closes @state))))
             (finally (deliver release true) (.close ^java.io.Closeable execute)))))))))
