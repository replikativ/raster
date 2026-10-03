(ns raster.gpu.runtime-root-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root]))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))

(deftest invalid-root-plan-is-rejected-before-publication-or-acquisition
  (doseq [[plan build] [[nil (fn [_] {})]
                        [[{:id :context :release nil}] (fn [_] {})]
                        [[] nil]]]
    (let [state (atom {:initialized? false}) publications (atom 0)]
      (add-watch state :observe (fn [& _] (swap! publications inc)))
      (is (= :invalid-cleanup-plan
             (:reason (ex-data (error-of #(root/initialize! state plan build))))))
      (is (= {:initialized? false} @state))
      (is (zero? @publications)))))

(defn- fixture [options test!]
  (let [state (atom {:initialized? false}) calls (atom [])
        primary (ex-info "native acquisition outcome unknown" {})
        destruction (ex-info "native destruction outcome unknown" {})
        release (fn [id resource]
                  (swap! calls conj [:release id resource])
                  (when (= id (:release-failure options)) (throw destruction)))
        plan [{:id :compute :release #(release :compute %)}
              {:id :transfer :release #(release :transfer %)}
              {:id :context :after #{:compute :transfer} :release #(release :context %)}
              {:id :arena :after #{:context} :release #(release :arena %)}]
        acquire (fn [entry id]
                  (root/acquire! entry id
                                 #(do (swap! calls conj [:acquire id])
                                      (when-let [callback (:on-acquire options)] (callback state id))
                                      (when (= id (:acquire-failure options)) (throw primary))
                                      (Object.))))
        build (fn [entry]
                (let [arena (acquire entry :arena)]
                  (when (:precontact-failure options) (throw primary))
                  {:arena arena :context (acquire entry :context)
                   :queue (acquire entry :compute) :transfer-queue (acquire entry :transfer)}))]
    (test! {:state state :calls calls :primary primary :destruction destruction
            :initialize! #(root/initialize! state plan build) :plan plan :build build})))

(deftest live-roots-admit-use-but-shutdown-is-pure-fail-closed-before-leases
  (fixture {}
           (fn [{:keys [state calls initialize!]}]
             (initialize!)
             (let [entry (root/assert-live! state) before @calls]
               (initialize!)
               (is (identical? entry (root/assert-live! state)))
               (is (= :runtime-root-leases-incomplete
                      (:reason (ex-data (error-of #(root/shutdown-construction! state))))))
               (is (= before @calls))
               (is (= :live (:phase (cleanup/status (::cleanup/owner entry)))))))))

(deftest every-unknown-create-retains-exact-debt-and-does-not-retry
  (doseq [id [:arena :context :compute :transfer]]
    (fixture {:acquire-failure id}
             (fn [{:keys [state calls primary initialize!]}]
               (is (identical? primary (error-of initialize!)))
               (is (= :runtime-root-unavailable (:reason (ex-data (error-of initialize!)))))
               (let [before @calls]
                 (is (identical? primary (error-of #(root/shutdown-construction! state))))
                 (is (= before @calls))
                 (is (some? (::root/entry @state))))))))

(deftest known-precontact-failure-releases-arena-and-allows-clean-retry
  (fixture {:precontact-failure true}
           (fn [{:keys [state calls primary initialize! plan]}]
             (is (identical? primary (error-of initialize!)))
             (is (nil? (::root/entry @state)))
             (is (= [:acquire :release] (mapv first @calls)))
             (root/initialize! state plan (fn [entry] {:arena (root/acquire! entry :arena #(Object.))}))
             (is (some? (root/assert-live! state))))))

(deftest final-publication-watch-and-validator-rollback-remove-only-the-released-root
  (doseq [kind [:watch :validator]]
    (fixture {}
             (fn [{:keys [state calls initialize!]}]
               (let [failure (ex-info "root publication rejected" {})]
                 (case kind
                   :watch (add-watch state :fail
                                     (fn [_ _ _ value] (when (:initialized? value) (throw failure))))
                   :validator (set-validator! state #(or (not (:initialized? %)) (throw failure))))
                 (is (identical? failure (error-of initialize!)))
                 (is (nil? (::root/entry @state)))
                 (is (false? (:initialized? @state)))
                 (is (= [:compute :transfer :context :arena]
                        (mapv second (filter #(= :release (first %)) @calls))))
                 (remove-watch state :fail)
                 (set-validator! state nil)
                 (initialize!)
                 (is (some? (root/assert-live! state))))))))

(deftest independent-queue-destruction-is-attempted-and-context-remains-pinned
  (fixture {:precontact-failure false :release-failure :compute}
           (fn [{:keys [state calls destruction plan build]}]
             (let [primary (ex-info "post-acquisition query failed" {})]
               (is (identical? primary
                               (error-of #(root/initialize! state plan (fn [entry] (build entry) (throw primary))))))
               (is (some #(identical? destruction %) (.getSuppressed ^Throwable primary)))
               (is (= [:compute :transfer]
                      (mapv second (filter #(= :release (first %)) @calls))))
               (let [before @calls]
                 (is (identical? destruction (error-of #(root/shutdown-construction! state))))
                 (is (= before @calls)))))))

(deftest lost-generation-retains-unresolved-authority-without-overwriting-unrelated-state
  (fixture {:acquire-failure :context
            :on-acquire (fn [state id]
                          (when (= id :context) (reset! state {:unrelated :preserved})))}
           (fn [{:keys [state primary initialize!]}]
             (is (identical? primary (error-of initialize!)))
             (is (= :preserved (:unrelated @state)))
             (is (= 1 (count (filter #(and (map? %) (::root/root? %)) (vals @state)))))
             (is (= :runtime-root-unavailable (:reason (ex-data (error-of initialize!))))))))

(deftest reentrant-and-concurrent-initialization-do-not-create-another-root
  (let [observed (atom nil)]
    (fixture {:on-acquire (fn [state id]
                            (when (= id :context)
                              (reset! observed (error-of #(root/shutdown-construction! state)))))}
             (fn [{:keys [state calls initialize!]}]
               (let [jobs (mapv (fn [_] (future (initialize!))) (range 4))]
                 (doseq [job jobs] @job)
                 (is (= :registration-in-use (:reason (ex-data @observed))))
                 (is (= 4 (count (filter #(= :acquire (first %)) @calls))))
                 (is (some? (root/assert-live! state))))))))

(deftest lazy-child-is-acquired-once-and-unknown-outcome-poisons-use-not-ownership
  (doseq [fail? [false true]]
    (let [state (atom {}) creates (atom 0) failure (ex-info "async create unknown" {})
          plan [{:id :async :release (fn [_])}]
          prepare (fn [_] #(do (swap! creates inc) (when fail? (throw failure)) (Object.)))]
      (root/initialize! state plan (fn [_] {}))
      (if fail?
        (do
          (is (identical? failure (error-of #(root/acquire-child! state :async prepare))))
          (is (= :runtime-root-unavailable
                 (:reason (ex-data (error-of #(root/acquire-child! state :async prepare))))))
          (is (= :runtime-root-leases-incomplete
                 (:reason (ex-data (error-of #(root/shutdown-construction! state))))))
          (is (= 1 @creates)))
        (let [jobs (mapv (fn [_] (future (root/acquire-child! state :async prepare))) (range 4))]
          (is (apply identical? (take 2 (mapv deref jobs))))
          (is (= 1 @creates)))))))
