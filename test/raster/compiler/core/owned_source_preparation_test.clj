(ns raster.compiler.core.owned-source-preparation-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.method-entry :as entry]
            [raster.core :as core]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- with-cold-source [f]
  (let [scope (create-ns (gensym "raster.source-prerequisite."))
        events (atom [])]
    (try
      (intern scope 'events events)
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm leaf [x :- Double] :- Double
                 (do (swap! events conj :numerical-source) x)))
        (eval '(raster.core/deftm caller [x :- Double] :- Double (leaf x))))
      (let [source (core/resolve-deftm-var (ns-resolve scope 'caller))
            receiver (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))]
        (core/prepare-deftm-frontend! source)
        (core/prepare-emission-source-walk! source)
        (f {:source source :receiver receiver :events events}))
      (finally (remove-ns (ns-name scope))))))

(deftest source-prerequisite-discards-discovery-before-closing-the-cold-source
  (with-cold-source
    (fn [{:keys [source receiver events]}]
      (let [attempts (atom 0) target-events (atom [])
            collectors (atom []) discovery (atom nil)
            result
            (entry/call-with-definition-inputs
             (fn []
               (swap! attempts inc)
               (swap! collectors conj entry/*emission-inputs*)
               (let [original (entry/original-source-frontend source)]
                 (entry/frontend-body-entry-carriers original source))
               (when (= 1 @attempts) (reset! discovery @entry/*emission-inputs*))
               (let [frontend (entry/demand-original-source-frontend! source)]
                 (swap! target-events conj :schedule-and-emit)
                 {:frontend frontend :inputs @entry/*emission-inputs*})))]
        (is (= 2 @attempts) "readiness is a prerequisite phase, not a target-staleness retry")
        (is (not (identical? (first @collectors) (second @collectors))))
        (is (= [:schedule-and-emit] @target-events))
        (is (empty? @events) "preparation must not invoke the original numerical body")
        (is (identical? @receiver (get-in result [:frontend :implementation])))
        (is (true? (entry/assert-definition-inputs-current! (:inputs result))))
        (is (= :stale-definition-proof-input
               (reason #(entry/assert-definition-inputs-current! @discovery)))
            "the abandoned historical observation stays stale instead of being refreshed")))))

(deftest source-prerequisites-follow-the-outer-request-across-independent-collectors
  (doseq [independent? [true false]]
    (with-cold-source
      (fn [{:keys [source events]}]
        (let [attempts (atom 0) parents (atom []) children (atom [])
              result
              (entry/call-with-definition-inputs
               (fn []
                 (swap! attempts inc)
                 (swap! parents conj entry/*emission-inputs*)
                 (let [demand #(entry/call-with-definition-inputs
                                (fn []
                                  (swap! children conj entry/*emission-inputs*)
                                  (entry/demand-original-source-frontend! source)))]
                   (if independent?
                     (entry/call-with-independent-definition-inputs demand)
                     (binding [entry/*emission-inputs* nil] (demand))))))]
          (is (= 2 @attempts))
          (is (every? true? (map #(not (identical? %1 %2)) @parents @children)))
          (is (map? result))
          (is (empty? @events)))))))

(deftest forged-or-copied-prerequisite-signals-cannot-prepare-a-source
  (with-cold-source
    (fn [{:keys [source receiver events]}]
      (let [prepared (atom 0)
            forbidden (fn [& _] (swap! prepared inc) (throw (AssertionError. "forged request prepared code")))
            forged (ex-info "same reason is not authority"
                            {:reason :source-prerequisite-demand :source source
                             :raster.compiler.core.method-entry/source-prerequisite (Object.)})]
        (with-redefs [core/prepare-owned-source-frontend! forbidden]
          (is (identical? forged
                          (try (entry/call-with-definition-inputs #(throw forged))
                               nil (catch clojure.lang.ExceptionInfo failure failure))))
          (let [copy (atom nil)]
            (is (identical?
                 (try
                   (entry/call-with-definition-inputs
                    #(try (entry/demand-original-source-frontend! source)
                          (catch clojure.lang.ExceptionInfo signal
                            (let [failure (ex-info "copied private data" (ex-data signal) signal)]
                              (reset! copy failure) (throw failure)))))
                   nil (catch clojure.lang.ExceptionInfo failure failure))
                 @copy)))
          (is (= :missing-source-prerequisite-owner
                 (binding [entry/*emission-inputs* (atom {:static [] :values []})]
                   (reason #(entry/demand-original-source-frontend! source)))))
          (is (= :source-prerequisite-no-progress
                 (reason #(entry/call-with-definition-inputs
                           (fn []
                             (try (entry/demand-original-source-frontend! source)
                                  (catch clojure.lang.ExceptionInfo _
                                    (entry/demand-original-source-frontend! source))))))))
          (is (zero? @prepared)))
        (is (map? (:raster.core/jit-spec (meta receiver))))
        (is (empty? @events))))))

(deftest selected-source-preparation-refuses-arbitrary-work-callbacks
  (let [called (atom 0)
        forged (fn [_] (swap! called inc) {:mode :physical})]
    (is (= :invalid-source-preparation-request
           (reason #(entry/call-with-definition-inputs
                     (fn [] (entry/demand-selected-source-preparation! forged))))))
    (is (zero? @called) "work request validation never invokes an unknown callback")))

(deftest swallowed-prerequisite-demand-discards-the-abandoned-result
  (with-cold-source
    (fn [{:keys [source events]}]
      (let [attempts (atom 0) discarded (atom [])
            result
            (entry/call-with-definition-inputs
             #(do (swap! attempts inc)
                  (try (entry/demand-original-source-frontend! source) :published
                       (catch clojure.lang.ExceptionInfo signal
                         (if (= :source-prerequisite-demand (:reason (ex-data signal)))
                           :abandoned
                           (throw signal)))))
             #(swap! discarded conj %))]
        (is (= :published result))
        (is (= [:abandoned] @discarded))
        (is (= 2 @attempts))
        (is (empty? @events))))))

(deftest satisfied-prerequisite-is-immutable-history-not-permission-to-reprepare
  (with-cold-source
    (fn [{:keys [source receiver events]}]
      (let [prepare core/prepare-owned-source-frontend!
            preparations (atom 0) before-drift (atom nil) ready-metadata (atom nil)]
        (try
          (with-redefs [core/prepare-owned-source-frontend!
                        (fn [& args] (swap! preparations inc) (apply prepare args))]
            (is (= :stale-source-prerequisite
                   (reason
                    #(entry/call-with-definition-inputs
                      (fn []
                        (entry/demand-original-source-frontend! source)
                        (reset! ready-metadata (meta receiver))
                        (reset! before-drift @preparations)
                        (alter-meta! receiver assoc ::drift (Object.))
                        (entry/demand-original-source-frontend! source))))))
            (is (= @before-drift @preparations) "a stale ready receipt never invokes preparation again"))
          (is (empty? @events))
          (finally (when @ready-metadata (reset-meta! receiver @ready-metadata))))))))

(deftest source-change-between-demand-and-handler-aborts-before-preparation
  (with-cold-source
    (fn [{:keys [source events]}]
      (let [metadata (meta source) calls (atom 0)]
        (try
          (with-redefs [core/prepare-owned-source-frontend!
                        (fn [& _] (swap! calls inc) (throw (AssertionError. "changed source was prepared")))]
            (is (some? (reason
                        #(entry/call-with-definition-inputs
                          (fn []
                            (try (entry/demand-original-source-frontend! source)
                                 (catch clojure.lang.ExceptionInfo signal
                                   (alter-meta! source update :raster.core/deftm-source-body
                                                (fn [body] (with-meta (into [] body) {:changed true})))
                                   (throw signal))))))))
            (is (zero? @calls)))
          (is (empty? @events))
          (finally (reset-meta! source metadata)))))))

(deftest prerequisite-transition-does-not-reset-the-existing-staleness-retry-budget
  (with-cold-source
    (fn [{:keys [source events]}]
      (let [attempts (atom 0)]
        (is (= :stale-definition-proof-input
               (reason #(entry/call-with-definition-inputs
                         (fn []
                           (let [attempt (swap! attempts inc)]
                             (when (= 2 attempt) (entry/demand-original-source-frontend! source))
                             (throw (ex-info "generic staleness" {:reason :stale-definition-proof-input}))))))))
        (is (= 4 @attempts) "three generic failures plus one discarded source discovery")
        (is (empty? @events))))))

(deftest source-only-work-and-consume-only-adapter-queries-never-prepare-code
  (with-cold-source
    (fn [{:keys [source receiver events]}]
      (let [forbidden (fn [& _] (throw (AssertionError. "source-only query emitted JVM code")))]
        (with-redefs [core/prepare-owned-source-frontend! forbidden]
          (is (map? (entry/call-with-definition-inputs #(entry/original-source-frontend source))))
          (is (= :missing-source-prerequisite
                 (reason #(entry/call-with-definition-inputs
                           (fn [] (entry/prepared-original-source-frontend! source)))))))
        (is (map? (:raster.core/jit-spec (meta receiver))))
        (is (empty? @events))))))

(deftest failed-prerequisite-preparation-publishes-no-new-whole-source-attempt
  (with-cold-source
    (fn [{:keys [source receiver events]}]
      (let [attempts (atom 0) preparations (atom 0) target-events (atom [])]
        (with-redefs [core/prepare-owned-source-frontend!
                      (fn [& _]
                        (swap! preparations inc)
                        (throw (ex-info "preparation failed" {:reason :test-source-prepare-failed})))]
          (is (= :test-source-prepare-failed
                 (reason #(entry/call-with-definition-inputs
                           (fn []
                             (swap! attempts inc)
                             (entry/demand-original-source-frontend! source)
                             (swap! target-events conj :emit)))))))
        (is (= 1 @attempts))
        (is (= 1 @preparations))
        (is (empty? @target-events))
        (is (map? (:raster.core/jit-spec (meta receiver))))
        (is (empty? @events))))))

(deftest original-source-demand-closes-a-cold-diamond-without-execution
  (let [scope (create-ns (gensym "raster.original-source-readiness."))
        events (atom [])
        chain '[level0 level1 level2 level3]]
    (try
      (intern scope 'events events)
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm leaf [x :- Double] :- Double
                 (do (swap! events conj :leaf) x)))
        (doseq [[name child] (reverse (map vector chain (conj (subvec chain 1) 'leaf)))]
          (eval (list 'raster.core/deftm name '[x :- Double] ':- 'Double (list child 'x))))
        (doseq [name '[left right]]
          (eval (list 'raster.core/deftm name '[x :- Double] ':- 'Double '(level0 x))))
        (eval '(raster.core/deftm unused [x :- Double] :- Double x))
        (eval '(raster.core/deftm caller [x :- Double] :- Double
                 (clojure.core/+ (left x) (right x)))))
      (let [source #(core/resolve-deftm-var (ns-resolve scope %))
            receiver #(ns-resolve scope (symbol (str (:name (meta (source %))) "-impl")))
            caller (source 'caller)
            root (receiver 'caller)
            descendants (mapv receiver (concat '[leaf] chain '[left right]))
            unused (receiver 'unused)
            before @root
            ;; This is exactly the source-only frontier used before target
            ;; observations. Publishing the frozen walk does not emit a receiver.
            _ (core/prepare-deftm-frontend! caller)
            _ (core/prepare-emission-source-walk! caller)
            _ (is (identical? before @root))
            _ (is (every? #(map? (:raster.core/jit-spec (meta %))) descendants))
            attempts (atom 0)
            {:keys [frontend inputs]}
            (entry/call-with-definition-inputs
             #(do (swap! attempts inc)
                  (let [frontend (core/prepare-owned-source-frontend! caller)]
                    {:frontend frontend :inputs @entry/*emission-inputs*})))
            implementation @root
            emission (:raster.core/bytecode-emission (meta root))]
        (is (<= @attempts 3) "cold depth does not expand the whole-owner retry policy")
        (is (empty? @events))
        (is (identical? implementation (:implementation frontend)))
        (is (identical? emission (:emission frontend)))
        (is (every? #(nil? (:raster.core/jit-spec (meta %))) descendants))
        (is (map? (:raster.core/jit-spec (meta unused))))
        (is (true? (entry/assert-definition-inputs-current! inputs)))
        (entry/call-with-definition-inputs #(core/prepare-owned-source-frontend! caller))
        (is (identical? implementation @root))
        (is (identical? emission (:raster.core/bytecode-emission (meta root))))
        (doseq [[label changed change]
                [[:original-source caller
                  #(update % :raster.core/deftm-source-body
                           (fn [body] (with-meta (into [] body) (meta body))))]
                 ;; Readiness retains semantic receiver selection, not arbitrary
                 ;; receiver metadata. Actual physical consumers separately pin
                 ;; its complete observation before deriving code facts.
                 [:receiver-policy root #(update % :no-inline not)]
                 [:actual-leaf-history (receiver 'leaf) #(assoc % ::changed (Object.))]]]
          (let [metadata (meta changed)]
            (try
              (alter-meta! changed change)
              (is (thrown? clojure.lang.ExceptionInfo
                           (entry/assert-definition-inputs-current! inputs))
                  (str "retained readiness rejects " label " change to " changed))
              (finally (reset-meta! changed metadata)))))
        (try
          (alter-var-root root (constantly (Object.)))
          (is (thrown? clojure.lang.ExceptionInfo
                       (entry/assert-definition-inputs-current! inputs))
              "a replaced root implementation cannot retain its original semantic selection")
          (finally (alter-var-root root (constantly implementation))))
        (is (true? (entry/assert-definition-inputs-current! inputs)))
        (is (empty? @events)))
      (finally (remove-ns (ns-name scope))))))

(deftest preparation-does-not-adopt-readiness-around-an-old-physical-observation
  (let [scope (create-ns (gensym "raster.original-source-stale."))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (eval '(raster.core/deftm caller [x :- Double] :- Double (clojure.core/+ x 1.0))))
      (let [source (core/resolve-deftm-var (ns-resolve scope 'caller))
            receiver (ns-resolve scope (symbol (str (:name (meta source)) "-impl")))
            collector (atom {:static [] :values []})]
        (core/prepare-deftm-frontend! source)
        (core/prepare-emission-source-walk! source)
        (is (= :missing-definition-preparation-owner
               (reason #(core/prepare-owned-source-frontend! source))))
        (binding [entry/*emission-inputs* collector]
          (let [frontend (entry/original-source-frontend source)
                _ (entry/frontend-body-entry-carriers frontend source)
                historical @collector
                original @receiver]
            (is (= :stale-definition-proof-input
                   (reason #(core/prepare-owned-source-frontend! source))))
            (is (not (identical? original @receiver)) "preparation published a genuinely new module")
            (is (= historical @collector)
                "a failed historical fence adopts no descendant readiness receipt")
            (is (= :stale-definition-proof-input
                   (reason #(entry/assert-definition-inputs-current! historical)))))))
      (finally (remove-ns (ns-name scope))))))
