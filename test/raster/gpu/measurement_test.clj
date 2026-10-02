(ns raster.gpu.measurement-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.measurement :as measurement]
            [raster.gpu.parallel-program :as parallel-program]))

(deftest summarizes-device-samples
  (let [m (measurement/summarize [100.0 200.0 300.0 400.0]
                                 :cv-threshold 1.0
                                 :warmup-iterations 2
                                 :budget-ms 25
                                 :timing-source :synthetic
                                 :hashes {:artifact "abc"})]
    (is (measurement/measurement? m))
    (is (= 100.0 (:min-ns m)))
    (is (= 200.0 (:median-ns m)) "nearest-rank p50")
    (is (= 300.0 (:p75-ns m)))
    (is (= 250.0 (:mean-ns m)))
    (is (:stationary? m))
    (is (= 4 (:n m)))
    (is (= {:artifact "abc"} (:hashes m)))))

(deftest stationarity-is-explicit
  (is (:stationary? (measurement/summarize [100 101 99] :cv-threshold 0.02)))
  (is (false? (:stationary? (measurement/summarize [1 100 1] :cv-threshold 0.02)))))

(deftest bounded-device-sampling
  (let [calls (atom 0)
        flushes (atom 0)
        m (measurement/measure! #(do (swap! calls inc) 1000000.0)
                                :warmup-iterations 2
                                :budget-ms 4
                                :min-samples 3
                                :max-samples 10
                                :flush-fn #(swap! flushes inc)
                                :timing-source :synthetic)]
    (testing "warmup + five probes are excluded from the bounded reported sample set"
      (is (= 4 (:n m)))
      (is (= (+ 2 5 4) @calls))
      (is (= 4 @flushes)))
    (is (= :synthetic (:timing-source m)))
    (is (= 1000000.0 (:min-ns m)))))

(deftest rejects-invalid-measurements
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"at least one"
                        (measurement/summarize [])))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"finite"
                        (measurement/summarize [Double/POSITIVE_INFINITY])))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"positive"
                        (measurement/measure! (constantly 1.0) :budget-ms 0))))

(deftest sampling-options-decline-before-callbacks
  (let [calls (atom 0)
        flushes (atom 0)
        sample #(do (swap! calls inc) 1.0)]
    (doseq [options [[:cv-threshold -1] [:cv-threshold Double/NaN]
                     [:cold-warm :unknown] [:timing-source "host"] [:hashes []]
                     [:compile-ms -1] [:compile-ms Double/POSITIVE_INFINITY]
                     [:budget-ms Double/POSITIVE_INFINITY]
                     [:warmup-iterations (inc (bigint Long/MAX_VALUE))]
                     [:min-samples (inc (bigint Long/MAX_VALUE))]
                     [:max-samples (inc (bigint Long/MAX_VALUE))]
                     [:flush-fn 42]]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (apply measurement/measure! sample
                          (concat [:flush-fn #(swap! flushes inc)] options)))))
    (is (zero? @calls))
    (is (zero? @flushes))))

(deftest sampling-checks-every-phase-before-continuing
  (doseq [[phase warmups valid-prefix] [[:warmup 1 0] [:probe 0 0] [:measurement 0 5]]
          invalid [nil -1 Double/NaN Double/POSITIVE_INFINITY]]
    (let [calls (atom 0)
          error (try
                  (measurement/measure!
                   #(if (<= (swap! calls inc) valid-prefix) 1000000.0 invalid)
                   :warmup-iterations warmups :budget-ms 1 :min-samples 1 :max-samples 1)
                  nil
                  (catch clojure.lang.ExceptionInfo e (ex-data e)))]
      (is (= phase (:phase error)))
      (is (= (inc valid-prefix) @calls)))))

(deftest finite-extreme-budget-remains-callback-bounded
  (let [calls (atom 0)
        result (measurement/measure! #(do (swap! calls inc) 1.0)
                                     :warmup-iterations 0 :budget-ms Double/MAX_VALUE
                                     :min-samples 1 :max-samples 2)]
    (is (= 2 (:n result)))
    (is (= 7 @calls))))

(deftest core-measures-runtime-graphs-with-device-events
  (let [replays (atom 0)
        reads (atom 0)
        session (atom {:device-id :probe})]
    (with-redefs-fn
      {#'raster.gpu.core/rt-resolve
       (fn [_ function-name]
         (case function-name
           "replay-graph!" (fn [_] (swap! replays inc))
           "read-graph-timestamps!" (fn [_]
                                      (swap! reads inc)
                                      {:wall-ms 0.001})
           (throw (ex-info "unexpected runtime function" {:function function-name}))))}
      #(let [m (gpu/measure-graph! session {:profile? true}
                                   :warmup-iterations 1
                                   :budget-ms 3
                                   :min-samples 3
                                   :max-samples 3)]
         (is (= :device-event (:timing-source m)))
         (is (= 1000.0 (:min-ns m)))
         (is (= (+ 1 5 3) @replays))
         (is (= @replays @reads))))))

(deftest stateful-linked-measurement-requires-restore
  (let [executable
        (link/map->LinkedExecutable
         {:plan {:nodes {:cache {:role :state}}}
          :profile? true
          :pending-inputs (atom #{}) :closed? (atom false)
          :lifetime-lock (Object.) :output-leases (atom 0)
          :output-ready? (atom false)})]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"require :before-sample!"
                          (link/measure! executable :budget-ms 1)))))

(deftest equation-first-linked-programs-use-aggregate-device-events
  (let [profiles (atom 0)
        restores (atom 0)
        executable
        (link/map->LinkedExecutable
         {:plan {:nodes {}}
          :profile? true
          :session :session
          :prepared-program :prepared
          :pending-inputs (atom #{})
          :lifetime-lock (Object.) :output-leases (atom 0)
          :output-ready? (atom false) :execution-state (atom {:value-epoch 0}) :completed-replays (atom 0)
          :closed? (atom false)})]
    (with-redefs [parallel-program/profile-prepared!
                  (fn [prepared profile-handle!]
                    (is (= :prepared prepared))
                    (is (ifn? profile-handle!))
                    (swap! profiles inc)
                    {:profile [{:kernel-name "generated" :ms 0.001}]
                     :kernel-total-ms 0.001
                     :device-wall-ms 0.002
                     :host-wall-ms 99.0
                     :timing-scope :single-graph-span
                     :program-graph-count 1})
                  measurement/measure!
                  (fn [sample! & options]
                    {:sample-ns (sample!) :options (apply hash-map options)})]
      (is (= 0.002 (:device-wall-ms (link/profile! executable))))
      (let [result (link/measure! executable
                                  :before-sample! #(swap! restores inc)
                                  :budget-ms 7)]
        (is (= 2000.0 (:sample-ns result)))
        (is (= :device-event (get-in result [:options :timing-source])))
        (is (= :single-graph-span (:timing-scope result)))
        (is (= 7 (get-in result [:options :budget-ms])))
        (is (= 1 @restores)))
      (is (= 2 @profiles)))))

(deftest profiling-and-measurement-respect-output-leases
  (let [executable
        (link/map->LinkedExecutable
         {:plan {:nodes {}} :session :session :prepared-program :prepared
          :pending-inputs (atom #{}) :closed? (atom false)
          :lifetime-lock (Object.) :output-leases (atom 1)
          :output-ready? (atom true) :completed-replays (atom 1)})]
    (is (= :link-output-lease-active
           (try (link/profile! executable)
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (is (= :link-output-lease-active
           (try (link/measure! executable :budget-ms 1)
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (is (true? @(:output-ready? executable)))
    (is (= 1 @(:completed-replays executable)))))

(defn- tracked-executable [recorded?]
  (link/map->LinkedExecutable
   {:plan {:nodes {}} :session :session :profile? true
    :graph-key (when recorded? :graph)
    :prepared-program (when-not recorded? :prepared)
    :pending-inputs (atom #{}) :closed? (atom false)
    :lifetime-lock (Object.) :output-leases (atom 0)
    :output-ready? (atom true) :completed-replays (atom 0)
    :execution-state (atom {:value-epoch 0})}))

(deftest every-measurement-replay-advances-the-same-owner-state
  (doseq [recorded? [false true]]
    (let [executable (tracked-executable recorded?)
          profiles (atom 0) restores (atom 0) flushes (atom 0)
          profile (fn [& _] (swap! profiles inc)
                    {:profile [] :device-wall-ms 0.002})]
      (with-redefs [gpu/profile-recorded-graph! profile
                    parallel-program/profile-prepared! profile]
        (let [result (link/measure! executable :warmup-iterations 1
                                    :min-samples 3 :max-samples 3
                                    :before-sample! #(swap! restores inc)
                                    :flush-fn #(swap! flushes inc))]
          (is (= 3 (count (:samples-ns result))))
          (is (= :device-event (:timing-source result)))
          (is (= 9 @profiles @restores @(:completed-replays executable)))
          (is (= 3 @flushes))
          (is (= 12 (:value-epoch @(:execution-state executable))))
          (is (true? @(:output-ready? executable))))))))

(deftest invalid-measurement-options-and-preflight-do-not-change-values
  (let [executable (tracked-executable true)]
    (doseq [opts [{:budget-ms -1} {:min-samples 0}
                  {:before-sample! 4} {:flush-fn 4}]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (apply link/measure! executable (mapcat identity opts)))))
    (reset! (:pending-inputs executable) #{:input})
    (is (thrown? clojure.lang.ExceptionInfo (link/profile! executable)))
    (reset! (:pending-inputs executable) #{})
    (is (= :link-profiling-disabled
           (try (link/profile! (assoc executable :profile? false))
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))
    (is (= {:value-epoch 0} @(:execution-state executable)))
    (is (zero? @(:completed-replays executable)))
    (is (true? @(:output-ready? executable)))))

(deftest measurement-failures-poison-without-crediting-an-incomplete-replay
  (doseq [failure-kind [:restore :profile :duration :flush]]
    (let [executable (tracked-executable true)
          failure (ex-info "injected measurement failure" {:kind failure-kind})
          profile (fn [& _]
                    (when (= :profile failure-kind) (throw failure))
                    {:profile [] :device-wall-ms (when-not (= :duration failure-kind) 0.002)})]
      (with-redefs [gpu/profile-recorded-graph! profile]
        (let [error (try (link/measure! executable :warmup-iterations 0
                                       :min-samples 1 :max-samples 1
                                       :before-sample! #(when (= :restore failure-kind) (throw failure))
                                       :flush-fn #(when (= :flush failure-kind) (throw failure)))
                         (catch Throwable error error))]
          (if (= :duration failure-kind)
            (is (= :link-measurement-device-duration (:reason (ex-data error))))
            (is (identical? failure error)))
          (is (identical? error (:failure @(:execution-state executable))))
          (is (= (if (= :flush failure-kind) 5 0) @(:completed-replays executable)))
          (is (= (if (= :flush failure-kind) 6 1)
                 (:value-epoch @(:execution-state executable))))
          (is (false? @(:output-ready? executable)))
          (is (= :link-execution-poisoned
                 (try (link/run! executable)
                      (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))))

(deftest caught-nested-restore-failure-cannot-replace-the-original-poison
  (let [executable (assoc (tracked-executable true) :owns-session? true)
        failure (ex-info "nested restore replay failed" {})
        closes (atom 0)]
    (with-redefs [gpu/replay! (fn [& _] (throw failure))
                  gpu/close-session! (fn [& _] (swap! closes inc))]
      (let [error (try (link/measure! executable :warmup-iterations 0
                                     :before-sample! #(try (link/run! executable)
                                                          (catch Throwable _)))
                       (catch Throwable error error))]
        (is (= :link-execution-poisoned (:reason (ex-data error))))
        (is (identical? failure (.getCause ^Throwable error)))
        (is (identical? failure (:failure @(:execution-state executable))))
        (is (false? @(:output-ready? executable)))
        (is (zero? @(:completed-replays executable)))
        (let [later (try (link/run! executable) (catch Throwable error error))]
          (is (identical? failure (.getCause ^Throwable later))))
        (link/close! executable)
        (link/close! executable)
        (is (= 1 @closes))))))

(deftest bound-graph-profile-preserves-device-span-and-kernel-breakdown
  (let [calls (atom []) session (atom {:device-id :probe})]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'resolve-kernel-graph-entry)
       (fn [_ handle] {:profile? (not= :unprofiled handle) :runtime-graph handle})
       #'raster.gpu.core/rt-resolve
       (fn [_ name]
         (case name
           "replay-graph!" #(swap! calls conj [:replay %])
           "read-graph-timestamps!"
           (fn [id] (swap! calls conj [:timestamp id])
             {:wall-ms (when-not (= :missing-span id) 0.5)
              :kernels [{:kernel-name "first" :ms 0.1} {:kernel-name "second" :ms 0.2}]})))}
      (fn []
        (is (thrown? clojure.lang.ExceptionInfo (gpu/profile-bound-kernel-graph! session :unprofiled)))
        (is (empty? @calls))
        (let [profile (gpu/profile-bound-kernel-graph! session :bound)]
          (is (= [[:replay :bound] [:timestamp :bound]] @calls))
          (is (= 0.5 (:device-wall-ms profile)))
          (is (< (Math/abs (- 0.3 (:kernel-total-ms profile))) 1.0e-12))
          (is (= ["first" "second"] (mapv :kernel-name (:profile profile)))))
        (is (nil? (:device-wall-ms (gpu/profile-bound-kernel-graph! session :missing-span)))
            "missing device timestamps are never replaced by host time")))))

(deftest interleaved-bound-graphs-use-device-events-and-restore-before-every-replay
  (let [calls (atom [])
        session (atom {:device-id :probe})
        candidates (mapv (fn [id] {:id id :handle id
                                   :before-sample! #(swap! calls conj [:restore id])}) [:a :b])]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'resolve-kernel-graph-entry)
       (fn [_ handle] {:profile? (not= :unprofiled handle) :runtime-graph handle})
       #'raster.gpu.core/rt-resolve
       (fn [_ name]
         (case name
           "replay-graph!" #(swap! calls conj [:replay %])
           "read-graph-timestamps!" (fn [id] (swap! calls conj [:timestamp id]) {:wall-ms 0.001})))}
      (fn []
        (doseq [opts [[:timing-source :host] [:rounds 0]]]
          (is (thrown? clojure.lang.ExceptionInfo
                       (apply gpu/measure-bound-kernel-graphs-interleaved! session candidates opts))))
        (is (thrown? clojure.lang.ExceptionInfo
                     (gpu/measure-bound-kernel-graphs-interleaved!
                      session (assoc-in candidates [1 :handle] :unprofiled))))
        (is (empty? @calls))
        (let [result (gpu/measure-bound-kernel-graphs-interleaved!
                      session candidates :warmup-rounds 1 :rounds 2)]
          (is (= (vec (mapcat (fn [id] [[:restore id] [:replay id] [:timestamp id]])
                              [:a :b :a :b :b :a])) @calls))
          (is (= [1000.0 1000.0] (get-in result [:measurements :a :samples-ns])))
          (is (= :device-event (get-in result [:measurements :b :timing-source]))))))))
