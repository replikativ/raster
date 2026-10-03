(ns raster.gpu.recorded-graph-order-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.gpu.core :as gpu]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.measurement :as measurement]
            [raster.gpu.link :as link]))

(defn- owned-phase [entry]
  (assoc entry ::cleanup/owner (cleanup/owner [])))

(deftest ownerless-plain-phases-cannot-execute-or-be-recorded
  (let [sess (atom {:device-id :ocl:0 :closed? false :graphs {}
                    :prepared {:phase {:phase :kernel}}})
        contacts (atom [])]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve)
       (fn [_ name] (swap! contacts conj name)
         (throw (ex-info "Unexpected native contact" {})))}
      (fn []
        (doseq [f [#(gpu/invoke-bound! sess :phase)
                   #(gpu/record-graph! sess [:phase] :graph)]]
          (is (= :missing-cleanup-owner
                 (try (f) (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
        (is (empty? @contacts))
        (is (empty? (:graphs @sess)))))))

(defn- with-recording-runtime [record replay destroy f]
  (with-redefs-fn
    {(ns-resolve 'raster.gpu.core 'rt-resolve)
     (fn [_ name]
       (case name "record-graph!" record "replay-graph!" replay
             (throw (ex-info "Unexpected native contact" {:name name}))))
     (ns-resolve 'raster.gpu.core 'rt-resolve-soft)
     (fn [_ name] (when (= name "destroy-graph!") destroy))}
    f))

(deftest independent-recordings-pin-the-same-prepared-source
  (let [calls (atom [])
        sess (atom {:device-id :ocl:0 :closed? false :graphs {}
                    :buffers {:data :buffer}
                    :prepared {:phase {:phase :kernel ::cleanup/owner (cleanup/owner [])}}})]
    (with-recording-runtime
      (fn [prepareds _] {:prepareds prepareds}) (fn [_])
      (fn [_] (swap! calls conj :destroy))
      (fn []
        (gpu/record-graph! sess [:phase] :first)
        (gpu/record-graph! sess [:phase] :second)
        (doseq [key [:first :second]]
          (is (= :recorded-source-retained
                 (try (gpu/release-prepared! sess :phase)
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
          (is (= :recorded-buffer-retained
                 (try (gpu/free-buffer! sess :data)
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
          (gpu/release-recorded-graph! sess key))
        (is (= [:destroy :destroy] @calls))
        (is (empty? (:graphs @sess)))
        (is (= :kernel (get-in @sess [:prepared :phase :phase])))
        (is (nil? (gpu/release-prepared! sess :phase)))
        (is (nil? (get-in @sess [:prepared :phase])))
        (gpu/free-buffer! sess :data)
        (is (nil? (get-in @sess [:buffers :data])))))))

(deftest prepared-root-footprints-allow-disjoint-staging-and-pin-owned-aliases
  (let [root (Object.) staging (Object.) freed (atom [])
        sess (atom {:device-id :ocl:0 :closed? false :graphs {} :prepared {}
                    :kernels {:phase [(artifact/make
                                      {:kernel-name "root_probe"
                                       :source "__kernel void root_probe(__global float* x, int n) { x[0] = x[0]; }"
                                       :abi [(abi/slot 'x :inout :float)
                                             (abi/slot 'n :scalar :int :role :bound)]
                                       :arguments '[x n]
                                       :launch (launch/spec {:workgroup-size [1] :group-count [1]})
                                       :effects {:kind :in-place}})]}
                    :buffers {:data root :staging staging}
                    :allocations {:data {:id :root :ownership :owned}
                                  :staging {:id :staging :ownership :owned}}})]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve)
       (fn [_ name]
         (case name
           "expand-pointer-binding" (fn [_ value] [value])
           "bind-kernel-call" (fn [call _]
                                  {:phase :kernel :kernel-call call ::cleanup/owner (cleanup/owner [])})
           "record-graph!" (fn [_ _] {:native :graph})
           "free-buffer!" (fn [buffer] (swap! freed conj buffer))))
       (ns-resolve 'raster.gpu.core 'rt-resolve-soft)
       (fn [_ name] (when (= name "destroy-graph!") (fn [_])))}
      (fn []
        (gpu/prepare! sess :phase {"x" :data} [] 1)
        (is (= :prepared-buffer-retained
               (try (gpu/free-buffer! sess :data)
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (gpu/record-graph! sess [:phase] :graph)
        (is (= #{:data} (get-in @sess [:graphs :graph :resident-footprint :buffer-keys])))
        (gpu/free-buffer! sess :staging)
        (is (= [staging] @freed))
        ;; Aliases registered after binding are not in its key set; native identity still pins
        ;; owned roots. Detaching a new borrowed alias never destroys that root.
        (swap! sess (fn [state]
                      (-> state (assoc-in [:buffers :owned-alias] root)
                          (assoc-in [:allocations :owned-alias] {:id :another-id :ownership :owned})
                          (assoc-in [:buffers :borrowed-alias] root)
                          (assoc-in [:allocations :borrowed-alias] {:id :borrowed :ownership :borrowed}))))
        (doseq [key [:data :owned-alias]]
          (is (= :recorded-buffer-retained
                 (try (gpu/free-buffer! sess key)
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
        (gpu/free-buffer! sess :borrowed-alias)
        (is (= [staging] @freed))
        (is (nil? (get-in @sess [:buffers :borrowed-alias])))
        (gpu/release-recorded-graph! sess :graph)
        (gpu/release-prepared! sess :phase)))))

(deftest qualified-emitted-source-remains-pinned-until-wrapper-release
  (let [calls (atom []) source-owner (cleanup/owner [])
        handle (gpu/map->KernelGraphHandle {:key :source :session-id :session :generation :generation})
        sess (atom {:device-id :ocl:0 :closed? false :session-id :session :graphs {}
                    :kernel-graphs {:source {:prepareds [{:phase :producer}]
                                             :generation :generation ::cleanup/owner source-owner}}})]
    (with-recording-runtime
      (fn [_ _] (swap! calls conj :record) {:native :wrapper}) (fn [_])
      (fn [_] (swap! calls conj :destroy-wrapper))
      (fn []
        (gpu/record-bound-sequence! sess [{:kind :graph :handle handle}] :wrapper)
        (is (= :recorded-source-retained
               (try (gpu/release-kernel-graph! sess handle)
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (is (= [:record] @calls))
        (is (= :live (:phase (cleanup/status source-owner))))
        (gpu/release-recorded-graph! sess :wrapper)
        (gpu/release-kernel-graph! sess handle)
        (is (= [:record :destroy-wrapper] @calls))
        (is (empty? (:kernel-graphs @sess)))
        (is (= :released (:phase (cleanup/status source-owner))))))))

(deftest recorded-replay-serializes-native-use-with-release
  (let [entered (promise) allow-return (promise) release-started (promise) destroyed (promise)
        sess (atom {:device-id :ocl:0 :closed? false :graphs {}
                    :prepared {:phase (owned-phase {:phase :kernel})}})]
    (with-recording-runtime
      (fn [_ _] {:native :graph})
      (fn [_] (deliver entered true)
        (when (= ::timeout (deref allow-return 5000 ::timeout))
          (throw (ex-info "Test replay was not unblocked" {}))))
      (fn [_] (deliver destroyed true))
      (fn []
        (gpu/record-graph! sess [:phase] :graph)
        (let [replay (future (gpu/replay! sess :graph))]
          (try
            (is (= true (deref entered 5000 ::timeout)))
            (let [release (future (deliver release-started true)
                           (gpu/release-recorded-graph! sess :graph))]
              (is (= true (deref release-started 5000 ::timeout)))
              (is (= ::pending (deref destroyed 50 ::pending)))
              (deliver allow-return true)
              (is (not= ::timeout (deref replay 5000 ::timeout)))
              (is (nil? (deref release 5000 ::timeout)))
              (is (= true (deref destroyed 5000 ::timeout))))
            (finally (deliver allow-return true))))))))

(deftest failed-recording-release-retains-registration-and-does-not-repeat-success
  (let [calls (atom [])
        failure (ex-info "Unknown native destruction outcome" {})
        sess (atom {:device-id :ocl:0 :closed? false :graphs {}
                    :prepared {:constant (owned-phase {:phase :constant :const-prologue? true})
                               :phase (owned-phase {:phase :kernel})}})]
    (with-recording-runtime
      (fn [prepareds _] {:prepareds prepareds}) (fn [_])
      (fn [graph]
        (let [phase (:phase (first (:prepareds graph)))]
          (swap! calls conj phase)
          (when (= :kernel phase) (throw failure))))
      (fn []
        (gpu/record-graph! sess [:constant :phase] :graph)
        (dotimes [_ 2]
          (is (identical? failure
                          (try (gpu/release-recorded-graph! sess :graph)
                               (catch Throwable e e)))))
        (is (= [:kernel :constant] @calls))
        (is (= [:replay] (cleanup/pending (::cleanup/owner (get-in @sess [:graphs :graph])))))
        (is (= :owner-releasing
               (try (gpu/replay! sess :graph)
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (is (= :recorded-source-retained
               (try (gpu/release-prepared! sess :phase)
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (is (identical? failure
                        (try (gpu/close-session! sess) (catch Throwable e e))))
        (is (= :releasing (:lifecycle @sess)))
        (is (= :kernel (get-in @sess [:prepared :phase :phase])))
        (is (= [:kernel :constant] @calls))))))

(deftest failed-construction-adopts-unresolved-native-debt-before-rethrow
  (let [primary (ex-info "Native construction failed" {})
        secondary (ex-info "Unknown release outcome" {})
        releases (atom 0)
        debt (cleanup/owner [{:id :native :release #(do (swap! releases inc) (throw secondary))}])
        sess (atom {:device-id :ocl:0 :closed? false :graphs {}
                    :prepared {:phase (owned-phase {:phase :kernel})}})]
    (with-recording-runtime
      (fn [_ {:keys [adopt-cleanup!]}] (adopt-cleanup! debt) (throw primary))
      (fn [_]) (fn [_] (throw (ex-info "No successful graph was acquired" {})))
      (fn []
        (is (identical? primary
                        (try (gpu/record-graph! sess [:phase] :public)
                             (catch Throwable e e))))
        (is (nil? (get-in @sess [:graphs :public])))
        (is (= 1 (count (:graphs @sess))))
        (let [[key entry] (first (:graphs @sess))]
          (is (:failed-construction? entry))
          (is (= [:replay] (cleanup/pending (::cleanup/owner entry))))
          (is (identical? secondary
                          (try (gpu/release-recorded-graph! sess key)
                               (catch Throwable e e)))))
        (is (= 1 @releases))
        (is (= :recorded-source-retained
               (try (gpu/release-prepared! sess :phase)
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))))

(deftest failed-replacement-keeps-old-registration-and-rolls-back-new-graph
  (let [failure (ex-info "Old recording cannot be destroyed" {})
        generation (atom 0)
        destroyed (atom [])
        sess (atom {:device-id :ocl:0 :closed? false :graphs {}
                    :prepared {:phase (owned-phase {:phase :kernel})}})]
    (with-recording-runtime
      (fn [_ _] {:generation (swap! generation inc)}) (fn [_])
      (fn [{:keys [generation]}]
        (swap! destroyed conj generation)
        (when (= 1 generation) (throw failure)))
      (fn []
        (gpu/record-graph! sess [:phase] :graph)
        (let [original (get-in @sess [:graphs :graph])]
          (is (identical? failure
                          (try (gpu/record-graph! sess [:phase] :graph)
                               (catch Throwable e e))))
          (is (identical? original (get-in @sess [:graphs :graph])))
          (is (= [1 2] @destroyed))
          (is (= 1 (count (:graphs @sess))))
          (is (= :owner-releasing
                 (try (gpu/replay! sess :graph)
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))))

(deftest failed-prologue-replay-preserves-primary-and-retains-prologue-debt
  (let [primary (ex-info "Prologue replay failed" {})
        secondary (ex-info "Prologue release indeterminate" {})
        recordings (atom 0)
        sess (atom {:device-id :ocl:0 :closed? false :graphs {}
                    :prepared {:constant (owned-phase {:phase :constant :const-prologue? true})
                               :phase (owned-phase {:phase :kernel})}})]
    (with-recording-runtime
      (fn [_ _] (swap! recordings inc) {:prologue true})
      (fn [_] (throw primary)) (fn [_] (throw secondary))
      (fn []
        (is (identical? primary
                        (try (gpu/record-graph! sess [:constant :phase] :graph)
                             (catch Throwable e e))))
        (is (= 1 @recordings))
        (is (= [secondary] (vec (.getSuppressed primary))))
        (is (= 1 (count (:graphs @sess))))
        (is (= [:prologue]
               (cleanup/pending (::cleanup/owner (val (first (:graphs @sess)))))))))))

(deftest measurement-hook-cannot-release-then-replay-a-recording
  (let [calls (atom [])
        sess (atom {:device-id :ocl:0 :closed? false :graphs {}
                    :prepared {:phase (owned-phase {:phase :kernel})}})]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve)
       (fn [_ name]
         (case name
           "record-graph!" (fn [_ _] {:native :graph})
           "replay-graph!" (fn [_] (swap! calls conj :replay))
           "read-graph-timestamps!" (fn [_] (swap! calls conj :timestamps) {:wall-ms 1.0})))
       (ns-resolve 'raster.gpu.core 'rt-resolve-soft)
       (fn [_ name] (when (= name "destroy-graph!") (fn [_] (swap! calls conj :destroy))))
       #'measurement/measure! (fn [sample & _] (sample))}
      (fn []
        (gpu/record-graph! sess [:phase] :graph {:profile? true})
        (is (= :stale-recording
               (try (gpu/measure-recorded-graph!
                     sess :graph :before-sample! #(gpu/release-recorded-graph! sess :graph))
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (is (= [:destroy] @calls))))))

(deftest emitted-graph-order-reads-the-bound-recording
  (let [handle (gpu/->KernelGraphHandle :emitted)
        sess (atom {:closed? false :kernel-graphs
                    {:emitted {:prepareds [{:phase :producer} {:phase :consumer}]}}})]
    (is (= {:record-time-prologue []
            :per-replay [{:kernel-phase :producer} {:kernel-phase :consumer}]
            :completion :unproven}
           (gpu/kernel-graph-execution-order sess handle)))
    (swap! sess assoc :closed? true)
    (is (= :gpu-execution-order-closed
           (try (gpu/kernel-graph-execution-order sess handle)
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest graph-order-separates-record-time-prologue-from-replay
  (let [calls (atom [])
        sess (atom {:device-id :ocl:0 :prepared
                    {:phase-a (owned-phase {:phase :a})
                     :phase-b (owned-phase {:phase :b :const-prologue? true})
                     :phase-c (owned-phase {:phase :c})}
                    :graphs {} :closed? false})
        resolve-runtime (fn [_device-id name]
                          (case name
                            "record-graph!" (fn [prepareds & _]
                                              (let [graph {:prepareds prepareds}]
                                                (swap! calls conj [:record (mapv :phase prepareds)])
                                                graph))
                            "replay-graph!" (fn [graph]
                                              (swap! calls conj [:replay
                                                                 (mapv :phase (:prepareds graph))]))
                            (throw (ex-info "unexpected runtime call" {:name name}))))
        resolve-soft (fn [_device-id name]
                       (case name
                         "destroy-graph!" (fn [graph]
                                            (swap! calls conj [:destroy
                                                               (mapv :phase (:prepareds graph))]))
                         nil))]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) resolve-runtime
       (ns-resolve 'raster.gpu.core 'rt-resolve-soft) resolve-soft}
      (fn []
        (gpu/record-graph! sess [:phase-a :phase-b :phase-c] :graph)
        (is (= [[:record [:b]] [:replay [:b]] [:record [:a :c]]] @calls))
        (is (= {:record-time-prologue [{:phase :phase-b :kernel-phase :b}]
                :per-replay [{:phase :phase-a :kernel-phase :a}
                             {:phase :phase-c :kernel-phase :c}]
                :completion :unproven}
               (gpu/graph-execution-order sess :graph)))
        (let [linked (link/map->LinkedExecutable
                      {:plan {:id :model-plan :target :ocl:0
                              :instances [{:id :model
                                           :descriptor {:steps [{} {} {}]}}]}
                       :session sess :graph-key :graph :phases [:phase-a :phase-b :phase-c]
                       :pending-inputs (atom #{}) :closed? (atom false)})]
          (is (= [{:phase :phase-b :kernel-phase :b
                   :source {:instance :model :step 1}}]
                 (:record-time-prologue (link/execution-order linked))))
          (is (= [{:instance :model :step 0} {:instance :model :step 2}]
                 (mapv :source (:per-replay (link/execution-order linked))))))
        (gpu/replay! sess :graph)
        (is (= [:replay [:a :c]] (last @calls)))
        (gpu/release-recorded-graph! sess :graph)
        (is (= [[:destroy [:a :c]] [:destroy [:b]]]
               (take-last 2 @calls)))
        (is (= :gpu-execution-order-missing
               (try (gpu/graph-execution-order sess :graph)
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))))

(deftest one-semantic-phase-can-split-between-prologue-and-replay
  (let [sess (atom {:device-id :ocl:0
                    :prepared {:phase (owned-phase (gpu/->BoundExecutableStep
                                       [{:phase :constant-transform :const-prologue? true}
                                        {:phase :value-kernel}]
                                       {} []))}
                    :graphs {} :closed? false})
        resolve-runtime (fn [_device-id name]
                          (case name
                            "record-graph!" (fn [prepareds & _] {:prepareds prepareds})
                            "replay-graph!" (fn [_])
                            (throw (ex-info "unexpected runtime call" {:name name}))))]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) resolve-runtime
       (ns-resolve 'raster.gpu.core 'rt-resolve-soft) (fn [& _] nil)}
      (fn []
        (gpu/record-graph! sess [:phase] :graph)
        (is (= {:record-time-prologue
                [{:phase :phase :kernel-phase :constant-transform}]
                :per-replay [{:phase :phase :kernel-phase :value-kernel}]
               :completion :unproven}
               (gpu/graph-execution-order sess :graph)))))))

(deftest bound-phases-and-emitted-graphs-record-in-one-order
  (let [calls (atom [])
        handle (gpu/->KernelGraphHandle :emitted)
        sess (atom {:device-id :ocl:0 :closed? false
                    :prepared {:before (owned-phase {:phase :before})
                               :constant (owned-phase {:phase :constant :const-prologue? true})
                               :after (owned-phase {:phase :after})}
                    :kernel-graphs {:emitted {:prepareds [{:phase :producer}
                                                          {:phase :consumer}]}}
                    :graphs {}})
        resolve-runtime (fn [_device-id name]
                          (case name
                            "record-graph!" (fn [prepareds & [options]]
                                              (swap! calls conj [:record (mapv :phase prepareds) options])
                                              {:prepareds prepareds})
                            "replay-graph!" (fn [graph]
                                              (swap! calls conj [:replay
                                                                 (mapv :phase (:prepareds graph))]))
                            (throw (ex-info "unexpected runtime call" {:name name}))))]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) resolve-runtime
       (ns-resolve 'raster.gpu.core 'rt-resolve-soft)
       (fn [_device-id name]
         (when (= name "destroy-graph!")
           (fn [graph]
             (swap! calls conj [:destroy (mapv :phase (:prepareds graph))]))))}
      (fn []
        (gpu/record-bound-sequence!
         sess [{:kind :phase :phase :before}
               {:kind :graph :handle handle}
               {:kind :phase :phase :constant}
               {:kind :phase :phase :after}]
         :mixed {:profile? true})
        (is (= [[:record [:before :producer :consumer :constant :after]
                 {:barriers? true :profile? true}]]
               (mapv (fn [[kind phases opts]] [kind phases (dissoc opts :adopt-cleanup!)]) @calls)))
        (is (fn? (get-in @calls [0 2 :adopt-cleanup!])))
        (is (= {:record-time-prologue []
                :per-replay [{:phase :before :kernel-phase :before}
                             {:phase handle :kernel-phase :producer}
                             {:phase handle :kernel-phase :consumer}
                             {:phase :constant :kernel-phase :constant}
                             {:phase :after :kernel-phase :after}]
                :completion :unproven}
               (gpu/graph-execution-order sess :mixed)))
        (let [before @calls]
          (is (= :gpu-recording-invalid-source
                 (try (gpu/record-bound-sequence!
                       sess [{:kind :graph :phase :before}] :mixed)
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
          (is (= before @calls))
          (is (= :gpu-recording-unbound-phase
                 (try (gpu/record-bound-sequence!
                       sess [{:kind :phase :phase :missing}] :mixed)
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
          (is (= before @calls)))
        (let [before @calls]
          (doseq [operation [#(gpu/release-kernel-graph! sess handle)
                             #(gpu/prepare! sess :before {} [] 1)
                             #(gpu/bind-step! sess {:phase :before} [] {})]]
            (is (= :recorded-source-retained
                   (try (operation)
                        (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
          (is (= before @calls)))
        (gpu/release-recorded-graph! sess :mixed)
        (is (= [:destroy [:before :producer :consumer :constant :after]]
               (last @calls)))))))
