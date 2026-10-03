(ns raster.gpu.program-cleanup-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.emitted-parallel-program-call :as call]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.gpu.core :as gpu]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.link :as link]
            [raster.gpu.parallel-program :as program]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.value :as value]))

(defn- prepared [keys run! release!]
  ((ns-resolve 'raster.gpu.parallel-program 'own-prepared)
   (program/map->PreparedParallelProgram
    {:call {:steps (mapv (fn [_] (call/map->EmittedEquationCall {})) keys) :outputs {}}
     :plan {:step-keys (zipmap (range) keys)} :handles (zipmap keys keys)
     :binding-order keys :run! run! :release! release! :closed? (atom false)})))

(defn- linked [owns?]
  ((ns-resolve 'raster.gpu.link 'own-linked-executable)
   (link/map->LinkedExecutable
    {:session ::session :owns-session? owns? :graph-key :recording :phases [:a :b]
     :allocation-keys [:x :y] :closed? (atom false) :lifetime-lock (Object.)
     :output-leases (atom 0) :execution-state (atom {:value-epoch 0})})))

(deftest partial-program-construction-retains-failed-cleanup
  ;; Stub only compiler validation/staging: this oracle concerns returned runtime handles,
  ;; not acceptance of a numerical program. Native acquisition before a returned handle
  ;; remains the injected executor's responsibility.
  (doseq [adopt? [false true]]
    (let [primary (ex-info "third bind failed" {}) secondary (ex-info "b destroy uncertain" {})
          released (atom []) adopted (atom nil)
          error (with-redefs-fn
                  {#'call/validate! identity
                   (ns-resolve 'raster.gpu.parallel-program 'preparation-plan)
                   (fn [& _] {:entries [{:key :a} {:key :b} {:key :c}] :step-keys {}})}
                  #(try
                     (program/prepare-with!
                      {:steps []}
                      (cond-> {:bind! (fn [key & _] (if (= :c key) (throw primary) key))
                               :run! identity
                               :release! (fn [key] (swap! released conj key)
                                           (when (= :b key) (throw secondary)))}
                        adopt? (assoc :adopt-cleanup! (fn [owner] (reset! adopted owner)))))
                     (catch Throwable e e)))
          owner (if adopt? @adopted (::cleanup/unresolved (ex-data error)))]
      (is (some? owner))
      (is (identical? primary (if adopt? error (.getCause error))))
      (is (= [secondary] (vec (.getSuppressed primary))))
      (is (= [:b :a] @released))
      (is (= [[:graph :b]] (cleanup/pending owner)))
      (is (identical? secondary (try (cleanup/release! owner) (catch Throwable e e))))
      (is (= [:b :a] @released)))))

(deftest partial-sequence-construction-retains-child-ownership
  (let [primary (ex-info "third program failed" {}) secondary (ex-info "child uncertain" {})
        released (atom []) adopted (atom nil)]
    (with-redefs [program/prepare-with!
                  (fn [id _]
                    (when (= :c id) (throw primary))
                    (prepared [id] identity
                              (fn [key] (swap! released conj key)
                                (when (= :b key) (throw secondary)))))]
      (is (identical? primary
                     (try (program/prepare-sequence-with!
                           [{:id :a :call :a} {:id :b :call :b} {:id :c :call :c}]
                           {:bind! identity :run! identity :release! identity
                            :adopt-cleanup! (fn [owner] (reset! adopted owner))})
                          (catch Throwable e e)))))
    (is (= [:b :a] @released))
    (is (= [secondary] (vec (.getSuppressed primary))))
    (is (= [[:instance :b]] (cleanup/pending @adopted)))
    (is (identical? secondary (try (cleanup/release! @adopted) (catch Throwable e e))))
    (is (= [:b :a] @released))))

(deftest session-adoption-pins-original-and-late-alias-roots
  (let [root {:dtype :float :n-elements 1} failure (ex-info "unknown destroy" {})
        attempts (atom 0)
        owner (cleanup/owner [{:id :program
                               :release #(do (swap! attempts inc) (throw failure))}])
        sess (atom {:device-id :ocl:0 :buffers {:x root} :prepared {}
                    :allocations {:x {:id :root :ownership :owned}}
                    :kernel-graphs {} :graphs {} :events {}})]
    (is (identical? failure (try (cleanup/release! owner) (catch Throwable e e))))
    (gpu/retain-program-cleanup! sess owner [:x])
    (gpu/retain-program-cleanup! sess owner [:x])
    (is (= 1 (count (:prepared @sess))))
    ;; A distinct registration ID does not make the same native root independent.
    (swap! sess assoc-in [:buffers :late-alias] root)
    (swap! sess assoc-in [:allocations :late-alias] {:id :other-id :ownership :owned})
    (doseq [key [:x :late-alias]]
      (is (= :prepared-buffer-retained
             (try (gpu/free-buffer! sess key)
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (dotimes [_ 2]
      (is (identical? failure (try (gpu/close-session! sess) (catch Throwable e e)))))
    (is (= 1 @attempts))
    (is (= #{:x :late-alias} (set (keys (:buffers @sess)))))))

(deftest retry-safe-adopted-program-releases-before-session-roots
  (let [root {:dtype :float :n-elements 1} releases (atom []) attempts (atom 0)
        failure (ex-info "not destroyed" {:cleanup-retry-safe? true})
        owner (cleanup/owner [{:id :program
                               :release #(do (swap! releases conj :program)
                                             (when (= 1 (swap! attempts inc)) (throw failure)))}])
        sess (atom {:device-id :ocl:0 :buffers {:x root} :prepared {}
                    :allocations {:x {:id :root :ownership :owned}}
                    :kernel-graphs {} :graphs {} :events {}})]
    (is (identical? failure (try (cleanup/release! owner) (catch Throwable e e))))
    (gpu/retain-program-cleanup! sess owner [:x])
    (with-redefs-fn {(ns-resolve 'raster.gpu.core 'rt-resolve)
                    (fn [_ name]
                      (case name
                        "free-buffer!" (fn [_] (swap! releases conj :root))
                        "close-kernel-arena!" (fn [_] (swap! releases conj :arena))))}
      #(gpu/close-session! sess))
    (is (= [:program :program :root :arena] @releases))
    (is (empty? (:prepared @sess)))
    (is (empty? (:buffers @sess)))
    (gpu/close-session! sess)
    (is (= [:program :program :root :arena] @releases))))

(deftest failed-prepared-parent-prevents-second-graph-retry-in-one-close
  (let [attempts (atom 0) failure (ex-info "not destroyed" {:cleanup-retry-safe? true})
        child (cleanup/owner [{:id :kernel :release #(when (<= (swap! attempts inc) 2)
                                                      (throw failure))}])
        sess (atom {:device-id :ocl:0 :session-id :session :buffers {} :allocations {}
                    :prepared {} :graphs {} :events {}
                    :kernel-graphs {:g {:generation :generation ::cleanup/owner child}}})
        handle (gpu/->KernelGraphHandle :g :session :generation)
        parent (cleanup/owner [{:id :program :release #(gpu/release-kernel-graph! sess handle)}])]
    (is (identical? failure (try (cleanup/release! parent) (catch Throwable e e))))
    (gpu/retain-program-cleanup! sess parent [])
    (is (identical? failure (try (gpu/close-session! sess) (catch Throwable e e))))
    (is (= 2 @attempts) "a prepared-layer failure must not descend to graphs in the same close")
    (is (contains? (:kernel-graphs @sess) :g))
    (with-redefs-fn {(ns-resolve 'raster.gpu.core 'rt-resolve)
                    (fn [& _] (fn [& _]))}
      #(gpu/close-session! sess))
    (is (= 3 @attempts))
    (is (empty? (:kernel-graphs @sess)))
    (is (empty? (:prepared @sess)))))

(deftest direct-graph-sequence-preflights-before-binding
  (let [contacts (atom 0)]
    (is (= :parallel-program-executor
           (try (program/prepare-sequence-with!
                 [{:id :graph :kind :graph :call {}}]
                 {:bind! (fn [& _] (swap! contacts inc)) :run! identity})
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (zero? @contacts))))

(deftest direct-graph-handle-is-owned-before-constructing-its-wrapper
  (let [own (ns-resolve 'raster.gpu.parallel-program 'own-prepared) original @own
        primary (ex-info "wrapper failed" {}) secondary (ex-info "destroy uncertain" {})
        released (atom []) adopted (atom nil)]
    (with-redefs-fn
      {own (fn [value & owners]
             (if (program/prepared-kernel-graph? value) (throw primary)
                 (apply original value owners)))}
      #(is (identical? primary
                      (try (program/prepare-sequence-with!
                            [{:id :graph :kind :graph :call {}}]
                            {:bind! (fn [& _] :returned-handle) :run! identity
                             :release! (fn [handle] (swap! released conj handle) (throw secondary))
                             :adopt-cleanup! (fn [owner] (reset! adopted owner))})
                           (catch Throwable e e)))))
    (is (= [:returned-handle] @released))
    (is (= [secondary] (vec (.getSuppressed primary))))
    (is (= [:graph] (cleanup/pending @adopted)))
    (is (identical? secondary (try (cleanup/release! @adopted) (catch Throwable e e))))
    (is (= [:returned-handle] @released))))

(deftest owned-link-validation-failure-retains-the-session-after-failed-close
  ;; Validation is stubbed explicitly to isolate ownership immediately after session creation.
  ;; Both admission declines precede allocation or kernel acquisition.
  (doseq [kind [:target :external]]
    (let [sess (atom {:device-id (if (= :target kind) :ze:0 :ocl:0)})
          secondary (ex-info "session close uncertain" {}) closes (atom 0)
          plan {:target :ocl:0 :instances [] :nodes {}}]
      (with-redefs [link-plan/validate-with-effect-evidence! (fn [plan] {:plan plan})
                    gpu/make-session (fn [_] sess)
                    gpu/close-session! (fn [actual] (is (identical? sess actual))
                                         (swap! closes inc) (throw secondary))]
        (let [error (try (link/instantiate! plan
                                          (if (= :external kind) {:external-buffers {:extra ::buffer}} {}))
                         (catch Throwable e e))
              owner (::cleanup/unresolved (ex-data error))
              primary (.getCause error)]
          (is (= (if (= :target kind) :link-session-target :link-external-bindings)
                 (:reason (ex-data primary))))
          (is (= [secondary] (vec (.getSuppressed primary))))
          (is (= [:session] (cleanup/pending owner)))
          (is (identical? secondary (try (cleanup/release! owner) (catch Throwable e e))))
          (is (= 1 @closes)))))))

(deftest repeated-parallel-close-preserves-failures-without-repeating-success
  (let [calls (atom []) primary (ex-info "b uncertain" {}) secondary (ex-info "a uncertain" {})
        p (prepared [:a :b :c] identity
                    #(do (swap! calls conj %)
                         (case % :a (throw secondary) :b (throw primary) nil)))]
    (dotimes [_ 2]
      (is (identical? primary (try (program/release-prepared! p) (catch Throwable e e)))))
    (is (= [:c :b :a] @calls))
    (is (= [secondary] (vec (.getSuppressed primary))))
    (is (= [[:graph :b] [:graph :a]] (cleanup/pending (::cleanup/owner p))))
    (is (= :parallel-program-closed
           (try (program/run-prepared! p)
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest only-explicit-retry-safe-parallel-failures-are-retried
  (let [calls (atom []) attempts (atom 0)
        failure (ex-info "not destroyed yet" {:cleanup-retry-safe? true})
        p (prepared [:a :b] identity
                    #(do (swap! calls conj %)
                         (when (and (= :b %) (= 1 (swap! attempts inc))) (throw failure))))]
    (is (identical? failure (try (program/release-prepared! p) (catch Throwable e e))))
    (program/release-prepared! p)
    (program/release-prepared! p)
    (is (= [:b :a :b] @calls))
    (is (empty? (cleanup/pending (::cleanup/owner p))))))

(deftest nested-sequence-retains-failed-child-and-releases-independent-sibling
  (let [calls (atom []) failure (ex-info "child failed" {})
        a (prepared [:a] identity #(swap! calls conj %))
        b (prepared [:b] identity #(do (swap! calls conj %) (throw failure)))
        p ((ns-resolve 'raster.gpu.parallel-program 'own-prepared)
           (program/map->PreparedParallelSequence
            {:instances [{:id :a :program a} {:id :b :program b}] :closed? (atom false)}))]
    (dotimes [_ 2]
      (is (identical? failure (try (program/release-prepared! p) (catch Throwable e e)))))
    (is (= [:b :a] @calls))
    (is (= [[:instance :b]] (cleanup/pending (::cleanup/owner p))))))

(deftest parallel-callbacks-cannot-release-their-active-owner
  (doseq [operation [:run :profile :info]]
    (let [p (atom nil) released (atom []) declines (atom [])
          callback (fn [_]
                     (swap! declines conj
                            (try (program/release-prepared! @p)
                                 (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))
                     {:profile [] :kernel-total-ms 0 :device-wall-ms 0})]
      (reset! p (prepared [:a :b] callback #(swap! released conj %)))
      (case operation
        :run (program/run-prepared! @p)
        :profile (program/profile-prepared! @p callback)
        :info (program/execution-info @p callback))
      (is (= [:parallel-program-in-use :parallel-program-in-use] @declines))
      (is (false? @(:closed? @p)))
      (is (empty? @released))
      (program/release-prepared! @p)
      (is (= [:b :a] @released)))))

(deftest concurrent-parallel-release-waits-for-the-complete-use-scope
  (let [entered (promise) resume (promise) releasing (promise) destroyed (promise)
        p (prepared [:a] (fn [_] (deliver entered true) (deref resume 5000 ::timeout))
                    (fn [_] (deliver destroyed true)))
        run (future (program/run-prepared! p))]
    (try
      (is (= true (deref entered 5000 ::timeout)))
      (let [close (future (deliver releasing true) (program/release-prepared! p))]
        (is (= true (deref releasing 5000 ::timeout)))
        (is (= ::pending (deref destroyed 50 ::pending)))
        (deliver resume true)
        (is (not= ::timeout (deref run 5000 ::timeout)))
        (is (nil? (deref close 5000 ::timeout)))
        (is (= true (deref destroyed 5000 ::timeout))))
      (finally (deliver resume true)))))

(deftest linked-close-retains-recording-debt-before-any-dependent-release
  (let [p (linked false) calls (atom []) failure (ex-info "recording uncertain" {})]
    (with-redefs [gpu/release-recorded-graph! (fn [& _] (swap! calls conj :recording) (throw failure))
                  gpu/release-prepared! (fn [& _] (swap! calls conj :phase))
                  gpu/free-buffer! (fn [& _] (swap! calls conj :buffer))]
      (dotimes [_ 2]
        (is (identical? failure (try (link/close! p) (catch Throwable e e)))))
      (is (= [:recording] @calls))
      (is (= :poisoned (:phase (cleanup/status (::cleanup/owner p))))))))

(deftest linked-retry-safe-close-releases-dependent-layers-once
  (let [p (linked false) calls (atom []) attempts (atom 0)
        failure (ex-info "recording still live" {:cleanup-retry-safe? true})]
    (with-redefs [gpu/release-recorded-graph!
                  (fn [& _] (swap! calls conj :recording)
                    (when (= 1 (swap! attempts inc)) (throw failure)))
                  gpu/release-prepared! (fn [_ phase] (swap! calls conj phase))
                  gpu/free-buffer! (fn [_ key] (swap! calls conj key))]
      (is (identical? failure (try (link/close! p) (catch Throwable e e))))
      (link/close! p)
      (link/close! p)
      (is (= [:recording :recording :b :a :y :x] @calls)))))

(deftest owned-linked-close-has-one-session-resource-and-sticky-failure
  (let [p (linked true) calls (atom 0) failure (ex-info "session uncertain" {})]
    (with-redefs [gpu/close-session! (fn [_] (swap! calls inc) (throw failure))]
      (dotimes [_ 2]
        (is (identical? failure (try (link/close! p) (catch Throwable e e)))))
      (is (= 1 @calls))
      (is (= [:session] (cleanup/pending (::cleanup/owner p)))))))

(deftest linked-independent-phase-failures-retain-all-dependent-allocations
  (let [p (linked false) calls (atom []) primary (ex-info "b uncertain" {})
        secondary (ex-info "a uncertain" {})]
    (with-redefs [gpu/release-recorded-graph! (fn [& _] (swap! calls conj :recording))
                  gpu/release-prepared!
                  (fn [_ phase] (swap! calls conj phase)
                    (throw (if (= :b phase) primary secondary)))
                  gpu/free-buffer! (fn [_ key] (swap! calls conj key))]
      (dotimes [_ 2]
        (is (identical? primary (try (link/close! p) (catch Throwable e e)))))
      (is (= [secondary] (vec (.getSuppressed primary))))
      (is (= [:recording :b :a] @calls))
      (is (= [[:phase :b] [:phase :a] [:allocation :y] [:allocation :x]]
             (cleanup/pending (::cleanup/owner p)))))))

(deftest concurrent-linked-close-waits-for-the-composite-callback
  (let [p (linked true) entered (promise) resume (promise) releasing (promise) destroyed (promise)]
    (with-redefs [gpu/close-session! (fn [_] (deliver destroyed true))]
      (let [use (future (link/with-unleased-execution!
                        p :test #(do (deliver entered true) (deref resume 5000 ::timeout))))]
        (try
          (is (= true (deref entered 5000 ::timeout)))
          (let [close (future (deliver releasing true) (link/close! p))]
            (is (= true (deref releasing 5000 ::timeout)))
            (is (= ::pending (deref destroyed 50 ::pending)))
            (deliver resume true)
            (is (not= ::timeout (deref use 5000 ::timeout)))
            (is (nil? (deref close 5000 ::timeout)))
            (is (= true (deref destroyed 5000 ::timeout))))
          (finally (deliver resume true)))))))

(deftest linked-reentrant-close-and-lease-declines-do-not-start-teardown
  (let [p (linked true) calls (atom 0)]
    (with-redefs [gpu/close-session! (fn [_] (swap! calls inc))]
      (is (= :link-execution-in-use
             (link/with-unleased-execution!
              p :test #(try (link/close! p)
                            (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
      (is (false? @(:closed? p)))
      (is (= :live (:phase (cleanup/status (::cleanup/owner p)))))
      (is (zero? @(::link/active-uses p)))
      (reset! (:output-leases p) 1)
      (is (= :link-output-lease-active
             (try (link/close! p)
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (false? @(:closed? p)))
      (is (zero? @calls))
      (reset! (:output-leases p) 0)
      (link/close! p)
      (is (= 1 @calls)))))

(deftest execution-state-watch-cannot-close-before-replay
  (let [p (assoc (linked true) :pending-inputs (atom #{}) :plan {:nodes {}}
                 :output-ready? (atom false) :completed-replays (atom 0))
        attempts (atom []) replays (atom 0) closes (atom 0)]
    (add-watch (:execution-state p) ::close
               (fn [_ _ before after]
                 (when (not= (:value-epoch before) (:value-epoch after))
                   (swap! attempts conj
                          (try (link/close! p)
                               (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
    (with-redefs [gpu/replay! (fn [& _] (swap! replays inc))
                  gpu/close-session! (fn [_] (swap! closes inc))
                  link/outputs (fn [_] {})]
      (try
        (link/run! p)
        (is (= [:link-execution-in-use] @attempts))
        (is (= 1 @replays))
        (is (zero? @closes))
        (is (false? @(:closed? p)))
        (is (true? @(:output-ready? p)))
        (is (zero? @(::link/active-uses p)))
        (finally (remove-watch (:execution-state p) ::close)))
      (link/close! p)
      (is (= 1 @closes)))))

(deftest upload-state-watch-cannot-close-before-transfer
  (let [p (assoc (linked true) :plan {:nodes {:x {:view {:shape [1]}}}}
                 :pending-inputs (atom #{:x}) :tainted-inputs (atom #{})
                 :output-ready? (atom false))
        attempts (atom []) transfers (atom 0) closes (atom 0)]
    (add-watch (:execution-state p) ::close
               (fn [_ _ before after]
                 (when (not= (:value-epoch before) (:value-epoch after))
                   (swap! attempts conj
                          (try (link/close! p)
                               (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
    (with-redefs [link-plan/validate-node-source! (fn [& _])
                  link/node-view (fn [& _] ::view)
                  gpu/upload-range! (fn [& _] (swap! transfers inc))
                  gpu/close-session! (fn [_] (swap! closes inc))]
      (try
        (is (identical? p (link/upload! p :x (float-array [1]))))
        (is (= [:link-execution-in-use] @attempts))
        (is (= 1 @transfers))
        (is (zero? @closes))
        (is (empty? @(:pending-inputs p)))
        (is (zero? @(::link/active-uses p)))
        (finally (remove-watch (:execution-state p) ::close)))
      (link/close! p)
      (is (= 1 @closes)))))

(deftest throwing-mutation-watch-balances-use-and-prevents-replay
  (let [p (assoc (linked true) :pending-inputs (atom #{}) :plan {:nodes {}}
                 :output-ready? (atom false) :completed-replays (atom 0))
        failure (ex-info "watch failed" {}) replays (atom 0) closes (atom 0)]
    (add-watch (:execution-state p) ::throw
               (fn [_ _ before after]
                 (when (not= (:value-epoch before) (:value-epoch after)) (throw failure))))
    (with-redefs [gpu/replay! (fn [& _] (swap! replays inc))
                  gpu/close-session! (fn [_] (swap! closes inc))]
      (try
        (is (identical? failure (try (link/run! p) (catch Throwable e e))))
        (is (zero? @replays))
        (is (identical? failure (:failure @(:execution-state p))))
        (is (zero? @(::link/active-uses p)))
        (finally (remove-watch (:execution-state p) ::throw)))
      (link/close! p)
      (is (= 1 @closes)))))

(deftest missing-cleanup-owners-decline-before-close-state-mutation
  (doseq [[p close!] [[(dissoc (prepared [:a] identity identity) ::cleanup/owner)
                      program/release-prepared!]
                     [(dissoc (linked true) ::cleanup/owner) link/close!]]]
    (is (= :missing-cleanup-owner
           (try (close! p) (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (false? @(:closed? p)))))

(deftest linked-live-operations-require-a-live-cleanup-owner
  (let [p (linked true) unowned (dissoc p ::cleanup/owner)]
    (doseq [f [#(link/run! unowned)
               #(link/with-unleased-execution! unowned :test
                                              (fn [] (throw (AssertionError. "callback entered"))))]]
      (is (= :missing-cleanup-owner
             (try (f) (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (with-redefs [gpu/close-session! (fn [_])]
      (cleanup/release! (::cleanup/owner p)))
    (is (false? @(:closed? p)))
    (is (= :owner-releasing
           (try (link/run! p) (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest compiled-close-reports-retained-failure-and-invalidates-outputs-once
  (let [p (linked true) failure (ex-info "session teardown uncertain" {})
        invalidated (atom []) closes (atom 0)
        c (compiled/map->Compiled {:executable p :live-outputs (atom [:output])})]
    (with-redefs [value/free! #(swap! invalidated conj %)
                  gpu/close-session! (fn [_] (swap! closes inc) (throw failure))]
      (dotimes [_ 2]
        (is (identical? failure (try (compiled/close! c) (catch Throwable e e)))))
      (is (= [:output] @invalidated))
      (is (= 1 @closes))
      (is (nil? @(:live-outputs c))))))

(deftest compiled-close-reentrant-decline-preserves-projected-outputs
  (let [p (linked true) invalidated (atom []) closes (atom 0)
        c (compiled/map->Compiled {:executable p :live-outputs (atom [:output])})]
    (with-redefs [value/free! #(swap! invalidated conj %)
                  gpu/close-session! (fn [_] (swap! closes inc))]
      (is (= :link-execution-in-use
             (link/with-unleased-execution!
              p :test #(try (compiled/close! c)
                            (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
      (is (= [:output] @(:live-outputs c)))
      (is (empty? @invalidated))
      (is (zero? @closes))
      (is (false? @(:closed? p)))
      (compiled/close! c)
      (is (= [:output] @invalidated))
      (is (= 1 @closes)))))
