(ns raster.gpu.runtime-root-native-test
  "Hardware-free fault injection through the production OpenCL and Level Zero initializers."
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.ze-runtime :as ze]
            [raster.gpu.runtime-root :as root]
            [raster.gpu.resource-cleanup :as cleanup])
  (:import [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.lang.invoke MethodHandles MethodType]))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))

(defn- native-handle [arity f]
  (delay (.bindTo (.findVirtual (MethodHandles/lookup) clojure.lang.IFn "invoke"
                                (MethodType/genericMethodType arity)) f)))

(defn- with-runtime [backend options test!]
  (let [namespace (if (= backend :ocl) 'raster.gpu.ocl-runtime 'raster.gpu.ze-runtime)
        v #(ns-resolve namespace %)
        state (atom {:initialized? false}) registry (atom {}) calls (atom []) queues (atom 0)
        primary (ex-info "native call/readback outcome unknown" {})
        platform (MemorySegment/ofAddress 100) device (MemorySegment/ofAddress 101)
        context (MemorySegment/ofAddress 102)
        contact! (fn [id]
                   (swap! calls conj id)
                   (when-let [callback (:on-contact options)] (callback state id))
                   (when (= id (:throw options)) (throw primary)))
        status (fn [id] (if (= id (:status options)) -1 0))
        handle (fn [id address]
                 (if (= id (:null options)) MemorySegment/NULL (MemorySegment/ofAddress address)))
        readback (fn [segment]
                   (let [handle (.get ^MemorySegment segment ValueLayout/ADDRESS 0)
                         id (case (.address ^MemorySegment handle)
                              102 :context 103 :sync 104 :async nil)]
                     (when (or (:readback-failure? options) (= id (:readback-failure options)))
                       (throw primary))
                     handle))
        methods
        (if (= backend :ocl)
          {(v 'device-info) (fn [_]
                              (contact! :properties)
                              {:name "fake" :integrated? true :buffer-offset-alignment 64})
           (v 'h-clGetPlatformIDs)
           (native-handle 3 (fn [n out count]
                              (contact! :platforms)
                              (.set ^MemorySegment count ValueLayout/JAVA_INT 0 (int 1))
                              (when (pos? n) (.set ^MemorySegment out ValueLayout/ADDRESS 0 platform))
                              0))
           (v 'h-clGetDeviceIDs)
           (native-handle 5 (fn [_ _ n out count]
                              (.set ^MemorySegment count ValueLayout/JAVA_INT 0 (int 1))
                              (when (pos? n) (.set ^MemorySegment out ValueLayout/ADDRESS 0 device))
                              0))
           (v 'h-clCreateContext)
           (native-handle 6 (fn [_ _ _ _ _ err]
                              (contact! :context)
                              (.set ^MemorySegment err ValueLayout/JAVA_INT 0 (int (status :context)))
                              (when (:readback-failure? options) (throw primary))
                              (if (= :context (:null options)) MemorySegment/NULL context)))
           (v 'h-clCreateCommandQueue)
           (native-handle 4 (fn [_ _ _ err]
                              (let [id (if (= 1 (swap! queues inc)) :compute :transfer)]
                                (contact! id)
                                (.set ^MemorySegment err ValueLayout/JAVA_INT 0 (int (status id)))
                                (handle id (+ 102 @queues)))))
           (v 'h-clReleaseCommandQueue)
           (native-handle 1 (fn [queue]
                              (let [id (if (= 103 (.address ^MemorySegment queue)) :release-compute :release-transfer)]
                                (contact! id) (status id))))
           (v 'h-clReleaseContext)
           (native-handle 1 (fn [_] (contact! :release-context) (status :release-context)))}
          (let [method (fn [arity f] (native-handle arity f))]
            {(v 'h-zeInit) (method 1 (fn [_] 0))
             (v 'h-zeDriverGet)
             (method 2 (fn [count out]
                         (.set ^MemorySegment count ValueLayout/JAVA_INT 0 (int 1))
                         (when-not (= MemorySegment/NULL out)
                           (.set ^MemorySegment out ValueLayout/ADDRESS 0 platform)) 0))
             (v 'h-zeDeviceGet)
             (method 3 (fn [_ count out]
                         (.set ^MemorySegment count ValueLayout/JAVA_INT 0 (int 1))
                         (when-not (= MemorySegment/NULL out)
                           (.set ^MemorySegment out ValueLayout/ADDRESS 0 device)) 0))
             (v 'h-zeContextCreate)
             (method 3 (fn [_ _ out]
                         (contact! :context)
                         (.set ^MemorySegment out ValueLayout/ADDRESS 0 (handle :context 102))
                         (status :context)))
             (v 'h-zeCommandListCreateImmediate)
             (method 4 (fn [_ _ desc out]
                         (let [id (if (= 1 (.get ^MemorySegment desc ValueLayout/JAVA_INT 28)) :sync :async)]
                           (contact! id)
                           (.set ^MemorySegment out ValueLayout/ADDRESS 0
                                 (handle id (if (= id :sync) 103 104)))
                           (status id))))
             (v 'h-zeDeviceGetProperties)
             (method 2 (fn [_ out]
                         (contact! :properties)
                         (.set ^MemorySegment out ValueLayout/JAVA_INT 24 (int 0x1234))
                         (status :properties)))
             (v 'h-zeCommandListDestroy)
             (method 1 (fn [list]
                         (let [id (if (= 103 (.address ^MemorySegment list)) :release-sync :release-async)]
                           (contact! id) (status id))))
             (v 'h-zeContextDestroy)
             (method 1 (fn [_] (contact! :release-context) (status :release-context)))
             (v 'read-ptr) readback}))]
    (with-redefs-fn (merge {(v 'state) state (v 'kernel-registry) registry} methods)
      (fn []
        (try
          (test! {:state state :calls calls :primary primary :registry registry
                  :init! (if (= backend :ocl) ocl/init! ze/init!)
                  :shutdown! (if (= backend :ocl) ocl/shutdown! ze/shutdown!)
                  :reset! (if (= backend :ocl) ocl/reset! ze/reset!)})
          (finally
            ;; Native handles here are synthetic. Reclaim only this fixture's real host
            ;; descriptor Arenas after assertions; never touch a real runtime or driver object.
            (doseq [entry (vals @state)
                    :when (and (map? entry) (::root/root? entry))
                    :let [arena (:resource @(get (:slots entry) :arena))]
                    :when (and arena (.isAlive (.scope ^Arena arena)))]
              (.close ^Arena arena))))))))

(deftest successful-roots-are-idempotent-and-live-reset-is-pure-fail-closed
  (doseq [backend [:ocl :ze]]
    (with-runtime backend {}
      (fn [{:keys [init! shutdown! reset! state calls]}]
        (init!)
        (let [entry (root/assert-live! state) before @calls]
          (init!)
          (is (identical? entry (root/assert-live! state)))
          (doseq [close [shutdown! reset!]]
            (is (= :runtime-root-leases-incomplete (:reason (ex-data (error-of close))))))
          (is (= before @calls)))))))

(deftest native-create-status-null-and-readback-failures-keep-root-debt
  (doseq [backend [:ocl :ze]
          id (if (= backend :ocl) [:context :compute :transfer] [:context :sync])
          failure-kind [:throw :status :null]]
    (with-runtime backend {failure-kind id}
      (fn [{:keys [init! shutdown! state calls]}]
        (let [primary (error-of init!) before @calls]
          (is (some? primary))
          (is (some? (::root/entry @state)))
          (is (= :runtime-root-unavailable (:reason (ex-data (error-of init!)))))
          (is (= before @calls))
          (is (identical? primary (error-of shutdown!)))
          (is (= before @calls))))))
  (doseq [backend [:ocl :ze]]
    (with-runtime backend {:readback-failure? true}
      (fn [{:keys [init! state primary]}]
        (is (identical? primary (error-of init!)))
        (is (some? (::root/entry @state)))))))

(deftest known-query-failure-rolls-back-all-acquired-parents
  (doseq [backend [:ocl :ze]]
    (with-runtime backend {:throw :properties}
      (fn [{:keys [init! primary state calls]}]
        (is (identical? primary (error-of init!)))
        (is (nil? (::root/entry @state)))
        (is (false? (:initialized? @state)))
        (is (= (if (= backend :ocl) [] [:release-sync :release-context])
               (filterv #(contains? #{:release-sync :release-context} %) @calls)))))))

(deftest lazy-async-list-concurrency-and-unknown-failure-use-the-root-slot
  (doseq [fail? [false true]]
    (with-runtime :ze (if fail? {:throw :async} {})
      (fn [{:keys [init! shutdown! calls primary state]}]
        (init!)
        (if fail?
          (do
            (is (identical? primary (error-of ze/async-cmd-list)))
            (let [before @calls]
              (is (= :runtime-root-unavailable (:reason (ex-data (error-of ze/async-cmd-list)))))
              (is (= :runtime-root-leases-incomplete (:reason (ex-data (error-of shutdown!)))))
              (is (= before @calls))))
          (let [jobs (mapv (fn [_] (future (ze/async-cmd-list))) (range 4))
                handles (mapv deref jobs)]
            (is (every? #(identical? (first handles) %) handles))
            (is (= 1 (count (filter #{:async} @calls))))
            (let [entry (::root/entry @state)
                  slot (get (:slots entry) :async-list)]
              (is (= :live (:phase @slot)))
              (is (identical? (first handles) (:resource @slot))))))))))

(deftest initialization-callbacks-cannot-reenter-shutdown-or-registration
  (doseq [backend [:ocl :ze]]
    (let [observed (atom [])
          shutdown! (if (= backend :ocl) ocl/shutdown! ze/shutdown!)
          register! (if (= backend :ocl) ocl/register-kernel! ze/register-kernel!)
          namespace (if (= backend :ocl) 'raster.gpu.ocl-runtime 'raster.gpu.ze-runtime)
          v #(ns-resolve namespace %)
          operations (cond-> [shutdown!
                              #(register! "reentry" {:source "fake"} :arena)
                              #((v 'close-kernel-arena!) :arena)
                              #((v 'ensure-kernel-loaded!) "reentry")
                              #((v (if (= backend :ocl) 'ensure-host-seg 'ensure-seg)) "reentry" :stage 1)]
                       (= backend :ze) (conj #((v 'ensure-arr) "reentry" :array 1)))]
      (with-runtime backend
        {:on-contact (fn [_ id]
                       (when (= id :context)
                         (swap! observed into (mapv error-of operations))))}
        (fn [{:keys [init! registry]}]
          (init!)
          (is (= (vec (repeat (count operations) :registration-in-use))
                 (mapv #(-> % ex-data :reason) @observed)))
          (is (empty? @registry)))))))

(deftest checked-native-release-failure-retains-parent-and-attempts-independent-queue
  (doseq [backend [:ocl :ze]
          failed-id (if (= backend :ocl) [:release-compute :release-transfer :release-context]
                        [:release-sync :release-context])]
    (with-runtime backend {:status failed-id}
      (fn [{:keys [init! shutdown! state calls]}]
        (let [primary (ex-info "final root publication rejected" {})]
          (add-watch state :reject-final
                     (fn [_ _ _ value] (when (:initialized? value) (throw primary))))
          (is (identical? primary (error-of init!)))
          (let [entry (::root/entry @state)
                before @calls
                releases (filterv #(contains? #{:release-compute :release-transfer
                                                :release-sync :release-context} %) before)]
            (is (= (cond
                     (= backend :ocl) (cond-> [:release-compute :release-transfer]
                                        (= failed-id :release-context) (conj :release-context))
                     :else (cond-> [:release-sync]
                             (= failed-id :release-context) (conj :release-context))) releases))
            (is (some? entry))
            (is (= :poisoned (:phase (cleanup/status (::cleanup/owner entry)))))
            (is (= 1 (alength (.getSuppressed ^Throwable primary))))
            (remove-watch state :reject-final)
            (is (identical? (first (.getSuppressed ^Throwable primary)) (error-of shutdown!)))
            (is (= before @calls))))))))

(deftest native-final-publication-watch-and-validator-rollback-permit-clean-retry
  (doseq [backend [:ocl :ze] kind [:watch :validator]]
    (with-runtime backend {}
      (fn [{:keys [init! state calls]}]
        (let [primary (ex-info "final publication rejected" {})]
          (case kind
            :watch (add-watch state :reject
                              (fn [_ _ _ value] (when (:initialized? value) (throw primary))))
            :validator (set-validator! state #(or (not (:initialized? %)) (throw primary))))
          (is (identical? primary (error-of init!)))
          (is (nil? (::root/entry @state)))
          (is (false? (:initialized? @state)))
          (is (= (if (= backend :ocl) [:release-compute :release-transfer :release-context]
                     [:release-sync :release-context])
                 (filterv #(contains? #{:release-compute :release-transfer :release-sync :release-context} %)
                          @calls)))
          (remove-watch state :reject)
          (set-validator! state nil)
          (init!)
          (is (some? (root/assert-live! state))))))))

(deftest native-generation-loss-retains-unknown-authority-and-preserves-unrelated-state
  (doseq [backend [:ocl :ze]]
    (with-runtime backend
      {:throw :context :on-contact (fn [state id]
                                     (when (= id :context) (reset! state {:unrelated :preserved})))}
      (fn [{:keys [init! shutdown! state calls primary]}]
        (is (identical? primary (error-of init!)))
        (is (= :preserved (:unrelated @state)))
        (is (= 1 (count (filter #(and (map? %) (::root/root? %)) (vals @state)))))
        (let [before @calls]
          (is (= :runtime-root-unavailable (:reason (ex-data (error-of init!)))))
          (is (identical? primary (error-of shutdown!)))
          (is (= before @calls)))))))

(deftest lazy-async-status-null-and-readback-failures-retain-the-exact-root-slot
  (doseq [options [{:status :async} {:null :async} {:readback-failure :async}]]
    (with-runtime :ze options
      (fn [{:keys [init! shutdown! state calls]}]
        (init!)
        (let [entry (root/assert-live! state)
              slot (get (:slots entry) :async-list)
              primary (error-of ze/async-cmd-list)
              before @calls]
          (is (some? primary))
          (is (= :indeterminate (:phase @slot)))
          (is (identical? entry (::root/entry @state)))
          (is (= :runtime-root-unavailable (:reason (ex-data (error-of ze/async-cmd-list)))))
          (is (= :runtime-root-leases-incomplete (:reason (ex-data (error-of shutdown!)))))
          (is (= before @calls)))))))

(deftest explicitly-safe-native-release-retries-only-the-pending-resource
  (doseq [backend [:ocl :ze]]
    (let [failed? (atom false)
          retry-error (ex-info "known retryable release" {:cleanup-retry-safe? true})]
      (with-runtime backend
        {:on-contact (fn [_ id]
                       (when (and (= id :release-context) (compare-and-set! failed? false true))
                         (throw retry-error)))}
        (fn [{:keys [init! shutdown! state calls]}]
          (let [primary (ex-info "final publication rejected" {})]
            (add-watch state :reject
                       (fn [_ _ _ value] (when (:initialized? value) (throw primary))))
            (is (identical? primary (error-of init!)))
            (is (identical? retry-error (first (.getSuppressed ^Throwable primary))))
            (remove-watch state :reject)
            (shutdown!)
            (is (nil? (::root/entry @state)))
            (is (false? (:initialized? @state)))
            (is (= 2 (count (filter #{:release-context} @calls))))
            (is (= 1 (count (filter #{(if (= backend :ocl) :release-compute :release-sync)} @calls))))))))))
