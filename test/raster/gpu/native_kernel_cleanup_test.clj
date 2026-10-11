(ns raster.gpu.native-kernel-cleanup-test
  "Hardware-free fault injection through the production KernelCall binders/destructors."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.storage-representation :as probe]
            [raster.compiler.ir.kernel-call :as call]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.ze-runtime :as ze])
  (:import [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.lang.invoke MethodHandles]))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))

(defn- with-ocl-map-staging-fault
  ([failure-point retry-safe? check]
   (with-ocl-map-staging-fault failure-point retry-safe? (fn [_ _] {}) check))
  ([failure-point retry-safe? customize check]
  (with-open [arena (Arena/ofShared)]
    (let [v #(ns-resolve 'raster.gpu.ocl-runtime %)
          primary (ex-info "injected map failure" {})
          release-failure (ex-info "injected temporary release failure"
                                   {:cleanup-retry-safe? retry-safe?})
          fails? #(if (set? failure-point) (contains? failure-point %) (= failure-point %))
          calls (atom []) creates (atom 0) release-attempts (atom 0)
          runtime-state (atom {:initialized? false})
          _ (root/initialize! runtime-state []
                              (fn [_] {:arena arena :context MemorySegment/NULL
                                       :queue MemorySegment/NULL}))
          info {:kernel-name "staging-test" :kernel-handle MemorySegment/NULL
                :dtype :float :workgroup-size 4 :arena-id :staging-test
                ::cleanup/owner (cleanup/owner [])}
          registry (atom {"staging-test" info})
          redefs (merge {(v 'state) runtime-state
                  (v 'kernel-registry) registry
                  (v 'h-clReleaseMemObject) (delay :fake)
                  (v 'h-clEnqueueNDRangeKernel) (delay :fake)
                  (v 'h-clFinish) (delay :fake)
                  (v 'h-clEnqueueReadBuffer) (delay :fake)
                  (v 'ensure-kernel-loaded!) (fn [_] info)
                  (v 'ensure-host-seg)
                  (fn [_ key ^long size]
                    (when (and (fails? :later-staging) (= key :void-arr-1))
                      (throw primary))
                    (.allocate arena (long size)))
                  (v 'create-map-buffer!)
                  (fn [& _]
                    (let [idx (swap! creates inc)]
                      (swap! calls conj [:create idx])
                      (when (and (fails? :later-acquisition) (= idx 2))
                        (throw primary))
                      (MemorySegment/ofAddress (+ 100 idx))))
                  (v 'set-kernel-arg-buffer!)
                  (fn [_ ^long _idx _value] (when (fails? :binding) (throw primary)))
                  (v 'set-kernel-arg-scalar!) (fn [_ ^long _idx _value] nil)
                  (v 'cl-call!)
                  (fn [label _ args]
                    (swap! calls conj [label (when (= label "clReleaseMemObject")
                                              (.address ^MemorySegment (first args)))])
                    (when (fails? label) (throw primary))
                    (when (= label "clReleaseMemObject")
                      (let [attempt (swap! release-attempts inc)]
                        (when (and (fails? :release) (= attempt 1))
                          (throw release-failure)))))}
                        (customize v arena))]
      (with-redefs-fn redefs
        #(let [result (error-of (fn [] (ocl/invoke-registered-map-void-kernel
                                       "staging-test" [(float-array 2) (float-array 2)] [] 2)))
               retained (vec (remove (fn [[key _]] (= key "staging-test")) @registry))]
           (check {:result result :primary primary :release-failure release-failure
                   :calls calls :retained retained :state runtime-state})))))))

(deftest ocl-positive-map-staging-rolls-back-every-execution-boundary
  (doseq [point [nil :later-staging :binding "clEnqueueNDRangeKernel"
                 "clFinish" "clEnqueueReadBuffer"]]
    (with-ocl-map-staging-fault
      point false
      (fn [{:keys [result primary calls retained state]}]
        (if point (is (identical? primary result)) (is (nil? result)))
        (is (= (if (= point :later-staging) [101] [101 102])
               (mapv second (filter #(= "clReleaseMemObject" (first %)) @calls))))
        (is (empty? retained))
        (is (zero? (root/lease-count state)))))))

(deftest ocl-positive-map-staging-retains-indeterminate-acquisition
  (with-ocl-map-staging-fault
    :later-acquisition false
    (fn [{:keys [result primary calls retained state]}]
      (is (identical? primary result))
      (is (= [101] (mapv second (filter #(= "clReleaseMemObject" (first %)) @calls))))
      (is (= 1 (count retained)))
      (is (= 1 (root/lease-count state)))
      (let [owner (::cleanup/owner (second (first retained))) before @calls]
        (is (identical? primary (error-of #(cleanup/release! owner))))
        (is (= before @calls))))))

(deftest ocl-positive-map-staging-retains-release-failure-disposition
  (doseq [retry-safe? [false true]]
    (with-ocl-map-staging-fault
      :release retry-safe?
      (fn [{:keys [result release-failure calls retained state]}]
        (is (identical? release-failure result))
        (is (= 1 (count retained)))
        (is (= [101 102] (mapv second (filter #(= "clReleaseMemObject" (first %)) @calls))))
        (is (= 1 (root/lease-count state)))
        (let [owner (::cleanup/owner (second (first retained))) before @calls
              retry (error-of #(cleanup/release! owner))]
          (if retry-safe?
            (do (is (nil? retry))
                (is (= [101 102 101]
                       (mapv second (filter #(= "clReleaseMemObject" (first %)) @calls))))
                (is (zero? (root/lease-count state))))
            (do (is (identical? release-failure retry))
                (is (= before @calls))
                (is (= 1 (root/lease-count state))))))))))

(deftest ocl-positive-map-staging-checks-native-allocation-result
  (with-open [arena (Arena/ofShared)]
    (let [v #(ns-resolve 'raster.gpu.ocl-runtime %)
          handle (MethodHandles/dropArguments
                  (MethodHandles/constant MemorySegment MemorySegment/NULL)
                  0 (into-array Class [MemorySegment Long/TYPE Long/TYPE MemorySegment MemorySegment]))]
      (doseq [error [-61 0]]
        (with-redefs-fn {(v 'h-clCreateBuffer) (delay handle)
                        (v 'read-int) (fn ^long [_] error)}
          #(let [failure (error-of (fn [] ((v 'create-map-buffer!) MemorySegment/NULL
                                           (.allocate arena 8) 8 (.allocate arena 4))))]
             (if (zero? error)
               (is (= :native-buffer-null (:reason (ex-data failure))))
               (is (= error (:error (ex-data failure)))))))))))

(deftest ocl-positive-map-staging-preserves-primary-with-failed-rollback
  (with-ocl-map-staging-fault
    #{:binding :release} true
    (fn [{:keys [result primary release-failure retained state]}]
      (is (identical? primary result))
      (is (some #(identical? release-failure %) (.getSuppressed ^Throwable result)))
      (is (= 1 (count retained)))
      (is (= 1 (root/lease-count state)))
      (is (nil? (error-of #(cleanup/release! (::cleanup/owner (second (first retained)))))))
      (is (zero? (root/lease-count state))))))

(deftest ocl-positive-map-staging-retains-uncertain-nonnull-allocation-error
  (with-ocl-map-staging-fault
    nil false
    (fn [v _]
      (let [create! @(v 'create-map-buffer!)
            handle (MethodHandles/dropArguments
                    (MethodHandles/constant MemorySegment (MemorySegment/ofAddress 901))
                    0 (into-array Class [MemorySegment Long/TYPE Long/TYPE MemorySegment MemorySegment]))]
        {(v 'create-map-buffer!) create!
         (v 'h-clCreateBuffer) (delay handle)
         (v 'read-int) (fn ^long [_] -61)}))
    (fn [{:keys [result calls retained state]}]
      (is (= -61 (:error (ex-data result))))
      (is (empty? (filter #(= "clReleaseMemObject" (first %)) @calls)))
      (is (= 1 (count retained)))
      (is (= 1 (root/lease-count state)))
      (let [before @calls]
        (is (identical? result
                        (error-of #(cleanup/release! (::cleanup/owner (second (first retained)))))))
        (is (= before @calls))))))

(deftest ocl-positive-map-staging-publication-failure-exposes-unresolved-owner
  (let [publication (ex-info "injected cleanup retention failure" {})]
    (with-ocl-map-staging-fault
      :release true
      (fn [_ _] {#'cleanup/retain-registration! (fn [& _] (throw publication))})
      (fn [{:keys [result release-failure retained state]}]
        (is (identical? release-failure (.getCause ^Throwable result)))
        (is (some #(identical? publication %) (.getSuppressed ^Throwable result)))
        (is (empty? retained))
        (is (= 1 (root/lease-count state)))
        (let [owner (::cleanup/unresolved (ex-data result))]
          (is (some? owner))
          (is (nil? (error-of #(cleanup/release! owner))))
          (is (zero? (root/lease-count state))))))))

(deftest both-binders-reject-registration-replacement-before-native-loading
  (doseq [[namespace bind!] [['raster.gpu.ze-runtime ze/bind-kernel-call]
                           ['raster.gpu.ocl-runtime ocl/bind-kernel-call]]]
    (let [v #(ns-resolve namespace %)
          artifact (probe/emit-artifact :float :opencl-portable)
          kernel-call (call/make artifact [(MemorySegment/ofArray (float-array 2))])
          registry (atom {(:kernel-name artifact) artifact})
          replacement (update artifact :source str "\n// another module generation")
          loads (atom 0)
          creates (atom 0)
          validate! call/validate-registered!]
      (with-redefs-fn
        {(v 'kernel-registry) registry
         #'call/validate-registered!
         (fn [call registered]
           (validate! call registered)
           ;; Deterministic interleaving: replacement after validation but before load.
           (swap! registry assoc (:kernel-name artifact) replacement))
         (v 'ensure-kernel-loaded!) (fn [& _] (swap! loads inc) {})
         (v 'create-kernel-fresh) (fn [& _]
                                  (swap! creates inc)
                                  (throw (ex-info "unexpected native acquisition" {})))}
        #(do
           (is (= :registry-generation-changed
                  (:reason (ex-data (error-of (fn [] (bind! kernel-call)))))))
           (is (zero? @loads))
           (is (zero? @creates))
           (is (identical? replacement (get @registry (:kernel-name artifact)))))))))

(deftest ze-recording-reserves-ownership-before-every-native-acquisition
  (doseq [failure-point [nil "zeCommandQueueCreate" "zeCommandListCreate"
                         "zeEventPoolCreate" "zeEventCreate"
                         "zeCommandListAppendLaunchKernel" "zeCommandListAppendBarrier"
                         "zeCommandListClose"]]
    (with-open [arena (Arena/ofShared)]
      (let [v #(ns-resolve 'raster.gpu.ze-runtime %)
            primary (ex-info "injected recording failure" {})
            calls (atom []) adopted (atom [])
            state (atom {:initialized? false})
            _ (root/initialize! state []
                                (fn [_] {:arena arena :context MemorySegment/NULL
                                         :device MemorySegment/NULL}))
            create? #{"zeCommandQueueCreate" "zeCommandListCreate" "zeEventPoolCreate" "zeEventCreate"}
            release? #{"zeCommandListDestroy" "zeEventDestroy" "zeEventPoolDestroy" "zeCommandQueueDestroy"}
            redefs (merge
                    {(v 'ensure-init!) (fn [] nil)
                     (v 'state) state
                     (v 'ze-call!) (fn [label _ args]
                                     (swap! calls conj label)
                                     (when (= failure-point label) (throw primary))
                                     (when (create? label)
                                       (.set ^MemorySegment (last args) ValueLayout/ADDRESS 0
                                             (MemorySegment/ofAddress (long (+ 100 (count @calls)))))))}
                    (into {} (map (fn [name] [(v name) (delay :fake)])
                                  '[h-zeCommandQueueCreate h-zeCommandListCreate h-zeEventPoolCreate
                                    h-zeEventCreate h-zeCommandListAppendLaunchKernel
                                    h-zeCommandListAppendBarrier h-zeCommandListClose
                                    h-zeCommandListDestroy h-zeEventDestroy h-zeEventPoolDestroy
                                    h-zeCommandQueueDestroy])))]
        (with-redefs-fn redefs
          (fn []
            (let [result (try
                           (ze/record-graph! [{:bound {:kernel MemorySegment/NULL
                                                       :gc-seg (.allocate arena 12)}}]
                                             {:profile? true :adopt-cleanup! #(swap! adopted conj %)})
                           (catch Throwable error error))]
              (if failure-point
                (do
                  (is (identical? primary result))
                  (is (= (if (create? failure-point) 1 0) (count @adopted)))
                  (is (= (if (create? failure-point) 1 0) (root/lease-count state)))
                  (when (not= failure-point "zeCommandQueueCreate")
                    (is (some #{"zeCommandQueueDestroy"} @calls)))
                  (doseq [owner @adopted]
                    (let [before @calls]
                      (is (identical? primary (error-of #(cleanup/release! owner))))
                      (is (= before @calls)))))
                (do
                  (is (some? (::cleanup/owner result)))
                  (is (= 1 (root/lease-count state)))
                  (ze/destroy-graph! result)
                  (is (zero? (root/lease-count state)))
                  (let [before @calls]
                    (ze/destroy-graph! result)
                    (is (= before @calls))
                    (doseq [operation [ze/submit-graph! ze/reset-graph-events! ze/read-graph-timestamps!]]
                      (is (= :owner-releasing (:reason (ex-data (error-of #(operation result)))))))
                    (is (= before @calls)))
                  (is (= ["zeCommandListDestroy" "zeEventDestroy" "zeEventPoolDestroy"
                          "zeCommandQueueDestroy"] (filterv release? @calls)))
                  (is (= :missing-cleanup-owner
                         (:reason (ex-data (error-of #(ze/destroy-graph!
                                                       (dissoc result ::cleanup/owner))))))))))))))))

(deftest both-production-binders-own-success-and-failed-native-acquisition
  (doseq [backend [:ze :ocl] fail-build? [false true] fail-release? [false true]]
    (let [namespace (if (= backend :ze) 'raster.gpu.ze-runtime 'raster.gpu.ocl-runtime)
          v #(ns-resolve namespace %)
          artifact (probe/emit-artifact :float :opencl-portable)
          handle (MemorySegment/ofArray (byte-array 8))
          output (MemorySegment/ofArray (float-array 2))
          kernel-call (call/make artifact [output])
          bind! (if (= backend :ze) ze/bind-kernel-call ocl/bind-kernel-call)
          destroy! (if (= backend :ze) ze/destroy-prepared! ocl/destroy-prepared!)
          primary (ex-info "argument binding failed" {})
          native-fault (ex-info "destruction outcome unknown" {})
          releases (atom []) adopted (atom []) acquired (atom 0)
          release-call (fn [context _ args]
                         (swap! releases conj [context (first args)])
                         (when fail-release? (throw native-fault)))
          state (atom {:initialized? false})
          _ (root/initialize! state [] (fn [_] {}))
          redefs (merge
                  {(v 'state) state
                   (v 'kernel-registry) (atom {(:kernel-name artifact) artifact})
                   (v 'ensure-kernel-loaded!) (fn [_] {:module handle :program handle :entry-name "probe"})
                   (v 'create-kernel-fresh) (fn [& _] (swap! acquired inc)
                                              (if (= backend :ze) {:handle handle} handle))}
                  (if (= backend :ze)
                    {(v 'h-zeKernelDestroy) (delay :fake)
                     (v 'ze-call!) release-call
                     (v 'destroy-kernel!) #(release-call "zeKernelDestroy" nil [(:handle %)])
                     (v 'bind-kernel!) (fn [& _]
                                         (when fail-build? (throw primary))
                                         {:kernel handle :gc-seg (MemorySegment/ofArray (int-array 3))})}
                    {(v 'h-clReleaseKernel) (delay :fake)
                     (v 'cl-call!) release-call
                     (v 'set-kernel-arg-buffer!) (fn [_handle ^long _index _buffer]
                                                   (when fail-build? (throw primary)))}))]
      (with-redefs-fn redefs
        (fn []
          (if fail-build?
            (do
              (is (identical? primary (error-of #(bind! kernel-call {:adopt-cleanup! (fn [owner] (swap! adopted conj owner))}))))
              (is (= (if fail-release? 1 0) (count @adopted)))
              (when fail-release?
                (is (identical? native-fault (error-of #(cleanup/release! (first @adopted)))))))
            (let [prepared (bind! kernel-call {:adopt-cleanup! (fn [owner] (swap! adopted conj owner))})]
              (is (some? (::cleanup/owner prepared)))
              (is (empty? @releases))
              (dotimes [_ 2]
                (is (if fail-release?
                      (identical? native-fault (error-of #(destroy! prepared)))
                      (nil? (destroy! prepared)))))
              (is (= :missing-cleanup-owner
                     (:reason (ex-data (error-of #(destroy! (dissoc prepared ::cleanup/owner)))))))))
          (is (= 1 @acquired))
          (when (= backend :ocl)
            (is (= (if fail-release? 1 0) (root/lease-count state))))
          (is (= [[(if (= backend :ze) "zeKernelDestroy" "clReleaseKernel") handle]] @releases)))))))

(deftest opencl-unknown-prepared-create-retains-the-composite-owner-and-root
  (let [v #(ns-resolve 'raster.gpu.ocl-runtime %)
        artifact (probe/emit-artifact :float :opencl-portable)
        kernel-call (call/make artifact [(MemorySegment/ofArray (float-array 2))])
        state (atom {:initialized? false})
        primary (ex-info "fresh kernel acquisition outcome unknown" {})
        acquired (atom 0) released (atom 0) adopted (atom [])]
    (root/initialize! state [] (fn [_] {}))
    (with-redefs-fn
      {(v 'state) state
       (v 'kernel-registry) (atom {(:kernel-name artifact) artifact})
       (v 'ensure-kernel-loaded!) (fn [_] {:program :program})
       (v 'create-kernel-fresh) (fn [& _] (swap! acquired inc) (throw primary))
       (v 'h-clReleaseKernel) (delay :fake)
       (v 'cl-call!) (fn [& _] (swap! released inc))}
      (fn []
        (is (identical? primary
                        (error-of #(ocl/bind-kernel-call kernel-call
                                                         {:adopt-cleanup! (fn [owner]
                                                                            (swap! adopted conj owner))}))))
        (is (= 1 @acquired))
        (is (= 1 (count @adopted)))
        (is (= 1 (root/lease-count state)))
        (let [owner (first @adopted)]
          (is (= [:kernel :runtime-root-lease] (cleanup/pending owner)))
          (is (identical? primary (error-of #(cleanup/release! owner))))
          (is (= 1 (root/lease-count state)))
          (is (zero? @released)))))))

(deftest ze-recording-keeps-dependent-pool-and-attempts-independent-destruction
  (doseq [failed-id [:list [:event 0] :pool :queue]]
    ;; recording-plan captures the handle delays while reserving callbacks: replace them
    ;; before reservation, not just before release. Otherwise laptop-loaded delays hide a
    ;; dependency on the native loader that fails on hardware-free CI.
    (with-redefs-fn
      (into {} (map (fn [name] [(ns-resolve 'raster.gpu.ze-runtime name) (delay :fake)])
                    '[h-zeCommandListDestroy h-zeEventDestroy h-zeEventPoolDestroy h-zeCommandQueueDestroy]))
      (fn []
        (let [{:keys [slots resources]} ((ns-resolve 'raster.gpu.ze-runtime 'recording-plan) 2)
              owner (cleanup/owner resources)
              calls (atom []) failure (ex-info "destroy outcome unknown" {})
              label->id {"zeCommandListDestroy" :list "zeEventPoolDestroy" :pool
                         "zeCommandQueueDestroy" :queue}
              resources {:list 1 [:event 0] 2 [:event 1] 3 :pool 4 :queue 5}
              reverse-resources (into {} (map (fn [[k v]] [v k]) resources))]
          (doseq [[id address] resources]
            (cleanup/acquire-native! (get slots id) #(MemorySegment/ofAddress (long address))))
          (with-redefs-fn
            {(ns-resolve 'raster.gpu.ze-runtime 'h-zeCommandListDestroy) (delay :fake)
             (ns-resolve 'raster.gpu.ze-runtime 'h-zeEventDestroy) (delay :fake)
             (ns-resolve 'raster.gpu.ze-runtime 'h-zeEventPoolDestroy) (delay :fake)
             (ns-resolve 'raster.gpu.ze-runtime 'h-zeCommandQueueDestroy) (delay :fake)
             (ns-resolve 'raster.gpu.ze-runtime 'ze-call!)
             (fn [label _ args]
               (let [id (if (= "zeEventDestroy" label)
                          (reverse-resources (.address ^MemorySegment (first args)))
                          (label->id label))]
                 (swap! calls conj id)
                 (when (= failed-id id) (throw failure))))}
            (fn []
              (is (identical? failure (error-of #(ze/destroy-graph! {::cleanup/owner owner}))))
              (is (= (case failed-id
                       :list [:list :queue]
                       [:event 0] [:list [:event 0] [:event 1] :queue]
                       [:list [:event 0] [:event 1] :pool :queue]) @calls))
              (is (= (case failed-id
                       :list [:list [:event 0] [:event 1] :pool]
                       [:event 0] [[:event 0] :pool]
                       :pool [:pool]
                       :queue [:queue]) (cleanup/pending owner)))
              (let [before @calls]
                (is (identical? failure (error-of #(ze/destroy-graph! {::cleanup/owner owner}))))
                (is (= before @calls))))))))))
