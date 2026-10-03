(ns raster.gpu.native-kernel-cleanup-test
  "Hardware-free fault injection through the production KernelCall binders/destructors."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.storage-representation :as probe]
            [raster.compiler.ir.kernel-call :as call]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.ze-runtime :as ze])
  (:import [java.lang.foreign Arena MemorySegment ValueLayout]))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))

(deftest ze-recording-reserves-ownership-before-every-native-acquisition
  (doseq [failure-point [nil "zeCommandQueueCreate" "zeCommandListCreate"
                        "zeEventPoolCreate" "zeEventCreate"
                        "zeCommandListAppendLaunchKernel" "zeCommandListAppendBarrier"
                        "zeCommandListClose"]]
    (with-open [arena (Arena/ofShared)]
      (let [v #(ns-resolve 'raster.gpu.ze-runtime %)
            primary (ex-info "injected recording failure" {})
            calls (atom []) adopted (atom [])
            create? #{"zeCommandQueueCreate" "zeCommandListCreate" "zeEventPoolCreate" "zeEventCreate"}
            release? #{"zeCommandListDestroy" "zeEventDestroy" "zeEventPoolDestroy" "zeCommandQueueDestroy"}
            redefs (merge
                     {(v 'ensure-init!) (fn [] nil)
                      (v 'state) (atom {:arena arena :context MemorySegment/NULL :device MemorySegment/NULL})
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
                  (when (not= failure-point "zeCommandQueueCreate")
                    (is (some #{"zeCommandQueueDestroy"} @calls)))
                  (doseq [owner @adopted]
                    (let [before @calls]
                      (is (identical? primary (error-of #(cleanup/release! owner))))
                      (is (= before @calls)))))
                (do
                  (is (some? (::cleanup/owner result)))
                  (ze/destroy-graph! result)
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
          redefs (merge
                   {(v 'kernel-registry) (atom {(:kernel-name artifact) artifact})
                    (v 'ensure-kernel-loaded!) (fn [_] {:module handle :program handle :entry-name "probe"})
                    (v 'create-kernel-fresh) (fn [& _] (swap! acquired inc) handle)}
                   (if (= backend :ze)
                     {(v 'h-zeKernelDestroy) (delay :fake)
                      (v 'ze-call!) release-call
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
          (is (= [[(if (= backend :ze) "zeKernelDestroy" "clReleaseKernel") handle]] @releases)))))))

(deftest ze-recording-keeps-dependent-pool-and-attempts-independent-destruction
  (doseq [failed-id [:list [:event 0] :pool :queue]]
    ;; recording-owner captures the handle delays while reserving callbacks: replace them
    ;; before reservation, not just before release. Otherwise laptop-loaded delays hide a
    ;; dependency on the native loader that fails on hardware-free CI.
    (with-redefs-fn
      (into {} (map (fn [name] [(ns-resolve 'raster.gpu.ze-runtime name) (delay :fake)])
                    '[h-zeCommandListDestroy h-zeEventDestroy h-zeEventPoolDestroy h-zeCommandQueueDestroy]))
      (fn []
    (let [{:keys [slots owner]} ((ns-resolve 'raster.gpu.ze-runtime 'recording-owner) 2)
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
