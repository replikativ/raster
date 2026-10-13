(ns raster.gpu.cuda-driver-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.cuda-driver :as driver]
            [raster.gpu.resource-cleanup :as cleanup])
  (:import [java.lang.foreign MemorySegment ValueLayout]))

(defn- with-raw [f]
  (let [calls (atom []) fault (atom {}) context (MemorySegment/ofAddress 11)
        raw (fn [symbol _ args]
              (swap! calls conj symbol)
              (when-let [error (get @fault symbol)]
                (if (fn? error) (error args) (throw error)))
              (case symbol
                "cuDevicePrimaryCtxRetain" (.set (first args) ValueLayout/ADDRESS (long 0) context)
                "cuStreamCreate" (.set (first args) ValueLayout/ADDRESS (long 0) (MemorySegment/ofAddress 12))
                "cuDeviceGetAttribute" (.set (first args) ValueLayout/JAVA_INT (long 0)
                                            (int (get {1 1024 8 49152 10 32 16 40} (second args) 1)))
                "cuCtxPopCurrent_v2" (.set (first args) ValueLayout/ADDRESS (long 0) context)
                "cuMemAlloc_v2" (.set (first args) ValueLayout/JAVA_LONG (long 0) (long 1024))
                "cuMemHostAlloc" (.set (first args) ValueLayout/ADDRESS (long 0) (MemorySegment/ofAddress 2048))
                "cuModuleLoadData" (.set (first args) ValueLayout/ADDRESS (long 0) (MemorySegment/ofAddress 3072))
                "cuEventCreate" (.set (first args) ValueLayout/ADDRESS (long 0) (MemorySegment/ofAddress 4096))
                nil))]
    (binding [driver/*invoke* nil driver/*context* context]
      (with-redefs-fn {(ns-resolve 'raster.gpu.cuda-driver 'raw!) raw}
        #(f calls fault context)))))

(deftest each-thread-operation-restores-the-exact-previous-context
  (with-raw
    (fn [calls _ _]
      (is (= 1024 (driver/invoke! :allocate [4])))
      (is (= ["cuCtxPushCurrent_v2" "cuMemAlloc_v2" "cuCtxPopCurrent_v2"] @calls)))))

(deftest resource-acquisitions-retire-unpublished-results-if-context-restoration-fails
  (doseq [[operation args destructor] [[:allocate [4] "cuMemFree_v2"]
                                      [:host-allocate [4] "cuMemFreeHost"]
                                      [:module-load [(byte-array [0])] "cuModuleUnload"]
                                      [:event-create [] "cuEventDestroy_v2"]]]
    (with-raw
      (fn [calls fault _]
        (let [primary (ex-info "context pop failed" {})]
          (reset! fault {"cuCtxPopCurrent_v2" primary})
          ;; The rollback's own pop also fails in this fixture. Native destruction
          ;; is not retried: the common owner retains its indeterminate disposition.
          (let [error (try (driver/invoke! operation args) nil (catch Throwable e e))]
            (is (identical? primary (.getCause ^Throwable error)) (str operation))
            (is (some? (::cleanup/unresolved (ex-data error))) (str operation))
            (is (= 1 (count (filter #{destructor} @calls))) (str operation))))))))

(deftest failed-acquisition-rollback-retains-common-ownership-without-retrying-native-free
  (with-raw
    (fn [calls fault _]
      (let [primary (ex-info "context pop failed" {})
            secondary (ex-info "native free failed" {})]
        (reset! fault {"cuCtxPopCurrent_v2" primary "cuMemFree_v2" secondary})
        (let [error (try (driver/invoke! :allocate [4]) nil (catch Throwable e e))
              owner (::cleanup/unresolved (ex-data error))]
          (is (identical? primary (.getCause ^Throwable error)))
          (is (= [:cuda-acquisition] (cleanup/pending owner)))
          (is (= :indeterminate (-> (cleanup/status owner) :remaining first :disposition)))
          (is (identical? secondary
                          (try (cleanup/release! owner) nil (catch Throwable e e))))
          (is (= 1 (count (filter #{"cuMemFree_v2"} @calls)))
              "an indeterminate free must not be issued a second time"))))))

(deftest same-throwable-body-and-pop-failures-do-not-self-suppress
  (with-raw
    (fn [_ fault _]
      (let [primary (ex-info "reused mock failure" {})]
        (reset! fault {"cuMemAlloc_v2" primary "cuCtxPopCurrent_v2" primary})
        (is (identical? primary
                        (try (driver/invoke! :allocate [4]) nil (catch Throwable e e))))
        (is (empty? (.getSuppressed primary)))))))

(deftest unexpected-popped-context-refuses-publication-and-retires-the-result
  (with-raw
    (fn [calls _ _]
      (let [original @(ns-resolve 'raster.gpu.cuda-driver 'raw!)
            replace-once? (atom true)]
        (with-redefs-fn
          {(ns-resolve 'raster.gpu.cuda-driver 'raw!)
           (fn [symbol layouts args]
             (original symbol layouts args)
             (when (and (= "cuCtxPopCurrent_v2" symbol)
                        (compare-and-set! replace-once? true false))
               (.set (first args) ValueLayout/ADDRESS (long 0) (MemorySegment/ofAddress 99))))}
          (fn []
            (let [error (try (driver/invoke! :allocate [4]) nil (catch Throwable e e))]
              (is (= :cuda-context-stack-mismatch (:reason (ex-data error))))
              (is (= 1 (count (filter #{"cuMemFree_v2"} @calls)))))))))))

(deftest primary-native-failure-preserves-context-pop-failure-as-suppressed
  (with-raw
    (fn [calls fault _]
      (let [primary (ex-info "allocation failed" {}) secondary (ex-info "context pop failed" {})]
        (reset! fault {"cuMemAlloc_v2" primary "cuCtxPopCurrent_v2" secondary})
        (is (identical? primary (try (driver/invoke! :allocate [4]) nil (catch Throwable e e))))
        (is (= [secondary] (vec (.getSuppressed primary))))
        (is (= "cuCtxPopCurrent_v2" (last @calls)))))))

(deftest primary-context-release-occurs-after-leaving-its-thread-context
  (with-raw
    (fn [calls _ context]
      (driver/invoke! :destroy-runtime [{:context context :stream (MemorySegment/ofAddress 12) :device 0}])
      (is (= ["cuCtxPushCurrent_v2" "cuStreamDestroy_v2" "cuCtxPopCurrent_v2" "cuDevicePrimaryCtxRelease_v2"] @calls)))))

(deftest initialization-records-queried-device-limits-and-api-version
  (with-raw
    (fn [_ _ _]
      (let [info (driver/invoke! :initialize [0])]
        (is (= {:max-workgroup-size 1024 :shared-local-memory 49152 :warp-size 32
                :multiprocessor-count 40}
               (select-keys info [:max-workgroup-size :shared-local-memory :warp-size :multiprocessor-count])))
        (is (contains? info :driver-api-version))))))

(deftest failed-initialization-context-pop-retires-the-acquired-stream
  (with-raw
    (fn [calls fault _]
      (let [primary (ex-info "initial context pop failed" {})]
        (reset! fault {"cuCtxPopCurrent_v2" (fn [_]
                                            (swap! fault dissoc "cuCtxPopCurrent_v2")
                                            (throw primary))})
        (is (identical? primary
                        (try (driver/invoke! :initialize [0]) nil (catch Throwable e e))))
        (is (= 1 (count (filter #{"cuStreamDestroy_v2"} @calls)))
            "the stream was acquired even though publication failed")
        (is (= "cuDevicePrimaryCtxRelease_v2" (last @calls)))))))

(deftest indeterminate-stream-acquisition-retains-context-without-guessing-or-retrying
  (with-raw
    (fn [calls fault _]
      (let [primary (ex-info "stream creation failed" {})]
        (reset! fault {"cuStreamCreate" primary})
        (let [error (try (driver/invoke! :initialize [0]) nil (catch Throwable e e))
              owner (::cleanup/unresolved (ex-data error))]
          (is (identical? primary (.getCause ^Throwable error)))
          (is (= [:stream :context] (cleanup/pending owner)))
          (is (not-any? #{"cuStreamDestroy_v2" "cuDevicePrimaryCtxRelease_v2"} @calls))
          (is (identical? primary
                          (try (cleanup/release! owner) nil (catch Throwable e e))))
          (is (= 1 (count (filter #{"cuStreamCreate"} @calls)))))))))

(deftest initialization-rollback-failures-preserve-the-original-query-error
  (with-raw
    (fn [calls fault _]
      (let [primary (ex-info "device query failed" {})
            stream-error (ex-info "stream rollback failed" {})]
        (reset! fault {"cuDeviceGetAttribute" primary
                       "cuStreamDestroy_v2" stream-error})
        (let [error (try (driver/invoke! :initialize [0]) nil (catch Throwable e e))
              owner (::cleanup/unresolved (ex-data error))]
          (is (identical? primary (.getCause ^Throwable error)))
          (is (= [stream-error] (vec (.getSuppressed primary))))
          (is (= [:stream :context] (cleanup/pending owner)))
          (is (not-any? #{"cuDevicePrimaryCtxRelease_v2"} @calls)
              "context retirement depends on successful stream retirement")
          (is (= 1 (count (filter #{"cuStreamDestroy_v2"} @calls)))))))))

(deftest initialization-query-failure-retires-stream-before-context
  (with-raw
    (fn [calls fault _]
      (let [primary (ex-info "query failed" {})]
        (reset! fault {"cuDeviceGetAttribute" primary})
        (is (identical? primary
                        (try (driver/invoke! :initialize [0]) nil (catch Throwable e e))))
        (is (= ["cuCtxPushCurrent_v2" "cuStreamDestroy_v2" "cuCtxPopCurrent_v2"
                "cuDevicePrimaryCtxRelease_v2"] (take-last 4 @calls)))))))
