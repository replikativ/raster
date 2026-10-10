(ns raster.gpu.ze-direct-generation-test
  "Exact registration admission through the public direct-call route, without a driver."
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.ze-runtime :as ze]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.backend.gpu.storage-representation :as probe])
  (:import [java.lang.foreign MemorySegment]))

(defn- fixture []
  (let [artifact (assoc-in (probe/emit-artifact :float :opencl-portable)
                          [:attributes :out-elems] 2)]
    {:artifact artifact :name (:kernel-name artifact)
     :registry (atom {(:kernel-name artifact) artifact})}))

(defn- error-of [f]
  (try (f) nil (catch Throwable error error)))

(deftest replacement-after-validation-is-rejected-before-any-native-use
  (let [{:keys [artifact name registry]} (fixture)
        replacement (update artifact :source str "\n// replacement generation")
        plan call/binding-plan
        seen (atom [])
        v #(ns-resolve 'raster.gpu.ze-runtime %)]
    (with-redefs-fn
      {#'ze/kernel-registry registry
       #'call/binding-plan (fn [c]
                            (let [result (plan c)]
                              (swap! registry assoc name replacement)
                              result))
       (v 'ensure-seg) (fn [_ _ ^long _]
                        (swap! seen conj :staging)
                        (MemorySegment/ofArray (float-array 2)))
       (v 'ensure-kernel-loaded!) (fn [_]
                                   (swap! seen conj :load)
                                   {:kernel-handle :replacement})
       #'ze/launch-geometry! (fn [& _] (swap! seen conj :launch))}
      #(do
         (is (= :registry-generation-changed
                (:reason (ex-data
                          (error-of (fn [] (ze/invoke-registered-contraction!
                                           name [(float-array 2)])))))))
         (is (empty? @seen))
         (is (identical? replacement (get @registry name)))))))

(deftest unchanged-registration-preserves-geometry-staging-and-readback
  (let [{:keys [name registry]} (fixture)
        seen (atom [])
        staged (float-array [1.25 -3.5])
        output (float-array 2)
        v #(ns-resolve 'raster.gpu.ze-runtime %)]
    (with-redefs-fn
      {#'ze/kernel-registry registry
       (v 'ensure-seg) (fn [kernel key ^long n-bytes]
                        (is (Thread/holdsLock registry))
                        (swap! seen conj [:stage kernel key n-bytes])
                        (MemorySegment/ofArray staged))
       (v 'ensure-kernel-loaded!) (fn [kernel]
                                   (is (Thread/holdsLock registry))
                                   (swap! seen conj [:load kernel])
                                   {:kernel-handle :original})
       #'ze/launch-geometry! (fn [handle workgroup groups arguments]
                              (is (Thread/holdsLock registry))
                              (is (= :registration-in-use
                                     (:reason (ex-data
                                               (error-of #(cleanup/assert-registry-mutable!
                                                           registry))))))
                              (swap! seen conj [:launch handle workgroup groups
                                                (count arguments)]))}
      #(do
         (is (identical? output (ze/invoke-registered-contraction! name [output])))
         (is (= [1.25 -3.5] (vec output)))
         (is (= [[:stage name :c-out 8] [:load name]
                 [:launch :original [1] [1] 1]] @seen))))))

(deftest failed-validation-still-precedes-staging-and-native-loading
  (let [{:keys [name registry]} (fixture)
        seen (atom [])
        v #(ns-resolve 'raster.gpu.ze-runtime %)]
    (with-redefs-fn
      {#'ze/kernel-registry registry
       (v 'ensure-seg) (fn [_ _ ^long _] (swap! seen conj :stage))
       (v 'ensure-kernel-loaded!) (fn [_] (swap! seen conj :load))}
      #(let [error (error-of (fn [] (ze/invoke-registered-contraction! name [])))]
         (is (= 1 (:expected (ex-data error))))
         (is (zero? (:actual (ex-data error))))
         (is (empty? @seen))))))

(deftest launch-failure-keeps-identity-and-releases-use-guard
  (let [{:keys [name registry]} (fixture)
        primary (ex-info "injected launch failure" {})
        v #(ns-resolve 'raster.gpu.ze-runtime %)]
    (with-redefs-fn
      {#'ze/kernel-registry registry
       (v 'ensure-seg) (fn [_ _ ^long _] (MemorySegment/ofArray (float-array 2)))
       (v 'ensure-kernel-loaded!) (fn [_] {:kernel-handle :original})
       #'ze/launch-geometry! (fn [& _] (throw primary))}
      #(do
         (is (identical? primary
                         (error-of (fn [] (ze/invoke-registered-contraction!
                                          name [(float-array 2)])))))
         (is (nil? (cleanup/assert-registry-mutable! registry)))))))

(deftest readback-failure-keeps-identity-and-releases-use-guard
  (let [{:keys [name registry]} (fixture)
        primary (ex-info "injected readback failure" {})
        v #(ns-resolve 'raster.gpu.ze-runtime %)]
    (with-redefs-fn
      {#'ze/kernel-registry registry
       (v 'ensure-seg) (fn [_ _ ^long _] (MemorySegment/ofArray (float-array 2)))
       (v 'ensure-kernel-loaded!) (fn [_] {:kernel-handle :original})
       #'ze/launch-geometry! (fn [& _] nil)
       (v 'readback-operand!) (fn [& _]
                               (is (= :registration-in-use
                                      (:reason (ex-data
                                                (error-of #(cleanup/assert-registry-mutable!
                                                            registry))))))
                               (throw primary))}
      #(do
         (is (identical? primary
                         (error-of (fn [] (ze/invoke-registered-contraction!
                                          name [(float-array 2)])))))
         (is (nil? (cleanup/assert-registry-mutable! registry)))))))
