(ns raster.gpu.ze-direct-generation-test
  "Exact registration admission through the public direct-call route, without a driver."
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.ze-runtime :as ze]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.backend.gpu.storage-representation :as probe])
  (:import [java.lang.foreign MemorySegment]))

(defn- fixture []
  (let [artifact (assoc-in (probe/emit-artifact :float :opencl-portable)
                          [:attributes :out-elems] 2)]
    {:artifact artifact :name (:kernel-name artifact)
     :registry (atom {(:kernel-name artifact) artifact})}))

(defn- error-of [f]
  (try (f) nil (catch Throwable error error)))

(defn- map-fixture []
  (let [name "generation-map"
        registration {:abi [(abi/slot 'x :inout :float)
                            (abi/slot 'out :output :float :role :result)
                            (abi/slot 'scale :scalar :float)
                            (abi/slot 'n :scalar :long :role :bound)]
                      :workgroup-size 4}]
    {:name name :registration registration :registry (atom {name registration})}))

(defn- reduction-fixture [dtype groups]
  (let [name (str "generation_reduction_" (clojure.core/name dtype))
        ctype (clojure.core/name dtype)
        registration (artifact/make
                      {:kernel-name name :target :opencl-c
                       :source (str "__kernel void " name "(__global const " ctype
                                    " *x, __global " ctype " *out, " ctype " scale, long n) {}")
                       :abi [(abi/slot 'x :input dtype)
                             (abi/slot 'out :output dtype :role :result)
                             (abi/slot 'scale :scalar dtype)
                             (abi/slot 'n :scalar :long :role :bound)]
                       :arguments '[x out scale n]
                       :launch (launch/spec {:workgroup-size [4] :group-count [groups]})
                       :attributes {:identity-val 0.0 :c-op "+"}})]
    {:name name :registration registration :registry (atom {name registration})}))

(deftest reduction-replacement-after-geometry-precedes-native-use
  (let [{:keys [name registration registry]} (reduction-fixture :float 3)
        replacement (assoc-in registration [:attributes :c-op] "*")
        realize call/realize-launch
        seen (atom [])
        v #(ns-resolve 'raster.gpu.ze-runtime %)]
    (with-redefs-fn
      {#'ze/kernel-registry registry
       #'call/realize-launch (fn [a arguments]
                              (let [geometry (realize a arguments)]
                                (swap! registry assoc name replacement)
                                geometry))
       (v 'ensure-kernel-loaded!) (fn [& _] (swap! seen conj :load))
       (v 'ensure-seg) (fn [_ _ ^long _] (swap! seen conj :stage))}
      #(do
         (is (= :registry-generation-changed
                (:reason (ex-data (error-of (fn [] (ze/invoke-registered-reduction-kernel
                                                   name [(float-array 12) nil {:type :float :value (float 2.0)}
                                                         {:type :long :value 12}])))))))
         (is (empty? @seen))
         (is (identical? replacement (get @registry name)))))))

(deftest reduction-partial-count-and-independent-rounded-combine-survive-admission
  (doseq [[dtype groups partials expected]
          [[:float 1 [3.25] 3.25]
           [:float 3 [16777216.0 1.0 -16777216.0] 0.0]
           [:double 3 [16777216.0 1.0 -16777216.0] 1.0]]]
    (let [{:keys [name registry]} (reduction-fixture dtype groups)
          array-fn (if (= :float dtype) float-array double-array)
          output (array-fn partials)
          input (array-fn 12)
          element-bytes (if (= :float dtype) 4 8)
          seen (atom [])
          v #(ns-resolve 'raster.gpu.ze-runtime %)]
      (with-redefs-fn
        {#'ze/kernel-registry registry
         (v 'ensure-kernel-loaded!) (fn [_]
                                     (is (Thread/holdsLock registry))
                                     (swap! registry update name assoc :kernel-handle :original)
                                     {:kernel-handle :original})
         (v 'ensure-seg) (fn [_ key ^long bytes]
                          (is (Thread/holdsLock registry))
                          (is (= (* element-bytes (if (= :partial-seg key) groups 12)) bytes))
                          (MemorySegment/ofArray (if (= :partial-seg key) output input)))
         #'ze/launch! (fn [handle ^long count ^long workgroup arguments]
                        (is (= :registration-in-use
                               (:reason (ex-data (error-of (fn [] (cleanup/assert-registry-mutable! registry)))))))
                        (swap! seen conj [handle count workgroup (drop 2 arguments)]))}
        #(do
           (is (= expected (ze/invoke-registered-reduction-kernel
                            name [(array-fn 12) nil {:type dtype :value (if (= :float dtype) (float 2.0) 2.0)}
                                  {:type :long :value 12}])))
           (is (= [[:original groups 4 [{:type dtype :value (if (= :float dtype) (float 2.0) 2.0)}
                                       {:type :long :value 12}]]] @seen))
           (is (nil? (cleanup/assert-registry-mutable! registry))))))))

(deftest reduction-failures-preserve-identity-and-release-native-use
  (doseq [phase [:launch :combine]]
    (let [{:keys [name registry]} (reduction-fixture :float 3)
          failure (ex-info "reduction injected failure" {:phase phase})
          seen (atom [])
          v #(ns-resolve 'raster.gpu.ze-runtime %)
          fail! (fn []
                  (is (= :registration-in-use
                         (:reason (ex-data (error-of (fn [] (cleanup/assert-registry-mutable! registry)))))))
                  (throw failure))]
      (with-redefs-fn
        {#'ze/kernel-registry registry
         (v 'ensure-kernel-loaded!) (fn [_] {:kernel-handle :original})
         (v 'ensure-seg) (fn [_ _ ^long _] (MemorySegment/ofArray (float-array 12)))
         #'ze/launch! (fn [_ ^long _count ^long _workgroup _args]
                        (swap! seen conj :launch)
                        (when (= phase :launch) (fail!)))
         (v 'combine-scalar-partials) (fn [& _] (swap! seen conj :combine) (fail!))}
        #(do
           (is (identical? failure (error-of (fn [] (ze/invoke-registered-reduction-kernel
                                                   name [(float-array 12) nil {:type :float :value (float 2.0)}
                                                         {:type :long :value 12}])))))
           (is (= (if (= phase :launch) [:launch] [:launch :combine]) @seen))
           (is (nil? (cleanup/assert-registry-mutable! registry))))))))

(deftest map-replacement-after-pure-admission-precedes-native-use
  (let [{:keys [name registration registry]} (map-fixture)
        replacement (assoc registration :workgroup-size 8)
        seen (atom [])
        v #(ns-resolve 'raster.gpu.ze-runtime %)]
    (with-redefs-fn
      {#'ze/kernel-registry registry
       (v 'registered-1d-workgroup-size) (fn [_] (swap! registry assoc name replacement) 4)
       (v 'ensure-kernel-loaded!) (fn [& _] (swap! seen conj :load))
       (v 'ensure-seg) (fn [_ _ ^long _] (swap! seen conj :stage))
       #'ze/launch! (fn [_ ^long _groups ^long _workgroup _args] (swap! seen conj :launch))}
      #(do
         (is (= :registry-generation-changed
                (:reason (ex-data (error-of (fn [] (ze/invoke-registered-kernel
                                                   name [(float-array 2)] (float-array 2) [2.0] 2)))))))
         (is (empty? @seen))
         (is (identical? replacement (get @registry name)))))))

(deftest map-snapshot-preserves-ordered-scalars-and-all-writable-readback
  (let [{:keys [name registry]} (map-fixture)
        x (float-array [1.25 -3.5]) out (float-array 2)
        staged-x (float-array 2) staged-out (float-array [7.0 8.0])
        seen (atom [])
        v #(ns-resolve 'raster.gpu.ze-runtime %)]
    (with-redefs-fn
      {#'ze/kernel-registry registry
       (v 'ensure-kernel-loaded!) (fn [_]
                                   (is (Thread/holdsLock registry))
                                   ;; Lazy loading legitimately publishes a cache update.
                                   (swap! registry update name assoc :kernel-handle :original)
                                   {:kernel-handle :original})
       (v 'ensure-seg) (fn [_ key ^long bytes]
                        (is (Thread/holdsLock registry))
                        (is (= 8 bytes))
                        (MemorySegment/ofArray (if (= :abi-arg-0 key) staged-x staged-out)))
       #'ze/launch! (fn [handle ^long groups ^long workgroup arguments]
                      (is (Thread/holdsLock registry))
                      (is (= :registration-in-use
                             (:reason (ex-data (error-of (fn [] (cleanup/assert-registry-mutable! registry)))))))
                      (is (= [1.25 -3.5] (vec staged-x)))
                      (aset staged-x 0 (float 9.0))
                      (swap! seen conj [handle groups workgroup (drop 2 arguments)]))}
      #(do
         (is (identical? out (ze/invoke-registered-kernel name [x] out [2.0] 2)))
         (is (= [[:original 1 4 [{:type :float :value (float 2.0)} {:type :long :value 2}]]] @seen))
         (is (= [9.0 -3.5] (vec x)))
         (is (= [7.0 8.0] (vec out)))
         (is (nil? (cleanup/assert-registry-mutable! registry)))))))

(deftest map-launch-failure-preserves-identity-and-releases-use
  (let [{:keys [name registry]} (map-fixture)
        failure (ex-info "map launch failure" {})
        v #(ns-resolve 'raster.gpu.ze-runtime %)]
    (with-redefs-fn
      {#'ze/kernel-registry registry
       (v 'ensure-kernel-loaded!) (fn [_] {:kernel-handle :original})
       (v 'ensure-seg) (fn [_ _ ^long _] (MemorySegment/ofArray (float-array 2)))
       #'ze/launch! (fn [_ ^long _groups ^long _workgroup _args] (throw failure))}
      #(do
         (is (identical? failure (error-of (fn [] (ze/invoke-registered-kernel
                                                 name [(float-array 2)] (float-array 2) [2.0] 2)))))
         (is (nil? (cleanup/assert-registry-mutable! registry)))))))

(deftest map-malformed-marker-is-refused-before-native-work
  (let [{:keys [name registry]} (map-fixture)
        seen (atom [])
        v #(ns-resolve 'raster.gpu.ze-runtime %)]
    (with-redefs-fn
      {#'ze/kernel-registry registry
       (v 'ensure-kernel-loaded!) (fn [& _] (swap! seen conj :load))
       (v 'ensure-seg) (fn [_ _ ^long _] (swap! seen conj :stage))}
      #(do
         (is (some? (error-of (fn [] (ze/invoke-registered-kernel
                                     name [(double-array 2)] (float-array 2) [2.0] 2)))))
         (is (some? (error-of (fn [] (ze/invoke-registered-kernel
                                     name [(float-array 2)] (float-array 2) [] 2)))))
         (is (empty? @seen))))))

(deftest map-readback-failure-releases-the-native-use-scope
  (let [{:keys [name registry]} (map-fixture)
        failure (ex-info "map readback selection failure" {})
        launched? (atom false)
        writable? abi/writable?
        v #(ns-resolve 'raster.gpu.ze-runtime %)]
    (with-redefs-fn
      {#'ze/kernel-registry registry
       (v 'ensure-kernel-loaded!) (fn [_] {:kernel-handle :original})
       (v 'ensure-seg) (fn [_ _ ^long _] (MemorySegment/ofArray (float-array 2)))
       #'ze/launch! (fn [_ ^long _groups ^long _workgroup _args] (reset! launched? true))
       #'abi/writable? (fn [slot]
                        (if @launched?
                          (do (is (= :registration-in-use
                                     (:reason (ex-data (error-of (fn [] (cleanup/assert-registry-mutable! registry)))))))
                              (throw failure))
                          (writable? slot)))}
      #(do
         (is (identical? failure (error-of (fn [] (ze/invoke-registered-kernel
                                                 name [(float-array 2)] (float-array 2) [2.0] 2)))))
         (is @launched?)
         (is (nil? (cleanup/assert-registry-mutable! registry)))))))

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
