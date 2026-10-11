(ns raster.gpu.ze-direct-generation-test
  "Exact registration admission through the public direct-call route, without a driver."
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.ze-runtime :as ze]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.compatibility-map :as compatibility-map]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]
            [raster.compiler.backend.gpu.segop-opencl :as segop-opencl]
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

(deftest ocl-both-direct-map-routes-reject-replacement-before-loading
  (doseq [[route boundary] [[:value :wrapper] [:value :geometry] [:void :geometry]]]
    (let [{:keys [name registration registry]} (map-fixture)
          replacement (assoc registration :workgroup-size 8)
          seen (atom []) changed? (atom false)
          validate abi/validate-arguments! realize compatibility-map/realize-launch
          v #(ns-resolve 'raster.gpu.ocl-runtime %)
          replace! (fn [] (when (compare-and-set! changed? false true)
                            (swap! registry assoc name replacement)))]
      (with-redefs-fn
        {(v 'kernel-registry) registry
         #'abi/validate-arguments!
         (fn [slots arguments]
           (let [result (validate slots arguments)]
             (when (= boundary :wrapper) (replace!))
             result))
         #'compatibility-map/realize-launch
         (fn [& arguments]
           (let [geometry (apply realize arguments)]
             (when (= boundary :geometry) (replace!))
             geometry))
         (v 'ensure-kernel-loaded!) (fn [& _] (swap! seen conj :load))
         (v 'ensure-host-seg) (fn [_ _ ^long _] (swap! seen conj :stage))}
        #(do
           (is (= :registry-generation-changed
                  (:reason (ex-data
                            (error-of (fn [] (if (= route :value)
                                              (ocl/invoke-registered-kernel name [(float-array 2)]
                                                                            (float-array 2) [(float 2)] 2)
                                              (ocl/invoke-registered-map-void-kernel
                                               name [(float-array 2) (float-array 2)] [(float 2)] 2))))))))
           (is (empty? @seen))
           (is (identical? replacement (get @registry name))))))))

(deftest ocl-direct-map-artifact-preconditions-precede-loading
  (let [a (artifact/make
           {:kernel-name "precondition_map" :target :opencl-c
            :source "__kernel void precondition_map(__global const float *x, __global float *out, long n) {}"
            :abi [(abi/slot 'x :input :float) (abi/slot 'out :output :float :role :result)
                  (abi/slot 'n :scalar :long :role :bound)]
            :arguments '[x out n]
            :launch (launch/spec {:workgroup-size [4] :group-count [1]})
            :preconditions [{:expression 'n :op :> :value 4}]})
        seen (atom []) v #(ns-resolve 'raster.gpu.ocl-runtime %)]
    (with-redefs-fn
      {(v 'kernel-registry) (atom {(:kernel-name a) a})
       (v 'ensure-kernel-loaded!) (fn [& _] (swap! seen conj :load))}
      #(doseq [invoke [(fn [] (ocl/invoke-registered-kernel (:kernel-name a) [(float-array 2)]
                                                           (float-array 2) [] 2))
                      (fn [] (ocl/invoke-registered-map-void-kernel
                              (:kernel-name a) [(float-array 2) (float-array 2)] [] 2))]]
         (is (= :kernel-precondition-failed (:reason (ex-data (error-of invoke)))))
         (is (empty? @seen))))))

(deftest ocl-direct-map-earlier-binding-failures-precede-geometry-and-loading
  (doseq [route [:value :void]
          fault [:count :dtype :alias :scalar :negative-bound :fractional-bound]]
    (let [{:keys [name registry]} (map-fixture)
          ;; The shared fixture's x is intentionally inout; overlap is legal there. Make
          ;; this fault's immutable-input authority explicit rather than inventing a rule.
          _ (when (= fault :alias)
              (swap! registry assoc-in [name :abi 0]
                     (abi/slot 'x :input :float :aliasing :no-write-alias)))
          input (if (= fault :dtype) (double-array 2) (float-array 2))
          output (if (= fault :alias) input (float-array 2))
          scalars (case fault :count [] :scalar [{:type :double :value 2.0}] [(float 2)])
          bound (case fault :negative-bound -1 :fractional-bound 1.5 2)
          seen (atom []) realize compatibility-map/realize-launch
          v #(ns-resolve 'raster.gpu.ocl-runtime %)]
      (with-redefs-fn
        {(v 'kernel-registry) registry
         #'compatibility-map/realize-launch (fn [& arguments]
                                             (swap! seen conj :geometry)
                                             (apply realize arguments))
         (v 'ensure-kernel-loaded!) (fn [& _] (swap! seen conj :load))}
        #(do
           (is (instance? clojure.lang.ExceptionInfo
                          (error-of (fn [] (if (= route :value)
                                            (ocl/invoke-registered-kernel name [input] output scalars bound)
                                            (ocl/invoke-registered-map-void-kernel
                                             name [input output] scalars bound))))))
           (is (empty? @seen)))))))

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

(defn- void-map-fixture [typed?]
  (let [name "generation-void-map"
        registration (cond-> {:dtype :float :workgroup-size 4}
                       typed? (assoc :abi [(abi/slot 'input :input :float)
                                          (abi/slot 'state :inout :float)
                                          (abi/slot 'scale :scalar :float)
                                          (abi/slot 'n :scalar :long :role :bound)]))]
    {:name name :registration registration :registry (atom {name registration})}))

(deftest direct-map-geometry-uses-exact-integer-ceiling
  (doseq [[n wg expected] [[1 4 1] [9 4 3]
                           [9007199254740993 2 4503599627370497]
                           [Long/MAX_VALUE 2 4611686018427387904]]]
    (let [geometry (#'ze/direct-map-geometry {:workgroup-size wg} [] n {})]
      (is (= [wg] (:workgroup-size geometry)))
      (is (= [expected] (:group-count geometry)))))
  (doseq [wg [0 -1 1.5 :invalid nil]]
    (is (some? (error-of #(#'ze/direct-map-geometry {:workgroup-size 4} [] 2
                                                   {:workgroup-size wg})))))
  (is (nil? (#'ze/direct-map-geometry {:workgroup-size 4} [] 0 {})))
  (is (= [2] (:workgroup-size
              (#'ze/direct-map-geometry {:workgroup-size 4} [] 9 {:workgroup-size 2})))))

(deftest direct-map-artifact-retains-full-launch-contract
  (let [a (artifact/make
           {:kernel-name "bounded_direct_map" :target :opencl-c
            :source "__kernel void bounded_direct_map(__global float *x, long n) {}"
            :abi [(abi/slot 'x :inout :float) (abi/slot 'n :scalar :long :role :bound)]
            :arguments '[x n]
            :launch (launch/spec
                     {:workgroup-size [4]
                      :group-count [(launch/minimum (launch/ceil-div 'n 4) 3)]})})
        args [(float-array 1) {:type :long :value 100}]]
    (is (= [3] (:group-count (#'ze/direct-map-geometry a args 100 {}))))
    (is (= [3] (:group-count (#'ze/direct-map-geometry a args 100 {:workgroup-size 4}))))
    (is (= :kernel-workgroup-override
           (:reason (ex-data (error-of #(#'ze/direct-map-geometry a args 100
                                                                    {:workgroup-size 2}))))))
    (let [host (float-array 100)
          staged (float-array 100)
          seen (atom [])]
      (with-redefs-fn
        {#'ze/kernel-registry (atom {(:kernel-name a) a})
         (ns-resolve 'raster.gpu.ze-runtime 'ensure-kernel-loaded!)
         (fn [_] {:kernel-handle :artifact})
         (ns-resolve 'raster.gpu.ze-runtime 'ensure-seg)
         (fn [_ _ ^long bytes]
           (is (= 400 bytes))
           (MemorySegment/ofArray staged))
         #'ze/launch! (fn [handle ^long groups ^long wg values]
                        (swap! seen conj [handle groups wg (vec (rest values))])
                        (aset staged 0 (float 7.0)))}
        #(do
           (is (nil? (ze/invoke-registered-map-void-kernel
                      (:kernel-name a) [host] [] {:type :long :value 100})))
           (is (= [[:artifact 3 4 [{:type :long :value 100}]]] @seen))
           (is (= 7.0 (double (aget host 0)))))))
    (let [masked (assoc a :launch (launch/spec {:workgroup-size [4] :group-count [1]}))]
      (is (= [1] (:group-count (#'ze/direct-map-geometry
                               masked [(float-array 1) {:type :long :value 0}] 0 {})))
          "zero semantic bounds remain legal with an explicit positive masked launch"))))

(deftest direct-map-invalid-geometry-precedes-all-native-contact
  (doseq [typed? [false true]
          [bound opts default-wg] [[-1 {} 4] [1.5 {} 4]
                                  [2 {:workgroup-size 0} 4]
                                  [2 {:workgroup-size -1} 4]
                                  [2 {:workgroup-size 1.5} 4]
                                  [2 {} 0] [2 {} -1] [2 {} 1.5]]]
    (let [{:keys [name registration]} (void-map-fixture typed?)
          registry (atom {name (assoc registration :workgroup-size default-wg)})
          seen (atom [])]
      (with-redefs-fn
        {#'ze/kernel-registry registry
         (ns-resolve 'raster.gpu.ze-runtime 'ensure-kernel-loaded!)
         (fn [& _] (swap! seen conj :load))
         (ns-resolve 'raster.gpu.ze-runtime 'ensure-seg)
         (fn [& _] (swap! seen conj :stage))
         #'ze/launch! (fn [& _] (swap! seen conj :launch))}
        #(do
           (is (some? (error-of
                       (fn [] (ze/invoke-registered-map-void-kernel
                               name [(float-array 2) (float-array 2)] [2.0] bound opts)))))
           (is (empty? @seen))))))
  (doseq [[bound workgroup] [[-1 4] [1.5 4] [2 0] [2 -1] [2 1.5]]]
    (let [{:keys [name registration]} (map-fixture)
          seen (atom [])]
      (with-redefs-fn
        {#'ze/kernel-registry (atom {name (assoc registration :workgroup-size workgroup)})
         (ns-resolve 'raster.gpu.ze-runtime 'ensure-kernel-loaded!)
         (fn [& _] (swap! seen conj :load))
         (ns-resolve 'raster.gpu.ze-runtime 'ensure-seg)
         (fn [& _] (swap! seen conj :stage))}
        #(do
           (is (some? (error-of (fn [] (ze/invoke-registered-kernel
                                      name [(float-array 2)] (float-array 2) [2.0] bound)))))
           (is (empty? @seen)))))))

(defn- canonical-empty-map-fixture []
  (let [typed (frontend/form->program
               '(let* [result (raster.par/pmap i n float (clojure.core/aget x i))] result)
               {:dtype :float :array-types {'x :float} :scalar-types {'n :long}})
        operation (first (soac-lower/lower-typed-map typed :ze:0 :dtype :float))
        registered (segop-opencl/generate-scheduled-segmap-kernel
                     operation :dtype :float :target-dialect :opencl-portable
                     :array-types {'x :float} :scalar-types {'n :long})]
    {:name (:kernel-name registered) :registration registered}))

(deftest empty-direct-maps-validate-and-return-without-native-contact
  (doseq [backend ['raster.gpu.ze-runtime 'raster.gpu.ocl-runtime]
          canonical? [false true]]
    (let [{:keys [name registration]} (if canonical? (canonical-empty-map-fixture) (map-fixture))
          registry (atom {name registration})
          seen (atom [])
          v #(ns-resolve backend %)
          invoke @(v 'invoke-registered-kernel)
          void @(v 'invoke-registered-map-void-kernel)
          input (float-array [3.0]) output (float-array 0)
          scalars (if canonical? [] [{:type :float :value (float 2.0)}])]
      (with-redefs-fn
        {(v 'kernel-registry) registry
         (v 'ensure-kernel-loaded!) (fn [& _] (swap! seen conj :load)
                                      (throw (AssertionError. "empty map loaded a kernel")))
         (v (if (= backend 'raster.gpu.ze-runtime) 'ensure-seg 'ensure-host-seg))
         (fn [& _] (swap! seen conj :stage)
           (throw (AssertionError. "empty map staged memory")))}
        #(do
           (is (identical? output (invoke name [input] output scalars 0)))
           (is (nil? (void name [input output] scalars 0)))
           (doseq [bad [-1 1.5]]
             (is (some? (error-of (fn [] (invoke name [input] output scalars bad))))))
           (is (some? (error-of (fn [] (void name [input output] scalars 0 {:workgroup-size 0})))))
           (is (some? (error-of (fn [] (void name [(double-array 1) output] scalars 0)))))
           (is (some? (error-of (fn [] (void name [input output] (conj scalars :extra) 0)))))
           (when canonical?
             (is (some? (error-of (fn [] (void name [input input] scalars 0)))))
             (is (some? (error-of (fn [] (invoke name [input] input scalars 0)))))
             (let [bound-name (:name (first (filter (fn [slot] (= :bound (:role slot))) (:abi registration))))
                   constrained (update registration :preconditions conj
                                       {:expression bound-name :op :> :value 0})]
               (swap! registry assoc name constrained)
               (is (= :kernel-precondition-failed
                      (:reason (ex-data (error-of (fn [] (invoke name [input] output scalars 0)))))))
               (swap! registry assoc name registration))
             (is (= :kernel-workgroup-override
                    (:reason (ex-data (error-of (fn [] (void name [input output] scalars 0
                                                          {:workgroup-size 3}))))))))
           (is (empty? @seen)))))))

(deftest empty-direct-map-still-rejects-a-replaced-registration
  (doseq [backend ['raster.gpu.ze-runtime 'raster.gpu.ocl-runtime]]
    (let [{:keys [name registration]} (canonical-empty-map-fixture)
          registry (atom {name registration})
          seen (atom [])
          validate call/validate-preconditions!
          v #(ns-resolve backend %)]
      (with-redefs-fn
        {(v 'kernel-registry) registry
         #'call/validate-preconditions! (fn [a args]
                                         (validate a args)
                                         (swap! registry assoc name (assoc registration :source "replacement")))
         (v 'ensure-kernel-loaded!) (fn [& _] (swap! seen conj :load))}
        #(do
           (is (= :registry-generation-changed
                  (:reason (ex-data (error-of
                                     (fn [] ((deref (v 'invoke-registered-kernel))
                                             name [(float-array 1)] (float-array 0) [] 0)))))))
           (is (empty? @seen)))))))

(deftest zero-bound-does-not-skip-an-opaque-positive-grid-artifact
  (doseq [backend ['raster.gpu.ze-runtime 'raster.gpu.ocl-runtime]]
    (let [a (artifact/make
             {:kernel-name "opaque_zero" :target :opencl-c
              :source "__kernel void opaque_zero(__global const float *x, __global float *out, long n) {}"
              :abi [(abi/slot 'x :input :float) (abi/slot 'out :output :float :role :result)
                    (abi/slot 'n :scalar :long :role :bound)]
              :arguments '[x out n]
              :launch (launch/spec {:workgroup-size [4] :group-count [1]})})
          v #(ns-resolve backend %)
          failure (ex-info "ordinary loading boundary" {})
          seen (atom [])]
      (with-redefs-fn
        {(v 'kernel-registry) (atom {(:kernel-name a) a})
         (v 'ensure-kernel-loaded!) (fn [& _] (swap! seen conj :load) (throw failure))}
        #(do
           (is (identical? failure (error-of
                                   (fn [] ((deref (v 'invoke-registered-kernel))
                                           (:kernel-name a) [(float-array 1)] (float-array 1) [] 0)))))
           (is (= [:load] @seen)))))))

(deftest void-map-replacement-after-pure-admission-precedes-native-use
  (doseq [typed? [false true]]
    (let [{:keys [name registration registry]} (void-map-fixture typed?)
          replacement (assoc registration :workgroup-size 8)
          seen (atom []) realize compatibility-map/realize-launch
          v #(ns-resolve 'raster.gpu.ze-runtime %)]
      (with-redefs-fn
        {#'ze/kernel-registry registry
         #'compatibility-map/realize-launch
         (fn [& arguments]
           (let [geometry (apply realize arguments)]
             (swap! registry assoc name replacement)
             geometry))
         (v 'ensure-kernel-loaded!) (fn [& _] (swap! seen conj :load))
         (v 'ensure-seg) (fn [_ _ ^long _] (swap! seen conj :stage))
         #'ze/launch! (fn [_ ^long _groups ^long _workgroup _args] (swap! seen conj :launch))}
        #(do
           (is (= :registry-generation-changed
                  (:reason (ex-data (error-of (fn [] (ze/invoke-registered-map-void-kernel
                                                     name [(float-array 6) (float-array 6)] [2.0] 6)))))))
           (is (empty? @seen))
           (is (identical? replacement (get @registry name))))))))

(deftest void-map-preserves-legacy-and-typed-writes-with-default-and-override
  (doseq [typed? [false true] opts [{} {:workgroup-size 2}]]
    (let [{:keys [name registry]} (void-map-fixture typed?)
          input (float-array [1 2 3 4 5 6]) state (float-array [7 8 9 10 11 12])
          staged-input (float-array 6) staged-state (float-array 6)
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
                          (is (= 24 bytes))
                          (MemorySegment/ofArray (if (= key :void-arr-0) staged-input staged-state)))
         #'ze/launch! (fn [handle ^long groups ^long workgroup arguments]
                        (is (= :registration-in-use
                               (:reason (ex-data (error-of #(cleanup/assert-registry-mutable! registry))))))
                        (is (= [1.0 2.0 3.0 4.0 5.0 6.0] (vec staged-input)))
                        (is (= [7.0 8.0 9.0 10.0 11.0 12.0] (vec staged-state)))
                        (aset staged-input 0 (float 99))
                        (aset staged-state 0 (float 88))
                        (swap! seen conj [handle groups workgroup (drop 2 arguments)]))}
        #(do
           (is (nil? (ze/invoke-registered-map-void-kernel name [input state] [2.0] 6 opts)))
           (is (= [[:original (if (seq opts) 3 2) (if (seq opts) 2 4)
                    [{:type :float :value (float 2.0)} {:type (if typed? :long :int) :value 6}]]]
                  @seen))
           (is (= (if typed? 1.0 99.0) (aget input 0)))
           (is (= 88.0 (aget state 0)))
           (is (nil? (cleanup/assert-registry-mutable! registry))))))))

(deftest void-map-malformed-markers-and-override-precede-native-use
  (let [{:keys [name registry]} (void-map-fixture true)
        seen (atom [])
        v #(ns-resolve 'raster.gpu.ze-runtime %)]
    (with-redefs-fn
      {#'ze/kernel-registry registry
       (v 'ensure-kernel-loaded!) (fn [& _] (swap! seen conj :load))
       (v 'ensure-seg) (fn [_ _ ^long _] (swap! seen conj :stage))}
      #(do
         (doseq [[arrays scalars opts] [[[(double-array 6) (float-array 6)] [2.0] {}]
                                       [[(float-array 6) (float-array 6)] [] {}]
                                       [[(float-array 6) (float-array 6)] [2.0] {:workgroup-size :invalid}]]]
           (is (some? (error-of (fn [] (ze/invoke-registered-map-void-kernel
                                      name arrays scalars 6 opts))))))
         (is (empty? @seen))))))

(deftest void-map-native-failures-preserve-identity-and-release-use
  (doseq [phase [:launch :readback]]
    (let [{:keys [name registry]} (void-map-fixture true)
          failure (ex-info "void-map injected failure" {:phase phase})
          seen (atom [])
          v #(ns-resolve 'raster.gpu.ze-runtime %)
          fail! (fn []
                  (is (= :registration-in-use
                         (:reason (ex-data (error-of #(cleanup/assert-registry-mutable! registry))))))
                  (throw failure))]
      (with-redefs-fn
        {#'ze/kernel-registry registry
         (v 'ensure-kernel-loaded!) (fn [_] {:kernel-handle :original})
         (v 'ensure-seg) (fn [_ _ ^long _] (MemorySegment/ofArray (float-array 6)))
         #'ze/launch! (fn [_ ^long _groups ^long _workgroup _args]
                        (swap! seen conj :launch)
                        (when (= phase :launch) (fail!)))
         (v 'readback-void-map!) (fn [& _] (swap! seen conj :readback) (fail!))}
        #(do
           (is (identical? failure
                          (error-of (fn [] (ze/invoke-registered-map-void-kernel
                                           name [(float-array 6) (float-array 6)] [2.0] 6)))))
           (is (= (if (= phase :launch) [:launch] [:launch :readback]) @seen))
           (is (nil? (cleanup/assert-registry-mutable! registry))))))))

(deftest map-replacement-after-pure-admission-precedes-native-use
  (let [{:keys [name registration registry]} (map-fixture)
        replacement (assoc registration :workgroup-size 8)
        seen (atom []) realize compatibility-map/realize-launch
        v #(ns-resolve 'raster.gpu.ze-runtime %)]
    (with-redefs-fn
      {#'ze/kernel-registry registry
       #'compatibility-map/realize-launch
       (fn [& arguments]
         (let [geometry (apply realize arguments)]
           (swap! registry assoc name replacement)
           geometry))
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
