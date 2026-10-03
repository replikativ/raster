(ns raster.gpu.prepared-kernel-call-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.gpu.core :as gpu]
            [raster.gpu.resource-cleanup :as cleanup]))

(defn- projection-artifact []
  (artifact/make
   {:kernel-name "positional"
    :source "__kernel void positional(int left, __global float* x, int right, long n) { x[0] = left + right; }"
    :abi [(abi/slot 'value :scalar :int :c-name "left")
          (abi/slot 'x :inout :float)
          (abi/slot 'value :scalar :int :c-name "right")
          (abi/slot 'n :scalar :long :role :bound)]
    :arguments '[left x right n]
    :launch (launch/spec {:workgroup-size [2 3 4] :group-count [5 6 7]})
    :effects {:kind :in-place}}))

(deftest split-preparation-preserves-positions-not-slot-names
  (let [buffer (Object.)
        seen (atom [])
        project (ns-resolve 'raster.gpu.core 'preparation-call)]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve)
       (fn [_ name]
         (is (= "expand-pointer-binding" name))
         (fn [entry value] (swap! seen conj [entry value]) [value]))}
      (fn []
        (let [call (project :ocl:0 (projection-artifact) [buffer] [11 29] 17)]
          (is (= [{:type :int :value 11} buffer {:type :int :value 29}
                  {:type :long :value 17}]
                 (:arguments call)))
          (is (= [2 3 4] (get-in call [:geometry :workgroup-size])))
          (is (= [5 6 7] (get-in call [:geometry :group-count])))
          (is (= 1 (count @seen))))
        (is (thrown? ArithmeticException
                     (project :ocl:0 (projection-artifact) [buffer]
                              [2147483648 29] 17)))
        ;; Distinct positional values for the SAME compiler value must remain visible to
        ;; KernelCall's conflict validation, rather than collapsing through a name map.
        (is (thrown? clojure.lang.ExceptionInfo
                     (project :ocl:0 (assoc (projection-artifact)
                                           :arguments '[shared x shared n]
                                           :launch (launch/spec
                                                    {:workgroup-size [2 3 4]
                                                     :group-count '[shared 6 7]}))
                              [buffer] [11 29] 17)))))))

(deftest failed-preparation-adopts-cleanup-debt-and-pins-the-root
  (let [root (Object.) primary (ex-info "Bind failed" {})
        secondary (ex-info "Release outcome unknown" {})
        releases (atom 0)
        debt (cleanup/owner [{:id :native
                              :release #(do (swap! releases inc) (throw secondary))}])
        sess (atom {:device-id :ocl:0 :closed? false :graphs {} :prepared {}
                    :kernels {:phase [(projection-artifact)]}
                    :buffers {:data root}
                    :allocations {:data {:id :root :ownership :owned}}})]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve)
       (fn [_ name]
         (case name
           "expand-pointer-binding" (fn [_ value] [value])
           "bind-kernel-call" (fn [_ {:keys [adopt-cleanup!]}]
                                (adopt-cleanup! debt) (throw primary))
           (throw (ex-info "Unexpected native contact" {:name name}))))}
      (fn []
        (is (identical? primary
                        (try (gpu/prepare! sess :phase {"x" :data} [11 29] 17)
                             (catch Throwable e e))))
        (is (nil? (get-in @sess [:prepared :phase])))
        (is (= 1 (count (:prepared @sess))))
        (let [[key entry] (first (:prepared @sess))]
          (is (:failed-construction? entry))
          (is (= #{:data} (get-in entry [:resident-footprint :buffer-keys])))
          (is (= :prepared-buffer-retained
                 (try (gpu/free-buffer! sess :data)
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
          (is (identical? secondary
                          (try (gpu/release-prepared! sess key) (catch Throwable e e))))
          (is (= :owner-releasing
                 (try (gpu/invoke-bound! sess key)
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
        (is (= 1 @releases))
        (is (some #(identical? secondary %) (.getSuppressed primary)))))))

(deftest failed-replacement-keeps-old-nonlive-binding-and-releases-candidate
  (let [root (Object.) failure (ex-info "Old kernel release unknown" {})
        destroyed (atom [])
        old {:phase :old ::cleanup/owner
             (cleanup/owner [{:id :kernel :release #(do (swap! destroyed conj :old)
                                                       (throw failure))}])}
        sess (atom {:device-id :ocl:0 :closed? false :graphs {} :prepared {:phase old}
                    :kernels {:phase [(projection-artifact)]} :buffers {:data root}})]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve)
       (fn [_ name]
         (case name
           "expand-pointer-binding" (fn [_ value] [value])
           "bind-kernel-call" (fn [_ _]
                                {:phase :new ::cleanup/owner
                                 (cleanup/owner [{:id :kernel
                                                  :release #(swap! destroyed conj :new)}])})
           (throw (ex-info "Unexpected native contact" {:name name}))))
       (ns-resolve 'raster.gpu.core 'rt-resolve-soft)
       (fn [_ name]
         (when (= name "destroy-prepared!")
           (fn [entry] (cleanup/release! (::cleanup/owner entry)))))}
      (fn []
        (is (identical? failure
                        (try (gpu/prepare! sess :phase {"x" :data} [11 29] 17)
                             (catch Throwable e e))))
        (is (identical? old (get-in @sess [:prepared :phase])))
        (is (= [:old :new] @destroyed))
        (is (= 1 (count (:prepared @sess))))
        (is (= :owner-releasing
               (try (gpu/invoke-bound! sess :phase)
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
        (is (identical? failure
                        (try (gpu/release-prepared! sess :phase) (catch Throwable e e))))
        (is (= [:old :new] @destroyed))))))

(deftest bound-launch-serializes-with-kernel-release
  (let [entered (promise) resume (promise) releasing (promise) destroyed (promise)
        sess (atom {:device-id :ocl:0 :closed? false
                    :prepared {:phase {::cleanup/owner
                                       (cleanup/owner [{:id :kernel
                                                        :release #(deliver destroyed true)}])}}})]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve)
       (fn [_ name]
         (is (= "launch-registered-bound!" name))
         (fn [_] (deliver entered true)
           (when (= ::timeout (deref resume 5000 ::timeout))
             (throw (ex-info "Launch not unblocked" {})))))}
      (fn []
        (let [invocation (future (gpu/invoke-bound! sess :phase))]
          (try
            (is (= true (deref entered 5000 ::timeout)))
            (let [release (future (deliver releasing true)
                           (gpu/release-prepared! sess :phase))]
              (is (= true (deref releasing 5000 ::timeout)))
              (is (= ::pending (deref destroyed 50 ::pending)))
              (deliver resume true)
              (is (not= ::timeout (deref invocation 5000 ::timeout)))
              (is (nil? (deref release 5000 ::timeout)))
              (is (= true (deref destroyed 5000 ::timeout))))
            (finally (deliver resume true))))))))
