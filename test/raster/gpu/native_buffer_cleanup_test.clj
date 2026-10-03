(ns raster.gpu.native-buffer-cleanup-test
  "Hardware-free fault injection through canonical backend buffer constructors/destructors."
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.ze-runtime :as ze]
            [raster.gpu.core :as gpu]
            [raster.gpu.resource-cleanup :as cleanup])
  (:import [java.lang.foreign Arena MemorySegment]
           [java.lang.invoke MethodHandles]))

(defn- error-of [f] (try (f) nil (catch Throwable e e)))

(defn- with-backend [backend options test!]
  (with-open [arena (Arena/ofShared)]
    (let [namespace (if (= :ocl backend) 'raster.gpu.ocl-runtime 'raster.gpu.ze-runtime)
          v #(ns-resolve namespace %)
          creates (atom 0) releases (atom [])
          make! (if (= :ocl backend) ocl/make-buffer ze/make-buffer)
          free! (if (= :ocl backend) ocl/free-buffer! ze/free-buffer!)
          release (fn [handle]
                    (swap! releases conj handle)
                    (when-let [failure (:release-failure options)]
                      (when (or (nil? (:release-failure-at options))
                                (= (:release-failure-at options) (count @releases)))
                        (throw failure))))
          redefs (if (= :ocl backend)
                   {(v 'ensure-init!) (fn [])
                    (v 'state) (atom {:arena arena :context MemorySegment/NULL})
                    (v 'buffer-offset-alignment) (constantly 16)
                    (v 'h-clCreateBuffer)
                    (delay (MethodHandles/dropArguments
                            (MethodHandles/constant MemorySegment
                                                    (if (:null? options) MemorySegment/NULL
                                                        (MemorySegment/ofAddress 100)))
                            0 (into-array Class [MemorySegment Long/TYPE Long/TYPE
                                                 MemorySegment MemorySegment])))
                    (v 'h-clCreateSubBuffer)
                    (delay (MethodHandles/dropArguments
                            (MethodHandles/constant MemorySegment (MemorySegment/ofAddress 200))
                            0 (into-array Class [MemorySegment Long/TYPE Integer/TYPE
                                                 MemorySegment MemorySegment])))
                    (v 'read-int) (fn ^long [_] (swap! creates inc)
                                    (when-let [failure (:create-failure options)] (throw failure)) 0)
                    (v 'h-clReleaseMemObject) (delay :fake)
                    (v 'cl-call!) (fn [label _ args]
                                    (is (= "clReleaseMemObject" label))
                                    (release (first args)))}
                   {(v 'ensure-init!) (fn [])
                    (v 'state) (atom {:arena arena :context (MemorySegment/ofAddress 300)})
                    (v 'alloc-shared-raw) (fn [_ _ _ n] (swap! creates inc)
                                            (when-let [failure (:create-failure options)] (throw failure))
                                            (if (:null? options) MemorySegment/NULL (.allocate arena (long n))))
                    (v 'free-in-context!) (fn [_ segment] (release segment))})]
      (with-redefs-fn redefs #(test! {:make! make! :free! free! :creates creates
                                      :releases releases :arena arena :v v})))))

(deftest backend-buffer-preflight-has-no-native-contact
  (doseq [backend [:ocl :ze]]
    (with-backend
      backend {}
      (fn [{:keys [make! creates releases]}]
        (is (= :unknown-dtype (:reason (ex-data (error-of #(make! 1 :unknown))))))
        (is (= :gpu-buffer-negative-elements (:reason (ex-data (error-of #(make! -1 :float))))))
        (is (instance? ArithmeticException (error-of #(make! Long/MAX_VALUE :double))))
        (is (zero? @creates))
        (is (empty? @releases))))))

(deftest canonical-buffer-owner-makes-native-free-exactly-once
  (doseq [backend [:ocl :ze] n [0 4]]
    (with-backend
      backend {}
      (fn [{:keys [make! free! releases]}]
        (let [buffer (make! n :float32)]
          (is (= :float (:dtype buffer)))
          (is (= n (:n-elements buffer)))
          (is (= (* 4 n) (:byte-size buffer)))
          (is (some? (::cleanup/owner buffer)))
          (free! buffer)
          (free! buffer)
          (is (= 1 (count @releases))))))))

(deftest canonical-owner-is-published-before-native-contact
  (doseq [backend [:ocl :ze]]
    (with-backend
      backend {}
      (fn [{:keys [make! free! creates releases]}]
        (let [retained (atom nil)
              buffer (make! 4 :float
                            {:retain-owner! (fn [owner]
                                              (is (zero? @creates))
                                              (reset! retained owner))})]
          (is (identical? @retained (::cleanup/owner buffer)))
          (is (= 1 @creates))
          (free! buffer)
          (cleanup/release! @retained)
          (is (= 1 (count @releases))))))))

(deftest owner-publication-failure-prevents-native-contact
  (doseq [backend [:ocl :ze]]
    (with-backend
      backend {}
      (fn [{:keys [make! creates releases]}]
        (let [primary (ex-info "owner publication failed" {})
              error (error-of #(make! 4 :float {:retain-owner! (fn [_] (throw primary))}))]
          (is (identical? primary error))
          (is (zero? @creates))
          (is (empty? @releases))
          (is (= :invalid-cleanup-plan
                 (:reason (ex-data (error-of #(make! 4 :float {:retain-owner! :invalid}))))))
          (is (zero? @creates)))))))

(deftest prepublished-owner-retains-uncertain-acquisition
  (doseq [backend [:ocl :ze]]
    (let [primary (ex-info "native outcome uncertain" {})]
      (with-backend
        backend {:create-failure primary}
        (fn [{:keys [make! creates releases]}]
          (let [retained (atom [])
                error (error-of #(make! 4 :float
                                        {:retain-owner! (fn [owner] (swap! retained conj owner))}))]
            (is (identical? primary error))
            (is (= 2 (count @retained)))
            (is (identical? (first @retained) (second @retained)))
            (is (identical? primary (error-of #(cleanup/release! (first @retained)))))
            (is (= 1 @creates))
            (is (empty? @releases))))))))

(deftest graph-private-buffer-retains-the-exact-backend-owner
  (doseq [backend [:ocl :ze] failed? [false true]]
    (let [primary (ex-info "native outcome uncertain" {})]
      (with-backend
        backend (if failed? {:create-failure primary} {})
        (fn [{:keys [make! releases]}]
          (let [construction (volatile! {:cleanup-debts []})
                acquire! (ns-resolve (the-ns (quote raster.gpu.core)) (quote acquire-private-buffer!))]
            (if failed?
              (do
                (is (identical? primary
                                (error-of #(acquire! backend construction (fn [opts] (make! 4 :float opts))))))
                (is (= 1 (count (:cleanup-debts @construction))))
                (is (identical? primary
                                (error-of #(cleanup/release! (first (:cleanup-debts @construction))))))
                (is (empty? @releases)))
              (let [{:keys [buffer owner]} (acquire! backend construction (fn [opts] (make! 4 :float opts)))]
                (is (identical? owner (::cleanup/owner buffer)))
                (is (= [owner] (:cleanup-debts @construction)))
                (cleanup/release! owner)
                (is (= 1 (count @releases)))))))))))

(deftest uncertain-create-and-null-readback-retain-the-reserved-owner
  (doseq [backend [:ocl :ze] kind [:throw :null]]
    (let [primary (ex-info "create/readback uncertain" {}) adopted (atom nil)]
      (with-backend
        backend (if (= :throw kind) {:create-failure primary} {:null? true})
        (fn [{:keys [make! free! creates releases]}]
          (let [error (error-of #(make! 4 :float {:adopt-cleanup! (fn [owner] (reset! adopted owner))}))]
            (if (= :throw kind)
              (is (identical? primary error))
              (is (= :native-buffer-null (:reason (ex-data error)))))
            (is (= 1 @creates))
            (is (some? @adopted))
            (is (identical? error (error-of #(cleanup/release! @adopted))))
            (is (empty? @releases))))))))

(deftest uncertain-free-is-reported-without-another-native-call
  (doseq [backend [:ocl :ze]]
    (let [failure (ex-info "destroy uncertain" {})]
      (with-backend
        backend {:release-failure failure}
        (fn [{:keys [make! free! releases]}]
          (let [buffer (make! 4 :float)]
            (dotimes [_ 2] (is (identical? failure (error-of #(free! buffer)))))
            (is (= 1 (count @releases)))))))))

(deftest array-initialization-failure-rolls-back-through-the-canonical-owner
  (doseq [backend [:ocl :ze] failed-free? [false true]]
    (let [primary (ex-info "upload failed" {}) secondary (ex-info "destroy uncertain" {})]
      (with-backend
        backend (if failed-free? {:release-failure secondary} {})
        (fn [{:keys [v releases]}]
          (with-redefs-fn {(v 'array->buffer!) (fn [& _] (throw primary))}
            (fn []
              (let [error (error-of #((if (= :ocl backend) ocl/buffer-of-array ze/buffer-of-array)
                                      (float-array 4)))]
                (if failed-free?
                  (do
                    (is (identical? primary (.getCause error)))
                    (is (= [secondary] (vec (.getSuppressed primary))))
                    (is (identical? secondary
                                    (error-of #(cleanup/release! (::cleanup/unresolved (ex-data error)))))))
                  (is (identical? primary error)))
                (is (= 1 (count @releases)))))))))))

(deftest owned-opencl-and-borrowed-level-zero-slices-have-distinct-authority
  (doseq [backend [:ocl :ze]]
    (with-backend
      backend {}
      (fn [{:keys [make! free! releases]}]
        (let [root (make! 16 :float)
              slice ((if (= :ocl backend) ocl/slice-buffer ze/slice-buffer) root 16 16 :float)]
          (if (= :ocl backend)
            (do (is (some? (::cleanup/owner slice))) (free! slice) (free! slice)
                (is (= 1 (count @releases))))
            (do (is (nil? (::cleanup/owner slice)))
                (is (= :non-owning-buffer (:reason (ex-data (error-of #(free! slice))))))
                (is (empty? @releases))))
          (free! root)
          (is (= (if (= :ocl backend) 2 1) (count @releases))))))))

(deftest opencl-subbuffer-publishes-its-own-canonical-owner
  (with-backend
    :ocl {}
    (fn [{:keys [make! free! creates releases]}]
      (let [root (make! 16 :float)
            retained (atom nil)
            slice (ocl/slice-buffer root 16 16 :float
                                    {:retain-owner! (fn [owner]
                                                      (is (= 1 @creates))
                                                      (reset! retained owner))})]
        (is (identical? @retained (::cleanup/owner slice)))
        (is (not (identical? @retained (::cleanup/owner root))))
        (free! slice)
        (cleanup/release! @retained)
        (free! root)
        (is (= 2 (count @releases)))))))

(deftest overflowing-slice-extents-are-rejected-before-native-contact
  (doseq [backend [:ocl :ze]]
    (with-backend
      backend {}
      (fn [{:keys [make! free! creates releases]}]
        (let [root (make! 16 :float)]
          (is (instance? ArithmeticException
                         (error-of #((if (= :ocl backend) ocl/slice-buffer ze/slice-buffer)
                                     root Long/MAX_VALUE 16 :float))))
          (is (= 1 @creates))
          (is (empty? @releases))
          (free! root))))))

(defn- with-session-backend [backend options test!]
  (with-backend
    backend options
    (fn [{:keys [make!] :as backend-state}]
      (let [sess (atom {:device-id (if (= :ocl backend) :ocl:0 :ze:0)
                        :session-id :test :arena-id :test
                        :buffers {} :allocations {} :buffer-owners {} :closed? false})
            arena-closes (atom 0)]
        (with-redefs-fn
          {(ns-resolve (the-ns (quote raster.gpu.core)) (quote rt-resolve))
           (fn [_ name]
             (case name
               "make-buffer" make!
               "device-buffer?" (if (= :ocl backend) ocl/device-buffer? ze/device-buffer?)
               "assert-buffer-live!" (if (= :ocl backend) ocl/assert-buffer-live! ze/assert-buffer-live!)
               "array->buffer!" (fn [buffer source]
                                  (when-let [failure (:upload-failure options)] (throw failure)) buffer)
               "close-kernel-arena!" (fn [_] (swap! arena-closes inc)
                                       (when-let [failure (:arena-failure options)] (throw failure)))
               (throw (ex-info "Unexpected session runtime function" {:name name}))))}
          #(test! (assoc backend-state :sess sess :arena-closes arena-closes)))))))

(deftest session-roots-share-canonical-owners-and-close-once
  (doseq [backend [:ocl :ze]]
    (with-session-backend
      backend {}
      (fn [{:keys [sess releases arena-closes]}]
        (let [buffers (gpu/alloc! sess {:a [:float 4 (float-array 4)] :b [:int 2 nil]})]
          (doseq [[key buffer] buffers]
            (is (identical? (::cleanup/owner buffer) (get-in @sess [:buffer-owners key]))))
          (gpu/free-buffer! sess :a)
          (is (not (contains? (:buffer-owners @sess) :a)))
          (gpu/close-session! sess)
          (gpu/close-session! sess)
          (is (= 2 (count @releases)))
          (is (= 1 @arena-closes))
          (is (empty? (:buffers @sess))))))))

(deftest session-upload-failure-rolls-back-current-buffer-too
  (doseq [backend [:ocl :ze] failed-free? [false true]]
    (let [primary (ex-info "upload failed" {}) secondary (ex-info "destroy uncertain" {})]
      (with-session-backend
        backend (cond-> {:upload-failure primary} failed-free? (assoc :release-failure secondary))
        (fn [{:keys [sess releases arena-closes]}]
          (is (identical? primary
                          (error-of #(gpu/alloc! sess {:a [:float 4 (float-array 4)]}))))
          (is (empty? (:buffers @sess)))
          (is (= 1 (count @releases)))
          (if failed-free?
            (do
              (is (:closed? @sess))
              (is (= 1 (count (:buffer-owners @sess))))
              (is (identical? secondary (error-of #(gpu/close-session! sess))))
              (is (identical? secondary (error-of #(gpu/close-session! sess))))
              (is (= 1 (count @releases)))
              (is (zero? @arena-closes)))
            (do (is (empty? (:buffer-owners @sess)))
                (is (not (:closed? @sess)))
                (gpu/close-session! sess)
                (is (= 1 @arena-closes)))))))))

(deftest session-retains-unknown-create-without-publishing-a-value
  (doseq [backend [:ocl :ze]]
    (let [primary (ex-info "create uncertain" {})]
      (with-session-backend
        backend {:create-failure primary}
        (fn [{:keys [sess releases arena-closes]}]
          (is (identical? primary (error-of #(gpu/alloc! sess {:a [:float 4 nil]}))))
          (is (empty? (:buffers @sess)))
          (is (= 1 (count (:buffer-owners @sess))))
          (is (:closed? @sess))
          (dotimes [_ 2] (is (identical? primary (error-of #(gpu/close-session! sess)))))
          (is (empty? @releases))
          (is (zero? @arena-closes)))))))

(deftest session-close-removes-successful-roots-before-reporting-sibling-failure
  (doseq [backend [:ocl :ze]]
    (let [failure (ex-info "second root destruction uncertain" {})]
      (with-session-backend
        backend {:release-failure failure :release-failure-at 2}
        (fn [{:keys [sess releases arena-closes]}]
          (gpu/alloc! sess (array-map :a [:float 4 nil] :b [:float 4 nil] :c [:float 4 nil]))
          (is (identical? failure (error-of #(gpu/close-session! sess))))
          (is (= 3 (count @releases)))
          (is (= #{:b} (set (keys (:buffer-owners @sess)))))
          (is (= #{:b} (set (keys (:buffers @sess)))))
          (is (identical? failure (error-of #(gpu/close-session! sess))))
          (is (= 3 (count @releases)))
          (is (zero? @arena-closes)))))))

(deftest arena-close-failure-cannot-resurrect-released-roots
  (doseq [backend [:ocl :ze]]
    (let [failure (ex-info "arena close failed" {})]
      (with-session-backend
        backend {:arena-failure failure}
        (fn [{:keys [sess releases]}]
          (gpu/alloc! sess {:a [:float 4 nil]})
          (dotimes [_ 2] (is (identical? failure (error-of #(gpu/close-session! sess)))))
          (is (= 1 (count @releases)))
          (is (empty? (:buffers @sess)))
          (is (empty? (:buffer-owners @sess))))))))

(deftest session-owner-publication-cannot-reentrantly-close
  (doseq [backend [:ocl :ze]]
    (with-session-backend
      backend {}
      (fn [{:keys [sess creates releases arena-closes]}]
        (let [observed (atom [])]
          (add-watch sess :reentrant
                     (fn [_ _ before after]
                       (when (and (empty? (:buffer-owners before)) (seq (:buffer-owners after)))
                         (swap! observed conj (error-of #(gpu/close-session! sess))))))
          (gpu/alloc! sess {:a [:float 4 nil]})
          (is (= [:reentrant-root-lifecycle] (mapv #(-> % ex-data :reason) @observed)))
          (is (= 1 @creates))
          (is (empty? @releases))
          (is (zero? @arena-closes))
          (remove-watch sess :reentrant)
          (gpu/close-session! sess)
          (is (= 1 (count @releases))))))))

(deftest throwing-session-watch-rolls-back-before-native-contact
  (doseq [backend [:ocl :ze]]
    (with-session-backend
      backend {}
      (fn [{:keys [sess creates releases]}]
        (let [primary (ex-info "watch failed after commit" {})]
          (add-watch sess :throwing
                     (fn [_ _ before after]
                       (when (and (empty? (:buffer-owners before)) (seq (:buffer-owners after)))
                         (throw primary))))
          (is (identical? primary (error-of #(gpu/alloc! sess {:a [:float 4 nil]}))))
          (is (zero? @creates))
          (is (empty? @releases))
          (is (empty? (:buffer-owners @sess)))
          (remove-watch sess :throwing)
          (gpu/close-session! sess))))))

(deftest released-root-and-borrowed-slice-reject-further-use
  (doseq [backend [:ocl :ze]]
    (with-backend
      backend {}
      (fn [{:keys [make! free!]}]
        (let [root (make! 16 :float)
              slice ((if (= :ocl backend) ocl/slice-buffer ze/slice-buffer) root 16 16 :float)
              live! (if (= :ocl backend) ocl/assert-buffer-live! ze/assert-buffer-live!)]
          (free! root)
          (is (= :owner-releasing (:reason (ex-data (error-of #(live! root))))))
          (if (= :ocl backend)
            (do
              (is (identical? slice (live! slice)))
              (is (.isAlive (.scope ^MemorySegment (:segment slice))))
              (free! slice)
              (is (not (.isAlive (.scope ^MemorySegment (:segment slice))))))
            (is (= :owner-releasing (:reason (ex-data (error-of #(live! slice))))))))))))

(deftest opencl-staging-does-not-borrow-the-global-arena
  (with-backend
    :ocl {}
    (fn [{:keys [make! free! v]}]
      (let [global (Arena/ofShared)]
        (try
          (swap! @(v (quote state)) assoc :arena global)
          (let [root (make! 4 :float)]
            (.close global)
            (is (.isAlive (.scope ^MemorySegment (:segment root))))
            (free! root)
            (is (not (.isAlive (.scope ^MemorySegment (:segment root))))))
          (finally (when (.isAlive (.scope global)) (.close global))))))))

(deftest level-zero-setup-failure-releases-the-captured-native-pointer
  (with-backend
    :ze {}
    (fn [{:keys [make! v creates releases]}]
      (let [primary (ex-info "record setup failed after native allocation" {})]
        (with-redefs-fn {(v (quote ->DeviceBuffer)) (fn [& _] (throw primary))}
          #(is (identical? primary (error-of (fn [] (make! 4 :float))))))
        (is (= 1 @creates))
        (is (= 1 (count @releases)))))))

(deftest level-zero-root-never-frees-through-a-replacement-context
  (let [real-free! @(ns-resolve (the-ns (quote raster.gpu.ze-runtime)) (quote free-in-context!))]
    (with-backend
      :ze {}
      (fn [{:keys [make! free! v]}]
        (let [root (make! 4 :float)
              runtime-state @(v (quote state))
              original (:context @runtime-state)
              native-calls (atom 0)]
          (swap! runtime-state assoc :context (MemorySegment/ofAddress 301))
          (with-redefs-fn {(v (quote free-in-context!)) real-free!
                           (v (quote ze-call!)) (fn [& _] (swap! native-calls inc))
                           (v (quote ensure-init!)) (fn [] (throw (ex-info "free must not initialize" {})))}
            (fn []
              (let [error (error-of #(free! root))]
                (is (= :runtime-generation-mismatch (:reason (ex-data error))))
                (is (zero? @native-calls))
                (swap! runtime-state assoc :context original)
                (is (identical? error (error-of #(free! root))))
                (is (zero? @native-calls))))))))))

(deftest registered-aliases-pin-the-exact-native-lifetime
  (doseq [backend [:ocl :ze] kind (if (= :ze backend) [:copy :slice] [:copy])]
    (with-session-backend
      backend {}
      (fn [{:keys [sess releases]}]
        (let [root (:root (gpu/alloc! sess {:root [:float 16 nil]}))
              alias (if (= kind :slice) (ze/slice-buffer root 16 16 :float) (assoc root :copied true))]
          (is (identical? (cleanup/lifetime-owner root) (cleanup/lifetime-owner alias)))
          (gpu/register-buffer! sess :alias alias {:ownership :borrowed :allocation-id :distinct})
          (is (= :registered-buffer-alias
                 (:reason (ex-data (error-of #(gpu/free-buffer! sess :root))))))
          (is (empty? @releases))
          (gpu/free-buffer! sess :alias)
          (gpu/free-buffer! sess :root)
          (is (= 1 (count @releases)))
          (gpu/close-session! sess))))))

(deftest in-progress-session-transfer-pins-roots-until-return
  (doseq [backend [:ocl :ze] teardown [:free :close]]
    (with-session-backend
      backend {}
      (fn [{:keys [sess releases]}]
        (gpu/alloc! sess {:root [:float 4 nil]})
        (let [entered (promise) finish (promise) teardown-entered (promise)
              resolver-var (ns-resolve (the-ns (quote raster.gpu.core)) (quote rt-resolve))
              resolver @resolver-var]
          (with-redefs-fn
            {resolver-var (fn [device name]
                            (if (= name "upload-range!")
                              (fn [buffer _ _] (deliver entered true) @finish buffer)
                              (resolver device name)))}
            (fn []
              (let [transfer (future (gpu/upload-range! sess :root (float-array 1) {:elements 1}))]
                (try
                  (is (= true (deref entered 2000 :timeout)))
                  (let [closing (future (deliver teardown-entered true)
                                        (if (= :free teardown) (gpu/free-buffer! sess :root)
                                            (gpu/close-session! sess)))]
                    (is (= true (deref teardown-entered 2000 :timeout)))
                    (is (= :waiting (deref closing 30 :waiting)))
                    (is (empty? @releases))
                    (deliver finish true)
                    (is (not= :timeout (deref transfer 2000 :timeout)))
                    (is (not= :timeout (deref closing 2000 :timeout)))
                    (is (= 1 (count @releases))))
                  (finally (deliver finish true))))))
          (gpu/close-session! sess))))))

(deftest publication-watch-cannot-reenter-a-range-transfer
  (doseq [backend [:ocl :ze]]
    (with-session-backend
      backend {}
      (fn [{:keys [sess]}]
        (let [observed (atom [])]
          (add-watch sess :range
                     (fn [_ _ before after]
                       (when (and (empty? (:buffers before)) (seq (:buffers after)))
                         (swap! observed conj
                                (error-of #(gpu/upload-range! sess :root (float-array 1) {:elements 1}))))))
          (gpu/alloc! sess {:root [:float 4 nil]})
          (is (= [:reentrant-root-lifecycle] (mapv #(-> % ex-data :reason) @observed)))
          (remove-watch sess :range)
          (gpu/close-session! sess))))))

(deftest event-publication-watch-cannot-close-retained-transfer
  (doseq [backend [:ocl :ze]]
    (with-session-backend
      backend {}
      (fn [{:keys [sess releases]}]
        (gpu/alloc! sess {:root [:float 4 nil]})
        (let [observed (atom []) closed (atom 0) released (atom 0)
              resource (reify java.lang.AutoCloseable (close [_] (swap! closed inc)))
              resolver-var (ns-resolve 'raster.gpu.core 'rt-resolve)
              resolver @resolver-var]
          (with-redefs-fn
            {resolver-var (fn [device name]
                            (case name
                              "plan-range" (fn [_ _ _ _] {:n-bytes 4})
                              "submit-range-batch!" (fn [_ _] :token)
                              "await-event!" (constantly {:bytes 4 :commands 1})
                              "release-event!" (fn [_] (swap! released inc))
                              (resolver device name)))}
            (fn []
              (add-watch sess :event-close
                         (fn [_ _ before after]
                           (when (< (count (:events before)) (count (:events after)))
                             (swap! observed conj (error-of #(gpu/close-session! sess))))))
              (let [event (gpu/submit-upload-ranges-retained!
                           sess [[(gpu/buffer-view sess :root) (float-array 1) {:elements 1}]]
                           [resource])]
                (is (= [:reentrant-root-lifecycle] (mapv #(-> % ex-data :reason) @observed)))
                (is (not (:closed? @sess)))
                (is (empty? @releases))
                (is (zero? @closed))
                (remove-watch sess :event-close)
                (gpu/await-event! sess event)
                (gpu/release-event! sess event)
                (is (= 1 @closed @released))
                (gpu/close-session! sess)
                (is (= 1 (count @releases)))
                (is (= 1 @closed @released))))))))))

(deftest replacement-publication-failure-removes-only-retired-generations
  (doseq [registry [:prepared :kernel-graphs] failure [:watch :validator :destruction]]
    (let [old {:generation :old} candidate {:generation :new}
          sess (atom {registry {:key old :unrelated :keep}})
          destroyed (atom []) primary (ex-info "publication failed" {})
          publish! cleanup/publish-replacement!]
      (case failure
        :watch (add-watch sess :throw
                          (fn [_ _ before after]
                            (when (and (not (identical? (get-in before [registry :key]) candidate))
                                       (identical? (get-in after [registry :key]) candidate))
                              (throw primary))))
        :validator (set-validator! sess #(not (identical? (get-in % [registry :key]) candidate)))
        :destruction nil)
      (let [error (error-of #(publish! sess [registry :key] candidate
                                       (fn [entry]
                                         (if (= failure :destruction) (throw primary)
                                             (swap! destroyed conj entry)))))]
        (is (some? error))
        (when (not= :validator failure) (is (identical? primary error)))
        (is (= :keep (get-in @sess [registry :unrelated])))
        (if (= :destruction failure)
          (do (is (identical? old (get-in @sess [registry :key])))
              (is (empty? @destroyed)))
          (do (is (not (contains? (get @sess registry) :key)))
              (is (= [old] @destroyed))))))))

(defn- buffer-uses [backend v buffer]
  (let [arr (float-array 4)
        plan {:buf-off 0 :host-off 0 :n-bytes 4 :host-seg (MemorySegment/ofArray arr)}]
    (into [(fn [] ((v (quote array->buffer!)) buffer arr))
           (fn [] ((v (quote zero-buffer!)) buffer))
           (fn [] ((v (quote buffer-as-float-buffer)) buffer))
           (fn [] ((v (quote buffer-as-int-buffer)) buffer))
           (fn [] ((v (quote buffer->array)) buffer))
           (fn [] ((v (quote plan-range)) buffer arr {:elements 1} :upload))
           (fn [] ((v (quote execute-range!)) buffer plan :upload))
           (fn [] ((v (quote copy-buffer-range!)) buffer buffer 0 0 1))]
          (when (= :ze backend)
            [(fn [] ((v (quote buffer-as-long-buffer)) buffer))
             (fn [] ((v (quote buffer->short-array)) buffer))
             (fn [] ((v (quote buffer->double-array)) buffer))
             (fn [] ((v (quote buffer->float-array)) buffer))
             (fn [] ((v (quote copy-doubles-to-fp16!)) buffer (double-array 4)))
             (fn [] ((v (quote copy-fp16-to-doubles!)) buffer (double-array 4)))]))))

(deftest public-backend-buffer-uses-reject-freed-and-retired-values
  (doseq [backend [:ocl :ze] kind [:freed :retired]]
    (with-backend
      backend {}
      (fn [{:keys [make! free! v releases]}]
        (let [buffer (make! 4 :float)
              runtime-state @(v (quote state))
              original (:context @runtime-state)]
          (if (= :freed kind) (free! buffer)
              (swap! runtime-state assoc :context (MemorySegment/ofAddress 400)))
          (doseq [use! (buffer-uses backend v buffer)]
            (is (= (if (= :freed kind) :owner-releasing :runtime-generation-mismatch)
                   (:reason (ex-data (error-of use!))))))
          (swap! runtime-state assoc :context original)
          (free! buffer)
          (is (= 1 (count @releases))))))))
