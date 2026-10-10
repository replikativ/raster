(ns raster.gpu.registry-cleanup-test
  "Fault injection through native registration/load/arena teardown, without a device."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.storage-representation :as probe]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.ze-runtime :as ze]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root]
            [raster.gpu.test-lifecycle :as lifecycle])
  (:import [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.lang.invoke MethodHandles MethodType]))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))

(defn- native-handle [arity f]
  (delay (.bindTo (.findVirtual (MethodHandles/lookup) clojure.lang.IFn "invoke"
                                (MethodType/genericMethodType arity)) f)))

(defn- with-backend [backend options test!]
  (with-open [arena (Arena/ofShared)]
    (let [namespace (if (= :ocl backend) 'raster.gpu.ocl-runtime 'raster.gpu.ze-runtime)
          v #(ns-resolve namespace %)
          registry (atom {}) dispatches (atom {})
          context (MemorySegment/ofAddress 900)
          module (MemorySegment/ofAddress 901)
          state (atom {:initialized? false})
          _ (root/initialize! state []
                              (fn [_] {:arena arena :context context :device MemorySegment/NULL
                                       :device-info {}}))
          calls (atom []) next-handle (atom 100)
          failure (ex-info "native fault" {})
          acquire (fn [kind]
                    (swap! calls conj [:acquire kind])
                    (when-let [callback (:on-acquire options)] (callback kind))
                    (when (= kind (:acquire-failure options)) (throw failure))
                    (MemorySegment/ofAddress (swap! next-handle inc)))
          release (fn [kind handle]
                    (swap! calls conj [:release kind handle])
                    (when-let [callback (:on-release options)] (callback kind))
                    (when (= kind (:release-failure options)) (throw failure)))
          common {(v 'kernel-registry) registry (v 'kernel-dispatch-registry) dispatches
                  (v 'ensure-init!) (fn [])
                  (v 'state) state}
          native (if (= :ocl backend)
                   {(v 'h-clCreateProgramWithSource)
                    (native-handle 5 (fn [_ _ _ _ err]
                                       (.set ^MemorySegment err ValueLayout/JAVA_INT 0 (int 0))
                                       (acquire :program)))
                    (v 'h-clBuildProgram)
                    (native-handle 6 (fn [& _]
                                       (swap! calls conj [:build])
                                       (when (:build-failure options) (throw failure)) 0))
                    (v 'h-clCreateKernel)
                    (native-handle 3 (fn [_ _ err]
                                       (.set ^MemorySegment err ValueLayout/JAVA_INT 0 (int 0))
                                       (acquire :kernel)))
                    (v 'h-clReleaseProgram) (delay :program)
                    (v 'h-clReleaseKernel) (delay :kernel)
                    (v 'cl-call!) (fn [_ handle args] (release handle (first args)))}
                   {(v 'load-module!) (fn [_] (swap! calls conj [:borrow-module]) module)
                    (v 'create-kernel-fresh) (fn [& _] {:handle (acquire :kernel)})
                    (v 'destroy-kernel!) #(release :kernel (:handle %))
                    (v 'assert-buffer-live!) #(do (cleanup/assert-live! (::cleanup/owner %)) %)
                    (v 'make-buffer)
                    (fn [n dtype opts]
                      (lifecycle/native-buffer
                       #(do (acquire :memory)
                            {:dtype dtype :byte-size n :n-elements n
                             :segment (MemorySegment/ofArray (byte-array n))})
                       #(release :memory (:segment %)) opts))})]
      (with-redefs-fn
        (merge common native)
        #(test! {:v v :registry registry :calls calls :failure failure :module module
                 :state @(v 'state)
                 :register! @(v 'register-kernel!) :load! @(v 'ensure-kernel-loaded!)
                 :stage! @(v (if (= :ocl backend) 'ensure-host-seg 'ensure-seg))
                 :close! @(v 'close-kernel-arena!)})))))

(defn- spec [name] {:kernel-name name :source "source" :spv-bytes (byte-array [1 2])})
(defn- releases [calls] (mapv second (filter #(= :release (first %)) @calls)))

(deftest ze-precompiled-payload-is-isolated-from-caller-and-exported-metadata
  (with-backend :ze {}
    (fn [{:keys [register! load! close! registry state v]}]
      (let [name "payload_snapshot"
            input (spec name)
            read! @(v 'kernel-registry-entry)
            loaded-bytes (atom nil)]
        (register! name input :fixture)
        (let [owner (::cleanup/owner (get @registry name))
              exported (read! name)]
          (aset-byte ^bytes (:spv-bytes input) 0 (byte 9))
          (is (= [1 2] (vec (:spv-bytes (read! name)))))
          (aset-byte ^bytes (:spv-bytes exported) 0 (byte 8))
          (is (= [1 2] (vec (:spv-bytes (read! name)))))
          (is (not (identical? (:spv-bytes exported) (:spv-bytes (read! name)))))
          (register! name (read! name) :fixture)
          (is (identical? owner (::cleanup/owner (get @registry name))))
          (with-redefs-fn {(v 'load-module!)
                          (fn [payload] (reset! loaded-bytes (vec payload)) MemorySegment/NULL)}
            #(load! name))
          (is (= [1 2] @loaded-bytes))
          (close! :fixture)
          (is (zero? (root/lease-count state))))))))

(deftest ze-precompiled-payload-drift-fails-before-native-loading
  (with-backend :ze {}
    (fn [{:keys [register! load! close! registry state calls]}]
      (let [name "payload_drift"]
        (register! name (spec name) :fixture)
        ;; Fault injection into private operational storage, not a public API.
        (aset-byte ^bytes (:spv-bytes (get @registry name)) 0 (byte 8))
        (is (= :registration-payload-mutated
               (:reason (ex-data (error-of #(load! name))))))
        (is (empty? @calls) "No module or kernel may be acquired after payload drift")
        (is (zero? (root/lease-count state)))
        (close! :fixture)))))

(deftest registration-remains-lazy-and-loaded-generations-have-one-balanced-root-pin
  (doseq [backend [:ocl :ze]]
    (with-backend backend {}
      (fn [{:keys [register! load! close! state registry calls]}]
        (let [info (spec "lazy_pin")]
          (register! "lazy_pin" info :fixture)
          (is (zero? (root/lease-count state)))
          (is (empty? @calls))
          (let [loaded (load! "lazy_pin")
                owner (::cleanup/owner loaded)]
            (is (= 1 (root/lease-count state)))
            (is (identical? owner (::cleanup/owner (load! "lazy_pin"))))
            (register! "lazy_pin" info :fixture)
            (is (identical? owner (::cleanup/owner (get @registry "lazy_pin"))))
            (is (= 1 (root/lease-count state)))
            (close! :fixture)
            (is (zero? (root/lease-count state)))
            (is (empty? @registry))))))))

(deftest unknown-registration-create-and-destroy-keep-the-root-pinned
  (doseq [backend [:ocl :ze] fault [:create :destroy]]
    (with-backend backend
      (if (= fault :create)
        {:acquire-failure (if (= backend :ocl) :program :kernel)}
        {:release-failure :kernel})
      (fn [{:keys [register! load! close! state registry failure]}]
        (register! "uncertain_pin" (spec "uncertain_pin") :fixture)
        (if (= fault :create)
          (is (identical? failure (error-of #(load! "uncertain_pin"))))
          (do (load! "uncertain_pin")
              (is (identical? failure (error-of #(close! :fixture))))))
        (is (= 1 (root/lease-count state)))
        (let [owner (::cleanup/owner (get @registry "uncertain_pin"))]
          (is (some? owner))
          (is (= :runtime-root-lease (last (cleanup/pending owner))))
          (is (identical? failure (error-of #(close! :fixture))))
          (is (= 1 (root/lease-count state))))))))

(deftest known-build-failure-retires-program-before-releasing-the-root-pin
  (with-backend :ocl {:build-failure true}
    (fn [{:keys [register! load! state calls failure]}]
      (register! "failed_build" (spec "failed_build") :fixture)
      (is (identical? failure (error-of #(load! "failed_build"))))
      (is (= [:program] (releases calls)))
      (is (zero? (root/lease-count state))))))

(deftest pure-root-admission-decline-does-not-create-indeterminate-registration-debt
  (doseq [backend [:ocl :ze]]
    (with-backend backend {}
      (fn [{:keys [register! load! state registry calls v]}]
        (register! "unavailable_root" (spec "unavailable_root") :fixture)
        (with-redefs-fn {(v 'ensure-init!) #(swap! state assoc :initialized? false)}
          (fn []
            (let [error (error-of #(load! "unavailable_root"))
                  info (get @registry "unavailable_root")]
              (is (= :runtime-root-unavailable (:reason (ex-data error))))
              (is (nil? (::cleanup/unresolved (ex-data error))))
              (is (empty? (cleanup/pending (::cleanup/owner info))))
              (is (= :not-acquired
                     (:phase @(:root-lease
                               ((if (= backend :ocl) ::ocl/registration ::ze/registration)
                                info)))))
              (is (empty? @calls)))))))))

(deftest compiler-registry-read-preserves-artifacts-without-importing-runtime-authority
  (doseq [backend [:ocl :ze]]
    (with-backend backend {}
      (fn [{:keys [register! load! stage! close! registry v]}]
        (let [artifact (cond-> (probe/emit-artifact :float :opencl-portable)
                         (= :ze backend) (assoc :spv-bytes (byte-array [1 2])))
              name (:kernel-name artifact)
              read! @(v 'kernel-registry-entry)]
          (register! name artifact :arena)
          (let [metadata (read! name)
                owner (::cleanup/owner (get @registry name))]
            (load! name)
            (stage! name :input 16)
            (if (= :ze backend)
              (let [snapshot (read! name)]
                (is (= (dissoc metadata :spv-bytes) (dissoc snapshot :spv-bytes)))
                (is (= (vec (:spv-bytes metadata)) (vec (:spv-bytes snapshot))))
                (is (not (identical? (:spv-bytes metadata) (:spv-bytes snapshot)))))
              (is (identical? metadata (read! name))))
            (is (kart/kernel-artifact? metadata))
            (is (= (select-keys artifact [:source :abi :arguments :launch :effects :attributes])
                   (select-keys metadata [:source :abi :arguments :launch :effects :attributes])))
            (is (not-any? #(contains? metadata %)
                          [:arena-id :program :module :kernel-handle :entry-name ::cleanup/owner
                           :raster.gpu.ocl-runtime/registration :raster.gpu.ze-runtime/registration
                           :raster.gpu.ze-runtime/registration-payload-identity]))
            (register! name metadata :arena)
            (is (identical? owner (::cleanup/owner (get @registry name))))
            (is (some? (:kernel-handle (get @registry name))))
            (is (= :arena (:arena-id (get @registry name))))
            (is (not-any? #(contains? (read! name) %)
                          [:arena-id :program :module :kernel-handle :entry-name ::cleanup/owner
                           :raster.gpu.ocl-runtime/registration :raster.gpu.ze-runtime/registration
                           :raster.gpu.ze-runtime/registration-payload-identity])
                "same-program refresh still exposes only admitted compiler metadata")
            (close! :arena)))))))

(deftest registration-rejects-imported-native-authority-before-contact
  (doseq [backend [:ocl :ze]]
    (with-backend backend {}
      (fn [{:keys [register! registry calls]}]
        (doseq [field (if (= :ocl backend)
                        [:arena-id :program :kernel-handle ::cleanup/owner :raster.gpu.ocl-runtime/registration]
                        [:arena-id :module :entry-name :kernel-handle ::cleanup/owner
                         :raster.gpu.ze-runtime/registration
                         :raster.gpu.ze-runtime/registration-payload-identity])]
          (is (= :invalid-kernel-registration
                 (:reason (ex-data (error-of #(register! "probe" (assoc (spec "probe") field :forged) :arena))))))
          (is (empty? @registry))
          (is (empty? @calls)))
        (register! "probe" (spec "probe") :arena)
        (let [prior (get @registry "probe")]
          (is (= :invalid-kernel-registration
                 (:reason (ex-data (error-of #(register! "probe" (assoc (spec "probe") ::cleanup/owner :forged) :arena))))))
          (is (identical? prior (get @registry "probe")))
          (is (empty? @calls)))))))

(deftest caught-reentrant-lifecycle-mutation-cannot-poison-acquisition
  (doseq [backend [:ocl :ze]]
    (let [callback (atom nil) errors (atom [])]
      (with-backend backend {:on-acquire #(when (= :kernel %) (@callback))}
        (fn [{:keys [register! load! close! registry calls]}]
          (reset! callback
                  #(doseq [operation [(fn [] (load! "probe"))
                                      (fn [] (close! :arena))
                                      (fn [] (register! "probe" (assoc (spec "probe") :source "replacement") :arena))]]
                     (swap! errors conj (error-of operation))))
          (register! "probe" (spec "probe") :arena)
          (load! "probe")
          (is (= [:registration-in-use :registration-in-use :registration-in-use]
                 (mapv (comp :reason ex-data) @errors)))
          (is (= :live (:phase (cleanup/status (::cleanup/owner (get @registry "probe"))))))
          (close! :arena)
          (is (empty? @registry))
          (is (= (if (= :ocl backend) [:kernel :program] [:kernel]) (releases calls))))))))

(deftest lost-loaded-generation-retains-unresolved-cleanup-for-arena-close
  (doseq [backend [:ocl :ze]]
    (with-backend backend {:release-failure :kernel}
      (fn [{:keys [register! load! close! registry calls]}]
        (register! "probe" (spec "probe") :arena)
        (let [owner (::cleanup/owner (get @registry "probe"))]
          (add-watch registry :remove-loaded
                     (fn [_ ref _ current]
                       (when (:kernel-handle (get current "probe"))
                         (swap! ref dissoc "probe"))))
          (is (= :registry-generation-changed (:reason (ex-data (error-of #(load! "probe"))))))
          (is (nil? (get @registry "probe")))
          (is (= 1 (count @registry)))
          (is (identical? owner (::cleanup/owner (val (first @registry)))))
          (is (= :arena (:arena-id (val (first @registry)))))
          (let [before @calls]
            (is (some? (error-of #(close! :arena))))
            (is (= before @calls) "uncertain native teardown must not be retried")))))))

(deftest released-entry-reinserted-by-watch-is-not-success
  (let [entry (Object.) registry (atom {:entry entry}) calls (atom 0)]
    (add-watch registry :reinsert (fn [_ ref _ current]
                                    (when-not (contains? current :entry)
                                      (swap! ref assoc :entry entry))))
    (is (= :registry-generation-changed
           (:reason (ex-data (error-of #(cleanup/release-entries!
                                         registry [[:entry entry]] (fn [_] (swap! calls inc))))))))
    (is (= 1 @calls))
    (is (identical? entry (:entry @registry)))))

(deftest registration-reserves-one-owner-before-loading-and-preserves-identical-refresh
  (doseq [backend [:ocl :ze]]
    (with-backend backend {}
      (fn [{:keys [registry register! load! stage! close! calls]}]
        (register! "probe" (spec "probe") :arena)
        (let [owner (::cleanup/owner (get @registry "probe"))]
          (is (some? owner))
          (is (empty? @calls))
          (load! "probe")
          (let [segment (stage! "probe" :input 16)
                before @calls]
            (register! "probe" (assoc (spec "probe") :refreshed true) :arena)
            (is (identical? owner (::cleanup/owner (get @registry "probe"))))
            (is (identical? segment (stage! "probe" :input 8)))
            (is (not (contains? (get @registry "probe") :input)))
            (is (= before @calls))
            (is (:refreshed (get @registry "probe"))))
          (close! :arena)
          (is (empty? @registry))
          (is (= (if (= :ocl backend) [:kernel :program] [:kernel :memory]) (releases calls)))
          (let [before @calls] (close! :arena) (is (= before @calls))))))))

(deftest failed-load-retains-unknown-acquisition-and-never-retries-it
  (doseq [backend [:ocl :ze]]
    (with-backend backend {:acquire-failure :kernel}
      (fn [{:keys [registry register! load! close! calls failure]}]
        (register! "probe" (spec "probe") :arena)
        (is (identical? failure (error-of #(load! "probe"))))
        (is (seq (cleanup/pending (::cleanup/owner (get @registry "probe")))))
        (let [before @calls]
          (is (= :owner-releasing (:reason (ex-data (error-of #(load! "probe"))))))
          (is (identical? failure (error-of #(close! :arena))))
          (is (identical? failure (error-of #(close! :arena))))
          (is (= before @calls)))
        (is (empty? (releases calls)) "unknown kernel creation blocks dependent destruction")))))

(deftest failed-opencl-build-releases-the-captured-program
  (with-backend :ocl {:build-failure true}
    (fn [{:keys [registry register! load! close! failure calls]}]
      (register! "probe" (spec "probe") :arena)
      (is (identical? failure (error-of #(load! "probe"))))
      (is (= [:program] (releases calls)))
      (is (empty? (cleanup/pending (::cleanup/owner (get @registry "probe")))))
      (close! :arena)
      (is (= [:program] (releases calls))))))

(deftest failed-arena-close-retains-bad-entry-and-removes-independent-successes
  (doseq [backend [:ocl :ze]]
    (with-backend backend {}
      (fn [{:keys [registry register! load! close! calls v]}]
        (doseq [name ["bad" "good"]]
          (register! name (spec name) :arena) (load! name))
        (let [bad-handle (:kernel-handle (get @registry "bad"))
              failure (ex-info "kernel release unknown" {})
              release! (fn [handle]
                         (swap! calls conj [:release :kernel handle])
                         (when (identical? handle bad-handle) (throw failure)))
              replacements (if (= :ze backend) {(v 'destroy-kernel!) #(release! (:handle %))}
                               {(v 'cl-call!) (fn [_ kind args]
                                                (if (= :kernel kind) (release! (first args))
                                                    (swap! calls conj [:release kind (first args)])))})]
          (with-redefs-fn replacements
            (fn []
              (is (identical? failure (error-of #(close! :arena))))
              (is (= #{"bad"} (set (keys @registry))))
              (let [before @calls]
                (is (identical? failure (error-of #(close! :arena))))
                (is (= before @calls))))))))))

(deftest arena-close-never-destroys-shared-level-zero-modules-or-metadata-pointers
  (with-backend :ze {}
    (fn [{:keys [registry register! load! close! calls module]}]
      (doseq [[name arena] [["a" :first] ["b" :second]]]
        (register! name (assoc (spec name) :unrelated (MemorySegment/ofAddress 902)) arena)
        (load! name))
      (is (every? #(identical? module (:module %)) (vals @registry)))
      (close! :first)
      (is (= #{"b"} (set (keys @registry))))
      (close! :second)
      (is (= [:kernel :kernel] (releases calls))))))

(deftest staging-growth-uses-canonical-buffer-owners
  (with-backend :ze {}
    (fn [{:keys [registry register! load! stage! close! calls]}]
      (register! "probe" (spec "probe") :arena) (load! "probe")
      (let [small (stage! "probe" :input 4)
            large (stage! "probe" :input 16)]
        (is (not (identical? small large)))
        (is (= [:memory] (releases calls)))
        (is (= 1 (count @(:staging (:raster.gpu.ze-runtime/registration (get @registry "probe"))))))
        (is (identical? large (stage! "probe" :input 8)))
        (close! :arena)
        (is (= [:memory :kernel :memory] (releases calls)))
        (is (empty? @registry))))))

(deftest staging-generation-loss-retains-exact-parent-even-after-successful-child-rollback
  (doseq [backend [:ocl :ze]
          action [:remove :replace]
          storage (if (= :ze backend) [:segment :array] [:segment])]
    (with-backend backend {}
      (fn [{:keys [register! load! stage! close! registry v]}]
        (register! "probe" (spec "probe") :arena) (load! "probe")
        (let [info (get @registry "probe")
              owner (::cleanup/owner info)
              replacement (@(v 'reserve-registration) (assoc (spec "replacement") :arena-id :replacement) (spec "replacement"))
              registration (get info (if (= :ocl backend)
                                       :raster.gpu.ocl-runtime/registration
                                       :raster.gpu.ze-runtime/registration))
              cache (get registration (if (= :array storage) :arrays :cached))
              stage! (if (= :array storage) @(v 'ensure-arr) stage!)]
          (add-watch cache :lose-parent
                     (fn [_ _ _ _]
                       (if (= :remove action)
                         (swap! registry dissoc "probe")
                         (swap! registry assoc "probe" replacement))))
          (is (= :registry-generation-changed
                 (:reason (ex-data (error-of #(stage! "probe" :input 16))))))
          (is (some #(identical? owner (::cleanup/owner %)) (vals @registry)))
          (when (= :replace action) (is (identical? replacement (get @registry "probe"))))
          (close! :arena)
          (is (= :released (:phase (cleanup/status owner))))
          (close! :replacement))))))

(deftest identical-refresh-generation-loss-retains-exact-parent
  (doseq [backend [:ocl :ze] action [:remove :replace]]
    (with-backend backend {}
      (fn [{:keys [register! load! close! registry v]}]
        (register! "probe" (spec "probe") :arena) (load! "probe")
        (let [owner (::cleanup/owner (get @registry "probe"))
              replacement (@(v 'reserve-registration) (assoc (spec "replacement") :arena-id :replacement) (spec "replacement"))]
          (add-watch registry :lose-parent
                     (fn [_ ref _ current]
                       (when (:refresh-marker (get current "probe"))
                         (if (= :remove action)
                           (swap! ref dissoc "probe")
                           (swap! ref assoc "probe" replacement)))))
          (is (= :registry-generation-changed
                 (:reason (ex-data (error-of #(register! "probe" (assoc (spec "probe") :refresh-marker true) :arena))))))
          (is (some #(identical? owner (::cleanup/owner %)) (vals @registry)))
          (when (= :replace action) (is (identical? replacement (get @registry "probe"))))
          (close! :arena)
          (is (= :released (:phase (cleanup/status owner))))
          (close! :replacement))))))

(deftest lifecycle-transactions-reject-reentry-from-native-release-and-publication-watch
  (doseq [backend [:ocl :ze] transaction [:close :replace]]
    (let [callback (atom nil) errors (atom [])]
      (with-backend backend {:on-release #(when (= :kernel %) (@callback))}
        (fn [{:keys [register! load! close! registry]}]
          (register! "probe" (spec "probe") :arena) (load! "probe")
          (reset! callback
                  #(doseq [operation [(fn [] (close! :arena))
                                      (fn [] (register! "probe" (spec "probe") :arena))]]
                     (swap! errors conj (error-of operation))))
          (add-watch registry :public-reentry (fn [& _] (@callback)))
          (if (= :close transaction)
            (close! :arena)
            (register! "probe" (assoc (spec "probe") :source "replacement") :arena))
          (is (>= (count @errors) 4))
          (is (every? #(= :registration-in-use (:reason (ex-data %))) @errors))
          (remove-watch registry :public-reentry)
          (reset! callback (fn []))
          (close! :arena)
          (is (empty? @registry)))))))

(deftest staging-prune-watch-cannot-return-storage-from-a-lost-parent
  (doseq [action [:remove :replace]]
    (with-backend :ze {}
      (fn [{:keys [register! load! stage! close! registry v]}]
        (register! "probe" (spec "probe") :arena) (load! "probe")
        (stage! "probe" :input 4)
        (let [info (get @registry "probe")
              owner (::cleanup/owner info)
              staging (:staging (:raster.gpu.ze-runtime/registration info))
              replacement (@(v 'reserve-registration) (assoc (spec "replacement") :arena-id :replacement) (spec "replacement"))]
          (add-watch staging :lose-parent
                     (fn [_ _ before after]
                       (when (< (count after) (count before))
                         (if (= :remove action)
                           (swap! registry dissoc "probe")
                           (swap! registry assoc "probe" replacement)))))
          (is (= :registry-generation-changed
                 (:reason (ex-data (error-of #(stage! "probe" :input 16))))))
          (is (some #(identical? owner (::cleanup/owner %)) (vals @registry)))
          (when (= :replace action) (is (identical? replacement (get @registry "probe"))))
          (remove-watch staging :lose-parent)
          (close! :arena)
          (is (= :released (:phase (cleanup/status owner))))
          (close! :replacement))))))

(deftest staging-growth-allocation-decline-preserves-the-live-old-buffer
  (with-backend :ze {}
    (fn [{:keys [register! load! stage! close! calls v]}]
      (register! "probe" (spec "probe") :arena) (load! "probe")
      (let [small (stage! "probe" :input 4)
            before @calls
            failure (ex-info "allocation declined before native contact" {})]
        (with-redefs-fn {(v 'make-buffer) (fn [& _] (throw failure))}
          #(is (identical? failure (error-of (fn [] (stage! "probe" :input 16))))))
        (is (= before @calls))
        (is (identical? small (stage! "probe" :input 4)))
        (close! :arena)
        (is (= [:kernel :memory] (releases calls)))))))

(deftest staging-growth-uncertain-old-release-retains-debt-without-retrying
  (with-backend :ze {:release-failure :memory}
    (fn [{:keys [register! load! stage! close! calls registry]}]
      (register! "probe" (spec "probe") :arena) (load! "probe")
      (stage! "probe" :input 4)
      (is (some? (error-of #(stage! "probe" :input 16))))
      (let [before @calls
            owner (::cleanup/owner (get @registry "probe"))]
        (is (seq (cleanup/pending owner)))
        (is (= :owner-releasing (:reason (ex-data (error-of #(stage! "probe" :input 4))))))
        (is (= before @calls))
        (is (some? (error-of #(close! :arena))))
        (let [after-close @calls]
          (is (some? (error-of #(close! :arena))))
          (is (= after-close @calls)))
        (is (= [:memory :memory :kernel] (releases calls)))))))

(deftest staging-publication-failure-removes-rolled-back-cache-candidate
  (with-backend :ze {}
    (fn [{:keys [register! load! stage! close! calls registry]}]
      (register! "probe" (spec "probe") :arena) (load! "probe")
      (stage! "probe" :input 4)
      (let [cache (:cached (:raster.gpu.ze-runtime/registration (get @registry "probe")))
            failure (ex-info "cache publication watch failed" {})]
        (add-watch cache :fail (fn [_ _ _ current]
                                 (when (= 16 (:byte-size (:input current))) (throw failure))))
        (is (identical? failure (error-of #(stage! "probe" :input 16))))
        (is (nil? (:input @cache)))
        (is (= [:memory :memory] (releases calls)))
        (remove-watch cache :fail)
        (is (some? (stage! "probe" :input 8)))
        (close! :arena)
        (is (= [:memory :memory :kernel :memory] (releases calls)))))))

(deftest stale-runtime-registration-declines-before-init-or-native-use
  (doseq [backend [:ocl :ze]]
    (with-backend backend {}
      (fn [{:keys [register! load! state calls v]}]
        (register! "probe" (spec "probe") :arena) (load! "probe")
        (swap! state assoc :context (MemorySegment/ofAddress 800))
        (with-redefs-fn {(v 'ensure-init!) #(throw (AssertionError. "unexpected initialization"))}
          (fn []
            (let [before @calls]
              (is (= :runtime-generation-mismatch (:reason (ex-data (error-of #(load! "probe"))))))
              (is (= before @calls)))))))))
