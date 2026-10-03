(ns raster.gpu.module-cleanup-test
  "Hardware-free native fault injection through the Level Zero module/kernel ownership path."
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root]
            [raster.gpu.ze-runtime :as ze])
  (:import [java.lang.foreign Arena MemorySegment ValueLayout]))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))

(defn- with-runtime [options test!]
  (with-open [arena (Arena/ofShared)]
    (let [v #(ns-resolve 'raster.gpu.ze-runtime %)
          modules (atom {}) registry (atom {}) calls (atom []) next-handle (atom 100)
          context (MemorySegment/ofAddress 900)
          state (atom {:initialized? false :arena arena :context context
                       :device (MemorySegment/ofAddress 901) :modules modules})
          failure (ex-info "native outcome unknown" {})
          native (fn [operation _ args]
                   (swap! calls conj operation)
                   (when-let [callback (:on-native options)] (callback operation args modules))
                   (when (= operation (:fail options)) (throw failure))
                   (case operation
                     "zeModuleCreate" (.set ^MemorySegment (nth args 3) ValueLayout/ADDRESS 0
                                            (if (:null-module? options) MemorySegment/NULL
                                                (MemorySegment/ofAddress (swap! next-handle inc))))
                     "zeKernelCreate" (.set ^MemorySegment (nth args 2) ValueLayout/ADDRESS 0
                                            (if (:null-kernel? options) MemorySegment/NULL
                                                (MemorySegment/ofAddress (swap! next-handle inc))))
                     nil)
                   0)]
      (with-redefs-fn
        {(v 'state) state (v 'kernel-registry) registry (v 'kernel-dispatch-registry) (atom {})
         (v 'ensure-init!) (fn []) (v 'ze-call!) native
         (v 'h-zeModuleCreate) (delay :fake) (v 'h-zeModuleDestroy) (delay :fake)
         (v 'h-zeKernelCreate) (delay :fake) (v 'h-zeKernelDestroy) (delay :fake)}
        #(do
           (root/initialize! state [] (fn [_] {:arena arena :context context
                                             :device (:device @state) :modules modules}))
           (test! {:modules modules :registry registry :calls calls :failure failure :state state
                   :close! @(v 'close-module-cache!)}))))))

(deftest module-cache-snapshots-payload-and-shares-exact-content-and-flags
  (let [input (byte-array [1 2 3]) seen (atom [])]
    (with-runtime
      {:on-native (fn [operation args modules]
                    (when (= operation "zeModuleCreate")
                      (is (every? #(some? (::cleanup/owner %)) (vals @modules)))
                      (aset-byte input 0 (byte 9))
                      (let [descriptor ^MemorySegment (nth args 2)
                            n (.get descriptor ValueLayout/JAVA_LONG 24)
                            data (.reinterpret (.get descriptor ValueLayout/ADDRESS 32) n)]
                        (swap! seen conj (vec (.toArray data ValueLayout/JAVA_BYTE))))))}
      (fn [{:keys [modules calls close!]}]
        (let [first-module (ze/load-module! input :spirv "flags")]
          (is (= first-module (ze/load-module! (byte-array [1 2 3]) :spirv "flags")))
          (is (not= first-module (ze/load-module! (byte-array [1 2 3]) :spirv "other")))
          (is (= [[1 2 3] [1 2 3]] @seen))
          (is (= 2 (count @modules)))
          (close!)
          (is (empty? @modules))
          (is (= 2 (count (filter #{"zeModuleCreate"} @calls)))))))))

(deftest concurrent-identical-loads-acquire-one-module
  (with-runtime {}
    (fn [{:keys [calls close!]}]
      (let [loads (mapv (fn [_] (future (ze/load-module! (byte-array [1 2])))) (range 4))]
        (is (apply = (mapv deref loads)))
        (is (= 1 (count (filter #{"zeModuleCreate"} @calls))))
        (close!)))))

(deftest unknown-module-create-and-readback-retain-authority-without-retry
  (doseq [options [{:fail "zeModuleCreate"} {:null-module? true}]]
    (with-runtime options
      (fn [{:keys [modules calls close!]}]
        (let [primary (error-of #(ze/load-module! (byte-array [1])))
              before @calls]
          (is (some? primary))
          (is (= 1 (count @modules)))
          (is (= :poisoned (:phase (cleanup/status (::cleanup/owner (first (vals @modules)))))))
          (is (= :owner-releasing (:reason (ex-data (error-of #(ze/load-module! (byte-array [1])))))))
          (is (identical? primary (error-of close!)))
          (is (= before @calls)))))))

(deftest fresh-kernels-own-module-borrows-and-raw-handles-cannot-release
  (with-runtime {}
    (fn [{:keys [modules calls close!]}]
      (let [module (ze/load-module! (byte-array [1]))
            a (ze/create-kernel-fresh module "a")
            b (ze/create-kernel-fresh module "b")
            entry (first (vals @modules))
            before @calls]
        (is (some? (::cleanup/owner a)))
        (is (= 2 (count @(:borrowers entry))))
        (is (= :missing-cleanup-owner (:reason (ex-data (error-of #(ze/destroy-kernel! (:handle a)))))))
        (is (= :module-in-use (:reason (ex-data (error-of close!)))))
        (is (= before @calls))
        (is (= :live (:phase (cleanup/status (::cleanup/owner entry)))))
        (ze/destroy-kernel! a)
        (ze/destroy-kernel! a)
        (is (= 1 (count @(:borrowers entry))))
        (is (= :module-in-use (:reason (ex-data (error-of close!)))))
        (ze/destroy-kernel! b)
        (close!)
        (is (= ["zeModuleCreate" "zeKernelCreate" "zeKernelCreate"
                "zeKernelDestroy" "zeKernelDestroy" "zeModuleDestroy"] @calls))))))

(deftest unknown-kernel-create-keeps-module-pinned
  (doseq [options [{:fail "zeKernelCreate"} {:null-kernel? true}]]
    (with-runtime options
      (fn [{:keys [modules calls close!]}]
        (let [module (ze/load-module! (byte-array [1]))
              primary (error-of #(ze/create-kernel-fresh module "k"))
              before @calls]
          (is (some? primary))
          (is (= 1 (count @(:borrowers (first (vals @modules))))))
          (is (= :module-in-use (:reason (ex-data (error-of close!)))))
          (is (= before @calls)))))))

(deftest failed-kernel-destroy-never-retries-and-blocks-all-module-destruction
  (with-runtime {:fail "zeKernelDestroy"}
    (fn [{:keys [modules calls failure close!]}]
      (let [module (ze/load-module! (byte-array [1]))
            kernel (ze/create-kernel-fresh module "k")]
        (ze/load-module! (byte-array [2]))
        (is (identical? failure (error-of #(ze/destroy-kernel! kernel))))
        (let [before @calls]
          (is (identical? failure (error-of #(ze/destroy-kernel! kernel))))
          (is (= :module-in-use (:reason (ex-data (error-of close!)))))
          (is (= before @calls))
          (is (= 2 (count @modules))))))))

(deftest failed-module-destruction-retains-runtime-and-attempts-independent-siblings
  (with-runtime {:fail "zeModuleDestroy"}
    (fn [{:keys [modules calls state failure close!]}]
      (ze/load-module! (byte-array [1]))
      (ze/load-module! (byte-array [2]))
      (is (= :runtime-root-leases-incomplete (:reason (ex-data (error-of ze/shutdown!)))))
      (is (identical? failure (error-of close!)))
      (is (:initialized? @state))
      (is (= 2 (count @modules)))
      (is (= 2 (count (filter #{"zeModuleDestroy"} @calls))))
      (let [before @calls]
        (is (identical? failure (error-of close!)))
        (is (= before @calls))))))

(deftest publication-loss-rolls-back-created-module-without-dropping-another-generation
  (with-runtime {}
    (fn [{:keys [modules calls]}]
      (let [replaced? (atom false)
            unrelated {:unrelated true}]
        (add-watch modules :replace
                   (fn [_ _ _ entries]
                     (when (and (not @replaced?) (some :handle (vals entries)))
                       (reset! replaced? true)
                       (reset! modules {:unrelated unrelated}))))
        (is (= :registry-generation-changed
               (:reason (ex-data (error-of #(ze/load-module! (byte-array [1])))))))
        (is (identical? unrelated (:unrelated @modules)))
        (is (= ["zeModuleCreate" "zeModuleDestroy"] @calls))))))

(deftest lost-module-generation-with-unknown-child-keeps-the-borrower-reachable
  (with-runtime
    {:on-native (fn [operation _ modules]
                  (when (= operation "zeKernelCreate") (reset! modules {})))
     :fail "zeKernelCreate"}
    (fn [{:keys [modules calls close!]}]
      (let [module (ze/load-module! (byte-array [1]))]
        (is (some? (error-of #(ze/create-kernel-fresh module "k"))))
        (is (= 1 (count @modules)))
        (is (= 1 (count @(:borrowers (first (vals @modules))))))
        (let [before @calls]
          (is (= :module-generation-retained
                 (:reason (ex-data (error-of #(ze/load-module! (byte-array [1])))))))
          (is (= :module-in-use (:reason (ex-data (error-of close!)))))
          (is (= before @calls)))))))

(deftest retired-context-or-cache-cannot-destroy-a-live-kernel
  (doseq [replacement [{:context (MemorySegment/ofAddress 999)} {:modules (atom {})}]]
    (with-runtime {}
      (fn [{:keys [state calls]}]
        (let [module (ze/load-module! (byte-array [1]))
              kernel (ze/create-kernel-fresh module "k")
              before @calls]
          (swap! state merge replacement)
          (is (= :runtime-generation-mismatch
                 (:reason (ex-data (error-of #(ze/destroy-kernel! kernel))))))
          (is (= before @calls)))))))

(deftest shutdown-declines-live-registrations-before-touching-native-resources
  (with-runtime {}
    (fn [{:keys [registry state calls]}]
      (ze/load-module! (byte-array [1]))
      (swap! registry assoc "live" {:not-a-native-handle true})
      (let [before @calls]
        (is (= :runtime-root-leases-incomplete (:reason (ex-data (error-of ze/shutdown!)))))
        (is (:initialized? @state))
        (is (= before @calls))))))

(deftest only-exact-module-references-may-create-kernels
  (with-runtime {}
    (fn [{:keys [modules calls close!]}]
      (let [module (ze/load-module! (byte-array [1]))
            before @calls]
        (doseq [forged [(:handle module)
                        (MemorySegment/ofAddress (.address ^MemorySegment (:handle module)))
                        (into {} module)]]
          (is (= :unowned-native-module
                 (:reason (ex-data (error-of #(ze/create-kernel-fresh forged "k")))))))
        (is (= before @calls))
        (close!)
        (let [replacement (ze/load-module! (byte-array [1]))]
          ;; Even with an identical synthetic address, the retired reference has no authority.
          (swap! modules assoc (:module-key replacement)
                 (assoc replacement :handle (:handle module)))
          (is (= :unowned-native-module
                 (:reason (ex-data (error-of #(ze/create-kernel-fresh module "k")))))))))))

(deftest reservation-watch-or-validator-failure-never-contacts-native-code
  (doseq [failure-kind [:watch :validator]]
    (with-runtime {}
      (fn [{:keys [modules calls]}]
        (let [failure (ex-info "reservation publication rejected" {})
              rejected? (atom false)]
          (case failure-kind
            :watch (add-watch modules :reject
                              (fn [_ _ _ entries]
                                (when (and (seq entries) (compare-and-set! rejected? false true))
                                  (throw failure))))
            :validator (set-validator! modules #(or (empty? %) (throw failure))))
          (is (identical? failure (error-of #(ze/load-module! (byte-array [1])))))
          (is (empty? @calls))
          (is (empty? @modules))
          (remove-watch modules :reject)
          (set-validator! modules nil)
          (is (some? (:handle (ze/load-module! (byte-array [1]))))))))))

(deftest loaded-publication-failure-retires-only-the-successfully-rolled-back-generation
  (doseq [kind [:watch :validator]]
    (with-runtime {}
      (fn [{:keys [modules calls]}]
        (let [failure (ex-info "loaded publication rejected" {}) rejected? (atom false)]
          (case kind
            :watch (add-watch modules :reject-loaded
                              (fn [_ _ _ entries]
                                (when (and (some :handle (vals entries))
                                           (compare-and-set! rejected? false true))
                                  (throw failure))))
            :validator (set-validator! modules
                                       #(or (not-any? :handle (vals %)) (throw failure))))
          (is (identical? failure (error-of #(ze/load-module! (byte-array [1])))))
          (is (= ["zeModuleCreate" "zeModuleDestroy"] @calls))
          (is (empty? @modules))
          (remove-watch modules :reject-loaded)
          (set-validator! modules nil)
          (is (some? (:handle (ze/load-module! (byte-array [1]))))))))))

(deftest destruction-reentry-is-rejected-before-new-native-contact
  (doseq [caught? [true false]]
    (let [reentry (atom nil)]
      (with-runtime
        {:on-native (fn [operation _ _]
                      (when (= operation "zeKernelDestroy")
                        (let [error (error-of #(ze/load-module! (byte-array [2])))]
                          (reset! reentry error)
                          (when-not caught? (throw error)))))}
        (fn [{:keys [calls close!]}]
          (let [module (ze/load-module! (byte-array [1]))
                kernel (ze/create-kernel-fresh module "k")
                error (error-of #(ze/destroy-kernel! kernel))]
            (is (= :registration-in-use (:reason (ex-data @reentry))))
            (is (= ["zeModuleCreate" "zeKernelCreate" "zeKernelDestroy"] @calls))
            (if caught?
              (do (is (nil? error)) (close!))
              (do
                (is (identical? @reentry error))
                (is (identical? error (error-of #(ze/destroy-kernel! kernel))))
                (is (= :module-in-use (:reason (ex-data (error-of close!)))))
                (is (= ["zeModuleCreate" "zeKernelCreate" "zeKernelDestroy"] @calls))))))))))

(deftest direct-module-and-kernel-native-callbacks-cannot-enter-the-registration-registry
  (doseq [operation ["zeModuleCreate" "zeKernelCreate" "zeKernelDestroy"]]
    (let [observed (atom [])
          v #(ns-resolve 'raster.gpu.ze-runtime %)]
      (with-runtime
        {:on-native (fn [current _ _]
                      (when (= current operation)
                        (swap! observed into
                               (mapv error-of
                                     [#(ze/close-kernel-arena! :unregistered)
                                      #((v 'ensure-kernel-loaded!) "unregistered")
                                      #((v 'ensure-seg) "unregistered" :stage 1)
                                      #((v 'ensure-arr) "unregistered" :array 1)]))))}
        (fn [{:keys [registry calls close!]}]
          (let [module (ze/load-module! (byte-array [1]))
                kernel (ze/create-kernel-fresh module "k")]
            (ze/destroy-kernel! kernel)
            (is (= (vec (repeat 4 :registration-in-use))
                   (mapv #(-> % ex-data :reason) @observed)))
            (is (empty? @registry))
            (is (= ["zeModuleCreate" "zeKernelCreate" "zeKernelDestroy"] @calls))
            (close!)))))))
