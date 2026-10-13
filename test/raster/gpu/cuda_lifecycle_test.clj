(ns raster.gpu.cuda-lifecycle-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-call :as call]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.gpu.cuda-driver :as driver]
            [raster.gpu.cuda-nvrtc :as nvrtc]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root])
  (:import [java.lang.foreign MemorySegment]))

(defn- kernel
  ([] (kernel :float))
  ([scalar-dtype]
   (artifact/make {:kernel-name "lifetime_contract" :target :cuda-c
                   :source "extern \"C\" __global__ void lifetime_contract(float s, float* x) {}"
                   :abi [(abi/slot 's :scalar scalar-dtype) (abi/slot 'x :inout :float)]
                   :arguments '[s x]
                   :launch (launch/spec {:workgroup-size [1] :group-count [1]})})))

(defn- with-driver [f]
  (let [operations (atom []) fault (atom nil) next-address (atom 4096)
        fake (fn [op args]
               (swap! operations conj [op args])
               (when (= op (:operation @fault)) (throw (:error @fault)))
               (case op
                 :initialize {:context (MemorySegment/ofAddress 1) :stream (MemorySegment/ofAddress 2)
                              :device 0 :compute-capability [8 6]
                              :max-workgroup-size 1024 :max-block-dimensions [1024 1024 64]
                              :max-grid-dimensions [2147483647 65535 65535]
                              :shared-local-memory 49152}
                 :allocate (swap! next-address + 4096)
                 (:module-load :module-function :event-create) (MemorySegment/ofAddress (swap! next-address + 4096))
                 nil))]
    (binding [driver/*invoke* fake]
      (with-redefs [runtime/state (atom {}) runtime/kernel-registry (atom {})
                    nvrtc/compile-artifact-ptx (fn [& _] {:ptx (byte-array [1 0])})]
        (f operations fault)))))

(defn- data-of [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (ex-data e))))

(deftest prepared-call-survives-registration-replacement-and-arena-retirement
  (with-driver
    (fn [ops _]
      (let [a (kernel) b (runtime/make-buffer 4 :float)]
        (runtime/register-kernel! (:kernel-name a) a)
        (let [p (runtime/bind-kernel-call (call/make a [{:type :float :value (float 2)} b]))]
          (try
            (runtime/register-kernel! (:kernel-name a) a)
            (runtime/close-kernel-arena! nil)
            (runtime/launch-registered-bound! p)
            (is (= 1 (count (filter #(= :launch (first %)) @ops))))
            (finally
              (runtime/destroy-prepared! p)
              (runtime/free-buffer! b)
              (runtime/close-kernel-arena! nil))))))))

(deftest runtime-does-not-silently-reuse-different-device-or-compiler-configuration
  (with-driver
    (fn [ops _]
      (runtime/initialize! {:device-ordinal 0 :include-paths ["toolkit-a"]})
      (runtime/initialize! {:device-ordinal 0 :include-paths ["toolkit-a"]})
      (doseq [config [{:device-ordinal 1 :include-paths ["toolkit-a"]}
                      {:device-ordinal 0 :include-paths ["toolkit-b"]}]]
        (is (= (if (= 1 (:device-ordinal config)) :cuda-runtime-device-unsupported :cuda-runtime-configuration-mismatch)
               (:reason (data-of #(runtime/initialize! config))))))
      (is (= 1 (count (filter #(= :initialize (first %)) @ops)))))))

(deftest unsupported-dtypes-are-explicit-and-reject-before-driver-contact
  (with-driver
    (fn [ops _]
      (is (= {:reason :cuda-unsupported-operation :backend :cuda :operation :make-buffer :dtype :complex}
             (data-of #(runtime/make-buffer 3 :complex))))
      (is (empty? @ops))
      (let [a (kernel :half) b (runtime/make-buffer 4 :float)]
        (try
          (runtime/register-kernel! (:kernel-name a) a)
          (let [before (count @ops)
                error (data-of #(runtime/bind-kernel-call
                                 (call/make a [{:type :half :value (float 2)} b])))]
            (is (= :cuda-unsupported-operation (:reason error)))
            (is (= :bind-scalar (:operation error)))
            (is (= before (count @ops))))
          (finally (runtime/free-buffer! b) (runtime/close-kernel-arena! nil)))))))

(deftest indeterminate-acquisition-is-retained-without-a-native-retry
  (doseq [operation [:allocate :module-load]]
    (with-driver
      (fn [ops fault]
        (let [failure (ex-info "injected acquisition outcome unknown" {:operation operation})
              retained (atom [])
              b (when (= :module-load operation) (runtime/make-buffer 4 :float))
              a (kernel)]
          (when b (runtime/register-kernel! (:kernel-name a) a))
          (reset! fault {:operation operation :error failure})
          (let [caught (try
                         (if b
                           (runtime/bind-kernel-call (call/make a [{:type :float :value (float 2)} b])
                                                     {:adopt-cleanup! #(swap! retained conj %)})
                           (runtime/make-buffer 4 :float {:adopt-cleanup! #(swap! retained conj %)}))
                         nil (catch Throwable e e))]
            (is (identical? failure caught))
            (is (= 1 (count @retained)))
            (is (seq (cleanup/pending (first @retained))))
            (reset! fault nil)
            (is (identical? failure (try (cleanup/release! (first @retained)) nil (catch Throwable e e))))
            (is (= 1 (count (filter #(= operation (first %)) @ops))) "unknown acquisition must not retry")
            (when b
              (is (= :owner-in-use (:reason (data-of #(runtime/free-buffer! b)))))
              (runtime/close-kernel-arena! nil))))))))

(deftest post-module-binding-failure-rolls-back-once-and-preserves-the-primary
  (with-driver
    (fn [ops fault]
      (let [a (kernel) b (runtime/make-buffer 4 :float)
            failure (ex-info "function lookup failed" {})]
        (runtime/register-kernel! (:kernel-name a) a)
        (reset! fault {:operation :module-function :error failure})
        (is (identical? failure (try (runtime/bind-kernel-call
                                      (call/make a [{:type :float :value (float 2)} b]))
                                     nil (catch Throwable e e))))
        (is (= 1 (count (filter #(= :module-unload (first %)) @ops))))
        (reset! fault nil)
        (runtime/free-buffer! b)
        (runtime/close-kernel-arena! nil)))))

(deftest explicitly-retryable-release-retains-dependent-bindings-until-success
  (with-driver
    (fn [ops fault]
      (let [a (kernel) b (runtime/make-buffer 4 :float)
            failure (ex-info "retryable module release" {:cleanup-retry-safe? true})]
        (runtime/register-kernel! (:kernel-name a) a)
        (let [p (runtime/bind-kernel-call (call/make a [{:type :float :value (float 2)} b]))]
          (reset! fault {:operation :module-unload :error failure})
          (is (identical? failure (try (runtime/destroy-prepared! p) nil (catch Throwable e e))))
          (is (seq (cleanup/pending (::cleanup/owner p))))
          (is (= :owner-in-use (:reason (data-of #(runtime/free-buffer! b)))))
          (reset! fault nil)
          (runtime/destroy-prepared! p)
          (is (empty? (cleanup/pending (::cleanup/owner p))))
          (runtime/destroy-prepared! p)
          (is (= 2 (count (filter #(= :module-unload (first %)) @ops))))
          (runtime/free-buffer! b) (runtime/close-kernel-arena! nil))))))

(deftest registration-publication-failure-retires-its-candidate-lease
  (with-driver
    (fn [_ _]
      (runtime/initialize!)
      (let [failure (ex-info "registration publication watch failed" {}) a (kernel)]
        (add-watch runtime/kernel-registry :injected-publication
                   (fn [_ _ _ after]
                     (when (get after (:kernel-name a)) (throw failure))))
        (try
          (is (identical? failure (try (runtime/register-kernel! (:kernel-name a) a)
                                       nil (catch Throwable e e))))
          (is (empty? @runtime/kernel-registry))
          (is (= 0 (root/lease-count runtime/state)) "failed candidate must not leave an invisible lease")
          (finally (remove-watch runtime/kernel-registry :injected-publication)))))))

(deftest include-path-policy-preserves-spaces-and-explicit-overrides
  (let [paths (ns-resolve 'raster.gpu.cuda-runtime 'include-paths-from-env)]
    (is (some? paths) "CUDA compiler configuration must support environment include paths")
    (when paths
      (is (= ["/a toolkit/include" "/b/include"]
             (paths {"RASTER_CUDA_INCLUDE_PATHS" (str "/a toolkit/include" java.io.File/pathSeparator "/b/include")
                     "CUDA_HOME" "/ignored"})))
      (is (= ["/cuda toolkit/include"] (paths {"CUDA_HOME" "/cuda toolkit"})))
      (is (= [] (paths {})))
      (with-driver
        (fn [_ _]
          (runtime/initialize! {:include-paths ["explicit"]})
          (is (= ["explicit"] (:include-paths @runtime/state))))))))

(deftest prepared-async-still-validates-scalar-types-before-compilation-and-driver-contact
  (with-driver
    (fn [ops _]
      (let [a (kernel) buffer (runtime/make-buffer 4 :float)
            compiled (atom 0) unexpected-prepared (atom nil)]
        (runtime/register-kernel! (:kernel-name a) a)
        (try
          (with-redefs [nvrtc/compile-artifact-ptx
                        (fn [& _] (swap! compiled inc) {:ptx (byte-array [1 0])})]
            (let [before (count @ops)
                  error (try
                          (reset! unexpected-prepared
                                  (runtime/bind-kernel-call
                                   (assoc (call/make a [{:type :float :value (float 2)} buffer])
                                          :arguments [{:type :int :value 2} buffer])
                                   {:async? true}))
                          nil
                          (catch clojure.lang.ExceptionInfo e (ex-data e)))]
              (is (= :float (:expected error)))
              (is (= :int (:actual error)))
              (is (zero? @compiled) "invalid ABI value must reject before compiler contact")
              (is (= before (count @ops)) "invalid ABI value must reject before driver contact")))
          (finally
            (when-let [p @unexpected-prepared] (runtime/destroy-prepared! p))
            (runtime/free-buffer! buffer)
            (runtime/close-kernel-arena! nil)))))))
