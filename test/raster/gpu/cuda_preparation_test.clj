(ns raster.gpu.cuda-preparation-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.cuda-runtime :as runtime]
            [raster.gpu.cuda-driver :as driver]
            [raster.gpu.cuda-lifecycle-test :as fixture]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root]
            [raster.gpu.cuda-nvrtc :as nvrtc]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.kernel-call :as call]))

(defn- error-of [f] (try (f) nil (catch Throwable e e)))
(defn- bind! [a b async?]
  (runtime/bind-kernel-call (call/make a [{:type :float :value (float 2)} b]) {:async? async?}))

(deftest impossible-device-launch-refuses-before-compilation-or-resource-acquisition
  (#'fixture/with-driver
   (fn [ops _]
     (let [b (runtime/make-buffer 4 :float)]
       (try
         (doseq [geometry [{:workgroup-size [1025] :group-count [1]}
                           {:workgroup-size [1024 2] :group-count [1 1]}
                           {:workgroup-size [1 1 1] :group-count [1 1 65536]}
                           {:workgroup-size [1] :group-count [1] :shared-memory-bytes 49153}]]
           (let [a (artifact/validate! (assoc (#'fixture/kernel) :launch (launch/spec geometry)))
                 compiled (atom 0)]
             (runtime/register-kernel! (:kernel-name a) a)
             (let [before @ops leases (root/lease-count runtime/state)]
               (with-redefs [nvrtc/compile-artifact-ptx (fn [& _] (swap! compiled inc))]
                 (is (= :kernel-launch-device-limit
                        (:reason (ex-data (error-of #(bind! a b false)))))))
               (is (zero? @compiled))
               (is (= before @ops))
               (is (= leases (root/lease-count runtime/state))))))
         (finally (runtime/free-buffer! b) (runtime/close-kernel-arena! nil)))))))

(deftest prepared-pending-work-is-shared-with-one-physical-stream-completion
  (#'fixture/with-driver
   (fn [ops _]
     (let [a (#'fixture/kernel) b (runtime/make-buffer 4 :float)]
       (runtime/register-kernel! (:kernel-name a) a)
       (let [synchronous (bind! a b false) left (bind! a b true) right (bind! a b true)]
         (try
           (let [before (count @ops)]
             (runtime/launch-registered-bound! synchronous)
             (is (= [:launch :stream-sync] (mapv first (drop before @ops)))))
           (let [before (count @ops)]
             (runtime/launch-registered-bound! left)
             (runtime/launch-registered-bound! right)
             (runtime/launch-registered-bound! left)
             (is (= [:launch :launch :launch] (mapv first (drop before @ops)))))
           (let [before (count @ops)]
             (runtime/synchronize-async!)
             (runtime/destroy-prepared! left)
             (runtime/destroy-prepared! right)
             (is (= 1 (count (filter #(= :stream-sync (first %)) (drop before @ops))))))
           (finally
             (doseq [p [synchronous left right]] (runtime/destroy-prepared! p))
             (runtime/free-buffer! b)
             (runtime/close-kernel-arena! nil))))
       (is (zero? (root/lease-count runtime/state)))))))

(deftest launch-reservation-protects-parameters-and-module-against-concurrent-retirement
  (#'fixture/with-driver
   (fn [ops _]
     (let [a (#'fixture/kernel) b (runtime/make-buffer 4 :float)
           entered (promise) proceed (promise) native driver/*invoke*]
       (runtime/register-kernel! (:kernel-name a) a)
       (let [p (bind! a b true)]
         (try
           (binding [driver/*invoke* (fn [operation args]
                                       (when (= :launch operation)
                                         (deliver entered true)
                                         (deref proceed 5000 :timeout))
                                       (native operation args))]
             (let [worker (future (runtime/launch-registered-bound! p))]
               (try
                 (is (= true (deref entered 5000 :timeout)))
                 (is (= :owner-in-use (:reason (ex-data (error-of #(runtime/destroy-prepared! p))))))
                 (is (= :live (:phase (cleanup/status (::cleanup/owner p)))))
                 (is (.isAlive (.scope (:parameters p))))
                 (is (not-any? #(= :module-unload (first %)) @ops))
                 (finally (deliver proceed true)))
               (is (nil? (deref worker 5000 :timeout)))))
           (runtime/destroy-prepared! p)
           (is (not (.isAlive (.scope (:parameters p)))))
           (finally (runtime/destroy-prepared! p) (runtime/free-buffer! b)
                    (runtime/close-kernel-arena! nil))))))))

(deftest unknown-completion-retains-prepared-resources-without-another-native-attempt
  (#'fixture/with-driver
   (fn [ops fault]
     (let [a (#'fixture/kernel) b (runtime/make-buffer 4 :float)
           failure (ex-info "completion unknown" {})]
       (runtime/register-kernel! (:kernel-name a) a)
       (let [p (bind! a b true)]
         (runtime/launch-registered-bound! p)
         (reset! fault {:operation :stream-sync :error failure})
         (is (identical? failure (error-of #(runtime/destroy-prepared! p))))
         (is (.isAlive (.scope (:parameters p))))
         (is (= :owner-in-use (:reason (ex-data (error-of #(runtime/free-buffer! b))))))
         (is (not-any? #(= :module-unload (first %)) @ops))
         (reset! fault nil)
         (let [before @ops]
           (is (identical? failure (error-of #(runtime/destroy-prepared! p))))
           (is (= before @ops)))
         (runtime/close-kernel-arena! nil)
         (is (= 2 (root/lease-count runtime/state))))))))

(deftest stale-registration-cannot-be-compiled-after-buffer-lifetime-admission
  (#'fixture/with-driver
   (fn [ops _]
     (let [a (#'fixture/kernel) b (runtime/make-buffer 4 :float)
           original cleanup/retain-use! replace-once? (atom true)]
       (runtime/register-kernel! (:kernel-name a) a)
       (try
         (with-redefs [cleanup/retain-use!
                       (fn [owner]
                         (let [lease (original owner)]
                           (when (compare-and-set! replace-once? true false)
                             (runtime/register-kernel! (:kernel-name a) a))
                           lease))]
           (is (= :registry-generation-changed (:reason (ex-data (error-of #(bind! a b true)))))))
         (is (not-any? #(= :module-load (first %)) @ops))
         (finally (runtime/free-buffer! b) (runtime/close-kernel-arena! nil)))
       (is (zero? (root/lease-count runtime/state)))))))
