(ns raster.gpu.distributed-profile-test
  "Hardware-free execution plumbing, not compiler/device completion evidence."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.buffer-view :as view]
            [raster.gpu.core :as gpu]
            [raster.gpu.distributed :as runtime]
            [raster.gpu.link :as link])
  (:import [java.lang.foreign MemorySegment]))

(defn owner []
  (#'runtime/seal-runtime-value (runtime/map->DistributedExecutable
   {:plan {:id :test-plan} :state (atom :ready)
    :sessions {:physical (atom {:session-id :test-session :device-id :physical-device})}
    :readiness {:actions []}
    :projections {:first {:plan {:target :physical :nodes {}}}
                  :second {:plan {:target :physical :nodes {}}}}
    :schedule {:operations
               [{:id :first :operation {:kind :compute :dependencies []}
                 :waits [] :completion {:id :first-complete}}
                {:id :second :operation {:kind :compute :dependencies [:first]}
                 :waits [{:id :first-complete}] :completion {:id :second-complete}}]}})))

(deftest profile-is-the-only-replay-and-retains-physical-context
  (let [executable (owner) calls (atom []) opts (atom []) closes (atom 0)]
    (with-redefs [gpu/execution-device-info (constantly {:driver :test-driver})
                  hardware/descriptor-for (fn [device]
                                            (is (= :physical-device device))
                                            {:device-id device :driver-version "test"
                                             :calibration-version 3 :unrelated :ignored})
                  link/instantiate! (fn [_ options]
                                      (swap! opts conj options)
                                      (reify java.io.Closeable (close [_] (swap! closes inc))))
                  link/run! (fn [_] (swap! calls conj :run))
                  link/profile! (fn [_] (swap! calls conj :profile)
                                  {:timing-source :device-events :profile [{:duration-ns 7}]})]
      (let [report (#'runtime/execute! executable true)]
        (is (= [:profile :profile] @calls))
        (is (= 2 @closes))
        (is (every? :profile? @opts))
        (is (every? #(identical? (get (:sessions executable) :physical) (:session %)) @opts))
        (is (every? #(= {} (:external-buffers %)) @opts))
        (is (identical? (:plan executable) (:plan report)))
        (is (= :synchronous-serialized (:execution-model report)))
        (is (false? (:calibration? report)))
        (is (= [:first :second] (mapv :step (:steps report))))
        (is (= [:first] (get-in report [:steps 1 :dependencies])))
        (is (= :test-session (get-in report [:devices-before :physical :session-id])))
        (is (= {:device-id :physical-device :driver-version "test" :calibration-version 3}
               (get-in report [:devices-before :physical :hardware-evidence])))
        (is (= (:devices-before report) (:devices-after report)))
        (is (every? #(= :device-events (get-in % [:kernel-profile :timing-source])) (:steps report)))
        (is (every? #(<= 0 (:host-wall-ns %)) (:steps report)))
        (is (= :complete @(:state executable)))
        (is (thrown? clojure.lang.ExceptionInfo (runtime/run! executable)))
        (is (= [:profile :profile] @calls)))))
  (is (thrown? clojure.lang.ExceptionInfo
               (runtime/profile! (runtime/map->DistributedExecutable (into {} (owner)))))
      "a constructed owner cannot publish an authentic observation"))

(deftest normal-run-stays-unprofiled-and-failures-poison-the-one-shot-owner
  (let [executable (owner) calls (atom [])]
    (with-redefs [link/instantiate! (fn [_ options]
                                    (is (false? (:profile? options)))
                                    (reify java.io.Closeable (close [_])))
                  link/run! (fn [_] (swap! calls conj :run))
                  link/profile! (fn [_] (throw (ex-info "unexpected profile" {})))]
      (is (identical? executable (runtime/run! executable)))
      (is (= [:run :run] @calls))))
  (let [executable (owner) calls (atom 0) closes (atom 0)]
    (with-redefs [gpu/execution-device-info (constantly {})
                  hardware/descriptor-for (constantly {})
                  link/instantiate! (fn [& _] (reify java.io.Closeable (close [_] (swap! closes inc))))
                  link/profile! (fn [_] (when (= 2 (swap! calls inc))
                                         (throw (ex-info "second step failed" {}))))]
      (is (thrown? clojure.lang.ExceptionInfo (#'runtime/execute! executable true)))
      (is (= :failed @(:state executable)))
      (is (= 2 @closes))
      (is (thrown? clojure.lang.ExceptionInfo (runtime/run! executable))))))

(deftest staged-profile-measures-and-drains-each-leg-before-reusing-the-arena
  (let [make-view (fn [id device]
                    (view/view (view/allocation {:id id :device device :byte-size 36 :memory-space :device})
                               {:dtype :float :shape [9]}))
        source (make-view :source :a) target (make-view :target :b)
        calls (atom []) segments (atom [])
        submit (fn [direction session entries]
                 (let [[_ host opts] (first entries)]
                   (swap! segments conj host)
                   (swap! calls conj [direction session opts])
                   {:direction direction :host host}))]
    (with-redefs [gpu/buffer-view (fn [_ _ opts] opts)
                  gpu/submit-download-ranges! (partial submit :download)
                  gpu/submit-upload-ranges! (partial submit :upload)
                  gpu/await-event! (fn [_ event] (is (.isAlive (.scope ^MemorySegment (:host event)))))
                  gpu/event-measurement (fn [_ event] {:timing-source :test-events :direction (:direction event)})
                  gpu/release-event! (fn [_ event] (is (.isAlive (.scope ^MemorySegment (:host event)))))]
      (let [staging (atom nil)
            legs (#'runtime/transfer-host-staged! {:a :source-session :b :target-session} source target 16 true staging)]
        (is (.isAlive (.scope ^MemorySegment (:segment @staging))))
        (is (= [:download :upload :download :upload :download :upload] (mapv :direction legs)))
        (is (= [16 16 16 16 4 4] (mapv :bytes legs)))
        (is (= [0 0 4 4 8 8] (mapv :element-offset legs)))
        (is (every? #(= :test-events (get-in % [:measurement :timing-source])) legs))
        (runtime/close! (runtime/map->DistributedExecutable
                         {:state (atom :complete) :sessions {} :staging staging}))))
    (is (= 6 (count @calls)))
    (is (every? #(not (.isAlive (.scope ^MemorySegment %))) @segments))))

(deftest hardware-drift-does-not-publish-a-completed-observation
  (let [executable (owner) snapshots (atom 0)]
    (with-redefs [gpu/execution-device-info (fn [_] {:driver (swap! snapshots inc)})
                  hardware/descriptor-for (constantly {})
                  link/instantiate! (fn [& _] (reify java.io.Closeable (close [_])))
                  link/profile! (constantly {})]
      (is (= :distributed-profile-device-drift
             (try (#'runtime/execute! executable true)
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (= :failed @(:state executable)))
      (is (thrown? clojure.lang.ExceptionInfo (runtime/output-values executable))))))

(deftest failed-transfer-await-drains-and-preserves-the-primary-error
  (let [primary (ex-info "await failed" {}) cleanup (ex-info "release failed" {}) calls (atom [])]
    (with-redefs [gpu/await-event! (fn [& _] (throw primary))
                  gpu/event-measurement (fn [& _] (swap! calls conj :measurement))
                  gpu/release-event! (fn [& _] (swap! calls conj :release) (throw cleanup))]
      (is (identical? primary
                      (try (#'runtime/profile-transfer-leg! (fn [& _] :event) :session [])
                           (catch Throwable e e))))
      (is (= [:release] @calls))
      (is (= [cleanup] (vec (.getSuppressed primary)))))))

(deftest failed-staging-survives-indeterminate-events-and-retryable-session-close
  (doseq [failure-point [:submit :await]]
    (let [primary (ex-info "transfer failed" {})
          secondary (ex-info "event retirement failed" {})
          staging (atom nil) retained (atom nil) close-count (atom 0)
          v (fn [id device]
              (view/view (view/allocation {:id id :device device :byte-size 16 :memory-space :device})
                         {:dtype :float :shape [4]}))
          executable (runtime/map->DistributedExecutable
                      {:state (atom :failed) :sessions {:a :source-session :b :target-session}
                       :staging staging})]
      (with-redefs [gpu/buffer-view (fn [_ _ opts] opts)
                    gpu/submit-download-ranges! (fn [_ entries]
                                                  (reset! retained (second (first entries)))
                                                  (when (= :submit failure-point) (throw primary))
                                                  :event)
                    gpu/await-event! (fn [& _] (throw primary))
                    gpu/release-event! (fn [& _] (throw secondary))
                    gpu/close-session! (fn [_]
                                         (when (= 1 (swap! close-count inc))
                                           (throw (ex-info "session still owns transfer debt" {}))))]
        (is (identical? primary
                        (try (#'runtime/transfer-host-staged! {:a :source-session :b :target-session}
                                                             (v :source :a) (v :target :b) 16 true staging)
                             (catch Throwable e e))))
        (is (.isAlive (.scope ^MemorySegment @retained)))
        (is (thrown? clojure.lang.ExceptionInfo (runtime/close! executable)))
        (is (= :closing @(:state executable)))
        (is (.isAlive (.scope ^MemorySegment @retained)))
        (runtime/close! executable)
        (is (= :closed @(:state executable)))
        (is (nil? @staging))
        (is (not (.isAlive (.scope ^MemorySegment @retained))))))))

(deftest calibration-drift-invalidates-otherwise-stable-device-observation
  (doseq [field [:calibration-version :bandwidth-bytes-s]]
    (let [executable (owner) snapshots (atom 0)]
      (with-redefs [gpu/execution-device-info (constantly {:driver :stable})
                    hardware/descriptor-for (fn [_]
                                              {:device-id :physical-device
                                               field (swap! snapshots inc)})
                    link/instantiate! (fn [& _] (reify java.io.Closeable (close [_])))
                    link/profile! (constantly {})]
        (let [failure (try (#'runtime/execute! executable true)
                           (catch clojure.lang.ExceptionInfo e (ex-data e)))]
          (is (= :distributed-profile-device-drift (:reason failure)))
          (is (= (get-in failure [:before :physical :device])
                 (get-in failure [:after :physical :device])))
          (is (= 1 (get-in failure [:before :physical :hardware-evidence field])))
          (is (= 2 (get-in failure [:after :physical :hardware-evidence field])))
          (is (= :failed @(:state executable))))))))

(deftest shared-cleanup-errors-do-not-stop-independent-session-teardown
  (let [failure (ex-info "shared teardown failure" {}) calls (atom [])
        executable (runtime/map->DistributedExecutable
                    {:state (atom :failed) :sessions {:a :a :b :b :c :c}})]
    (with-redefs [gpu/close-session! (fn [session] (swap! calls conj session) (throw failure))]
      (is (identical? failure (try (runtime/close! executable) (catch Throwable e e))))
      (is (= #{:a :b :c} (set @calls)))
      (is (= 3 (count @calls)))
      (is (= :closing @(:state executable))))))
