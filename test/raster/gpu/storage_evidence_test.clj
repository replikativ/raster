(ns raster.gpu.storage-evidence-test
  "Hardware-free ownership and fault oracles; compiler-build evidence is stubbed explicitly."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.storage-representation :as probe]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.test-lifecycle]
            [raster.gpu.link :as link])
  (:import [java.lang.foreign MemorySegment]))

(defn- seal [value] (#'compiled/seal-artifact value))
(defn- owner []
  (let [session (atom {:device-id :ocl:unit :session-id (random-uuid)
                      :buffers {} :kernel-graphs {} :events {}})
        executable (raster.gpu.test-lifecycle/linked-executable
                    {:plan {:id :unit :outputs [:out]
                            :nodes {:out {:view {:allocation {:ownership :owned}}}}}
                     :session session :owns-session? true :closed? (atom false)
                     :lifetime-lock (Object.) :execution-state (atom {:value-epoch 0})
                     :completed-replays (atom 0) :output-leases (atom 0)
                     :output-ready? (atom true)})]
    (seal (compiled/map->Compiled {:executable executable :live-outputs (atom nil)}))))

(defn- error-of [f] (try (f) nil (catch Throwable error error)))
(defn- reason [f] (:reason (ex-data (error-of f))))

(defn- simulated [c faults f]
  (let [trace (atom [])
        session (:session (:executable c))
        hit! (fn [stage]
               (swap! trace conj stage)
               (when-let [error (get faults stage)] (throw error)))
        classify probe/classify-bytes]
    (with-redefs [compiled/execution-identity (constantly {:fingerprint "unit-program"})
                  gpu/execution-device-info
                  (fn [sess]
                    (when (:closed? @sess)
                      (throw (ex-info "closed unit session" {:reason :gpu-execution-device-closed})))
                    {:backend :unit :driver {:version 7} :storage-types #{:float :int :byte}})
                  gpu/alloc! (fn [sess specs]
                               (hit! :allocate) (swap! sess update :buffers merge specs))
                  gpu/bind-kernel-call!
                  (fn [sess key artifact arguments]
                    (hit! :bind)
                    (is (= [key] arguments))
                    (swap! sess assoc-in [:kernel-graphs key] artifact)
                    (gpu/->KernelGraphHandle key))
                  gpu/run-kernel-graph! (fn [& _] (hit! :run))
                  gpu/download-range!
                  (fn [sess key destination _]
                    (hit! :download)
                    (let [dt (first (get-in @sess [:buffers key]))
                          xs (byte-array (map unchecked-byte (probe/expected-bytes dt :little-endian)))]
                      (MemorySegment/copy (MemorySegment/ofArray xs) 0 destination 0 (alength xs))))
                  probe/classify-bytes (fn [& arguments] (hit! :classify) (apply classify arguments))
                  gpu/release-kernel-graph!
                  (fn [sess handle]
                    (swap! sess update :kernel-graphs dissoc (:key handle))
                    (hit! :release-graph))
                  gpu/free-buffer!
                  (fn [sess key] (swap! sess update :buffers dissoc key) (hit! :free))]
      (f trace session))))

(deftest successful-measurement-is-owned-offline-and-clean
  (let [c (owner) executable (:executable c)]
    (simulated c {}
      (fn [trace session]
        (let [fact (compiled/measure-storage-representation! c :f32)]
          (is (compiled/representation-evidence? fact))
          (is (identical? c (:compiled fact)))
          (is (identical? session (:session fact)))
          (is (= (:session-id @session) (:session-id fact)))
          (is (= :float (:dtype @fact)))
          (is (= :little-endian (:byte-order @fact)))
          (is (= (probe/expected-bytes :float :little-endian) (:observed-bytes @fact)))
          (is (string? (:probe-fingerprint @fact)))
          (is (string? (:device-fingerprint @fact)))
          (is (= [:allocate :bind :run :download :classify :release-graph :free] @trace))
          (is (empty? (:buffers @session)))
          (is (empty? (:kernel-graphs @session)))
          (is (empty? (:events @session)))
          (is (zero? @(:completed-replays executable)))
          (is (= 2 (:value-epoch @(:execution-state executable))))
          (is (false? @(:output-ready? executable)))
          (is (not (compiled/representation-evidence? (assoc fact :data @fact))))
          (is (= :compiled-representation-owner (reason #(deref (assoc fact :data @fact)))))
          (is (not (compiled/representation-evidence?
                    (compiled/map->ResidentRepresentationEvidence (into {} fact))))))))))

(deftest probe-failures-preserve-first-cause-and-release-every-acquired-resource
  (doseq [stage [:allocate :bind :run :download :classify :release-graph :free]]
    (testing (str stage)
      (let [c (owner) failure (ex-info (name stage) {})]
        (simulated c {stage failure}
          (fn [trace session]
            (is (identical? failure (error-of #(compiled/measure-storage-representation! c :float))))
            (is (identical? failure (:failure @(:execution-state (:executable c)))))
            (is (zero? @(:completed-replays (:executable c))))
            (is (false? @(:output-ready? (:executable c))))
            (is (empty? (:buffers @session)))
            (is (empty? (:kernel-graphs @session)))
            (is (= (not= :allocate stage) (boolean (some #{:free} @trace)))))))))
  (doseq [body-failure? [true false]]
    (let [c (owner) body (ex-info "download failed" {})
          graph (ex-info "graph cleanup failed" {}) buffer (ex-info "buffer cleanup failed" {})
          faults (cond-> {:release-graph graph :free buffer} body-failure? (assoc :download body))]
      (simulated c faults
        (fn [trace _]
          (let [error (error-of #(compiled/measure-storage-representation! c :float))]
            (is (identical? (if body-failure? body graph) error))
            (is (= (if body-failure? [graph buffer] [buffer]) (vec (.getSuppressed error))))
            (is (= [:release-graph :free] (vec (take-last 2 @trace)))))))))
  (let [c (owner) failure (ex-info "same primary and cleanup" {})]
    (simulated c {:download failure :release-graph failure}
      (fn [_ _]
        (is (identical? failure (error-of #(compiled/measure-storage-representation! c :float))))
        (is (empty? (.getSuppressed failure)))))))

(deftest admission-failures-do-not-mutate-or-allocate
  (doseq [scenario [:unsupported :lease :events :closed-session :closed-executable :attached :poisoned]]
    (let [c (owner) executable (:executable c)
          c (if (= :attached scenario)
              (seal (assoc c :executable
                           (raster.gpu.test-lifecycle/linked-executable
                            (assoc executable :owns-session? false)))) c)
          executable (:executable c)]
      (case scenario
        :lease (reset! (:output-leases executable) 1)
        :events (swap! (:session executable) assoc :events {:pending :unit})
        :closed-session (swap! (:session executable) assoc :closed? true)
        :closed-executable (reset! (:closed? executable) true)
        :poisoned (swap! (:execution-state executable) assoc :failure (ex-info "old" {}))
        nil)
      (simulated c {}
        (fn [trace _]
          (is (some? (error-of #(compiled/measure-storage-representation! c
                                (if (= :unsupported scenario) :half :float)))))
          (is (empty? @trace))
          (is (zero? (:value-epoch @(:execution-state executable))))
          (is (true? @(:output-ready? executable))))))))

(defn- completed-receipt [c]
  (let [executable (:executable c)
        _ (reset! (:output-ready? executable) true)
        lease (with-redefs [link/outputs (constantly {}) link/output-values (constantly {})]
                (link/output-lease! executable))
        node {:dtype :float :shape [4] :strides [1] :byte-length 16
              :representation :device-native :content :unit-content}]
    (seal (compiled/map->CompletedEvidence
           {:data {:program-fingerprint "unit-program" :fingerprint "unit-completion"
                   :inputs {:in node} :outputs {:out node} :post-state {}}
            :lease lease :executable executable :epoch (:value-epoch @(:execution-state executable))
            :values {} :released? (atom false)}))))

(deftest typed-storage-description-requires-live-same-owner-facts
  (let [c (owner)]
    (simulated c {}
      (fn [_ _]
        (let [fact (compiled/measure-storage-representation! c :float)
              receipt (completed-receipt c)]
          (try
            (let [description (compiled/completed-storage-description receipt {:float fact})]
              (is (= "unit-completion" (:completed-fingerprint description)))
              (is (= @fact (get-in description [:representations :float])))
              (is (= {:format :raw-array :byte-order :little-endian}
                     (get-in description [:outputs :out :storage])))
              (is (= (get-in @receipt [:outputs :out :content])
                     (get-in description [:outputs :out :content]))))
            (doseq [facts [{} {:float @fact} {:float (assoc fact :data @fact)}
                            {:float (compiled/map->ResidentRepresentationEvidence (into {} fact))}]]
              (is (= :compiled-representation-mismatch
                     (reason #(compiled/completed-storage-description receipt facts)))))
            (is (= :compiled-completed-evidence-owner
                   (reason #(compiled/completed-storage-description (assoc receipt :data @receipt)
                                                                    {:float fact}))))
            (with-redefs [gpu/execution-device-info (constantly {:driver {:version 8}})]
              (is (= :compiled-representation-mismatch
                     (reason #(compiled/completed-storage-description receipt {:float fact})))))
            (let [session (:session (:executable c)) original-id (:session-id @session)]
              (swap! session assoc :session-id (random-uuid))
              (try
                (is (= :compiled-representation-mismatch
                       (reason #(compiled/completed-storage-description receipt {:float fact}))))
                (finally (swap! session assoc :session-id original-id))))
            (is (= :link-output-lease-active
                   (reason #(compiled/measure-storage-representation! c :float))))
            (finally (.close ^java.io.Closeable receipt)))
          (is (some? (error-of #(compiled/completed-storage-description receipt {:float fact})))))))))

(deftest foreign-and-wrong-dtype-measurements-cannot-label-completed-bytes
  (let [c (owner) foreign (owner)]
    (simulated c {}
      (fn [_ _]
        (let [foreign-fact (compiled/measure-storage-representation! foreign :float)
              integer-fact (compiled/measure-storage-representation! c :int)
              receipt (completed-receipt c)]
          (try
            (doseq [fact [foreign-fact integer-fact]]
              (is (= :compiled-representation-mismatch
                     (reason #(compiled/completed-storage-description receipt {:float fact})))))
            (reset! (:closed? (:executable c)) true)
            (is (some? (error-of #(compiled/completed-storage-description receipt
                                                                        {:float integer-fact}))))
            (finally (.close ^java.io.Closeable receipt))))))))
