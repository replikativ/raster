(ns raster.runtime.distributed-state-device-test
  "Two exact generated producers, owner-retained capture and mapped fresh-context continuation.
   The provider and packaged identity are explicit synthetic test fixtures."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.build-manifest :as build]
            [raster.compiler.fixtures.distributed-capture :as fixture]
            [raster.compiler.ir.distributed-plan :as plan]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.completed-evidence-device-test :as identity]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.distributed :as runtime]
            [raster.runtime.numerical-content :as content]
            [raster.runtime.resident-state :as resident]
            [raster.runtime.resident-state-test :as provider-fixture]
            [raster.test-support.numerical-checkpoint :as checkpoint])
  (:import [java.lang.foreign MemorySegment ValueLayout]
           [java.nio ByteOrder]
           [java.nio.file Files OpenOption]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- decode-fields [captured provider]
  (resident/verify-distributed-restore! (:state captured) (:expected-semantics captured)
                                      (:source-plan captured) (:prepared-entries captured) (:field-specs captured))
  (let [fields (get-in captured [:state :manifest :fields])
        paths (checkpoint/temp-files (mapv :id fields))]
    (try
      (into {}
            (for [field fields
                  :let [chunk (first (:chunks field)) path (paths (:id field))
                        result (float-array (reduce * (get-in field [:value :shape])))]]
              (do
                (content/with-local-content
                 provider (:content chunk) {:tier :local}
                 (fn [lease]
                   (content/verify-chunk-lease! chunk lease)
                   (Files/write path (.toArray (content/lease-segment lease) ValueLayout/JAVA_BYTE)
                                (make-array OpenOption 0))))
                (with-open [lease (checkpoint/open-chunk-lease path chunk)]
                  (content/decode-raw-array-chunk!
                   chunk lease :float (MemorySegment/ofArray result)
                   (if (= ByteOrder/LITTLE_ENDIAN (ByteOrder/nativeOrder)) :little-endian :big-endian)))
                [(:id field) (vec result)])))
      (finally (doseq [path (vals paths)] (Files/deleteIfExists path))))))

(defn- capture-step! [target inputs step parent faults?]
  (with-redefs [build/current-identity #'identity/test-build]
    (let [{:keys [plan prepared-by-entry options capture-options expected]} (fixture/fixture target inputs)
          opts (assoc capture-options :id (keyword (str "state-" step))
                      :parents (if parent [parent] []) :logical-coordinate {:step step})]
      (with-open [owner (runtime/instantiate! plan options)]
        (runtime/run! owner)
        (let [fact (runtime/measure-storage-representation! owner target :float)
              facts {[target :float] fact}
              {:keys [provider events submissions]} (#'provider-fixture/provider (fn [& _]))
              capture! #(resident/capture-distributed! owner prepared-by-entry facts provider :local %)]
          (doseq [[entry prepared] prepared-by-entry]
            (is (= (compiled/plan prepared)
                   (get-in (plan/compute-bindings plan) [:bindings (second entry) :link-plan]))))
          (when faults?
            (is (= :distributed-runtime-owner
                   (reason #(resident/capture-distributed! (assoc owner :state (:state owner))
                                                           prepared-by-entry facts provider :local opts))))
            (is (= :compiled-execution-identity-owner
                   (reason #(resident/capture-distributed!
                             owner (update prepared-by-entry [:worker :twice] (fn [p] (assoc p :schedule (:schedule p))))
                             facts provider :local opts))))
            (is (= :resident-state-distributed-bindings
                   (reason #(resident/capture-distributed! owner (dissoc prepared-by-entry [:worker :twice])
                                                           facts provider :local opts))))
            (is (= :resident-state-distributed-field
                   (reason #(capture! (assoc-in opts [:fields 1 :key] :absent)))))
            (is (empty? @submissions) "even a later field error performs no provider write")
            (is (= :distributed-representation-mismatch
                   (reason #(resident/capture-distributed! owner prepared-by-entry
                                                           {[target :float] (assoc fact :data @fact)}
                                                           provider :local opts))))
            (doseq [phase [:submit :await]]
              (let [calls (atom 0)
                    {:keys [provider events blobs]}
                    (#'provider-fixture/provider
                     (fn [actual _]
                       (when (and (= phase actual) (= 2 (swap! calls inc)))
                         (throw (ex-info "injected second-field failure" {:reason :injected-capture-failure})))))]
                (is (= :injected-capture-failure
                       (reason #(resident/capture-distributed! owner prepared-by-entry facts provider :local opts))))
                (is (= :complete @(:state owner)))
                (is (empty? @events))
                (is (seq @blobs) "already ingested content may remain orphaned on failure")))
            (let [{:keys [provider events submissions]} (#'provider-fixture/provider (fn [& _]))]
              (with-redefs [gpu/download-range! (fn [& _] (throw (ex-info "injected download failure"
                                                                         {:reason :injected-download-failure})))]
                (is (= :injected-download-failure
                       (reason #(resident/capture-distributed! owner prepared-by-entry facts provider :local opts)))))
              (is (empty? @submissions))
              (is (empty? @events))
              (is (= :complete @(:state owner))))
            (let [awaits (atom 0) drift? (atom false) device-info gpu/execution-device-info
                  {:keys [provider events]}
                  (#'provider-fixture/provider
                   (fn [phase _] (when (and (= :await phase) (= 2 (swap! awaits inc))) (reset! drift? true))))]
              (with-redefs [gpu/execution-device-info
                            (fn [session] (cond-> (device-info session) @drift? (assoc :changed-after-ingestion true)))]
                (is (= :distributed-representation-mismatch
                       (reason #(resident/capture-distributed! owner prepared-by-entry facts provider :local opts)))))
              (is @drift?)
              (is (empty? @events))
              (is (= :complete @(:state owner)))))
          (let [close-results (atom [])
                {:keys [provider events]}
                (#'provider-fixture/provider
                 (fn [phase _]
                   (when (= phase :submit)
                     (swap! close-results conj
                            (deref (future (reason #(runtime/close! owner))) 3000 :timeout)))))
                readers (atom []) ingest! content/ingest-content!
                captured (with-redefs [content/ingest-content!
                                      (fn [provider address tier bytes reader & [options]]
                                        (swap! readers conj reader)
                                        (ingest! provider address tier bytes reader (or options {})))]
                           (resident/capture-distributed! owner prepared-by-entry facts provider :local opts))]
            (is (= [:distributed-runtime-output-scope-active :distributed-runtime-output-scope-active]
                   @close-results))
            (is (empty? @events))
            (is (= 2 (count @readers)))
            (is (= :distributed-runtime-state
                   (reason #((first @readers) 0 (MemorySegment/ofArray (byte-array 1)))))
                "a retained provider reader cannot read outside its owning output scope")
            (is (= :completed-distributed-entry-outputs (get-in captured [:state :manifest :provenance :scope])))
            (is (= [[:worker :twice] [:worker :increment]]
                   (mapv :entry (get-in captured [:state :manifest :provenance :programs]))))
            {:captured (assoc captured :source-plan plan :prepared-entries prepared-by-entry
                             :field-specs (mapv #(select-keys % [:id :step :key]) (:fields opts))
                             :expected-semantics
                             {:fields (mapv #(assoc (select-keys % [:id :value]) :coordinate-space {}) (:fields opts))
                              :logical-coordinate (:logical-coordinate opts)
                              :numerical-contract (:numerical-contract opts)})
             :provider provider :expected expected}))))))

(defn- check-restart! [target]
  (let [initial {:twice (mapv #(* 0.125 (- % 100)) (range 257))
                 :increment (mapv #(* 0.25 (- % 30)) (range 129))}
        first (capture-step! target initial 1 nil true)
        ;; The producing owner is closed before any mapped read or fresh session.
        decoded (decode-fields (:captured first) (:provider first))
        second (capture-step! target decoded 2 (get-in first [:captured :state :manifest :id]) false)
        again (decode-fields (:captured second) (:provider second))]
    (is (= (:expected first) decoded))
    (is (= (:expected second) again))
    (is (= [:state-1] (get-in second [:captured :state :manifest :parents])))))

(deftest distributed-entry-capture-survives-fresh-context-restart
  (if @opencl/opencl-available? (check-restart! :ocl:0)
      (opencl/opencl-skip! "distributed entry capture/restart"))
  (if @ze/gpu-available? (check-restart! :ze:0)
      (ze/gpu-skip! "distributed entry capture/restart")))
