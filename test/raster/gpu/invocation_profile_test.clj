(ns raster.gpu.invocation-profile-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.link-plan :as plan]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.invocation-observation :as observation]
            [raster.gpu.link :as link]
            [raster.gpu.test-lifecycle :as lifecycle]
            [raster.gpu.value :as value]))

(defn- fixture []
  (let [allocation (bview/allocation {:id :storage :byte-size 32 :memory-space :device
                                     :device :ze:0 :ownership :owned})
        view (bview/view allocation {:dtype :float :shape [4]})
        buffer {:id :storage :dtype :float :n-elements 8 :byte-size 32}
        executable (lifecycle/linked-executable
                    {:plan {:id :profile :target :ze:0
                            :nodes {:input (plan/node {:id :input :view view :role :input})}}
                     :session ::session :node-views {:input (gpu/->ResidentBufferView ::session :input view)}
                     :output-leases (atom 0) :pending-inputs (atom #{}) :tainted-inputs (atom #{})
                     :output-ready? (atom true) :completed-replays (atom 0)})
        artifact (compiled/map->Compiled
                  {:executable executable :target :ze:0 :donated {}
                   :in-tree [{:key :x :node :input :role :input :dtype :float :shape [4]}]
                   :out-tree [{:key :y :node :input}] :live-outputs (atom nil)})]
    {:artifact artifact :executable executable :buffer buffer :view view}))

(deftest profiled-invocation-is-one-ordinary-resident-invocation
  (let [{:keys [artifact buffer]} (fixture)
        actions (atom [])
        phases (atom [])
        record observation/record-phase!]
    (with-redefs [gpu/buffer (fn [& _] buffer)
                  gpu/upload-range! (fn [& _] (swap! actions conj :upload))
                  link/run! (fn [executable]
                              (link/with-unleased-execution!
                               executable :test-replay #(swap! actions conj :replay)))
                  observation/record-phase! (fn [phase ns]
                                              (swap! phases conj phase) (record phase ns))]
      (let [{:keys [outputs report]} (compiled/invoke-profiled artifact {:x (float-array 4)})]
        (is (= [:upload :replay] @actions))
        (is (value/device-array? (:y outputs)))
        (is (identical? buffer (:buffer (:y outputs))))
        (is (= 1 (count (filter #{:outer-lock-acquisition} @phases))))
        (is (= #{:outer-lock-acquisition :invocation-layout :aggregate-input-projection
                 :input-key-validation :donation-preflight :input-preflight :input-refresh
                 :replay-host :output-projection} (set (keys (:phases-ns report)))))
        (is (= {:host-upload {:operations 1 :logical-bytes 16}} (:input-refresh report)))
        (is (= :host-monotonic (:timing-source report)))
        (is (every? #(<= 0 %) (concat (vals (:phases-ns report))
                                    [(:total-ns report) (:unpartitioned-ns report)])))
        (reset! actions [])
        (reset! phases [])
        (let [normal (compiled/invoke-compiled artifact {:x (float-array 4)})]
          (is (identical? buffer (:buffer (:y normal))))
          (is (not (value/live? (:y outputs))))
          (is (= [:upload :replay] @actions))
          (is (empty? @phases))
          (is (nil? observation/*collector*)))))))

(deftest profiled-admission-preserves-output-and-rejects-before-writes
  (doseq [scenario [:lease :unknown-key :bad-shape]]
    (let [{:keys [artifact executable buffer view]} (fixture)
          previous (value/wrap-external-view buffer :ze:0 view)
          writes (atom 0)
          inputs (case scenario :unknown-key {:wrong (float-array 4)}
                       :bad-shape {:x (float-array 3)} {:x (float-array 4)})]
      (reset! (:live-outputs artifact) [previous])
      (when (= :lease scenario) (reset! (:output-leases executable) 1))
      (with-redefs [gpu/buffer (fn [& _] buffer)
                    gpu/upload-range! (fn [& _] (swap! writes inc))
                    link/run! (fn [& _] (throw (AssertionError. "decline must precede replay")))]
        (is (thrown? clojure.lang.ExceptionInfo (compiled/invoke-profiled artifact inputs))))
      (is (zero? @writes))
      (is (value/live? previous))
      (is (nil? observation/*collector*)))))

(deftest observation-preserves-original-failure-and-transfer-poisoning
  (let [{:keys [artifact executable buffer view]} (fixture)
        previous (value/wrap-external-view buffer :ze:0 view)
        marker (ex-info "transfer failed" {:marker true})]
    (reset! (:live-outputs artifact) [previous])
    (with-redefs [gpu/buffer (fn [& _] buffer)
                  gpu/upload-range! (fn [& _] (throw marker))]
      (is (identical? marker (try (compiled/invoke-profiled artifact {:x (float-array 4)})
                                 (catch Throwable error error)))))
    (is (= #{:input} @(:tainted-inputs executable)))
    (is (not (value/live? previous)))
    (is (nil? observation/*collector*))))

(deftest profiling-keeps-donation-before-mutation-but-after-complete-preflight
  (doseq [valid-input? [false true]]
    (let [{:keys [artifact buffer view]} (fixture)
          donated (value/wrap-external-view buffer :ze:0 view)
          artifact (assoc artifact :donated {:a :y}
                          :in-tree [{:key :a :node :input :role :state :dtype :float :shape [4]}
                                    {:key :x :node :input :role :input :dtype :float :shape [4]}])
          marker (ex-info "backend write" {})
          writes (atom 0)]
      (with-redefs [gpu/buffer (fn [& _] buffer)
                    gpu/upload-range! (fn [& _] (swap! writes inc) (throw marker))]
        (let [error (try (compiled/invoke-profiled
                         artifact {:a donated :x (float-array (if valid-input? 4 3))})
                         (catch Throwable error error))]
          (is (instance? clojure.lang.ExceptionInfo error))
          (is (= valid-input? (identical? marker error)))
          (is (= (not valid-input?) (value/live? donated)))
          (is (= (if valid-input? 1 0) @writes)))))))

(deftest refresh-routes-observe-actual-validated-branches
  (doseq [route [:exact-view-no-op :same-buffer-copy :foreign-buffer-copy]]
    (let [{:keys [executable buffer view]} (fixture)
          source-view (if (= :same-buffer-copy route) (assoc view :byte-offset 16) view)
          source-buffer (if (= :foreign-buffer-copy route) (assoc buffer :id :foreign) buffer)
          source (value/wrap-external-view source-buffer :ze:0 source-view)
          actions (atom [])]
      (with-redefs [gpu/buffer (fn [& _] buffer)
                    gpu/buffer-view (fn [session key attributes]
                                      (gpu/->ResidentBufferView session key (merge view attributes)))
                    gpu/copy-range! (fn [& _] (swap! actions conj :copy))
                    gpu/register-buffer! (fn [& _] (swap! actions conj :register))
                    gpu/free-buffer! (fn [& _] (swap! actions conj :detach))]
        (let [report (:report (observation/observe #(link/write! executable :input source)))]
          (is (= {route {:operations 1 :logical-bytes 16}} (:input-refresh report)))
          (is (= (case route :exact-view-no-op [] :same-buffer-copy [:copy]
                       :foreign-buffer-copy [:register :copy :detach]) @actions))))
      (is (value/live? source)))))

(deftest input-batch-preflight-rejects-cross-input-read-write-hazards
  (let [make-view #(bview/view
                    (bview/allocation {:id % :byte-size 16 :memory-space :device
                                       :device :ze:0 :ownership :owned})
                    {:dtype :float :shape [4]})
        a-view (make-view :a) b-view (make-view :b)
        a-buffer {:id :a :dtype :float :n-elements 4 :byte-size 16}
        b-buffer {:id :b :dtype :float :n-elements 4 :byte-size 16}
        a (value/wrap-external-view a-buffer :ze:0 a-view)
        b (value/wrap-external-view b-buffer :ze:0 b-view)
        executable (lifecycle/linked-executable
                    {:plan {:id :batch-ranges :target :ze:0
                            :nodes {:a (plan/node {:id :a :view a-view :role :input})
                                    :b (plan/node {:id :b :view b-view :role :input})}}
                     :session ::session
                     :node-views {:a (gpu/->ResidentBufferView ::session :a a-view)
                                  :b (gpu/->ResidentBufferView ::session :b b-view)}
                     :output-leases (atom 0) :pending-inputs (atom #{})
                     :tainted-inputs (atom #{}) :output-ready? (atom true)
                     :completed-replays (atom 0)})
        artifact (compiled/map->Compiled
                  {:executable executable :target :ze:0 :donated {}
                   :in-tree [{:key :a :node :a :role :input :default (float-array 4)}
                             {:key :b :node :b :role :input :default (float-array 4)}]
                   :out-tree [] :live-outputs (atom [a b])})
        writes (atom 0) replays (atom 0)]
    (with-redefs [gpu/buffer (fn [_ key] ({:a a-buffer :b b-buffer} key))
                  link/write! (fn [& _] (swap! writes inc))
                  link/run! (fn [& _] (swap! replays inc))]
      (doseq [inputs [{:a b :b a}
                      ;; A host upload can destroy the later resident read just as a D2D copy can.
                      {:a (float-array 4) :b a}]]
        (is (= :link-input-transfer-overlap
               (try (compiled/invoke-compiled artifact inputs) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
      (is (identical? executable (link/validate-write-batch! executable [[:a a] [:b b]]))
          "exact self-range no-ops do not invent a mutating destination")
      (is (= :link-input-transfer-duplicate
             (try (link/validate-write-batch! executable [[:a (float-array 4)] [:a a]]) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))
          "a repeated destination cannot hide a host-write/resident-read hazard")
      (is (zero? @writes))
      (is (zero? @replays))
      (is (every? value/live? [a b]))
      (is (true? @(:output-ready? executable)))
      (is (zero? (:value-epoch @(:execution-state executable)))))))
