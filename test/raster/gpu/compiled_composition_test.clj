(ns raster.gpu.compiled-composition-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.equation-artifact-store :as artifact-store]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.resident-plan :as resident-plan]
            [raster.compiler.pipeline :as pipeline]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.test-lifecycle]
            [raster.gpu.link :as gpu-link]
            [raster.gpu.parallel-program :as parallel-program]
            [raster.gpu.value :as value]))

(deftest static-invocation-layout-belongs-to-one-exact-artifact
  (let [tree [{:key :x :node :x :role :input}
              {:key :a :node :a :role :state}
              {:key :weight :node :weight :role :constant}]
        donated {:a :a-next}
        layout (#'compiled/derive-invocation-layout tree donated)
        artifact (#'compiled/seal-artifact
                  (compiled/map->Compiled {:in-tree tree :donated donated
                                           :invocation-layout layout}))
        original @#'compiled/derive-invocation-layout
        calls (atom 0)]
    (with-redefs-fn {#'compiled/derive-invocation-layout
                    (fn [& args] (swap! calls inc) (apply original args))}
      (fn []
        (is (identical? layout (#'compiled/invocation-layout-for artifact)))
        (is (zero? @calls))
        (let [changed (assoc-in artifact [:in-tree 0 :role] :constant)
              changed-layout (#'compiled/invocation-layout-for changed)]
          (is (= #{} (:input-keys changed-layout)))
          (is (= 1 @calls)))
        (let [changed (assoc artifact :donated {})
              changed-layout (#'compiled/invocation-layout-for changed)]
          (is (= #{} (:donated-keys changed-layout)))
          (is (= 2 @calls)))
        (is (= layout (#'compiled/invocation-layout-for
                       (compiled/map->Compiled (into {} artifact)))))
        (is (= 3 @calls))
        (is (= layout (#'compiled/invocation-layout-for
                       (compiled/map->Compiled {:in-tree tree :donated donated}))))
        (is (= 4 @calls))))))

(deftest caller-prepared-mutable-metadata-does-not-acquire-a-cached-layout
  (let [entry (java.util.HashMap. {:key :x :node :x :role :input})
        donated (java.util.HashMap.)
        prepared (compiled/map->Prepared {:lowering {:plan ::plan}
                                          :in-tree [entry] :donated donated})
        issued (with-redefs [gpu-link/instantiate! (fn [plan opts]
                                                    (is (= ::plan plan))
                                                    (is (= {} opts))
                                                    ::executable)]
                 (compiled/instantiate! prepared))]
    (is (#'compiled/sealed-artifact? issued))
    (is (nil? (:invocation-layout issued)))
    (is (= #{:x} (:input-keys (#'compiled/invocation-layout-for issued))))
    (is (= #{} (:donated-keys (#'compiled/invocation-layout-for issued))))
    (.put entry :role :constant)
    (.put donated :a :a-next)
    (is (= #{} (:input-keys (#'compiled/invocation-layout-for issued))))
    (is (= #{:a} (:donated-keys (#'compiled/invocation-layout-for issued))))))

(deftest invalid-later-donation-does-not-consume-earlier-adapter-or-write-inputs
  (let [buffer-a {:id :a :dtype :float :n-elements 4 :byte-size 16}
        buffer-b {:id :b :dtype :float :n-elements 4 :byte-size 16}
        foreign-b {:id :foreign :dtype :float :n-elements 4 :byte-size 16}
        view (fn [id]
               (bview/view
                (bview/allocation {:id id :byte-size 16 :memory-space :device
                                   :device :ze:0 :ownership :owned})
                {:dtype :float :shape [4]}))
        view-a (view :a)
        view-b (view :b)
        a (value/wrap-external-view buffer-a :ze:0 view-a)
        b (value/wrap-external-view foreign-b :ze:0 view-b)
        executable (raster.gpu.test-lifecycle/linked-executable
                    {:plan {:id :two-adapters :target :ze:0}
                     :session ::session
                     :node-views {:a (gpu/->ResidentBufferView ::session :a view-a)
                                  :b (gpu/->ResidentBufferView ::session :b view-b)}
                     :closed? (atom false) :lifetime-lock (Object.)
                     :output-leases (atom 0) :pending-inputs (atom #{})
                     :output-ready? (atom true) :execution-state (atom {:value-epoch 0}) :completed-replays (atom 0)})
        artifact (compiled/map->Compiled
                  {:executable executable :target :ze:0
                   :in-tree [{:key :A :node :a :role :state :dtype :float :shape [4]}
                             {:key :B :node :b :role :state :dtype :float :shape [4]}
                             {:key :x :node :x :role :input :dtype :float :shape [4]}]
                   :donated (array-map :A :A' :B :B')
                   :out-tree [] :live-outputs (atom nil)})
        writes (atom 0)
        replays (atom 0)]
    (with-redefs [gpu/buffer (fn [_ key] ({:a buffer-a :b buffer-b} key))
                  gpu-link/write! (fn [& _] (swap! writes inc))
                  gpu-link/run! (fn [& _] (swap! replays inc))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"not this artifact's resident view"
                            (compiled/invoke-compiled artifact
                                                      {:A a :B b :x (float-array 4)}))))
    (is (value/live? a))
    (is (value/live? b))
    (is (zero? @writes))
    (is (zero? @replays))
    (is (true? @(:output-ready? executable)))))

(deftest invalid-later-input-does-not-write-earlier-input-or-invalidate-output
  (let [view (bview/view
              (bview/allocation {:id :input :byte-size 16 :memory-space :device
                                 :device :ze:0 :ownership :owned})
              {:dtype :float :shape [4]})
        buffer {:id :input :dtype :float :n-elements 4 :byte-size 16}
        old-output (value/wrap-external-view buffer :ze:0 view)
        foreign-view (assoc-in view [:allocation :device] :cuda:0)
        foreign (value/wrap-external-view buffer :cuda:0 foreign-view)
        executable (raster.gpu.test-lifecycle/linked-executable
                    {:plan {:id :input-preflight :target :ze:0
                            :nodes {:a (link-plan/node {:id :a :view view :role :input})
                                    :b (link-plan/node {:id :b :view view :role :input})}}
                     :session ::session :closed? (atom false) :lifetime-lock (Object.)
                     :output-leases (atom 0) :pending-inputs (atom #{:a :b})
                     :output-ready? (atom true) :execution-state (atom {:value-epoch 0}) :completed-replays (atom 0)})
        artifact (compiled/map->Compiled
                  {:executable executable :target :ze:0 :donated {}
                   :in-tree [{:key :a :node :a :role :input :default (float-array 4)}
                             {:key :b :node :b :role :input :default (float-array 4)}]
                   :out-tree [] :live-outputs (atom [old-output])})
        writes (atom 0) replays (atom 0)]
    (with-redefs [gpu-link/write! (fn [& _] (swap! writes inc))
                  gpu-link/run! (fn [& _] (swap! replays inc))]
      (doseq [[source expected] [[(int-array 4) :link-initializer-dtype]
                                 [(float-array 3) :link-initializer-size]
                                 [foreign :link-device-input-target]]]
        (is (= expected
               (try (compiled/invoke-compiled artifact {:b source}) nil
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
    (is (zero? @writes))
    (is (zero? @replays))
    (is (value/live? old-output))
    (is (true? @(:output-ready? executable)))
    (is (= #{:a :b} @(:pending-inputs executable)))))

(deftest failed-input-transfer-retires-previous-output-aliases
  (let [view (bview/view
              (bview/allocation {:id :input :byte-size 16 :memory-space :device
                                 :device :ze:0 :ownership :owned})
              {:dtype :float :shape [4]})
        old-output (value/wrap-external-view
                    {:dtype :float :n-elements 4 :byte-size 16} :ze:0 view)
        executable (raster.gpu.test-lifecycle/linked-executable
                    {:plan {:id :failed-input :target :ze:0
                            :nodes {:a (link-plan/node {:id :a :view view :role :input})}}
                     :session ::session
                     :node-views {:a (gpu/->ResidentBufferView ::session :a view)}
                     :closed? (atom false) :lifetime-lock (Object.) :output-leases (atom 0)
                     :pending-inputs (atom #{}) :tainted-inputs (atom #{})
                     :output-ready? (atom true) :execution-state (atom {:value-epoch 0}) :completed-replays (atom 0)})
        artifact (compiled/map->Compiled
                  {:executable executable :target :ze:0 :donated {}
                   :in-tree [{:key :a :node :a :role :input :default (float-array 4)}]
                   :out-tree [] :live-outputs (atom [old-output])})
        failure (ex-info "partial upload" {})]
    (with-redefs [gpu/upload-range! (fn [& _]
                                    (is (not (value/live? old-output)))
                                    (throw failure))]
      (is (identical? failure
                      (try (compiled/invoke-compiled artifact {:a (float-array 4)})
                           (catch Throwable error error)))))
    (is (not (value/live? old-output)))
    (is (nil? @(:live-outputs artifact)))
    (is (= #{:a} @(:pending-inputs executable)))
    (is (= #{:a} @(:tainted-inputs executable)))
    (doseq [operation [compiled/profile compiled/measure]]
      (let [output (value/wrap-external-view
                    {:dtype :float :n-elements 4 :byte-size 16} :ze:0 view)]
        (reset! (:live-outputs artifact) [output])
        (with-redefs [gpu/upload-range! (fn [& _]
                                        (is (not (value/live? output)))
                                        (throw failure))]
          (is (identical? failure (try (operation artifact)
                                       (catch Throwable error error)))))
        (is (not (value/live? output)))
        (is (nil? @(:live-outputs artifact)))))
    (let [output (value/wrap-external-view
                  {:dtype :float :n-elements 4 :byte-size 16} :ze:0 view)
          executable (-> executable
                         (assoc-in [:plan :nodes :state]
                                   (link-plan/node {:id :state :view view :role :state}))
                         (assoc-in [:node-views :state] (gpu/->ResidentBufferView ::session :state view)))
          artifact (-> artifact
                       (assoc :executable executable :donated {:state :state-out})
                       (update :in-tree conj {:key :state :node :state :role :state
                                             :dtype :float :shape [4]}))]
      (reset! (:live-outputs artifact) [output])
      (with-redefs [gpu/buffer (fn [& _] (:buffer output))
                    gpu/upload-range! (fn [& _]
                                        (is (not (value/live? output)))
                                        (throw failure))]
        (is (identical? failure
                        (try (compiled/invoke-compiled artifact
                                                       {:state output :a (float-array 4)})
                             (catch Throwable error error)))))
      (is (not (value/live? output))))))

(deftest previous-output-can-feed-the-next-invocation-without-reviving-its-wrapper
  (doseq [[alias? donate?] [[false false] [true false] [false true]]]
   (let [view (bview/view
              (bview/allocation {:id :recurrent :byte-size 16 :memory-space :device
                                 :device :ze:0 :ownership :owned})
              {:dtype :float :shape [4]})
        buffer {:dtype :float :n-elements 4 :byte-size 16}
        output (value/wrap-external-view buffer :ze:0 view)
        source (if alias? (value/alias-of output) output)
        executable (raster.gpu.test-lifecycle/linked-executable
                    {:plan {:id :recurrent :target :ze:0
                            :nodes {:a (link-plan/node {:id :a :view view :role :input})}}
                     :session ::session :node-views {:a (gpu/->ResidentBufferView ::session :a view)}
                     :closed? (atom false) :lifetime-lock (Object.) :output-leases (atom 0)
                     :pending-inputs (atom #{}) :tainted-inputs (atom #{})
                     :output-ready? (atom true) :execution-state (atom {:value-epoch 0}) :completed-replays (atom 0)})
        artifact (compiled/map->Compiled
                  {:executable executable :target :ze:0 :donated (if donate? {:a :result} {})
                   :in-tree [{:key :a :node :a :role :input :dtype :float :shape [4]}]
                   :out-tree [] :live-outputs (atom [output])})
        replayed (atom false)]
    (with-redefs [gpu/buffer (fn [& _] buffer)
                  gpu/copy-range! (fn [& _] (throw (AssertionError. "self-write must not copy")))
                  gpu-link/run! (fn [& _]
                                  (is (not (value/live? output)))
                                  (reset! replayed true))]
      (is (= {} (compiled/invoke-compiled artifact {:a source}))))
    (is @replayed)
    (is (not (value/live? output)))
    (is (not (value/live? source))))))

(deftest profiling-and-measurement-preflight-leases-before-mutating-inputs-or-outputs
  (let [view (bview/view
              (bview/allocation {:id :output :byte-size 16 :memory-space :device
                                 :device :ze:0 :ownership :owned})
              {:dtype :float :shape [4]})
        output (value/wrap-external-view
                {:id :output :dtype :float :n-elements 4 :byte-size 16} :ze:0 view)
        executable (raster.gpu.test-lifecycle/linked-executable
                    {:profile? true :plan {:id :profile-lease :target :ze:0
                            :nodes {:input (link-plan/node {:id :input :dtype :float :shape [4]
                                                            :device :ze:0 :role :input})}}
                     :session ::session :closed? (atom false) :lifetime-lock (Object.)
                     :output-leases (atom 1) :pending-inputs (atom #{})
                     :execution-state (atom {:value-epoch 0})})
        artifact (compiled/map->Compiled
                  {:executable executable
                   :in-tree [{:node :input :role :input :default (float-array 4)}]
                   :out-tree [] :live-outputs (atom [output])})
        writes (atom 0)
        profiles (atom 0)
        measures (atom 0)
        reason (fn [f] (try (f) nil
                            (catch clojure.lang.ExceptionInfo error
                              (:reason (ex-data error)))))]
    (with-redefs [gpu-link/write! (fn [& _] (swap! writes inc))
                  gpu-link/profile! (fn [& _] (swap! profiles inc) {:profile []})
                  gpu-link/measure! (fn [& _] (swap! measures inc) {:samples []})]
      (is (= :link-output-lease-active (reason #(compiled/profile artifact))))
      (is (= :link-output-lease-active (reason #(compiled/measure artifact))))
      (is (zero? @writes))
      (is (zero? @profiles))
      (is (zero? @measures))
      (is (value/live? output))
      (reset! (:output-leases executable) 0)
      (is (= {:profile [] :result {}} (compiled/profile artifact)))
      (is (not (value/live? output)))
      (is (= {:samples []} (compiled/measure artifact)))
      (is (= 2 @writes))
      (is (= 1 @profiles))
      (is (= 1 @measures)))))

(deftest template-cache-guards-compare-live-roots-not-only-identity-hashes
  (let [first-root (Object.)
        second-root (Object.)
        token (fn [root]
                (compiled/->WeakIdentity
                 (java.lang.ref.WeakReference. root) 17))
        first (token first-root)
        same (token first-root)
        second (token second-root)]
    (is (= 17 (.hashCode first)))
    (is (= (hash first) (hash second)) "force an identity-hash collision")
    (is (= first same))
    (is (not= first second))
    (is (= :first (get {first :first} same)))
    (is (nil? (get {first :first} second)))))

(deftest profiling-request-errors-reject-before-refresh-or-output-retirement
  (doseq [[operation profiling? state? options expected-data]
          [[:profile false false {} {:reason :link-profiling-disabled}]
           [:measure false false {} {:reason :link-profiling-disabled}]
           [:measure true true {} {:reason :link-stateful-measurement}]
           [:measure true false {:before-sample! 42} {:option :before-sample!}]
           [:measure true false {:flush-fn 42} {:option :flush-fn}]
           [:measure true false {:budget-ms 0} {:budget-ms 0}]
           [:measure true false {:warmup-iterations -1} {:field :warmup-iterations}]
           [:measure true false {:min-samples 0} {:min-samples 0}]
           [:measure true false {:cv-threshold Double/NaN} {}]]]
    (let [view (bview/view (bview/allocation {:id :storage :byte-size 16 :memory-space :device
                                            :device :ze:0 :ownership :owned})
                           {:dtype :float :shape [4]})
          input (link-plan/node {:id :input :view view :role :input})
          buffer {:id :storage :dtype :float :n-elements 4 :byte-size 16}
          output (value/wrap-external-view buffer :ze:0 view)
          executable (raster.gpu.test-lifecycle/linked-executable
                       {:profile? profiling?
                        :plan {:id :profile-request :target :ze:0
                               :nodes (cond-> {:input input}
                                        state? (assoc :state (assoc input :id :state :role :state)))}
                        :session ::session :output-leases (atom 0)
                        :pending-inputs (atom #{:input}) :tainted-inputs (atom #{})})
          artifact (compiled/map->Compiled
                    {:executable executable :in-tree [{:key :x :node :input :role :input
                                                       :default (float-array 4)}]
                     :out-tree [] :live-outputs (atom [output])})
          calls (atom [])]
      (with-redefs [gpu-link/write! (fn [& _] (swap! calls conj :write))
                    gpu-link/profile! (fn [& _] (swap! calls conj :profile))
                    gpu-link/measure! (fn [& _] (swap! calls conj :measure))]
        (let [error (try (if (= :profile operation)
                          (compiled/profile artifact)
                          (apply compiled/measure artifact (mapcat identity options)))
                        (catch Throwable error error))]
          (is (instance? clojure.lang.ExceptionInfo error))
          (is (= expected-data (select-keys (ex-data error) (keys expected-data))))))
      (is (empty? @calls))
      (is (value/live? output))
      (is (= #{:input} @(:pending-inputs executable)))
      (is (= 0 (:value-epoch @(:execution-state executable)))))))

(deftest profiling-request-admission-does-not-require-inputs-before-refresh
  (let [view (bview/view (bview/allocation {:id :input :byte-size 16 :memory-space :device
                                          :device :ze:0 :ownership :owned})
                         {:dtype :float :shape [4]})
        executable (raster.gpu.test-lifecycle/linked-executable
                     {:profile? true :session ::session
                      :plan {:id :refresh :target :ze:0
                             :nodes {:input (link-plan/node {:id :input :view view :role :input})}}
                      :node-views {:input (gpu/->ResidentBufferView ::session :input view)}
                      :pending-inputs (atom #{:input}) :tainted-inputs (atom #{})
                      :output-ready? (atom false) :output-leases (atom 0)})
        artifact (compiled/map->Compiled
                  {:executable executable :in-tree [{:key :x :node :input :role :input
                                                     :default (float-array 4)}]
                   :out-tree [] :live-outputs (atom nil)})
        writes (atom 0)]
    (with-redefs [gpu/upload-range! (fn [& _] (swap! writes inc))
                  gpu-link/profile! (fn [& _]
                                      (is (empty? @(:pending-inputs executable)))
                                      {:profile []})]
      (is (= {:profile [] :result {}} (compiled/profile artifact)))
      (is (= 1 @writes)))))

(defn component [_x _w _n])

(deftest execution-info-observes-linked-binding-and-rejects-unavailable-evidence
  (let [info {:strategy :chosen :entry-points ["chosen_kernel"]}
        session (atom {:prepared {:phase {:execution-info info}}})
        live (raster.gpu.test-lifecycle/linked-executable
              {:session session :phases [:phase] :closed? (atom false)})
        compiled (compiled/map->Compiled {:executable live})]
    (is (= [{:phase :phase :executable info}] (compiled/execution-info compiled)))
    (is (= :compiled-execution-info-unbound
           (try (compiled/execution-info (compiled/map->Prepared {}))
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (let [prepared ((ns-resolve 'raster.gpu.parallel-program 'own-prepared)
                   (parallel-program/map->PreparedParallelProgram
                    {:binding-order [:fixed-phase]
                     :handles {:fixed-phase :bound-handle}
                     :closed? (atom false)}))
          fixed-info (assoc info :selection :fixed :admission [])
          artifact (assoc compiled :executable
                          (raster.gpu.test-lifecycle/linked-executable
                           (assoc live :prepared-program prepared)))]
      (with-redefs [gpu/kernel-graph-execution-info
                    (fn [actual-session handle]
                      (is (identical? session actual-session))
                      (is (= :bound-handle handle))
                      fixed-info)]
        (is (= [{:phase :fixed-phase :executable fixed-info}]
               (compiled/execution-info artifact))))
      (reset! (:closed? prepared) true)
      (is (= :parallel-program-closed
             (try (compiled/execution-info artifact)
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (reset! (:closed? live) true)
    (is (thrown? clojure.lang.ExceptionInfo (compiled/execution-info compiled)))))

(deftest compiled-artifact-projects-the-link-instantiation-report
  (let [report {:timing-source :host-monotonic :total-ns 42}
        live (raster.gpu.test-lifecycle/linked-executable {:instantiation-report report})
        artifact (compiled/map->Compiled {:executable live})]
    (is (= report (compiled/instantiation-report artifact)))
    (is (= :compiled-instantiation-report-type
           (:reason (ex-data (try (compiled/instantiation-report {})
                                  (catch clojure.lang.ExceptionInfo error error))))))))

(deftest preparation-report-is-available-before-and-after-instantiation
  (let [report {:kind :resident-descriptor :timing-source :host-monotonic}
        prepared (compiled/map->Prepared {:preparation-report report})
        artifact (compiled/map->Compiled {:preparation-report report})]
    (is (= report (compiled/preparation-report prepared)))
    (is (= report (compiled/preparation-report artifact)))
    (is (= :compiled-preparation-report-type
           (:reason (ex-data (try (compiled/preparation-report {})
                                  (catch clojure.lang.ExceptionInfo error error))))))))

(def ^:private kernel
  (artifact/make
   {:kernel-name "compiled_composition_axpy"
    :source "__kernel void compiled_composition_axpy(float* x, float* w, float* y, long n) {}"
    :abi [(kabi/slot 'x :input :float)
          (kabi/slot 'w :input :float)
          (kabi/slot 'y :output :float)
          (kabi/slot 'n :scalar :long)]
    :arguments '[x w y n]
    :launch (launch/spec {:workgroup-size [64]
                          :group-count [(launch/ceil-div 'n 64)]})
    :effects {:kind :map :reads '[x w] :writes '[y]}}))

(defn- descriptor []
  {:dtype :float
   :all-params '[x w n]
   :array-params '[x w]
   :scalar-params '[n]
   :array-roles {'x :input 'w :input}
   :allocs [{:sym 'y :dtype :float :size-fn (fn [args] (long (nth args 2)))}]
   :steps [{:phase :map :kernel-name "compiled_composition_axpy" :convention :map
            :artifact kernel
            :argument-specs [{:kind :input :sym 'x}
                             {:kind :input :sym 'w}
                             {:kind :output :sym 'y}
                             {:kind :scalar :type :long
                              :value-fn (fn [args] (long (nth args 2)))}]}]
   :result-sym 'y})

(deftest semantic-artifacts-compose-before-one-runtime-instantiation
  (let [weight (float-array 16)
        prepare #(compiled/lower #'component [(float-array 16) weight 16]
                                 {:target :ze:0 :constants '[w]})
        [first second]
        (with-redefs [pipeline/compile-gpu-program (fn [& _] (descriptor))
                      gpu-link/instantiate! (fn [& _]
                                              (throw (AssertionError.
                                                      "lower must not contact the runtime")))]
          [(prepare) (prepare)])
        composite
        (compiled/compose
         {:id :semantic-two-layers
          :components [{:id :first :program first}
                       {:id :second :program second}]
          :connections [{:from [:first :y] :to [:second :x]}]
          :shares [[[:first :w] [:second :w]]]
          :outputs [{:key :result :from [:second :y]}]})
        plan (compiled/plan composite)
        first-y (get-in first [:out-tree 0 :node])
        second-x (get-in second [:in-tree 0 :node])
        mapping (get-in composite [:lowering :certificate :node-mapping])]
    (is (compiled/prepared? first))
    (is (compiled/prepared? composite))
    (is (= (get mapping [:first first-y]) (get mapping [:second second-x])))
    (is (= 4 (count (:nodes plan))))
    (is (= [[:first :x] [:first :w]] (mapv :key (:in-tree composite))))
    (is (= [:result] (mapv :key (:out-tree composite))))
    (is (= 2 (count (:instances plan))))
    (is (= :composition (:kind (compiled/preparation-report composite))))
    (is (= [:first :second]
           (mapv :id (:components (compiled/preparation-report composite)))))
    (is (pos? (:total-ns (compiled/preparation-report composite))))
    (is (= 2 (count (compiled/ir composite))))
    (is (= {:map 2} (:steps (compiled/cache-key composite))))))

(deftest certified-instantiation-requires-an-exact-validated-plan-and-evidence
  (let [prepared (with-redefs [pipeline/compile-gpu-program (fn [& _] (descriptor))]
                   (compiled/lower #'component [(float-array 16) (float-array 16) 16]
                                   {:target :ze:0}))
        copied (assoc prepared :preparation-report {:copied true})
        lowering (:lowering prepared)
        rebound-evidence (get-in lowering [:certificate :effect-evidence])
        validated (link-plan/validate-with-effect-evidence! (:plan lowering))
        exact (assoc lowering :plan (:plan validated)
                     :certificate (assoc (:certificate lowering)
                                         :effect-evidence (:effect-evidence validated)))
        evidence (get-in exact [:certificate :effect-evidence])
        forged-plan (assoc exact :plan (assoc (:plan exact) :outputs []))
        forged-evidence (assoc-in exact [:certificate :effect-evidence]
                                  (assoc evidence :step-facts []))]
    (is (not (link-plan/retained-effect-evidence? (:plan lowering) rebound-evidence))
        "rebinding host sources changes the exact plan object")
    (is (link-plan/retained-effect-evidence? (:plan exact) evidence))
    (is (not (link-plan/retained-effect-evidence? nil evidence)))
    (is (not (link-plan/retained-effect-evidence? (:plan forged-plan) evidence)))
    (is (not (link-plan/retained-effect-evidence? (:plan exact)
                                                  (get-in forged-evidence
                                                          [:certificate :effect-evidence]))))
    (with-redefs [link-plan/validate-with-effect-evidence!
                  (fn [_] (throw (ex-info "raw plan validation reached" {})))
                  gpu/make-session
                  (fn [_] (throw (ex-info "session setup reached" {})))]
      (is (= "raw plan validation reached"
             (try (compiled/instantiate! prepared)
                  (catch clojure.lang.ExceptionInfo error (.getMessage error)))))
      (is (= "session setup reached"
             (try (gpu-link/instantiate-certified! exact {})
                  (catch clojure.lang.ExceptionInfo error (.getMessage error)))))
      (is (= "raw plan validation reached"
             (try (compiled/instantiate! copied)
                  (catch clojure.lang.ExceptionInfo error (.getMessage error)))))
      (is (= "raw plan validation reached"
             (try (compiled/instantiate!
                   (assoc prepared :provenance-seal (constantly true)))
                  (catch clojure.lang.ExceptionInfo error (.getMessage error)))))
      (doseq [invalid [forged-plan forged-evidence]]
        (is (= :link-certified-effect-evidence
               (try (gpu-link/instantiate-certified! invalid {})
                    (catch clojure.lang.ExceptionInfo error
                      (:reason (ex-data error))))))))))

(deftest exact-prepared-values-compose-without-rederiving-component-certificates
  (let [weight (float-array 16)
        prepare #(compiled/lower #'component [(float-array 16) weight 16]
                                 {:target :ze:0 :constants '[w]})
        [first second] (with-redefs [pipeline/compile-gpu-program (fn [& _] (descriptor))]
                         [(prepare) (prepare)])
        request {:id :sealed-components
                 :components [{:id :first :program first}
                              {:id :second :program second}]
                 :connections [{:from [:first :y] :to [:second :x]}]
                 :shares [[[:first :w] [:second :w]]]
                 :outputs [{:key :result :from [:second :y]}]}
        facts-var (ns-resolve 'raster.compiler.ir.link-plan 'instance-access-facts)]
    (with-redefs-fn
      {#'resident-plan/verify!
       (fn [_] (throw (AssertionError. "fresh Prepared was redundantly verified")))
       facts-var
       (fn [_] (throw (AssertionError. "certified component ABIs were reparsed")))}
      #(is (compiled/prepared? (compiled/compose request))))
    (let [copied (assoc first :preparation-report {:copied true})
          calls (atom 0)
          original resident-plan/verify!]
      (with-redefs [resident-plan/verify! (fn [lowering]
                                            (swap! calls inc)
                                            (original lowering))]
        (is (compiled/prepared?
             (compiled/compose
              (assoc-in request [:components 0 :program] copied))))
        (is (= 2 @calls)
            "copying one sealed component makes the whole public composition verify independently")))))

(deftest descriptor-cache-uses-the-shared-schedule-alias-normalization
  (compiled/clear-compilation-cache!)
  (try
    (let [calls (atom []) args [(float-array 16) (float-array 16) 16]
          base {:target :ze:0 :constants '[w]}]
      (with-redefs [pipeline/compile-gpu-program
                    (fn [& arguments] (swap! calls conj arguments) (descriptor))]
        (doseq [schedule [{:gemm-precision :f32-scalar} {:precision :f32-scalar}]]
          (compiled/lower #'component args (assoc base :schedule schedule)))
        (is (= 1 (count @calls)))
        (is (= {:precision :f32-scalar}
               (:schedule (apply hash-map (drop 2 (first @calls))))))
        (compiled/lower #'component args (assoc base :schedule {:precision :mixed-f16-f32}))
        (is (= 2 (count @calls)))
        (is (thrown? clojure.lang.ExceptionInfo
                     (compiled/lower #'component args
                                     (assoc base :schedule {:gemm-precision :mixed-f16-f32
                                                            :precision :f32-scalar}))))
        (is (= 2 (count @calls)))))
    (finally (compiled/clear-compilation-cache!))))

(deftest repeated-lowerings-share-only-the-immutable-compilation-template
  (compiled/clear-compilation-cache!)
  (try
    (let [compilations (atom 0)
          first-input (float-array 16)
          second-input (float-array 32)
          first-weight (float-array 16)
          second-weight (float-array 32)]
      (with-redefs [pipeline/compile-gpu-program
                    (fn [& _] (swap! compilations inc) (descriptor))]
        (let [first (compiled/lower #'component [first-input first-weight 16]
                                    {:target :ze:0 :constants '[w]})
              second (compiled/lower #'component [second-input second-weight 32]
                                     {:target :ze:0 :constants '[w]
                                      :gemm-precision :mixed-f16-f32})
              stats (compiled/compilation-cache-stats)]
          (is (= 1 @compilations)
              "symbolic compilation is shared across concrete sizes and buffer identities")
          (is (= {:hits 1 :misses 1 :misses-by-reason {:compulsory 1}
                  :compilations 1 :failures 0
                  :entries 1 :entries-by-compiler {:resident-descriptor 1}}
                 (dissoc stats :compile-nanos)))
          (is (= first-input (get-in first [:in-tree 0 :default])))
          (is (= second-input (get-in second [:in-tree 0 :default])))
          (is (false? (get-in (compiled/preparation-report first)
                              [:template :cache-hit?])))
          (is (= :compulsory (get-in (compiled/preparation-report first)
                                     [:template :miss-reason])))
          (is (string? (get-in (compiled/preparation-report first)
                               [:template :semantic-fingerprint])))
          (is (false? (get-in (compiled/preparation-report first)
                              [:template :persistent-cache-eligible?]))
              "an ordinary Var without retained deftm source is process-cache-only")
          (is (contains? (get-in (compiled/preparation-report first)
                                 [:template :persistence-blockers])
                         :retained-deftm-source))
          (is (true? (get-in (compiled/preparation-report second)
                             [:template :cache-hit?])))
          (is (nil? (get-in (compiled/preparation-report second)
                            [:template :miss-reason])))
          (is (= :compulsory
                 (get-in (compiled/preparation-report first)
                         [:resident-plan-template :miss-reason])))
          (is (= :specialization
                 (get-in (compiled/preparation-report second)
                         [:resident-plan-template :miss-reason]))
              "a new bound shape is not a failed compiler-template reuse")
          (is (every? #(and (integer? %) (not (neg? %)))
                      ((juxt :total-ns :link-plan-lowering-ns)
                       (compiled/preparation-report second))))
          (is (not= (:lowering first) (:lowering second))
              "argument-dependent LinkPlan certification remains per invocation"))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest repeated-shapes-rebind-a-source-free-resident-plan-template
  (compiled/clear-compilation-cache!)
  (try
    (let [first-x (float-array 16)
          first-w (float-array 16)
          second-x (float-array 16)
          second-w (float-array 16)]
      (with-redefs [pipeline/compile-gpu-program (fn [& _] (descriptor))]
        (let [first (compiled/lower #'component [first-x first-w 16] {:target :ze:0})
              second (compiled/lower #'component [second-x second-w 16] {:target :ze:0})]
          (is (false? (get-in first [:preparation-report :resident-plan-template :cache-hit?])))
          (is (true? (get-in second [:preparation-report :resident-plan-template :cache-hit?])))
          (is (identical? second-x (get-in second [:lowering :plan :nodes
                                                   (get-in second [:lowering :certificate
                                                                   :bindings 'x]) :source])))
          (is (identical? second-w (get-in second [:lowering :plan :nodes
                                                   (get-in second [:lowering :certificate
                                                                   :bindings 'w]) :source])))
          (is (every? (fn [entry]
                        (let [template @entry]
                          (and (every? nil? (map :source (vals (get-in template [:plan :nodes]))))
                               (= [] (get-in template [:plan :instances 0 :arguments])))))
                      (vals @(var-get #'compiled/resident-plan-template-cache)))
              "the cache retains structure but no caller arrays"))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest resident-plan-key-preserves-scalar-bits
  (let [descriptor (assoc (descriptor) :all-params '[x w scale n]
                           :scalar-params '[scale n])
        x (float-array 8) w (float-array 8)
        key-for (fn [scale]
                  (#'compiled/compilation-id #'component :ze:0 :float descriptor
                                             [x w scale 8] false))
        nan-a (Float/intBitsToFloat 0x7fc00001)
        same-nan-a (Float/intBitsToFloat 0x7fc00001)
        nan-b (Float/intBitsToFloat 0x7fc00002)]
    (is (not= (key-for (float 0.0)) (key-for (float -0.0))))
    (is (= (key-for nan-a) (key-for same-nan-a)))
    (is (not= (key-for nan-a) (key-for nan-b)))
    (is (not= (key-for (float 0.0)) (key-for (double 0.0))))
    (is (not= (key-for 0.0) (key-for -0.0)))
    (is (= (key-for (Double/longBitsToDouble 0x7ff8000000000001))
           (key-for (Double/longBitsToDouble 0x7ff8000000000001))))
    (is (not= (key-for (Double/longBitsToDouble 0x7ff8000000000001))
              (key-for (Double/longBitsToDouble 0x7ff8000000000002))))))

(deftest template-validation-evidence-requires-independent-math-request
  (compiled/clear-compilation-cache!)
  (try
    (let [request {:scalar-math {:overrides {[:tanh :float] :f64-target-library-rte-f32}}}
          emitted (program/make {:dialect :opencl-parallel})
          compilation (equation-first/map->EquationFirstCompilation {:emitted emitted})
          owner (atom nil)
          key {:guards {:compiler-revision (dispatch/compiler-definition-revision)
                        :pipeline-identity (#'compiled/weak-identity @#'equation-first/compile)}}
          calls (atom 0)
          compile! (fn [] (swap! calls inc) compilation)]
      (binding [compiled/*compilation-template-owner* owner]
        (is (identical? compilation
                        (#'compiled/stable-compilation-template
                         (constantly key) :equation-first compile! request))))
      (is (nil? (#'compiled/owned-emitted-validation @owner compilation)))
      (is (not (realized? (get-in @owner [:entry :emitted-validation])))
          "a foreign request must not force the cached proof")
      (let [proof (#'compiled/owned-emitted-validation @owner compilation request)]
        (is (emitted-program/retained-validation? emitted proof request))
        (is (not (emitted-program/retained-validation? emitted proof)))
        (is (identical? proof (#'compiled/owned-emitted-validation @owner compilation request)))
        (is (nil? (#'compiled/owned-emitted-validation @owner (assoc compilation :id :other) request)))
        (is (nil? (#'compiled/owned-emitted-validation @owner compilation))))
      (binding [compiled/*compilation-template-owner* owner]
        (is (identical? compilation
                        (#'compiled/cached-compilation-template key :equation-first compile!))))
      (is (= 1 @calls) "compilation remains single-flight even when proof consent differs")
      (is (nil? (#'compiled/owned-emitted-validation @owner compilation)))
      (is (thrown? clojure.lang.ExceptionInfo
                   (#'compiled/cached-compilation-template :invalid :equation-first compile!
                                                          {:scalar-math {:unknown true}})))
      (is (= 1 @calls) "invalid intent fails before invoking compilation")
      (let [default-owner (atom nil)]
        (binding [compiled/*compilation-template-owner* default-owner]
          (#'compiled/cached-compilation-template (assoc key :kind :default)
                                                  :equation-first compile!))
        (is (nil? (#'compiled/owned-emitted-validation @default-owner compilation request)))
        (is (not (realized? (get-in @default-owner [:entry :emitted-validation]))))
        (let [proof (#'compiled/owned-emitted-validation @default-owner compilation)]
          (is (emitted-program/retained-validation? emitted proof))
          (is (not (emitted-program/retained-validation? emitted proof request)))
          (is (identical? proof (#'compiled/owned-emitted-validation @default-owner compilation))))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest template-callbacks-and-delayed-proofs-do-not-inherit-compiler-scopes
  (compiled/clear-compilation-cache!)
  (try
    (let [scope-vars (mapv (fn [[namespace symbol]] (ns-resolve namespace symbol))
                           [['raster.compiler.ir.link-plan '*validated-program-instances*]
                            ['raster.compiler.ir.link-plan '*retained-program-validations*]
                            ['raster.compiler.ir.link-plan '*caller-options*]
                            ['raster.compiler.ir.emitted-parallel-program-call '*validated-boundary-projections*]
                            ['raster.compiler.ir.emitted-parallel-program-call '*validated-projection-policy*]])
          observed (atom [])
          observe (fn [stage]
                    (swap! observed conj [stage (mapv var-get scope-vars)])
                    (swap! observed conj [stage @(future (mapv var-get scope-vars))]))
          emitted (program/make {:dialect :opencl-parallel})
          compilation (equation-first/map->EquationFirstCompilation {:emitted emitted})
          owner (atom nil)
          request {:scalar-math {:overrides {[:tanh :float] :f64-target-library-rte-f32}}}
          key {:persistent-cache-eligible? true :semantic-fingerprint "scope-test"
               :guards {:compiler-revision (dispatch/compiler-definition-revision)
                        :pipeline-identity (#'compiled/weak-identity @#'equation-first/compile)}}
          validator @#'emitted-program/validate-with-physical-results!]
      (is (every? var? scope-vars))
      (with-redefs [artifact-store/load-artifact
                    (fn [& _] (observe :load) {:status :miss})
                    artifact-store/store-artifact!
                    (fn [& _] (observe :store) {:status :stored})
                    emitted-program/validate-with-physical-results!
                    (fn [& arguments] (observe :validator) (apply validator arguments))]
        (with-bindings (zipmap scope-vars (repeat (Object.)))
          (binding [compiled/*compilation-template-owner* owner
                    compiled/*compilation-template-observer* (fn [_] (observe :observer))]
            (is (identical? compilation
                            (#'compiled/stable-compilation-template
                             (fn [_] (observe :key) key) :equation-first
                             (fn [] (observe :compile) compilation) request))))
          ;; Force the delay directly: its execution context must not depend on its forcing thread.
          (let [proof (:validation @(get-in @owner [:entry :emitted-validation]))]
            (is (emitted-program/retained-validation? emitted proof request))
            (is (identical? proof
                            (#'compiled/owned-emitted-validation @owner compilation request))))))
      (is (= #{:key :load :compile :store :observer :validator} (set (map first @observed))))
      (is (every? #(= [nil nil nil nil nil] (second %)) @observed) (pr-str @observed)))
    (finally (compiled/clear-compilation-cache!))))

(deftest structural-compilation-cache-is-single-flight
  (compiled/clear-compilation-cache!)
  (try
    (let [started (promise)
          release (promise)
          calls (atom 0)
          compile! (fn []
                     (swap! calls inc)
                     (deliver started true)
                     @release
                     :template)
          first-result (future (#'compiled/cached-compilation-template
                                :same :resident-descriptor compile!))]
      @started
      (let [second-result (future (#'compiled/cached-compilation-template
                                   :same :resident-descriptor compile!))]
        (deliver release true)
        (is (= [:template :template] [@first-result @second-result]))
        (is (= 1 @calls))
        (is (= {:hits 1 :misses 1 :misses-by-reason {:compulsory 1}
                :compilations 1 :failures 0
                :entries 1 :entries-by-compiler {:resident-descriptor 1}}
               (dissoc (compiled/compilation-cache-stats) :compile-nanos)))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest structural-compilation-cache-converges-after-nested-specialization
  (compiled/clear-compilation-cache!)
  (try
    (let [calls (atom 0)
          first? (atom true)
          first-report (atom nil)
          key-for-revision (fn [revision] [:nested-specialization revision])
          compile! (fn []
                     (swap! calls inc)
                     (when (compare-and-set! first? true false)
                       (dispatch/bump-compiler-definition-revision!))
                     :template)
          first-value
          (binding [compiled/*compilation-template-observer* #(reset! first-report %)]
            (#'compiled/stable-compilation-template
             key-for-revision :resident-descriptor compile!))
          second-report (atom nil)
          second-value
          (binding [compiled/*compilation-template-observer* #(reset! second-report %)]
            (#'compiled/stable-compilation-template
             key-for-revision :resident-descriptor compile!))]
      (is (= [:template :template] [first-value second-value]))
      (is (= 2 @calls)
          "the first request recompiles under the installed specialization epoch; the next hits")
      (is (= 2 (:stabilization-attempts @first-report)))
      (is (false? (:cache-hit? @first-report)))
      (is (= 1 (:stabilization-attempts @second-report)))
      (is (true? (:cache-hit? @second-report)))
      (is (= {:hits 1 :misses 2 :misses-by-reason {:compulsory 2}
              :compilations 2 :failures 0
              :entries 1 :entries-by-compiler {:resident-descriptor 1}}
             (dissoc (compiled/compilation-cache-stats) :compile-nanos))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest structural-compilation-cache-fails-loud-on-an-unstable-epoch
  (compiled/clear-compilation-cache!)
  (try
    (let [calls (atom 0)
          error (try
                  (#'compiled/stable-compilation-template
                   (fn [revision] [:unstable revision])
                   :resident-descriptor
                   (fn []
                     (swap! calls inc)
                     (dispatch/bump-compiler-definition-revision!)
                     :never-stable))
                  (catch clojure.lang.ExceptionInfo error error))]
      (is (= :compilation-template-unstable (:reason (ex-data error))))
      (is (= 8 @calls))
      (is (zero? (:entries (compiled/compilation-cache-stats)))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest stable-template-preserves-failure-observation-and-does-not-cache-errors
  (compiled/clear-compilation-cache!)
  (try
    (let [report (atom nil)
          error (binding [compiled/*compilation-template-observer* #(reset! report %)]
                  (try
                    (#'compiled/stable-compilation-template
                     (fn [revision] [:failure revision])
                     :resident-descriptor
                     #(throw (ex-info "compile failed" {:reason :expected-failure})))
                    (catch clojure.lang.ExceptionInfo error error)))]
      (is (= :expected-failure (:reason (ex-data error))))
      (is (false? (:success? @report)))
      (is (zero? (:entries (compiled/compilation-cache-stats))))
      (is (= 1 (:failures (compiled/compilation-cache-stats)))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest compiler-visible-redefinition-invalidates-structural-templates
  (compiled/clear-compilation-cache!)
  (try
    (let [compilations (atom 0)
          arguments [(float-array 16) (float-array 16) 16]]
      (with-redefs [pipeline/compile-gpu-program
                    (fn [& _] (swap! compilations inc) (descriptor))]
        (compiled/lower #'component arguments {:target :ze:0})
        (dispatch/bump-compiler-definition-revision!)
        (let [second (compiled/lower #'component arguments {:target :ze:0})]
          (is (= 2 @compilations)
              "a changed inlined callee cannot reuse a pre-redefinition template")
          (is (= :invalidation
                 (get-in (compiled/preparation-report second) [:template :miss-reason])))
          (is (= 2 (:entries (compiled/compilation-cache-stats)))))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest already-instantiated-artifacts-are-too-late-to-compose-zero-copy
  (is (= :compiled-composition-component
         (:reason
          (ex-data
           (try
             (compiled/compose
              {:id :late :components [{:id :late :program (compiled/map->Compiled {})}]
               :outputs []})
             (catch clojure.lang.ExceptionInfo error error)))))))
