(ns raster.gpu.ordered-preparation-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.ir.invocation-source-order-test :as fixture]
            [raster.compiler.ir.ordered-invocation-realization-test :as allocation-fixture]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.value :as value]
            [raster.gpu.ordered-invocation-root-test :as root-fixture]
            [raster.gpu.program-memory-order-test :as dense-fixture]
            [raster.gpu.device-probe :as ocl-probe]
            [raster.dl.gpu-grad-parity :as ze-probe]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest ordered-scalar-preparation-requires-the-inner-resident-output-certificate
  (let [compilation (equation-first/compile #'fixture/write-before-resident-reduction
                                           {:target :ocl:0 :dtype :double})
        output (first (get-in compilation [:semantic :outputs]))
        expected (get-in compilation [:semantic :values output])
        producer-index (first (keep-indexed
                              (fn [index equation]
                                (when (some #{output} (:results equation)) index))
                              (get-in compilation [:scheduled :equations])))
        proofless (update-in compilation [:scheduled :equations producer-index :algorithm]
                             (fn [algorithm]
                               (soac/make (update (soac/facts algorithm) :attributes
                                                  dissoc :public-resident-reduction-outputs)
                                          (soac/equations algorithm) (soac/outputs algorithm))))
        forbidden (fn [& _] (throw (AssertionError. "preparation acquired device resources")))]
    (is (= expected (equation-first/validated-resident-output compilation output)))
    (doseq [damaged (concat
                    [proofless (assoc-in compilation [:emitted :outputs] [])
                     (assoc-in compilation [:emitted :values output :dtype] :float)]
                    (for [[path replacement] [[[:representation] {:kind :plain}]
                                              [[:representation :elements] 2]
                                              [[:memory-space] :host]
                                              [[:shape] [1]]
                                              [[:dtype] :float]]]
                      (assoc-in compilation (into [:scheduled :values output] path) replacement)))]
      (is (nil? (equation-first/validated-resident-output damaged output))
          "outer rank/representation/dtype hints cannot replace the exact inner certificate"))
    (with-redefs [gpu/make-session forbidden gpu/alloc! forbidden link/instantiate! forbidden]
      (doseq [n [0 1 Long/MAX_VALUE]]
        (let [prepared (#'compiled/prepare-ordered-equation-first
                        compilation [(double-array [7.0]) (double-array 1) n] {} nil)]
          (is (= :ordered-equation-first (:kind (compiled/preparation-report prepared))))
          (is (= [] (get-in prepared [:out-tree 0 :abstract :shape])))))
      (with-redefs [equation-first/validated-resident-output (fn [& _] nil)]
        (is (= :compiled-ordered-output-kind
               (reason #(#'compiled/prepare-ordered-equation-first
                          compilation [(double-array [7.0]) (double-array 1) 1] {} nil))))))))

(deftest ordered-resident-scalar-constructor-keeps-its-logical-view-and-exact-footprint
  (let [compilation (equation-first/compile #'fixture/write-before-resident-reduction
                                           {:target :ocl:0 :dtype :double})
        initial (invocation-link/begin-ordered-realization
                 compilation [(double-array [7.0]) (double-array 1) 1] nil)
        [position action] (first (filter #(= :result-allocation (:kind (val %)))
                                         (:source-realizations initial)))
        ;; Pure constructor-step test: selecting the source cursor does not assert earlier
        ;; device execution or supply acquired/initialized storage to the runtime.
        state (assoc initial :source-cursor position)
        evaluated (atom 0)
        evaluate (fn [expression _]
                   (swap! evaluated inc)
                   (is (= 1 expression))
                   {:source-value 1 :elements 1})
        prepared (invocation-link/prepare-source-allocation state evaluate nil)]
    (is (= 1 @evaluated))
    (is (= 1 (:elements prepared)))
    (is (= [] (get-in prepared [:nodes 0 :view :shape])))
    (is (empty? (get-in prepared [:state :acquired-storage])))
    (doseq [expression ['(clojure.core/double-array 0)
                        '(clojure.core/double-array 2)
                        '(clojure.core/float-array 1)]]
      (reset! evaluated 0)
      (is (= :invocation-ordered-next-allocation
             (reason #(invocation-link/prepare-source-allocation
                        (assoc-in state [:source-realizations position :expression] expression)
                        evaluate nil))))
      (is (zero? @evaluated) "wrong constructor contracts refuse before source count evaluation"))
    (doseq [elements [0 2]]
      (is (= :invocation-ordered-allocation-shape
             (reason #(invocation-link/prepare-source-allocation
                        state (fn [& _] {:source-value elements :elements elements}) nil)))))
    (is (= :invocation-ordered-realization-storage
           (reason #(invocation-link/prepare-source-allocation
                      (assoc-in state [:source-realizations position :binding] 'not-the-result)
                      evaluate nil))))
    (is (= (:realized initial) (:realized state)))
    (is (= 'sum (:binding action)))))

(defn- check-ordered-resident-scalar! [target]
  (doseq [n [0 1]]
    (let [prepared (compiled/lower #'fixture/write-before-resident-reduction
                                   [(double-array [7.0]) (double-array 1) n]
                                   {:compiler :equation-first :target target :dtype :double})
          artifact (compiled/instantiate! prepared)
          check-consumer! (fn [expected]
                            (let [executable (:executable artifact)
                                  state @(:execution-state executable)
                                  token (get-in state [:ordered :current :realized :buffers 'out])
                                  view (get-in state [:ordered :node-views token])
                                  actual (double-array 1)]
                              (is (some? view))
                              (when view
                                (gpu/download-range! (:session executable) view actual {:elements 1})
                                (is (= [expected] (vec actual))
                                    "the later map consumes the same resident scalar"))))]
      (try
        (let [first (:result (artifact))]
          (is (= [] (:shape first)))
          (is (= [(if (zero? n) 0.0 7.0)] (vec (value/->host first))))
          (check-consumer! (if (zero? n) 8.0 15.0))
          (is (= :invocation-materialization-buffer-dtype
                 (reason #(artifact {:x (float-array 1)}))))
          (is (value/live? first) "failed preflight keeps the existing scalar output alive")
          (let [second (:result (artifact {:x (double-array [11.0])}))]
            (is (= [(if (zero? n) 0.0 11.0)] (vec (value/->host second))))
            (check-consumer! (if (zero? n) 12.0 23.0))
            (is (not (value/live? first)))))
        (is (= [(if (zero? n) 0.0 7.0)] (vec (value/->host (:result (artifact)))))
            "omitted replacement uses the original source, not the preceding replay")
        (is (= 3 @(:completed-replays (:executable artifact))))
        (finally (compiled/close! artifact)))))
  (let [prepared (compiled/lower #'fixture/write-before-resident-reduction
                                 [(double-array [7.0]) (double-array 1) Long/MAX_VALUE]
                                 {:compiler :equation-first :target target :dtype :double})
        artifact (compiled/instantiate! prepared)]
    (try
      (is (instance? ArithmeticException (try (artifact) nil (catch Throwable error error))))
      (let [executable (:executable artifact)
            state @(:execution-state executable)
            token (get-in state [:ordered :current :realized :buffers 'out])
            view (get-in state [:ordered :node-views token])
            actual (double-array 1)]
        (is (some? view))
        (when view
          (gpu/download-range! (:session executable) view actual {:elements 1})
          (is (= [8.0] (vec actual)) "only the earlier write precedes the checked failure"))
        (is (= 2 (count (get-in state [:ordered :roots]))) "the scalar allocation has not run"))
      (is (nil? @(:live-outputs artifact)))
      (is (zero? @(:completed-replays (:executable artifact))))
      (finally (compiled/close! artifact)))))

(deftest opencl-ordered-resident-scalar-replay-and-failure-order
  (if @ocl-probe/opencl-fp64-available?
    (check-ordered-resident-scalar! :ocl:0)
    (ocl-probe/opencl-skip! "ordered certified resident scalar output")))

(deftest level-zero-ordered-resident-scalar-replay-and-failure-order
  (if @ze-probe/gpu-available?
    (check-ordered-resident-scalar! :ze:0)
    (ze-probe/gpu-skip! "ordered certified resident scalar output")))

(deftest public-preparation-selects-order-before-evaluating-source-work
  (let [arguments [(double-array [7.0]) (double-array 1) Long/MAX_VALUE]
        forbidden (fn [& _] (throw (Exception. "preparation evaluated source work or touched a driver")))]
    (with-redefs [scalar/evaluate-invocation-step forbidden scalar/evaluate-host-equation forbidden
                  gpu/make-session forbidden equation-first/lower forbidden]
      (let [prepared (compiled/lower #'fixture/write-before-shape arguments
                                     {:compiler :equation-first :target :ocl:0 :dtype :double})]
        (is (compiled/prepared? prepared))
        (is (some? (:ordered-invocation prepared)))
        (is (nil? (:lowering prepared)))
        (is (= :semantic-preparation (get-in prepared [:preparation-report :phase])))
        (is (false? (get-in prepared [:preparation-report :concrete-plan-available?]))))
      (doseq [options [{:outputs '[out first-output out]} {:taps '[x out]}]]
        (let [prepared (compiled/lower #'fixture/write-before-shape arguments
                                       (merge {:compiler :equation-first :target :ocl:0 :dtype :double}
                                              options))]
          (is (compiled/prepared? prepared))
          (is (= :after-source (get-in prepared [:preparation-report :output-projection-phase])))
          (is (every? #(not (contains? % :node)) (:out-tree prepared))
              "physical outputs and alias deduplication are not claimed before source work")))
      (doseq [options [{:donate '[x]} {:profile? true} {:session :borrowed}]]
        (is (= :compiled-ordered-capability
               (reason #(compiled/lower #'fixture/write-before-shape arguments
                                         (merge {:compiler :equation-first :target :ocl:0 :dtype :double}
                                                options))))))
      (is (= :compiled-equation-first-output
             (reason #(compiled/lower #'fixture/write-before-shape arguments
                                       {:compiler :equation-first :target :ocl:0 :dtype :double
                                        :outputs '[missing]}))))
      (is (= :compiled-ordered-output-kind
             (reason #(compiled/lower #'fixture/write-before-shape arguments
                                       {:compiler :equation-first :target :ocl:0 :dtype :double
                                        :taps '[n]})))))))

(defn- check-ordered-output-selection! [target]
  (let [prepared (compiled/lower #'fixture/write-before-shape
                                 [(double-array [7.0]) (double-array 1) 1]
                                 {:compiler :equation-first :target target :dtype :double
                                  :outputs '[out first-output out] :taps '[x out first-output]})
        artifact (compiled/instantiate! prepared)]
    (try
      (is (= :compiled-ordered-entry-intent
             (reason #(compiled/instantiate! prepared {:project-boundary identity}))))
      (let [first (artifact)
            check! (fn [outputs x]
                     (is (= #{:out :first-output :x} (set (keys outputs))))
                     (is (= [1.0] (vec (value/->host (:out outputs)))))
                     (is (= [(+ x 1.0)] (vec (value/->host (:first-output outputs)))))
                     (is (= [x] (vec (value/->host (:x outputs))))))]
        (check! first 7.0)
        (let [executable (:executable artifact)
              boundary (:resident-boundary @(:execution-state executable))
              plan (:plan boundary)
              tree (get-in boundary [:projection :out-tree])]
          (is (= [:out :first-output :x] (mapv :key tree)))
          (is (= (:outputs plan) (mapv :node tree)))
          (is (= (count (:outputs plan)) (count (set (:outputs plan)))))
          (is (= 1 (count (get-in plan [:attributes :semantic-outputs])))
              "public taps do not become fabricated semantic results")
          (is (= :invocation-link-output-boundary
                 (reason #(invocation-link/certify plan (pop (:outputs plan))))))
          (is (= :link-outputs
                 (reason #(invocation-link/certify plan (conj (:outputs plan) ::missing-output)))))
          (is (= (set (:outputs plan)) (set (keys (link/outputs executable))))))
        (is (= :invocation-materialization-buffer-dtype
               (reason #(artifact {:x (float-array 1)}))))
        (is (every? value/live? (vals first)) "failed preflight retains all previous outputs")
        (check! (artifact {:x (double-array [11.0])}) 11.0)
        (is (not-any? value/live? (vals first)) "successful replay retires every escaped wrapper")
        (check! (artifact) 7.0))
      (finally (compiled/close! artifact))))
  (let [prepared (compiled/lower #'fixture/write-before-shape
                                 [(double-array [7.0]) (double-array 1) Long/MAX_VALUE]
                                 {:compiler :equation-first :target target :dtype :double
                                  :outputs '[first-output] :taps '[x]})
        artifact (compiled/instantiate! prepared)]
    (try
      (is (instance? ArithmeticException (try (artifact) nil (catch Throwable error error))))
      (is (nil? @(:live-outputs artifact)))
      (is (= :link-execution-poisoned
             (reason #(link/invocation-output-projection (:executable artifact)))))
      (is (zero? @(:completed-replays (:executable artifact))))
      (finally (compiled/close! artifact))))
  (doseq [[source arguments options expected]
          [[#'root-fixture/ordered-void! [(double-array 1)] {:outputs '[out out] :taps '[out]}
            {:out [2.0]}]
           [#'fixture/write-before-resident-reduction
            [(double-array [7.0]) (double-array 1) 1]
            {:outputs '[sum] :taps '[out sum]} {:sum [7.0] :out [15.0]}]]]
    (let [compilation (equation-first/compile source {:target target :dtype :double})
          prepared (#'compiled/prepare-ordered-equation-first compilation arguments options nil)
          artifact (compiled/instantiate! prepared)]
      (try
        (is (= expected (into {} (map (fn [[key array]] [key (vec (value/->host array))]))
                                   (artifact))))
        (finally (compiled/close! artifact)))))
  (let [w (double-array [1.0 0.25 -0.5 1.0])
        b (double-array [0.25 -0.125])
        prepared (compiled/lower #'dense-fixture/four-layers
                                 [w b (double-array [1.0 -2.0])]
                                 {:compiler :equation-first :target target :dtype :double
                                  :constants '[w b] :taps '[middle]})
        artifact (compiled/instantiate! prepared)
        ;; Independent two-coordinate oracle, not the compiler's dense implementation.
        layer (fn [[x y]] [(+ x (* 0.25 y) 0.25) (+ (* -0.5 x) y -0.125)])]
    (try
      (is (some? (:ordered-invocation prepared)))
      (doseq [coordinates [[1.0 -2.0] [2.0 3.0]]]
        (let [expected (iterate layer coordinates)
              outputs (artifact {:x (double-array coordinates)})]
          (is (= #{:result :middle} (set (keys outputs))))
          (is (= (nth expected 2) (vec (value/->host (:middle outputs)))))
          (is (= (nth expected 4) (vec (value/->host (:result outputs)))))
          (let [plan (get-in @(:execution-state (:executable artifact)) [:resident-boundary :plan])
                middle (get-in plan [:attributes :compiler-buffer-bindings 'middle])]
            (is (not (contains? (set (vals (get-in plan [:attributes :public-buffer-bindings]))) middle)))
            (is (contains? (set (:outputs plan)) middle)))))
      (finally (compiled/close! artifact)))))

(deftest opencl-ordered-explicit-outputs-and-taps
  (if @ocl-probe/opencl-fp64-available?
    (check-ordered-output-selection! :ocl:0)
    (ocl-probe/opencl-skip! "ordered public output selection")))

(deftest level-zero-ordered-explicit-outputs-and-taps
  (if @ze-probe/gpu-available?
    (check-ordered-output-selection! :ze:0)
    (ze-probe/gpu-skip! "ordered public output selection")))

(deftest direct-eager-lowering-refuses-before-a-late-check-can-run
  (let [compilation (equation-first/compile #'fixture/write-before-shape
                                           {:target :ocl:0 :dtype :double})
        forbidden (fn [& _] (throw (Exception. "eager lowering evaluated a late check")))]
    (with-redefs [scalar/evaluate-invocation-step forbidden scalar/evaluate-host-equation forbidden]
      (is (= :equation-first-ordered-preparation
             (reason #(equation-first/lower compilation
                                            [(double-array [7.0]) (double-array 1) Long/MAX_VALUE])))))))

(deftest public-preparation-retains-the-concrete-route-for-an-ordered-prefix
  (with-redefs [gpu/make-session (fn [& _] (throw (Exception. "lowering opened a session")))]
    (let [prepared (compiled/lower #'allocation-fixture/shape-before-first-stage
                                   [(double-array [7.0]) (double-array 1)]
                                   {:compiler :equation-first :target :ocl:0 :dtype :double})]
      (is (some? (:lowering prepared)))
      (is (nil? (:ordered-invocation prepared))))))

(deftest source-action-coverage-is-shared-before-preparation-and-native-capture
  (let [compilation (equation-first/compile #'fixture/write-before-shape {:target :ocl:0 :dtype :double})
        arguments [(double-array [7.0]) (double-array 1) Long/MAX_VALUE]
        initial (invocation-link/begin-ordered-realization compilation arguments nil)
        calls (atom 0)
        forbidden (fn [& _] (throw (Exception. "source preflight evaluated source work or touched a driver")))]
    (with-redefs [scalar/evaluate-invocation-step forbidden scalar/evaluate-host-equation forbidden
                  gpu/make-session forbidden]
      (is (identical? initial (invocation-link/validate-ordered-source-actions! initial)))
      (is (= :invocation-link-native-initialization
             (reason #(invocation-link/validate-ordered-source-actions!
                       (assoc-in initial [:program :attributes :native-initialization-providers] ['external])))))
      (is (= :invocation-ordered-source-action
             (reason #(invocation-link/validate-ordered-source-actions!
                       (assoc-in initial [:program :equations 0 :operations]
                                 [(emitted-loop/map->EmittedStructuredLoop {})])))))
      (is (= :invocation-ordered-source-action
             (reason #(invocation-link/validate-ordered-source-actions!
                       (update-in initial [:materialization :plan :steps 0]
                                  (fn [step] (invocation/map->BufferClone (into {} step))))))))
      (with-redefs [invocation-link/validate-ordered-source-actions!
                    (fn [_] (swap! calls inc)
                      (throw (ex-info "unsupported source action" {:reason :test-source-capability})))]
        (is (= :test-source-capability
               (reason #(#'compiled/prepare-ordered-equation-first compilation arguments {} nil))))
        (is (= :test-source-capability
               (reason #(link/instantiate-ordered-invocation! compilation arguments {})))))
      (is (= 2 @calls))))
  (let [compilation (equation-first/compile #'root-fixture/functional-source-allocation
                                           {:target :ocl:0 :dtype :double})
        initial (invocation-link/begin-ordered-realization
                 compilation [(double-array [7.0]) (double-array 1) 2] nil)]
    (is (seq (:source-realizations initial)))
    (is (identical? initial (invocation-link/validate-ordered-source-actions! initial)))
    (with-redefs [emitted-program/operation-projection (fn [& _] {:physical-results {}})]
      (is (= :invocation-ordered-realization-storage
             (reason #(invocation-link/validate-ordered-source-actions! initial)))))))

(deftest preparation-retains-semantics-without-evaluating-the-late-source-check
  (let [compilation (equation-first/compile #'fixture/write-before-shape {:target :ocl:0 :dtype :double})
        args [(double-array [7.0]) (double-array 1) Long/MAX_VALUE]
        forbidden (fn [& _] (throw (Exception. "preparation evaluated source work or touched a driver")))]
    (with-redefs [scalar/evaluate-invocation-step forbidden scalar/evaluate-host-equation forbidden
                  gpu/make-session forbidden equation-first/lower forbidden]
      (let [prepared (#'compiled/prepare-ordered-equation-first compilation args {} {:cache :existing})]
        (is (compiled/prepared? prepared))
        (is (#'compiled/sealed-artifact? prepared))
        (is (nil? (:lowering prepared)))
        (is (identical? compilation (get-in prepared [:ordered-invocation :compilation])))
        (is (= args (:args prepared)))
        (is (= {:cache :existing} (get-in prepared [:preparation-report :template])))
        (is (false? (get-in prepared [:preparation-report :concrete-plan-available?])))
        (is (= [:x :first-output] (mapv :key (:in-tree prepared))))
        (is (= [:input :state] (mapv :role (:in-tree prepared))))
        (is (= [[1] [1]] (mapv :shape (:in-tree prepared))))
        (is (every? #(not (contains? % :node)) (concat (:in-tree prepared) (:out-tree prepared))))
        (is (= [:result] (mapv :key (:out-tree prepared))))
        (doseq [operation [compiled/plan compiled/certificate compiled/execution-identity
                           compiled/producer-interface compiled/cache-key]]
          (is (= :compiled-deferred-phase (reason #(operation prepared)))))
        (is (= :compiled-deferred-phase
               (reason #(compiled/compose {:id :not-yet-composable
                                           :components [{:id :ordered :program prepared}]}))))
        (is (= :compiled-ordered-owner (reason #(compiled/instantiate! (assoc prepared :target :other)))))))))

(deftest unsupported-preparation-capabilities-refuse-without-native-acquisition
  (let [compilation (equation-first/compile #'fixture/write-before-shape {:target :ocl:0 :dtype :double})
        args [(double-array [7.0]) (double-array 1) Long/MAX_VALUE]]
    (with-redefs [gpu/make-session (fn [& _] (throw (Exception. "unexpected driver acquisition")))]
      (doseq [opts [{:donate '[x]} {:profile? true} {:session :borrowed}]]
        (is (= :compiled-ordered-capability
               (reason #(#'compiled/prepare-ordered-equation-first compilation args opts nil)))))
      (is (= :link-write-read-only
             (reason #(#'compiled/prepare-ordered-equation-first
                       compilation args {:constants '[first-output]} nil))))))
  (let [compilation (equation-first/compile #'root-fixture/captured-entry-replay
                                           {:target :ocl:0 :dtype :double})
        shared (double-array [7.0])
        args [shared (double-array [3.0]) shared (double-array 1)]]
    (with-redefs [gpu/make-session (fn [& _] (throw (Exception. "alias rejection opened a session")))]
      (is (= :invocation-ordered-source-alias
             (reason #(#'compiled/prepare-ordered-equation-first compilation args {} nil))))
      (is (= :invocation-ordered-source-alias
             (reason #(link/instantiate-ordered-invocation! compilation args {})))))))

(defn- check-callable-ordered-preparation! [target]
  (let [compilation (equation-first/compile #'root-fixture/captured-entry-replay
                                           {:target target :dtype :double})
        x (double-array [7.0]) weights (double-array [3.0]) state (double-array 1)
        prepared (#'compiled/prepare-ordered-equation-first
                  compilation [x weights state (double-array 1)] {:constants '[weights]} nil)
        c (compiled/instantiate! prepared)]
    (try
      (aset weights 0 50.0) (aset state 0 100.0)
      (let [first (:result (c {:x (double-array [11.0])}))]
        (is (= [14.0] (vec (value/->host first))))
        (is (= :compiled-ordered-input-role (reason #(c {:weights (double-array [9.0])}))))
        (is (= :invocation-materialization-buffer-dtype (reason #(c {:x (float-array 1)}))))
        (is (= :invocation-ordered-input-specialization (reason #(c {:x (double-array 2)}))))
        (is (= :invocation-ordered-source-alias (reason #(c {:x state}))))
        (is (= :compiled-ordered-owner (reason #((assoc c :target :other) {}))))
        (is (value/live? first) "failed replacement preflight preserves the previous output wrapper")
        (doseq [[operation expected-reason]
                [[#(compiled/invoke-with-evidence c {}) :compiled-deferred-phase]
                 [#(compiled/measure-storage-representation! c :double) :compiled-deferred-phase]
                 [#(compiled/profile c) :link-profiling-disabled]
                 [#(compiled/measure c) :link-profiling-disabled]]]
          (is (= expected-reason (reason operation)))
          (is (= 1 @(:completed-replays (:executable c))))
          (is (value/live? first) "unsupported capability cannot retire outputs or replay"))
        (let [second (:result (c {:x (double-array [21.0])}))]
          (is (= [24.0] (vec (value/->host second))))
          (is (not (value/live? first)) "successful mutation retires earlier public output aliases")))
      (let [third (:result (c))]
        (is (= [10.0] (vec (value/->host third))) "omitted x uses the original Prepared default, not 21")
        (is (= :compiled-deferred-phase (reason #(compiled/invoke-leased c {}))))
        (is (value/live? third)))
      (is (= 3 @(:completed-replays (:executable c))))
      (is (= [100.0] (vec state)))
      (finally (compiled/close! c))))
  (let [prepared (compiled/lower #'fixture/write-before-shape
                                 [(double-array [7.0]) (double-array 1) Long/MAX_VALUE]
                                 {:compiler :equation-first :target target :dtype :double})
        c (compiled/instantiate! prepared)]
    (try
      (is (instance? ArithmeticException (try (c) nil (catch Throwable error error))))
      (let [executable (:executable c)
            state @(:execution-state executable)
            token (get-in state [:ordered :current :realized :buffers 'first-output])
            view (get-in state [:ordered :node-views token])
            actual (double-array 1)]
        (is (some? view))
        (when view
          (gpu/download-range! (:session executable) view actual {:elements 1})
          (is (= [8.0] (vec actual)) "the public route completes the earlier write before overflow"))
        (is (= 2 (count (get-in state [:ordered :roots])))
            "failure acquires no suffix allocation"))
      (is (nil? @(:live-outputs c)) "late source failure publishes no public output wrapper")
      (is (zero? @(:completed-replays (:executable c))))
      (is (= :link-execution-poisoned (reason #(c))))
      (finally (compiled/close! c))))
  (let [compilation (equation-first/compile #'root-fixture/captured-entry-replay
                                           {:target target :dtype :double})
        shared (double-array [7.0])
        prepared (#'compiled/prepare-ordered-equation-first
                  compilation [shared shared (double-array 1) (double-array 1)]
                  {:constants '[weights]} nil)
        c (compiled/instantiate! prepared)]
    (try
      (is (= [14.0] (vec (value/->host (:result (c))))))
      (is (= [16.0] (vec (value/->host (:result (c {:x (double-array [9.0])}))))))
      (finally (compiled/close! c)))))

(deftest opencl-callable-preparation-preserves-defaults-and-output-lifetimes
  (if @ocl-probe/opencl-available?
    (check-callable-ordered-preparation! :ocl:0)
    (ocl-probe/opencl-skip! "callable source-ordered prepared boundary")))

(deftest level-zero-callable-preparation-preserves-defaults-and-output-lifetimes
  (if @ze-probe/gpu-available?
    (check-callable-ordered-preparation! :ze:0)
    (ze-probe/gpu-skip! "callable source-ordered prepared boundary")))

(defn- check-preserved-scalar-between-writes! [target]
  (doseq [n [2 Long/MAX_VALUE]]
    (let [prepared (compiled/lower #'fixture/preserved-scalar-between-writes
                                   [(double-array [7.0]) (double-array 1) n]
                                   {:compiler :equation-first :target target :dtype :double})
          c (compiled/instantiate! prepared)
          calls (atom 0)
          evaluate scalar/evaluate-host-equation]
      (try
        (is (some? (:ordered-invocation prepared)))
        (with-redefs [scalar/evaluate-host-equation
                      (fn [source-ns equation context]
                        (when (contains? (:flags (effects/descriptor (:source equation)))
                                         :preserve-source-evaluation)
                          (swap! calls inc))
                        (evaluate source-ns equation context))]
          (if (= n 2)
            (do
              (let [first (:result (c))]
                (is (= [9.0] (vec (value/->host first))))
                (is (= [13.0] (vec (value/->host (:result (c {:x (double-array [11.0])}))))))
                (is (not (value/live? first))))
              (is (= 2 @calls) "the retained scalar executes once per replay, not during preparation"))
            (do
              (is (thrown? ArithmeticException (c)))
              (is (= 1 @calls))
              (is (nil? @(:live-outputs c)))
              (let [executable (:executable c)
                    state @(:execution-state executable)
                    token (get-in state [:ordered :current :realized :buffers 'out])
                    view (get-in state [:ordered :node-views token])
                    actual (double-array 1)]
                (is (some? view))
                (when view
                  (gpu/download-range! (:session executable) view actual {:elements 1})
                  (is (= [8.0] (vec actual)) "overflow preserves the earlier write and prevents the later write")))
              (is (= :link-execution-poisoned (reason #(c))))
              (is (= 1 @calls)))))
        (finally (compiled/close! c))))))

(deftest public-retained-scalar-ordering-on-opencl
  (if @ocl-probe/opencl-available?
    (check-preserved-scalar-between-writes! :ocl:0)
    (ocl-probe/opencl-skip! "retained scalar between public source writes")))

(deftest public-retained-scalar-ordering-on-level-zero
  (if @ze-probe/gpu-available?
    (check-preserved-scalar-between-writes! :ze:0)
    (ze-probe/gpu-skip! "retained scalar between public source writes")))
