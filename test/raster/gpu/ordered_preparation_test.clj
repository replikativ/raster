(ns raster.gpu.ordered-preparation-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.par :as par]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.ir.invocation-source-order-test :as fixture]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.emitted-structured-loop :as emitted-loop]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.value :as value]
            [raster.gpu.ordered-invocation-root-test :as root-fixture]
            [raster.gpu.device-probe :as ocl-probe]
            [raster.dl.gpu-grad-parity :as ze-probe]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftm late-captured-entry-replay
  [x :- (Array double), weights :- (Array double), state :- (Array double),
   out :- (Array double)] :- (Array double)
  (let [updated (par/map! state i 1 double (+ (aget state i) 1.0))
        temporary (double-array 1)
        computed (par/map! temporary i 1 double (+ (aget x i) (aget weights i)))]
    (par/map! out i 1 double (aget computed i))))

(deftest public-lowering-selects-ordered-semantics-only-for-late-actions
  (let [target :ocl:0
        options {:compiler :equation-first :target target :dtype :double}
        arguments [(double-array [7.0]) (double-array 1) Long/MAX_VALUE]
        late (equation-first/compile #'fixture/write-before-shape (dissoc options :compiler))
        prefix (equation-first/compile #'fixture/shape-before-write (dissoc options :compiler))
        forbidden (fn [& _] (throw (AssertionError. "public preparation executed source or acquired a device")))]
    (with-redefs [scalar/evaluate-invocation-step forbidden scalar/evaluate-host-equation forbidden
                  gpu/make-session forbidden equation-first/lower forbidden
                  equation-first/compile (fn [& _] late)]
      (let [prepared (compiled/lower #'fixture/write-before-shape arguments options)]
        (is (compiled/prepared? prepared))
        (is (identical? late (get-in prepared [:ordered-invocation :compilation])))
        (is (nil? (:lowering prepared)))
        (is (= :ordered-source (get-in prepared [:descriptor :execution-phase])))
        (is (some? (get-in prepared [:preparation-report :template])))
        (is (= :compiled-deferred-phase (reason #(compiled/plan prepared))))
        (is (= :compiled-deferred-phase (reason #(compiled/certificate prepared))))
        (is (= :compiled-ordered-entry-intent
               (reason #(compiled/instantiate! prepared {:entry-roles {}}))))
        (is (= :compiled-prepared-math-request
               (reason #(compiled/instantiate!
                          prepared {:scalar-math {:overrides {[:exp :float] :f64-target-library-rte-f32}}}))))
        (is (= :compiled-ordered-owner (reason #(compiled/instantiate! (assoc prepared :target :other))))))
      (doseq [intent [{:donate '[x]} {:outputs '[first-output]} {:taps '[x]}
                      {:profile? true} {:session :borrowed}]]
        (is (= :compiled-ordered-capability
               (reason #(compiled/lower #'fixture/write-before-shape arguments (merge options intent)))))))
    ;; The eager path is still selected for prefix-only source work; its real
    ;; lowering may evaluate that prefix, but never a late action prematurely.
    (with-redefs [gpu/make-session forbidden equation-first/compile (fn [& _] prefix)]
      (let [prepared (compiled/lower #'fixture/shape-before-write [(double-array [7.0]) 1] options)]
        (is (nil? (:ordered-invocation prepared)))
        (is (some? (compiled/plan prepared)))))))

(deftest public-ordered-lowering-keeps-owner-and-result-observation-gates
  (let [options {:compiler :equation-first :target :ocl:0 :dtype :double}
        arguments [(double-array [7.0]) (double-array 1) 1]
        compilation (equation-first/compile #'fixture/write-before-shape (dissoc options :compiler))
        forbidden (fn [& _] (throw (AssertionError. "refused preparation touched source or device")))]
    (with-redefs [gpu/make-session forbidden scalar/evaluate-invocation-step forbidden
                  scalar/evaluate-host-equation forbidden equation-first/lower forbidden]
      (doseq [[changed expected]
              [[(dissoc compilation :ordered-source-owner) :ordered-source-owner-mismatch]
               [(assoc compilation :attributes {:checked-stage-result true}) :checked-stage-observation-required]]]
        ;; The existing template key includes the actual compile function;
        ;; each replacement below therefore owns its own template observation.
        (with-redefs [equation-first/compile (fn [& _] changed)]
          (is (= expected
                 (reason #(compiled/lower #'fixture/write-before-shape arguments
                                          options)))))))))

(deftest ordinary-source-owner-cannot-be-removed-or-copied-before-resource-acquisition
  (doseq [[v arguments] [[#'fixture/shape-before-write [(double-array [7.0]) 1]]
                        [#'fixture/write-before-shape [(double-array [7.0]) (double-array 1) 1]]]]
    (let [compilation (equation-first/compile v {:target :ocl:0 :dtype :double})
          other (equation-first/compile v {:target :ocl:0 :dtype :double})
          sessions (atom 0)
          forbidden (fn [& _] (swap! sessions inc)
                      (throw (AssertionError. "unowned source touched a device")))]
      (is (some? (:ordered-source-owner compilation)))
      (doseq [changed [(dissoc compilation :ordered-source-owner)
                       (assoc compilation :ordered-source-owner (:ordered-source-owner other))]]
        (with-redefs [gpu/make-session forbidden]
          (is (= :ordered-source-owner-mismatch
                 (reason #(link/instantiate-ordered-invocation! changed arguments {}))))
          (is (= :ordered-source-owner-mismatch
                 (reason #(#'compiled/prepare-ordered-equation-first changed arguments {} nil))))))
      (is (zero? @sessions)))))

(deftest unsupported-ordered-policy-does-not-disable-eager-compilation
  (let [source (binding [*ns* (the-ns 'raster.gpu.ordered-preparation-test)
                         *unchecked-math* true]
                 (eval (list 'raster.core/deftm (gensym "unchecked-prefix")
                             '[n :- Long] ':- '(Array double)
                             '(let [out (double-array n)]
                                (raster.par/map! out i n double 2.0)))))
        compilation (equation-first/compile source {:target :ocl:0 :dtype :double})
        sessions (atom 0)]
    (is (nil? (:ordered-source-owner compilation)))
    (is (seq (invocation/source-host-actions
              (get-in compilation [:semantic :attributes :invocation-plan]) (:semantic compilation))))
    (is (empty? (invocation/ordered-host-actions
                 (get-in compilation [:semantic :attributes :invocation-plan]) (:semantic compilation))))
    (with-redefs [gpu/make-session (fn [& _] (swap! sessions inc)
                                   (throw (AssertionError. "compile/lower touched a device")))]
      (is (some? (link-plan/validate! (equation-first/lower compilation [1]))))
      (is (= :ordered-source-owner-mismatch
             (reason #(link/instantiate-ordered-invocation! compilation [1] {})))))
    (is (zero? @sessions))))

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
      (doseq [opts [{:donate '[x]} {:outputs '[first-output]} {:taps '[x]}
                    {:profile? true} {:session :borrowed}]]
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
  (let [x (double-array [7.0]) weights (double-array [3.0]) state (double-array 1)
        prepared (compiled/lower #'late-captured-entry-replay
                                 [x weights state (double-array 1)]
                                 {:compiler :equation-first :target target :dtype :double
                                  :constants '[weights]})
        _ (assert (:ordered-invocation prepared) "fixture must retain a genuine late host action")
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
      (is (nil? @(:live-outputs c)) "late source failure publishes no public output wrapper")
      (is (zero? @(:completed-replays (:executable c))))
      (is (= :link-execution-poisoned (reason #(c))))
      (finally (compiled/close! c))))
  (let [shared (double-array [7.0])
        prepared (compiled/lower #'late-captured-entry-replay
                                 [shared shared (double-array 1) (double-array 1)]
                                 {:compiler :equation-first :target target :dtype :double
                                  :constants '[weights]})
        _ (assert (:ordered-invocation prepared) "fixture must retain a genuine late host action")
        c (compiled/instantiate! prepared)]
    (try
      (is (= [14.0] (vec (value/->host (:result (c))))))
      (is (= [16.0] (vec (value/->host (:result (c {:x (double-array [9.0])}))))))
      (finally (compiled/close! c)))))

(defn- check-late-integral-host-boundary! [target]
  (doseq [[rows width] [[2 3] [0 2] [-1 1] [Long/MAX_VALUE 2] [(inc (long Integer/MAX_VALUE)) 1]]]
    (let [x (double-array [7.0]) state (double-array 1)
          arguments [x state rows width]
          oracle-state (double-array 1)
          oracle (try {:values (vec (fixture/write-before-checked-integral-shape x oracle-state rows width))}
                      (catch Exception error {:error (class error)}))
          prepared (compiled/lower #'fixture/write-before-checked-integral-shape arguments
                                   {:compiler :equation-first :target target :dtype :double})
          compilation (get-in prepared [:ordered-invocation :compilation])
          initial (invocation-link/begin-ordered-realization compilation arguments nil)
          layout (invocation-link/ordered-entry-layout initial (invocation-link/ordered-entry-roles initial nil))
          state-id (:id (first (filter #(identical? state (:source %)) (:nodes layout))))
          c (compiled/instantiate! prepared)
          executable (:executable c)
          session (:session executable)]
      (try
        (is (some? state-id))
        (let [actual (try {:values (vec (value/->host (:result (c))))}
                          (catch Exception error {:error (class error)}))
              resident-state (double-array 1)
              view (get-in @(:execution-state executable) [:ordered :node-views state-id])]
          (is (= oracle actual))
          ;; Privileged regression inspection, not a public output from a failed invocation.
          ;; The session still owns these buffers until close, even when replay is poisoned.
          (gpu/download-range! session view resident-state {:elements 1})
          (is (= (vec oracle-state) (vec resident-state)) "earlier device write precedes the host check")
          (is (= [0.0] (vec state)) "resident mutation does not invent host copyback")
          (when (:error oracle)
            (is (nil? @(:live-outputs c)))
            (is (zero? @(:completed-replays executable)))
            (is (= 2 (count (:buffers @session))) "failed size check acquires no suffix buffer")
            (is (= :link-execution-poisoned (reason #(c)))))
          (when (= [2 3] [rows width])
            (is (= (vec (repeat 18 13.0))
                   (vec (value/->host (:result (c {:x (double-array [11.0])})))))))
          (is (empty? (:events @session)))
          (is (empty? (get-in @(:execution-state executable) [:ordered :prepared]))))
        (finally (compiled/close! c)))
      (is (true? (:closed? @session)))
      (is (empty? (:buffers @session))))))

(deftest late-integral-host-checks-preserve-source-order-on-local-backends
  (if @ocl-probe/opencl-available?
    (check-late-integral-host-boundary! :ocl:0)
    (ocl-probe/opencl-skip! "late integral host boundary"))
  (if @ze-probe/gpu-available?
    (check-late-integral-host-boundary! :ze:0)
    (ze-probe/gpu-skip! "late integral host boundary")))

(deftest opencl-callable-preparation-preserves-defaults-and-output-lifetimes
  (if @ocl-probe/opencl-available?
    (check-callable-ordered-preparation! :ocl:0)
    (ocl-probe/opencl-skip! "callable source-ordered prepared boundary")))

(deftest level-zero-callable-preparation-preserves-defaults-and-output-lifetimes
  (if @ze-probe/gpu-available?
    (check-callable-ordered-preparation! :ze:0)
    (ze-probe/gpu-skip! "callable source-ordered prepared boundary")))
