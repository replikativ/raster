(ns raster.gpu.ordered-preparation-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.ir.invocation-source-order-test :as fixture]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.value :as value]
            [raster.gpu.ordered-invocation-root-test :as root-fixture]
            [raster.gpu.device-probe :as ocl-probe]
            [raster.dl.gpu-grad-parity :as ze-probe]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

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
  (let [compilation (equation-first/compile #'fixture/write-before-shape {:target target :dtype :double})
        prepared (#'compiled/prepare-ordered-equation-first
                  compilation [(double-array [7.0]) (double-array 1) Long/MAX_VALUE] {} nil)
        c (compiled/instantiate! prepared)]
    (try
      (is (instance? ArithmeticException (try (c) nil (catch Throwable error error))))
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
