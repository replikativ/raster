(ns raster.gpu.ordered-structured-loop-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.par :as par]
            [raster.arrays :as arrays]
            [raster.ode.pde :as pde]
            [raster.nn :as nn]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.structured-control :as control]
            [raster.gpu.core :as gpu]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.parallel-program :as program-runtime]
            [raster.gpu.value :as value]
            [raster.gpu.device-probe :as ocl-probe]
            [raster.dl.gpu-grad-parity :as ze-probe]
            [raster.gpu.link :as link]))

(deftm repeated-shift
  [input :- (Array double), steps :- Long] :- (Array double)
  (let [n (alength input)
        state (aclone input)
        scratch (double-array n)
        count (int steps)
        time-loop (dotimes [tick count]
                    (arrays/acopy! (par/pmap i n double (+ (aget state i) 1.0))
                                   0 state 0 n))
        result (par/pmap i n double (* 2.0 (aget state i)))]
    result))

(deftm metadata-between-independent-stages
  [input :- (Array double), metadata :- (Array double), early :- (Array double), width :- Long] :- (Array double)
  (let [written (par/map! early i width double (+ (aget input i) 1.0))
        size (alength metadata)]
    (par/pmap i width double (+ (aget input i) (double size)))))

(deftest public-loop-preflight-retains-its-source-order-without-touching-a-device
  (let [compilation (equation-first/compile #'repeated-shift {:target :ocl:0 :dtype :double})
        input (double-array [0.5 -1.0 3.0])
        state (invocation-link/begin-ordered-realization compilation [input 3] nil)
        loop-equation (some #(when (control/loop-program? (:algorithm %)) %)
                            (get-in compilation [:semantic :equations]))]
    (is (some? loop-equation))
    (is (identical? state (invocation-link/validate-ordered-source-actions! state)))
    (is (some #(= (:id loop-equation) (:equation %)) (:source-order state)))
    (is (empty? (:acquired-storage state)))
    (is (= [0.5 -1.0 3.0] (vec input)))))

(defn- check-public-repetition! [target]
  (let [compilation (equation-first/compile #'repeated-shift {:target target :dtype :double})]
    (doseq [[trips input-values] (concat (map #(vector % [0.5 -1.0 3.0]) [-2 0 1 2 5 6])
                                       [[0 []] [2 []]])]
      (let [input (double-array input-values)
            executable (link/instantiate-ordered-invocation! compilation [input trips] {})]
        (try
          (dotimes [replay 2]
            (let [expected (mapv #(* 2.0 (+ % (max 0 trips))) input)
                  output (first (keys (link/run! executable)))]
              (is (= expected (vec (link/download executable output)))
                  (str "whole-source recurrence " target " trips=" trips " replay=" replay))
              (is (= expected (vec (repeated-shift input trips))))
              (is (= (inc replay) @(:completed-replays executable)))
              (is (empty? (get-in @(:execution-state executable) [:ordered :prepared]))))
            (when (seq input-values) (aset input 0 (+ 4.0 (aget input 0)))))
          (finally (link/close! executable)))))))

(defn- check-iteration-failure! [target]
  (let [compilation (equation-first/compile #'repeated-shift {:target target :dtype :double})
        input (double-array [0.5 -1.0 3.0])
        executable (link/instantiate-ordered-invocation! compilation [input 4] {})
        bind gpu/bind-kernel-graph! run gpu/run-kernel-graph! release gpu/release-kernel-graph!
        bindings (atom 0) runs (atom 0) releases (atom 0)
        failure (ex-info "injected later loop iteration failure" {})]
    (try
      (with-redefs [gpu/bind-kernel-graph! (fn [& args] (let [handle (apply bind args)]
                                                       (swap! bindings inc) handle))
                    gpu/run-kernel-graph! (fn [& args]
                                            (if (= 3 (swap! runs inc))
                                              (throw failure) (apply run args)))
                    gpu/release-kernel-graph! (fn [& args] (swap! releases inc) (apply release args))]
        (is (identical? failure (try (link/run! executable) nil (catch Throwable error error))))
        (is (= 3 @runs) "prefix and first iteration finish; the next fails before any suffix")
        (is (= 4 @bindings) "prefix plus three bounded carry variants, no suffix binding")
        (is (= @bindings @releases))
        (is (empty? (get-in @(:execution-state executable) [:ordered :prepared])))
        (is (nil? (:resident-boundary @(:execution-state executable))))
        (is (zero? @(:completed-replays executable)))
        (is (false? @(:output-ready? executable))))
      (finally (link/close! executable)))
    (is (true? (:closed? @(:session executable))))))

(defn- check-independent-math-request! [target]
  (let [request {:scalar-math {:overrides {[:tanh :float] :f64-target-library-rte-f32}}}
        compilation (equation-first/compile #'repeated-shift (merge {:target target :dtype :double} request))
        input (double-array [0.5 -1.0 3.0])
        executable (link/instantiate-ordered-invocation! compilation [input 2] request)
        project-plan link-plan/make-with-final-projection
        observed (atom [])]
    (try
      (with-redefs [link-plan/make-with-final-projection
                    (fn [candidate project evidence actual-request]
                      (swap! observed conj actual-request)
                      (project-plan candidate project evidence actual-request))]
        (let [output (first (keys (link/run! executable)))]
          (is (= [5.0 2.0 10.0] (vec (link/download executable output))))))
      (is (seq @observed))
      (is (every? #{request} @observed) "every numerical stage receives independent caller intent")
      (finally (link/close! executable)))))

(defn- check-completed-host-snapshots! [executable]
  (let [whole (get-in @(:execution-state executable) [:resident-boundary :plan :instances 0 :call])
        program (:program whole)
        equation (last (:equations program))
        snapshots (into {} (keep #(when (program-call/evaluated-host-equation? %)
                                   [(get-in % [:equation :id]) %])) (:steps whole))
        host (first (vals snapshots))
        host-id (get-in host [:equation :id])
        construct (fn [snapshots]
                    (program-call/make-equation-program-call
                     program (:id equation)
                     {:buffers (:buffers whole) :scalar-values (:scalar-values whole)
                      :loop-scratch (:loop-scratch whole) :host-steps snapshots} nil nil))]
    (is (some? host) "the real RK4 reduction has an already-completed scalar prefix")
    (is (program-call/emitted-program-call? (:call (construct snapshots))))
    (is (thrown? clojure.lang.ExceptionInfo (construct {})))
    (doseq [changed [(assoc-in host [:equation :id] :different-host)
                     (update host :operands #(update-vals % (fn [value]
                                                             (if (map? value) (assoc value :type :float)
                                                               [::different-operand]))))
                     (update host :results #(update-vals % (fn [value] (assoc value :type :float))))
                     (update host :results #(update-vals % (fn [value] (update value :value inc))))]]
      (is (thrown? clojure.lang.ExceptionInfo (construct (assoc snapshots host-id changed)))))))

(defn- check-completed-pde-boundary! [executable trips]
  (let [runtime @(:execution-state executable)
        plan (get-in runtime [:resident-boundary :plan])
        call (get-in plan [:instances 0 :call])
        state (get-in runtime [:ordered :current])
        materialized (materialization/finish-materialization (:materialization state))
        invocation-plan (:plan materialized)
        public-nsteps (:id (first (filter #(= 'nsteps (:symbol %)) (:parameters invocation-plan))))
        narrowed (first (filter #(some (fn [operand] (= public-nsteps (:value operand))) (:operands %))
                                (:steps invocation-plan)))
        sources (program-call/buffer-identities call)
        mapping (zipmap sources (mapv #(vector :pde-remapped-storage %) (range (count sources))))
        remapped (program-call/map-buffers call mapping)]
    (is (link-plan/link-plan? plan))
    (is (link-plan/program-link-instance? (first (:instances plan))))
    (is (= 0 (get-in plan [:attributes :driver-allocations])))
    (is (= (set (get-in call [:program :inputs]))
           (set (concat (keys (:program-buffers materialized)) (keys (:program-scalars materialized))))))
    (doseq [alias (filter invocation/value-alias? (:steps invocation-plan))]
      (is (identical? (get-in materialized [:values (:id alias)])
                      (get-in materialized [:values (:source alias)]))))
    (is (invocation/scalar-compute? narrowed))
    (is (= :int (get-in materialized [:values (:id narrowed) :type])))
    (is (= trips (get-in materialized [:values (:id narrowed) :value])))
    (is (not= public-nsteps (:id narrowed)))
    (is (some #(= (:id narrowed) (:invocation-value %)) (:bindings invocation-plan)))
    (is (= (set (get-in call [:program :outputs])) (set (keys (:outputs call)))))
    (is (= 1 (count (:outputs remapped))))
    (is (every? (set (vals mapping)) (vals (:outputs remapped)))
        "the completed rank-zero reduction result remains resident during identity transport")
    (is (every? #(= [] (:shape (get-in call [:program :values %]))) (keys (:outputs call))))))

(defn- check-discrete-rk4! [target]
  ;; A discrete Laplacian eigenmode evolves by the RK4 stability polynomial.
  ;; This oracle uses neither Raster's stencils nor its RK4 implementation.
  (let [compilation (equation-first/compile #'pde/heat-loss-rk4 {:target target :dtype :double})
        size 16
        mode (mapv #(if (or (zero? %) (= % (dec size))) 0.0
                        (Math/sin (/ (* Math/PI %) (dec size)))) (range size))
        input (double-array mode) target-values (double-array (map #(* 0.5 %) mode))
        alpha 0.05 inv-dx2 225.0 dt 0.01
        eigenvalue (* -4.0 alpha inv-dx2 (Math/pow (Math/sin (/ Math/PI (* 2.0 (dec size)))) 2.0))
        z (* dt eigenvalue)
        amplification (+ 1.0 z (/ (* z z) 2.0) (/ (* z z z) 6.0) (/ (* z z z z) 24.0))]
    (doseq [trips [0 1 3]]
      (let [executable (link/instantiate-ordered-invocation!
                        compilation [input target-values alpha inv-dx2 dt trips] {})
            bind gpu/bind-kernel-graph!
            release gpu/release-kernel-graph!
            bound (atom []) released (atom [])]
        (with-redefs [gpu/bind-kernel-graph!
                      (fn [& args]
                        (let [handle (apply bind args)] (swap! bound conj handle) handle))
                      gpu/release-kernel-graph!
                      (fn [session handle]
                        (let [result (release session handle)]
                          (swap! released conj handle) result))
                      gpu/record-graph!
                      (fn [& _]
                        (throw (AssertionError. "PDE program must not use descriptor recording")))]
         (try
          (let [output (first (keys (link/run! executable)))
                actual (first (link/download executable output))
                coefficient (- (Math/pow amplification trips) 0.5)
                expected (* coefficient coefficient (/ (reduce + (map #(* % %) mode)) size))]
            (is (< (Math/abs (- expected actual)) 1.0e-12)
                (str "discrete RK4 " target " trips=" trips " actual=" actual " expected=" expected))
            (is (= mode (vec input)) "the working carry must not overwrite the public initial state")
            (when (zero? trips) (check-completed-host-snapshots! executable))
            (check-completed-pde-boundary! executable trips))
          (finally (link/close! executable))))
        (is (seq @bound) "the actual PDE binds typed executable graphs")
        (is (= (frequencies @bound) (frequencies @released))
            "every successfully acquired graph handle is retired exactly once")
        (is (true? (:closed? @(:session executable))))
        (doseq [resource [:buffers :allocations :kernel-graphs :graphs :prepared :events]]
          (is (empty? (get @(:session executable) resource))
              (str "PDE owner retires " resource)))))))

(defn- check-metadata-only-buffer! [target]
  (let [compilation (equation-first/compile #'metadata-between-independent-stages
                                           {:target target :dtype :double})
        executable (link/instantiate-ordered-invocation!
                    compilation [(double-array [1.0 2.0]) (double-array 3) (double-array 2) 2] {})]
    (try
      (let [output (first (keys (link/run! executable)))]
        (is (= [4.0 5.0] (vec (link/download executable output))))
        (is (= 3 (count (get-in @(:execution-state executable) [:ordered :roots])))
            "a completed metadata-only read does not require uploading its unrelated array"))
      (finally (link/close! executable)))))

(defn- check-public-mlp!
  ([target] (check-public-mlp! target nil))
  ([target request]
  (let [arguments [(double-array [1.0 0.0 0.0 1.0])
                   (double-array [0.0 0.0]) (double-array [0.5 0.5])
                   (double-array [0.25]) (double-array [1.0 -1.0])]
        prepared (compiled/lower #'nn/predict-fn arguments
                                 (merge {:compiler :equation-first :target target :dtype :double
                                         :constants '[W1 b1 W2 b2]} request))
        invocation-plan (get-in prepared [:ordered-invocation :compilation :semantic :attributes :invocation-plan])
        bias-id (:id (first (filter #(= 'b1 (:symbol %)) (:parameters invocation-plan))))
        rows (:symbol (first (filter #(and (invocation/shape-projection? %) (= bias-id (:source %)))
                                     (:steps invocation-plan))))
        artifact (compiled/instantiate! prepared)
        latest (volatile! nil)]
    (try
      (is (some? (:ordered-invocation prepared)))
      (when request (is (= request (:math-request prepared))))
      (is (nil? (:lowering prepared)) "preparation cannot publish a concrete plan before source completion")
      (is (some? rows))
      (dotimes [replay 2]
        (let [expected (vec (apply nn/predict-fn arguments))
              result (:result (artifact {}))
              actual (vec (value/->host result))
              plan (get-in @(:execution-state (:executable artifact)) [:resident-boundary :plan])
              call (get-in plan [:instances 0 :call])
              staging (program-runtime/staging-plan call [:public-mlp replay] request)]
          (when-let [previous @latest]
            (is (not (value/live? previous)) "the next replay retires the previous output wrapper"))
          (vreset! latest result)
          (is (= (count expected) (count actual)))
          (is (every? #(< (Math/abs (double %)) 1.0e-12) (map - expected actual)))
          (is (link-plan/link-plan? plan))
          (is (= 0 (get-in plan [:attributes :driver-allocations])))
          (is (= 2 (count staging)))
          (is (every? #(contains? (:scalar-values %) rows) staging)
              "the later layer retains shape context even without a kernel ABI use"))
        (aset ^doubles (last arguments) 0 2.0))
      (finally (compiled/close! artifact)))
    (is (not (value/live? @latest)))
    (is (true? (:closed? @(:session (:executable artifact))))))))

(deftest opencl-public-ordered-loop-composes-prefix-body-and-suffix
  (if @ocl-probe/opencl-available?
    (do (check-public-repetition! :ocl:0)
        (check-iteration-failure! :ocl:0)
        (check-independent-math-request! :ocl:0)
        (check-discrete-rk4! :ocl:0)
        (check-metadata-only-buffer! :ocl:0)
        (check-public-mlp! :ocl:0)
        ;; predict-fn is the Double dense/ReLU example. This checks request ownership,
        ;; not a claim that this workload contains the Float tanh operation.
        (check-public-mlp! :ocl:0
                           {:scalar-math {:overrides {[:tanh :float] :f64-target-library-rte-f32}}}))
    (ocl-probe/opencl-skip! "public ordered structured-loop composition")))

(deftest level-zero-public-ordered-loop-composes-prefix-body-and-suffix
  (if @ze-probe/gpu-available?
    (do (check-public-repetition! :ze:0)
        (check-iteration-failure! :ze:0)
        (check-independent-math-request! :ze:0)
        (check-discrete-rk4! :ze:0)
        (check-metadata-only-buffer! :ze:0)
        (check-public-mlp! :ze:0))
    (ze-probe/gpu-skip! "public ordered structured-loop composition")))
