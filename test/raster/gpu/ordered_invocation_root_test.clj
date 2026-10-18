(ns raster.gpu.ordered-invocation-root-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.invocation-source-order-test :as fixture]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.core :refer [deftm]]
            [raster.dl.gpu-grad-parity :as ze-probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as ocl-probe]
            [raster.gpu.link :as link]
            [raster.gpu.parallel-program :as parallel-program]
            [raster.math :as math]
            [raster.par :as par]))

(deftm source-zero-observed
  [x :- (Array double), early :- (Array double), n :- Long] :- (Array double)
  (let [written (par/map! early i 1 double (+ (aget x i) 1.0))
        out (double-array n)
        result (par/map! out i n double (+ (aget out i) 1.0))]
    result))

(deftm ordered-void!
  [out :- (Array double)] :- Void
  (par/map-void! i 1 (aset out i 2.0)))

(deftm selected-source-math
  [x :- (Array float), n :- Long] :- (Array float)
  (let [out (float-array n)]
    (par/map! out i n float (math/tanh (aget x i)))))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- check-owned-root! [target]
  (let [compilation (equation-first/compile #'source-zero-observed {:target target :dtype :double})]
    (doseq [n [0 3]]
      (let [x (double-array [7.0])
            executable (link/instantiate-ordered-invocation! compilation [x (double-array 1) n] {})
            session (:session executable)]
        (try
          (is (empty? (:buffers @session)) "construction cannot allocate a suffix early")
          (is (= :link-runtime-boundary-unavailable (reason #(link/outputs executable))))
          (is (= :link-runtime-boundary-unavailable (reason #(link/output-lease! executable))))
          (let [lease (link/run-and-lease! executable)
                output-id (first (keys (:outputs @lease)))
                plan (get-in @(:execution-state executable) [:resident-boundary :plan])
                early-id (get-in plan [:attributes :public-buffer-bindings 'early])]
            (try
              (is (= 1 (:replay @lease)))
              (is (= (vec (repeat n 1.0)) (vec (link/download executable output-id))))
              (is (= [8.0] (vec (link/download executable early-id))))
              (is (= :link-output-lease-active (reason #(link/run! executable))))
              (is (= :link-output-lease-active (reason #(link/close! executable))))
              (finally (.close ^java.io.Closeable lease)))
            (aset x 0 11.0)
            (link/run! executable)
            (is (= 2 @(:completed-replays executable)))
            (is (= (vec (repeat n 1.0)) (vec (link/download executable output-id)))
                "source constructor zeroing must occur on every replay, not only acquisition")
            (is (= [12.0] (vec (link/download executable early-id)))
                "reused roots receive the changed public input")
            (is (= 3 (count (get-in @(:execution-state executable) [:ordered :roots]))))
            (is (empty? (get-in @(:execution-state executable) [:ordered :prepared]))))
          (finally (link/close! executable)))
        (is (true? (:closed? @session)))))))

(defn- check-failed-root! [target]
  (doseq [[v n expected] [[#'fixture/write-before-shape Long/MAX_VALUE ArithmeticException]
                          [#'source-zero-observed -1 NegativeArraySizeException]]]
    (let [compilation (equation-first/compile v {:target target :dtype :double})
          executable (link/instantiate-ordered-invocation!
                      compilation [(double-array [7.0]) (double-array 1) n] {})
          session (:session executable)]
      (try
        (is (instance? expected (try (link/run! executable) nil (catch Throwable e e))))
        (let [runtime (:ordered @(:execution-state executable))
              written (get-in runtime [:current :realized :buffers 'written])
              resident (get-in runtime [:node-views written])
              actual (double-array 1)]
          (is (= 2 (count @(:ledger runtime))))
          (is (nil? (:resident-boundary @(:execution-state executable))))
          (is (zero? @(:completed-replays executable)))
          (is (false? @(:output-ready? executable)))
          (is (= :link-execution-poisoned (reason #(link/outputs executable))))
          (is (= :link-execution-poisoned (reason #(link/run! executable))))
          (gpu/download-range! session resident actual {:elements 1})
          (is (= [8.0] (vec actual)) "the late failure cannot erase the completed earlier write"))
        (finally (link/close! executable)))
      (is (true? (:closed? @session))))))

(defn- check-shared-check-and-admission! [target]
  (let [compilation (equation-first/compile #'fixture/write-before-shape
                                           {:target target :dtype :double})
        executable (link/instantiate-ordered-invocation!
                    compilation [(double-array [7.0]) (double-array 1) 3] {})
        evaluate scalar/evaluate-invocation-step
        calls (atom 0)]
    (try
      (with-redefs [scalar/evaluate-invocation-step
                    (fn [& args] (swap! calls inc) (apply evaluate args))]
        (let [output-id (first (keys (link/run! executable)))]
          (is (= (vec (repeat 9 1.0)) (vec (link/download executable output-id)))))
        (is (= 1 @calls) "final whole-plan validation uses the same cached source result"))
      (finally (link/close! executable))))
  (let [compilation (equation-first/compile #'ordered-void! {:target target :dtype :double})
        out (double-array 1)
        executable (link/instantiate-ordered-invocation! compilation [out] {})]
    (try
      (is (= :link-output-lease-ownership (reason #(link/run-and-lease! executable))))
      (is (zero? @(:completed-replays executable)))
      (is (empty? (:buffers @(:session executable))) "rejected leased execution performs no upload/graph")
      (is (= [0.0] (vec out)))
      (finally (link/close! executable)))))

(defn- check-math-request! [target]
  (let [request {:scalar-math {:overrides {[:tanh :float] :f64-target-library-rte-f32}}}
        compilation (equation-first/compile #'selected-source-math
                                           (merge {:target target :dtype :float} request))
        x (float-array [-1.0 -0.0 0.25 1.0])
        executable (link/instantiate-ordered-invocation! compilation [x 4] request)
        prepare parallel-program/prepare-sequence-with!
        seen (atom [])]
    (try
      (with-redefs [parallel-program/prepare-sequence-with!
                    (fn [instances executor actual-request]
                      (swap! seen conj actual-request)
                      (prepare instances executor actual-request))]
        (let [output-id (first (keys (link/run! executable)))]
          (is (= (vec (selected-source-math x 4)) (vec (link/download executable output-id)))))
        (is (= [request] @seen) "each stage retains independently supplied caller math intent"))
      (finally (link/close! executable)))))

(defn- check-release-failure! [target]
  (let [compilation (equation-first/compile #'source-zero-observed {:target target :dtype :double})
        executable (link/instantiate-ordered-invocation!
                    compilation [(double-array [7.0]) (double-array 1) 3] {})
        release gpu/release-kernel-graph!
        attempts (atom 0)
        failure (ex-info "injected retry-safe stage release" {:cleanup-retry-safe? true})]
    (with-redefs [gpu/release-kernel-graph!
                  (fn [& args]
                    (if (= 1 (swap! attempts inc)) (throw failure) (apply release args)))]
      (try
        (is (identical? failure (try (link/run! executable) nil (catch Throwable e e))))
        (is (= 1 (count (get-in @(:execution-state executable) [:ordered :prepared]))))
        (is (nil? (:resident-boundary @(:execution-state executable))))
        (is (zero? @(:completed-replays executable)))
        (is (= :link-execution-poisoned (reason #(link/run! executable))))
        (finally (link/close! executable)))
      (is (= 2 @attempts) "retry uses the same prepared Cleanup owner")
      (is (empty? (get-in @(:execution-state executable) [:ordered :prepared])))
      (is (true? (:closed? @(:session executable)))))))

(deftest opencl-owned-root-obeys-source-order-and-replay
  (if @ocl-probe/opencl-available?
    (do (check-owned-root! :ocl:0) (check-failed-root! :ocl:0)
        (check-shared-check-and-admission! :ocl:0) (check-math-request! :ocl:0)
        (check-release-failure! :ocl:0))
    (ocl-probe/opencl-skip! "owned source-ordered invocation root")))

(deftest level-zero-owned-root-obeys-source-order-and-replay
  (if @ze-probe/gpu-available?
    (do (check-owned-root! :ze:0) (check-failed-root! :ze:0)
        (check-shared-check-and-admission! :ze:0) (check-math-request! :ze:0)
        (check-release-failure! :ze:0))
    (ze-probe/gpu-skip! "owned source-ordered invocation root")))
