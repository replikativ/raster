(ns raster.gpu.checked-equation-test
  "The actual source-derived checked-map schedule through confined graph ownership."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.fixtures.mixed-storage :as storage]
            [raster.compiler.ir.emitted-parallel-equation :as emitted]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.device-probe :as opencl]
            [raster.runtime.checked-stage-result :as observation]
            [raster.dl.gpu-grad-parity :as level-zero]))

(defn- equation
  ([target] (equation target #'storage/mixed-scale))
  ([target function]
  (->> (:equations (:emitted (equation-first/compile
                             function
                             (merge storage/policy {:target target :dtype :double}))))
       (remove #(get-in % [:attributes :host-only])) first :operations first)))

(defn- thrown-description [f]
  (try (f) nil (catch Throwable error [(class error) (.getMessage error)])))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- check-device [target]
  (let [equation (equation target)
        graph (:graph equation)
        {:keys [extent]} (first (:stage-results (emitted/validate-with-result-contracts equation)))
        input (:id (first (:inputs graph)))
        output (:id (first (:outputs graph)))
        coefficient (:id (first (filter #(= :double (:dtype %)) (:scalars graph))))
        width 4
        consumer (atom 0)]
    (gpu/with-gpu-session [session target]
      (gpu/alloc! session {:input [:float width nil] :output [:float width nil]})
      (doseq [[factor values] [[2.0 [1.5 -2.0 0.0 -0.0]]
                               [2.0 [Float/MAX_VALUE 1.0 (- Float/MAX_VALUE) 0.0]]
                               [0.5 [Float/MAX_VALUE Float/MIN_VALUE Float/NaN -0.0]]
                               [Double/POSITIVE_INFINITY [1.0 -1.0 0.0 1.0]]
                               [1.0 [3.0 4.0 5.0 6.0]]]]
        (let [input-array (float-array values)
              expected-error (thrown-description
                              #(mapv (fn [x] (float (* (double factor) (double x)))) input-array))
              before @consumer]
          (gpu/upload! session :input input-array)
          (let [actual-error
                (thrown-description
                 #(do
                    (let [result (gpu/execute-checked-equation!
                                  session equation {input :input output :output}
                                  {coefficient {:type :double :value factor}
                                   extent {:type :long :value width}})]
                      (is (nil? result)))
                    (swap! consumer inc)))]
            (is (= expected-error actual-error) (str target " factor=" factor))
            (is (= (+ before (if expected-error 0 1)) @consumer))
            (when-not actual-error
              (is (= (mapv #(Float/floatToRawIntBits (float (* (double factor) (double %)))) input-array)
                     (mapv #(Float/floatToRawIntBits %) (gpu/download session :output))))))
          (is (empty? (:kernel-graphs @session)) "no private graph/temporary owner escapes")
          (is (empty? (:events @session)) "completion events retire before return")))
      (let [failure (ex-info "injected observation failure" {:reason :injected-observation-failure})]
        (with-redefs [observation/observe! (fn [& _] (throw failure))]
          (is (identical?
               failure
               (try
                 (gpu/execute-checked-equation!
                  session equation {input :input output :output}
                  {coefficient {:type :double :value 1.0}
                   extent {:type :long :value width}})
                 nil
                 (catch Throwable error error)))))
        (is (empty? (:kernel-graphs @session)) "observation failure retires private storage")
        (is (empty? (:events @session)) "observation failure retains no event"))
      ;; Empty logical maps do not launch or read an uninitialized status lane.
      (with-redefs [gpu/run-kernel-graph! (fn [& _] (throw (AssertionError. "empty stage launched")))]
        (is (nil? (gpu/execute-checked-equation!
                   session equation {input :input output :output}
                   {coefficient {:type :double :value 2.0}
                    extent {:type :long :value 0}}))))
      (is (empty? (:kernel-graphs @session))))))

(deftest actual-checked-map-observation-preserves-source-results-and-failure-order
  (if @level-zero/gpu-available?
    (check-device :ze:0)
    (level-zero/gpu-skip! "source-derived checked equation on Level Zero"))
  (if @opencl/opencl-available?
    (check-device :ocl:0)
    (opencl/opencl-skip! "source-derived checked equation on OpenCL")))

(defn- check-library-device [target]
  (let [equation (equation target #'storage/double-exponential-float-storage)
        graph (:graph equation)
        input (:id (first (:inputs graph)))
        output (:id (first (:outputs graph)))
        extent (:extent (first (:stage-results (emitted/validate-with-result-contracts equation))))]
    (gpu/with-gpu-session [session target]
      (gpu/alloc! session {:input [:float 4 nil] :output [:float 4 nil]})
      (doseq [values [[-1.0 0.0 1.0 Float/NaN] [0.0 80.0 Float/NEGATIVE_INFINITY 1.0]
                      [90.0 0.0 -1.0 1.0] [Float/POSITIVE_INFINITY 0.0 -1.0 1.0]]]
        (let [array (float-array values)
              oracle #(mapv (fn [x] (float (Math/exp (double x)))) array)
              expected-error (first (thrown-description oracle))]
          (gpu/upload! session :input array)
          (let [actual-error (first (thrown-description
                                    #(gpu/execute-checked-equation!
                                      session equation {input :input output :output}
                                      {extent {:type :long :value 4}})))]
            (is (= expected-error actual-error))
            (when-not actual-error
              ;; The retained target-library realization is implementation-defined,
              ;; not a bitwise libm promise. The check applies to the actual
              ;; target result; cases here stay far from the Float threshold.
              (doseq [[expected actual] (map vector (oracle) (gpu/download session :output))]
                (is (if (Float/isNaN expected) (Float/isNaN actual)
                        (<= (Math/abs (- (double expected) (double actual)))
                            (* 2.0e-6 (max 1.0 (Math/abs (double expected))))))))))
          (is (empty? (:kernel-graphs @session)))
          (is (empty? (:events @session))))))))

(deftest primitive-library-math-preserves-checked-observation-on-both-backends
  (if @level-zero/gpu-available?
    (check-library-device :ze:0)
    (level-zero/gpu-skip! "checked exponential on Level Zero"))
  (if @opencl/opencl-available?
    (check-library-device :ocl:0)
    (opencl/opencl-skip! "checked exponential on OpenCL")))

(defn- check-ordered-device [target]
  (let [compilation (equation-first/compile #'storage/mixed-scale
                                           (merge storage/policy {:target target :dtype :double}))
        input (float-array [1.0 -2.0 0.0])
        executable (link/instantiate-checked-invocation! compilation [2.0 input])
        session (:session executable)]
    (try
      (with-redefs [gpu/make-session (fn [& _] (throw (AssertionError. "premature driver contact")))]
        (is (= :checked-stage-observation-required
               (reason #(link/instantiate-ordered-invocation! compilation [2.0 input] {}))))
        (is (= :ordered-source-owner-mismatch
               (reason #(link/instantiate-checked-invocation!
                         (dissoc compilation :ordered-source-owner) [2.0 input])))))
      (let [first-result (link/run! executable)
            output (first (keys first-result))
            old-view (get first-result output)
            first-occurrences (get-in @(:execution-state executable) [:ordered :observed])]
        (is (= [2.0 -4.0 0.0] (vec (link/download executable output))))
        (is (= 1 (count first-occurrences)))
        (with-open [lease (link/output-lease! executable)]
          (is (= :link-output-lease-active (reason #(link/run! executable))))
          (is (= 1 @(:completed-replays executable)))
          (is (= (:key old-view) (:key (get (:outputs @lease) output))))
          (is (= [2.0 -4.0 0.0] (vec (link/download executable output)))))
        (aset input 0 (float 3.0))
        (let [next-result (link/run! executable)
              next-view (get next-result output)
              next-occurrences (get-in @(:execution-state executable) [:ordered :observed])]
          (is (= [6.0 -4.0 0.0] (vec (link/download executable output))))
          (is (not= (:key old-view) (:key next-view)))
          (is (not= first-occurrences next-occurrences) "observations belong to one replay epoch")
          (is (thrown? clojure.lang.ExceptionInfo
                       (gpu/download-range! session old-view (float-array 3) {:elements 3}))))
        (is (= 2 @(:completed-replays executable)))
        (aset input 0 Float/MAX_VALUE)
        (is (= (thrown-description #(float (* (double 2.0) (double Float/MAX_VALUE))))
               (thrown-description #(link/run! executable))))
        (is (= 2 @(:completed-replays executable)))
        (is (false? @(:output-ready? executable)))
        (is (nil? (:resident-boundary @(:execution-state executable))))
        (is (empty? (get-in @(:execution-state executable) [:ordered :observed])))
        (is (some? (thrown-description #(link/outputs executable))))
        (is (some? (thrown-description #(link/run! executable))))
        (is (empty? (:kernel-graphs @session)))
        (is (empty? (:events @session))))
      (finally (link/close! executable)))
    (is (true? (:closed? @session)))
    (let [executable (link/instantiate-checked-invocation! compilation [2.0 (float-array [1.0])])
          session (:session executable)
          failure (ex-info "injected ordered observation failure" {:reason :injected-observation-failure})]
      (try
        (with-redefs [observation/observe! (fn [& _] (throw failure))]
          (is (identical? failure (try (link/run! executable) nil (catch Throwable error error)))))
        (is (zero? @(:completed-replays executable)))
        (is (empty? (get-in @(:execution-state executable) [:ordered :observed])))
        (is (nil? (:resident-boundary @(:execution-state executable))))
        (is (= :link-execution-poisoned (reason #(link/run! executable))))
        (is (empty? (:kernel-graphs @session)))
        (is (empty? (:events @session)))
        (finally (link/close! executable))))))

(deftest ordered-checked-results-publish-only-after-observation-and-use-private-generations
  (if @level-zero/gpu-available?
    (check-ordered-device :ze:0)
    (level-zero/gpu-skip! "ordered checked result on Level Zero"))
  (if @opencl/opencl-available?
    (check-ordered-device :ocl:0)
    (opencl/opencl-skip! "ordered checked result on OpenCL")))

(defn- check-private-void-store! [target]
  (doseq [n [0 3]
          coefficient (cond-> [2.0 -0.5] (zero? n) (conj Double/MAX_VALUE))]
    (let [input (float-array (take n [1.5 -2.0 0.25]))
          expected (mapv (fn [offset]
                           (storage/checked-scale-effect-result
                            coefficient (float-array (map #(+ (double %) offset) input)) n))
                         [0.0 0.125])
          compilation (equation-first/compile #'storage/checked-scale-effect-result
                                               (merge storage/policy {:target target :dtype :float}))
          executable (link/instantiate-checked-invocation! compilation [coefficient input n])
          session (:session executable)]
      (try
        (doseq [replay [0 1]]
          (when (= replay 1)
            (dotimes [index n] (aset input index (float (+ (double (aget input index)) 0.125)))))
          (let [output (first (keys (link/run! executable)))
                actual (link/download executable output)]
            (is (= (class (nth expected replay)) (class actual)))
            (is (= (vec (nth expected replay)) (vec actual)))
            (is (= 1 (count (get-in @(:execution-state executable) [:ordered :observed]))))))
        (when (pos? n)
          (aset input 0 Float/POSITIVE_INFINITY)
          (is (thrown? IllegalArgumentException (link/run! executable)))
          (is (= 2 @(:completed-replays executable)))
          (is (false? @(:output-ready? executable)))
          (is (empty? (get-in @(:execution-state executable) [:ordered :observed])))
          (is (= :link-execution-poisoned (reason #(link/outputs executable))))
          (is (= :link-execution-poisoned (reason #(link/run! executable)))))
        (finally (link/close! executable)))
      (is (true? (:closed? @session))))))

(deftest void-helper-private-output-is-observed-before-publication-on-local-devices
  (if @level-zero/gpu-available?
    (check-private-void-store! :ze:0)
    (level-zero/gpu-skip! "private checked Void store on Level Zero"))
  (if @opencl/opencl-available?
    (check-private-void-store! :ocl:0)
    (opencl/opencl-skip! "private checked Void store on OpenCL")))

(defn- check-fold-device [target]
  ;; Materialize the JVM oracle before retaining historical compilation owners.
  (is (= [1.0] (vec (storage/double-fold-float-storage
                      (float-array [16777216.0 1.0 -16777216.0])))))
  (let [compiled (equation-first/compile #'storage/double-fold-float-storage
                                        (merge storage/policy {:target target :dtype :double}))
        input (float-array [16777216.0 1.0 -16777216.0])
        executable (link/instantiate-checked-invocation! compiled [input])]
    (try
      (doseq [values [[16777216.0 1.0 -16777216.0] [1.0 2.0 3.0]
                      [Float/MAX_VALUE (- Float/MAX_VALUE) 1.0]]]
        (dotimes [i 3] (aset input i (float (nth values i))))
        (let [output (first (keys (link/run! executable)))
              expected (float (double (reduce + 0.0 (map double input))))]
          (is (= [expected] (vec (link/download executable output))))))
      (aset input 0 Float/MAX_VALUE)
      (aset input 1 Float/MAX_VALUE)
      (is (= (thrown-description #(float (* (double Float/MAX_VALUE) 2.0)))
             (thrown-description #(link/run! executable))))
      (is (= 3 @(:completed-replays executable)))
      (is (false? @(:output-ready? executable)))
      (is (nil? (:resident-boundary @(:execution-state executable))))
      (is (empty? (get-in @(:execution-state executable) [:ordered :observed])))
      (finally (link/close! executable)))))

(deftest checked-cast-after-ordered-fold-executes-with-original-double-precision
  (if @level-zero/gpu-available?
    (check-fold-device :ze:0)
    (level-zero/gpu-skip! "checked ordered fold on Level Zero"))
  (if @opencl/opencl-available?
    (check-fold-device :ocl:0)
    (opencl/opencl-skip! "checked ordered fold on OpenCL")))

(defn- check-dynamic-fold-device [target]
  (let [cases [[16777216.0 1.0 -16777216.0] [1.0 2.0 3.0]
               [Float/MAX_VALUE (- Float/MAX_VALUE) 1.0]]
        ;; Freeze every source oracle before retaining the compilation history.
        expected (into {} (for [n [0 1 3] values cases]
                            [[n values] (vec (storage/dynamic-double-fold-float-storage
                                              (float-array values) n))]))
        compiled (equation-first/compile #'storage/dynamic-double-fold-float-storage
                                         (merge storage/policy {:target target :dtype :double}))]
    (doseq [n [0 1 3]]
      (let [input (float-array 3)
            executable (link/instantiate-checked-invocation! compiled [input n])
            session (:session executable)]
        (try
          (doseq [values cases]
            (dotimes [i 3] (aset input i (float (nth values i))))
            (let [output (first (keys (link/run! executable)))]
              (is (= (get expected [n values]) (vec (link/download executable output))))))
          (is (= 3 @(:completed-replays executable)))
          (when (= n 3)
            (aset input 0 Float/MAX_VALUE)
            (aset input 1 Float/MAX_VALUE)
            (aset input 2 (float 0.0))
            (is (= (thrown-description #(float (* (double Float/MAX_VALUE) 2.0)))
                   (thrown-description #(link/run! executable))))
            (is (= 3 @(:completed-replays executable)))
            (is (false? @(:output-ready? executable)))
            (is (nil? (:resident-boundary @(:execution-state executable))))
            (is (empty? (get-in @(:execution-state executable) [:ordered :observed])))
            (is (= :link-execution-poisoned (reason #(link/outputs executable))))
            (is (= :link-execution-poisoned (reason #(link/run! executable))))
            (is (empty? (:kernel-graphs @session)))
            (is (empty? (:events @session))))
          (finally (link/close! executable)))
        (is (true? (:closed? @session)))))
    ;; Header termination does not grant read capacity. This separate source
    ;; obligation must fail before any graph is submitted to the device.
    (let [executable (link/instantiate-checked-invocation! compiled [(float-array 3) 4])
          session (:session executable)]
      (try
        (with-redefs [gpu/run-kernel-graph!
                      (fn [& _] (throw (AssertionError. "out-of-capacity graph submitted")))]
          (is (= :program-link-graph-range (reason #(link/run! executable)))))
        (is (zero? @(:completed-replays executable)))
        (is (false? @(:output-ready? executable)))
        (is (empty? (get-in @(:execution-state executable) [:ordered :observed])))
        (finally (link/close! executable)))
      (is (true? (:closed? @session))))))

(deftest dynamic-ordered-fold-keeps-double-carries-and-checked-publication
  (if @level-zero/gpu-available?
    (check-dynamic-fold-device :ze:0)
    (level-zero/gpu-skip! "dynamic checked ordered fold on Level Zero"))
  (if @opencl/opencl-available?
    (check-dynamic-fold-device :ocl:0)
    (opencl/opencl-skip! "dynamic checked ordered fold on OpenCL")))
