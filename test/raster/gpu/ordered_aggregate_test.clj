(ns raster.gpu.ordered-aggregate-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.par :as par]
            [raster.compiler.public-aggregate-test :as aggregate]
            [raster.compiler.backend.jvm.typed-scalar :as scalar]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.value :as value])
  (:import [raster.compiler.public_aggregate_test ArrayState]))

(deftm ordered-record-step
  [state :- ArrayState sample :- ArrayState n :- Long] :- (Array float)
  (let [positions (.-positions state) labels (.-labels state)
        samples (.-positions sample) sample-labels (.-labels sample)
        written (par/map! positions i 1 float (+ (aget positions i) (float 1.0)))
        ^long count (* (long n) (long n))
        out (float-array count)]
    (par/map! out i count float
      (+ (+ (aget positions i) (aget samples i))
         (+ (float (aget labels i)) (float (aget sample-labels i)))))))

(deftm ordered-tree-step
  [state :- (HMap {:buffers (HVec [(Array float) (Array int)]) :offset Float})
   sample :- (HMap {:buffers (HVec [(Array float) (Array int)]) :offset Float})
   n :- Long] :- (Array float)
  (let [positions (nth (:buffers state) 0) labels (nth (:buffers state) 1)
        samples (nth (:buffers sample) 0) sample-labels (nth (:buffers sample) 1)
        offset (+ (:offset state) (:offset sample))
        written (par/map! positions i 1 float (+ (aget positions i) (float 1.0)))
        ^long count (* (long n) (long n))
        out (float-array count)]
    (par/map! out i count float
      (+ offset (+ (+ (aget positions i) (aget samples i))
                   (+ (float (aget labels i)) (float (aget sample-labels i))))))))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest captured-scalar-invariance-preserves-floating-bits-and-carriers
  (let [data (float-array 1)
        leaf {:binding 'model :symbol 'model_offset :path [:offset] :tag 'float}
        array-leaf {:binding 'model :symbol 'model_data :path [:data] :tag 'floats}
        project (fn [expected actual]
                  (#'compiled/project-aggregate-inputs
                   {'model [{:key [:model :data] :role :input :aggregate-leaf array-leaf
                             :aggregate-scalars {leaf expected}}]}
                   {:model {:data data :offset actual}}))
        nan (Float/intBitsToFloat 0x7fc00001)]
    (is (identical? data (get (project nan (Float/intBitsToFloat 0x7fc00001)) [:model :data])))
    (doseq [[expected actual] [[(float -0.0) (float 0.0)]
                              [(float 0.0) (double 0.0)]
                              [nan (Float/intBitsToFloat 0x7fc00002)]]]
      (is (= :compiled-aggregate-scalar-change (reason #(project expected actual)))))))

(defn- configurations []
  [{:source #'ordered-record-step :positions [:positions] :labels [:labels] :extra 0.0
    :make (fn [positions labels _] (aggregate/->ArrayState positions labels))}
   {:source #'ordered-tree-step :positions [:buffers 0] :labels [:buffers 1] :extra 0.75
    :make (fn [positions labels offset] {:buffers [positions labels] :offset (float offset)})}])

(defn- check-aggregate-replay! [target {:keys [source positions labels extra make]}]
  (let [state-values (float-array [2.0]) labels-values (int-array [3])
        sample-values (float-array [5.0])
        state (make state-values labels-values 0.25)
        sample (make sample-values (int-array [7]) 0.5)
        state-key (into [:state] positions) sample-key (into [:sample] positions)
        args [state sample 1]
        options {:compiler :equation-first :target target :dtype :float
                 :roles {state-key :state} :constants [(into [:state] labels)] :outputs [state-key]}
        forbidden (fn [& _] (throw (AssertionError. "preparation executed source or acquired resources")))
        prepared (with-redefs [gpu/make-session forbidden gpu/alloc! forbidden
                               scalar/evaluate-invocation-step forbidden]
                   (compiled/lower source args options))
        _ (aset state-values 0 (float 4.0))
        _ (aset sample-values 0 (float 6.0))
        artifact (with-redefs [materialization/aggregate-leaf forbidden]
                   (compiled/instantiate! prepared))
        check! (fn [outputs expected-state expected-result]
                 (is (= #{state-key :result} (set (keys outputs))))
                 (is (= [expected-state] (vec (value/->host (get outputs state-key)))))
                 (is (= [(+ expected-result extra)] (vec (value/->host (:result outputs))))))]
    (try
      (is (some? (:ordered-invocation prepared)))
      (is (= '[state sample n] (get-in prepared [:descriptor :all-params])))
      (let [ordered (:ordered @(:execution-state (:executable artifact)))]
        (is (= (count args) (count (:arguments ordered))))
        (is (> (count (:parameter-defaults ordered)) (count args)))
        (is (identical? (:parameter-defaults ordered) (:parameter-arguments ordered))))
      ;; Captured state/constants are resident; subsequent mutations of their host bytes are not replay.
      (aset state-values 0 (float 1000.0))
      (aset labels-values 0 99)
      (let [first (with-redefs [materialization/aggregate-leaf forbidden] (artifact))]
        (check! first 5.0 21.0)
        (doseq [bad [{:sample (make (double-array 1) (int-array [7]) 0.5)}
                     {:sample (make (float-array [9]) (long-array [7]) 0.5)}
                     {:sample (make state-values (int-array [7]) 0.5)}
                     {:sample (make (float-array [9]) (int-array [7]) 0.5)
                      sample-key (float-array [9])}
                     {:state state}]]
          (is (thrown? clojure.lang.ExceptionInfo (artifact bad)))
          (is (every? value/live? (vals first)))
          (check! first 5.0 21.0))
        (if (= positions [:positions])
          (is (= :invocation-materialization-aggregate-class
                 (reason #(artifact {:sample {:positions (float-array 1) :labels (int-array 1)}}))))
          (do
            (is (thrown? clojure.lang.ExceptionInfo (artifact {:sample (dissoc sample :offset)})))
            (is (= :compiled-aggregate-scalar-change
                   (reason #(artifact {:sample (assoc sample :offset (float 2.0))}))))))
        (check! (artifact {:sample (make (float-array [11]) (int-array [13]) 0.5)}) 6.0 33.0)
        (is (not-any? value/live? (vals first)))
        (check! (artifact {sample-key (float-array [17])}) 7.0 34.0)
        (check! (artifact) 8.0 24.0))
      (finally (compiled/close! artifact))))
  (let [state-values (float-array [2])
        state (make state-values (int-array [3]) 0.25)
        sample (make (float-array [5]) (int-array [7]) 0.5)
        state-key (into [:state] positions)
        prepared (compiled/lower source [state sample Long/MAX_VALUE]
                                 {:compiler :equation-first :target target :dtype :float
                                  :roles {state-key :state} :constants [(into [:state] labels)]
                                  :outputs [state-key]})
        artifact (compiled/instantiate! prepared)
        executable (:executable artifact)
        roots (set (keys (get-in @(:execution-state executable) [:ordered :roots])))]
    (try
      (is (instance? ArithmeticException (try (artifact) nil (catch Throwable error error))))
      (let [runtime @(:execution-state executable)
            id (:value (first (filter #(= state-key (:key %)) (:in-tree prepared))))
            token (get-in runtime [:ordered :current :realized :buffers id])
            view (get-in runtime [:ordered :node-views token])
            actual (float-array 1)]
        (is (some? view))
        (when view
          (gpu/download-range! (:session executable) view actual {:elements 1})
          (is (= [3.0] (vec actual))))
        (is (= roots (set (keys (get-in runtime [:ordered :roots]))))))
      (is (nil? @(:live-outputs artifact)))
      (is (zero? @(:completed-replays executable)))
      (finally (compiled/close! artifact)))))

(deftest opencl-ordered-aggregate-entry-replay-and-failure-order
  (if @opencl/opencl-available?
    (doseq [configuration (configurations)] (check-aggregate-replay! :ocl:0 configuration))
    (opencl/opencl-skip! "ordered nominal/tree aggregate entry")))

(deftest level-zero-ordered-aggregate-entry-replay-and-failure-order
  (if @ze/gpu-available?
    (doseq [configuration (configurations)] (check-aggregate-replay! :ze:0 configuration))
    (ze/gpu-skip! "ordered nominal/tree aggregate entry")))
