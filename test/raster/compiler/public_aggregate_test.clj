(ns raster.compiler.public-aggregate-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm defvalue]]
            [raster.par :as par]
            [raster.compiler.equation-first :as equation]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.value :as value]))

(defvalue ArrayState [positions :- (Array float), labels :- (Array int)])
(defvalue Particle [x :- Float, id :- Long])

(deftm project-particle!
  [particles :- ParticleSoA, out :- (Array float), n :- Long] :- (Array float)
  (par/map-void! i n (aset out i (.x (aget particles i))))
  out)

(deftm project-state!
  [state :- ArrayState, out :- (Array float), n :- Long] :- (Array float)
  (let [positions (.-positions state) labels (.-labels state)]
    (par/map-void! i n
      (aset out i (+ (aget positions i) (float (aget labels i))))))
  out)

(deftm write-state!
  [state :- ArrayState, n :- Long] :- Void
  (par/map-void! i n (aset (.-positions state) i 0.0)))

(defn- state [offset]
  (->ArrayState (float-array (map #(+ offset (* 0.25 %)) (range 17)))
               (int-array (map #(mod % 3) (range 17)))))

(defn- oracle [state]
  (let [out (float-array 17)] (project-state! state out 17) (vec out)))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest public-projection-retains-logical-order-and-checked-physical-leaves
  (let [compilation (equation/compile #'project-state! {:target :cuda:0 :dtype :float})
        plan (get-in compilation [:semantic :attributes :invocation-plan])
        projection (get-in plan [:attributes :parameter-projection])
        input (state 1.0)
        out (float-array 17)
        args (materialization/parameter-arguments plan [input out 17])]
    (is (= '[state out n] (:public-parameters projection)))
    (is (= '[state_positions state_labels out n] (mapv :symbol (:parameters plan))))
    (is (= [:positions :labels] (mapv :field (take 2 (:physical-parameters projection)))))
    (is (identical? (.-positions input) (first args)))
    (is (identical? (.-labels input) (second args)))
    (is (identical? out (nth args 2)))
    (is (= :none (get-in compilation [:stats :fallback])))
    (is (= :invocation-materialization-aggregate-class
           (reason #(materialization/parameter-arguments plan [{:positions (float-array 17)} out 17]))))
    (is (= :invocation-materialization-arity
           (reason #(materialization/parameter-arguments plan [input out]))))
    (is (= :invocation-parameter-projection
           (reason #(invocation/validate!
                     (assoc-in plan [:attributes :parameter-projection :physical-parameters 0 :tag] 'doubles)))))
    (is (= :invocation-parameter-projection
           (reason #(invocation/validate!
                     (assoc-in plan [:attributes :parameter-projection :physical-parameters 1 :field] :positions)))))
    (is (= :invocation-parameter-projection
           (reason #(invocation/validate!
                     (assoc-in plan [:attributes :parameter-projection :public-parameters] '[out state n])))))
    (is (= :invocation-materialization-buffer-dtype
           (reason #(equation/lower compilation
                                   [(->ArrayState (double-array 17) (int-array 17)) out 17]))))))

(deftest template-identity-tracks-only-relevant-aggregate-declarations
  (let [identity #(#'compiled/source-specialization-identity #'project-state! :float)
        original (get-in (identity) [:semantic :parameter-projection])
        order @inference/field-order-registry]
    (with-redefs [inference/field-order-registry (atom (assoc order 'ArrayState ["labels" "positions"]))]
      (is (not= original (get-in (identity) [:semantic :parameter-projection]))))
    (with-redefs [inference/field-order-registry (atom (assoc order 'UnrelatedState ["other"]))]
      (is (= original (get-in (identity) [:semantic :parameter-projection]))))))

(defn- run-device [target]
  (let [input (state 1.0)
        args [input (float-array (repeat 17 Float/NaN)) 17]
        options {:compiler :equation-first :target target :dtype :float}
        prepared (compiled/lower #'project-state! args options)]
    (is (= 0 (get-in (compiled/plan prepared) [:attributes :driver-allocations])))
    (is (= '[state out n] (get-in prepared [:descriptor :all-params])))
    (is (= #{[:state :positions] [:state :labels] :out} (set (map :key (:in-tree prepared)))))
    (let [program (compiled/instantiate! prepared)]
      (try
        (let [result (program {})]
          (is (= (oracle input) (vec (value/->host (:result result))))))
        (let [changed (state 3.0)
              result (program {:state changed})]
          (is (= (oracle changed) (vec (value/->host (:result result))))))
        (is (= :invocation-materialization-aggregate-class (reason #(program {:state {}}))))
        (is (= :compiled-aggregate-input-conflict
               (reason #(program {:state input [:state :positions] (.-positions input)}))))
        (finally (compiled/close! program))))
    (is (= :compiled-aggregate-output
           (reason #(compiled/lower #'project-state! args (assoc options :donate '[state])))))
    (is (= :compiled-aggregate-write
           (reason #(compiled/lower #'write-state! [input 17] options))))
    (is (= :compiled-aggregate-write
           (reason #(compiled/lower #'write-state! [input 17] (assoc options :roles {'state :input})))))
    (let [constant (compiled/lower #'project-state! args (assoc options :constants '[state]))
          program (compiled/instantiate! constant)]
      (try
        (is (= (oracle input) (vec (value/->host (:result (program {}))))))
        (is (= :compiled-aggregate-input-role (reason #(program {:state (state 4.0)}))))
        (finally (compiled/close! program))))
    (let [warm (compiled/lower #'project-state! [(state 2.0) (float-array 17) 17] options)]
      (is (true? (get-in (compiled/preparation-report warm) [:template :cache-hit?]))))
    (let [particles (->ParticleSoA (float-array (range 17)) (long-array (range 17)))
          prepared (compiled/lower #'project-particle! [particles (float-array 17) 17] options)
          program (compiled/instantiate! prepared)]
      (try
        (let [result (program {})]
          (is (= (mapv float (range 17)) (vec (value/->host (:result result))))))
        (finally (compiled/close! program))))))

(deftest public-aggregate-on-opencl
  (if @opencl/opencl-available? (run-device :ocl:0)
      (opencl/opencl-skip! "public-aggregate-opencl")))

(deftest public-aggregate-on-level-zero
  (if @ze/gpu-available? (run-device :ze:0)
      (ze/gpu-skip! "public-aggregate-level-zero")))
