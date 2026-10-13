(ns raster.gpu.public-lora-training-test
  "Public whole-program AD/SGD with two donated adapters and changing data.
   Tiny capability witness, not the unchanged real-weight model acceptance gate."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.fixtures.lora-training :as fixture]
            [raster.ad.reverse :as reverse]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.value :as value]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]))

(deftest analytic-low-rank-gradient-agrees-with-finite-differences-and-ad
  (let [{:keys [A B x tgt rows in r out] :as input} (fixture/inputs)
        {:keys [loss dA dB]} (fixture/oracle input)
        vg ((reverse/value+grad #'fixture/objective :wrt [1 2]) x A B tgt rows in r out)
        h 1.0e-5]
    (is (< (Math/abs (- loss (double (first vg)))) 1.0e-8))
    (is (< (fixture/max-error dA (nth vg 2)) 1.0e-7))
    (is (< (fixture/max-error dB (nth vg 3)) 1.0e-7))
    (doseq [[key gradient] [[:A dA] [:B dB]]
            i (range (count gradient))]
      (let [parameters (double-array (get input key))
            original (aget parameters i)
            evaluate (fn [offset]
                       (aset parameters i (+ original offset))
                       (:loss (fixture/oracle (assoc input key parameters))))
            finite-difference (/ (- (evaluate h) (evaluate (- h))) (* 2.0 h))]
        (is (< (Math/abs (- finite-difference (aget ^doubles gradient i))) 1.0e-8))))))

(deftest every-stale-startup-input-distinguishes-the-correct-update-trajectory
  (let [initial (fixture/inputs)
        keys [:A :B :x :tgt]
        captured (mapv #(vec (get initial %)) keys)
        ;; Same three batches as the device trajectory, using only the
        ;; independent coordinate-loop oracle and rounded adapter storage.
        trajectory (vec (reductions
                         (fn [state step]
                           (fixture/update-oracle
                            (merge state (select-keys (fixture/inputs step) [:x :tgt]))))
                         initial (range 3)))]
    (doseq [step [1 2]
            :let [current (merge (nth trajectory step)
                                 (select-keys (fixture/inputs step) [:x :tgt]))
                  before (mapv #(vec (get current %)) keys)
                  correct (nth trajectory (inc step))]
            key keys]
      ;; Replace exactly one input, holding every other current input fixed.
      ;; This tests stale adapters separately from stale batches/targets;
      ;; simultaneous changes cannot mask an insensitive fixture.
      (let [stale-input (assoc current key (get initial key))
            stale-update (fixture/update-oracle stale-input)
            error (max (fixture/max-error (:A correct) (:A stale-update))
                       (fixture/max-error (:B correct) (:B stale-update)))]
        (is (not= (vec (get current key)) (vec (get stale-input key)))
            (str "step " step " changes " key " from startup"))
        (is (> error 2.0e-7)
            (str "step " step " stale " key " must exceed the unchanged device-update tolerance"))
        (is (= before (mapv #(vec (get current %)) keys))
            "counterfactual oracle evaluation never mutates the correct trajectory")))
    (is (= captured (mapv #(vec (get initial %)) keys))
        "startup arrays remain unchanged by every oracle/counterfactual")))

(defn- check-public-training! [target]
  (let [initial (fixture/inputs)
        captured (mapv #(vec (get initial %)) [:A :B :x :tgt])
        prepared (with-redefs [gpu/make-session (fn [& _] (throw (AssertionError. "lower opened a session")))
                              gpu/alloc! (fn [& _] (throw (AssertionError. "lower allocated")))
                              link/instantiate! (fn [& _] (throw (AssertionError. "lower instantiated")))]
                   (compiled/lower #'fixture/train-step (fixture/arguments initial)
                                   {:compiler :equation-first :target target :dtype :float
                                    :donate '[A B] :outputs '[A B]}))
        instances (get-in prepared [:lowering :plan :instances])]
    (is (= 1 (count instances)))
    (is (every? link-plan/program-link-instance? instances))
    (is (= {:A :A' :B :B'} (:donated prepared)))
    (let [artifact (compiled/instantiate! prepared)
          latest (atom nil)]
      (try
        (loop [step 0 expected initial]
          (when (< step 3)
            (let [{:keys [x tgt]} (fixture/inputs step)
                  data-before [(vec x) (vec tgt)]
                  expected (fixture/update-oracle (assoc expected :x x :tgt tgt))
                  previous @latest
                  overrides (cond-> {:x x :tgt tgt}
                              previous (assoc :A (:A' previous) :B (:B' previous)))
                  actual (artifact overrides)]
              (reset! latest actual)
              (is (= #{:A' :B'} (set (keys actual))))
              (is (= data-before [(vec x) (vec tgt)]) "fresh host data remain read-only")
              (doseq [[input-key output-key] [[:A :A'] [:B :B']]]
                (is (< (fixture/max-error (get expected input-key)
                                         (value/->host (get actual output-key))) 2.0e-7))
                (when previous (is (not (value/live? (get previous output-key)))))
                (is (value/live? (get actual output-key))))
              (recur (inc step) expected))))
        (is (= captured (mapv #(vec (get initial %)) [:A :B :x :tgt]))
            "captured host arrays are never mutated by device donation")
        (compiled/close! artifact)
        (is (every? (complement value/live?) (vals @latest)))
        (finally (compiled/close! artifact))))))

(deftest changed-batches-update-both-resident-adapters-through-the-public-api
  (if @opencl/opencl-available?
    (check-public-training! :ocl:0)
    (opencl/opencl-skip! "public plain LoRA training"))
  (if @ze/gpu-available?
    (check-public-training! :ze:0)
    (ze/gpu-skip! "public plain LoRA training")))

(def ^:private adam-state-keys [:A :B :mA :vA :mB :vB])

(defn- check-public-joint-clipped-adam! [target]
  (let [input (fixture/adam-inputs)
        expected (fixture/joint-clipped-adam-oracle input)
        host-before (into {} (map (fn [key] [key (vec (get input key))]))
                          (concat adam-state-keys [:x :tgt]))
        prepared
        (with-redefs [gpu/make-session (fn [& _] (throw (AssertionError. "lower opened a session")))
                      gpu/alloc! (fn [& _] (throw (AssertionError. "lower allocated")))
                      link/instantiate! (fn [& _] (throw (AssertionError. "lower instantiated")))]
          (compiled/lower
           #'fixture/joint-clipped-lora-adam-step (fixture/adam-arguments input)
           {:compiler :equation-first :target target :dtype :float
            :roles {'A :state 'B :state 'mA :state 'vA :state 'mB :state 'vB :state}
            :outputs '[A B mA vA mB vB]}))]
    (is (= #{:A :B :mA :vA :mB :vB :result}
           (set (map :key (:out-tree prepared)))))
    (is (< 0.0 (:scale expected) 0.01)
        "the independent joint-norm oracle proves clipping is active")
    (let [artifact (compiled/instantiate! prepared)
          latest (atom nil)]
      (try
        (let [actual (artifact {})
              host (into {} (map (fn [key] [key (value/->host (get actual key))]))
                         adam-state-keys)
              loss (double (first (value/->host (:result actual))))]
          (reset! latest actual)
          (is (= #{:A :B :mA :vA :mB :vB :result} (set (keys actual))))
          (is (every? value/live? (vals actual)))
          (is (= (:loss expected) loss)
              "the public rank-zero Double result is the pre-update objective")
          (doseq [key [:A :B]]
            (is (< (fixture/max-error (get expected key) (get host key)) 2.0e-7)
                (str (name key) " matches the independently clipped Adam update")))
          (doseq [key [:mA :mB]]
            (is (< (fixture/max-error (get expected key) (get host key)) 1.0e-10)
                (str (name key) " matches the independent first moment")))
          (doseq [key [:vA :vB]]
            (is (< (fixture/max-error (get expected key) (get host key)) 1.0e-16)
                (str (name key) " matches the independent second moment")))
          (doseq [key [:mA :vA :mB :vB]]
            (is (some #(not (zero? (float %))) (get host key))
                (str (name key) " is updated, not merely finite zero storage"))
            (is (every? #(Float/isFinite (float %)) (get host key))
                (str (name key) " remains finite")))
          (is (= host-before
                 (into {} (map (fn [key] [key (vec (get input key))]))
                       (concat adam-state-keys [:x :tgt])))
              "captured host data and optimizer state remain untouched"))
        (compiled/close! artifact)
        (is (every? (complement value/live?) (vals @latest)))
        (finally (compiled/close! artifact))))))

(deftest one-public-joint-clipped-adam-step-returns-loss-and-all-state
  ;; `adam-t` is captured at one in this bounded witness. Replaying it would not qualify dynamic
  ;; Adam time progression; that requires a separately supported public scalar/buffer boundary.
  (if @opencl/opencl-available?
    (check-public-joint-clipped-adam! :ocl:0)
    (opencl/opencl-skip! "public joint-clipped LoRA Adam step"))
  (if @ze/gpu-available?
    (check-public-joint-clipped-adam! :ze:0)
    (ze/gpu-skip! "public joint-clipped LoRA Adam step")))
