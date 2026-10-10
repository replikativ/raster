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
