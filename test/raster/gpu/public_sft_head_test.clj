(ns raster.gpu.public-sft-head-test
  "Frozen SFT head with changing loss-row inputs through the public compiler API.
   A tiny capability witness, not real-weight model or large-vocabulary performance acceptance."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.fixtures.sft-head :as fixture]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.value :as value]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]))

(defn- check-close [expected actual tolerance]
  (is (= (count expected) (count actual)))
  (doseq [[index [a b]] (map-indexed vector (map vector expected actual))]
    (is (< (Math/abs (- (double a) (double b))) tolerance) (str "coordinate " index))))

(deftest coordinate-sft-adjoint-agrees-with-finite-differences-and-jvm
  (doseq [batch (range 3)]
    (let [{:keys [y nll g] :as inputs} (fixture/inputs batch)
          expected (fixture/oracle inputs)
          parameters (double-array y)
          h 1.0e-5]
      (dotimes [i (count y)]
        (let [original (aget parameters i)
              loss-at (fn [offset]
                        (aset parameters i (+ original offset))
                        (reduce + 0.0 (:nll (fixture/oracle (assoc inputs :y parameters)))))
              finite-difference (/ (- (loss-at h) (loss-at (- h))) (* 2.0 h))]
          (aset parameters i original)
          (is (< (Math/abs (- finite-difference (aget ^doubles (:g expected) i))) 1.0e-7))))
      (apply fixture/head-step! (fixture/arguments inputs))
      (check-close (:nll expected) nll 1.0e-5)
      (check-close (:g expected) g 1.0e-5))))

(deftest changing-inputs-distinguish-every-stale-public-input
  (doseq [batch [1 2]
          :let [previous (fixture/inputs (dec batch))
                current (fixture/inputs batch)
                expected (fixture/oracle current)]
          key [:y :rows :targets :w]]
    (let [stale (fixture/oracle (assoc current key (get previous key)))
          difference (reduce max 0.0
                             (map #(Math/abs (- (double %1) (double %2)))
                                  (concat (:nll expected) (:g expected))
                                  (concat (:nll stale) (:g stale))))]
      (is (> difference 1.0e-4) (str "stale " key " is observable in batch " batch)))))

(defn- check-public-head! [target]
  (let [initial (fixture/inputs)
        initial-before (into {} (map (fn [[k v]] [k (vec v)]))
                             (select-keys initial [:y :rows :targets :w :E :wf :lse :nll :g]))
        prepared (with-redefs [gpu/make-session (fn [& _] (throw (AssertionError. "lower opened a session")))
                              gpu/alloc! (fn [& _] (throw (AssertionError. "lower allocated")))
                              link/instantiate! (fn [& _] (throw (AssertionError. "lower instantiated")))]
                   (compiled/lower #'fixture/head-step! (fixture/arguments initial)
                                   {:compiler :equation-first :target target :dtype :float
                                    :constants '[E wf] :outputs '[nll g]
                                    :roles '{lse :state nll :output g :output}}))
        instances (get-in prepared [:lowering :plan :instances])]
    (is (= 1 (count instances)))
    (is (every? link-plan/program-link-instance? instances))
    (is (= #{:E :wf} (set (map :key (filter #(= :constant (:role %)) (:in-tree prepared))))))
    (let [artifact (compiled/instantiate! prepared)
          latest (atom nil)]
      (try
        (doseq [batch (range 3)]
          (let [{:keys [rows w d seq-len] :as inputs} (fixture/inputs batch)
                overrides (select-keys inputs [:y :rows :targets :w])
                before (into {} (map (fn [[k v]] [k (vec v)])) overrides)
                expected (fixture/oracle inputs)
                previous @latest
                actual (artifact overrides)
                gradient (value/->host (:g actual))
                active-rows (set (keep-indexed (fn [i weight]
                                                (when (pos? weight) (nth rows i))) w))]
            (reset! latest actual)
            (is (= #{:nll :g} (set (keys actual))))
            (check-close (:nll expected) (value/->host (:nll actual)) 1.0e-5)
            (check-close (:g expected) gradient 1.0e-5)
            (doseq [row (range seq-len) :when (not (contains? active-rows row))]
              (is (every? zero? (subvec (vec gradient) (* row d) (* (inc row) d)))
                  "unselected and zero-weight rows are cleared after every replay"))
            (is (= before (into {} (map (fn [[k v]] [k (vec v)])) overrides)))
            (is (every? value/live? (vals actual)))
            (when previous
              (is (every? (complement value/live?) (vals previous))))))
        (is (= initial-before
               (into {} (map (fn [[k v]] [k (vec v)]))
                     (select-keys initial [:y :rows :targets :w :E :wf :lse :nll :g]))))
        (compiled/close! artifact)
        (is (every? (complement value/live?) (vals @latest)))
        (finally (compiled/close! artifact))))))

(deftest changing-rows-targets-and-padding-replay-through-the-public-api
  (if @opencl/opencl-available?
    (check-public-head! :ocl:0)
    (opencl/opencl-skip! "public SFT head"))
  (if @ze/gpu-available?
    (check-public-head! :ze:0)
    (ze/gpu-skip! "public SFT head")))
