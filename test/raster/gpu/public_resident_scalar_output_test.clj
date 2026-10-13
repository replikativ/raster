(ns raster.gpu.public-resident-scalar-output-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.fixtures.resident-scalar-output :as fixture]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as link]
            [raster.gpu.value :as value]))

(deftest source-semantics-return-a-double-and-fill-the-output
  (doseq [variant [0 1]]
    (let [{:keys [out] :as input} (fixture/inputs variant)
          expected (fixture/oracle input)
          actual (apply fixture/joint-norm-with-output (fixture/arguments input))]
      (is (instance? Double actual))
      (is (= expected actual))
      (is (= (vec (repeat 3 (float expected))) (vec out))))))

(defn- check-public-resident-output! [target]
  (let [initial (fixture/inputs)
        host-before [(vec (:x initial)) (vec (:y initial)) (vec (:out initial))]
        prepared
        (with-redefs [gpu/make-session (fn [& _] (throw (AssertionError. "lower opened a session")))
                      gpu/alloc! (fn [& _] (throw (AssertionError. "lower allocated")))
                      link/instantiate! (fn [& _] (throw (AssertionError. "lower instantiated")))]
          (compiled/lower
           #'fixture/joint-norm-with-output (fixture/arguments initial)
           {:compiler :equation-first :target target :dtype :float
            :roles {'out :output} :outputs '[out]}))
        result-entry (some #(when (= :result (:key %)) %) (:out-tree prepared))]
    (is (= #{:out :result} (set (map :key (:out-tree prepared))))
        "the same resident root is the semantic result; no duplicate scalar output is added")
    (is (= [] (:shape result-entry)))
    (is (= :double (:dtype result-entry)))
    (let [artifact (compiled/instantiate! prepared)
          latest (atom nil)]
      (try
        (doseq [variant [0 1]]
          (let [input (fixture/inputs variant)
                expected (fixture/oracle input)
                previous @latest
                actual (artifact (if (zero? variant)
                                   {}
                                   {:x (:x input) :y (:y input)}))
                actual-norm (double (first (value/->host (:result actual))))
                actual-out (vec (value/->host (:out actual)))]
            (reset! latest actual)
            (is (= #{:out :result} (set (keys actual))))
            (is (< (Math/abs (- expected actual-norm))
                   (* 1.0e-12 (max 1.0 (Math/abs expected)))))
            (is (= 3 (count actual-out)))
            (is (every? #(< (Math/abs (- (double %) (double (float expected)))) 1.0e-6)
                        actual-out))
            (is (every? value/live? (vals actual)))
            (when previous
              (is (every? (complement value/live?) (vals previous))
                  "replay retires every prior public wrapper before projecting fresh views"))))
        (is (= host-before [(vec (:x initial)) (vec (:y initial)) (vec (:out initial))])
            "captured host inputs and output-role initializer remain unchanged")
        (compiled/close! artifact)
        (is (every? (complement value/live?) (vals @latest)))
        (finally (compiled/close! artifact))))))

(deftest changed-data-replays-a-public-resident-scalar-on-opencl-and-level-zero
  (if @opencl/opencl-fp64-available?
    (check-public-resident-output! :ocl:0)
    (opencl/opencl-skip! "public resident Double scalar output" :fp64))
  (if @ze/gpu-available?
    (check-public-resident-output! :ze:0)
    (ze/gpu-skip! "public resident Double scalar output")))
