(ns raster.gpu.public-buffered-adam-test
  "Two real Adam steps, not replay of a frozen t=1 optimizer."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.fixtures.lora-training :as fixture]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as link]
            [raster.gpu.value :as value]))

(def ^:private state-keys [:A :B :mA :vA :mB :vB])

(defn- inputs []
  (assoc (fixture/adam-inputs) :time-buffer (long-array [1])))

(defn- arguments [input]
  (conj (pop (fixture/adam-arguments input)) (:time-buffer input)))

(defn- check-state [expected actual]
  (doseq [[key tolerance] [[:A 2.0e-7] [:B 2.0e-7]
                         [:mA 1.0e-10] [:mB 1.0e-10]
                         [:vA 1.0e-16] [:vB 1.0e-16]]]
    (is (every? #(Float/isFinite (float %)) (get actual key))
        (str key " contains only finite state"))
    (is (< (fixture/max-error (get expected key) (get actual key)) tolerance)
        (str key " agrees with the independent Adam recurrence"))))

(deftest source-training-carries-moments-and-distinguishes-a-stale-clock
  (let [actual (inputs)
        initial (inputs)
        first-reference (fixture/joint-clipped-adam-oracle initial)
        second-input (merge initial (select-keys first-reference state-keys))
        correct-second (fixture/joint-clipped-adam-oracle (assoc second-input :adam-t 2))
        stale-second (fixture/joint-clipped-adam-oracle (assoc second-input :adam-t 1))]
    (is (< 0.0 (:scale first-reference) 0.01)
        "the joint clipping branch is active in this recurrence witness")
    (is (> (fixture/max-error (:A correct-second) (:A stale-second)) 2.0e-7)
        "the witness detects an optimizer frozen at the first clock value")
    (doseq [[clock expected] [[1 first-reference] [2 correct-second]]]
      (aset ^longs (:time-buffer actual) 0 (long clock))
      (let [loss (apply fixture/joint-clipped-lora-adam-buffered-step (arguments actual))]
        (is (< (Math/abs (- (double loss) (:loss expected))) 1.0e-7))
        (check-state expected actual)))))

(defn- check-device! [target]
  (let [initial (inputs)
        host-before (into {} (map (fn [key] [key (vec (get initial key))]))
                          (concat state-keys [:x :tgt :time-buffer]))
        prepared
        (with-redefs [gpu/make-session (fn [& _] (throw (AssertionError. "lower opened a session")))
                      gpu/alloc! (fn [& _] (throw (AssertionError. "lower allocated")))
                      link/instantiate! (fn [& _] (throw (AssertionError. "lower instantiated")))]
          (compiled/lower
           #'fixture/joint-clipped-lora-adam-buffered-step (arguments initial)
           {:compiler :equation-first :target target :dtype :float
            :roles {'A :state 'B :state 'mA :state 'vA :state 'mB :state 'vB :state}
            :outputs '[A B mA vA mB vB]}))
        stage-matches
        (vec (for [instance (get-in prepared [:lowering :plan :instances])
                   :let [program (get-in instance [:call :program])]
                   closure (get-in program [:attributes :resident-uniform-input-scalar-closures])
                   :when (= 'time-buffer (:input closure))]
               [program closure]))
        _schedule
        (do
          (is (= 1 (count stage-matches)))
          (is (= :none (get-in prepared [:schedule :stats :fallback])))
          (when-let [[program closure] (first stage-matches)]
            (is (= [:root :island] (mapv :kind (:stages closure))))
            (is (= [1 2] (mapv #(count (:frontier %)) (:stages closure))))
            (let [private-values (vec (mapcat :frontier (:stages closure)))
                  public-values (set (concat (:inputs program) (:outputs program)))
                  equations (into {} (map (juxt :id identity)) (:equations program))]
              (is (every? #(= [1] (get-in program [:values % :shape])) private-values))
              (is (not-any? public-values private-values))
              (doseq [{:keys [equation frontier]} (:stages closure)]
                (let [producer (get equations equation)
                      operation (first (dialect/equations (:algorithm producer)))]
                  (is (= (set frontier) (set (:results producer))))
                  (is (= (set frontier) (set (nth operation 2)))
                      "the singleton semantic operation produces every certified frontier")
                  (is (= 'map (:kind (dialect/operation-parts operation)))
                      "clock and bias correction are device maps, not host scalar equations")
                  (is (= 1 (dialect/operation-extent operation))))))))
        artifact (compiled/instantiate! prepared)
        latest (atom nil)]
    (try
      (loop [expected (inputs) clock 1]
        (when (<= clock 2)
          (let [reference (fixture/joint-clipped-adam-oracle (assoc expected :adam-t clock))
                previous @latest
                ;; Captured :state buffers update in place. Only the ordinary clock input changes.
                actual (artifact {:time-buffer (long-array [clock])})
                _ (reset! latest actual)
                host (into {} (map (fn [key] [key (value/->host (get actual key))])) state-keys)
                loss (double (first (value/->host (:result actual))))]
            (is (= (conj (set state-keys) :result) (set (keys actual))))
            (is (< (Math/abs (- loss (:loss reference))) 1.0e-7))
            (check-state reference host)
            (is (every? value/live? (vals actual)))
            (when previous
              (is (every? (complement value/live?) (vals previous))))
            (recur (merge expected (select-keys reference state-keys)) (inc clock)))))
      (is (= host-before
             (into {} (map (fn [key] [key (vec (get initial key))]))
                   (concat state-keys [:x :tgt :time-buffer]))))
      (compiled/close! artifact)
      (is (every? (complement value/live?) (vals @latest)))
      (finally (compiled/close! artifact)))))

(deftest buffered-clock-and-resident-moments-replay-on-both-local-backends
  (if @opencl/opencl-available?
    (check-device! :ocl:0)
    (opencl/opencl-skip! "buffered Adam recurrence"))
  (if @ze/gpu-available?
    (check-device! :ze:0)
    (ze/gpu-skip! "buffered Adam recurrence")))
