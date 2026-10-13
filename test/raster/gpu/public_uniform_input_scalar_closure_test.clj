(ns raster.gpu.public-uniform-input-scalar-closure-test
  (:require [clojure.set :as set]
            [clojure.test :refer [deftest is]]
            [raster.compiler.fixtures.uniform-input-scalar-closure :as fixture]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as link]
            [raster.gpu.value :as value]))

(deftest source-semantics-use-the-current-buffered-clock
  (doseq [clock [1 2]]
    (let [{:keys [out-a out-b] :as input} (fixture/inputs clock)
          expected (fixture/oracle clock)
          result (apply fixture/buffered-clock-two-maps (fixture/arguments input))]
      (is (identical? out-a result))
      (is (= (:out-a expected) (vec out-a)))
      (is (= (:out-b expected) (vec out-b))))))

(defn- check-buffered-clock! [target]
  (let [initial (fixture/inputs 1)
        host-before [(vec (:time-buffer initial)) (vec (:out-a initial)) (vec (:out-b initial))]
        prepared
        (with-redefs [gpu/make-session (fn [& _] (throw (AssertionError. "lower opened a session")))
                      gpu/alloc! (fn [& _] (throw (AssertionError. "lower allocated")))
                      link/instantiate! (fn [& _] (throw (AssertionError. "lower instantiated")))]
          (compiled/lower
           #'fixture/buffered-clock-two-maps (fixture/arguments initial)
           {:compiler :equation-first :target target :dtype :long
            :roles {'out-a :output 'out-b :output}
            :outputs '[out-a out-b]}))
        emitted-program (get-in prepared [:lowering :plan :instances 0 :call :program])
        closure (first (get-in emitted-program
                               [:attributes :resident-uniform-input-scalar-closures]))
        frontier (vec (:frontier closure))
        frontier-set (set frontier)
        equations (:equations emitted-program)
        producer (some #(when (and (seq frontier-set)
                                   (set/subset? frontier-set (set (:results %)))) %)
                       equations)
        producer-operation (first (dialect/equations (:algorithm producer)))
        compilation-stats (get-in prepared [:schedule :stats])
        artifact (compiled/instantiate! prepared)
        latest (atom nil)]
    (try
      (is (= {:input 'time-buffer :frontier-count 2}
             {:input (:input closure) :frontier-count (count frontier)})
          "the emitted public program retains the exact private singleton-stage certificate")
      (is (= 1 (dialect/operation-extent producer-operation)))
      (is (= frontier-set (set (nth producer-operation 2)))
          "the singleton TypedSOAC operation produces the complete certified frontier")
      (is (every? #(= [1] (get-in emitted-program [:values % :shape])) frontier))
      (is (not-any? (set (concat (:inputs emitted-program) (:outputs emitted-program))) frontier))
      (is (every? (fn [value]
                    (some #(some #{value} (:operands %)) equations)) frontier)
          "each private singleton buffer feeds a later emitted consumer")
      (is (= {:kernel-body 2} (get-in compilation-stats [:emission :emission-routes])))
      (is (zero? (get-in compilation-stats [:emission :host-scalar-equations])))
      (is (= :none (:fallback compilation-stats)))
      (doseq [clock [1 2]]
        (let [expected (fixture/oracle clock)
              previous @latest
              actual (artifact (if (= 1 clock)
                                 {}
                                 {:time-buffer (long-array [clock])}))]
          (reset! latest actual)
          (is (= #{:out-a :out-b} (set (keys actual))))
          (is (= (:out-a expected) (vec (value/->host (:out-a actual)))))
          (is (= (:out-b expected) (vec (value/->host (:out-b actual)))))
          (is (every? value/live? (vals actual)))
          (when previous
            (is (every? (complement value/live?) (vals previous))))))
      (is (= host-before
             [(vec (:time-buffer initial)) (vec (:out-a initial)) (vec (:out-b initial))]))
      (compiled/close! artifact)
      (is (every? (complement value/live?) (vals @latest)))
      (finally (compiled/close! artifact)))))

(deftest changed-buffered-clock-replays-on-opencl-and-level-zero
  (if @opencl/opencl-available?
    (check-buffered-clock! :ocl:0)
    (opencl/opencl-skip! "uniform input scalar closure"))
  (if @ze/gpu-available?
    (check-buffered-clock! :ze:0)
    (ze/gpu-skip! "uniform input scalar closure")))
