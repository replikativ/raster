(ns raster.gpu.public-scalar-checkpoint-test
  "A public scalar checkpoint remains ordered before a later independent state mutation."
  (:require [clojure.test :refer [deftest is]]
            [raster.arrays :as arrays]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-resident :as resident]
            [raster.core :refer [deftm]]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as link]
            [raster.gpu.value :as value]
            [raster.numeric :as n]
            [raster.par :as par]))

(deftm scalar-checkpoint-before-state-update
  [input :- (Array float) state :- (Array float)] :- Double
  (let [checkpoint (float-array 1)]
    (par/map-void! i 1
      (arrays/aset checkpoint i
                   (n/+ (n/* (arrays/aget input i) (float 1.125))
                        (arrays/aget state i))))
    (let [loss-value (double (arrays/aget checkpoint 0))]
      ;; The public checkpoint has no internal consumer. This later effect must remain ordered
      ;; after it while independently advancing the resident state.
      (par/map-void! i 1
        (arrays/aset state i (n/+ (arrays/aget state i) (float 0.25))))
      loss-value)))

(def ^:private samples
  [(float-array [1.2345679])
   (float-array [3.1415927])
   (float-array [16777214.0])])

(defn- reference [^floats input state]
  (let [x (aget input 0)
        product (float (* x (float 1.125)))
        checkpoint (float (+ product (float state)))]
    {:loss (double checkpoint)
     :state (float (+ (float state) (float 0.25)))}))

(defn- double-bits [number]
  (Double/doubleToRawLongBits (double number)))

(defn- float-bits [number]
  (Float/floatToRawIntBits (float number)))

(defn- scheduled-algorithms [prepared]
  (vec
   (mapcat
    (fn [[instance-index instance]]
      (let [program (get-in instance [:call :program])]
        (keep-indexed
         (fn [equation-index equation]
           (let [algorithm (:algorithm equation)]
             (when (dialect/program-form? algorithm)
               {:position [instance-index equation-index]
                :program program :equation equation :algorithm algorithm
                :inner (first (dialect/equations algorithm))})))
         (:equations program))))
    (map-indexed vector (get-in prepared [:lowering :plan :instances])))))

(defn- check-device! [target]
  (let [initial-input (first samples)
        initial-state (float-array [0.5])
        host-before {:input (vec initial-input) :state (vec initial-state)
                     :samples (mapv vec samples)}
        prepared
        (with-redefs [gpu/make-session (fn [& _] (throw (AssertionError. "lower opened a session")))
                      gpu/alloc! (fn [& _] (throw (AssertionError. "lower allocated")))
                      link/instantiate! (fn [& _] (throw (AssertionError. "lower instantiated")))]
          (compiled/lower
           #'scalar-checkpoint-before-state-update [initial-input initial-state]
           {:compiler :equation-first :target target :dtype :float
            :roles {'state :state} :outputs '[state]}))
        state-input (some #(when (= :state (:key %)) %) (:in-tree prepared))
        state-symbol (:sym state-input)
        scheduled (scheduled-algorithms prepared)
        scalar
        (some (fn [{:keys [algorithm inner] :as scheduled-equation}]
                (let [result (first (nth inner 2))
                      scalar-value (get-in (dialect/facts algorithm) [:values result])]
                  (when (and (= 'scalar (dialect/operation-kind inner))
                             (= [result] (dialect/outputs algorithm))
                             (= :resident-scalar-buffer
                                (get-in scalar-value [:representation :kind])))
                    (assoc scheduled-equation :result result :value scalar-value))))
              scheduled)
        state-effect
        (some (fn [{:keys [algorithm inner] :as scheduled-equation}]
                (when (some #{state-symbol} (dialect/physical-results algorithm inner))
                  scheduled-equation))
              scheduled)
        result-entry (some #(when (= :result (:key %)) %) (:out-tree prepared))]
    (is (= :none (get-in prepared [:schedule :stats :fallback])))
    (is (= :state (:role state-input)))
    (is (= #{:state :result} (set (map :key (:out-tree prepared)))))
    (is (some? scalar) "the public loss has one semantic ScalarIR device schedule")
    (is (some? state-effect) "the independent state mutation retains its physical write result")
    (when (and scalar state-effect)
      (is (neg? (compare (:position scalar) (:position state-effect)))
          "the scalar checkpoint remains before the later state mutation")
      (is (= [] (get-in scalar [:value :shape])))
      (is (= :double (get-in scalar [:value :dtype])))
      (is (= {:kind :resident-scalar-buffer :elements 1}
             (get-in scalar [:value :representation])))
      (is (= (get-in scalar [:value])
             (get-in scalar [:program :values (:result scalar)])))
      (is (= [(:result scalar)] (get-in scalar [:equation :results])))
      (is (not (true? (get-in scalar [:equation :attributes :host-only]))))
      (is (not-any? #(some #{(:result scalar)} (get-in % [:equation :operands]))
                    (filter #(and (= (:program scalar) (:program %))
                                  (neg? (compare (:position scalar) (:position %))))
                            scheduled))
          "the public checkpoint has no internal consumer")
      (let [semantic (get-in prepared [:schedule :semantic])
            algorithms (mapv :algorithm (:equations semantic))
            facts {:values (:values semantic) :attributes (:attributes semantic)
                   :equations (apply merge (map #(get-in (dialect/facts %) [:equations]) algorithms))}]
        (is (resident/public-scalar-unused-in-program?
             facts (vec (mapcat dialect/equations algorithms)) (:result scalar))
            "the complete zero-use check includes allocation, shape and alias dependencies"))
      (let [emitted (first (get-in scalar [:equation :operations]))
            operation (some #(when (= (get-in scalar [:equation :id]) (:id %))
                               (first (:operations %)))
                            (get-in emitted [:body :equations]))]
        (is (segop/seg-map? operation))
        (is (= 1 (get-in operation [:space :dims 0 :bound])))
        (is (= #{(:result scalar)} (:outputs operation)))))
    (when state-effect
      (is (seq (get-in state-effect [:equation :operations])))
      (is (not (true? (get-in state-effect [:equation :attributes :host-only])))))
    (is (= [] (:shape result-entry)))
    (is (= :double (:dtype result-entry)))
    (let [artifact (compiled/instantiate! prepared)
          latest (atom nil)]
      (try
        (loop [state (float 0.5)
               index 0]
          (when (< index (count samples))
            (let [input (nth samples index)
                  expected (reference input state)
                  previous @latest
                  outputs (artifact (if (zero? index) {} {:input input}))
                  loss-output (:result outputs)
                  state-output (:state outputs)
                  actual-loss (first (value/->host loss-output))
                  actual-state (first (value/->host state-output))]
              (reset! latest outputs)
              (is (= #{:state :result} (set (keys outputs))))
              (is (every? value/device-array? (vals outputs)))
              (is (= :double (:dtype loss-output)))
              (is (= [] (:shape loss-output)))
              (is (= :float (:dtype state-output)))
              (is (= [1] (:shape state-output)))
              (is (= (double-bits (:loss expected)) (double-bits actual-loss))
                  "the public Double widens the pre-update Float checkpoint")
              (is (= (float-bits (:state expected)) (float-bits actual-state)))
              (is (> (float actual-state) state) "resident state advances after every checkpoint")
              (is (every? value/live? (vals outputs)))
              (when previous
                (is (every? (complement value/live?) (vals previous))))
              (recur (:state expected) (inc index)))))
        (is (= host-before {:input (vec initial-input) :state (vec initial-state)
                            :samples (mapv vec samples)})
            "captured state and all ordinary host arrays remain unchanged")
        (compiled/close! artifact)
        (is (every? (complement value/live?) (vals @latest)))
        (finally (compiled/close! artifact))))))

(deftest source-checkpoint-precedes-the-independent-state-update
  (let [state (float-array [0.5])]
    (loop [expected-state (float 0.5)
           index 0]
      (when (< index (count samples))
        (let [input (nth samples index)
              input-before (vec input)
              expected (reference input expected-state)
              actual (scalar-checkpoint-before-state-update input state)]
          (is (= (double-bits (:loss expected)) (double-bits actual)))
          (is (= (float-bits (:state expected)) (float-bits (aget state 0))))
          (is (= input-before (vec input)))
          (recur (:state expected) (inc index)))))))

(deftest public-scalar-checkpoint-replays-on-both-local-backends
  (if @opencl/opencl-fp64-available?
    (check-device! :ocl:0)
    (opencl/opencl-skip! "public Double scalar checkpoint" :fp64))
  (if @ze/gpu-available?
    (check-device! :ze:0)
    (ze/gpu-skip! "public Double scalar checkpoint")))
