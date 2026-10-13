(ns raster.gpu.public-mutable-aggregate-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.fixtures.mutable-aggregate-training :as fixture]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as link]
            [raster.gpu.value :as value]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(defn- failure-data [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (ex-data error))))

(defn- host-state [outputs]
  {:weight (vec (value/->host (get outputs [:parameters :layer :weight])))
   :bias (vec (value/->host (get outputs [:parameters :layer :bias])))
   :mean (vec (value/->host (get outputs [:moments :layer :mean])))
   :variance (vec (value/->host (get outputs [:moments :layer :variance])))})

(defn- max-error [expected actual]
  (reduce max 0.0 (map #(Math/abs (- (double %1) (double %2))) expected actual)))

(defn- within-float-ulps? [expected actual]
  (and (= (count expected) (count actual))
       (every? #(Float/isFinite (float %)) actual)
       (every? true?
               (map (fn [expected actual]
                      (<= (Math/abs (- (double expected) (double actual)))
                          (* 4.0 (double (Math/ulp (float expected))))))
                    expected actual))))

(defn- check-state [expected actual]
  (doseq [field [:weight :bias :mean :variance]]
    (is (= 4 (count (get actual field))) (str (name field) " retains its declared length"))
    (is (within-float-ulps? (get expected field) (get actual field))
        (str (name field) " is finite and follows the independent Float recurrence"))))

(defn- compile-options [target]
  {:compiler :equation-first :target target :dtype :float
   :roles {'parameters :state 'moments :state}
   :outputs fixture/state-paths})

(defn- state-host-before [input]
  {:state (fixture/snapshot input)
   :batch (vec (:batch input))
   :clock (vec (:clock input))})

(defn- mutable-state-alias-input [input]
  (assoc-in input [:moments :layer :mean] (get-in input [:parameters :layer :weight])))

(defn- state-flat-alias-input [input]
  (assoc input :batch (get-in input [:parameters :layer :weight])))

(deftest independent-oracle-distinguishes-state-data-and-clock-progress
  (let [initial (fixture/inputs)
        state (fixture/snapshot initial)
        first (fixture/oracle-step state (:batch initial) 1)
        changed-batch (:batch (fixture/inputs 0.375))
        second (fixture/oracle-step (:state first) changed-batch 2)
        stale-state (fixture/oracle-step state changed-batch 2)
        stale-clock (fixture/oracle-step (:state first) changed-batch 1)
        stale-data (fixture/oracle-step (:state first) (:batch initial) 2)]
    (is (> (max-error (get-in second [:state :weight])
                      (get-in stale-state [:state :weight])) 1.0e-4))
    (is (> (max-error (get-in second [:state :mean])
                      (get-in stale-clock [:state :mean])) 1.0e-4))
    (is (> (max-error (get-in second [:state :variance])
                      (get-in stale-data [:state :variance])) 1.0e-4))))

(deftest source-recurrence-agrees-exactly-with-the-independent-oracle
  (let [actual (fixture/inputs)]
    (loop [reference (fixture/snapshot actual)
           [[batch-shift clock-value] & remaining] [[0.0 1] [0.375 2]]]
      (when clock-value
        (let [batch (:batch (fixture/inputs batch-shift))
              clock (long-array [clock-value])
              expected (fixture/oracle-step reference batch clock-value)
              loss (apply fixture/step!
                          (fixture/arguments (assoc actual :batch batch :clock clock)))
              actual-state (fixture/snapshot actual)]
          (is (Double/isFinite (double loss)))
          (is (= (:loss expected) (double loss)))
          (is (= (:state expected) actual-state))
          (doseq [field [:weight :bias :mean :variance]]
            (is (= 4 (count (get actual-state field))))
            (is (every? #(Float/isFinite (float %)) (get actual-state field))))
          (recur (:state expected) remaining))))))

(defn- run-device! [target]
  (let [initial (fixture/inputs)
        host-before (state-host-before initial)
        prepared
        (with-redefs [gpu/make-session (fn [& _] (throw (AssertionError. "lower opened a session")))
                      gpu/alloc! (fn [& _] (throw (AssertionError. "lower allocated")))
                      link/instantiate! (fn [& _] (throw (AssertionError. "lower instantiated")))]
          (compiled/lower #'fixture/step! (fixture/arguments initial) (compile-options target)))
        in-roles (into {} (map (juxt :key :role)) (:in-tree prepared))
        out-nodes (mapv :node (:out-tree prepared))]
    (is (= :none (get-in prepared [:schedule :stats :fallback])))
    (is (= (set fixture/state-paths) (set (keys (select-keys in-roles fixture/state-paths)))))
    (is (every? #(= :state (get in-roles %)) fixture/state-paths))
    (is (= (conj (set fixture/state-paths) :result) (set (map :key (:out-tree prepared)))))
    (is (= (count out-nodes) (count (distinct out-nodes))))
    (is (= (set out-nodes) (set (get-in prepared [:lowering :plan :outputs]))))
    (let [artifact (compiled/instantiate! prepared)
          latest (atom nil)]
      (try
        (is (= :compiled-aggregate-input-role
               (reason #(artifact {:parameters (:parameters (fixture/inputs))}))))
        (is (= [:parameters :layer :weight]
               (:key (failure-data
                      #(artifact {[:parameters :layer :weight] (float-array 4)}))))
            "captured state leaves are not replay inputs")
        (loop [reference (fixture/snapshot initial)
               [[batch-shift clock] & remaining] [[0.0 1] [0.375 2]]]
          (when clock
            (let [batch (:batch (fixture/inputs batch-shift))
                  expected (fixture/oracle-step reference batch clock)
                  previous @latest
                  actual (artifact {:batch batch :clock (long-array [clock])})
                  actual-state (host-state actual)
                  loss (double (first (value/->host (:result actual))))]
              (reset! latest actual)
              (is (Double/isFinite loss))
              (is (<= (Math/abs (- loss (:loss expected)))
                      (* 4.0 (double (Math/ulp (float (:loss expected)))))))
              (check-state (:state expected) actual-state)
              (is (some #(not (zero? (float %))) (:mean actual-state)))
              (is (every? value/live? (vals actual)))
              (is (every? #(= ::value/external (:owner %)) (vals actual))
                  "state and loss outputs borrow session-owned resident storage")
              (when previous
                (is (every? (complement value/live?) (vals previous))))
              (recur (:state expected) remaining))))
        (let [before (host-state @latest)]
          (is (= :link-initializer-dtype
                 (reason #(artifact {:batch (:batch (fixture/inputs 0.75))
                                     :clock (float-array [3.0])}))))
          (is (= before (host-state @latest))
              "complete input preflight precedes state mutation and output invalidation"))
        (let [lease (compiled/invoke-leased
                     artifact {:batch (:batch (fixture/inputs -0.25)) :clock (long-array [3])})
              leased @lease]
          (try
            (is (every? value/live? (vals leased)))
            (is (= :link-output-lease-active
                   (reason #(artifact {:batch (:batch initial) :clock (long-array [4])}))))
            (is (= :link-output-lease-active (reason #(compiled/close! artifact))))
            (finally (.close ^java.io.Closeable lease)))
          (is (every? (complement value/live?) (vals leased))))
        (is (= host-before (state-host-before initial))
            "captured host aggregates and ordinary inputs are never mutated")
        (compiled/close! artifact)
        (is (every? (complement value/live?) (vals @latest)))
        (finally (compiled/close! artifact))))))

(deftest mutable-aggregate-admission-refuses-implicit-ownership-and-aliases
  (let [input (fixture/inputs)
        options (compile-options :ocl:0)]
    (is (= :compiled-aggregate-write
           (reason #(compiled/lower #'fixture/step! (fixture/arguments input)
                                    (dissoc options :roles)))))
    (is (= :compiled-aggregate-write
           (reason #(compiled/lower #'fixture/step! (fixture/arguments input)
                                    (assoc options :roles {'parameters :state})))))
    (is (= :compiled-aggregate-write
           (reason #(compiled/lower #'fixture/step! (fixture/arguments input)
                                    (assoc options :roles
                                           {'parameters :output 'moments :state})))))
    (doseq [aliased [(mutable-state-alias-input (fixture/inputs))
                     (state-flat-alias-input (fixture/inputs))]]
      (is (= :compiled-aggregate-state-alias
             (reason #(compiled/lower #'fixture/step! (fixture/arguments aliased) options)))))
    (let [same (float-array 4)
          aliased (assoc-in (assoc-in (fixture/inputs)
                                      [:parameters :layer :weight] same)
                            [:parameters :layer :bias] same)]
      (is (= #{[:layer :weight] [:layer :bias]}
             (let [{:keys [path-a path-b]}
                   (failure-data
                    #(compiled/lower #'fixture/step! (fixture/arguments aliased) options))]
               (set [path-a path-b])))))
    (is (= :compiled-aggregate-output
           (reason #(compiled/lower #'fixture/step! (fixture/arguments input)
                                    (assoc options :outputs '[parameters])))))
    (is (= :compiled-field-selector-intent
           (reason #(compiled/lower #'fixture/step! (fixture/arguments input)
                                    (assoc options :donate [[:parameters :layer :weight]])))))
    (doseq [intent [{:donate '[parameters__layer__weight]}
                    {:outputs '[parameters__layer__weight]}
                    {:taps '[parameters__layer__weight]}]]
      (is (= :compiled-aggregate-output
             (reason #(compiled/lower #'fixture/step! (fixture/arguments input)
                                      (merge options intent))))))
    (is (= :compiled-field-selector
           (reason #(compiled/lower #'fixture/step! (fixture/arguments input)
                                    (assoc options :outputs [[:parameters :missing]])))))
    (is (= :compiled-field-output-duplicates
           (reason #(compiled/lower #'fixture/step! (fixture/arguments input)
                                    (assoc options :outputs
                                           [[:parameters :layer :weight]
                                            [:parameters :layer :weight]])))))
    (is (= :compiled-field-output-semantic-overlap
           (reason #(compiled/lower
                     #'fixture/return-weight! [(:parameters input) 4]
                     {:compiler :equation-first :target :ocl:0 :dtype :float
                      :roles {'parameters :state}
                      :outputs [[:parameters :layer :weight]]}))))))

(deftest nested-parameter-and-moment-state-replays-on-both-local-backends
  (if @opencl/opencl-available?
    (run-device! :ocl:0)
    (opencl/opencl-skip! "mutable nested aggregate state"))
  (if @ze/gpu-available?
    (run-device! :ze:0)
    (ze/gpu-skip! "mutable nested aggregate state")))
