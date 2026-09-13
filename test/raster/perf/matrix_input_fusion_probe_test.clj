(ns raster.perf.matrix-input-fusion-probe-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.perf.matrix-input-fusion-probe :as probe]
            [raster.perf.production-canary :as canary]))

(deftest invalid-probe-options-do-not-open-a-session
  (with-redefs [gpu/with-gpu-session* (fn [& _] (throw (AssertionError. "unexpected session")))]
    (doseq [options [{:rounds 1} {:rounds 121} {:warmup-rounds -1} {:timing? :yes}
                    {:shape [4096 4096 4096]} {:shape [0 32 32]} {:shape [13 32 nil]}
                    {:tile-policy :unknown} {:input-policy :unknown} {:variant :tn}
                    {:residency :unknown}]]
      (is (thrown? clojure.lang.ExceptionInfo (probe/run! :mock options))))))

(deftest independent-oracle-rounds-halfway-values-to-even
  (is (= [0.5 0.5009765625 -0.5 -0.5009765625]
         (vec (probe/rounded-inputs [0.500244140625 0.500732421875
                                     -0.500244140625 -0.500732421875])))))

(deftest paired-probe-validates-every-replay-and-releases-handles
  (let [storage (atom {}) calls (atom []) released (atom [])
        profile-valid? (atom true)]
    (with-redefs [gpu/with-gpu-session* (fn [_ f] (f (atom {})))
                  gpu/alloc! (fn [_ specs] (reset! storage (update-vals specs last)))
                  gpu/bind-kernel-graph! (fn [_ id _ _ _ opts]
                                           (is (:profile? opts)) id)
                  gpu/upload! (fn [_ id values] (swap! storage assoc id values))
                  gpu/profile-bound-kernel-graph!
                  (fn [_ id]
                    (is (every? #(Float/isNaN (float %)) (:c @storage)))
                    (swap! calls conj id)
                    (swap! storage assoc :c (canary/gemm-reference (:a @storage) (:b @storage) [13 32 32]))
                    {:device-wall-ms (when @profile-valid? 0.01) :kernel-total-ms 0.005 :profile []})
                  gpu/download (fn [_ id] (get @storage id))
                  gpu/release-kernel-graph! (fn [_ id] (swap! released conj id))]
      (let [result (probe/run! :mock {:timing? true :rounds 2 :warmup-rounds 1})]
        (is (= 14 (count @calls)))
        (is (= 2 (count @released)))
        (is (= {:validation 8 :warmup 2 :measurement 4}
               (frequencies (map :sampling-phase (:replay-profiles result)))))
        (is (= [0 0 1 1] (keep :sample-index (:replay-profiles result))))
        (is (= [7 7] (mapv :replays (:candidates result)))))
      (reset! profile-valid? false)
      (reset! released [])
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"finite device span"
                            (probe/run! :mock {:timing? true :rounds 2 :warmup-rounds 0})))
      (is (= 2 (count @released)) "failure releases every previously bound candidate"))))

(deftest constant-weight-mode-uses-normal-linkplan-initialization-and-cleanup
  (let [state (atom {}) plans (atom []) closed (atom [])]
    (with-redefs [gpu/with-gpu-session* (fn [_ f] (f (atom {})))
                  gpu/alloc! (fn [& _] (throw (AssertionError. "LinkPlan owns these allocations")))
                  link/instantiate!
                  (fn [plan opts]
                    (is (:profile? opts))
                    (is (= :constant (get-in plan [:nodes :b :role])))
                    (is (= :input (get-in plan [:nodes :a :role])))
                    (let [id (count @plans)]
                      (swap! plans conj plan)
                      (swap! state assoc id {:a (get-in plan [:nodes :a :source])
                                            :b (get-in plan [:nodes :b :source])})
                      id))
                  link/upload! (fn [h id values] (swap! state assoc-in [h id] values))
                  link/profile!
                  (fn [h]
                    (is (every? #(Float/isNaN (float %)) (get-in @state [h :c])))
                    (swap! state assoc-in [h :c]
                           (canary/gemm-reference (get-in @state [h :a]) (get-in @state [h :b]) [13 32 32]))
                    {:device-wall-ms 0.01 :kernel-total-ms 0.01
                     :profile [{:kernel-name "matrix" :ms 0.01}]})
                  link/download (fn [h id] (get-in @state [h id]))
                  link/close! #(swap! closed conj %)]
      (let [result (probe/run! :mock {:residency :constant-weights :timing? true
                                     :rounds 2 :warmup-rounds 0})]
        (is (= 2 (count @plans)))
        (is (= [1 0] @closed))
        (is (false? (get-in result [:scope :weight-conversion-included?])))
        (is (= [["matrix"] ["matrix"]] (mapv :profiled-replay-kernels (:candidates result))))))))
