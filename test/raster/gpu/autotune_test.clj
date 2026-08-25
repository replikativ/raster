(ns raster.gpu.autotune-test
  "Phase-2: the autotune search + cache. Device-free — coordinate-descent over synthetic landscapes,
   the noise-rejection threshold, and the disk-cache round-trip."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.gpu.autotune :as at]
            [raster.gpu.schedule :as sched]
            [raster.gpu.tuning-cache :as cache])
  (:import [java.nio.file Files]))

(defn- temporary-cache-root
  []
  (.toFile (Files/createTempDirectory "raster-autotune-"
                                      (make-array java.nio.file.attribute.FileAttribute 0))))

(deftest coordinate-descent-finds-the-minimum
  ;; landscape minimized at {:a 8 :b :y}; ×2/÷2 moves on :a, two-valued :b
  (let [cost (fn [c] (+ (Math/abs (double (- (:a c) 8))) (if (= (:b c) :y) 0.0 5.0)))
        r (at/coordinate-descent {:a 1 :b :x}
                                 {:a (fn [a] [(* a 2) (max 1 (quot a 2))])
                                  :b (fn [_] [:x :y])}
                                 cost)]
    (is (= 8 (:a (:config r))) "walks the ×2 ladder to 8")
    (is (= :y (:b (:config r))) "flips the two-valued knob")
    (is (< (:cost r) 1.0))
    (is (pos? (:evals r)))))

(deftest threshold-rejects-noise
  (testing "a sub-0.1% improvement is NOT taken (Inductor's noise floor)"
    (let [cost (fn [c] (if (= (:p c) :b) 999.5 1000.0))   ;; 0.05% better — below 0.1%
          r (at/coordinate-descent {:p :a} {:p (fn [_] [:a :b])} cost)]
      (is (= :a (:p (:config r)))))))

(deftest infeasible-candidates-self-prune
  (testing "a +Inf cost (infeasible schedule) is never selected"
    (let [cost (fn [c] (case (:p c) :bad Double/POSITIVE_INFINITY :good 1.0 2.0))
          r (at/coordinate-descent {:p :start} {:p (fn [_] [:bad :good])} cost)]
      (is (= :good (:p (:config r)))))))

(deftest nested-path-moves
  (testing "coordinate-descent updates nested schedule paths like [:grf :mode]"
    (let [cost (fn [c] (if (= (get-in c [:grf :mode]) :grf256) 1.0 2.0))
          r (at/coordinate-descent {:grf {:mode :grf128}}
                                   {[:grf :mode] (fn [_] [:grf128 :grf256])} cost)]
      (is (= :grf256 (get-in (:config r) [:grf :mode]))))))

(deftest cache-roundtrip
  (binding [cache/*cache-root* (temporary-cache-root)]
    (let [desc {:device-id :ze:0 :bandwidth-bytes-s 9.0e10 :peak-flops {:f32 4.0e12}}
          identity {:numerical-mode :f32 :emitter-hash "emitter-v1" :layout :row-major}
          k (at/cache-key :op-Z :shape-Z desc identity)
          cfg {:precision :f16-xmx :grf {:mode :grf128}}]
      (is (nil? (at/cache-get k)) "cold miss")
      (at/cache-put! k cfg)
      (is (= cfg (at/cache-get k)) "round-trips from disk")
      (testing "the key includes the measured perf signature — a re-calibrated machine misses"
        (is (not= k (at/cache-key :op-Z :shape-Z
                                  (assoc desc :bandwidth-bytes-s 1.37e11)
                                  identity))))
      (testing "numerical mode, emitter, and layout are mandatory correctness identity"
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"requires numerical mode"
                              (at/cache-key :op-Z :shape-Z desc {})))))))

(deftest autotune-schedule-seeds-from-derive-default
  (binding [cache/*cache-root* (temporary-cache-root)]
    (let [desc {:device-id :ze:0 :grf-bytes-per-lane 256 :bandwidth-bytes-s 9.0e10 :peak-flops {:f32 4.0e12} :balance 60}
          ;; cost prefers a smaller measured crossover multiplier while numerical mode stays fixed
          cost (fn [s] (try (sched/feasible? s desc)
                            (if (= 8 (get-in s [:segmented-weighted-reduction
                                                :score-reuse-subgroup-multiple]))
                              1.0 2.0)
                            (catch clojure.lang.ExceptionInfo _ Double/POSITIVE_INFINITY)))
          op :op-Q
          identity {:numerical-mode :f16-f32
                    :emitter-hash "segmented-reduction-v1"
                    :layout :packed-edge-list}
          reduction-neighbor
          {[:segmented-weighted-reduction :score-reuse-subgroup-multiple]
           (fn [_] [8 16])}
          best (at/autotune-schedule op :shape-Q desc cost
                                     :identity identity
                                     :neighbors reduction-neighbor :force? true)]
      (is (not (contains? (at/schedule-neighbors desc) :precision))
          "default search never changes numerical mode")
      (is (= :f16-xmx (:precision best)) "numerical mode remains fixed")
      (is (= 8 (get-in best [:segmented-weighted-reduction
                             :score-reuse-subgroup-multiple])))
      (testing "second call hits the cache without re-measuring"
        (is (= 8 (get-in (at/autotune-schedule
                          op :shape-Q desc
                          (fn [_] (throw (ex-info "should-not-measure" {})))
                          :identity identity
                          :neighbors reduction-neighbor)
                         [:segmented-weighted-reduction
                          :score-reuse-subgroup-multiple])))))))
