(ns raster.gpu.autotune-test
  "Phase-2: the autotune search + cache. Device-free — coordinate-descent over synthetic landscapes,
   the noise-rejection threshold, and the disk-cache round-trip."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.gpu.autotune :as at]
            [raster.gpu.schedule :as sched]))

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
  (let [desc {:device-id :ze:0 :bandwidth-bytes-s 9.0e10 :peak-flops {:f32 4.0e12}}
        k (at/cache-key (keyword (gensym "op")) :shape-Z desc)
        cfg {:precision :f16-xmx :grf {:mode :grf128}}]
    (is (nil? (at/cache-get k)) "cold miss")
    (at/cache-put! k cfg)
    (is (= cfg (at/cache-get k)) "round-trips from disk")
    (testing "the key includes the measured perf signature — a re-calibrated machine misses"
      (is (not= k (at/cache-key (keyword "op") :shape-Z (assoc desc :bandwidth-bytes-s 1.37e11)))))))

(deftest autotune-schedule-seeds-from-derive-default
  (let [desc {:device-id :ze:0 :grf-bytes-per-lane 256 :bandwidth-bytes-s 9.0e10 :peak-flops {:f32 4.0e12} :balance 60}
        ;; cost prefers f32-scalar (opposite of the f16-xmx default) + gates infeasible
        cost (fn [s] (try (sched/feasible? s desc)
                          (if (= (:precision s) :f32-scalar) 1.0 2.0)
                          (catch clojure.lang.ExceptionInfo _ Double/POSITIVE_INFINITY)))
        op (keyword (gensym "op"))
        best (at/autotune-schedule op :shape-Q desc cost :force? true)]
    (is (= :f32-scalar (:precision best)) "search moved off the f16-xmx seed to the cheaper config")
    (testing "second call hits the cache without re-measuring"
      (is (= :f32-scalar (:precision (at/autotune-schedule op :shape-Q desc
                                                           (fn [_] (throw (ex-info "should-not-measure" {}))))))))))
