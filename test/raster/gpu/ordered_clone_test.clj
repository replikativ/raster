(ns raster.gpu.ordered-clone-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.par :as par]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.value :as value]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]))

(deftm clone-between-device-writes!
  [seed :- (Array double), state :- (Array double), out :- (Array double), n :- Long] :- (Array double)
  (par/map-void! i n (aset state i (aget seed i)))
  (let [snapshot (aclone state)]
    (par/map-void! i n
      (do (aset state i 0.0)
          (aset out i (aget snapshot i))))
    out))

(defn- bits [array]
  (mapv #(Double/doubleToRawLongBits %) array))

(defn- check-snapshot-replay! [target]
  (doseq [data [[-0.0 (Double/longBitsToDouble 0x7ff8000000000001) 3.0] []]]
    (let [seed (double-array data)
          n (alength seed)
          state (double-array (repeat n 77.0))
          out (double-array n)
          prepared (compiled/lower #'clone-between-device-writes! [seed state out n]
                                   {:compiler :equation-first :target target :dtype :double
                                    :outputs '[state]})
          artifact (compiled/instantiate! prepared)
          forbidden (fn [& _] (throw (AssertionError. "source clone downloaded intermediate data")))]
      (try
        (is (some? (:ordered-invocation prepared)))
        (doseq [replacement [(double-array data) (double-array (reverse data)) nil]]
          (let [outputs (with-redefs [gpu/download forbidden gpu/download-range! forbidden]
                          (if replacement (artifact {:seed replacement}) (artifact)))
                actual (value/->host (:result outputs))
                expected (or replacement seed)]
            (is (= (bits expected) (bits actual)))
            (is (every? zero? (bits (value/->host (:state outputs))))
                "the later device write happened after the snapshot")
            (is (= (vec (repeat n 77.0)) (vec state))
                "resident writes and source snapshots do not mutate borrowed host entries")))
        (let [plan (get-in @(:execution-state (:executable artifact)) [:resident-boundary :plan])
              bindings (get-in plan [:attributes :compiler-buffer-bindings])
              snapshots (vec (distinct (keep (fn [[id node]]
                                               (when (and (symbol? id) (re-find #"^snapshot" (name id))) node))
                                             bindings)))
              snapshot (first snapshots)
              source (get-in plan [:attributes :public-buffer-bindings 'state])]
          (is (= 1 (count snapshots)))
          (is (some? source))
          (is (contains? (:nodes plan) source))
          (is (contains? (:nodes plan) snapshot))
          (is (not= (get-in plan [:nodes source :view :allocation :id])
                    (get-in plan [:nodes snapshot :view :allocation :id])))
          (is (nil? (get-in plan [:nodes snapshot :source]))
              "the clone is not re-uploaded from a stale inherited host initializer"))
        (finally (compiled/close! artifact))))))

(deftest opencl-clone-snapshots-current-resident-data
  (if @opencl/opencl-fp64-available?
    (check-snapshot-replay! :ocl:0)
    (opencl/opencl-skip! "ordered clone snapshot and replay")))

(deftest level-zero-clone-snapshots-current-resident-data
  (if @ze/gpu-available?
    (check-snapshot-replay! :ze:0)
    (ze/gpu-skip! "ordered clone snapshot and replay")))
