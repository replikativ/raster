(ns raster.compiler.ir.kv-transfer-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.kv-transfer :as kv]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.ze-runtime :as ze]))

;; gemma-270m's KV shape: one kv-head of 256, FP16 page pool, 16 tokens per page
(def ^:private geometry
  (kv/geometry {:layers 3 :page-size 16 :physical-pages 64
                :key-elements-per-token 256 :value-elements-per-token 256
                :storage-dtype :half}))

(deftest bytes-follow-the-geometry
  (testing "one page of one buffer is page-size × elements-per-token elements"
    (is (= 4096 (kv/page-elements geometry 256))))
  (testing "a layer's K and V pages, in FP16"
    ;; 3 pages × (4096 + 4096) elements × 2 bytes
    (is (= (* 3 8192 2) (kv/layer-bytes geometry [5 2 9]))))
  (testing "all layers"
    (is (= (* 3 3 8192 2) (kv/total-bytes geometry [5 2 9])))))

(deftest ranges-are-the-fragmented-transfer-in-layer-order
  (let [ranges (kv/ranges geometry [5 2])]
    (is (= 12 (count ranges)) "3 layers × 2 buffers × 2 pages")
    (is (= [{:layer 0 :buffer :key :page 5 :element (* 5 4096) :elements 4096}
            {:layer 0 :buffer :key :page 2 :element (* 2 4096) :elements 4096}
            {:layer 0 :buffer :value :page 5 :element (* 5 4096) :elements 4096}
            {:layer 0 :buffer :value :page 2 :element (* 2 4096) :elements 4096}]
           (take 4 ranges))
        "layer-major, K before V, pages in the continuation's order")
    (is (= (kv/total-bytes geometry [5 2])
           (* 2 (reduce + (map :elements ranges))))
        "the range list moves exactly the certified bytes")))

(deftest pages-must-be-distinct-and-inside-the-pool
  (doseq [pages [[] [64] [3 3] [-1]]]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"distinct physical page indices"
                          (kv/ranges geometry pages)))))

(deftest a-geometry-whose-extents-overflow-is-refused-whole
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"64-bit capacity"
                        (kv/geometry {:layers 1 :page-size Long/MAX_VALUE :physical-pages 1
                                      :key-elements-per-token 2 :value-elements-per-token 1
                                      :storage-dtype :half}))))

(defn- two-device-plan
  [continuation]
  (distributed/plan
   {:id :restore
    :mesh (distributed/mesh [{:name :data :size 2}] [:gpu-0 :gpu-1])
    :topology (distributed/topology
               [(distributed/device {:id :gpu-0 :memory-capacity-bytes (* 8 1024 1024 1024)})
                (distributed/device {:id :gpu-1 :memory-capacity-bytes (* 8 1024 1024 1024)})]
               [(distributed/link {:id :gpu-0->gpu-1 :source :gpu-0 :target :gpu-1
                                   :bandwidth-bytes-s 1.0e9 :latency-ns 10000})])
    :values {(:value continuation) (kv/value-of geometry [5 2 9] [:gpu-0 :gpu-1])}
    :shards {(:value continuation) (kv/shards (:value continuation) geometry [5 2 9]
                                              [:gpu-0 :gpu-1])}
    :steps (into [(distributed/compute-step {:id :reserve-pages :device :gpu-1 :duration-ns 100})]
                 (:steps continuation))
    :outputs (mapv :id (:steps continuation))}))

(deftest a-continuation-transfers-one-leg-per-layer
  (let [continuation (kv/transfer {:id :continuation-7 :source :gpu-0 :target :gpu-1
                                   :route [:gpu-0->gpu-1] :geometry geometry :pages [5 2 9]
                                   :dependencies [:reserve-pages]})
        certified (distributed/certify (two-device-plan continuation))
        simulation (distributed/simulate (:plan certified))]
    (is (= [:continuation-7-layer-0 :continuation-7-layer-1 :continuation-7-layer-2]
           (mapv :id (:steps continuation))))
    (is (= (kv/total-bytes geometry [5 2 9]) (:bytes continuation)))
    (testing "the KV certificate binds geometry, pages and ranges to the plan's legs"
      (let [witness (kv/certificate :continuation-7 geometry [5 2 9])]
        (is (= (:bytes continuation) (kv/verify! certified witness)))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"legs disagree"
                              (kv/verify! (update-in certified [:plan :steps 1] assoc :bytes 1)
                                          witness)))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"does not match its geometry"
                              (kv/verify! certified (assoc witness :bytes 1))))))
    (is (= (:bytes continuation) (get-in simulation [:link-transfer-bytes :gpu-0->gpu-1])))
    (testing "legs serialize on the link and each waits only for the reservation"
      (let [timeline (:timeline simulation)
            starts (mapv #(get-in timeline [% :start-ns]) (mapv :id (:steps continuation)))]
        (is (apply < starts))
        (is (= 100 (first starts)) "the first leg starts when the reservation completes")))
    (is (distributed/verify! certified))))

(deftest a-transfer-plan-still-checks-its-route
  (let [continuation (kv/transfer {:id :c :source :gpu-0 :target :gpu-1
                                   :route [:gpu-1->gpu-0] :geometry geometry :pages [1]})]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"route"
                          (two-device-plan continuation)))))

(deftest overlap-follows-the-endpoints-stated-capabilities
  ;; a target that serializes transfers with compute (Level Zero today) cannot decode layer 0
  ;; while layer 1 is in flight; an OpenCL target can
  (let [plan-with (fn [capabilities]
                    (let [continuation (kv/transfer {:id :c :source :gpu-0 :target :gpu-1
                                                     :route [:gpu-0->gpu-1] :geometry geometry
                                                     :pages [5 2 9] :dependencies [:reserve-pages]
                                                     :capabilities {:gpu-1 capabilities}})
                          plan (two-device-plan continuation)
                          decode (distributed/compute-step
                                  {:id :decode-layer-0 :device :gpu-1 :duration-ns 50000
                                   :dependencies [:c-layer-0]})]
                      (distributed/simulate
                       (distributed/plan (assoc plan
                                                :steps (conj (:steps plan) decode)
                                                :outputs [:decode-layer-0 :c-layer-2])))))
        overlaps? (fn [simulation]
                    (let [timeline (:timeline simulation)
                          decode (get timeline :decode-layer-0)
                          leg (get timeline :c-layer-1)]
                      (< (:start-ns decode) (:finish-ns leg))))]
    (is (overlaps? (plan-with (ocl/transfer-capabilities))))
    (is (not (overlaps? (plan-with (ze/transfer-capabilities)))))
    (is (= [:gpu-1] (get-in (kv/transfer {:id :c :source :gpu-0 :target :gpu-1
                                          :route [:gpu-0->gpu-1] :geometry geometry :pages [1]
                                          :capabilities {:gpu-1 (ze/transfer-capabilities)}})
                            [:steps 0 :attributes :serialized-on])))))
