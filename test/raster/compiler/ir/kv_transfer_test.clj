(ns raster.compiler.ir.kv-transfer-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.kv-transfer :as kv]))

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
    (is (= (:bytes continuation) (kv/certified-bytes certified :continuation-7))
        "the certificate recomputes the bytes from the legs")
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
