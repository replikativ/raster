(ns raster.compiler.ir.kv-transfer-device-test
  "The range list a KV transfer plan derives is executable byte for byte by the resident range
   API: a continuation's pages move from one page pool to another with `copy-range!`, and
   nothing else in the target pool changes."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kv-transfer :as kv]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as device-probe]))

(def ^:private geometry
  (kv/geometry {:layers 2 :page-size 4 :physical-pages 6
                :key-elements-per-token 8 :value-elements-per-token 8
                :storage-dtype :float}))

(defn- pool-buffers
  "One K and one V buffer per layer, each `physical-pages × page-size × elements-per-token`."
  [prefix]
  (into {}
        (for [layer (range (:layers geometry))
              [buffer per-token] [[:key (:key-elements-per-token geometry)]
                                  [:value (:value-elements-per-token geometry)]]]
          [(keyword (str prefix "-" layer "-" (name buffer)))
           [:float (* (:physical-pages geometry) (kv/page-elements geometry per-token)) nil]])))

(defn- buffer-key [prefix {:keys [layer buffer]}]
  (keyword (str prefix "-" layer "-" (name buffer))))

(deftest a-continuation-moves-between-page-pools-by-its-range-list
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "KV transfer ranges on OpenCL")
    (let [session (gpu/make-session :ocl:0)
          pages [4 1]
          ranges (kv/ranges geometry pages)
          n (* (:physical-pages geometry) (kv/page-elements geometry 8))]
      (try
        (gpu/alloc! session (merge (pool-buffers "source") (pool-buffers "target")))
        (let [sources (into {} (for [[k _] (pool-buffers "source")]
                                 [k (float-array (map #(float (+ % (* 1000 (hash k)))) (range n)))]))
              targets (into {} (for [[k _] (pool-buffers "target")]
                                 [k (float-array n (float -1.0))]))]
          (doseq [[k array] (merge sources targets)]
            (gpu/upload-range! session k array {:elements n}))
          (doseq [{:keys [element elements] :as range} ranges]
            (gpu/copy-range! session (buffer-key "source" range) (buffer-key "target" range)
                             {:src-element element :dst-element element :elements elements}))
          (testing "every page of the continuation arrived and nothing else was written"
            (doseq [[k _] (pool-buffers "target")]
              (let [layer (Long/parseLong (second (re-find #"target-(\d+)-" (name k))))
                    buffer (keyword (last (clojure.string/split (name k) #"-")))
                    source (get sources (keyword (str "source-" layer "-" (name buffer))))
                    moved (float-array n)
                    _ (gpu/download-range! session k moved {:elements n})
                    page-of (fn [e] (quot e (kv/page-elements geometry 8)))]
                (dotimes [e n]
                  (is (= (if (some #{(page-of e)} pages) (aget source e) (float -1.0))
                         (aget moved e)))))))
          (testing "the certified byte count equals the bytes the range list moved"
            (is (= (kv/total-bytes geometry pages)
                   (* 4 (reduce + (map :elements ranges)))))))
        (finally (gpu/close-session! session))))))
