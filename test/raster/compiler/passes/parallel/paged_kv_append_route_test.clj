(ns raster.compiler.passes.parallel.paged-kv-append-route-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.paged-kv-append :as append]
            [raster.compiler.passes.parallel.paged-kv-append-route :as route]))

(defn- problem
  ([] (problem {}))
  ([opts]
   (append/make
    (merge {:id :append :key-rows 'key-rows :value-rows 'value-rows
            :slot-mapping 'slots :key-pages 'key-pages :value-pages 'value-pages
            :batch-size 3 :key-elements-per-token 8 :value-elements-per-token 6
            :page-size 4 :physical-pages 5}
           opts))))

(defn- reason
  [thunk]
  (try (thunk) nil
       (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest problem-exposes-exact-buffer-geometry
  (let [value (problem)
        specs (append/buffer-specs value)]
    (is (= 20 (append/physical-slots value)))
    (is (= {:dtype :float :elements 24 :role :input} (get specs 'key-rows)))
    (is (= {:dtype :float :elements 18 :role :input} (get specs 'value-rows)))
    (is (= {:dtype :int :elements 3 :role :input} (get specs 'slots)))
    (is (= {:dtype :half :elements 160 :role :inout} (get specs 'key-pages)))
    (is (= {:dtype :half :elements 120 :role :inout} (get specs 'value-pages)))))

(deftest active-slot-values-must-be-unique-and-in-bounds
  (let [value (problem)]
    (is (= [19 0 7] (append/validate-slot-values! value [19 0 7])))
    (is (= [-1 -1 7] (append/validate-slot-values! value [-1 -1 7]))
        "inactive lanes may share the no-write sentinel")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"unique destination"
                          (append/validate-slot-values! value [1 1 2])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"outside the physical pool"
                          (append/validate-slot-values! value [1 -2 2])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"outside the physical pool"
                          (append/validate-slot-values! value [1 2 20])))))

(deftest routed-artifact-is-assignment-with-in-place-page-effects
  (let [{:keys [artifact graph strategy]} (route/route! (problem))
        source (:source artifact)]
    (is (= :fp32-to-fp16-reference strategy))
    (is (str/includes? source "key_pages[dst] = convert_half_rte(key_rows[src]);"))
    (is (str/includes? source "value_pages[dst] = convert_half_rte(value_rows[src]);"))
    (is (str/includes? source "if (slot < 0"))
    (is (not (str/includes? source "atomic")))
    (is (= ['key-rows 'value-rows 'slots 'key-pages 'value-pages]
           (:arguments artifact)))
    (is (= :paged-kv-append (get-in graph [:provenance :semantic-op])))
    (is (= #{'key-pages 'value-pages}
           (set (map :id (:outputs graph)))))
    (is (every? #(= :inout (:role %)) (:outputs graph)))
    (is (graph/kernel-graph? graph))))

(deftest unsupported-storage-declines-before-emission
  (let [result (route/route (problem {:key-storage-dtype :float}))]
    (is (nil? (:strategy result)))
    (is (= :paged-kv-append-dtype-unsupported
           (get-in result [:declines 0 :reason]))))
  (testing "a failed route retains a stable top-level reason"
    (is (= :paged-kv-append-no-kernel-route
           (reason #(route/route! (problem {:value-storage-dtype :double})))))))
