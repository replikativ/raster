(ns raster.compiler.ir.attention-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.attention :as attention]))

(defn- query
  [& overrides]
  (attention/packed-query-batch
   (merge {:values 'q :row-offsets 'q-row-offsets :positions 'q-positions :total-tokens 4}
          (apply hash-map overrides))))

(defn- dense-route
  [& overrides]
  (attention/dense-paged-route
   (merge {:page-table 'page-table :lengths 'kv-lengths
           :start-positions 'kv-start-positions :pages-per-sequence 3}
          (apply hash-map overrides))))

(defn- csr-route
  [& overrides]
  (attention/csr-paged-route
   (merge {:page-offsets 'page-offsets :page-indices 'page-indices
           :last-page-lengths 'last-page-lengths
           :start-positions 'kv-start-positions :page-index-capacity 7}
          (apply hash-map overrides))))

(defn- problem
  [& overrides]
  (attention/make
   (merge {:id :attention :query (query) :k-pages 'k-pages :v-pages 'v-pages
           :route (dense-route) :output 'output
           :batch-size 3 :q-heads 4 :kv-heads 2
           :qk-head-dim 8 :value-head-dim 6
           :page-size 2 :physical-pages 7 :scale (/ 1.0 (Math/sqrt 8.0))
           :k-layout :kv-head-major :v-layout :page-major
           :visibility (attention/visibility
                        {:causal? true :window-left 2 :window-right 0})}
          (apply hash-map overrides))))

(deftest semantic-problem-separates-query-visibility-storage-and-routing
  (let [dense (problem)
        csr (problem :route (csr-route))]
    (is (attention/attention-problem? dense))
    (is (= :dense-paged (attention/route-kind (:route dense))))
    (is (= :csr-paged (attention/route-kind (:route csr))))
    (is (= {:causal? true :window-left 2 :window-right 0}
           (into {} (:visibility dense))))
    (is (= {:q [4 4 8]
            :q-row-offsets [4]
            :q-positions [4]
            :k-pages [2 7 2 8]
            :v-pages [7 2 2 6]
            :kv-start-positions [3]
            :output [4 4 6]
            :page-table [3 3]
            :kv-lengths [3]}
           (attention/layouts dense)))
    (is (= [4] (:q-row-offsets (attention/layouts csr))))
    (is (= [7] (:page-indices (attention/layouts csr))))
    (is (= 96 (get-in (attention/buffer-specs dense) ['output :elements])))
    (is (= {:dtype :half :quantization :none} (:k-format dense)))
    (is (= {:dtype :half :quantization :none} (:v-format dense)))))

(deftest packed-query-and-route-values-are-validated-before-upload
  (let [dense (problem)
        csr (problem :route (csr-route))]
    (is (= dense (attention/validate-query-values! dense [0 2 2 4] [3 4 10 12])))
    (is (= dense
           (attention/validate-routing!
            dense {:page-table [4 1 6, -1 -1 -1, 2 5 0]
                   :lengths [5 0 5]
                   :start-positions [0 0 8]})))
    (is (= csr
           (attention/validate-routing!
            csr {:page-offsets [0 3 3 6]
                 :page-indices [4 1 6 2 5 0 -1]
                 :last-page-lengths [1 0 1]
                 :start-positions [0 0 8]})))
    (testing "empty packed rows are legal but offsets cover every query exactly"
      (is (= :attention-invalid-query-offsets
             (try
               (attention/validate-query-values! dense [0 2 2 3] [3 4 10 12])
               (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (testing "positions strictly increase within each nonempty row"
      (is (= :attention-nonmonotone-query-positions
             (try
               (attention/validate-query-values! dense [0 2 2 4] [4 3 10 12])
               (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (testing "unused dense padding may contain a sentinel, selected pages may not"
      (is (= :attention-invalid-physical-page
             (try
               (attention/validate-routing!
                dense {:page-table [-1 1 6, -1 -1 -1, 2 5 0]
                       :lengths [5 0 5] :start-positions [0 0 8]})
               (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (testing "CSR empty rows have final-page length zero"
      (is (= :attention-invalid-last-page-length
             (try
               (attention/validate-routing!
                csr {:page-offsets [0 3 3 6]
                     :page-indices [4 1 6 2 5 0 -1]
                     :last-page-lengths [1 1 1]
                     :start-positions [0 0 8]})
               (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest invalid-static-semantics-fail-before-emission
  (testing "GQA mapping must be integral"
    (is (= :attention-invalid-gqa-ratio
           (try
             (problem :q-heads 3)
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
  (testing "logical buffers cannot alias accidentally"
    (is (= :attention-duplicate-buffer-identity
           (try
             (problem :output 'q)
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
  (testing "allocation arithmetic is checked"
    (is (= :attention-extent-overflow
           (try
             (problem :batch-size Long/MAX_VALUE)
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
  (testing "K and V layouts are independent and explicit"
    (is (= :attention-unsupported-cache-layout
           (try
             (problem :v-layout :implicit)
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
