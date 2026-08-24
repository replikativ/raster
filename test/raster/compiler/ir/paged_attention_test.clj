(ns raster.compiler.ir.paged-attention-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.paged-attention :as paged]))

(defn- operation
  [& overrides]
  (paged/make
   (merge {:id :decode-attention
           :q 'q :k-pages 'k-pages :v-pages 'v-pages
           :page-table 'page-table :lengths 'lengths :output 'output
           :batch-size 2 :q-heads 4 :kv-heads 2 :head-dim 8
           :page-size 2 :physical-pages 7 :pages-per-sequence 3
           :scale (/ 1.0 (Math/sqrt 8.0))}
          (apply hash-map overrides))))

(deftest descriptor-proves-layout-and-storage-extents
  (let [op (operation)
        specs (paged/buffer-specs op)]
    (is (paged/paged-attention? op))
    (is (= {:q [2 4 8]
            :k-pages [2 7 2 8]
            :v-pages [2 7 2 8]
            :page-table [2 3]
            :lengths [2]
            :output [2 4 8]}
           (paged/layouts op)))
    (is (= {:role :input :dtype :half :shape [2 4 8] :elements 64}
           (get specs 'q)))
    (is (= 224 (get-in specs ['k-pages :elements])))
    (is (= {:dtype :half :quantization :none} (:cache-format op)))
    (is (= :kv-head-major (:cache-layout op)))
    (is (= {:dtype :float :quantization :none}
           (:cache-format (operation :cache-dtype :float))))
    (is (= [7 2 2 8]
           (:k-pages (paged/layouts (operation :cache-layout :page-major)))))))

(deftest descriptor-refuses-invalid-semantics-before-emission
  (testing "GQA mapping must be integral"
    (is (= :paged-attention-invalid-gqa-ratio
           (try
             (operation :q-heads 3)
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
  (testing "logical buffers cannot alias accidentally"
    (is (= :paged-attention-duplicate-buffer-identity
           (try
             (operation :output 'q)
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
  (testing "allocation arithmetic is checked"
    (is (= :paged-attention-extent-overflow
           (try
             (operation :batch-size Long/MAX_VALUE :q-heads 2)
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
  (testing "cache stride order is explicit"
    (is (= :paged-attention-unsupported-cache-layout
           (try
             (operation :cache-layout :implicit)
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest host-visible-routing-is-validated-before-upload
  (let [op (operation)
        table [4 1 6 2 5 0]
        lengths [5 3]]
    (is (= op (paged/validate-routing! op table lengths)))
    (is (= :paged-attention-page-table-shape
           (try
             (paged/validate-routing! op (pop table) lengths)
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= :paged-attention-invalid-physical-page
           (try
             (paged/validate-routing! op (assoc table 1 7) lengths)
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= :paged-attention-invalid-length
           (try
             (paged/validate-routing! op table [7 3])
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
