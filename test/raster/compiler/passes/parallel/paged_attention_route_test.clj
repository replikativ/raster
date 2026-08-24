(ns raster.compiler.passes.parallel.paged-attention-route-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.paged-attention :as paged]
            [raster.compiler.passes.parallel.paged-attention-route :as route]))

(defn- operation
  [& overrides]
  (paged/make
   (merge {:id :decode-attention
           :q 'q :k-pages 'k-pages :v-pages 'v-pages
           :page-table 'page-table :lengths 'lengths :output 'output
           :batch-size 2 :q-heads 4 :kv-heads 2 :head-dim 20
           :page-size 2 :physical-pages 7 :pages-per-sequence 3
           :scale 0.25}
          (apply hash-map overrides))))

(deftest fp16-reference-route-is-a-complete-executable-compiler-value
  (let [{:keys [strategy reference? declines artifact graph schedule]}
        (route/route! (operation)
                      {:device-type :gpu :subgroup-size 16 :max-workgroup-size 256})]
    (is (= :fp16-reference strategy))
    (is reference?)
    (is (empty? declines))
    (is (kart/kernel-artifact? artifact))
    (is (kgraph/kernel-graph? graph))
    (is (= '[q k-pages v-pages page-table lengths output]
           (:arguments artifact)))
    (is (= ["q" "k_pages" "v_pages" "page_table" "lengths" "output"]
           (mapv :c-name (:abi artifact))))
    (is (= [:half :half :half :int :int :half]
           (mapv :dtype (:abi artifact))))
    (is (= {:workgroup-size [16 1 1] :group-count [2 4 2]} schedule))
    (is (= 3 (count (:workgroup-size (:launch artifact)))))
    (is (= 1 (count (:nodes graph))))
    (is (= #{'q 'k-pages 'v-pages 'page-table 'lengths}
           (set (map :id (:inputs graph)))))
    (is (= ['output] (mapv :id (:outputs graph))))
    (is (str/includes? (:source artifact) "const int kv_head = q_head / 2;"))
    (is (str/includes? (:source artifact) "physical_page = page_table"))
    (is (str/includes? (:source artifact) "float accumulator = 0.0f;"))))

(deftest page-major-cache-layout-is-lowered-without-repacking
  (let [artifact (:artifact (route/route! (operation :cache-layout :page-major)))]
    (is (= :page-major (get-in artifact [:attributes :cache-layout])))
    (is (str/includes? (:source artifact)
                       "((physical_page * 2 + page_token) * 2 + kv_head) * 20"))))

(deftest unsupported-representations-return-machine-readable-declines
  (testing "quantization declines on its missing ABI before generic dtype routing"
    (let [r (route/route
             (operation :cache-dtype :byte
                        :cache-format {:dtype :byte :quantization :int8
                                       :group-size 32}))]
      (is (nil? (:strategy r)))
      (is (= :paged-attention-quantized-cache-abi-unimplemented
             (get-in r [:declines 0 :reason])))))
  (testing "plain non-FP16 storage declines rather than being silently reinterpreted"
    (let [r (route/route (operation :q-dtype :float :output-dtype :float))]
      (is (= :paged-attention-reference-storage-unsupported
             (get-in r [:declines 0 :reason])))))
  (testing "a CPU target cannot accidentally receive OpenCL GPU scheduling"
    (is (= :paged-attention-requires-gpu
           (get-in (route/route (operation) {:device-type :cpu})
                   [:declines 0 :reason]))))
  (is (= :paged-attention-no-kernel-route
         (try
           (route/route! (operation) {:device-type :cpu})
           (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
