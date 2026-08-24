(ns raster.gpu.paged-attention-device-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [raster.compiler.ir.paged-attention :as paged]
            [raster.compiler.passes.parallel.paged-attention-route :as route]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.core :as gpu]))

(def ^:private ocl-fp16-available?
  (delay
    (try
      (require 'raster.gpu.ocl-runtime)
      ((resolve 'raster.gpu.ocl-runtime/init!))
      (boolean
       (some #(str/includes? (or (:extensions %) "") "cl_khr_fp16")
             ((resolve 'raster.gpu.ocl-runtime/query-devices))))
      (catch Throwable _ false))))

(defn- encode-halfs
  [values]
  (short-array (map #(Float/floatToFloat16 (float %)) values)))

(defn- decode-half
  [value]
  (double (Float/float16ToFloat (short value))))

(defn- make-case
  [cache-layout]
  (let [dims {:batch-size 2 :q-heads 4 :kv-heads 2 :head-dim 8
              :page-size 2 :physical-pages 7 :pages-per-sequence 3}
        {:keys [batch-size q-heads kv-heads head-dim page-size physical-pages]} dims
        q-elements (* batch-size q-heads head-dim)
        cache-elements (* kv-heads physical-pages page-size head-dim)
        q (encode-halfs
           (map #(* 0.07 (- (mod (+ (* 3 %) 1) 13) 6)) (range q-elements)))
        k (encode-halfs
           (map #(* 0.05 (- (mod (+ (* 5 %) 2) 17) 8)) (range cache-elements)))
        v (encode-halfs
           (map #(* 0.04 (- (mod (+ (* 7 %) 3) 19) 9)) (range cache-elements)))
        table (int-array [4 1 6, 2 5 0])
        lengths (int-array [5 3])
        operation (paged/make
                   (merge dims
                          {:id :device-reference
                           :q 'q :k-pages 'k-pages :v-pages 'v-pages
                           :page-table 'page-table :lengths 'lengths :output 'output
                           :cache-layout cache-layout
                           :scale (/ 1.0 (Math/sqrt (double head-dim)))}))]
    (paged/validate-routing! operation table lengths)
    {:operation operation :q q :k k :v v :table table :lengths lengths}))

(defn- cache-index
  [{:keys [physical-pages page-size kv-heads head-dim cache-layout]}
   kv-head page token d]
  (case cache-layout
    :kv-head-major
    (+ (* (+ (* (+ (* kv-head physical-pages) page) page-size) token) head-dim) d)
    :page-major
    (+ (* (+ (* (+ (* page page-size) token) kv-heads) kv-head) head-dim) d)))

(defn- reference
  [{:keys [operation q k v table lengths]}]
  (let [{:keys [batch-size q-heads kv-heads head-dim page-size pages-per-sequence scale]}
        operation
        output (double-array (* batch-size q-heads head-dim))
        gqa-ratio (quot q-heads kv-heads)]
    (dotimes [batch batch-size]
      (dotimes [q-head q-heads]
        (let [kv-head (quot q-head gqa-ratio)
              q-base (* (+ (* batch q-heads) q-head) head-dim)
              length (aget ^ints lengths batch)
              logits
              (mapv
               (fn [token]
                 (let [logical-page (quot token page-size)
                       page (aget ^ints table (+ (* batch pages-per-sequence) logical-page))
                       page-token (rem token page-size)]
                   (* scale
                      (reduce +
                              (map (fn [d]
                                     (* (decode-half (aget ^shorts q (+ q-base d)))
                                        (decode-half
                                         (aget ^shorts k
                                               (cache-index operation kv-head page page-token d)))))
                                   (range head-dim))))))
               (range length))
              maximum (reduce max logits)
              weights (mapv #(Math/exp (- (double %) maximum)) logits)
              denominator (reduce + weights)]
          (dotimes [d head-dim]
            (let [value
                  (/ (reduce +
                             (map-indexed
                              (fn [token weight]
                                (let [logical-page (quot token page-size)
                                      page (aget ^ints table
                                                 (+ (* batch pages-per-sequence) logical-page))
                                      page-token (rem token page-size)]
                                  (* weight
                                     (decode-half
                                      (aget ^shorts v
                                            (cache-index operation kv-head page page-token d))))))
                              weights))
                     denominator)]
              (aset output (+ q-base d) value))))))
    output))

(defn- run-case
  [device-id cache-layout]
  (let [{:keys [operation q k v table lengths] :as test-case} (make-case cache-layout)
        graph (:graph (route/route!
                       operation {:device-type :gpu :subgroup-size 16
                                  :max-workgroup-size 256}))
        expected (reference test-case)
        specs (paged/buffer-specs operation)]
    (gpu/with-gpu-session [session device-id]
      (gpu/alloc! session
                  {:q [:half (get-in specs ['q :elements]) q]
                   :k-pages [:half (get-in specs ['k-pages :elements]) k]
                   :v-pages [:half (get-in specs ['v-pages :elements]) v]
                   :page-table [:int (get-in specs ['page-table :elements]) table]
                   :lengths [:int (get-in specs ['lengths :elements]) lengths]
                   :output [:half (get-in specs ['output :elements]) nil]})
      (let [handle (gpu/bind-kernel-graph!
                    session :paged-attention graph
                    {'q :q 'k-pages :k-pages 'v-pages :v-pages
                     'page-table :page-table 'lengths :lengths 'output :output}
                    {})]
        (try
          (gpu/run-kernel-graph! session handle)
          (let [actual-bits ^shorts (gpu/download session :output)
                actual (mapv decode-half actual-bits)]
            (is (= (count expected) (count actual)))
            (is (every? true?
                        (map (fn [wanted got]
                               (< (Math/abs (- (double wanted) (double got))) 0.01))
                             expected actual))))
          (finally
            (gpu/release-kernel-graph! session handle)))))))

(deftest level-zero-routed-paged-attention-matches-reference
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "routed FP16 paged attention on Level Zero")
    (run-case :ze:0 :kv-head-major)))

(deftest opencl-routed-paged-attention-matches-reference
  (if-not @ocl-fp16-available?
    (is true "OpenCL FP16 device unavailable")
    (run-case :ocl:0 :page-major)))
