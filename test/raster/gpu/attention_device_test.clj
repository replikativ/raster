(ns raster.gpu.attention-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.attention :as attention]
            [raster.compiler.passes.parallel.attention-route :as route]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as device-probe]))

(defn- encode-halfs
  [values]
  (short-array (map #(Float/floatToFloat16 (float %)) values)))

(defn- decode-half
  [value]
  (double (Float/float16ToFloat (short value))))

(defn- make-case
  [route-kind visibility-kind]
  (let [dims {:batch-size 3 :q-heads 4 :kv-heads 2 :qk-head-dim 8
              :value-head-dim 6 :page-size 2 :physical-pages 7}
        {:keys [q-heads kv-heads qk-head-dim value-head-dim page-size physical-pages]} dims
        total-query-tokens 4
        q-elements (* total-query-tokens q-heads qk-head-dim)
        k-elements (* kv-heads physical-pages page-size qk-head-dim)
        v-elements (* kv-heads physical-pages page-size value-head-dim)
        q (encode-halfs
           (map #(* 0.07 (- (mod (+ (* 3 %) 1) 13) 6)) (range q-elements)))
        k (encode-halfs
           (map #(* 0.05 (- (mod (+ (* 5 %) 2) 17) 8)) (range k-elements)))
        v (encode-halfs
           (map #(* 0.04 (- (mod (+ (* 7 %) 3) 19) 9)) (range v-elements)))
        ;; Batch row 1 owns a query but has no resident KV tokens.  This exercises the cooperative
        ;; schedule's zero-length early exit independently from packed rows 0 and 2.
        q-offsets (int-array [0 2 3 4])
        q-positions (int-array [3 4 10 12])
        starts (int-array [0 0 8])
        query (attention/packed-query-batch
               {:values 'q :row-offsets 'q-row-offsets :positions 'q-positions
                :total-tokens total-query-tokens})
        route-values
        (case route-kind
          :dense-paged
          {:page-table (int-array [4 1 6, -1 -1 -1, 2 5 0])
           :lengths (int-array [5 0 5])
           :start-positions starts}

          :csr-paged
          {:page-offsets (int-array [0 3 3 6])
           :page-indices (int-array [4 1 6 2 5 0 -1])
           :last-page-lengths (int-array [1 0 1])
           :start-positions starts})
        kv-route
        (case route-kind
          :dense-paged
          (attention/dense-paged-route
           {:page-table 'page-table :lengths 'kv-lengths
            :start-positions 'kv-start-positions :pages-per-sequence 3})

          :csr-paged
          (attention/csr-paged-route
           {:page-offsets 'page-offsets :page-indices 'page-indices
            :last-page-lengths 'last-page-lengths
            :start-positions 'kv-start-positions :page-index-capacity 7}))
        visibility-values
        (when (= :csr visibility-kind)
          ;; The third query belongs to batch row 1, whose routed KV length is zero.
          {:row-offsets (int-array [0 2 3 3 6])
           :key-indices (int-array [1 3, 2, 2 3 4, -1 -1 -1])})
        attention-visibility
        (case visibility-kind
          :interval
          (attention/visibility {:causal? true :window-left 2 :window-right 0})

          :csr
          (attention/csr-visibility
           {:row-offsets 'attention-row-offsets :key-indices 'attention-key-indices
            :key-index-capacity 9 :duplicate-policy :set
            :position-filter (attention/visibility
                              {:causal? true :window-left 2 :window-right 0})}))
        [k-layout v-layout] (if (= :dense-paged route-kind)
                              [:kv-head-major :page-major]
                              [:page-major :kv-head-major])
        problem (attention/make
                 (merge dims
                        {:id [:device-reference route-kind visibility-kind]
                         :query query :k-pages 'k-pages :v-pages 'v-pages
                         :route kv-route :output 'output
                         :k-layout k-layout :v-layout v-layout
                         :visibility attention-visibility
                         :scale (/ 1.0 (Math/sqrt (double qk-head-dim)))}))]
    (attention/validate-query-values! problem q-offsets q-positions)
    (attention/validate-routing! problem route-values)
    (attention/validate-visibility-values!
     problem q-offsets route-values visibility-values)
    {:problem problem :q q :k k :v v
     :q-offsets q-offsets :q-positions q-positions
     :route-values route-values :visibility-values visibility-values}))

(defn- cache-index
  [{:keys [physical-pages page-size kv-heads]} layout dim kv-head page token d]
  (case layout
    :kv-head-major
    (+ (* (+ (* (+ (* kv-head physical-pages) page) page-size) token) dim) d)
    :page-major
    (+ (* (+ (* (+ (* page page-size) token) kv-heads) kv-head) dim) d)))

(defn- route-row
  [{:keys [problem route-values]} batch]
  (let [{:keys [route page-size]} problem]
    (if (attention/dense-paged-route? route)
      (let [length (aget ^ints (:lengths route-values) batch)
            pps (:pages-per-sequence route)]
        {:length length
         :page #(aget ^ints (:page-table route-values) (+ (* batch pps) %))})
      (let [begin (aget ^ints (:page-offsets route-values) batch)
            end (aget ^ints (:page-offsets route-values) (inc batch))
            pages (- end begin)
            last (aget ^ints (:last-page-lengths route-values) batch)]
        {:length (if (zero? pages) 0 (+ (* (dec pages) page-size) last))
         :page #(aget ^ints (:page-indices route-values) (+ begin %))}))))

(defn- query-batch
  [^ints offsets q-token]
  (first (filter (fn [batch]
                   (<= (aget offsets batch) q-token (dec (aget offsets (inc batch)))))
                 (range (dec (alength offsets))))))

(defn- visible?
  [{:keys [causal? window-left window-right]} q-position kv-position]
  (and (or (not causal?) (<= kv-position q-position))
       (or (nil? window-left) (>= kv-position (- q-position window-left)))
       (or (nil? window-right) (<= kv-position (+ q-position window-right)))))

(defn- logical-tokens
  [{:keys [problem visibility-values]} q-token length]
  (let [visibility (:visibility problem)]
    (if (attention/csr-visibility? visibility)
      (let [begin (aget ^ints (:row-offsets visibility-values) q-token)
            end (aget ^ints (:row-offsets visibility-values) (inc q-token))]
        (mapv #(aget ^ints (:key-indices visibility-values) %) (range begin end)))
      (range length))))

(defn- reference
  [{:keys [problem q k v q-offsets q-positions route-values] :as test-case}]
  (let [{:keys [query q-heads kv-heads qk-head-dim value-head-dim page-size scale
                k-layout v-layout visibility]} problem
        output (double-array (* (:total-tokens query) q-heads value-head-dim))
        gqa-ratio (quot q-heads kv-heads)]
    (dotimes [q-token (:total-tokens query)]
      (let [batch (query-batch q-offsets q-token)
            q-position (aget ^ints q-positions q-token)
            kv-start (aget ^ints (:start-positions route-values) batch)
            {:keys [length page]} (route-row test-case batch)]
        (dotimes [q-head q-heads]
          (let [kv-head (quot q-head gqa-ratio)
                q-base (* (+ (* q-token q-heads) q-head) qk-head-dim)
                visible-tokens (filterv #(visible? (attention/position-filter visibility)
                                                   q-position (+ kv-start %))
                                        (logical-tokens test-case q-token length))
                logits
                (mapv
                 (fn [token]
                   (let [physical-page (page (quot token page-size))
                         page-token (rem token page-size)]
                     (* scale
                        (reduce +
                                (map (fn [d]
                                       (* (decode-half (aget ^shorts q (+ q-base d)))
                                          (decode-half
                                           (aget ^shorts k
                                                 (cache-index problem k-layout qk-head-dim
                                                              kv-head physical-page page-token d)))))
                                     (range qk-head-dim))))))
                 visible-tokens)
                maximum (when (seq logits) (reduce max logits))
                weights (mapv #(Math/exp (- (double %) maximum)) logits)
                denominator (reduce + 0.0 weights)]
            (dotimes [d value-head-dim]
              (let [value
                    (if (zero? denominator)
                      0.0
                      (/ (reduce + 0.0
                                 (map (fn [token weight]
                                        (let [physical-page (page (quot token page-size))
                                              page-token (rem token page-size)]
                                          (* weight
                                             (decode-half
                                              (aget ^shorts v
                                                    (cache-index
                                                     problem v-layout value-head-dim
                                                     kv-head physical-page page-token d))))))
                                      visible-tokens weights))
                         denominator))
                    out-index (+ (* (+ (* q-token q-heads) q-head) value-head-dim) d)]
                (aset output out-index value)))))))
    output))

(defn- route-buffer-data
  [{:keys [problem route-values]}]
  (let [route (:route problem)]
    (if (attention/dense-paged-route? route)
      {:page-table [:int (alength ^ints (:page-table route-values)) (:page-table route-values)]
       :kv-lengths [:int (alength ^ints (:lengths route-values)) (:lengths route-values)]
       :kv-start-positions [:int (alength ^ints (:start-positions route-values))
                            (:start-positions route-values)]}
      {:page-offsets [:int (alength ^ints (:page-offsets route-values))
                      (:page-offsets route-values)]
       :page-indices [:int (alength ^ints (:page-indices route-values))
                      (:page-indices route-values)]
       :last-page-lengths [:int (alength ^ints (:last-page-lengths route-values))
                           (:last-page-lengths route-values)]
       :kv-start-positions [:int (alength ^ints (:start-positions route-values))
                            (:start-positions route-values)]})))

(defn- visibility-buffer-data
  [{:keys [visibility-values]}]
  (if visibility-values
    {:attention-row-offsets [:int (alength ^ints (:row-offsets visibility-values))
                             (:row-offsets visibility-values)]
     :attention-key-indices [:int (alength ^ints (:key-indices visibility-values))
                             (:key-indices visibility-values)]}
    {}))

(defn- graph-bindings
  [problem]
  (let [{:keys [query route]} problem
        common {(:values query) :q (:row-offsets query) :q-row-offsets
                (:positions query) :q-positions
                (:k-pages problem) :k-pages (:v-pages problem) :v-pages
                (:start-positions route) :kv-start-positions
                (:output problem) :output}]
    (merge common
           (if (attention/dense-paged-route? route)
             {(:page-table route) :page-table (:lengths route) :kv-lengths}
             {(:page-offsets route) :page-offsets (:page-indices route) :page-indices
              (:last-page-lengths route) :last-page-lengths})
           (when (attention/csr-visibility? (:visibility problem))
             {(:row-offsets (:visibility problem)) :attention-row-offsets
              (:key-indices (:visibility problem)) :attention-key-indices}))))

(defn- run-case
  [device-id route-kind visibility-kind policy expected-strategy]
  (let [{:keys [problem q k v q-offsets q-positions] :as test-case}
        (make-case route-kind visibility-kind)
        descriptor (assoc (hardware/descriptor-for device-id)
                          :segmented-weighted-reduction-schedule policy)
        routed (route/route! problem descriptor)
        _ (is (= expected-strategy (:strategy routed))
              "the device test must execute the intended attention leaf")
        graph (:graph routed)
        expected (reference test-case)
        specs (attention/buffer-specs problem)
        allocations
        (merge {:q [:half (get-in specs ['q :elements]) q]
                :q-row-offsets [:int (get-in specs ['q-row-offsets :elements]) q-offsets]
                :q-positions [:int (get-in specs ['q-positions :elements]) q-positions]
                :k-pages [:half (get-in specs ['k-pages :elements]) k]
                :v-pages [:half (get-in specs ['v-pages :elements]) v]
                :output [:half (get-in specs ['output :elements]) nil]}
               (route-buffer-data test-case)
               (visibility-buffer-data test-case))]
    (gpu/with-gpu-session [session device-id]
      (gpu/alloc! session allocations)
      (let [handle (gpu/bind-kernel-graph!
                    session [:attention route-kind] graph (graph-bindings problem) {})]
        (try
          (gpu/run-kernel-graph! session handle)
          (let [actual-bits ^shorts (gpu/download session :output)
                actual (mapv decode-half actual-bits)]
            (is (= (count expected) (count actual)))
            (is (every? true?
                        (map (fn [wanted got]
                               (< (Math/abs (- (double wanted) (double got))) 0.012))
                             expected actual))))
          (finally
            (gpu/release-kernel-graph! session handle)))))))

(defn- run-mixed-io-case
  [device-id policy expected-strategy]
  (let [{:keys [problem q k v q-offsets q-positions] :as test-case}
        (make-case :dense-paged :interval)
        expected (reference test-case)
        problem (assoc problem :q-dtype :float :output-dtype :float)
        descriptor (assoc (hardware/descriptor-for device-id)
                          :segmented-weighted-reduction-schedule policy)
        routed (route/route! problem descriptor)
        _ (is (= expected-strategy (:strategy routed))
              "the device test must execute the intended attention leaf")
        graph (:graph routed)
        specs (attention/buffer-specs problem)
        float-q (float-array (map decode-half q))
        allocations
        (merge {:q [:float (get-in specs ['q :elements]) float-q]
                :q-row-offsets [:int (get-in specs ['q-row-offsets :elements]) q-offsets]
                :q-positions [:int (get-in specs ['q-positions :elements]) q-positions]
                :k-pages [:half (get-in specs ['k-pages :elements]) k]
                :v-pages [:half (get-in specs ['v-pages :elements]) v]
                :output [:float (get-in specs ['output :elements]) nil]}
               (route-buffer-data test-case))]
    (gpu/with-gpu-session [session device-id]
      (gpu/alloc! session allocations)
      (let [handle (gpu/bind-kernel-graph!
                    session [:attention :mixed-io] graph (graph-bindings problem) {})]
        (try
          (gpu/run-kernel-graph! session handle)
          (let [actual ^floats (gpu/download session :output)]
            (is (= (count expected) (alength actual)))
            (is (every? true?
                        (map (fn [wanted got]
                               (< (Math/abs (- (double wanted) (double got))) 1.0e-5))
                             expected actual))))
          (finally
            (gpu/release-kernel-graph! session handle)))))))

(deftest level-zero-packed-dense-attention-matches-reference
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "packed dense-routed FP16 attention on Level Zero")
    (do
      (run-case :ze:0 :dense-paged :interval :reference :fp16-reference)
      (run-case :ze:0 :dense-paged :interval :subgroup-score-reuse
                :routed-paged-subgroup-online-score-reuse))))

(deftest level-zero-fp32-query-and-output-with-fp16-kv-matches-reference
  (if-not @gp/gpu-available?
    (gp/gpu-skip! "FP32-I/O packed attention over FP16 KV on Level Zero")
    (run-mixed-io-case :ze:0 :subgroup-score-reuse
                       :routed-paged-subgroup-online-score-reuse)))

(deftest opencl-packed-csr-attention-matches-reference
  (if-not @device-probe/opencl-fp16-available?
    (device-probe/opencl-skip! "packed CSR attention" :fp16)
    (run-case :ocl:0 :csr-paged :interval :auto :fp16-reference)))

(deftest opencl-logical-csr-visibility-over-dense-pages-matches-reference
  (if-not @device-probe/opencl-fp16-available?
    (device-probe/opencl-skip! "logical CSR visibility over dense pages" :fp16)
    (run-case :ocl:0 :dense-paged :csr :auto :fp16-reference)))
