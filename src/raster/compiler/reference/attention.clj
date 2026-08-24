(ns raster.compiler.reference.attention
  "Independent host oracle for semantic routed attention and its VJP.

   Values are supplied by compiler buffer identity. The oracle operates in double precision and
   is deliberately schedule-free: dense and CSR page routes must produce the same logical result.
   It is a correctness/differential-test implementation, not a runtime fallback."
  (:require [raster.compiler.ir.attention :as attention]
            [raster.compiler.ir.attention-ad :as attention-ad]))

(defn- value-vector!
  [values buffer-id expected-elements]
  (let [value (get values buffer-id ::missing)]
    (when (= ::missing value)
      (throw (ex-info "attention reference is missing a buffer value"
                      {:reason :attention-reference-missing-value :buffer buffer-id})))
    (let [value (vec value)]
      (when-not (= expected-elements (count value))
        (throw (ex-info "attention reference buffer has the wrong element count"
                        {:reason :attention-reference-value-shape
                         :buffer buffer-id :expected expected-elements :actual (count value)})))
      value)))

(defn- checked-values
  [problem values extra-specs]
  (let [specs (merge (dissoc (attention/buffer-specs problem) (:output problem)) extra-specs)]
    (into {}
          (map (fn [[buffer-id {:keys [elements]}]]
                 [buffer-id (value-vector! values buffer-id elements)]))
          specs)))

(defn- route-values
  [problem values]
  (let [route (:route problem)]
    (if (attention/dense-paged-route? route)
      {:page-table (get values (:page-table route))
       :lengths (get values (:lengths route))
       :start-positions (get values (:start-positions route))}
      {:page-offsets (get values (:page-offsets route))
       :page-indices (get values (:page-indices route))
       :last-page-lengths (get values (:last-page-lengths route))
       :start-positions (get values (:start-positions route))})))

(defn- prepare
  [problem values extra-specs]
  (let [{:keys [query] :as problem} (attention/validate! problem)
        values (checked-values problem values extra-specs)]
    (attention/validate-query-values!
     problem (get values (:row-offsets query)) (get values (:positions query)))
    (attention/validate-routing! problem (route-values problem values))
    [problem values]))

(defn- query-batches
  [problem values]
  (let [{:keys [query batch-size]} problem
        offsets (get values (:row-offsets query))
        result (int-array (:total-tokens query))]
    (dotimes [batch batch-size]
      (doseq [token (range (nth offsets batch) (nth offsets (inc batch)))]
        (aset result token batch)))
    result))

(defn- route-row
  [problem values batch]
  (let [{:keys [route page-size]} problem]
    (if (attention/dense-paged-route? route)
      (let [lengths (get values (:lengths route))
            pages (get values (:page-table route))
            pages-per-sequence (:pages-per-sequence route)]
        {:length (nth lengths batch)
         :start-position (nth (get values (:start-positions route)) batch)
         :physical-page
         (fn [logical-page]
           (nth pages (+ (* batch pages-per-sequence) logical-page)))})
      (let [offsets (get values (:page-offsets route))
            indices (get values (:page-indices route))
            lasts (get values (:last-page-lengths route))
            page-begin (nth offsets batch)
            page-end (nth offsets (inc batch))
            page-count (- page-end page-begin)]
        {:length (if (zero? page-count)
                   0
                   (+ (* (dec page-count) page-size) (nth lasts batch)))
         :start-position (nth (get values (:start-positions route)) batch)
         :physical-page (fn [logical-page] (nth indices (+ page-begin logical-page)))}))))

(defn- cache-index
  [{:keys [physical-pages page-size kv-heads]} layout dim kv-head page token d]
  (case layout
    :kv-head-major
    (+ (* (+ (* (+ (* kv-head physical-pages) page) page-size) token) dim) d)

    :page-major
    (+ (* (+ (* (+ (* page page-size) token) kv-heads) kv-head) dim) d)))

(defn- visible?
  [{:keys [causal? window-left window-right]} query-position kv-position]
  (and (or (not causal?) (<= kv-position query-position))
       (or (nil? window-left) (>= kv-position (- query-position window-left)))
       (or (nil? window-right) (<= kv-position (+ query-position window-right)))))

(defn- attention-row
  [problem values query-token query-head batch]
  (let [{:keys [query k-pages q-heads kv-heads qk-head-dim page-size scale k-layout
                visibility]} problem
        q (get values (:values query))
        k (get values k-pages)
        query-position (nth (get values (:positions query)) query-token)
        kv-head (quot query-head (quot q-heads kv-heads))
        q-base (* (+ (* query-token q-heads) query-head) qk-head-dim)
        {:keys [length start-position physical-page]} (route-row problem values batch)
        tokens
        (into []
              (keep (fn [token]
                      (let [kv-position (+ start-position token)]
                        (when (visible? visibility query-position kv-position)
                          (let [page (physical-page (quot token page-size))
                                page-token (rem token page-size)
                                k-base (cache-index problem k-layout qk-head-dim
                                                    kv-head page page-token 0)
                                logit
                                (* scale
                                   (reduce + 0.0
                                           (map (fn [d]
                                                  (* (double (nth q (+ q-base d)))
                                                     (double (nth k (+ k-base d)))))
                                                (range qk-head-dim))))]
                            {:token token :page page :page-token page-token :logit logit})))))
              (range length))]
    (if (empty? tokens)
      {:kv-head kv-head :q-base q-base :tokens [] :lse Double/NEGATIVE_INFINITY}
      (let [maximum (reduce max (map :logit tokens))
            denominator (reduce + 0.0 (map #(Math/exp (- (double (:logit %)) maximum)) tokens))
            lse (+ maximum (Math/log denominator))]
        {:kv-head kv-head :q-base q-base :tokens tokens :lse lse}))))

(defn reference-forward-with-state
  "Return `{:output double-array :lse double-array}` for host values keyed by compiler buffer ID."
  [problem raw-values]
  (let [[{:keys [query v-pages q-heads value-head-dim v-layout] :as problem} values]
        (prepare problem raw-values {})
        total-tokens (:total-tokens query)
        output (double-array (* total-tokens q-heads value-head-dim))
        lse (double-array (* total-tokens q-heads))
        batches (query-batches problem values)
        v (get values v-pages)]
    (dotimes [query-token total-tokens]
      (dotimes [query-head q-heads]
        (let [batch (aget batches query-token)
              {:keys [kv-head tokens] :as row}
              (attention-row problem values query-token query-head batch)
              row-lse (:lse row)
              out-base (* (+ (* query-token q-heads) query-head) value-head-dim)]
          (aset lse (+ (* query-token q-heads) query-head) row-lse)
          (doseq [{:keys [page page-token logit]} tokens
                  :let [weight (Math/exp (- (double logit) row-lse))
                        v-base (cache-index problem v-layout value-head-dim
                                            kv-head page page-token 0)]
                  d (range value-head-dim)]
            (aset output (+ out-base d)
                  (+ (aget output (+ out-base d))
                     (* weight (double (nth v (+ v-base d))))))))))
    {:output output :lse lse}))

(defn reference-forward
  "Return the schedule-free double-precision attention output."
  [problem values]
  (:output (reference-forward-with-state problem values)))

(defn reference-vjp
  "Evaluate the semantic attention VJP. Returned keys are the requested active roles.

   K/V results have physical cache shapes and use addition at every routed write. Consequently,
   two logical rows that reference one physical page produce the required shared-page sum."
  [vjp raw-values]
  (let [{:keys [primal output-cotangent active-values softmax-state-mode saved-lse]
         :as vjp} (attention-ad/validate! vjp)
        extra-specs (select-keys (attention-ad/buffer-specs vjp)
                                 (cond-> [output-cotangent] saved-lse (conj saved-lse)))
        [{:keys [query k-pages v-pages q-heads qk-head-dim value-head-dim
                 scale k-layout v-layout]
          :as problem} values]
        (prepare primal raw-values extra-specs)
        total-tokens (:total-tokens query)
        q (get values (:values query))
        k (get values k-pages)
        v (get values v-pages)
        d-output (get values output-cotangent)
        saved-lse-values (when saved-lse (get values saved-lse))
        batches (query-batches problem values)
        d-query (when (contains? active-values :query)
                  (double-array (* total-tokens q-heads qk-head-dim)))
        d-key (when (contains? active-values :key)
                (double-array (get-in (attention/buffer-specs problem) [k-pages :elements])))
        d-value (when (contains? active-values :value)
                  (double-array (get-in (attention/buffer-specs problem) [v-pages :elements])))]
    (dotimes [query-token total-tokens]
      (dotimes [query-head q-heads]
        (let [batch (aget batches query-token)
              {:keys [kv-head q-base tokens]
               recomputed-lse :lse}
              (attention-row problem values query-token query-head batch)
              lse-index (+ (* query-token q-heads) query-head)
              lse (if (= :saved-lse softmax-state-mode)
                    (double (nth saved-lse-values lse-index))
                    recomputed-lse)
              out-base (* (+ (* query-token q-heads) query-head) value-head-dim)
              weighted-output-adjoint
              (reduce
               + 0.0
               (map (fn [{:keys [page page-token logit]}]
                      (let [weight (Math/exp (- (double logit) lse))
                            v-base (cache-index problem v-layout value-head-dim
                                                kv-head page page-token 0)
                            d-weight
                            (reduce + 0.0
                                    (map (fn [d]
                                           (* (double (nth d-output (+ out-base d)))
                                              (double (nth v (+ v-base d)))))
                                         (range value-head-dim)))]
                        (* weight d-weight)))
                    tokens))]
          (doseq [{:keys [page page-token logit]} tokens
                  :let [weight (Math/exp (- (double logit) lse))
                        k-base (cache-index problem k-layout qk-head-dim
                                            kv-head page page-token 0)
                        v-base (cache-index problem v-layout value-head-dim
                                            kv-head page page-token 0)
                        d-weight
                        (reduce + 0.0
                                (map (fn [d]
                                       (* (double (nth d-output (+ out-base d)))
                                          (double (nth v (+ v-base d)))))
                                     (range value-head-dim)))
                        d-logit (* weight (- d-weight weighted-output-adjoint))]]
            (when d-query
              (dotimes [d qk-head-dim]
                (aset d-query (+ q-base d)
                      (+ (aget d-query (+ q-base d))
                         (* scale d-logit (double (nth k (+ k-base d))))))))
            (when d-key
              (dotimes [d qk-head-dim]
                (aset d-key (+ k-base d)
                      (+ (aget d-key (+ k-base d))
                         (* scale d-logit (double (nth q (+ q-base d))))))))
            (when d-value
              (dotimes [d value-head-dim]
                (aset d-value (+ v-base d)
                      (+ (aget d-value (+ v-base d))
                         (* weight (double (nth d-output (+ out-base d))))))))))))
    (cond-> {}
      d-query (assoc :query d-query)
      d-key (assoc :key d-key)
      d-value (assoc :value d-value))))
