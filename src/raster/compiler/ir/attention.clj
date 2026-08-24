(ns raster.compiler.ir.attention
  "Backend-neutral semantic attention and routed KV storage contracts.

   Attention is expressed in logical token coordinates. Packed query rows, query positions and
   visibility are independent of the physical KV route. Dense and CSR page routes therefore share
   semantics without being forced through one native ABI. Cache allocation, relocation, sharing,
   native handles and queue ownership remain outside this IR."
  (:require [raster.compiler.core.dtype :as dtype]))

(defrecord PackedQueryBatch [values row-offsets positions total-tokens])
(defrecord AttentionVisibility [causal? window-left window-right])
(defrecord DensePagedRoute [page-table lengths start-positions pages-per-sequence])
(defrecord CSRPagedRoute
           [page-offsets page-indices last-page-lengths start-positions page-index-capacity])
(defrecord AttentionProblem
           [id query k-pages v-pages route output
            batch-size q-heads kv-heads qk-head-dim value-head-dim
            page-size physical-pages
            q-dtype k-dtype v-dtype output-dtype accumulator-dtype
            scale k-format v-format k-layout v-layout visibility])

(def ^:private cache-layouts #{:kv-head-major :page-major})

(defn packed-query-batch?
  [x]
  (instance? PackedQueryBatch x))

(defn visibility?
  [x]
  (instance? AttentionVisibility x))

(defn dense-paged-route?
  [x]
  (instance? DensePagedRoute x))

(defn csr-paged-route?
  [x]
  (instance? CSRPagedRoute x))

(defn paged-route?
  [x]
  (or (dense-paged-route? x) (csr-paged-route? x)))

(defn attention-problem?
  [x]
  (instance? AttentionProblem x))

(defn route-kind
  [route]
  (cond
    (dense-paged-route? route) :dense-paged
    (csr-paged-route? route) :csr-paged
    :else nil))

(defn- positive-integer!
  [owner field value]
  (when-not (and (integer? value) (pos? value))
    (throw (ex-info (str owner " requires a positive static " (name field))
                    {:reason :attention-nonpositive-or-dynamic-extent
                     :owner owner :field field :value value})))
  value)

(defn- nonnegative-integer!
  [owner field value]
  (when-not (and (integer? value) (not (neg? value)))
    (throw (ex-info (str owner " requires a nonnegative static " (name field))
                    {:reason :attention-negative-or-dynamic-extent
                     :owner owner :field field :value value})))
  value)

(defn- finite-positive!
  [field value]
  (when-not (and (number? value)
                 (Double/isFinite (double value))
                 (pos? (double value)))
    (throw (ex-info "attention scale must be finite and positive"
                    {:reason :attention-invalid-scale :field field :value value})))
  value)

(defn- checked-product
  [field factors]
  (try
    (reduce (fn [acc factor] (Math/multiplyExact (long acc) (long factor))) 1 factors)
    (catch ArithmeticException e
      (throw (ex-info "attention storage extent exceeds signed 64-bit capacity"
                      {:reason :attention-extent-overflow
                       :field field :factors (vec factors)} e)))))

(defn packed-query-batch
  "Construct packed query metadata. `row-offsets` has B+1 int32 values at execution; `positions`
   has one logical int32 position per packed query token. Individual rows may be empty."
  [{:keys [values row-offsets positions total-tokens]}]
  (when (some nil? [values row-offsets positions])
    (throw (ex-info "packed query batch requires values, row offsets and positions"
                    {:reason :attention-query-missing-buffer})))
  (positive-integer! "packed query batch" :total-tokens total-tokens)
  (->PackedQueryBatch values row-offsets positions total-tokens))

(defn visibility
  "Construct logical visibility. Window values are maximum relative distances and are inclusive:
   `window-left=2` admits positions q-2 through q. nil means unbounded."
  ([] (visibility {}))
  ([{:keys [causal? window-left window-right]
     :or {causal? false window-left nil window-right nil}}]
   (when-not (instance? Boolean causal?)
     (throw (ex-info "attention causal flag must be boolean"
                     {:reason :attention-invalid-causal :value causal?})))
   (doseq [[field value] [[:window-left window-left] [:window-right window-right]]]
     (when (some? value)
       (nonnegative-integer! "attention visibility" field value)))
   (->AttentionVisibility causal? window-left window-right)))

(defn dense-paged-route
  "Construct a fixed-width dense page route. Row b lists physical pages in increasing logical KV
   order; `lengths[b]` clips its final page and `start-positions[b]` is the logical position of
   routed token zero."
  [{:keys [page-table lengths start-positions pages-per-sequence]}]
  (when (some nil? [page-table lengths start-positions])
    (throw (ex-info "dense page route requires table, lengths and start positions"
                    {:reason :attention-dense-route-missing-buffer})))
  (positive-integer! "dense page route" :pages-per-sequence pages-per-sequence)
  (->DensePagedRoute page-table lengths start-positions pages-per-sequence))

(defn csr-paged-route
  "Construct a compact heterogeneous page route. `page-offsets [B+1]` partitions page indices by
   sequence. A nonempty row has `last-page-lengths[b]` in 1..page-size; an empty row has zero. The
   capacity is the resident page-index buffer extent and may exceed the runtime nnz page count."
  [{:keys [page-offsets page-indices last-page-lengths start-positions page-index-capacity]}]
  (when (some nil? [page-offsets page-indices last-page-lengths start-positions])
    (throw (ex-info "CSR page route requires offsets, indices, final lengths and start positions"
                    {:reason :attention-csr-route-missing-buffer})))
  (positive-integer! "CSR page route" :page-index-capacity page-index-capacity)
  (->CSRPagedRoute page-offsets page-indices last-page-lengths start-positions
                   page-index-capacity))

(defn- validate-format!
  [field storage-dtype format]
  (when-not (map? format)
    (throw (ex-info "attention storage format must be a map"
                    {:reason :attention-format-not-map :field field :format format})))
  (when-not (= (dtype/canon storage-dtype) (dtype/canon (:dtype format)))
    (throw (ex-info "attention format dtype differs from storage dtype"
                    {:reason :attention-format-dtype-mismatch
                     :field field :storage-dtype storage-dtype :format format})))
  (when-not (keyword? (:quantization format))
    (throw (ex-info "attention storage format requires a quantization mode"
                    {:reason :attention-format-missing-quantization
                     :field field :format format})))
  (when (and (= :none (:quantization format))
             (not (dtype/fp-dtype? storage-dtype)))
    (throw (ex-info "plain attention storage must be floating point"
                    {:reason :attention-plain-storage-nonfloating
                     :field field :storage-dtype storage-dtype :format format})))
  format)

(defn- route-buffer-ids
  [route]
  (cond
    (dense-paged-route? route)
    [(:page-table route) (:lengths route) (:start-positions route)]

    (csr-paged-route? route)
    [(:page-offsets route) (:page-indices route) (:last-page-lengths route)
     (:start-positions route)]

    :else
    (throw (ex-info "attention requires a supported KV route"
                    {:reason :attention-unsupported-route :route route :actual (type route)}))))

(defn validate!
  "Validate an AttentionProblem's static semantics and storage extents. Runtime route/query values
   are validated separately before upload and guarded again by the reference kernel."
  [problem]
  (when-not (attention-problem? problem)
    (throw (ex-info "attention operation must be an AttentionProblem"
                    {:reason :attention-invalid-problem :actual (type problem)})))
  (let [{:keys [id query k-pages v-pages route output batch-size q-heads kv-heads
                qk-head-dim value-head-dim page-size physical-pages q-dtype k-dtype v-dtype
                output-dtype accumulator-dtype scale k-format v-format k-layout v-layout
                visibility]} problem]
    (when (nil? id)
      (throw (ex-info "attention requires a stable identity" {:reason :attention-missing-id})))
    (when-not (packed-query-batch? query)
      (throw (ex-info "attention requires a PackedQueryBatch"
                      {:reason :attention-invalid-query-batch :query query})))
    (when-not (visibility? visibility)
      (throw (ex-info "attention requires an AttentionVisibility"
                      {:reason :attention-invalid-visibility :visibility visibility})))
    (when-not (paged-route? route)
      (throw (ex-info "attention requires a dense or CSR paged route"
                      {:reason :attention-unsupported-route :route route})))
    (doseq [[field value] [[:batch-size batch-size] [:q-heads q-heads]
                           [:kv-heads kv-heads] [:qk-head-dim qk-head-dim]
                           [:value-head-dim value-head-dim] [:page-size page-size]
                           [:physical-pages physical-pages]]]
      (positive-integer! "attention" field value))
    (positive-integer! "packed query batch" :total-tokens (:total-tokens query))
    (when-not (zero? (mod q-heads kv-heads))
      (throw (ex-info "attention requires q-heads divisible by kv-heads"
                      {:reason :attention-invalid-gqa-ratio
                       :q-heads q-heads :kv-heads kv-heads})))
    (doseq [[field value] [[:q-dtype q-dtype] [:k-dtype k-dtype] [:v-dtype v-dtype]
                           [:output-dtype output-dtype]
                           [:accumulator-dtype accumulator-dtype]]]
      (when-not (dtype/known? value)
        (throw (ex-info "attention has an unknown dtype"
                        {:reason :attention-unknown-dtype :field field :dtype value}))))
    (when-not (every? dtype/fp-dtype? [q-dtype output-dtype accumulator-dtype])
      (throw (ex-info "attention query, output and accumulator dtypes must be floating point"
                      {:reason :attention-nonfloating-dtype
                       :dtypes [q-dtype output-dtype accumulator-dtype]})))
    (finite-positive! :scale scale)
    (validate-format! :k k-dtype k-format)
    (validate-format! :v v-dtype v-format)
    (doseq [[field layout] [[:k-layout k-layout] [:v-layout v-layout]]]
      (when-not (contains? cache-layouts layout)
        (throw (ex-info "attention cache layout is unsupported"
                        {:reason :attention-unsupported-cache-layout
                         :field field :layout layout :supported cache-layouts}))))
    (let [buffers (vec (concat [(:values query) (:row-offsets query) (:positions query)
                                k-pages v-pages]
                               (route-buffer-ids route)
                               [output]))]
      (when (some nil? buffers)
        (throw (ex-info "attention requires every logical buffer identity"
                        {:reason :attention-missing-buffer :buffers buffers})))
      (when-not (= (count buffers) (count (distinct buffers)))
        (throw (ex-info "attention logical buffer identities must be distinct"
                        {:reason :attention-duplicate-buffer-identity :buffers buffers}))))
    (checked-product :q [(:total-tokens query) q-heads qk-head-dim])
    (checked-product :k-cache [kv-heads physical-pages page-size qk-head-dim])
    (checked-product :v-cache [kv-heads physical-pages page-size value-head-dim])
    (checked-product :output [(:total-tokens query) q-heads value-head-dim])
    (when (> (:total-tokens query) Integer/MAX_VALUE)
      (throw (ex-info "packed query offsets use int32 and cannot address this many tokens"
                      {:reason :attention-query-token-capacity-overflow
                       :total-tokens (:total-tokens query)})))
    (let [page-capacity (if (dense-paged-route? route)
                          (:pages-per-sequence route)
                          (:page-index-capacity route))
          token-capacity (checked-product :logical-kv-capacity [page-capacity page-size])]
      (when (> token-capacity Integer/MAX_VALUE)
        (throw (ex-info "page route lengths use int32 and exceed their token capacity"
                        {:reason :attention-route-token-capacity-overflow
                         :page-capacity page-capacity :page-size page-size}))))
    (if (dense-paged-route? route)
      (checked-product :page-table [batch-size (:pages-per-sequence route)])
      (positive-integer! "CSR page route" :page-index-capacity
                         (:page-index-capacity route)))
    problem))

(defn make
  "Construct a checked attention problem. Plain FP16 K/V defaults keep the first executable leaf
   concise; quantized formats may already be represented and decline until their ordered ABI lands."
  [{:keys [id query k-pages v-pages route output batch-size q-heads kv-heads
           qk-head-dim value-head-dim page-size physical-pages
           q-dtype k-dtype v-dtype output-dtype accumulator-dtype scale
           k-format v-format k-layout v-layout visibility]
    :or {q-dtype :half k-dtype :half v-dtype :half output-dtype :half
         accumulator-dtype :float k-format nil v-format nil
         k-layout :kv-head-major v-layout :kv-head-major visibility nil}}]
  (let [q-dtype (dtype/canon q-dtype)
        k-dtype (dtype/canon k-dtype)
        v-dtype (dtype/canon v-dtype)
        output-dtype (dtype/canon output-dtype)
        accumulator-dtype (dtype/canon accumulator-dtype)
        k-format (or k-format {:dtype k-dtype :quantization :none})
        v-format (or v-format {:dtype v-dtype :quantization :none})
        visibility (or visibility (raster.compiler.ir.attention/visibility))
        problem (->AttentionProblem
                 id query k-pages v-pages route output batch-size q-heads kv-heads
                 qk-head-dim value-head-dim page-size physical-pages
                 q-dtype k-dtype v-dtype output-dtype accumulator-dtype scale
                 k-format v-format k-layout v-layout visibility)]
    (assoc (validate! problem) :scale (double scale))))

(defn layouts
  "Named logical layouts. These are semantic/storage axes, not thread/register layouts."
  [problem]
  (let [{:keys [query route batch-size q-heads kv-heads qk-head-dim value-head-dim
                page-size physical-pages k-layout v-layout]} (validate! problem)
        cache-shape (fn [layout dim]
                      (case layout
                        :kv-head-major [kv-heads physical-pages page-size dim]
                        :page-major [physical-pages page-size kv-heads dim]))
        common {:q [(:total-tokens query) q-heads qk-head-dim]
                :q-row-offsets [(inc batch-size)]
                :q-positions [(:total-tokens query)]
                :k-pages (cache-shape k-layout qk-head-dim)
                :v-pages (cache-shape v-layout value-head-dim)
                :kv-start-positions [batch-size]
                :output [(:total-tokens query) q-heads value-head-dim]}]
    (merge common
           (if (dense-paged-route? route)
             {:page-table [batch-size (:pages-per-sequence route)]
              :kv-lengths [batch-size]}
             {:page-offsets [(inc batch-size)]
              :page-indices [(:page-index-capacity route)]
              :last-page-lengths [batch-size]}))))

(defn buffer-specs
  "Graph-buffer metadata keyed by compiler identities, including route-variant metadata."
  [problem]
  (let [{:keys [query k-pages v-pages route output q-dtype k-dtype v-dtype output-dtype]
         :as problem} (validate! problem)
        shapes (layouts problem)
        spec (fn [role dtype shape-key]
               {:role role :dtype dtype :shape (get shapes shape-key)
                :elements (checked-product shape-key (get shapes shape-key))})
        common {(:values query) (spec :input q-dtype :q)
                (:row-offsets query) (spec :input :int :q-row-offsets)
                (:positions query) (spec :input :int :q-positions)
                k-pages (spec :input k-dtype :k-pages)
                v-pages (spec :input v-dtype :v-pages)
                (:start-positions route) (spec :input :int :kv-start-positions)
                output (spec :output output-dtype :output)}]
    (merge common
           (if (dense-paged-route? route)
             {(:page-table route) (spec :input :int :page-table)
              (:lengths route) (spec :input :int :kv-lengths)}
             {(:page-offsets route) (spec :input :int :page-offsets)
              (:page-indices route) (spec :input :int :page-indices)
              (:last-page-lengths route) (spec :input :int :last-page-lengths)}))))

(defn- int32-nonnegative?
  [x]
  (and (integer? x) (<= 0 x Integer/MAX_VALUE)))

(defn validate-query-values!
  "Validate host-visible packed-query metadata before upload. Row offsets must cover every packed
   query exactly; logical positions are nonnegative int32 and strictly increase within each row."
  [problem row-offset-values position-values]
  (let [{:keys [query batch-size]} (validate! problem)
        offsets (vec row-offset-values)
        positions (vec position-values)
        total (:total-tokens query)]
    (when-not (= (inc batch-size) (count offsets))
      (throw (ex-info "packed query row offsets have the wrong element count"
                      {:reason :attention-query-offset-shape
                       :expected (inc batch-size) :actual (count offsets)})))
    (when-not (= total (count positions))
      (throw (ex-info "packed query positions have the wrong element count"
                      {:reason :attention-query-position-shape
                       :expected total :actual (count positions)})))
    (when-not (and (= 0 (first offsets)) (= total (peek offsets))
                   (every? int32-nonnegative? offsets)
                   (every? true? (map <= offsets (rest offsets))))
      (throw (ex-info "packed query row offsets must be monotone from zero to total tokens"
                      {:reason :attention-invalid-query-offsets
                       :offsets offsets :total-tokens total})))
    (when-not (every? int32-nonnegative? positions)
      (throw (ex-info "packed query positions must be nonnegative int32"
                      {:reason :attention-invalid-query-position :positions positions})))
    (doseq [batch (range batch-size)
            :let [start (nth offsets batch) end (nth offsets (inc batch))
                  row (subvec positions start end)]]
      (when-not (every? true? (map < row (rest row)))
        (throw (ex-info "query positions must strictly increase within each sequence"
                        {:reason :attention-nonmonotone-query-positions
                         :batch batch :positions row}))))
    problem))

(defn- validate-start-positions!
  [batch-size starts lengths]
  (when-not (= batch-size (count starts))
    (throw (ex-info "KV start positions have the wrong element count"
                    {:reason :attention-kv-start-position-shape
                     :expected batch-size :actual (count starts)})))
  (doseq [[batch start length] (map vector (range) starts lengths)]
    (when-not (and (int32-nonnegative? start)
                   (<= (+ (long start) (long length)) (inc (long Integer/MAX_VALUE))))
      (throw (ex-info "KV logical position range exceeds nonnegative int32 coordinates"
                      {:reason :attention-invalid-kv-position-range
                       :batch batch :start start :length length}))))
  starts)

(defn validate-routing!
  "Validate host-visible physical route metadata before upload. Dense padding entries are ignored;
   only pages selected by each length must be valid. CSR offsets may use any prefix of their
   resident page-index capacity."
  [problem values]
  (let [{:keys [route batch-size page-size physical-pages]} (validate! problem)]
    (if (dense-paged-route? route)
      (let [{:keys [page-table lengths start-positions]} values
            table (vec page-table)
            lengths (vec lengths)
            starts (vec start-positions)
            pps (:pages-per-sequence route)
            table-elements (checked-product :page-table [batch-size pps])
            capacity (checked-product :logical-kv-capacity [pps page-size])]
        (when-not (= table-elements (count table))
          (throw (ex-info "dense page table has the wrong element count"
                          {:reason :attention-page-table-shape
                           :expected table-elements :actual (count table)})))
        (when-not (= batch-size (count lengths))
          (throw (ex-info "dense KV lengths have the wrong element count"
                          {:reason :attention-kv-length-shape
                           :expected batch-size :actual (count lengths)})))
        (doseq [[batch length] (map-indexed vector lengths)]
          (when-not (and (integer? length) (<= 0 length capacity))
            (throw (ex-info "dense KV length exceeds its logical page capacity"
                            {:reason :attention-invalid-kv-length
                             :batch batch :length length :capacity capacity})))
          (let [used-pages (quot (+ length (dec page-size)) page-size)]
            (doseq [logical-page (range used-pages)
                    :let [index (+ (* batch pps) logical-page)
                          page (nth table index)]]
              (when-not (and (integer? page) (<= 0 page) (< page physical-pages))
                (throw (ex-info "dense route selects an invalid physical page"
                                {:reason :attention-invalid-physical-page
                                 :batch batch :logical-page logical-page :page page
                                 :physical-pages physical-pages}))))))
        (validate-start-positions! batch-size starts lengths)
        problem)
      (let [{:keys [page-offsets page-indices last-page-lengths start-positions]} values
            offsets (vec page-offsets)
            indices (vec page-indices)
            lasts (vec last-page-lengths)
            starts (vec start-positions)
            capacity (:page-index-capacity route)]
        (when-not (= (inc batch-size) (count offsets))
          (throw (ex-info "CSR page offsets have the wrong element count"
                          {:reason :attention-page-offset-shape
                           :expected (inc batch-size) :actual (count offsets)})))
        (when-not (= capacity (count indices))
          (throw (ex-info "CSR page-index buffer has the wrong element count"
                          {:reason :attention-page-index-shape
                           :expected capacity :actual (count indices)})))
        (when-not (= batch-size (count lasts))
          (throw (ex-info "CSR last-page lengths have the wrong element count"
                          {:reason :attention-last-page-length-shape
                           :expected batch-size :actual (count lasts)})))
        (when-not (and (= 0 (first offsets))
                       (every? int32-nonnegative? offsets)
                       (every? true? (map <= offsets (rest offsets)))
                       (<= (peek offsets) capacity))
          (throw (ex-info "CSR page offsets must be monotone and bounded by index capacity"
                          {:reason :attention-invalid-page-offsets
                           :offsets offsets :capacity capacity})))
        (doseq [[index page] (map-indexed vector (subvec indices 0 (peek offsets)))]
          (when-not (and (integer? page) (<= 0 page) (< page physical-pages))
            (throw (ex-info "CSR route selects an invalid physical page"
                            {:reason :attention-invalid-physical-page
                             :index index :page page :physical-pages physical-pages}))))
        (let [lengths
              (mapv (fn [batch]
                      (let [pages (- (nth offsets (inc batch)) (nth offsets batch))
                            last (nth lasts batch)]
                        (when-not (if (zero? pages)
                                    (= 0 last)
                                    (and (integer? last) (<= 1 last page-size)))
                          (throw (ex-info "CSR final-page length disagrees with its page row"
                                          {:reason :attention-invalid-last-page-length
                                           :batch batch :pages pages :last-page-length last
                                           :page-size page-size})))
                        (if (zero? pages) 0 (+ (* (dec pages) page-size) last))))
                    (range batch-size))]
          (validate-start-positions! batch-size starts lengths))
        problem))))
