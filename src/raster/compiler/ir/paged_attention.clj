(ns raster.compiler.ir.paged-attention
  "Backend-neutral semantic descriptor for decode-time paged grouped-query attention.

   The page table is the routing contract: row `b` lists physical page ids in the logical
   attention order for sequence `b`. This represents ordinary contiguous caches, arbitrary page
   placement, shared prefixes, and pinned-prefix + ring layouts without putting an allocator or a
   native handle in compiler IR. `lengths[b]` clips the final logical page.

   The default storage layout follows the portable JAX/Pallas convention:
     q          [batch, q-head, head-dim]
     k/v-pages  [kv-head, physical-page, page-token, head-dim]
     page-table [batch, pages-per-sequence]
     lengths    [batch]
     output     [batch, q-head, head-dim]

   `:page-major` cache storage instead uses [physical-page, page-token, kv-head, head-dim].
   With page-size 1 this is a zero-copy view of a conventional [token, kv-head, head-dim] cache.

   This first descriptor is deliberately decode-only: one query per sequence, all routed tokens
   are visible, and RoPE (if used) is already baked into q/k at their true logical positions."
  (:require [raster.compiler.core.dtype :as dtype]))

(defrecord PagedAttention
           [id q k-pages v-pages page-table lengths output
            batch-size q-heads kv-heads head-dim
            page-size physical-pages pages-per-sequence
            q-dtype cache-dtype output-dtype accumulator-dtype
            scale cache-format cache-layout])

(def ^:private cache-layouts #{:kv-head-major :page-major})

(defn paged-attention?
  [x]
  (and x (= "raster.compiler.ir.paged_attention.PagedAttention" (.getName (class x)))))

(defn- positive-integer!
  [owner field value]
  (when-not (and (integer? value) (pos? value))
    (throw (ex-info (str owner " requires a positive static " (name field))
                    {:reason :paged-attention-nonpositive-or-dynamic-extent
                     :field field :value value})))
  value)

(defn- finite-positive!
  [field value]
  (when-not (and (number? value)
                 (Double/isFinite (double value))
                 (pos? (double value)))
    (throw (ex-info "paged attention scale must be finite and positive"
                    {:reason :paged-attention-invalid-scale :field field :value value})))
  value)

(defn- checked-product
  [field factors]
  (try
    (reduce (fn [acc factor] (Math/multiplyExact (long acc) (long factor))) 1 factors)
    (catch ArithmeticException e
      (throw (ex-info "paged attention storage extent exceeds signed 64-bit capacity"
                      {:reason :paged-attention-extent-overflow
                       :field field :factors (vec factors)} e)))))

(defn validate!
  "Validate a PagedAttention descriptor. Shapes are static in the reference vertical so GQA and
   every storage extent are proved before emission; dynamic-shape routing needs a later checked
   scalar-constraint IR rather than unchecked division inside a kernel."
  [operation]
  (when-not (paged-attention? operation)
    (throw (ex-info "paged attention operation must be a PagedAttention value"
                    {:operation operation :actual (type operation)})))
  (let [{:keys [id q k-pages v-pages page-table lengths output
                batch-size q-heads kv-heads head-dim page-size physical-pages
                pages-per-sequence q-dtype cache-dtype output-dtype accumulator-dtype
                scale cache-format cache-layout]} operation
        buffers [q k-pages v-pages page-table lengths output]]
    (when (nil? id)
      (throw (ex-info "paged attention requires a stable identity" {})))
    (when (some nil? buffers)
      (throw (ex-info "paged attention requires every logical buffer identity"
                      {:buffers buffers})))
    (when-not (= (count buffers) (count (distinct buffers)))
      (throw (ex-info "paged attention logical buffer identities must be distinct"
                      {:reason :paged-attention-duplicate-buffer-identity
                       :buffers buffers})))
    (doseq [[field value] [[:batch-size batch-size] [:q-heads q-heads]
                           [:kv-heads kv-heads] [:head-dim head-dim]
                           [:page-size page-size] [:physical-pages physical-pages]
                           [:pages-per-sequence pages-per-sequence]]]
      (positive-integer! "paged attention" field value))
    (when-not (zero? (mod q-heads kv-heads))
      (throw (ex-info "paged attention requires q-heads divisible by kv-heads"
                      {:reason :paged-attention-invalid-gqa-ratio
                       :q-heads q-heads :kv-heads kv-heads})))
    (doseq [[field value] [[:q-dtype q-dtype] [:cache-dtype cache-dtype]
                           [:output-dtype output-dtype]
                           [:accumulator-dtype accumulator-dtype]]]
      (when-not (dtype/known? value)
        (throw (ex-info "paged attention has an unknown dtype"
                        {:reason :paged-attention-unknown-dtype
                         :field field :dtype value}))))
    (when-not (every? dtype/fp-dtype? [q-dtype output-dtype accumulator-dtype])
      (throw (ex-info "paged attention query, output and accumulator dtypes must be floating point"
                      {:reason :paged-attention-nonfloating-dtype
                       :dtypes [q-dtype output-dtype accumulator-dtype]})))
    (finite-positive! :scale scale)
    (when-not (map? cache-format)
      (throw (ex-info "paged attention cache-format must be a map"
                      {:cache-format cache-format})))
    (when-not (= (dtype/canon cache-dtype) (dtype/canon (:dtype cache-format)))
      (throw (ex-info "paged attention cache format dtype differs from cache storage"
                      {:reason :paged-attention-cache-format-dtype-mismatch
                       :cache-dtype cache-dtype :cache-format cache-format})))
    (when-not (keyword? (:quantization cache-format))
      (throw (ex-info "paged attention cache format requires a quantization mode"
                      {:reason :paged-attention-cache-format-missing-quantization
                       :cache-format cache-format})))
    (when (and (= :none (:quantization cache-format))
               (not (dtype/fp-dtype? cache-dtype)))
      (throw (ex-info "plain paged attention cache storage must be floating point"
                      {:reason :paged-attention-plain-cache-nonfloating
                       :cache-dtype cache-dtype :cache-format cache-format})))
    (when-not (contains? cache-layouts cache-layout)
      (throw (ex-info "paged attention cache layout is unsupported"
                      {:reason :paged-attention-unsupported-cache-layout
                       :cache-layout cache-layout :supported cache-layouts})))
    ;; Prove all allocation sizes fit the extent representation now, not during allocation.
    (checked-product :q [batch-size q-heads head-dim])
    (checked-product :cache [kv-heads physical-pages page-size head-dim])
    (checked-product :page-table [batch-size pages-per-sequence])
    operation))

(defn make
  "Construct a checked decode paged-attention descriptor.

   `cache-format` is representation metadata and defaults to plain storage. Quantized formats
   will extend it with explicit scale buffers/grouping; the FP16 reference route refuses them
   until that ABI exists."
  [{:keys [id q k-pages v-pages page-table lengths output
           batch-size q-heads kv-heads head-dim page-size physical-pages pages-per-sequence
           q-dtype cache-dtype output-dtype accumulator-dtype scale cache-format cache-layout]
    :or {q-dtype :half cache-dtype :half output-dtype :half accumulator-dtype :float
         cache-format nil cache-layout :kv-head-major}}]
  (let [q-dtype (dtype/canon q-dtype)
        cache-dtype (dtype/canon cache-dtype)
        output-dtype (dtype/canon output-dtype)
        accumulator-dtype (dtype/canon accumulator-dtype)
        cache-format (or cache-format {:dtype cache-dtype :quantization :none})
        operation
        (validate!
         (->PagedAttention id q k-pages v-pages page-table lengths output
                           batch-size q-heads kv-heads head-dim page-size physical-pages
                           pages-per-sequence q-dtype cache-dtype output-dtype accumulator-dtype
                           scale cache-format cache-layout))]
    (assoc operation :scale (double scale))))

(defn layouts
  "Named logical layouts. These are semantic axes, not thread/register layouts."
  [operation]
  (let [{:keys [batch-size q-heads kv-heads head-dim page-size physical-pages
                pages-per-sequence cache-layout]} (validate! operation)
        cache-shape (case cache-layout
                      :kv-head-major [kv-heads physical-pages page-size head-dim]
                      :page-major [physical-pages page-size kv-heads head-dim])]
    {:q [batch-size q-heads head-dim]
     :k-pages cache-shape
     :v-pages cache-shape
     :page-table [batch-size pages-per-sequence]
     :lengths [batch-size]
     :output [batch-size q-heads head-dim]}))

(defn buffer-specs
  "Graph-buffer metadata keyed by the operation's compiler identities."
  [operation]
  (let [{:keys [q k-pages v-pages page-table lengths output
                q-dtype cache-dtype output-dtype]} (validate! operation)
        shapes (layouts operation)
        elements #(checked-product % (get shapes %))]
    {q {:role :input :dtype q-dtype :shape (:q shapes) :elements (elements :q)}
     k-pages {:role :input :dtype cache-dtype :shape (:k-pages shapes)
              :elements (elements :k-pages)}
     v-pages {:role :input :dtype cache-dtype :shape (:v-pages shapes)
              :elements (elements :v-pages)}
     page-table {:role :input :dtype :int :shape (:page-table shapes)
                 :elements (elements :page-table)}
     lengths {:role :input :dtype :int :shape (:lengths shapes)
              :elements (elements :lengths)}
     output {:role :output :dtype output-dtype :shape (:output shapes)
             :elements (elements :output)}}))

(defn validate-routing!
  "Validate host-visible page-table/length contents before upload. Runtime-owned page allocators
   should call this at route construction; device kernels still guard corrupted resident tables
   with NaN output rather than performing an out-of-bounds read."
  [operation page-table-values length-values]
  (let [{:keys [batch-size physical-pages pages-per-sequence page-size]}
        (validate! operation)
        table (vec page-table-values)
        lengths (vec length-values)
        table-elements (checked-product :page-table [batch-size pages-per-sequence])
        max-tokens (checked-product :logical-capacity [pages-per-sequence page-size])]
    (when-not (= table-elements (count table))
      (throw (ex-info "paged attention page table has the wrong element count"
                      {:reason :paged-attention-page-table-shape
                       :expected table-elements :actual (count table)})))
    (when-not (= batch-size (count lengths))
      (throw (ex-info "paged attention lengths have the wrong element count"
                      {:reason :paged-attention-lengths-shape
                       :expected batch-size :actual (count lengths)})))
    (doseq [[index page] (map-indexed vector table)]
      (when-not (and (integer? page) (<= 0 page) (< page physical-pages))
        (throw (ex-info "paged attention page table contains an invalid physical page"
                        {:reason :paged-attention-invalid-physical-page
                         :index index :page page :physical-pages physical-pages}))))
    (doseq [[batch length] (map-indexed vector lengths)]
      (when-not (and (integer? length) (<= 0 length max-tokens))
        (throw (ex-info "paged attention sequence length exceeds its logical page capacity"
                        {:reason :paged-attention-invalid-length
                         :batch batch :length length :max-tokens max-tokens}))))
    operation))
