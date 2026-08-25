(ns raster.compiler.ir.paged-kv-append
  "Backend-neutral assignment of projected K/V rows into physical cache slots.

  The operation is deliberately narrower than scatter-add: every batch lane
  names one unique physical slot and overwrites its complete key and value rows.
  Page allocation and reservation remain runtime responsibilities; this IR owns
  only checked tensor geometry, dtype conversion semantics, and write effects."
  (:require [raster.compiler.core.dtype :as dtype]))

(defrecord PagedKVAppend
           [id key-rows value-rows slot-mapping key-pages value-pages
            batch-size key-elements-per-token value-elements-per-token
            page-size physical-pages key-input-dtype value-input-dtype
            key-storage-dtype value-storage-dtype rounding-mode])

(declare validate!)

(defn paged-kv-append?
  "Return true when `value` is a paged K/V append problem."
  [value]
  (and value
       (= "raster.compiler.ir.paged_kv_append.PagedKVAppend"
          (.getName (class value)))))

(defn- positive-integer!
  [field value]
  (when-not (and (integer? value) (pos? value))
    (throw (ex-info "Paged K/V append extents must be positive integers"
                    {:field field :value value})))
  (long value))

(defn physical-slots
  "Return the checked number of addressable token slots in `problem`."
  [problem]
  (let [{:keys [page-size physical-pages]} (validate! problem)]
    (try
      (Math/multiplyExact (long page-size) (long physical-pages))
      (catch ArithmeticException error
        (throw (ex-info "Paged K/V append slot extent exceeds signed 64-bit capacity"
                        {:page-size page-size :physical-pages physical-pages}
                        error))))))

(defn validate!
  "Validate a paged K/V append problem and return it unchanged."
  [problem]
  (when-not (paged-kv-append? problem)
    (throw (ex-info "Paged K/V append must be a PagedKVAppend value"
                    {:problem problem :actual (type problem)})))
  (let [{:keys [id key-rows value-rows slot-mapping key-pages value-pages
                batch-size key-elements-per-token value-elements-per-token
                page-size physical-pages key-input-dtype value-input-dtype
                key-storage-dtype value-storage-dtype rounding-mode]} problem
        buffers [key-rows value-rows slot-mapping key-pages value-pages]]
    (when (nil? id)
      (throw (ex-info "Paged K/V append requires an identity" {:problem problem})))
    (when (some nil? buffers)
      (throw (ex-info "Paged K/V append requires every buffer identity"
                      {:buffers buffers})))
    (when-not (= (count buffers) (count (set buffers)))
      (throw (ex-info "Paged K/V append buffer identities must be distinct"
                      {:buffers buffers})))
    (doseq [[field value] [[:batch-size batch-size]
                           [:key-elements-per-token key-elements-per-token]
                           [:value-elements-per-token value-elements-per-token]
                           [:page-size page-size]
                           [:physical-pages physical-pages]]]
      (positive-integer! field value))
    (doseq [[field value] [[:key-input-dtype key-input-dtype]
                           [:value-input-dtype value-input-dtype]
                           [:key-storage-dtype key-storage-dtype]
                           [:value-storage-dtype value-storage-dtype]]]
      (when-not (dtype/known? value)
        (throw (ex-info "Paged K/V append dtype is unknown"
                        {:field field :value value}))))
    (when-not (= :round-to-nearest-even rounding-mode)
      (throw (ex-info "Paged K/V append requires explicit round-to-nearest-even conversion"
                      {:rounding-mode rounding-mode})))
    problem))

(defn make
  "Construct a checked paged K/V append problem.

  Input rows default to FP32 and page storage defaults to FP16. Conversion uses
  round-to-nearest-even. All extents and buffer identities are static compiler
  values; only row contents and physical slot values vary between submissions."
  [{:keys [key-input-dtype value-input-dtype key-storage-dtype value-storage-dtype
           rounding-mode]
    :or {key-input-dtype :float value-input-dtype :float
         key-storage-dtype :half value-storage-dtype :half
         rounding-mode :round-to-nearest-even}
    :as opts}]
  (validate!
   (map->PagedKVAppend
    (assoc opts
           :key-input-dtype (dtype/canon key-input-dtype)
           :value-input-dtype (dtype/canon value-input-dtype)
           :key-storage-dtype (dtype/canon key-storage-dtype)
           :value-storage-dtype (dtype/canon value-storage-dtype)
           :rounding-mode rounding-mode))))

(defn ordered-input-buffer-ids
  "Return source and slot buffers in their semantic order."
  [problem]
  (let [{:keys [key-rows value-rows slot-mapping]} (validate! problem)]
    [key-rows value-rows slot-mapping]))

(defn ordered-output-buffer-ids
  "Return mutated page pools in key/value order."
  [problem]
  (let [{:keys [key-pages value-pages]} (validate! problem)]
    [key-pages value-pages]))

(defn buffer-specs
  "Return exact dtype, element count, and graph role for every problem buffer."
  [problem]
  (let [{:keys [key-rows value-rows slot-mapping key-pages value-pages batch-size
                key-elements-per-token value-elements-per-token key-input-dtype
                value-input-dtype key-storage-dtype value-storage-dtype]}
        (validate! problem)
        slots (physical-slots problem)]
    {key-rows {:dtype key-input-dtype
               :elements (* batch-size key-elements-per-token) :role :input}
     value-rows {:dtype value-input-dtype
                 :elements (* batch-size value-elements-per-token) :role :input}
     slot-mapping {:dtype :int :elements batch-size :role :input}
     key-pages {:dtype key-storage-dtype
                :elements (* slots key-elements-per-token) :role :inout}
     value-pages {:dtype value-storage-dtype
                  :elements (* slots value-elements-per-token) :role :inout}}))

(defn validate-slot-values!
  "Validate one host-visible physical slot vector and return it unchanged.

  Slots must be unique because assignment has no atomic conflict resolution.
  This check belongs before descriptor upload; the device kernel separately
  bounds-checks slots so corrupted descriptor memory cannot overwrite pages."
  [problem slots]
  (let [{:keys [batch-size]} (validate! problem)
        capacity (physical-slots problem)
        values (vec slots)]
    (when-not (= batch-size (count values))
      (throw (ex-info "Paged K/V append slot vector has the wrong lane count"
                      {:expected batch-size :actual (count values)})))
    (doseq [[lane slot] (map-indexed vector values)]
      (when-not (and (integer? slot) (<= 0 (long slot)) (< (long slot) capacity))
        (throw (ex-info "Paged K/V append slot is outside the physical pool"
                        {:lane lane :slot slot :physical-slots capacity}))))
    (when-not (= (count values) (count (set values)))
      (throw (ex-info "Paged K/V append requires unique destination slots"
                      {:slots values})))
    slots))
