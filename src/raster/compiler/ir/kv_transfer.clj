(ns raster.compiler.ir.kv-transfer
  "A continuation's key/value pages as a certified cross-device transfer plan.

   Rung D8 of the north star. The plan owns no transport: it derives, from checked paged
   geometry, the byte-exact set of resident ranges that move and the `DistributedPlan` transfer
   steps that carry them — one leg per layer, so a transfer interleaves with a per-layer decode
   loop on the target. Bytes are derived from geometry and never stated by the caller; the
   distributed certificate recomputes them.

   Geometry (position-major caches, one K and one V buffer per layer, as
   `raster.compiler.ir.paged-kv-append` lays them out): a page holds `page-size` token
   positions; token position `t` of a layer buffer occupies elements
   `[t·elements-per-token, (t+1)·elements-per-token)`. A page `p` is therefore the contiguous
   element range `[p·page-size·elements-per-token, (p+1)·page-size·elements-per-token)` of each
   layer buffer, and a continuation's pages need not be contiguous: the range list is the
   fragmented transfer the runtime executes with `upload-ranges!`/`download-ranges!`/
   `copy-range!`, or a peer route executes over a socket, byte for byte."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.abstract-value :as abstract-value]
            [raster.compiler.ir.distributed-plan :as distributed]
            [raster.compiler.ir.validate :refer [fail!]]))

(defrecord PagedGeometry
           [layers page-size physical-pages key-elements-per-token value-elements-per-token
            storage-dtype])

(defn paged-geometry?
  [value]
  (instance? PagedGeometry value))

(defn- positive-integer!
  [field value]
  (when-not (and (integer? value) (pos? value))
    (fail! "paged geometry extents must be positive integers"
           :kv-transfer-geometry {:field field :value value}))
  (long value))

(defn geometry
  "Checked paged geometry. `storage-dtype` is the cache's storage dtype (`:half` for an FP16
   page pool); byte counts follow from it, never from the arithmetic dtype."
  [{:keys [layers page-size physical-pages key-elements-per-token value-elements-per-token
           storage-dtype]}]
  (let [storage-dtype (dtype/canon storage-dtype)]
    (when-not (dtype/known? storage-dtype)
      (fail! "paged geometry requires a known storage dtype"
             :kv-transfer-storage-dtype {:storage-dtype storage-dtype}))
    (->PagedGeometry (positive-integer! :layers layers)
                     (positive-integer! :page-size page-size)
                     (positive-integer! :physical-pages physical-pages)
                     (positive-integer! :key-elements-per-token key-elements-per-token)
                     (positive-integer! :value-elements-per-token value-elements-per-token)
                     storage-dtype)))

(defn validate-geometry!
  [value]
  (when-not (paged-geometry? value)
    (fail! "expected a PagedGeometry value" :kv-transfer-geometry-type {:actual (type value)}))
  (geometry value))

(defn- pages!
  "Pages of one continuation: a vector of distinct physical page indices in transfer order."
  [{:keys [physical-pages]} pages]
  (when-not (and (vector? pages) (seq pages)
                 (every? #(and (integer? %) (<= 0 % (dec physical-pages))) pages)
                 (= (count pages) (count (distinct pages))))
    (fail! "continuation pages must be distinct physical page indices inside the page pool"
           :kv-transfer-pages {:pages pages :physical-pages physical-pages}))
  (mapv long pages))

(defn page-elements
  "Elements one page occupies in a layer buffer with `elements-per-token` elements per token."
  [{:keys [page-size]} elements-per-token]
  (* (long page-size) (long elements-per-token)))

(defn layer-bytes
  "Bytes one layer's K and V pages of a continuation occupy."
  [{:keys [key-elements-per-token value-elements-per-token storage-dtype] :as geometry} pages]
  (let [pages (pages! geometry pages)
        element-bytes (long (dtype/bytes-of storage-dtype))]
    (* (count pages)
       (+ (page-elements geometry key-elements-per-token)
          (page-elements geometry value-elements-per-token))
       element-bytes)))

(defn total-bytes
  [{:keys [layers] :as geometry} pages]
  (* (long layers) (layer-bytes geometry pages)))

(defn ranges
  "The fragmented element ranges of a continuation, one entry per (layer, buffer, page):
   `{:layer l :buffer :key|:value :page p :element e :elements n}` in transfer order (layer-major,
   K before V, pages in the continuation's order). `:element` is the page's first element in
   the layer buffer, the same on the source and the target when both pools share the geometry;
   a target with a different page assignment rewrites `:page`/`:element` per entry."
  [{:keys [layers key-elements-per-token value-elements-per-token] :as geometry} pages]
  (let [geometry (validate-geometry! geometry)
        pages (pages! geometry pages)]
    (vec (for [layer (range layers)
               [buffer per-token] [[:key key-elements-per-token] [:value value-elements-per-token]]
               page pages
               :let [n (page-elements geometry per-token)]]
           {:layer layer :buffer buffer :page page :element (* page n) :elements n}))))

(defn- leg-id
  [id layer]
  (keyword (str (name id) "-layer-" layer)))

(defn- continuation-elements
  [{:keys [layers key-elements-per-token value-elements-per-token] :as geometry} pages]
  (* (long layers) (count pages)
     (+ (page-elements geometry key-elements-per-token)
        (page-elements geometry value-elements-per-token))))

(defn value-of
  "The AbstractValue a `DistributedPlan` declares for the moving pages: the storage dtype and
   the flat element shape of one continuation across every layer, replicated over `devices`
   (the source that owns it and every target that receives it)."
  [{:keys [storage-dtype] :as geometry} pages devices]
  (let [geometry (validate-geometry! geometry)
        pages (pages! geometry pages)]
    (abstract-value/tensor {:dtype storage-dtype :shape [(continuation-elements geometry pages)]
                            :representation {:kind :plain} :memory-space :device
                            :sharding {:kind :replicated :devices (vec devices)}
                            :ownership :owned})))

(defn shards
  "The plan's physical placements of continuation `id`: one full replica per device."
  [id geometry pages devices]
  (let [geometry (validate-geometry! geometry)
        pages (pages! geometry pages)
        elements (continuation-elements geometry pages)]
    (mapv (fn [device]
            (distributed/shard {:id (keyword (str (name id) "-on-" (name device)))
                                :value id :device device :offsets [0] :shape [elements]
                                :ownership :replica}))
          devices)))

(defn transfer
  "The transfer steps of one continuation: one `DistributedPlan` transfer leg per layer, in
   layer order, each depending on `dependencies` (page reservation on the target, the decode
   event that produced the pages) and on nothing else; legs of one continuation serialize on
   their route's links, so a per-layer compute loop on the target may consume layer `l` while
   layer `l+1` is in flight. Returns `{:value :steps :ranges :bytes}`; `:value` is the
   continuation's page-set identity the plan must declare with `value-of`."
  [{:keys [id source target route geometry pages dependencies attributes]
    :or {dependencies [] attributes {}}}]
  (let [geometry (validate-geometry! geometry)
        pages (pages! geometry pages)
        per-layer (layer-bytes geometry pages)
        value id]
    (when (nil? id)
      (fail! "kv transfer requires an identity" :kv-transfer-id {}))
    {:value value
     :bytes (* (long (:layers geometry)) per-layer)
     :ranges (ranges geometry pages)
     :steps (mapv (fn [layer]
                    (distributed/transfer-step
                     {:id (leg-id id layer) :source source :target target :route route
                      :value value :bytes per-layer :dependencies (vec dependencies)
                      :attributes (assoc attributes :kv-transfer id :layer layer
                                         :pages pages)}))
                  (range (:layers geometry)))}))

(defn certified-bytes
  "Bytes a certified plan moves for continuation `id`, recomputed from its legs."
  [certified id]
  (reduce + 0 (map :bytes (filter #(= id (get-in % [:attributes :kv-transfer]))
                                  (get-in certified [:plan :steps])))))
