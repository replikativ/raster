(ns raster.runtime.resident-state
  "Capture pinned completed numerical bytes through the existing manifest/provider contracts.
   This module owns no store, compiler cache or GPU session. Semantic field names, coordinates
   and numerical policy are application declarations; physical producer/byte facts are derived."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.buffer-view :as view]
            [raster.compiler.ir.numerical-state :as state]
            [raster.compiler.ir.validate :refer [exact-keys! fail!]]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.runtime.numerical-content :as content])
  (:import [java.lang.foreign MemorySegment]))

(defn- capture-field [description executable spec]
  (when-not (map? spec)
    (fail! "capture field must be a map" :resident-state-field {}))
  (exact-keys! "capture field" :resident-state-field-keys spec
               #{:id :source :node :value :coordinate-space :attributes})
  (let [{:keys [id source node value coordinate-space attributes]
         :or {coordinate-space {} attributes {}}} spec
        _ (when-not (contains? #{:outputs :post-state} source)
            (fail! "capture requires current output or post-state bytes, not historical inputs"
                   :resident-state-source {:field id :source source}))
        leaf (or (get-in description [source node])
                 (fail! "capture field has no leaf in its completed frontier"
                        :resident-state-node {:field id :source source :node node}))
        _ (av/validate! value)
        dt (when (keyword? (:dtype value)) (dtype/canon (:dtype value)))
        _ (when-not (and (= :tensor (:kind value)) dt
                         (nil? (:logical-layout value)) (= {:kind :plain} (:representation value))
                         (every? nil? (map #(get value %) [:memory-space :placement :sharding :ownership]))
                         (empty? (:effects value)) (empty? (:attributes value))
                         (seq (:shape value)) (every? pos-int? (:shape value))
                         (= dt (:dtype leaf)) (= (:shape value) (:shape leaf)))
            (fail! "capture v1 requires an exact positive plain portable tensor matching its leaf"
                   :resident-state-value {:field id :node node}))
        physical (get-in executable [:plan :nodes node :view])
        width (long (dtype/bytes-of dt))
        bytes (reduce (fn [n extent] (Math/multiplyExact (long n) (long extent))) width (:shape value))
        _ (when-not (and (= :device-native (:representation leaf))
                         (= {:format :raw-array :byte-order (get-in leaf [:storage :byte-order])}
                            (:storage leaf))
                         (contains? state/byte-orders (get-in leaf [:storage :byte-order]))
                         (view/contiguous? physical)
                         (= (:strides physical) (:strides leaf))
                         (= bytes (:byte-length leaf) (:byte-length physical))
                         (= dt (:dtype physical)) (= (:shape value) (:shape physical)))
            (fail! "capture requires exact dense measured raw-array storage"
                   :resident-state-storage {:field id :node node}))
        canonical (av/tensor {:dtype dt :shape (:shape value)})
        field (state/field
               {:id id :value canonical :coordinate-space coordinate-space :attributes attributes
                :chunk-shape (:shape value)
                :chunks [(state/chunk {:id 0 :offsets (vec (repeat (count (:shape value)) 0))
                                       :shape (:shape value) :logical-byte-length bytes
                                       :stored-byte-length bytes :content (:content leaf)
                                       :storage (:storage leaf)})]})]
    {:field field :node node :source source :leaf leaf :element-bytes width}))

(defn- capture-plan [description receipt-data executable opts]
  (when-not (map? opts)
    (fail! "capture options must be a map" :resident-state-options {}))
  (exact-keys! "capture options" :resident-state-option-keys opts
               #{:id :parents :logical-coordinate :fields :numerical-contract :attributes})
  (when-not (and (vector? (:fields opts)) (seq (:fields opts)))
    (fail! "capture requires ordered semantic field specifications" :resident-state-fields {}))
  (let [fields (mapv #(capture-field description executable %) (:fields opts))
        provenance {:scope :exact-completed-linked-replay
                    :semantic-authority :application-declared
                    :program-fingerprint (:program-fingerprint description)
                    :completed-fingerprint (:completed-fingerprint description)
                    :bound-schedules (:bound-schedules receipt-data)
                    :parent-replay (:parent receipt-data)
                    :representations (:representations description)
                    :field-producers (mapv (fn [{:keys [field source node leaf]}]
                                             {:field-id (:id field) :source source :node node
                                              :content (:content leaf)}) fields)}
        manifest (state/manifest (assoc (dissoc opts :fields) :fields (mapv :field fields)
                                        :provenance provenance))]
    {:state (state/certify manifest) :fields fields}))

(defn- resident-reader [executable {:keys [node leaf element-bytes]}]
  (let [resident (link/node-view executable node)
        scratch (MemorySegment/ofArray (byte-array (+ 65536 (* 2 (dec element-bytes)))))
        extent (:byte-length leaf)]
    (fn [offset ^MemorySegment destination]
      (let [n (.byteSize destination)
            end (Math/addExact (long offset) n)
            start (- offset (mod offset element-bytes))
            aligned-end (Math/addExact end (mod (- element-bytes (mod end element-bytes)) element-bytes))
            span (- aligned-end start)]
        (when-not (and (<= 0 offset end extent) (<= span (.byteSize scratch)))
          (fail! "capture read exceeds its pinned leaf" :resident-state-read-range {:node node}))
        (gpu/download-range! (:session executable) resident (.asSlice scratch 0 span)
                             {:src-element (quot start element-bytes) :elements (quot span element-bytes)})
        (MemorySegment/copy scratch (- offset start) destination 0 n)
        n))))

(defn capture!
  "Capture selected current leaves from an original live completed receipt.

   facts are the exact owner-bound measured representation evidence used by
   completed-storage-description. opts declares ordered fields with :id, :source (:outputs or
   :post-state), :node and a plain portable tensor :value; coordinates/policy remain declared.
   Version 1 emits one complete raw-array chunk per field. Packed/strided/composite storage needs
   an explicit codec/packing vertical and is not guessed here.

   Validate/certify every field before provider writes. An additional existing Link output lease
   pins the entire capture, even if provider callbacks close the caller receipt. Sources stream
   synchronously through bounded aligned downloads; only requested bytes enter provider windows.
   Placements remain runtime data. Later-field failure may leave earlier orphan blobs and returns
   no state; this is not a storage transaction or durable publication. Use
   finalize-state-availability! to independently reopen/verify/promote before metadata publication.
   Direct mutation of the owned session remains outside the LinkedExecutable lease contract."
  [receipt facts provider target-tier opts]
  (when-not (compiled/completed-evidence? receipt)
    (fail! "capture requires original completed evidence" :compiled-completed-evidence-owner {}))
  (let [executable (:executable receipt)
        held (volatile! nil)
        owner (cleanup/owner [{:id :capture-output-lease
                               :release #(when-let [lease @held] (.close ^java.io.Closeable lease))}])
        result (cleanup/build!
                owner
                (fn []
                  (let [{:keys [state fields]}
                        (locking (:lifetime-lock executable)
                          (let [description (compiled/completed-storage-description receipt facts)
                                plan (capture-plan description @receipt executable opts)]
                            (vreset! held (link/output-lease! executable))
                            plan))]
                    {:state state
                     :placements
                     (mapv (fn [{:keys [field leaf] :as entry}]
                             {:field-id (:id field) :chunk-id 0
                              :placement (content/ingest-content!
                                          provider (:content leaf) target-tier (:byte-length leaf)
                                          (resident-reader executable entry))}) fields)}))
                nil)]
    (try (cleanup/release! owner)
         (catch Throwable error
           (throw (ex-info "capture retains unresolved lease cleanup"
                           {::cleanup/unresolved owner} error))))
    (dissoc result ::cleanup/owner)))
