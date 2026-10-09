(ns raster.runtime.resident-state
  "Capture pinned completed numerical bytes through the existing manifest/provider contracts.
   This module owns no store, compiler cache or GPU session. Semantic field names, coordinates
   and numerical policy are application declarations; physical producer/byte facts are derived."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.buffer-view :as view]
            [raster.compiler.ir.numerical-state :as state]
            [raster.compiler.ir.validate :refer [exact-keys! fail! unique-by!]]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.runtime.numerical-content :as content]))

(defn- plain-portable-tensor? [value]
  (and (= :tensor (:kind value))
       (nil? (:logical-layout value)) (= {:kind :plain} (:representation value))
       (every? nil? (map #(get value %) [:memory-space :placement :sharding :ownership]))
       (empty? (:effects value)) (empty? (:attributes value))
       (seq (:shape value)) (every? pos-int? (:shape value))))

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
        _ (when-not (and (plain-portable-tensor? value) dt
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
        extent (:byte-length leaf)]
    (content/element-byte-reader
     extent element-bytes
     (fn [start elements destination]
       (gpu/download-range! (:session executable) resident destination
                            {:src-element start :elements elements})))))

(defn- producer-bindings [prepared specs]
  (when-not (and (vector? specs) (seq specs) (every? map? specs))
    (fail! "restore requires ordered semantic producer field specifications"
           :resident-state-producer-fields {}))
  (unique-by! "producer fields" :resident-state-producer-field-identities :id specs)
  (let [interface (compiled/producer-interface prepared)]
    (exact-keys! "producer interface" :resident-state-producer-interface interface
                 #{:kind :program :outputs :post-state})
    (when-not (= :raster.compiled/producer-interface-v1 (:kind interface))
      (fail! "restore requires the known producer interface schema" :resident-state-producer-interface {}))
    (exact-keys! "producer program" :resident-state-producer-interface (:program interface)
                 #{:scope :fingerprint})
    (when-not (= :exact-bound-program (get-in interface [:program :scope]))
      (fail! "restore requires exact retained source program identity" :resident-state-producer-interface {}))
    {:program-fingerprint (get-in interface [:program :fingerprint])
     :fields
     (mapv (fn [{:keys [id source key] :as spec}]
             (exact-keys! "producer field" :resident-state-producer-field-keys spec #{:id :source :key})
             (when-not (and (some? id) (contains? #{:outputs :post-state} source))
               (fail! "producer fields require an identity and current frontier"
                      :resident-state-producer-source {:field id :source source}))
             (let [ports (filter #(= key (:key %)) (get interface source))]
               (when-not (= 1 (count ports))
                 (fail! "producer field has no unique public source port"
                        :resident-state-producer-port {:field id :source source :key key}))
               (when-not (= 1 (count (filter #(= (:node (first ports)) (:node %))
                                             (get interface source))))
                 (fail! "node-only capture cannot distinguish aliased public source ports"
                        :resident-state-producer-port {:field id :source source :key key}))
               (assoc (select-keys (first ports) [:node :dtype :shape]) :field-id id :source source)))
           specs)}))

(defn verify-restore!
  "Check captured-state compatibility before opening leases or uploading any bytes.

   expected-semantics independently declares the complete fields/coordinates, logical phase and
   numerical policy. source-prepared is the original sealed source producer, not the continuation
   consumer. Ordered field specs contain :id, :source (:outputs or :post-state) and its public :key;
   source nodes and exact program identity are resolved from retained compiler evidence.
   Aliased public ports are rejected: node-only capture cannot distinguish their semantic keys.

   All three semantic facets and exact ordered producer bindings must agree. Every v1 full-field
   raw chunk must match its recorded producer content and physical dtype/shape. Measured
   representation kind/dtype/program/byte-order claims must also agree with each stored chunk.
   Dynamic replay fingerprints, schedules and observations remain audit data, not target predictions.
   This verifies consistency/compatibility, not authentication of serialized claims or actual
   execution, parent existence, byte integrity, codec conversion or numerical equivalence.
   The generic numerical-state verify-restore! still requires exact complete provenance."
  [certified expected-semantics source-prepared field-specs]
  (state/verify-restore-semantics! certified expected-semantics)
  (let [manifest (:manifest certified)
        {:keys [program-fingerprint fields]} (producer-bindings source-prepared field-specs)
        provenance (:provenance manifest)
        actual (:field-producers provenance)]
    (exact-keys! "captured producer provenance" :resident-state-provenance provenance
                 #{:scope :semantic-authority :program-fingerprint :completed-fingerprint
                   :bound-schedules :parent-replay :representations :field-producers})
    (when-not (and (= :exact-completed-linked-replay (:scope provenance))
                   (= :application-declared (:semantic-authority provenance))
                   (string? (:completed-fingerprint provenance)) (seq (:completed-fingerprint provenance))
                   (or (nil? (:parent-replay provenance))
                       (and (string? (:parent-replay provenance)) (seq (:parent-replay provenance))))
                   (vector? (:bound-schedules provenance)) (every? map? (:bound-schedules provenance))
                   (map? (:representations provenance))
                   (vector? actual) (every? map? actual))
      (fail! "restore requires the complete captured producer schema" :resident-state-provenance {}))
    (when-not (= program-fingerprint (:program-fingerprint provenance))
      (fail! "captured state does not name the independently supplied source program"
             :resident-state-producer-program {}))
    (when-not (= (mapv #(select-keys % [:field-id :source :node]) fields)
                 (mapv #(select-keys % [:field-id :source :node]) actual))
      (fail! "captured state does not match ordered public source field bindings"
             :resident-state-producer-bindings {}))
    (when-not (= (mapv :field-id fields) (mapv :id (:fields manifest)))
      (fail! "captured field order differs from its producer bindings" :resident-state-producer-bindings {}))
    (doseq [[expected producer field] (map vector fields actual (:fields manifest))]
      (exact-keys! "captured field producer" :resident-state-producer-bindings producer
                   #{:field-id :source :node :content})
      (let [value (:value field) chunks (:chunks field) chunk (first chunks)
            representation (get (:representations provenance) (:dtype expected))
            byte-order (if (= :order-invariant (:byte-order representation))
                         :little-endian (:byte-order representation))]
        (when-not (and (= :raster.compiled/resident-representation-v1 (:kind representation))
                       (= (:dtype expected) (:dtype representation))
                       (= program-fingerprint (:program-fingerprint representation))
                       (contains? state/byte-orders byte-order)
                       (= byte-order (get-in chunk [:storage :byte-order])))
          (fail! "captured representation disagrees with its program, dtype or stored byte order"
                 :resident-state-representation {:field (:id field)}))
        (when-not (and (plain-portable-tensor? value)
                       (= (:dtype expected) (:dtype value)) (= (:shape expected) (:shape value))
                       (= 1 (count chunks)) (= 0 (:id chunk))
                       (= (:shape value) (:chunk-shape field) (:shape chunk))
                       (= (vec (repeat (count (:shape value)) 0)) (:offsets chunk))
                       (= {:format :raw-array :byte-order (get-in chunk [:storage :byte-order])}
                          (:storage chunk))
                       (contains? state/byte-orders (get-in chunk [:storage :byte-order]))
                       (= (:logical-byte-length chunk) (:stored-byte-length chunk))
                       (= (:content chunk) (:content producer)))
          (fail! "captured full-field chunk disagrees with its producer binding or storage"
                 :resident-state-producer-content {:field (:id field)}))))
    certified))

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
