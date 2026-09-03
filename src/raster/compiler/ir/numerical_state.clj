(ns raster.compiler.ir.numerical-state
  "Certified logical snapshots of durable numerical state.

   A NumericalStateManifest connects backend-neutral AbstractValues to immutable rectangular
   chunks.  It records lineage, logical coordinates, numerical compatibility and producer
   provenance, but deliberately contains no store handles, paths, buckets, mmap segments, device
   buffers, queues or events.  A runtime may therefore realize the same content addresses through
   a local file/Konserve mmap tier, LMDB, S3, or another object store without changing compiler IR.

   Version 1 describes complete dense fields on a regular chunk grid.  Reusing a parent's content
   address gives incremental checkpoints without making a manifest depend on its parent for
   reconstruction.  Sparse and AMR states are represented as multiple explicitly coordinated
   fields; future schemas may add partial/delta field coverage without weakening this contract."
  (:refer-clojure :exclude [chunk])
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.abstract-value :as abstract-value]
            [raster.compiler.ir.validate :refer [fail! exact-keys! non-blank-string? unique-by!]]))

(def schema-version 1)

(def determinism-contracts
  #{:bitwise :reproducible-order :toleranced :nondeterministic})

(def byte-orders #{:little-endian :big-endian})

(defrecord ContentAddress [algorithm digest])
(defrecord StateChunk
           [id offsets shape logical-byte-length stored-byte-length content storage attributes])
(defrecord StateField
           [id value chunk-shape coordinate-space chunks attributes])
(defrecord NumericalStateManifest
           [id schema-version parents logical-coordinate fields numerical-contract provenance
            attributes])
(defrecord NumericalStateCertificate
           [state-id schema-version parents logical-coordinate fields numerical-contract provenance
            attributes logical-byte-length stored-byte-length content-addresses])
(defrecord CertifiedNumericalState [manifest certificate])

(defn content-address? [value] (instance? ContentAddress value))
(defn state-chunk? [value] (instance? StateChunk value))
(defn state-field? [value] (instance? StateField value))
(defn manifest? [value] (instance? NumericalStateManifest value))
(defn certificate? [value] (instance? NumericalStateCertificate value))
(defn certified-state? [value] (instance? CertifiedNumericalState value))

(defn content-address
  "Construct a storage-independent identity for immutable bytes.

   `algorithm` identifies the digest algorithm (normally `:sha-256`); `digest` is its canonical
   printable representation.  An object key, bucket, local path or LMDB key is a runtime placement
   of this identity and must not be embedded here."
  [algorithm digest]
  (when-not (keyword? algorithm)
    (fail! "content-address algorithm must be a keyword"
           :numerical-state-content-algorithm {:algorithm algorithm}))
  (when-not (non-blank-string? digest)
    (fail! "content-address digest must be a non-blank string"
           :numerical-state-content-digest {:digest digest}))
  (->ContentAddress algorithm digest))

(defn- validate-content-address!
  [candidate]
  (when-not (content-address? candidate)
    (fail! "state chunk requires an immutable ContentAddress"
           :numerical-state-chunk-content {:content candidate}))
  (exact-keys! "content address" :numerical-state-content-fields candidate
               #{:algorithm :digest})
  (content-address (:algorithm candidate) (:digest candidate))
  candidate)

(defn chunk
  [{:keys [id offsets shape logical-byte-length stored-byte-length content storage attributes]
    :or {attributes {}}}]
  (when (nil? id)
    (fail! "state chunk requires an identity" :numerical-state-chunk-id {}))
  (when-not (and (vector? offsets) (every? #(and (integer? %) (not (neg? %))) offsets))
    (fail! "state chunk offsets must be a vector of non-negative integers"
           :numerical-state-chunk-offsets {:chunk id :offsets offsets}))
  (when-not (and (vector? shape) (seq shape) (every? pos-int? shape))
    (fail! "state chunk shape must be a non-empty vector of positive integers"
           :numerical-state-chunk-shape {:chunk id :shape shape}))
  (when-not (= (count offsets) (count shape))
    (fail! "state chunk offsets and shape must have equal rank"
           :numerical-state-chunk-rank {:chunk id :offsets offsets :shape shape}))
  (when-not (and (integer? logical-byte-length) (not (neg? logical-byte-length)))
    (fail! "state chunk logical byte length must be a non-negative integer"
           :numerical-state-chunk-logical-bytes
           {:chunk id :logical-byte-length logical-byte-length}))
  (when-not (pos-int? stored-byte-length)
    (fail! "state chunk stored byte length must be a positive integer"
           :numerical-state-chunk-stored-bytes
           {:chunk id :stored-byte-length stored-byte-length}))
  (validate-content-address! content)
  (when-not (and (map? storage) (keyword? (:format storage)))
    (fail! "state chunk storage requires a map with a keyword :format"
           :numerical-state-chunk-storage {:chunk id :storage storage}))
  (when-not (contains? byte-orders (:byte-order storage))
    (fail! "state chunk storage requires a durable, explicit byte order"
           :numerical-state-chunk-byte-order
           {:chunk id :byte-order (:byte-order storage) :allowed byte-orders}))
  (when-not (map? attributes)
    (fail! "state chunk attributes must be a map"
           :numerical-state-chunk-attributes {:chunk id :attributes attributes}))
  (->StateChunk id offsets shape logical-byte-length stored-byte-length content storage attributes))

(defn field
  [{:keys [id value chunk-shape coordinate-space chunks attributes]
    :or {coordinate-space {} chunks [] attributes {}}}]
  (when (nil? id)
    (fail! "state field requires an identity" :numerical-state-field-id {}))
  (abstract-value/validate! value)
  (when-not (= :tensor (:kind value))
    (fail! "version 1 numerical state fields must contain tensor AbstractValues"
           :numerical-state-field-kind {:field id :kind (:kind value)}))
  (when-not (and (vector? chunk-shape) (seq chunk-shape) (every? pos-int? chunk-shape))
    (fail! "state field chunk shape must be a non-empty vector of positive integers"
           :numerical-state-field-chunk-shape {:field id :chunk-shape chunk-shape}))
  (when-not (= (count (:shape value)) (count chunk-shape))
    (fail! "state field value and chunk grid must have equal rank"
           :numerical-state-field-rank
           {:field id :shape (:shape value) :chunk-shape chunk-shape}))
  (when-not (map? coordinate-space)
    (fail! "state field coordinate space must be a map"
           :numerical-state-field-coordinate-space
           {:field id :coordinate-space coordinate-space}))
  (when-not (and (vector? chunks) (every? state-chunk? chunks))
    (fail! "state field chunks must be a vector of StateChunks"
           :numerical-state-field-chunks {:field id :chunks chunks}))
  (when-not (map? attributes)
    (fail! "state field attributes must be a map"
           :numerical-state-field-attributes {:field id :attributes attributes}))
  (->StateField id value chunk-shape coordinate-space chunks attributes))

(defn manifest
  [{:keys [id schema-version parents logical-coordinate fields numerical-contract provenance
           attributes]
    :or {schema-version raster.compiler.ir.numerical-state/schema-version
         parents [] logical-coordinate {} fields [] attributes {}}}]
  (->NumericalStateManifest id schema-version parents logical-coordinate fields numerical-contract
                            provenance attributes))

(defn- concrete-shape!
  [field-id value]
  (let [shape (:shape value)]
    (when-not (and (seq shape) (every? pos-int? shape))
      (fail! "durable numerical fields require a positive concrete logical shape"
             :numerical-state-field-shape {:field field-id :shape shape}))
    shape))

(defn- grid-offsets
  [shape chunk-shape]
  (reduce (fn [prefixes [extent step]]
            (vec (for [prefix prefixes
                       offset (range 0 extent step)]
                   (conj prefix offset))))
          [[]]
          (map vector shape chunk-shape)))

(defn- expected-chunk-shape
  [shape chunk-shape offsets]
  (mapv (fn [extent chunk-extent offset]
          (min chunk-extent (- extent offset)))
        shape chunk-shape offsets))

(defn- validate-chunks!
  [field-id value chunk-shape chunks]
  (let [shape (concrete-shape! field-id value)
        element-bytes (dtype/bytes-of (:dtype value))
        expected-offsets (grid-offsets shape chunk-shape)
        actual-offsets (mapv :offsets chunks)]
    (unique-by! "state chunks" :numerical-state-chunk-identities :id chunks)
    (doseq [candidate chunks]
      ;; Reconstructing invokes chunk-local checks even after record mutation through assoc.
      (exact-keys! "state chunk" :numerical-state-chunk-fields candidate
                   #{:id :offsets :shape :logical-byte-length :stored-byte-length
                     :content :storage :attributes})
      (chunk candidate))
    (when-not (= actual-offsets (vec (sort actual-offsets)))
      (fail! "state chunks must use canonical row-major offset order"
             :numerical-state-chunk-order {:field field-id :offsets actual-offsets}))
    (when-not (= expected-offsets actual-offsets)
      (fail! "state chunks must exactly cover the regular logical chunk grid"
             :numerical-state-chunk-coverage
             {:field field-id :expected-offsets expected-offsets
              :actual-offsets actual-offsets}))
    (doseq [candidate chunks]
      (let [expected-shape (expected-chunk-shape shape chunk-shape (:offsets candidate))
            expected-bytes (* element-bytes (reduce * 1 expected-shape))]
        (when-not (= expected-shape (:shape candidate))
          (fail! "state chunk shape does not match its clipped logical grid cell"
                 :numerical-state-chunk-grid-shape
                 {:field field-id :chunk (:id candidate) :offsets (:offsets candidate)
                  :expected expected-shape :actual (:shape candidate)}))
        (when-not (= expected-bytes (:logical-byte-length candidate))
          (fail! "state chunk logical byte length disagrees with its dtype and shape"
                 :numerical-state-chunk-logical-bytes
                 {:field field-id :chunk (:id candidate) :dtype (:dtype value)
                  :shape expected-shape :expected expected-bytes
                  :actual (:logical-byte-length candidate)}))))
    chunks))

(defn- validate-field!
  [candidate]
  (when-not (state-field? candidate)
    (fail! "numerical state fields must be StateFields"
           :numerical-state-field-type {:field candidate :actual (type candidate)}))
  (exact-keys! "state field" :numerical-state-field-fields candidate
               #{:id :value :chunk-shape :coordinate-space :chunks :attributes})
  ;; Reconstructing invokes all field-local structural checks even if a record was mutated with
  ;; assoc after construction.
  (field candidate)
  (let [value (abstract-value/validate! (:value candidate))]
    (when-not (dtype/known? (:dtype value))
      (fail! "durable numerical state requires a compiler-known element dtype"
             :numerical-state-field-dtype
             {:field (:id candidate) :dtype (:dtype value)}))
    (validate-chunks! (:id candidate) value (:chunk-shape candidate) (:chunks candidate)))
  candidate)

(defn- validate-numerical-contract!
  [contract]
  (when-not (map? contract)
    (fail! "numerical state requires an explicit numerical contract"
           :numerical-state-numerical-contract {:contract contract}))
  (when-not (keyword? (:mode contract))
    (fail! "numerical contract requires a keyword :mode"
           :numerical-state-numerical-mode {:contract contract}))
  (when-not (contains? determinism-contracts (:determinism contract))
    (fail! "numerical contract has an unsupported :determinism policy"
           :numerical-state-determinism
           {:determinism (:determinism contract) :allowed determinism-contracts}))
  (when-not (non-blank-string? (:compatibility-id contract))
    (fail! "numerical contract requires a non-blank :compatibility-id"
           :numerical-state-compatibility-id {:contract contract}))
  contract)

(defn- validate-provenance!
  [provenance]
  (when-not (map? provenance)
    (fail! "numerical state provenance must be a map"
           :numerical-state-provenance {:provenance provenance}))
  (when-not (non-blank-string? (:program-fingerprint provenance))
    (fail! "numerical state provenance requires a non-blank :program-fingerprint"
           :numerical-state-program-fingerprint {:provenance provenance}))
  provenance)

(defn validate!
  "Validate a complete, backend-neutral numerical snapshot manifest.

   Validation proves regular dense chunk coverage and exact logical byte counts.  Stored byte
   counts remain representation-dependent and are retained for placement/cost planning."
  [candidate]
  (when-not (manifest? candidate)
    (fail! "expected a NumericalStateManifest"
           :numerical-state-manifest-type {:actual (type candidate)}))
  (exact-keys! "numerical state manifest" :numerical-state-manifest-fields candidate
               #{:id :schema-version :parents :logical-coordinate :fields
                 :numerical-contract :provenance :attributes})
  (let [{:keys [id schema-version parents logical-coordinate fields numerical-contract provenance
                attributes]}
        candidate]
    (when (nil? id)
      (fail! "numerical state manifest requires an identity" :numerical-state-id {}))
    (when-not (= raster.compiler.ir.numerical-state/schema-version schema-version)
      (fail! "unsupported numerical state schema version"
             :numerical-state-schema-version
             {:expected raster.compiler.ir.numerical-state/schema-version :actual schema-version}))
    (when-not (and (vector? parents) (not-any? nil? parents))
      (fail! "numerical state parents must be a vector of non-nil identities"
             :numerical-state-parents {:parents parents}))
    (unique-by! "numerical state parents" :numerical-state-parent-identities identity parents)
    (when (some #{id} parents)
      (fail! "a numerical state cannot name itself as a parent"
             :numerical-state-parent-cycle {:state id :parents parents}))
    (when-not (map? logical-coordinate)
      (fail! "numerical state logical coordinate must be a map"
             :numerical-state-logical-coordinate {:logical-coordinate logical-coordinate}))
    (when-not (and (vector? fields) (seq fields))
      (fail! "numerical state requires a non-empty ordered field vector"
             :numerical-state-fields {:fields fields}))
    (unique-by! "numerical state fields" :numerical-state-field-identities :id fields)
    (doseq [candidate-field fields]
      (validate-field! candidate-field))
    (validate-numerical-contract! numerical-contract)
    (validate-provenance! provenance)
    (when-not (map? attributes)
      (fail! "numerical state attributes must be a map"
             :numerical-state-attributes {:attributes attributes})))
  candidate)

(defn- derive-certificate
  [manifest]
  (let [chunks (mapcat :chunks (:fields manifest))]
    (->NumericalStateCertificate
     (:id manifest)
     (:schema-version manifest)
     (:parents manifest)
     (:logical-coordinate manifest)
     (:fields manifest)
     (:numerical-contract manifest)
     (:provenance manifest)
     (:attributes manifest)
     (reduce + 0 (map :logical-byte-length chunks))
     (reduce + 0 (map :stored-byte-length chunks))
     (mapv :content chunks))))

(defn certify
  "Validate a manifest and attach a reproducible coverage, lineage and byte-cost witness."
  [manifest]
  (let [manifest (validate! manifest)]
    (->CertifiedNumericalState manifest (derive-certificate manifest))))

(defn verify!
  "Revalidate a CertifiedNumericalState and independently derive its certificate."
  [certified]
  (when-not (certified-state? certified)
    (fail! "expected a CertifiedNumericalState"
           :numerical-state-certified-type {:actual (type certified)}))
  (let [manifest (validate! (:manifest certified))
        certificate (:certificate certified)
        expected (derive-certificate manifest)]
    (when-not (certificate? certificate)
      (fail! "numerical state certificate has the wrong type"
             :numerical-state-certificate-type {:actual (type certificate)}))
    (when-not (= expected certificate)
      (fail! "numerical state certificate does not match its manifest"
             :numerical-state-certificate {:expected expected :actual certificate}))
    certified))
