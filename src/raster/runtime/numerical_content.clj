(ns raster.runtime.numerical-content
  "Runtime realization of immutable numerical-state chunks.

   The compiler manifest names content, not storage products. A ContentProvider describes its
   tiers and capabilities, submits asynchronous promotion/localization operations, and opens a
   local realization as a scoped MemorySegment lease. Implementations may wrap Konserve file/S3
   tiering, LMDB transactions, Ceph, EOS, GPFS, Lustre, or an institutional archive.

   This namespace owns no store implementation. In particular, remote object storage is localized
   before `open-local-content!`; a remote object is never represented as a fictitious mmap."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.execution-plan :as execution]
            [raster.compiler.ir.numerical-state :as numerical-state])
  (:import [java.lang AutoCloseable]
           [java.lang.foreign MemorySegment]
           [java.security MessageDigest]
           [java.util HexFormat]))

(defrecord StorageTier [id kind locality durability capabilities attributes])
(defrecord ContentProviderDescriptor [id tiers capabilities attributes])
(defrecord ContentPlacement [provider-id tier-id content attributes])
(defrecord StorageEvent [provider-id id operation queue])

(defrecord LocalContentLease
           [content placement segment byte-offset byte-length release-fn closed-state]
  AutoCloseable
  (close [_]
    (when (compare-and-set! closed-state false true)
      (release-fn))
    nil))

(defprotocol ContentProvider
  "Backend contract for immutable numerical content.

   Provider events are opaque, provider-owned completions. Promotion establishes the requested
   durable tier; localization establishes a local realization that can subsequently be opened as a
   scoped lease. Event methods deliberately mirror Raster GPU events without sharing native handles
   or pretending storage and device queues are the same resource. Content-addressed realizations
   are immutable. A :durable-receipt placement promises those named bytes in the declared tier.
   Await establishes completion and returns a ContentPlacement for localization/promotion.

   -release-storage-event! is safe drain, not best-effort cancellation: whether await was never
   called, succeeded or failed, release must settle work and end provider borrows before returning.
   If draining fails, ownership remains internal to the provider and no result may be consumed.
   Providers unable to guarantee this must not advertise publication capabilities."
  (-provider-descriptor [provider])
  (-submit-promotion! [provider content target-tier opts])
  (-submit-localization! [provider content opts])
  (-open-local-content! [provider content opts])
  (-storage-event-complete? [provider event])
  (-await-storage-event! [provider event])
  (-storage-event-measurement [provider event])
  (-release-storage-event! [provider event]))

(defprotocol ContentIngestor
  "Optional ingestion capability alongside ContentProvider, using its existing event lifecycle.
   Each provider-owned window is lent exclusively for the synchronous callback; the reader must
   not retain or access it afterwards. Providers must not commit addressed content after a callback
   failure. Submission failure without a returned event leaves cleanup internal to the provider."
  (-submit-ingestion! [provider content target-tier byte-length read! opts]
    "Consume ordered source windows synchronously on the submitting thread, then return an
     :ingest placement event. Never retain or asynchronously invoke read!. Each writable window
     is at most 64 KiB and filled completely. Source lifetime ends when submission returns."))

(defn storage-tier? [value] (instance? StorageTier value))
(defn provider-descriptor? [value] (instance? ContentProviderDescriptor value))
(defn content-placement? [value] (instance? ContentPlacement value))
(defn storage-event? [value] (instance? StorageEvent value))
(defn local-content-lease? [value] (instance? LocalContentLease value))

(defn- fail!
  [message reason data]
  (throw (ex-info message (assoc data :reason reason))))

(defn- release-after-error! [error release!]
  (try (release!)
       (catch Throwable cleanup
         (when-not (identical? error cleanup) (.addSuppressed ^Throwable error cleanup))))
  (throw error))

(defn- with-release* [body release!]
  (let [result (try (body)
                    (catch Throwable error (release-after-error! error release!)))]
    (release!)
    result))

(defn- keyword-set?
  [value]
  (and (set? value) (every? keyword? value)))

(defn storage-tier
  [{:keys [id kind locality durability capabilities attributes]
    :or {capabilities #{} attributes {}}}]
  (when (nil? id)
    (fail! "storage tier requires an identity" :numerical-content-tier-id {}))
  (doseq [[field value] [[:kind kind] [:locality locality] [:durability durability]]]
    (when-not (keyword? value)
      (fail! "storage tier facets must be explicit keywords"
             :numerical-content-tier-facet {:tier id :field field :value value})))
  (when-not (keyword-set? capabilities)
    (fail! "storage tier capabilities must be a set of keywords"
           :numerical-content-tier-capabilities
           {:tier id :capabilities capabilities}))
  (when-not (map? attributes)
    (fail! "storage tier attributes must be a map"
           :numerical-content-tier-attributes {:tier id :attributes attributes}))
  (->StorageTier id kind locality durability capabilities attributes))

(defn provider-description
  [{:keys [id tiers capabilities attributes]
    :or {capabilities #{} attributes {}}}]
  (when (nil? id)
    (fail! "content provider requires an identity" :numerical-content-provider-id {}))
  (when-not (and (vector? tiers) (seq tiers) (every? storage-tier? tiers))
    (fail! "content provider requires a non-empty ordered vector of StorageTiers"
           :numerical-content-provider-tiers {:provider id :tiers tiers}))
  (let [tier-ids (mapv :id tiers)]
    (when-not (= (count tier-ids) (count (distinct tier-ids)))
      (fail! "content provider tier identities must be unique"
             :numerical-content-provider-tier-identities
             {:provider id :tiers tier-ids})))
  (when-not (keyword-set? capabilities)
    (fail! "content provider capabilities must be a set of keywords"
           :numerical-content-provider-capabilities
           {:provider id :capabilities capabilities}))
  (when-not (map? attributes)
    (fail! "content provider attributes must be a map"
           :numerical-content-provider-attributes
           {:provider id :attributes attributes}))
  (->ContentProviderDescriptor id tiers capabilities attributes))

(defn content-placement
  [{:keys [provider-id tier-id content attributes]
    :or {attributes {}}}]
  (when (nil? provider-id)
    (fail! "content placement requires a provider identity"
           :numerical-content-placement-provider {}))
  (when (nil? tier-id)
    (fail! "content placement requires a tier identity"
           :numerical-content-placement-tier {:provider provider-id}))
  (when-not (numerical-state/content-address? content)
    (fail! "content placement requires an immutable ContentAddress"
           :numerical-content-placement-content {:content content}))
  (when-not (map? attributes)
    (fail! "content placement attributes must be a map"
           :numerical-content-placement-attributes {:attributes attributes}))
  (->ContentPlacement provider-id tier-id content attributes))

(defn storage-event
  [{:keys [provider-id id operation queue]
    :or {queue (execution/storage-queue)}}]
  (when (nil? provider-id)
    (fail! "storage event requires a provider identity"
           :numerical-content-event-provider {}))
  (when (nil? id)
    (fail! "storage event requires an identity"
           :numerical-content-event-id {:provider provider-id}))
  (when-not (keyword? operation)
    (fail! "storage event operation must be a keyword"
           :numerical-content-event-operation {:operation operation}))
  (when-not (and (execution/logical-queue? queue) (= :storage (:class queue)))
    (fail! "storage event queue must be a logical storage queue"
           :numerical-content-event-queue {:queue queue}))
  (->StorageEvent provider-id id operation queue))

(defn local-content-lease
  "Create an AutoCloseable lease over a local content realization.

   `release-fn` closes the mapped-file arena, LMDB read transaction, cache pin, or equivalent
   provider resource exactly once. `lease-segment` returns the declared slice and fails after close."
  [{:keys [content placement segment byte-offset byte-length release-fn]
    :or {byte-offset 0}}]
  (when-not (numerical-state/content-address? content)
    (fail! "local content lease requires an immutable ContentAddress"
           :numerical-content-lease-content {:content content}))
  (when-not (content-placement? placement)
    (fail! "local content lease requires a ContentPlacement"
           :numerical-content-lease-placement {:placement placement}))
  (when-not (= content (:content placement))
    (fail! "local content lease and placement name different content"
           :numerical-content-lease-content-mismatch
           {:content content :placement-content (:content placement)}))
  (when-not (instance? MemorySegment segment)
    (fail! "local content lease requires a MemorySegment"
           :numerical-content-lease-segment {:actual (type segment)}))
  (when-not (and (integer? byte-offset) (not (neg? byte-offset)))
    (fail! "local content lease byte offset must be a non-negative integer"
           :numerical-content-lease-offset {:byte-offset byte-offset}))
  (when-not (and (integer? byte-length) (not (neg? byte-length)))
    (fail! "local content lease byte length must be a non-negative integer"
           :numerical-content-lease-length {:byte-length byte-length}))
  (when (> (+ byte-offset byte-length) (.byteSize ^MemorySegment segment))
    (fail! "local content lease range exceeds its MemorySegment"
           :numerical-content-lease-bounds
           {:byte-offset byte-offset :byte-length byte-length
            :segment-bytes (.byteSize ^MemorySegment segment)}))
  (when-not (ifn? release-fn)
    (fail! "local content lease requires a release function"
           :numerical-content-lease-release {:release-fn release-fn}))
  (->LocalContentLease content placement segment byte-offset byte-length release-fn (atom false)))

(defn lease-closed?
  [lease]
  (when-not (local-content-lease? lease)
    (fail! "expected a LocalContentLease"
           :numerical-content-lease-type {:actual (type lease)}))
  @(:closed-state lease))

(defn lease-segment
  "Return the live, range-limited MemorySegment owned by a LocalContentLease."
  [lease]
  (when-not (local-content-lease? lease)
    (fail! "expected a LocalContentLease"
           :numerical-content-lease-type {:actual (type lease)}))
  (when (lease-closed? lease)
    (fail! "local content lease is closed"
           :numerical-content-lease-closed {:content (:content lease)}))
  (.asSlice ^MemorySegment (:segment lease) (long (:byte-offset lease))
            (long (:byte-length lease))))

(defn element-byte-reader
  "Adapt synchronous whole-element downloads to bounded arbitrary byte windows.
   download! receives [first-element element-count destination-segment] and must fill that
   segment before returning. Reads preserve raw bits, never decode or convert byte order.
   The returned reader accepts at most 64 KiB per call and is sequential, not thread-safe.
   The caller supplies the exact extent/element width and owns source lifetime, completion,
   immutability and representation evidence. This adapter establishes none of those facts."
  [byte-length element-bytes download!]
  (when-not (and (integer? byte-length) (<= 0 byte-length Long/MAX_VALUE)
                 (integer? element-bytes) (<= 1 element-bytes 65536)
                 (zero? (mod byte-length element-bytes)) (ifn? download!))
    (fail! "element byte reader requires a whole-element extent and synchronous downloader"
           :numerical-content-element-reader
           {:byte-length byte-length :element-bytes element-bytes}))
  (let [extent (long byte-length)
        width (long element-bytes)
        scratch (MemorySegment/ofArray (byte-array (+ 65536 (* 2 (dec width)))))]
    (fn [offset destination]
      (when-not (and (integer? offset) (<= 0 offset extent)
                     (instance? MemorySegment destination))
        (fail! "element byte read requires a valid offset and destination segment"
               :numerical-content-element-read-range {:offset offset}))
      (let [offset (long offset)
            n (.byteSize ^MemorySegment destination)]
        ;; Subtraction avoids overflowing offset + n before rejection.
        (when-not (and (<= n 65536) (<= n (- extent offset)))
          (fail! "element byte read exceeds its extent or staging window"
                 :numerical-content-element-read-range {:offset offset :bytes n :extent extent}))
        (when (pos? n)
          (let [end (+ (long offset) n)
                start (- offset (mod offset width))
                aligned-end (+ end (mod (- width (mod end width)) width))
                span (- aligned-end start)]
            (download! (quot start width) (quot span width) (.asSlice scratch 0 span))
            (MemorySegment/copy scratch (- offset start) destination 0 n)))
        n))))

(defn content-address-from-reader
  "Hash exactly byte-length stored bytes through one bounded 64-KiB staging segment.
   read! receives [byte-offset destination-segment], synchronously fills that entire segment,
   and returns its byte count. The caller owns source lifetime, synchronization and immutability;
   this hashes supplied bytes, not authority/provenance or decoded numerical values."
  [byte-length read!]
  (when-not (and (integer? byte-length) (<= 0 byte-length Long/MAX_VALUE) (ifn? read!))
    (fail! "stream content hashing requires a bounded byte extent and callable reader"
           :numerical-content-hash-reader {:byte-length byte-length}))
  (let [digest (MessageDigest/getInstance "SHA-256")
        scratch (byte-array 65536)
        scratch-segment (MemorySegment/ofArray scratch)
        total (long byte-length)]
    (loop [offset 0]
      (when (< offset total)
        (let [n (int (min (long (alength scratch)) (- total offset)))
              actual (read! offset (.asSlice scratch-segment 0 n))]
          (when-not (= n actual)
            (fail! "stream content reader did not complete the requested range"
                   :numerical-content-hash-short-read {:offset offset :expected n :actual actual}))
          (.update digest scratch 0 n)
          (recur (+ offset n)))))
    (numerical-state/content-address :sha-256
                                     (.formatHex (HexFormat/of) (.digest digest)))))

(defn content-address-of
  "Hash stored segment bytes without materializing the whole segment on the Java heap.
   The fixed staging bound also supports mapped chunks beyond ByteBuffer's int-indexed limit."
  [^MemorySegment segment]
  (when-not (instance? MemorySegment segment)
    (fail! "content hashing requires a MemorySegment"
           :numerical-content-hash-segment {:actual (type segment)}))
  (content-address-from-reader
   (.byteSize segment)
   (fn [offset ^MemorySegment destination]
     (let [n (.byteSize destination)]
       (MemorySegment/copy segment offset destination 0 n)
       n))))

(defn verify-chunk-lease!
  "Verify a localized chunk's exact stored extent and SHA-256 before restore or device upload.

   Structural manifest certification alone cannot attest bytes returned by a storage provider.
   The caller retains ownership of `lease` on success and failure and must close it."
  [chunk lease]
  (when-not (numerical-state/state-chunk? chunk)
    (fail! "chunk verification requires a StateChunk"
           :numerical-content-chunk-type {:actual (type chunk)}))
  (numerical-state/chunk chunk)
  (let [segment (lease-segment lease)
        expected (:content chunk)]
    (when-not (= expected (:content lease))
      (fail! "local lease names a different chunk"
             :numerical-content-chunk-identity
             {:chunk (:id chunk) :expected expected :actual (:content lease)}))
    (when-not (= (:stored-byte-length chunk) (.byteSize ^MemorySegment segment))
      (fail! "localized chunk extent differs from its manifest"
             :numerical-content-chunk-extent
             {:chunk (:id chunk) :expected (:stored-byte-length chunk)
              :actual (.byteSize ^MemorySegment segment)}))
    (when-not (= :sha-256 (:algorithm expected))
      (fail! "chunk content-address algorithm has no runtime verifier"
             :numerical-content-chunk-algorithm
             {:chunk (:id chunk) :algorithm (:algorithm expected)}))
    (let [actual (content-address-of segment)]
      (when-not (= expected actual)
        (fail! "localized chunk bytes differ from their content address"
               :numerical-content-chunk-digest
               {:chunk (:id chunk) :expected expected :actual actual}))))
  lease)

(defn decode-raw-array-chunk!
  "Verify a leased :raw-array chunk and copy its exact element bits into destination.

   element-dtype comes from the independently checked target field, not inferred payload bytes.
   destination-byte-order is explicit; this does not attest a GPU's representation or perform an
   upload. Floating NaNs, signed zeros and half payloads are never converted through host numbers.
   All format/extent/alias checks and SHA-256 verification precede writes. Endian conversion uses
   bounded 64-KiB staging. The caller owns the non-overlapping destination and retains the source
   lease, which must remain immutable throughout verification and copying. A concurrent close or
   write failure is not transactional rollback, and yields no successful decode result."
  [chunk lease element-dtype ^MemorySegment destination destination-byte-order]
  (numerical-state/chunk chunk)
  (let [width (dtype/bytes-of element-dtype)
        bytes (reduce *' width (:shape chunk))
        source (lease-segment lease)]
    (when-not (= {:format :raw-array :byte-order (get-in chunk [:storage :byte-order])}
                 (:storage chunk))
      (fail! "raw-array decoding requires an unencoded raw-array storage contract"
             :numerical-content-codec-format {:storage (:storage chunk)}))
    (when-not (contains? numerical-state/byte-orders destination-byte-order)
      (fail! "raw-array decoding requires explicit destination byte order"
             :numerical-content-codec-byte-order {:byte-order destination-byte-order}))
    (when-not (and (<= bytes Long/MAX_VALUE)
                   (= bytes (:logical-byte-length chunk) (:stored-byte-length chunk)))
      (fail! "raw-array byte extent disagrees with shape and element dtype"
             :numerical-content-codec-extent
             {:shape (:shape chunk) :dtype element-dtype :expected bytes
              :logical (:logical-byte-length chunk) :stored (:stored-byte-length chunk)}))
    (when-not (and (instance? MemorySegment destination)
                   (= bytes (.byteSize destination))
                   (not (.isReadOnly destination))
                   (.isAlive (.scope destination))
                   (.isAccessibleBy destination (Thread/currentThread)))
      (fail! "raw-array destination must be a live writable exact-extent segment"
             :numerical-content-codec-destination {:expected bytes}))
    (when (.isPresent (.asOverlappingSlice source destination))
      (fail! "raw-array decoding cannot mutate its leased immutable source"
             :numerical-content-codec-alias {}))
    (verify-chunk-lease! chunk lease)
    (if (or (= 1 width) (= destination-byte-order (get-in chunk [:storage :byte-order])))
      (MemorySegment/copy source 0 destination 0 (long bytes))
      (let [scratch (byte-array 65536)
            staging (MemorySegment/ofArray scratch)]
        (loop [offset 0]
          (when (< offset bytes)
            (let [n (long (min 65536 (- bytes offset)))]
              (MemorySegment/copy source offset staging 0 n)
              (doseq [base (range 0 n width)
                      lane (range (quot width 2))]
                (let [left (+ base lane) right (+ base (- width 1 lane))
                      value (aget scratch left)]
                  (aset-byte scratch left (aget scratch right))
                  (aset-byte scratch right value)))
              (MemorySegment/copy staging 0 destination offset n)
              (recur (+ offset n)))))))
    destination))

(defn provider-descriptor
  [provider]
  (when-not (satisfies? ContentProvider provider)
    (fail! "value does not implement ContentProvider"
           :numerical-content-provider-type {:actual (type provider)}))
  (let [description (-provider-descriptor provider)]
    (when-not (provider-descriptor? description)
      (fail! "ContentProvider returned a non-descriptor"
             :numerical-content-provider-descriptor-type
             {:actual (type description)}))
    ;; Reconstruction revalidates mutated records and their tiers.
    (doseq [tier (:tiers description)]
      (storage-tier tier))
    (provider-description description)))

(defn- require-capability!
  [description capability]
  (when-not (contains? (:capabilities description) capability)
    (fail! "content provider lacks a required capability"
           :numerical-content-provider-capability
           {:provider (:id description) :required capability
            :available (:capabilities description)})))

(defn- tier-by-id
  [description tier-id]
  (or (some #(when (= tier-id (:id %)) %) (:tiers description))
      (fail! "content provider does not declare the requested tier"
             :numerical-content-provider-tier
             {:provider (:id description) :tier tier-id
              :available (mapv :id (:tiers description))})))

(defn- validate-provider-event!
  [description operation event]
  (when-not (storage-event? event)
    (fail! "ContentProvider returned a non-StorageEvent"
           :numerical-content-event-type {:actual (type event)}))
  (storage-event event)
  (when-not (= (:id description) (:provider-id event))
    (fail! "storage event belongs to a different content provider"
           :numerical-content-event-provider-mismatch
           {:expected (:id description) :actual (:provider-id event)}))
  (when-not (= operation (:operation event))
    (fail! "storage event operation differs from its submission"
           :numerical-content-event-operation-mismatch
           {:expected operation :actual (:operation event)}))
  event)

(defn- accept-provider-event! [provider description operation event]
  (try
    (validate-provider-event! description operation event)
    (catch Throwable error
      (if (storage-event? event)
        ;; The originating provider owns a rejected handoff even if its provider-id is wrong.
        (release-after-error! error #(-release-storage-event! provider event))
        (throw error)))))

(defn submit-ingestion!
  "Ingest exact immutable source bytes through bounded provider-owned windows.

   read! receives [offset destination] and returns the bytes filled. The source must remain live
   and immutable throughout this call; no source borrow survives submission. Consumption must be
   synchronous, ordered, complete and on this thread. The wrapper verifies SHA-256, expires the
   reader on every exit, and safely drains rejected event handoffs. An accepted event describes
   provider placement work, not source ownership. Stored bytes are independently verified by the
   ordinary localization/publication path; hashing source windows alone does not attest storage."
  ([provider content target-tier byte-length read!]
   (submit-ingestion! provider content target-tier byte-length read! {}))
  ([provider content target-tier byte-length read! opts]
   (let [description (provider-descriptor provider)]
     (when-not (and (numerical-state/content-address? content) (= :sha-256 (:algorithm content)))
       (fail! "content ingestion requires a SHA-256 content address"
              :numerical-content-ingestion-content {:content content}))
     (when-not (and (integer? byte-length) (<= 0 byte-length Long/MAX_VALUE) (ifn? read!))
       (fail! "content ingestion requires a bounded byte extent and reader"
              :numerical-content-ingestion-reader {:byte-length byte-length}))
     (when-not (map? opts)
       (fail! "content ingestion options must be a map"
              :numerical-content-ingestion-options {:opts opts}))
     (require-capability! description :ingest)
     (when-not (satisfies? ContentIngestor provider)
       (fail! "ingesting provider must implement ContentIngestor"
              :numerical-content-ingestion-provider {}))
     (tier-by-id description target-tier)
     (let [thread (Thread/currentThread)
           active? (volatile! true)
           source-reader (volatile! read!)
           reading? (volatile! false)
           offset (volatile! 0)
           fault (atom nil)
           digest (MessageDigest/getInstance "SHA-256")
           empty-address (when (zero? byte-length)
                           (numerical-state/content-address :sha-256
                                                            (.formatHex (HexFormat/of) (.digest digest))))
           _ (when (and empty-address (not= content empty-address))
               (fail! "empty ingestion source differs from its content address"
                      :numerical-content-ingestion-digest {:expected content :actual empty-address}))
           verified? (volatile! (zero? byte-length))
           consume! (fn [position destination]
                      (when-not @active?
                        (fail! "ingestion reader has expired" :numerical-content-ingestion-expired {}))
                      (try
                        (when-not (identical? thread (Thread/currentThread))
                          (fail! "ingestion must consume source on the submitting thread"
                                 :numerical-content-ingestion-thread {}))
                        (when-let [error @fault] (throw error))
                        (when @reading?
                          (fail! "ingestion reader cannot reenter itself"
                                 :numerical-content-ingestion-reentrant {}))
                        (when-not (and (integer? position) (= position @offset))
                          (fail! "ingestion windows must be contiguous and ordered"
                                 :numerical-content-ingestion-order {:expected @offset :actual position}))
                        (when-not (and (instance? MemorySegment destination)
                                       (.isAlive (.scope ^MemorySegment destination))
                                       (.isAccessibleBy ^MemorySegment destination thread)
                                       (not (.isReadOnly ^MemorySegment destination))
                                       (<= 1 (.byteSize ^MemorySegment destination) 65536)
                                       (<= (.byteSize ^MemorySegment destination) (- byte-length @offset)))
                          (fail! "ingestion destination must be a live writable bounded window"
                                 :numerical-content-ingestion-window {:offset @offset}))
                        (let [n (.byteSize ^MemorySegment destination)
                              actual (do (vreset! reading? true)
                                         (try (@source-reader position destination)
                                              (finally (vreset! reading? false))))]
                          (when-let [error @fault] (throw error))
                          (when-not (= n actual)
                            (fail! "ingestion reader did not fill its window"
                                   :numerical-content-ingestion-short-read
                                   {:offset position :expected n :actual actual}))
                          (.update digest (.asByteBuffer ^MemorySegment destination))
                          (vreset! offset (Math/addExact (long @offset) (long n)))
                          (when (= byte-length @offset)
                            (let [actual-address (numerical-state/content-address :sha-256
                                                                                  (.formatHex (HexFormat/of) (.digest digest)))]
                              (when-not (= content actual-address)
                                (fail! "ingested source bytes differ from their content address"
                                       :numerical-content-ingestion-digest
                                       {:expected content :actual actual-address}))
                              (vreset! verified? true)))
                          n)
                        (catch Throwable error
                          (compare-and-set! fault nil error)
                          (throw error))))
           submission (try {:event (-submit-ingestion! provider content target-tier byte-length consume! opts)}
                           (catch Throwable error {:error error})
                           (finally (vreset! active? false) (vreset! source-reader nil)))
           read-failure @fault
           ;; A provider-retained expired callback must not retain source objects via error data.
           _ (reset! fault nil)
           event (:event submission)
           _ (when-let [error (:error submission)]
               (when (and read-failure (not (identical? error read-failure))
                          (not-any? #(identical? error %) (.getSuppressed ^Throwable read-failure)))
                 (.addSuppressed ^Throwable read-failure error))
               (throw (or read-failure error)))]
       (try
         (when read-failure (throw read-failure))
         (when-not (= byte-length @offset)
           (fail! "provider did not consume the complete source"
                  :numerical-content-ingestion-incomplete {:expected byte-length :actual @offset}))
         (when-not @verified?
           (fail! "ingestion stream has no completed digest verification"
                  :numerical-content-ingestion-digest {}))
         (validate-provider-event! description :ingest event)
         (catch Throwable error
           (if (storage-event? event)
             (release-after-error! error #(-release-storage-event! provider event))
             (throw error))))))))

(defn submit-promotion!
  "Submit promotion of immutable content to a declared durable tier."
  ([provider content target-tier] (submit-promotion! provider content target-tier {}))
  ([provider content target-tier opts]
   (let [description (provider-descriptor provider)]
     (when-not (numerical-state/content-address? content)
       (fail! "content promotion requires an immutable ContentAddress"
              :numerical-content-promotion-content {:content content}))
     (when-not (map? opts)
       (fail! "content promotion options must be a map"
              :numerical-content-promotion-options {:opts opts}))
     (require-capability! description :promote)
     (let [tier (tier-by-id description target-tier)]
       (when-not (contains? (:capabilities tier) :durable-receipt)
         (fail! "promotion target does not promise a durable receipt"
                :numerical-content-promotion-durability
                {:provider (:id description) :tier target-tier
                 :capabilities (:capabilities tier)})))
     (accept-provider-event!
      provider description :promote (-submit-promotion! provider content target-tier opts)))))

(defn submit-localization!
  "Submit localization of immutable content into a provider tier that can later be opened."
  ([provider content] (submit-localization! provider content {}))
  ([provider content opts]
   (let [description (provider-descriptor provider)]
     (when-not (numerical-state/content-address? content)
       (fail! "content localization requires an immutable ContentAddress"
              :numerical-content-localization-content {:content content}))
     (when-not (map? opts)
       (fail! "content localization options must be a map"
              :numerical-content-localization-options {:opts opts}))
     (require-capability! description :localize)
     (when-let [tier-id (:tier opts)]
       (tier-by-id description tier-id))
     (accept-provider-event!
      provider description :localize (-submit-localization! provider content opts)))))

(defn- checked-event
  [provider event]
  (validate-provider-event! (provider-descriptor provider) (:operation event) event))

(defn storage-event-complete?
  [provider event]
  (-storage-event-complete? provider (checked-event provider event)))

(defn await-storage-event!
  [provider event]
  (-await-storage-event! provider (checked-event provider event)))

(defn storage-event-measurement
  [provider event]
  (-storage-event-measurement provider (checked-event provider event)))

(defn release-storage-event!
  "Settle provider work and release its event, with or without any preceding await.
   A failure forbids consuming the result; opaque resource ownership remains with the provider."
  [provider event]
  (-release-storage-event! provider (checked-event provider event))
  nil)

(defn open-local-content!
  "Open already-localized content as a scoped MemorySegment lease.
   A returned provider lease is released if validation rejects the handoff. Cleanup failures are
   suppressed onto the original validation error rather than replacing its diagnostic."
  ([provider content] (open-local-content! provider content {}))
  ([provider content opts]
   (let [description (provider-descriptor provider)]
     (when-not (numerical-state/content-address? content)
       (fail! "opening local content requires an immutable ContentAddress"
              :numerical-content-open-content {:content content}))
     (when-not (map? opts)
       (fail! "local content options must be a map"
              :numerical-content-open-options {:opts opts}))
     (require-capability! description :scoped-segment)
     (let [lease (-open-local-content! provider content opts)]
       (when-not (local-content-lease? lease)
         (fail! "ContentProvider returned a non-LocalContentLease"
                :numerical-content-lease-type {:actual (type lease)}))
       (try
         (when-not (= content (:content lease))
           (fail! "ContentProvider opened a lease for different content"
                  :numerical-content-open-mismatch
                  {:expected content :actual (:content lease)}))
         (let [placement (:placement lease)]
           (when-not (= (:id description) (:provider-id placement))
             (fail! "local content placement belongs to a different provider"
                    :numerical-content-placement-provider-mismatch
                    {:expected (:id description) :actual (:provider-id placement)}))
           (tier-by-id description (:tier-id placement)))
         ;; Access once so a malformed or already-closed lease fails before ownership is returned.
         (lease-segment lease)
         lease
         (catch Throwable error
           (release-after-error! error #(.close ^AutoCloseable lease))))))))

(defn with-local-content
  "Open localized content, call `f` and release its lease once, preserving primary failures."
  ([provider content f] (with-local-content provider content {} f))
  ([provider content opts f]
   (when-not (ifn? f)
     (fail! "with-local-content requires a callback"
            :numerical-content-callback {:callback f}))
   (let [lease (open-local-content! provider content opts)]
     (with-release* #(f lease) #(.close ^AutoCloseable lease)))))

(defn- checked-placement! [description content tier placement]
  (when-not (content-placement? placement)
    (fail! "storage completion must return a ContentPlacement"
           :numerical-content-completion-placement {:actual (type placement)}))
  (content-placement placement)
  (when-not (= [(:id description) tier content]
               [(:provider-id placement) (:tier-id placement) (:content placement)])
    (fail! "storage completion differs from its requested placement"
           :numerical-content-completion-mismatch
           {:expected {:provider-id (:id description) :tier-id tier :content content}
            :actual placement}))
  placement)

(defn- await-placement! [provider description content tier event]
  (with-release*
    #(checked-placement! description content tier (await-storage-event! provider event))
    ;; This scope owns the exact accepted handoff. Descriptor drift must not prevent safe drain.
    #(-release-storage-event! provider event)))

(defn ingest-content!
  "Consume bounded source bytes and settle the exact requested placement before returning.
   No source borrow survives submission; provider work is safely drained on success/failure.
   Placement durability is only the selected tier's declared contract, not producer provenance."
  ([provider content target-tier byte-length read!]
   (ingest-content! provider content target-tier byte-length read! {}))
  ([provider content target-tier byte-length read! opts]
   (let [description (provider-descriptor provider)
         event (submit-ingestion! provider content target-tier byte-length read! opts)]
     (await-placement! provider description content target-tier event))))

(defn finalize-state-availability!
  "Verify/promote a certified state's blobs before invoking publish-manifest! with its manifest.

   Preflights all capabilities/digest algorithms before submitting work. Localizes through the
   first declared scoped-segment tier and serially verifies exact stored extents and SHA-256 under
   a source lease. The lease closes before content-addressed promotion. Every completion must name
   the requested provider, tier and content, and every event drains before the next stage. Runtime
   placements are returned in manifest field/chunk order, never embedded into the compiler state.

   This finalizes availability according to trusted immutable-content/durable-receipt provider
   contracts, not producer authentication, codec semantics or a storage transaction. The callback
   owns metadata atomicity, parent existence and retries. Failure can leave orphaned promoted blobs;
   callback failure or lost acknowledgment cannot roll metadata back. No store/cache is introduced."
  [provider certified-state target-tier publish-manifest!]
  (numerical-state/verify! certified-state)
  (when-not (ifn? publish-manifest!)
    (fail! "state availability finalization requires a metadata publication callback"
           :numerical-content-publication-callback {}))
  (let [description (provider-descriptor provider)
        _ (doseq [capability [:localize :scoped-segment :promote]]
            (require-capability! description capability))
        local-tier (or (some #(when (contains? (:capabilities %) :scoped-segment) (:id %))
                            (:tiers description))
                       (fail! "state availability requires an openable localization tier"
                              :numerical-content-localization-tier {}))
        target (tier-by-id description target-tier)
        _ (when-not (contains? (:capabilities target) :durable-receipt)
            (fail! "publication target does not promise a durable receipt"
                   :numerical-content-promotion-durability {:tier target-tier}))
        chunks (vec (for [field (get-in certified-state [:manifest :fields])
                          chunk (:chunks field)]
                      {:field-id (:id field) :chunk chunk}))
        _ (doseq [{:keys [chunk]} chunks]
            (when-not (= :sha-256 (get-in chunk [:content :algorithm]))
              (fail! "state publication has an unsupported content digest algorithm"
                     :numerical-content-chunk-algorithm
                     {:chunk (:id chunk) :algorithm (get-in chunk [:content :algorithm])})))
        placements
        (mapv (fn [{:keys [field-id chunk]}]
                (let [address (:content chunk)]
                  (await-placement! provider description address local-tier
                                    (submit-localization! provider address {:tier local-tier}))
                  (with-local-content provider address {:tier local-tier}
                    (fn [lease]
                      (checked-placement! description address local-tier (:placement lease))
                      (verify-chunk-lease! chunk lease)))
                  {:field-id field-id :chunk-id (:id chunk)
                   :placement (await-placement!
                               provider description address target-tier
                               (submit-promotion! provider address target-tier))}))
              chunks)]
    {:certified-state certified-state :placements placements
     :publication (publish-manifest! (:manifest certified-state))}))
