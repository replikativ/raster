(ns raster.runtime.numerical-content
  "Runtime realization of immutable numerical-state chunks.

   The compiler manifest names content, not storage products. A ContentProvider describes its
   tiers and capabilities, submits asynchronous promotion/localization operations, and opens a
   local realization as a scoped MemorySegment lease. Implementations may wrap Konserve file/S3
   tiering, LMDB transactions, Ceph, EOS, GPFS, Lustre, or an institutional archive.

   This namespace owns no store implementation. In particular, remote object storage is localized
   before `open-local-content!`; a remote object is never represented as a fictitious mmap."
  (:require [raster.compiler.ir.execution-plan :as execution]
            [raster.compiler.ir.numerical-state :as numerical-state])
  (:import [java.lang AutoCloseable]
           [java.lang.foreign MemorySegment]))

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
   or pretending storage and device queues are the same resource."
  (-provider-descriptor [provider])
  (-submit-promotion! [provider content target-tier opts])
  (-submit-localization! [provider content opts])
  (-open-local-content! [provider content opts])
  (-storage-event-complete? [provider event])
  (-await-storage-event! [provider event])
  (-storage-event-measurement [provider event])
  (-release-storage-event! [provider event]))

(defn storage-tier? [value] (instance? StorageTier value))
(defn provider-descriptor? [value] (instance? ContentProviderDescriptor value))
(defn content-placement? [value] (instance? ContentPlacement value))
(defn storage-event? [value] (instance? StorageEvent value))
(defn local-content-lease? [value] (instance? LocalContentLease value))

(defn- fail!
  [message reason data]
  (throw (ex-info message (assoc data :reason reason))))

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
  (when-not (= (:id description) (:provider-id event))
    (fail! "storage event belongs to a different content provider"
           :numerical-content-event-provider-mismatch
           {:expected (:id description) :actual (:provider-id event)}))
  (when-not (= operation (:operation event))
    (fail! "storage event operation differs from its submission"
           :numerical-content-event-operation-mismatch
           {:expected operation :actual (:operation event)}))
  event)

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
     (validate-provider-event!
      description :promote (-submit-promotion! provider content target-tier opts)))))

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
     (validate-provider-event!
      description :localize (-submit-localization! provider content opts)))))

(defn- checked-event
  [provider event]
  (let [description (provider-descriptor provider)]
    (when-not (storage-event? event)
      (fail! "storage event operation requires a StorageEvent"
             :numerical-content-event-type {:actual (type event)}))
    (when-not (= (:id description) (:provider-id event))
      (fail! "storage event belongs to a different content provider"
             :numerical-content-event-provider-mismatch
             {:expected (:id description) :actual (:provider-id event)}))
    event))

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
  [provider event]
  (-release-storage-event! provider (checked-event provider event))
  nil)

(defn open-local-content!
  "Open already-localized content as a scoped MemorySegment lease."
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
       (when-not (= content (:content lease))
         (.close ^AutoCloseable lease)
         (fail! "ContentProvider opened a lease for different content"
                :numerical-content-open-mismatch
                {:expected content :actual (:content lease)}))
       (let [placement (:placement lease)]
         (when-not (= (:id description) (:provider-id placement))
           (.close ^AutoCloseable lease)
           (fail! "local content placement belongs to a different provider"
                  :numerical-content-placement-provider-mismatch
                  {:expected (:id description) :actual (:provider-id placement)}))
         (tier-by-id description (:tier-id placement)))
       ;; Access once so a malformed or already-closed lease fails before ownership is returned.
       (lease-segment lease)
       lease))))

(defn with-local-content
  "Open localized content, call `f` with its lease, and release the provider resource exactly once."
  ([provider content f] (with-local-content provider content {} f))
  ([provider content opts f]
   (when-not (ifn? f)
     (fail! "with-local-content requires a callback"
            :numerical-content-callback {:callback f}))
   (let [lease (open-local-content! provider content opts)]
     (try
       (f lease)
       (finally
         (.close ^AutoCloseable lease))))))
