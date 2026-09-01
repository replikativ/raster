(ns raster.runtime.numerical-content-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.buffer-view :as buffer-view]
            [raster.compiler.ir.numerical-state :as numerical-state]
            [raster.gpu.core :as gpu]
            [raster.runtime.numerical-content :as content])
  (:import [java.lang AutoCloseable]
           [java.lang.foreign MemorySegment ValueLayout]))

(defn- address
  [n]
  (numerical-state/content-address :sha-256 (format "%064x" n)))

(defn- fake-provider
  [provider-id]
  (let [events (atom {})
        releases (atom 0)
        local-tier (content/storage-tier
                    {:id :local
                     :kind :file
                     :locality :node
                     :durability :cached
                     :capabilities #{:scoped-segment}})
        durable-tier (content/storage-tier
                      {:id :durable
                       :kind :object-store
                       :locality :site
                       :durability :durable
                       :capabilities #{:durable-receipt :range-read :multipart-write}})
        description (content/provider-description
                     {:id provider-id
                      :tiers [local-tier durable-tier]
                      :capabilities #{:promote :localize :scoped-segment}
                      :attributes {:implementation :fake}})
        submit (fn [operation value]
                 (let [event (content/storage-event
                              {:provider-id provider-id
                               :id (random-uuid)
                               :operation operation})]
                   (swap! events assoc (:id event)
                          {:status :pending
                           :value value
                           :measurement {:timing-source :fake
                                         :bytes 16
                                         :elapsed-ns 20}})
                   event))
        provider
        (reify content/ContentProvider
          (-provider-descriptor [_] description)
          (-submit-promotion! [_ chunk target-tier _opts]
            (submit :promote
                    (content/content-placement
                     {:provider-id provider-id :tier-id target-tier :content chunk})))
          (-submit-localization! [_ chunk opts]
            (submit :localize
                    (content/content-placement
                     {:provider-id provider-id :tier-id (or (:tier opts) :local)
                      :content chunk})))
          (-open-local-content! [_ chunk _opts]
            (let [bytes (byte-array (map byte (range 16)))
                  placement (content/content-placement
                             {:provider-id provider-id :tier-id :local :content chunk})]
              (content/local-content-lease
               {:content chunk
                :placement placement
                :segment (MemorySegment/ofArray bytes)
                :byte-offset 4
                :byte-length 8
                :release-fn #(swap! releases inc)})))
          (-storage-event-complete? [_ event]
            (= :complete (get-in @events [(:id event) :status])))
          (-await-storage-event! [_ event]
            (swap! events assoc-in [(:id event) :status] :complete)
            (get-in @events [(:id event) :value]))
          (-storage-event-measurement [_ event]
            (when (= :complete (get-in @events [(:id event) :status]))
              (get-in @events [(:id event) :measurement])))
          (-release-storage-event! [_ event]
            (swap! events dissoc (:id event))))]
    {:provider provider :events events :releases releases :description description}))

(deftest provider-capabilities-govern-promotion-and-localization
  (let [{:keys [provider events description]} (fake-provider :tiered-store)
        chunk (address 1)
        promotion (content/submit-promotion! provider chunk :durable)
        localization (content/submit-localization! provider chunk {:tier :local})]
    (is (= description (content/provider-descriptor provider)))
    (is (= :storage (get-in promotion [:queue :class])))
    (is (= :promote (:operation promotion)))
    (is (= :localize (:operation localization)))
    (is (false? (content/storage-event-complete? provider promotion)))
    (is (= :durable (:tier-id (content/await-storage-event! provider promotion))))
    (is (content/storage-event-complete? provider promotion))
    (is (= {:timing-source :fake :bytes 16 :elapsed-ns 20}
           (content/storage-event-measurement provider promotion)))
    (is (= :local (:tier-id (content/await-storage-event! provider localization))))
    (content/release-storage-event! provider promotion)
    (content/release-storage-event! provider localization)
    (is (empty? @events))))

(deftest events-and-target-tiers-cannot-cross-provider-boundaries
  (let [{left :provider} (fake-provider :left)
        {right :provider} (fake-provider :right)
        event (content/submit-localization! left (address 2))]
    (is (= :numerical-content-event-provider-mismatch
           (:reason
            (try
              (content/await-storage-event! right event)
              nil
              (catch clojure.lang.ExceptionInfo error
                (ex-data error))))))
    (is (= :numerical-content-provider-tier
           (:reason
            (try
              (content/submit-promotion! left (address 2) :unknown)
              nil
              (catch clojure.lang.ExceptionInfo error
                (ex-data error))))))
    (content/release-storage-event! left event)))

(deftest local-content-is-scoped-and-range-limited
  (let [{:keys [provider releases]} (fake-provider :local-provider)
        chunk (address 3)
        observed
        (content/with-local-content
          provider chunk
          (fn [lease]
            (let [segment (content/lease-segment lease)]
              (is (= 8 (.byteSize ^MemorySegment segment)))
              (is (= 4 (.get ^MemorySegment segment ValueLayout/JAVA_BYTE 0)))
              lease)))]
    (is (content/lease-closed? observed))
    (is (= 1 @releases))
    (.close ^AutoCloseable observed)
    (is (= 1 @releases) "lease release is idempotent")
    (is (= :numerical-content-lease-closed
           (:reason
            (try
              (content/lease-segment observed)
              nil
              (catch clojure.lang.ExceptionInfo error
                (ex-data error))))))))

(defn- fake-transfer-session
  []
  (let [buffer {:dtype :byte :n-elements 8 :byte-size 8}
        allocation (buffer-view/allocation
                    {:id :resident-allocation
                     :byte-size 8
                     :memory-space :device
                     :device :ze:0
                     :coherence :host-coherent
                     :ownership :owned})]
    (atom {:device-id :ze:0
           :session-id :retained-transfer-session
           :buffers {:buffer buffer}
           :allocations {:buffer allocation}
           :kernel-graphs {}
           :events {}
           :closed? false})))

(deftest polling-and-cancellation-release-local-content-only-at-the-transfer-boundary
  (let [{:keys [provider releases]} (fake-provider :transfer-provider)
        lease (content/open-local-content! provider (address 4))
        session (fake-transfer-session)
        backend-token ::backend-transfer
        backend-complete? (atom false)
        resolver
        (fn [_ name]
          (case name
            "plan-range" (fn [_ host spec direction]
                           {:host-segment host :spec spec :direction direction :n-bytes 8})
            "submit-range-batch!" (fn [_ direction]
                                    (is (= :upload direction))
                                    backend-token)
            "await-event!" (fn [token]
                             (is (= backend-token token))
                             {:timing-source :device-event
                              :elapsed-ns 10 :bytes 8 :commands 1
                              :direction :upload :asynchronous? true})
            "release-event!" (fn [token] (is (= backend-token token)))
            "event-complete?" (fn [token]
                                (is (= backend-token token))
                                @backend-complete?)
            (throw (ex-info "unexpected mocked runtime function" {:name name}))))]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve) resolver}
      (fn []
        (let [event (gpu/submit-upload-ranges-retained!
                     session
                     [[:buffer (content/lease-segment lease) {:elements 8}]]
                     [lease])]
          (is (false? (content/lease-closed? lease)))
          (is (= 0 @releases))
          (is (false? (gpu/event-complete? session event)))
          ;; A cancellation request may stop scheduling new work, but it cannot close the mapped
          ;; arena. Even a positive device poll is only an observation: release-event! establishes
          ;; the host-visible transfer boundary and consumes both event and lease ownership.
          (reset! backend-complete? true)
          (is (gpu/event-complete? session event))
          (is (false? (content/lease-closed? lease)))
          (gpu/release-event! session event)
          (is (content/lease-closed? lease))
          (is (= 1 @releases))
          (is (empty? (:events @session))))))))

(deftest failed-transfer-submission-does-not-take-lease-ownership
  (let [{:keys [provider releases]} (fake-provider :failing-provider)
        lease (content/open-local-content! provider (address 5))
        session (fake-transfer-session)]
    (with-redefs-fn
      {(ns-resolve 'raster.gpu.core 'rt-resolve)
       (fn [_ name]
         (case name
           "plan-range" (fn [& _] (throw (ex-info "invalid range" {})))
           (throw (ex-info "unexpected mocked runtime function" {:name name}))))}
      (fn []
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"invalid range"
                              (gpu/submit-upload-ranges-retained!
                               session
                               [[:buffer (content/lease-segment lease) {:elements 8}]]
                               [lease])))
        (is (false? (content/lease-closed? lease)))
        (is (= 0 @releases))))
    (.close ^AutoCloseable lease)
    (is (= 1 @releases))))
