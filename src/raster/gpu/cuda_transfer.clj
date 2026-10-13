(ns raster.gpu.cuda-transfer
  "Pinned host staging and true asynchronous CUDA ranges on the existing stream.
   Polling observes device status; only awaiting/releasing publishes downloads."
  (:require [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root])
  (:import [java.lang.foreign MemorySegment]))

(defn- validate-batch! [entries direction validate-range-plan!]
  (when-not (and (vector? entries) (contains? #{:upload :download} direction)
                 (every? #(and (vector? %) (= 2 (count %))) entries))
    (throw (ex-info "CUDA transfer batch requires ordered buffer/plan pairs"
                    {:reason :cuda-transfer-batch :direction direction})))
  (mapv
   (fn [[buffer plan]] [buffer (validate-range-plan! buffer plan direction)]) entries))

(defn submit-range-batch! [state entries direction {:keys [adopt-cleanup!]} native! assert-buffer-live! validate-range-plan!]
  (let [checked (validate-batch! entries direction validate-range-plan!)
        active (filterv #(pos? (get-in % [1 :n-bytes])) checked)
        total-bytes (reduce +' 0 (map #(get-in % [1 :n-bytes]) checked))]
    (if (empty? active)
      {:transfer? true :complete? true
       :completion {:timing-source :host-monotonic :elapsed-ns 0 :bytes total-bytes :commands 0
                    :direction direction :asynchronous? false}}
      (let [stream (:stream @state) started? (atom false) completed? (atom false)
            completion-error (atom nil)
            committed? (atom false) published? (atom false) publish-error (atom nil)
            end-recorded? (atom false) copies (atom []) operation-lock (Object.)
            buffers (vec (distinct (map first active)))
            buffer-slots (mapv (fn [_] (cleanup/acquisition-slot)) buffers)
            root-slot (cleanup/acquisition-slot)
            start-slot (cleanup/acquisition-slot) end-slot (cleanup/acquisition-slot)
            staging-slots (mapv (fn [_] (cleanup/acquisition-slot)) active)
            wait! (fn []
                    (locking operation-lock
                      (when (and @started? (not @completed?))
                      ;; Direct await and canonical cleanup share the same outcome.
                      ;; An uncertain wait cannot become a fresh native attempt when
                      ;; the session follows an operation failure with token release.
                        (when @completion-error (throw @completion-error))
                        (try
                          (if @end-recorded?
                            (native! :event-sync [(:resource @end-slot)])
                          ;; An unknown enqueue/record outcome still retains every source/destination.
                            (native! :stream-sync [stream]))
                          (reset! completed? true)
                          (catch Throwable error
                            (when-not (true? (:cleanup-retry-safe? (ex-data error)))
                              (reset! completion-error error))
                            (throw error))))))
            publish! (fn []
                       (locking operation-lock
                         (when (and @committed? (not @published?))
                           (when @publish-error (throw @publish-error))
                           (try
                             (when (= :download direction)
                               (doseq [{:keys [staging host-seg host-off n-bytes]} @copies]
                                 (MemorySegment/copy staging (long 0) host-seg (long host-off) (long n-bytes))))
                             (reset! published? true)
                             (catch Throwable error (reset! publish-error error) (throw error))))))
            resources
            (into [{:id :completion :release wait!}
                   {:id :visibility :after #{:completion}
                    ;; Host publication is an operation outcome, not native destruction.
                    ;; Once completion is known, a closed destination must not retain
                    ;; staging/events forever. Await/release report the stored error.
                    :release #(try (publish!) (catch Throwable _ nil))}
                   {:id :start-event :after #{:visibility}
                    :release #(cleanup/release-native! start-slot (fn [e] (native! :event-destroy [e])))}
                   {:id :end-event :after #{:visibility}
                    :release #(cleanup/release-native! end-slot (fn [e] (native! :event-destroy [e])))}]
                  (concat
                   (map-indexed (fn [i slot]
                                  {:id [:pinned i] :after #{:start-event :end-event}
                                   :release #(cleanup/release-native! slot (fn [p] (native! :host-free [p])))}) staging-slots)
                   (map-indexed
                    (fn [i slot]
                      {:id [:buffer-use i]
                       :after (into #{:start-event :end-event} (map #(vector :pinned %) (range (count staging-slots))))
                       :release #(cleanup/release-native! slot (fn [lease] (cleanup/release! (::cleanup/owner lease))))})
                    buffer-slots)))
            owner (cleanup/owner
                   (conj resources
                         {:id :runtime-root-lease :after (set (map :id resources))
                          :release #(cleanup/release-native! root-slot
                                                             (fn [lease] (cleanup/release! (::cleanup/owner lease))))}))
            result
            (cleanup/build!
             owner
             (fn []
               (doseq [[slot buffer] (map vector buffer-slots buffers)]
                 (cleanup/acquire-native! slot #(cleanup/retain-use! (cleanup/lifetime-owner buffer))))
               ;; All buffer lifetime reservations precede root/queue locking.
               ;; Capture the same common root lease used by other native children.
               (root/capture-lease! state root-slot)
               (cleanup/with-registry-use state
                 (root/assert-live! state)
                 (doseq [buffer buffers] (assert-buffer-live! buffer))
               ;; Every upload is copied into private pinned memory before any enqueue.
                 (doseq [[i [buffer {:keys [buf-off host-off n-bytes host-seg]}]] (map-indexed vector active)]
                   (let [pointer (cleanup/acquire-native! (nth staging-slots i)
                                                          #(native! :host-allocate [n-bytes]))
                         staging (.reinterpret pointer (long n-bytes))]
                     (when (= :upload direction)
                       (MemorySegment/copy host-seg (long host-off) staging (long 0) (long n-bytes)))
                     (swap! copies conj {:buffer buffer :pointer (+' (:device-pointer buffer) buf-off)
                                         :staging staging :host-seg host-seg :host-off host-off :n-bytes n-bytes})))
                 (let [start (cleanup/acquire-native! start-slot #(native! :event-create []))
                       end (cleanup/acquire-native! end-slot #(native! :event-create []))]
                   (reset! started? true)
                   (native! :event-record [start stream])
                   (doseq [{:keys [pointer staging n-bytes]} @copies]
                     (native! (if (= :upload direction) :upload-async :download-async)
                              [pointer staging n-bytes stream]))
                   (native! :event-record [end stream])
                   (reset! end-recorded? true)
                   {:transfer? true :event end :start-event start :wait! wait! :publish! publish!
                    :publication-error publish-error
                    :operation-lock operation-lock
                    :bytes total-bytes :commands (count active) :direction direction})))
             adopt-cleanup!)]
        ;; No fallible publication follows this handoff. Canonical release can now publish
        ;; downloads; failed construction instead drains/discards staging without host writes.
        (reset! committed? true)
        result))))

(defn event-complete? [token native!]
  (if (:complete? token) true
      (cleanup/using! (::cleanup/owner token)
                      #(locking (:operation-lock token)
                         (boolean (native! :event-query [(:event token)]))))))

(defn await-event! [token native!]
  (if (:complete? token)
    (:completion token)
    (cleanup/using!
     (::cleanup/owner token)
     (fn []
       (locking (:operation-lock token)
         ((:wait! token))
         ((:publish! token))
         {:timing-source :device-event
          :elapsed-ns (long (native! :event-elapsed [(:start-event token) (:event token)]))
          :bytes (:bytes token) :commands (:commands token)
          :direction (:direction token) :asynchronous? true})))))

(defn release-event! [token]
  (when-not (:complete? token)
    (let [prior-error @(:publication-error token)
          retirement-error (try (cleanup/release! (::cleanup/owner token)) nil
                                (catch Throwable error error))
          publication-error @(:publication-error token)]
      (cond
        (and (nil? prior-error) publication-error)
        (do (when (and retirement-error (not (identical? publication-error retirement-error)))
              (.addSuppressed publication-error retirement-error))
            (throw publication-error))
        retirement-error (throw retirement-error))))
  nil)
