(ns raster.gpu.cuda-runtime
  "CUDA memory foundation over the common root/cleanup authorities. Public CUDA\n   sessions remain disabled until execution and device qualification land."
  (:refer-clojure :exclude [reset!])
  (:require [clojure.string :as string]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.buffer-view :as view]
            [raster.gpu.cuda-driver :as driver]
            [raster.gpu.cuda-prepared-execution :as prepared-execution]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.runtime-root :as root])
  (:import [java.lang.foreign MemorySegment]))

(def state (atom {}))

(defn include-paths-from-env
  "Pure compiler configuration policy. Explicit include-path options override this default."
  [env]
  (if (contains? env "RASTER_CUDA_INCLUDE_PATHS")
    (if (seq (get env "RASTER_CUDA_INCLUDE_PATHS"))
      (vec (string/split (get env "RASTER_CUDA_INCLUDE_PATHS")
                         (re-pattern (java.util.regex.Pattern/quote java.io.File/pathSeparator))))
      [])
    (if (seq (get env "CUDA_HOME")) [(str (get env "CUDA_HOME") "/include")] [])))

(defn- native! [operation arguments]
  (let [ledger (:prepared-execution @state)
        invoke #(binding [driver/*context* (:context @state)]
                  (driver/invoke! operation arguments))]
    (if (and ledger (contains? #{:stream-sync :event-sync} operation))
      ;; Graphs, transfers and prepared calls share the physical stream. A failed
      ;; completion cannot be retried through a different public adapter path.
      (prepared-execution/complete! ledger invoke)
      (if (and ledger (contains? #{:launch :copy :upload :download :upload-async :download-async
                                  :allocate :host-allocate :module-load :module-function
                                  :event-create :event-record :event-query :event-elapsed} operation))
        (locking (:lock ledger)
          (prepared-execution/assert-admitted! ledger)
          (invoke))
        ;; Destruction stays under its canonical completion dependencies; unrelated
        ;; resources whose use already completed may still retire after poison.
        (invoke)))))

(defn initialize!
  "Initialize the shared process root. Toolkit include paths are explicit configuration,
   never inferred from a compiler device descriptor or committed machine paths."
  ([] (initialize! {}))
  ([opts]
   (let [device-ordinal (get opts :device-ordinal 0)
         include-paths (get opts :include-paths (include-paths-from-env (System/getenv)))]
   (when (and (integer? device-ordinal) (not= 0 device-ordinal))
     (throw (ex-info "CUDA memory foundation supports only ordinal zero"
                     {:reason :cuda-runtime-device-unsupported :device-ordinal device-ordinal})))
   (when-not (and (integer? device-ordinal) (<= 0 device-ordinal Integer/MAX_VALUE)
                  (vector? include-paths) (every? string? include-paths))
     (throw (ex-info "Invalid CUDA runtime configuration" {:reason :cuda-runtime-configuration})))
   (locking state
     (when (and (:initialized? @state)
                (not= [device-ordinal include-paths] [(:device-ordinal @state) (:include-paths @state)]))
       (throw (ex-info "CUDA runtime already has a different device/compiler configuration"
                       {:reason :cuda-runtime-configuration-mismatch})))
     (root/initialize!
      state [{:id :runtime :release #(binding [driver/*context* (:context %)]
                                     (driver/invoke! :destroy-runtime [%]))}]
      (fn [entry]
        (assoc (root/acquire! entry :runtime #(driver/invoke! :initialize [device-ordinal]))
               :include-paths include-paths :device-ordinal device-ordinal
               :prepared-execution (prepared-execution/ledger state)))))
   nil)))

(defn ensure-init! []
  (when-not (:initialized? @state) (initialize!))
  (root/assert-live! state)
  (prepared-execution/assert-admitted! (:prepared-execution @state)))

(defn execution-device-info []
  (ensure-init!)
  (select-keys @state [:name :device-ordinal :compute-capability :driver-api-version
                      :max-workgroup-size :max-block-dimensions :max-grid-dimensions
                      :warp-size :shared-local-memory :multiprocessor-count
                      :free-memory-bytes :total-memory-bytes]))

(defn- owner! [value]
  (or (cleanup/lifetime-owner value)
      (throw (ex-info "CUDA value has no lifetime owner" {:reason :missing-cleanup-owner}))))

(defn device-buffer? [value] (true? (::buffer value)))

(defn assert-buffer-live! [buffer]
  (when-not (device-buffer? buffer)
    (throw (ex-info "CUDA requires a resident buffer" {:reason :cuda-buffer-type})))
  (cleanup/assert-live! (owner! buffer))
  (root/assert-live! state)
  (prepared-execution/assert-admitted! (:prepared-execution @state))
  (when-not (identical? (::context buffer) (:context @state))
    (throw (ex-info "CUDA buffer belongs to another context generation"
                    {:reason :runtime-generation-mismatch})))
  buffer)

(defn- assert-unborrowed! [borrows reason message]
  (when (seq @borrows)
    (throw (ex-info message {:reason reason :cleanup-retry-safe? true}))))

(defn make-buffer
  ([n] (make-buffer n :float))
  ([n dt] (make-buffer n dt {}))
  ([n dt {:keys [retain-owner! adopt-cleanup!]}]
   (when-not (dtype/known? dt)
     (throw (ex-info "CUDA buffer dtype is unsupported"
                     {:reason :cuda-unsupported-operation :backend :cuda :operation :make-buffer :dtype dt})))
   (when-not (and (integer? n) (<= 0 n Long/MAX_VALUE))
     (throw (ex-info "GPU buffer element count must be non-negative"
                     {:reason :gpu-buffer-negative-elements :elements n})))
   (when-not (every? #(or (nil? %) (fn? %)) [retain-owner! adopt-cleanup!])
     (throw (ex-info "Buffer ownership requires callbacks" {:reason :invalid-cleanup-plan})))
   (let [dt (dtype/canon dt) bytes (Math/multiplyExact (long n) (long (dtype/bytes-of dt)))
         slot (cleanup/acquisition-slot) borrows (atom #{})]
     (ensure-init!)
     (root/construct-child!
      state [{:id :memory
              :release #(do
                          (assert-unborrowed! borrows :cuda-buffer-in-use
                                              "CUDA buffer is retained by a prepared call")
                          (cleanup/release-native! slot (fn [p] (native! :free [p]))))}]
      (fn [owner entry]
        (prepared-execution/assert-admitted! (:prepared-execution @state))
        (when retain-owner! (retain-owner! owner))
        (let [p (cleanup/acquire-native! slot #(native! :allocate [(max 1 bytes)]))
              allocation (view/allocation {:id (random-uuid) :byte-size bytes
                                           :memory-space :device :device :cuda:0
                                           :alignment 256 :coherence :explicit-transfer})]
          {::buffer true ::context (:context @(:projection entry))
           :device-pointer p :n-elements n :byte-size bytes :dtype dt :alignment 256
           :borrows borrows
           :view (view/view allocation {:id (random-uuid) :dtype dt :shape [n]})}))
      (or adopt-cleanup! retain-owner!)))))

(defn make-buffer-like [buffer] (make-buffer (:n-elements buffer) (:dtype buffer)))

(defn slice-buffer [buffer byte-offset byte-length dt]
  (assert-buffer-live! buffer)
  (let [dt (dtype/canon dt) width (dtype/bytes-of dt)]
    (when-not (= dt (:dtype buffer))
      (throw (ex-info "CUDA slice cannot reinterpret its parent dtype" {:reason :cuda-slice-dtype})))
    (when-not (and (integer? byte-offset) (integer? byte-length)
                   (<= 0 byte-offset) (<= 0 byte-length)
                   (<= (+' byte-offset byte-length) (:byte-size buffer))
                   (zero? (mod byte-offset width)) (zero? (mod byte-length width)))
      (throw (ex-info "CUDA slice range is invalid" {:reason :cuda-slice-range})))
    (-> buffer
        (dissoc ::cleanup/owner)
        (assoc ::cleanup/lifetime-owner (owner! buffer)
               :device-pointer (Math/addExact (long (:device-pointer buffer)) (long byte-offset))
               ;; BufferAllocation guarantees are powers of two. Their gcd with an offset is
               ;; the smaller of the parent's guarantee and the offset's lowest set bit.
               :alignment (if (zero? byte-offset) (:alignment buffer)
                              (min (:alignment buffer) (Long/lowestOneBit (long byte-offset))))
               :byte-size byte-length :n-elements (quot byte-length width)
               :view (view/subview (:view buffer) {:id (random-uuid) :byte-offset byte-offset
                                                 :shape [(quot byte-length width)]})))))

(defn free-buffer! [buffer]
  (when-not (::cleanup/owner buffer)
    (throw (ex-info "Borrowed CUDA slice has no destruction authority" {:reason :missing-cleanup-owner})))
  (assert-unborrowed! (:borrows buffer) :cuda-buffer-in-use "CUDA buffer is retained by a prepared call")
  (cleanup/release! (::cleanup/owner buffer)))

(defn- host-segment [host]
  (if (instance? MemorySegment host) host (MemorySegment/ofArray host)))

(defn plan-range [buffer host {:keys [src-element dst-element elements]
                               :or {src-element 0 dst-element 0}} direction]
  (assert-buffer-live! buffer)
  (when-not (contains? #{:upload :download} direction)
    (throw (ex-info "CUDA range direction is unsupported" {:reason :cuda-range-direction})))
  (let [segment (host-segment host) width (dtype/bytes-of (:dtype buffer))
        [buf-index host-index] (if (= :upload direction) [dst-element src-element] [src-element dst-element])]
    (when-not (and (every? #(and (integer? %) (<= 0 % Long/MAX_VALUE)) [buf-index host-index elements])
                   (<= (+' buf-index elements) (:n-elements buffer))
                   (<= (*' (+' host-index elements) width) (.byteSize segment)))
      (throw (ex-info "CUDA transfer range exceeds buffer/host capacity" {:reason :cuda-range-capacity})))
    {:buf-off (* buf-index width) :host-off (* host-index width)
     :n-bytes (* elements width) :host-seg segment}))

(defn validate-range-plan!
  "Normalize an untrusted byte plan through the exact typed buffer range contract."
  [buffer {:keys [buf-off host-off n-bytes host-seg]} direction]
  (assert-buffer-live! buffer)
  (let [width (dtype/bytes-of (:dtype buffer))]
    (when-not (and (contains? #{:upload :download} direction)
                   (instance? MemorySegment host-seg)
                   (every? #(and (integer? %) (<= 0 % Long/MAX_VALUE)
                                 (zero? (mod % width))) [buf-off host-off n-bytes]))
      (throw (ex-info "CUDA transfer byte plan is invalid" {:reason :cuda-transfer-plan})))
    (when (and (pos? n-bytes)
               (or (not (.isAlive (.scope host-seg)))
                   (and (= :download direction) (.isReadOnly host-seg))))
      (throw (ex-info "CUDA transfer host segment is unavailable" {:reason :cuda-transfer-host-segment})))
    (plan-range buffer host-seg
                (if (= :upload direction)
                  {:src-element (quot host-off width) :dst-element (quot buf-off width)
                   :elements (quot n-bytes width)}
                  {:src-element (quot buf-off width) :dst-element (quot host-off width)
                   :elements (quot n-bytes width)}) direction)))

(defn execute-range! [buffer plan direction]
  (let [checked (validate-range-plan! buffer plan direction)
        {:keys [buf-off host-off n-bytes host-seg]} checked]
    (when (pos? n-bytes)
      (let [ledger (:prepared-execution @state)
            use-slot (cleanup/acquisition-slot) staging-slot (cleanup/acquisition-slot)
            started? (volatile! false) completed? (volatile! false)
            operation-owner
            (cleanup/owner
             [{:id :completion
               :release #(when (and @started? (not @completed?))
                           ;; The synchronous Driver API copies use the default
                           ;; stream, not our explicitly nonblocking execution stream.
                           (prepared-execution/complete!
                            ledger (fn [] (native! :stream-sync [MemorySegment/NULL])))
                           (vreset! completed? true))}
              {:id :staging :after #{:completion}
               :release #(cleanup/release-native! staging-slot (fn [p] (native! :host-free [p])))}
              {:id :buffer-use :after #{:staging}
               :release #(cleanup/release-native! use-slot
                                                 (fn [lease] (cleanup/release! (::cleanup/owner lease))))}])]
        (let [value
                (cleanup/build!
                 operation-owner
                 (fn []
                   (cleanup/acquire-native! use-slot #(cleanup/retain-use! (owner! buffer)))
                   ;; Reserve memory before acquiring the root/queue monitor. Free
                   ;; retires its root lease under the owner monitor in the opposite
                   ;; order; holding root while waiting for owner could deadlock.
                   (locking (:lock ledger)
                     (assert-buffer-live! buffer)
                     (let [pointer (cleanup/acquire-native! staging-slot #(native! :host-allocate [n-bytes]))
                           staging (.reinterpret ^MemorySegment pointer (long n-bytes))]
                       (prepared-execution/synchronize! ledger (:stream @state) native!)
                       (when (= :upload direction)
                         (MemorySegment/copy host-seg (long host-off) staging (long 0) (long n-bytes)))
                       ;; Pinned staging makes successful synchronous upload completion
                       ;; unambiguous. On any uncertain native outcome, rollback drains
                       ;; the actual copy stream before retiring staging or buffer use.
                       (vreset! started? true)
                       (native! direction [(+' (:device-pointer buffer) buf-off) staging n-bytes])
                       (vreset! completed? true)
                       (when (= :download direction)
                         (MemorySegment/copy staging (long 0) host-seg (long host-off) (long n-bytes)))
                       {})))
                 #(prepared-execution/retain-debt! ledger %))]
          (try (cleanup/release! (::cleanup/owner value))
               (catch Throwable e
                 (prepared-execution/retain-debt! ledger (::cleanup/owner value))
                 (throw e)))))))
  buffer)

(defn upload-range! [buffer host opts] (execute-range! buffer (plan-range buffer host opts :upload) :upload))
(defn download-range! [buffer host opts]
  (execute-range! buffer (plan-range buffer host opts :download) :download)
  host)
(defn array->buffer! [buffer host]
  (upload-range! buffer host {:elements (:n-elements buffer)}))
(defn buffer-of-array [host]
  (let [dt (case (.getName (.getComponentType (class host)))
             "float" :float "double" :double "int" :int "long" :long "short" :half "byte" :byte)
        buffer (make-buffer (alength host) dt)]
    (try (array->buffer! buffer host)
         (catch Throwable e (try (free-buffer! buffer) (catch Throwable x (.addSuppressed e x))) (throw e)))))
(defn buffer->array [buffer]
  (let [host (case (:dtype buffer)
               :float (float-array (:n-elements buffer)) :double (double-array (:n-elements buffer))
               :int (int-array (:n-elements buffer)) :long (long-array (:n-elements buffer))
               :half (short-array (:n-elements buffer)) :byte (byte-array (:n-elements buffer)))]
    (download-range! buffer host {:elements (:n-elements buffer)}) host))

(defn shutdown! [] (root/shutdown-construction! state))
(defn reset! [] (shutdown!))
