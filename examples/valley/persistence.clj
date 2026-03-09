(ns valley.persistence
  "World persistence: Datahike for metadata/player/entities,
  raw files for chunk block/light arrays."
  (:require [datahike.api :as d]
            [clojure.java.io :as io]
            [clojure.edn :as edn])
  (:import [java.io DataOutputStream DataInputStream
            BufferedOutputStream BufferedInputStream
            FileOutputStream FileInputStream]
           [java.util.concurrent ConcurrentLinkedQueue]))

(set! *warn-on-reflection* true)

;; ================================================================
;; Schema
;; ================================================================

(def schema
  [{:db/ident :chunk/coord
    :db/valueType :db.type/string
    :db/unique :db.unique/identity
    :db/cardinality :db.cardinality/one}
   {:db/ident :chunk/blocks-file
    :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one}
   {:db/ident :chunk/light-file
    :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one}
   {:db/ident :player/id
    :db/valueType :db.type/string
    :db/unique :db.unique/identity
    :db/cardinality :db.cardinality/one}
   {:db/ident :player/pos
    :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one}
   {:db/ident :player/health
    :db/valueType :db.type/long
    :db/cardinality :db.cardinality/one}
   {:db/ident :player/inventory
    :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one}
   {:db/ident :player/hunger
    :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one}
   {:db/ident :entity/uuid
    :db/valueType :db.type/string
    :db/unique :db.unique/identity
    :db/cardinality :db.cardinality/one}
   {:db/ident :entity/data
    :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one}
   {:db/ident :world/id
    :db/valueType :db.type/string
    :db/unique :db.unique/identity
    :db/cardinality :db.cardinality/one}
   {:db/ident :world/seed
    :db/valueType :db.type/long
    :db/cardinality :db.cardinality/one}
   {:db/ident :world/game-time
    :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one}
   {:db/ident :world/spawn-pos
    :db/valueType :db.type/string
    :db/cardinality :db.cardinality/one}])

;; ================================================================
;; Database lifecycle
;; ================================================================

(defn world-dir
  "Get or create the world save directory."
  [^String base-path]
  (let [dir (io/file base-path)]
    (.mkdirs dir)
    (.mkdirs (io/file dir "chunks"))
    (.getAbsolutePath dir)))

(def ^:private valley-db-id #uuid "a1b2c3d4-e5f6-7890-abcd-ef0123456789")

(defn db-config [^String world-path]
  {:store {:backend :file
           :path (str world-path "/datahike")
           :id valley-db-id}})

(defn create-world-db!
  "Create a fresh world database. Returns connection."
  [world-path]
  (let [cfg (db-config world-path)]
    (when (d/database-exists? cfg)
      (d/delete-database cfg))
    (d/create-database cfg)
    (let [conn (d/connect cfg)]
      (d/transact conn schema)
      conn)))

(defn open-world-db
  "Open existing world database. Returns connection or nil."
  [world-path]
  (let [cfg (db-config world-path)]
    (when (d/database-exists? cfg)
      (d/connect cfg))))

(defn close-world-db!
  "Release the connection."
  [conn]
  (when conn
    (d/release conn)))

;; ================================================================
;; Chunk file I/O
;; ================================================================

(defn- coord-str
  "Convert chunk pos [cx cy cz] to string key."
  [[cx cy cz]]
  (str cx "_" cy "_" cz))

(defn- chunk-blocks-path ^String [^String world-path pos]
  (str world-path "/chunks/" (coord-str pos) ".blocks"))

(defn- chunk-light-path ^String [^String world-path pos]
  (str world-path "/chunks/" (coord-str pos) ".light"))

(defn save-chunk-files!
  "Write chunk block and light arrays to disk."
  [^String world-path pos ^ints blocks ^bytes light]
  (let [blocks-file (chunk-blocks-path world-path pos)
        light-file (chunk-light-path world-path pos)]
    ;; Write blocks (int array → 4 bytes each)
    (with-open [out (DataOutputStream. (BufferedOutputStream. (FileOutputStream. blocks-file)))]
      (dotimes [i (alength blocks)]
        (.writeInt out (aget blocks i))))
    ;; Write light (byte array)
    (with-open [out (BufferedOutputStream. (FileOutputStream. light-file))]
      (.write out light))
    {:blocks-file blocks-file :light-file light-file}))

(defn load-chunk-files
  "Read chunk block and light arrays from disk. Returns {:blocks int[] :light byte[]} or nil."
  [^String world-path pos]
  (let [blocks-file (chunk-blocks-path world-path pos)
        light-file (chunk-light-path world-path pos)]
    (when (and (.exists (io/file blocks-file))
               (.exists (io/file light-file)))
      (let [light (byte-array (count (slurp (io/file light-file) :encoding "ISO-8859-1")))]
        ;; Read light
        (with-open [in (BufferedInputStream. (FileInputStream. light-file))]
          (.read in light))
        ;; Read blocks
        (let [n (/ (.length (io/file blocks-file)) 4)
              blocks (int-array n)]
          (with-open [in (DataInputStream. (BufferedInputStream. (FileInputStream. blocks-file)))]
            (dotimes [i n]
              (aset blocks i (.readInt in))))
          {:blocks blocks :light light})))))

;; ================================================================
;; Chunk persistence (Datahike + files)
;; ================================================================

(defn save-chunk!
  "Save a chunk to files and register in Datahike."
  [conn ^String world-path chunk]
  (let [pos (:pos chunk)
        {:keys [blocks-file light-file]} (save-chunk-files! world-path pos
                                           (:blocks chunk) (:light chunk))
        coord (coord-str pos)]
    (d/transact conn [{:chunk/coord coord
                       :chunk/blocks-file blocks-file
                       :chunk/light-file light-file}])))

(defn load-chunk
  "Load a chunk from files. Returns chunk map or nil."
  [^String world-path pos]
  (let [data (load-chunk-files world-path pos)]
    (when data
      {:pos pos
       :blocks (:blocks data)
       :light (:light data)
       :dirty? true  ;; needs meshing
       :mesh nil})))

(defn chunk-saved?
  "Check if a chunk exists in the world directory."
  [^String world-path pos]
  (.exists (io/file (chunk-blocks-path world-path pos))))

;; ================================================================
;; Player persistence
;; ================================================================

(defn save-player!
  "Save player state to Datahike."
  [conn player-entity inventory hunger-state cam-pos]
  (d/transact conn [{:player/id "player-1"
                     :player/pos (pr-str cam-pos)
                     :player/health (long (:health player-entity))
                     :player/inventory (pr-str inventory)
                     :player/hunger (pr-str hunger-state)}]))

(defn load-player
  "Load player state from Datahike. Returns map or nil."
  [conn]
  (let [result (d/q '[:find ?pos ?health ?inv ?hunger
                       :where
                       [?e :player/id "player-1"]
                       [?e :player/pos ?pos]
                       [?e :player/health ?health]
                       [?e :player/inventory ?inv]
                       [?e :player/hunger ?hunger]]
                    @conn)]
    (when (seq result)
      (let [[pos health inv hunger] (first result)]
        {:pos (edn/read-string pos)
         :health health
         :inventory (edn/read-string inv)
         :hunger-state (edn/read-string hunger)}))))

;; ================================================================
;; Entity persistence (mobs)
;; ================================================================

(defn save-entities!
  "Save all mob entities to Datahike."
  [conn mobs]
  ;; Retract all existing entities first, then add current ones
  (let [existing (d/q '[:find ?e ?uuid
                         :where [?e :entity/uuid ?uuid]]
                       @conn)
        retractions (mapv (fn [[e _]] [:db/retractEntity e]) existing)
        additions (mapv (fn [mob]
                          ;; Convert double-arrays to vectors for EDN serialization
                          (let [m (-> (dissoc mob :id)
                                     (cond-> (:pos mob)
                                       (update :pos #(if (.isArray (class %)) (vec %) %)))
                                     (cond-> (:vel mob)
                                       (update :vel #(if (.isArray (class %)) (vec %) %))))]
                            {:entity/uuid (str (:id mob))
                             :entity/data (pr-str m)}))
                        mobs)]
    (when (or (seq retractions) (seq additions))
      (d/transact conn (vec (concat retractions additions))))))

(defn load-entities
  "Load mob entities from Datahike. Skips entities that fail to parse."
  [conn]
  (let [results (d/q '[:find ?uuid ?data
                        :where
                        [?e :entity/uuid ?uuid]
                        [?e :entity/data ?data]]
                      @conn)]
    (into []
      (keep (fn [[uuid data]]
              (try
                (let [m (edn/read-string data)
                      ;; Convert vectors back to double-arrays (saved as vectors for EDN)
                      m (cond-> m
                          (and (:pos m) (vector? (:pos m)))
                          (assoc :pos (double-array (:pos m)))
                          (and (:vel m) (vector? (:vel m)))
                          (assoc :vel (double-array (:vel m))))]
                  (assoc m :id (parse-uuid uuid)))
                (catch Exception e
                  (println "Skipping corrupt entity" uuid ":" (.getMessage e))
                  nil))))
      results)))

;; ================================================================
;; World metadata
;; ================================================================

(defn save-world-meta!
  "Save world metadata (seed, game-time, spawn)."
  [conn seed game-time spawn-pos]
  (d/transact conn [{:world/id "world-1"
                     :world/seed (long seed)
                     :world/game-time (pr-str game-time)
                     :world/spawn-pos (pr-str spawn-pos)}]))

(defn load-world-meta
  "Load world metadata. Returns map or nil."
  [conn]
  (let [result (d/q '[:find ?seed ?time ?spawn
                       :where
                       [?e :world/id "world-1"]
                       [?e :world/seed ?seed]
                       [?e :world/game-time ?time]
                       [?e :world/spawn-pos ?spawn]]
                     @conn)]
    (when (seq result)
      (let [[seed game-time spawn] (first result)]
        {:seed seed
         :game-time (edn/read-string game-time)
         :spawn-pos (edn/read-string spawn)}))))

;; ================================================================
;; Async write queue
;; ================================================================

(defonce ^ConcurrentLinkedQueue write-queue (ConcurrentLinkedQueue.))
(defonce writer-thread (atom nil))

(defn queue-save!
  "Queue a save operation. op is a fn of [conn world-path]."
  [op]
  (.add write-queue op))

(defn- flush-queue!
  "Process all queued save operations."
  [conn world-path]
  (loop []
    (when-let [op (.poll write-queue)]
      (try
        (op conn world-path)
        (catch Exception e
          (println "Save error:" (.getMessage e))))
      (recur))))

(defn start-writer!
  "Start the background writer thread."
  [conn world-path]
  (let [running? (atom true)
        t (Thread.
            (fn []
              (while @running?
                (try
                  (flush-queue! conn world-path)
                  (Thread/sleep 200)
                  (catch InterruptedException _
                    (reset! running? false))
                  (catch Exception e
                    (println "Writer error:" (.getMessage e))))))
            "valley-writer")]
    (.setDaemon t true)
    (.start t)
    (reset! writer-thread {:thread t :running? running?})
    t))

(defn stop-writer!
  "Stop the writer thread and flush remaining ops."
  [conn world-path]
  (when-let [{:keys [^Thread thread running?]} @writer-thread]
    (reset! running? false)
    (.interrupt thread)
    (.join thread 2000)
    ;; Final flush
    (flush-queue! conn world-path)
    (reset! writer-thread nil)))
