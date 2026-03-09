(ns valley.state
  "World state management for Valley.

  Combines Datahike (entity state) with CoW chunk storage (block world)
  into a unified state container with transient/persistent semantics.

  Entity state (mobs, player, items) lives in Datahike:
    - Queryable via Datalog
    - Forkable (immutable db snapshots)
    - Updated per-frame via fast-transact (144µs for 50 entities)

  Chunk storage (block arrays) lives in a persistent map with CoW:
    - Direct array access for physics/meshing
    - Fork marks chunks shared, first mutation copies the array
    - Serializable via konserve

  Usage:
    (def world (create-world))
    ;; Per frame:
    (let [db (world-db world)
          entities (pull-entities db)
          ;; ... physics ...
          tx-data (build-updates entities)]
      (fast-transact! world tx-data))
    ;; Fork:
    (let [snapshot (fork-world world)]
      ;; snapshot is immutable, world continues evolving)
    ;; Reset:
    (reset-world! world snapshot)"
  (:require [datahike.api :as d]
            [datahike.db :as db]
            [datahike.db.interface :as dbi]
            [datahike.datom :as dd]
            [datahike.index :as di]
            [valley.world :as w]))

;; ================================================================
;; Schema — entity attributes for game state
;; ================================================================

(def entity-schema
  "Datahike schema for game entities."
  [;; Position
   {:db/ident :px :db/valueType :db.type/double :db/cardinality :db.cardinality/one}
   {:db/ident :py :db/valueType :db.type/double :db/cardinality :db.cardinality/one}
   {:db/ident :pz :db/valueType :db.type/double :db/cardinality :db.cardinality/one}
   ;; Velocity
   {:db/ident :vx :db/valueType :db.type/double :db/cardinality :db.cardinality/one}
   {:db/ident :vy :db/valueType :db.type/double :db/cardinality :db.cardinality/one}
   {:db/ident :vz :db/valueType :db.type/double :db/cardinality :db.cardinality/one}
   ;; Entity properties
   {:db/ident :yaw :db/valueType :db.type/double :db/cardinality :db.cardinality/one}
   {:db/ident :health :db/valueType :db.type/long :db/cardinality :db.cardinality/one}
   {:db/ident :max-health :db/valueType :db.type/long :db/cardinality :db.cardinality/one}
   {:db/ident :mob-type :db/valueType :db.type/keyword :db/cardinality :db.cardinality/one}
   {:db/ident :entity-state :db/valueType :db.type/long :db/cardinality :db.cardinality/one}
   {:db/ident :hostile :db/valueType :db.type/boolean :db/cardinality :db.cardinality/one}
   {:db/ident :on-ground :db/valueType :db.type/boolean :db/cardinality :db.cardinality/one}
   ;; Entity kind (for polymorphic queries)
   {:db/ident :entity-kind :db/valueType :db.type/keyword :db/cardinality :db.cardinality/one}
   ;; Player-specific
   {:db/ident :hunger :db/valueType :db.type/double :db/cardinality :db.cardinality/one}
   {:db/ident :saturation :db/valueType :db.type/double :db/cardinality :db.cardinality/one}])

;; AI state encoding (keywords → longs for deftm compatibility)
(def state-codes {:idle 0 :wander 1 :flee 2 :chase 3 :attack 4})
(def code->state (into {} (map (fn [[k v]] [v k]) state-codes)))

;; ================================================================
;; Fast-path transact — bypass async writer, batch PSS updates
;; ================================================================

(defn fast-transact
  "Fast synchronous transact for game state updates.
  Takes a db value and pre-validated [e a v] triples.
  Returns TxReport with :db-before, :db-after, :tx-data.
  Skips: async writer, schema validation, history, hash updates."
  [db datom-triples]
  (let [tx (inc (:max-tx db))
        datoms (mapv (fn [[e a v]] (dd/datom e a v tx)) datom-triples)
        ;; Batch update: transient all indices
        eavt (transient (:eavt db))
        aevt (transient (:aevt db))
        avet (transient (:avet db))
        op-count (:op-count db)]
    ;; Upsert: remove old values, insert new
    (doseq [[e a _v] datom-triples]
      (when-let [old (first (dbi/search db [e a]))]
        (let [oc (inc op-count)]
          (di/-remove eavt old :eavt oc)
          (di/-remove aevt old :aevt oc)
          (di/-remove avet old :avet oc))))
    (doseq [d datoms]
      (let [oc (inc op-count)]
        (di/-insert eavt d :eavt oc)
        (di/-insert aevt d :aevt oc)
        (di/-insert avet d :avet oc)))
    (let [new-db (assoc db
                   :eavt (persistent! eavt)
                   :aevt (persistent! aevt)
                   :avet (persistent! avet)
                   :max-tx tx
                   :op-count (+ op-count (count datoms)))]
      (db/map->TxReport
        {:db-before db
         :db-after new-db
         :tx-data datoms
         :tempids {:db/current-tx tx}
         :tx-meta {:db/txInstant (java.util.Date.)}}))))

(defn fast-transact!
  "Fast-transact and swap the connection's db atom.
  Accepts either a WorldState or a raw Datahike connection.
  Fires listeners if any."
  [world-or-conn datom-triples]
  (let [conn (if (:conn world-or-conn)
               (:conn world-or-conn)
               world-or-conn)
        db @conn
        report (fast-transact db datom-triples)]
    ;; Swap connection to new db
    (swap! (:wrapped-atom conn)
           assoc :db (:db-after report))
    ;; Fire listeners
    (doseq [[_ callback] (some-> (:listeners (meta conn)) (deref))]
      (callback report))
    report))

;; ================================================================
;; Chunk storage with CoW semantics
;; ================================================================

(defrecord CowChunk [^ints blocks ^bytes light
                     ^long cx ^long cy ^long cz
                     ^boolean dirty ^boolean shared])

(defn make-cow-chunk [cx cy cz]
  (->CowChunk (int-array w/CHUNK-VOLUME)
              (byte-array w/CHUNK-VOLUME)
              cx cy cz true false))

(defn cow-set-block
  "Set a block in a CoW chunk. Copies arrays if shared."
  [^CowChunk chunk x y z block-id]
  (let [blocks (if (:shared chunk)
                 (let [copy (aclone (:blocks chunk))]
                   copy)
                 (:blocks chunk))
        idx (w/block-index x y z)]
    (aset ^ints blocks idx (int block-id))
    (assoc chunk :blocks blocks :dirty true :shared false)))

(defn fork-chunks
  "Mark all chunks as shared (CoW). Returns new chunk map."
  [chunk-map]
  (persistent!
    (reduce-kv (fn [m k chunk]
                 (assoc! m k (assoc chunk :shared true)))
               (transient {}) chunk-map)))

;; ================================================================
;; World state container
;; ================================================================

(defrecord WorldState [conn          ;; Datahike connection
                       chunk-map     ;; {[cx cy cz] -> CowChunk}
                       solid-arr     ;; byte-array — block property LUT
                       time-of-day   ;; atom<double>
                       tick-count])  ;; atom<long>

(defn create-world
  "Create a new world state with empty Datahike DB and no chunks."
  []
  (let [cfg {:store {:backend :memory :id (java.util.UUID/randomUUID)}
             :schema-flexibility :write
             :keep-history? false
             :attribute-refs? true}]
    (d/create-database cfg)
    (let [conn (d/connect cfg)]
      (d/transact conn entity-schema)
      (->WorldState conn
                    {}
                    nil  ;; solid-arr set by caller via set-solid-arr!
                    (atom 0.5)
                    (atom 0)))))

(defn set-solid-arr!
  "Set the block solidity lookup table. Called once at game init."
  [^WorldState world ^bytes arr]
  (assoc world :solid-arr arr))

(defn world-db
  "Get immutable db snapshot from world. Free (just deref)."
  [^WorldState world]
  @(:conn world))

(defn fork-world
  "Create an immutable snapshot of the world state.
  Entity state: immutable Datahike db value.
  Chunks: marked shared (CoW on mutation)."
  [^WorldState world]
  {:db (world-db world)
   :chunks (fork-chunks (:chunk-map world))
   :time-of-day @(:time-of-day world)
   :tick-count @(:tick-count world)})

(defn reset-world!
  "Restore world state from a snapshot."
  [^WorldState world snapshot]
  (swap! (:wrapped-atom (:conn world)) assoc :db (:db snapshot))
  (reset! (:time-of-day world) (:time-of-day snapshot))
  (reset! (:tick-count world) (:tick-count snapshot))
  ;; Chunks: replace the chunk-map reference
  ;; (caller must handle this since chunk-map is in the WorldState record)
  world)

;; ================================================================
;; Entity helpers — add/query/update entities via Datahike
;; ================================================================

(defn add-mob!
  "Add a mob entity to the world. Returns the entity ID."
  [^WorldState world mob-type x y z hostile?]
  (let [conn (:conn world)
        result (d/transact conn
                 [{:entity-kind :mob
                   :mob-type mob-type
                   :px (double x) :py (double y) :pz (double z)
                   :vx 0.0 :vy 0.0 :vz 0.0
                   :yaw (* (rand) 2.0 Math/PI)
                   :health (long 20) :max-health (long 20)
                   :entity-state (long (get state-codes :idle 0))
                   :hostile (boolean hostile?)
                   :on-ground false}])]
    ;; Return the new entity ID
    (-> result :tempids vals first)))

(defn pull-mob
  "Pull a mob entity as a map from a db snapshot."
  [db eid]
  (d/pull db [:px :py :pz :vx :vy :vz :yaw :health :max-health
              :mob-type :entity-state :hostile :on-ground :entity-kind] eid))

(defn pull-all-mobs
  "Pull all mob data in one query. Returns seq of tuples:
  [eid px py pz vx vy vz yaw health mob-type entity-state hostile on-ground]"
  [db]
  (d/q '[:find ?e ?px ?py ?pz ?vx ?vy ?vz ?yaw ?hp ?mt ?st ?hostile ?ground
         :where [?e :entity-kind :mob]
                [?e :px ?px] [?e :py ?py] [?e :pz ?pz]
                [?e :vx ?vx] [?e :vy ?vy] [?e :vz ?vz]
                [?e :yaw ?yaw] [?e :health ?hp]
                [?e :mob-type ?mt] [?e :entity-state ?st]
                [?e :hostile ?hostile] [?e :on-ground ?ground]]
       db))

(defn mobs->maps
  "Convert pull-all-mobs tuples to maps (for rendering/AI compatibility)."
  [tuples]
  (mapv (fn [[eid px py pz vx vy vz yaw hp mt st hostile ground]]
          {:db/id eid
           :px px :py py :pz pz
           :vx vx :vy vy :vz vz
           :yaw yaw :health hp
           :mob-type mt
           :entity-state st
           :hostile hostile
           :on-ground ground
           ;; Legacy compat fields for mobs.clj/hostile.clj
           :id eid
           :pos (double-array [px py pz])
           :vel (double-array [vx vy vz])
           :hostile? hostile
           :on-ground? ground
           :max-health 20
           :walk-phase 0.0
           :hurt-timer 0.0
           :state (get code->state st :idle)
           :state-timer (+ 2.0 (* (rand) 3.0))
           :wander-dir [(- (rand) 0.5) (- (rand) 0.5)]})
        tuples))

(defn build-full-updates
  "Build fast-transact triples from ticked mob maps.
  Takes the db (for attr refs) and seq of maps with :db/id and updated fields."
  [db mobs]
  (let [px-ref (dbi/-ref-for db :px)
        py-ref (dbi/-ref-for db :py)
        pz-ref (dbi/-ref-for db :pz)
        vx-ref (dbi/-ref-for db :vx)
        vy-ref (dbi/-ref-for db :vy)
        vz-ref (dbi/-ref-for db :vz)
        yaw-ref (dbi/-ref-for db :yaw)
        hp-ref (dbi/-ref-for db :health)
        st-ref (dbi/-ref-for db :entity-state)
        ground-ref (dbi/-ref-for db :on-ground)]
    (vec (mapcat (fn [m]
                  (let [eid (:db/id m)
                        ^doubles pos (:pos m)
                        ^doubles vel (:vel m)]
                    [[eid px-ref (aget pos 0)]
                     [eid py-ref (aget pos 1)]
                     [eid pz-ref (aget pos 2)]
                     [eid vx-ref (double (aget vel 0))]
                     [eid vy-ref (double (or (:vy m) (aget vel 1)))]
                     [eid vz-ref (double (aget vel 2))]
                     [eid yaw-ref (double (:yaw m))]
                     [eid ground-ref (boolean (:on-ground? m))]
                     [eid hp-ref (long (:health m))]
                     [eid st-ref (long (get state-codes (:state m) 0))]]))
                mobs))))

(defn all-mob-eids
  "Get all mob entity IDs from a db snapshot."
  [db]
  (d/q '[:find [?e ...] :where [?e :entity-kind :mob]] db))

(defn nearby-mobs
  "Find mob entity IDs near (px, pz) within radius r."
  [db px pz r]
  (d/q '[:find [?e ...]
         :in $ ?cx ?cz ?r2
         :where [?e :entity-kind :mob]
                [?e :px ?px]
                [?e :pz ?pz]
                [(- ?px ?cx) ?dx]
                [(- ?pz ?cz) ?dz]
                [(* ?dx ?dx) ?dx2]
                [(* ?dz ?dz) ?dz2]
                [(+ ?dx2 ?dz2) ?d2]
                [(< ?d2 ?r2)]]
       db (double px) (double pz) (* (double r) (double r))))

(defn hostile-mobs
  "Find all hostile mob entity IDs."
  [db]
  (d/q '[:find [?e ...] :where [?e :hostile true]] db))

(defn build-position-updates
  "Build fast-transact triples for position updates.
  Takes a seq of [eid new-px new-py new-pz new-vy on-ground?] tuples
  and the db (for attribute ref resolution)."
  [db updates]
  (let [px-ref (dbi/-ref-for db :px)
        py-ref (dbi/-ref-for db :py)
        pz-ref (dbi/-ref-for db :pz)
        vy-ref (dbi/-ref-for db :vy)
        ground-ref (dbi/-ref-for db :on-ground)]
    (vec (mapcat (fn [[eid px py pz vy on-ground]]
                  [[eid px-ref px]
                   [eid py-ref py]
                   [eid pz-ref pz]
                   [eid vy-ref vy]
                   [eid ground-ref on-ground]])
                updates))))
