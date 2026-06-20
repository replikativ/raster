(ns valley.mobs-store
  "Resident-column mob store (mobs phase) — the optimal state model made concrete (JVM).

   Hot state (position/velocity) lives in SoA double-array COLUMNS indexed by row; the
   batch physics kernel (valley.core/integrate-physics-batch!) sweeps the columns in
   place every frame. Datahike holds the eid→row bridge's cold half — the relational
   attrs (type, hp) and the row pointer — and serves queries + snapshots; it is NOT
   written per frame. Positions are reconciled into Datahike lazily (checkpoint/save),
   never on the hot path. See design/state-model.md.

   The browser counterpart is the same shape with the columns living in wasm linear
   memory and async datahike-cljs; the eid→row bridge and reconcile policy are identical."
  (:require [datahike.api :as d]
            [valley.core :as vc]))

(def ^:const CAP 4096)            ; max resident mobs (column capacity)
(def ^:const HALF-W 0.6)
(def ^:const HEIGHT 1.7)

(defn make-store
  "conn = a Datahike connection (e.g. (:conn (valley.state/create-world))); world = a
   chunk world {:blocks :solid :wx :wy :wz} (the resident block array the kernel reads)."
  [conn world]
  {:conn     conn
   :world    world
   :n        (atom 0)
   :eid->row (atom {})            ; the bridge (host-side; could also be a :mob/row datom)
   :row->eid (long-array CAP)
   :pos      (double-array (* CAP 3))
   :vel      (double-array (* CAP 3))
   :dxs      (double-array CAP)
   :dzs      (double-array CAP)})

(defn spawn!
  "Allocate a row, write the hot columns, and transact the cold attrs to Datahike.
   Returns the Datahike eid (the stable handle); the row is the column index."
  [store mob-type hp x y z]
  (let [row @(:n store)
        rep (d/transact (:conn store)
                        [{:db/id -1 :mob-type mob-type :health hp :entity-kind :mob
                          :px (double x) :py (double y) :pz (double z)}])
        eid (get (:tempids rep) -1)
        b   (* row 3)]
    (aset ^longs (:row->eid store) row (long eid))
    (swap! (:eid->row store) assoc eid row)
    (aset ^doubles (:pos store) b (double x))
    (aset ^doubles (:pos store) (+ b 1) (double y))
    (aset ^doubles (:pos store) (+ b 2) (double z))
    (swap! (:n store) inc)
    eid))

(defn set-drift!
  "Per-mob horizontal drift (the AI's intended xz move this tick)."
  [store eid dx dz]
  (let [row (get @(:eid->row store) eid)]
    (aset ^doubles (:dxs store) row (double dx))
    (aset ^doubles (:dzs store) row (double dz))))

(defn step!
  "One physics tick over ALL mobs — a single batch kernel call over the columns.
   Mutates the pos/vel columns in place; Datahike is untouched (write-behind)."
  [store dt]
  (let [w (:world store) n @(:n store)]
    (vc/integrate-physics-batch!
     (:pos store) (:vel store) (:dxs store) (:dzs store)
     (:blocks w) (:solid w) n (:wx w) (:wy w) (:wz w) HALF-W HEIGHT dt)))

(defn mob-pos
  "Read a mob's position from the hot columns by eid (no Datahike round-trip)."
  [store eid]
  (let [b (* (get @(:eid->row store) eid) 3) p (:pos store)]
    [(aget ^doubles p b) (aget ^doubles p (+ b 1)) (aget ^doubles p (+ b 2))]))

(defn reconcile!
  "Checkpoint: write the hot position columns back into Datahike for every row. This is
   the durable write-behind point (on save / before a relational query that needs current
   positions) — NOT the hot path — so it uses the normal `d/transact` (which persists
   through the writer and is visible via @conn), not the per-frame fast-transact."
  [store]
  (let [n @(:n store) p (:pos store) r->e (:row->eid store)
        tx (vec (for [row (range n)
                      :let [eid (aget ^longs r->e row) b (* row 3)]]
                  {:db/id eid
                   :px (aget ^doubles p b)
                   :py (aget ^doubles p (+ b 1))
                   :pz (aget ^doubles p (+ b 2))}))]
    (d/transact (:conn store) tx)))
