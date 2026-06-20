# State model: Datahike as truth, SoA columns as materialized views

How simulation/game state (valley, and dvergr-style agents generally) is structured
so that a relational, queryable, forkable store (Datahike) and the numerical hot state
that `deftm`/wasm/GPU kernels require (flat typed arrays, SoA) coexist **without two
sources of truth**. Cross-platform: the same model runs on the JVM (deftm→bytecode,
Datahike+konserve on disk) and in the browser (deftm→wasm, Datahike-cljs+konserve-
indexeddb).

## Principle

There is exactly one authoritative state: the **Datahike DB** (EAVT datoms). The
numerical SoA columns kernels operate on are a **materialized view** of the hot subset
of that DB — a denormalized, dense, typed cache. The test that the columns are a *view*
and not a second authority: they are reconstructible from a query at any time (a full
rebuild of N=2000 positions is ~1.3 ms with the new planner). Kernels never see datoms;
Datahike never sees float arrays. The only coupling is the view-maintenance machinery.

## Hot vs cold, and two maintenance directions

Each attribute flows in exactly one direction, chosen by its owner:

- **DB → column (forward incremental view maintenance):** cold inputs to hot
  computation (mass, drag, is-static, block-type). Written in the DB by game logic;
  the column holds a read-only mirror, patched only when those datoms change (rare).
- **column → DB (deferred write-back):** hot attrs the kernel owns (position,
  velocity). The kernel mutates the column every tick; the DB value is the last
  checkpoint. Deltas are transacted back **lazily** — not every frame.

Why deferred, not per-tick: a Datahike transact is ~2.5 µs/entity on the JVM (so a
1000-entity per-tick writeback ≈ 2.5 ms — affordable but wasteful), and ~20 µs/entity
in cljs today (see Performance). Hot attrs therefore *live* in columns between
writebacks; the DB is reconciled only when truth is actually needed: (a) a relational
query touches a hot attr, (b) a fork/snapshot/save, (c) a fixed checkpoint interval.
This is write-behind caching.

## The bridge: entity ↔ row

The single integration surface is a bijection **Datahike eid ↔ dense row index**.
Columns are indexed by row; the DB by eid. Spawning allocates a row (free-list);
despawning frees it. This map is the only thing that must stay coherent every frame.

## The frame loop

```
  snapshot db0 = @conn-value            (~0.2 µs — immutable persistent DB)

  GAMEPLAY PHASE   (DB is the writer)
    relational queries on db0 (in-range, hostile, can-craft?)   ~1 ms, cached ≈ 0
    transact cold changes (damage, pickups, AI transitions)
    refresh cold-input mirror columns for changed datoms

  NUMERIC PHASE    (columns are the writer — no DB access)
    deftm/wasm/GPU kernels sweep WHOLE columns in place
    integrate-physics! over all rows in one call (not per-entity)

  RECONCILE        (cadence-gated)
    every K frames / before a hot-attr query / on save:
      diff dirty rows → one batched fast-transact   (write-behind)
```

Fork/snapshot is the simulation superpower: `db' = @conn-value` is ~0.2 µs (structural
sharing) + copy-on-write of the working columns. Branch the world, run kernels forward
on the forked columns, score, keep or discard — rollback / what-if / ensemble — with
the DB side costing nothing. The bitemporal branch additionally lets you query "state
as of sim-tick T" while the column always holds "now".

## Optimal deftm/wasm integration: resident columns, Datahike indexes rows

The materialized-view model above describes *correctness*. The *optimal* form removes
all per-frame copying by making the kernel's own memory the column store:

> **The columns the kernels operate on ARE the resident storage for hot state, living
> in the kernel's native memory; Datahike is the relational index over *rows*, not the
> storage of hot values.**

```
  Datahike (truth for structure)              Resident columns (truth for hot values)
  ┌────────────────────────────┐             ┌─────────────────────────────────────┐
  │ entity registry: eid → row  │──row index─▶│ Position[3·N]  Velocity[3·N]  …       │
  │ cold attrs (faction, hp,    │             │ SoA, addressed by row                 │
  │   inventory, relations)     │             │ lives in: Java arrays (JVM)           │
  │ history / forks / queries   │             │           wasm linear memory (browser)│
  └────────────────────────────┘             │           GPU buffer (device)         │
         ▲  reconcile (rare)                  └─────────────────────────────────────┘
         │  column → datoms at checkpoint/                     ▲
         │  save / relational-query-on-hot-attr               │ kernels read/write IN PLACE
         └────────────────────────────────────────────────────┘  (zero marshal)
```

What makes it optimal, specifically for deftm/wasm:

- **The kernel's memory is the column store — no marshaling step.** On the JVM a column
  is a `double[]` deftm reads via `aget` at primitive speed. In wasm a column is a
  *permanent* region of linear memory at a fixed offset; the kernel reads/writes byte
  offsets directly and JS reads the same bytes via typed-array views for rendering. This
  is the delta from the current `integrate-physics!` shim, which uploads pos/vel/blocks
  into scratch *every call* and reads them back — if the columns *live* in linear
  memory, the call just passes base offsets and mutates in place. Same ptr,len ABI,
  pointed at resident allocations instead of per-frame scratch.
- **One kernel over the whole column.** `integrate-physics!` becomes
  `integrate-physics-batch!` sweeping N rows in one deftm pass (the SIMD/GPU sweet spot)
  instead of one marshaled call per entity.
- **Backend-parametric.** "Resident column" = Java array | wasm linear-memory region |
  GPU buffer. The kernel backend changes; the column abstraction and the eid→row bridge
  (portable `.cljc`) do not — the same model reaches GPU for free.

What Datahike stores, and does not:

- **Stores**: eid→row, all *cold* attrs, relations, and free snapshots/forks for
  simulation rollback (`@conn` ~0.2 µs + CoW the column buffers — valley.state's
  existing CoW-chunk trick, generalized to entity columns).
- **Does not store**: per-frame hot values as datoms (redundant — the column is the
  truth, and the 2.5 µs/ent JVM / 20 µs/ent cljs writeback is exactly what deferral
  avoids). Hot values reach Datahike only at reconcile (checkpoint/save), or never if
  queried by row instead of by Datalog.
- **Block world**: the bulk-hot case of the same rule — chunk arrays are resident
  columns; Datahike holds only chunk metadata (coord→key, dirty). This is why option C's
  windowed physics is pure array work that never touches Datahike.

For hot-attr *spatial* queries ("entities within radius"), use a spatial index
(grid/BVH) derived from the position column — not a Datalog range scan. Datalog is for
relational/cold selection; the spatial index for geometric selection; the column for the
math.

**Staging.** Single player + option C does not need the column system (one entity).
The mobs phase (N>1) is the natural entry point: introduce the eid→row bridge + resident
pos/vel columns in linear memory + the batch kernel + reconcile-at-checkpoint. It is not
a rewrite — the valley.state CoW pattern and the wasm ABI are already ~80 % of it.

## `fast-transact`: the synchronous write-behind path

`valley.state/fast-transact` is a direct-index transact that bypasses the async writer,
schema validation, history and hashing — for pre-validated `[e a v]` triples it just
upserts into the EAVT/AEVT/AVET indices and returns a `TxReport`. This is the per-frame
write-behind primitive.

### Cross-platform correctness: capture the index return values

The original (JVM-only) form built `transient` indices and called `-remove`/`-insert`
**discarding their return values**, relying on in-place mutation. That is correct on the
JVM and wrong in ClojureScript:

- **JVM** persistent-sorted-set transient: `conj!`/`disj!` mutate the B-tree in place
  and return the same object → discarding the return is forgiven.
- **cljs** persistent-sorted-set (`BTSet`) has *no real transient*: `-as-transient`
  and `-persistent!` are identity, and `conj!`/`disj!` (`conjoin`/`disjoin`) always
  allocate and return a **new `BTSet`**. The updated value lives *only in the return*;
  discarding it silently drops every update (the index is unchanged, only `:max-tx`
  advances).

This is **not** a hitchhiker-tree vs PSS issue — it is PSS on both runtimes, differing
only in transient semantics. The fix is the proper transient contract: **thread the
return of every `-remove`/`-insert`** (`eavt = (di/-insert eavt d …)`). Correct on both;
on the JVM it is free (the transient returns `self`) — see Performance.

### Hold the db value, don't deref the Connection per frame

`@conn` calls `deref-conn`, which with the default `:writer {:backend :self}`
(non-streaming) **reads the db back from the konserve store and deserializes it on every
deref** (`k/get` → `stored->db`) — it does not return an in-memory atom. The hot loop
must therefore hold the immutable db **value** in a plain atom (the working set),
`fast-transact` on it, and query `@db-atom`; the durable Connection/store is touched
only at reconcile/checkpoint. (A streaming writer makes `@conn` return the in-memory
value, an alternative to holding the value yourself.)

## Cross-platform symmetry

```
  JVM:      Datahike(jvm, new planner) ⇄ entity↔row ⇄ typed arrays    ⇄ deftm→bytecode
                          │ konserve-file/rocksdb
  Browser:  Datahike(cljs, planner ON by default) ⇄ entity↔row ⇄ wasm linear memory ⇄ deftm→wasm
                          │ konserve-indexeddb
```

- Datahike runs in cljs: `d/connect` is sync; `d/create-database`/`d/transact!` are
  async (core.async channels); `d/transact` (sync) is clj-only. The new IR/cost-based
  query planner is `.cljc` and `*force-legacy*` defaults to **false in cljs** (planner
  on by default in the browser).
- The integration layer (entity↔row, attribute classification, reconcile policy) is
  portable `.cljc`. The only per-platform differences are the kernel backend (bytecode
  vs wasm) and the konserve backend — both already abstracted.
- Block chunks are the same pattern at coarser grain: the dense block array is a
  materialized view of a konserve blob; Datahike holds chunk metadata (coord → key,
  dirty flag). This is what `valley.persistence` already does.

## Do we need the dual representation?

Yes for the bytes, no for the truth. Kernels physically require contiguous f64/f32
buffers; EAVT datoms in a sorted-set are pointer-chasing trees you cannot hand to
wasm/GPU. So the column must exist — but it is a *view* (one writer per attribute per
phase, rebuildable from the DB), not a second authority. "Dual source of truth" is the
mess; "single source + materialized view with deferred write-back" is the principled
resolution, and free snapshots make it strictly better than a standalone ECS.

## Performance (measured)

`fast-transact`, per-frame position writeback, persistent-sorted-set index:

| platform | 50 ent | 200 ent | per-entity |
|---|---|---|---|
| JVM HotSpot 25 (ignore-return, original) | 126 µs | 495 µs | ~2.5 µs |
| JVM HotSpot 25 (capture-return, fixed)   | 107 µs | 487 µs | ~2.4 µs |
| cljs / V8 (capture-return, fixed)        |  —     | 4.1 ms | ~20.6 µs |

The fix is free on the JVM (capture ≈ ignore, even marginally faster — the transient
returns `self`). The ~8× JVM→cljs gap is **not** in `fast-transact`: cljs PSS lacks
real transients, so each `conj!`/`disj!` allocates a new `BTSet` (~7 ops/entity → ~7
allocations/entity). Closing it is a PSS-level change (the `^:mutable root` is already
present; `conjoin`/`disjoin` would need an edit/transient flag that mutates in place and
returns `this`), not a valley change. Either way 200 mobs/frame = 0.49 ms (JVM) /
4.1 ms (cljs) — within a 16.6 ms frame.

Datahike read/snapshot, new planner, N=2000 (JVM, cache off):
snapshot/fork ~0.2 µs; gameplay query ~1.1 ms; full view rebuild (pull all positions)
~1.3 ms. Snapshots are effectively free; reads are frame-budget-compatible; per-tick
writes are the cost the write-behind cadence exists to amortize.
