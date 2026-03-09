# Living Valley

![screenshot](screenshot.png)

A voxel survival game built as a showcase for Raster's programming model.
Procedural terrain with biomes and caves, day/night lighting, passive and
hostile mobs, block interaction, hunger, farming, crafting, and inventory --
all rendered through Raster's Vulkan engine.

## Running

Start a REPL with the Vulkan alias:

```bash
clojure -M:vulkan:nREPL
```

Then from the REPL:

```clojure
(require '[valley.game])
(valley.game/start!)

;; To stop:
(valley.game/stop!)
```

The game launches on a background daemon thread. If it crashes, check
`(deref valley.game/last-crash)` or `/tmp/valley_crash.log`.

## Controls

| Key | Action |
|-----|--------|
| W / A / S / D | Move forward / left / back / right |
| Mouse | Look around |
| Space | Jump (walk mode) / fly up (fly mode) |
| Shift | Fly down (fly mode) |
| F | Toggle fly / walk mode |
| Left click | Break block / attack mob |
| Right click | Place block / eat food |
| 1-9 / scroll wheel | Select hotbar slot |
| Escape | Quit |

## Architecture

### Typed numerical core (`raster.core/deftm`)

Performance-critical kernels (physics, collision, terrain sampling, ray
intersection) are written with `deftm` in `core.clj`. The Raster walker
devirtualizes and inlines these into single JVM methods -- no boxing, no
megamorphic dispatch in the hot path. The game layer extracts typed data
from Clojure maps, calls the kernels, and reassembles state.

### Datahike world state

Entity state (player, mobs, items) lives in a Datahike database, queryable
via Datalog. Chunk block data lives in a separate persistent map with
copy-on-write semantics. The two are combined in a unified `WorldState`
container (`state.clj`) that supports fork (immutable snapshots) and reset.

### Background chunk generation

A 4-thread pool generates terrain chunks in the background using a
Minetest-style emerge pattern. Completed chunks are posted to a
`ConcurrentLinkedQueue` and integrated into the world on the main thread.

### Hot reload

Shaders, HUD, and game logic can be hot-reloaded from the REPL without
restarting the window. Require any namespace with `:reload` to pick up
changes.

## REPL development

Save and restore game state:

```clojure
(def save @valley.game/game-state-ref)
(reset! valley.game/game-state-ref save)
```

Take a screenshot:

```clojure
(require 'raster.vk.screenshot)
(raster.vk.screenshot/request-screenshot! "/tmp/shot.png")
```

Frame profiler:

```clojure
(valley.game/prof-report)       ;; per-phase timing summary
(valley.game/prof-slow-frames)  ;; last 10 frames exceeding 16ms
(valley.game/prof-reset!)       ;; clear profiler data
```

Hot reload a namespace:

```clojure
(require '[valley.hud] :reload)
```

## File overview

| File | Purpose |
|------|---------|
| `game.clj` | Main loop, frame profiler, chunk emerge pool, `start!`/`stop!` |
| `state.clj` | Unified world state: Datahike entities + CoW chunk storage |
| `core.clj` | Typed numerical kernels (physics, geometry, terrain queries) |
| `world.clj` | Block types, chunk storage, coordinate math |
| `terrain.clj` | Biome-based procedural terrain generation |
| `mesher.clj` | Face-culling mesh generation for voxel chunks |
| `lighting.clj` | Sunlight propagation and point light BFS |
| `textures.clj` | Procedural block texture generation |
| `entity.clj` | Entity system: player/mob state, damage, healing, death |
| `mobs.clj` | Passive mob spawning, AI, physics, drops |
| `hostile.clj` | Hostile mobs: zombies, skeletons, lurkers, spiders |
| `mob_models.clj` | Box-mesh generation for cuboid mob assemblies |
| `hud.clj` | HUD rendering: health hearts and hotbar overlay |
| `inventory.clj` | Player inventory: hotbar (9 slots) + storage (27 slots) |
| `items.clj` | Item registry: block drops, tool types, mining speeds |
| `crafting.clj` | Shaped 3x3 crafting recipes with offset and mirror matching |
| `hunger.clj` | Hunger system: exhaustion, saturation drain, hunger drain |
| `farming.clj` | Farmland blocks, crop growth, planting, harvesting |
| `persistence.clj` | Save/load: Datahike for metadata, konserve for chunks |
