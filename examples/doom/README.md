# Doom WAD Renderer (WIP)

A Vulkan-based renderer for classic Doom WAD files, featuring BSP traversal,
sprite rendering, monster AI, HUD, sound, and music.

**Status: Work in progress.** The renderer loads and displays Doom levels with
textured walls, sprites, and monsters. Some rendering bugs remain (texture
alignment, sprite clipping). Contributions welcome — the REPL + screenshot
workflow makes it easy to iterate on shaders and rendering interactively.

## Raster Features

Physics and geometry kernels use `deftm` for typed dispatch:
- `point-side-cross` — BSP partition cross product
- `line-seg-dist` — point-to-segment distance (collision)
- `doom->vk-x/y/z` — coordinate system transforms
- `wall-length` — wall segment length

The game logic (monster AI, door/lift state machines, weapon handling) uses
plain `defn` — it's data-driven from WAD files, not numerically intensive.

## Requirements

You need a Doom WAD file. Options:

- **Shareware DOOM1.WAD** — freely distributable, available from many mirrors
- **FreeDoom** — BSD-licensed replacement WADs: https://freedoom.github.io/
- **Commercial** — `DOOM.WAD` or `DOOM2.WAD` from your Steam/GOG copy

On Debian/Ubuntu:
```bash
sudo apt install doom-wad-shareware   # installs to /usr/share/games/doom/doom1.wad
# or
sudo apt install freedoom             # installs FreeDoom WADs
```

## Running

```bash
source valhalla-env.sh
clojure -J--enable-preview \
  -J--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED \
  -J--enable-native-access=ALL-UNNAMED \
  -Sdeps '{:paths ["src" "examples"]}' \
  -M:vulkan -m doom.run textured /usr/share/games/doom/doom1.wad E1M1
```

## Architecture

| Module | Purpose |
|--------|---------|
| `wad.clj` | WAD binary parser, texture/sprite loading |
| `bsp.clj` | BSP tree queries, collision sliding (`deftm`) |
| `level.clj` | Geometry builder — walls, floors, ceilings (`deftm`) |
| `player.clj` | Player state, weapons, hitscan |
| `monster.clj` | Monster AI, pathfinding, combat |
| `sector.clj` | Door/lift/damage state machines |
| `sprite.clj` | Billboard sprite rendering |
| `hud.clj` | HUD overlay rendering |
| `sound.clj` | WAD sound → OpenAL |
| `music.clj` | MUS → MIDI conversion + playback |
