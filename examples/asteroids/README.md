# Geometric Asteroids

A typed dispatch showcase disguised as a game.

![Geometric Asteroids](screenshot.png)

All shapes are regular polygons with typed dispatch via `deftm`. When shot,
each polygon decomposes into simpler shapes following a geometric hierarchy:

```
Octagon → 2 Squares
Hexagon → 3 Pentagons
Pentagon → Square + Triangle
Square  → 2 Triangles
Triangle → destroyed
```

Each shape type is a `defvalue` with `deftm` methods for `split`, `sides`,
and `score-value`. New polygon types can be added from the REPL during
gameplay — define the type, register its dispatch methods, and it appears
in the next wave.

## Running

```bash
source valhalla-env.sh
clojure -J--enable-preview \
  -J--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED \
  -J--enable-native-access=ALL-UNNAMED \
  -Sdeps '{:paths ["src" "examples" "classes"]}' \
  -M:vulkan -e '(require (quote [asteroids.core :as ast])) (ast/start!)'
```

Arrow keys to move, Space to shoot, Enter to restart.

## Aesthetics

Matplotlib tab10 color palette on black — each polygon type has a distinct
color (blue octagon, orange hexagon, green pentagon, red square, purple
triangle). Serif font HUD. All graphics are procedurally generated.

## Hot-reload from REPL

```clojure
;; Add a Heptagon during gameplay:
(ns asteroids.shapes)
(defrecord Heptagon [x y vx vy angle spin size])
(deftm sides [s :- Heptagon] :- Long 7)
(deftm score-value [s :- Heptagon] :- Long 18)
(deftm split [s :- Heptagon]
  ;; splits into 2 Pentagons + 1 Square
  ...)
(swap! shape-ctors assoc :heptagon ->Heptagon)
(swap! spawn-pool conj :heptagon)
;; Clear the wave — heptagons appear!
```

## Architecture

- `shapes.clj` — `defvalue` types, `deftm` dispatch (split/sides/score),
  procedural texture generation, shape movement
- `core.clj` — Vulkan game loop, per-type texture registry, batched
  rendering, collision detection, wave management
