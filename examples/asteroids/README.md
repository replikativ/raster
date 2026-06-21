# Geometric Asteroids

A typed-dispatch + cross-platform showcase disguised as a game.

![Geometric Asteroids](screenshot.png)

All asteroids are regular polygons. When shot, each decomposes into simpler
shapes following a geometric hierarchy (`split-rule` in `game.cljc`):

```
Octagon  → 2 Pentagons-worth of Squares   (8 → [4 4])
Hexagon  → 3 Pentagons                     (6 → [5 5 5])
Pentagon → Square + Triangle               (5 → [4 3])
Square   → 2 Triangles                     (4 → [3 3])
Triangle → destroyed                       (3 → nil)
```

**One `.cljc` game, two platforms.** `asteroids.game` (logic) + `asteroids.render`
(scene) are portable Clojure(Script). The physics — integrating each shape's
position/angle with toroidal wrap — is `asteroids.kernels/move-shape`, a value-type
`deftm` that compiles to **JVM bytecode** on the desktop and to **WebAssembly** in the
browser. The renderer is the portable `raster.render` protocol with **Vulkan** and
**WebGPU** backends driven by one shader source.

## Running

**Desktop (Vulkan):**
```bash
clojure -Sdeps '{:paths ["src" "examples"]}' -M:vulkan -m asteroids.vk-main
```

**Desktop (no native deps — Swing/Java2D):**
```bash
clojure -Sdeps '{:paths ["src" "examples"]}' -M -m asteroids.jvm
```

**Browser (WebGPU — Chrome/Edge):** see [web/README.md](web/README.md):
```bash
cd web && npx shadow-cljs watch casteroids   # http://localhost:8080
```

Arrow keys / WASD to move, Space to shoot, Enter to restart.

## Aesthetics

Matplotlib tab10 palette on black — each polygon size has a distinct colour. Serif-font
HUD. All graphics are procedurally generated.

## Adding a shape live from the REPL (JVM)

On the JVM each polygon is a `defvalue` and `sides`/`score-value`/`split`/`move-shape`
dispatch on its type — so you can define a new type *and its methods* while the game is
running and the next wave uses it, no recompile:

```clojure
(in-ns 'asteroids.shapes)
(defvalue Heptagon [x :- Double, y :- Double, vx :- Double, vy :- Double,
                    angle :- Double, spin :- Double, size :- Double])
(deftm sides       [s :- Heptagon] :- Long 7)
(deftm score-value [s :- Heptagon] :- Long 18)
(deftm split [s :- Heptagon]                       ; heptagon → pentagon + square
  [(child ->Pentagon (scatter (:x s) (:y s) (:vx s) (:vy s) (* 0.7 (:size s)) 0 2))
   (child ->Square   (scatter (:x s) (:y s) (:vx s) (:vy s) (* 0.7 (:size s)) 1 2))])
(deftm move-shape [s :- Heptagon, dt :- Double, w :- Double, h :- Double]
  (->Heptagon (:x s) (:y s) (:vx s) (:vy s) (+ (:angle s) (* (:spin s) dt)) (:spin s) (:size s)))
(swap! shape-ctors assoc 7 ->Heptagon)             ; spawnable…
(swap! spawn-pool conj 7)                           ; …and added to the wave pool
```

This is JVM-only: ClojureScript has no `deftm`/`defvalue` runtime, so the browser build
uses the equivalent **data form** in `game.cljc` (`split-rule` + `data-score`), which the
same shape set compiles into for WebAssembly.

## Architecture

| File | Purpose |
|------|---------|
| `game.cljc` | Portable game logic — loop, collision, scoring, ship, bullets; the `sh-*` seam (JVM dispatch / cljs data) |
| `shapes.clj` | **JVM** typed-dispatch shape hierarchy (`defvalue` + `deftm`); live REPL extension |
| `kernels.clj` | `move-shape` value-type `deftm` kinematics → bytecode (JVM) / wasm (browser) |
| `render.cljc` | Portable scene description (procedural n-gon geometry) for the renderer protocol |
| `vk_main.clj` | JVM entry — Vulkan backend |
| `jvm.clj` | JVM entry — Swing/Java2D backend (no native deps) |
| `web/` | Browser build (shadow-cljs): WebGPU renderer + generated wasm kernels |
