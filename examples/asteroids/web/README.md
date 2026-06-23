# Geometric Asteroids — browser playground

The **same** game runs on the JVM and in the browser. The game logic lives in
[`../game.cljc`](../game.cljc) (immutable, portable `defn` glue), and the physics
— integrating each shape's position + angle with toroidal wrap — is a raster
`deftm` value-type kernel in [`../kernels.clj`](../kernels.clj):

- **JVM**: `deftm` → typed dispatch → bytecode.
  - Vulkan shell [`../vk_main.clj`](../vk_main.clj) — renders through the portable
    renderer (`raster.render` + the Vulkan backend). Run from the repo root:
    `clojure -M:vulkan -m asteroids.vk-main`.
  - Swing/Java2D shell [`../jvm.clj`](../jvm.clj) (no GPU deps):
    `clojure -Sdeps '{:paths ["src" "examples"]}' -M -m asteroids.jvm`.
- **Browser**: `deftm` → **WebAssembly** (via `raster.compiler.cljs-emit`), rendered
  through **WebGPU** ([`src/asteroids/web_gpu.cljs`](src/asteroids/web_gpu.cljs)). The
  generated `asteroids.kernels` cljs namespace marshals the `Shape` value type over
  the wasm boundary; `raster.wasm` (in the project `src/`) is the loader. A Canvas2D
  shell ([`src/asteroids/web.cljs`](src/asteroids/web.cljs)) remains as a no-WebGPU
  fallback (point the `:casteroids` build's `:init-fn` at `asteroids.web/init`).

The browser and the JVM Vulkan shell render through the **same** scene code
([`../render.cljc`](../render.cljc), `asteroids.render`) against the **same**
portable `raster.render` protocol — only the backend differs (`raster.render.webgpu`
vs `raster.render.vulkan`). `asteroids.game` requires `asteroids.kernels` and gets
the right physics per platform — the `.clj` `deftm` on the JVM, the generated `.cljs`
(wasm) in the browser.

## Play in the browser

```bash
cd ../../.. && clojure -Sdeps '{:paths ["src" "examples"]}' -M examples/asteroids/web/gen.clj  # once: generate the wasm kernels
clojure -Sdeps '{:paths ["src" "examples"]}' -M examples/valley/gen.clj                        # (and the valley demo's)
cd examples/asteroids/web
npm install                      # once — fetches shadow-cljs
npx shadow-cljs watch casteroids # then open http://localhost:8080
```

`↑` thrust · `← →` rotate · `space` fire. The wasm kernels + generated `kernels.cljs`
are build artifacts (gitignored) — run `gen.clj` once on a fresh checkout (above) before
`shadow-cljs`. The Pages deploy (`.github/workflows/pages.yml`) does this from source.

For an optimized bundle: `npx shadow-cljs release casteroids`, then serve `public/`.

## Regenerate the wasm kernels

After editing [`../kernels.clj`](../kernels.clj), recompile the kernels (needs the
JVM + raster's deps), run from the **repo root**:

```bash
clojure -Sdeps '{:paths ["src" "examples"]}' -M examples/asteroids/web/gen.clj
```

This rewrites `public/kernels.wasm` and `src/asteroids/kernels.cljs`.

## Layout

```
web/
  deps.edn           cljs deps; paths to ../game.cljc and project src/ (raster.wasm)
  shadow-cljs.edn    :casteroids browser build
  package.json       shadow-cljs (npm)
  gen.clj            regenerate kernels.wasm + kernels.cljs from ../kernels.clj
  public/
    index.html       canvas + HUD; loads /js/main.js and /kernels.wasm
    kernels.wasm     GENERATED — the compiled physics kernel
  src/asteroids/
    web.cljs         Canvas2D shell: render, DOM input, rAF loop, kernel load
    kernels.cljs     GENERATED — cljs counterpart of asteroids.kernels
```
