# raster cljs / browser sandbox

Scratch experiments for the raster → ClojureScript / browser port. Each
experiment de-risks one unknown from `.internal/cljs-browser-port-design.md`.
Nothing here is wired into raster proper — it's a playground.

## Experiment 1 — wasm numeric codegen (runs under node, no browser)

The load-bearing one: can we emit a fast f64 numeric kernel as WebAssembly from
a "compiler-shaped" generator (binaryen.js), the way raster emits JVM bytecode
today? Builds a dot-product kernel scalar + SIMD128 `f64x2`, validates vs JS.

```bash
npm install
npm run wasm:dot
```

Last local run (node 23): wasm scalar ~1.24x and SIMD `f64x2` ~1.64x faster than
the JS scalar baseline, identical sums. Proves emit→optimize→validate→instantiate
+ the SIMD128 path + f64-on-CPU + shared-linear-memory views all work.

## Experiment 2 + 3 — WebGPU compute + WebGL2 (browser)

- **WebGPU saxpy (`src/sandbox/webgpu_compute.cljs`)** — `y = a*x + y` as a WGSL
  compute shader (f32, the GPU numeric identity raster must adopt), checked vs a
  CPU reference. The shape every raster `par/map!` kernel takes on WebGPU.
- **WebGL2 triangle (`src/sandbox/webgl_triangle.cljs`)** — minimal render-path
  interop check (the greenfield renderer target; `raster.vk` does not port).

```bash
npm install
npx shadow-cljs watch app
# open http://localhost:8080  (Chrome/Edge, or Safari 26 / Firefox Nightly for WebGPU)
```

WebGPU availability in 2026: Chrome/Edge default; Safari 26+; **Firefox Linux is
Nightly-only** (so on this dev box, test WebGPU in Chrome). The page degrades
gracefully and logs why if `navigator.gpu` is missing.

## Why these three

They map 1:1 to the port's three top unknowns: wasm CPU backend viability +
dispatch (Exp 1), f32-on-GPU compute correctness (Exp 2), and the render seam
(Exp 3). See the design doc for the full architecture.
