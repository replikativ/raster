# Raster Examples

## Notebooks

Interactive notebooks (Kindly/Clay compatible) are in [`notebooks/raster/`](../notebooks/raster/).
Run them with Clay:
```clojure
(require '[scicloj.clay.v2.api :as clay])
(clay/make! {:source-path "notebooks/raster/getting_started.clj"})
```

## Deep Learning Reference Models

| Example | Description |
|---------|------------|
| [LeNet-5](raster/dl/lenet.clj) | Compiled CNN on MNIST |
| [GPT-2](raster/dl/gpt2.clj) | Transformer language model |
| [BERT](raster/dl/bert.clj) | Masked language model |
| [Matrix Factorization](raster/dl/matfac.clj) | Collaborative filtering |

Also: [Reaction-Diffusion](reaction_diffusion/) — Gray-Scott pattern formation.

## Games

| Example | Description |
|---------|------------|
| [Asteroids](asteroids/) | Geometric Asteroids — polygon shapes via `defvalue`/`deftm`; cross-platform |
| [Doom](doom/) | WAD renderer — BSP, sprites, HUD (WIP) |
| [Valley](valley/) | Streaming voxel world — biome terrain, caves, ores, trees, lighting, water, mobs; cross-platform |

**Asteroids** and **Valley** are cross-platform: one `.cljc` codebase runs on the JVM
(Vulkan + `deftm`→bytecode) and in the browser (WebGPU + `deftm`→WebAssembly). See
[asteroids/web/README.md](asteroids/web/README.md) for the browser builds.

### Running (JVM)

```bash
clojure -Sdeps '{:paths ["src" "examples"]}' -M:vulkan -m <entry-point>
```

Entry points:
- Asteroids (Vulkan): `-m asteroids.vk-main` — or `-m asteroids.jvm` for a no-native-deps Swing window
- Doom: `-m doom.run`
- Valley (Vulkan): `-m valley.vk-walk`

(The `:vulkan` alias pulls in the LWJGL dependencies. The Valhalla JDK is optional —
needed only for the `Dual4` value-type fast paths, not for the games.)

### Architecture

The games are built on `raster.vk.*`, a data-oriented Vulkan wrapper via LWJGL:

- `raster.vk.gpu` — GPU context (instance, device, queues, allocator)
- `raster.vk.frame` — fixed-timestep game loop with compute/render phases
- `raster.vk.pipeline` — graphics pipeline construction
- `raster.vk.texture` — image loading and sampler management
- `raster.vk.mesh` — vertex/index buffer management
- `raster.vk.input` — keyboard/mouse via GLFW
- `raster.vk.audio` — positional audio via OpenAL

Valley also uses `raster.noise` for procedural terrain generation (Perlin noise,
FBM, ridged noise — implemented with `deftm` for typed dispatch).
