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

Interactive games built on the `raster.vk` Vulkan engine.

| Example | Description | LOC |
|---------|------------|-----|
| [Asteroids](asteroids/) | Classic arcade — 2D sprites, audio, keyboard | 758 |
| [Doom](doom/) | WAD renderer — BSP, sprites, monsters, HUD, music | 4,920 |
| [Valley](valley/) | Voxel survival — procedural terrain, crafting, mobs | 7,226 |

All games use the `:vulkan` deps.edn alias for LWJGL dependencies.

### Running Games

```bash
source valhalla-env.sh
clojure -J--enable-preview \
  -J--add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED \
  -J--enable-native-access=ALL-UNNAMED \
  -Sdeps '{:paths ["src" "examples"]}' \
  -M:vulkan -m <entry-point>
```

Entry points:
- Asteroids: `-m asteroids`
- Doom: `-m doom.run`
- Valley: `-m valley.game`

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
