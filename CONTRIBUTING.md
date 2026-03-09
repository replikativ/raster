# Contributing to Raster

Thank you for your interest in contributing to Raster.

## Getting Started

1. Clone the repository
2. Install JDK 25+ (or JDK 27 for Valhalla value types)
3. Run `clojure -M:test` to verify everything works

For Valhalla features, see `valhalla-env.sh` and the Valhalla section in `CLAUDE.md`.

## Development

### Key conventions

- **Always use `deftm`/`ftm` for numerical code**, never `defn`/`fn`. See `CLAUDE.md` for exceptions.
- **No Clojure type hints** (`^double`, `^long`, etc.) — Raster's dispatch system handles specialization via `:- Type` annotations.
- Run `clojure -M:test` before submitting changes.

### REPL

```bash
clojure -M:repl     # start nREPL on port 7899
clojure -M:nREPL    # with cider-nrepl middleware
```

Use `:reload` (not `:reload-all`) when requiring `raster.core` to avoid ClassCastException from MethodEntry recompilation.

### Tests

```bash
clojure -M:test                           # run all tests
clojure -M:test -e :exec-fn -- -n raster.core-test  # single namespace
```

### Benchmarks

```bash
clojure -M:bench -m raster.abm-bench     # FIRMS benchmark
```

Julia comparison benchmarks are in `bench/` and require a Julia installation.

## Code Organization

- `src/raster/core.clj` — central dispatch system, walker, deftm macro
- `src/raster/compiler/` — nanopass compiler pipeline
- `src/raster/dl/` — deep learning modules
- `src/raster/abm/` — agent-based simulation
- `src/raster/vk/` — Vulkan rendering engine
- `test/raster/` — tests mirror source structure

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
