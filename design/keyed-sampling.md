# Keyed distribution sampling

`sample` has an explicit `(distribution, seed, counter)` overload for Normal,
Uniform, Exponential, Gamma, Poisson, and Beta. It is a pure draw: revisiting the same key returns the
same value regardless of scheduling. The older one-argument, thread-local
sampling overloads remain convenience APIs, not replayable compiler inputs.

The counter is a caller-owned draw identity, not a global generator position.
For a Normal draw, Raster reserves counters `2 * counter` and
`2 * counter + 1` for Box–Muller; callers must avoid counter overflow and key
separate sites or streams with distinct seeds. A Spindel bridge can derive the
seed from its stable structural site identity and run seed, while Raster owns
only the numerical draw. Replay-vs-fresh policy stays above this API.

Rejection-based samplers first derive a site seed with `splitmix64(seed,
counter)`. Gamma uses separate normal and acceptance substreams per attempt,
plus an independent transform draw for shapes below one. Poisson uses a local
counter for its small-rate product; Beta gives each Gamma component a separate
child seed. A rejection at one site therefore never shifts another site's
stream. Gamma's CPU sampler and keyed sampler share the Marsaglia–Tsang log
acceptance expression, including its essential `+ log(v)` term.

`raster.par/uniform-open01` converts SplitMix64's upper 52 bits into an open
interval value, so `log(u)` never sees zero. The integer stream and uniform
conversion are specified here; transcendental library differences mean that
Normal and Exponential values may be numerically, but not bitwise, equal across
targets. The keyed Gamma/Poisson/Beta overloads establish replayable host
semantics; this does not claim their unbounded rejection loops are an optimized
GPU route. Remaining distributions retain only their one-argument API until
their draw structure and target lowering are validated.
