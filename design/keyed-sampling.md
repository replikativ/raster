# Keyed distribution sampling

`sample` has an explicit `(distribution, seed, counter)` overload for Normal,
Uniform, and Exponential. It is a pure draw: revisiting the same key returns the
same value regardless of scheduling. The older one-argument, thread-local
sampling overloads remain convenience APIs, not replayable compiler inputs.

The counter is a caller-owned draw identity, not a global generator position.
For a Normal draw, Raster reserves counters `2 * counter` and
`2 * counter + 1` for Box–Muller; callers must avoid counter overflow and key
separate sites or streams with distinct seeds. A Spindel bridge can derive the
seed from its stable structural site identity and run seed, while Raster owns
only the numerical draw. Replay-vs-fresh policy stays above this API.

`raster.par/uniform-open01` converts SplitMix64's upper 52 bits into an open
interval value, so `log(u)` never sees zero. The integer stream and uniform
conversion are specified here; transcendental library differences mean that
Normal and Exponential values may be numerically, but not bitwise, equal across
targets. No seeded overload is claimed for the other distributions yet.
