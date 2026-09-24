# Memory planning campaign: evidence before reuse

This is an implementation ledger for the memory part of the compiler north star. It does not
replace the TypedSOAC, KernelBody, scheduling, AD, or distributed-planning campaigns. One local
device is the smallest topology; the same value, physical-instance, access, and completion facts
must remain meaningful when a plan spans devices and transfers.

## Invariants

- A logical value and a physical allocation are different identities. A value can have several
  ordered physical leaves; views can share an allocation with different byte ranges.
- Read/write permissions are not proof of a complete write, exclusive ownership, deadness, or
  completion. An AD tape, exported output, borrowed view, or in-flight transfer keeps storage live
  until its actual consumer/release condition is satisfied.
- Scheduling may choose placement and transfer routes, but may not silently alter value semantics.
  The ownership proof and runtime event/lease contract must agree before reusing storage.
- Unknown call effects, aliasing, branch paths, or event completion prevent reuse. They must be
  reported as unknown rather than guessed from source spelling or ABI naming.

## Landing sequence

1. **Observe.** Report stable LinkPlan value/node/allocation identities, byte ranges, locations,
   ownership, and ordered certified accesses. Label reuse, release, and completion *unproven*.
   Differential tests assert that the report matches LinkPlan effect evidence. No allocation or
   surface-language behavior changes.
2. **Connect witnesses.** Carry value/place/access facts from typed program through schedule,
   LinkPlan, and execution events. Add explicit value versions and completion dependencies where
   the existing IR lacks them. Challenge the model with a resident AD train step, Laya QKV views,
   an async transfer, and a city-style branch/loop. Do not copy the S-expression alias inference
   into a new pass.
3. **Shadow-plan.** Produce `reuse`, `retain`, `copy`, and `unknown` proposals with reasons, but do
   not execute them. Compare these against the current buffer-fusion, memory-merge, and hoisting
   decisions. Measure compile time and planned peak bytes; keep counterexamples as tests.
4. **Enable one verified local reuse.** Require range-disjointness or non-overlapping completed
   lifetimes, exclusive ownership, initialized-read obligations, and AD tape safety. Differential
   numerical and resident-graph tests gate it. Cross-component donation and distributed reuse are
   separate steps, not inferred from this local proof.
5. **Retire duplicates.** Remove old syntactic lifetime/reuse guesses only after the new evidence
   covers their successful cases or an explicit unsupported diagnostic replaces them. Then add
   topology-aware placement/transfer costs and memory-capacity constraints to the existing
   DistributedPlan rather than a second scheduler.

## Surface-semantics change rule

Steps 1–3 change no `deftm` semantics. For any later proposed semantic change, record before/after
examples and why it is necessary, then add CPU/GPU differential tests *before* enabling it. In
particular, physical padding or reuse must not change a source-visible `alength`; mutation/donation
must not invalidate a value unless the language explicitly exposes that transfer of ownership.
Compatibility can be removed, but never by an undocumented semantic shift.

## Rhythm and other work

Land one invariant or one vertical per PR. Run focused tests in a warm REPL, then let CI run the
full suite; respond to red CI without making the local hot loop depend on it. At each behavior
change record correctness, compile time, peak resident bytes, copy count, and relevant kernel
latency against a frozen baseline. Merge green, reviewable slices rather than stacking broad
refactors.

After the observation and witness seams, alternate memory slices with the existing city typed
effectful-loop/helper/constant work and Laya attention/quantized-kernel benchmarks. The immediate
city gate is the day-kernel episode loop with a guarded inner search followed by an atomic effect;
the nested two-exit spelling has an equivalent supported single-exit form and is a later language
ergonomics task. Those are
real workload oracles for the broader compiler agenda, not reasons to postpone memory planning
until a cluster scheduler exists. Revisit external JAX/MLIR/Mojo and scientific/LLM baselines at
measured milestones, not in every edit/test cycle.

## Execution-order witness

`LinkPlan/memory-report` records source step submission order. The selected executable can expand
one step into several kernels, and graph recording may lift cacheable constant transforms into a
one-time prologue. `gpu/graph-execution-order` and `gpu.link/execution-order` report those selected
record-time and per-replay kernel partitions without exposing backend graph handles. This is a
necessary correction to source-order liveness, not a completion certificate: event boundaries,
escape/AD retention, full initialization, and alias realization still gate actual reuse. An
equation-first prepared program has a separate runner and explicitly declines this report until
that runner supplies equivalent evidence.

The shadow planner can consume the linked witness. It declines a proposed reuse when either
allocation is touched by a one-time prologue (including a mixed prologue/replay semantic step),
when the source step cannot be matched, or when the selected replay order overlaps. A witnessed
per-replay ordering still leaves cross-replay full initialization, completion/escape, and physical
alias realization open. In particular, a prologue-produced temporary read on each replay must
not be recycled for a later temporary in the same replay: it would corrupt the next invocation.
