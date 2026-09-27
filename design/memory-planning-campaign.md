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
the nested pure two-exit spelling now uses the same ordered while-fold as the single-exit form.
Counted effect loops now retain branch-local carried recurrence tuples when every arm proves
the same unit step; mixed effectful early exits remain open. Those are
real workload oracles for the broader compiler agenda, not reasons to postpone memory planning
until a cluster scheduler exists. Revisit external JAX/MLIR/Mojo and scientific/LLM baselines at
measured milestones, not in every edit/test cycle.

## Execution-order witness

`LinkPlan/memory-report` records source step submission order. The selected executable can expand
one step into several kernels, and graph recording may lift cacheable constant transforms into a
one-time prologue. `gpu/graph-execution-order` and `gpu.link/execution-order` report those selected
record-time and per-replay kernel partitions without exposing backend graph handles. This is a
necessary correction to source-order liveness, not a completion certificate: event boundaries,
escape/AD retention, full initialization, and alias realization still gate actual reuse.
Straight-line equation-first prepared programs now compose the actual bound kernel order with
their source equation indices through the same linked witness. Host-only equations keep their
indices but launch no kernels. Structured program loops explicitly decline this flat report;
preparation order is not a substitute for their repeated execution order.

The shadow planner can consume the linked witness. It declines a proposed reuse when either
allocation is touched by a one-time prologue (including a mixed prologue/replay semantic step),
when the source step cannot be matched, or when the selected replay order overlaps. A witnessed
per-replay ordering still leaves cross-replay full initialization, completion/escape, and physical
alias realization open. In particular, a prologue-produced temporary read on each replay must
not be recycled for a later temporary in the same replay: it would corrupt the next invocation.
Only an ABI-certified complete first overwrite of both single-view allocations, in the witnessed
per-replay order, discharges the cross-replay initialization obligation. A legacy descriptor's
write permission alone never does. Completion/escape and physical alias realization remain open.

A four-layer typed dense program exercises the intersection without injecting write facts:
the first and third intermediate arrays have certified complete overwrites and disjoint selected
replay lifetimes. Two numerical replays on OpenCL and Level Zero check that observation changes
neither buffers nor results on the ordinary resident path.

Before enabling that candidate, establish a nonescape contract: `node-view`, `value-view`, and
`download` currently expose even internal storage. An internal role alone is not exclusivity.
Perform any alias rewrite before binding, validate its alias/effect obligations, and verify that
the selected order agrees with the proof. Synchronous completion does not authorize reuse while
an external view or AD tape can still observe the old contents. Existing JVM/source lifetime
passes must remain until their successful cases have equivalent coverage.

## First realized reuse: confined host-result execution

`gpu.link/evaluate!` is an opt-in, owned, host-result-only boundary for a straight-line
equation-first plan. It projects prebind order from validated, selected equation graphs,
realizes one full-allocation single-view pair with the same complete allocation contract
(including coherence), and revalidates aliases/effects. Its single binding must have the same
actual order and the predicted owned allocation count before execution. It then uses the
existing synchronous runner and downloads only declared outputs; no session, callback or
resident view escapes. Ordinary `instantiate!` and its inspectable internal views are unchanged.

The tiny four-layer oracle saves one 16-byte allocation on both OpenCL and Level Zero, with
two optimized replays matching the JVM. This proves a local storage decision, **not** broad
model savings or a measured latency improvement. The former baseline binding is removed;
runtime verification remains mandatory. No-candidate plans retain distinct storage. Public outputs, host-initialized
buffers, borrowed storage, incompatible coherence and partial-allocation views are excluded.

`gpu.link/private-executor!` extends this same host-result boundary across repeated invocations.
It prepares/binds once, keeps private storage resident, and accepts checked host input/state
updates. All updates are validated before upload, including rejection of overlapping views.
Invocation and close are serialized; runtime failures close the scope, while invalid requests
leave it unchanged. The callable is not an inspectable `LinkedExecutable`. Each result remains
a detached host snapshot, valid after later invocations or close. `evaluate!` delegates to this
scope rather than maintaining a second lifecycle implementation.

This is not yet a zero-copy resident-output facade. The equation-first `Compiled` boundary now
certifies all exposed result, explicit-output, donation and tap nodes as plan outputs before
initialization and escape analysis. Certification cannot drop the original semantic outputs.
This is physical-storage retention, not a snapshot of an earlier SSA version: two compiler names
sharing one storage identity still need a separate value-version/tape proof. In particular,
the existing `Compiled` wrapper exposes its executable and cannot silently opt into private
reuse. Resident leases, structured replay and AD tape lifetime integration remain open. This
GPU-local proof does not supersede the JVM/source memory passes, so none is deleted on that claim.

## Semantic value retention

The invocation memory witness follows each semantic buffer definition through later equation
uses. It distinguishes an equation's read phase from its result-write phase, retains observable
storage through the call boundary, and excludes storage-only destination names from the set of
semantic versions. The same ordinary use information covers values produced in a forward pass
and consumed by generated backward equations; no separate AD tape-name registry is needed.

This report is not a selected execution-order or completion certificate. Persistent public
storage remains excluded from reuse, and unresolved mutations, partial writes, overlapping
versions/views and structured control must remain explicit unknowns. Physical escape still does
not identify which earlier mutable value a caller intended to retain. The confined host-result
executor now requires the aggregate witness before attempting its one-pair reuse. Unknown
retention (or a plan lacking the invocation certificate) preserves distinct storage and reports
the reason; it does not reject otherwise valid execution. Selected order, complete initialization,
alias validation, nonescape and synchronous completion remain independent gates. Resident
leases, escaped pullbacks and version-preserving copies remain separate obligations.

The ordinary, inspectable `LinkedExecutable` now has an explicit synchronous output lease. A
successful replay makes its owned outputs leaseable; replay, upload, write and close are serialized
against live leases, and a failed replay or subsequent mutation invalidates the completed-result
claim. Attached sessions and borrowed/external outputs decline. Releasing the lease permits the
next replay; it is not a copy or an earlier-version snapshot. This protects output views used
through the Link API; `Compiled` guards its input donation and output-wrapper invalidation under
the same lease lock before replay or close. It does **not** authorize private temporary reuse:
callers can still inspect internal views and the session, and direct session mutation lies outside
this lease. An opaque composition capability plus asynchronous event/AD-tape retention is required
before a private resident-result executor can safely realize reuse while outputs escape.

Session-owned asynchronous range transfers also retain their resident buffer registrations.
Submission records the validated buffer keys, physical allocation identities and buffer objects;
`free-buffer!` rejects release through any of those aliases until `await-event!` has established
completion and released backend staging. A nonblocking `event-complete?` observation is not a
release certificate. Session close continues to drain outstanding events. This closes a concrete
use-after-free path, but it does not yet provide a general read/write dependency scheduler for
other operations submitted against the same ranges while a transfer is pending.

The session event boundary now also rejects a bound `KernelGraph` submission against a pending
range transfer that names the same resident allocation, and rejects the reverse submission order.
The binding carries physical allocation and buffer identities, including the single-kernel-call
path; disjoint buffers may still use independent queues. This is a conservative whole-allocation
hazard check, not range-aware dependency scheduling or automatic wait insertion. Direct legacy
session `replay!` and synchronous transfer calls remain outside this event-order contract; their
eventual retirement or unification must be explicit before claiming general cross-queue safety.

`Compiled/invoke-leased` now projects its ordinary external `DeviceArray` results and acquires
the Link output lease in one locked invocation. Only declared LinkPlan outputs may be projected;
ownership is rejected before input writes or donation, and exactly one completed replay is
required. Closing the lease invalidates the returned wrappers, not the session-owned buffers. A
real equation-first effect map on CPU OpenCL checks
the value before release, rejects an intervening replay/close, and runs again after release.
This still leaves the existing `Compiled` wrapper inspectable and makes no alias-reuse claim.
