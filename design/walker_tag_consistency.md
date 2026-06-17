# Walker / binding-tag consistency — gaps & design notes

Tracking the compiler-correctness work behind **Option X** (TypedClojure exposes
per-binding types on the checked AST; raster consumes them off `:checked-ast`
instead of `::binding-types` metadata). Branch: `feature/tc-ast-binding-types`.
Depends on the updated `typed.clj.checker` (the `fix/let-binding-expr-types`
fork; PR typedclojure/typedclojure#195).

## Design recap: how a binding gets its JVM type

```
deftm body → inf/tc-analyze-deftm-body (TC once)
           → {:ret-tag, :binding-tags {sym → dispatch-tag}}
           → walker walk-form :let (walker.clj:467, also handles loop*)
               tc-tag = (get tc-binding-tags sym)          ; TC wins
               tag    = (or tc-tag (infer-binding-tag …))  ; structural fallback
               stamp :raster.type/tag on sym + ctx type-env
           → bytecode backend: :raster.type/tag → JVM local types, typed
             loads/stores; JVM verifier requires stack-map consistency
```

`infer-binding-tag` is **not** redundant with TC — it covers TC's genuine gaps:
parametric `(All [T])` concrete element types, `Object[]` element types, the
record/value `field-type-registry`, walker `fn-info`, deftm-call returns that
appear *mid-walk* (typed macros), and the TC-absent / TCError case. Where TC
speaks, TC wins (sound); the fallback fills the rest.

## GAP 1 — branch-merge representation inconsistency  [ROOT CAUSE, ACTIVE]

`case` / `if` branches whose results merge must produce a **uniform JVM
representation** (all primitive-`T`, or all boxed). Today each branch's result
type is decided **locally** by that branch's bindings' tags. When tags make one
branch's result primitive (e.g. `double`, 2 stack slots) and a sibling branch's
result boxed (ref, 1 slot), the operand stack has different sizes at the merge →
JVM verifier "stack size mismatch".

- Repro: `raster.linalg.core/matrix-norm` with `sum` (a `(loop …)`-bound binding
  in the Frobenius `case` branch) tagged `double` while sibling branches are
  boxed → mismatch at the `case` merge (GOTO → shared DRETURN).
- Pre-Option-X this was hidden: sparse TC tags left branches uniformly boxed.
- **Latent bug**: any deftm with a mixed typed/boxed `case`/`if` would hit it,
  independent of Option X.
- Sharper observation: even the *full* comprehensive tag set fails at the same
  merge (offset 255 = the 1-norm branch's GOTO to the shared DRETURN). That
  branch is a bare `(loop …)`; a `loop`'s result representation is **not coerced
  to the branch/merge context** regardless of the loop-var tag. So the fix is
  about typing the *tail result* of control-flow forms (loop / case-branch /
  if-branch) to the context's representation — not just tagging the bindings
  inside them.

**Confirmed precise mechanism (bytecode.clj):**
`emit-case*` decides the merge result type up front via `infer-arg-stack-type`
(a *static shadow predictor*, separate from the real `emit-form`). It computes
`uniform-prim?` from the branches it can infer, then for each emitted branch
boxes the result iff `(and (not uniform-prim?) (primitive? t))`. The flaw:
`infer-arg-stack-type` has **no `loop*` case** (and collapses all arrays to
`:ref`), so a `(loop …)` branch infers `nil`, is **filtered out** of the
uniformity vote, and the surviving sibling branches make `uniform-prim?` true →
the loop branch (whose *actual* emitted type is primitive `double`, 2 slots) is
**not boxed**, while the merge frame was set for the inferred primitive — and the
*other* untyped sibling left a 1-slot ref. Operand-stack slot counts diverge at
the merge → "stack size mismatch". Two sources of truth (`infer-arg-stack-type`
predicts; `emit-form` produces) that disagree is the real disease.

**Fix options considered:**
- **A — teach `infer-arg-stack-type` about `loop*`.** Fragile: predicting a
  loop's result needs the loop-var LUB logic that `emit-loop*` does (init+recur
  widening), duplicating emit. And it's whack-a-mole (next: `try`, …). Rejected.
- **B — target-type threading (the consistent-design north-star).** Give
  `emit-form` an expected/`target-type` from context (binding tag / fn return);
  control-flow forms propagate it to their tail position(s) and coerce. Single
  source of truth, eliminates the predict-vs-emit divergence class entirely.
  Larger refactor (thread target through emit-form's cases). **Tracked as the
  end-state; not done yet.**
- **D — conservative merge (VALIDATED, applied first).** In `emit-case*`, ignore
  statically-diverging branches (`throw`/`recur`), and require *every* remaining
  branch to have a KNOWN uniform primitive type; if any is unknown (e.g. a loop),
  fall to `:ref` and box all primitive branch results. Correct by construction
  (merge is uniform), localized, and the only cost is boxing a case result when a
  branch type is unpredictable — i.e. exactly the pre-Option-X behavior, so no
  perf regression vs. baseline. **Verified: `matrix-norm` passes all ords with
  full comprehensive tags.**

Decision: ship **D** for correctness now; keep **B** as the design goal (it also
subsumes D). Apply the same conservative rule to any other predict-then-merge
site. `emit-if-handler` emits both then/else and reconciles *actual* types (it
already fixed a related :bool stack-map bug), so it is likely already sound —
verify, don't assume.

## GAP 2 — f32 array binding mis-typed as long-array  [ACTIVE]

`compiled-mlp-gradient-f32-test`: `ClassCastException [J cannot be cast to [F` in
a `softmax_m_floats` compiled method — a `float[]` binding tagged/handled as
`long[]`. Likely the same family as GAP 1 (representation mismatch at a
control-flow merge or a binding whose AST-derived array-element tag disagrees
with the emitted array type). Confirm whether it's branch-merge (GAP 1) or a
distinct array-element-tag conversion bug in `tcr->dispatch-tag` / the f32 path.

## GAP 3 — Option X ext double-checks the bindings  [PERF, follow-up]

The `core/let` extension (Option X) runs `check-let-bindings` (for destructuring
expansion) and then re-checks the rebuilt `(let* expanded-bindings body)` via the
`let*` checker — the bindings are type-checked twice. Correct but on TC's hot
path. Production version should build the checked `:let` node from the
already-checked binding exprs without re-checking. Raise with Ambrose in #195.

## GAP 4 — parametric types still need the structural fallback  [DESIGN]

TC checks `(All [T] …)` bodies abstractly, so `T`-derived bindings have no
concrete tag; the walker recovers them structurally. Principled end-state: run TC
on the **concretely-instantiated** body at specialization time (T = double/float/
…) so TC supplies concrete tags for parametric bindings too, shrinking the
fallback to genuinely TC-blind cases (field registry, mid-walk macros, interop).
Larger change; on the TC agenda.

## GAP 5 — collector hygiene  [MINOR]

`collect-binding-types-from-checked-ast` currently collects every `:let`/`:loop`
binding node. A gensym filter (skip `G__\d+`, `…__auto__`) avoids surfacing
macroexpansion temporaries (e.g. `case` selector gensyms) that can't correspond
to source bindings. Not the root cause of GAP 1, but correct hygiene to keep.

## Status / next
- Option X ext + AST consumer: implemented, devirt gate green. Full-suite
  regressions (matrix-norm, gradient, matrix-norm-*, f32-mlp, a couple FAILs)
  all trace to GAP 1/2.
- Next: prototype GAP 1 fix (context-directed branch-result coercion) and
  re-run the full suite under the local Option-X TC.
- Then: typedclojure#195 — rebase fork onto upstream + reframe (AST not metadata,
  drop tag-suppression), address GAP 3.
