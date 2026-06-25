(ns raster.compiler.ir.invariants
  "Compiler invariants enforced at pass boundaries (via validate-dialect!).

  Two tiers, mirroring the design in
  .internal/compiler_correctness_by_construction.md:

  - STRUCTURAL (cheap, always-on under *validate-dialects?*): shape properties a
    dialect grammar could express. These THROW — they guard guarantees the
    pipeline establishes (e.g. idempotent ftm ⇒ no nested source-body). When the
    dialect grammars are wired (Option 3) these migrate into the grammar.

  - SEMANTIC (deep, opt-in under *validate-deep?*): cross-cutting properties a
    context-free grammar CANNOT express — qualification (scope/free-var analysis),
    type-tag presence (metadata), op classification (registry). These live here
    permanently and run alongside whatever pass implementation produced the form
    (direct-walking today, pattern `defpass` later — same boundary seam)."
  (:require [raster.compiler.core.util :as util]
            [raster.compiler.ir.form :as form]))

;; ---------------------------------------------------------------------------
;; I2 — no nested / duplicated :raster.walker/source-body  (STRUCTURAL)
;; Idempotent walk-form :ftm guarantees each ftm carries exactly one raw
;; source-body marker. A second marker at the same level means a non-idempotent
;; re-walk doubled the (64KB-method-limit-sensitive) payload.
;; ---------------------------------------------------------------------------
(defn nested-source-body
  "First ftm form carrying >1 :raster.walker/source-body marker at its own level,
  or nil if clean."
  [form]
  (->> (tree-seq coll? seq form)
       (filter #(and (seq? %) (= 'ftm (first %))))
       (some (fn [ftm]
               (when (> (count (filter #(= :raster.walker/source-body %) ftm)) 1)
                 ftm)))))

;; ---------------------------------------------------------------------------
;; I4 — closed core: no macro binding/control form survives the walker (STRUCTURAL)
;; walk-body runs macroexpand-core, so every walked body is closed to special
;; forms (let*/loop*/fn*/case*/if/do...). A bare let/loop/fn/case/when/cond/and/or
;; reaching a pass boundary means a code-generating pass emitted an UN-closed macro
;; form (bypassing dispatch + the scope foundation). Skips quoted data and the raw
;; ftm `:raster.walker/source-body` payload, which are deliberately pre-macroexpand.
;; ---------------------------------------------------------------------------
(def ^:private preserved-macro-heads
  "Macros legitimately PRESERVED in walked IR (not eliminated by macroexpand-core):
  the `dotimes` counted-loop primitive and the `ftm` closure form. The
  `raster.par/*` SOAC primitives are also macros but preserved — matched by
  namespace below, not enumerated here."
  '#{dotimes ftm clojure.core/dotimes})

(def ^:private forbidden-macro?*
  "Memoized core of forbidden-macro?, keyed on [ns head]. Resolution is done with
  `ns-resolve` against the EXPLICIT ns (the bound source-ns at scan time), NOT
  ambient `*ns*` — and the ns is part of the memo key — so a BARE unqualified head
  can never cache a stale verdict from a different namespace (e.g. `when` resolves
  to clojure.core/when in a real source ns but to nil in a refer-less ns). The
  distinct [ns head] set is small, so this stays O(1) after warmup."
  (memoize
   (fn [ns h]
     (and (symbol? h)
          (not (contains? preserved-macro-heads h))
          (not (and (namespace h) (.startsWith ^String (namespace h) "raster.par")))
          (boolean (when-let [v (try (ns-resolve ns h) (catch Throwable _ nil))]
                     (and (var? v) (:macro (meta v)))))))))

(defn- forbidden-macro?
  "True if head `h` is a macro that macroexpand-core should have eliminated:
  it resolves (in the current ns) to a macro var and is NOT a preserved primitive
  (dotimes/ftm/raster.par/*). A WHITELIST by construction — any macro head that
  isn't explicitly preserved is forbidden, so the check can't rot as new macros
  appear (unlike a hardcoded blacklist). Special forms (let*/loop*/fn*/case*/if/
  do/try/recur…) resolve to nil → not flagged; the polymorphic deftm ops
  (raster.numeric/+ …) resolve to non-macro vars → not flagged."
  [h]
  (forbidden-macro?* *ns* h))

(defn- strip-source-body
  "Drop `:raster.walker/source-body <vec>` marker pairs from an ftm form's child
  seq — the raw payload is pre-macroexpand source, not closed-core code to check."
  [children]
  (loop [cs (seq children), out []]
    (cond
      (nil? cs) out
      (= :raster.walker/source-body (first cs)) (recur (nnext cs) out)
      :else (recur (next cs) (conj out (first cs))))))

(defn non-closed-core
  "First seq form whose head is a macro that macroexpand-core should have
  eliminated, or nil if the form is fully closed-core. Opaque to quoted data and
  (for ftm forms only) the raw source-body payload."
  [form]
  (letfn [(scan [x]
            (cond
              (and (seq? x) (= 'quote (first x))) nil          ;; quoted data — opaque
              (seq? x) (if (forbidden-macro? (first x))
                         x
                         ;; strip source-body ONLY inside ftm (the marker is
                         ;; ftm-specific; elsewhere the keyword is plain data and
                         ;; its sibling must still be scanned)
                         (some scan (if (= 'ftm (first x)) (strip-source-body x) x)))
              (vector? x) (some scan x)
              (map? x)    (some scan (mapcat identity x))
              (set? x)    (some scan x)
              :else nil))]
    (scan form)))

;; ---------------------------------------------------------------------------
;; I3 — every free symbol reaching the backend is namespace-qualified (SEMANTIC)
;; A bare (unqualified, non-local, non-form-head) symbol forces the backend to
;; guess a namespace — the qualify-upfront invariant. Grammars can't express this
;; (it needs scope/free-var analysis), so it stays a predicate.
;; ---------------------------------------------------------------------------
(def ^:private always-bare-ok
  "Unqualified symbols that are legitimately bare at the backend."
  '#{.invk double float long int boolean recur finally & do})

(defn unqualified-leaks
  "Free (non-local) unqualified symbols that are not form heads, cast intrinsics,
  interop (.foo), or one of the function's params — i.e. ones the backend would
  have to resolve against a guessed namespace. `params` is the deftm's parameter
  symbols (free-syms already excludes let-bound locals via scope; params are bound
  at the fn boundary, outside the validated body, so must be excluded explicitly).
  Returns a distinct seq, or nil if none."
  [form params]
  (let [param-set (set (map #(if (symbol? %) % (symbol (name %))) params))]
    (seq (distinct
          (->> (util/free-syms form)
               (remove always-bare-ok)
               (remove form/known-form-heads)
               (remove param-set)
               (remove #(.startsWith (name %) ".")))))))

;; ---------------------------------------------------------------------------
;; Boundary entry points (called by pipeline/validate-dialect!)
;; ---------------------------------------------------------------------------
(defn check-structural!
  "Always-on structural invariants. Throws on violation."
  [dialect-key form pass-key]
  (when-let [bad (nested-source-body form)]
    (throw (ex-info (str "Invariant I2 violated after :" pass-key
                         " — nested :raster.walker/source-body (non-idempotent ftm walk)")
                    {:invariant :I2 :pass pass-key :dialect dialect-key
                     :form-preview (pr-str (take 3 bad))})))
  (when-let [bad (non-closed-core form)]
    (throw (ex-info (str "Invariant I4 violated after :" pass-key
                         " — non-closed-core macro form `" (first bad) "` in walked IR "
                         "(a code-gen pass emitted an un-closed " (first bad)
                         "; use the closed-core form, e.g. let* / loop* / fn* / case* / if)")
                    {:invariant :I4 :pass pass-key :dialect dialect-key
                     :head (first bad)
                     :form-preview (pr-str (take 3 bad))}))))

(defn check-deep!
  "Opt-in semantic invariants. WARN mode (no throw) until the corpus is clean;
  flip individual checks to throw once baselined. `params` = the deftm's
  parameter symbols (legitimately unqualified; excluded from the leak check)."
  [dialect-key form pass-key params]
  (when (= dialect-key :backend-applied)
    (when-let [leaks (unqualified-leaks form params)]
      (binding [*out* *err*]
        (println (str "WARNING: invariant I3 (qualify-upfront) after :" pass-key
                      " — unqualified symbols reaching backend: " (vec leaks)))))))
