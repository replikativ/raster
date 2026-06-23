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
