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
;; I-T3 — dtype closure under :float monomorphization (SEMANTIC, THROWS)
;; Under :dtype :float, no double/doubles-tagged VALUE may be bound except where
;; it flows from an EXPLICIT double cast — the f64-in-f32 bug class (a stale
;; 'double stamp on a float kernel's binder makes the bytecode/GPU backend emit
;; f64 arithmetic, silently on CPU and as garbage kernels on GPU).
;;
;; Exemption model (evidence from the corpus, e.g. raster.dl.nn/rms-norm!):
;; kernels legitimately compute small "double islands" for numerical stability —
;; `ms (loop* ... (/ (double s) (double features)))` then `inv` referencing ms.
;; The walker spells intentional widening as a bare `(double ...)` cast (it also
;; wraps Double-typed scalar param uses that way), so:
;;   - a double-stamped binder whose init subtree contains an explicit double
;;     cast, or references an already-exempt double binder/param, is an
;;     INTENTIONAL island → added to the exempt set, no throw;
;;   - a double-stamped binder whose init is a devirtualized (.invk impl ...)
;;     call whose ORIGINAL deftm declares a double return at this dtype (e.g.
;;     mse-loss is `:- Double` even at floats args) is likewise declared → exempt;
;;   - a double-stamped binder with NONE of these is a stale stamp → VIOLATION.
;; Double/doubles-tagged deftm params (eps :- Double, ...) seed the exempt set —
;; they are declared, not inferred, so they are intentional by construction.
;; ---------------------------------------------------------------------------
(def ^:private double-cast-heads
  "Explicit double-construction heads that mark an intentional double island.
  The walker emits the BARE spelling (probed: `(double s)`, `(double features)`
  survive to every validated boundary); qualified spellings accepted defensively."
  '#{double clojure.core/double double-array clojure.core/double-array})

(def ^:private double-tags
  "Value/array tags meaning 'this is f64' — stamped as :raster.type/tag or :tag."
  '#{double doubles Double java.lang.Double})

(defn- stamped-tag
  "The double tag stamped on binder sym or its init form (checks
  :raster.type/tag first, then :tag), or nil if neither is double-tagged."
  [s init]
  (letfn [(dtag [x]
            (when (instance? clojure.lang.IObj x)
              (let [m (meta x)]
                (some double-tags [(:raster.type/tag m) (:tag m)]))))]
    (or (dtag s) (dtag init))))

(defn- subtree-nodes
  "All nodes of a form, skipping quoted data (code-as-data is not value flow)."
  [form]
  (tree-seq (fn [x] (and (coll? x) (not (and (seq? x) (= 'quote (first x))))))
            seq form))

(defn- contains-double-cast?
  "Init subtree contains an explicit (double ...) / (double-array ...) form."
  [init]
  (boolean (some #(and (seq? %) (contains? double-cast-heads (first %)))
                 (subtree-nodes init))))

(defn- refs-exempt?
  "Init subtree references a symbol in the exempt set (symbols compare by
  name+ns, metadata-blind, so stamped occurrences match)."
  [init exempt]
  (boolean (some #(and (symbol? %) (contains? exempt %)) (subtree-nodes init))))

(def ^:private declared-double-return?*
  "Memoized on [original-op-sym dtype]: does the deftm behind a devirtualized
  call DECLARE a double/doubles return at this dtype? Follows the walker's
  semantic-identity metadata (:raster.op/original) to the deftm var, resolves
  the dtype specialization (raster.core/resolve-deftm-var — requiring-resolve
  because ir/* must not require the top-level API), and reads the DECLARED
  :raster.core/return-tag. Never parses mangled impl names (compiler design:
  no pass recovers meaning from mangled names)."
  (memoize
   (fn [orig-sym dtype]
     (boolean
      (when-let [v (try (resolve orig-sym) (catch Throwable _ nil))]
        (when (var? v)
          (let [rv (try ((requiring-resolve 'raster.core/resolve-deftm-var)
                         v {:dtype dtype})
                        (catch Throwable _ nil))]
            (contains? double-tags
                       (:raster.core/return-tag (meta (or rv v)))))))))))

(defn- declared-double-invk?
  "Init is a direct devirtualized call `(.invk impl args...)` whose ORIGINAL
  deftm declares a double return at this dtype — the binder's double stamp is
  the callee's DECLARED return type (e.g. `mse-loss :- Double` even at floats
  args in an AD train step), not a stale stamp."
  [init dtype]
  (and (seq? init)
       (= '.invk (first init))
       (when-let [orig (:raster.op/original (meta init))]
         (declared-double-return?* orig dtype))))

(defn dtype-closure-violation
  "First let*/loop* binder violating I-T3, or nil. Scans bindings SEQUENTIALLY
  in evaluation order, threading the exempt set of legitimately-double names
  (seeded from double-tagged params via `param-env`) — so `inv` referencing the
  exempt island `ms` is itself exempt. Binder names are walker-uniquified, so
  the exempt set is threaded flat (no scope popping) without capture risk.
  Returns {:sym :tag :init} for the offending binder. `dtype` is the compile's
  dtype (resolves declared return tags of devirtualized callees)."
  [form param-env dtype]
  (let [seed (into #{} (comp (filter (fn [[_ t]] (contains? double-tags t)))
                             (map key))
                   param-env)
        scan (fn scan [x exempt]
               ;; → [violation-or-nil exempt']
               (cond
                 (and (seq? x) (= 'quote (first x)))
                 [nil exempt]

                 (and (seq? x)
                      (or (form/let-head? (first x)) (form/loop-head? (first x)))
                      (vector? (second x)))
                 (let [[v ex]
                       (reduce
                        (fn [[_ ex] [s init]]
                          ;; nested binding forms INSIDE the init first (their
                          ;; binders are checked under the current exempt set)
                          (let [[iv ex] (scan init ex)]
                            (cond
                              iv (reduced [iv ex])
                              (stamped-tag s init)
                              (if (or (contains-double-cast? init)
                                      (refs-exempt? init ex)
                                      (declared-double-invk? init dtype))
                                [nil (conj ex s)]        ;; intentional island
                                (reduced [{:sym s
                                           :tag (stamped-tag s init)
                                           :init init} ex]))
                              :else [nil ex])))
                        [nil exempt]
                        (partition 2 (second x)))]
                   (if v
                     [v ex]
                     (reduce (fn [[_ ex] child]
                               (let [[bv ex] (scan child ex)]
                                 (if bv (reduced [bv ex]) [nil ex])))
                             [nil ex] (drop 2 x))))

                 (coll? x)
                 (reduce (fn [[_ ex] child]
                           (let [[bv ex] (scan child ex)]
                             (if bv (reduced [bv ex]) [nil ex])))
                         [nil exempt]
                         (if (map-entry? x) [(key x) (val x)] (seq x)))

                 :else [nil exempt]))]
    (first (scan form seed))))

;; ---------------------------------------------------------------------------
;; I-T2-lite — binder/init tag consistency (SEMANTIC, WARN)
;; A binder symbol carrying no :raster.type/tag while its walked init form DOES
;; carry one means a pass rebound a typed value through an unstamped symbol —
;; downstream emitters that read binder stamps fall back to inference. Known
;; offenders exist in the corpus, so this WARNS to gather evidence before any
;; promotion to throw.
;; ---------------------------------------------------------------------------
(defn binder-tag-inconsistencies
  "Seq of {:sym :init-tag} for let*/loop* binders with an untagged symbol but a
  :raster.type/tag-stamped init form, or nil if none."
  [form]
  (seq
   (->> (subtree-nodes form)
        (filter #(and (seq? %)
                      (or (form/let-head? (first %)) (form/loop-head? (first %)))
                      (vector? (second %))))
        (mapcat #(partition 2 (second %)))
        (keep (fn [[s init]]
                (when (and (nil? (:raster.type/tag (meta s)))
                           (instance? clojure.lang.IObj init))
                  (when-let [it (:raster.type/tag (meta init))]
                    {:sym s :init-tag it})))))))

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
  "Opt-in semantic invariants at a pass boundary. `ctx` carries
    :params    — the deftm's parameter symbols (legitimately unqualified;
                 excluded from the I3 leak check)
    :dtype     — the compile's dtype (:float gates I-T3)
    :param-env — {param-sym → walker tag} (double-tagged params seed I-T3's
                 exempt set).
  I-T3 (dtype closure) THROWS — it guards against the f64-in-f32 silent
  miscompile class. It runs on the STAMPED dialects only (post-rewalk,
  pre-backend); it is skipped at:
    :lowered — the :lower pass splices AD templates whose stamps
      (:raster.type/tag, :raster.op/original) are only (re)established by the
      :fixpoint rewalk; pre-rewalk IR is half-stamped by design (probed: an
      AD-spliced mse-loss .invk carries a bare {:tag double} at :lowered and
      the full stamps at :fixpointed);
    :backend-applied and later — the :backend pass CONSUMES the stamps and
      strips form metadata (probed: the same .invk carries nil meta from
      :backend-applied onward), so the exemption evidence is gone.
  I3 (qualify-upfront) and I-T2-lite (binder/init tag consistency) WARN until
  their corpora are clean."
  [dialect-key form pass-key {:keys [params dtype param-env]}]
  (when (and (= dtype :float)
             (not (contains? #{:lowered :backend-applied :alength-resolved
                               :mem-merged}
                             dialect-key)))
    (when-let [{:keys [sym tag init]} (dtype-closure-violation form (or param-env {}) dtype)]
      (throw (ex-info (str "Invariant I-T3 (dtype closure) violated after :" pass-key
                           " — binder `" sym "` is stamped `" tag
                           "` under :float monomorphization, but its init neither"
                           " contains an explicit (double ...) cast nor references"
                           " an intentional double island. Stale f64 stamp in an"
                           " f32 kernel → silent double arithmetic on CPU /"
                           " garbage kernel on GPU.")
                      {:invariant :I-T3 :pass pass-key :dialect dialect-key
                       :binder sym :tag tag
                       :form-preview (pr-str (if (seq? init) (take 6 init) init))}))))
  (when (= dialect-key :backend-applied)
    (when-let [leaks (unqualified-leaks form params)]
      (binding [*out* *err*]
        (println (str "WARNING: invariant I3 (qualify-upfront) after :" pass-key
                      " — unqualified symbols reaching backend: " (vec leaks)))))
    (when-let [bad (binder-tag-inconsistencies form)]
      (binding [*out* *err*]
        (println (str "WARNING: invariant I-T2 (binder tag consistency) after :" pass-key
                      " — binders without :raster.type/tag whose init IS stamped: "
                      (pr-str (vec (take 8 bad)))))))))
