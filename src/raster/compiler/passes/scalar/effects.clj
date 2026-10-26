(ns raster.compiler.passes.scalar.effects
  "Effect and exceptional-control classification for scalar compiler passes.

   Beichte is the sole authority for externally observable effect analysis.
   When beichte cannot analyze an expression, it is conservatively assumed
   effectful (:io). Only expressions beichte proves pure get :pure.

   Source operations may additionally carry semantic obligations which are not
   external effects. Checked integral casts and array constructors can throw.
   Such expressions retain Beichte's effect level but carry a flag so
   DCE, CSE and algebraic simplification cannot erase or duplicate the possible
   exceptional control transfer.

   The Raster context registers declared contracts but disables unowned source
   lookup. Reading a Var's file/line does not authenticate its current root;
   expanding type/protocol declarations can also mutate JVM class identity."
  (:require [beichte.core :as b]
            [beichte.registry :as beichte-registry]
            [clojure.walk :as walk]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.intrinsic-source :as intrinsic-source]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.core.numeric-constant :as numeric-constant]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.scalar-conversion :as scalar-conversion]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.form :as form]))

(defn- declared-context [registry]
  (let [context (b/make-context {:registry registry :source-resolver nil})]
    ;; Older Beichte versions silently ignored unknown options. Never silently
    ;; re-enable source expansion when this compiler boundary requests none.
    (when-not (and (contains? context :source-resolver)
                   (nil? (:source-resolver context)))
      (throw (ex-info "Effect analysis requires source lookup to be disabled"
                      {:reason :unowned-effect-source-analysis})))
    context))

(defonce ^:private raster-context
  (delay
    (let [;; Register all raster.numeric and raster.math vars as :pure
          pure-entries
          (into {}
                (mapcat (fn [ns-sym]
                          (try
                            (require ns-sym)
                            (map (fn [[_ v]] [v :pure])
                                 (ns-publics (the-ns ns-sym)))
                            (catch Exception e
                              (binding [*out* *err*]
                                (println "WARNING: effects.clj failed to load" ns-sym "-" (.getMessage e)))
                              nil)))
                        '[raster.numeric raster.math]))
          ;; Register raster.arrays: aget/alength → :pure, aset → :local
          array-entries
          (try
            (require 'raster.arrays)
            (let [publics (ns-publics (the-ns 'raster.arrays))
                  mutating descriptor/array-mutating-base-names]
              (into {} (map (fn [[sym v]]
                              [v (if (contains? mutating sym) :local :pure)])
                            publics)))
            (catch Exception e
              (binding [*out* *err*]
                (println "WARNING: effects.clj failed to load raster.arrays -" (.getMessage e)))
              {}))
          ;; Register the pure FORWARD loss reductions as :pure. Each is a
          ;; reduce over pred/target with no IO or buffer mutation, but beichte
          ;; cannot see through the (All [T]) dispatch var to its reduce! body,
          ;; so it conservatively defaults them to :io — which wrongly pins the
          ;; DEAD value+grad primal (e.g. the mse-loss forward emitted alongside
          ;; the gradients but unused when the caller computes its own loss) so
          ;; DCE can't drop it, and it then leaks a resident forward array into a
          ;; host scalar closure on the GPU-resident path. Only the forward
          ;; reductions are registered — the *-backward helpers allocate + aset
          ;; (:local), so they are intentionally excluded.
          loss-entries
          (try
            (require 'raster.dl.loss)
            (let [publics (ns-publics (the-ns 'raster.dl.loss))]
              (into {} (keep (fn [[sym v]]
                               (when (contains? '#{mse-loss cross-entropy-loss huber-loss l1-loss} sym)
                                 [v :pure]))
                             publics)))
            (catch Exception e
              (binding [*out* *err*]
                (println "WARNING: effects.clj failed to load raster.dl.loss -" (.getMessage e)))
              {}))
          ;; Register raster.ad.reverse/grad-acc as :pure. It is the AD gradient-
          ;; accumulation primitive (nil-safe a+b): it ALWAYS returns a fresh value —
          ;; a passed-through input, a fresh clone it mutates locally, or numeric/+ —
          ;; and NEVER mutates a caller-owned buffer, so it is externally pure. beichte
          ;; can't see through the (.isArray)/aclone branches and defaults it to :io,
          ;; which wrongly pins a DEAD gradient (e.g. the unused input gradient dx in a
          ;; single LoRA/QLoRA train step: value+grad computes every arg's gradient, but
          ;; the step only consumes dA/dB — the dx grad-acc is dead) so DCE can't drop
          ;; it, and it then defeats resident GPU extraction (a non-kernel array op).
          ad-entries
          (try
            (require 'raster.ad.reverse)
            (let [publics (ns-publics (the-ns 'raster.ad.reverse))]
              (into {} (keep (fn [[sym v]]
                               (when (= 'grad-acc sym) [v :pure]))
                             publics)))
            (catch Exception e
              (binding [*out* *err*]
                (println "WARNING: effects.clj failed to load raster.ad.reverse -" (.getMessage e)))
              {}))
          raster-entries (merge pure-entries array-entries loss-entries ad-entries)
          reg (b/extend-registry (b/default-registry) raster-entries)]
      (declared-context reg))))

(defn init-raster-context!
  "Force initialization of the shared Raster Beichte context."
  []
  @raster-context
  nil)

(defn get-raster-context
  "Return the declared Raster Beichte context. Its registry contains trusted
   contracts; mutable helper analysis must use a fresh context."
  []
  @raster-context)

(defn- quoted-form? [expression]
  (and (seq? expression) (= 2 (count expression))
       (contains? '#{quote clojure.core/quote} (first expression))))

(defn- collect-locals
  "Collect free non-operator symbols from an expression for beichte's :locals.
   Excludes call-position symbols (they should resolve as vars, not locals)."
  [expr]
  (cond
    (quoted-form? expr) #{}

    (symbol? expr)
    (when (and (not (namespace expr))
               (not (contains? form/known-form-heads expr)))
      #{expr})
    (seq? expr)
    ;; Skip head (operator position) — only collect from args
    (apply clojure.set/union #{} (map collect-locals (rest expr)))
    (coll? expr)
    (apply clojure.set/union #{}
           (map collect-locals (if (map? expr) (mapcat identity expr) expr)))
    :else #{}))

(declare descriptor)

(defn analyze-effect
  "Analyze an expression and return its effect level (:pure, :local, :mutation, :io)."
  [expr]
  (:effect (descriptor expr)))

(defn analyze-var-effect
  "Return the declared effect of a Var; unknown source is not compiler evidence."
  [v]
  (b/analyze-var v (declared-context (:registry @raster-context))))

(defn- semantic-calls
  "Restore walker-devirtualized calls to the source-level operation recorded in
   their metadata before handing the expression to Beichte.  `.invk` is Raster
   dispatch IR, not a Clojure call Beichte can resolve; the carried
   :raster.op/original identity is the authoritative semantic seam.  Unknown
   `.invk` forms deliberately remain unchanged and therefore classify
   conservatively."
  [expr]
  (walk/postwalk
   (fn [node]
     (if (and (seq? node) (= '.invk (first node)))
       (if-let [operation (descriptor/semantic-op node)]
         (with-meta (apply list operation (descriptor/call-args node)) (meta node))
         node)
       node))
   expr))

(defn- proven-total-cast?
  "Use retained operand types or checked constant evidence, never a consumer's expected type."
  [expression]
  (when (= 1 (count (descriptor/call-args expression)))
    (let [operand (first (descriptor/call-args expression))
          source (some-> (when (instance? clojure.lang.IObj operand)
                           (types/sym-type-tag operand))
                         dtype/dtype-for-scalar-tag)
          target (some-> (descriptor/semantic-op expression) descriptor/cast-result-tag
                         dtype/dtype-for-scalar-tag)]
      (or (some? (when-let [operation (descriptor/semantic-op expression)]
                   (when (descriptor/cast-op? operation)
                     (numeric-constant/source-value
                      (apply list (symbol "clojure.core" (name operation))
                             (descriptor/call-args expression))))))
          (and source target
               (= [:exact :exact] (scalar-conversion/policy source target)))))))

(defn- bounded-source-allocation?
  "Discharge deterministic size checks only for an exact core constructor.
   Only raw integral literals are admitted: neither a type tag nor a nested
   conversion's numeric value authenticates its source implementation. A
   one-argument constructor may instead consume initializer data.
   As with existing dead-allocation elimination, resource exhaustion is outside
   this permission: a valid count is not a proof that allocation can succeed."
  [expression]
  (when-let [size (descriptor/array-allocation-size expression)]
    (when (and (integer? size) (<= 0 size Integer/MAX_VALUE))
      (let [lookup (first expression)
            operation (symbol "clojure.core" (name lookup))]
        (try
          (intrinsic-source/assert-original! (resolve lookup) operation
                                             :source-constructor-definition-mismatch)
          (when method-entry/*emission-inputs*
            (intrinsic-source/observe-original! *ns* lookup operation))
          true
          (catch clojure.lang.ExceptionInfo _ false))))))

(defn- source-evaluation-flags
  "Checked conversion/constructor obligations and explicit source-owner denials.

   This deliberately derives from op-descriptor's existing operation contracts;
   it is not another function registry. Without retained operand dtypes here,
   checked integral casts remain conservatively non-removable. This reader does
   not globally reinterpret arithmetic purity as source totality. New local
   product rewrites retain their actual evaluations explicitly; migration of
   ordinary arithmetic classification requires its target consumers too. Local product
   lowering additionally retains its generic factory's reference conversion
   obligation explicitly. Untyped scoped binders require their own type evidence;
   this does not globally reinterpret their casts as reference conversions."
  [expr]
  (into #{}
   (mapcat (fn [node]
             (let [operation (when (seq? node) (descriptor/semantic-op node))]
                    (cond-> []
                        (and (seq? node) (not (proven-total-cast? node))
                             (descriptor/cast-op? operation)
                             (= :reject (descriptor/cast-integral-narrowing operation))
                             (contains? '#{byte int long} (descriptor/cast-result-tag operation)))
                        (conj :checked-source-cast)
                        ;; Allocation purity says nothing about a negative/overflowing
                        ;; count, unsupported initializer overload or helper failure.
                        ;; Nominal allocation recognition is denial-only; only the
                        ;; exact constructor/count proof above discharges it.
                        (and (seq? node)
                             (descriptor/alloc-op? (form/effective-op node))
                             (not (bounded-source-allocation? node)))
                        (conj :source-allocation-evaluation)
                        ;; This is denial-only evidence from the source factory:
                        ;; no optimizer may erase its conversion merely because
                        ;; the projected field is not otherwise used.
                        (true? (:raster.effect/source-reference-cast (meta node)))
                        (conj :source-reference-cast)
                        ;; A value proof consumed by simplification is not a
                        ;; proof that its original evaluation can disappear.
                        ;; This obligation is independent of numeric width,
                        ;; external purity, and actual primitive/Object storage.
                        (true? (:raster.effect/preserve-evaluation (meta node)))
                        (conj :preserve-source-evaluation))))
         (tree-seq #(and (coll? %) (not (quoted-form? %)))
                   seq expr))))

(def ^:private descriptor-cache-capacity 65536)

(def ^:private descriptor-cache
  "Descriptors by expression key, least recently used evicted first. Scalar
   passes ask about the same expressions many times per fixpoint, and each
   analysis runs tools.analyzer over the whole expression."
  (java.util.Collections/synchronizedMap
   (proxy [java.util.LinkedHashMap] [1024 0.75 true]
     (removeEldestEntry [_]
       (> (.size ^java.util.Map this) (long descriptor-cache-capacity))))))

(defn- descriptor-key
  "Return a digest of `expr` including all of its metadata and `*ns*`.

   Metadata is part of the key because the analysis reads it: devirtualized
   calls carry their source operation there, and operand type tags decide
   whether a cast is total. Structural equality alone would ignore both.
   Unqualified symbols resolve in `*ns*`, so it is part of the key too."
  ^String [expr]
  (let [text (binding [*print-meta* true *print-length* nil *print-level* nil
                       *print-namespace-maps* false]
               (str (ns-name *ns*) "\n" (pr-str expr)))
        digest (.digest (java.security.MessageDigest/getInstance "SHA-256")
                        (.getBytes ^String text java.nio.charset.StandardCharsets/UTF_8))]
    (.formatHex (java.util.HexFormat/of) digest)))

(declare analyze-descriptor)

(defn- closed-registry-proof
  "Exact operation identities for expressions closed over declared contracts.
   Ordinary helpers, macros, special forms and interop need fresh analysis:
   Beichte's analyzed-Var cache is not a complete dependency footprint."
  [expr]
  (let [registry (:registry @raster-context)]
    (letfn [(proof [node]
              (cond
                (seq? node)
                (let [head (first node)
                      v (when (symbol? head) (resolve head))]
                  (when (and (var? v) (not (:macro (meta v)))
                             (some? (beichte-registry/lookup registry v)))
                    (let [children (mapv proof (rest node))]
                      (when (every? some? children)
                        (into [v] (mapcat identity children))))))

                ;; Collections may contain executable expressions, so their
                ;; analysis is deliberately not certified by a leaf proof.
                (coll? node) nil
                :else []))]
      (proof (semantic-calls expr)))))

(defn descriptor
  "Return the effect descriptor for a compiler IR expression.

   Uses beichte with :locals (free symbols treated as locally-bound)
   so it can analyze walked IR without var resolution failures.

   Conservative default: if beichte fails, returns {:effect :io :flags #{}}
   (assumed effectful). Only proven-pure expressions get :pure. Results are
   memoized only for closed declared-registry expressions, by metadata and
   exact resolved operation identities. Open expressions use a fresh Beichte
   context; cached source analysis must not survive helper redefinition."
  [expr]
  (if (or (not (coll? expr)) (quoted-form? expr))
    ;; A leaf performs no nested evaluation. Collection initializers, unlike
    ;; leaves, can execute calls in their elements/keys/values and must use the
    ;; same effect authority as ordinary call forms. Quote stays opaque there.
    {:effect :pure :flags (source-evaluation-flags expr)}
    (if-let [k (try
                 (when-let [proof (closed-registry-proof expr)]
                   ;; Revalidate source obligations even on cache hits. Core
                   ;; Var identity alone survives root/inline replacement and
                   ;; must not preserve yesterday's constructor permission.
                   [(descriptor-key expr) proof (source-evaluation-flags expr)])
                 (catch Exception _ nil)
                 (catch StackOverflowError _ nil))]
      (or (.get ^java.util.Map descriptor-cache k)
          (let [result (analyze-descriptor expr)]
            (.put ^java.util.Map descriptor-cache k result)
            result))
      (analyze-descriptor expr))))

(defn- analyze-descriptor
  [expr]
  ;; A failed optional purity analysis cannot discharge independently retained
  ;; source obligations. Read those from original syntax, not its Beichte-only
  ;; call projection, and merge them even when that analysis abstains.
  (try
   (let [obligations (source-evaluation-flags expr)
        result
        (try
          (let [semantic-expr (semantic-calls expr)
                locals (collect-locals semantic-expr)
                context (declared-context (:registry @raster-context))
                result (b/analyze-full semantic-expr context locals)]
            (if (map? result)
              (update result :flags #(or % #{}))
              {:effect (or result :io) :flags #{}}))
          (catch Exception _
            ;; Conservative: unknown = effectful, not obligation-free.
            {:effect :io :flags #{}})
          (catch StackOverflowError _
            ;; Deep expressions overflow tools.analyzer — assume effectful.
            {:effect :io :flags #{}}))]
    (update result :flags into obligations))
   ;; Obligation discovery is optional analysis too. If it cannot complete,
   ;; retain a NON-WAIVABLE unknown obligation, not a preservation-only marker
   ;; whose ordering could be relaxed by an independent total-value proof.
   (catch Exception _ {:effect :io :flags #{:unclosed-source-obligations}})
   (catch StackOverflowError _ {:effect :io :flags #{:unclosed-source-obligations}})))

(defn removable-expr?
  "True if an unused expression can be dropped without changing semantics."
  [expr]
  (let [{:keys [effect flags]} (descriptor expr)]
    (and (= :pure effect)
         (empty? flags))))

(defn cse-properties
  "Project sharing eligibility and observation invalidation from one effect query.

   External purity is necessary but not sufficient: an allocator may be removable
   when its result is dead and its checks discharged, yet two live allocations have distinct mutable
   identities and must never be coalesced.  Keep that distinction here rather
   than teaching individual compiler markers or consumers to protect their
   operands. Declared storage writes, mutation and allocation also rule out CSE
   without reanalyzing the callee body. These contracts only deny sharing;
   their absence never substitutes for Beichte's purity proof.

   A local or external effect invalidates prior observations even when that
   expression is not itself a sharing candidate. Source obligations are barriers
   too: this pass has no successful-return/actual-carrier proof discharging them.
   This conservative source epoch is not alias analysis or a fusion boundary."
  [expr]
  (let [call? (and (seq? expr) (symbol? (first expr)))
        contract (when call?
                   (when-let [op (descriptor/semantic-op expr)]
                     (first (descriptor/resolve-op-descriptor op))))
        writes? (or (:buffer-write contract)
                    (true? (get-in contract [:effects :mutating?])))
        allocates? (or (and call? (descriptor/alloc-op? (form/effective-op expr)))
                       (true? (get-in contract [:buffer :allocates?])))]
    (if (or writes? allocates?)
      ;; Storage contracts may deny sharing without inspecting the callee.
      ;; Fresh-output identity alone does not prove an allocator's arguments or
      ;; helper body cannot affect earlier observations: start a new epoch.
      {:shareable? false :invalidates-observations? true}
      (let [{:keys [effect flags]} (descriptor expr)]
        {:shareable? (boolean (and call? (= :pure effect) (empty? flags)))
         :invalidates-observations? (boolean (or (not= :pure effect) (seq flags)))}))))

(defn cse-safe-expr?
  "Whether the expression is eligible for sharing in an unchanged source epoch.
   Consumers must also honor `cse-properties` observation barriers."
  [expr]
  (:shareable? (cse-properties expr)))

(defn replay-safe-value?
  "Stable scalar/identity operands safe to repeat or omit during substitution.
   This is not general purity or hoisting: open helpers, reads and arithmetic
   require an evaluate-once prefix. Only closed, proven-total source casts may
   wrap stable operands. Qualified Vars are not immutable local values."
  ([expression] (replay-safe-value? expression {}))
  ([expression environment]
   (let [expression (walk/postwalk
                     #(if-let [tag (and (symbol? %) (get environment %))]
                        (vary-meta % assoc :raster.type/tag tag) %) expression)]
     (or (nil? expression) (number? expression) (boolean? expression)
         (string? expression) (char? expression) (keyword? expression)
         (and (symbol? expression) (contains? environment expression))
         (and (seq? expression)
              (not (contains? environment (first expression)))
              (descriptor/cast-op? (descriptor/semantic-op expression))
              (some? (closed-registry-proof expression))
              (proven-total-cast? expression)
              (replay-safe-value? (first (descriptor/call-args expression)) environment))))))
