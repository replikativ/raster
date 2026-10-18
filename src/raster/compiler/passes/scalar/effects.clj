(ns raster.compiler.passes.scalar.effects
  "Effect and exceptional-control classification for scalar compiler passes.

   Beichte is the sole authority for externally observable effect analysis.
   When beichte cannot analyze an expression, it is conservatively assumed
   effectful (:io). Only expressions beichte proves pure get :pure.

   Source operations may additionally carry semantic obligations which are not
   external effects. Checked integral casts/arithmetic and allocation extent
   validation can throw. Such expressions retain Beichte's effect level but carry a flag so
   DCE, CSE and algebraic simplification cannot erase or duplicate the possible
   exceptional control transfer. These denial flags are not a complete
   totality analysis for arbitrary helpers or control flow.

   The raster context pre-registers raster.numeric and raster.math vars
   as :pure so beichte doesn't need to analyze their source each time."
  (:require [beichte.core :as b]
            [beichte.registry :as beichte-registry]
            [clojure.walk :as walk]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.core.numeric-constant :as numeric-constant]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.scalar-conversion :as scalar-conversion]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.scalar-range :as scalar-range]))

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
      (b/make-context {:registry reg}))))

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

(defn- collect-locals
  "Collect free non-operator symbols from an expression for beichte's :locals.
   Excludes call-position symbols (they should resolve as vars, not locals)."
  [expr]
  (cond
    (symbol? expr)
    (when (and (not (namespace expr))
               (not (contains? form/known-form-heads expr)))
      #{expr})
    (seq? expr)
    ;; Skip head (operator position) — only collect from args
    (apply clojure.set/union #{} (map collect-locals (rest expr)))
    (vector? expr)
    (apply clojure.set/union #{} (map collect-locals expr))
    :else #{}))

(declare descriptor)

(defn analyze-effect
  "Analyze an expression and return its effect level (:pure, :local, :mutation, :io)."
  [expr]
  (:effect (descriptor expr)))

(defn analyze-var-effect
  "Analyze a var and return its effect level using fresh source evidence."
  [v]
  (b/analyze-var v (b/make-context {:registry (:registry @raster-context)})))

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
      (or (some? (numeric-constant/value expression))
          (and source target
               (= [:exact :exact] (scalar-conversion/policy source target)))))))

(defn- source-cast-flags
  "Evaluation obligations for source casts derived from the shared descriptor.

   This deliberately derives from op-descriptor's closed conversion contract;
   it is not another function registry. Without retained operand dtypes here,
   checked integral casts remain conservatively non-removable. Local product
   lowering additionally retains its generic factory's reference conversion
   obligation explicitly. Untyped scoped binders require their own type evidence;
   this does not globally reinterpret their casts as reference conversions."
  [expr]
  (into #{}
   (mapcat (fn [node]
           (when (seq? node)
             (let [operation (descriptor/semantic-op node)]
               (when (descriptor/cast-op? operation)
                    (cond-> []
                        (and (not (proven-total-cast? node))
                             (= :reject (descriptor/cast-integral-narrowing operation))
                             (contains? '#{byte int long} (descriptor/cast-result-tag operation)))
                        (conj :checked-source-cast)
                        ;; This is denial-only evidence from the source factory:
                        ;; no optimizer may erase its conversion merely because
                        ;; the projected field is not otherwise used.
                        (true? (:raster.effect/source-reference-cast (meta node)))
                        (conj :source-reference-cast))))))
         (tree-seq coll? seq expr))))

(defn- source-evaluation-flags
  "Known exceptional evaluation obligations, independent of external purity.
   Read retained producer dtypes, source overflow policy and the same scalar
   range/constructor authorities as lowering. These are denial flags, not a
   totality certificate for arbitrary helpers, accesses or control flow."
  [expr]
  (into #{}
        (mapcat
         (fn [node]
           (when (seq? node)
             (let [operation (descriptor/semantic-op node)
                   arguments (vec (descriptor/call-args node))
                   source-type (or (some-> (types/sym-type-tag node) dtype/dtype-for-scalar-tag)
                                   (when (and (seq arguments)
                                              (every? #(= 'long (types/literal-tag %)) arguments))
                                     :long))
                   integral? (intrinsics/integral-arithmetic? operation source-type)
                   division? (intrinsics/source-divisor-check? operation source-type)
                   divisor (when (and division? (= 2 (count arguments)))
                             (numeric-constant/value (second arguments)))
                   divisor-range (when (and divisor (integer? (:value divisor)))
                                   (scalar-range/literal (:value divisor) source-type))
                   literal-range
                   (when (and integral? (= 2 (count arguments)))
                     (scalar-range/arithmetic
                      (intrinsics/canonical operation)
                      (mapv (fn [argument]
                              (when-let [{:keys [value]} (numeric-constant/value argument)]
                                (when (integer? value) (scalar-range/literal value source-type))))
                            arguments)))
                   extent (descriptor/array-alloc-size node)
                   constant-extent (when (some? extent) (numeric-constant/value extent))]
               (cond-> []
                 (and integral?
                      (not= :wrap (intrinsics/source-overflow-policy operation))
                      (not (scalar-range/contained-in-dtype? literal-range source-type)))
                 (conj :checked-source-arithmetic)
                 (and division?
                      (not (and (intrinsics/integral-division? operation source-type)
                                (scalar-range/strictly-positive-integral-divisor? divisor-range))))
                 (conj :checked-source-division)
                 ;; Clojure's Number-size constructors use Number.intValue,
                 ;; not a checked long->int cast. Match that actual narrowing:
                 ;; e.g. 2^32 requests zero elements, not a negative-size error.
                 ;; Unknown extents retain validation; positive literal extents
                 ;; remain removable (resource exhaustion is not modeled here).
                 (and (some? extent)
                      (not (and constant-extent (number? (:value constant-extent))
                                (not (neg? (.intValue ^Number (:value constant-extent)))))))
                 (conj :source-array-extent)
                 ;; A storage/allocation facet is not a totality promise for
                 ;; an allocator helper (or another constructor overload).
                 ;; Preserve its checks until a source/entry-domain proof can
                 ;; expose and discharge them, rather than guessing by a name.
                 (and (descriptor/alloc-op? operation) (nil? extent))
                 (conj :source-allocation-evaluation)))))
         (tree-seq #(and (coll? %)
                         (not (and (seq? %) (contains? '#{quote clojure.core/quote} (first %)))))
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
   exact resolved operation identities. Source evaluation obligations are
   recomputed from current policies outside that cache. Open expressions use
   a fresh Beichte context; cached source analysis must not survive helper
   redefinition."
  [expr]
  (cond
    (not (seq? expr)) {:effect :pure :flags #{}}
    ;; A validated Object context returns the original object (including nil),
    ;; after evaluating its operand once. It adds no source conversion check.
    ;; This is effect analysis only, not permission to strip live materialization
    ;; or merge separate boxing contexts. Other completion contracts retain their
    ;; checks and malformed reserved terms remain conservative unknown calls.
    (= completion/object-input-contract (:contract (completion/canonical-parts expr)))
    (descriptor (:operand (completion/canonical-parts expr)))
    :else
    (let [semantic-expr (semantic-calls expr)
          ;; Cache expensive Beichte analysis, not source evaluation policy.
          ;; Re-read the current shared policies without clearing any cache.
          source-flags (into (source-cast-flags semantic-expr)
                             (source-evaluation-flags semantic-expr))
          result
          (if-let [k (try
                 (when-let [proof (closed-registry-proof expr)]
                   [(descriptor-key expr) proof])
                 (catch Exception _ nil)
                 (catch StackOverflowError _ nil))]
      (or (.get ^java.util.Map descriptor-cache k)
          (let [result (analyze-descriptor expr)]
            (.put ^java.util.Map descriptor-cache k result)
            result))
      (analyze-descriptor expr))]
      (update result :flags into source-flags))))

(defn- analyze-descriptor
  [expr]
  (try
    (let [semantic-expr (semantic-calls expr)
          locals (collect-locals semantic-expr)
          context (b/make-context {:registry (:registry @raster-context)})
          result (b/analyze-full semantic-expr context locals)
          result (if (map? result)
                   (update result :flags #(or % #{}))
                   {:effect (or result :io) :flags #{}})]
      result)
    (catch Exception _
      ;; Conservative: unknown = effectful
      {:effect :io :flags #{}})
    (catch StackOverflowError _
      ;; Deep expressions overflow tools.analyzer — assume effectful
      {:effect :io :flags #{}})))

(defn ordered-scalar-evaluation-required?
  "Whether a scalar conversion/arithmetic check must execute even if its value is dead.
   Allocation obligations remain on their storage evaluation boundary, not a
   scalar ABI port. This proves neither hoistability nor immutable operands."
  [expr]
  (boolean (some #{:checked-source-cast :source-reference-cast
                   :checked-source-arithmetic :checked-source-division}
                 (:flags (descriptor expr)))))

(defn removable-expr?
  "True if an unused expression can be dropped without changing semantics."
  [expr]
  (let [{:keys [effect flags]} (descriptor expr)]
    (and (= :pure effect)
         (empty? flags))))

(defn cse-safe-expr?
  "True if an expression is safe to common-subexpression eliminate.

   External purity is necessary but not sufficient: an allocator is removable
   when its result is dead, yet two live allocations have distinct mutable
   identities and must never be coalesced.  Keep that distinction here rather
   than teaching individual compiler markers or consumers to protect their
   operands. Declared storage writes, mutation and allocation also rule out CSE
   without reanalyzing the callee body. These contracts only deny sharing;
   their absence never substitutes for Beichte's purity proof."
  [expr]
  (and (seq? expr)
       (not (completion/contains-completion? expr))
       (symbol? (first expr))
       (not (descriptor/alloc-op? (form/effective-op expr)))
       (not (when-let [op (descriptor/semantic-op expr)]
              (let [[contract] (descriptor/resolve-op-descriptor op)]
                (or (:buffer-write contract)
                    (true? (get-in contract [:effects :mutating?]))
                    (true? (get-in contract [:buffer :allocates?]))))))
       (removable-expr? expr)))

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
