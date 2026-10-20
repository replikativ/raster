(ns raster.compiler.backend.jvm.source-owner
  "Compile an owner-local JVM source projection with actual private captures.

   This is a source compiler boundary, not another evaluator or executable
   registry. The producer supplies the compilation policy, resolved Methods,
   lexical ports and dependency guard. It must retain runtime/residual guards
   and must not export these private captures as ordinary portable source."
  (:require [clojure.set :as set]
            [raster.compiler.core.ordered-call :as ordered-call]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.intrinsic-source :as intrinsic-source]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.types :as types]
            [raster.compiler.backend.jvm.terminal-source :as terminal-source]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.stage-result :as stage-result]
            [raster.compiler.ir.form :as form])
  (:import [java.lang.reflect Method]))

(defn assert-portable-source!
  "Refuse source extraction that would discard an unprojected private owner.
   Prepared programs and callable metadata use separate, explicit owner slots."
  [program]
  (when (:raster.core/anonymous-function program)
    (let [retained (:ns program)
          live (some-> (:raster.core/deftm-source-ns program) find-ns)]
      (when-not (and (instance? clojure.lang.Namespace retained)
                     (identical? retained live))
        (throw (ex-info "Anonymous source namespace has been removed or replaced"
                        {:reason :stale-anonymous-source-namespace
                         :source-ns (:raster.core/deftm-source-ns program)}))))
    (let [policy (:raster.core/source-policy program)]
      (when-not (completion/source-policy? policy)
        (throw (ex-info "Anonymous source has no retained compilation policy"
                        {:reason :unclosed-anonymous-source-policy})))
      (when-not (false? (:unchecked-math policy))
        (throw (ex-info "Anonymous unchecked arithmetic needs source-owned lowering"
                        {:reason :unsupported-anonymous-source-policy :source-policy policy})))))
  (when (or (contains? program :source-owner)
            (contains? program :raster.compiler/source-owner)
            (seq (:raster.core/lexical-captures program)))
    (throw (ex-info "Private JVM source ownership requires a capture-aware projection"
                    {:reason :unprojected-jvm-source-owner})))
  program)

(defn call-with-source-context
  "Reuse an anonymous producer's actual namespace/policy, never the caller's.
   Named and separately owned entries retain their existing context contracts."
  [metadata compile!]
  (if (:raster.core/anonymous-function metadata)
    (do (assert-portable-source! metadata)
        (binding [*ns* (:ns metadata)
                  *unchecked-math* (get-in metadata [:raster.core/source-policy :unchecked-math])]
          (let [compiled (compile!)]
            (assert-portable-source! metadata)
            compiled)))
    (compile!)))

(defn- bound-identifiers [source]
  (let [identifiers (volatile! #{})]
    ;; Binder knowledge remains exclusively in scope-info, also for closures,
    ;; branches and recurrence scopes. Quoted data is opaque to this walk.
    (util/postwalk-preserving-meta
     (fn [expression]
       (when-let [{:keys [scopes]} (form/scope-info expression)]
         (doseq [scope scopes parameter (:binders scope) :when (symbol? parameter)]
           (vswap! identifiers conj parameter)))
       expression)
     source)
    @identifiers))

(defn compile-function
  "Compile one function under its owner's actual source context.

   function-form is the existing owner's parameter/arity-aware function builder.
   Captures are [{:port unqualified-symbol :value actual-object}], including
   actual Classes used by canonical CHECKCAST completion terms. No captured
   executable is installed in a namespace or reconstructed by its class name.

   The guard fences compilation before and after eval. This function does not
   establish call-site correspondence or provide a runtime/pullback lifetime
   wrapper; those remain with the producer that owns the captured executable."
  [params source function-form
   {:keys [source-policy source-ns captures reference-bindings methods guard]}]
  (when-not (and (vector? params) (every? symbol? params)
                 (= (count params) (count (distinct params)))
                 (ifn? function-form) (completion/source-policy? source-policy)
                 (instance? clojure.lang.Namespace source-ns)
                 (vector? captures) (every? #(= #{:port :value} (set (keys %))) captures)
                 (every? #(and (symbol? (:port %)) (nil? (namespace (:port %)))) captures)
                 (map? reference-bindings) (vector? methods)
                 (every? #(instance? Method %) methods) (or (nil? guard) (ifn? guard)))
    (throw (ex-info "JVM source owner requires a closed compilation context"
                    {:reason :unclosed-jvm-source-owner})))
  (let [ports (mapv :port captures)
        bound (into (set params) (bound-identifiers source))
        conflict (set/intersection bound (set ports))]
    (when (or (not= (count ports) (count (distinct ports))) (seq conflict))
      (throw (ex-info "private source captures must not shadow an owner binder"
                      {:reason :source-owner-capture-scope :conflicts conflict})))
    (doseq [[carrier port] reference-bindings]
      (when-not (and (class? carrier)
                     (some #(and (= port (:port %)) (identical? carrier (:value %))) captures))
        (throw (ex-info "reference completion requires its actual captured Class"
                        {:reason :source-owner-reference-port :port port}))))
    (let [body (completion/project-to-jvm-source source source-policy reference-bindings)
          used (into (into bound ports) (util/free-syms-flat [params body]))
          arguments (loop [id (gensym "owner_captures__")]
                      (if (contains? used id) (recur (gensym "owner_captures__")) id))
          nth-method (.getMethod clojure.lang.RT "nth" (into-array Class [Object Integer/TYPE]))
          nth-head (with-meta 'clojure.lang.RT/nth {:param-tags [Object Integer/TYPE]})
          bindings (vec (mapcat (fn [port index]
                                 [port (list nth-head arguments (int index))]) ports (range)))
          factory (list 'fn* ['& arguments]
                        (list 'let* bindings (function-form params body)))]
      (binding [*ns* source-ns *unchecked-math* (:unchecked-math source-policy)]
        (when guard (guard))
        (ordered-call/verify-source-method! nth-method)
        (doseq [method methods] (ordered-call/verify-source-method! method))
        (let [execute (apply (eval factory) (mapv :value captures))]
          (ordered-call/verify-source-method! nth-method)
          (doseq [method methods] (ordered-call/verify-source-method! method))
          (when guard (guard))
          execute)))))

(deftype ^:private TerminalResultOwner [program contract execute observations source-ns])

(defn- original-source-context! [source-var reason]
  (let [collector entry/*emission-inputs*
        observation (when collector (get-in @collector [:definition-proofs source-var]))
        metadata (:metadata observation)
        source-ns (:ns metadata)
        policy (:raster.core/source-policy metadata)]
    (when-not (and observation (var? source-var)
                   (instance? clojure.lang.Namespace source-ns)
                   (completion/source-policy? policy))
      (throw (ex-info "source projection needs its original definition owner" {:reason reason})))
    (entry/assert-definition-inputs-current! @collector)
    {:collector collector :metadata metadata :source-ns source-ns :policy policy}))

(defn- source-history-current! [source-ns observations reason]
  (when-not (identical? source-ns (find-ns (ns-name source-ns)))
    (throw (ex-info "source namespace was replaced" {:reason reason})))
  (entry/assert-definition-inputs-current! observations))

(defn- terminal-owner-current! [^TerminalResultOwner owner]
  (source-history-current! (.-source-ns owner) (.-observations owner) :stale-terminal-source-namespace))

(defn compile-terminal-result
  "Compile the generated terminal source region under its existing producer.

   Requires an active definition collector which ALREADY observed source-var
   before constructing the numerical program. Never manufactures historical
   ownership from a fresh snapshot. The compiler producer remains responsible
   for source-to-program correspondence. The returned owner is process-local,
   nonportable, and tied to this exact program/contract pair.

   This compiles the region's source Object result context, not an arbitrary
   original bytecode method's return descriptor. It grants no physical export,
   producer completion, device observation or whole-invocation result proof."
  [program contract source-var]
  (let [{:keys [collector metadata source-ns policy]} (original-source-context! source-var :unowned-terminal-source)]
    (invocation/validate-terminal-source-result!
     contract program {:source-policy policy :source-namespace (ns-name source-ns)})
    (let [projection (terminal-source/project contract source-ns)
          return-tag (get metadata :raster.core/return-tag 'Object)
          result-facts (:result-facts projection)
          ;; Source annotations are not evidence that final unboxing/checkcast
          ;; is present. Permit Object, or prove the actual projected result is
          ;; already the exact non-null box of the declared primitive result.
          ;; Other method-return adaptations require a retained completion.
          _ (when-not (or (contains? #{'Object 'java.lang.Object Object} return-tag)
                          (and (get types/primitive-info return-tag)
                               (= (get types/primitive-info return-tag) (:boxed-carrier result-facts))
                               (:non-null? result-facts)))
              (throw (ex-info "terminal result needs its original method return completion"
                              {:reason :terminal-source-return-completion :return-tag return-tag
                               :result-facts result-facts})))
          guard #(source-history-current! source-ns @collector :stale-terminal-source-namespace)
          execute (compile-function (:params projection) (:source projection)
                                    (fn [params body] (list 'fn* params body))
                                    (merge projection {:source-policy policy :source-ns source-ns :guard guard}))
          ;; Freeze AFTER compilation, including dependencies discovered while
          ;; compiling. Invocation must never reread a mutable collector to
          ;; refresh an already-emitted function's historical observations.
          observations @collector]
      (entry/assert-definition-inputs-current! observations)
      (TerminalResultOwner. program contract execute observations source-ns))))

(deftype ^:private OrderedSourceOwner [program emission observations source-ns])

(defn ordered-source-required?
  "Derive historical source obligations, independently of owner-field presence.
   Prefix host work needs ownership too; only its late subset needs ordered routing.
   Source-free projections are exempt only through their validated closed contract."
  [program emission]
  (let [actions (invocation/source-host-actions
                 (get-in program [:attributes :invocation-plan]) program)]
    (boolean (or (seq actions) (stage-result/requires-observation? emission)))))

(defn compile-ordered-source
  "Freeze source ownership after emission under the original producer.
   This grants no physical binding, completion or initialization. The compiler
   producer owes source-to-semantic correspondence; consumers cannot refresh it.
   The existing checked-policy restriction remains; this grants no new policy.
   Optional mode returns nil ONLY for unsupported policy/import capabilities so an
   ordinary eager compile need not acquire ordered execution. Invalid/stale source
   always throws. Checked stages use required mode; ordered runtime rederives need."
  ([program emission source-var frontend]
   (compile-ordered-source program emission source-var frontend true))
  ([program emission source-var frontend required?]
   (entry/call-with-optional-definition-inputs
    (fn []
      (entry/assert-frontend-source! frontend source-var)
      (let [{:keys [collector source-ns policy]}
            (original-source-context! source-var :unowned-checked-result-source)]
        (intrinsic-source/observe-originals! source-ns)
        (let [observations @collector
              unsupported-import
              (some (fn [[v origin]]
                      (let [snapshot (get-in observations [:definition-proofs v])
                            metadata (case origin :var (:metadata snapshot)
                                                  :value (meta (:value snapshot)) nil)
                            imported-policy (:raster.core/source-policy metadata)]
                        (when-not (and snapshot (completion/source-policy? imported-policy)
                                       (false? (:unchecked-math imported-policy)))
                          {:reason :unsupported-checked-result-import-policy
                           :source v :origin origin :source-policy imported-policy})))
                    (:source-body-inputs observations))
              unavailable (cond
                            (not (false? (:unchecked-math policy)))
                            {:reason :unsupported-checked-result-source-policy :source-policy policy}
                            unsupported-import unsupported-import
                            (seq (:unclosed-source-imports observations))
                            {:reason :unclosed-checked-result-source-imports
                             :imports (:unclosed-source-imports observations)})]
          ;; Capability abstention must not mask a concurrent invalidation.
          (source-history-current! source-ns observations :stale-checked-result-source-namespace)
          (if unavailable
            (when required?
              (throw (ex-info "ordered source needs checked policy and original import lineage"
                              unavailable)))
            (OrderedSourceOwner. program emission observations source-ns))))))))

(defn assert-ordered-source!
  "Guard exact historical semantic/emitted ownership before and after source work."
  [owner program emission]
  (when-not (and (instance? OrderedSourceOwner owner)
                 (identical? program (.-program ^OrderedSourceOwner owner))
                 (identical? emission (.-emission ^OrderedSourceOwner owner)))
    (throw (ex-info "ordered source owner belongs to another source/program boundary"
                    {:reason :ordered-source-owner-mismatch})))
  (source-history-current! (.-source-ns ^OrderedSourceOwner owner)
                           (.-observations ^OrderedSourceOwner owner)
                           :stale-checked-result-source-namespace)
  owner)

(defn assert-ordered-compilation-source!
  "Fence an ordered compilation, including attempts to remove its required owner.
   An existing opaque owner authenticates the exact validated program identities;
   if absent, independently rederive the obligation rather than granting a bypass."
  [{:keys [ordered-source-owner semantic emitted]}]
  (when (or ordered-source-owner (ordered-source-required? semantic emitted))
    (assert-ordered-source! ordered-source-owner semantic emitted)))

(defn assert-terminal-result-owner!
  "Preflight exact source ownership without observing any numerical value or executing source."
  [owner program contract]
  (when-not (and (instance? TerminalResultOwner owner)
                 (identical? program (.-program ^TerminalResultOwner owner))
                 (identical? contract (.-contract ^TerminalResultOwner owner)))
    (throw (ex-info "terminal executable belongs to another source/program boundary"
                    {:reason :terminal-source-owner-mismatch})))
  (terminal-owner-current! owner)
  owner)

(defn invoke-terminal-result
  "Evaluate one owned terminal region using completed, ordered scalar exports.
   The runtime caller owes export/completion evidence BEFORE this boundary.
   Exact carrier/arity and owner guards are checked before source effects;
   successful source results are checked again before publication."
  [owner program contract arguments]
  (assert-terminal-result-owner! owner program contract)
  (let [^TerminalResultOwner owner owner
        parameters (:parameters contract)]
    (when-not (and (vector? arguments) (= (count parameters) (count arguments))
                   (every? true?
                           (map (fn [{:keys [dtype]} value]
                                  (case dtype :double (instance? Double value)
                                              :long (instance? Long value)))
                                parameters arguments)))
      (throw (ex-info "terminal arguments differ from their actual scalar carriers"
                      {:reason :terminal-source-arguments})))
    (terminal-owner-current! owner)
    (let [result ((.-execute owner) arguments)]
      (terminal-owner-current! owner)
      result)))
