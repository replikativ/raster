(ns raster.compiler.core.source-prefix
  "Producer-local verification of unchanged frontend source intervals.

   Prefix correspondence relates original lexical occurrences to invocation
   steps without granting physical carrier or execution permission. The direct
   reduction/consumer family separately joins observed physical completions,
   conditional on their original evaluation. Neither grants count/read, purity,
   placement or GPU admission. A materialization cast is not original source."
  (:require [raster.compiler.core.method-entry :as entry]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.numeric-constant :as numeric-constant]
            [raster.compiler.core.intrinsic-source :as intrinsic-source]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.core.source-transport :as transport]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.par :as par]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.ir.semantic-fingerprint :as semantic]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-source :as source]
            [raster.compiler.passes.scalar.effects :as effects]))

(deftype ^:private PrefixInput [frontend parameters scope pairs])
(deftype ^:private PrefixCorrespondence [input program entries])

(defn- parallel-source? [expression]
  (cond
    (and (seq? expression) (#{'quote 'clojure.core/quote} (first expression))) false
    (par/par-form? expression) true
    (coll? expression) (boolean (some parallel-source? expression))
    :else false))

(defn- flat-prefix [expression]
  (when (and (form/binding-form? expression)
             (contains? form/let-heads (first expression))
             (vector? (second expression)) (even? (count (second expression))))
    (let [pairs (mapv vec (partition 2 (second expression)))
          prefix (vec (take-while #(not (parallel-source? (second %))) pairs))]
      (when (and (seq prefix) (every? symbol? (map first pairs))
                 ;; A scalar-only function has no device-prefix boundary here.
                 (some parallel-source? (concat (map second pairs) (nnext expression))))
        prefix))))

(defn- direct-invocation-root? [expression]
  (when (seq? expression)
    (let [{:keys [dispatch? lexical-callee]} (descriptor/call-description expression)]
      (and dispatch? (not lexical-callee)))))

(defn begin
  "Retain the exact authenticated frontend BEFORE source normalization.
   Parameters are the original declaration's ordered symbols. Unsupported roots
   abstain; forged or stale frontend ownership still throws. No emission occurs."
  [frontend parameters]
  (when frontend
    (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
    (when (and (= 1 (count (:body frontend))) (vector? parameters)
               (= parameters (get-in frontend [:source-entry :params]))
               (every? symbol? parameters) (= (count parameters) (count (set parameters))))
      (let [scope (first (:body frontend))
            ;; A direct call can expose a host prefix only after its selected
            ;; body is imported. Retain its original occurrence now; an empty
            ;; prefix is not correspondence to any subsequently inserted step.
            pairs (or (flat-prefix scope)
                      (when (direct-invocation-root? scope) []))]
        (when (and pairs
                   (= (count (concat parameters (map first pairs)))
                      (count (set (concat parameters (map first pairs))))))
          ;; Shadowed declarations need a richer occurrence relation, not a
          ;; symbol-map guess that conflates two binders of the same spelling.
          (PrefixInput. frontend parameters scope pairs))))))

(defn prepare-prefix-entry!
  "Explicitly prepare the original caller of a retained direct-call root.
   This exposes original argument/completion observations before helper import.
   It executes no argument or model code and grants no prefix correspondence.
   Promotions escape to the producer's existing whole-attempt retry."
  [input prepare!]
  (when (instance? PrefixInput input)
    (let [frontend (.-frontend ^PrefixInput input)
          scope (.-scope ^PrefixInput input)]
      (when (and (empty? (.-pairs ^PrefixInput input))
                 (identical? scope (first (:body frontend)))
                 (direct-invocation-root? scope)
                 ;; Only the first-argument nested boundary is supported by
                 ;; this producer family. Do not warm every scalar-call root
                 ;; or search arbitrary later arguments for a matching helper.
                 (direct-invocation-root? (first (descriptor/call-args scope))))
        (let [caller (entry/assert-frontend-selection!
                      frontend (get-in frontend [:source-entry :source-var]))]
          (when-let [receiver (:receiver-var caller)]
            (entry/prepare-owned-definition-inputs! prepare! [receiver])
            true))))))

(defn- direct-reduction-pairs [scope]
  (when (and (seq? scope) (contains? form/let-heads (first scope))
             (vector? (second scope)) (= 4 (count (second scope)))
             (= 3 (count scope)))
    (let [pairs (mapv vec (partition 2 (second scope)))]
      (when (and (every? symbol? (map first pairs))
                 (par/par-reduce-form? (second (first pairs)))
                 (seq? (second (second pairs)))
                 (not (parallel-source? (second (second pairs))))
                 (= (first (second pairs)) (nth scope 2)))
        pairs))))

(defn begin-reduction-interval
  "Capture an original direct reduction/consumer scope before normalization.
   This is distinct from the pre-device prefix family. It grants no bound/read,
   value, placement or execution correspondence, and prepares no source code."
  [frontend parameters]
  (when frontend
    (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
    (when (and (vector? parameters) (= parameters (get-in frontend [:source-entry :params]))
               (= 1 (count (:body frontend))))
      (let [scope (first (:body frontend))
            pairs (direct-reduction-pairs scope)
            binders (concat parameters (map first pairs))]
        (when (and pairs (every? symbol? parameters)
                   (= (count binders) (count (set binders))))
          (PrefixInput. frontend parameters scope pairs))))))

(defn prepare-reduction-entry!
  "Explicit producer preparation for a direct reduction/terminal interval.
   Queries never emit. Public compilation may request the original caller's owned bytecode
   and its direct selected floating update/terminal leaves, without executing them.
   Select every leaf before preparation, then prepare callees before renewing the caller.
   Changed observations escape to the whole compiler retry."
  [input prepare!]
  (when (instance? PrefixInput input)
    (let [frontend (.-frontend ^PrefixInput input)
          scope (.-scope ^PrefixInput input)]
      (when (and (direct-reduction-pairs scope)
                 (identical? scope (first (:body frontend)))
                 (every? true? (map identical? (mapcat identity (.-pairs ^PrefixInput input))
                                    (second scope))))
        (let [caller (entry/assert-frontend-selection! frontend (get-in frontend [:source-entry :source-var]))
              pairs (.-pairs ^PrefixInput input)
              selected-receiver
              (fn [expression]
                (let [{:keys [dispatch? semantic-op implementation-op arguments lexical-callee]}
                      (descriptor/call-description expression)
                      selected (when (and dispatch? (not lexical-callee))
                                 (entry/selected-parameter-carriers
                                  semantic-op implementation-op (count arguments)))]
                  (when (and selected (= Object (:result-carrier selected))
                             (seq (:parameter-carriers selected))
                             (every? #{Float/TYPE Double/TYPE} (:parameter-carriers selected)))
                    (get-in selected [:definition :receiver-var]))))
              reduction (second (first pairs))
              update (when (= 6 (bounded-count 7 reduction))
                       (selected-receiver (nth reduction 5)))
              terminal (selected-receiver (second (second pairs)))]
          (when-let [receiver (:receiver-var caller)]
            ;; Signature observation selects preparation only, never numerical admission.
            ;; Even a warm caller can refer to a still-cold selected implementation.
            (entry/prepare-owned-definition-inputs!
             prepare! (vec (distinct (cond-> [] update (conj update)
                                            terminal (conj terminal) true (conj receiver)))))
            true))))))

(defn- reduction-binding-observation [frontend scope original]
  (let [reduction (entry/frontend-reduction-emission frontend original)
        binding (entry/frontend-binding-context frontend scope 0)
        slot (:after binding)
        root (:root-result reduction)]
    (when (and reduction binding)
      (if (and (identical? (:initializer binding) (get-in reduction [:bound :expression]))
               (= :ref (:type slot) (:result-type root))
               (identical? (:reference-class slot) (get-in reduction [:result-fact :carrier]))
               (= :present (get-in reduction [:result-fact :presence]))
               (contains? #{Float Double} (get-in reduction [:result-fact :builtin-class])))
        {:kind :direct-reduction :reduction reduction :binding binding
         :completed-value (:result-fact reduction)}
        (when-let [extracted (entry/frontend-extracted-reduction-binding frontend scope 0)]
          {:kind :extracted-reduction :reduction (:reduction extracted)
           :binding (:binding extracted) :extraction extracted
           ;; This is the actual primitive local AFTER its helper invocation
           ;; and return adaptation complete, not a relabelled boxed payload.
           :completed-value {:carrier (.getReturnType ^java.lang.reflect.Method
                                                      (get-in extracted [:helper :method]))
                             :presence :present}})))))

(defn- closed-source-literal [frontend original before]
  (let [source-ns (get-in frontend [:source-entry :source-ns])
        locals (set (keys before))]
    (when (and entry/*emission-inputs* (instance? clojure.lang.Namespace source-ns))
      (letfn [(canonical [expression remaining]
                (when (pos? remaining)
                  (cond
                    (or (integer? expression) (float? expression)) expression
                    (and (seq? expression) (= 2 (count expression))
                         (symbol? (first expression))
                         (not (contains? locals (first expression)))
                         (not (:lexical-callee (descriptor/call-description expression)))
                         (descriptor/cast-op? (first expression))
                         (contains? #{nil "clojure.core"} (namespace (first expression))))
                    (let [lookup (first expression)
                          operation (symbol "clojure.core" (name lookup))]
                      (when-let [operand (canonical (second expression) (dec remaining))]
                        (intrinsic-source/observe-original! source-ns lookup operation)
                        (with-meta (list operation operand) (meta expression)))))))]
        (when-let [expression (canonical original 128)]
          (let [literal (numeric-constant/source-value expression)
                obligations (effects/descriptor original)]
            (when (and literal (= :pure (:effect obligations)) (empty? (:flags obligations)))
              {:source original :proof-expression expression :value (:value literal)})))))))

(defn- closed-reduction-count [frontend producer]
  (entry/call-with-optional-definition-inputs
   (fn []
     (let [bound (get-in producer [:reduction :bound])
           original (:initializer bound)
           literal (closed-source-literal frontend original (:before bound))]
       (when (and literal
                  (= :int (get-in bound [:after :type]))
                  (false? (:unretained-sibling-calls? bound))
                  (completion/bytecode-operand-context? (:context bound)))
         ;; Qualify only this closed literal/cast proof, under actual bootstrap
         ;; lookups. The original expression and its source position are retained.
         ;; Bare or qualified casts cannot borrow a different lookup's identity.
         (when (and (instance? Integer (:value literal)) (contains? #{0 1} (:value literal)))
           (assoc literal :bound bound :scope :closed-source-count :evaluation :before-initializer)))))))

(defn observed-reduction-interval
  "Join actual producer storage to original scalar invocation completions.
   Conditional on those original evaluations completing. The selected Object
   payload computation stays separate from actual invocation return/storage.
   An optional closed-count value is not a constructed-domain/read proof.
   No use closure, placement or GPU admission is established here.
   Cold/missing/unsupported evidence abstains transactionally without preparing
   code. The result is not a standalone execution certificate."
  [input]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when (and entry/*emission-inputs* (instance? PrefixInput input))
       (let [^PrefixInput input input
             frontend (.-frontend input) scope (.-scope input) pairs (.-pairs input)
             [[producer-id original] [consumer-id expression]] pairs]
         (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
         (when (and (identical? scope (first (:body frontend)))
                    (= (.-parameters input) (get-in frontend [:source-entry :params]))
                    (= 2 (count pairs))
                    (every? true? (map identical? (mapcat identity pairs) (second scope)))
                    (some? (direct-reduction-pairs scope)))
           (let [producer (reduction-binding-observation frontend scope original)
                 bound (entry/frontend-binding-context frontend scope 1)
                 call (entry/frontend-completed-invocation-context frontend expression)
                 {:keys [dispatch? semantic-op implementation-op arguments lexical-callee]}
                 (descriptor/call-description expression)
                 selected (when (and dispatch? (not lexical-callee))
                            (entry/selected-parameter-carriers semantic-op implementation-op (count arguments)))
                 producer-slot (get-in producer [:binding :after])
                 producer-inputs (keep-indexed (fn [i argument] (when (= producer-id argument) i)) arguments)
                 facts (mapv (fn [argument input completed]
                               (if (= producer-id argument)
                                 (when (and (= argument (:operand input))
                                            (= {producer-id producer-slot} (:ports input))
                                            (= (:type producer-slot) (:emitted-type completed)))
                                   (:completed-value producer))
                                 (when (and (empty? (:ports input))
                                            (contains? #{:float :double} (:emitted-type completed)))
                                   {:carrier (types/tag->java-class (symbol (name (:emitted-type completed))))
                                    :presence :present})))
                             arguments (:inputs call) (:input-completions call))]
             (when (and producer bound call selected
                        (= producer-id (get-in producer [:binding :binder]))
                        (= consumer-id (:binder bound))
                        (= producer-slot (get-in bound [:before producer-id]))
                        (identical? (:initializer bound) (:expression call))
                        (= :global-typed-indy (:invocation-mode call))
                        (= (:invocation-method selected) (:method call))
                        (identical? Object (:return-carrier call))
                        (= :ref (get-in bound [:after :type]))
                        (identical? Object (get-in bound [:after :reference-class]))
                        (= implementation-op (:receiver call) (get-in call [:receiver-input :operand]))
                        (empty? (get-in call [:receiver-input :ports]))
                        (completion/bytecode-operand-context? (get-in call [:receiver-input :context]))
                        (= 1 (count producer-inputs))
                        (= (count arguments) (count facts) (count (:inputs call))
                           (count (:input-completions call)))
                        (= (:parameter-carriers selected)
                           (mapv :parameter-carrier (:input-completions call)))
                        (every? some? facts))
               (when-let [computation
                          ((requiring-resolve 'raster.compiler.passes.parallel.scalar-expression-body/checked-selected-fp-payload)
                           expression facts)]
                 (when (and (= (:method call) (get-in computation [:bridge :invocation-method]))
                            (identical? (get-in selected [:definition :entry :typed-impl])
                                        (get-in computation [:bridge :implementation])))
                   {:scope scope :parameters (.-parameters input) :ordinals [0 1]
                    :producer producer :closed-count (closed-reduction-count frontend producer)
                    :consumer {:binding bound :invocation call
                                                 :argument-facts facts :selected selected
                                                 :computation computation}}))))))))))

(defn- significant-metadata [value]
  (let [metadata (apply dissoc (or (meta value) {})
                        [:line :column :end-line :end-column :file])
        ;; Normalization may elaborate an already retained logical declaration
        ;; into :tag. Accept only that exact elaboration, never a changed width.
        ;; This is logical source correspondence, NOT a physical hint proof.
        logical (:raster.type/tag metadata)]
    (if (and logical (= logical (:tag metadata)))
      (dissoc metadata :tag)
      metadata)))

(defn- source-metadata-compatible? [original transformed declarations]
  (let [left (significant-metadata original)
        right (significant-metadata transformed)
        ;; The existing canonical core allocator owns its array result type.
        ;; Tag elaboration does not stand for a checked cast or a helper's
        ;; inferred result; no other missing source declaration is repaired.
        constructor-tag (when (and (seq? original) (symbol? (first original))
                                   (= "clojure.core" (namespace (first original))))
                          (get descriptor/alloc-sym->array-tag (first original)))]
    (or (= left right)
        ;; Scalar construction carries existing declaration facts onto lexical
        ;; references. Only the declaration visible BEFORE this initializer
        ;; may justify that elaboration; it proves no primitive JVM carrier.
        (and (symbol? original) (get declarations original)
             (not (contains? left :tag)) (not (contains? left :raster.type/tag))
             (= (get declarations original) (:raster.type/tag right))
             (= left (dissoc right :raster.type/tag)))
        (and constructor-tag (not (contains? left :tag))
             (= constructor-tag (:tag right)) (= left (dissoc right :tag))))))

(defn- same-expression?
  ([original transformed renaming declarations]
   (same-expression? original transformed renaming declarations false))
  ([original transformed renaming declarations unchanged-scopes?]
  (and (source-metadata-compatible? original transformed declarations)
       (cond
         (symbol? original)
         (and (symbol? transformed) (= (get renaming original original) transformed))

         (and (seq? original) (#{'quote 'clojure.core/quote} (first original)))
         ;; Data is not a lexical expression. Refuse copied quoted products
         ;; until their reference identity has its own representation proof.
         (identical? original transformed)

         (seq? original)
         (and (seq? transformed)
              ;; Ordinary prefix verification admits no nested scopes. The
              ;; source-interval check may retain unchanged scoped syntax, but
              ;; never transports a renaming across a lexical boundary.
              (or (and unchanged-scopes? (empty? renaming))
                  (and (nil? (form/scope-info original))
                       (nil? (form/scope-info transformed))))
              (= (count original) (count transformed))
              (every? true? (map #(same-expression? %1 %2 renaming declarations unchanged-scopes?)
                                original transformed)))

         (instance? Double original)
         (and (instance? Double transformed)
              (= (Double/doubleToRawLongBits original) (Double/doubleToRawLongBits transformed)))

         (instance? Float original)
         (and (instance? Float transformed)
              (= (Float/floatToRawIntBits original) (Float/floatToRawIntBits transformed)))

         ;; No structural comparison of reference-valued products, executable
         ;; objects, Classes or arbitrary user values can manufacture origins.
         (or (nil? original) (string? original) (keyword? original)
             (boolean? original) (char? original) (integer? original))
         (and (= (class original) (class transformed)) (= original transformed))

         :else (identical? original transformed)))))

(defn close-reduction-terminal
  "Relate an unchanged owned reduction/consumer scope to its exact terminal suffix.
   All source phases are checked, not searched for a matching call. Nested scopes
   must be unchanged: this verifier performs no binder renaming or substitution.
   This grants terminal invocation correspondence only, not numerical producer,
   read-domain, input materialization or resident execution permission."
  [input normalization normalized represented program construction]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when (and (instance? PrefixInput input) (map? program)
                (vector? (:equations program)) (map? (:attributes program))
                (transport/source-construction-view construction normalization represented))
       (let [^PrefixInput input input
             frontend (.-frontend input)
             scope (.-scope input)
             origin (transport/normalization-occurrence normalization normalized)
             declarations (zipmap (.-parameters input) (get-in frontend [:source-entry :tags]))
             unchanged? (fn [left right]
                          (let [a (direct-reduction-pairs left) b (direct-reduction-pairs right)]
                            (and a b
                                 (= (first left) (first right))
                                 (or (source-metadata-compatible? left right declarations)
                                     ;; This phase drops the outer let's logical
                                     ;; result annotation. It is not a source
                                     ;; cast or a declaration of a local carrier;
                                     ;; every child/checkpoint is checked below.
                                     (and (not (contains? (significant-metadata left) :tag))
                                          (= (dissoc (significant-metadata left) :raster.type/tag)
                                             (significant-metadata right))))
                                 (every? true?
                                         (map (fn [[id expression] [other replacement]]
                                                (and (same-expression? id other {} declarations)
                                                     (same-expression? expression replacement {} declarations true)))
                                              a b))
                                 (same-expression? (nth left 2) (nth right 2) {} declarations))))
             pairs (direct-reduction-pairs represented)
             [[producer-id producer] [consumer-id consumer]] pairs
             final (:source program)
             tail (when (and (seq? final) (= 3 (count final))) (nth final 2))
             contract (get-in program [:attributes :invocation-plan :attributes :terminal-host-result])
             equation (first (:equations program))
             observation (observed-reduction-interval input)
             fact (get-in observation [:producer :completed-value])
             physical (or (:builtin-class fact) (:carrier fact))
             result-dtype (cond (contains? #{Float Float/TYPE} physical) :float
                                (contains? #{Double Double/TYPE} physical) :double)
             final-producer (when (and (seq? final) (vector? (second final))
                                       (= 2 (count (second final))))
                              (second (second final)))]
         (when (and origin observation (:closed-count observation) pairs
                    (identical? frontend (:frontend origin))
                    (identical? scope (:original origin))
                    (unchanged? scope normalized)
                    (unchanged? normalized represented)
                    (= (.-parameters input)
                       (mapv :symbol (get-in program [:attributes :invocation-plan :parameters])))
                    (nil? (get-in program [:attributes :invocation-plan :attributes :parameter-projection]))
                    (empty? (get-in program [:attributes :invocation-plan :steps]))
                    (= 1 (count (:equations program)))
                    (seq? final) (contains? form/let-heads (first final))
                    (= (significant-metadata represented) (significant-metadata final))
                    (vector? (second final)) (= 2 (count (second final)))
                    (= producer-id (first (second final)))
                    result-dtype (= result-dtype (get-in program [:values producer-id :dtype]))
                    (identical? (:source equation) (second (second final)))
                    (= [producer-id] (:results equation))
                    (seq? tail) (contains? form/let-heads (first tail)) (= 3 (count tail))
                    (vector? (second tail)) (= 2 (count (second tail)))
                    (= consumer-id (first (second tail)) (nth tail 2))
                    (same-expression? consumer-id (first (second tail)) {} declarations)
                    (same-expression? (nth represented 2) (nth tail 2) {} declarations)
                    (same-expression? consumer (second (second tail)) {} declarations)
                    (identical? tail (:body contract))
                    (= [{:symbol producer-id :value producer-id
                         :dtype (get-in program [:values producer-id :dtype])
                         :producer-equation (:id equation) :result-position 0}]
                       (:parameters contract))
                    (empty? (:public-scalar-parameters contract)))
           (PrefixCorrespondence.
            input program [{:kind :reduction-terminal :contract contract
                            :expression (second (second tail))
                            ;; The numerical frontend changes reduction index
                            ;; and operand annotations. This join does NOT prove
                            ;; those changes preserve source execution. Retain
                            ;; the actual endpoints as an outstanding obligation.
                            :producer-source {:original (second (first (.-pairs input)))
                                              :represented producer :constructed final-producer
                                              :status :unproved}
                            :observation observation}])))))))

(defn- prefix-renaming [original transformed parameters parameter-declarations]
  (when (= (count original) (count transformed))
    (let [free (:free
                (reduce (fn [{:keys [bound free]} [id expression]]
                          {:bound (conj bound id)
                           :free (into free (remove bound (util/free-syms expression)))})
                        {:bound (set parameters) :free #{}} original))]
      (loop [left (seq original) right (seq transformed)
             renaming (zipmap parameters parameters) bound (set parameters)
             declarations parameter-declarations]
        (if-let [[id expression] (first left)]
          (let [[result replacement] (first right)]
            (when (and (symbol? result) (nil? (namespace result))
                       (not (contains? bound result))
                       (or (= id result) (not (contains? free result)))
                       (= (significant-metadata id) (significant-metadata result))
                       (same-expression? expression replacement renaming declarations))
              (recur (next left) (next right) (assoc renaming id result) (conj bound result)
                     (if-let [tag (types/sym-type-tag id)]
                       (assoc declarations id tag) (dissoc declarations id)))))
          renaming)))))

(defn- scalar-source-corresponds?
  "Relate source to a validated scalar producer, retaining its result conversion.
   Invocation-only producers have no ProgramEquation. Their closed region and
   final initializer must BOTH retain the source unchanged, or BOTH be its
   declared materialization. This never strips an arbitrary source conversion."
  [expression final-expression step equation declarations]
  (if equation
    (same-expression? expression (source/scalar-equation-expression equation) {} declarations)
    (let [{:keys [parameters locals body-results]} (dialect/lambda-parts (:region step))
          tag (dtype/scalar-tag-for-dtype (get-in step [:value :dtype]))
          materialized (source/typed-store-value (symbol "clojure.core" (name tag)) expression)]
      (and (= parameters (mapv :symbol (:operands step)))
           (empty? locals) (= 1 (count body-results))
           (some (fn [expected]
                   (and (same-expression? expected (first body-results) {} declarations)
                        (same-expression? expected final-expression {} declarations)))
                 [expression materialized])))))

(defn- reference-indy-boundary [frame]
  ;; :receiver-input is synthetic for this mode. The emitter evaluates all
  ;; arguments and then links the selected target; it never loads a receiver.
  ;; Existing definition observations use effective Var values. Restricting
  ;; dynamic bindings ensures those observations also describe the raw root
  ;; used by indy; this is admission, not an extra source receiver operation.
  (let [original (:original-invocation frame)
        call (:call-context original)
        selected (:selected-callee original)
        frontend (get-in original [:builder :callee-import :frontend])
        implementation (:implementation frontend)
        receiver (get-in selected [:definition :receiver-var])
        symbol (:receiver call)
        bridge (when frontend (entry/frontend-reference-return-bridge frontend))]
    (when (and bridge (= :global-typed-indy (:invocation-mode call))
               (var? receiver) (not (.isDynamic ^clojure.lang.Var receiver))
               (nil? (.getThreadBinding ^clojure.lang.Var receiver))
               (symbol? symbol) (namespace symbol)
               (identical? receiver (some-> (find-ns (clojure.core/symbol (namespace symbol)))
                                           (.findInternedVar (clojure.core/symbol (name symbol)))))
               (.hasRoot ^clojure.lang.Var receiver)
               (identical? implementation (.getRawRoot ^clojure.lang.Var receiver))
               (identical? implementation (get-in selected [:definition :entry :typed-impl]))
               (identical? implementation (:implementation bridge))
               (= symbol (get-in call [:receiver-input :operand]))
               (empty? (get-in call [:receiver-input :ports]))
               (= (:method call) (:invocation-method selected) (:invocation-method bridge))
               (= Object (:return-carrier call) (:result-carrier selected) (:result-carrier bridge))
               (= (:parameter-carriers selected) (:parameter-carriers bridge)
                  (mapv :parameter-carrier (:input-completions call))))
      {:receiver receiver :implementation implementation :bridge bridge :frame frame})))

(defn- close-nested-prefix
  "Close only an original first-argument call's first allocator initializer.
   Match actual construction frames, then verify their positional prefix and
   adjacent allocation against the final plan. This is not execution admission."
  [^PrefixInput input normalization normalized represented program construction]
  (let [frontend (.-frontend input)
        parameters (.-parameters input)
        scope (.-scope input)
        origin (transport/normalization-occurrence normalization normalized)
        view (transport/source-construction-view construction normalization represented)
        events (:builder-expansions view)
        frame (fn [predicate]
                ;; Select occurrences before traversing their selected ancestry.
                ;; Computing a recursive frame for every event makes a large
                ;; construction pay the full ancestry search repeatedly.
                (let [found (filterv predicate events)]
                  (when (= 1 (count found))
                    ((requiring-resolve 'raster.compiler.passes.scalar.inline/nested-original-call-frame)
                     construction (:expansion (first found))))))
        original-event? (fn [original event]
                          (let [occurrence (transport/construction-occurrence construction (:caller event))]
                            (and (identical? original (:original occurrence))
                                 (identical? frontend (:frontend occurrence)))))
        root (frame #(original-event? scope %))
        edge (when root (entry/frontend-invocation-input-source frontend scope 0))
        parent (when edge (frame #(original-event? (:source-input edge) %)))
        imported (get-in parent [:original-invocation :builder :callee-import :frontend])
        body (when (= 1 (count (:body imported))) (first (:body imported)))
        first-initializer (when (and (seq? body) (contains? form/let-heads (first body))
                                     (vector? (second body)) (even? (count (second body))))
                            (nth (second body) 1 nil))
        parent-expansion (get-in parent [:original-invocation :builder :expansion])
        child (when (seq? first-initializer)
                (frame (fn [event]
                         (let [derived ((requiring-resolve 'raster.compiler.passes.scalar.inline/nested-substitution-call-origin)
                                        construction parent-expansion (:caller event))]
                           (and (identical? first-initializer (get-in derived [:origin :original]))
                                (identical? imported (get-in derived [:origin :frontend])))))))
        parent-boundary (when parent (reference-indy-boundary parent))
        child-boundary (when child (reference-indy-boundary child))
        declarations (zipmap parameters (get-in frontend [:source-entry :tags]))
        original-arguments (get-in parent [:original-invocation :source-arguments])
        parent-slots (:argument-slots parent)
        child-slots (:argument-slots child)
        bindings (get-in child [:substitution :argument-bindings])
        represented-pairs (flat-prefix represented)
        final-pairs (flat-prefix (:source program))
        ;; The allocator's single nontrivial argument is evaluated and bound
        ;; before its actual body. Its generated name is related positionally.
        renaming (when (= 1 (count bindings))
                   (prefix-renaming bindings (vec (take 1 represented-pairs)) parameters declarations))
        plan (get-in program [:attributes :invocation-plan])]
    (when (and (empty? (.-pairs input)) root edge parent-boundary child-boundary
               (identical? scope (get-in root [:original-invocation :original-call]))
               (identical? frontend (get-in root [:original-invocation :caller-origin :frontend]))
               (identical? (:operand edge) (get-in root [:original-invocation :call-context :inputs 0 :operand]))
               (identical? (:source-input edge) (get-in parent [:original-invocation :original-call]))
               (identical? frontend (get-in parent [:original-invocation :caller-origin :frontend]))
               (identical? first-initializer (get-in child [:original-invocation :original-call]))
               (identical? imported (get-in child [:original-invocation :caller-origin :frontend]))
               (identical? parent-expansion (get-in child [:parent-frame :original-invocation :builder :expansion]))
               (identical? frontend (:frontend origin)) (identical? scope (:original origin))
               (empty? (get-in parent [:substitution :argument-bindings]))
               (= (count original-arguments) (count parent-slots))
               (every? true? (map (fn [argument slot]
                                   (and (or (instance? Long argument)
                                            (and (symbol? argument) (contains? declarations argument)))
                                        (same-expression? argument (:argument slot) {} declarations)))
                                 original-arguments parent-slots))
               (= 2 (count child-slots)) (= 1 (count bindings))
               (= (:binder (nth child-slots 1)) (ffirst bindings))
               (identical? (:argument (nth child-slots 1)) (second (first bindings)))
               (nil? (:binder (first child-slots))) renaming
               (>= (count represented-pairs) 2) (>= (count final-pairs) 2)
               (nil? (get-in plan [:attributes :parameter-projection]))
               (= parameters (mapv :symbol (:parameters plan))))
      (invocation/validate-against! plan program)
      (let [order (invocation/source-order plan program)
            steps (into {} (map (juxt :id identity)) (:steps plan))
            [scalar-position allocation-position] order
            scalar (get steps (:invocation-step scalar-position))
            allocation (get steps (:invocation-step allocation-position))
            [represented-id expression] (first represented-pairs)
            [final-id final-expression] (first final-pairs)
            [allocation-id allocation-expression] (second final-pairs)
            expected-allocation (get-in child [:substitution :body])
            values (into (:values program)
                         (keep #(when (and (:symbol %) (:value %)) [(:symbol %) (:value %)]))
                         (concat (:parameters plan) (:steps plan)))]
        (when (and (invocation/scalar-compute? scalar) (invocation/buffer-allocation? allocation)
                   (= 0 (:position scalar-position)) (= 1 (:position allocation-position))
                   (= represented-id final-id (:symbol scalar) (:binding scalar-position))
                   (= allocation-id (:symbol allocation) (:binding allocation-position) (:storage allocation-position))
                   (= :long (dtype/canon (get-in scalar [:value :dtype])))
                   (scalar-source-corresponds? expression final-expression scalar nil declarations)
                   (same-expression? expected-allocation (second (second represented-pairs)) renaming declarations)
                   (= (first (second represented-pairs)) allocation-id)
                   (same-expression? (second (second represented-pairs)) allocation-expression {} declarations)
                   (source/allocation-corresponds? allocation-expression (:value allocation)
                                                  (:initialization allocation) values))
          (PrefixCorrespondence.
           input program
           [{:kind :nested-invocation-input :ordinal 0 :scope scope
             :binding represented-id :expression expression :step scalar :result-expression final-expression
             :construction construction :normalization normalization :represented represented
             :parent parent :child child :boundaries [parent-boundary child-boundary]
             :allocation allocation}]))))))

(defn close
  "Verify the complete bounded transformation at the equation-first producer.
   Intermediate forms are checked in their actual phase order. The exact final
   program and step objects are retained, not found later by source equality.
   Aggregate projection, inserted/removed/reordered prefix work, nested scopes
   and changed source metadata prevent construction. Existing plan validation
   separately owns logical result-materialization conversions."
  ([input normalization normalized represented program construction]
   ;; Retain the ACTUAL source-phase boundary, not a replay or an options-carried history.
   ;; Expanded helper intervals remain unsupported by the bounded verifier below; builder
   ;; observations alone must not bypass its complete prefix/metadata/order checks.
   (when (or (nil? construction)
             (transport/source-construction-view construction normalization represented))
     (if (and (instance? PrefixInput input) (empty? (.-pairs ^PrefixInput input)) construction)
       (close-nested-prefix input normalization normalized represented program construction)
       (close input normalization normalized represented program))))
  ([input normalization normalized represented program]
  (when (and (instance? PrefixInput input)
             (flat-prefix (:source program))
             (nil? (get-in program [:attributes :invocation-plan :attributes :parameter-projection])))
    (let [^PrefixInput input input
          frontend (.-frontend input)
          parameters (.-parameters input)
          parameter-declarations (zipmap parameters (get-in frontend [:source-entry :tags]))
          original (.-pairs input)
          normalized-pairs (flat-prefix normalized)
          represented-pairs (flat-prefix represented)
          normalized-origin (transport/normalization-occurrence normalization normalized)
          represented-map (prefix-renaming original represented-pairs parameters parameter-declarations)
          plan (get-in program [:attributes :invocation-plan])]
      (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
      (when (and normalized-origin
                 (identical? frontend (:frontend normalized-origin))
                 (identical? (.-scope input) (:original normalized-origin))
                 (= (count original) (count normalized-pairs))
                 represented-map plan
                 (= parameters (mapv :symbol (:parameters plan))))
        (let [order (invocation/source-order plan program)
              steps (into {} (map (juxt :id identity)) (:steps plan))
              final-pairs (mapv vec (partition 2 (second (:source program))))
              declarations (reductions (fn [visible [id]]
                                         (if-let [tag (types/sym-type-tag id)]
                                           (assoc visible id tag) (dissoc visible id)))
                                       parameter-declarations represented-pairs)
              entries
              (mapv
               (fn [ordinal [original-id original-expression] [represented-id expression]]
                 (let [position (nth order ordinal nil)
                       [final-id final-expression] (nth final-pairs ordinal nil)
                       step (get steps (:invocation-step position))
                       equation (some #(when (= (:id %) (:equation position)) %) (:equations program))
                       scalar? (invocation/scalar-compute? step)]
                   (when (and (= ordinal (:position position))
                              (= represented-id final-id (:binding position))
                              (if scalar?
                                (and (= represented-id (:symbol step))
                                     (scalar-source-corresponds?
                                      expression final-expression step equation (nth declarations ordinal)))
                                (same-expression? expression final-expression {} (nth declarations ordinal))))
                     {:ordinal ordinal :scope (.-scope input)
                      :binding original-id :expression original-expression
                      :step step :result-expression final-expression})))
               (range) original represented-pairs)]
          (when (every? some? entries)
            (PrefixCorrespondence. input program entries))))))))

(defn- owned-entries [correspondence frontend program]
  (when (and (instance? PrefixCorrespondence correspondence)
             (identical? program (.-program ^PrefixCorrespondence correspondence)))
    (let [^PrefixInput input (.-input ^PrefixCorrespondence correspondence)]
      (when (identical? frontend (.-frontend input))
        (entry/assert-frontend-source! frontend (get-in frontend [:source-entry :source-var]))
        (entry/observe-frontend-source-identity! frontend)
        (let [entries (.-entries ^PrefixCorrespondence correspondence)]
          (if-not (some #(= :nested-invocation-input (:kind %)) entries)
            entries
            (entry/call-with-definition-inputs
             (fn []
               (entry/observe-frontend-source-identity! frontend)
               (let [refreshed
                     (mapv
                      (fn [{:keys [kind construction normalization represented boundaries] :as original}]
                        (if-not (= :nested-invocation-input kind)
                          original
                          (do
                            (entry/assert-producer-current! construction)
                            (when (transport/source-construction-view construction normalization represented)
                              ;; The construction phase predates these dispatch/return
                              ;; joins. Reacquire their actual guards under THIS owner.
                              (let [current (mapv #(reference-indy-boundary (:frame %)) boundaries)
                                    fields [:kind :result-carrier :body-return-carrier :invocation-method
                                            :compute-method :parameter-carriers :compute-parameter-carriers]]
                                (when (and (every? some? current)
                                           (every? true? (map (fn [old new]
                                                               (and (identical? (:receiver old) (:receiver new))
                                                                    (identical? (:implementation old) (:implementation new))
                                                                    (= (select-keys (:bridge old) fields)
                                                                       (select-keys (:bridge new) fields))))
                                                             boundaries current)))
                                  (assoc original :boundaries current))))))) entries)]
                 (when (every? some? refreshed) refreshed))))))))))

(defn original-terminal-invocation
  "Read only an exact source-owned terminal join. Maps/copies are not authority.
   Recheck original completion observations under the current enclosing history."
  [correspondence frontend program contract]
  (entry/call-with-optional-definition-inputs
   (fn []
     (let [entries (owned-entries correspondence frontend program)
           candidate (when (= 1 (count entries)) (first entries))]
       (when (and (= :reduction-terminal (:kind candidate))
                  (identical? contract (:contract candidate)))
         (when-let [observation (observed-reduction-interval
                                 (.-input ^PrefixCorrespondence correspondence))]
           (assoc candidate :observation observation)))))))

(defn- original-literal-reduction
  "Common original count/index/seed join; it proves no recurrence or read."
  [correspondence frontend program expected-count]
  (entry/call-with-optional-definition-inputs
   (fn []
     (let [contract (get-in program [:attributes :invocation-plan :attributes :terminal-host-result])
           terminal (original-terminal-invocation correspondence frontend program contract)
           observed (:observation terminal)
           producer (get-in terminal [:producer-source :original])
           reduction (get-in observed [:producer :reduction])
           initializers (get-in reduction [:loop :initializers])
           [index-init seed-init] initializers
           seed (when terminal (closed-source-literal frontend (:initializer seed-init) (:before seed-init)))
           index (when terminal (closed-source-literal frontend (:initializer index-init) (:before index-init)))
           completed (get-in observed [:producer :completed-value])
           carrier (or (:builtin-class completed) (:carrier completed))
           dtype (cond (contains? #{Float Float/TYPE} carrier) :float
                       (contains? #{Double Double/TYPE} carrier) :double)
           seed-class (case dtype :float Float :double Double nil)
           equation (first (:equations program))
           algorithm (:algorithm equation)
           inner (when (dialect/program-form? algorithm) (dialect/equations algorithm))
           semantic-equation (first inner)
           parts (when semantic-equation (dialect/operation-parts semantic-equation))
           attributes (:attributes parts)
           constructed (get-in terminal [:producer-source :constructed])
           constructed-seed (when (and (seq? constructed) (= 6 (count constructed)))
                              (closed-source-literal frontend (nth constructed 2) (:before seed-init)))]
       (when (and terminal dtype (contains? #{0 1} expected-count)
                  (= expected-count (get-in observed [:closed-count :value]))
                  (= :known (get-in reduction [:loop :status])) (= 2 (count initializers))
                  (= (nth producer 1 nil) (:binder seed-init))
                  (= (nth producer 3 nil) (:binder index-init))
                  (= :int (:input-type index-init) (:completed-type index-init))
                  (instance? Integer (:value index)) (zero? (:value index))
                  (completion/bytecode-operand-context? (:context index-init))
                  (= dtype (:input-type seed-init) (:completed-type seed-init))
                  (completion/bytecode-operand-context? (:context seed-init))
                  (identical? seed-class (class (:value seed)))
                  (semantic/equivalent? (:value seed) (:value constructed-seed))
                  (= 1 (count (:equations program)) (count inner))
                  (= 'reduce (:kind parts))
                  (integer? (:extent attributes)) (= expected-count (:extent attributes))
                  (= [dtype] (:dtypes attributes))
                  (= 1 (count (:identities attributes)))
                  (semantic/equivalent? (:value seed)
                                        (:value (numeric-constant/value (first (:identities attributes)))))
                  (nil? (:result-transform attributes))
                  (empty? (get-in program [:attributes :invocation-plan :steps])))
         {:dtype dtype :seed (:value seed) :initializer seed-init
          :count (:closed-count observed) :equation (:id equation)
          :result (first (:results equation)) :terminal terminal})))))

(defn original-empty-reduction
  "Revalidate the original zero-count reduction and its completed literal seed.
   The dead recurrence's annotations grant no evidence. The eventual owner must
   independently validate the native empty-fold schedule and physical demand."
  [correspondence frontend program]
  (original-literal-reduction correspondence frontend program 0))

(defn original-singleton-reduction
  "Join original literal-one count, initializer, read/update and signed Int control.
   The existing opaque correspondence owns the exact frontend/program endpoints.
   The singleton domain discharges the step proof's coordinate obligation: index
   zero widens exactly, increments to one without overflow, then the loop exits.
   Original nullable read completion, physical scheduling/storage and execution
   still require the enclosing owner; this query does not read an array."
  [correspondence frontend program]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when-let [literal (original-literal-reduction correspondence frontend program 1)]
       (let [terminal (:terminal literal)
             original (get-in terminal [:producer-source :original])
             observed (get-in terminal [:observation :producer :reduction])
             index (nth original 3)
             bound (get-in observed [:bound :binder])
             control (get-in observed [:exit :control])
             increment (first (get-in observed [:recur :values]))
             increment-expression (:operand increment)
             description
             ((requiring-resolve 'raster.compiler.passes.parallel.soac-lower/project-typed-reduce)
              (:algorithm (first (:equations program))))
             step ((requiring-resolve
                    'raster.compiler.passes.parallel.scalar-expression-body/check-completed-reduction-step)
                   frontend original (:reduction description))]
         (when (and step (= :completed-selected-reduction-step (:scope step))
                    (= (:dtype literal) (:dtype step))
                    (= {:source :int :product :long :relation :exact-widening
                        :domain-correspondence :unproved} (:coordinate-obligation step))
                    (= :int-less-than (:kind control)) (= :if-icmpge (:false-branch control))
                    (identical? (:predicate control) (second (get-in observed [:exit :expression])))
                    (= [index bound] (:operands control)) (= [:int :int] (:input-types control))
                    (= 2 (count (descriptor/call-args (:predicate control))))
                    (every? true? (map identical? (:operands control)
                                      (descriptor/call-args (:predicate control))))
                    (identical? (get-in observed [:loop :locals index])
                                (get-in control [:locals index]))
                    (identical? (get-in observed [:bound :after])
                                (get-in control [:locals bound]))
                    (= 2 (count (get-in observed [:recur :values]))
                       (count (get-in observed [:recur :targets])))
                    (identical? (get-in observed [:loop :locals index])
                                (get-in observed [:recur :targets 0]))
                    (= :int (get-in observed [:recur :targets 0 :type]))
                    (= :int (:input-type increment) (:type increment))
                    (seq? increment-expression) (= 2 (count increment-expression))
                    (= 'clojure.core/unchecked-inc-int (first increment-expression))
                    (= index (second increment-expression))
                    (identical? (get-in observed [:loop :locals index])
                                (get-in increment [:locals index]))
                    (= 1 (:bound description)) (= index (:idx description))
                    (= #{(:array step)} (:inputs description))
                    (= #{(:result literal)} (:outputs description))
                    (empty? (:scalars description))
                    (semantic/equivalent? (:seed literal)
                                         (:value (numeric-constant/value
                                                  (get-in description [:reduction :components 0 :neutral])))))
           (assoc literal :array (:array step) :description description :step step
                  :domain {:count 1 :source-index :int :native-index :long
                           :coordinates [0] :widening :exact :backedge :no-overflow}
                  :read-completion :unproved)))))))

(defn original-step
  "Query an exact constructed step under its exact original frontend/program.
   This proves correspondence only. It does not authorize interpreting source,
   manufacturing bytecode contexts or adapting an Object result to a primitive.
   The enclosing owner still owes complete namespace/definition history guards."
  [correspondence frontend program step]
  (when (invocation/scalar-compute? step)
    (let [matches (filterv #(identical? step (:step %))
                           (owned-entries correspondence frontend program))]
      (when (= 1 (count matches)) (first matches)))))

(defn prepare-selected-calls!
  "Explicit equation-first producer preparation, not an occurrence query.
   Prepare original direct registered invk initializers over primitive Long
   public parameters/literals, then their actual caller. Signature observations
   do not import an old emitted-module guard before renewal. Unsupported arguments
   and transformed/foreign prefix ownership abstain without compilation.
   Promotion/renewal invalidates the ORIGINAL owner and must escape to its whole
   compilation retry. This grants no source executor, result unboxing or purity."
  [correspondence frontend program prepare!]
  (entry/call-with-optional-definition-inputs
   (fn []
     (when-let [entries (owned-entries correspondence frontend program)]
       (when entry/*emission-inputs*
         (let [caller (entry/assert-frontend-selection!
                       frontend (get-in frontend [:source-entry :source-var]))
               signature (entry/selected-parameter-carriers
                          (:operation caller) (:implementation-op caller)
                          (count (:parameters caller)))
               long-ports (into #{} (keep (fn [[id carrier]]
                                           (when (= Long/TYPE carrier) id)))
                                (map vector (:parameters signature) (:parameter-carriers signature)))
               callees
               (into []
                     (keep
                      (fn [{:keys [expression step kind]}]
                        (entry/call-with-optional-definition-inputs
                         (fn []
                           (let [{:keys [dispatch? semantic-op implementation-op arguments lexical-callee]}
                                 (descriptor/call-description expression)]
                             (when (and (nil? kind) signature (invocation/scalar-compute? step)
                                        dispatch? (not lexical-callee)
                                        (every? #(or (instance? Long %)
                                                     (and (symbol? %) (contains? long-ports %)))
                                                arguments))
                               (when-let [selected (entry/selected-parameter-carriers
                                                    semantic-op implementation-op (count arguments))]
                                 (let [receiver (get-in selected [:definition :receiver-var])]
                                   (when (and (every? #(= Long/TYPE %) (:parameter-carriers selected))
                                              (= Object (:result-carrier selected))
                                              (not (identical? receiver (:receiver-var caller))))
                                     receiver)))))))))
                     entries)]
           (when (and (:receiver-var caller) (seq callees))
             ;; All selections are authenticated before the first callback.
             ;; The service keeps this order and owns compilation separately.
             (entry/prepare-owned-definition-inputs! prepare! (conj callees (:receiver-var caller)))
             true)))))))
