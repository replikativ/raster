(ns raster.ad.normalized-scalar
  "Opt-in scalar AD over checked, unscheduled TypedSOAC equations.

   No derivative registry, runtime evaluator or target schedule lives here. Existing templates
   emit the linear rules; the ordinary compiler consumes the resulting retained let forms.
   The closed first vertical deliberately rejects effects, scopes and active precision changes."
  (:require [clojure.set :as set]
            [raster.ad.templates :as templates]
            ;; Existing rule emitters produce grad-acc; its runtime Var must exist
            ;; before source/rule epochs and executable dependencies are captured.
            [raster.ad.reverse]
            [raster.ad.reverse.normalize :as anf]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-fusion :as fusion]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.core :as core]))

(defn- fail! [reason data]
  (throw (ex-info "program is outside the checked normalized scalar AD boundary"
                  (assoc data :reason reason :representation :normalized-scalar))))

(def ^:private ^:dynamic *proof-dependencies* nil)

(defn- same-dependency? [a b]
  (and (identical? (:var a) (:var b))
       (= (:bound? a) (:bound? b))
       (identical? (:value a) (:value b))
       (= (:metadata a) (:metadata b))))

(defn- observe-dependency!
  "Keep the first effective observation, never bless a later replacement.
   Collection is request-local; refusal is deferred to preserve earlier diagnostics."
  [observations sym snapshot]
  (swap! observations
         (fn [state]
           (if-let [first-observation (get-in state [:dependencies sym])]
             (cond-> state
               (not (same-dependency? first-observation snapshot))
               (update :disagreements conj sym))
             (assoc-in state [:dependencies sym] snapshot)))))

(defn- dependency-snapshot [v]
  (let [is-bound (bound? v)]
    {:var v :bound? is-bound :value (when is-bound (var-get v)) :metadata (meta v)}))

(defn- check-dependencies! [observations source-ns]
  (let [{:keys [dependencies disagreements]} @observations]
    (doseq [[sym snapshot] dependencies]
      (let [v (ns-resolve source-ns sym)]
        (when-not (and (not (contains? disagreements sym))
                       (var? v) (same-dependency? snapshot (dependency-snapshot v)))
          (fail! :ad-normalized-execution-callee {:callee sym}))))))

(defn- scalar-dtype! [value]
  (av/validate! value)
  (when-not (and (= :tensor (:kind value)) (= [] (:shape value))
                 (contains? #{:float :double} (:dtype value)) (empty? (:effects value)))
    (fail! :ad-normalized-scalar-space {:value value}))
  (:dtype value))

(defn- tag [dt] (dtype/scalar-tag-for-dtype dt))
(defn- tagged [expression dt]
  (if (instance? clojure.lang.IObj expression)
    (vary-meta expression assoc :raster.type/tag (tag dt)) expression))
(defn- literal-dtype [expression]
  (cond (instance? Double expression) :double
        (instance? Float expression) :float))
(defn- expression-dtype [values expression]
  (if (symbol? expression)
    (some-> (get values expression) scalar-dtype!)
    (literal-dtype expression)))

(defn- call! [values expression]
  (let [callee (:implementation-op (descriptor/call-description expression))]
    (when (contains? values callee)
      (fail! :ad-normalized-callee {:expression expression :lexical-callee callee})))
  (let [{:keys [operation lexical-callee arguments] :as call}
        (descriptor/call-description expression)
        [template template-identity] (templates/resolve-template operation)
        canonical (cond (or (descriptor/addition-op? operation)
                            ;; Existing polymorphic cotangent addition, restricted here to
                            ;; checked non-nil scalar operands; its nil case cannot occur.
                            (= 'raster.ad.reverse/grad-acc template-identity)) '+
                        (descriptor/subtraction-op? operation) '-
                        (descriptor/multiplication-op? operation) '*
                        (descriptor/division-op? operation) '/)
        cast-tag (descriptor/cast-result-tag operation)
        [forward-template forward-owner] (when-not (or canonical cast-tag)
                                            (templates/resolve-forward-template operation))
        unary? (and (= 1 (count arguments) (count (:params forward-template)))
                    (= {:class :elementwise :in #{0}} (:structure forward-template))
                    (descriptor/scalar-op? forward-owner)
                    (= :pure (effects/analyze-effect expression))
                    (:grads-fn forward-template))]
    (when (or lexical-callee (nil? operation))
      (fail! :ad-normalized-callee {:expression expression :call call}))
    (when-not (or (and (contains? '#{+ - * /} canonical)
                       (contains? #{1 2} (count arguments))
                       (or (= 2 (count arguments)) (= '- canonical)))
                  (and (contains? '#{float double} cast-tag) (= 1 (count arguments)))
                  unary?)
      (fail! :ad-normalized-operation {:expression expression :operation operation}))
    (assoc call :template (if unary? forward-template template)
           :template-identity (if unary? forward-owner template-identity)
           :canonical (if unary? :nonlinear-unary canonical) :cast-tag cast-tag)))

(defn- equation-expression [equation]
  (let [{:keys [kind captures lambda]} (soac/operation-parts equation)]
    (when (or (not= 'scalar kind) (not= 1 (count (nth equation 2))))
      (fail! :ad-normalized-region {:equation equation}))
    (let [{:keys [parameters locals body-results]} (soac/lambda-parts lambda)]
      (when (or (seq locals) (not= 1 (count body-results)))
        (fail! :ad-normalized-region {:equation equation}))
      (util/subst-syms (zipmap parameters captures) (first body-results)))))

(defn- selected-call-contract!
  "Check retained calls against the existing non-invoking dispatch authority.
   Runtime implementation objects are compared here, never serialized as AD identities."
  [call argument-dtypes result-dtype]
  (let [{:keys [operation dispatch? implementation-op canonical arguments]} call
        operation-var (when (symbol? operation) (resolve operation))
        operation-snapshot (when (var? operation-var) (dependency-snapshot operation-var))]
    (when (and *proof-dependencies* operation-snapshot)
      (observe-dependency! *proof-dependencies* operation operation-snapshot))
    (when (and (= :nonlinear-unary canonical)
               (not (or dispatch? (:raster.core/dispatch-table (:metadata operation-snapshot)))))
      ;; Java primitives need the existing reflected return signature, not merely
      ;; homogeneous caller assertions. Math/exp(Float) cannot acquire Float semantics.
      (let [native-result (inference/static-method-return-tag
                           operation arguments (zipmap arguments (mapv tag argument-dtypes)))]
        (when-not (= (tag result-dtype) native-result)
          (fail! :ad-normalized-call-contract
                 {:operation operation :arguments (mapv tag argument-dtypes)
                  :result (tag result-dtype) :native-result native-result}))))
    (when (or dispatch? (:raster.core/dispatch-table (:metadata operation-snapshot)))
      (let [argument-tags (mapv tag argument-dtypes)
            selected (dispatch/selected-call-signature
                      operation argument-tags
                      (inference/registered-call-selection operation argument-tags))
            implementation-var (when (and dispatch? (symbol? implementation-op))
                                 (resolve implementation-op))
            implementation-snapshot (when (var? implementation-var)
                                      (dependency-snapshot implementation-var))]
        (when (and *proof-dependencies* implementation-snapshot)
          (observe-dependency! *proof-dependencies* implementation-op implementation-snapshot))
        ;; This scalar vertical admits exact registered homogeneous methods only.
        ;; Promotion or a merely applicable parametric template is not a proof.
        (when-not (and (= :registered (:selection selected))
                       (= argument-tags (:tags selected))
                       (= (tag result-dtype) (:return-tag selected))
                       (not (some some? (:promotion-casts selected)))
                       (or (not dispatch?)
                           (and (var? implementation-var) (:bound? implementation-snapshot)
                                (some? (:typed-impl selected))
                                (identical? (:value implementation-snapshot) (:typed-impl selected)))))
          (fail! :ad-normalized-call-contract
                 {:operation operation :implementation implementation-op
                  :arguments argument-tags :result (tag result-dtype)
                  :selected (select-keys selected
                                         [:selection :mangled-sym :tags :return-tag
                                          :promotion-casts])}))))))

(defn- binding-dtype! [values binding expression]
  (let [declared (or (some-> (get values binding) scalar-dtype!)
                     (some-> (types/sym-type-tag binding) dtype/dtype-for-scalar-tag))
        retained (some-> (types/sym-type-tag expression) dtype/dtype-for-scalar-tag)
        actual (cond
                 (or (symbol? expression) (number? expression))
                 (expression-dtype values expression)

                 (seq? expression)
                 (let [{:keys [arguments cast-tag] :as call} (call! values expression)
                       arg-types (mapv #(expression-dtype values %) arguments)]
                   (when (some nil? arg-types)
                     (fail! :ad-normalized-unknown-type {:binding binding :expression expression}))
                   (when declared (selected-call-contract! call arg-types declared))
                   (or (some-> cast-tag dtype/dtype-for-scalar-tag)
                       ;; This is a homogeneity check, not result-type inference.
                       ;; The selected/walker result must already be retained.
                       (when (and declared (apply = arg-types) (= declared (first arg-types)))
                         declared))))]
    (when-not (and (contains? #{:float :double} actual)
                   (or (nil? declared) (= actual declared))
                   (or (nil? retained) (= actual retained)))
      (fail! :ad-normalized-unknown-type
             {:binding binding :expression expression :declared declared :actual actual}))
    actual))

(defn- program [input-values bindings outputs]
  (let [{:keys [values equations]}
        (reduce (fn [{:keys [values equations]} [binding expression]]
                  (when (contains? values binding)
                    (fail! :ad-normalized-redefinition {:binding binding}))
                  (when (seq? expression) (call! values expression))
                  (let [binding (if (and (nil? (types/sym-type-tag binding))
                                         (literal-dtype expression))
                                  (tagged binding (literal-dtype expression)) binding)
                        equation (binding [util/*shadowing-locals* (set (keys values))]
                                   (frontend/scalar-equation
                                    {:id binding :sym binding :expr expression}
                                    (into {} (map (fn [[id value]] [id (:dtype value)])) values)
                                    {}))
                        dt (first (get-in (soac/operation-parts equation) [:attributes :dtypes]))
                        result-value (av/tensor {:dtype dt :shape []})
                        _ (scalar-dtype! result-value)
                        values (assoc values binding result-value)
                        _ (binding-dtype! values binding (equation-expression equation))]
                    {:values values
                     :equations (conj equations equation)}))
                {:values input-values :equations []} (partition 2 bindings))
        ;; The functional dialect requires outputs to be equation results, not input aliases.
        missing (remove (set (mapcat #(nth % 2) equations)) outputs)
        used-inputs (set (mapcat soac/operation-inputs equations))]
    (when (seq missing) (fail! :ad-normalized-output {:outputs missing}))
    (soac/make
     (soac/default-program-facts
      {:values values :inputs (vec (filter used-inputs (keys input-values)))
       :equations (into {} (map (fn [eq] [(second eq) (soac/default-equation-facts)])) equations)
       :attributes {:source-arithmetic numerics/retained-source-arithmetic}})
     equations outputs)))

(defn- bindings [program]
  (vec (mapcat (fn [equation] [(first (nth equation 2)) (equation-expression equation)])
               (soac/equations (soac/validate! program)))))

(defn- required-bindings [bindings outputs]
  (let [{:keys [kept]}
        (reduce (fn [{:keys [required kept] :as state} [id expression]]
                  (if (contains? required id)
                    {:required (into (disj required id) (util/free-syms expression))
                     :kept (conj kept [id expression])} state))
                {:required (set outputs) :kept []} (reverse (partition 2 bindings)))]
    (vec (mapcat identity (reverse kept)))))

(defn validate-linear!
  "Check scalar typing and tangent dependence, including bilinear transpose legality.
   coefficient IDs are known primal values; tangent IDs are the only linear inputs."
  [program tangent-ids coefficient-ids]
  (soac/validate! program)
  (when-not (and (empty? (set/intersection (set tangent-ids) (set coefficient-ids)))
                 (set/subset? (set (:inputs (soac/facts program)))
                              (set (concat tangent-ids coefficient-ids)))
                 (every? #(contains? (:values (soac/facts program)) %)
                         (concat tangent-ids coefficient-ids)))
    (fail! :ad-normalized-linear-interface {:tangents tangent-ids :coefficients coefficient-ids}))
  (let [values (:values (soac/facts program))]
    (doseq [[_ value] values] (scalar-dtype! value))
    (let [classes
          (reduce
           (fn [classes [result expression]]
             (binding-dtype! values result expression)
             (let [class (if (seq? expression)
                           (let [{:keys [canonical cast-tag arguments]} (call! values expression)
                                 kinds (mapv #(if (number? %) (if (zero? %) :zero :coefficient)
                                                  (get classes % :coefficient)) arguments)
                                 linear? (some #{:linear} kinds)]
                             (when (and (= '* canonical) (every? #{:linear} kinds))
                               (fail! :ad-normalized-nonlinear {:result result :expression expression}))
                             (when (and (= '/ canonical) (= :linear (second kinds)))
                               (fail! :ad-normalized-nonlinear {:result result :expression expression}))
                             (when (and (= :nonlinear-unary canonical) linear?)
                               (fail! :ad-normalized-nonlinear {:result result :expression expression}))
                             (when (and cast-tag linear?
                                        (not= (expression-dtype values (first arguments))
                                              (:dtype (get values result))))
                               (fail! :ad-normalized-active-conversion {:expression expression}))
                             (when (and (contains? '#{+ -} canonical) linear?
                                        (some #{:coefficient} kinds))
                               (fail! :ad-normalized-affine-offset {:expression expression}))
                             (cond linear? :linear
                                   (= :nonlinear-unary canonical) :coefficient
                                   (and (not (contains? '#{* /} canonical)) (every? #{:zero} kinds)) :zero
                                   :else :coefficient))
                           (if (number? expression) (if (zero? expression) :zero :coefficient)
                               (get classes expression :coefficient)))]
               (assoc classes result class)))
           (into (zipmap coefficient-ids (repeat :coefficient))
                 (zipmap tangent-ids (repeat :linear)))
           (partition 2 (bindings program)))]
      (when-not (every? #(contains? #{:linear :zero} (get classes %)) (soac/outputs program))
        (fail! :ad-normalized-nonlinear-output {:outputs (soac/outputs program)}))
      (into #{} (keep (fn [[id class]] (when (= :linear class) id))) classes))))

(defn- typed-gensym
  ([dt] (typed-gensym dt (fn [prefix _] (gensym (str prefix "__nsad_")))))
  ([dt allocate]
   (fn fresh
     ([prefix] (fresh prefix nil))
     ([prefix requested-tag]
      (when (and requested-tag (not= requested-tag (tag dt)))
        (fail! :ad-normalized-rule-type {:requested requested-tag :dtype dt}))
      (tagged (allocate (str prefix) requested-tag) dt)))))

(defn- recording-fresh [dt phase stream]
  (typed-gensym dt
                (fn [prefix requested-tag]
                  (let [id (tagged (gensym (str prefix "__nsad_")) dt)]
                    (swap! stream conj {:phase phase :prefix prefix :requested-tag requested-tag :id id})
                    id))))

(defn- sum-contributions [contributions fresh]
  (reduce (fn [[bs current] next]
            (let [result (fresh "sum")]
              [(into bs [result (list 'raster.numeric/+ current next)]) result]))
          [[] (first contributions)] (rest contributions)))

(defn- linearize [primal active-inputs fresh]
  (let [values (:values (soac/facts primal))
        tangent-inputs (mapv (fn [id] (fresh (str (name id) "__tangent")
                                           (tag (:dtype (get values id))))) active-inputs)
        {:keys [emitted tangents]}
        (reduce
         (fn [{:keys [emitted tangents]} [result expression]]
           (cond
             (symbol? expression)
             {:emitted emitted :tangents (cond-> tangents (get tangents expression)
                                                 (assoc result (get tangents expression)))}
             (number? expression) {:emitted emitted :tangents tangents}
             :else
             (let [{:keys [template operation cast-tag arguments]} (call! values expression)
                   seeded (keep-indexed (fn [i argument]
                                          (when-let [seed (get tangents argument)] [i seed])) arguments)]
               (if (empty? seeded)
                 {:emitted emitted :tangents tangents}
                 (do
                   (when cast-tag
                     (when-not (= (expression-dtype values (first arguments))
                                  (:dtype (get values result)))
                       (fail! :ad-normalized-active-conversion {:expression expression})))
                   (if cast-tag
                     {:emitted emitted :tangents (assoc tangents result (second (first seeded)))}
                     (do
                       (let [forward-rule (or (templates/op-jvp-fn operation)
                                              (fail! :ad-normalized-missing-forward-rule
                                                     {:operation operation}))
                             [ctx tangent] (forward-rule
                                            {:bindings [] :gensym-fn fresh} arguments
                                            (mapv #(get tangents %) arguments) result fresh)]
                         (when-not tangent
                           (fail! :ad-normalized-missing-contribution {:operation operation}))
                         {:emitted (into emitted (required-bindings (:bindings ctx) [tangent]))
                          :tangents (assoc tangents result tangent)}))))))))
         {:emitted [] :tangents (zipmap active-inputs tangent-inputs)}
         (partition 2 (bindings primal)))
        output (fresh "linear_output")
        dt (:dtype (get values (first (soac/outputs primal))))
        emitted (into emitted [output (or (get tangents (first (soac/outputs primal)))
                                          (if (= dt :float) '(clojure.core/float 0.0) 0.0))])
        ;; Templates may retain nested scalar terms. Reuse the shared ANF boundary;
        ;; fresh values carry the rule's declared homogeneous space, which program
        ;; checks against retained operands and selected method contracts below.
        emitted (anf/anf-normalize-bindings emitted fresh)
        defined (set (take-nth 2 emitted))
        external (set/difference (set (mapcat util/free-syms (take-nth 2 (rest emitted)))) defined)
        coefficients (vec (filter external (concat (:inputs (soac/facts primal))
                                                   (take-nth 2 (bindings primal)))))
        input-values (into (array-map) (map (fn [id] [id (get values id)]) coefficients))
        input-values (into input-values (map (fn [id tangent] [tangent (get values id)])
                                             active-inputs tangent-inputs))
        linear (program input-values emitted [output])]
    (validate-linear! linear tangent-inputs coefficients)
    {:program linear :tangent-inputs tangent-inputs :coefficients coefficients}))

(defn- transpose [linear tangent-inputs coefficients fresh]
  (let [dependent (validate-linear! linear tangent-inputs coefficients)
        values (:values (soac/facts linear))
        output (first (soac/outputs linear))
        seed (fresh "cotangent" (tag (:dtype (get values output))))
        known-bindings (vec (mapcat identity (remove #(contains? dependent (first %))
                                                     (partition 2 (bindings linear)))))
        {:keys [emitted adjoints]}
        (reduce
         (fn [{:keys [emitted adjoints] :as state} [result expression]]
           (if-let [adjoint (get adjoints result)]
             (let [[generated args grads]
                   (if (symbol? expression)
                     [[] [expression] [adjoint]]
                     (if (number? expression)
                       [[] [] []]
                       (let [{:keys [template arguments cast-tag]} (call! values expression)]
                         (if cast-tag
                           [[] arguments [adjoint]]
                           (let [[ctx grads] (templates/instantiate-template-ctx
                                              template arguments result adjoint
                                              {:bindings [] :gensym-fn fresh})]
                             [(required-bindings (:bindings ctx)
                                                 (keep-indexed (fn [i argument]
                                                                 (when (contains? dependent argument)
                                                                   (nth grads i nil))) arguments))
                              arguments grads])))))
                   [extra adjoints]
                   (reduce (fn [[extra adjoints] [argument contribution]]
                             (if (and (contains? dependent argument) contribution)
                               (if-let [previous (get adjoints argument)]
                                 (let [[bs sum] (sum-contributions [previous contribution] fresh)]
                                   [(into extra bs) (assoc adjoints argument sum)])
                                 [extra (assoc adjoints argument contribution)])
                               [extra adjoints]))
                           [[] adjoints] (map vector args grads))]
               {:emitted (into emitted (concat generated extra)) :adjoints adjoints}) state))
         {:emitted known-bindings :adjoints {output seed}}
         (reverse (partition 2 (bindings linear))))
        outputs (mapv (fn [_] (fresh "gradient")) tangent-inputs)
        zero (if (= :float (:dtype (get values output))) '(clojure.core/float 0.0) 0.0)
        emitted (into emitted (mapcat (fn [input result] [result (get adjoints input zero)])
                                      tangent-inputs outputs))
        emitted (anf/anf-normalize-bindings emitted fresh)
        input-values (into (array-map) (map (fn [id] [id (get values id)]) coefficients))]
    {:program (program (assoc input-values seed (get values output)) emitted outputs)
     :seed seed}))

(defn ^:no-doc revisions []
  ;; Effect analysis cold-loads numerical libraries and their AD/dispatch declarations. Establish
  ;; that shared semantic context before any construction or execution authority snapshots epochs.
  ;; The realized delay is cheap on hot calls; later mutations must still fail the existing fences.
  (effects/init-raster-context!)
  {:template-revision (templates/registry-revision)
   :compiler-revision (dispatch/compiler-definition-revision)})

(defn- validate-derivation!
  "Replay existing rules with exact recorded typed IDs. No recursive bundle validation,
   fresh fallback, alpha-normalization or supplied linear program feeds the transpose."
  [{:keys [primal params active-inputs generation] :as bundle}]
  (let [before (revisions)
        dt (:dtype (get-in (soac/facts primal) [:values (first params)]))
        source-ids (set (keys (:values (soac/facts primal))))
        ids (when (vector? generation) (mapv :id generation))]
    (when-not (= before (select-keys bundle [:template-revision :compiler-revision]))
      (fail! :ad-normalized-stale-rules
             {:expected (select-keys bundle [:template-revision :compiler-revision]) :actual before}))
    (when-not (and (vector? generation)
                   (every? (fn [entry]
                             (and (map? entry) (= #{:phase :prefix :requested-tag :id} (set (keys entry)))
                                  (contains? #{:linear :transpose} (:phase entry))
                                  (string? (:prefix entry))
                                  (or (nil? (:requested-tag entry)) (= (tag dt) (:requested-tag entry)))
                                  (symbol? (:id entry)) (nil? (namespace (:id entry)))
                                  (= {:raster.type/tag (tag dt)} (meta (:id entry))))) generation)
                   (= (count ids) (count (distinct ids)))
                   (empty? (set/intersection source-ids (set ids))))
      (fail! :ad-normalized-generation-stream {}))
    (let [position (atom 0)
          replay (fn [phase]
                   (typed-gensym dt
                                 (fn [prefix requested-tag]
                                   (let [entry (get generation @position)]
                                     (when-not (= {:phase phase :prefix prefix :requested-tag requested-tag}
                                                  (select-keys entry [:phase :prefix :requested-tag]))
                                       (fail! :ad-normalized-generation-stream {:position @position :phase phase}))
                                     (swap! position inc)
                                     (:id entry)))))
          regenerated (linearize primal active-inputs (replay :linear))
          transposed (transpose (:program regenerated) (:tangent-inputs regenerated)
                                (:coefficients regenerated) (replay :transpose))]
      (when-not (= before (revisions))
        (fail! :ad-normalized-stale-rules {:before before :after (revisions)}))
      (when-not (= @position (count generation))
        (fail! :ad-normalized-generation-stream {:position @position :count (count generation)}))
      (when-not (fingerprint/equivalent?
                 (select-keys bundle [:linear :transpose :tangent-inputs :residuals :cotangent-seed])
                 {:linear (:program regenerated) :transpose (:program transposed)
                  :tangent-inputs (:tangent-inputs regenerated) :residuals (:coefficients regenerated)
                  :cotangent-seed (:seed transposed)})
        (fail! :ad-normalized-derivation-correspondence {})))))

(defn- scalar-program-source [program parameters]
  {:parameters parameters
   :body (list 'let* (bindings program) (soac/outputs program))})

(defn ^:no-doc execution-sources
  "Project the raw primal, linear and transpose scalar programs for executable dependency capture.

   Generated map lambdas may use an independently proved canonical operation spelling. These raw
   projections keep every original helper Var and rule-produced call in the execution guard. This
   is not source importing, host materialization, or an independent validator."
  [bundle]
  (if-let [nodes (get-in bundle [:map-programs :proof :nodes])]
    (vec
     (mapcat
      (fn [node]
        (if-let [scalar (:scalar-derivation node)]
          [(scalar-program-source (:primal scalar) (:params scalar))
           (scalar-program-source (:linear scalar) (:inputs (soac/facts (:linear scalar))))
           (scalar-program-source (:transpose scalar) (:inputs (soac/facts (:transpose scalar))))]
          (let [{:keys [primal interface]} (:primal-derivation node)]
            [(scalar-program-source primal (:params interface))])))
      nodes))
    [(scalar-program-source (:primal bundle) (:params bundle))
     (scalar-program-source (:linear bundle) (:inputs (soac/facts (:linear bundle))))
     (scalar-program-source (:transpose bundle) (:inputs (soac/facts (:transpose bundle))))]))

(defn ^:no-doc execution-source
  "Project the validated raw scalar primal for compatibility with existing guard callers."
  [bundle]
  (scalar-program-source (:primal bundle) (:params bundle)))

(defn ^:no-doc execution-guard
  "Capture effective executable Var identities in already checked source projections.
   Scope decomposition is owned by form/scope-info. This is a sequential staleness
   fence, not a concurrent Var-mutation/ABA or callable-internal-mutation guarantee."
  [source-ns expected projections]
  (let [dependencies (atom {:dependencies {} :disagreements #{}})
        resolve-var (fn [sym] (when (symbol? sym) (ns-resolve source-ns sym)))]
    (letfn [(capture-var [sym]
              (let [v (resolve-var sym)
                    snapshot (when (var? v)
                               ;; Realize before the first observation; never bless a later
                               ;; replacement. The independent post-capture proof remains.
                               (core/ensure-bytecode-implementation! v)
                               (dependency-snapshot v))]
                (when snapshot
                  (when-not (:bound? snapshot)
                    (fail! :ad-normalized-execution-callee {:callee sym}))
                  (observe-dependency! dependencies sym snapshot))))
            (capture [expression locals]
              (let [{:keys [implementation-op lexical-callee dispatch? operation]}
                    (descriptor/call-description expression)]
                (when-not lexical-callee
                  (when-not (contains? locals implementation-op)
                    (capture-var implementation-op))
                  ;; Retained dispatch selection belongs to the global semantic
                  ;; operation as well as the exact executable receiver.
                  (when dispatch? (capture-var operation)))))
            (visit [expression locals]
              (cond
                (seq? expression)
                (when-not (= 'quote (first expression))
                  ;; Scope-owned macros execute at compilation too. Their binder
                  ;; semantics remain exclusively in scope-info; special heads
                  ;; without a Var contribute no executable dependency.
                  (capture expression locals)
                  (if-let [{:keys [scopes outer sequential? rec?]} (form/scope-info expression)]
                    (do
                      (doseq [e outer] (visit e locals))
                      (doseq [{:keys [binders inits body]} scopes]
                        (let [all (into locals (remove nil? binders))]
                          (loop [pairs (seq (map vector binders inits)) visible (if rec? all locals)]
                            (when-let [[binder init] (first pairs)]
                              (visit init visible)
                              (recur (next pairs) (if (and sequential? binder) (conj visible binder) visible))))
                          (doseq [e body] (visit e all)))))
                    (doseq [e expression] (visit e locals))))
                (coll? expression) (doseq [e expression] (visit e locals))))]
      (doseq [{:keys [parameters body]} projections]
        (visit body (set parameters))))
    (fn []
      (when-not (= expected (revisions))
        (fail! :ad-normalized-stale-rules {:expected expected :actual (revisions)}))
      (check-dependencies! dependencies source-ns)
      true)))

(defn- pure-map-proof-namespaces [map-programs]
  (let [nodes (get-in map-programs [:proof :nodes])]
    (if-not (vector? nodes)
      map-programs
      (assoc-in map-programs [:proof :nodes]
                (mapv (fn [node]
                        (if-let [derivation (or (:scalar-derivation node)
                                               (:primal-derivation node))]
                          (let [key (if (:scalar-derivation node)
                                      :scalar-derivation :primal-derivation)
                                derivation (assoc derivation :source-ns
                                                  (ns-name (:source-ns derivation)))
                                derivation (if (= key :primal-derivation)
                                             (update derivation :interface
                                                     #(assoc % :source-ns
                                                             (ns-name (:source-ns %))))
                                             derivation)]
                            (assoc node key derivation))
                          node))
                      nodes)))))

(defn- certificate [bundle]
  (let [bundle (assoc (dissoc bundle :certificate)
                      :source-ns (ns-name (:source-ns bundle)))
        bundle (cond-> bundle
                 (contains? bundle :map-programs)
                 (update :map-programs pure-map-proof-namespaces))]
    (fingerprint/fingerprint bundle)))

(defn- validate-primal! [primal params tags]
  (soac/validate! primal)
  (let [values (:values (soac/facts primal))
        dtypes (mapv dtype/dtype-for-scalar-tag tags)
        produced (set (mapcat #(nth % 2) (soac/equations primal)))]
    (when-not (and (vector? params) (seq params) (every? symbol? params)
                   (= (count params) (count (distinct params)) (count tags))
                   (every? #{:float :double} dtypes) (apply = dtypes)
                   (set/subset? (set (:inputs (soac/facts primal))) (set params))
                   (empty? (set/intersection (set params) produced))
                   (every? true? (map (fn [id dt] (= dt (:dtype (get values id)))) params dtypes))
                   (= 1 (count (soac/outputs primal)))
                   (= (first dtypes) (:dtype (get values (first (soac/outputs primal))))))
      (fail! :ad-normalized-program-interface {:params params :tags tags}))
    (doseq [[_ value] values] (scalar-dtype! value))
    (doseq [[id expression] (partition 2 (bindings primal))]
      (binding-dtype! values id expression)))
  primal)

(declare validate! validate-map-proof! validate-map-programs! validate-sum-programs!
         validate-gather-programs! sum-source-map sole-map-scalar!)

(defn- validate-scalar-correspondence!
  [{:keys [primal linear transpose params tags active-inputs tangent-inputs residuals
           cotangent-seed source-ns] :as bundle}]
  (when-not (= (revisions) (select-keys bundle [:template-revision :compiler-revision]))
    (fail! :ad-normalized-stale-rules {:expected (select-keys bundle [:template-revision :compiler-revision])
                                       :actual (revisions)}))
  (validate-primal! primal params tags)
  (doseq [p [linear transpose]] (soac/validate! p))
  (when-not (and (set/subset? (set (:inputs (soac/facts primal))) (set params))
                 (instance? clojure.lang.Namespace source-ns)
                 (every? #(contains? (:values (soac/facts primal)) %) params)
                 (= (count params) (count tags))
                 (= (count active-inputs) (count tangent-inputs) (count (soac/outputs transpose)))
                 (= (count active-inputs) (count (distinct active-inputs)))
                 (set/subset? (set active-inputs) (set params))
                 (= 1 (count (soac/outputs primal)) (count (soac/outputs linear))))
    (fail! :ad-normalized-bundle-interface {:params params :active active-inputs}))
  (validate-linear! linear tangent-inputs residuals)
  (validate-linear! transpose [cotangent-seed] residuals)
  (let [primal-values (:values (soac/facts primal))
        linear-values (:values (soac/facts linear))
        transpose-values (:values (soac/facts transpose))]
    (doseq [id residuals]
      (when-not (and (get primal-values id)
                     (= (get primal-values id) (get linear-values id) (get transpose-values id)))
        (fail! :ad-normalized-residual-interface {:value id})))
    (doseq [[p t g] (map vector active-inputs tangent-inputs (soac/outputs transpose))]
      (when-not (= (get primal-values p) (get linear-values t) (get transpose-values g))
        (fail! :ad-normalized-tangent-interface {:primal p :tangent t :cotangent g}))))
  bundle)

(defn- validate-scalar-proof!
  [bundle]
  (when-not (= #{:template-revision :compiler-revision :primal :linear :transpose
                 :params :tags :active-inputs :tangent-inputs :residuals :cotangent-seed
                 :generation :source-ns :certificate}
               (set (keys bundle)))
    (fail! :ad-normalized-bundle-interface {:keys (set (keys bundle))}))
  (validate-scalar-correspondence! bundle)
  (when-not (= (:certificate bundle) (certificate bundle))
    (fail! :ad-normalized-certificate {:certificate (:certificate bundle)}))
  (validate-derivation! bundle)
  bundle)

(defn- validate-legacy!
  "Validate the bundle's typed interfaces and retained producer certificate before emission.
   Replays registered scalar rules on every call; no validation cache is used. Correspondence
   with those rules is not independent proof of custom derivative mathematics."
  [{:keys [source-ns] :as bundle}]
  (when-not (instance? clojure.lang.Namespace source-ns)
    (fail! :ad-normalized-bundle-interface {:source-ns source-ns}))
  (binding [*ns* source-ns
            *proof-dependencies* (atom {:dependencies {} :disagreements #{}})]
    (validate-scalar-correspondence! bundle)
    (when (contains? bundle :gather-programs)
      (validate-gather-programs! bundle))
    ;; Preserve all existing family/certificate refusal ordering. Matching a certificate
    ;; never grants derivation authority: current-rule replay remains mandatory afterward.
    (when-not (= (:certificate bundle) (certificate bundle))
      (fail! :ad-normalized-certificate {:certificate (:certificate bundle)}))
    (validate-derivation! bundle)
    ;; Only successful existing proofs reach this sequential identity fence.
    ;; It neither skips proof replay nor bridges separate preparation phases.
    (check-dependencies! *proof-dependencies* source-ns)
    bundle))

(defn- build-bundle [primal {:keys [params tags active-inputs source-ns]} revision]
  (when-not (instance? clojure.lang.Namespace source-ns)
    (fail! :ad-normalized-program-interface {:source-ns source-ns}))
  (binding [*ns* source-ns]
    (validate-primal! primal params tags)
    (when-not (and (instance? clojure.lang.Namespace source-ns)
                   (vector? active-inputs)
                   (= (count active-inputs) (count (distinct active-inputs)))
                   (set/subset? (set active-inputs) (set params)))
      (fail! :ad-normalized-program-interface
             {:params params :active-inputs active-inputs :source-ns source-ns}))
    (let [dt (:dtype (get-in (soac/facts primal) [:values (first params)]))
          generation (atom [])
          {:keys [program tangent-inputs coefficients]} (linearize primal active-inputs (recording-fresh dt :linear generation))
          transposed (transpose program tangent-inputs coefficients (recording-fresh dt :transpose generation))
          _ (when-not (= revision (revisions))
              (fail! :ad-normalized-stale-rules {:before revision :after (revisions)}))
          bundle (merge revision {:primal primal :linear program :transpose (:program transposed)
                                  :params params :tags tags :active-inputs active-inputs
                                  :tangent-inputs tangent-inputs :residuals coefficients
                                  :cotangent-seed (:seed transposed) :generation @generation :source-ns source-ns})]
      (validate! (assoc bundle :certificate (certificate bundle))))))

(defn linearize-program
  "Build checked scalar primal/linear/transpose programs from an existing TypedSOAC program.
   The interface explicitly supplies :params, retained :tags, :active-inputs and :source-ns.
   This compiler entry never walks source, resolves a deftm Var or re-infers its types."
  [primal interface]
  (build-bundle primal interface (revisions)))

(defn- source-envelope?
  [facts equations]
  (let [attributes (dissoc (:attributes facts) :source-arithmetic)
        equation-facts (mapv #(get-in facts [:equations (second %)]) equations)]
    (or (and (empty? attributes) (every? #(empty? (:attributes %)) equation-facts))
        (and (= {:source-dialect :closed-clojure
                 :source-bindings (vec (mapcat #(nth % 2) equations))
                 :host-binding-ids [] :host-read-values [] :host-read-sites [] :allocations []}
                attributes)
             (every? true?
                     (map-indexed
                      (fn [index equation-facts]
                        ;; With no intervening host activity, the first parallel equation
                        ;; ends the frontend's stageable scalar prefix for every later one.
                        (and (= {:source-prefix? (zero? index)} (:attributes equation-facts))
                             (= :analyzed-source (get-in equation-facts [:provenance :front-end]))
                             (= index (get-in equation-facts [:provenance :source-binding-id]))))
                      equation-facts))))))

(defn- map-source-envelope? [facts equation]
  (source-envelope? facts [equation]))

(defn- checked-map-activity!
  [arrays captures options]
  (when-not (and (map? options)
                 (set/subset? (set (keys options)) #{:active-array :active-input :active-inputs})
                 (<= (count (filter #(contains? options %)
                                    [:active-array :active-input :active-inputs])) 1))
    (fail! :ad-normalized-map-active-array {:options options :arrays arrays}))
  (let [plural? (contains? options :active-inputs)
        requested (cond plural? (:active-inputs options)
                        (contains? options :active-input) [(:active-input options)]
                        (contains? options :active-array) [(:active-array options)]
                        (= 1 (count arrays)) [(first arrays)])
        candidates (if plural? arrays
                       (if (contains? options :active-array) arrays (concat arrays captures)))
        active-inputs (when (vector? requested)
                        (mapv (fn [requested]
                                (some #(when (= requested %) %) candidates))
                              requested))]
    (when-not (and (vector? requested) (seq requested)
                   (= (count requested) (count (distinct requested)))
                   (every? some? active-inputs))
      (fail! :ad-normalized-map-active-array
             {:active-inputs requested :arrays arrays :captures captures}))
    active-inputs))

(defn- project-checked-map-lane
  "Shared lane body for a validated source and its validated one-equation projection.
   Private callers own validation/membership. Full original bookkeeping is checked here;
   the source/equation survive beside the explicit typed scalar output, without stage fusion."
  [source map-program equation source-ns output-id options]
  (let [source-facts (soac/facts source)
        {:keys [kind attributes arrays captures lambda]} (soac/operation-parts equation)
        results (nth equation 2)
        facts (soac/facts map-program)
        equation-facts (get-in facts [:equations (second equation)])
        values (:values facts)
        extent (:extent attributes)
        extent-value (get values extent)]
    (when-not (and (= 'map kind)
                   (contains? #{1 2} (count arrays)) (= 1 (count results))
                   (= results (soac/outputs map-program))
                   (empty? (:effects facts)) (empty? (:aliases facts))
                   (empty? (:effects equation-facts)) (empty? (:aliases equation-facts))
                   (source-envelope? source-facts (soac/equations source))
                   (empty? (:effects source-facts)) (empty? (:aliases source-facts))
                   (or (nil? (get-in facts [:attributes :source-arithmetic]))
                       (= numerics/retained-source-arithmetic
                          (get-in facts [:attributes :source-arithmetic])))
                   (set/subset? (set (keys attributes)) #{:index :extent :attributes})
                   (set/subset? (set (keys (:attributes attributes)))
                                #{:stable-array-captures :pointwise-storage-inputs})
                   (every? empty? (vals (:attributes attributes)))
                   (empty? (soac/result-storage map-program (second equation)))
                   (nil? (:result-transform attributes))
                   (empty? (get-in attributes [:attributes :stable-array-captures]))
                   (or (integer? extent)
                       (and (= :tensor (:kind extent-value)) (= [] (:shape extent-value))
                            (dtype/known? (:dtype extent-value))
                            (dtype/integral? (:dtype extent-value))
                            (= {:kind :plain} (:representation extent-value))
                            (nil? (:logical-layout extent-value))
                            (empty? (:effects extent-value)) (empty? (:attributes extent-value))))
                   (instance? clojure.lang.Namespace source-ns))
      (fail! :ad-normalized-map-boundary {:equation equation}))
    (let [active-inputs (checked-map-activity! arrays captures options)
          layout (soac/parameter-layout equation)
          outer->element (zipmap (concat arrays captures)
                                 (concat (:elements layout) (:capture-parameters layout)))
          active-elements (mapv outer->element active-inputs)
          array-value (get values (first arrays))
          result-value (get values (first results))
          dt (:dtype array-value)
          operand-values (mapv values (concat arrays captures results))
          {:keys [parameters locals body-results]} (soac/lambda-parts lambda)]
      (when-not (and (contains? #{:float :double} dt)
                     (= (:shape array-value) (:shape result-value))
                     (every? #(= (:shape array-value) (:shape (get values %))) arrays)
                     (every? (fn [value]
                               (and (= :tensor (:kind value)) (= dt (:dtype value))
                                    (= {:kind :plain} (:representation value))
                                    (nil? (:logical-layout value)) (empty? (:effects value))
                                    (empty? (:attributes value))))
                             operand-values)
                     (every? #(= [] (:shape (get values %))) captures)
                     (every? #(= dt (:dtype %)) locals))
        (fail! :ad-normalized-map-space {:equation equation :values operand-values
                                         :locals locals}))
      ;; Check rather than overwrite contradictory retained binder evidence. The explicit
      ;; local-value dtype is authoritative when the source symbol has no metadata.
      (doseq [id (concat parameters (map :id locals))
              :let [retained (types/sym-type-tag id)]
              :when (and retained (not= dt (dtype/dtype-for-scalar-tag retained)))]
        (fail! :ad-normalized-map-space {:binding id :retained retained :dtype dt}))
      (binding [*ns* source-ns]
        (let [params (mapv #(tagged % dt) parameters)
              output (or output-id ((typed-gensym dt) "map_element_output"))
              input-values (into (array-map)
                                 (map (fn [id] [id (av/tensor {:dtype dt :shape []})])) params)
              expressions (concat (map :init locals) body-results)
              _ (doseq [expression expressions
                        term (tree-seq coll? seq expression)
                        :when (soac/scalar-convert-form? term)
                        :let [attributes (:attributes (soac/scalar-convert-parts term))]]
                  (when-not (= dt (:source-dtype attributes) (:target-dtype attributes))
                    (fail! :ad-normalized-map-space {:conversion term :dtype dt})))
              local-bindings (vec (mapcat (fn [{:keys [id dtype init]}]
                                            [(tagged id dtype) (soac/scalar-converts->source init)]) locals))
              output-expression (soac/scalar-converts->source (first body-results))
              ordinal (atom 0)
              normalized-bindings
              (anf/anf-normalize-bindings
               (into local-bindings [output output-expression])
               (fn [_]
                 ;; Reprojection uses the same output ID. Stable local IDs let relational
                 ;; validation reproduce ANF without comparing generated source strings.
                 (tagged (symbol (str (name output) "__anf_" (swap! ordinal inc))) dt)))
            ;; Check initializers at their retained dtype before scalar-equation's exact
            ;; widening repair. Projection must not manufacture conversion evidence.
              _ (reduce (fn [known [id expression]]
                          (let [known (assoc known id (av/tensor {:dtype dt :shape []}))]
                            (binding-dtype! known id expression)
                            known))
                        input-values (partition 2 normalized-bindings))
              scalar (program input-values
                              normalized-bindings [output])]
          {:source-program source :source-equation equation :primal scalar
           :interface {:params params :tags (vec (repeat (count params) (tag dt)))
                       :active-inputs (mapv #(tagged % dt) active-elements)
                       :source-ns source-ns}})))))

(defn- project-map-equation-lane
  "Checked exact-member lane entry for singleton callers. Spine composition projects its
   equation subprograms once and invokes the same shared lane body directly."
  ([source equation source-ns output-id options]
   (project-map-equation-lane source (soac/equation-subprogram source equation)
                              equation source-ns output-id options))
  ([source subprogram equation source-ns output-id options]
   (project-checked-map-lane source subprogram equation source-ns output-id options)))

(defn- exact-semantic-members?
  [left right]
  (and (= (count left) (count right))
       (every? #(= 1 (count (filter (partial fingerprint/equivalent? %) right))) left)
       (every? #(= 1 (count (filter (partial fingerprint/equivalent? %) left))) right)))

(defn- equation-dataflow
  "Project ordered equation dependencies after one whole-program validation.

   Executable operand occurrences retain duplicates and positions. Extent/control references
   remain separate: they constrain iteration and shape but are not executable data operands.
   This ephemeral view grants no activity, differentiation, fusion or ownership authority."
  [source]
  (let [subprograms (soac/equation-subprograms source)
        equations (vec (soac/equations source))
        nodes
        (mapv (fn [index equation subprogram]
                (let [equation-id (second equation)
                      results (vec (nth equation 2))
                      operands (soac/operation-inputs equation)
                      controls (soac/operation-extents equation)]
                  {:index index :id equation-id :equation equation :subprogram subprogram
                   :results results
                   :operand-occurrences
                   (mapv (fn [position value]
                           {:value value :position position :consumer-index index
                            :consumer-id equation-id})
                         (range) operands)
                   :control-references
                   (into []
                         (keep-indexed
                          (fn [position value]
                            (when (soac/value-id? value)
                              {:value value :position position :consumer-index index
                               :consumer-id equation-id})))
                         controls)}))
              (range) equations subprograms)
        producers
        (into {}
              (mapcat (fn [{:keys [index id results]}]
                        (map-indexed (fn [position value]
                                       [value {:producer-index index :producer-id id
                                               :position position}])
                                     results)))
              nodes)
        consumers (reduce (fn [result occurrence]
                            (update result (:value occurrence) (fnil conj []) occurrence))
                          {} (mapcat :operand-occurrences nodes))
        controls (reduce (fn [result reference]
                           (update result (:value reference) (fnil conj []) reference))
                         {} (mapcat :control-references nodes))]
    {:nodes nodes :producers producers :consumers consumers :controls controls}))

(defn- map-region-description
  "Audit one closed, pure homogeneous map region without choosing differentiation activity.

   Every source equation is independently admitted through the existing map-lane authority.
   Intermediate arrays may fan out, but none may escape the region or participate in host,
   control, effect, alias or physical-storage boundaries. The exact validated source remains
   the only input to the existing fusion authority."
  [source]
  (let [{:keys [nodes consumers controls] :as dataflow} (equation-dataflow source)
        equations (mapv :equation nodes)
        facts (soac/facts source)]
    (when-not (<= 2 (count equations))
      (fail! :ad-normalized-map-chain-boundary {:equations (count equations)}))
    (let [subprograms (mapv :subprogram nodes)
          results (mapv :results nodes)
          public-outputs (set (soac/outputs source))
          intermediate-results (vec (mapcat identity (butlast results)))
          equation-ids (mapv second equations)
          equation-facts (mapv #(get-in facts [:equations %]) equation-ids)
          parts (mapv soac/operation-parts equations)
          extents (mapv #(get-in % [:attributes :extent]) parts)
          result-values (mapv #(get-in facts [:values (first %)]) results)]
      (when-not (and (every? #(= 'map (:kind %)) parts)
                     (every? #(= 1 (count %)) results)
                     (= 1 (count public-outputs))
                     (fingerprint/equivalent? (last results) (soac/outputs source))
                     (empty? (set/intersection public-outputs (set intermediate-results)))
                     (every? #(seq (get consumers %)) intermediate-results)
                     (every? #(empty? (get controls %)) intermediate-results)
                     (every? #(fingerprint/equivalent? (first extents) %)
                             (rest extents))
                     (apply = (map :dtype result-values))
                     (empty? (:effects facts)) (empty? (:aliases facts))
                     (empty? (get-in facts [:attributes :host-read-values]))
                     (empty? (get-in facts [:attributes :host-read-sites]))
                     (every? #(and (empty? (:effects %)) (empty? (:aliases %)))
                             equation-facts)
                     (every? (fn [equation]
                               (empty? (soac/result-storage source (second equation))))
                             equations))
        (fail! :ad-normalized-map-chain-boundary
               {:equations equations :outputs (soac/outputs source)
                :intermediates intermediate-results}))
      (assoc dataflow :equations equations :subprograms subprograms))))

(defn- exact-map-region-stage
  "Derive one transient execution map directly from an exact validated closed map region.
   The untouched multi-equation source remains the public semantic/provenance authority."
  ([source] (exact-map-region-stage source (map-region-description source)))
  ([source {:keys [equations]}]
   (let [source-inputs (:inputs (soac/facts source))
          [stage stats] (fusion/fusion-fixpoint source)
          [stage-equation] (soac/equations stage)
          stage-parts (when stage-equation (soac/operation-parts stage-equation))
          visible-ids (concat (:inputs (soac/facts source)) (soac/outputs source))
          source-equation-ids (mapv second equations)
          stage-equation-id (second stage-equation)
          constituents (get-in (soac/facts stage)
                               [:equations stage-equation-id :attributes
                                :fusion/constituents])]
      (when-not (and (= 1 (count (soac/equations stage)))
                     (= 'map (:kind stage-parts))
                     (= 1 (count (nth stage-equation 2)))
                     (= (set source-equation-ids) (set (keys constituents)))
                     (every? (fn [equation-id]
                               (fingerprint/equivalent?
                                (get-in (soac/facts source) [:equations equation-id])
                                (get constituents equation-id)))
                             source-equation-ids)
                     ;; Fusion may canonicalize the transient executable input order. The
                     ;; retained source remains the public ABI; prove that the stage has the
                     ;; exact same typed members without pretending its internal order is ABI.
                     (exact-semantic-members? source-inputs
                                              (:inputs (soac/facts stage)))
                     (fingerprint/equivalent? (soac/outputs source) (soac/outputs stage))
                     (every? #(fingerprint/equivalent?
                               (get-in (soac/facts source) [:values %])
                               (get-in (soac/facts stage) [:values %]))
                             visible-ids))
        (fail! :ad-normalized-map-chain-boundary
               {:equations equations :stats stats
                :constituents (keys constituents)}))
      stage)))

(defn- project-map-region
  [source source-ns output-id options]
  (let [{:keys [equations subprograms] :as region} (map-region-description source)
        ;; These are source-member validity checks, not activity selection. The final fused
        ;; lane applies the caller's option against the complete external operand boundary.
        local-actives (mapv (fn [equation]
                              (let [{:keys [arrays captures]} (soac/operation-parts equation)]
                                (first (concat arrays captures))))
                            equations)
        _ (doseq [[equation subprogram active] (map vector equations subprograms local-actives)]
            (project-map-equation-lane source subprogram equation source-ns nil
                                       {:active-input active}))
        stage (exact-map-region-stage source region)
        equation (first (soac/equations stage))]
    (assoc (select-keys
            (project-checked-map-lane source stage equation source-ns output-id options)
            [:primal :interface])
           :stage-source stage)))

(defn- project-map-element
  [map-program source-ns output-id options]
  (soac/validate! map-program)
  (when-not (= 1 (count (soac/equations map-program)))
    (fail! :ad-normalized-map-boundary {:equations (count (soac/equations map-program))}))
  (let [equation (first (soac/equations map-program))]
    (when-not (= (nth equation 2) (soac/outputs map-program))
      (fail! :ad-normalized-map-boundary {:equation equation}))
    (select-keys (project-checked-map-lane map-program map-program equation source-ns output-id options)
                 [:primal :interface])))

(defn linearize-map-element
  "Linearize the typed scalar region of one pure, homogeneous functional map.

   Returns the existing scalar bundle, not array execution. :active-inputs selects a nonempty,
   distinct ordered vector of exact source array operand IDs. Singular :active-input may instead
   select one array or scalar capture; :active-array is its array-only alias. Other inputs are
   excluded. This returns a scalar lane,
   not a map gradient: scalar-active map programs require an enclosing ordered sum.
   Retained dtypes are not storage-default guesses."
  ([map-program source-ns] (linearize-map-element map-program source-ns {}))
  ([map-program source-ns options]
   (let [{:keys [primal interface]} (project-map-element map-program source-ns nil options)]
     (linearize-program primal interface))))

(defn- map-residual-roles [bundle source]
  (let [equation (first (soac/equations source))
        {:keys [captures]} (soac/operation-parts equation)
        capture-values (zipmap (:capture-parameters (soac/parameter-layout equation)) captures)
        ;; Lookup follows the validated lambda identity, but the residual ID is the typed scalar
        ;; authority. `select-keys` retained the raw outer-lambda key and silently discarded that
        ;; metadata, making later exact source-scope checks impossible.
        scalars (into {}
                      (keep (fn [id]
                              (when-let [entry (find capture-values id)]
                                [id (val entry)])))
                      (:residuals bundle))]
    ;; Only direct scalar captures have an existing outer scalar value. Snapshot other
    ;; coefficients per lane, including invariant locals, without another activity analysis.
    {:scalars scalars :arrays (vec (remove #(contains? scalars %) (:residuals bundle)))}))

(defn- canonical-scalar-accumulation!
  "Replace one already checked homogeneous scalar grad-acc with its non-nil addition semantics.

   The exact helper Var, retained AVs and selected call contracts prove both operands and the result
   inhabit the same Float/Double scalar space. No spelling, result hint, array value or nil identity
   authorizes this projection. The original scalar program remains the derivation and dependency
   authority."
  [values result expression]
  (if-not (seq? expression)
    expression
    (let [{:keys [template-identity arguments dispatch? implementation-op] :as call}
          (call! values expression)]
      (if-not (= 'raster.ad.reverse/grad-acc template-identity)
        expression
        (let [result-dtype (scalar-dtype! (get values result))
              argument-dtypes (mapv #(expression-dtype values %) arguments)]
          (when-not (and (not dispatch?)
                         (symbol? implementation-op)
                         (identical? #'raster.ad.reverse/grad-acc (resolve implementation-op))
                         (= 2 (count arguments))
                         (= [result-dtype result-dtype] argument-dtypes))
            (fail! :ad-normalized-scalar-accumulation
                   {:result result :expression expression
                    :implementation implementation-op :dispatch dispatch?
                    :result-dtype result-dtype :argument-dtypes argument-dtypes}))
          ;; Validate the exact retained grad-acc selection before projecting its checked non-nil
          ;; scalar case, then prove the replacement through the ordinary addition authority too.
          (binding-dtype! values result expression)
          (selected-call-contract! call argument-dtypes result-dtype)
          (let [addition (tagged (list 'raster.numeric/+ (first arguments) (second arguments))
                                 result-dtype)
                addition-call (call! values addition)]
            (selected-call-contract! addition-call argument-dtypes result-dtype)
            addition))))))

(defn- scalar-program-lambda [program parameters outputs]
  (let [values (:values (soac/facts program))]
    (soac/lambda-form
     parameters
     (mapv (fn [[id expression]]
             (soac/local-value id (:dtype (get values id))
                               (canonical-scalar-accumulation! values id expression)))
           (partition 2 (bindings program)))
     outputs)))

(defn- generated-map-program
  "Assemble a generated map using canonical executable dependencies. Extra retained value IDs
   are physical/capacity facts only: retaining them never makes them execution inputs."
  [values id attributes arrays captures region outputs retained-value-ids]
  (let [equation (list '= id outputs (list 'map attributes arrays captures region))
        inputs (vec (distinct (concat (soac/operation-inputs equation)
                                      (filter soac/value-id? (soac/operation-extents equation)))))]
    (when-let [missing (seq (remove #(contains? values %) retained-value-ids))]
      (fail! :ad-normalized-generated-map-facts {:missing-values (vec missing)}))
    (soac/make
     (soac/default-program-facts
      {:values (select-keys values (concat inputs outputs retained-value-ids)) :inputs inputs
       :equations {id (soac/default-equation-facts)}
       :attributes {:source-arithmetic numerics/retained-source-arithmetic}})
     [equation] outputs)))

(defn- owned-functional-result! [value]
  (when-not (contains? #{nil :owned} (:ownership value))
    (fail! :ad-normalized-map-storage {:output value}))
  (assoc value :ownership :owned))

(defn- derivative-map-operands [values pairs program]
  ;; The scalar program retains its parameter identities; only their placement
  ;; changes. SOAC's array-first/capture-second layout follows the existing AVs.
  (let [scalar? (fn [[_ outer]] (= [] (:shape (get values outer))))
        captures (filterv scalar? pairs)
        arrays (filterv (complement scalar?) pairs)
        ordered (into arrays captures)]
    {:arrays (mapv second arrays) :captures (mapv second captures)
     :lambda (scalar-program-lambda program (mapv first ordered) (soac/outputs program))}))

(defn- allocate-map-stage-interface [bundle source]
  (let [equation (first (soac/equations source))
        layout (soac/parameter-layout equation)
        parts (soac/operation-parts equation)
        element->outer (zipmap (concat (:elements layout) (:capture-parameters layout))
                               (concat (:arrays parts) (:captures parts)))
        active-inputs (mapv element->outer (:active-inputs bundle))
        {:keys [scalars arrays]} (map-residual-roles bundle source)]
    {:active-inputs active-inputs
     :residual-arrays (into {} (map (fn [id] [id (gensym "ad_residual_")])) arrays)
     :scalar-captures scalars
     :tangent-inputs (mapv (fn [_] (gensym "ad_tangent_")) active-inputs)
     :cotangent-input (gensym "ad_cotangent_")
     :tangent-output (gensym "ad_push_")
     :gradient-outputs (mapv (fn [_] (gensym "ad_pull_")) active-inputs)}))

(defn- construct-map-stage
  "Construct the exact executable map stage owned by a scalar AD bundle.

   Initial construction allocates the stage interface once. Reconstruction accepts that
   retained interface without allocating new stage identities, so validation exercises the
   same constructor without changing residual identities."
  ([bundle source]
   (construct-map-stage bundle source (allocate-map-stage-interface bundle source)))
  ([bundle source interface]
   (construct-map-stage bundle source interface
                        {:primal 'ad-map-primal :linear 'ad-map-linear
                         :transpose 'ad-map-transpose}))
  ([bundle source interface equation-ids]
   (let [equation (first (soac/equations source))
        {:keys [attributes arrays captures]} (soac/operation-parts equation)
        {:keys [active-inputs residual-arrays scalar-captures tangent-inputs cotangent-input
                tangent-output gradient-outputs]} interface
        outputs (soac/outputs source)
        input-values (mapv #(get-in (soac/facts source) [:values %]) arrays)
        output-value (get-in (soac/facts source) [:values (first outputs)])
        _ (when-not (and (contains? #{nil :owned} (:ownership output-value))
                         (fingerprint/equivalent?
                          (mapv #(select-keys % [:memory-space :placement :sharding]) input-values)
                          (vec (repeat (count arrays)
                                       (select-keys output-value [:memory-space :placement :sharding])))))
            (fail! :ad-normalized-map-storage {:inputs input-values :output output-value}))
        coefficients (vec (filter #(contains? residual-arrays %) (:residuals bundle)))
        scalar-coefficients (vec (filter #(contains? scalar-captures %) (:residuals bundle)))
        tensor (owned-functional-result! output-value)
        residual-outputs (mapv residual-arrays coefficients)
        active-values (mapv #(get-in (soac/facts source) [:values %]) active-inputs)
        values (merge (:values (soac/facts source))
                      (zipmap (concat outputs residual-outputs [tangent-output]) (repeat tensor))
                      (zipmap tangent-inputs
                              (map #(if (= [] (:shape %)) % (assoc % :ownership :borrowed))
                                   active-values))
                      ;; Every derivative map result has the original map-result geometry.
                      ;; Scalar-active lanes acquire their scalar cotangent only in the enclosing
                      ;; ordered sum; source-array ownership is not transferred to fresh outputs.
                      (zipmap gradient-outputs (repeat tensor))
                      {cotangent-input (assoc tensor :ownership :borrowed)})
        make-map
        (fn [id arrays captures lambda outputs]
          (generated-map-program values id attributes arrays captures lambda outputs []))
        primal-results (into outputs residual-outputs)
        primal-yields (into (soac/outputs (:primal bundle)) coefficients)
        derivative-pairs (fn [seeds outers]
                           (into (mapv vector seeds outers)
                                 (concat (map vector coefficients residual-outputs)
                                         (map (fn [id] [id (get scalar-captures id)]) scalar-coefficients))))
        push (derivative-map-operands values
                                      (derivative-pairs (:tangent-inputs bundle) tangent-inputs)
                                      (:linear bundle))
        pull (derivative-map-operands values (derivative-pairs [(:cotangent-seed bundle)] [cotangent-input])
                                      (:transpose bundle))]
    {:source source :interface interface
     :primal (make-map (:primal equation-ids) arrays captures
                       (scalar-program-lambda (:primal bundle) (:params bundle) primal-yields)
                       primal-results)
     :linear (make-map (:linear equation-ids) (:arrays push) (:captures push) (:lambda push)
                       [tangent-output])
     :transpose (make-map (:transpose equation-ids) (:arrays pull) (:captures pull) (:lambda pull)
                          gradient-outputs)})))

(defn- edge-readable-value?
  [producer consumer]
  (and (= :owned (:ownership producer))
       (or (fingerprint/equivalent? producer consumer)
           (and (contains? #{nil :borrowed} (:ownership consumer))
                (fingerprint/equivalent? (dissoc producer :ownership)
                                         (dissoc consumer :ownership))))))

(defn- append-map-node
  [producer consumer boundary outputs]
  (let [producer-values (:values (soac/facts producer))
        consumer-values (:values (soac/facts consumer))
        _ (doseq [[consumer-id producer-id] boundary]
            (when-not (edge-readable-value? (get producer-values producer-id)
                                            (get consumer-values consumer-id))
              (fail! :ad-normalized-map-edge
                     {:producer producer-id :consumer consumer-id
                      :producer-value (get producer-values producer-id)
                      :consumer-value (get consumer-values consumer-id)})))
        consumer (soac/make
                  (reduce-kv (fn [facts consumer-id producer-id]
                               (assoc-in facts [:values consumer-id]
                                         (get producer-values producer-id)))
                             (soac/facts consumer) boundary)
                  (soac/equations consumer) (soac/outputs consumer))]
    (soac/append-pure-spines producer consumer boundary outputs)))

(defn- graph-spine-description
  [source]
  (let [{:keys [nodes consumers controls] :as graph} (equation-dataflow source)
        results (mapv :results nodes)
        outputs (soac/outputs source)]
    (when-not
     (and (< 1 (count nodes))
          (every? #(= 1 (count %)) results)
          (= (last results) outputs)
          (every? true?
                  (map-indexed
                   (fn [index [result]]
                     (if (= index (dec (count nodes)))
                       true
                       (let [uses (get consumers result)]
                         (and (= 1 (count uses))
                              (= (inc index) (:consumer-index (first uses)))
                              (empty? (get controls result))))))
                   results)))
      (fail! :ad-normalized-map-chain-boundary {:outputs outputs :results results}))
    graph))

(defn- node-equation-ids []
  {:primal (gensym "ad_node_primal_")
   :linear (gensym "ad_node_linear_")
   :transpose (gensym "ad_node_transpose_")})

(defn- node-allocated-ids [node]
  (let [interface (:stage-interface node)]
    (concat (vals (:equation-ids node))
            (vals (:residual-arrays interface))
            (:tangent-inputs interface)
            (when-let [id (:cotangent-input interface)] [id])
            (when-let [id (:tangent-output interface)] [id])
            (:gradient-outputs interface))))

(defn- construct-primal-map-node
  "Construct one checked primal-only map node.  Unlike `construct-map-stage`, this does not
   allocate or manufacture tangent/cotangent identities for a globally inactive node."
  [primal scalar-interface source equation-id]
  (let [equation (first (soac/equations source))
        {:keys [attributes arrays captures]} (soac/operation-parts equation)
        outputs (soac/outputs source)
        input-values (mapv #(get-in (soac/facts source) [:values %]) arrays)
        output-value (get-in (soac/facts source) [:values (first outputs)])
        tensor (owned-functional-result! output-value)
        values (merge (:values (soac/facts source))
                      (zipmap outputs (repeat tensor)))]
    (when-not (and (contains? #{nil :owned} (:ownership output-value))
                   (fingerprint/equivalent?
                    (mapv #(select-keys % [:memory-space :placement :sharding]) input-values)
                    (vec (repeat (count arrays)
                                 (select-keys output-value
                                              [:memory-space :placement :sharding])))))
      (fail! :ad-normalized-map-storage {:inputs input-values :output output-value}))
    (generated-map-program
     values equation-id attributes arrays captures
     (scalar-program-lambda primal (:params scalar-interface) (soac/outputs primal))
     outputs [])))

(defn- build-map-node
  [source source-ns {:keys [equation subprogram results]} active-outers equation-ids]
  (let [{:keys [arrays]} (soac/operation-parts equation)
        witness-active (if (seq active-outers) active-outers [(first arrays)])
        projected (project-map-equation-lane
                   source subprogram equation source-ns nil {:active-inputs (vec witness-active)})
        scalar-interface (:interface projected)]
    (if (seq active-outers)
      (let [scalar (linearize-program (:primal projected) scalar-interface)
            interface (allocate-map-stage-interface scalar subprogram)
            stage (construct-map-stage scalar subprogram interface equation-ids)]
        {:source-equations [(second equation)]
         :execution-equation (second equation)
         :result (first results)
         :active? true
         :scalar-derivation scalar
         :stage-interface interface
         :equation-ids equation-ids
         :stage stage})
      (let [primal-id (:primal equation-ids)]
        {:source-equations [(second equation)]
         :execution-equation (second equation)
         :result (first results)
         :active? false
         :primal-derivation {:source-ns source-ns
                             :primal (:primal projected)
                             :interface scalar-interface}
         :stage-interface {}
         :equation-ids {:primal primal-id}
         :stage {:primal (construct-primal-map-node
                          (:primal projected) scalar-interface subprogram primal-id)}}))))

(defn- compose-primal-nodes [nodes]
  (loop [program (get-in (first nodes) [:stage :primal])
         residuals (vec (vals (get-in (first nodes) [:stage-interface :residual-arrays])))
         previous (:result (first nodes))
         remaining (next nodes)]
    (if-let [node (first remaining)]
      (let [node-program (get-in node [:stage :primal])
            node-residuals (vec (vals (get-in node [:stage-interface :residual-arrays])))
            outputs (into [(:result node)] (concat residuals node-residuals))]
        (recur (append-map-node program node-program {previous previous} outputs)
               (into residuals node-residuals) (:result node) (next remaining)))
      program)))

(defn- local-port-map [node port-key]
  (into (array-map)
        (map vector (get-in node [:stage-interface :active-inputs])
             (get-in node [:stage-interface port-key]))))

(defn- compose-linear-nodes [nodes]
  (let [active-nodes (filterv :active? nodes)]
    (loop [program (get-in (first active-nodes) [:stage :linear])
           tangent-output (get-in (first active-nodes) [:stage-interface :tangent-output])
           previous (:result (first active-nodes))
           remaining (next active-nodes)]
      (if-let [node (first remaining)]
        (let [tangent-input (get (local-port-map node :tangent-inputs) previous)
              output (get-in node [:stage-interface :tangent-output])]
          (when-not tangent-input
            (fail! :ad-normalized-map-chain-boundary
                   {:previous previous :node (:execution-equation node)}))
          (recur (append-map-node program (get-in node [:stage :linear])
                                  {tangent-input tangent-output} [output])
                 output (:result node) (next remaining)))
        program))))

(defn- compose-transpose-nodes [nodes requested]
  (let [active-nodes (filterv :active? nodes)
        downstream (last active-nodes)
        initial-pairs (vec (local-port-map downstream :gradient-outputs))]
    (loop [program (get-in downstream [:stage :transpose])
           pairs initial-pairs
           remaining (next (reverse active-nodes))]
      (if-let [node (first remaining)]
        (let [boundary (:result node)
              cotangent (some (fn [[outer gradient]] (when (= outer boundary) gradient)) pairs)
              retained (filterv #(not= boundary (first %)) pairs)
              node-pairs (vec (local-port-map node :gradient-outputs))
              outputs (mapv second (into retained node-pairs))]
          (when-not cotangent
            (fail! :ad-normalized-map-chain-boundary
                   {:boundary boundary :node (:execution-equation node)}))
          (recur (append-map-node program (get-in node [:stage :transpose])
                                  {(get-in node [:stage-interface :cotangent-input]) cotangent}
                                  outputs)
                 (into retained node-pairs) (next remaining)))
        (let [by-outer (into {} pairs)
              outputs (mapv by-outer requested)]
          (when-not (every? some? outputs)
            (fail! :ad-normalized-map-chain-boundary
                   {:requested requested :available (mapv first pairs)}))
          (soac/make (soac/facts program) (soac/equations program) outputs))))))

(defn- node-proof-interface [nodes requested]
  (let [active-nodes (filterv :active? nodes)
        tangent-ports (into {} (mapcat #(local-port-map % :tangent-inputs) active-nodes))
        gradient-ports (into {} (mapcat #(local-port-map % :gradient-outputs) active-nodes))
        coefficient-key (fn [node coefficient]
                          [(:execution-equation node) coefficient])]
    {:active-inputs requested
     :residual-arrays
     (into {}
           (mapcat (fn [node]
                     (map (fn [[coefficient result]]
                            [(coefficient-key node coefficient) result])
                          (get-in node [:stage-interface :residual-arrays])))
                   nodes))
     :scalar-captures
     (into {}
           (mapcat (fn [node]
                     (map (fn [[coefficient input]]
                            [(coefficient-key node coefficient) input])
                          (get-in node [:stage-interface :scalar-captures])))
                   nodes))
     :tangent-inputs (mapv tangent-ports requested)
     :cotangent-input (get-in (last active-nodes) [:stage-interface :cotangent-input])
     :tangent-output (get-in (last active-nodes) [:stage-interface :tangent-output])
     :gradient-outputs (mapv gradient-ports requested)}))

(defn- build-nodewise-map-proof [source source-ns options]
  (let [{:keys [nodes consumers]} (graph-spine-description source)
        fused (exact-map-region-stage source)
        fused-parts (soac/operation-parts (first (soac/equations fused)))
        requested (checked-map-activity! (:arrays fused-parts) (:captures fused-parts) options)
        _ (when-not (< 2 (count (:arrays fused-parts)))
            (fail! :ad-normalized-map-chain-boundary
                   {:arrays (:arrays fused-parts)}))
        _ (when-not (every? (set (:arrays fused-parts)) requested)
            (fail! :ad-normalized-map-scalar-requires-sum
                   {:active-inputs requested}))
        _ (doseq [active requested]
            (when-not (= 1 (count (get consumers active)))
              (fail! :ad-normalized-map-chain-boundary
                     {:active active :uses (get consumers active)})))
        active-values (atom (set requested))
        built
        (mapv (fn [node]
                (let [{:keys [arrays captures]} (soac/operation-parts (:equation node))
                      local-active (filterv @active-values (concat arrays captures))
                      result (first (:results node))
                      built (build-map-node source source-ns node local-active
                                            (node-equation-ids))]
                  (when (seq local-active) (swap! active-values conj result))
                  built))
              nodes)
        _ (when-not (contains? @active-values (first (soac/outputs source)))
            (fail! :ad-normalized-map-chain-boundary {:active-inputs requested}))
        interface (node-proof-interface built requested)
        primal (compose-primal-nodes built)
        linear (compose-linear-nodes built)
        transpose (compose-transpose-nodes built requested)
        witnesses (mapv #(dissoc % :stage :result :active?) built)]
    {:source-ns source-ns
     :template-revision (templates/registry-revision)
     :compiler-revision (dispatch/compiler-definition-revision)
     :map-programs {:source source :primal primal :linear linear :transpose transpose
                    :interface interface
                    :proof {:authority :normalized-scalar :nodes witnesses}}}))

(defn- validate-map-programs! [{:keys [map-programs source-ns] :as bundle}]
  (let [{:keys [source interface]} map-programs
        region? (< 1 (count (soac/equations source)))
        {:keys [primal stage-source] :as projected}
        ((if region? project-map-region
             (fn [source source-ns output-id options]
               (assoc (project-map-element source source-ns output-id options)
                      :stage-source source)))
         source source-ns (first (soac/outputs (:primal bundle)))
         (let [source-parts (soac/operation-parts (first (soac/equations source)))
               active-inputs (:active-inputs interface)]
           (if (and (= 1 (count active-inputs))
                    (some #{(first active-inputs)} (:captures source-parts)))
             {:active-input (first active-inputs)}
             {:active-inputs active-inputs})))
        scalar-interface (:interface projected)
        interface (:interface map-programs)
        source-parts (soac/operation-parts (first (soac/equations stage-source)))
        source-operands (concat (:arrays source-parts) (:captures source-parts))
        source-active (mapv (fn [active] (some #(when (= active %) %) source-operands))
                            (:active-inputs interface))
        {:keys [scalars arrays]} (map-residual-roles bundle stage-source)
        ids (concat (vals (:residual-arrays interface))
                    (:tangent-inputs interface) [(:cotangent-input interface) (:tangent-output interface)]
                    (:gradient-outputs interface))]
    (when-not (and (fingerprint/equivalent? primal (:primal bundle))
                   (fingerprint/equivalent? source-active (:active-inputs interface))
                   (fingerprint/equivalent? (select-keys bundle [:params :tags :active-inputs])
                                            (select-keys scalar-interface [:params :tags :active-inputs]))
                   (= #{:active-inputs :residual-arrays :scalar-captures :tangent-inputs :cotangent-input
                        :tangent-output :gradient-outputs} (set (keys interface)))
                   (seq (:active-inputs interface))
                   (= (count (:active-inputs interface))
                      (count (:tangent-inputs interface))
                      (count (:gradient-outputs interface)))
                   (fingerprint/equivalent? (set arrays) (set (keys (:residual-arrays interface))))
                   (fingerprint/equivalent? scalars (:scalar-captures interface))
                   (every? symbol? ids) (= (count ids) (count (distinct ids)))
                   (empty? (set/intersection
                            (set ids)
                            (set/union (set (keys (:values (soac/facts source))))
                                       (set (keys (:values (soac/facts stage-source))))))))
      (fail! :ad-normalized-map-interface {:interface interface :projected projected}))
    (when (and (= [] (get-in (soac/facts stage-source)
                             [:values (first (:active-inputs interface)) :shape]))
               (not (:sum-programs bundle)))
      (fail! :ad-normalized-map-scalar-requires-sum {:active-inputs (:active-inputs interface)}))
    (doseq [program (map map-programs [:primal :linear :transpose])] (soac/validate! program))
    ;; Fixed interface IDs make reconstruction deterministic. Exact semantic equality includes
    ;; expression metadata and float bits; no fresh AD generation or printed-string comparison.
    (when-not (fingerprint/equivalent?
               map-programs
               (assoc (construct-map-stage bundle stage-source interface) :source source))
      (fail! :ad-normalized-map-programs {:interface interface})))
  bundle)

(defn- rebuild-submitted-node
  [source source-ns subprogram equation node local-active]
  (let [active? (boolean (seq local-active))
        derivation ((if active? :scalar-derivation :primal-derivation) node)
        _ (when-not (= (second equation) (:execution-equation node))
            (fail! :ad-normalized-map-node-coverage
                   {:expected (second equation) :actual (:execution-equation node)}))
        _ (when-not (identical? source-ns (:source-ns derivation))
            (fail! :ad-normalized-map-node-namespace
                   {:source-ns source-ns :node-ns (:source-ns derivation)}))
        _ (when active? (validate-scalar-proof! derivation))
        {:keys [arrays]} (soac/operation-parts equation)
        witness-active (if (seq local-active) local-active [(first arrays)])
        projected (project-map-equation-lane
                   source subprogram equation source-ns
                   (first (soac/outputs (:primal derivation)))
                   {:active-inputs (vec witness-active)})
        expected-interface (:interface projected)]
    (when-not (and (fingerprint/equivalent? (:primal projected) (:primal derivation))
                   (if active?
                     (fingerprint/equivalent?
                      (select-keys expected-interface [:params :tags :active-inputs])
                      (select-keys derivation [:params :tags :active-inputs]))
                     (and (= #{:source-ns :primal :interface} (set (keys derivation)))
                          (identical? source-ns (:source-ns (:interface derivation)))
                          (fingerprint/equivalent?
                           (dissoc expected-interface :source-ns)
                           (dissoc (:interface derivation) :source-ns)))))
      (fail! :ad-normalized-map-node-derivation
             {:equation (second equation)}))
    (if active?
      (let [interface (:stage-interface node)
            roles (map-residual-roles derivation subprogram)
            ids (concat (vals (:residual-arrays interface))
                        (:tangent-inputs interface)
                        [(:cotangent-input interface) (:tangent-output interface)]
                        (:gradient-outputs interface))]
        (when-not (and (= #{:active-inputs :residual-arrays :scalar-captures
                            :tangent-inputs :cotangent-input :tangent-output
                            :gradient-outputs}
                          (set (keys interface)))
                       (fingerprint/equivalent? (vec local-active)
                                                (:active-inputs interface))
                       (= (count local-active) (count (:tangent-inputs interface))
                          (count (:gradient-outputs interface)))
                       (= (set (:arrays roles)) (set (keys (:residual-arrays interface))))
                       (fingerprint/equivalent? (:scalars roles) (:scalar-captures interface))
                       (every? symbol? ids)
                       (= (count ids) (count (distinct ids)))
                       (empty? (set/intersection
                                (set ids)
                                (set (keys (:values (soac/facts subprogram)))))))
          (fail! :ad-normalized-map-node-interface {:interface interface}))
        (assoc node :result (first (nth equation 2)) :active? true
               :stage (construct-map-stage derivation subprogram interface
                                           (:equation-ids node))))
      (let [primal-id (get-in node [:equation-ids :primal])]
        (when-not (and (= {} (:stage-interface node))
                       (= #{:primal} (set (keys (:equation-ids node))))
                       (symbol? primal-id))
          (fail! :ad-normalized-map-node-interface {:node node}))
        (assoc node :result (first (nth equation 2)) :active? false
               :stage {:primal (construct-primal-map-node
                                (:primal derivation) expected-interface subprogram
                                primal-id)})))))

(defn- reconstructed-map-proof
  [{:keys [map-programs source-ns] :as bundle}]
  (let [{:keys [source interface proof]} map-programs
        submitted (:nodes proof)
        equations (vec (soac/equations source))
        region? (< 1 (count equations))
        fused (when region? (exact-map-region-stage source))
        fused-arrays (when fused (:arrays (soac/operation-parts
                                           (first (soac/equations fused)))))
        nodewise? (and region? (< 2 (count fused-arrays)))
        requested (:active-inputs interface)]
    (when-not (and (= :normalized-scalar (:authority proof))
                   (= #{:authority :nodes} (set (keys proof)))
                   (vector? submitted) (seq submitted)
                   (vector? requested) (seq requested)
                   (= (count requested) (count (distinct requested))))
      (fail! :ad-normalized-map-node-proof {:proof proof :interface interface}))
    (if nodewise?
      (let [{source-nodes :nodes consumers :consumers} (graph-spine-description source)
            allocated-ids (vec (mapcat node-allocated-ids submitted))
            _ (when-not (every? (set fused-arrays) requested)
                (fail! :ad-normalized-map-node-proof
                       {:active-inputs requested :source-arrays fused-arrays}))
            _ (when-not (and (every? symbol? allocated-ids)
                             (= (count allocated-ids) (count (distinct allocated-ids)))
                             (empty? (set/intersection
                                      (set allocated-ids)
                                      (set (keys (:values (soac/facts source)))))))
                (fail! :ad-normalized-map-node-interface {:allocated-ids allocated-ids}))
            _ (when-not (= (count source-nodes) (count submitted))
                (fail! :ad-normalized-map-node-coverage
                       {:source (count source-nodes) :submitted (count submitted)}))
            _ (doseq [active requested]
                (when-not (= 1 (count (get consumers active)))
                  (fail! :ad-normalized-map-chain-boundary {:active active})))
            active-values (atom (set requested))
            nodes
            (mapv (fn [source-node node]
                    (let [{:keys [arrays captures]} (soac/operation-parts (:equation source-node))
                          local-active (filterv @active-values (concat arrays captures))
                          expected-source [(second (:equation source-node))]
                          expected-keys (if (seq local-active)
                                          #{:source-equations :execution-equation
                                            :scalar-derivation :stage-interface :equation-ids}
                                          #{:source-equations :execution-equation
                                            :primal-derivation :stage-interface :equation-ids})
                          _ (when-not (= expected-keys (set (keys node)))
                              (fail! :ad-normalized-map-node-proof
                                     {:node (:execution-equation node)
                                      :keys (set (keys node))}))
                          _ (when-not (= expected-source (:source-equations node))
                              (fail! :ad-normalized-map-node-coverage
                                     {:expected expected-source
                                      :actual (:source-equations node)}))
                          rebuilt (rebuild-submitted-node
                                   source source-ns (:subprogram source-node)
                                   (:equation source-node) node local-active)]
                      (when (seq local-active)
                        (swap! active-values conj (first (:results source-node))))
                      rebuilt))
                  source-nodes submitted)
            expected-interface (node-proof-interface nodes requested)]
        (when-not (contains? @active-values (first (soac/outputs source)))
          (fail! :ad-normalized-map-chain-boundary {:active-inputs requested}))
        {:source source
         :primal (compose-primal-nodes nodes)
         :linear (compose-linear-nodes nodes)
         :transpose (compose-transpose-nodes nodes requested)
         :interface expected-interface
         :proof (assoc proof :nodes (mapv #(dissoc % :stage :result :active?) nodes))})
      (let [stage-source (or fused source)
            equation (first (soac/equations stage-source))
            [node] submitted
            expected-coverage (mapv second equations)
            _ (when-not (and (= 1 (count submitted))
                             (= #{:source-equations :execution-equation :scalar-derivation
                                  :stage-interface :equation-ids}
                                (set (keys node)))
                             (= {:primal 'ad-map-primal :linear 'ad-map-linear
                                 :transpose 'ad-map-transpose}
                                (:equation-ids node))
                             (= expected-coverage (:source-equations node)))
                (fail! :ad-normalized-map-node-coverage
                       {:expected expected-coverage :nodes submitted}))
            scalar (:scalar-derivation node)
            _ (when-not (and (identical? source-ns (:source-ns scalar))
                             (= (second equation) (:execution-equation node)))
                (fail! :ad-normalized-map-node-namespace
                       {:source-ns source-ns :node (:execution-equation node)}))
            _ (validate-scalar-proof! scalar)
            ;; Preserve the complete established singleton/fused-stage correspondence check.
            ;; The node wrapper changes evidence ownership, not the one-stage map contract.
            _ (validate-map-programs!
               (cond-> (assoc (dissoc scalar :certificate)
                              :map-programs (dissoc map-programs :proof))
                 (:sum-programs bundle) (assoc :sum-programs (:sum-programs bundle))))
            source-parts (soac/operation-parts equation)
            options (if (and (= 1 (count requested))
                             (some #{(first requested)} (:captures source-parts)))
                      {:active-input (first requested)}
                      {:active-inputs requested})
            projected (if fused
                        (project-map-region source source-ns
                                            (first (soac/outputs (:primal scalar))) options)
                        (project-map-element source source-ns
                                             (first (soac/outputs (:primal scalar))) options))
            _ (when-not (and (fingerprint/equivalent? (:primal projected) (:primal scalar))
                             (fingerprint/equivalent?
                              (select-keys (:interface projected)
                                           [:params :tags :active-inputs])
                              (select-keys scalar [:params :tags :active-inputs])))
                (fail! :ad-normalized-map-node-derivation
                       {:equation (second equation)}))
            stage (construct-map-stage scalar stage-source (:stage-interface node)
                                       (:equation-ids node))]
        (assoc stage :source source
               :proof (assoc proof :nodes [node]))))))

(defn- validate-proof-header! [bundle expected-keys]
  (let [source-ns (:source-ns bundle)]
    (when-not (and (= expected-keys (set (keys bundle)))
                   (instance? clojure.lang.Namespace source-ns))
      (fail! :ad-normalized-map-node-proof {:source-ns source-ns})))
  (when-not (= (revisions) (select-keys bundle [:template-revision :compiler-revision]))
    (fail! :ad-normalized-stale-rules
           {:expected (select-keys bundle [:template-revision :compiler-revision])
            :actual (revisions)})))

(defn- validate-map-proof-correspondence!
  [{:keys [map-programs] :as bundle}]
  (doseq [program (map map-programs [:primal :linear :transpose])]
    (soac/validate! program))
  (let [expected (reconstructed-map-proof bundle)]
    (when-not (fingerprint/equivalent? (pure-map-proof-namespaces map-programs)
                                       (pure-map-proof-namespaces expected))
      (fail! :ad-normalized-map-programs {:interface (:interface map-programs)})))
  bundle)

(defn validate-map-proof!
  [{:keys [source-ns] :as bundle}]
  (validate-proof-header! bundle #{:source-ns :template-revision :compiler-revision
                                   :map-programs :certificate})
  (binding [*ns* source-ns
            *proof-dependencies* (atom {:dependencies {} :disagreements #{}})]
    (validate-map-proof-correspondence! bundle)
    (when-not (= (:certificate bundle) (certificate bundle))
      (fail! :ad-normalized-certificate {:certificate (:certificate bundle)}))
    (check-dependencies! *proof-dependencies* source-ns))
  bundle)

(defn- validate-map-sum-proof!
  [{:keys [source-ns sum-programs] :as bundle}]
  (validate-proof-header! bundle #{:source-ns :template-revision :compiler-revision
                                   :map-programs :sum-programs :certificate})
  (binding [*ns* source-ns
            *proof-dependencies* (atom {:dependencies {} :disagreements #{}})]
    ;; Establish the exact surrounding map→sum source before that context permits the
    ;; existing scalar-active map stage during correspondence reconstruction.
    (sum-source-map (:source sum-programs) source-ns)
    (sole-map-scalar! bundle)
    (validate-map-proof-correspondence! bundle)
    (validate-sum-programs! bundle)
    (when-not (= (:certificate bundle) (certificate bundle))
      (fail! :ad-normalized-certificate {:certificate (:certificate bundle)}))
    (check-dependencies! *proof-dependencies* source-ns))
  bundle)

(defn validate!
  [bundle]
  (cond
    (contains? bundle :sum-programs) (validate-map-sum-proof! bundle)
    (contains? bundle :map-programs) (validate-map-proof! bundle)
    :else (validate-legacy! bundle)))

(defn- build-map-programs
  [source source-ns options]
  (let [region? (< 1 (count (soac/equations source)))
        fused (when region? (exact-map-region-stage source))
        fused-arrays (when fused (:arrays (soac/operation-parts
                                           (first (soac/equations fused)))))]
    (if (and region? (< 2 (count fused-arrays)))
      (build-nodewise-map-proof source source-ns options)
      (let [{:keys [primal interface stage-source]}
            (if region?
              (project-map-region source source-ns nil options)
              (assoc (project-map-element source source-ns nil options)
                     :stage-source source))
            scalar (linearize-program primal interface)
            equation-ids {:primal 'ad-map-primal :linear 'ad-map-linear
                          :transpose 'ad-map-transpose}
            stage (construct-map-stage scalar stage-source
                                       (allocate-map-stage-interface scalar stage-source)
                                       equation-ids)
            execution-equation (second (first (soac/equations stage-source)))
            node {:source-equations (mapv second (soac/equations source))
                  :execution-equation execution-equation
                  :scalar-derivation scalar
                  :stage-interface (:interface stage)
                  :equation-ids equation-ids}]
        {:source-ns source-ns
         :template-revision (templates/registry-revision)
         :compiler-revision (dispatch/compiler-definition-revision)
         :map-programs (assoc stage :source source
                              :proof {:authority :normalized-scalar :nodes [node]})}))))

(defn linearize-map-program
  "Lift one checked homogeneous map or an admitted closed map region. Regions with at most two
   execution-boundary arrays retain the exact independently rederived fused-map model. A checked
   single-use region with more boundary arrays instead composes its ordered source-node scalar
   derivatives, preserving each node's retained checkpoint. Descriptive numerical facets grant no
   additional fusion or reassociation permission.
   :active-inputs is
   an ordered nonempty vector of exact source array IDs. Singular :active-input and compatibility
   :active-array retain their existing one-active behavior. Scalar capture activity needs a sum."
  ([source source-ns] (linearize-map-program source source-ns {}))
  ([source source-ns options]
   (let [bundle (build-map-programs source source-ns options)]
     (validate! (assoc bundle :certificate (certificate bundle))))))

(defn linearize-map-chain-program
  "Compatibility entry for a multi-equation map composition.

   The original multi-equation source remains the semantic authority. Supported fused frontiers
   retain their transient single-map execution stage; a bounded larger-frontier spine composes
   equation-local stages instead. Validation rederives this choice and exact stage correspondence.
   Canonical adaptation separately checks source, execution scope and residual lineage."
  ([source source-ns] (linearize-map-chain-program source source-ns {}))
  ([source source-ns options]
   (when-not (< 1 (count (soac/equations source)))
     (fail! :ad-normalized-map-chain-boundary
            {:equations (count (soac/equations source))}))
   (linearize-map-program source source-ns options)))

(defn execution-map-scope
  "Return the exact execution source and certified scalar scope of a validated map bundle.

   A singleton retains its original source identity. An admitted closed map region is rederived
   through the same checked fusion path used by construction; callers cannot supply
   or cache a substitute execution stage. Scalar values are limited to the validated primal's
   parameters and ordered binding results; residual-role validation independently proves which
   of those values are saved coefficients."
  [bundle]
  (validate! bundle)
  (let [source (get-in bundle [:map-programs :source])
        proof-nodes (get-in bundle [:map-programs :proof :nodes])
        nodewise? (< 1 (count proof-nodes))
        execution-source (if (and (not nodewise?) (< 1 (count (soac/equations source))))
                           (exact-map-region-stage source)
                           source)
        nodes
        (mapv (fn [node]
                (let [derivation (or (:scalar-derivation node) (:primal-derivation node))
                      primal (:primal derivation)
                      params (or (:params derivation)
                                 (get-in derivation [:interface :params]))]
                  {:source-equations (:source-equations node)
                   :execution-equation (:execution-equation node)
                   :active? (boolean (:scalar-derivation node))
                   :scalar-values (vec (concat params
                                               (map first (partition 2 (bindings primal)))))}))
              proof-nodes)]
    {:source execution-source
     :nodes nodes
     ;; Preserve the established one-stage projection while graph consumers use the
     ;; node-qualified scopes above.  Never flatten graph-local identities into one map.
     :scalar-values (when (= 1 (count nodes)) (:scalar-values (first nodes)))}))

(defn execution-map-source
  "Return only the revalidated execution-source view of `execution-map-scope`."
  [bundle]
  (:source (execution-map-scope bundle)))

(defn- project-indexed-element
  "Check a functional map's exact indirect read using the existing operand/layout authority."
  [source source-ns output]
  (soac/validate! source)
  (when-not (= 1 (count (soac/equations source)))
    (fail! :ad-normalized-indexed-map-boundary {:equations (count (soac/equations source))}))
  (let [facts (soac/facts source)
        [equation :as equations] (soac/equations source)
        {:keys [kind attributes arrays captures lambda]} (soac/operation-parts equation)
        {:keys [parameters locals body-results]} (soac/lambda-parts lambda)
        layout (soac/parameter-layout equation)
        [indices] arrays [data] captures [result] (nth equation 2)
        [index-element] (:elements layout) [data-parameter] (:capture-parameters layout)
        values (:values facts)
        index-value (get values indices) data-value (get values data) result-value (get values result)
        dt (:dtype data-value) it (:dtype index-value)
        extent (:extent attributes)
        plain? (fn [value]
                 (and (= :tensor (:kind value)) (= {:kind :plain} (:representation value))
                      (= 1 (count (:shape value))) (nil? (:logical-layout value))
                      (empty? (:effects value)) (empty? (:attributes value))))
        integral-extent? (fn [id]
                           (or (and (integer? id) (<= 0 id Integer/MAX_VALUE))
                               (let [value (get values id)]
                                 (and (= :tensor (:kind value)) (= [] (:shape value))
                                      (contains? #{:int :long} (:dtype value))
                                      (= {:kind :plain} (:representation value))
                                      (nil? (:logical-layout value))
                                      (empty? (:effects value)) (empty? (:attributes value))))))
        read (first body-results)
        ;; The semantic descriptor supplies operands, not executable receiver proof.
        ;; Initial indexed admission is an actual direct core Var only; retained dispatch
        ;; requires a separate selected implementation contract before it can be admitted.
        read-var (when (and (seq? read) (symbol? (first read))
                            (not= '.invk (first read))
                            (instance? clojure.lang.Namespace source-ns))
                   (ns-resolve source-ns (first read)))]
    (when-not (and (= 1 (count equations)) (= 'map kind)
                   (= 1 (count arrays) (count captures) (count (nth equation 2)))
                   (= [result] (soac/outputs source)) (= 2 (count parameters)) (empty? locals)
                   (instance? clojure.lang.Namespace source-ns)
                   (empty? (:effects facts)) (empty? (:aliases facts))
                   (empty? (get-in facts [:equations (second equation) :effects]))
                   (empty? (get-in facts [:equations (second equation) :aliases]))
                   (map-source-envelope? facts equation)
                   (empty? (soac/result-storage source (second equation)))
                   (set/subset? (set (keys attributes)) #{:index :extent :attributes})
                   (set/subset? (set (keys (:attributes attributes)))
                                #{:stable-array-captures :pointwise-storage-inputs})
                   (empty? (get-in attributes [:attributes :pointwise-storage-inputs]))
                   (fingerprint/equivalent? [data] (get-in attributes [:attributes :stable-array-captures]))
                   (or (nil? (get-in facts [:attributes :source-arithmetic]))
                       (= numerics/retained-source-arithmetic (get-in facts [:attributes :source-arithmetic])))
                   (every? plain? [index-value data-value result-value])
                   (contains? #{:float :double} dt) (= dt (:dtype result-value))
                   (contains? #{:int :long} it)
                   (= [extent] (:shape index-value) (:shape result-value))
                   (integral-extent? extent) (integral-extent? (first (:shape data-value)))
                   (contains? #{nil :owned} (:ownership result-value))
                   (every? #(fingerprint/equivalent?
                              (select-keys data-value [:memory-space :placement :sharding])
                              (select-keys % [:memory-space :placement :sharding]))
                           [index-value result-value])
                   (= 1 (count body-results)) (descriptor/aget-call? read)
                   (identical? #'clojure.core/aget read-var)
                   (not (contains? (set parameters)
                                   (:implementation-op (descriptor/call-description read))))
                   (= [data-parameter index-element] (descriptor/call-args read)))
      (fail! :ad-normalized-indexed-map-boundary {:equation equation}))
    (doseq [[id expected] (concat [[index-element it] [data-parameter (dtype/array-tag-for-dtype dt)]
                                  [read dt]]
                                 (map vector (descriptor/call-args read)
                                      [(dtype/array-tag-for-dtype dt) it]))]
      (when-let [retained (types/sym-type-tag id)]
        (when-not (= retained (if (keyword? expected) (tag expected) expected))
          (fail! :ad-normalized-indexed-map-boundary {:parameter id :retained retained}))))
    (let [element (tagged (symbol (str (name output) "__element")) dt)
          scalar (program {element (av/tensor {:dtype dt :shape []})} [output element] [output])]
      {:primal scalar
       :interface {:params [element] :tags [(tag dt)] :active-inputs [element] :source-ns source-ns}})))

(defn- lift-gather-programs [bundle source interface]
  (let [facts (soac/facts source) [equation] (soac/equations source)
        {:keys [attributes arrays captures lambda]} (soac/operation-parts equation)
        [indices] arrays [data] captures [result] (soac/outputs source)
        {:keys [parameters body-results]} (soac/lambda-parts lambda)
        [index-element data-parameter] parameters
        {:keys [index-snapshot tangent-input tangent-output cotangent-input zero-gradient gradient-output]} interface
        index-value (assoc (get-in facts [:values indices]) :ownership :owned)
        output-value (assoc (get-in facts [:values result]) :ownership :owned)
        data-value (assoc (get-in facts [:values data]) :ownership :owned)
        dt (:dtype data-value)
        extent (:extent attributes)
        capacity-facts (filter #(contains? (:values facts) %) [extent (first (:shape data-value))])
        values (merge (:values facts) {index-snapshot index-value tangent-output output-value
                                      tangent-input (assoc data-value :ownership :borrowed)
                                      cotangent-input (assoc output-value :ownership :borrowed)
                                      zero-gradient data-value gradient-output data-value})
        make-map (fn [id arrays captures region outputs]
                   (generated-map-program values id
                                          (assoc-in attributes [:attributes :stable-array-captures] captures)
                                          arrays captures region outputs capacity-facts))
        ;; Reuse fixed fresh interface IDs as lexical parameters, not conventional stems
        ;; which could capture a source extent or input of the same name.
        destination (with-meta zero-gradient {:raster.type/tag (dtype/array-tag-for-dtype dt)})
        dy (tagged cotangent-input dt) index (tagged index-snapshot (:dtype index-value))
        effects #{:memory/read :memory/write}
        addition (tagged (list 'raster.numeric/+ (list 'clojure.core/aget destination index) dy) dt)
        ;; Destinations are physical storage, not semantic operation-inputs. The declared
        ;; read/write destination owns both sides of the ordered update. Its required
        ;; typed-zero physical binding is supplied by a later execution adapter, not
        ;; duplicated as a capture (the shared dialect forbids overlapping operand roles).
        transpose-equation (list '= 'ad-gather-transpose [gradient-output]
                                 (list 'effect-map
                                       {:index (:index attributes) :extent extent :dtypes [dt]
                                        :iteration-order :sequential}
                                       [cotangent-input index-snapshot] [] [zero-gradient]
                                       (soac/effect-lambda-form [dy index destination]
                                                                [(list 'effect destination :ordered index 1 addition)])))
        transpose-inputs (vec (distinct (concat (soac/operation-inputs transpose-equation)
                                               (filter soac/value-id? (soac/operation-extents transpose-equation)))))
        transpose-facts (soac/default-program-facts
                         {:values (select-keys values (concat transpose-inputs [zero-gradient gradient-output] capacity-facts))
                          :inputs transpose-inputs :effects effects
                          :attributes {:source-arithmetic numerics/retained-source-arithmetic}
                          :equations {'ad-gather-transpose
                                      (assoc (soac/default-equation-facts) :effects effects
                                             :aliases {gradient-output zero-gradient}
                                             :attributes {:result-storage
                                                          [{:destination zero-gradient :access :read-write
                                                            :host-return :effect}]})}})
        _ (binding [*ns* (:source-ns bundle)]
            (selected-call-contract! (call! {} addition) [dt dt] dt))]
    {:source source :interface interface
     :required-zero-gradient {:value zero-gradient :dtype dt :shape (:shape data-value)
                              :ownership :owned :initialization :typed-zero :exclusive true}
     :primal (make-map 'ad-gather-primal arrays captures
                       (soac/lambda-form parameters (conj (vec body-results) index-element))
                       [result index-snapshot])
     :linear (make-map 'ad-gather-linear [index-snapshot] [tangent-input]
                       (soac/lambda-form parameters [(list 'clojure.core/aget data-parameter index-element)])
                       [tangent-output])
     :transpose (soac/make transpose-facts [transpose-equation] [gradient-output])}))

(defn- validate-gather-programs! [{:keys [gather-programs source-ns] :as bundle}]
  (let [{:keys [source interface]} gather-programs
        projected (project-indexed-element source source-ns (first (soac/outputs (:primal bundle))))
        ids (vals interface)
        source-equation (first (soac/equations source))
        source-bound (conj (set (:parameters (soac/lambda-parts
                                             (:lambda (soac/operation-parts source-equation)))))
                           (get-in (soac/operation-parts source-equation) [:attributes :index]))]
    (when-not (and (= #{:index-snapshot :tangent-input :tangent-output :cotangent-input
                       :zero-gradient :gradient-output} (set (keys interface)))
                   (every? symbol? ids) (= (count ids) (count (distinct ids)))
                   (empty? (set/intersection (set ids)
                                            (set/union source-bound
                                                       (set (keys (:values (soac/facts source)))))))
                   (fingerprint/equivalent? (:primal projected) (:primal bundle))
                   (fingerprint/equivalent? (select-keys (:interface projected) [:params :tags :active-inputs])
                                            (select-keys bundle [:params :tags :active-inputs])))
      (fail! :ad-normalized-gather-interface {:interface interface}))
    (doseq [p (map gather-programs [:primal :linear :transpose])] (soac/validate! p))
    (when-not (fingerprint/equivalent? gather-programs (lift-gather-programs bundle source interface))
      (fail! :ad-normalized-gather-programs {:interface interface})))
  bundle)

(defn linearize-indexed-map-program
  "Construct checked indexed-map programs only; no host/device execution is exposed.
   A functional map reads one Float/Double data array through unchanged excluded Int/Long
   indices. Primal explicitly snapshots indices. Ordered transpose requires a fresh owned
   typed-zero physical gradient binding; the effect equation does not allocate or zero it."
  [source source-ns]
  (let [revision (revisions)
        output (gensym "indexed_element_output_")
        {:keys [primal interface]} (project-indexed-element source source-ns output)
        bundle (build-bundle primal interface revision)
        equation (first (soac/equations source))
        source-parts (soac/operation-parts equation)
        used (into (set (keys (:values (soac/facts source))))
                   (conj (:parameters (soac/lambda-parts (:lambda source-parts)))
                         (get-in source-parts [:attributes :index])))
        fresh (fn [key] (loop [id (gensym (str "ad_gather_" (name key) "_"))]
                          (if (contains? used id) (recur (gensym (str "ad_gather_" (name key) "_"))) id)))
        interface (into {} (map (fn [key] [key (fresh key)]))
                        [:index-snapshot :tangent-input :tangent-output :cotangent-input
                         :zero-gradient :gradient-output])
        bundle (assoc bundle :gather-programs (lift-gather-programs bundle source interface))]
    (validate! (assoc bundle :certificate (certificate bundle)))))

(defn- sum-source-map
  "Check the two-equation source before narrowing its known frontend bookkeeping.
   This is a relational projection, not permission to erase arbitrary host activity."
  [source source-ns]
  (let [{:keys [nodes]} (equation-dataflow source)
        equations (mapv :equation nodes)]
    (when-not (and (= 2 (count equations))
                   (instance? clojure.lang.Namespace source-ns))
      (fail! :ad-normalized-sum-boundary {:source source :source-ns source-ns}))
    (let [[map-equation reduction] equations
          facts (soac/facts source)
          map-parts (soac/operation-parts map-equation)
          {:keys [kind attributes arrays captures lambda]} (soac/operation-parts reduction)
          {:keys [parameters locals body-results]} (soac/lambda-parts lambda)
          [acc element] parameters
          [identity] (:identities attributes)
          [dt] (:dtypes attributes)
          [result] (nth reduction 2)
          values (:values facts)
          body (first body-results)
          scalar-values {result (get values result)
                         acc (av/tensor {:dtype dt :shape []})
                         element (av/tensor {:dtype dt :shape []})}]
    (when-not (and (= 2 (count equations)) (= 'map (:kind map-parts)) (= 'reduce kind)
                   (= [result] (soac/outputs source)) (= 1 (count (nth reduction 2)))
                   (empty? (:effects facts)) (empty? (:aliases facts))
                   (source-envelope? facts equations)
                   (or (nil? (get-in facts [:attributes :source-arithmetic]))
                       (= numerics/retained-source-arithmetic
                          (get-in facts [:attributes :source-arithmetic])))
                   (every? #(and (empty? (:effects %)) (empty? (:aliases %)))
                           (map #(get-in facts [:equations (second %)]) equations))
                   (= arrays (nth map-equation 2)) (= 1 (count arrays)) (empty? captures)
                   (fingerprint/equivalent? (:extent attributes) (get-in map-parts [:attributes :extent]))
                   (= 2 (count parameters)) (empty? locals) (= 1 (count body-results))
                   (every? #(or (nil? (types/sym-type-tag %))
                                (= dt (dtype/dtype-for-scalar-tag (types/sym-type-tag %)))) parameters)
                   (= [acc] (:accumulators attributes)) (= 1 (count (:identities attributes)))
                   (= 1 (count (:algebra attributes))) (contains? #{:float :double} dt)
                   (= dt (scalar-dtype! (get values result)) (:dtype (get values (first arrays))))
                   (fingerprint/equivalent?
                    (select-keys (get values result) [:memory-space :placement :sharding])
                    (select-keys (get values (first arrays)) [:memory-space :placement :sharding]))
                   (contains? #{nil :owned} (:ownership (get values result)))
                   (= {:kind :plain} (:representation (get values result)))
                   (nil? (:logical-layout (get values result)))
                   (empty? (:attributes (get values result)))
                   (empty? (soac/result-storage source (second reduction)))
                   (set/subset? (set (keys attributes))
                                #{:index :extent :attributes :accumulators :identities :dtypes
                                  :algebra :source-arithmetic})
                   (set/subset? (set (keys (:attributes attributes)))
                                #{:stable-array-captures :pointwise-storage-inputs})
                   (every? empty? (vals (:attributes attributes)))
                   (or (nil? (:source-arithmetic attributes))
                       (= numerics/retained-source-arithmetic (:source-arithmetic attributes))))
      (fail! :ad-normalized-sum-boundary {:source source}))
    (binding [*ns* source-ns]
      ;; The concrete step must add the exact accumulator and element, not merely carry an
      ;; additive certificate for some other expression. Selected overloads remain homogeneous.
      (let [{:keys [operation arguments canonical] :as call} (call! scalar-values body)
            typed-zero (descriptor/typed-reduce-identity operation dt)
            identity-value (:value (constant/value identity))
            identity-type
            (if (number? identity)
              (literal-dtype identity)
              (when (seq? identity)
                (let [{:keys [cast-tag arguments]} (call! values identity)]
                  ;; A source literal cast must still have actual nonlexical call authority.
                  ;; Only a single literal operand is needed for this typed-zero boundary.
                  (when (and (= 1 (count arguments)) (number? (first arguments)))
                    (some-> cast-tag dtype/dtype-for-scalar-tag)))))]
        (when-not (and (= '+ canonical) (descriptor/addition-op? operation)
                       ;; References identify the exact binders; use-site type metadata is
                       ;; separately checked, not mistaken for a different logical value ID.
                       (or (= arguments [acc element]) (= arguments [element acc]))
                       (every? #(or (nil? (types/sym-type-tag %))
                                    (= dt (dtype/dtype-for-scalar-tag (types/sym-type-tag %))))
                               arguments)
                       (= dt identity-type (literal-dtype identity-value))
                       (constant/zero-value? identity)
                       (fingerprint/equivalent? identity-value (:value (constant/value typed-zero))))
          (fail! :ad-normalized-sum-step {:body body :identity identity :dtype dt}))
        (binding-dtype! scalar-values result body)
        (selected-call-contract! call [dt dt] dt)
        (let [derived (scan/certify-reassociation {:acc acc :init identity :lambda body} dt)]
          (when-not (fingerprint/equivalent? derived (first (:algebra attributes)))
            (fail! :ad-normalized-sum-algebra {:derived derived :retained (:algebra attributes)})))))
    (let [view (:subprogram (first nodes))]
      (if (= :closed-clojure (get-in facts [:attributes :source-dialect]))
        ;; All original bindings, provenance and absent host operations were checked above.
        ;; The one-map view's binding list is the exact checked first source prefix.
        (let [view-facts (assoc-in (soac/facts view) [:attributes :source-bindings]
                                  (nth map-equation 2))]
          (soac/make view-facts [map-equation] (nth map-equation 2)))
        view)))))

(defn- append-ordered-sum [map-program reduction source result]
  (let [[boundary] (:arrays (soac/operation-parts reduction))
        source-value (get-in (soac/facts source) [:values boundary])
        producer-value (get-in (soac/facts map-program) [:values boundary])
        owned-value (owned-functional-result! source-value)
        view (soac/equation-subprogram source reduction)]
    ;; Map lifting already proves this exact source→generated owned result relation.
    ;; Do not generalize it into an ownership wildcard or right-biased value merge.
    (when-not (and (= boundary (first (soac/outputs map-program)))
                   (fingerprint/equivalent? owned-value producer-value))
      (fail! :ad-normalized-sum-boundary {:boundary boundary :source source-value :producer producer-value}))
    (let [facts (-> (soac/facts view)
                    (assoc-in [:values boundary] owned-value)
                    ;; Only the frontend bookkeeping already checked by sum-source-map is
                    ;; omitted; equation provenance, numerical operation attributes stay exact.
                    (assoc-in [:equations (second reduction) :attributes] {})
                    (assoc :attributes {:source-arithmetic numerics/retained-source-arithmetic}))]
      (soac/append-pure-spines map-program
                               (soac/make facts [reduction] (soac/outputs view))
                               {boundary boundary}
                               (into [result] (rest (soac/outputs map-program)))))))

(defn- sum-map-activity! [map-interface]
  (let [active-inputs (:active-inputs map-interface)
        tangent-inputs (:tangent-inputs map-interface)
        gradient-outputs (:gradient-outputs map-interface)]
    (when-not (and (vector? active-inputs) (seq active-inputs)
                   (vector? tangent-inputs) (vector? gradient-outputs)
                   (= (count active-inputs) (count tangent-inputs) (count gradient-outputs))
                   (= (count active-inputs) (count (distinct active-inputs)))
                   (= (count tangent-inputs) (count (distinct tangent-inputs)))
                   (= (count gradient-outputs) (count (distinct gradient-outputs))))
      (fail! :ad-normalized-sum-interface
             {:active-inputs active-inputs :tangent-inputs tangent-inputs
              :gradient-outputs gradient-outputs}))
    active-inputs))

(defn- sole-map-scalar!
  [bundle]
  (let [proof (get-in bundle [:map-programs :proof])
        nodes (:nodes proof)
        node (when (vector? nodes) (first nodes))
        scalar (:scalar-derivation node)]
    (when-not (and (= :normalized-scalar (:authority proof))
                   (vector? nodes)
                   (= 1 (count nodes))
                   (map? node)
                   (= #{:source-equations :execution-equation :scalar-derivation
                        :stage-interface :equation-ids}
                      (set (keys node)))
                   (map? scalar))
      (fail! :ad-normalized-sum-map-boundary {:nodes nodes}))
    scalar))

(defn- lift-sum-programs [bundle source interface]
  (let [scalar (sole-map-scalar! bundle)
        [_ reduction] (soac/equations source)
        map-programs (:map-programs bundle)
        map-interface (:interface map-programs)
        active-inputs (sum-map-activity! map-interface)
        map-result (first (soac/outputs (:source map-programs)))
        result (first (soac/outputs source))
        {:keys [seed tangent-result gradient-results]} interface
        tangent-source (soac/remap-values source
                                         {map-result (:tangent-output map-interface)
                                          result tangent-result})
        tangent-reduction (second (soac/equations tangent-source))
        transpose (:transpose map-programs)
        transpose-equation (first (soac/equations transpose))
        {:keys [attributes arrays captures lambda]} (soac/operation-parts transpose-equation)
        {:keys [parameters locals body-results]} (soac/lambda-parts lambda)
        ;; Move the explicit per-lane dy parameter to a scalar capture, leaving its scalar AD
        ;; identity unchanged. No broadcast array, reduction of cotangents or second evaluator.
        transpose-lambda (soac/lambda-form
                          (into (vec (rest parameters)) [(:cotangent-seed scalar)])
                          (mapv (fn [{:keys [id dtype init]}] (soac/local-value id dtype init)) locals)
                          body-results)
        transpose-equation (list '= (second transpose-equation) (nth transpose-equation 2)
                                 (list 'map attributes (vec (rest arrays))
                                       (into (vec captures) [seed]) transpose-lambda))
        transpose-facts (-> (soac/facts transpose)
                            (update :values dissoc (:cotangent-input map-interface))
                            (assoc-in [:values seed] (get-in (soac/facts source) [:values result]))
                            (update :inputs #(conj (vec (remove #{(:cotangent-input map-interface)} %)) seed)))]
    (let [transpose (soac/make transpose-facts [transpose-equation] (soac/outputs transpose))
          scalar-active? (= [] (get-in (soac/facts source) [:values (first active-inputs) :shape]))
          gradient-source (when scalar-active?
                            (soac/remap-values source {map-result (first (:gradient-outputs map-interface))
                                                      result (first gradient-results)}))]
      (when (and scalar-active? (not= 1 (count active-inputs)))
        (fail! :ad-normalized-sum-interface {:active-inputs active-inputs}))
      {:source source :interface interface
       :primal (append-ordered-sum (:primal map-programs) reduction source result)
       :linear (append-ordered-sum (:linear map-programs) tangent-reduction tangent-source tangent-result)
       :transpose (if scalar-active?
                    (append-ordered-sum transpose (second (soac/equations gradient-source))
                                        gradient-source (first gradient-results))
                    transpose)})))

(defn- validate-sum-programs! [{:keys [sum-programs source-ns] :as bundle}]
  (let [{:keys [source interface]} sum-programs
        projected (sum-source-map source source-ns)
        active-inputs (sum-map-activity! (get-in bundle [:map-programs :interface]))
        gradient-results (:gradient-results interface)
        fixed-ids ((juxt :seed :tangent-result) interface)
        ids (into fixed-ids gradient-results)
        map-value-ids (set (mapcat #(keys (:values (soac/facts %)))
                                  (vals (select-keys (:map-programs bundle) [:primal :linear :transpose]))))
        scalar-active? (= [] (get-in (soac/facts source) [:values (first active-inputs) :shape]))]
    (when-not (and (fingerprint/equivalent? projected (get-in bundle [:map-programs :source]))
                   (= #{:seed :tangent-result :gradient-results} (set (keys interface)))
                   (vector? gradient-results)
                   (= (count active-inputs) (count gradient-results))
                   (= (count gradient-results) (count (distinct gradient-results)))
                   (if scalar-active?
                     (and (= 1 (count active-inputs))
                          (symbol? (first gradient-results))
                          (not (contains? (into map-value-ids
                                                (concat fixed-ids (keys (:values (soac/facts source)))))
                                          (first gradient-results))))
                     (= gradient-results
                        (get-in bundle [:map-programs :interface :gradient-outputs])))
                   (every? symbol? ids) (= (count ids) (count (distinct ids)))
                   (empty? (set/intersection (set fixed-ids)
                                            (set (concat (keys (:values (soac/facts source)))
                                                         (mapcat #(keys (:values (soac/facts %)))
                                                                 (vals (select-keys (:map-programs bundle)
                                                                                    [:primal :linear :transpose]))))))))
      (fail! :ad-normalized-sum-interface {:interface interface}))
    (doseq [program (map sum-programs [:primal :linear :transpose])] (soac/validate! program))
    (when-not (fingerprint/equivalent? sum-programs (lift-sum-programs bundle source interface))
      (fail! :ad-normalized-sum-programs {:interface interface})))
  bundle)

(defn linearize-map-sum-program
  "Build checked map→ordered additive sum programs in the existing normalized bundle.
   One homogeneous Float/Double accumulator, exact typed zero, and the original selected
   addition step are required. Algebra evidence grants no reassociation permission.
   Transpose consumes an explicit scalar seed. This entry constructs programs, not execution."
  ([source source-ns] (linearize-map-sum-program source source-ns {}))
  ([source source-ns options]
   (let [map-source (sum-source-map source source-ns)
         bundle (build-map-programs map-source source-ns options)
         map-interface (get-in bundle [:map-programs :interface])
         active-inputs (sum-map-activity! map-interface)
         scalar-active? (= [] (get-in (soac/facts map-source)
                                      [:values (first active-inputs) :shape]))
         interface {:seed (gensym "ad_sum_seed_") :tangent-result (gensym "ad_sum_tangent_")
                    :gradient-results (if scalar-active? [(gensym "ad_sum_gradient_")]
                                          (:gradient-outputs map-interface))}
         bundle (assoc bundle :sum-programs (lift-sum-programs bundle source interface))]
     (validate! (assoc bundle :certificate (certificate bundle))))))

(defn prepare
  "Build one validated primal/linear/transpose bundle for a resolved scalar deftm Var.
   :wrt is a vector of distinct original parameter indices; excluded slots retain their primals."
  [resolved wrt]
  (let [revision (revisions)
        metadata (meta resolved)
        params (vec (:raster.core/deftm-params metadata))
        tags (:raster.core/deftm-tags metadata)
        dtypes (mapv dtype/dtype-for-scalar-tag tags)
        _ (when-not (and (seq params) (= (count params) (count tags))
                         (every? #{:float :double} dtypes) (apply = dtypes))
            (fail! :ad-normalized-unknown-type {:params params :tags tags}))
        selected (if (nil? wrt) (vec (range (count params))) wrt)
        _ (when-not (and (vector? selected) (= (count selected) (count (distinct selected)))
                         (every? #(and (integer? %) (<= 0 %) (< % (count params))) selected))
            (fail! :ad-invalid-wrt {:wrt wrt :params params}))
        active (mapv #(nth params %) (filter (set selected) (range (count params))))
        fresh (typed-gensym (first dtypes))
        source-ns (or (:ns metadata) *ns*)
        body (first (core/ensure-walked-body! resolved))
        qualified (inference/qualify-body-symbols body source-ns (set params))
        alpha (util/alpha-convert qualified)
        [source-bindings body-expressions]
        (if (and (seq? alpha) (contains? '#{let let*} (first alpha)))
          [(second alpha) (nnext alpha)] [[] [alpha]])
        [source-bindings output] (anf/normalize-for-ad source-bindings body-expressions
                                                       #(gensym (str % "__primal_")))
        ;; Always give the output an equation identity, including an input alias.
        primal-output (fresh "primal_output")
        source-bindings (into (vec source-bindings) [primal-output output])
        input-values (into (array-map) (map (fn [p dt] [p (av/tensor {:dtype dt :shape []})]) params dtypes))
        primal (program input-values source-bindings [primal-output])]
    (build-bundle primal {:params params :tags tags :active-inputs active :source-ns source-ns}
                  revision)))

(defn- emitted-body [bundle program result-forms params tags]
  {:walked-body [(list 'let* (into (bindings (:primal bundle)) (bindings program)) result-forms)]
   :params params :tags tags :source-ns (:source-ns bundle) :bundle bundle})

(defn jvp-body [bundle]
  (validate! bundle)
  (emitted-body bundle (:linear bundle)
                [(first (soac/outputs (:primal bundle))) (first (soac/outputs (:linear bundle)))]
                (into (:params bundle) (:tangent-inputs bundle))
                (into (vec (:tags bundle)) (map #(types/sym-type-tag %) (:tangent-inputs bundle)))))

(defn value+grad-body [bundle]
  (validate! bundle)
  (let [seed (:cotangent-seed bundle)
        dt (:dtype (get-in (soac/facts (:transpose bundle)) [:values seed]))
        gradients (zipmap (:active-inputs bundle) (soac/outputs (:transpose bundle)))
        body (emitted-body bundle (:transpose bundle)
                           (into [(first (soac/outputs (:primal bundle)))]
                                 (map gradients (:params bundle)))
                           (:params bundle) (:tags bundle))]
    (assoc body :walked-body
           [(list 'let* (into (bindings (:primal bundle))
                              (concat [seed (if (= :float dt) '(clojure.core/float 1.0) 1.0)]
                                      (bindings (:transpose bundle))))
                  (into [(first (soac/outputs (:primal bundle)))]
                        (map gradients (:params bundle))))])))

(defn ^:no-doc guarded-runtime
  "Internal lifecycle for an already derived scalar projection and its compiler adapter.
   Capture effective callees, independently revalidate their selected identities, and fence
   compilation/publication and every invocation. The adapter thunk creates the existing runtime
   function and must compile the supplied body. It is never user/serialized input and grants
   no source, operation or numerical capability. Original failures propagate.
   This is the existing sequential staleness contract, not concurrent mutation protection."
  [bundle parameters body compile-runtime]
  (let [source-ns (:source-ns bundle)
        guard! (execution-guard source-ns (select-keys bundle [:template-revision :compiler-revision])
                                [{:parameters parameters :body body}])
        _ (validate! bundle)
        _ (guard!)
        runtime-fn (binding [*ns* source-ns] (compile-runtime))
        _ (guard!)]
    (fn [& args]
      (guard!)
      (let [result (apply runtime-fn args)]
        (guard!)
        result))))
