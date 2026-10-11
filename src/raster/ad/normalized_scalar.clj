(ns raster.ad.normalized-scalar
  "Opt-in scalar AD over checked, unscheduled TypedSOAC equations.

   No derivative registry, runtime evaluator or target schedule lives here. Existing templates
   emit the linear rules; the ordinary compiler consumes the resulting retained let forms.
   The closed first vertical deliberately rejects effects, scopes and active precision changes."
  (:require [clojure.set :as set]
            [raster.ad.templates :as templates]
            [raster.ad.reverse.normalize :as anf]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.core :as core]))

(defn- fail! [reason data]
  (throw (ex-info "program is outside the checked normalized scalar AD boundary"
                  (assoc data :reason reason :representation :normalized-scalar))))

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
        cast-tag (descriptor/cast-result-tag operation)]
    (when (or lexical-callee (nil? operation))
      (fail! :ad-normalized-callee {:expression expression :call call}))
    (when-not (or (and (contains? '#{+ - * /} canonical)
                       (contains? #{1 2} (count arguments))
                       (or (= 2 (count arguments)) (= '- canonical)))
                  (and (contains? '#{float double} cast-tag) (= 1 (count arguments))))
      (fail! :ad-normalized-operation {:expression expression :operation operation}))
    (assoc call :template template :template-identity template-identity
           :canonical canonical :cast-tag cast-tag)))

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
  (let [{:keys [operation dispatch? implementation-op]} call
        operation-var (when (symbol? operation) (resolve operation))]
    (when (or dispatch? (:raster.core/dispatch-table (meta operation-var)))
      (let [argument-tags (mapv tag argument-dtypes)
            selected (dispatch/selected-call-signature
                      operation argument-tags
                      (inference/registered-call-selection operation argument-tags))
            implementation-var (when (and dispatch? (symbol? implementation-op))
                                 (resolve implementation-op))]
        ;; This scalar vertical admits exact registered homogeneous methods only.
        ;; Promotion or a merely applicable parametric template is not a proof.
        (when-not (and (= :registered (:selection selected))
                       (= argument-tags (:tags selected))
                       (= (tag result-dtype) (:return-tag selected))
                       (not (some some? (:promotion-casts selected)))
                       (or (not dispatch?)
                           (and (var? implementation-var) (bound? implementation-var)
                                (some? (:typed-impl selected))
                                (identical? @implementation-var (:typed-impl selected)))))
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
                             (when (and cast-tag linear?
                                        (not= (expression-dtype values (first arguments))
                                              (:dtype (get values result))))
                               (fail! :ad-normalized-active-conversion {:expression expression}))
                             (when (and (contains? '#{+ -} canonical) linear?
                                        (some #{:coefficient} kinds))
                               (fail! :ad-normalized-affine-offset {:expression expression}))
                             (cond linear? :linear
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

(defn- typed-gensym [dt]
  (fn fresh
    ([prefix] (tagged (gensym (str prefix "__nsad_")) dt))
    ([prefix requested-tag]
     (when (and requested-tag (not= requested-tag (tag dt)))
       (fail! :ad-normalized-rule-type {:requested requested-tag :dtype dt}))
     (fresh prefix))))

(defn- sum-contributions [contributions fresh]
  (reduce (fn [[bs current] next]
            (let [result (fresh "sum")]
              [(into bs [result (list 'raster.numeric/+ current next)]) result]))
          [[] (first contributions)] (rest contributions)))

(defn- linearize [primal active-inputs fresh]
  (let [values (:values (soac/facts primal))
        tangent-inputs (mapv (fn [id] (tagged (gensym (str (name id) "__tangent_"))
                                              (:dtype (get values id)))) active-inputs)
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
        seed (tagged (gensym "cotangent__") (:dtype (get values output)))
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

(defn- revisions []
  {:template-revision (templates/registry-revision)
   :compiler-revision (dispatch/compiler-definition-revision)})

(defn- certificate [bundle]
  (fingerprint/fingerprint
   (assoc (dissoc bundle :certificate :source-ns)
          :source-namespace (ns-name (:source-ns bundle)))))

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

(declare validate-map-programs!)

(defn validate!
  "Validate the bundle's typed interfaces and retained producer certificate before emission.
   This detects stale or changed producer data; it is not a proof of custom derivative math."
  [{:keys [primal linear transpose params tags active-inputs tangent-inputs residuals
           cotangent-seed source-ns] :as bundle}]
  (when-not (instance? clojure.lang.Namespace source-ns)
    (fail! :ad-normalized-bundle-interface {:source-ns source-ns}))
  (binding [*ns* source-ns]
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
    (when (contains? bundle :map-programs)
      (validate-map-programs! bundle))
    (when-not (= (:certificate bundle) (certificate bundle))
      (fail! :ad-normalized-certificate {:certificate (:certificate bundle)}))
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
    (let [fresh (typed-gensym (:dtype (get-in (soac/facts primal) [:values (first params)])))
          {:keys [program tangent-inputs coefficients]} (linearize primal active-inputs fresh)
          transposed (transpose program tangent-inputs coefficients fresh)
          _ (when-not (= revision (revisions))
              (fail! :ad-normalized-stale-rules {:before revision :after (revisions)}))
          bundle (merge revision {:primal primal :linear program :transpose (:program transposed)
                                  :params params :tags tags :active-inputs active-inputs
                                  :tangent-inputs tangent-inputs :residuals coefficients
                                  :cotangent-seed (:seed transposed) :source-ns source-ns})]
      (validate! (assoc bundle :certificate (certificate bundle))))))

(defn linearize-program
  "Build checked scalar primal/linear/transpose programs from an existing TypedSOAC program.
   The interface explicitly supplies :params, retained :tags, :active-inputs and :source-ns.
   This compiler entry never walks source, resolves a deftm Var or re-infers its types."
  [primal interface]
  (build-bundle primal interface (revisions)))

(defn- map-source-envelope?
  [facts equation]
  (let [attributes (dissoc (:attributes facts) :source-arithmetic)
        equation-facts (get-in facts [:equations (second equation)])]
    (or (and (empty? attributes) (empty? (:attributes equation-facts)))
        (and (= {:source-dialect :closed-clojure
                 :source-bindings (nth equation 2)
                 :host-binding-ids [] :host-read-values [] :host-read-sites [] :allocations []}
                attributes)
             (= {:source-prefix? true} (:attributes equation-facts))
             (= :analyzed-source (get-in equation-facts [:provenance :front-end]))
             (= 0 (get-in equation-facts [:provenance :source-binding-id]))))))

(defn- project-map-element
  [map-program source-ns output-id]
  (soac/validate! map-program)
  (when-not (= 1 (count (soac/equations map-program)))
    (fail! :ad-normalized-map-boundary {:equations (count (soac/equations map-program))}))
  (let [equations (soac/equations map-program)
        equation (first equations)
        {:keys [kind attributes arrays captures lambda]} (soac/operation-parts equation)
        results (nth equation 2)
        facts (soac/facts map-program)
        equation-facts (get-in facts [:equations (second equation)])
        values (:values facts)
        extent (:extent attributes)
        extent-value (get values extent)]
    (when-not (and (= 'map kind)
                   (= 1 (count arrays)) (= 1 (count results))
                   (= results (soac/outputs map-program))
                   (empty? (:effects facts)) (empty? (:aliases facts))
                   (empty? (:effects equation-facts)) (empty? (:aliases equation-facts))
                   (map-source-envelope? facts equation)
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
    (let [array-value (get values (first arrays))
          result-value (get values (first results))
          dt (:dtype array-value)
          operand-values (mapv values (concat arrays captures results))
          {:keys [parameters locals body-results]} (soac/lambda-parts lambda)]
      (when-not (and (contains? #{:float :double} dt)
                     (= (:shape array-value) (:shape result-value))
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
          {:primal scalar
           :interface {:params params :tags (vec (repeat (count params) (tag dt)))
                       :active-inputs [(first params)] :source-ns source-ns}})))))

(defn linearize-map-element
  "Linearize the typed scalar region of one pure, homogeneous functional map.

   Returns the existing scalar bundle, not array execution. The element is active;
   scalar captures are inactive. Retained local dtypes are not storage-default guesses."
  [map-program source-ns]
  (let [{:keys [primal interface]} (project-map-element map-program source-ns nil)]
    (linearize-program primal interface)))

(defn- map-residual-roles [bundle source]
  (let [{:keys [captures lambda]} (soac/operation-parts (first (soac/equations source)))
        capture-values (zipmap (rest (:parameters (soac/lambda-parts lambda))) captures)
        scalars (select-keys capture-values (:residuals bundle))]
    ;; Only direct scalar captures have an existing outer scalar value. Snapshot other
    ;; coefficients per lane, including invariant locals, without another activity analysis.
    {:scalars scalars :arrays (vec (remove #(contains? scalars %) (:residuals bundle)))}))

(defn- scalar-program-lambda [program parameters outputs]
  (soac/lambda-form
   parameters
   (mapv (fn [[id expression]]
           (soac/local-value id (:dtype (get-in (soac/facts program) [:values id])) expression))
         (partition 2 (bindings program)))
   outputs))

(defn- lift-map-programs [bundle source interface]
  (let [equation (first (soac/equations source))
        {:keys [attributes arrays captures]} (soac/operation-parts equation)
        {:keys [residual-arrays scalar-captures tangent-input cotangent-input
                tangent-output gradient-output]} interface
        outputs (soac/outputs source)
        input-value (get-in (soac/facts source) [:values (first arrays)])
        output-value (get-in (soac/facts source) [:values (first outputs)])
        _ (when-not (and (contains? #{nil :owned} (:ownership output-value))
                         (fingerprint/equivalent?
                          (select-keys input-value [:memory-space :placement :sharding])
                          (select-keys output-value [:memory-space :placement :sharding])))
            (fail! :ad-normalized-map-storage {:input input-value :output output-value}))
        coefficients (vec (filter #(contains? residual-arrays %) (:residuals bundle)))
        scalar-coefficients (vec (filter #(contains? scalar-captures %) (:residuals bundle)))
        tensor (assoc output-value :ownership :owned)
        residual-outputs (mapv residual-arrays coefficients)
        values (merge (:values (soac/facts source))
                      (zipmap (concat outputs residual-outputs [tangent-output gradient-output])
                              (repeat tensor))
                      {tangent-input (assoc tensor :ownership :borrowed)
                       cotangent-input (assoc tensor :ownership :borrowed)})
        make-map
        (fn [id arrays captures lambda outputs]
          (let [extent (:extent attributes)
                inputs (vec (distinct (concat arrays captures
                                              (when-not (integer? extent) [extent]))))
                equation (list '= id outputs (list 'map attributes arrays captures lambda))]
            (soac/make
             (soac/default-program-facts
              {:values (select-keys values (concat inputs outputs)) :inputs inputs
               :equations {id (soac/default-equation-facts)}
               :attributes {:source-arithmetic numerics/retained-source-arithmetic}})
             [equation] outputs)))
        primal-results (into outputs residual-outputs)
        primal-yields (into (soac/outputs (:primal bundle)) coefficients)
        derivative-captures (mapv scalar-captures scalar-coefficients)
        derivative-parameters (fn [seed] (into [seed] (concat coefficients scalar-coefficients)))]
    {:source source :interface interface
     :primal (make-map 'ad-map-primal arrays captures
                       (scalar-program-lambda (:primal bundle) (:params bundle) primal-yields)
                       primal-results)
     :linear (make-map 'ad-map-linear (into [tangent-input] residual-outputs) derivative-captures
                       (scalar-program-lambda (:linear bundle)
                                              (derivative-parameters (first (:tangent-inputs bundle)))
                                              (soac/outputs (:linear bundle)))
                       [tangent-output])
     :transpose (make-map 'ad-map-transpose (into [cotangent-input] residual-outputs)
                          derivative-captures
                          (scalar-program-lambda (:transpose bundle)
                                                 (derivative-parameters (:cotangent-seed bundle))
                                                 (soac/outputs (:transpose bundle)))
                          [gradient-output])}))

(defn- validate-map-programs! [{:keys [map-programs source-ns] :as bundle}]
  (let [{:keys [source interface]} map-programs
        {:keys [primal] :as projected}
        (project-map-element source source-ns (first (soac/outputs (:primal bundle))))
        scalar-interface (:interface projected)
        interface (:interface map-programs)
        {:keys [scalars arrays]} (map-residual-roles bundle source)
        ids (concat (vals (:residual-arrays interface))
                    (map interface [:tangent-input :cotangent-input :tangent-output :gradient-output]))]
    (when-not (and (fingerprint/equivalent? primal (:primal bundle))
                   (fingerprint/equivalent? (select-keys bundle [:params :tags :active-inputs])
                                            (select-keys scalar-interface [:params :tags :active-inputs]))
                   (= #{:residual-arrays :scalar-captures :tangent-input :cotangent-input
                        :tangent-output :gradient-output} (set (keys interface)))
                   (fingerprint/equivalent? (set arrays) (set (keys (:residual-arrays interface))))
                   (fingerprint/equivalent? scalars (:scalar-captures interface))
                   (every? symbol? ids) (= (count ids) (count (distinct ids)))
                   (empty? (set/intersection (set ids) (set (keys (:values (soac/facts source)))))))
      (fail! :ad-normalized-map-interface {:interface interface :projected projected}))
    (doseq [program (map map-programs [:primal :linear :transpose])] (soac/validate! program))
    ;; Fixed interface IDs make reconstruction deterministic. Exact semantic equality includes
    ;; expression metadata and float bits; no fresh AD generation or printed-string comparison.
    (when-not (fingerprint/equivalent? map-programs (lift-map-programs bundle source interface))
      (fail! :ad-normalized-map-programs {:interface interface})))
  bundle)

(defn linearize-map-program
  "Lift one checked map's scalar bundle into ordinary primal/linear/transpose SOAC maps.
   Primal exports fresh owned coefficient arrays; derivatives consume explicit dx/dy arrays.
   This builds programs only. Allocation, execution and asynchronous lifetimes are downstream."
  [source source-ns]
  (let [bundle (linearize-map-element source source-ns)
        {:keys [scalars arrays]} (map-residual-roles bundle source)
        interface {:residual-arrays (into {} (map (fn [id] [id (gensym "ad_residual_")])) arrays)
                   :scalar-captures scalars
                   :tangent-input (gensym "ad_tangent_") :cotangent-input (gensym "ad_cotangent_")
                   :tangent-output (gensym "ad_push_") :gradient-output (gensym "ad_pull_")}
        bundle (assoc bundle :map-programs (lift-map-programs bundle source interface))]
    (validate! (assoc bundle :certificate (certificate bundle)))))

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
