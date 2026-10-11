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

(defn- call! [expression]
  (let [{:keys [operation lexical-callee arguments] :as call}
        (descriptor/call-description expression)
        [template template-identity] (templates/resolve-template operation)
        canonical (cond (or (descriptor/addition-op? operation)
                            ;; Existing polymorphic cotangent addition, restricted here to
                            ;; checked non-nil scalar operands; its nil case cannot occur.
                            (= 'raster.ad.reverse/grad-acc template-identity)) '+
                        (descriptor/subtraction-op? operation) '-
                        (descriptor/multiplication-op? operation) '*)
        cast-tag (descriptor/cast-result-tag operation)]
    (when (or lexical-callee (nil? operation))
      (fail! :ad-normalized-callee {:expression expression :call call}))
    (when-not (or (and (contains? '#{+ - *} canonical)
                       (contains? #{1 2} (count arguments))
                       (or (= 2 (count arguments)) (= '- canonical)))
                  (and (contains? '#{float double} cast-tag) (= 1 (count arguments))))
      (fail! :ad-normalized-operation {:expression expression :operation operation}))
    (assoc call :template template :template-identity template-identity
           :canonical canonical :cast-tag cast-tag)))

(defn- equation-expression [equation]
  (let [{:keys [captures lambda]} (soac/operation-parts equation)
        {:keys [parameters locals body-results]} (soac/lambda-parts lambda)]
    (when (or (seq locals) (not= 1 (count body-results)))
      (fail! :ad-normalized-region {:equation equation}))
    (util/subst-syms (zipmap parameters captures) (first body-results))))

(defn- binding-dtype! [values binding expression]
  (let [declared (or (some-> (get values binding) scalar-dtype!)
                     (some-> (types/sym-type-tag binding) dtype/dtype-for-scalar-tag))
        retained (some-> (types/sym-type-tag expression) dtype/dtype-for-scalar-tag)
        actual (cond
                 (or (symbol? expression) (number? expression))
                 (expression-dtype values expression)

                 (seq? expression)
                 (let [{:keys [arguments cast-tag]} (call! expression)
                       arg-types (mapv #(expression-dtype values %) arguments)]
                   (when (some nil? arg-types)
                     (fail! :ad-normalized-unknown-type {:binding binding :expression expression}))
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
                  (when (seq? expression) (call! expression))
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
                    (let [{:keys [canonical cast-tag arguments]} (call! expression)
                          kinds (mapv #(if (number? %) (if (zero? %) :zero :coefficient)
                                           (get classes % :coefficient)) arguments)
                          linear? (some #{:linear} kinds)]
                      (when (and (= '* canonical) (every? #{:linear} kinds))
                        (fail! :ad-normalized-nonlinear {:result result :expression expression}))
                      (when (and cast-tag linear?
                                 (not= (expression-dtype values (first arguments))
                                       (:dtype (get values result))))
                        (fail! :ad-normalized-active-conversion {:expression expression}))
                      (when (and (contains? '#{+ -} canonical) linear?
                                 (some #{:coefficient} kinds))
                        (fail! :ad-normalized-affine-offset {:expression expression}))
                      (cond linear? :linear
                            (and (not= '* canonical) (every? #{:zero} kinds)) :zero
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
             (let [{:keys [template operation cast-tag arguments]} (call! expression)
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
                       (let [{:keys [template arguments cast-tag]} (call! expression)]
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

(defn validate!
  "Validate the bundle's typed interfaces and retained producer certificate before emission.
   This detects stale or changed producer data; it is not a proof of custom derivative math."
  [{:keys [primal linear transpose params tags active-inputs tangent-inputs residuals
           cotangent-seed] :as bundle}]
  (when-not (= (revisions) (select-keys bundle [:template-revision :compiler-revision]))
    (fail! :ad-normalized-stale-rules {:expected (select-keys bundle [:template-revision :compiler-revision])
                                     :actual (revisions)}))
  (doseq [p [primal linear transpose]] (soac/validate! p))
  (doseq [[id value] (:values (soac/facts primal))] (scalar-dtype! value))
  (doseq [[id expression] (partition 2 (bindings primal))]
    (binding-dtype! (:values (soac/facts primal)) id expression))
  (when-not (and (set/subset? (set (:inputs (soac/facts primal))) (set params))
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
  (when-not (= (:certificate bundle) (certificate bundle))
    (fail! :ad-normalized-certificate {:certificate (:certificate bundle)}))
  bundle)

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
        primal (program input-values source-bindings [primal-output])
        {:keys [program tangent-inputs coefficients]} (linearize primal active fresh)
        transposed (transpose program tangent-inputs coefficients fresh)
        _ (when-not (= revision (revisions))
            (fail! :ad-normalized-stale-rules {:before revision :after (revisions)}))
        bundle (merge revision {:primal primal :linear program :transpose (:program transposed)
     :params params :tags tags :active-inputs active :tangent-inputs tangent-inputs
     :residuals coefficients :cotangent-seed (:seed transposed) :source-ns source-ns})]
    (validate! (assoc bundle :certificate (certificate bundle)))))

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
