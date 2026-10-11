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
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.ir.scan :as scan]
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

(declare validate-map-programs! validate-sum-programs! validate-gather-programs!)

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
    (when (contains? bundle :sum-programs)
      (validate-sum-programs! bundle))
    (when (contains? bundle :gather-programs)
      (validate-gather-programs! bundle))
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
    (when-not (and (map? options) (set/subset? (set (keys options)) #{:active-array}))
      (fail! :ad-normalized-map-active-array {:options options :arrays arrays}))
    (let [active-array (if (contains? options :active-array) (:active-array options)
                         (when (= 1 (count arrays)) (first arrays)))
          _ (when-not (some #(fingerprint/equivalent? active-array %) arrays)
              (fail! :ad-normalized-map-active-array {:active-array active-array :arrays arrays}))
          layout (soac/parameter-layout equation)
          active-element (get (zipmap arrays (:elements layout)) active-array)
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
                       :active-inputs [(tagged active-element dt)] :source-ns source-ns}})))))

(defn- project-map-equation-lane
  "Checked exact-member lane entry. Batch composition should project once with
   equation-subprograms before calling the shared private lane body."
  [source equation source-ns output-id options]
  (project-checked-map-lane source (soac/equation-subprogram source equation)
                            equation source-ns output-id options))

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

   Returns the existing scalar bundle, not array execution. One of at most two arrays
   is active; the other array and scalar captures are excluded. Two arrays require
   {:active-array source-value-id}. Retained dtypes are not storage-default guesses."
  ([map-program source-ns] (linearize-map-element map-program source-ns {}))
  ([map-program source-ns options]
   (let [{:keys [primal interface]} (project-map-element map-program source-ns nil options)]
     (linearize-program primal interface))))

(defn- map-residual-roles [bundle source]
  (let [equation (first (soac/equations source))
        {:keys [captures]} (soac/operation-parts equation)
        capture-values (zipmap (:capture-parameters (soac/parameter-layout equation)) captures)
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
        (project-map-element source source-ns (first (soac/outputs (:primal bundle)))
                             {:active-array (:active-array interface)})
        scalar-interface (:interface projected)
        interface (:interface map-programs)
        {:keys [scalars arrays]} (map-residual-roles bundle source)
        ids (concat (vals (:residual-arrays interface))
                    (map interface [:tangent-input :cotangent-input :tangent-output :gradient-output]))]
    (when-not (and (fingerprint/equivalent? primal (:primal bundle))
                   (fingerprint/equivalent? (select-keys bundle [:params :tags :active-inputs])
                                            (select-keys scalar-interface [:params :tags :active-inputs]))
                   (= #{:active-array :residual-arrays :scalar-captures :tangent-input :cotangent-input
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
   Two-array maps require {:active-array source-value-id}; other inputs are excluded.
   This builds programs only. Allocation, execution and asynchronous lifetimes are downstream."
  ([source source-ns] (linearize-map-program source source-ns {}))
  ([source source-ns options]
   (let [bundle (linearize-map-element source source-ns options)
         equation (first (soac/equations source))
         active-array (get (zipmap (:elements (soac/parameter-layout equation))
                                  (:arrays (soac/operation-parts equation)))
                           (first (:active-inputs bundle)))
         {:keys [scalars arrays]} (map-residual-roles bundle source)
         interface {:active-array active-array
                    :residual-arrays (into {} (map (fn [id] [id (gensym "ad_residual_")])) arrays)
                    :scalar-captures scalars
                    :tangent-input (gensym "ad_tangent_") :cotangent-input (gensym "ad_cotangent_")
                    :tangent-output (gensym "ad_push_") :gradient-output (gensym "ad_pull_")}
         bundle (assoc bundle :map-programs (lift-map-programs bundle source interface))]
     (validate! (assoc bundle :certificate (certificate bundle))))))

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
                   (let [equation (list '= id outputs
                                        (list 'map (assoc-in attributes [:attributes :stable-array-captures] captures)
                                              arrays captures region))
                         inputs (vec (distinct (concat (soac/operation-inputs equation)
                                                       (filter soac/value-id? (soac/operation-extents equation)))))]
                     (soac/make (soac/default-program-facts
                                 {:values (select-keys values (concat inputs outputs capacity-facts)) :inputs inputs
                                  :equations {id (soac/default-equation-facts)}
                                  :attributes {:source-arithmetic numerics/retained-source-arithmetic}})
                                [equation] outputs)))
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
  (soac/validate! source)
  (when-not (and (= 2 (count (soac/equations source)))
                 (instance? clojure.lang.Namespace source-ns))
    (fail! :ad-normalized-sum-boundary {:source source :source-ns source-ns}))
  (let [[map-equation reduction :as equations] (soac/equations source)
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
    (let [view (soac/equation-subprogram source map-equation)]
      (if (= :closed-clojure (get-in facts [:attributes :source-dialect]))
        ;; All original bindings, provenance and absent host operations were checked above.
        ;; The one-map view's binding list is the exact checked first source prefix.
        (let [view-facts (assoc-in (soac/facts view) [:attributes :source-bindings]
                                  (nth map-equation 2))]
          (soac/make view-facts [map-equation] (nth map-equation 2)))
        view))))

(defn- compose-sum [map-program reduction source result]
  (let [map-equation (first (soac/equations map-program))
        equations [map-equation reduction]
        inputs (:inputs (soac/facts map-program))
        outputs (into [result] (rest (soac/outputs map-program)))
        values (merge (:values (soac/facts source)) (:values (soac/facts map-program)))]
    (soac/make
     (soac/default-program-facts
      {:values (select-keys values (concat inputs (mapcat #(nth % 2) equations)))
       :inputs inputs
       :equations {(second map-equation) (get-in (soac/facts map-program) [:equations (second map-equation)])
                   ;; The original checked envelope remains in :sum-programs/:source. This
                   ;; generated program has no scalar prefix to stage: omit only the exact
                   ;; frontend bookkeeping facet already checked by sum-source-map.
                   (second reduction) (assoc (get-in (soac/facts source) [:equations (second reduction)])
                                             :attributes {})}
       :attributes {:source-arithmetic numerics/retained-source-arithmetic}})
     equations outputs)))

(defn- lift-sum-programs [bundle source interface]
  (let [[_ reduction] (soac/equations source)
        map-programs (:map-programs bundle)
        map-interface (:interface map-programs)
        map-result (first (soac/outputs (:source map-programs)))
        result (first (soac/outputs source))
        {:keys [seed tangent-result]} interface
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
                          (into (vec (rest parameters)) [(:cotangent-seed bundle)])
                          (mapv (fn [{:keys [id dtype init]}] (soac/local-value id dtype init)) locals)
                          body-results)
        transpose-equation (list '= (second transpose-equation) (nth transpose-equation 2)
                                 (list 'map attributes (vec (rest arrays))
                                       (into (vec captures) [seed]) transpose-lambda))
        transpose-facts (-> (soac/facts transpose)
                            (update :values dissoc (:cotangent-input map-interface))
                            (assoc-in [:values seed] (get-in (soac/facts source) [:values result]))
                            (update :inputs #(conj (vec (remove #{(:cotangent-input map-interface)} %)) seed)))]
    {:source source :interface interface
     :primal (compose-sum (:primal map-programs) reduction source result)
     :linear (compose-sum (:linear map-programs) tangent-reduction tangent-source tangent-result)
     :transpose (soac/make transpose-facts [transpose-equation] (soac/outputs transpose))}))

(defn- validate-sum-programs! [{:keys [sum-programs source-ns] :as bundle}]
  (let [{:keys [source interface]} sum-programs
        projected (sum-source-map source source-ns)
        ids (map interface [:seed :tangent-result])]
    (when-not (and (fingerprint/equivalent? projected (get-in bundle [:map-programs :source]))
                   (= #{:seed :tangent-result} (set (keys interface)))
                   (every? symbol? ids) (= (count ids) (count (distinct ids)))
                   (empty? (set/intersection (set ids)
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
         bundle (linearize-map-program map-source source-ns options)
         interface {:seed (gensym "ad_sum_seed_") :tangent-result (gensym "ad_sum_tangent_")}
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
