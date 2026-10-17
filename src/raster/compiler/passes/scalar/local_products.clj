(ns raster.compiler.passes.scalar.local-products
  "Closed lexical presence refinement and nonescaping local vector replacement.
   No calls are evaluated or arithmetic reassociated. An approved optional
   scalar sum projects to addition only with independent present, homogeneous
   producer carriers and the existing checked scalar lowerer. Owners retaining
   definition-freshness guards must capture dependencies before projection. Fields
   are snapshotted in source order; escaping products retain their source form."
  (:require [raster.compiler.ir.form :as form]
            [raster.ad.accumulation :as accumulation]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.value-product :as value-product]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.inference :as inference]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-body]
            [raster.compiler.passes.scalar.normal-return :as normal-return]
            [raster.compiler.core.util :as util]))

(defn- literal? [x]
  (or (nil? x) (boolean? x) (number? x) (string? x) (keyword? x) (char? x)))

(defn- relist [original children]
  (with-meta (apply list children) (meta original)))

(defn- fact [env expression]
  (cond (literal? expression) (cond-> {:literal expression
                                     :presence (if (nil? expression) :absent :present)}
                               (number? expression)
                               (assoc :scalar-type (some-> (inference/infer-arg-tag expression {})
                                                          dtype/dtype-for-scalar-tag)))
        (symbol? expression) (get env expression)
        :else nil))

(defn- product-shape? [expression]
  (or (vector? expression)
      (and (seq? expression) (= 'let* (first expression))
           (= 3 (count expression)) (product-shape? (last expression)))))

(defn- split-product [expression]
  (if-let [fields (value-product/literal-fields expression)]
    [[] fields]
    (let [[bindings fields] (split-product (last expression))]
      [(into (vec (second expression)) bindings) fields])))

(defn- primitive-cast? [expression]
  (when (and (seq? expression) (= 2 (count expression)))
    (let [operation (descriptor/semantic-op expression)]
      (and (symbol? operation) (= "clojure.core" (namespace operation))
           (descriptor/cast-result-tag operation)))))

(defn- literal-tail [expression]
  (cond (literal? expression) {:bindings [] :literal expression}
        (and (seq? expression) (= 'let* (first expression)) (= 3 (count expression)))
        (when-let [tail (literal-tail (last expression))]
          (update tail :bindings #(into (vec (second expression)) %)))
        :else nil))

(declare lower)

(defn- nil-test-branches [env test]
  ;; A canonical direct nil test refines only the value it actually reads.
  ;; Do not infer presence from dtype tags or carry predicate facts across
  ;; later rebinding of a lexical name.
  (if (and (seq? test) (= 2 (count test)) (symbol? (second test))
           (= 'clojure.core/nil? (descriptor/semantic-op test)))
    (let [value (second test) known (get env value)]
      [(assoc env value (-> known (dissoc :fields :present? :scalar-type :array-tag)
                           (assoc :literal nil :presence :absent)))
       (assoc env value (-> known (dissoc :literal) (assoc :present? true :presence :present)))])
    [env env]))

(def ^:private conditional-expansion-limit 4096)
(def ^:dynamic ^:private *conditional-expansion-budget* nil)

(defn- product-arity [expression]
  (cond
    (vector? expression) (count expression)
    (and (seq? expression) (= 'let* (first expression)) (= 3 (count expression)))
    (product-arity (last expression))
    (and (seq? expression) (= 'if (first expression)) (= 4 (count expression)))
    (let [left (product-arity (nth expression 2))
          right (product-arity (nth expression 3))]
      (when (and (some? left) (= left right)) left))
    :else nil))

(defn- conditional-product? [expression]
  (and (seq? expression) (= 'if (first expression))
       (= 4 (count expression)) (some? (product-arity expression))))

(defn- spend-conditional-expansion! [continuation]
  ;; Account for the duplicated continuation before constructing it. The
  ;; bounded walk also avoids traversing an arbitrarily large rejected body.
  (let [remaining @*conditional-expansion-budget*
        cost (count (take (inc remaining) (tree-seq coll? seq continuation)))]
    (when (> cost remaining)
      (throw (ex-info "Conditional product replacement exceeds its expansion budget"
                      {:reason :local-product-budget})))
    (swap! *conditional-expansion-budget* - cost)))

(defn- binding-fact [scope id init]
  (let [known (or (fact scope init)
                  ;; A real primitive conversion cannot return nil. This is a
                  ;; normal-return fact, not permission to remove its source
                  ;; evaluation or the exception raised for a nullable input.
                  (when (primitive-cast? init)
                    (let [type (some-> (descriptor/semantic-op init)
                                       descriptor/cast-result-tag dtype/dtype-for-scalar-tag)]
                      (cond-> {:present? true} type (assoc :scalar-type type))))
                  (when-let [proof (and (seq? init)
                                       (descriptor/scalar-op? (descriptor/semantic-op init))
                                       (scalar-body/checked-primitive-result
                                        init (into {} (keep (fn [argument]
                                                             (when-let [type (:producer-scalar-type (get scope argument))]
                                                               [argument type])))
                                                   (descriptor/call-args init))))]
                    {:present? true :scalar-type (:type proof)}))
        known (or known
                  (when-let [proof (scalar-body/checked-selected-primitive-result init)]
                    {:present? true :scalar-type (:type proof)})
                  (let [result (normal-return/result-fact init scope)]
                    (when (= :present (:presence result))
                      (assoc result :present? true))))
        retained (some-> (:raster.type/tag (meta id)) dtype/dtype-for-scalar-tag)
        ;; Retained consumer declarations are not physical producer evidence.
        ;; Keep that distinction through aliases for checked sum projection.
        producer-type (or (:producer-scalar-type known) (:scalar-type known))]
    (cond-> known
      (:present? known) (assoc :presence :present)
      producer-type (assoc :producer-scalar-type producer-type)
      (and (:scalar-type known) retained) (assoc :scalar-type retained))))

(defn- project-scalar-accumulation [env expression]
  (when (accumulation/approved-call? expression)
    (let [arguments (:arguments (descriptor/call-description expression))
          facts (mapv (fn [argument]
                        (let [known (binding-fact env (gensym "sum_operand__") argument)
                              dt (:producer-scalar-type known)]
                          (when (and dt (= dt (:scalar-type known))
                                     (or (:present? known)
                                         (and (contains? known :literal) (some? (:literal known)))))
                            {:present? true :dtype dt}))) arguments)]
      (accumulation/project-checked-addition
       expression facts
       (fn [addition dt]
         ;; Validate the selected operation using independently established
         ;; producer widths. Proxies are proof inputs, not emitted temporaries:
         ;; original operands and their evaluation order remain unchanged.
         (let [ports [(gensym "sum_left__") (gensym "sum_right__")]
               probe (with-meta (apply list (first addition) ports) (meta addition))
               proof (scalar-body/checked-primitive-result probe (zipmap ports (repeat dt)))]
           (= dt (:type proof))))))))

(defn- project-array-accumulation [env expression]
  (when (accumulation/approved-call? expression)
    (let [facts (mapv #(normal-return/result-fact % env)
                      (:arguments (descriptor/call-description expression)))]
      (accumulation/project-checked-array-addition
       expression facts
       (fn [addition tag]
         (= {:presence :present :array-tag tag}
            (normal-return/result-fact addition env)))))))

(defn- lower-let [env expression products?]
  (let [[head bindings & body] expression]
    (when-not (and (vector? bindings) (even? (count bindings))
                   (every? symbol? (take-nth 2 bindings)))
      (throw (ex-info "Local product replacement requires lexical symbol bindings"
                      {:reason :local-product-decline})))
    (loop [remaining (seq (partition 2 bindings)) scope env output []]
      (if-let [[id original-init] (first remaining)]
        (let [init (if (and products? (symbol? original-init) (:fields (get scope original-init)))
                     original-init (lower scope original-init products?))
              alias (when (symbol? init) (get scope init))]
          (cond
                 ;; Replace a nonescaping product join by its lexical
                 ;; continuation in the selected arm. Predicate, arm effects
                 ;; and continuation each execute once, in their source order.
                 (and products? (conditional-product? init))
                 (let [[_ test then else] init
                       suffix (vec (mapcat identity (next remaining)))
                       continue (fn [arm]
                                  (util/alpha-convert
                                   (relist expression
                                           (concat [head (into [id arm] suffix)] body))))
                       _ (spend-conditional-expansion! (concat suffix body))
                       choice (lower scope (relist init ['if test (continue then) (continue else)])
                                     products?)]
                   (relist expression [head (with-meta (vec output) (meta bindings)) choice]))

                 (and products? (:fields alias))
                 (recur (next remaining) (assoc scope id alias) output)

                 (and products? (product-shape? init))
                 (let [[prefix fields] (split-product (util/alpha-convert init))
                       snapshots (mapv (fn [value]
                                         [(with-meta (gensym "product_field__")
                                            (select-keys (meta value) [:raster.type/tag])) value])
                                       fields)
                       lifted (vec (concat prefix (mapcat identity snapshots)))
                       scope (reduce (fn [local [id init]]
                                       (assoc local id (binding-fact local id init)))
                                     scope (partition 2 lifted))]
                   (recur (next remaining) (assoc scope id {:fields (mapv first snapshots)})
                          (into output lifted)))

                 :else
                 (recur (next remaining) (assoc scope id (binding-fact scope id init))
                        (conj output id init))))
        (relist expression (concat [head (with-meta (vec output) (meta bindings))]
                                   (map #(lower scope % products?) body)))))))

(defn- lower-scopes [env expression products?]
  (let [{:keys [scopes outer rebuild sequential? rec?]} (form/scope-info expression)
        mask #(apply dissoc %1 %2)]
    (rebuild
     (mapv (fn [{:keys [binders inits body] :as scope}]
             (let [base (if rec? (mask env binders) env)
                   [local inits']
                   (reduce (fn [[local out] [id init]]
                             [(mask local [id])
                              (conj out (lower (if (or sequential? rec?) local env) init products?))])
                           [base []] (map vector binders inits))]
               (assoc scope :inits inits'
                            :body (mapv #(lower (mask local binders) % products?) body)))) scopes)
     (mapv #(lower env % products?) outer))))

(defn- lower [env expression products?]
  (cond
    (symbol? expression)
    (let [known (get env expression)]
      (cond (:fields known) (throw (ex-info "Local product escapes projection scope"
                                           {:reason :local-product-escape :value expression}))
            :else expression))

    (and (seq? expression) (= 'quote (first expression))) expression

    (and (seq? expression) (contains? #{'let 'let*} (first expression)))
    (lower-let env expression products?)

    ;; Constant-index projection consumes a saved product without reconstructing
    ;; its allocation. Out-of-bounds/defaulted/dynamic accesses remain unchanged.
    (and (seq? expression) (symbol? (second expression))
         (get-in env [(second expression) :fields])
         (value-product/constant-projection
          expression (count (get-in env [(second expression) :fields]))))
    (let [fields (get-in env [(second expression) :fields])
          projection (value-product/constant-projection expression (count fields))]
      (nth fields (:index projection)))

    (and (seq? expression) (= 'clojure.core/nil? (descriptor/semantic-op expression)) (= 2 (count expression)))
    (let [argument (second expression) known (fact env argument)]
      (cond (:fields known) false
            (:present? known) false
            (contains? known :literal) (nil? (:literal known))
            ;; A primitive cast cannot return nil, but can throw. Retain its
            ;; evaluation once, at the original test position, before false.
            (primitive-cast? argument)
            (list 'let* [(gensym "presence_evaluation__") (lower env argument products?)] false)
            :else (relist expression [(first expression) (lower env argument products?)])))

    (and (seq? expression) (= 'if (first expression)))
    (let [[head test then else] expression
          test' (lower env test products?)
          known (fact env test')
          tail (or (when (contains? known :literal) {:bindings [] :literal (:literal known)})
                   (literal-tail (util/alpha-convert test')))]
      (if tail
        (let [branch (lower env (if (:literal tail) then else) products?)]
          (if (seq (:bindings tail)) (list 'let* (:bindings tail) branch) branch))
        (let [[then-env else-env] (nil-test-branches env test')]
          (relist expression (cond-> [head test' (lower then-env then products?)]
                              (= 4 (count expression)) (conj (lower else-env else products?)))))))

    (and (seq? expression) (form/scope-info expression))
    (lower-scopes env expression products?)

    (seq? expression)
    (let [lowered (relist expression (map #(lower env % products?) expression))]
      (or (project-scalar-accumulation env lowered)
          (project-array-accumulation env lowered) lowered))
    (vector? expression) (with-meta (mapv #(lower env % products?) expression) (meta expression))
    (map? expression) (into (empty expression) (map (fn [[k v]] [(lower env k products?) (lower env v products?)])) expression)
    (set? expression) (into (empty expression) (map #(lower env % products?)) expression)
    :else expression))

(defn lower-local-products
  "Refine only literal/alias presence and scalar-replace projectively used local
   vectors. Unknown calls/parameters never prove presence. If any product escapes,
   retain all product allocations and perform only literal presence refinement.
   Conditional product joins distribute only a bounded lexical continuation;
   an escape or expansion limit rolls the entire product rewrite back."
  [expression]
  (try (binding [*conditional-expansion-budget* (atom conditional-expansion-limit)]
         (lower {} expression true))
       (catch clojure.lang.ExceptionInfo e
         (case (:reason (ex-data e))
           (:local-product-escape :local-product-budget) (try (lower {} expression false)
                                      (catch clojure.lang.ExceptionInfo e
                                        (if (= :local-product-decline (:reason (ex-data e))) expression (throw e))))
           :local-product-decline expression
           (throw e)))))
