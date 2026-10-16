(ns raster.compiler.passes.scalar.local-products
  "Closed lexical presence refinement and nonescaping local vector replacement.
   No calls are evaluated and no arithmetic or dtype facts are inferred. Fields
   are snapshotted in source order; escaping products retain their source form."
  (:require [raster.compiler.ir.form :as form]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.numeric-constant :as numeric-constant]
            [raster.compiler.core.util :as util]))

(defn- literal? [x]
  (or (nil? x) (boolean? x) (number? x) (string? x) (keyword? x) (char? x)))

(defn- relist [original children]
  (with-meta (apply list children) (meta original)))

(defn- fact [env expression]
  (cond (literal? expression) {:literal expression}
        (symbol? expression) (get env expression)
        :else nil))

(defn- product-shape? [expression]
  (or (vector? expression)
      (and (seq? expression) (= 'let* (first expression))
           (= 3 (count expression)) (product-shape? (last expression)))))

(defn- split-product [expression]
  (if (vector? expression)
    [[] expression]
    (let [[bindings fields] (split-product (last expression))]
      [(into (vec (second expression)) bindings) fields])))

(defn- primitive-cast? [expression]
  (when (and (seq? expression) (= 2 (count expression)))
    (let [operation (descriptor/semantic-op expression)]
      (and (symbol? operation) (= "clojure.core" (namespace operation))
           (descriptor/cast-result-tag operation)))))

(defn- projection-index [expression]
  (when (or (integer? expression)
            (and (primitive-cast? expression) (some? (projection-index (second expression)))))
    (let [constant (numeric-constant/value expression)]
      (when (integer? (:value constant)) (:value constant)))))

(defn- literal-tail [expression]
  (cond (literal? expression) {:bindings [] :literal expression}
        (and (seq? expression) (= 'let* (first expression)) (= 3 (count expression)))
        (when-let [tail (literal-tail (last expression))]
          (update tail :bindings #(into (vec (second expression)) %)))
        :else nil))

(declare lower)

(defn- lower-let [env expression products?]
  (let [[head bindings & body] expression]
    (when-not (and (vector? bindings) (even? (count bindings))
                   (every? symbol? (take-nth 2 bindings)))
      (throw (ex-info "Local product replacement requires lexical symbol bindings"
                      {:reason :local-product-decline})))
    (let [[local output]
          (reduce
           (fn [[scope output] [id init]]
             (let [init (if (and products? (symbol? init) (:fields (get scope init)))
                          init (lower scope init products?))
                   alias (when (symbol? init) (get scope init))]
               (cond
                 (and products? (:fields alias))
                 [(assoc scope id alias) output]

                 (and products? (product-shape? init))
                 (let [[prefix fields] (split-product (util/alpha-convert init))
                       snapshots (mapv (fn [value]
                                         [(with-meta (gensym "product_field__")
                                            (select-keys (meta value) [:raster.type/tag])) value])
                                       fields)]
                   [(assoc scope id {:fields (mapv first snapshots)})
                    (into output (concat prefix (mapcat identity snapshots)))])

                 :else
                 [(assoc scope id (fact scope init)) (conj output id init)])))
           [env []] (partition 2 bindings))]
      (relist expression (concat [head (with-meta (vec output) (meta bindings))]
                                 (map #(lower local % products?) body))))))

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
    (and (seq? expression) (= 'clojure.core/nth (descriptor/semantic-op expression))
         (= 3 (count expression)) (symbol? (second expression))
         (get-in env [(second expression) :fields])
         (some? (projection-index (nth expression 2)))
         (<= 0 (projection-index (nth expression 2)))
         (< (projection-index (nth expression 2)) (count (get-in env [(second expression) :fields]))))
    (nth (get-in env [(second expression) :fields]) (projection-index (nth expression 2)))

    (and (seq? expression) (= 'clojure.core/nil? (descriptor/semantic-op expression)) (= 2 (count expression)))
    (let [argument (second expression) known (fact env argument)]
      (cond (:fields known) false
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
        (relist expression (cond-> [head test' (lower env then products?)]
                            (= 4 (count expression)) (conj (lower env else products?))))))

    (and (seq? expression) (form/scope-info expression))
    (lower-scopes env expression products?)

    (seq? expression) (relist expression (map #(lower env % products?) expression))
    (vector? expression) (with-meta (mapv #(lower env % products?) expression) (meta expression))
    (map? expression) (into (empty expression) (map (fn [[k v]] [(lower env k products?) (lower env v products?)])) expression)
    (set? expression) (into (empty expression) (map #(lower env % products?)) expression)
    :else expression))

(defn lower-local-products
  "Refine only literal/alias presence and scalar-replace projectively used local
   vectors. Unknown calls/parameters never prove presence. If any product escapes,
   retain all product allocations and perform only literal presence refinement."
  [expression]
  (try (lower {} expression true)
       (catch clojure.lang.ExceptionInfo e
         (case (:reason (ex-data e))
           :local-product-escape (try (lower {} expression false)
                                      (catch clojure.lang.ExceptionInfo e
                                        (if (= :local-product-decline (:reason (ex-data e))) expression (throw e))))
           :local-product-decline expression
           (throw e)))))
