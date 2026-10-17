(ns raster.ad.product-ports
  "Owner-rooted cotangent ports for immutable value products.

   Schemas contain shape and retained tangent declarations, never a child's
   private source IDs. Constructor/projection edges are local checked program
   data. Ports are namespaced keys distinct from scalar SSA symbols."
  (:require [raster.ad.tangent :as tangent]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.value-product :as product]))

(defn field-port [owner path]
  [::field owner (vec path)])

(defn leaf-paths
  ([schema] (leaf-paths schema []))
  ([schema prefix]
   (if (= :product (:kind schema))
     (mapcat (fn [[index field]] (leaf-paths field (conj prefix index)))
             (map-indexed vector (:fields schema)))
     [[prefix schema]])))

(defn join-facts
  "Join two already-owned product interfaces, without importing private value
   identities or treating an unknown shape as evidence of compatibility."
  [then-fact else-fact]
  (when-not (and (= :product (get-in then-fact [:schema :kind]))
                 (= (:schema then-fact) (:schema else-fact)))
    (throw (ex-info "conditional product arms require identical retained field schemas"
                    {:reason :ad-product-choice-schema
                     :then (:schema then-fact) :else (:schema else-fact)})))
  {:schema (:schema then-fact)
   :active-paths (into (:active-paths then-fact) (:active-paths else-fact))})

(defn choice-plan
  "Route a joined field port only to the selected existing product value.
   The primal choice is retained verbatim; no product identity is reconstructed."
  [owner branch then-source then-fact else-source else-fact]
  (let [joined (join-facts then-fact else-fact)]
    (assoc joined :edges
           (vec (mapcat (fn [[path leaf]]
                          (keep (fn [[source fact arm]]
                                  (when (contains? (:active-paths fact) path)
                                    {:port (field-port owner path) :path path :tag (:tag leaf)
                                     :source (field-port source path) :active? true
                                     :branch branch :arm arm}))
                                [[then-source then-fact :then] [else-source else-fact :else]]))
                        (leaf-paths (:schema joined)))))))

(defn- leaf-tag [expression value-facts]
  ;; Exact aliases transport an existing declaration; they do not justify
  ;; operation/type inference. Facts may be supplied by another owner, so stop
  ;; on cycles rather than allowing malformed provenance to hang compilation.
  (or (when (number? expression) (types/literal-tag expression))
      (loop [value expression seen #{}]
        (when (and (symbol? value) (not (contains? seen value)))
          (or (types/sym-type-tag value)
              (some-> (get-in value-facts [value :value]) types/sym-type-tag)
              (when-let [alias (get-in value-facts [value :alias])]
                (recur alias (conj seen value))))))))

(defn literal-plan
  "Derive a constructor's field edges from actual literal syntax and visible
   product facts. Compound scalar fields must first become owned SSA values;
   this function neither normalizes calls nor infers their result types."
  [owner expression value-facts activity]
  (when-not (vector? expression)
    (throw (ex-info "product construction requires literal fields" {:reason :ad-product-constructor})))
  (letfn [(walk [field path]
            (cond
              (vector? field)
              (let [children (mapv (fn [index value] (walk value (conj path index))) (range) field)]
                {:schema {:kind :product :fields (mapv :schema children)}
                 :edges (vec (mapcat :edges children))})

              (and (symbol? field) (get-in value-facts [field :product]))
              (let [{:keys [schema active-paths]} (get-in value-facts [field :product])]
                {:schema schema
                 :edges (mapv (fn [[child-path leaf]]
                                {:port (field-port owner (into path child-path))
                                 :path (into path child-path) :tag (:tag leaf)
                                 :source (field-port field child-path)
                                 :active? (contains? active-paths child-path)})
                              (leaf-paths schema))})

              (or (symbol? field) (number? field) (nil? field) (boolean? field)
                  (string? field) (keyword? field) (char? field))
              (let [active? (boolean (and (symbol? field) (get activity field)))
                    tag (leaf-tag field value-facts)]
                (when (and active? (not (tangent/differentiable? tag)))
                  (throw (ex-info "product field has no retained tangent-space declaration"
                                  {:reason :ad-product-unknown-tangent :field field :tag tag})))
                {:schema {:kind :leaf :tag tag}
                 :edges [{:port (field-port owner path) :path path :tag tag
                          :source field :active? active?}]})

              :else
              (throw (ex-info "product field is not an owned SSA value"
                              {:reason :ad-product-unclosed-field :field field :path path}))))]
    (let [{:keys [schema edges]} (walk expression [])]
      {:schema schema :edges edges
       :active-paths (into #{} (keep #(when (:active? %) (:path %))) edges)})))

(defn projection-plan
  "Project only a closed constant field of a known product. Retain the source
   expression in the primal; returned edges describe its transpose, not PE."
  [owner expression value-facts]
  (when (seq? expression)
   (let [source (second expression)
        {:keys [schema active-paths]} (get-in value-facts [source :product])
        count-fields (count (:fields schema))
        projection (when (= :product (:kind schema))
                     (or (product/constant-projection
                          expression count-fields #(get-in value-facts [% :constant-index]))
                         (when (and (= 2 (count expression))
                                    (= (first expression) (descriptor/semantic-op expression))
                                    (contains? '#{clojure.core/first clojure.core/second} (first expression)))
                           (let [index (if (= 'clojure.core/first (first expression)) 0 1)]
                             (when (< index count-fields) {:source source :index index})))))]
    (when-let [index (:index projection)]
      (let [result (nth (:fields schema) index)
            structured? (= :product (:kind result))
            edges (mapv (fn [[path leaf]]
                          {:port (if structured? (field-port owner path) owner)
                           :path path :tag (:tag leaf)
                           :source (field-port source (into [index] path))
                           :active? (contains? active-paths (into [index] path))})
                        (leaf-paths result))]
        {:schema result :edges edges
         :active-paths (into #{} (keep #(when (:active? %) (:path %))) edges)})))))
