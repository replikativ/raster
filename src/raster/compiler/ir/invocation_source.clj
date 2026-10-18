(ns raster.compiler.ir.invocation-source
  "Shared correspondence between retained host source and typed invocation/realization values.
   These projections preserve the existing shape and auxiliary-binding rules. They neither
   execute source nor certify that an exceptional computation may move across device work."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.scalar-conversion :as scalar-conversion]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.index-expression :as index-expression]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.semantic-fingerprint :as semantic]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.scalar.effects :as effects]))

(def dtype->allocation
  {:float ['clojure.core/float-array 'floats 'float]
   :double ['clojure.core/double-array 'doubles 'double]
   :int ['clojure.core/int-array 'ints 'int]
   :long ['clojure.core/long-array 'longs 'long]
   :byte ['clojure.core/byte-array 'bytes 'byte]})

(defn shape-projection-source
  [expression]
  (loop [expression expression]
    (if (and (seq? expression)
             (descriptor/cast-op? (descriptor/semantic-op expression))
             (not= :wrap (descriptor/cast-integral-narrowing (descriptor/semantic-op expression)))
             (= 1 (count (descriptor/call-args expression))))
      (recur (first (descriptor/call-args expression)))
      (when (and (seq? expression)
                 (descriptor/alength-op? (descriptor/semantic-op expression))
                 (= 1 (count (descriptor/call-args expression)))
                 (symbol? (first (descriptor/call-args expression))))
        (first (descriptor/call-args expression))))))

(defn- scalar-value?
  [value]
  (and (= :tensor (:kind value)) (empty? (:shape value))))

(defn- dimension-expression
  [dimension]
  (if (and (seq? dimension) (= 'value (first dimension)) (= 2 (count dimension)))
    (second dimension)
    dimension))

(defn- host-computable-dimension?
  [program-values dimension]
  (let [dimension (dimension-expression dimension)
        references (util/free-syms dimension)]
    (or (integer? dimension)
        (and (symbol? dimension) (scalar-value? (get program-values dimension)))
        (and (seq? dimension)
             (= :pure (effects/analyze-effect dimension))
             (every? #(scalar-value? (get program-values %)) references)))))

(defn- certified-shape-dimension
  [program-values source]
  (let [shape (:shape (get program-values source))
        dimension (when (= 1 (count shape)) (first shape))]
    (when (and dimension (host-computable-dimension? program-values dimension))
      (dimension-expression dimension))))

(defn project-scalar-metadata
  "Project retained, exactly typed length leaves, not the enclosing scalar computation.
   Unknown widths, layouts and lexical regions remain unchanged. In particular a long storage
   dimension cannot silently replace an int length result without a separate range proof."
  [program-values public-inputs expression]
  (letfn [(project [expression]
            (if (or (not (seq? expression))
                    (contains? program-values (first expression))
                    (:raster.op/lexical-callee (meta expression))
                    (contains? '#{quote clojure.core/quote var clojure.core/var}
                               (first expression))
                    (completion/canonical-parts expression)
                    (form/scope-info expression))
              expression
              (let [operation (descriptor/semantic-op expression)
                    source (when (and (descriptor/alength-op? operation)
                                      (= 1 (count (descriptor/call-args expression))))
                             (first (descriptor/call-args expression)))
                    value (get program-values source)
                    dimension (when (and (symbol? source)
                                         (not (contains? public-inputs source))
                                         (= {:kind :plain} (:representation value))
                                         (nil? (:logical-layout value)))
                                (certified-shape-dimension program-values source))
                    leaf-dtype (some-> (types/sym-type-tag expression)
                                       dtype/dtype-for-scalar-tag dtype/canon)
                    dimension-dtype (when (symbol? dimension)
                                      (:dtype (get program-values dimension)))
                    dimension (if (and (symbol? dimension) dimension-dtype)
                                (vary-meta dimension assoc :raster.type/tag
                                           (dtype/scalar-tag-for-dtype dimension-dtype))
                                dimension)]
                (cond
                  (and (contains? #{:int :long} leaf-dtype)
                       (contains? #{:int :long} dimension-dtype)
                       (= [:exact :exact]
                          (scalar-conversion/policy dimension-dtype leaf-dtype)))
                  (if (= leaf-dtype dimension-dtype) dimension
                      (with-meta (list (symbol "clojure.core"
                                               (name (dtype/scalar-tag-for-dtype leaf-dtype)))
                                       dimension)
                        (meta expression)))

                  ;; Do not inspect binding bodies under a different lexical environment, quote
                  ;; data, or computed callees. Call argument order and original metadata survive.
                  (not (symbol? (first expression))) expression

                  :else (with-meta (apply list (first expression)
                                          (map project (rest expression)))
                          (meta expression))))))]
    (project expression)))

(defn canonicalize-shape-projection
  [program-values public-inputs symbol expression]
  (let [source (shape-projection-source expression)
        dimension (when (and source (not (contains? public-inputs source)))
                    (certified-shape-dimension program-values source))]
    (if-not dimension
      (project-scalar-metadata program-values public-inputs expression)
      (let [result-dtype (:dtype (get program-values symbol))
            dimension-dtype (when (symbol? dimension)
                              (:dtype (get program-values dimension)))]
        (if (or (nil? dimension-dtype) (= result-dtype dimension-dtype))
          dimension
          (list (dtype/scalar-tag-for-dtype result-dtype) dimension))))))

(defn allocation-pair
  [values result extent]
  (let [dtype (:dtype (get values result))
        [allocation tag] (or (get dtype->allocation dtype)
                             (throw (ex-info "TypedSOAC materialization has no array allocation"
                                             {:reason :typed-soac-materialization-dtype
                                              :result result :dtype dtype})))
        result' (with-meta result {:tag tag :raster.type/tag tag
                                   :raster.buffer/hoistable true})]
    [result' (list allocation extent)]))

(defn equation-allocation-pair
  "Exact allocation scaffolding for the equation kinds which realize fresh result storage."
  [values equation result]
  (let [{:keys [kind attributes]} (dialect/operation-parts equation)
        extent (case kind
                 map (:extent attributes)
                 reduce 1
                 scalar 1
                 nil)]
    (when extent (allocation-pair values result extent))))

(defn- realization-equations
  "Return the unchanged equation owning one exact retained auxiliary source pair.
   Allocation and alias correspondence is shared with the materializing producer."
  [program symbol expression]
  (let [source-pairs (mapv vec (partition 2 (second (:source program))))
        source-bindings (into {} source-pairs)]
    (filterv
      (fn [equation]
        (let [primary (second (:site equation))
              witness (get-in equation [:attributes :realization-bindings])
              destinations (set (map :destination
                                     (get-in equation [:attributes :result-storage])))]
          (and (= :binding (first (:site equation)))
               (vector? witness)
               ;; A nonmatching binding cannot be this equation's auxiliary result. Keep the
               ;; exact semantic checks below for every potentially matching witness; do not
               ;; encode unrelated kernel bodies while searching by binding identity.
               (some #(= symbol (:binding %)) witness)
               (parallel-program/equation-for-binding program primary (get source-bindings primary))
               (= (count witness) (count (distinct (map :binding witness))))
               (every? (fn [{:keys [binding expression]}]
                         (and (not= primary binding)
                              (= 1 (count (filter #(and (= binding (first %))
                                                         (semantic/equivalent? [binding expression] (vec %)))
                                                   source-pairs)))))
                       witness)
               (some (fn [{binding :binding generated :expression kind :kind}]
                       (and (= symbol binding)
                            (semantic/equivalent? [symbol expression] [binding generated])
                            (case kind
                              :result-allocation
                              (and (some #{binding} (:results equation))
                                   (dialect/program-form? (:algorithm equation))
                                   (some (fn [inner]
                                           (and (some #{binding} (nth inner 2))
                                                (semantic/equivalent?
                                                 [binding generated]
                                                 (equation-allocation-pair
                                                  (:values program) inner binding))))
                                         (dialect/equations (:algorithm equation))))
                              :storage-alias
                              (and (symbol? generated) (contains? destinations generated))
                              false)))
                     witness)
               equation)))
      (:equations program))))

(defn realization-equation
  "Return the unique equation owning an auxiliary pair; ambiguous ownership is invalid."
  [program symbol expression]
  (let [matches (realization-equations program symbol expression)]
    (when (< 1 (count matches))
      (throw (ex-info "auxiliary source binding has more than one equation owner"
                      {:reason :invocation-source-ambiguous-realization
                       :binding symbol :equations (mapv :id matches)})))
    (first matches)))

(defn realization-binding?
  "Whether an exact auxiliary pair belongs to an unchanged realized equation."
  [program symbol expression]
  (boolean (realization-equation program symbol expression)))

(defn realization-action
  "Recover the exact validated auxiliary action at an attested source position. A witness is
   correspondence only: the caller must still execute allocation or aliasing in source order."
  [program entry]
  (when (:realization entry)
    (let [[binding expression] (nth (vec (partition 2 (second (:source program))))
                                  (:position entry) nil)
          equation (realization-equation program binding expression)]
      (when-not (and (= binding (:binding entry)) (= (:id equation) (:realization entry)))
        (throw (ex-info "auxiliary action differs from its exact source owner"
                        {:reason :invocation-source-realization-action :entry entry})))
      (assoc (first (filter #(= binding (:binding %))
                            (get-in equation [:attributes :realization-bindings])))
             :equation (:id equation)))))

(defn scalar-equation-expression
  [equation]
  (let [algorithm (:algorithm equation)]
    (when (and (dialect/program-form? algorithm)
               (empty? (:effects equation))
               (= 1 (count (:results equation)))
               (= 1 (count (dialect/equations algorithm))))
      (let [algorithm-equation (first (dialect/equations algorithm))
            {:keys [kind captures lambda]} (dialect/operation-parts algorithm-equation)
            {:keys [parameters locals body-results]} (dialect/lambda-parts lambda)
            expression (when (and (= 'scalar kind) (empty? locals)
                                  (= 1 (count body-results)))
                         (util/subst-syms (zipmap parameters captures)
                                          (first body-results)))]
        expression))))

(defn typed-store-value
  [cast value]
  (let [source-operation (when (seq? value) (descriptor/semantic-op value))
        target (some-> cast descriptor/cast-result-tag
                       dtype/dtype-for-scalar-tag dtype/canon)
        source (when (seq? value)
                 (some-> source-operation descriptor/cast-result-tag
                         dtype/dtype-for-scalar-tag dtype/canon))]
    ;; The scalar equation's result dtype still owns the host/JVM boundary conversion.  When its
    ;; already-validated body ends in exactly that primitive cast, reusing it preserves the source
    ;; check/rounding point without manufacturing a redundant `(long (long ...))` wrapper.
    (cond
      (nil? cast) value

      (= target source)
      ;; This cast is now a compiler-attested result conversion. Preserve its exact operand and
      ;; metadata, but not a bare operator spelling that a public scalar named `long`/`float`
      ;; could capture when the materialized source is compiled in its destination scope.
      (if (and (= 2 (count value)) (descriptor/cast-op? source-operation))
        (with-meta (list (symbol "clojure.core" (name source-operation)) (second value))
          (meta value))
        value)

      :else
      (list (symbol "clojure.core" (name cast)) value))))

(defn scalar-equation-source
  "Project the same explicit result conversion used by host scalar materialization.
   This preserves the conversion; it does not infer its removability from a logical dtype."
  [equation]
  (when-let [expression (scalar-equation-expression equation)]
    (let [result (first (:results equation))
          dtype (get-in (dialect/facts (:algorithm equation)) [:values result :dtype])
          tag (dtype/scalar-tag-for-dtype dtype)]
      (typed-store-value (symbol "clojure.core" (name tag)) expression))))

(defn allocation-source
  "Project one definition-derived allocation to its exact source constructor. The caller must
   retain and compare helper correspondence with its compilation; this is not a check-hoisting
   or definition-lifetime witness. Constructor conversions remain executable source operations."
  [expression values]
  (when-let [{:keys [constructor] :as projection}
             (descriptor/direct-allocation-projection expression values)]
    (let [operation (descriptor/semantic-op constructor)
          tag (get descriptor/alloc-sym->array-tag operation)]
      (when tag
        (assoc projection :extent (descriptor/array-allocation-size constructor)
                          :dtype (some-> tag dtype/dtype-for-array-tag dtype/canon)
                          :initialization (descriptor/allocation-initialization operation))))))

(defn- same-integral-extent? [source dimension values]
  (or (semantic/equivalent? source dimension)
      (try
        (let [decline! (fn [& _] (throw (ex-info "not an exact typed extent" {:reason ::extent})))
              project #(index-expression/lower-typed % (set (keys values))
                                                    (fn [id] (:dtype (get values id))) :long decline!)]
          (semantic/equivalent? (project source) (project dimension)))
        (catch clojure.lang.ExceptionInfo error
          (if (= ::extent (:reason (ex-data error))) false (throw error))))))

(defn allocation-corresponds?
  "Check the bounded size-only allocation contract, without deriving shape from a dtype.
   Copy/initializer overloads and custom allocators require their own retained correspondence."
  ([expression value initialization] (allocation-corresponds? expression value initialization {}))
  ([expression value initialization values]
   (when-let [source (allocation-source expression values)]
     (and (some? (:extent source)) (= 1 (count (:shape value)))
          (= (dtype/canon (:dtype value)) (:dtype source))
          (= initialization (:initialization source))
          (same-integral-extent? (:extent source) (first (:shape value)) values)))))

(defn clone-corresponds?
  "A bounded clone correspondence requires the exact unary core operation and a direct
   lexical source. Cast erasure and ignored arguments cannot certify source behavior."
  [expression environment source]
  (and (seq? expression)
       (= 2 (count expression))
       (= 'clojure.core/aclone (first expression))
       (symbol? (second expression))
       (contains? environment (second expression))
       (= source (get environment (second expression)))))

(defn projection-corresponds?
  "Admit only exact conversions of an array-length result. Cast unwrapping alone cannot
   establish a checked narrowing or rounding contract for a ShapeProjection."
  [expression value]
  (loop [expression expression conversions []]
    (let [operation (descriptor/semantic-op expression)
          arguments (when (seq? expression) (descriptor/call-args expression))]
      (if (and (descriptor/cast-op? operation) (= 1 (count arguments)))
        (let [target (some-> operation descriptor/cast-result-tag dtype/dtype-for-scalar-tag)]
          (and target (not= :wrap (descriptor/cast-integral-narrowing operation))
               (recur (first arguments) (conj conversions target))))
        (and (descriptor/alength-op? operation) (= 1 (count arguments))
             (symbol? (first arguments))
             (let [result (reduce (fn [current target]
                                    (if (= [:exact :exact] (scalar-conversion/policy current target))
                                      target (reduced nil)))
                                  :int (reverse conversions))]
               (= result (dtype/canon (:dtype value)))))))))
