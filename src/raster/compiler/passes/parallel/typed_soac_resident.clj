(ns raster.compiler.passes.parallel.typed-soac-resident
  "Realize audited TypedSOAC scalars as resident one-element buffers.

   Logical reduction results remain rank-zero AbstractValues.  Their representation records the
   physical resident realization, scheduled reductions own the buffer output role, and consuming
   scalar regions load element zero through an explicit stable capture.  Pure scalar equations
   depending on a resident reduction are beta-reduced into their parallel consumers, so no device
   value is reconstructed or synchronized through host scalar control. A separately audited
   terminal ScalarIR result may use the same physical representation without importing the
   reduction rewrite rules."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.scalar-conversion :as scalar-conversion]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.extent-proof :as extent-proof]
            [raster.compiler.ir.index-expression :as index-expression]
            [raster.compiler.ir.semantic-fingerprint :as semantic]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.ir.write-coverage :as write-coverage]
            [raster.compiler.passes.parallel.device-scalar-schedule :as device-scalar]
            [raster.compiler.passes.parallel.typed-soac-initialization :as initialization]
            [raster.compiler.passes.parallel.typed-soac-fusion :as fusion]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-expression]
            [raster.compiler.passes.scalar.effects :as effects]))

(def resident-representation :resident-scalar-buffer)

(defn resident-scalar-value?
  [value]
  (= resident-representation (get-in value [:representation :kind])))

(defn- operation-info
  [equation]
  (or (some-> (fusion/equation-info equation)
              (assoc :original-equation equation))
      (throw (ex-info "resident realization received an unknown TypedSOAC equation"
                      {:reason :typed-soac-resident-equation :equation equation}))))

(defn- emit-equation
  [{:keys [kind id results attributes arrays captures destinations parameters locals body-results
           original-equation rewritten?]}]
  ;; Operations with multiple regions carry semantic parts beyond the primary lambda.
  ;; Unchanged equations must retain all of them, not be rebuilt by a single-lambda emitter.
  (if-not rewritten?
    original-equation
    (list '= id (vec results)
        (case kind
          :contract (list 'contract attributes (vec arrays) (vec captures))
          :scalar (list 'scalar attributes (vec captures)
                        (dialect/lambda-form (vec parameters) (dialect/emit-locals locals)
                                             (vec body-results)))
          :effect-map (list 'effect-map attributes (vec arrays) (vec captures)
                            (vec destinations)
                            (dialect/effect-lambda-form
                             (vec parameters) (dialect/emit-locals locals)
                             (vec body-results)))
          (list (symbol (name kind)) attributes (vec arrays) (vec captures)
                (dialect/lambda-form (vec parameters) (dialect/emit-locals locals)
                                     (vec body-results)))))))

(defn- parameter-parts
  [info]
  (dialect/parameter-layout (emit-equation info)))

(defn- result-transform-inputs
  [attributes]
  (let [{:keys [operands scalars]} (:result-transform attributes)]
    (map :value (concat operands scalars))))

(defn- use-sites
  [equations]
  (reduce
   (fn [uses equation]
     (let [{:keys [id kind arrays captures attributes]} (operation-info equation)]
       (-> uses
           (into (map (fn [value] [value {:equation id :role :array}]) arrays))
           (into (map (fn [value] [value {:equation id
                                         :role (case kind
                                                 :contract :contract-capture
                                                 (:product-reduce :segmented-fold-map)
                                                 :multi-region-capture
                                                 :capture)}]) captures))
           (into (map (fn [operand]
                        [(:value operand) {:equation id :role :result-transform-operand
                                           :descriptor operand}])
                      (get-in attributes [:result-transform :operands])))
           (into (map (fn [scalar]
                        [(:value scalar) {:equation id :role :result-transform-scalar
                                          :descriptor scalar}])
                      (get-in attributes [:result-transform :scalars])))
           (into (keep (fn [extent]
                         (when (dialect/value-id? extent)
                           [extent {:equation id :role :extent}])))
                 (dialect/operation-extents equation)))))
   [] equations))

(defn- promotable-result-transform-use?
  "A result-transform scalar may become a device load only when the complete typed producer and
   consumer boundary already proves that exact use. Rank zero alone is not storage authority."
  [facts info-by-id position-by-id producer-position value [_ {:keys [equation role descriptor]}]]
  (let [consumer (get info-by-id equation)
        transform (get-in consumer [:attributes :result-transform])
        actual (get-in facts [:values value])
        declared (:dtype descriptor)]
    (and (= :result-transform-scalar role)
         (= :reduce (:kind consumer))
         (< producer-position (get position-by-id equation -1))
         (= value (:value descriptor))
         (= 1 (count (filter #(= value (:value %)) (:scalars transform))))
         (not-any? #(= value (:value %)) (:operands transform))
         (= :tensor (:kind actual))
         (= [] (:shape actual))
         (keyword? (:dtype actual)) (dtype/known? (:dtype actual))
         (= (dtype/canon declared) (dtype/canon (:dtype actual)))
         (= {:kind :plain} (:representation actual))
         (nil? (:logical-layout actual))
         (nil? (:memory-space actual))
         (nil? (:placement actual))
         (nil? (:sharding actual))
         (nil? (:ownership actual))
         (empty? (:effects actual))
         (empty? (:attributes actual)))))

(defn- transform-load-candidate?
  [facts info-by-id position-by-id outputs host-reads producer result sites]
  (let [producer-position (get position-by-id (:id producer))
        equation-facts (get-in facts [:equations (:id producer)])
        transform-sites (filter #(= :result-transform-scalar (get-in % [1 :role])) sites)]
    (and (seq transform-sites)
         (not (contains? outputs result))
         (not (contains? host-reads result))
         (empty? (:effects equation-facts))
         (empty? (:aliases equation-facts))
         (empty? (get-in equation-facts [:attributes :result-storage]))
         (every? (fn [site]
                   (or (= :capture (get-in site [1 :role]))
                       (promotable-result-transform-use?
                        facts info-by-id position-by-id producer-position result site)))
                 sites))))

(defn- exact-plain-scalar-value?
  "Require the complete default scalar value contract before changing physical representation.
   Rank zero alone is not evidence that a value may become a device buffer."
  [value]
  (and (= :tensor (:kind value))
       (= [] (:shape value))
       (keyword? (:dtype value)) (dtype/known? (:dtype value))
       (= (:dtype value) (dtype/canon (:dtype value)))
       (= {:kind :plain} (:representation value))
       (nil? (:logical-layout value))
       (nil? (:memory-space value))
       (nil? (:placement value))
       (nil? (:sharding value))
       (nil? (:ownership value))
       (empty? (:effects value))
       (empty? (:attributes value))))

(defn- public-reused-reduction-root?
  "Admit one direct public reduction result only when it is also a fully audited device capture.

   The public occurrence is deliberately not inferred through dependent scalar equations. Every
   actual equation use must be a later ordinary capture; array, extent, multi-region and result-
   transform roles refuse. The producer and value contracts must be pristine because realization
   changes only their physical representation, never any other semantic facet."
  [facts info-by-id position-by-id producers-by-result outputs host-reads producer result sites]
  (let [producer-position (get position-by-id (:id producer))
        equation-facts (get-in facts [:equations (:id producer)])]
    (and (contains? outputs result)
         (= 1 (count (:results producer)))
         (= 1 (count (get producers-by-result result)))
         (seq sites)
         (every? (fn [[_ {:keys [equation role]}]]
                   (and (= :capture role)
                        (< producer-position (get position-by-id equation -1))
                        (some? (get info-by-id equation))))
                 sites)
         (not (contains? host-reads result))
         (empty? (:effects equation-facts))
         (empty? (:aliases equation-facts))
         (empty? (get-in equation-facts [:attributes :result-storage]))
         (exact-plain-scalar-value? (get-in facts [:values result])))))

(defn- result-transform-expression
  [info transform promoted]
  (let [{:keys [parameters body-results]} (dialect/lambda-parts (:lambda transform))
        accumulator-parameter (first parameters)
        accumulator (first (get-in info [:attributes :accumulators]))
        substitutions
        (into {accumulator-parameter accumulator}
              (concat
               (map (fn [{:keys [value parameter]}]
                      [parameter value])
                    (:operands transform))
               (map (fn [{:keys [value parameter]}]
                      [parameter (if (contains? promoted value)
                                   (list 'clojure.core/aget value 0)
                                   value)])
                    (:scalars transform))))]
    (util/subst-syms substitutions (first body-results))))

(defn- promote-result-transform-loads
  "Move only audited resident roots from the scalar ABI to the typed singleton operand ABI.
   The constructor re-closes the expression after inserting explicit element-zero loads."
  [info roots]
  (let [transform (get-in info [:attributes :result-transform])
        promoted (set/intersection roots (set (map :value (:scalars transform))))]
    (if (empty? promoted)
      info
      (let [old-operands (mapv #(select-keys % [:value :dtype :map]) (:operands transform))
            promoted-operands
            (into []
                  (comp (filter #(contains? promoted (:value %)))
                        (map #(assoc (select-keys % [:value :dtype])
                                     :map (axis-map/singleton-broadcast))))
                  (:scalars transform))
            retained-scalars
            (into []
                  (comp (remove #(contains? promoted (:value %)))
                        (map #(select-keys % [:value :dtype])))
                  (:scalars transform))
            rebuilt (dialect/make-result-transform
                     {:accumulator (first (get-in info [:attributes :accumulators]))
                      :expression (result-transform-expression info transform promoted)
                      :operands (into old-operands promoted-operands)
                      :scalars retained-scalars
                      :result-dtype (:result-dtype transform)})]
        (-> info
            (assoc :rewritten? true)
            (assoc-in [:attributes :result-transform] rebuilt)
            (update-in [:attributes :attributes :stable-array-captures]
                       (fn [stable]
                         (vec (filter (set/union (set stable) promoted)
                                      (:captures info))))))))))

(defn- scalar-definitions
  [equations]
  (into {}
        (keep (fn [equation]
                (let [info (operation-info equation)]
                  (when (and (= :scalar (:kind info))
                             (= 1 (count (:results info)))
                             (= 1 (count (:body-results info))))
                    [(first (:results info)) info]))))
        equations))

(defn- dependent-scalars
  [scalar-defs roots]
  (loop [dependent #{}]
    (let [known (set/union roots dependent)
          dependent'
          (into dependent
                (keep (fn [[result {:keys [captures]}]]
                        (when (some known captures) result)))
                scalar-defs)]
      (if (= dependent dependent') dependent (recur dependent')))))

(defn- scalar-expression
  [scalar-defs dependent roots id]
  (letfn [(expand [value visiting]
            (cond
              (contains? roots value)
              (list 'clojure.core/aget value 0)

              (contains? dependent value)
              (do
                (when (contains? visiting value)
                  (throw (ex-info "resident scalar equations contain a dependency cycle"
                                  {:reason :typed-soac-resident-scalar-cycle :value value})))
                (let [{:keys [captures parameters body-results]} (get scalar-defs value)
                      expression (util/subst-syms (zipmap parameters captures)
                                                  (first body-results))
                      substitutions (into {}
                                          (keep (fn [capture]
                                                  (when (or (contains? roots capture)
                                                            (contains? dependent capture))
                                                    [capture (expand capture (conj visiting value))])))
                                          captures)]
                  (util/subst-syms substitutions expression)))

              :else value))]
    (expand id #{})))

(defn- rewrite-lambda-captures
  "Substitute captured scalar equations without reconstructing a source-shaped kernel.
   `stable-captures` are newly introduced array reads that must keep the no-write-alias contract."
  [info values replacements stable-captures]
  (let [{:keys [accumulators elements capture-parameters destination-parameters]}
        (parameter-parts info)
        capture-substitutions
        (into {}
              (map (fn [[parameter capture]]
                     [parameter (get replacements capture capture)]))
              (map vector capture-parameters (:captures info)))
        global-locals (mapv #(update % :init
                                     (fn [init]
                                       (util/subst-syms capture-substitutions init)))
                            (:locals info))
        global-bodies (mapv #(util/subst-syms capture-substitutions %) (:body-results info))
        bound (set (concat accumulators elements destination-parameters
                           [(get-in info [:attributes :index])]
                           (map first (get-in info [:attributes :segment-axes]))))
        stable-before (set (get-in info [:attributes :attributes :stable-array-captures]))
        referenced-values (->> (concat (mapcat #(util/free-syms (:init %) bound) global-locals)
                                       (mapcat #(util/free-syms % bound) global-bodies)
                                       stable-before
                                       (result-transform-inputs (:attributes info)))
                               (filter #(contains? values %)) distinct (sort-by pr-str) vec)
        new-parameters (mapv #(symbol (str "%capture" %)) (range (count referenced-values)))
        body-substitutions (zipmap referenced-values new-parameters)
        stable-after (set (filter #(or (contains? stable-before %)
                                       (contains? stable-captures %)
                                       (resident-scalar-value? (get values %)))
                                  referenced-values))]
    (assoc info
           :rewritten? true
           :captures referenced-values
           :parameters (vec (concat accumulators elements new-parameters
                                    destination-parameters))
           :locals (mapv #(update % :init
                                  (fn [init]
                                    (util/subst-syms body-substitutions init)))
                         global-locals)
           :body-results (mapv #(util/subst-syms body-substitutions %) global-bodies)
           :attributes (assoc-in (:attributes info) [:attributes :stable-array-captures]
                                 (vec (filter stable-after referenced-values))))))

(defn- rewrite-lambda-consumer
  [info values scalar-defs dependent roots]
  (let [replacements
        (into {}
              (keep (fn [capture]
                      (cond
                        (contains? roots capture)
                        [capture (list 'clojure.core/aget capture 0)]
                        (contains? dependent capture)
                        [capture (scalar-expression scalar-defs dependent roots capture)])))
              (:captures info))]
    (rewrite-lambda-captures info values replacements #{})))

(defn- rewrite-consumer
  [info values scalar-defs dependent roots]
  ;; Only a lexical consumer of a realized/dependent scalar needs a new buffer load. Besides
  ;; avoiding needless alpha-renaming, this keeps unrelated host scalar equations entirely out
  ;; of the parallel parameter-layout machinery (their lambda may intentionally have no element
  ;; or accumulator partition). A contraction's scalar closure is likewise not a resident-buffer
  ;; lambda; its capture uses are explicit escape sites above.
  (let [affected (set/union dependent roots)]
    (if (or (= :contract (:kind info))
            (empty? (set/intersection affected (set (:captures info)))))
      info
      (rewrite-lambda-consumer info values scalar-defs dependent roots))))

(defn- uniform-input-load
  "Recognize an ordered scalar equation that reads one immutable program input.
   A launch extent or host-visible result is never a candidate for this transformation."
  [info input-values values]
  (let [{:keys [kind results captures parameters locals body-results]} info
        expression (first body-results)
        arguments (when (descriptor/aget-call? expression)
                    (vec (descriptor/call-args expression)))
        array (first captures)
        index (second arguments)]
    (when (and (= :scalar kind) (= 1 (count results)) (= 1 (count captures))
               (= 1 (count parameters)) (empty? locals) (= 1 (count body-results))
               (= 2 (count arguments)) (= (first arguments) (first parameters))
               (integer? index) (not (neg? index))
               (contains? input-values array)
               (seq (:shape (get values array))))
      {:result (first results) :array array :index index})))

(defn- exact-plain-tensor?
  [value shape]
  (if (empty? shape)
    (exact-plain-scalar-value? value)
    (and (= :tensor (:kind value))
         (= shape (:shape value))
         (keyword? (:dtype value)) (dtype/known? (:dtype value))
         (= (:dtype value) (dtype/canon (:dtype value)))
         (= {:kind :plain} (:representation value))
         (nil? (:logical-layout value))
         (nil? (:memory-space value))
         (nil? (:placement value))
         (nil? (:sharding value))
         (nil? (:ownership value))
         (empty? (:effects value))
         (empty? (:attributes value)))))

(defn- typed-scalar-effect-expression
  "Restore validated lambda-parameter dtypes solely for the shared effect/exception proof.

   TypedSOAC carries parameter types through capture AbstractValues rather than symbol metadata.
   The effect authority intentionally treats an untyped integral cast as possibly checked, so
   this proof view reattaches only those certified types.  The original expression is retained in
   the generated typed local and no flag is waived or reconstructed from an expected result."
  [facts info]
  (let [pairs (map vector (:parameters info) (:captures info))
        typed-symbols
        (into {}
              (keep (fn [[parameter capture]]
                      (let [value (get-in facts [:values capture])
                            tag (when (exact-plain-scalar-value? value)
                                  (dtype/scalar-tag-for-dtype (:dtype value)))]
                        (when (and (symbol? parameter) tag)
                          [parameter (with-meta parameter
                                       (merge (meta parameter)
                                              {:tag tag :raster.type/tag tag}))]))))
              pairs)]
    (when (= (count pairs) (count typed-symbols))
      (util/subst-syms typed-symbols (first (:body-results info))))))

(defn- nontrapping-scalar-operations?
  "Audit a complete emitted scalar operation tree with the existing totality boundary."
  [operations]
  (let [nodes (tree-seq #(or (map? %) (sequential? %))
                        #(if (map? %) (vals %) (seq %))
                        operations)]
    (not-any?
     (fn [operation]
       (let [expression (:expression operation)
             result-type (:result-type expression)]
         (or (= :trap (get-in expression [:options :overflow]))
             (and (contains? #{:byte :int :long} result-type)
                  (contains? #{:quot :rem :mod :div} (:op expression))))))
     nodes)))

(defn- typed-scalar-expression-total?
  "Ask the existing strict typed scalar lowerer to certify compound conversion boundaries.

   `require-source-types?` is essential. The lowerer's source-type entry derives the top-level
   dtype before lowering, so an enclosing result fact is never evidence for its operand. Any
   emitted trapping operation refuses removability. This helper emits nothing and catches only an
   ordinary coverage refusal."
  [facts info expression]
  (let [scalar-types
        (into {}
              (map (fn [[parameter capture]]
                     [parameter (get-in facts [:values capture :dtype])]))
              (map vector (:parameters info) (:captures info)))
        result-dtype (get-in facts [:values (first (:results info)) :dtype])]
    (when (and (every? (fn [[_ scalar-dtype]]
                         (and (keyword? scalar-dtype) (dtype/known? scalar-dtype)
                              (= scalar-dtype (dtype/canon scalar-dtype))))
                       scalar-types)
               (keyword? result-dtype) (= result-dtype (dtype/canon result-dtype)))
      (try
        (let [decline! (fn [rule message data]
                         (throw (ex-info message (assoc data :reason rule))))
              lowerer (scalar-expression/make-lowerer
                       {:arrays #{} :array-types {} :scalar-types scalar-types
                        :index-scope #{} :source-region [expression]
                        :require-source-types? true
                        :conversion-policy scalar-conversion/policy
                        :lower-index (fn [& _]
                                       (decline! :uniform-scalar-index
                                                 "scalar closure arithmetic cannot lower an index"
                                                 {:expression expression}))
                        :decline! decline! :id-prefix "uniform-proof"})
              derived ((:source-type lowerer) expression scalar-types)
              lowered (when derived ((:lower lowerer) expression derived scalar-types))
              operations (:operations lowered)
              trapping? (not (nontrapping-scalar-operations? operations))]
          (and (= result-dtype derived (:type lowered))
               (not trapping?)))
        (catch Exception _ false)))))

(defn- removable-typed-scalar-expression?
  [facts info expression]
  (let [{:keys [effect flags]} (effects/descriptor expression)]
    (and (= :pure effect)
         (or (empty? flags) (= #{:checked-source-cast} flags))
         (typed-scalar-expression-total? facts info expression))))

(defn- pristine-scalar-equation?
  "A scalar moved into the singleton device stage must already be a total removable binding.

   Empty retained effects alone are not enough: the shared scalar effect authority must also prove
   that evaluating the exact body has neither external effects nor exceptional-control flags.
   Indexed reads are admitted only by the separately checked root-load grammar below."
  [facts info root?]
  (let [equation-facts (get-in facts [:equations (:id info)])
        body (first (:body-results info))
        effect-expression (typed-scalar-effect-expression facts info)
        value (get-in facts [:values (first (:results info))])]
    (and (= :scalar (:kind info))
         (= 1 (count (:results info)))
         (= 1 (count (:body-results info)))
         (empty? (:locals info))
         (empty? (:effects equation-facts))
         (empty? (:aliases equation-facts))
         (empty? (get-in equation-facts [:attributes :result-storage]))
         (set/subset? (set (keys (:attributes equation-facts))) #{:source-prefix?})
         (= #{:dtypes} (set (keys (:attributes info))))
         (empty? (get-in info [:attributes :attributes]))
         (= [(:dtype value)] (get-in info [:attributes :dtypes]))
         (or root?
             (and effect-expression
                  (removable-typed-scalar-expression? facts info effect-expression)
                  (not-any? descriptor/aget-call? (tree-seq coll? seq body))
                  (not-any? (fn [form]
                              (and (seq? form)
                                   (descriptor/alloc-op? (descriptor/semantic-op form))))
                             (tree-seq coll? seq body)))))))

(defn- exact-zero-input-read
  "Return the canonical input parameter read beneath at most one proved identity cast.

   Frontend scalar equations retain explicit primitive identity casts.  The cast is peeled only
   for admission: its complete original expression remains the first typed local in the new map."
  [expression parameter input-dtype result-dtype]
  (letfn [(lexical-call? [form]
            (and (seq? form)
                 (or (:raster.op/lexical-callee (meta form))
                     (:raster.op/lexical-callee (meta (first form))))))
          (read? [form]
            (let [operation (descriptor/semantic-op form)
                  arguments (vec (descriptor/call-args form))]
              (and (= 'clojure.core/aget operation)
                   (not (lexical-call? form))
                   (= 2 (count arguments))
                   (= parameter (first arguments))
                   (= 0 (second arguments)))))]
    (cond
      (read? expression) expression

      (and (seq? expression)
           (not (lexical-call? expression))
           (= 1 (count (descriptor/call-args expression))))
      (let [operation (descriptor/semantic-op expression)
            target (some-> operation descriptor/cast-result-tag
                           dtype/dtype-for-scalar-tag dtype/canon)
            operand (first (descriptor/call-args expression))]
        (when (and (contains? '#{clojure.core/byte clojure.core/int clojure.core/long
                                 clojure.core/float clojure.core/double} operation)
                   (= input-dtype result-dtype target)
                   (read? operand))
          operand))

      :else nil)))

(defn- alias-component
  "Compute the complete undirected component certified by equation alias facts."
  [aliases value]
  (let [neighbors
        (reduce (fn [graph [left right]]
                  (if (and (symbol? left) (symbol? right))
                    (-> graph (update left (fnil conj #{}) right)
                        (update right (fnil conj #{}) left))
                    graph))
                {} aliases)]
    (loop [pending [value] seen #{}]
      (if-let [current (peek pending)]
        (if (contains? seen current)
          (recur (pop pending) seen)
          (recur (into (pop pending) (get neighbors current)) (conj seen current)))
        seen))))

(defn- exact-uniform-root
  "Return the exact input[0] scalar boundary, or nil.

   The public input keeps its truthful self-extent AbstractValue. The ordinary static map-read
   proof later projects this literal zero access to a one-element KernelBody capacity obligation;
   this pass never pretends that the public array's logical shape is `[1]`."
  [facts info input-values written aliases outputs host-reads]
  (let [{:keys [results captures parameters body-results]} info
        result (first results)
        array (first captures)
        expression (first body-results)
        input-value (get-in facts [:values array])
        result-value (get-in facts [:values result])
        input-dtype (some-> input-value :dtype dtype/canon)
        result-dtype (some-> result-value :dtype dtype/canon)
        read (exact-zero-input-read expression (first parameters) input-dtype result-dtype)]
    (when (and (pristine-scalar-equation? facts info true)
               (= 1 (count results) (count captures) (count parameters))
               read
               (contains? input-values array)
               (empty? (set/intersection written (alias-component aliases array)))
               (not (contains? outputs result))
               (not (contains? host-reads result))
               (exact-plain-tensor? input-value [(list 'extent array)])
               (exact-plain-tensor? result-value [])
               (= (dtype/canon (:dtype input-value))
                  (dtype/canon (:dtype result-value))))
      {:result result :array array :info info})))

(defn- fresh-private-id
  [reserved value ordinal]
  (let [stem (str (name value) "__uniform_scalar_buffer")]
    (loop [suffix ordinal]
      (let [candidate (symbol (str stem "_" suffix))]
        (if (contains? reserved candidate)
          (recur (inc suffix))
          candidate)))))

(defn- allocate-private-id
  [reserved value ordinal]
  (let [id (fresh-private-id reserved value ordinal)]
    [id (conj reserved id)]))

(defn- uniform-scalar-closure
  "Prove one contiguous evaluate-once scalar closure rooted at an input[0] read.

   Every use is audited against original equation order. Closure values may feed later scalar
   equations and ordinary map/effect-map captures only; at least two terminal uses or one genuine
   dependent scalar is required so the existing direct single-use fast path remains unchanged."
  [facts infos uses position-by-id outputs host-reads root]
  (let [root-info (:info root)
        root-id (:id root-info)
        root-position (get position-by-id root-id)
        scalar-infos
        (loop [remaining (subvec infos (inc root-position))
               known #{(:result root)} result [root-info]]
          (if-let [info (first remaining)]
            (if (and (= :scalar (:kind info))
                     (some known (:captures info)))
              (recur (subvec remaining 1)
                     (into known (:results info)) (conj result info))
              (recur (subvec remaining 1) known result))
            result))
        scalar-values (set (mapcat :results scalar-infos))
        scalar-equations (set (map :id scalar-infos))
        producer-position
        (into {}
              (mapcat (fn [info]
                        (map (fn [result] [result (get position-by-id (:id info))])
                             (:results info))))
              infos)
        positions (mapv position-by-id (map :id scalar-infos))
        contiguous? (= positions (vec (range root-position (+ root-position (count positions)))))
        sites (mapcat #(get uses % []) scalar-values)
        info-by-id (into {} (map (juxt :id identity)) infos)
        scalar-site?
        (fn [[value {:keys [equation role]}]]
          (and (= :capture role) (contains? scalar-equations equation)
               (< (get producer-position value Long/MAX_VALUE)
                  (get position-by-id equation -1))))
        terminal-site?
        (fn [[value {:keys [equation role]}]]
          (and (= :capture role)
               (contains? #{:map :effect-map} (:kind (get info-by-id equation)))
               (< (get producer-position value Long/MAX_VALUE)
                  (get position-by-id equation -1))))
        terminal-sites (filter terminal-site? sites)
        frontier (->> scalar-values
                      (filter #(some terminal-site? (get uses % [])))
                      (sort-by (fn [value]
                                 (get position-by-id (:id (some (fn [info]
                                                                  (when (some #{value} (:results info)) info))
                                                                scalar-infos)))))
                      vec)
        external-captures
        (set (mapcat (fn [info]
                       (remove scalar-values (:captures info))) scalar-infos))
        external-scalars (disj external-captures (:array root))
        input-values (set (:inputs facts))]
    (when (and contiguous?
               (every? #(pristine-scalar-equation? facts % (= root-id (:id %))) scalar-infos)
               (every? #(exact-plain-tensor? (get-in facts [:values (first (:results %))]) [])
                       scalar-infos)
               (not-any? outputs scalar-values)
               (not-any? host-reads scalar-values)
               (every? #(exact-plain-tensor? (get-in facts [:values %]) []) external-scalars)
               (every? input-values external-captures)
               (seq terminal-sites)
               (every? #(or (scalar-site? %) (terminal-site? %)) sites)
               (or (< 1 (count terminal-sites)) (< 1 (count scalar-infos))))
      {:root root :infos scalar-infos :values scalar-values :equations scalar-equations
       :frontier frontier :terminal-equations (set (map #(get-in % [1 :equation]) terminal-sites))})))

(defn- singleton-scalar-stage
  "Build one extent-one typed stage without deciding which scalar equations may be staged.

   Capture descriptors map each original logical capture to its physical stage value. `:load?`
   introduces the sole certified singleton-buffer read; ordinary scalar inputs remain parameters.
   Placement facts and constituents belong only to this stage's removed equations."
  [facts {:keys [infos frontier equations capture-descriptors capture-values
                 stable-captures pass] :as stage}
   reserved]
  (let [root-info (first infos)
        internal-values (set (mapcat :results infos))
        external-logicals
        (vec (distinct (mapcat (fn [info]
                                 (remove internal-values (:captures info))) infos)))
        descriptor-by-logical (into {} (map (juxt :logical identity)) capture-descriptors)
        _descriptor-coverage
        (when-not (and (= (count capture-descriptors) (count descriptor-by-logical))
                       (= (set external-logicals) (set (keys descriptor-by-logical))))
          (throw (ex-info "singleton scalar stage requires one exact capture descriptor"
                          {:reason :uniform-scalar-stage-capture-descriptors
                           :expected external-logicals
                           :actual (mapv :logical capture-descriptors)})))
        descriptors (mapv descriptor-by-logical external-logicals)
        certified-values (merge (:values facts) capture-values)
        _load-authority
        (doseq [{:keys [logical value load?] :as descriptor} descriptors]
          (let [logical-value (get certified-values logical)
                physical-value (get certified-values value)]
            (when-not
             (if load?
               (and (contains? stable-captures value)
                    (exact-plain-tensor? logical-value [])
                    (exact-plain-tensor? physical-value [1])
                    (= (dtype/canon (:dtype logical-value))
                       (dtype/canon (:dtype physical-value))))
               (= logical value))
              (throw (ex-info "singleton capture descriptor lacks exact value authority"
                              {:reason :uniform-scalar-stage-capture-authority
                               :descriptor descriptor})))))
        external-captures (vec (distinct (map :value descriptors)))
        [index reserved] (allocate-private-id reserved 'uniform-index 0)
        [capture-parameters reserved]
        (reduce (fn [[parameters reserved] [ordinal capture]]
                  (let [[parameter reserved]
                        (allocate-private-id reserved
                                             (symbol (str "uniform-capture-" (name capture)))
                                             ordinal)]
                    [(conj parameters parameter) reserved]))
                [[] reserved] (map-indexed vector external-captures))
        [frontier-map reserved-after-frontier]
        (reduce (fn [[mapping reserved] [ordinal value]]
                  (let [[id reserved] (allocate-private-id reserved value ordinal)]
                    [(assoc mapping value id) reserved]))
                [{} reserved] (map-indexed vector frontier))
        parameter-by-value (zipmap external-captures capture-parameters)
        external-substitutions
        (into {}
              (map (fn [{:keys [logical value load?]}]
                     (let [parameter (get parameter-by-value value)]
                       [logical (if load?
                                  (list 'clojure.core/aget parameter 0)
                                  parameter)])))
              descriptors)
        locals
        (mapv (fn [info]
                (let [result (first (:results info))
                      parameter-substitutions
                      (into {}
                            (map (fn [[parameter capture]]
                                   [parameter (get external-substitutions capture capture)]))
                            (map vector (:parameters info) (:captures info)))]
                  {:id result :dtype (dtype/canon (get-in facts [:values result :dtype]))
                   :init (util/subst-syms parameter-substitutions (first (:body-results info)))}))
              infos)
        results (mapv frontier-map frontier)
        attributes {:index index :extent 1
                    :attributes {:stable-array-captures
                                 (vec (filter stable-captures external-captures))}}
        operation (list 'map attributes [] external-captures
                        (dialect/lambda-form capture-parameters
                                             (dialect/emit-locals locals)
                                             frontier))
        equation (list '= (:id root-info) results operation)
        old-facts (select-keys (:equations facts) equations)
        root-equation-facts (get-in facts [:equations (:id root-info)])
        equation-facts
        (-> root-equation-facts
            (assoc :effects #{} :aliases {})
            (update :provenance assoc :pass pass)
            (update :attributes dissoc :result-storage :host-binding :host-bindings)
            (assoc-in [:attributes :fusion/constituents] old-facts))
        values
        (into {}
              (map (fn [[old new]]
                     [new (av/tensor {:dtype (dtype/canon (get-in facts [:values old :dtype]))
                                      :shape [1]})]))
              (map (fn [old] [old (frontier-map old)]) frontier))]
    {:equation equation :equation-facts equation-facts :values values
     :frontier-map frontier-map :external-captures external-captures
     :stage stage :reserved reserved-after-frontier}))

(defn- singleton-closure-map
  [facts closure reserved]
  (let [external-logicals
        (vec (distinct (mapcat (fn [info]
                                (remove (:values closure) (:captures info)))
                              (:infos closure))))]
    (singleton-scalar-stage
     facts
     (assoc closure
            :capture-descriptors
            (mapv (fn [logical] {:logical logical :value logical :load? false})
                  external-logicals)
            :stable-captures #{(get-in closure [:root :array])}
            :pass :uniform-input-scalar-closure)
     reserved)))

(defn- retained-host-scalar-prefix?
  "Prove that `value` is an already evaluated host scalar, without importing its producer.

   The recursive dependency graph must terminate at exact scalar program inputs. Every produced
   value has one prior pristine scalar producer and every edge is source-order dominated. This
   excludes rank-one reads, reductions, device equations, storage and dependencies on either
   scalar stage while leaving the certified host prefix in its original evaluate-once position."
  [facts producers position-by-id outputs host-reads forbidden value consumer-position]
  (let [inputs (set (:inputs facts))]
    (letfn [(eligible? [value consumer-position pending]
              (let [actual (get-in facts [:values value])
                    producer-pairs (get producers value)
                    producer (second (first producer-pairs))]
                (and (exact-plain-tensor? actual [])
                     (not (contains? forbidden value))
                     (not (contains? outputs value))
                     (not (contains? host-reads value))
                     (not (contains? pending value))
                     (or (contains? inputs value)
                         (and (= 1 (count producer-pairs))
                              producer
                              (= :scalar (:kind producer))
                              (< (get position-by-id (:id producer) Long/MAX_VALUE)
                                 consumer-position)
                              (pristine-scalar-equation? facts producer false)
                              (every? #(eligible?
                                        % (get position-by-id (:id producer))
                                        (conj pending value))
                                      (:captures producer)))))))]
      (eligible? value consumer-position #{}))))

(defn- rebuild-uniform-scalar-stages
  "Atomically replace audited scalar stages and rewrite only their certified terminal captures."
  [program infos stages generated certificate stats]
  (let [facts (dialect/facts program)
        outputs (set (dialect/outputs program))
        removed-equations (apply set/union (map :equations stages))
        removed-values (apply set/union (map :values stages))
        terminal-equations (apply set/union (map :terminal-equations stages))
        generated-by-id (into {} (map (fn [item]
                                        [(get-in item [:stage :infos 0 :id]) item]))
                              generated)
        replacements (apply merge (map :frontier-map generated))
        stable-frontiers (set (vals replacements))
        values (merge (apply dissoc (:values facts) removed-values)
                      (apply merge (map :values generated)))
        rewritten
        (->> infos
             (keep (fn [info]
                     (if-let [item (get generated-by-id (:id info))]
                       (:equation item)
                       (cond
                         (contains? removed-equations (:id info)) nil
                         (contains? terminal-equations (:id info))
                         (emit-equation
                          (rewrite-lambda-captures
                           info values
                           (into {}
                                 (map (fn [[old new]]
                                        [old (list 'clojure.core/aget new 0)]))
                                 replacements)
                           stable-frontiers))
                         :else (emit-equation info)))))
             vec)
        equation-ids (set (map second rewritten))
        definitions (set (mapcat #(nth % 2) rewritten))
        references (set (mapcat (fn [equation]
                                  (cond-> (dialect/operation-inputs equation)
                                    (dialect/value-id? (dialect/operation-extent equation))
                                    (conj (dialect/operation-extent equation))))
                                rewritten))
        storage (set (mapcat (fn [equation]
                               (map :destination
                                    (or (dialect/result-storage facts (second equation)) [])))
                             rewritten))
        live-values (set/union definitions references storage outputs)
        equation-facts
        (-> (apply dissoc (:equations facts) removed-equations)
            (merge (into {} (map (fn [item]
                                   [(get-in item [:stage :infos 0 :id])
                                    (:equation-facts item)]))
                         generated))
            (select-keys equation-ids))
        new-facts
        (-> facts
            (assoc :values (select-keys values live-values)
                   :inputs (vec (sort-by pr-str (set/difference references definitions))))
            (assoc :equations equation-facts)
            (assoc :effects (reduce set/union #{} (map :effects (vals equation-facts))))
            (assoc-in [:attributes :resident-uniform-input-scalar-closures] certificate))
        result (dialect/make new-facts rewritten (dialect/outputs program))]
    [(dialect/validate! result) stats]))

(defn- ordered-uniform-scalar-islands
  "Recognize exactly one late contiguous scalar island depending on an earlier uniform root.

   The equations between root and island are never moved or absorbed. External scalar captures
   are either program inputs or uniquely produced by an earlier pristine scalar equation; their
   producers stay in place."
  [facts infos uses position-by-id outputs host-reads root]
  (let [root-info (:info root)
        root-value (:result root)
        root-position (get position-by-id (:id root-info))
        producers
        (group-by first
                  (mapcat (fn [info]
                            (map (fn [result] [result info]) (:results info)))
                          infos))
        dependent-infos
        (loop [remaining (subvec infos (inc root-position)) known #{root-value} result []]
          (if-let [info (first remaining)]
            (if (and (= :scalar (:kind info)) (some known (:captures info)))
              (recur (subvec remaining 1) (into known (:results info)) (conj result info))
              (recur (subvec remaining 1) known result))
            result))
        island-values (set (mapcat :results dependent-infos))
        island-equations (set (map :id dependent-infos))
        positions (mapv #(get position-by-id (:id %)) dependent-infos)
        island-position (first positions)
        contiguous? (and (seq positions)
                         (= positions (vec (range island-position
                                                  (+ island-position (count positions))))))
        info-by-id (into {} (map (juxt :id identity)) infos)
        producer-position (into {}
                                (mapcat (fn [info]
                                          (map (fn [result]
                                                 [result (get position-by-id (:id info))])
                                               (:results info))))
                                infos)
        scalar-site?
        (fn [[value {:keys [equation role]}]]
          (and (= :capture role)
               (contains? island-equations equation)
               (< (get producer-position value Long/MAX_VALUE)
                  (get position-by-id equation -1))))
        terminal-site?
        (fn [[value {:keys [equation role]}]]
          (and (= :capture role)
               (contains? #{:map :effect-map} (:kind (get info-by-id equation)))
               (< (get producer-position value Long/MAX_VALUE)
                  (get position-by-id equation -1))))
        root-sites (get uses root-value [])
        island-sites (mapcat #(get uses % []) island-values)
        terminal-sites (filter terminal-site? island-sites)
        frontier (->> island-values
                      (filter #(some terminal-site? (get uses % [])))
                      (sort-by producer-position) vec)
        external-captures
        (set (mapcat (fn [info]
                       (remove #(or (= root-value %) (contains? island-values %))
                               (:captures info)))
                     dependent-infos))
        forbidden-prefix-values (conj island-values root-value)
        eligible-external?
        (fn [value]
          (retained-host-scalar-prefix?
           facts producers position-by-id outputs host-reads forbidden-prefix-values
           value island-position))]
    (when (and contiguous?
               (< (inc root-position) island-position)
               (every? #(pristine-scalar-equation? facts % false) dependent-infos)
               (every? #(exact-plain-tensor? (get-in facts [:values (first (:results %))]) [])
                       dependent-infos)
               (not-any? outputs island-values)
               (not-any? host-reads island-values)
               (seq root-sites)
               (every? scalar-site? root-sites)
               (seq terminal-sites)
               (every? #(or (scalar-site? %) (terminal-site? %)) island-sites)
               (every? eligible-external? external-captures))
      {:root root
       :root-stage {:root root :infos [root-info] :values #{root-value}
                    :equations #{(:id root-info)} :frontier [root-value]
                    :terminal-equations #{}}
       :island-stage {:root root :infos dependent-infos :values island-values
                      :equations island-equations :frontier frontier
                      :terminal-equations
                      (set (map #(get-in % [1 :equation]) terminal-sites))}
       :external-captures external-captures})))

(defn- materialize-ordered-uniform-scalar-islands
  [program infos candidate reserved]
  (let [facts (dialect/facts program)
        root-stage
        (assoc (:root-stage candidate)
               :capture-descriptors
               [{:logical (get-in candidate [:root :array])
                 :value (get-in candidate [:root :array]) :load? false}]
               :stable-captures #{(get-in candidate [:root :array])}
               :pass :ordered-uniform-scalar-root)
        root-generated (singleton-scalar-stage facts root-stage reserved)
        root-value (get-in candidate [:root :result])
        root-buffer (get (:frontier-map root-generated) root-value)
        island-stage
        (assoc (:island-stage candidate)
               :capture-descriptors
               (into [{:logical root-value :value root-buffer :load? true}]
                     (map (fn [value] {:logical value :value value :load? false}))
                     (sort-by pr-str (:external-captures candidate)))
               :stable-captures #{root-buffer}
               :capture-values (:values root-generated)
               :pass :ordered-uniform-scalar-island)
        island-generated
        (singleton-scalar-stage facts island-stage (:reserved root-generated))
        stages [root-stage island-stage]
        generated [root-generated island-generated]
        terminal-frontier (mapv (:frontier-map island-generated) (:frontier island-stage))]
    (rebuild-uniform-scalar-stages
     program infos stages generated
     [{:root root-value :input (get-in candidate [:root :array])
       :root-frontier [root-buffer] :frontier terminal-frontier
       :stages [{:kind :root :equation (get-in root-stage [:infos 0 :id])
                 :frontier [root-buffer]}
                {:kind :island :equation (get-in island-stage [:infos 0 :id])
                 :frontier terminal-frontier}]}]
     {:resident-uniform-input-closures 1
      :resident-uniform-input-scalar-islands 1
      :resident-uniform-input-scalars
      (+ (count (:values root-stage)) (count (:values island-stage)))})))

(defn- materialize-uniform-scalar-closure-candidate
  [program]
  (let [facts (dialect/facts program)
        equations (dialect/equations program)
        infos (mapv operation-info equations)
        position-by-id (into {} (map-indexed (fn [index info] [(:id info) index])) infos)
        uses (group-by first (use-sites equations))
        input-values (set (:inputs facts))
        outputs (set (dialect/outputs program))
        host-reads (set (get-in facts [:attributes :host-read-values]))
        written (set (concat (mapcat :destinations infos)
                             (mapcat (fn [equation]
                                       (map :destination
                                            (or (dialect/result-storage facts (second equation)) [])))
                                     equations)))
        aliases (mapcat (comp seq :aliases) (vals (:equations facts)))
        roots (keep #(exact-uniform-root facts % input-values written aliases outputs host-reads)
                    infos)
        closures (keep #(uniform-scalar-closure facts infos uses position-by-id outputs host-reads %)
                       roots)
        islands (when (= 1 (count roots))
                  (keep #(ordered-uniform-scalar-islands
                          facts infos uses position-by-id outputs host-reads %)
                        roots))
        reserved (into (set (keys (:values facts)))
                       (filter symbol? (tree-seq coll? seq equations)))
        miss [program {:resident-uniform-input-closures 0
                       :resident-uniform-input-scalars 0}]]
    ;; Preserve the established one-stage admission. The new two-island route additionally audits
    ;; every exact root and activates only when there is exactly one, never coordinating roots.
    (try
      (cond
        (and (= 1 (count roots)) (= 1 (count closures)))
        (let [closure (first closures)
              generated (singleton-closure-map facts closure reserved)]
          (rebuild-uniform-scalar-stages
           program infos [closure] [generated]
           [{:root (get-in closure [:root :result])
             :input (get-in closure [:root :array])
             :frontier (mapv (:frontier-map generated) (:frontier closure))}]
           {:resident-uniform-input-closures 1
            :resident-uniform-input-scalars (count (:values closure))}))

        (and (empty? closures) (= 1 (count islands)))
        (materialize-ordered-uniform-scalar-islands
         program infos (first islands) reserved)

        :else miss)
      ;; Admission is transactional. A closure that the reconstructed TypedSOAC boundary cannot
      ;; validate leaves the original program intact; no partially rewritten facts escape.
      (catch Exception _ miss))))

(defn- materialize-uniform-scalar-closure
  [program]
  ;; Validate the input outside the optional optimization transaction. Malformed input is never
  ;; disguised as an optimization miss; any failure while proving or rebuilding a candidate is.
  (let [program (dialect/validate! program)]
    (try
      (materialize-uniform-scalar-closure-candidate program)
      (catch Exception _
        [program {:resident-uniform-input-closures 0
                  :resident-uniform-input-scalars 0}]))))

(defn- inline-single-uniform-input-loads
  "Move a single-use uniform input load into its resident map/effect-map consumer.

   The input remains a stable, no-write-alias array capture. This is a device-side read of the
   currently bound buffer, not a host-side download or a cached copy of allocation-time data.
   Extent values, host-visible scalar results, non-input storage and arrays written anywhere in
   this program remain ordered scalar equations. The JVM route does not apply this pass."
  [program]
  (let [program (dialect/validate! program)
        facts (dialect/facts program)
        equations (dialect/equations program)
        infos (mapv operation-info equations)
        input-values (set (:inputs facts))
        outputs (set (dialect/outputs program))
        uses (group-by first (use-sites equations))
        kinds (into {} (map (juxt :id :kind)) infos)
        written (set (concat (mapcat :destinations infos)
                             (mapcat (fn [equation]
                                       (map :destination
                                            (or (dialect/result-storage facts (second equation)) [])))
                                     equations)))
        candidates
        (into {}
              (keep (fn [info]
                      (when-let [{:keys [result array] :as load}
                                 (uniform-input-load info input-values (:values facts))]
                        (let [sites (get uses result)]
                          (when (and (not (contains? outputs result))
                                     (not (contains? written array))
                                     (= 1 (count sites))
                                     (= :capture (get-in (first sites) [1 :role]))
                                     (contains? #{:map :effect-map}
                                                (get kinds (get-in (first sites) [1 :equation]))))
                            [result load])))))
              infos)]
    (if (empty? candidates)
      [program {:resident-uniform-input-loads 0}]
      (let [removed (set (keys candidates))
            rewritten
            (->> infos
                 (remove #(and (= :scalar (:kind %)) (some removed (:results %))))
                 (mapv (fn [info]
                         (if (and (contains? #{:map :effect-map} (:kind info))
                                  (some removed (:captures info)))
                           (let [loads (keep candidates (:captures info))
                                 replacements (into {}
                                                    (map (fn [{:keys [result array index]}]
                                                           [result (list 'clojure.core/aget array index)]))
                                                    loads)
                                 stable (set (map :array loads))]
                             (rewrite-lambda-captures info (:values facts) replacements stable))
                           info)))
                 (mapv emit-equation))
            equation-ids (set (map second rewritten))
            definitions (set (mapcat #(nth % 2) rewritten))
            references (set (mapcat (fn [equation]
                                      (cond-> (dialect/operation-inputs equation)
                                        (dialect/value-id? (dialect/operation-extent equation))
                                        (conj (dialect/operation-extent equation))))
                                    rewritten))
            inputs (vec (sort-by pr-str (set/difference references definitions)))
            storage (set (mapcat (fn [equation]
                                   (map :destination
                                        (or (dialect/result-storage facts (second equation)) [])))
                                 rewritten))
            live-values (set/union definitions references storage outputs)
            facts (-> facts
                      (assoc :values (select-keys (:values facts) live-values) :inputs inputs)
                      (update :equations select-keys equation-ids)
                      (assoc-in [:attributes :resident-uniform-input-loads]
                                (vec (sort-by pr-str removed))))]
        [(dialect/make facts rewritten (dialect/outputs program))
         {:resident-uniform-input-loads (count removed)}]))))

(defn inline-uniform-input-loads
  "Keep the direct single-use load fast path and additionally materialize one audited multi-use
   scalar closure as an evaluate-once singleton map. Both transformations are GPU-route policy;
   neither changes the JVM pipeline or the public invocation ABI."
  [program]
  (let [program (dialect/validate! program)
        [program closure-stats] (materialize-uniform-scalar-closure program)
        [program load-stats] (inline-single-uniform-input-loads program)]
    [program (merge closure-stats load-stats)]))

(defn- realize-reductions
  "Return `[program stats]`, realizing every eligible audited scalar reduction.

   Ordinary output-only roots and indirect/dependent outputs remain scalar. A direct public,
   single-result reduction root may become resident only when it also has at least one later
   ordinary capture and its complete producer, use and AbstractValue boundaries are pristine.
   A direct full-reduction result-transform scalar is an eligible use only when its unique prior
   producer and every use are audited together; marking the root resident and rebuilding all such
   uses as typed singleton loads is one transaction. Scalar chains depending on eligible roots
   must likewise remain internal and capture-only."
  [program]
  (let [program (dialect/validate! program)
        facts (dialect/facts program)
        equations (dialect/equations program)
        infos (mapv operation-info equations)
        info-by-id (into {} (map (juxt :id identity)) infos)
        position-by-id (into {} (map-indexed (fn [index info] [(:id info) index])) infos)
        producers-by-result
        (group-by first
                  (mapcat (fn [info] (map (fn [result] [result info]) (:results info))) infos))
        outputs (set (dialect/outputs program))
        host-reads (set (get-in facts [:attributes :host-read-values]))
        uses (group-by first (use-sites equations))
        stored-reduction-destinations
        (into {}
              (mapcat (fn [equation]
                        (when (= 'reduce (dialect/operation-kind equation))
                          (map (fn [result storage] [result (:destination storage)])
                               (nth equation 2)
                               (or (dialect/result-storage facts (second equation)) [])))))
              equations)
        stored-reduction-results (set (keys stored-reduction-destinations))
        public-reused-roots
        (set (mapcat (fn [{:keys [kind results] :as producer}]
                       (when (= :reduce kind)
                         (filter (fn [result]
                                   (public-reused-reduction-root?
                                    facts info-by-id position-by-id producers-by-result outputs
                                    host-reads producer result (get uses result [])))
                                 results)))
                     infos))
        candidate-roots
        (set (mapcat (fn [{:keys [kind results] :as producer}]
                       (when (= :reduce kind)
                         (filter (fn [result]
                                   (let [sites (get uses result [])]
                                     (and (= 1 (count (get producers-by-result result)))
                                          (or (and (or (not (contains? outputs result))
                                                       (contains? stored-reduction-results result))
                                              (every? #(= :capture (get-in % [1 :role])) sites))
                                              (contains? public-reused-roots result)
                                              (transform-load-candidate?
                                               facts info-by-id position-by-id outputs host-reads
                                               producer result sites)))))
                                 results)))
                     infos))
        scalar-defs (scalar-definitions equations)
        dependent (dependent-scalars scalar-defs candidate-roots)
        escaping-dependent
        (set (filter (fn [value]
                       (or (contains? outputs value)
                           (not-every? #(= :capture (get-in % [1 :role]))
                                       (get uses value []))))
                     dependent))
        blocked-roots
        (set (filter (fn [root]
                       (some (fn [value]
                               (let [expression (scalar-expression scalar-defs dependent
                                                                   candidate-roots value)]
                                 (contains? (util/free-syms expression) root)))
                             escaping-dependent))
                     candidate-roots))
        roots (set/difference candidate-roots blocked-roots)
        dependent (dependent-scalars scalar-defs roots)
        removed-scalars (set/difference dependent escaping-dependent)]
    (if (empty? roots)
      [program {:resident-reductions 0 :inlined-scalars 0}]
      (let [values (reduce (fn [vs root]
                             (let [destination (get stored-reduction-destinations root)
                                   mark-resident
                                   (fn [values id]
                                     (cond-> values
                                       id (assoc-in [id :representation]
                                                    {:kind resident-representation :elements 1})
                                       id (assoc-in [id :memory-space] :device)))]
                               (-> vs
                                   (mark-resident root)
                                   (mark-resident destination))))
                           (:values facts) roots)
            rewritten
            (->> infos
                 (remove #(and (= :scalar (:kind %))
                               (some removed-scalars (:results %))))
                 (mapv #(rewrite-consumer % values scalar-defs removed-scalars roots))
                 (mapv #(promote-result-transform-loads % roots))
                 (mapv emit-equation))
            equation-ids (set (map second rewritten))
            definitions (set (mapcat #(nth % 2) rewritten))
            references (set (mapcat (fn [equation]
                                      (cond-> (dialect/operation-inputs equation)
                                        (dialect/value-id? (dialect/operation-extent equation))
                                        (conj (dialect/operation-extent equation))))
                                    rewritten))
            inputs (vec (sort-by pr-str (set/difference references definitions)))
            storage-destinations
            (set (mapcat (fn [equation]
                           (map :destination
                                (or (dialect/result-storage facts (second equation)) [])))
                         rewritten))
            live-values (set/union definitions references storage-destinations outputs)
            facts (-> facts
                      (assoc :values (select-keys values live-values)
                             :inputs inputs)
                      (update :equations select-keys equation-ids)
                      (update :equations
                              (fn [equation-facts]
                                (reduce (fn [m equation]
                                          (let [id (second equation)
                                                results (set (nth equation 2))]
                                            (if (seq (set/intersection roots results))
                                              (assoc-in m [id :attributes :resident-realization]
                                                        {:kind resident-representation :elements 1})
                                              m)))
                                        equation-facts rewritten)))
                      (assoc-in [:attributes :resident-reductions] (vec (sort-by pr-str roots))))
            public-resident-outputs (set/intersection roots public-reused-roots)
            facts (cond-> facts
                    (seq public-resident-outputs)
                    (assoc-in [:attributes :public-resident-reduction-outputs]
                              (vec (sort-by pr-str public-resident-outputs))))
            result (dialect/make facts rewritten (dialect/outputs program))]
        [result {:resident-reductions (count roots)
                 :inlined-scalars (count removed-scalars)}]))))

(defn- lexical-call-marker?
  [form]
  (and (seq? form)
       (or (:raster.op/lexical-callee (meta form))
           (:raster.op/lexical-callee (meta (first form))))))

(defn- strict-device-scalar-body?
  "Prove one projected ScalarIR body through the existing strict scalar lowerer.

   Source dtypes come from capture AVs and typed locals. The result AV is checked only after the
   body derives its own dtype. This proof admits the bounded singleton read grammar and rejects
   trapping KernelBody operations; it neither emits nor schedules the result."
  [program equation]
  (let [decline! (fn [rule message data]
                   (throw (ex-info message (assoc data :reason rule
                                                  :device-scalar-eligibility true))))]
    (try
      (let [subprogram (dialect/equation-subprogram program equation)
          projected (device-scalar/projection subprogram)
          arrays (:arrays projected)
          scalars (:scalars projected)
          locals (:locals projected)
          body (:body projected)
          facts (dialect/facts subprogram)
          values (:values facts)
          array-types (into {} (map (fn [id] [id (get-in values [id :dtype])])) arrays)
          scalar-types (into {} (map (fn [id] [id (get-in values [id :dtype])])) scalars)
          expressions (concat (map :init locals) [body])
          reads (mapcat descriptor/aget-reads expressions)
          lexical? (some lexical-call-marker?
                         (mapcat #(tree-seq coll? seq %) expressions))
          exact-reads?
          (and (seq reads)
               (every? (fn [{:keys [sym idx]}]
                         (and (contains? arrays sym) (= 0 idx)))
                       reads))
          lowerer (scalar-expression/make-lowerer
                   {:arrays arrays :array-types array-types :scalar-types scalar-types
                    :index-scope (set (keys scalar-types))
                    :source-region [locals body]
                    :require-source-types? true
                    :conversion-policy scalar-conversion/policy
                    :lower-index (fn [expression scope]
                                   (index-expression/lower expression scope decline!))
                    :decline! decline! :id-prefix "device-scalar-proof"})
          lowered
          (when (and (not lexical?) exact-reads?)
            (reduce
             (fn [{:keys [environment operations]} {:keys [id dtype init]}]
               (let [derived ((:source-type lowerer) init environment)
                     value (when derived ((:lower lowerer) init derived environment))]
                 (when-not (= dtype derived (:type value))
                   (decline! :device-scalar-local-type
                             "device scalar local differs from its retained dtype"
                             {:local id :dtype dtype :derived derived}))
                 {:environment (assoc environment id dtype)
                  :operations (into operations (:operations value))}))
             {:environment scalar-types :operations []}
             locals))
          body-derived (when lowered ((:source-type lowerer) body (:environment lowered)))
          body-value (when body-derived
                       ((:lower lowerer) body body-derived (:environment lowered)))
          result-dtype (:dtype projected)
          operations (concat (:operations lowered) (:operations body-value))]
        (and lowered body-value (= result-dtype body-derived (:type body-value))
             (nontrapping-scalar-operations? operations)))
      (catch clojure.lang.ExceptionInfo exception
        (if (or (:device-scalar-eligibility (ex-data exception))
                (= :typed-soac-device-scalar-subset (:reason (ex-data exception))))
          false (throw exception))))))

(defn- terminal-device-scalar-candidate
  "Return the sole direct terminal ScalarIR equation whose singleton storage dependency is proved.

   Writer lookup is performed over the complete ordered equation vector before any candidate index
   is selected. Exactly one earlier complete singleton writer is required; a future/latest writer,
   partial effect, public array read, or non-prefix produced scalar capture refuses."
  [program]
  (let [facts (dialect/facts program)
        equations (dialect/equations program)
        positions (into {} (map-indexed (fn [index equation] [(second equation) index])) equations)
        outputs (dialect/outputs program)
        candidate (when (= 1 (count outputs))
                    (some #(when (= outputs (nth % 2)) %) equations))
        candidate-index (some-> candidate second positions)
        info (when candidate (operation-info candidate))
        info-by-id (into {} (map (juxt :id identity)) (map operation-info equations))
        producers
        (group-by first
                  (mapcat (fn [producer]
                            (map (fn [result] [result producer]) (:results producer)))
                          (vals info-by-id)))
        host-reads (set (get-in facts [:attributes :host-read-values]))
        captures (:captures info)
        shaped (filterv #(seq (get-in facts [:values % :shape])) captures)
        scalars (filterv #(empty? (get-in facts [:values % :shape])) captures)
        storage (first shaped)
        allocation (when storage
                     (some #(when (= storage (:destination %)) %)
                           (initialization/allocation-contracts! program)))
        allocation-environment
        (when allocation
          (reduce #(extent-proof/advance %1 facts %2)
                  (extent-proof/initial-environment program)
                  (take-while #(let [site (get-in facts [:equations (second %)
                                                       :provenance :source-binding-id])]
                                 (and (integer? site)
                                      (< site (:source-binding-id allocation))))
                              equations)))
        singleton-capacity?
        (if allocation
          (extent-proof/same-volume? allocation-environment
                                     (:extent allocation) [1])
          (= [1] (get-in facts [:values storage :shape])))
        writers
        (when storage
          (into []
                (mapcat
                 (fn [equation]
                   (keep (fn [[logical physical]]
                           (when (= storage physical)
                             {:equation equation :logical logical
                              :position (get positions (second equation))}))
                         (map vector (nth equation 2)
                              (dialect/physical-results facts equation)))))
                equations))
        writer (when (= 1 (count writers)) (first writers))
        writer-facts (get-in facts [:equations (some-> writer :equation second)])
        writer-sites (or (seq (keys (get-in writer-facts [:attributes :fusion/constituents])))
                         [(get-in writer-facts [:provenance :source-binding-id])])
        allocation-before-writer?
        (or (nil? allocation)
            (and (every? integer? writer-sites)
                 (< (:source-binding-id allocation) (apply min writer-sites))))
        scalar-prefix?
        (fn [id]
          (retained-host-scalar-prefix?
           facts producers positions (set outputs) host-reads
           #{(first outputs) storage} id candidate-index))]
    (when (and candidate (= :scalar (:kind info))
               (= candidate-index (dec (count equations)))
               (= 1 (count shaped)) singleton-capacity? allocation-before-writer?
               (not (contains? host-reads storage))
               (exact-plain-tensor? (get-in facts [:values storage])
                                    (get-in facts [:values storage :shape]))
               writer (< (:position writer) candidate-index)
               (= [1] (write-coverage/symbolic-complete-write-shape
                       facts (:equation writer) (:logical writer)))
               (every? scalar-prefix? scalars)
               (exact-plain-scalar-value? (get-in facts [:values (first outputs)])))
      candidate)))

(defn terminal-device-scalar-projection
  "Return the audited resident projection, or nil for an unsupported terminal scalar.
   Source normalization and residency use this same admission proof."
  [program]
  (when-let [equation (terminal-device-scalar-candidate program)]
    (let [result (first (nth equation 2))
          marked (-> program
                     dialect/facts
                     (assoc-in [:values result :representation]
                               {:kind resident-representation :elements 1})
                     (assoc-in [:values result :memory-space] :device)
                     (dialect/make (dialect/equations program) (dialect/outputs program)))]
      (when (strict-device-scalar-body? marked equation) marked))))

(defn- realize-terminal-device-scalar [program]
  (if-let [marked (terminal-device-scalar-projection program)]
    [marked {:resident-terminal-scalars 1}]
    [program {:resident-terminal-scalars 0}]))

(defn validated-terminal-resident-output
  "Independently reconstruct terminal residency from the enclosing semantic equation spine.
   A resident AV or an isolated ScalarIR equation is not activation authority."
  [parallel-program result]
  (let [equations (:equations parallel-program)
        algorithms (mapv :algorithm equations)
        outer (get-in parallel-program [:values result])
        terminal (some #(when (= [result] (:results %)) %) equations)
        terminal-facts (when (dialect/program-form? (:algorithm terminal))
                         (dialect/facts (:algorithm terminal)))
        terminal-equation (when terminal-facts
                            (first (dialect/equations (:algorithm terminal))))
        critical-storage (when terminal-equation
                           (filter #(seq (get-in terminal-facts [:values % :shape]))
                                   (:captures (dialect/operation-parts terminal-equation))))
        allocated (set (map :destination (get-in parallel-program [:attributes :allocations])))]
    (when (and (= [result] (:outputs parallel-program))
               (= {:kind :resident-scalar-buffer :elements 1} (:representation outer))
               (= :device (:memory-space outer))
               (every? dialect/program-form? algorithms)
               ;; A compatibility join cannot create the singleton-capacity premise. A capture
               ;; without its own retained allocator needs an exact inner shape witness.
               (every? (fn [id]
                         (or (contains? allocated id)
                             (some #(semantic/equivalent?
                                     (get-in (dialect/facts %) [:values id])
                                     (get-in parallel-program [:values id]))
                                   algorithms)))
                       critical-storage)
               (every? true?
                       (map (fn [equation algorithm]
                              (let [inner-equations (dialect/equations algorithm)
                                    inner-facts (dialect/facts algorithm)]
                                (and (= 1 (count inner-equations))
                                     (= (:id equation) (second (first inner-equations)))
                                     (= (:results equation) (dialect/outputs algorithm))
                                     (= (:operands equation) (:inputs inner-facts))
                                     (= (:effects equation) (:effects inner-facts))
                                     (or (semantic/equivalent? (:operations equation) inner-equations)
                                         (and (true? (get-in equation [:attributes :host-only]))
                                              (empty? (:operations equation))
                                              (= 'scalar (dialect/operation-kind (first inner-equations)))
                                              (every? #(exact-plain-scalar-value?
                                                        (get-in inner-facts [:values %]))
                                                      (:results equation))))
                                     (every? (fn [[id value]]
                                               (let [outer-value (get-in parallel-program [:values id])]
                                                 (and outer-value
                                                      (semantic/equivalent?
                                                       (av/merge-refinement value outer-value)
                                                       outer-value))))
                                             (:values inner-facts)))))
                            equations algorithms)))
      (let [facts {:values (:values parallel-program) :inputs (:inputs parallel-program)
                   :equations (apply merge (map #(get-in (dialect/facts %) [:equations]) algorithms))
                   :effects (:effects parallel-program)
                   :diagnostics (:diagnostics parallel-program)
                   :provenance (:provenance parallel-program)
                   :attributes (:attributes parallel-program)}
            unmarked (-> facts
                         (assoc-in [:values result :representation] {:kind :plain})
                         (assoc-in [:values result :memory-space] nil)
                         (dialect/make (vec (mapcat dialect/equations algorithms)) [result]))
            audited (terminal-device-scalar-projection unmarked)]
        (when (and audited
                   (semantic/equivalent? outer (get-in (dialect/facts audited) [:values result])))
          outer)))))

(defn realize
  "Realize audited reduction residents and one bounded direct terminal ScalarIR result."
  [program]
  (let [[program reduction-stats] (realize-reductions program)
        [program scalar-stats] (realize-terminal-device-scalar program)]
    [program (merge reduction-stats scalar-stats)]))
