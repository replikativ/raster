(ns raster.compiler.passes.parallel.product-consumer-region
  "Recognize a product reduction followed by an ordered scalar consumer.

   This pass derives schedule facts from retained SegOps and their exact multi-equation graph. It
   does not fuse or emit a kernel. A later body refinement may use the returned axis partition only
   after replaying this analysis against the same source graph."
  (:require [clojure.set :as set]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.extent-expression :as extent-expression]
            [raster.compiler.ir.index-algebra :as index-algebra]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]))

(defn- decline! [rule message data]
  (throw (ex-info message (assoc data :reason :product-consumer-region-declined
                                 :missing-rule rule :fallback :separate-kernels))))

(defn declined? [exception]
  (= :product-consumer-region-declined (:reason (ex-data exception))))

(defn- record-name [value]
  (some-> value class .getName))

(defn- segred? [value]
  (= "raster.compiler.ir.segop.SegRed" (record-name value)))

(defn- segmap? [value]
  (= "raster.compiler.ir.segop.SegMap" (record-name value)))

(defn- ordered-fold-form? [value]
  (or (dialect/scalar-fold-form? value) (dialect/product-fold-form? value)))

(defn- ordered-fold-parts [value]
  (if (dialect/product-fold-form? value)
    (dialect/product-fold-parts value)
    (dialect/scalar-fold-parts value)))

(defn- ordered-folds [region]
  (->> (tree-seq coll? seq region)
       (filter ordered-fold-form?)
       distinct
       vec))

(defn- without-product-folds [value]
  (cond
    (ordered-fold-form? value) ::ordered-fold
    (map? value) (into (empty value) (map (fn [[k v]] [k (without-product-folds v)])) value)
    (vector? value) (mapv without-product-folds value)
    (seq? value) (apply list (map without-product-folds value))
    (set? value) (set (map without-product-folds value))
    :else value))

(defn- loads-of [array value]
  (->> (tree-seq coll? seq value)
       (filter descriptor/aget-call?)
       (filter #(= array (descriptor/aget-array-sym %)))
       vec))

(defn- row-major-expression [digits bounds]
  (reduce (fn [outer [digit bound]]
            (list 'clojure.core/+ (list 'clojure.core/* outer bound) digit))
          (first digits)
          (map vector (rest digits) (rest bounds))))

(defn- extent-product [bounds]
  (case (count bounds)
    0 1
    1 (first bounds)
    (with-meta (apply list 'clojure.core/* bounds)
      {:raster.type/tag 'long :tag 'long})))

(defn- expand-locals
  "Expand the consumer's ordered scalar locals as pure source expressions.  The scalar-region
   validator has already established their SSA order; this projection is used only for an affine
   address proof and never evaluates or retypes a local."
  ([locals] (expand-locals {} locals))
  ([initial locals]
  (reduce (fn [bindings {:keys [id init]}]
            (assoc bindings id (util/subst-syms bindings init)))
          initial locals)))

(defn- address-substitutions
  [fold prefix consumer-prefix-binding consumer-locals]
  (let [consumer->producer
        (into {}
              (mapv (fn [consumer-digit producer-axis]
                      (when-not (symbol? consumer-digit)
                        (decline! :consumer-prefix-binding
                                  "consumer prefix decomposition must expose symbolic digits"
                                  {:digit consumer-digit :axis producer-axis}))
                      [consumer-digit (:name producer-axis)])
                    consumer-prefix-binding prefix))
        fold-locals (:locals (dialect/lambda-parts (:lambda (ordered-fold-parts fold))))]
    (-> (merge (expand-locals consumer-locals) consumer->producer)
        (expand-locals fold-locals))))

(defn- intermediate-load-offsets
  [intermediate fold prefix ordered local consumer-prefix-binding consumer-locals]
  (let [loads (loads-of intermediate fold)
        local-volume (reduce * 1 (map :bound local))
        axes (set (conj (mapv :name prefix) (:name ordered)))
        substitutions (address-substitutions fold prefix consumer-prefix-binding consumer-locals)
        expected (fn [offset]
                   (row-major-expression
                    (conj (mapv :name prefix) (:name ordered) offset)
                    (conj (mapv :bound prefix) (:bound ordered) local-volume)))]
    (mapv
     (fn [load]
       (let [coordinate (util/subst-syms substitutions (descriptor/aget-index load))
             matches (filterv #(axis-map/index= coordinate (expected %) axes)
                              (range local-volume))]
         (when-not (= 1 (count matches))
           (decline! :intermediate-layout
                     "consumer intermediate load is not one proved producer-local element"
                     {:load load :expanded-coordinate coordinate :axes axes
                      :local-volume local-volume :matching-offsets matches}))
         {:load load :expanded-coordinate coordinate :local-offset (first matches)}))
     loads)))

(defn- ordered-axis-selections [axes]
  (letfn [(walk [prefix remaining]
            (concat (when (seq prefix) [prefix])
                    (mapcat (fn [axis]
                              (walk (conj prefix axis)
                                    (vec (remove #(= axis %) remaining))))
                            remaining)))]
    (walk [] (vec axes))))

(defn- consumer-read-requirements
  [intermediate fold prefix ordered consumer-prefix-binding consumer-locals]
  (let [substitutions (address-substitutions fold prefix consumer-prefix-binding consumer-locals)
        axes (conj (vec prefix) ordered)
        axis-names (set (map :name axes))
        loads (remove #(= intermediate (descriptor/aget-array-sym %))
                      (filter descriptor/aget-call? (tree-seq coll? seq fold)))]
    (reduce
     (fn [requirements load]
       (let [buffer (descriptor/aget-array-sym load)
             coordinate (util/subst-syms substitutions (descriptor/aget-index load))
             matches
             (keep (fn [selection]
                     (let [digits (mapv :name selection)
                           bounds (mapv :bound selection)]
                       (when (axis-map/index= coordinate
                                             (row-major-expression digits bounds) axis-names)
                         (extent-product bounds))))
                   (ordered-axis-selections axes))
             extents (vec (distinct matches))]
         (when-not (= 1 (count extents))
           (decline! :consumer-read-layout
                     "consumer input load has no unique dense axis projection"
                     {:load load :expanded-coordinate coordinate :matches extents}))
         (if-let [prior (get requirements buffer)]
           (if (extent-expression/equivalent? prior (first extents))
             requirements
             (decline! :consumer-read-layout
                       "consumer input uses incompatible dense projections"
                       {:buffer buffer :left prior :right (first extents)}))
           (assoc requirements buffer (first extents)))))
     {} loads)))

(defn- permutations [values]
  (if (empty? values)
    [[]]
    (mapcat (fn [i]
              (let [value (nth values i)
                    remaining (vec (concat (subvec values 0 i) (subvec values (inc i))))]
                (map #(into [value] %) (permutations remaining))))
            (range (count values)))))

(defn- zero-offset? [form]
  (and (empty? (:offset-terms form))
       (= {:const 0 :factors []} (:offset form))))

(defn- prefix-axis-binding
  [consumer prefix derived-scalars]
  (let [dims (get-in consumer [:space :dims])
        _ (when-not (= 1 (count dims))
            (decline! :consumer-space
                      "ordered product consumer currently requires one flattened map axis"
                      {:dimensions dims}))
        {map-index :name raw-map-extent :bound} (first dims)
        map-extent (util/subst-syms derived-scalars raw-map-extent)
        ;; Prefix decomposition precedes the ordered Fold. Later component/projection locals can
        ;; be very large and cannot contribute digits of the outer map index.
        locals (->> (get-in consumer [:scalar-region :locals])
                    (take-while #(empty? (ordered-folds (:init %))))
                    (mapv #(update % :init (partial util/subst-syms derived-scalars))))
        bounds (mapv #(util/subst-syms derived-scalars (:bound %)) prefix)
        state (index-algebra/digits map-index map-extent locals)
        leaves (vec (:leaves state))
        candidates (if (= 1 (count prefix))
                     [[map-index]]
                     (when (= (count prefix) (count leaves))
                       (permutations leaves)))
        attempts
        (mapv (fn [digits]
                (let [expression (row-major-expression digits bounds)
                      form (index-algebra/index-form expression map-index map-extent locals {})]
                  {:consumer-digits digits :index-form form
                   :accepted? (and form (index-algebra/injective? form) (zero-offset? form)
                                   (= (set digits) (set (keys (:terms form))))
                                   (every? true?
                                           (map (fn [digit bound]
                                                  (= (index-algebra/monomial bound)
                                                     (get-in form [:terms digit :radix])))
                                                digits bounds)))}))
              candidates)
        matching (first (filter :accepted? attempts))]
    (or matching
        (decline! :consumer-prefix-bijection
                  "consumer map index is not a proved row-major bijection of producer prefix axes"
                  {:map-index map-index :map-extent map-extent
                   :prefix prefix :locals locals
                   :digit-state (select-keys state [:digits :leaves :parents])
                   :attempts attempts}))))

(defn- static-cooperative-width [local reduced]
  (let [bounds (mapv :bound (conj (vec local) reduced))]
    (when-not (every? #(and (integer? %) (pos? %)) bounds)
      (decline! :static-cooperative-width
                "the initial cooperative schedule requires static positive local/reduction axes"
                {:bounds bounds}))
    (let [width (reduce * 1 bounds)]
      (when-not (and (pos? width) (zero? (bit-and width (dec width))))
        (decline! :cooperative-width
                  "the initial cooperative schedule requires a power-of-two worker domain"
                  {:width width :bounds bounds}))
      width)))

(defn- expanded-combine-result
  [combine]
  (let [bindings (partition 2 (:bindings combine))]
    (reduce (fn [result [id init]]
              (util/subst-syms {id init} result))
            (first (:results combine))
            (reverse bindings))))

(defn- same-carrier-parameter
  [expression parameter carrier]
  (cond
    (= expression parameter) parameter
    (and (seq? expression) (= 2 (count expression))
         (= carrier
            (some-> (descriptor/cast-result-tag (first expression)) keyword dtype/canon))
         (= parameter (second expression))) parameter
    :else nil))

(defn- subgroup-collective-contract
  "Prove the initial subgroup candidate's exact scalar monoid spelling.

   Product-tree legality alone permits an arbitrary certified combine region. A hardware
   collective is narrower: it implements one registered binary operator directly. Keep that
   structural obligation in the target-neutral region proof so target selection never guesses an
   operator from a quantization or workload name."
  [operator]
  (let [components (:components operator)
        combine (:combine operator)]
    (when (= 1 (count components))
      (let [{:keys [dtype neutral]} (first components)
            dtype (dtype/canon dtype)
            [left right] (first (:parameters combine))
            result (expanded-combine-result combine)
            source-op (descriptor/semantic-op result)
            op (intrinsics/canonical source-op)
            args (vec (descriptor/call-args result))
            carrier-args (mapv (fn [argument]
                                 (or (same-carrier-parameter argument left dtype)
                                     (same-carrier-parameter argument right dtype)))
                               args)
            identity (when op (descriptor/typed-reduce-identity source-op dtype))
            overflow (intrinsics/source-overflow-policy source-op)]
        (when (and (= :+ op)
                   (contains? #{:byte :int :long} dtype)
                   (= 2 (count carrier-args))
                   (= #{left right} (set carrier-args))
                   (= :wrap overflow)
                   (= :wrap (get-in operator [:algebra :overflow]))
                   identity
                   (constant/equivalent? neutral identity))
          {:operator op
           :dtype dtype
           :neutral (:value (constant/value neutral))
           :arithmetic {:overflow :wrap}
           :association :implementation-defined
           :source-operation source-op})))))

(defn analyze
  "Derive a target-neutral schedule plan for two numerical producer/consumer equations.

   The producer's tree algebra may reorder only its certified inner reduction. The consumer's
   product Fold remains explicitly ordered; this function never grants it reassociation. Pure
   host scalar equations may occur between the two numerical endpoints when the graph builder
   proves they can be hoisted into the region's scalar prefix."
  [parallel-program equations]
  (let [equations (vec equations)
        _ (when-not (= 2 (count equations))
            (decline! :equation-count "product-consumer region requires exactly two equations"
                      {:equation-count (count equations)}))
        {:keys [algorithm body graph]} (equation-graph/make-for-equations
                                        parallel-program equations)
        operations (mapv (fn [equation]
                           (when-not (= 1 (count (:operations equation)))
                             (decline! :operation-count
                                       "each equation in a product-consumer region requires one SegOp"
                                       {:equation (:id equation)
                                        :operations (:operations equation)}))
                           (first (:operations equation)))
                         equations)
        [producer consumer] operations
        _ (when-not (and (segred? producer) (segmap? consumer))
            (decline! :operation-kinds
                      "product-consumer region requires SegRed followed by SegMap"
                      {:operations (mapv record-name operations)}))
        operator (reduction/validate! (:reduction producer))
        _ (try
            (reduction/validate-product-tree! operator (:schedule producer))
            (catch clojure.lang.ExceptionInfo exception
              (if (= :product-reduction-schedule-not-emittable
                     (:reason (ex-data exception)))
                (decline! :producer-schedule
                          "product consumer requires an admitted segmented workgroup tree"
                          {:strategy (get-in producer [:schedule :strategy])})
                (throw exception))))
        intermediate-set (set/intersection (:outputs producer) (:inputs consumer))
        _ (when-not (and (= 1 (count intermediate-set))
                         (= intermediate-set (:outputs producer)))
            (decline! :private-intermediate
                      "producer results must form one private consumer intermediate"
                      {:producer-outputs (:outputs producer)
                       :consumer-inputs (:inputs consumer)}))
        intermediate (first intermediate-set)
        _ (when-not (some #(= intermediate (:id %)) (:temporaries graph))
            (decline! :private-intermediate
                      "product intermediate must be private to the exact source graph"
                      {:intermediate intermediate
                       :temporaries (mapv :id (:temporaries graph))}))
        folds (ordered-folds (:scalar-region consumer))
        _ (when-not (= 1 (count folds))
            (decline! :ordered-consumer-fold
                      "consumer requires one structurally shared product Fold"
                      {:fold-count (count folds)}))
        fold (first folds)
        fold-parts (ordered-fold-parts fold)
        fold-attributes (:attributes fold-parts)
        _ (when-not (= :ordered (:association fold-attributes))
            (decline! :ordered-consumer-fold
                      "consumer product Fold must retain source order"
                      {:attributes fold-attributes}))
        fold-loads (loads-of intermediate fold)
        outside-loads (loads-of intermediate (without-product-folds (:scalar-region consumer)))
        _ (when (or (empty? fold-loads) (seq outside-loads))
            (decline! :intermediate-use
                      "the private product result may be read only by the ordered consumer Fold"
                      {:fold-load-count (count fold-loads)
                       :outside-load-count (count outside-loads)}))
        segments (segop/seg-space-segment-dims (:space producer))
        reduced (segop/seg-space-reduced-dim (:space producer))
        ordered-index (:index fold-attributes)
        ordered-position (first (keep-indexed #(when (= ordered-index (:name %2)) %1) segments))
        _ (when (nil? ordered-position)
            (decline! :ordered-axis
                      "consumer Fold index must name one producer segment axis"
                      {:index ordered-index :segments segments}))
        ordered (nth segments ordered-position)
        _ (when-not (extent-expression/equivalent? (:extent fold-attributes) (:bound ordered))
            (decline! :ordered-axis
                      "consumer Fold extent must equal its producer segment extent"
                      {:fold-extent (:extent fold-attributes) :segment ordered}))
        prefix (subvec (vec segments) 0 ordered-position)
        local (subvec (vec segments) (inc ordered-position))
        _ (when (empty? prefix)
            (decline! :prefix-axes
                      "product-consumer fusion requires at least one output-prefix axis" {}))
        host-prefix (take-while #(true? (get-in % [:attributes :host-only]))
                                (:equations body))
        derived-scalars (equation-graph/derived-scalar-expressions (:values body) host-prefix)
        prefix-binding (prefix-axis-binding consumer prefix derived-scalars)
        consumer-locals (get-in consumer [:scalar-region :locals])
        intermediate-offsets
        (intermediate-load-offsets intermediate fold prefix ordered local
                                   (:consumer-digits prefix-binding) consumer-locals)
        consumer-requirements
        (consumer-read-requirements intermediate fold prefix ordered
                                    (:consumer-digits prefix-binding) consumer-locals)
        cooperative-width (static-cooperative-width local reduced)
        integral-components? (every? #(contains? #{:byte :short :int :long}
                                                   (dtype/canon (:dtype %)))
                                     (:components operator))
        _ (when-not integral-components?
            (decline! :inner-exactness
                      "the initial fused schedule requires exact integral inner components"
                      {:dtypes (mapv :dtype (:components operator))}))]
    {:kind :product-ordered-consumer
     :source {:algorithm algorithm :body body :graph graph}
     :equations (mapv :id equations)
     :producer producer
     :consumer consumer
     :intermediate intermediate
     :intermediate-loads intermediate-offsets
     :consumer-read-requirements consumer-requirements
     :axes {:prefix prefix
            :prefix-binding (:consumer-digits prefix-binding)
            :ordered ordered
            :local local
            :reduced reduced}
     :workgroup-size cooperative-width
     :subgroup-collective (subgroup-collective-contract operator)
     :numerics {:inner {:association :implementation-defined
                        :dtypes (mapv :dtype (:components operator))
                        :overflow (get-in operator [:algebra :overflow])}
                :outer {:association :ordered
                        :dtypes (or (:dtypes fold-attributes)
                                    [(:dtype fold-attributes)])}}
     :provenance {:pass :product-consumer-region
                  :source-operations (mapv :id operations)}}))
