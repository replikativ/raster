(ns raster.compiler.ir.segmented-weighted-reduction
  "Backend-neutral algebra for a segmented, normalized weighted reduction.

   For every segment s and visible member e, the plan computes a scalar score, maps it to a
   weight, reduces `weight*value` and `weight`, then normalizes the two sums. The member traversal
   and physical value access are explicit descriptors rather than part of the scalar algebra.

   This is deliberately an internal compiler plan, not a new user-facing attention primitive.
   Canonical attention and a future recognizer for indexed-dot/scatter programs can therefore
   converge here without making either source spelling opaque or changing its numerics."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.util :as util]))

(defrecord ScalarRegion [parameters body result-dtype])
(defrecord Reduction [operator identity map-region])
(defrecord SegmentedWeightedReductionPlan
           [id segment-axes membership storage score weight value
            numerator denominator normalization operands output
            accumulator-dtype source-operation provenance])

(def ^:private scalar-operators
  "Closed scalar vocabulary accepted in plan regions. Keeping regions closed makes recognition
   conservative: an unknown call declines before any reordering or fusion can change effects."
  '#{raster.numeric/+ raster.numeric/- raster.numeric/* raster.numeric//
     raster.numeric/min raster.numeric/max raster.numeric/abs raster.numeric/sqrt
     raster.math/exp raster.math/log
     clojure.core/+ clojure.core/- clojure.core/* clojure.core//
     clojure.core/min clojure.core/max clojure.core/abs
     Math/exp Math/log Math/sqrt})

(defn scalar-region?
  [x]
  (instance? ScalarRegion x))

(defn reduction?
  [x]
  (instance? Reduction x))

(defn plan?
  [x]
  (instance? SegmentedWeightedReductionPlan x))

(defn- scalar-expression?
  [parameters expression]
  (cond
    (or (number? expression) (boolean? expression)) true
    (symbol? expression) (contains? parameters expression)
    (seq? expression) (and (contains? scalar-operators (first expression))
                           (every? #(scalar-expression? parameters %) (rest expression)))
    :else false))

(defn validate-region!
  "Validate a closed, side-effect-free scalar region. Parameters are positional SSA-like names;
   the body may only use those names, numeric literals, and the closed scalar vocabulary above."
  [owner region]
  (when-not (scalar-region? region)
    (throw (ex-info "segmented reduction requires a ScalarRegion"
                    {:reason :segmented-reduction-invalid-region
                     :owner owner :region region :actual (type region)})))
  (let [{:keys [parameters body result-dtype]} region]
    (when-not (and (vector? parameters)
                   (every? symbol? parameters)
                   (= (count parameters) (count (distinct parameters))))
      (throw (ex-info "scalar region parameters must be an ordered vector of unique symbols"
                      {:reason :segmented-reduction-invalid-region-parameters
                       :owner owner :parameters parameters})))
    (when-not (and (dtype/known? result-dtype) (dtype/fp-dtype? result-dtype))
      (throw (ex-info "scalar region result dtype must be floating point"
                      {:reason :segmented-reduction-invalid-region-dtype
                       :owner owner :dtype result-dtype})))
    (when (util/effectful? body)
      (throw (ex-info "scalar region cannot contain effects"
                      {:reason :segmented-reduction-effectful-region
                       :owner owner :body body})))
    (let [free (util/free-syms body)]
      (when-not (set/subset? free (set parameters))
        (throw (ex-info "scalar region references an undeclared parameter"
                        {:reason :segmented-reduction-region-free-symbol
                         :owner owner :free free :parameters parameters}))))
    (when-not (scalar-expression? (set parameters) body)
      (throw (ex-info "scalar region contains an unsupported operation or value"
                      {:reason :segmented-reduction-unsupported-scalar-expression
                       :owner owner :body body :supported scalar-operators})))
    region))

(defn region
  "Construct a checked scalar region."
  [{:keys [parameters body result-dtype]}]
  (validate-region! :region
                    (->ScalarRegion (vec parameters) body (dtype/canon result-dtype))))

(defn validate-reduction!
  [owner reduction expected-parameters accumulator-dtype]
  (when-not (reduction? reduction)
    (throw (ex-info "segmented reduction requires an explicit Reduction"
                    {:reason :segmented-reduction-invalid-reducer
                     :owner owner :reduction reduction :actual (type reduction)})))
  (let [{:keys [operator identity map-region]} reduction]
    (when-not (= :sum operator)
      (throw (ex-info "only additive weighted reductions are currently legal"
                      {:reason :segmented-reduction-unsupported-reducer
                       :owner owner :operator operator :supported #{:sum}})))
    (when-not (and (number? identity) (zero? identity))
      (throw (ex-info "additive weighted reduction requires numeric zero identity"
                      {:reason :segmented-reduction-invalid-identity
                       :owner owner :identity identity})))
    (validate-region! owner map-region)
    (when-not (= expected-parameters (:parameters map-region))
      (throw (ex-info "weighted reduction map region has the wrong positional contract"
                      {:reason :segmented-reduction-map-arity
                       :owner owner :expected expected-parameters
                       :actual (:parameters map-region)})))
    (when-not (= (dtype/canon accumulator-dtype) (:result-dtype map-region))
      (throw (ex-info "weighted reduction map must produce the accumulator dtype"
                      {:reason :segmented-reduction-map-dtype
                       :owner owner :expected accumulator-dtype
                       :actual (:result-dtype map-region)})))
    reduction))

(defn reduction
  "Construct a checked reduction. The plan validator additionally checks its positional inputs."
  [{:keys [operator identity map-region] :or {operator :sum identity 0.0}}]
  (->Reduction operator identity map-region))

(defn- positive-axis!
  [axis]
  (when-not (and (map? axis) (keyword? (:name axis))
                 (integer? (:extent axis)) (pos? (:extent axis)))
    (throw (ex-info "segment axes require keyword names and positive static extents"
                    {:reason :segmented-reduction-invalid-segment-axis :axis axis})))
  axis)

(defn- descriptor!
  [field descriptor]
  (when-not (and (map? descriptor) (keyword? (:kind descriptor)))
    (throw (ex-info "segmented reduction descriptor requires a keyword kind"
                    {:reason :segmented-reduction-invalid-descriptor
                     :field field :descriptor descriptor})))
  descriptor)

(defn- buffer-descriptor!
  [field descriptor]
  (when-not (and (map? descriptor)
                 (some? (:id descriptor))
                 (dtype/known? (:dtype descriptor))
                 (vector? (:shape descriptor))
                 (seq (:shape descriptor))
                 (every? #(and (integer? %) (pos? %)) (:shape descriptor))
                 (integer? (:elements descriptor))
                 (pos? (:elements descriptor))
                 (= (:elements descriptor) (reduce * 1 (:shape descriptor))))
    (throw (ex-info "segmented reduction buffer descriptor is incomplete or inconsistent"
                    {:reason :segmented-reduction-invalid-buffer-descriptor
                     :field field :descriptor descriptor})))
  descriptor)

(defn validate!
  "Validate a segmented weighted-reduction plan independently of target scheduling."
  [plan]
  (when-not (plan? plan)
    (throw (ex-info "segmented weighted reduction must be a plan value"
                    {:reason :segmented-reduction-invalid-plan
                     :plan plan :actual (type plan)})))
  (let [{:keys [id segment-axes membership storage score weight value numerator denominator
                normalization operands output accumulator-dtype source-operation provenance]} plan]
    (when (nil? id)
      (throw (ex-info "segmented weighted reduction requires a stable identity"
                      {:reason :segmented-reduction-missing-id})))
    (when-not (and (vector? segment-axes) (seq segment-axes))
      (throw (ex-info "segmented weighted reduction requires ordered segment axes"
                      {:reason :segmented-reduction-missing-segment-axes
                       :segment-axes segment-axes})))
    (doseq [axis segment-axes] (positive-axis! axis))
    (when-not (= (count segment-axes) (count (distinct (map :name segment-axes))))
      (throw (ex-info "segmented reduction axis names must be unique"
                      {:reason :segmented-reduction-duplicate-axis
                       :axes (mapv :name segment-axes)})))
    (doseq [[field descriptor] [[:membership membership] [:storage storage]
                                [:score score] [:value value]]]
      (descriptor! field descriptor))
    (positive-axis! (:axis score))
    (when-not (and (dtype/known? accumulator-dtype) (dtype/fp-dtype? accumulator-dtype))
      (throw (ex-info "segmented reduction accumulator must be floating point"
                      {:reason :segmented-reduction-invalid-accumulator-dtype
                       :dtype accumulator-dtype})))
    (let [accumulator-dtype (dtype/canon accumulator-dtype)]
      (validate-region! :score-combine (:combine score))
      (validate-region! :score-finalize (:finalize score))
      (when-not (= ['left 'right] (:parameters (:combine score)))
        (throw (ex-info "score combine region must accept left and right elements"
                        {:reason :segmented-reduction-score-combine-arity
                         :actual (:parameters (:combine score))})))
      (when-not (= ['dot] (:parameters (:finalize score)))
        (throw (ex-info "score finalize region must accept the reduced dot value"
                        {:reason :segmented-reduction-score-finalize-arity
                         :actual (:parameters (:finalize score))})))
      (when-not (and (= accumulator-dtype (:result-dtype (:combine score)))
                     (= accumulator-dtype (:result-dtype (:finalize score))))
        (throw (ex-info "score regions must produce the accumulator dtype"
                        {:reason :segmented-reduction-score-dtype
                         :accumulator-dtype accumulator-dtype
                         :combine-dtype (:result-dtype (:combine score))
                         :finalize-dtype (:result-dtype (:finalize score))})))
      (validate-region! :weight weight)
      (when-not (and (= ['score] (:parameters weight))
                     (= accumulator-dtype (:result-dtype weight)))
        (throw (ex-info "weight region must map one score to the accumulator dtype"
                        {:reason :segmented-reduction-weight-contract
                         :parameters (:parameters weight) :dtype (:result-dtype weight)})))
      (validate-reduction! :numerator numerator ['weight 'value] accumulator-dtype)
      (validate-reduction! :denominator denominator ['weight] accumulator-dtype))
    (when-not (and (map? normalization)
                   (= :divide (:kind normalization))
                   (number? (:epsilon normalization))
                   (Double/isFinite (double (:epsilon normalization)))
                   (not (neg? (double (:epsilon normalization))))
                   (number? (:empty-result normalization)))
      (throw (ex-info "normalization requires finite nonnegative epsilon and an empty result"
                      {:reason :segmented-reduction-invalid-normalization
                       :normalization normalization})))
    (when-not (and (vector? operands) (every? map? operands))
      (throw (ex-info "segmented reduction operands must be an ordered descriptor vector"
                      {:reason :segmented-reduction-invalid-operands :operands operands})))
    (doseq [[index operand] (map-indexed vector operands)]
      (buffer-descriptor! [:operand index] operand))
    (buffer-descriptor! :output output)
    (let [ids (mapv :id operands)]
      (when (or (some nil? ids) (not= (count ids) (count (distinct ids))))
        (throw (ex-info "segmented reduction operand identities must be non-nil and unique"
                        {:reason :segmented-reduction-invalid-operand-identities :ids ids})))
      (when (or (nil? (:id output)) (some #{(:id output)} ids))
        (throw (ex-info "segmented reduction output must be distinct from every input"
                        {:reason :segmented-reduction-output-alias
                         :output (:id output) :inputs ids}))))
    (when (nil? source-operation)
      (throw (ex-info "segmented reduction must retain its source operation"
                      {:reason :segmented-reduction-missing-source-operation})))
    (when-not (and (map? provenance) (keyword? (:semantic-op provenance)))
      (throw (ex-info "segmented reduction requires semantic provenance"
                      {:reason :segmented-reduction-invalid-provenance
                       :provenance provenance})))
    plan))

(defn make
  "Construct and validate an internal segmented weighted-reduction plan."
  [fields]
  (validate!
   (map->SegmentedWeightedReductionPlan
    (update fields :accumulator-dtype dtype/canon))))

(defn ordered-input-ids
  [plan]
  (mapv :id (:operands (validate! plan))))

(defn algebra-key
  "Buffer- and storage-independent computation identity. Physical page routing and cache layout
   cannot change this key; logical membership and scalar/reduction semantics can."
  [plan]
  (let [{:keys [segment-axes membership score weight value numerator denominator normalization
                accumulator-dtype]} (validate! plan)]
    {:segment-axes segment-axes
     :membership (dissoc membership :buffers)
     :score (dissoc score :left :right)
     :weight weight
     :value (select-keys value [:components])
     :numerator numerator
     :denominator denominator
     :normalization normalization
     :accumulator-dtype accumulator-dtype}))

(defn schedule-key
  "Static code-generation identity: algebra plus storage representation, never buffer names."
  [plan]
  (let [{:keys [storage score value operands output] :as plan} (validate! plan)]
    {:algebra (algebra-key plan)
     :storage (dissoc storage :buffers :route)
     :score-access (-> score (select-keys [:kind :axis :head-map :left :right])
                       (update :left dissoc :buffer)
                       (update :right dissoc :buffer))
     :value-access (dissoc value :buffer)
     :operand-dtypes (mapv :dtype operands)
     :output-dtype (:dtype output)}))

(defn online-softmax-algebra?
  "True only for the canonical exp-normalized weighted sum that an online-softmax schedule can
   execute without changing semantics. Score scaling may vary; all other scalar/reduction forms
   are exact. A future GSDM clamp/epsilon plan therefore declines this leaf unless an emitter
   explicitly implements those semantics."
  [plan]
  (let [{:keys [score weight numerator denominator normalization accumulator-dtype]}
        (validate! plan)
        score-finalize (:body (:finalize score))]
    (and (= :dot (:kind score))
         (= '(raster.numeric/* left right) (:body (:combine score)))
         (seq? score-finalize)
         (= 'raster.numeric/* (first score-finalize))
         (= 'dot (second score-finalize))
         (= 3 (count score-finalize))
         (number? (nth score-finalize 2))
         (Double/isFinite (double (nth score-finalize 2)))
         (pos? (double (nth score-finalize 2)))
         (= '(raster.math/exp score) (:body weight))
         (= :sum (:operator numerator))
         (= 0.0 (double (:identity numerator)))
         (= '(raster.numeric/* weight value) (:body (:map-region numerator)))
         (= :sum (:operator denominator))
         (= 0.0 (double (:identity denominator)))
         (= 'weight (:body (:map-region denominator)))
         (= {:kind :divide :epsilon 0.0 :empty-result 0.0} normalization)
         (= accumulator-dtype (:result-dtype weight)))))
