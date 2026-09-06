(ns raster.compiler.ir.kernel-body
  "Target-neutral scheduled kernel bodies.

  SegOps retain algorithmic semantics.  A KernelBody is the result of applying a concrete
  schedule: hardware indices, named layouts, masks, fragment operations and loop structure are
  explicit, but target spellings such as Intel DPAS, CUDA WMMA or AMD MFMA are not.  Backends lower
  these values to their own dialect; they must not recover a schedule by inspecting source text."
  (:require [clojure.set :as set]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.scalar-range :as scalar-range]))

(defrecord KernelParameter [id kind dtype shape memory-space layout role])
(defrecord BufferView [id buffer element-offset shape layout])
(defrecord StableRead [buffer])
(defrecord WorkgroupAllocation [id dtype shape layout alignment])
(defrecord IndexBinding [id source axis])
(defrecord IndexCompute [id expression])
(defrecord IndexExpr [op arguments])
(defrecord IndexCast [argument dtype overflow])
(defrecord Predicate [op arguments])
(defrecord Mask [id predicates])
(defrecord Fragment [id dtype shape layout])
(defrecord ScalarRegion [parameters expression operands result-dtype])
(defrecord ScalarSSARegion
           [parameters operands indices accumulator-dtype operations result result-dtype])

;; General scalar/control kernel vocabulary.  Results are SSA values; :predicate is an internal
;; control type rather than a public storage dtype.  Memory operations remain element/rank aware,
;; while representation decode and numerical conversion are explicit scalar expressions.
(defrecord ValueSpec [id type])
(defrecord Literal [value type])
(defrecord ScalarExpr [op arguments result-type options])
(defrecord ScalarCompute [result expression])
(defrecord ScalarLoad [result buffer coordinates predicate other cache])
(defrecord ScalarStore [buffer coordinates value predicate])
(defrecord AtomicRMW [buffer coordinates value operator predicate])
(defrecord Yield [values])
(defrecord IfRegion [condition then-operations else-operations results])
(defrecord LoopArg [binding initial])
(defrecord ForLoop [index lower upper step iter-args operations results attributes])
(defrecord AsyncLoopArg [binding initial])
(defrecord PipelineYield [values groups])
(defrecord PipelinedFor
           [index lower upper step iter-args async-iter-args operations results async-results
            attributes])
(defrecord Participation [kind])
(defrecord Collective
           [result kind scope width input operator source-lane participation association])
(defrecord WorkgroupBarrier [scope memory-spaces semantics participation])
(defrecord AsyncWorkgroupCopy
           [id source source-coordinates destination destination-coordinates elements
            transfer-bytes cache overlap participation])
(defrecord AsyncCommit [id copies])
(defrecord AsyncWait [groups pending-groups semantics participation])

(defrecord FragmentInit [fragment value])
(defrecord TileLoad [fragment buffer coordinates mask cache])
(defrecord TilePrefetch [buffer coordinates shape layout mask distance])
(defrecord MatrixMad [accumulator lhs rhs instruction])
(defrecord Loop [index lower upper step operations attributes])
(defrecord Guard [mask operations])
(defrecord TileStore [buffer fragment coordinates mask value-region])

(defrecord KernelBody
           [id parameters views stable-reads allocations indices masks fragments operations
            schedule launch provenance attributes])

(def ^:private parameter-kinds #{:input :output :inout :scalar})
(def ^:private index-sources #{:group :group-count :local :subgroup :lane})
(def ^:private index-ops #{:add :sub :mul :floor-div :ceil-div :mod :min :max})
(def ^:private predicate-ops #{:lt :lte :eq :and :or :not})
(def ^:private cache-policies #{:default :cached :streaming})
(def ^:private internal-types #{:predicate})
(def ^:private special-scalar-ops #{:cast :select :isnan})
(def ^:private arithmetic-overflow-policies #{:wrap :trap :no-overflow})
(def ^:private collective-kinds #{:reduce :broadcast})
(def ^:private workgroup-memory-spaces #{:workgroup})
(def ^:private barrier-semantics #{:acquire-release})
(def ^:private async-wait-semantics #{:acquire})
(def ^:private async-overlap-policies #{:preferred :required})
(def ^:private async-transfer-widths #{4 8 16})
(def ^:private pipeline-tail-policies #{:exact :separate-epilogue})

(defn- record-kind? [class-name value]
  (and value (= class-name (.getName (class value)))))

(defn kernel-body? [value]
  (record-kind? "raster.compiler.ir.kernel_body.KernelBody" value))

(defn- value-id? [value]
  (or (symbol? value) (keyword? value)))

(defn expression
  "Construct explicit target-neutral integer index arithmetic."
  [op & arguments]
  (when-not (contains? index-ops op)
    (throw (ex-info "kernel index expression has an unsupported operation"
                    {:operation op :arguments arguments})))
  (->IndexExpr op (vec arguments)))

(defn index-cast
  "Construct an explicit integral representation conversion for index arithmetic."
  [argument target-dtype overflow]
  (->IndexCast argument target-dtype overflow))

(defn predicate
  "Construct an explicit target-neutral bounds predicate."
  [op & arguments]
  (when-not (contains? predicate-ops op)
    (throw (ex-info "kernel predicate has an unsupported operation"
                    {:operation op :arguments arguments})))
  (->Predicate op (vec arguments)))

(defn value
  "Construct a typed SSA result declaration.  `:predicate` is reserved for internal control."
  [id type]
  (->ValueSpec id type))

(defn literal
  "Construct a typed scalar literal.  Literals are explicit so target lowering never infers the
  width of a Clojure number."
  [value type]
  (->Literal value type))

(defn scalar-expression
  "Construct a target-neutral scalar expression using the canonical intrinsic key space.

  `options` is intentionally explicit and op-specific.  In particular, casts must state both
  rounding and overflow behavior."
  ([op result-type arguments]
   (scalar-expression op result-type arguments {}))
  ([op result-type arguments options]
   (->ScalarExpr op (vec arguments) result-type (or options {}))))

(defn cast-expression
  "Construct an explicit numerical conversion."
  [argument result-type rounding overflow]
  (scalar-expression :cast result-type [argument]
                     {:rounding rounding :overflow overflow}))

(defn full-participation
  "Every lane in the collective's statically selected subgroup participates."
  []
  (->Participation :full))

(defn stable-read
  "Require an input buffer to remain disjoint from every writable buffer for the complete kernel
  execution. This is an external binding precondition, not a load-local optimization hint."
  [buffer]
  (->StableRead buffer))

(defn- align-up-static
  [value alignment]
  (* (quot (+ value (dec alignment)) alignment) alignment))

(defn workgroup-memory-plan
  "Deterministically pack statically shaped workgroup allocations.

  Returns `{:allocations [{:allocation a :byte-offset n :byte-size n} ...]
            :bytes n :alignment n}`. The verifier ties `:bytes` exactly to the launch resource
  charge; C-family emitters consume this same plan instead of independently laying storage out."
  [allocations]
  (loop [remaining allocations offset 0 max-alignment 1 packed []]
    (if-let [allocation (first remaining)]
      (let [{:keys [dtype shape alignment]} allocation]
        (when-not (and (record-kind?
                        "raster.compiler.ir.kernel_body.WorkgroupAllocation" allocation)
                       (dtype/known? dtype)
                       (vector? shape) (seq shape) (every? pos-int? shape)
                       (contains? #{1 2 4 8 16} alignment)
                       (>= alignment (dtype/bytes-of dtype)))
          (throw (ex-info
                  "workgroup allocation requires a known dtype, static shape, and supported alignment"
                  {:reason :kernel-body-workgroup-allocation :allocation allocation})))
        (let [byte-offset (align-up-static offset alignment)
              byte-size (* (reduce * shape) (dtype/bytes-of dtype))]
          (recur (next remaining) (+ byte-offset byte-size) (max max-alignment alignment)
                 (conj packed {:allocation allocation
                               :byte-offset byte-offset
                               :byte-size byte-size}))))
      {:allocations packed
       :bytes (align-up-static offset max-alignment)
       :alignment max-alignment})))

(def ^:private scalar-region-forbidden-ops
  '#{raster.par/scan raster.par/scan-exclusive raster.par/scatter! raster.par/reduce-by-key
     raster.par/reduce raster.par/reduce-into raster.par/contract})

(defn scalar-region-legal?
  "Check whether a scalar expression can execute once in a tile store without changing layout.

  Accepts either a ScalarRegion or the source epilogue descriptor used before scheduling."
  [region]
  (let [accumulator (or (:acc region) (first (:parameters region)))
        expression (or (:expr region) (:expression region))
        nodes (tree-seq coll? seq expression)
        heads (into #{} (keep #(when (seq? %) (first %))) nodes)
        accumulator-uses (count (filter #(= accumulator %) nodes))]
    (cond
      (zero? accumulator-uses)
      {:ok false :reason :epilogue-ignores-accumulator}

      (> accumulator-uses 1)
      {:ok false :reason :accumulator-used-more-than-once}

      (some #{'raster.par/reduce 'raster.par/reduce-into} heads)
      {:ok false :reason :reduction-in-epilogue}

      (seq (set/intersection heads scalar-region-forbidden-ops))
      {:ok false :reason :layout-changing-op-in-epilogue
       :ops (set/intersection heads scalar-region-forbidden-ops)}

      :else {:ok true})))

(defn- expression? [value]
  (cond
    (or (integer? value) (value-id? value)) true
    (record-kind? "raster.compiler.ir.kernel_body.IndexCast" value)
    (and (contains? #{:int :long} (dtype/canon (:dtype value)))
         (numerics/cast-overflow-policy? (:overflow value))
         (expression? (:argument value)))
    (record-kind? "raster.compiler.ir.kernel_body.IndexExpr" value)
    (and (contains? index-ops (:op value))
         (seq (:arguments value))
         (every? expression? (:arguments value)))
    :else false))

(defn- expression-references [value]
  (cond
    (value-id? value) #{value}
    (integer? value) #{}
    (record-kind? "raster.compiler.ir.kernel_body.IndexCast" value)
    (expression-references (:argument value))
    (record-kind? "raster.compiler.ir.kernel_body.IndexExpr" value)
    (reduce into #{} (map expression-references (:arguments value)))
    :else #{}))

(defn- gcd-static
  [left right]
  (loop [left (abs (long left)) right (abs (long right))]
    (if (zero? right) left (recur right (mod left right)))))

(defn- expression-divisible?
  "Conservative proof that every realization of an index expression is divisible by divisor."
  [value divisor]
  (cond
    (= 1 divisor) true
    (integer? value) (zero? (mod value divisor))
    (value-id? value) false
    (not (record-kind? "raster.compiler.ir.kernel_body.IndexExpr" value)) false
    (contains? #{:add :sub} (:op value))
    (every? #(expression-divisible? % divisor) (:arguments value))
    (= :mul (:op value))
    (some #(expression-divisible? % divisor) (:arguments value))
    :else false))

(defn- storage-offset-aligned?
  [storage coordinates alignment]
  (let [element-bytes (dtype/bytes-of (:dtype storage))
        strides (layout/resolve-strides (:layout storage))
        terms (map vector coordinates strides)
        aligned-term?
        (fn [[coordinate stride]]
          (and (integer? stride)
               (let [coefficient (* element-bytes stride)
                     remaining (quot alignment (gcd-static coefficient alignment))]
                 (expression-divisible? coordinate remaining))))
        view-offset (some-> storage :view :element-offset)]
    (and (every? aligned-term? terms)
         (or (nil? view-offset)
             (expression-divisible?
              view-offset (quot alignment (gcd-static element-bytes alignment)))))))

(defn- predicate? [value]
  (and (record-kind? "raster.compiler.ir.kernel_body.Predicate" value)
       (contains? predicate-ops (:op value))
       (seq (:arguments value))
       (every? #(or (expression? %) (predicate? %)) (:arguments value))))

(defn- predicate-references [value]
  (reduce into #{}
          (map #(if (record-kind? "raster.compiler.ir.kernel_body.Predicate" %)
                  (predicate-references %)
                  (expression-references %))
               (:arguments value))))

(defn- scalar-type?
  [type]
  (or (contains? internal-types type) (dtype/known? type)))

(defn- canonical-type
  [type]
  (if (contains? internal-types type) type (dtype/canon type)))

(defn- value-spec!
  [owner spec]
  (when-not (and (record-kind? "raster.compiler.ir.kernel_body.ValueSpec" spec)
                 (value-id? (:id spec))
                 (scalar-type? (:type spec)))
    (throw (ex-info (str owner " requires a typed ValueSpec") {:value spec})))
  spec)

(defn- literal!
  [literal]
  (let [type (when (scalar-type? (:type literal)) (canonical-type (:type literal)))
        value (:value literal)
        valid-value?
        (case type
          :predicate (instance? Boolean value)
          :byte (and (integer? value) (<= Byte/MIN_VALUE value Byte/MAX_VALUE))
          :int (and (integer? value) (<= Integer/MIN_VALUE value Integer/MAX_VALUE))
          :long (and (integer? value) (<= Long/MIN_VALUE value Long/MAX_VALUE))
          (:half :float :double) (number? value)
          false)]
    (when-not (and (record-kind? "raster.compiler.ir.kernel_body.Literal" literal)
                   type valid-value?)
      (throw (ex-info "kernel literal value and type disagree" {:literal literal})))
    literal))

(defn- launch-bound-value
  [value]
  (if (record-kind? "raster.compiler.ir.kernel_launch.RuntimeValue" value)
    (:value value)
    value))

(defn- collective-association?
  [association width]
  (or (= :implementation-defined association)
      (and (map? association)
           (= :shuffle-down-tree (:kind association))
           (zero? (bit-and width (dec width)))
           (= (vec (:distances association))
              (vec (take-while pos? (iterate #(quot % 2) (quot width 2))))))))

(defn- shape! [owner shape]
  (when-not (and (vector? shape) (seq shape)
                 (every? #(or (value-id? %)
                              (and (integer? %) (pos? %))
                              (launch/dimension-expression? %)
                              (and (record-kind?
                                    "raster.compiler.ir.kernel_body.IndexExpr" %)
                                   (expression? %)))
                         shape))
    (throw (ex-info (str owner " requires a non-empty shape of positive extents")
                    {:shape shape}))))

(defn- layout! [owner layout]
  (when-not (and (map? layout) (keyword? (:kind layout)))
    (throw (ex-info (str owner " requires a named layout") {:layout layout})))
  (when (layout/shared-memory-layout? layout)
    (layout/validate-shared-memory! layout)))

(defn- storage-layout!
  "Validate the shape/dtype facts repeated by dense storage layouts.

  Fragment layouts deliberately do not all carry a tensor shape, but every layout that does state
  one must agree with its owner.  This prevents a target emitter from indexing according to a
  stale layout descriptor after the storage contract changed."
  [owner shape dtype storage-layout]
  (layout! owner storage-layout)
  (when (and (contains? storage-layout :shape)
             (not= (vec shape) (vec (:shape storage-layout))))
    (throw (ex-info (str owner " shape disagrees with its layout")
                    {:reason :kernel-body-layout-shape
                     :shape shape :layout storage-layout})))
  (when (and (contains? storage-layout :rank)
             (not= (count shape) (:rank storage-layout)))
    (throw (ex-info (str owner " rank disagrees with its layout")
                    {:reason :kernel-body-layout-rank
                     :shape shape :layout storage-layout})))
  (when (and (contains? storage-layout :dtype)
             (not= (dtype/canon dtype) (dtype/canon (:dtype storage-layout))))
    (throw (ex-info (str owner " dtype disagrees with its layout")
                    {:reason :kernel-body-layout-dtype
                     :dtype dtype :layout storage-layout}))))

(defn- unique-ids! [owner values]
  (let [ids (mapv :id values)]
    (when (or (some nil? ids) (not= (count ids) (count (set ids))))
      (throw (ex-info (str owner " identities must be non-nil and unique") {:ids ids})))))

(defn- operation? [value]
  (contains? #{"raster.compiler.ir.kernel_body.FragmentInit"
               "raster.compiler.ir.kernel_body.TileLoad"
               "raster.compiler.ir.kernel_body.TilePrefetch"
               "raster.compiler.ir.kernel_body.MatrixMad"
               "raster.compiler.ir.kernel_body.Loop"
               "raster.compiler.ir.kernel_body.Guard"
               "raster.compiler.ir.kernel_body.TileStore"
               "raster.compiler.ir.kernel_body.IndexCompute"
               "raster.compiler.ir.kernel_body.ScalarCompute"
               "raster.compiler.ir.kernel_body.ScalarLoad"
               "raster.compiler.ir.kernel_body.ScalarStore"
               "raster.compiler.ir.kernel_body.AtomicRMW"
               "raster.compiler.ir.kernel_body.Yield"
               "raster.compiler.ir.kernel_body.IfRegion"
               "raster.compiler.ir.kernel_body.ForLoop"
               "raster.compiler.ir.kernel_body.PipelineYield"
               "raster.compiler.ir.kernel_body.PipelinedFor"
               "raster.compiler.ir.kernel_body.Collective"
               "raster.compiler.ir.kernel_body.WorkgroupBarrier"
               "raster.compiler.ir.kernel_body.AsyncWorkgroupCopy"
               "raster.compiler.ir.kernel_body.AsyncCommit"
               "raster.compiler.ir.kernel_body.AsyncWait"}
             (some-> value class .getName)))

(declare validate-operations!)

(defn- axis-map!
  [operand]
  (let [groups (get-in operand [:map :groups])]
    (when-not (and (vector? groups) (seq groups)
                   (every? #(and (vector? %) (seq %)
                                 (every? (fn [pair]
                                           (and (vector? pair) (= 2 (count pair))
                                                (value-id? (first pair))
                                                (or (value-id? (second pair))
                                                    (and (integer? (second pair))
                                                         (pos? (second pair))))))
                                         %))
                           groups))
      (throw (ex-info "scalar-region operand requires a structured axis map"
                      {:operand operand})))
    (axis-map/shape (:map operand))))

(defn- validate-scalar-region-boundary!
  [region storage epilogue-abi operand-roles scalar-roles]
  (let [parameters (:parameters region)
        operands (:operands region)
        operand-ids (when (vector? operands) (mapv :sym operands))]
    (when-not (and (vector? parameters) (seq parameters)
                   (every? symbol? parameters)
                   (= (count parameters) (count (set parameters)))
                   (vector? operands)
                   (<= (inc (count operand-ids)) (count parameters)))
      (throw (ex-info "tile-store scalar region is incomplete or has ambiguous parameters"
                      {:region region})))
    (let [accumulator (first parameters)
          scalar-ids (subvec parameters (inc (count operand-ids)))
          external-ids (vec (rest parameters))
          missing-external-ids (filterv #(not (contains? storage %)) external-ids)
          region-epilogue-ids
          (filterv #(= :epilogue (:role (get storage %))) external-ids)]
      (when-not (and (= operand-ids (subvec parameters 1 (inc (count operand-ids))))
                     (= (count operand-ids) (count (set operand-ids))))
        (throw (ex-info "scalar-region operands must be an ordered prefix of its ABI parameters"
                        {:region region :operand-ids operand-ids})))
      (when (contains? storage accumulator)
        (throw (ex-info "scalar-region accumulator must be region-local"
                        {:accumulator accumulator})))
      (when (seq missing-external-ids)
        (throw (ex-info "scalar-region parameters and the epilogue ABI disagree"
                        {:parameters external-ids
                         :missing-parameters missing-external-ids
                         :epilogue-abi epilogue-abi})))
      (when-not (= epilogue-abi region-epilogue-ids)
        (throw (ex-info "scalar-region parameters and the epilogue ABI disagree"
                        {:parameters external-ids
                         :region-epilogue-parameters region-epilogue-ids
                         :epilogue-abi epilogue-abi})))
      (doseq [operand operands]
        (let [id (:sym operand)
              parameter (get storage id)
              shape (axis-map! operand)]
          (when-not (and parameter (= :input (:kind parameter))
                         (contains? operand-roles (:role parameter))
                         (= (dtype/canon (:dtype operand :float))
                            (dtype/canon (:dtype parameter)))
                         (= shape (:shape parameter)))
            (throw (ex-info "scalar-region operand and its kernel ABI slot disagree"
                            {:operand operand :parameter parameter :shape shape})))))
      (doseq [id scalar-ids]
        (let [parameter (get storage id)]
          (when-not (and parameter (= :scalar (:kind parameter))
                         (contains? scalar-roles (:role parameter)))
            (throw (ex-info "scalar-region scalar and its kernel ABI slot disagree"
                            {:scalar id :parameter parameter})))))
      region)))

(defn- validate-scalar-region!
  [region storage epilogue-abi]
  (when-not (record-kind? "raster.compiler.ir.kernel_body.ScalarRegion" region)
    (throw (ex-info "tile-store value region must be a ScalarRegion" {:region region})))
  (when-not (and (some? (:expression region))
                 (dtype/known? (:result-dtype region)))
    (throw (ex-info "tile-store scalar region is incomplete or has ambiguous parameters"
                    {:region region})))
  (validate-scalar-region-boundary! region storage epilogue-abi #{:epilogue} #{:epilogue})
  (let [legal (scalar-region-legal? region)]
    (when-not (:ok legal)
      (throw (ex-info "tile-store scalar region is not store-local"
                      (assoc legal :region region)))))
  region)

(defn- validate-scalar-ssa-region!
  [region storage masks scope epilogue-abi]
  (when-not (record-kind? "raster.compiler.ir.kernel_body.ScalarSSARegion" region)
    (throw (ex-info "tile-store value region must be typed scalar SSA" {:region region})))
  (let [parameters (:parameters region)
        indices (:indices region)
        operations (:operations region)]
    (validate-scalar-region-boundary!
     region storage epilogue-abi #{:operand :lhs :rhs :epilogue} #{:parameter :epilogue})
    (when-not (and (vector? indices) (every? symbol? indices)
                   (= (count indices) (count (set indices)))
                   (not-any? (set parameters) indices)
                   (dtype/known? (:accumulator-dtype region))
                   (vector? operations)
                   (every? #(contains? #{"raster.compiler.ir.kernel_body.ScalarCompute"
                                         "raster.compiler.ir.kernel_body.ScalarLoad"}
                                       (some-> % class .getName))
                           operations)
                   (value-id? (:result region))
                   (dtype/known? (:result-dtype region)))
      (throw (ex-info "tile-store scalar SSA region is incomplete"
                      {:reason :kernel-body-scalar-ssa-region :region region})))
    (validate-operations! operations storage {} masks
                          (into scope (concat parameters indices)) epilogue-abi)
    region))

(defn- validate-operation!
  [operation storage fragments masks scope epilogue-abi]
  (let [parameter (fn [id]
                    (or (get storage id)
                        (throw (ex-info "kernel operation references an undeclared parameter"
                                        {:parameter id :operation operation}))))
        fragment (fn [id]
                   (or (get fragments id)
                       (throw (ex-info "kernel operation references an undeclared fragment"
                                       {:fragment id :operation operation}))))
        mask (fn [id]
               (when id
                 (let [m (or (get masks id)
                             (throw (ex-info "kernel operation references an undeclared mask"
                                             {:mask id :operation operation})))
                       outside (remove scope (mapcat predicate-references (:predicates m)))]
                   (when (seq outside)
                     (throw (ex-info "kernel mask references values outside the operation scope"
                                     {:mask id :references (vec outside)
                                      :scope scope :operation operation}))))))
        coordinates-syntax! (fn [owner coordinates]
                              (when-not (and (vector? coordinates)
                                             (every? expression? coordinates))
                                (throw (ex-info
                                        (str owner " coordinates must use explicit index expressions")
                                        {:coordinates coordinates}))))
        coordinates! (fn [owner coordinates]
                       (coordinates-syntax! owner coordinates)
                       (let [outside (remove scope (mapcat expression-references coordinates))]
                         (when (seq outside)
                           (throw (ex-info (str owner " coordinates reference values outside the operation scope")
                                           {:coordinates coordinates :references (vec outside)
                                            :scope scope})))))]
    (cond
      (record-kind? "raster.compiler.ir.kernel_body.FragmentInit" operation)
      (let [f (fragment (:fragment operation))]
        (when-not (= :mma-frag (get-in f [:layout :kind]))
          (throw (ex-info "only accumulator fragments may be initialized"
                          {:fragment f :operation operation}))))

      (record-kind? "raster.compiler.ir.kernel_body.TileLoad" operation)
      (let [f (fragment (:fragment operation))
            p (parameter (:buffer operation))]
        (when-not (= :input (:kind p))
          (throw (ex-info "tile loads require an input buffer" {:buffer p})))
        (when-not (= :dot-operand (get-in f [:layout :kind]))
          (throw (ex-info "tile loads must produce a named dot-operand fragment"
                          {:fragment f})))
        (coordinates! "tile-load" (:coordinates operation))
        (when-not (= (count (:shape p)) (count (:coordinates operation)))
          (throw (ex-info "tile-load coordinates must match the buffer rank"
                          {:reason :kernel-body-tile-load-rank
                           :buffer p :coordinates (:coordinates operation)})))
        (when-not (= (dtype/canon (:dtype p)) (dtype/canon (:dtype f)))
          (throw (ex-info "tile-load fragment dtype must equal the buffer element dtype"
                          {:reason :kernel-body-tile-load-dtype
                           :buffer p :fragment f})))
        (mask (:mask operation))
        (when-not (contains? cache-policies (:cache operation))
          (throw (ex-info "tile load has an unsupported cache policy"
                          {:cache (:cache operation)}))))

      (record-kind? "raster.compiler.ir.kernel_body.TilePrefetch" operation)
      (let [p (parameter (:buffer operation))]
        (when-not (= :input (:kind p))
          (throw (ex-info "tile prefetch requires an input buffer" {:buffer p})))
        (shape! "tile prefetch" (:shape operation))
        (layout! "tile prefetch" (:layout operation))
        (coordinates! "tile-prefetch" (:coordinates operation))
        (when-not (= (count (:shape p)) (count (:coordinates operation)))
          (throw (ex-info "tile-prefetch coordinates must match the buffer rank"
                          {:reason :kernel-body-tile-prefetch-rank
                           :buffer p :coordinates (:coordinates operation)})))
        (when-not (= (dtype/canon (:dtype p))
                     (dtype/canon (get-in operation [:layout :dtype])))
          (throw (ex-info "tile-prefetch layout dtype must equal the buffer element dtype"
                          {:reason :kernel-body-tile-prefetch-dtype
                           :buffer p :layout (:layout operation)})))
        (when-not (and (integer? (:distance operation)) (not (neg? (:distance operation))))
          (throw (ex-info "tile-prefetch distance must be a non-negative integer"
                          {:distance (:distance operation)})))
        (mask (:mask operation)))

      (record-kind? "raster.compiler.ir.kernel_body.MatrixMad" operation)
      (let [acc (fragment (:accumulator operation))
            lhs (fragment (:lhs operation))
            rhs (fragment (:rhs operation))
            acc-layout (:layout acc)
            lhs-layout (:layout lhs)
            rhs-layout (:layout rhs)
            matrix (:instruction operation)]
        (when-not (and (= :mma-frag (:kind acc-layout))
                       (= :dot-operand (:kind lhs-layout))
                       (= :dot-operand (:kind rhs-layout))
                       (= 0 (:op-idx lhs-layout))
                       (= 1 (:op-idx rhs-layout))
                       (= (:parent lhs-layout) acc-layout)
                       (= (:parent rhs-layout) acc-layout)
                       (= (:matrix acc-layout) matrix))
          (throw (ex-info "matrix MAD fragment layouts do not agree with its instruction"
                          {:accumulator acc :lhs lhs :rhs rhs :instruction matrix})))
        (let [{matrix-m :m matrix-n :n matrix-k :k} matrix]
          (when-not (= [[matrix-m matrix-n] [matrix-m matrix-k] [matrix-k matrix-n]]
                       [(:shape acc) (:shape lhs) (:shape rhs)])
            (throw (ex-info "matrix MAD fragment shapes do not agree with its instruction"
                            {:reason :kernel-body-matrix-fragment-shape
                             :accumulator acc :lhs lhs :rhs rhs :instruction matrix}))))
        (when-not (and (= (dtype/canon (:dtype acc))
                          (dtype/canon (:dtype acc-layout)))
                       (= (dtype/canon (:dtype lhs))
                          (dtype/canon (:dtype lhs-layout)))
                       (= (dtype/canon (:dtype rhs))
                          (dtype/canon (:dtype rhs-layout))))
          (throw (ex-info "matrix MAD fragment dtypes do not agree with their layouts"
                          {:reason :kernel-body-matrix-fragment-dtype
                           :accumulator acc :lhs lhs :rhs rhs :instruction matrix}))))

      (record-kind? "raster.compiler.ir.kernel_body.Loop" operation)
      (do
        (when-not (symbol? (:index operation))
          (throw (ex-info "kernel loop requires a symbolic induction value"
                          {:index (:index operation)})))
        (when-not (and (expression? (:lower operation)) (expression? (:upper operation))
                       (integer? (:step operation)) (pos? (:step operation)))
          (throw (ex-info "kernel loop requires explicit bounds and a positive static step"
                          {:loop operation})))
        (when-not (map? (:attributes operation))
          (throw (ex-info "kernel loop attributes must be a map" {:loop operation})))
        (let [outside (remove scope
                              (into (expression-references (:lower operation))
                                    (expression-references (:upper operation))))]
          (when (seq outside)
            (throw (ex-info "kernel loop bounds reference values outside their scope"
                            {:loop operation :references (vec outside) :scope scope}))))
        (when (contains? scope (:index operation))
          (throw (ex-info "kernel loop induction value shadows an existing value"
                          {:index (:index operation) :scope scope})))
        (validate-operations! (:operations operation) storage fragments masks
                              (conj scope (:index operation)) epilogue-abi))

      (record-kind? "raster.compiler.ir.kernel_body.Guard" operation)
      (do
        (mask (:mask operation))
        (validate-operations! (:operations operation) storage fragments masks scope epilogue-abi))

      (record-kind? "raster.compiler.ir.kernel_body.TileStore" operation)
      (let [p (parameter (:buffer operation))
            f (fragment (:fragment operation))]
        (when-not (= :output (:kind p))
          (throw (ex-info "tile stores require an output buffer" {:buffer p})))
        (when-not (= :mma-frag (get-in f [:layout :kind]))
          (throw (ex-info "tile stores require an accumulator fragment" {:fragment f})))
        (coordinates! "tile-store" (:coordinates operation))
        (when-not (= (count (:shape p)) (count (:coordinates operation)))
          (throw (ex-info "tile-store coordinates must match the buffer rank"
                          {:reason :kernel-body-tile-store-rank
                           :buffer p :coordinates (:coordinates operation)})))
        (mask (:mask operation))
        (when-let [region (:value-region operation)]
          (if (record-kind? "raster.compiler.ir.kernel_body.ScalarSSARegion" region)
            (validate-scalar-ssa-region! region storage masks scope epilogue-abi)
            (validate-scalar-region! region storage epilogue-abi))))

      (record-kind? "raster.compiler.ir.kernel_body.ScalarCompute" operation)
      (do
        (value-spec! "scalar compute result" (:result operation))
        (when-not (record-kind? "raster.compiler.ir.kernel_body.ScalarExpr"
                                (:expression operation))
          (throw (ex-info "scalar compute requires an explicit ScalarExpr"
                          {:operation operation}))))

      ;; Index computations are legal both in the kernel's index declaration section and
      ;; lexically inside scalar control regions.  The latter is needed when a grid-stride loop
      ;; recomputes a decomposed logical coordinate on each iteration.  Keeping this as explicit
      ;; index IR prevents schedule arithmetic from being smuggled into a scalar expression.
      (record-kind? "raster.compiler.ir.kernel_body.IndexCompute" operation)
      (when-not (and (value-id? (:id operation)) (expression? (:expression operation)))
        (throw (ex-info "kernel index compute requires a named explicit index expression"
                        {:reason :kernel-body-index-compute :operation operation})))

      (record-kind? "raster.compiler.ir.kernel_body.ScalarLoad" operation)
      (let [p (parameter (:buffer operation))]
        (value-spec! "scalar load result" (:result operation))
        (when (= :scalar (:kind p))
          (throw (ex-info "scalar loads require buffer storage" {:buffer p})))
        (when-not (and (vector? (:coordinates operation))
                       (every? expression? (:coordinates operation)))
          (throw (ex-info "scalar-load coordinates must use explicit index expressions"
                          {:coordinates (:coordinates operation)})))
        (when-not (= (count (:shape p)) (count (:coordinates operation)))
          (throw (ex-info "scalar-load coordinates must match the buffer rank"
                          {:buffer (:buffer operation) :shape (:shape p)
                           :coordinates (:coordinates operation)})))
        (mask (:predicate operation))
        (when (and (:predicate operation) (nil? (:other operation)))
          (throw (ex-info "masked scalar load requires an explicit other value"
                          {:operation operation})))
        (when (and (nil? (:predicate operation)) (some? (:other operation)))
          (throw (ex-info "unmasked scalar load cannot carry an ignored other value"
                          {:operation operation})))
        (when-not (contains? cache-policies (:cache operation))
          (throw (ex-info "scalar load has an unsupported cache policy"
                          {:cache (:cache operation)}))))

      (record-kind? "raster.compiler.ir.kernel_body.ScalarStore" operation)
      (let [p (parameter (:buffer operation))]
        (when (contains? #{:input :scalar} (:kind p))
          (throw (ex-info "scalar stores require writable output storage" {:buffer p})))
        (when-not (and (vector? (:coordinates operation))
                       (every? expression? (:coordinates operation)))
          (throw (ex-info "scalar-store coordinates must use explicit index expressions"
                          {:coordinates (:coordinates operation)})))
        (when-not (= (count (:shape p)) (count (:coordinates operation)))
          (throw (ex-info "scalar-store coordinates must match the buffer rank"
                          {:buffer (:buffer operation) :shape (:shape p)
                           :coordinates (:coordinates operation)})))
        (mask (:predicate operation)))

      (record-kind? "raster.compiler.ir.kernel_body.AtomicRMW" operation)
      (let [p (parameter (:buffer operation))]
        (when-not (= :inout (:kind p))
          (throw (ex-info "atomic updates require read-write kernel storage"
                          {:reason :kernel-body-atomic-storage :buffer p})))
        (when-not (= :+ (:operator operation))
          (throw (ex-info "KernelBody atomic update has an unsupported operator"
                          {:reason :kernel-body-atomic-operator
                           :operator (:operator operation)})))
        (when-not (and (vector? (:coordinates operation))
                       (every? expression? (:coordinates operation)))
          (throw (ex-info "atomic coordinates must use explicit index expressions"
                          {:coordinates (:coordinates operation)})))
        (when-not (= (count (:shape p)) (count (:coordinates operation)))
          (throw (ex-info "atomic coordinates must match the buffer rank"
                          {:buffer (:buffer operation) :shape (:shape p)
                           :coordinates (:coordinates operation)})))
        (mask (:predicate operation)))

      (record-kind? "raster.compiler.ir.kernel_body.IfRegion" operation)
      (do
        (when-not (and (value-id? (:condition operation))
                       (vector? (:then-operations operation))
                       (vector? (:else-operations operation))
                       (vector? (:results operation)))
          (throw (ex-info "kernel if region is incomplete" {:operation operation})))
        (doseq [result (:results operation)] (value-spec! "if result" result))
        (validate-operations! (:then-operations operation) storage fragments masks scope epilogue-abi)
        (validate-operations! (:else-operations operation) storage fragments masks scope epilogue-abi))

      (record-kind? "raster.compiler.ir.kernel_body.ForLoop" operation)
      (let [index (:index operation)]
        (value-spec! "kernel for-loop index" index)
        (when-not (contains? #{:int :long} (canonical-type (:type index)))
          (throw (ex-info "kernel for-loop index must have an integral dtype"
                          {:index index})))
        (when-not (and (expression? (:lower operation)) (expression? (:upper operation))
                       (expression? (:step operation))
                       (or (not (integer? (:step operation))) (pos? (:step operation)))
                       (vector? (:iter-args operation))
                       (every? #(record-kind? "raster.compiler.ir.kernel_body.LoopArg" %)
                               (:iter-args operation))
                       (vector? (:results operation))
                       (= (count (:iter-args operation)) (count (:results operation)))
                       (map? (:attributes operation)))
          (throw (ex-info "kernel for-loop requires typed carried values and explicit bounds"
                          {:loop operation})))
        (let [outside (remove scope (expression-references (:step operation)))]
          (when (seq outside)
            (throw (ex-info "kernel for-loop step references values outside its invariant scope"
                            {:reason :kernel-body-loop-step-scope
                             :step (:step operation) :references (vec outside)
                             :scope scope}))))
        (doseq [arg (:iter-args operation)]
          (value-spec! "loop carried binding" (:binding arg)))
        (let [uniform-iter-args (get-in operation [:attributes :uniform-iter-args] #{})
              bindings (set (map (comp :id :binding) (:iter-args operation)))]
          (when-not (and (set? uniform-iter-args)
                         (set/subset? uniform-iter-args bindings))
            (throw (ex-info
                    "kernel loop uniform-iter-args must name carried bindings"
                    {:reason :kernel-body-loop-uniform-iter-args
                     :uniform-iter-args uniform-iter-args
                     :bindings bindings}))))
        (doseq [result (:results operation)] (value-spec! "loop result" result))
        (when (contains? scope (get-in operation [:index :id]))
          (throw (ex-info "kernel for-loop induction value shadows an existing value"
                          {:index index :scope scope})))
        (validate-operations! (:operations operation) storage fragments masks
                              (conj scope (:id index)) epilogue-abi))

      (record-kind? "raster.compiler.ir.kernel_body.PipelinedFor" operation)
      (let [index (:index operation)
            async-args (:async-iter-args operation)
            async-results (:async-results operation)
            tail-policy (get-in operation [:attributes :tail-policy])]
        (value-spec! "kernel pipelined-for index" index)
        (when-not (contains? #{:int :long} (canonical-type (:type index)))
          (throw (ex-info "kernel pipelined-for index must have an integral dtype"
                          {:index index})))
        (when-not (and (expression? (:lower operation)) (expression? (:upper operation))
                       (integer? (:step operation)) (pos? (:step operation))
                       (vector? (:iter-args operation))
                       (every? #(record-kind? "raster.compiler.ir.kernel_body.LoopArg" %)
                               (:iter-args operation))
                       (vector? async-args) (seq async-args)
                       (every? #(record-kind?
                                 "raster.compiler.ir.kernel_body.AsyncLoopArg" %)
                               async-args)
                       (every? #(and (value-id? (:binding %)) (value-id? (:initial %)))
                               async-args)
                       (= (count async-args) (count async-results))
                       (vector? async-results) (every? value-id? async-results)
                       (= (count async-results) (count (set async-results)))
                       (vector? (:results operation))
                       (= (count (:iter-args operation)) (count (:results operation)))
                       (map? (:attributes operation))
                       (contains? pipeline-tail-policies tail-policy))
          (throw (ex-info
                  "kernel pipelined-for requires scalar and async carries with an explicit tail policy"
                  {:reason :kernel-body-pipelined-for :loop operation
                   :supported-tail-policies pipeline-tail-policies})))
        (doseq [arg (:iter-args operation)]
          (value-spec! "pipelined loop carried binding" (:binding arg)))
        (let [uniform-iter-args (get-in operation [:attributes :uniform-iter-args] #{})
              bindings (set (map (comp :id :binding) (:iter-args operation)))]
          (when-not (and (set? uniform-iter-args)
                         (set/subset? uniform-iter-args bindings))
            (throw (ex-info
                    "kernel pipelined-for uniform-iter-args must name scalar carried bindings"
                    {:reason :kernel-body-loop-uniform-iter-args
                     :uniform-iter-args uniform-iter-args :bindings bindings}))))
        (doseq [result (:results operation)]
          (value-spec! "pipelined loop result" result))
        (when (contains? scope (:id index))
          (throw (ex-info "kernel pipelined-for induction value shadows an existing value"
                          {:index index :scope scope})))
        (validate-operations! (:operations operation) storage fragments masks
                              (conj scope (:id index)) epilogue-abi))

      (record-kind? "raster.compiler.ir.kernel_body.Yield" operation)
      (when-not (vector? (:values operation))
        (throw (ex-info "kernel region yield values must be an ordered vector"
                        {:operation operation})))

      (record-kind? "raster.compiler.ir.kernel_body.PipelineYield" operation)
      (when-not (and (vector? (:values operation))
                     (vector? (:groups operation)) (seq (:groups operation))
                     (every? value-id? (:groups operation))
                     (= (count (:groups operation)) (count (set (:groups operation)))))
        (throw (ex-info "pipelined region yield requires scalar values and async groups"
                        {:reason :kernel-body-pipeline-yield :operation operation})))

      (record-kind? "raster.compiler.ir.kernel_body.Collective" operation)
      (do
        (value-spec! "collective result" (:result operation))
        (when-not (and (contains? collective-kinds (:kind operation))
                       (= :subgroup (:scope operation))
                       (integer? (:width operation)) (pos? (:width operation))
                       (record-kind? "raster.compiler.ir.kernel_body.Participation"
                                     (:participation operation))
                       (= :full (get-in operation [:participation :kind])))
          (throw (ex-info "kernel collective has an unsupported execution contract"
                          {:operation operation})))
        (case (:kind operation)
          :reduce
          (when-not (and (keyword? (:operator operation))
                         (nil? (:source-lane operation))
                         (collective-association? (:association operation)
                                                  (:width operation)))
            (throw (ex-info "subgroup reduction requires an operator and no source lane"
                            {:operation operation})))
          :broadcast
          (when-not (and (nil? (:operator operation))
                         (nil? (:association operation))
                         (expression? (:source-lane operation)))
            (throw (ex-info "subgroup broadcast requires a source lane and no reduction association"
                            {:operation operation})))))

      (record-kind? "raster.compiler.ir.kernel_body.WorkgroupBarrier" operation)
      (when-not (and (= :workgroup (:scope operation))
                     (set? (:memory-spaces operation))
                     (seq (:memory-spaces operation))
                     (set/subset? (:memory-spaces operation) workgroup-memory-spaces)
                     (contains? barrier-semantics (:semantics operation))
                     (record-kind? "raster.compiler.ir.kernel_body.Participation"
                                   (:participation operation))
                     (= :full (get-in operation [:participation :kind])))
        (throw (ex-info "workgroup barrier has an unsupported synchronization contract"
                        {:reason :kernel-body-workgroup-barrier :operation operation})))

      (record-kind? "raster.compiler.ir.kernel_body.AsyncWorkgroupCopy" operation)
      (let [source (parameter (:source operation))
            destination (parameter (:destination operation))
            elements (:elements operation)
            transfer-bytes (:transfer-bytes operation)]
        ;; Async-copy coordinates are scalar SSA expressions, unlike the legacy tile vocabulary
        ;; whose coordinates are restricted to kernel index bindings.  Their references and
        ;; workgroup uniformity are checked by the typed SSA verifier below.
        (coordinates-syntax! "async-copy source" (:source-coordinates operation))
        (coordinates-syntax! "async-copy destination" (:destination-coordinates operation))
        (when-not (and (value-id? (:id operation))
                       (= :input (:kind source))
                       (= :global (:memory-space source))
                       (= :allocation (:kind destination))
                       (= :workgroup (:memory-space destination))
                       ;; A non-identity destination is a logical row transfer, not a physically
                       ;; contiguous copy. It is legal only as preferred overlap: the C-family
                       ;; boundary then emits a cooperative layout-aware scatter. Required overlap
                       ;; remains reserved for a native tensor/scatter-copy operation.
                       (or (not (layout/shared-memory-layout? (:layout destination)))
                           (= :identity (get-in destination [:layout :swizzle]))
                           (and (= :preferred (:overlap operation))
                                (= 2 (get-in destination [:layout :rank]))
                                (= 2 (count (:destination-coordinates operation)))
                                (let [[row column] (:destination-coordinates operation)
                                      [rows columns] (:shape destination)]
                                  (and (nat-int? row) (< row rows)
                                       (nat-int? column)
                                       (<= (+ column elements) columns)))))
                       (= (dtype/canon (:dtype source)) (dtype/canon (:dtype destination)))
                       (= (count (:shape source)) (count (:source-coordinates operation)))
                       (= (count (:shape destination))
                          (count (:destination-coordinates operation)))
                       (pos-int? elements)
                       (contains? async-transfer-widths transfer-bytes)
                       (zero? (mod (* elements (dtype/bytes-of (:dtype source))) transfer-bytes))
                       (contains? cache-policies (:cache operation))
                       (contains? async-overlap-policies (:overlap operation))
                       (or (= :preferred (:overlap operation))
                           (and (>= (:alignment destination) transfer-bytes)
                                (storage-offset-aligned?
                                 source (:source-coordinates operation) transfer-bytes)
                                (storage-offset-aligned?
                                 destination (:destination-coordinates operation)
                                 transfer-bytes)))
                       (record-kind? "raster.compiler.ir.kernel_body.Participation"
                                     (:participation operation))
                       (= :full (get-in operation [:participation :kind])))
          (throw (ex-info
                  "async workgroup copy requires a typed contiguous global-to-workgroup transfer"
                  {:reason :kernel-body-async-copy :operation operation
                   :source source :destination destination}))))

      (record-kind? "raster.compiler.ir.kernel_body.AsyncCommit" operation)
      (when-not (and (value-id? (:id operation))
                     (vector? (:copies operation)) (seq (:copies operation))
                     (every? value-id? (:copies operation))
                     (= (count (:copies operation)) (count (set (:copies operation)))))
        (throw (ex-info "async commit requires a named ordered set of issued copies"
                        {:reason :kernel-body-async-commit :operation operation})))

      (record-kind? "raster.compiler.ir.kernel_body.AsyncWait" operation)
      (when-not (and (vector? (:groups operation)) (seq (:groups operation))
                     (every? value-id? (:groups operation))
                     (= (count (:groups operation)) (count (set (:groups operation))))
                     (nat-int? (:pending-groups operation))
                     (contains? async-wait-semantics (:semantics operation))
                     (record-kind? "raster.compiler.ir.kernel_body.Participation"
                                   (:participation operation))
                     (= :full (get-in operation [:participation :kind])))
        (throw (ex-info "async wait requires ordered groups and full workgroup participation"
                        {:reason :kernel-body-async-wait :operation operation})))

      :else
      (throw (ex-info "kernel body contains an unsupported operation"
                      {:operation operation :actual (type operation)})))))

(defn- validate-operations! [operations storage fragments masks scope epilogue-abi]
  (when-not (vector? operations)
    (throw (ex-info "kernel operations must be an ordered vector" {:operations operations})))
  ;; A nested IndexCompute is a lexical index binding, rather than a kernel-wide declaration.
  ;; Extend the structural coordinate scope in source order so a following load/store may address
  ;; through it; SSA dataflow independently rejects duplicate or forward uses.
  (loop [remaining operations scope scope]
    (when-let [operation (first remaining)]
      (when-not (operation? operation)
        (throw (ex-info "kernel body contains a non-operation value" {:operation operation})))
      (validate-operation! operation storage fragments masks scope epilogue-abi)
      (recur (next remaining)
             (if (record-kind? "raster.compiler.ir.kernel_body.IndexCompute" operation)
               (conj scope (:id operation))
               scope)))))

;; ---------------------------------------------------------------------------
;; Typed SSA and convergence verification for the general scalar vocabulary.
;; ---------------------------------------------------------------------------

(def ^:private all-uniform #{:workgroup :subgroup})
(def ^:private subgroup-uniform #{:subgroup})
(def ^:private lane-varying #{})

(defn- join-uniformity
  [infos]
  (if (seq infos)
    (reduce set/intersection all-uniform (map :uniformity infos))
    all-uniform))

(defn- typed-info!
  [values id owner]
  (or (get values id)
      (throw (ex-info (str owner " references a value before it is defined")
                      {:reason :kernel-body-use-before-definition :value id}))))

(declare scalar-info!)

(defn- literal-info!
  [literal]
  (literal! literal)
  (let [type (canonical-type (:type literal))]
    {:type type
     :uniformity all-uniform
     :range (scalar-range/literal (:value literal) type)}))

(defn- scalar-argument-info!
  [argument values]
  (cond
    (record-kind? "raster.compiler.ir.kernel_body.Literal" argument)
    (literal-info! argument)

    (record-kind? "raster.compiler.ir.kernel_body.ScalarExpr" argument)
    (scalar-info! argument values)

    (value-id? argument)
    (typed-info! values argument "scalar expression")

    :else
    (throw (ex-info "scalar expression operands must be typed values, literals, or expressions"
                    {:argument argument}))))

(defn- same-types!
  [owner infos]
  (let [types (mapv :type infos)]
    (when-not (apply = types)
      (throw (ex-info (str owner " operand types must agree") {:types types})))
    (first types)))

(defn- scalar-result-range
  "Derive a range from verifier-owned operand facts.  This deliberately knows only canonical
  scalar semantics; producer-supplied proof maps are diagnostic evidence, never facts."
  [canonical-op result-type infos]
  (let [ranges (mapv :range infos)]
    (cond
      (contains? #{:+ :- :*} canonical-op)
      (scalar-range/arithmetic canonical-op ranges)

      (contains? #{:min :max} canonical-op)
      (scalar-range/minmax canonical-op ranges)

      (= :quot canonical-op)
      (scalar-range/quotient ranges)

      (= :select canonical-op)
      (scalar-range/hull (subvec ranges 1))

      :else
      ;; Operations without an independently modelled transfer function remain unknown.  For
      ;; integral values that means the complete declared dtype, never a guessed subrange.
      (scalar-range/for-dtype result-type))))

(defn- proof-covers-derived-range?
  [proof derived-range result-type]
  (and (= :typed-scalar-range (:kind proof))
       (integer? (:lower proof))
       (integer? (:upper proof))
       (<= (:lower proof) (:upper proof))
       (scalar-range/contained-in-dtype? proof result-type)
       derived-range
       (<= (:lower proof) (:lower derived-range))
       (<= (:upper derived-range) (:upper proof))))

(defn- conservative-loop-info
  "A loop body is checked once, not interpreted to an inductive fixed point.  In particular a
  carried integer can change on every backedge and a loop may execute zero times.  Preserve only
  its dtype fact until a future verifier grows explicit checked invariants."
  [info type]
  (if (contains? #{:byte :int :long} (canonical-type type))
    (assoc info :range (scalar-range/for-dtype type))
    info))

(declare positive-index-expression?)

(defn- nonnegative-index-expression?
  [expression values]
  (cond
    (integer? expression) (not (neg? expression))
    (value-id? expression) (boolean (:nonnegative? (get values expression)))
    (record-kind? "raster.compiler.ir.kernel_body.IndexCast" expression)
    (nonnegative-index-expression? (:argument expression) values)
    (record-kind? "raster.compiler.ir.kernel_body.IndexExpr" expression)
    (let [arguments (:arguments expression)]
      (case (:op expression)
        (:add :mul :min) (every? #(nonnegative-index-expression? % values) arguments)
        :max (boolean (some #(nonnegative-index-expression? % values) arguments))
        :ceil-div (and (nonnegative-index-expression? (first arguments) values)
                       (positive-index-expression? (second arguments) values))
        false))
    :else false))

(defn- positive-index-expression?
  "Conservatively prove that an invariant index expression advances a loop.

   Hardware group counts are positive in every executing work item. Other hardware indices are
   merely non-negative. The small transfer algebra below is intentionally one-way: an expression
   outside it is not rejected as an index, but cannot serve as a dynamic loop step."
  [expression values]
  (cond
    (integer? expression) (pos? expression)
    (value-id? expression) (boolean (:positive? (get values expression)))
    (record-kind? "raster.compiler.ir.kernel_body.IndexCast" expression)
    (positive-index-expression? (:argument expression) values)
    (record-kind? "raster.compiler.ir.kernel_body.IndexExpr" expression)
    (let [arguments (:arguments expression)]
      (case (:op expression)
        :add (and (every? #(nonnegative-index-expression? % values) arguments)
                  (boolean (some #(positive-index-expression? % values) arguments)))
        :mul (every? #(positive-index-expression? % values) arguments)
        :min (every? #(positive-index-expression? % values) arguments)
        :max (boolean (some #(positive-index-expression? % values) arguments))
        :ceil-div (and (positive-index-expression? (first arguments) values)
                       (positive-index-expression? (second arguments) values))
        false))
    :else false))

(defn- scalar-info!
  [expression values]
  (when-not (record-kind? "raster.compiler.ir.kernel_body.ScalarExpr" expression)
    (throw (ex-info "expected an explicit ScalarExpr" {:expression expression})))
  (let [{:keys [op arguments result-type options]} expression
        result-type (when (scalar-type? result-type) (canonical-type result-type))
        infos (mapv #(scalar-argument-info! % values) arguments)
        intrinsic (intrinsics/descriptor op)
        canonical-op (intrinsics/canonical op)]
    (when-not result-type
      (throw (ex-info "scalar expression requires a known result type"
                      {:expression expression :result-type (:result-type expression)})))
    (cond
      (= :cast op)
      (let [source-type (:type (first infos))
            source-integral? (contains? #{:byte :int :long} source-type)
            result-integral? (contains? #{:byte :int :long} result-type)
            result-floating? (dtype/fp-dtype? result-type)
            both-floating? (and (dtype/fp-dtype? source-type) (dtype/fp-dtype? result-type))
            narrowing? (and both-floating?
                            (> (dtype/bytes-of source-type) (dtype/bytes-of result-type)))
            rounding (:rounding options)
            overflow (:overflow options)]
        (when-not (= 1 (count infos))
          (throw (ex-info "scalar cast requires one operand" {:expression expression})))
        (when-not (= #{:rounding :overflow} (set (keys options)))
          (throw (ex-info "scalar cast must state exactly its rounding and overflow policies"
                          {:reason :kernel-body-cast-policy :options options})))
        (when-not (and (numerics/rounding-policy? (:rounding options))
                       (numerics/cast-overflow-policy? (:overflow options)))
          (throw (ex-info "scalar cast has an unsupported numerical policy"
                          {:reason :kernel-body-cast-policy :options options})))
        (when (or (= :predicate result-type) (= :predicate (:type (first infos))))
          (throw (ex-info "numeric casts cannot create or consume predicates"
                          {:expression expression})))
        ;; Integral-to-integral conversion has no rounding. Integral-to-floating conversion can
        ;; lose precision and therefore states nearest-even explicitly; float/double cover the
        ;; complete integral range, while half retains IEEE overflow. FP same-width/widening casts
        ;; are exact; FP narrowing states IEEE overflow and a directional rounding contract.
        (when-not
         (cond
           (and source-integral? result-integral?)
           (and (= :exact rounding) (not= :ieee overflow))

           (and source-integral? result-floating?)
           (or (and (= :double result-type)
                    (<= (dtype/bytes-of source-type) 4)
                    (= [:exact :exact] [rounding overflow]))
               (and (= :nearest-even rounding)
                    (= (if (= :half result-type) :ieee :exact) overflow)))

           both-floating?
           (if narrowing?
             (and (= :ieee overflow)
                  (contains? #{:nearest-even :toward-zero :up :down} rounding))
             (and (= :exact rounding) (= :exact overflow)))

           ;; Floating-to-integral casts may state their directional rounding and their chosen
           ;; finite-range policy, but never borrow representation wrapping or IEEE FP overflow.
           result-integral?
           (and (not= :wrap overflow) (not= :ieee overflow))

           :else false)
          (throw (ex-info "scalar cast policies disagree with its source and result dtypes"
                          {:reason :kernel-body-cast-policy
                           :source-type source-type :result-type result-type
                           :rounding rounding :overflow overflow}))))

      (= :select op)
      (do
        (when-not (= 3 (count infos))
          (throw (ex-info "scalar select requires predicate, true, and false operands"
                          {:expression expression})))
        (when-not (= :predicate (:type (first infos)))
          (throw (ex-info "scalar select condition must be a predicate"
                          {:condition (first infos)})))
        (let [selected-type (same-types! "scalar select" (subvec infos 1))]
          (when-not (= result-type selected-type)
            (throw (ex-info "scalar select result type disagrees with its values"
                            {:result-type result-type :value-type selected-type})))))

      (= :isnan op)
      (do
        (when-not (and (= 1 (count infos))
                       (dtype/fp-dtype? (:type (first infos)))
                       (= :predicate result-type)
                       (empty? options))
          (throw (ex-info "isnan requires one floating operand and a predicate result"
                          {:expression expression}))))

      intrinsic
      (let [arity (:arity intrinsic)
            kind (:kind intrinsic)
            operand-type (same-types! "scalar intrinsic" infos)
            integral? (contains? #{:byte :int :long} operand-type)
            overflow-op? (contains? #{:+ :- :*} canonical-op)
            arithmetic-overflow? (and integral? overflow-op?)
            overflow-policy (:overflow options)
            proof (:proof options)
            derived-range (when arithmetic-overflow?
                            (scalar-range/arithmetic canonical-op (mapv :range infos)))
            no-overflow-proved?
            (and derived-range
                 (scalar-range/contained-in-dtype? derived-range result-type))
            valid-options?
            (and (contains? #{#{:overflow} #{:overflow :proof}} (set (keys options)))
                 (contains? arithmetic-overflow-policies overflow-policy)
                 (or (not (contains? options :proof))
                     (proof-covers-derived-range? proof derived-range result-type)))]
        (when-not (= arity (count infos))
          (throw (ex-info "scalar intrinsic arity mismatch"
                          {:operation canonical-op :expected arity :actual (count infos)})))
        (when (and arithmetic-overflow? (not valid-options?))
          (throw (ex-info "integral arithmetic requires exactly one overflow contract"
                          {:reason :kernel-body-intrinsic-overflow
                           :operation canonical-op :operand-type operand-type
                           :options options})))
        (when (and arithmetic-overflow?
                   (= :no-overflow overflow-policy)
                   (not no-overflow-proved?))
          (throw (ex-info "integral no-overflow contract is not derivable from operand ranges"
                          {:reason :kernel-body-intrinsic-overflow-proof
                           :operation canonical-op :operand-type operand-type
                           :derived-range derived-range :result-type result-type
                           :proof proof})))
        (when (and (seq options) (not arithmetic-overflow?))
          (throw (ex-info "overflow contracts are only defined for integral add, subtract, and multiply"
                          {:reason :kernel-body-intrinsic-overflow
                           :operation canonical-op :operand-type operand-type
                           :options options})))
        (when (= :predicate operand-type)
          (throw (ex-info "numeric intrinsics cannot consume predicate values"
                          {:operation canonical-op})))
        (when-not (intrinsics/accepts-scalar-dtype? canonical-op operand-type)
          (throw (ex-info "scalar intrinsic is not defined for its operand dtype"
                          {:reason :kernel-body-intrinsic-dtype
                           :operation canonical-op :operand-type operand-type})))
        (if (= :cmp kind)
          (when-not (= :predicate result-type)
            (throw (ex-info "comparison intrinsic must produce a predicate"
                            {:operation canonical-op :result-type result-type})))
          (when-not (= result-type operand-type)
            (throw (ex-info "scalar intrinsic result type must equal its operand type"
                            {:operation canonical-op :result-type result-type
                             :operand-type operand-type})))))

      :else
      (throw (ex-info "scalar expression has an unknown canonical operation"
                      {:operation op :allowed-special special-scalar-ops})))
    {:type result-type
     :uniformity (join-uniformity infos)
     :range
     (cond
       (= :cast op)
       (let [source-range (:range (first infos))]
         ;; Exact widening retains its fact.  Narrowing, wrapping and floating conversion are
         ;; intentionally conservative.
         (if (and source-range
                  (scalar-range/contained-in-dtype? source-range result-type))
           source-range
           (scalar-range/for-dtype result-type)))

       (= :isnan op) nil
       :else
       (let [integral-arithmetic?
             (and (contains? #{:byte :int :long} result-type)
                  (contains? #{:+ :- :*} canonical-op))]
         (if integral-arithmetic?
           ;; A checked operation may trap and a wrapping operation may narrow modulo its
           ;; representation.  Neither permits downstream arithmetic to borrow an unbounded
           ;; mathematical interval; only an independently proved operation retains it.
           (if (= :no-overflow (:overflow options))
             (scalar-result-range canonical-op result-type infos)
             (scalar-range/for-dtype result-type))
           (scalar-result-range canonical-op result-type infos))))}))

(defn- expression-info!
  [expression values]
  (cond
    (integer? expression)
    {:type :int :uniformity all-uniform :range (scalar-range/literal expression :int)}

    (value-id? expression)
    (typed-info! values expression "index expression")

    (record-kind? "raster.compiler.ir.kernel_body.IndexCast" expression)
    (let [source (expression-info! (:argument expression) values)
          target (dtype/canon (:dtype expression))]
      (when-not (contains? #{:int :long} target)
        (throw (ex-info "index conversion target must be integral"
                        {:reason :kernel-body-index-cast-dtype
                         :expression expression :target target})))
      (when-not (numerics/cast-overflow-policy? (:overflow expression))
        (throw (ex-info "index conversion requires an explicit overflow policy"
                        {:reason :kernel-body-index-cast-overflow
                         :expression expression :overflow (:overflow expression)})))
      (when-not (and (= :exact (:overflow expression))
                     (or (= (:type source) target)
                         (and (= :int (:type source)) (= :long target))))
        (throw (ex-info "the portable index IR currently permits only exact widening conversions"
                        {:reason :kernel-body-index-cast-semantics
                         :expression expression :source (:type source) :target target
                         :overflow (:overflow expression)})))
      {:type target :uniformity (:uniformity source)
       :range (when (scalar-range/contained-in-dtype? (:range source) target)
                (:range source))})

    (record-kind? "raster.compiler.ir.kernel_body.IndexExpr" expression)
    (let [infos (mapv #(expression-info! % values) (:arguments expression))
          types (set (map :type infos))]
      (when-not (every? #{:int :long} types)
        (throw (ex-info "index expression references a non-integral SSA value"
                        {:reason :kernel-body-index-dtype :expression expression :types types})))
      (when (> (count types) 1)
        (throw (ex-info "index expression mixes integral widths without an explicit conversion"
                        {:reason :kernel-body-index-dtype :expression expression :types types})))
      {:type (or (first types) :int)
       :uniformity (join-uniformity infos)
       ;; Index expressions are scheduling arithmetic; conservatively retain a range only for
       ;; the canonical operations whose interval transfer is total and explicitly modelled.
       :range (let [op (:op expression)
                    result-type (or (first types) :int)
                    ranges (mapv :range infos)]
                (case op
                  :add (scalar-range/arithmetic :+ ranges)
                  :sub (scalar-range/arithmetic :- ranges)
                  :mul (scalar-range/arithmetic :* ranges)
                  :min (scalar-range/minmax :min ranges)
                  :max (scalar-range/minmax :max ranges)
                  (scalar-range/for-dtype result-type)))})

    :else
    (throw (ex-info "unsupported kernel index expression"
                    {:reason :kernel-body-index-expression :expression expression}))))

(defn- expression-uniformity
  [expression values]
  (:uniformity (expression-info! expression values)))

(defn- mask-uniformity
  [mask-id mask-map values]
  (if-not mask-id
    all-uniform
    (let [mask (or (get mask-map mask-id)
                   (throw (ex-info "kernel operation references an undeclared mask"
                                   {:mask mask-id})))]
      (join-uniformity
       (map #(typed-info! values % "kernel mask")
            (mapcat predicate-references (:predicates mask)))))))

(defn- scalar-value-info!
  [value values]
  (cond
    (record-kind? "raster.compiler.ir.kernel_body.Literal" value) (literal-info! value)
    (record-kind? "raster.compiler.ir.kernel_body.ScalarExpr" value) (scalar-info! value values)
    (value-id? value) (typed-info! values value "kernel scalar operation")
    :else (throw (ex-info "kernel scalar value must be typed" {:value value}))))

(defn- claim-value!
  [claimed reserved values spec owner]
  (value-spec! owner spec)
  (let [id (:id spec)]
    (when (or (contains? reserved id) (contains? values id) (contains? @claimed id))
      (throw (ex-info "kernel SSA result identity is not globally unique"
                      {:reason :kernel-body-duplicate-ssa :value id :owner owner})))
    (swap! claimed conj id))
  spec)

(defn- terminal-yield!
  [owner operations]
  (when-not (and (vector? operations) (seq operations)
                 (record-kind? "raster.compiler.ir.kernel_body.Yield" (peek operations)))
    (throw (ex-info (str owner " must terminate in Yield") {:operations operations})))
  (when (some #(record-kind? "raster.compiler.ir.kernel_body.Yield" %)
              (pop operations))
    (throw (ex-info (str owner " may only yield as its terminal operation")
                    {:operations operations})))
  (peek operations))

(defn- terminal-pipeline-yield!
  [owner operations]
  (when-not (and (vector? operations) (seq operations)
                 (record-kind? "raster.compiler.ir.kernel_body.PipelineYield"
                               (peek operations)))
    (throw (ex-info (str owner " must terminate in PipelineYield")
                    {:reason :kernel-body-pipeline-terminator :operations operations})))
  (when (some #(record-kind? "raster.compiler.ir.kernel_body.PipelineYield" %)
              (pop operations))
    (throw (ex-info (str owner " may only pipeline-yield as its terminal operation")
                    {:reason :kernel-body-pipeline-terminator :operations operations})))
  (peek operations))

(declare validate-dataflow-operations!)

(defn- validate-region!
  [owner operations expected-results values context]
  (let [yield-op (terminal-yield! owner operations)
        branch-values (validate-dataflow-operations! (pop operations) values context)
        yielded (mapv #(scalar-value-info! % branch-values) (:values yield-op))
        expected-types (mapv (comp canonical-type :type) expected-results)
        actual-types (mapv :type yielded)]
    (when-not (= expected-types actual-types)
      (throw (ex-info (str owner " yield arity or types disagree with its declared results")
                      {:reason :kernel-body-yield-mismatch
                       :expected expected-types :actual actual-types})))
    yielded))

(defn- static-workgroup-width!
  [launch]
  (let [workgroup (cond
                    (launch/launch-spec? launch) (:workgroup-size (launch/validate-spec! launch))
                    (map? launch) (:workgroup-size launch)
                    :else nil)]
    (when-not (and (vector? workgroup) (seq workgroup) (every? pos-int? workgroup))
      (throw (ex-info "collective KernelBody requires a static workgroup launch"
                      {:reason :kernel-body-collective-launch :launch launch})))
    (reduce * workgroup)))

(defn- validate-dataflow-operation!
  [operation values {:keys [storage stable-reads masks claimed reserved control-uniformity launch
                            schedule]
                     :as context}]
  (cond
    (record-kind? "raster.compiler.ir.kernel_body.IndexCompute" operation)
    (let [info (expression-info! (:expression operation) values)
          result (value (:id operation) (:type info))]
      (claim-value! claimed reserved values result "index compute")
      (assoc values (:id result) info))

    (record-kind? "raster.compiler.ir.kernel_body.ScalarCompute" operation)
    (let [result (claim-value! claimed reserved values (:result operation) "scalar compute")
          info (scalar-info! (:expression operation) values)
          declared (canonical-type (:type result))]
      (when-not (= declared (:type info))
        (throw (ex-info "scalar compute result type disagrees with its expression"
                        {:result result :expression-type (:type info)})))
      (assoc values (:id result) info))

    (record-kind? "raster.compiler.ir.kernel_body.ScalarLoad" operation)
    (let [result (claim-value! claimed reserved values (:result operation) "scalar load")
          buffer (get storage (:buffer operation))
          result-type (canonical-type (:type result))
          buffer-type (canonical-type (:dtype buffer))
          coordinate-uniformity
          (join-uniformity
           (map #(hash-map :uniformity (expression-uniformity % values))
                (:coordinates operation)))
          predicate-uniformity (mask-uniformity (:predicate operation) masks values)
          other-info (when (:predicate operation)
                       (scalar-value-info! (:other operation) values))]
      (when-not (= result-type buffer-type)
        (throw (ex-info "scalar load result type must equal the buffer element type"
                        {:reason :kernel-body-load-dtype :result result :buffer buffer})))
      (when (and other-info (not= result-type (:type other-info)))
        (throw (ex-info "masked scalar load other value has the wrong type"
                        {:reason :kernel-body-load-other-dtype
                         :expected result-type :actual (:type other-info)})))
      (assoc values (:id result)
             {:type result-type
              ;; A device load is unconstrained unless a preceding scalar operation derives a
              ;; narrower fact.  `other` is included because a masked load may produce it.
              :range (if other-info
                       (scalar-range/hull [(scalar-range/for-dtype result-type)
                                           (:range other-info)])
                       (scalar-range/for-dtype result-type))
              ;; Only a body-level StableRead contract can establish memory uniformity. Its
              ;; corresponding ABI slot requires no write alias for the whole parallel launch.
              :uniformity (reduce set/intersection
                                  (if (contains? stable-reads (:buffer operation))
                                    all-uniform lane-varying)
                                  (cond-> [coordinate-uniformity predicate-uniformity]
                                    other-info (conj (:uniformity other-info))))}))

    (record-kind? "raster.compiler.ir.kernel_body.ScalarStore" operation)
    (let [buffer (get storage (:buffer operation))
          stored (scalar-value-info! (:value operation) values)
          buffer-type (canonical-type (:dtype buffer))]
      (doseq [coordinate (:coordinates operation)]
        (expression-uniformity coordinate values))
      (mask-uniformity (:predicate operation) masks values)
      (when-not (= buffer-type (:type stored))
        (throw (ex-info "scalar store value type must equal the buffer element type"
                        {:reason :kernel-body-store-dtype
                         :buffer buffer :value-type (:type stored)})))
      values)

    (record-kind? "raster.compiler.ir.kernel_body.AtomicRMW" operation)
    (let [buffer (get storage (:buffer operation))
          contribution (scalar-value-info! (:value operation) values)
          buffer-type (canonical-type (:dtype buffer))]
      (doseq [coordinate (:coordinates operation)]
        (expression-uniformity coordinate values))
      (mask-uniformity (:predicate operation) masks values)
      (when-not (= buffer-type (:type contribution))
        (throw (ex-info "atomic contribution type must equal the buffer element type"
                        {:reason :kernel-body-atomic-dtype
                         :buffer buffer :value-type (:type contribution)})))
      values)

    (record-kind? "raster.compiler.ir.kernel_body.IfRegion" operation)
    (let [condition (typed-info! values (:condition operation) "kernel if condition")]
      (when-not (= :predicate (:type condition))
        (throw (ex-info "kernel if condition must be a predicate value"
                        {:reason :kernel-body-if-condition :condition condition})))
      (let [results (:results operation)
            branch-context (update context :control-uniformity set/intersection
                                   (:uniformity condition))
            then-values (validate-region! "kernel if then-region" (:then-operations operation)
                                          results values branch-context)
            else-values (validate-region! "kernel if else-region" (:else-operations operation)
                                          results values branch-context)]
        (reduce (fn [env [result then-info else-info]]
                  (claim-value! claimed reserved values result "kernel if result")
                  (assoc env (:id result)
                         {:type (canonical-type (:type result))
                          :range (scalar-range/hull [(:range then-info) (:range else-info)])
                          :uniformity (reduce set/intersection (:uniformity condition)
                                              [(:uniformity then-info)
                                               (:uniformity else-info)])}))
                values (map vector results then-values else-values))))

    (record-kind? "raster.compiler.ir.kernel_body.ForLoop" operation)
    (let [index (:index operation)
          lower-info (expression-info! (:lower operation) values)
          upper-info (expression-info! (:upper operation) values)
          step-info (when-not (integer? (:step operation))
                      (expression-info! (:step operation) values))
          index-type (canonical-type (:type index))
          lower-uniformity (:uniformity lower-info)
          upper-uniformity (:uniformity upper-info)
          loop-control (reduce set/intersection control-uniformity
                               [lower-uniformity upper-uniformity])
          iter-args (:iter-args operation)
          uniform-iter-args (get-in operation [:attributes :uniform-iter-args] #{})
          results (:results operation)
          initials (mapv #(scalar-value-info! (:initial %) values) iter-args)]
      (when-not (= index-type (:type lower-info) (:type upper-info))
        (throw (ex-info "kernel for-loop index and bounds must have one integral type"
                        {:reason :kernel-body-loop-index-dtype :index index
                         :lower-type (:type lower-info) :upper-type (:type upper-info)})))
      (when (and step-info (not= index-type (:type step-info)))
        (throw (ex-info "kernel for-loop dynamic step must use the induction dtype"
                        {:reason :kernel-body-loop-step-dtype :index index
                         :index-type index-type :step (:step operation)
                         :step-type (:type step-info)})))
      (when-not (positive-index-expression? (:step operation) values)
        (throw (ex-info "kernel for-loop dynamic step is not provably positive"
                        {:reason :kernel-body-loop-step-positive
                         :index index :step (:step operation)})))
      (claim-value! claimed reserved values index "kernel for-loop index")
      (doseq [[arg initial] (map vector iter-args initials)]
        (let [binding (:binding arg)]
          (claim-value! claimed reserved values binding "kernel loop-carried binding")
          (when-not (= (canonical-type (:type binding)) (:type initial))
            (throw (ex-info "kernel loop initial value type disagrees with its binding"
                            {:reason :kernel-body-loop-initial :arg arg :initial initial})))))
      (let [loop-values (into (assoc values (:id index)
                                     {:type (canonical-type (:type index))
                                      ;; The verifier checks these are the actual loop bounds;
                                      ;; including the upper endpoint is conservative and avoids
                                      ;; assuming a particular trip-count convention.
                                      :range (scalar-range/hull [(:range lower-info)
                                                                 (:range upper-info)])
                                      :uniformity loop-control})
                              (map (fn [arg initial]
                                     [(:id (:binding arg))
                                      (let [initial (conservative-loop-info
                                                     initial (:type (:binding arg)))]
                                        (if (contains? uniform-iter-args (:id (:binding arg)))
                                        ;; This is an inductive claim, checked against the
                                        ;; backedge yield below. Unclaimed carries remain
                                        ;; conservative because a later iteration may diverge.
                                        initial
                                        (assoc initial :uniformity lane-varying)))])
                                   iter-args initials))
            yielded (validate-region! "kernel for-loop body" (:operations operation)
                                      (mapv :binding iter-args) loop-values
                                      (assoc context :control-uniformity loop-control))]
        (doseq [[arg initial yielded-info] (map vector iter-args initials yielded)
                :when (contains? uniform-iter-args (:id (:binding arg)))]
          (when-not (set/subset? (:uniformity initial) (:uniformity yielded-info))
            (throw (ex-info
                    "kernel loop uniform carried value is not preserved by its backedge"
                    {:reason :kernel-body-loop-uniformity-invariant
                     :binding (:binding arg)
                     :initial-uniformity (:uniformity initial)
                     :yielded-uniformity (:uniformity yielded-info)}))))
        (reduce (fn [env [result yielded-info initial-info]]
                  (claim-value! claimed reserved values result "kernel for-loop result")
                  (when-not (= (canonical-type (:type result)) (:type yielded-info))
                    (throw (ex-info "kernel for-loop result type disagrees with its yielded value"
                                    {:reason :kernel-body-loop-result
                                     :result result :yielded yielded-info})))
                  (assoc env (:id result)
                         (assoc (conservative-loop-info yielded-info (:type result)) :uniformity
                                ;; The loop may execute zero times, so its result cannot be
                                ;; more uniform than either the initial or backedge value.
                                (reduce set/intersection loop-control
                                        [(:uniformity initial-info)
                                         (:uniformity yielded-info)]))))
                values (map vector results yielded initials))))

    (record-kind? "raster.compiler.ir.kernel_body.PipelinedFor" operation)
    (let [index (:index operation)
          lower-info (expression-info! (:lower operation) values)
          upper-info (expression-info! (:upper operation) values)
          index-type (canonical-type (:type index))
          loop-control (reduce set/intersection control-uniformity
                               [(:uniformity lower-info) (:uniformity upper-info)])
          iter-args (:iter-args operation)
          uniform-iter-args (get-in operation [:attributes :uniform-iter-args] #{})
          results (:results operation)
          initials (mapv #(scalar-value-info! (:initial %) values) iter-args)
          pipeline-yield (terminal-pipeline-yield!
                          "kernel pipelined-for body" (:operations operation))]
      (when-not (= index-type (:type lower-info) (:type upper-info))
        (throw (ex-info "kernel pipelined-for index and bounds must have one integral type"
                        {:reason :kernel-body-loop-index-dtype :index index
                         :lower-type (:type lower-info) :upper-type (:type upper-info)})))
      (claim-value! claimed reserved values index "kernel pipelined-for index")
      (doseq [[arg initial] (map vector iter-args initials)]
        (let [binding (:binding arg)]
          (claim-value! claimed reserved values binding "kernel pipelined loop-carried binding")
          (when-not (= (canonical-type (:type binding)) (:type initial))
            (throw (ex-info "kernel pipelined loop initial value disagrees with its binding"
                            {:reason :kernel-body-loop-initial :arg arg :initial initial})))))
      (let [loop-values (into (assoc values (:id index)
                                     {:type index-type
                                      :range (scalar-range/hull [(:range lower-info)
                                                                 (:range upper-info)])
                                      :uniformity loop-control})
                              (map (fn [arg initial]
                                     [(:id (:binding arg))
                                      (let [initial (conservative-loop-info
                                                     initial (:type (:binding arg)))]
                                        (if (contains? uniform-iter-args (:id (:binding arg)))
                                        initial
                                        (assoc initial :uniformity lane-varying)))])
                                   iter-args initials))
            body-values (validate-dataflow-operations!
                         (pop (:operations operation)) loop-values
                         (assoc context :control-uniformity loop-control))
            yielded (mapv #(scalar-value-info! % body-values)
                          (:values pipeline-yield))
            expected-types (mapv (comp canonical-type :type :binding) iter-args)
            actual-types (mapv :type yielded)]
        (when-not (= expected-types actual-types)
          (throw (ex-info
                  "kernel pipelined-for scalar yield disagrees with its carried values"
                  {:reason :kernel-body-yield-mismatch
                   :expected expected-types :actual actual-types})))
        (doseq [[arg initial yielded-info] (map vector iter-args initials yielded)
                :when (contains? uniform-iter-args (:id (:binding arg)))]
          (when-not (set/subset? (:uniformity initial) (:uniformity yielded-info))
            (throw (ex-info
                    "kernel pipelined loop uniform carried value is not preserved by its backedge"
                    {:reason :kernel-body-loop-uniformity-invariant
                     :binding (:binding arg)
                     :initial-uniformity (:uniformity initial)
                     :yielded-uniformity (:uniformity yielded-info)}))))
        (reduce (fn [env [result yielded-info initial-info]]
                  (claim-value! claimed reserved values result "kernel pipelined-for result")
                  (when-not (= (canonical-type (:type result)) (:type yielded-info))
                    (throw (ex-info
                            "kernel pipelined-for result type disagrees with its yielded value"
                            {:reason :kernel-body-loop-result
                             :result result :yielded yielded-info})))
                  (assoc env (:id result)
                         (assoc (conservative-loop-info yielded-info (:type result)) :uniformity
                                (reduce set/intersection loop-control
                                        [(:uniformity initial-info)
                                         (:uniformity yielded-info)]))))
                values (map vector results yielded initials))))

    (record-kind? "raster.compiler.ir.kernel_body.Collective" operation)
    (let [result (claim-value! claimed reserved values (:result operation) "subgroup collective")
          input (scalar-value-info! (:input operation) values)
          result-type (canonical-type (:type result))
          width (:width operation)
          workgroup-width (static-workgroup-width! launch)
          scheduled-width (or (:subgroup-size schedule)
                              (get-in schedule [:score-reduction :width])
                              (get-in schedule [:matrix :subgroup]))]
      (when-not (contains? control-uniformity :subgroup)
        (throw (ex-info "subgroup collective appears in lane-divergent control flow"
                        {:reason :kernel-body-divergent-collective
                         :operation operation :control-uniformity control-uniformity})))
      (when-not (= result-type (:type input))
        (throw (ex-info "collective result type must equal its input type"
                        {:reason :kernel-body-collective-dtype
                         :result result :input input})))
      (when-not (and (<= width workgroup-width) (zero? (mod workgroup-width width))
                     (integer? scheduled-width) (= width scheduled-width))
        (throw (ex-info "collective width disagrees with launch or scheduled subgroup geometry"
                        {:reason :kernel-body-collective-width :width width
                         :workgroup-width workgroup-width :scheduled-width scheduled-width})))
      (case (:kind operation)
        :reduce
        (let [operator (intrinsics/canonical (:operator operation))]
          (when-not (and (contains? #{:+ :* :min :max :bit-and :bit-or :bit-xor} operator)
                         (intrinsics/accepts-scalar-dtype? operator (:type input)))
            (throw (ex-info "subgroup reduction operator is not associative for its input dtype"
                            {:reason :kernel-body-collective-operator
                             :operator (:operator operation) :input-type (:type input)}))))
        :broadcast
        (let [lane (:source-lane operation)]
          (when-not (and (integer? lane) (<= 0 lane) (< lane width))
            (throw (ex-info "subgroup broadcast source lane must be statically in range"
                            {:reason :kernel-body-broadcast-source
                             :source-lane lane :width width})))))
      (assoc values (:id result) {:type result-type :uniformity subgroup-uniform}))

    (record-kind? "raster.compiler.ir.kernel_body.WorkgroupBarrier" operation)
    (do
      (when-not (contains? control-uniformity :workgroup)
        (throw (ex-info "workgroup barrier appears in divergent control flow"
                        {:reason :kernel-body-divergent-workgroup-barrier
                         :operation operation :control-uniformity control-uniformity})))
      values)

    (record-kind? "raster.compiler.ir.kernel_body.AsyncWorkgroupCopy" operation)
    (let [_ (static-workgroup-width! launch)
          coordinate-uniformity
          (join-uniformity
           (map #(hash-map :uniformity (expression-uniformity % values))
                (concat (:source-coordinates operation)
                        (:destination-coordinates operation))))]
      (when-not (and (contains? control-uniformity :workgroup)
                     (contains? coordinate-uniformity :workgroup))
        (throw (ex-info "async workgroup copy must be issued uniformly by the workgroup"
                        {:reason :kernel-body-divergent-async-copy
                         :operation operation
                         :control-uniformity control-uniformity
                         :coordinate-uniformity coordinate-uniformity})))
      values)

    (or (record-kind? "raster.compiler.ir.kernel_body.AsyncCommit" operation)
        (record-kind? "raster.compiler.ir.kernel_body.AsyncWait" operation))
    (do
      (when-not (contains? control-uniformity :workgroup)
        (throw (ex-info "async commit/wait must execute uniformly across the workgroup"
                        {:reason :kernel-body-divergent-async-control
                         :operation operation :control-uniformity control-uniformity})))
      values)

    (record-kind? "raster.compiler.ir.kernel_body.Guard" operation)
    (let [guard-uniformity (mask-uniformity (:mask operation) masks values)]
      (validate-dataflow-operations!
       (:operations operation) values
       (update context :control-uniformity set/intersection guard-uniformity))
      values)

    (record-kind? "raster.compiler.ir.kernel_body.Loop" operation)
    (let [loop-uniformity (reduce set/intersection control-uniformity
                                  [(expression-uniformity (:lower operation) values)
                                   (expression-uniformity (:upper operation) values)])]
      (validate-dataflow-operations!
       (:operations operation)
       (assoc values (:index operation) {:type :int :uniformity loop-uniformity})
       (assoc context :control-uniformity loop-uniformity))
      values)

    (record-kind? "raster.compiler.ir.kernel_body.Yield" operation)
    (throw (ex-info "Yield is only legal as a structured region terminator"
                    {:reason :kernel-body-misplaced-yield :operation operation}))

    (record-kind? "raster.compiler.ir.kernel_body.PipelineYield" operation)
    (throw (ex-info "PipelineYield is only legal as a pipelined-for terminator"
                    {:reason :kernel-body-misplaced-pipeline-yield :operation operation}))

    ;; Matrix fragment operations do not define scalar SSA values.
    :else values))

(defn- validate-dataflow-operations!
  [operations values context]
  (reduce (fn [env operation]
            (validate-dataflow-operation! operation env context))
          values operations))

;; Async groups are symbolic schedule dependencies, not scalar SSA values.  Keep their lifetime
;; verification separate from scalar dataflow so a target can lower the same dependency graph to
;; OpenCL events, CUDA cp.async groups, or an honest synchronous implementation.
(defn- nested-operation-regions
  [operation]
  (cond
    (record-kind? "raster.compiler.ir.kernel_body.IfRegion" operation)
    [(:then-operations operation) (:else-operations operation)]

    (or (record-kind? "raster.compiler.ir.kernel_body.ForLoop" operation)
        (record-kind? "raster.compiler.ir.kernel_body.PipelinedFor" operation)
        (record-kind? "raster.compiler.ir.kernel_body.Loop" operation)
        (record-kind? "raster.compiler.ir.kernel_body.Guard" operation))
    [(:operations operation)]

    :else []))

(defn- scalar-ssa-store-regions
  [operations]
  (mapcat
   (fn [operation]
     (concat
      (when (and (record-kind? "raster.compiler.ir.kernel_body.TileStore" operation)
                 (record-kind? "raster.compiler.ir.kernel_body.ScalarSSARegion"
                               (:value-region operation)))
        [(:value-region operation)])
      (mapcat scalar-ssa-store-regions (nested-operation-regions operation))))
   operations))

(defn- validate-scalar-ssa-dataflow!
  [region values context]
  (let [accumulator (first (:parameters region))
        region-values
        (into (assoc values accumulator
                     {:type (canonical-type (:accumulator-dtype region))
                      :uniformity lane-varying})
              (map (fn [index] [index {:type :int :uniformity lane-varying}]))
              (:indices region))
        boundary-ids (set (concat (:parameters region) (:indices region)))
        region-context (-> context
                           (assoc :claimed (atom #{}))
                           (update :reserved into boundary-ids)
                           (assoc :control-uniformity lane-varying))
        final-values (validate-dataflow-operations!
                      (:operations region) region-values region-context)
        result (get final-values (:result region))]
    (when-not (= (canonical-type (:result-dtype region)) (:type result))
      (throw (ex-info "scalar SSA store-region result has the wrong dtype"
                      {:reason :kernel-body-scalar-region-result-dtype
                       :result (:result region) :declared (:result-dtype region)
                       :actual (:type result)})))
    region))

(defn- validate-async-protocol!
  [operations storage stable-buffers reserved]
  (let [claimed (atom reserved)]
    (letfn [(claim! [id owner operation]
              (when (contains? @claimed id)
                (throw (ex-info "async copy/group identity is not globally unique"
                                {:reason :kernel-body-async-id-collision
                                 :id id :owner owner :operation operation})))
              (swap! claimed conj id))
            (clean-state! [owner state]
              (when (or (seq (:issued state)) (seq (:committed state))
                        (seq (:awaiting-barrier state)))
                (throw (ex-info "async staging lifetime crosses a structured region boundary"
                                {:reason :kernel-body-async-region-lifetime
                                 :owner owner :state state}))))
            (active-staging? [state]
              (or (seq (:issued state)) (seq (:committed state))
                  (seq (:awaiting-barrier state))
                  (seq (:readable-stages state))
                  (seq (:consumed-stages state))))
            (queue-state [state]
              (select-keys state [:issued :committed :awaiting-barrier :readable-stages]))
            (copy-stage [copy]
              [(:destination copy) (:destination-coordinates copy)])
            (memory-stage [operation]
              [(:buffer operation) (:coordinates operation)])
            (same-stage? [left right]
              (let [[left-buffer left-coordinates] left
                    [right-buffer right-coordinates] right]
                (and (= left-buffer right-buffer)
                     ;; Exact coordinate equality proves disjoint 2-D stage rows. Any dynamic or
                     ;; differently ranked access remains conservatively allocation-wide.
                     (or (= left-coordinates right-coordinates)
                         (not (and (= 2 (count left-coordinates))
                                   (= 2 (count right-coordinates))
                                   (integer? (first left-coordinates))
                                   (integer? (first right-coordinates))
                                   (not= (first left-coordinates)
                                         (first right-coordinates))))))))
            (contains-stage? [stages stage]
              (boolean (some #(same-stage? % stage) stages)))
            (group-signature [group]
              (mapv (fn [copy]
                      [(copy-stage copy) (:elements copy) (:transfer-bytes copy)])
                    (:copies group)))
            (pipeline-state-compatible! [operation incoming outgoing]
              (let [incoming-widths (mapv (comp count :copies) incoming)
                    outgoing-widths (mapv (comp count :copies) outgoing)
                    incoming-stages (mapv group-signature incoming)
                    outgoing-stages (mapv group-signature outgoing)]
                (when-not (and (= incoming-widths outgoing-widths)
                               (= incoming-stages outgoing-stages))
                  (throw (ex-info
                          "pipelined-for backedge must preserve its rotating async stages"
                          {:reason :kernel-body-pipeline-stage-invariant
                           :operation operation
                           :incoming (mapv group-signature incoming)
                           :outgoing (mapv group-signature outgoing)})))))
            (validate-sequence-state! [owner operations initial-state]
              (reduce
               (fn [{:keys [issued committed awaiting-barrier consumed-stages]
                     :as state} operation]
                 (cond
                   (record-kind? "raster.compiler.ir.kernel_body.AsyncWorkgroupCopy" operation)
                   (let [id (:id operation)
                         source (get storage (:source operation))
                         source-root (or (some-> source :view :buffer) (:id source))
                         destination (:destination operation)
                         stage (copy-stage operation)]
                     (claim! id :copy operation)
                     (when-not (contains? stable-buffers source-root)
                       (throw (ex-info
                               "async copy source requires a whole-kernel stable-read contract"
                               {:reason :kernel-body-async-source-stability
                                :copy id :source source-root})))
                     (when (or (some #(same-stage? stage (copy-stage %)) issued)
                               (some (fn [group]
                                       (some #(same-stage? stage (copy-stage %))
                                             (:copies group)))
                                     committed)
                               (contains-stage? awaiting-barrier stage)
                               (contains-stage? consumed-stages stage))
                       (throw (ex-info "async copy destination is still live"
                                       {:reason :kernel-body-async-destination-live
                                        :copy id :destination destination})))
                     (-> state
                         (update :readable-stages
                                 (fn [stages]
                                   (into #{} (remove #(same-stage? % stage)) stages)))
                         (update :issued conj operation)))

                   (record-kind? "raster.compiler.ir.kernel_body.AsyncCommit" operation)
                   (let [group (:id operation)
                         copy-ids (mapv :id issued)]
                     (claim! group :group operation)
                     (when-not (= copy-ids (:copies operation))
                       (throw (ex-info
                               "async commit must close every copy issued since the prior commit"
                               {:reason :kernel-body-async-commit-order
                                :group group :issued copy-ids
                                :committed (:copies operation)})))
                     (-> state
                         (assoc :issued [])
                         (update :committed conj {:id group :copies issued})))

                   (record-kind? "raster.compiler.ir.kernel_body.AsyncWait" operation)
                   (let [groups (:groups operation)
                         committed-ids (mapv :id committed)
                         group-count (count groups)
                         enough-groups? (<= group-count (count committed))
                         waited (when enough-groups? (subvec committed 0 group-count))
                         remaining (when enough-groups? (subvec committed group-count))]
                     (when-not (and enough-groups?
                                    (= groups (subvec committed-ids 0 group-count))
                                    (= (:pending-groups operation) (count remaining)))
                       (throw (ex-info
                               "async wait must consume an oldest prefix and state the remainder"
                               {:reason :kernel-body-async-wait-order
                                :wait groups :committed committed-ids
                                :pending-groups (:pending-groups operation)})))
                     (-> state
                         (assoc :committed remaining)
                         (update :awaiting-barrier into
                                 (map copy-stage (mapcat :copies waited)))))

                   (record-kind? "raster.compiler.ir.kernel_body.WorkgroupBarrier" operation)
                   (-> state
                       (update :readable-stages into awaiting-barrier)
                       (assoc :awaiting-barrier #{} :consumed-stages #{}))

                   (or (record-kind? "raster.compiler.ir.kernel_body.ScalarLoad" operation)
                       (record-kind? "raster.compiler.ir.kernel_body.ScalarStore" operation)
                       (record-kind? "raster.compiler.ir.kernel_body.AtomicRMW" operation))
                   (let [buffer (:buffer operation)
                         stage (memory-stage operation)
                         incomplete (into (set (map copy-stage issued))
                                          (map copy-stage (mapcat :copies committed)))]
                     (when (or (contains-stage? incomplete stage)
                               (contains-stage? awaiting-barrier stage))
                       (throw (ex-info
                               (if (contains-stage? incomplete stage)
                                 "staged workgroup memory is consumed before its async wait"
                                 "staged workgroup memory is consumed before a workgroup barrier")
                               {:reason (if (contains? incomplete buffer)
                                          :kernel-body-async-missing-wait
                                          :kernel-body-async-missing-barrier)
                                :buffer buffer :operation operation})))
                     (if (contains-stage? (:readable-stages state) stage)
                       (update state :consumed-stages conj stage)
                       state))

                   (record-kind? "raster.compiler.ir.kernel_body.PipelinedFor" operation)
                   (let [async-args (:async-iter-args operation)
                         initial-groups (mapv :initial async-args)
                         committed-ids (mapv :id committed)
                         pipeline-yield (terminal-pipeline-yield!
                                         "kernel pipelined-for body"
                                         (:operations operation))]
                     (when-not (and (empty? issued) (empty? awaiting-barrier)
                                    (empty? consumed-stages)
                                    (= initial-groups committed-ids))
                       (throw (ex-info
                               "pipelined-for must carry the complete ordered async queue"
                               {:reason :kernel-body-pipeline-entry-state
                                :operation operation :initial-groups initial-groups
                                :state state})))
                     (doseq [arg async-args]
                       (claim! (:binding arg) :pipeline-binding operation))
                     (doseq [result (:async-results operation)]
                       (claim! result :pipeline-result operation))
                     (let [body-incoming
                           (mapv (fn [arg group] (assoc group :id (:binding arg)))
                                 async-args committed)
                           body-final
                           (validate-sequence-state!
                            [owner :pipeline-body]
                            (pop (:operations operation))
                            {:issued [] :committed body-incoming
                             :awaiting-barrier #{}
                             :readable-stages (:readable-stages state)
                             :consumed-stages #{}})
                           yielded-groups (:groups pipeline-yield)
                           yielded-committed (:committed body-final)]
                       (when-not (and (empty? (:issued body-final))
                                      (empty? (:awaiting-barrier body-final))
                                      (empty? (:consumed-stages body-final))
                                      (= yielded-groups (mapv :id yielded-committed)))
                         (throw (ex-info
                                 "pipelined-for must yield its complete ordered async queue after a reuse barrier"
                                 {:reason :kernel-body-pipeline-backedge-state
                                  :operation operation :yielded yielded-groups
                                  :state body-final})))
                       (pipeline-state-compatible! operation body-incoming yielded-committed)
                       (assoc body-final :committed
                              (mapv (fn [result group] (assoc group :id result))
                                    (:async-results operation) yielded-committed))))

                   (seq (nested-operation-regions operation))
                   (let [regions (nested-operation-regions operation)]
                     (if-not (active-staging? state)
                       (do
                         (clean-state! owner state)
                         (doseq [[index region] (map-indexed vector regions)]
                           (let [nested-final
                                 (validate-sequence-state!
                                  [owner index] region
                                  {:issued [] :committed [] :awaiting-barrier #{}
                                   :readable-stages #{} :consumed-stages #{}})]
                             (clean-state! [owner index] nested-final)))
                         state)
                       (if (record-kind? "raster.compiler.ir.kernel_body.IfRegion" operation)
                         (let [branch-states
                               (mapv (fn [index region]
                                       (validate-sequence-state! [owner index] region state))
                                     (range) regions)]
                           (when-not (apply = branch-states)
                             (throw (ex-info
                                     "structured branches disagree on live async staging state"
                                     {:reason :kernel-body-async-branch-state
                                      :owner owner :operation operation
                                      :branch-states branch-states})))
                           (first branch-states))
                         (let [nested-final
                               (validate-sequence-state! [owner 0] (first regions) state)]
                           (when-not (= (queue-state state) (queue-state nested-final))
                             (throw (ex-info
                                     "ordinary structured loop changes live async queue state"
                                     {:reason :kernel-body-async-loop-state
                                      :owner owner :operation operation
                                      :initial state :final nested-final})))
                           nested-final))))

                   :else state))
               initial-state
               operations))]
      (let [final-state
            (validate-sequence-state!
             :kernel operations
             {:issued [] :committed [] :awaiting-barrier #{}
              :readable-stages #{} :consumed-stages #{}})]
        (clean-state! :kernel final-state))
      (set/difference @claimed reserved))))

(defn validate!
  "Verify a scheduled KernelBody and return it unchanged."
  [body]
  (when-not (kernel-body? body)
    (throw (ex-info "kernel body must be a KernelBody value"
                    {:body body :actual (type body)})))
  (let [{:keys [id parameters views stable-reads allocations indices masks fragments operations
                schedule launch provenance attributes]} body]
    (when (nil? id)
      (throw (ex-info "kernel body requires a stable identity" {:body body})))
    (doseq [[field values] [[:parameters parameters] [:views views] [:stable-reads stable-reads]
                            [:allocations allocations] [:indices indices] [:masks masks]
                            [:fragments fragments] [:operations operations]]]
      (when-not (vector? values)
        (throw (ex-info "kernel body sections must be ordered vectors"
                        {:field field :value values}))))
    (unique-ids! "kernel parameters" parameters)
    (unique-ids! "kernel buffer views" views)
    (unique-ids! "kernel workgroup allocations" allocations)
    (unique-ids! "kernel indices" indices)
    (unique-ids! "kernel masks" masks)
    (unique-ids! "kernel fragments" fragments)
    (doseq [p parameters]
      (when-not (record-kind? "raster.compiler.ir.kernel_body.KernelParameter" p)
        (throw (ex-info "kernel parameter must be a KernelParameter value" {:parameter p})))
      (when-not (contains? parameter-kinds (:kind p))
        (throw (ex-info "kernel parameter has an unsupported kind" {:parameter p})))
      (when-not (dtype/known? (:dtype p))
        (throw (ex-info "kernel parameter requires a known dtype" {:parameter p})))
      (when-not (keyword? (:role p))
        (throw (ex-info "kernel parameter requires a semantic role" {:parameter p})))
      (if (= :scalar (:kind p))
        (when (or (seq (:shape p)) (:layout p) (:memory-space p))
          (throw (ex-info "scalar kernel parameters cannot carry storage layout"
                          {:parameter p})))
        (do (shape! "kernel buffer parameter" (:shape p))
            (storage-layout! "kernel buffer parameter" (:shape p) (:dtype p) (:layout p))
            (when (layout/shared-memory-layout? (:layout p))
              (throw (ex-info "shared-memory layouts are restricted to workgroup allocations"
                              {:reason :kernel-body-shared-layout-memory-space
                               :parameter p})))
            (when-not (keyword? (:memory-space p))
              (throw (ex-info "kernel buffer parameter requires a memory space"
                              {:parameter p}))))))
    (let [parameter-map (into {} (map (juxt :id identity)) parameters)
          allocation-plan (workgroup-memory-plan allocations)
          allocation-map (into {}
                               (map (fn [allocation]
                                      [(:id allocation)
                                       (assoc allocation :kind :allocation
                                              :memory-space :workgroup)]))
                               allocations)
          scalar-ids (set (map :id (filter #(= :scalar (:kind %)) parameters)))
          launch-dimensions (launch/dimensions launch)
          index-scope
          (reduce
           (fn [scope idx]
             (cond
               (record-kind? "raster.compiler.ir.kernel_body.IndexBinding" idx)
               (when-not (and (contains? index-sources (:source idx))
                              (integer? (:axis idx))
                              (case (:source idx)
                                (:group :group-count :local) (< -1 (:axis idx) launch-dimensions)
                                (:subgroup :lane) (zero? (:axis idx))))
                 (throw (ex-info "kernel index binding has an invalid hardware source" {:index idx})))

               (record-kind? "raster.compiler.ir.kernel_body.IndexCompute" idx)
               (do
                 (when-not (expression? (:expression idx))
                   (throw (ex-info "computed kernel index must use explicit index IR" {:index idx})))
                 (let [outside (remove scope (expression-references (:expression idx)))]
                   (when (seq outside)
                     (throw (ex-info "computed kernel index references a value before it is defined"
                                     {:index idx :references (vec outside) :scope scope})))))

               :else
               (throw (ex-info "kernel body contains an unsupported index value" {:index idx})))
             (conj scope (:id idx)))
           scalar-ids indices)
          group-axes (into {} (keep #(when (and (record-kind?
                                                 "raster.compiler.ir.kernel_body.IndexBinding" %)
                                                (= :group (:source %)))
                                       [(:id %) (:axis %)]))
                           indices)
          view-map
          (reduce
           (fn [resolved view]
             (when-not (record-kind? "raster.compiler.ir.kernel_body.BufferView" view)
               (throw (ex-info "kernel body contains an unsupported buffer view" {:view view})))
             (when (or (contains? parameter-map (:id view)) (contains? resolved (:id view)))
               (throw (ex-info "kernel buffer view identity collides with storage"
                               {:view (:id view)})))
             (let [parent (or (get parameter-map (:buffer view))
                              (throw (ex-info "kernel buffer view references undeclared storage"
                                              {:view view :buffer (:buffer view)})))]
               (when (= :scalar (:kind parent))
                 (throw (ex-info "kernel buffer view cannot derive from scalar storage"
                                 {:view view :buffer parent})))
               (when-not (expression? (:element-offset view))
                 (throw (ex-info "kernel buffer view requires an explicit element offset"
                                 {:view view})))
               (let [outside (remove index-scope
                                     (expression-references (:element-offset view)))]
                 (when (seq outside)
                   (throw (ex-info "kernel buffer view offset references values outside its scope"
                                   {:view view :references (vec outside) :scope index-scope}))))
               (shape! "kernel buffer view" (:shape view))
               (storage-layout! "kernel buffer view" (:shape view) (:dtype parent)
                                (:layout view))
               (when (layout/shared-memory-layout? (:layout view))
                 (throw (ex-info "shared-memory layouts cannot decorate global buffer views"
                                 {:reason :kernel-body-shared-layout-memory-space
                                  :view view})))
               (let [parent-shape (:shape parent)
                     view-shape (:shape view)
                     prefix-rank (- (count parent-shape) (count view-shape))
                     offset (:element-offset view)
                     offset-arguments (when (and (record-kind?
                                                  "raster.compiler.ir.kernel_body.IndexExpr" offset)
                                                 (= :mul (:op offset)))
                                        (:arguments offset))
                     slice-index (first offset-arguments)
                     group-axis (get group-axes slice-index)]
                 (when-not (and (= 1 prefix-rank)
                                (= view-shape (subvec (vec parent-shape) 1))
                                (= (vec offset-arguments) (into [slice-index] view-shape))
                                (integer? group-axis)
                                (< group-axis (count (:group-count launch)))
                                (= (first parent-shape)
                                   (launch-bound-value
                                    (nth (:group-count launch) group-axis))))
                   (throw (ex-info
                           "kernel buffer view is not a launch-bounded contiguous leading slice"
                           {:view view :parent parent :launch launch
                            :required {:parent-shape '[extent & view-shape]
                                       :element-offset '[group-index & view-shape]
                                       :group-count 'extent}}))))
               (assoc resolved (:id view)
                      (assoc parent :id (:id view) :shape (:shape view)
                             :layout (:layout view) :view view))))
           {} views)
          storage (merge parameter-map allocation-map view-map)]
      (doseq [allocation allocations]
        (layout! "workgroup allocation" (:layout allocation))
        (when-not (and (= (:shape allocation) (get-in allocation [:layout :shape]))
                       (= (dtype/canon (:dtype allocation))
                          (dtype/canon (get-in allocation [:layout :dtype]))))
          (throw (ex-info "workgroup allocation shape and dtype must agree with its layout"
                          {:reason :kernel-body-workgroup-layout
                           :allocation allocation}))))
      (when (and (seq allocations)
                 (not= (:bytes allocation-plan) (:shared-memory-bytes launch)))
        (throw (ex-info "workgroup allocations and launch shared-memory charge disagree"
                        {:reason :kernel-body-workgroup-byte-accounting
                         :allocation-bytes (:bytes allocation-plan)
                         :launch-shared-memory-bytes (:shared-memory-bytes launch)})))
      (let [stable-buffers (mapv :buffer stable-reads)]
        (when-not (= (count stable-buffers) (count (set stable-buffers)))
          (throw (ex-info "kernel stable-read requirements must name unique buffers"
                          {:reason :kernel-body-stable-read-duplicate
                           :buffers stable-buffers})))
        (doseq [requirement stable-reads]
          (let [buffer (get parameter-map (:buffer requirement))]
            (when-not (and (record-kind? "raster.compiler.ir.kernel_body.StableRead" requirement)
                           buffer (= :input (:kind buffer)))
              (throw (ex-info "kernel stable-read requirement must name an input parameter"
                              {:reason :kernel-body-stable-read-invalid
                               :requirement requirement :parameter buffer}))))))
      (let [section-ids (vec (concat (map :id parameters) (map :id views)
                                     (map :id allocations) (map :id indices)
                                     (map :id masks) (map :id fragments)))]
        (when-not (= (count section-ids) (count (set section-ids)))
          (throw (ex-info "kernel storage, index, mask, and fragment identities must be globally unique"
                          {:reason :kernel-body-section-id-collision :ids section-ids}))))
      (doseq [m masks]
        (when-not (and (record-kind? "raster.compiler.ir.kernel_body.Mask" m)
                       (vector? (:predicates m)) (seq (:predicates m))
                       (every? predicate? (:predicates m)))
          (throw (ex-info "kernel mask requires explicit predicates" {:mask m}))))
      (doseq [f fragments]
        (when-not (and (record-kind? "raster.compiler.ir.kernel_body.Fragment" f)
                       (dtype/known? (:dtype f)))
          (throw (ex-info "kernel fragment requires a known dtype" {:fragment f})))
        (shape! "kernel fragment" (:shape f))
        (layout! "kernel fragment" (:layout f))
        (when (and (contains? (:layout f) :dtype)
                   (not= (dtype/canon (:dtype f))
                         (dtype/canon (get-in f [:layout :dtype]))))
          (throw (ex-info "kernel fragment dtype disagrees with its layout"
                          {:reason :kernel-body-fragment-layout-dtype
                           :fragment f}))))
      (doseq [[field value] [[:schedule schedule] [:launch launch] [:provenance provenance]
                             [:attributes attributes]]]
        (when-not (map? value)
          (throw (ex-info "kernel body descriptive sections must be maps"
                          {:field field :value value}))))
      (launch/validate-spec! launch)
      (validate-operations! operations
                            storage
                            (into {} (map (juxt :id identity)) fragments)
                            (into {} (map (juxt :id identity)) masks)
                            index-scope
                            (mapv :id (filter #(= :epilogue (:role %)) parameters)))
      (let [section-reserved (set (concat (keys storage) (map :id indices) (map :id masks)
                                          (map :id fragments)))
            async-ids (validate-async-protocol!
                       operations storage (set (map :buffer stable-reads)) section-reserved)
            initial-values
            (reduce
             (fn [values idx]
               (let [uniformity
                     (if (record-kind? "raster.compiler.ir.kernel_body.IndexBinding" idx)
                       (case (:source idx)
                         (:group :group-count) all-uniform
                         :local lane-varying
                         :subgroup subgroup-uniform
                         :lane lane-varying)
                       (expression-uniformity (:expression idx) values))]
                 (assoc values (:id idx)
                        {:type (if (record-kind? "raster.compiler.ir.kernel_body.IndexBinding" idx)
                                 :int
                                 (:type (expression-info! (:expression idx) values)))
                         :range (if (record-kind? "raster.compiler.ir.kernel_body.IndexBinding" idx)
                                  ;; Hardware ids have target-dependent launch ranges.  Until a
                                  ;; launch contract proves one, they are ordinary full-width
                                  ;; ints, never implicit no-overflow evidence.
                                  (scalar-range/for-dtype :int)
                                  (:range (expression-info! (:expression idx) values)))
                         :nonnegative?
                         (if (record-kind? "raster.compiler.ir.kernel_body.IndexBinding" idx)
                           true
                           (nonnegative-index-expression? (:expression idx) values))
                         :positive?
                         (if (record-kind? "raster.compiler.ir.kernel_body.IndexBinding" idx)
                           (= :group-count (:source idx))
                           (positive-index-expression? (:expression idx) values))
                         :uniformity uniformity})))
             (into {}
                   (map (fn [parameter]
                          [(:id parameter)
                           {:type (canonical-type (:dtype parameter))
                            ;; ABI scalars and all loads begin at their complete dtype range.
                            ;; Producers may derive a narrower range with real clamp/select
                            ;; operations; roles alone are not assertions.
                            :range (scalar-range/for-dtype (:dtype parameter))
                            :uniformity all-uniform}])
                        (filter #(= :scalar (:kind %)) parameters)))
             indices)
            reserved (into section-reserved async-ids)
            context {:storage storage
                     :stable-reads (set (map :buffer stable-reads))
                     :masks (into {} (map (juxt :id identity)) masks)
                     :claimed (atom #{})
                     :reserved reserved
                     :control-uniformity all-uniform
                     :launch launch
                     :schedule schedule}]
        (doseq [region (scalar-ssa-store-regions operations)]
          (validate-scalar-ssa-dataflow! region initial-values context))
        (validate-dataflow-operations! operations initial-values context))))
  body)

(defn required-async-source-alignments
  "Return external input alignment preconditions implied by overlap-required async copies.

  Coordinate-offset alignment is proved inside `validate!`; this projection names the root ABI
  buffer whose base address must satisfy the scheduled transfer width."
  [kernel-body]
  (let [kernel-body (validate! kernel-body)
        view-roots (into {} (map (juxt :id :buffer)) (:views kernel-body))
        operations (letfn [(walk [operations]
                             (mapcat (fn [operation]
                                       (cons operation
                                             (mapcat walk (nested-operation-regions operation))))
                                     operations))]
                     (walk (:operations kernel-body)))]
    (reduce (fn [requirements operation]
              (if (and (record-kind?
                        "raster.compiler.ir.kernel_body.AsyncWorkgroupCopy" operation)
                       (= :required (:overlap operation)))
                (update requirements (get view-roots (:source operation) (:source operation))
                        (fnil max 1) (:transfer-bytes operation))
                requirements))
            {} operations)))

(defn make
  "Construct and verify a target-neutral scheduled kernel body."
  [{:keys [id parameters views stable-reads allocations indices masks fragments operations schedule
           launch provenance attributes]
    :or {parameters [] views [] stable-reads [] allocations [] indices [] masks [] fragments []
         operations [] schedule {} launch {} provenance {} attributes {}}}]
  (validate! (->KernelBody id parameters views stable-reads allocations indices masks fragments
                           operations schedule launch provenance attributes)))

(defn validate-launch-index-ranges!
  "Check the physical ranges of a verified body's hardware bindings at concrete launch time.

   IndexBinding currently has canonical dtype int. A long logical extent does not widen the
   hardware index representation. This cheap binding check does not rerun body verification and
   is independent of target limits, which may impose stricter resource constraints."
  [kernel-body geometry]
  (let [{:keys [workgroup-size group-count]} (launch/validate-geometry! geometry)
        workgroup-items (reduce *' 1 workgroup-size)]
    (doseq [index (:indices kernel-body)
            :when (record-kind? "raster.compiler.ir.kernel_body.IndexBinding" index)]
      (let [upper (case (:source index)
                    :group (dec (nth group-count (:axis index)))
                    :group-count (nth group-count (:axis index))
                    :local (dec (nth workgroup-size (:axis index)))
                    ;; Without a target subgroup width, the workgroup bounds both values.
                    (:subgroup :lane) (dec workgroup-items))]
        (when-not (scalar-range/contained-in-dtype? {:lower 0 :upper upper} :int)
          (throw (ex-info "kernel launch exceeds its hardware index representation"
                          {:reason :kernel-launch-index-range :index index :dtype :int
                           :upper upper :geometry geometry}))))))
  geometry)
