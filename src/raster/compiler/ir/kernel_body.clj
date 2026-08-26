(ns raster.compiler.ir.kernel-body
  "Target-neutral scheduled kernel bodies.

  SegOps retain algorithmic semantics.  A KernelBody is the result of applying a concrete
  schedule: hardware indices, named layouts, masks, fragment operations and loop structure are
  explicit, but target spellings such as Intel DPAS, CUDA WMMA or AMD MFMA are not.  Backends lower
  these values to their own dialect; they must not recover a schedule by inspecting source text.")

(defrecord KernelParameter [id kind dtype shape memory-space layout role])
(defrecord IndexBinding [id source axis])
(defrecord IndexCompute [id expression])
(defrecord IndexExpr [op arguments])
(defrecord Predicate [op arguments])
(defrecord Mask [id predicates])
(defrecord Fragment [id dtype shape layout])
(defrecord ScalarRegion [parameters expression operands result-dtype])

(defrecord FragmentInit [fragment value])
(defrecord TileLoad [fragment buffer coordinates mask cache])
(defrecord TilePrefetch [buffer coordinates shape layout mask distance])
(defrecord MatrixMad [accumulator lhs rhs instruction])
(defrecord Loop [index lower upper step operations attributes])
(defrecord Guard [mask operations])
(defrecord TileStore [buffer fragment coordinates mask value-region])

(defrecord KernelBody
           [id parameters indices masks fragments operations schedule launch provenance attributes])

(def ^:private parameter-kinds #{:input :output :scalar})
(def ^:private index-sources #{:group :subgroup :lane})
(def ^:private index-ops #{:add :sub :mul :floor-div :ceil-div :mod :min :max})
(def ^:private predicate-ops #{:lt :lte :eq :and :or :not})
(def ^:private cache-policies #{:default :cached :streaming})

(defn- record-kind? [class-name value]
  (and value (= class-name (.getName (class value)))))

(defn kernel-body? [value]
  (record-kind? "raster.compiler.ir.kernel_body.KernelBody" value))

(defn expression
  "Construct explicit target-neutral integer index arithmetic."
  [op & arguments]
  (when-not (contains? index-ops op)
    (throw (ex-info "kernel index expression has an unsupported operation"
                    {:operation op :arguments arguments})))
  (->IndexExpr op (vec arguments)))

(defn predicate
  "Construct an explicit target-neutral bounds predicate."
  [op & arguments]
  (when-not (contains? predicate-ops op)
    (throw (ex-info "kernel predicate has an unsupported operation"
                    {:operation op :arguments arguments})))
  (->Predicate op (vec arguments)))

(defn- expression? [value]
  (cond
    (or (number? value) (symbol? value)) true
    (record-kind? "raster.compiler.ir.kernel_body.IndexExpr" value)
    (and (contains? index-ops (:op value))
         (seq (:arguments value))
         (every? expression? (:arguments value)))
    :else false))

(defn- expression-references [value]
  (cond
    (symbol? value) #{value}
    (number? value) #{}
    (record-kind? "raster.compiler.ir.kernel_body.IndexExpr" value)
    (reduce into #{} (map expression-references (:arguments value)))
    :else #{}))

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

(defn- shape! [owner shape]
  (when-not (and (vector? shape) (seq shape)
                 (every? #(or (symbol? %) (and (integer? %) (pos? %))) shape))
    (throw (ex-info (str owner " requires a non-empty shape of positive extents")
                    {:shape shape}))))

(defn- layout! [owner layout]
  (when-not (and (map? layout) (keyword? (:kind layout)))
    (throw (ex-info (str owner " requires a named layout") {:layout layout}))))

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
               "raster.compiler.ir.kernel_body.TileStore"}
             (some-> value class .getName)))

(declare validate-operations!)

(defn- validate-operation!
  [operation parameters fragments masks scope]
  (let [parameter (fn [id]
                    (or (get parameters id)
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
        coordinates! (fn [owner coordinates]
                       (when-not (and (vector? coordinates) (every? expression? coordinates))
                         (throw (ex-info (str owner " coordinates must use explicit index expressions")
                                         {:coordinates coordinates})))
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
                          {:accumulator acc :lhs lhs :rhs rhs :instruction matrix}))))

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
        (validate-operations! (:operations operation) parameters fragments masks
                              (conj scope (:index operation))))

      (record-kind? "raster.compiler.ir.kernel_body.Guard" operation)
      (do
        (mask (:mask operation))
        (validate-operations! (:operations operation) parameters fragments masks scope))

      (record-kind? "raster.compiler.ir.kernel_body.TileStore" operation)
      (let [p (parameter (:buffer operation))
            f (fragment (:fragment operation))]
        (when-not (= :output (:kind p))
          (throw (ex-info "tile stores require an output buffer" {:buffer p})))
        (when-not (= :mma-frag (get-in f [:layout :kind]))
          (throw (ex-info "tile stores require an accumulator fragment" {:fragment f})))
        (coordinates! "tile-store" (:coordinates operation))
        (mask (:mask operation))
        (when-let [region (:value-region operation)]
          (when-not (record-kind? "raster.compiler.ir.kernel_body.ScalarRegion" region)
            (throw (ex-info "tile-store value region must be a ScalarRegion" {:region region})))
          (when-not (and (vector? (:parameters region)) (seq (:parameters region))
                         (some? (:expression region)) (keyword? (:result-dtype region)))
            (throw (ex-info "tile-store scalar region is incomplete" {:region region})))))

      :else
      (throw (ex-info "kernel body contains an unsupported operation"
                      {:operation operation :actual (type operation)})))))

(defn- validate-operations! [operations parameters fragments masks scope]
  (when-not (vector? operations)
    (throw (ex-info "kernel operations must be an ordered vector" {:operations operations})))
  (doseq [operation operations]
    (when-not (operation? operation)
      (throw (ex-info "kernel body contains a non-operation value" {:operation operation})))
    (validate-operation! operation parameters fragments masks scope)))

(defn validate!
  "Verify a scheduled KernelBody and return it unchanged."
  [body]
  (when-not (kernel-body? body)
    (throw (ex-info "kernel body must be a KernelBody value"
                    {:body body :actual (type body)})))
  (let [{:keys [id parameters indices masks fragments operations schedule launch provenance
                attributes]} body]
    (when (nil? id)
      (throw (ex-info "kernel body requires a stable identity" {:body body})))
    (doseq [[field values] [[:parameters parameters] [:indices indices] [:masks masks]
                            [:fragments fragments] [:operations operations]]]
      (when-not (vector? values)
        (throw (ex-info "kernel body sections must be ordered vectors"
                        {:field field :value values}))))
    (unique-ids! "kernel parameters" parameters)
    (unique-ids! "kernel indices" indices)
    (unique-ids! "kernel masks" masks)
    (unique-ids! "kernel fragments" fragments)
    (doseq [p parameters]
      (when-not (contains? parameter-kinds (:kind p))
        (throw (ex-info "kernel parameter has an unsupported kind" {:parameter p})))
      (when-not (keyword? (:dtype p))
        (throw (ex-info "kernel parameter requires a dtype" {:parameter p})))
      (when-not (keyword? (:role p))
        (throw (ex-info "kernel parameter requires a semantic role" {:parameter p})))
      (if (= :scalar (:kind p))
        (when (or (seq (:shape p)) (:layout p) (:memory-space p))
          (throw (ex-info "scalar kernel parameters cannot carry storage layout"
                          {:parameter p})))
        (do (shape! "kernel buffer parameter" (:shape p))
            (layout! "kernel buffer parameter" (:layout p))
            (when-not (keyword? (:memory-space p))
              (throw (ex-info "kernel buffer parameter requires a memory space"
                              {:parameter p}))))))
    (let [scalar-ids (set (map :id (filter #(= :scalar (:kind %)) parameters)))
          index-scope
          (reduce
           (fn [scope idx]
             (cond
               (record-kind? "raster.compiler.ir.kernel_body.IndexBinding" idx)
               (when-not (and (contains? index-sources (:source idx))
                              (integer? (:axis idx)) (not (neg? (:axis idx))))
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
           scalar-ids indices)]
      (doseq [m masks]
        (when-not (and (record-kind? "raster.compiler.ir.kernel_body.Mask" m)
                       (vector? (:predicates m)) (seq (:predicates m))
                       (every? predicate? (:predicates m)))
          (throw (ex-info "kernel mask requires explicit predicates" {:mask m}))))
      (doseq [f fragments]
        (when-not (keyword? (:dtype f))
          (throw (ex-info "kernel fragment requires a dtype" {:fragment f})))
        (shape! "kernel fragment" (:shape f))
        (layout! "kernel fragment" (:layout f)))
      (doseq [[field value] [[:schedule schedule] [:launch launch] [:provenance provenance]
                             [:attributes attributes]]]
        (when-not (map? value)
          (throw (ex-info "kernel body descriptive sections must be maps"
                          {:field field :value value}))))
      (validate-operations! operations
                            (into {} (map (juxt :id identity)) parameters)
                            (into {} (map (juxt :id identity)) fragments)
                            (into {} (map (juxt :id identity)) masks)
                            index-scope)))
  body)

(defn make
  "Construct and verify a target-neutral scheduled kernel body."
  [{:keys [id parameters indices masks fragments operations schedule launch provenance attributes]
    :or {parameters [] indices [] masks [] fragments [] operations [] schedule {} launch {}
         provenance {} attributes {}}}]
  (validate! (->KernelBody id parameters indices masks fragments operations schedule launch
                           provenance attributes)))
