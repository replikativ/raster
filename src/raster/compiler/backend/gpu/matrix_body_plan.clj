(ns raster.compiler.backend.gpu.matrix-body-plan
  "Target-neutral analysis of verified scheduled matrix KernelBody values.

  This boundary derives the structural emission plan by walking explicit indices, fragments,
  masks and operations.  Instruction-family legality belongs to the target emitter."
  (:require [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-body :as body]))

(defn- record-kind? [simple-name value]
  (= (str "raster.compiler.ir.kernel_body." simple-name)
     (some-> value class .getName)))

(defn- value-id? [value]
  (or (symbol? value) (keyword? value)))

(defn- only!
  [owner values]
  (let [values (vec values)]
    (when-not (= 1 (count values))
      (throw (ex-info (str "scheduled matrix body requires exactly one " owner)
                      {:reason :raster/bug :owner owner :values values})))
    (first values)))

(defn- require!
  [condition message data]
  (when-not condition
    (throw (ex-info message (assoc data :reason :kernel-body-matrix-plan-unimplemented)))))

(defn- expression-references [expression]
  (cond
    (value-id? expression) #{expression}
    (number? expression) #{}
    (record-kind? "IndexCast" expression)
    (expression-references (:argument expression))
    (record-kind? "IndexExpr" expression)
    (reduce into #{} (map expression-references (:arguments expression)))
    :else #{}))

(defn- coefficient
  "The static linear coefficient of `needle` in the small affine index-expression subset used by
  scheduled matrix bodies. Returns nil for nonlinear/unknown expressions."
  [expression needle]
  (cond
    (= expression needle) 1
    (or (number? expression) (value-id? expression)) 0
    (not (record-kind? "IndexExpr" expression)) nil
    (not (contains? (expression-references expression) needle)) 0
    (= :add (:op expression))
    (let [coefficients (map #(coefficient % needle) (:arguments expression))]
      (when (every? some? coefficients) (reduce + 0 coefficients)))
    (= :sub (:op expression))
    (let [[head & tail] (map #(coefficient % needle) (:arguments expression))]
      (when (and (some? head) (every? some? tail)) (reduce - head tail)))
    (= :mul (:op expression))
    (let [arguments (:arguments expression)
          containing (filter #(contains? (expression-references %) needle) arguments)
          constants (remove #(contains? (expression-references %) needle) arguments)]
      (when (and (= 1 (count containing)) (every? number? constants))
        (when-let [inner (coefficient (first containing) needle)]
          (* inner (reduce * 1 constants)))))
    :else nil))

(defn- scale-affine [form factor]
  {:constant (* factor (:constant form))
   :coefficients (into {} (map (fn [[id value]] [id (* factor value)]))
                       (:coefficients form))})

(defn- add-affine [lhs rhs]
  {:constant (+ (:constant lhs) (:constant rhs))
   :coefficients (merge-with + (:coefficients lhs) (:coefficients rhs))})

(defn- affine-form
  "Normalize explicit index IR to a linear form. nil means the expression is outside the affine
  subset accepted by the matrix-plan boundary."
  [expression]
  (cond
    (number? expression) {:constant expression :coefficients {}}
    (value-id? expression) {:constant 0 :coefficients {expression 1}}
    (not (record-kind? "IndexExpr" expression)) nil
    (= :add (:op expression))
    (let [parts (map affine-form (:arguments expression))]
      (when (every? some? parts)
        (reduce add-affine {:constant 0 :coefficients {}} parts)))
    (= :sub (:op expression))
    (let [[head & tail] (map affine-form (:arguments expression))]
      (when (and (some? head) (every? some? tail))
        (reduce #(add-affine %1 (scale-affine %2 -1)) head tail)))
    (= :mul (:op expression))
    (let [arguments (:arguments expression)
          constants (filter number? arguments)
          nonconstants (remove number? arguments)]
      (when (= 1 (count nonconstants))
        (some-> (affine-form (first nonconstants))
                (scale-affine (reduce * 1 constants)))))
    :else nil))

(defn- affine=
  [expression constant coefficients]
  (= {:constant constant :coefficients coefficients} (affine-form expression)))

(defn- lt-affine?
  [predicate lhs-constant lhs-coefficients rhs]
  (and (record-kind? "Predicate" predicate)
       (= :lt (:op predicate))
       (= 2 (count (:arguments predicate)))
       (affine= (first (:arguments predicate)) lhs-constant lhs-coefficients)
       (= rhs (second (:arguments predicate)))))

(defn- binding-of [indices source axis]
  (:id (only! (str (name source) " index on axis " axis)
              (filter #(and (record-kind? "IndexBinding" %)
                            (= source (:source %)) (= axis (:axis %)))
                      indices))))

(defn- compute-using [indices op source]
  (only! (str (name op) " index derived from " source)
         (filter #(and (record-kind? "IndexCompute" %)
                       (= op (get-in % [:expression :op]))
                       (= source (first (get-in % [:expression :arguments]))))
                 indices)))

(defn- fragment-op-idx [fragments fragment-id]
  (get-in fragments [fragment-id :layout :op-idx]))

(defn analyze
  "Derive a target-neutral matrix emission plan from a verified KernelBody.

  Every returned field that affects target code has an IR witness. The returned `:instruction`
  remains target-neutral; individual emitters must validate its family and shape."
  [kernel-body]
  (let [kernel-body (body/validate! kernel-body)
        {:keys [parameters views indices masks fragments operations launch attributes]} kernel-body
        parameters-by-role (group-by :role parameters)
        lhs-param (only! "lhs parameter" (:lhs parameters-by-role))
        rhs-param (only! "rhs parameter" (:rhs parameters-by-role))
        out-param (only! "result parameter" (:result parameters-by-role))
        stable-read-buffers (set (map :buffer (:stable-reads kernel-body)))
        read-only-parameters (filterv #(= :input (:kind %)) parameters)
        _ (require! (every? #(contains? stable-read-buffers (:id %)) read-only-parameters)
                    "matrix read-only parameters require stable no-write-alias contracts"
                    {:read-only-parameters (mapv :id read-only-parameters)
                     :stable-reads stable-read-buffers})
        view-map (into {} (map (juxt :id identity)) views)
        operation-buffers (:operation-buffers attributes)
        lhs-storage-id (get operation-buffers :row (:id lhs-param))
        rhs-storage-id (get operation-buffers :col (:id rhs-param))
        out-storage-id (get operation-buffers :out (:id out-param))
        storage-entry (fn [storage-id parameter]
                        (if (= storage-id (:id parameter))
                          parameter
                          (let [view (get view-map storage-id)]
                            (require! (and view (= (:id parameter) (:buffer view)))
                                      "matrix buffer view does not derive from its role parameter"
                                      {:storage storage-id :parameter parameter :view view})
                            (assoc parameter :id storage-id :shape (:shape view)
                                   :layout (:layout view) :view view))))
        lhs-storage (storage-entry lhs-storage-id lhs-param)
        rhs-storage (storage-entry rhs-storage-id rhs-param)
        out-storage (storage-entry out-storage-id out-param)
        [[m-extent k-extent] [k-extent' n-extent] [m-extent' n-extent']]
        (map :shape [lhs-storage rhs-storage out-storage])
        _ (require! (= [m-extent k-extent n-extent]
                       [m-extent' k-extent' n-extent'])
                    "matrix parameter shapes do not compose" {:parameters parameters})
        _ (require! (every? (fn [{:keys [shape dtype] storage-layout :layout}]
                             (= (layout/row-major shape dtype) storage-layout))
                           [lhs-storage rhs-storage out-storage])
                    "matrix plan currently requires exact row-major storage layouts"
                    {:lhs lhs-storage :rhs rhs-storage :out out-storage})
        dimension-parameters (:dimension-parameters attributes)
        m-parameter (:m dimension-parameters)
        n-parameter (:n dimension-parameters)
        k-parameter (:k dimension-parameters)
        dimension-values (:dimension-values attributes)
        schedule-parameters (filterv #(= :schedule (:role %)) parameters)
        _ (require! (every? #(and (= :scalar (:kind %)) (= :int (:dtype %)))
                            schedule-parameters)
                    "matrix schedule parameters must be scalar integers"
                    {:parameters schedule-parameters})
        _ (require! (= #{m-parameter n-parameter k-parameter}
                       (set (map :id (:dimension parameters-by-role))))
                    "matrix body dimension roles and parameter identities disagree"
                    {:dimension-parameters dimension-parameters
                     :parameters (:dimension parameters-by-role)})
        _ (require! (= {m-parameter m-extent n-parameter n-extent k-parameter k-extent}
                       dimension-values)
                    "matrix dimension specializations must agree with its storage shapes"
                    {:dimension-parameters dimension-parameters
                     :dimension-values dimension-values
                     :storage-dimensions [m-extent n-extent k-extent]})
        _ (require! (and (= :half (:dtype lhs-param)) (= :half (:dtype rhs-param))
                         (contains? #{:half :float} (:dtype out-param)))
                    "matrix plan requires f16 inputs and an f16 or f32 result"
                    {:lhs-dtype (:dtype lhs-param) :rhs-dtype (:dtype rhs-param)
                     :result-dtype (:dtype out-param)})
        fragment-map (into {} (map (juxt :id identity)) fragments)
        guard (only! "top-level guard" operations)
        _ (require! (record-kind? "Guard" guard)
                    "matrix body must begin with a guarded tile" {:operation guard})
        guarded (:operations guard)
        initializers (filter #(record-kind? "FragmentInit" %) guarded)
        outer-loop (only! "outer K loop" (filter #(record-kind? "Loop" %) guarded))
        stores (vec (filter #(record-kind? "TileStore" %) guarded))
        unknown-guarded (remove #(or (record-kind? "FragmentInit" %)
                                     (record-kind? "Loop" %)
                                     (record-kind? "TileStore" %))
                                guarded)
        _ (require! (empty? unknown-guarded)
                    "matrix tile contains operations without a matrix-plan lowering"
                    {:operations (vec unknown-guarded)})
        inner-loop (only! "matrix-fragment loop" (:operations outer-loop))
        _ (require! (record-kind? "Loop" inner-loop)
                    "outer K loop must contain the matrix-fragment loop" {:operation inner-loop})
        inner-ops (:operations inner-loop)
        prefetches (vec (filter #(record-kind? "TilePrefetch" %) inner-ops))
        loads (vec (filter #(record-kind? "TileLoad" %) inner-ops))
        mads (vec (filter #(record-kind? "MatrixMad" %) inner-ops))
        unknown-inner (remove #(or (record-kind? "TilePrefetch" %)
                                   (record-kind? "TileLoad" %)
                                   (record-kind? "MatrixMad" %))
                              inner-ops)
        _ (require! (empty? unknown-inner)
                    "matrix fragment loop contains operations without a matrix-plan lowering"
                    {:operations (vec unknown-inner)})
        lhs-loads (filterv #(= 0 (fragment-op-idx fragment-map (:fragment %))) loads)
        rhs-loads (filterv #(= 1 (fragment-op-idx fragment-map (:fragment %))) loads)
        lhs-ids (mapv :fragment lhs-loads)
        rhs-ids (mapv :fragment rhs-loads)
        instruction (only! "matrix instruction shape" (distinct (map :instruction mads)))
        {mi :m ni :n ki :k subgroup :subgroup} instruction
        _ (require! (= ki (:step inner-loop))
                    "matrix loop step and instruction K fragment disagree"
                    {:loop-step (:step inner-loop) :instruction instruction})
        block-k (:step outer-loop)
        outer-index (:index outer-loop)
        inner-index (:index inner-loop)
        [k-lower k-upper] (get-in attributes [:iteration-range :k]
                                  [(:lower outer-loop) (:upper outer-loop)])
        _ (require! (and (= k-lower (:lower outer-loop)) (= k-upper (:upper outer-loop))
                         (= outer-index (:lower inner-loop))
                         (= :min (get-in inner-loop [:upper :op]))
                         (affine= (first (get-in inner-loop [:upper :arguments]))
                                  block-k {outer-index 1})
                         (= k-upper (second (get-in inner-loop [:upper :arguments]))))
                    "matrix K loops do not describe blocked exact-fragment traversal"
                    {:outer outer-loop :inner inner-loop})
        _ (require! (and (seq lhs-loads) (seq rhs-loads)
                         (= (count prefetches) (count lhs-loads)))
                    "matrix loop requires one prefetch per lhs fragment and both operand loads"
                    {:prefetches prefetches :lhs-loads lhs-loads :rhs-loads rhs-loads})
        mad-by-operands (into {} (map (fn [op] [[(:lhs op) (:rhs op)] op]) mads))
        expected-pairs (for [lhs lhs-ids rhs rhs-ids] [lhs rhs])
        _ (require! (and (= (count expected-pairs) (count mads) (count mad-by-operands))
                         (= (set expected-pairs) (set (keys mad-by-operands))))
                    "matrix MAD operations do not cover the lhs/rhs fragment product"
                    {:expected (vec expected-pairs) :actual (vec (keys mad-by-operands))})
        accumulator-ids (mapv (comp :accumulator mad-by-operands) expected-pairs)
        _ (require! (and (= (count accumulator-ids) (count initializers) (count stores))
                         (= (set accumulator-ids) (set (map :fragment initializers))
                            (set accumulator-ids) (set (map :fragment stores))))
                    "matrix accumulators must each be initialized and stored exactly once"
                    {:accumulators accumulator-ids :initializers initializers :stores stores})
        _ (require! (every? #(and (number? (:value %)) (zero? (:value %))) initializers)
                    "matrix plan currently requires zero accumulator initialization"
                    {:initializers initializers})
        prefetch-distance (only! "pipeline distance" (distinct (map :distance prefetches)))
        _ (require! (and (integer? block-k) (pos? block-k) (zero? (mod block-k ki)))
                    "outer K step must contain a whole number of matrix fragments"
                    {:block-k block-k :matrix-k ki})
        group-n (binding-of indices :group 0)
        group-m (binding-of indices :group 1)
        subgroup-id (binding-of indices :subgroup 0)
        lane-id (binding-of indices :lane 0)
        subgroup-row (compute-using indices :floor-div subgroup-id)
        subgroup-col (compute-using indices :mod subgroup-id)
        ncols (second (get-in subgroup-row [:expression :arguments]))
        _ (require! (= ncols (second (get-in subgroup-col [:expression :arguments])))
                    "subgroup row and column decomposition disagree" {:indices indices})
        computes (filter #(record-kind? "IndexCompute" %) indices)
        m-base (only! "M tile origin"
                      (filter #(pos? (or (coefficient (:expression %) group-m) 0)) computes))
        m-base-id (:id m-base)
        n-base-ids (mapv (comp second :coordinates) rhs-loads)
        n-base-computes (mapv (fn [id]
                                (only! (str "N tile origin " id)
                                       (filter #(= id (:id %)) computes)))
                              n-base-ids)
        block-m (coefficient (:expression m-base) group-m)
        sg-m (coefficient (:expression m-base) (:id subgroup-row))
        block-n (coefficient (:expression (first n-base-computes)) group-n)
        sg-n (coefficient (:expression (first n-base-computes)) (:id subgroup-col))
        _ (require! (= [sg-m sg-n] [(* (count lhs-ids) mi) (* (count rhs-ids) ni)])
                    "fragment topology and subgroup tile geometry disagree"
                    {:derived [sg-m sg-n]
                     :fragments [(* (count lhs-ids) mi) (* (count rhs-ids) ni)]})
        _ (require! (= ncols (quot block-n sg-n))
                    "subgroup decomposition and block geometry disagree"
                    {:subgroup-columns ncols :block-n block-n :sg-n sg-n})
        workgroup-width (* (quot block-m sg-m) (quot block-n sg-n) subgroup)
        expected-workgroup-size
        (into [workgroup-width] (repeat (dec (count (:group-count launch))) 1))
        _ (require! (= expected-workgroup-size (:workgroup-size launch))
                    "matrix launch does not match its block/subgroup topology"
                    {:expected expected-workgroup-size
                     :actual (:workgroup-size launch)
                     :launch launch})
        lhs-rows (mapv (comp first :coordinates) lhs-loads)
        _ (require! (every? true?
                            (map-indexed #(and (affine= %2 (* %1 mi) {m-base-id 1})
                                               (= inner-index
                                                  (second (:coordinates (nth lhs-loads %1)))))
                                         lhs-rows))
                    "lhs tile coordinates do not match its matrix fragments"
                    {:coordinates lhs-rows :origin m-base-id})
        _ (require! (every? true?
                            (map #(and (= inner-index (first (:coordinates %)))
                                       (symbol? (second (:coordinates %))))
                                 rhs-loads))
                    "rhs tile coordinates do not match the matrix-fragment loop"
                    {:loads rhs-loads :loop-index inner-index})
        _ (require! (and (every? #(= lhs-storage-id (:buffer %)) lhs-loads)
                         (every? #(= rhs-storage-id (:buffer %)) rhs-loads)
                         (every? #(= out-storage-id (:buffer %)) stores))
                    "matrix operations are not connected to their role parameters"
                    {:lhs lhs-storage :rhs rhs-storage :out out-storage})
        _ (require! (every?
                     true?
                     (map-indexed
                      #(affine= (:expression %2) (* %1 ni)
                                {group-n block-n (:id subgroup-col) sg-n})
                      n-base-computes))
                    "N tile origins do not match the rhs fragment order"
                    {:origins n-base-computes})
        _ (require! (and (integer? prefetch-distance) (pos? prefetch-distance)
                         (every? #(and (= lhs-storage-id (:buffer %))
                                       (= [mi ki] (:shape %)))
                                 prefetches)
                         (= (set lhs-rows) (set (map (comp first :coordinates) prefetches)))
                         (every? #(affine= (second (:coordinates %))
                                           (* prefetch-distance ki) {inner-index 1})
                                 prefetches))
                    "prefetch operations do not describe the matrix pipeline lookahead"
                    {:prefetches prefetches :distance prefetch-distance})
        store-by-fragment (into {} (map (juxt :fragment identity)) stores)
        _ (require! (every?
                     true?
                     (for [[m lhs] (map-indexed vector lhs-ids)
                           [n rhs] (map-indexed vector rhs-ids)
                           :let [accumulator (:accumulator (get mad-by-operands [lhs rhs]))
                                 store (get store-by-fragment accumulator)]]
                       (= [(nth lhs-rows m) (nth n-base-ids n)] (:coordinates store))))
                    "tile stores do not preserve accumulator fragment coordinates"
                    {:stores stores})
        mask-map (into {} (map (juxt :id identity)) masks)
        predicates-of (fn [mask-id] (:predicates (get mask-map mask-id)))
        guard-predicates (predicates-of (:mask guard))
        sliced-k? (not= [0 k-parameter] [k-lower k-upper])
        range-predicate? (fn [predicate]
                           (and (record-kind? "Predicate" predicate)
                                (= :lt (:op predicate))
                                (= [k-lower k-upper] (:arguments predicate))))
        _ (require! (and (= (if sliced-k? 3 2) (count guard-predicates))
                         (some #(lt-affine? % 0 {m-base-id 1} m-parameter) guard-predicates)
                         (some #(lt-affine? % 0 {(first n-base-ids) 1} n-parameter)
                               guard-predicates)
                         (or (not sliced-k?) (some range-predicate? guard-predicates)))
                    "tile guard mask does not cover the M/N tile origins"
                    {:guard guard :mask (get mask-map (:mask guard))})
        load-mask (only! "matrix-load mask" (distinct (map :mask loads)))
        load-predicates (predicates-of load-mask)
        _ (require! (and (= 1 (count load-predicates))
                         (lt-affine? (first load-predicates) 0 {inner-index 1} k-upper))
                    "matrix loads require the fragment-index K mask"
                    {:mask (get mask-map load-mask)})
        prefetch-mask (only! "matrix-prefetch mask" (distinct (map :mask prefetches)))
        prefetch-predicates (predicates-of prefetch-mask)
        _ (require! (and (= 1 (count prefetch-predicates))
                         (lt-affine? (first prefetch-predicates)
                                     (* prefetch-distance ki) {inner-index 1} k-upper))
                    "matrix prefetch mask and pipeline distance disagree"
                    {:mask (get mask-map prefetch-mask) :distance prefetch-distance})
        _ (require!
           (every?
            true?
            (for [[m lhs] (map-indexed vector lhs-ids)
                  [n rhs] (map-indexed vector rhs-ids)
                  :let [accumulator (:accumulator (get mad-by-operands [lhs rhs]))
                        store (get store-by-fragment accumulator)
                        predicates (predicates-of (:mask store))]]
              (and (= 2 (count predicates))
                   (some #(lt-affine? % (* m mi) {m-base-id 1} m-parameter) predicates)
                   (some #(lt-affine? % 0 {(nth n-base-ids n) 1 lane-id 1} n-parameter)
                         predicates))))
           "tile-store masks do not match their fragment coordinates"
           {:stores stores :masks masks})]
    {:instruction instruction
     :dimensions [m-extent n-extent k-extent]
     :mi mi :ni ni :ki ki :subgroup subgroup
     :block-m block-m :block-n block-n :block-k block-k :sg-m sg-m :sg-n sg-n
     :ncols ncols :lhs-ids lhs-ids :rhs-ids rhs-ids :mad-by-operands mad-by-operands
     :stores stores :prefetch prefetch-distance :result-dtype (:dtype out-param)
     :dimension-parameters dimension-parameters
     :dimension-values dimension-values
     :parameters parameters
     :schedule-parameters schedule-parameters
     :launch launch
     :group-z (some #(when (and (record-kind? "IndexBinding" %)
                                (= :group (:source %)) (= 2 (:axis %)))
                       (:id %))
                    indices)
     :k-lower k-lower :k-upper k-upper
     :buffer-offsets {:lhs (some-> lhs-storage :view :element-offset)
                      :rhs (some-> rhs-storage :view :element-offset)
                      :result (some-> out-storage :view :element-offset)}}))
