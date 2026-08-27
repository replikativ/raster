(ns raster.compiler.backend.gpu.kernel-body-opencl
  "Direct OpenCL lowering of verified scheduled KernelBody values.

  This namespace is below semantic contraction analysis and hardware scheduling.  It chooses only
  target spellings: Intel subgroup builtins, OpenCL declarations and a pipelined loop form.  Tile
  geometry and fragment topology are recovered from explicit index/operation IR, never from the
  source contraction or a second schedule registry."
  (:require [clojure.set :as set]
            [clojure.string :as str]
            [clojure.walk :as walk]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.kernel-body :as body]))

(defn- record-kind? [simple-name value]
  (= (str "raster.compiler.ir.kernel_body." simple-name)
     (some-> value class .getName)))

(defn- value-id? [value]
  (or (symbol? value) (keyword? value)))

(defn- target-name [value]
  (-> (name value)
      (str/replace #"[^A-Za-z0-9_]" "_")
      (str/replace #"^[^A-Za-z_]" "_$0")))

(declare emit-index-expression)

(defn- emit-infix [operator arguments env]
  (str "(" (str/join (str " " operator " ")
                     (map #(emit-index-expression % env) arguments)) ")"))

(defn- emit-index-expression
  [expression env]
  (cond
    (number? expression) (str expression)
    (value-id? expression) (or (get env expression) (target-name expression))
    (record-kind? "IndexExpr" expression)
    (let [arguments (:arguments expression)]
      (case (:op expression)
        :add (emit-infix "+" arguments env)
        :sub (emit-infix "-" arguments env)
        :mul (emit-infix "*" arguments env)
        :floor-div (emit-infix "/" arguments env)
        :mod (emit-infix "%" arguments env)
        :min (str "min(" (str/join ", " (map #(emit-index-expression % env) arguments)) ")")
        :max (str "max(" (str/join ", " (map #(emit-index-expression % env) arguments)) ")")
        :ceil-div (let [[numerator denominator] arguments]
                    (str "((" (emit-index-expression numerator env) " + "
                         (emit-index-expression denominator env) " - 1) / "
                         (emit-index-expression denominator env) ")"))))
    :else (throw (ex-info "OpenCL lowering cannot spell kernel index expression"
                          {:reason :kernel-body-opencl-unimplemented
                           :expression expression}))))

(defn- emit-wide-expression
  [expression env]
  (if (and (record-kind? "IndexExpr" expression) (= :mul (:op expression)))
    (str/join " * " (map #(str "(long)" (emit-index-expression % env))
                         (:arguments expression)))
    (str "(long)" (emit-index-expression expression env))))

(defn- nested-operations [operations]
  (mapcat (fn [operation]
            (cons operation
                  (when-let [nested (:operations operation)]
                    (nested-operations nested))))
          operations))

(def ^:private scalar-builtins
  (into #{"float" "double" "half" "int" "long" "short" "char"
          "uint" "ulong" "ushort" "uchar" "if"}
        (comp (filter string?) (filter #(re-matches #"[A-Za-z_][A-Za-z0-9_]*" %)))
        (vals ce/op-map)))

(defn- validate-emitted-scalar-calls!
  [region emitted]
  (let [calls (into #{} (map second) (re-seq #"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(" emitted))
        ;; rstr_dp4a is accompanied by a helper in general kernels, but matrix store regions do not
        ;; inject helper source. Keep the scalar-region target vocabulary self-contained.
        unsupported (set/difference calls (disj scalar-builtins "rstr_dp4a"))]
    (when (seq unsupported)
      (throw (ex-info "OpenCL scalar region contains calls without a typed target lowering"
                      {:reason :scalar-region-opencl-call-unsupported
                       :calls (vec (sort unsupported)) :region region})))
    emitted))

(defn- scalar-tag [dtype]
  (symbol (name dtype)))

(defn- array-tag [dtype]
  (symbol (str (name dtype) "s")))

(defn- lower-scalar-region
  [kernel-body region]
  (let [parameters (into {} (map (juxt :id identity)) (:parameters kernel-body))
        operand-ids (set (map :sym (:operands region)))
        scalar-ids (remove operand-ids (rest (:parameters region)))
        free-syms (subvec (vec (get-in kernel-body [:attributes :axis-symbols])) 0 2)
        [i-sym j-sym] free-syms
        ctype (dtype/ctype :opencl (:result-dtype region))
        array-syms (set (map (comp #(symbol (name %)) :sym) (:operands region)))
        int-vars (into #{} (map #(symbol (name %))) free-syms)
        indices (into {} (map (juxt :sym (comp axis-map/index-expr :map)))
                      (:operands region))
        expression (descriptor/rewrite-aget-indices (:expression region) indices)
        accumulator (first (:parameters region))
        accumulator-token (str "__acc_" (name (gensym "")))
        typed-parameters
        (into {}
              (for [id (rest (:parameters region))
                    :let [parameter (get parameters id)]]
                [id (with-meta (symbol (name id))
                      {:raster.type/tag (if (= :scalar (:kind parameter))
                                          (scalar-tag (:dtype parameter))
                                          (array-tag (:dtype parameter)))})]))
        replacements (assoc typed-parameters accumulator
                            (with-meta (symbol accumulator-token)
                              {:raster.type/tag (scalar-tag (:result-dtype region))}))
        emitted (validate-emitted-scalar-calls!
                 region
                 (binding [ce/*emit-config* ce/opencl-config
                           ce/*scalar-type* ctype
                           ce/*int-vars* (into ce/*int-vars* int-vars)]
                   (ce/emit-expr
                    (walk/postwalk-replace replacements expression)
                    (gensym "z__") array-syms)))
        params (apply str
                      (concat
                       (for [{:keys [sym dtype] :or {dtype :float}} (:operands region)]
                         (str ", __global const " (dtype/ctype :opencl dtype)
                              "* restrict " (ce/c-symbol sym)))
                       (for [id scalar-ids
                             :let [parameter (get parameters id)]]
                         (str ", " (dtype/ctype :opencl (:dtype parameter))
                              " " (ce/c-symbol id)))))]
    {:epilogue (fn [accumulator-expression row col]
                 (-> emitted
                     (str/replace accumulator-token (str "(" accumulator-expression ")"))
                     (str/replace (re-pattern (str "\\b" (ce/c-symbol i-sym) "\\b")) row)
                     (str/replace (re-pattern (str "\\b" (ce/c-symbol j-sym) "\\b")) col)))
     :epilogue-params params
     :region region}))

(defn lower-store-region
  "Lower the one verified ScalarRegion shared by all matrix stores to OpenCL scalar syntax.

  Returns nil for identity stores. This is a target lowering of typed KernelBody data: callers
  never supply source callbacks, parameter declaration strings, or helper source."
  [kernel-body]
  (let [kernel-body (body/validate! kernel-body)
        stores (filter #(record-kind? "TileStore" %)
                       (nested-operations (:operations kernel-body)))
        regions (distinct (map :value-region stores))]
    (when-not (= 1 (count regions))
      (throw (ex-info "all matrix tile stores must carry the same scalar region"
                      {:reason :raster/bug :regions (vec regions)})))
    (when-let [region (first regions)]
      (lower-scalar-region kernel-body region))))

(defn- only!
  [owner values]
  (let [values (vec values)]
    (when-not (= 1 (count values))
      (throw (ex-info (str "OpenCL KernelBody lowering requires exactly one " owner)
                      {:reason :raster/bug :owner owner :values values})))
    (first values)))

(defn- require!
  [condition message data]
  (when-not condition
    (throw (ex-info message (assoc data :reason :kernel-body-opencl-unimplemented)))))

(defn- expression-references [expression]
  (cond
    (value-id? expression) #{expression}
    (number? expression) #{}
    (record-kind? "IndexExpr" expression)
    (reduce into #{} (map expression-references (:arguments expression)))
    :else #{}))

(defn- coefficient
  "The static linear coefficient of `needle` in the small affine index-expression subset used by
  scheduled matrix bodies.  Returns nil for nonlinear/unknown expressions."
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
  "Normalize explicit index IR to a linear form.  nil means the expression is outside the affine
  subset accepted by this target lowering."
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

(defn- matrix-plan
  "Verify the currently lowered DPAS KernelBody subset and derive a target emission plan by walking
  its indices, fragments and operations.  Every field that affects generated code has an IR witness."
  [kernel-body]
  (let [kernel-body (body/validate! kernel-body)
        {:keys [parameters views indices masks fragments operations attributes]} kernel-body
        parameters-by-role (group-by :role parameters)
        lhs-param (only! "lhs parameter" (:lhs parameters-by-role))
        rhs-param (only! "rhs parameter" (:rhs parameters-by-role))
        out-param (only! "result parameter" (:result parameters-by-role))
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
        dimension-parameters (:dimension-parameters attributes)
        m-parameter (:m dimension-parameters)
        n-parameter (:n dimension-parameters)
        k-parameter (:k dimension-parameters)
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
        _ (require! (and (= :half (:dtype lhs-param)) (= :half (:dtype rhs-param))
                         (contains? #{:half :float} (:dtype out-param)))
                    "Intel matrix lowering requires f16 inputs and an f16 or f32 result"
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
                    "matrix tile contains operations without an OpenCL lowering"
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
                    "matrix fragment loop contains operations without an OpenCL lowering"
                    {:operations (vec unknown-inner)})
        lhs-loads (filterv #(= 0 (fragment-op-idx fragment-map (:fragment %))) loads)
        rhs-loads (filterv #(= 1 (fragment-op-idx fragment-map (:fragment %))) loads)
        lhs-ids (mapv :fragment lhs-loads)
        rhs-ids (mapv :fragment rhs-loads)
        first-mad (only! "matrix instruction shape" (distinct (map :instruction mads)))
        {mi :m ni :n ki :k subgroup :subgroup} first-mad
        _ (require! (and (= :dpas (:family first-mad))
                         (= 8 mi) (= 16 ki)
                         (contains? #{8 16} subgroup)
                         (= ni subgroup))
                    "Intel OpenCL lowering has no builtin for this matrix instruction"
                    {:instruction first-mad})
        _ (require! (= ki (:step inner-loop))
                    "matrix loop step and instruction K fragment disagree"
                    {:loop-step (:step inner-loop) :instruction first-mad})
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
        _ (require! (= (set expected-pairs) (set (keys mad-by-operands)))
                    "matrix MAD operations do not cover the lhs/rhs fragment product"
                    {:expected (vec expected-pairs) :actual (vec (keys mad-by-operands))})
        accumulator-ids (mapv (comp :accumulator mad-by-operands) expected-pairs)
        _ (require! (= (set accumulator-ids) (set (map :fragment initializers))
                       (set accumulator-ids) (set (map :fragment stores)))
                    "matrix accumulators must each be initialized and stored exactly once"
                    {:accumulators accumulator-ids :initializers initializers :stores stores})
        _ (require! (every? #(and (number? (:value %)) (zero? (:value %))) initializers)
                    "Intel matrix accumulators currently require zero initialization"
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
    {:mi mi :ni ni :ki ki :subgroup subgroup
     :block-m block-m :block-n block-n :block-k block-k :sg-m sg-m :sg-n sg-n
     :ncols ncols :lhs-ids lhs-ids :rhs-ids rhs-ids :mad-by-operands mad-by-operands
     :stores stores :prefetch prefetch-distance :result-dtype (:dtype out-param)
     :dimension-parameters dimension-parameters
     :schedule-parameters schedule-parameters
     :group-z (some #(when (and (record-kind? "IndexBinding" %)
                                (= :group (:source %)) (= 2 (:axis %)))
                       (:id %))
                    indices)
     :k-lower k-lower :k-upper k-upper
     :buffer-offsets {:lhs (some-> lhs-storage :view :element-offset)
                      :rhs (some-> rhs-storage :view :element-offset)
                      :result (some-> out-storage :view :element-offset)}}))

(defn- emit-plan
  [kernel-name {:keys [mi ni ki subgroup block-m block-n block-k sg-m sg-n
                       ncols lhs-ids rhs-ids mad-by-operands stores prefetch result-dtype
                       dimension-parameters schedule-parameters group-z k-lower k-upper
                       buffer-offsets]}
   {:keys [epilogue epilogue-params parameter-names]}]
  (let [nms (count lhs-ids)
        nns (count rhs-ids)
        ksteps (quot block-k ki)
        acc-id (fn [m n] (:accumulator (get mad-by-operands
                                            [(nth lhs-ids m) (nth rhs-ids n)])))
        store-by-fragment (into {} (map (juxt :fragment identity)) stores)
        ms (range nms)
        ns (range nns)
        amul (fn [m] (if (zero? m) "m_base" (str "m_base+" (* m mi))))
        c-type (case result-dtype :float "float" :half "half")
        store-cast (if (= :half result-dtype) "(half)" "")
        parameter-names (merge (zipmap [(:m dimension-parameters)
                                        (:n dimension-parameters)
                                        (:k dimension-parameters)]
                                       ["M" "N" "K"])
                               (into {} (map (fn [parameter]
                                               [(:id parameter) (target-name (:id parameter))])
                                             schedule-parameters))
                               parameter-names)
        index-names (cond-> parameter-names group-z (assoc group-z (target-name group-z)))
        sliced-k? (not= [0 (:k dimension-parameters)] [k-lower k-upper])
        k-begin (if sliced-k? "k_begin" "0")
        k-end (if sliced-k? "k_end" "K")
        scalar-params (apply str
                             (for [parameter schedule-parameters]
                               (str ", int " (get parameter-names (:id parameter)))))
        grid-z-line (when group-z
                      (str "    int " (get index-names group-z)
                           " = (int)get_group_id(2);\n"))
        offset-line (fn [pointer role]
                      (when-let [offset (get buffer-offsets role)]
                        (when-not (and (number? offset) (zero? offset))
                          (str "    " pointer " += "
                               (emit-wide-expression offset index-names) ";\n"))))
        k-range-lines (when sliced-k?
                        (str "    int k_begin = "
                             (emit-index-expression k-lower index-names) ";\n"
                             "    int k_end = "
                             (emit-index-expression k-upper index-names) ";\n"
                             "    if (k_begin >= k_end) return;\n"))
        kstep (fn [kpos]
                (str "        { int pk = " kpos " + " (* prefetch ki) ";\n"
                     "          if (pk < " k-end ") {\n"
                     (apply str
                            (for [m ms]
                              (str "            intel_sub_group_2d_block_prefetch_16b_8r16x1c((__global void*)A, a_wb, M, a_pb, (int2)(pk, " (amul m) "));\n")))
                     "          } }\n"
                     (apply str
                            (for [n ns]
                              (str "        bp" n " = as_int8(intel_subgroup_block_read_transform_u16_k16((__global void*)B, b_wb, K, b_pb, (int2)(n_base" n ", " kpos ")));\n")))
                     (apply str
                            (for [m ms]
                              (str "        intel_sub_group_2d_block_read_16b_8r16x1c((__global void*)A, a_wb, M, a_pb, (int2)(" kpos ", " (amul m) "), &a" m ");\n")))
                     (apply str (for [m ms] (str "        sa" m " = as_short8(a" m ");\n")))
                     (apply str
                            (for [m ms n ns]
                              (do
                                (require! (contains? store-by-fragment (acc-id m n))
                                          "matrix accumulator has no store operation"
                                          {:accumulator (acc-id m n)})
                                (str "        acc" m n " = intel_sub_group_f16_f16_matrix_mad_k16(sa" m ", bp" n ", acc" m n ");\n"))))))]
    (str
     "#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable\n"
     "#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable\n\n"
     "// Tiled GEMM (parametric): WG " block-m "x" block-n ", SG " sg-m "x" sg-n
     ", K " block-k ", DPAS " mi "x" ni "x" ki ", sg " subgroup "\n"
     "__attribute__((intel_reqd_sub_group_size(" subgroup ")))\n"
     "__kernel void " kernel-name "(\n"
     "    __global const half* restrict A,\n    __global const half* restrict B,\n"
     "    __global " c-type "* restrict C,\n    int M, int N, int K"
     scalar-params
     epilogue-params
     ") {\n"
     "    int sg_id = get_sub_group_id();\n    int sg_lid = get_sub_group_local_id();\n"
     "    int sg_row = sg_id / " ncols ";\n    int sg_col = sg_id % " ncols ";\n"
     "    int m_base = get_group_id(1) * " block-m " + sg_row * " sg-m ";\n"
     (apply str
            (for [n ns]
              (str "    int n_base" n " = get_group_id(0) * " block-n " + sg_col * " sg-n
                   (when (pos? n) (str " + " (* n ni))) ";\n")))
     "    if (m_base >= M || n_base0 >= N) return;\n"
     grid-z-line
     k-range-lines
     (offset-line "A" :lhs)
     (offset-line "B" :rhs)
     (offset-line "C" :result)
     (apply str
            (for [m ms]
              (str "    float" mi " "
                   (str/join ", " (for [n ns] (str "acc" m n "=0.0f"))) ";\n")))
     "    int a_wb = K * 2, a_pb = K * 2;\n    int b_wb = N * 2, b_pb = N * 2;\n"
     "    ushort8 " (str/join ", " (for [m ms] (str "a" m))) ";\n"
     "    short8 " (str/join ", " (for [m ms] (str "sa" m))) ";\n"
     "    int8 " (str/join ", " (for [n ns] (str "bp" n))) ";\n"
     (apply str
            (for [p (range prefetch)
                  :let [position (if sliced-k?
                                   (if (zero? p)
                                     k-begin
                                     (str k-begin " + " (* p ki)))
                                   (str (* p ki)))]]
              (str "    if (" position " < " k-end ") {\n"
                   (apply str
                          (for [m ms]
                            (str "        intel_sub_group_2d_block_prefetch_16b_8r16x1c((__global void*)A, a_wb, M, a_pb, (int2)(" position ", " (amul m) "));\n")))
                   "    }\n")))
     "    int k = " k-begin ";\n"
     "    for (; k + " (dec block-k) " < " k-end "; k += " block-k ") {\n"
     (apply str
            (for [ks (range ksteps)]
              (kstep (if (zero? ks) "k" (str "k + " (* ks ki))))))
     "    }\n"
     "    for (; k < " k-end "; k += " ki ") {\n"
     (kstep "k")
     "    }\n"
     (apply str
            (for [m ms i (range mi)]
              (str "    { int row = m_base + " (+ (* m mi) i) ";\n      if (row < M) {\n"
                   (apply str
                          (for [n ns]
                            (let [acc-expr (str "acc" m n ".s" i)]
                              (str "        { int col = n_base" n " + sg_lid;\n          if (col < N) "
                                   (if epilogue
                                     (str "C[row*N+col] = " store-cast "("
                                          (epilogue acc-expr "row" "col") ");\n")
                                     (str "C[row*N+col] = " store-cast "(" acc-expr ");\n"))
                                   "        }\n"))))
                   "      }\n    }\n")))
     "}\n")))

(defn emit-matrix-kernel
  "Lower a verified f16 DPAS KernelBody directly to OpenCL C.

  ScalarRegion stores are lowered as part of this boundary. The optional target map contains only
  target naming policy; it cannot inject source or replace the store expression."
  ([kernel-name kernel-body]
   (emit-matrix-kernel kernel-name kernel-body {}))
  ([kernel-name kernel-body {:keys [parameter-names]}]
   (emit-plan kernel-name (matrix-plan kernel-body)
              (assoc (or (lower-store-region kernel-body) {})
                     :parameter-names parameter-names))))
