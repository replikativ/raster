(ns raster.compiler.backend.gpu.kernel-body-opencl
  "C-family lowering of verified scheduled KernelBody values.

  Matrix fragments retain their Intel OpenCL leaf. The general scalar/control path is shared by
  OpenCL, CUDA and HIP through thin target dialect descriptors; geometry is recovered from explicit
  index/operation IR, never from the source algorithm or a second schedule registry."
  (:require [clojure.set :as set]
            [clojure.string :as str]
            [clojure.walk :as walk]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [raster.compiler.backend.gpu.kernel-body-c-dialect :as c-dialect]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
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

;; ---------------------------------------------------------------------------
;; General scalar/control KernelBody lowering
;; ---------------------------------------------------------------------------

(def ^:private scalar-operation-kinds
  #{"ScalarCompute" "ScalarLoad" "ScalarStore" "Yield" "IfRegion" "ForLoop"
    "PipelineYield" "PipelinedFor"
    "Collective" "WorkgroupBarrier" "AsyncWorkgroupCopy" "AsyncCommit" "AsyncWait"})

(defn- scalar-body-operations
  [operations]
  (mapcat
   (fn [operation]
     (concat [operation]
             (when (record-kind? "IfRegion" operation)
               (concat (scalar-body-operations (:then-operations operation))
                       (scalar-body-operations (:else-operations operation))))
             (when (or (record-kind? "ForLoop" operation)
                       (record-kind? "PipelinedFor" operation))
               (scalar-body-operations (:operations operation)))))
   operations))

(defn- scalar-defined-ids
  [operations]
  (mapcat
   (fn [operation]
     (cond
       (contains? #{"ScalarCompute" "ScalarLoad" "Collective"}
                  (some-> operation class .getSimpleName))
       [(:id (:result operation))]

       (record-kind? "IfRegion" operation)
       (concat (map :id (:results operation))
               (scalar-defined-ids (:then-operations operation))
               (scalar-defined-ids (:else-operations operation)))

       (record-kind? "ForLoop" operation)
       (concat [(:id (:index operation))]
               (map (comp :id :binding) (:iter-args operation))
               (map :id (:results operation))
               (scalar-defined-ids (:operations operation)))

       (record-kind? "PipelinedFor" operation)
       (concat [(:id (:index operation))]
               (map (comp :id :binding) (:iter-args operation))
               (map :id (:results operation))
               (scalar-defined-ids (:operations operation)))

       :else []))
   operations))

(defn- scalar-local-name [id]
  (str "rstr_" (target-name id)))

(def ^:dynamic *scalar-dialect*
  "Validated target spelling used by the shared scalar/control emitter."
  (c-dialect/resolve! :opencl-intel))

(defn- target-type [type]
  (c-dialect/type-name *scalar-dialect* type))

(defn- emit-floating-literal [value suffix]
  (cond
    (Double/isNaN (double value)) "NAN"
    (= Double/POSITIVE_INFINITY (double value)) "INFINITY"
    (= Double/NEGATIVE_INFINITY (double value)) "(-INFINITY)"
    :else (str (Double/toString (double value)) suffix)))

(defn- emit-literal
  [{:keys [value type]}]
  (case (dtype/canon type)
    :byte (str "(" (target-type :byte) ")" value)
    :int (str value)
    :long (str value (if (c-dialect/opencl? *scalar-dialect*) "L" "LL"))
    :half (if (c-dialect/opencl? *scalar-dialect*)
            (str "(half)(" (emit-floating-literal value "f") ")")
            (str "__float2half_rn(" (emit-floating-literal value "f") ")"))
    :float (emit-floating-literal value "f")
    :double (emit-floating-literal value "")
    (throw (ex-info "C-family scalar literal has no target spelling"
                    {:reason :kernel-body-c-unimplemented
                     :dialect (:id *scalar-dialect*) :literal value :type type}))))

(defn- scalar-value-type
  [value types]
  (cond
    (record-kind? "Literal" value) (if (= :predicate (:type value))
                                     :predicate (dtype/canon (:type value)))
    (record-kind? "ScalarExpr" value) (if (= :predicate (:result-type value))
                                        :predicate (dtype/canon (:result-type value)))
    (value-id? value) (get types value)
    :else nil))

(declare emit-scalar-value)

(defn- cast-suffix
  [rounding overflow]
  (str (case overflow
         :saturate "_sat"
         (:wrap :trap :exact :ieee) "")
       (case rounding
         :toward-zero "_rtz"
         :nearest-even "_rte"
         :up "_rtp"
         :down "_rtn"
         :exact "")))

(defn- emit-cast
  [expression context]
  (let [argument (first (:arguments expression))
        source-type (scalar-value-type argument (:types context))
        result-type (dtype/canon (:result-type expression))
        {:keys [rounding overflow]} (:options expression)
        argument-source (emit-scalar-value argument context)
        source-fp? (dtype/fp-dtype? source-type)
        result-fp? (dtype/fp-dtype? result-type)
        narrowing-fp? (and source-fp? result-fp?
                           (> (dtype/bytes-of source-type) (dtype/bytes-of result-type)))]
    (cond
      ;; Same-width and widening FP conversions are exact in the KernelBody contract.
      (and source-fp? result-fp? (not narrowing-fp?)
           (= [:exact :exact] [rounding overflow]))
      (cond
        (and (not (c-dialect/opencl? *scalar-dialect*))
             (= source-type :half) (= result-type :float))
        (str "__half2float(" argument-source ")")

        (and (not (c-dialect/opencl? *scalar-dialect*))
             (= source-type :half) (= result-type :double))
        (str "(double)(__half2float(" argument-source "))")

        :else (str "(" (target-type result-type) ")(" argument-source ")"))

      ;; OpenCL conversion suffixes state the requested rounding of IEEE narrowing.
      (and narrowing-fp? (= :ieee overflow))
      (if (c-dialect/opencl? *scalar-dialect*)
        (str "convert_" (target-type result-type) (cast-suffix rounding overflow)
             "(" argument-source ")")
        (case [source-type result-type rounding]
          [:float :half :nearest-even] (str "__float2half_rn(" argument-source ")")
          [:double :float :nearest-even] (str "__double2float_rn(" argument-source ")")
          (throw (ex-info "CUDA/HIP cannot preserve this floating narrowing policy"
                          {:reason :kernel-body-c-cast-policy
                           :dialect (:id *scalar-dialect*)
                           :source-type source-type :result-type result-type
                           :rounding rounding :overflow overflow}))))

      ;; Saturating FP->integer conversion and wrapping integral conversion have direct spellings.
      (or (and source-fp? (not result-fp?) (= :saturate overflow))
          (and (not source-fp?) (not result-fp?) (= :wrap overflow)))
      (if (c-dialect/opencl? *scalar-dialect*)
        (str "convert_" (target-type result-type) (cast-suffix rounding overflow)
             "(" argument-source ")")
        (throw (ex-info "CUDA/HIP cannot preserve this saturating or wrapping cast policy"
                        {:reason :kernel-body-c-cast-policy
                         :dialect (:id *scalar-dialect*)
                         :source-type source-type :result-type result-type
                         :rounding rounding :overflow overflow})))

      ;; Integral widening is exact. Narrowing/exact and trapping conversions need a proof or
      ;; runtime check that this target layer does not currently carry.
      (and (not source-fp?) (not result-fp?) (= [:exact :exact] [rounding overflow])
           (<= (dtype/bytes-of source-type) (dtype/bytes-of result-type)))
      (str "(" (target-type result-type) ")(" argument-source ")")

      :else
      (throw (ex-info "C-family target cannot preserve this KernelBody cast policy"
                      {:reason :kernel-body-c-cast-policy
                       :dialect (:id *scalar-dialect*)
                       :source-type source-type :result-type result-type
                       :rounding rounding :overflow overflow})))))

(defn- emit-intrinsic-expression
  [expression context]
  (let [op (intrinsics/canonical (:op expression))
        lowering (intrinsics/op->c-lowering op false)
        arguments (mapv #(emit-scalar-value % context) (:arguments expression))
        operand-type (scalar-value-type (first (:arguments expression)) (:types context))
        integral? (contains? #{:byte :int :long} operand-type)]
    (cond
      ;; The shared C descriptor uses the floating spelling. OpenCL integer min/max are distinct
      ;; overloads, so target spelling must retain the verified operand dtype.
      (and integral? (contains? #{:min :max} op))
      (str (name op) "(" (str/join ", " arguments) ")")

      ;; Signed right shift is not the target-neutral unsigned-shift contract.
      (= :ushr op)
      (let [[value amount] arguments
            unsigned-type (c-dialect/unsigned-type-name *scalar-dialect* operand-type)]
        (str "(" (target-type operand-type) ")((" unsigned-type ")(" value ") >> "
             amount ")"))

      ;; OpenCL's integral abs returns an unsigned type, while KernelBody currently declares a
      ;; same-signed-type result. Refuse the mismatch until the IR states that representation step.
      (and (c-dialect/opencl? *scalar-dialect*) integral? (= :abs op))
      (throw (ex-info "OpenCL integral abs disagrees with the KernelBody result type"
                      {:reason :kernel-body-opencl-intrinsic
                       :operation op :operand-type operand-type}))

      :else
      (case (:kind lowering)
        :infix (str "(" (str/join (str " " (:op lowering) " ") arguments) ")")
        :fn (str (:op lowering) "(" (str/join ", " arguments) ")")
        :floored-mod (let [[a b] arguments]
                       (str "((" a " % " b " + " b ") % " b ")"))
        (throw (ex-info "OpenCL has no scalar spelling for a verified KernelBody intrinsic"
                        {:reason :kernel-body-opencl-intrinsic
                         :operation op :expression expression}))))))

(defn- emit-scalar-value
  [value {:keys [names] :as context}]
  (cond
    (record-kind? "Literal" value)
    (if (= :predicate (:type value)) (if (:value value) "true" "false") (emit-literal value))

    (value-id? value)
    (or (get names value)
        (throw (ex-info "OpenCL scalar lowering lost an SSA name"
                        {:reason :raster/bug :value value})))

    (record-kind? "ScalarExpr" value)
    (case (:op value)
      :cast (emit-cast value context)
      :select (let [[condition if-true if-false] (:arguments value)]
                (str "(" (emit-scalar-value condition context) " ? "
                     (emit-scalar-value if-true context) " : "
                     (emit-scalar-value if-false context) ")"))
      :isnan (str "isnan(" (emit-scalar-value (first (:arguments value)) context) ")")
      (emit-intrinsic-expression value context))

    :else
    (throw (ex-info "OpenCL scalar lowering received an untyped value"
                    {:reason :raster/bug :value value}))))

(defn- emit-layout-expression
  [expression names]
  (cond
    (number? expression) (str expression)
    (value-id? expression) (or (get names expression) (scalar-local-name expression))
    (seq? expression)
    (let [[op & arguments] expression
          operator ({'+ "+" '* "*" '- "-" 'quot "/"} op)]
      (when-not operator
        (throw (ex-info "OpenCL cannot lower a symbolic layout stride"
                        {:reason :kernel-body-opencl-layout :expression expression})))
      (str "(" (str/join (str " " operator " ")
                         (map #(emit-layout-expression % names) arguments)) ")"))
    :else
    (throw (ex-info "OpenCL cannot lower a layout expression"
                    {:reason :kernel-body-opencl-layout :expression expression}))))

(defn- emit-storage-index
  [storage coordinates names]
  (let [layout (:layout storage)
        dense-offset
        (fn [dense-layout]
          (let [strides (layout/resolve-strides dense-layout)
                terms (mapv (fn [coordinate stride]
                              (str "((long)(" (emit-index-expression coordinate names)
                                   ") * (long)(" (emit-layout-expression stride names) "))"))
                            coordinates strides)]
            (if (seq terms) (str/join " + " terms) "0")))
        local-offset
        (case (:kind layout)
          (:row-major :col-major) (dense-offset layout)
          :shared-memory
          (do
            (layout/validate-shared-memory! layout)
            (if (= :identity (:swizzle layout))
              (dense-offset layout)
              (let [[row column] coordinates
                    row-source (emit-index-expression row names)
                    column-source (emit-index-expression column names)
                    columns (second (:shape layout))
                    mask (dec (layout/swizzle-width (:swizzle layout)))]
                (str "((long)(" row-source ") * (long)(" columns ") + "
                     "((long)(" column-source ") ^ ((long)(" row-source ") & " mask ")))"))))
          (throw (ex-info "C-family scalar memory lowering requires a dense strided layout or verified shared layout"
                          {:reason :kernel-body-opencl-layout :storage (:id storage)
                           :layout layout})))
        view-offset (some-> storage :view :element-offset)]
    (if view-offset
      (str "((long)(" (emit-index-expression view-offset names) ") + " local-offset ")")
      (str "(" local-offset ")"))))

(defn- emit-mask-predicate
  [predicate names]
  (let [arguments (:arguments predicate)]
    (case (:op predicate)
      :lt (emit-infix "<" arguments names)
      :lte (emit-infix "<=" arguments names)
      :eq (emit-infix "==" arguments names)
      :and (str "(" (str/join " && " (map #(emit-mask-predicate % names) arguments)) ")")
      :or (str "(" (str/join " || " (map #(emit-mask-predicate % names) arguments)) ")")
      :not (str "(!" (emit-mask-predicate (first arguments) names) ")")
      (throw (ex-info "OpenCL cannot lower a KernelBody mask predicate"
                      {:reason :raster/bug :predicate predicate})))))

(defn- emit-mask
  [mask-id {:keys [masks names]}]
  (when mask-id
    (let [predicates (:predicates (get masks mask-id))]
      (str "(" (str/join " && " (map #(emit-mask-predicate % names) predicates)) ")"))))

(defn- indent-lines [depth source]
  (let [prefix (apply str (repeat depth "  "))]
    (apply str (map #(str prefix % "\n") (str/split-lines source)))))

(declare emit-scalar-operations)

(defn- add-value
  [context spec]
  (-> context
      (assoc-in [:names (:id spec)] (scalar-local-name (:id spec)))
      (assoc-in [:types (:id spec)] (if (= :predicate (:type spec))
                                      :predicate (dtype/canon (:type spec))))))

(defn- emit-yield-assignments
  [results values context depth]
  (apply str
         (map (fn [result value]
                (indent-lines depth
                              (str (get-in context [:names (:id result)]) " = "
                                   (emit-scalar-value value context) ";")))
              results values)))

(defn- async-group-events
  [context group]
  (or (:events group)
      (into [] (keep #(get-in context [:async-copies %])) (:copies group))))

(defn- emit-layout-aware-cooperative-copy
  [operation context source-storage destination-storage source-base destination-base copy-name]
  (let [destination-layout (:layout destination-storage)
        [row column] (:destination-coordinates operation)
        row-source (emit-index-expression row (:names context))
        column-source (emit-index-expression column (:names context))
        source-index (emit-storage-index source-storage (:source-coordinates operation)
                                         (:names context))
        columns (second (:shape destination-layout))
        mask (dec (layout/swizzle-width (:swizzle destination-layout)))
        offset (str copy-name "_element_offset")
        thread-id (c-dialect/workgroup-linear-thread-id
                   *scalar-dialect* (:workgroup-shape context))]
    (str "for (int " offset " = " thread-id "; " offset " < " (:elements operation)
         "; " offset " += " (:workgroup-width context) ") {\n"
         "  " destination-base "[((long)(" row-source ") * (long)(" columns ") + "
         "((long)(" column-source ") + " offset ") ^ ((long)(" row-source ") & " mask
         "))] = " source-base "[((long)(" source-index ") + " offset ")];\n"
         "}")))

(defn- emit-scalar-operation
  [operation context depth]
  (cond
    (record-kind? "ScalarCompute" operation)
    (let [result (:result operation)
          next-context (add-value context result)]
      [(indent-lines depth
                     (str (target-type (get-in next-context [:types (:id result)])) " "
                          (get-in next-context [:names (:id result)]) " = "
                          (emit-scalar-value (:expression operation) context) ";"))
       next-context])

    (record-kind? "ScalarLoad" operation)
    (let [result (:result operation)
          storage (get-in context [:storage (:buffer operation)])
          next-context (add-value context result)
          base-name (get-in context [:names (or (some-> storage :view :buffer)
                                                (:id storage))])
          index (emit-storage-index storage (:coordinates operation) (:names context))
          load (str base-name "[" index "]")
          source (if-let [predicate (emit-mask (:predicate operation) context)]
                   (str "(" predicate " ? " load " : "
                        (emit-scalar-value (:other operation) context) ")")
                   load)]
      [(indent-lines depth
                     (str (target-type (get-in next-context [:types (:id result)])) " "
                          (get-in next-context [:names (:id result)]) " = " source ";"))
       next-context])

    (record-kind? "ScalarStore" operation)
    (let [storage (get-in context [:storage (:buffer operation)])
          base-name (get-in context [:names (or (some-> storage :view :buffer)
                                                (:id storage))])
          assignment (str base-name "["
                          (emit-storage-index storage (:coordinates operation) (:names context))
                          "] = " (emit-scalar-value (:value operation) context) ";")]
      [(if-let [predicate (emit-mask (:predicate operation) context)]
         (str (indent-lines depth (str "if (" predicate ") {"))
              (indent-lines (inc depth) assignment)
              (indent-lines depth "}"))
         (indent-lines depth assignment))
       context])

    (record-kind? "IfRegion" operation)
    (let [results (:results operation)
          result-context (reduce add-value context results)
          declarations (apply str
                              (for [result results]
                                (indent-lines depth
                                              (str (target-type
                                                    (get-in result-context
                                                            [:types (:id result)]))
                                                   " "
                                                   (get-in result-context [:names (:id result)])
                                                   ";"))))
          then-ops (pop (:then-operations operation))
          else-ops (pop (:else-operations operation))
          then-yield (peek (:then-operations operation))
          else-yield (peek (:else-operations operation))
          [then-source then-context] (emit-scalar-operations then-ops context (inc depth))
          [else-source else-context] (emit-scalar-operations else-ops context (inc depth))]
      [(str declarations
            (indent-lines depth (str "if (" (get-in context [:names (:condition operation)]) ") {"))
            then-source
            (emit-yield-assignments results (:values then-yield)
                                    (update then-context :names merge (:names result-context))
                                    (inc depth))
            (indent-lines depth "} else {")
            else-source
            (emit-yield-assignments results (:values else-yield)
                                    (update else-context :names merge (:names result-context))
                                    (inc depth))
            (indent-lines depth "}"))
       result-context])

    (record-kind? "ForLoop" operation)
    (let [results (:results operation)
          result-context (reduce add-value context results)
          index (:index operation)
          loop-context (add-value result-context index)
          loop-context (reduce (fn [ctx arg] (add-value ctx (:binding arg)))
                               loop-context (:iter-args operation))
          initializers (apply str
                              (map (fn [result arg]
                                     (indent-lines depth
                                                   (str (target-type
                                                         (get-in result-context
                                                                 [:types (:id result)]))
                                                        " "
                                                        (get-in result-context
                                                                [:names (:id result)])
                                                        " = "
                                                        (emit-scalar-value (:initial arg) context)
                                                        ";")))
                                   results (:iter-args operation)))
          index-name (get-in loop-context [:names (:id index)])
          bindings (apply str
                          (map (fn [arg result]
                                 (let [binding (:binding arg)]
                                   (indent-lines (inc depth)
                                                 (str (target-type
                                                       (get-in loop-context
                                                               [:types (:id binding)]))
                                                      " "
                                                      (get-in loop-context
                                                              [:names (:id binding)])
                                                      " = "
                                                      (get-in result-context
                                                              [:names (:id result)])
                                                      ";"))))
                               (:iter-args operation) results))
          body-operations (pop (:operations operation))
          yield-op (peek (:operations operation))
          [body-source body-context] (emit-scalar-operations body-operations loop-context
                                                             (inc depth))
          loop-header (str "for (" (target-type (get-in loop-context [:types (:id index)]))
                           " " index-name " = "
                           (emit-index-expression (:lower operation) (:names context)) "; "
                           index-name " < "
                           (emit-index-expression (:upper operation) (:names context)) "; "
                           index-name " += " (:step operation) ") {")]
      [(str initializers
            (when (get-in operation [:attributes :unroll])
              (indent-lines depth "#pragma unroll"))
            (indent-lines depth loop-header)
            bindings body-source
            (emit-yield-assignments results (:values yield-op)
                                    (update body-context :names merge (:names result-context))
                                    (inc depth))
            (indent-lines depth "}"))
       result-context])

    (record-kind? "PipelinedFor" operation)
    (let [results (:results operation)
          result-context (reduce add-value context results)
          index (:index operation)
          loop-context (add-value result-context index)
          loop-context (reduce (fn [ctx arg] (add-value ctx (:binding arg)))
                               loop-context (:iter-args operation))
          initializers
          (apply str
                 (map (fn [result arg]
                        (indent-lines depth
                                      (str (target-type
                                            (get-in result-context [:types (:id result)]))
                                           " " (get-in result-context [:names (:id result)])
                                           " = " (emit-scalar-value (:initial arg) context) ";")))
                      results (:iter-args operation)))
          initial-groups (:async-groups context)
          native-events? (= :native-events (c-dialect/async-copy-mode *scalar-dialect*))
          binding-groups
          (mapv
           (fn [arg group]
             (let [binding (:binding arg)
                   event-count (count (:copies group))
                   events (mapv #(str (scalar-local-name binding) "_event_" %)
                                (range event-count))]
               {:id binding :copies (:copies group) :events events}))
           (:async-iter-args operation) initial-groups)
          event-initializers
          (when native-events?
            (apply str
                   (mapcat
                    (fn [binding-group initial-group]
                      (map (fn [binding-event initial-event]
                             (indent-lines depth
                                           (str "event_t " binding-event " = " initial-event ";")))
                           (:events binding-group)
                           (async-group-events context initial-group)))
                    binding-groups initial-groups)))
          loop-context (assoc loop-context :async-groups binding-groups)
          index-name (get-in loop-context [:names (:id index)])
          scalar-bindings
          (apply str
                 (map (fn [arg result]
                        (let [binding (:binding arg)]
                          (indent-lines (inc depth)
                                        (str (target-type
                                              (get-in loop-context [:types (:id binding)]))
                                             " " (get-in loop-context [:names (:id binding)])
                                             " = " (get-in result-context [:names (:id result)])
                                             ";"))))
                      (:iter-args operation) results))
          body-operations (pop (:operations operation))
          yield-op (peek (:operations operation))
          [body-source body-context] (emit-scalar-operations body-operations loop-context
                                                             (inc depth))
          yielded-by-id (into {} (map (juxt :id identity)) (:async-groups body-context))
          yielded-groups (mapv yielded-by-id (:groups yield-op))
          next-event-names
          (when native-events?
            (mapv (fn [binding-group group]
                    (mapv (fn [event-index event]
                            [(str (scalar-local-name (:id binding-group))
                                  "_next_event_" event-index)
                             event])
                          (range) (async-group-events body-context group)))
                  binding-groups yielded-groups))
          event-backedge
          (when native-events?
            (str
             (apply str
                    (for [[temporary source-event] (mapcat identity next-event-names)]
                      (indent-lines (inc depth)
                                    (str "event_t " temporary " = " source-event ";"))))
             (apply str
                    (mapcat
                     (fn [binding-group temporaries]
                       (map (fn [binding-event [temporary _]]
                              (indent-lines (inc depth)
                                            (str binding-event " = " temporary ";")))
                            (:events binding-group) temporaries))
                     binding-groups next-event-names))))
          loop-header (str "for (" (target-type (get-in loop-context [:types (:id index)]))
                           " " index-name " = "
                           (emit-index-expression (:lower operation) (:names context)) "; "
                           index-name " < "
                           (emit-index-expression (:upper operation) (:names context)) "; "
                           index-name " += " (:step operation) ") {")
          output-groups
          (mapv (fn [result binding-group]
                  (assoc binding-group :id result))
                (:async-results operation) binding-groups)
          next-context (assoc result-context :async-groups output-groups
                              :async-issued [])]
      [(str initializers event-initializers
            (when (get-in operation [:attributes :unroll])
              (indent-lines depth "#pragma unroll"))
            (indent-lines depth loop-header)
            scalar-bindings body-source
            (emit-yield-assignments results (:values yield-op)
                                    (update body-context :names merge (:names result-context))
                                    (inc depth))
            event-backedge
            (indent-lines depth "}"))
       next-context])

    (record-kind? "Collective" operation)
    (let [result (:result operation)
          next-context (add-value context result)
          input (emit-scalar-value (:input operation) context)
          result-name (get-in next-context [:names (:id result)])
          result-type (get-in next-context [:types (:id result)])
          width (:width operation)
          source
          (case (:kind operation)
            :broadcast
            (indent-lines depth
                          (str (target-type result-type) " " result-name " = "
                               (c-dialect/broadcast-expression
                                *scalar-dialect* input (:source-lane operation) width) ";"))

            :reduce
            (let [operator (intrinsics/canonical (:operator operation))
                  association (:association operation)]
              (if (c-dialect/opencl? *scalar-dialect*)
                (let [_ (when-not (= :implementation-defined association)
                          (throw
                           (ex-info
                            "OpenCL subgroup builtin cannot preserve an explicit reduction tree"
                            {:reason :kernel-body-opencl-collective-association
                             :association association})))
                      builtin (c-dialect/opencl-reduction-builtin operator)]
                  (when-not builtin
                    (throw (ex-info "OpenCL has no matching subgroup reduction builtin"
                                    {:reason :kernel-body-opencl-collective
                                     :operator (:operator operation)})))
                  (indent-lines depth
                                (str (target-type result-type) " " result-name " = "
                                     builtin "(" input ");")))
                (let [distances (if (= :implementation-defined association)
                                  (when (zero? (bit-and width (dec width)))
                                    (vec (take-while pos?
                                                     (iterate #(quot % 2) (quot width 2)))))
                                  (:distances association))
                      _ (when-not (seq distances)
                          (throw (ex-info "CUDA/HIP subgroup reduction requires a power-of-two tree"
                                          {:reason :kernel-body-c-collective-association
                                           :dialect (:id *scalar-dialect*)
                                           :width width :association association})))
                      combine (fn [rhs]
                                (case operator
                                  :+ (str result-name " += " rhs ";")
                                  :* (str result-name " *= " rhs ";")
                                  :bit-and (str result-name " &= " rhs ";")
                                  :bit-or (str result-name " |= " rhs ";")
                                  :bit-xor (str result-name " ^= " rhs ";")
                                  (:min :max)
                                  (let [fn-name (case [operator result-type]
                                                  [:min :float] "fminf"
                                                  [:max :float] "fmaxf"
                                                  [:min :double] "fmin"
                                                  [:max :double] "fmax"
                                                  nil)]
                                    (when-not fn-name
                                      (throw (ex-info "CUDA/HIP min/max collective dtype is unsupported"
                                                      {:reason :kernel-body-c-collective
                                                       :dialect (:id *scalar-dialect*)
                                                       :operator operator :dtype result-type})))
                                    (str result-name " = " fn-name "(" result-name ", " rhs ");"))
                                  (throw (ex-info "CUDA/HIP has no matching subgroup reduction"
                                                  {:reason :kernel-body-c-collective
                                                   :dialect (:id *scalar-dialect*)
                                                   :operator operator}))))]
                  (str (indent-lines depth
                                     (str (target-type result-type) " " result-name " = " input ";"))
                       (apply str
                              (for [distance distances]
                                (indent-lines depth
                                              (combine
                                               (c-dialect/shuffle-down-expression
                                                *scalar-dialect* result-name distance width)))))
                       (indent-lines depth
                                     (str result-name " = "
                                          (c-dialect/broadcast-expression
                                           *scalar-dialect* result-name 0 width) ";")))))))]
      [source next-context])

    (record-kind? "WorkgroupBarrier" operation)
    [(indent-lines depth (c-dialect/workgroup-barrier *scalar-dialect*)) context]

    (record-kind? "AsyncWorkgroupCopy" operation)
    (let [destination-storage (get-in context [:storage (:destination operation)])
          layout-aware? (and (layout/shared-memory-layout? (:layout destination-storage))
                             (not= :identity (get-in destination-storage [:layout :swizzle])))
          mode (if layout-aware?
                 :synchronous-layout-aware
                 (c-dialect/async-copy-mode *scalar-dialect*))
          _ (when (and (= :required (:overlap operation))
                       (contains? #{:synchronous-cooperative :synchronous-layout-aware} mode))
              (throw (ex-info "target cannot preserve required async-copy overlap"
                              {:reason :kernel-body-c-async-overlap
                               :dialect (:id *scalar-dialect*)
                               :operation operation :lowering mode})))
          source-storage (get-in context [:storage (:source operation)])
          source-base (get-in context [:names (or (some-> source-storage :view :buffer)
                                                  (:id source-storage))])
          destination-base (get-in context [:names (:id destination-storage)])
          source-index (emit-storage-index source-storage (:source-coordinates operation)
                                           (:names context))
          destination-index (emit-storage-index destination-storage
                                                (:destination-coordinates operation)
                                                (:names context))
          copy-name (scalar-local-name (:id operation))
          source (str "(&" source-base "[" source-index "])")
          destination (str "(&" destination-base "[" destination-index "])")
          source-code (if layout-aware?
                        (emit-layout-aware-cooperative-copy
                         operation context source-storage destination-storage
                         source-base destination-base copy-name)
                        (c-dialect/async-copy
                         *scalar-dialect*
                         {:name copy-name :source source :destination destination
                          :elements (:elements operation)
                          :element-bytes (dtype/bytes-of (:dtype source-storage))
                          :transfer-bytes (:transfer-bytes operation)
                          :overlap (:overlap operation)
                          :workgroup-width (:workgroup-width context)}))]
      [(indent-lines depth source-code)
       (-> context
           (assoc-in [:async-copies (:id operation)] (when-not layout-aware? copy-name))
           (update :async-issued conj (:id operation)))])

    (record-kind? "AsyncCommit" operation)
    [(indent-lines depth (c-dialect/async-commit *scalar-dialect* (:id operation)))
     (-> context
         (assoc :async-issued [])
         (update :async-groups conj {:id (:id operation) :copies (:copies operation)}))]

    (record-kind? "AsyncWait" operation)
    (let [group-count (count (:groups operation))
          waited (subvec (:async-groups context) 0 group-count)
          event-names (mapv identity (mapcat #(async-group-events context %) waited))]
      [(indent-lines depth
                     (c-dialect/async-wait *scalar-dialect* event-names
                                           (:pending-groups operation)))
       (assoc context :async-groups (subvec (:async-groups context) group-count))])

    (record-kind? "Yield" operation)
    (throw (ex-info "OpenCL scalar lowering found a misplaced Yield"
                    {:reason :raster/bug :operation operation}))

    (record-kind? "PipelineYield" operation)
    (throw (ex-info "OpenCL scalar lowering found a misplaced PipelineYield"
                    {:reason :raster/bug :operation operation}))

    :else
    (throw (ex-info "OpenCL scalar lowering found an unsupported KernelBody operation"
                    {:reason :kernel-body-opencl-unimplemented :operation operation}))))

(defn- emit-scalar-operations
  [operations context depth]
  (reduce (fn [[source ctx] operation]
            (let [[next-source next-context] (emit-scalar-operation operation ctx depth)]
              [(str source next-source) next-context]))
          ["" context] operations))

(defn- scalar-storage-map
  [kernel-body]
  (let [parameters (into {} (map (juxt :id identity)) (:parameters kernel-body))
        allocations (into {}
                          (map (fn [allocation]
                                 [(:id allocation)
                                  (assoc allocation :kind :allocation
                                         :memory-space :workgroup)]))
                          (:allocations kernel-body))]
    (reduce (fn [storage view]
              (let [parent (get parameters (:buffer view))]
                (assoc storage (:id view)
                       (assoc parent :id (:id view) :shape (:shape view)
                              :layout (:layout view) :view view))))
            (merge parameters allocations) (:views kernel-body))))

(defn- emit-scalar-kernel*
  [kernel-name kernel-body {:keys [parameter-names]}]
  (let [kernel-body (body/validate! kernel-body)
        operations (vec (scalar-body-operations (:operations kernel-body)))
        unsupported (remove #(contains? scalar-operation-kinds
                                        (some-> % class .getSimpleName))
                            operations)
        _ (when (seq (:fragments kernel-body))
            (throw (ex-info "scalar OpenCL lowering cannot consume matrix fragments"
                            {:reason :kernel-body-opencl-unimplemented
                             :fragments (:fragments kernel-body)})))
        _ (when (seq unsupported)
            (throw (ex-info "scalar OpenCL lowering cannot consume legacy tile operations"
                            {:reason :kernel-body-opencl-unimplemented
                             :operations (vec unsupported)})))
        parameters (:parameters kernel-body)
        parameter-names (into {}
                              (map (fn [parameter]
                                     [(:id parameter)
                                      (or (get parameter-names (:id parameter))
                                          (scalar-local-name (:id parameter)))])
                                   parameters))
        allocation-names (into {}
                               (map (fn [allocation]
                                      [(:id allocation) (scalar-local-name (:id allocation))]))
                               (:allocations kernel-body))
        _ (when-not (every? #(re-matches #"[A-Za-z_][A-Za-z0-9_]*" %)
                            (vals parameter-names))
            (throw (ex-info "C-family kernel parameter names must be C identifiers"
                            {:reason :kernel-body-c-parameter-name
                             :dialect (:id *scalar-dialect*)
                             :parameter-names parameter-names})))
        indices (:indices kernel-body)
        names (reduce (fn [env index]
                        (assoc env (:id index) (scalar-local-name (:id index))))
                      (merge parameter-names allocation-names) indices)
        types (reduce (fn [env index] (assoc env (:id index) :int))
                      (into {}
                            (map (fn [parameter]
                                   [(:id parameter) (dtype/canon (:dtype parameter))])
                                 (filter #(= :scalar (:kind %)) parameters)))
                      indices)
        all-names (vals names)
        _ (when-not (= (count all-names) (count (set all-names)))
            (throw (ex-info "C-family parameter and index names collide after target spelling"
                            {:reason :kernel-body-c-name-collision
                             :dialect (:id *scalar-dialect*) :names names})))
        local-ids (concat (map :id (:allocations kernel-body))
                          (map :id indices) (scalar-defined-ids (:operations kernel-body))
                          (keep #(when (contains? #{"AsyncWorkgroupCopy" "AsyncCommit"}
                                                  (some-> % class .getSimpleName))
                                   (:id %))
                                operations))
        emitted-local-names (map scalar-local-name local-ids)
        all-emitted-names (concat (vals parameter-names) emitted-local-names)
        _ (when-not (= (count all-emitted-names) (count (set all-emitted-names)))
            (throw (ex-info "C-family parameter or SSA names collide after target spelling"
                            {:reason :kernel-body-c-name-collision
                             :dialect (:id *scalar-dialect*)
                             :ids (vec local-ids)})))
        storage (scalar-storage-map kernel-body)
        masks (into {} (map (juxt :id identity)) (:masks kernel-body))
        context {:names names :types types :storage storage :masks masks
                 :workgroup-shape (get-in kernel-body [:launch :workgroup-size])
                 :workgroup-width (reduce * (get-in kernel-body [:launch :workgroup-size]))
                 :async-copies {} :async-issued [] :async-groups []}
        allocation-plan (body/workgroup-memory-plan (:allocations kernel-body))
        arena-name "rstr_workgroup_memory"
        allocation-source
        (when (seq (:allocations kernel-body))
          (str (indent-lines
                1 (c-dialect/workgroup-arena-declaration
                   *scalar-dialect* arena-name (:bytes allocation-plan)
                   (:alignment allocation-plan)))
               (apply str
                      (for [{:keys [allocation byte-offset]} (:allocations allocation-plan)]
                        (indent-lines
                         1 (c-dialect/workgroup-pointer-declaration
                            *scalar-dialect* (:dtype allocation)
                            (get allocation-names (:id allocation)) arena-name byte-offset))))))
        index-source
        (apply str
               (for [index indices]
                 (let [name (get names (:id index))]
                   (if (record-kind? "IndexBinding" index)
                     (indent-lines
                      1
                      (str "int " name " = (int)"
                           (c-dialect/index-binding
                            *scalar-dialect* (:source index) (:axis index)
                            (get-in kernel-body [:schedule :subgroup-size]))
                           ";"))
                     (indent-lines
                      1 (str "int " name " = "
                             (emit-index-expression (:expression index) names) ";"))))))
        [operation-source _] (emit-scalar-operations (:operations kernel-body) context 1)
        storage-declarations (concat parameters (:allocations kernel-body))
        uses-half? (some #(= :half (dtype/canon (:dtype %))) storage-declarations)
        uses-double? (some #(= :double (dtype/canon (:dtype %))) storage-declarations)
        collective (first (filter #(record-kind? "Collective" %) operations))
        uses-subgroups? (or collective
                            (some #(and (record-kind? "IndexBinding" %)
                                        (contains? #{:subgroup :lane} (:source %)))
                                  indices))
        subgroup-size (or (some-> collective :width)
                          (get-in kernel-body [:schedule :subgroup-size]))
        attribute (c-dialect/subgroup-attribute
                   *scalar-dialect* subgroup-size uses-subgroups?)]
    (str (c-dialect/preamble *scalar-dialect*
                             {:uses-half? uses-half? :uses-double? uses-double?
                              :uses-subgroups? uses-subgroups?})
         attribute
         (c-dialect/entry-prefix *scalar-dialect*) (target-name kernel-name) "(\n    "
         (str/join ",\n    "
                   (map #(c-dialect/parameter-declaration
                          *scalar-dialect* % (get parameter-names (:id %)))
                        parameters))
         ") {\n"
         allocation-source index-source operation-source
         "}\n")))

(defn emit-scalar-kernel
  "Lower a verified scalar/control KernelBody directly to a C-family target dialect.

  The body already fixes schedules, types, layouts, masks, convergence and numerical conversion
  policies. `parameter-names` may only select ABI spelling; `target-dialect` defaults to the
  production Intel OpenCL row and may select `:opencl-portable`, `:cuda`, or `:hip`. Target
  selection cannot alter the scheduled body. `target-features` may only select a capability-gated
  physical implementation (for example Ampere `cp.async`) with the same verified semantics."
  ([kernel-name kernel-body]
   (emit-scalar-kernel kernel-name kernel-body {}))
  ([kernel-name kernel-body {:keys [target-dialect subgroup-attribute target-features] :as options
                             :or {target-dialect :opencl-intel subgroup-attribute :intel}}]
   (let [target-dialect (if (and (= :opencl-intel target-dialect)
                                 (= :none subgroup-attribute))
                          :opencl-portable
                          target-dialect)]
     (binding [*scalar-dialect*
               (merge (c-dialect/resolve! target-dialect)
                      (select-keys target-features [:compute-capability :architecture]))]
       (emit-scalar-kernel* kernel-name kernel-body options)))))
