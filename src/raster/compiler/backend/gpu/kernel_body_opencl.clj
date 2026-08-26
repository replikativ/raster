(ns raster.compiler.backend.gpu.kernel-body-opencl
  "Direct OpenCL lowering of verified scheduled KernelBody values.

  This namespace is below semantic contraction analysis and hardware scheduling.  It chooses only
  target spellings: Intel subgroup builtins, OpenCL declarations and a pipelined loop form.  Tile
  geometry and fragment topology are recovered from explicit index/operation IR, never from the
  source contraction or a second schedule registry."
  (:require [clojure.string :as str]
            [raster.compiler.ir.kernel-body :as body]))

(defn- record-kind? [simple-name value]
  (= (str "raster.compiler.ir.kernel_body." simple-name)
     (some-> value class .getName)))

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
    (symbol? expression) #{expression}
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
    (or (number? expression) (symbol? expression)) 0
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
    (symbol? expression) {:constant 0 :coefficients {expression 1}}
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
        {:keys [parameters indices masks fragments operations]} kernel-body
        parameters-by-role (group-by :role parameters)
        lhs-param (only! "lhs parameter" (:lhs parameters-by-role))
        rhs-param (only! "rhs parameter" (:rhs parameters-by-role))
        out-param (only! "result parameter" (:result parameters-by-role))
        [[M K] [K' N] [M' N']] (map :shape [lhs-param rhs-param out-param])
        _ (require! (= [M K N] [M' K' N'])
                    "matrix parameter shapes do not compose" {:parameters parameters})
        _ (require! (= #{'M 'N 'K} (set (map :id (:dimension parameters-by-role))))
                    "matrix body requires the ordered M/N/K dimension ABI"
                    {:dimensions (:dimension parameters-by-role)})
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
        _ (require! (and (= 0 (:lower outer-loop)) (= 'K (:upper outer-loop))
                         (= outer-index (:lower inner-loop))
                         (= :min (get-in inner-loop [:upper :op]))
                         (affine= (first (get-in inner-loop [:upper :arguments]))
                                  block-k {outer-index 1})
                         (= 'K (second (get-in inner-loop [:upper :arguments]))))
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
        _ (require! (and (every? #(= (:id lhs-param) (:buffer %)) lhs-loads)
                         (every? #(= (:id rhs-param) (:buffer %)) rhs-loads)
                         (every? #(= (:id out-param) (:buffer %)) stores))
                    "matrix operations are not connected to their role parameters"
                    {:lhs lhs-param :rhs rhs-param :out out-param})
        _ (require! (every?
                     true?
                     (map-indexed
                      #(affine= (:expression %2) (* %1 ni)
                                {group-n block-n (:id subgroup-col) sg-n})
                      n-base-computes))
                    "N tile origins do not match the rhs fragment order"
                    {:origins n-base-computes})
        _ (require! (and (integer? prefetch-distance) (pos? prefetch-distance)
                         (every? #(and (= (:id lhs-param) (:buffer %))
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
        _ (require! (and (= 2 (count guard-predicates))
                         (some #(lt-affine? % 0 {m-base-id 1} 'M) guard-predicates)
                         (some #(lt-affine? % 0 {(first n-base-ids) 1} 'N)
                               guard-predicates))
                    "tile guard mask does not cover the M/N tile origins"
                    {:guard guard :mask (get mask-map (:mask guard))})
        load-mask (only! "matrix-load mask" (distinct (map :mask loads)))
        load-predicates (predicates-of load-mask)
        _ (require! (and (= 1 (count load-predicates))
                         (lt-affine? (first load-predicates) 0 {inner-index 1} 'K))
                    "matrix loads require the fragment-index K mask"
                    {:mask (get mask-map load-mask)})
        prefetch-mask (only! "matrix-prefetch mask" (distinct (map :mask prefetches)))
        prefetch-predicates (predicates-of prefetch-mask)
        _ (require! (and (= 1 (count prefetch-predicates))
                         (lt-affine? (first prefetch-predicates)
                                     (* prefetch-distance ki) {inner-index 1} 'K))
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
                   (some #(lt-affine? % (* m mi) {m-base-id 1} 'M) predicates)
                   (some #(lt-affine? % 0 {(nth n-base-ids n) 1 lane-id 1} 'N)
                         predicates))))
           "tile-store masks do not match their fragment coordinates"
           {:stores stores :masks masks})]
    {:mi mi :ni ni :ki ki :subgroup subgroup
     :block-m block-m :block-n block-n :block-k block-k :sg-m sg-m :sg-n sg-n
     :ncols ncols :lhs-ids lhs-ids :rhs-ids rhs-ids :mad-by-operands mad-by-operands
     :stores stores :prefetch prefetch-distance}))

(defn- emit-plan
  [kernel-name {:keys [mi ni ki subgroup block-m block-n block-k sg-m sg-n
                       ncols lhs-ids rhs-ids mad-by-operands stores prefetch]}
   {:keys [epilogue epilogue-params]}]
  (let [nms (count lhs-ids)
        nns (count rhs-ids)
        ksteps (quot block-k ki)
        acc-id (fn [m n] (:accumulator (get mad-by-operands
                                            [(nth lhs-ids m) (nth rhs-ids n)])))
        store-by-fragment (into {} (map (juxt :fragment identity)) stores)
        ms (range nms)
        ns (range nns)
        amul (fn [m] (if (zero? m) "m_base" (str "m_base+" (* m mi))))
        kstep (fn [kpos]
                (str "        { int pk = " kpos " + " (* prefetch ki) ";\n"
                     "          if (pk < K) {\n"
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
     "    __global half* restrict C,\n    int M, int N, int K"
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
     (apply str
            (for [m ms]
              (str "    float" mi " "
                   (str/join ", " (for [n ns] (str "acc" m n "=0.0f"))) ";\n")))
     "    int a_wb = K * 2, a_pb = K * 2;\n    int b_wb = N * 2, b_pb = N * 2;\n"
     "    ushort8 " (str/join ", " (for [m ms] (str "a" m))) ";\n"
     "    short8 " (str/join ", " (for [m ms] (str "sa" m))) ";\n"
     "    int8 " (str/join ", " (for [n ns] (str "bp" n))) ";\n"
     (apply str
            (for [p (range prefetch)]
              (str "    if (" (* p ki) " < K) {\n"
                   (apply str
                          (for [m ms]
                            (str "        intel_sub_group_2d_block_prefetch_16b_8r16x1c((__global void*)A, a_wb, M, a_pb, (int2)(" (* p ki) ", " (amul m) "));\n")))
                   "    }\n")))
     "    int k = 0;\n"
     "    for (; k + " (dec block-k) " < K; k += " block-k ") {\n"
     (apply str
            (for [ks (range ksteps)]
              (kstep (if (zero? ks) "k" (str "k + " (* ks ki))))))
     "    }\n"
     "    for (; k < K; k += " ki ") {\n"
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
                                     (str "C[row*N+col] = (half)("
                                          (epilogue acc-expr "row" "col") ");\n")
                                     (str "C[row*N+col] = (half)(" acc-expr ");\n"))
                                   "        }\n"))))
                   "      }\n    }\n")))
     "}\n")))

(defn emit-matrix-kernel
  "Lower a verified f16 DPAS KernelBody directly to OpenCL C.

  `target-store` is the already type-checked scalar-region target spelling
  `{:epilogue fn :epilogue-params string}`.  It is nil for an identity store."
  [kernel-name kernel-body target-store]
  (emit-plan kernel-name (matrix-plan kernel-body) (or target-store {})))
