(ns raster.compiler.backend.gpu.indexed-attention
  "Direct correctness lowering for recognized indexed graph attention.

   One work-item owns one destination/feature output and scans the edge list. The schedule is
   intentionally simple, but it is genuinely fused: scores, clamped exponential weights,
   denominator and weighted values remain private scalars and no edge-sized intermediates are
   materialized."
  (:require [clojure.string :as str]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.ir.segmented-weighted-reduction :as swr]))

(defn- fail
  [message reason data]
  (throw (ex-info message (assoc data :reason reason))))

(defn- indexed-attention-plan!
  [plan]
  (let [{:keys [segment-axes membership storage score weight value numerator denominator
                normalization operands output accumulator-dtype provenance]
         :as plan} (swr/validate! plan)
        [destination-axis head-axis] segment-axes
        [q k v destination-indices source-indices] operands
        score-arguments (:arguments score)
        [scale lower upper] score-arguments
        bound (:value upper)]
    (when-not
     (and (= [:destination :head] (mapv :name segment-axes))
          (= :edge-list-by-destination (:kind membership))
          (= :multiset (:duplicate-policy membership))
          (= [(:id destination-indices) (:id source-indices)] (:buffers membership))
          (= (:id destination-indices) (:destination-indices membership))
          (= (:id source-indices) (:source-indices membership))
          (= :indexed-dense-values (:kind storage))
          (= [(:id q) (:id k) (:id v)] (:buffers storage))
          (= (:extent destination-axis) (:entity-count storage))
          (= :dot (:kind score))
          (= {:name :head-component :extent (:components value)} (:axis score))
          (= {:kind :identity :heads (:extent head-axis)} (:head-map score))
          (= {:kind :indexed-query :buffer (:id q)
              :indices (:id destination-indices) :dtype (:dtype q)
              :total-dim (:total-dim storage)}
             (:left score))
          (= {:kind :indexed-key :buffer (:id k)
              :indices (:id source-indices) :dtype (:dtype k)
              :total-dim (:total-dim storage)}
             (:right score))
          (= '(raster.numeric/* left right) (get-in score [:combine :body]))
          (= ['left 'right] (get-in score [:combine :parameters]))
          (= [:inverse-sqrt :literal :literal] (mapv :kind score-arguments))
          (= ['scale 'lower 'upper] (mapv :parameter score-arguments))
          (= (:components value) (:extent scale))
          (number? bound) (pos? (double bound))
          (= (- (double bound)) (double (:value lower)))
          (= '(raster.numeric/min
               upper
               (raster.numeric/max lower (raster.numeric/* dot scale)))
             (get-in score [:finalize :body]))
          (= ['dot 'scale 'lower 'upper] (get-in score [:finalize :parameters]))
          (= '(raster.math/exp score) (:body weight))
          (= ['score] (:parameters weight))
          (= :indexed-value (:kind value))
          (= (:id v) (:buffer value))
          (= (:id source-indices) (:indices value))
          (= (:dtype v) (:dtype value))
          (= (:extent destination-axis) (:entity-count value))
          (= (:total-dim storage) (:total-dim value))
          (= :sum (:operator numerator))
          (zero? (double (:identity numerator)))
          (= '(raster.numeric/* weight value) (get-in numerator [:map-region :body]))
          (= :sum (:operator denominator))
          (zero? (double (:identity denominator)))
          (= 'weight (get-in denominator [:map-region :body]))
          (= :divide (:kind normalization))
          (pos? (double (:epsilon normalization)))
          (= 0.0 (double (:empty-result normalization)))
          (= 5 (count operands))
          (= [:long :long] (mapv :dtype [destination-indices source-indices]))
          (= [(:extent destination-axis) (:total-dim storage)] (:shape q))
          (= (:shape q) (:shape k) (:shape v) (:shape output))
          (= (:dtype q) (:dtype k) (:dtype v) (:dtype output) accumulator-dtype))
      (fail "indexed edge-list leaf cannot preserve this reduction plan exactly"
            :indexed-segmented-reduction-plan-unsupported
            {:plan-id (:id plan) :provenance provenance}))
    plan))

(defn- resolve-extent
  [shape-env owner value]
  (let [resolved (if (integer? value) value (get shape-env value ::missing))
        valid? pos?]
    (when (or (= ::missing resolved) (not (integer? resolved)) (not (valid? resolved)))
      (fail "indexed-attention extent is missing or invalid"
            :indexed-attention-invalid-shape-extent
            {:owner owner :extent value :resolved resolved}))
    (long resolved)))

(defn- resolved-shape
  [plan shape-env]
  (let [[destination-axis head-axis] (:segment-axes plan)
        membership (:membership plan)
        storage (:storage plan)
        value (:value plan)]
    {:entities (resolve-extent shape-env :entities (:extent destination-axis))
     :heads (resolve-extent shape-env :heads (:extent head-axis))
     :edges (resolve-extent shape-env :edges (:edges membership))
     :components (resolve-extent shape-env :components (:components value))
     :total-dim (resolve-extent shape-env :total-dim (:total-dim storage))}))

(defn- checked-shape!
  [plan shape-env]
  (let [{:keys [heads components total-dim] :as shape} (resolved-shape plan shape-env)]
    (when (> (* heads components) total-dim)
      (fail "indexed-attention head slices exceed the row width"
            :indexed-attention-head-layout-overflow shape))
    shape))

(defn reference-workgroup-x
  [plan shape-env desc]
  (let [{:keys [total-dim]} (checked-shape! (indexed-attention-plan! plan) shape-env)
        subgroup (long (or (:subgroup-size desc) 16))
        maximum (long (or (:max-workgroup-size desc) 256))]
    (long (max 1 (min total-dim subgroup maximum)))))

(defn- c-type
  [dtype]
  (case dtype :float "float" :double "double"))

(defn- fp-literal
  [dtype value]
  (str (if (= :float dtype)
         (Float/toString (float value))
         (Double/toString (double value)))
       (when (= :float dtype) "f")))

(defn- long-expression
  [value]
  (if (number? value) (str value "L") (str "(" value ")")))

(defn- fp-expression
  [dtype value]
  (if (number? value) (fp-literal dtype value) (str value)))

(defn- kernel-name
  [plan shape]
  (let [identity {:schedule (swr/schedule-key plan) :shape shape}
        schedule-name (case (:schedule shape)
                        :destination-score-reuse "subgroup_score_reuse"
                        "ref")]
    (format "raster_indexed_attention_%s_%08x" schedule-name
            (bit-and 0xffffffff (long (hash identity))))))

(defn- source*
  [plan {:keys [entities edges components total-dim active-width scale bound epsilon
                scalar-signature invalid-shape]} name]
  (let [entities (long-expression entities)
        edges (long-expression edges)
        components (long-expression components)
        total-dim (long-expression total-dim)
        active-width (long-expression active-width)
        dtype (:accumulator-dtype plan)
        ctype (c-type dtype)
        bound (fp-expression dtype bound)
        epsilon (fp-expression dtype epsilon)
        scale (fp-expression dtype scale)
        zero (fp-literal dtype 0.0)]
    (str (when (= :double dtype)
           "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n")
         "__kernel void " name "(\n"
         "    __global const " ctype "* q,\n"
         "    __global const " ctype "* k,\n"
         "    __global const " ctype "* v,\n"
         "    __global const long* destination_indices,\n"
         "    __global const long* source_indices,\n"
         "    __global " ctype "* output"
         (when (seq scalar-signature) (str ",\n" scalar-signature))
         ") {\n"
         "  const long feature = (long)get_global_id(0);\n"
         "  const long destination = (long)get_global_id(1);\n"
         "  if (feature >= " total-dim " || destination >= " entities ") return;\n"
         "  const long output_index = destination * " total-dim " + feature;\n"
         (when invalid-shape
           (str "  if (" invalid-shape ") {\n"
                "    output[output_index] = (" ctype ")NAN;\n"
                "    return;\n"
                "  }\n"))
         "  if (feature >= " active-width ") {\n"
         "    output[output_index] = " zero ";\n"
         "    return;\n"
         "  }\n"
         "  const long head = feature / " components ";\n"
         "  const long component = feature - head * " components ";\n"
         "  " ctype " numerator = " zero ";\n"
         "  " ctype " denominator = " zero ";\n"
         "  for (long edge = 0; edge < " edges "; ++edge) {\n"
         "    const long edge_destination = destination_indices[edge];\n"
         "    const long source = source_indices[edge];\n"
         "    if (edge_destination < 0L || edge_destination >= " entities
         " || source < 0L || source >= " entities ") {\n"
         "      output[output_index] = (" ctype ")NAN;\n"
         "      return;\n"
         "    }\n"
         "    if (edge_destination != destination) continue;\n"
         "    const long q_base = destination * " total-dim " + head * " components ";\n"
         "    const long kv_base = source * " total-dim " + head * " components ";\n"
         "    " ctype " dot = " zero ";\n"
         "    for (long x = 0; x < " components "; ++x)\n"
         "      dot += q[q_base + x] * k[kv_base + x];\n"
         "    const " ctype " scaled = dot * (" ctype ")" scale ";\n"
         ;; Java Math/min and Math/max propagate NaN; OpenCL fmin/fmax select the numeric operand.
         ;; Preserve the recognized source semantics explicitly instead of inheriting that drift.
         "    const " ctype " score = isnan(scaled) ? scaled\n"
         "        : fmin((" ctype ")" bound
         ", fmax(-(" ctype ")" bound ", scaled));\n"
         "    const " ctype " weight = exp(score);\n"
         "    numerator += weight * v[kv_base + component];\n"
         "    denominator += weight;\n"
         "  }\n"
         "  output[output_index] = denominator == " zero " ? " zero
         " : numerator / (denominator + (" ctype ")" epsilon ");\n"
         "}\n")))

(defn- source
  [plan {:keys [heads components] :as shape} name]
  (source* plan
           (assoc shape
                  :active-width (* heads components)
                  :scale (/ 1.0 (Math/sqrt (double components)))
                  :bound (double (get-in plan [:score :arguments 2 :value]))
                  :epsilon (double (get-in plan [:normalization :epsilon])))
           name))

(defn- ordered-abi
  [plan]
  (let [[q k v destination-indices source-indices] (:operands plan)
        output (:output plan)
        fp (:accumulator-dtype plan)]
    (kabi/validate!
     [(kabi/slot (:id q) :input fp :c-name "q" :role :query)
      (kabi/slot (:id k) :input fp :c-name "k" :role :key)
      (kabi/slot (:id v) :input fp :c-name "v" :role :value)
      (kabi/slot (:id destination-indices) :input :long
                 :c-name "destination_indices" :role :destination-indices)
      (kabi/slot (:id source-indices) :input :long
                 :c-name "source_indices" :role :source-indices)
      (kabi/slot (:id output) :output fp :c-name "output" :role :result)])))

(defn- dynamic-fields
  [plan]
  (let [[destination-axis head-axis] (:segment-axes plan)
        membership (:membership plan)
        storage (:storage plan)
        value (:value plan)
        output-elements (:elements (:output plan))]
    [{:name 'n_entities :c-name "n_entities" :value (:extent destination-axis)}
     {:name 'n_edges :c-name "n_edges" :value (:edges membership)}
     {:name 'total_dim :c-name "total_dim" :value (:total-dim storage)}
     {:name 'n_heads :c-name "n_heads" :value (:extent head-axis)}
     {:name 'n_components :c-name "n_components" :value (:components value)}
     {:name 'output_elements :c-name "output_elements" :value output-elements}]))

(defn- dynamic-workgroup-x
  [desc]
  (long (max 1 (min (long (or (:subgroup-size desc) 16))
                    (long (or (:max-workgroup-size desc) 256))))))

(defn- dynamic-abi
  [plan fields]
  (let [[q k v destination-indices source-indices] (:operands plan)
        output (:output plan)
        fp (:accumulator-dtype plan)]
    (kabi/validate!
     (into
      [(kabi/slot (:id q) :input fp :c-name "q" :role :query)
       (kabi/slot (:id k) :input fp :c-name "k" :role :key)
       (kabi/slot (:id v) :input fp :c-name "v" :role :value)
       (kabi/slot (:id destination-indices) :input :long
                  :c-name "destination_indices" :role :destination-indices)
       (kabi/slot (:id source-indices) :input :long
                  :c-name "source_indices" :role :source-indices)
       (kabi/slot (:id output) :output fp :c-name "output" :role :result)]
      (map (fn [{:keys [name c-name]}]
             (kabi/slot name :scalar :long :c-name c-name :role :shape))
           fields)))))

(defn emit-dynamic-reference
  "Emit one shape-polymorphic indexed-attention artifact.

   Extents remain ordered int64 ABI values and drive the symbolic 2-D launch. Clamp and epsilon
   are proven model semantics and stay embedded constants. `:out-elems` names an explicit scalar
   argument so the staging path can allocate/read back without interpreting product forms."
  [plan desc]
  (let [plan (indexed-attention-plan! plan)
        fields (dynamic-fields plan)
        values (mapv :value fields)
        field-code (into {} (map (juxt :name :c-name)) fields)
        [entities _ total-dim _ _ output-elements] values
        workgroup-x (dynamic-workgroup-x desc)
        name (kernel-name plan :dynamic)
        inputs (swr/ordered-input-ids plan)
        output (get-in plan [:output :id])
        dtype (:accumulator-dtype plan)
        ctype (c-type dtype)
        scalar-signature
        (->> fields
             (map (fn [{:keys [c-name]}] (str "    long " c-name)))
             (str/join ",\n"))
        code {:entities (get field-code 'n_entities)
              :edges (get field-code 'n_edges)
              :total-dim (get field-code 'total_dim)
              :heads (get field-code 'n_heads)
              :components (get field-code 'n_components)
              :active-width (str (get field-code 'n_heads) " * "
                                 (get field-code 'n_components))
              :scale (str "((" ctype ")" (fp-literal dtype 1.0)
                          " / sqrt((" ctype ")" (get field-code 'n_components) "))")
              :bound (double (get-in plan [:score :arguments 2 :value]))
              :epsilon (double (get-in plan [:normalization :epsilon]))
              :scalar-signature scalar-signature
              :invalid-shape
              "n_edges < 0L || n_heads <= 0L || n_components <= 0L || n_heads > total_dim / n_components"}
        abi (dynamic-abi plan fields)
        arguments (into (conj inputs output) values)]
    (kart/make
     {:kernel-name name
      :source (source* plan code name)
      :abi abi
      :arguments arguments
      :launch (klaunch/spec
               {:workgroup-size [workgroup-x 1]
                :group-count [(klaunch/ceil-div total-dim workgroup-x)
                              (klaunch/runtime-value entities)]})
      :effects {:kind :segmented-weighted-reduction :reads inputs :writes [output]}
      :provenance {:operation-id (get-in plan [:provenance :operation-id])
                   :semantic-op (get-in plan [:provenance :semantic-op])
                   :algebra-plan-id (:id plan)
                   :lowering :direct-dynamic-reference}
      :attributes {:strategy :indexed-segmented-reduction-reference
                   :optimization-tier :reference
                   :algebra :segmented-weighted-reduction
                   :algebra-key (swr/algebra-key plan)
                   :storage-dtype dtype :accumulator-dtype dtype
                   :membership :edge-list-by-destination
                   :duplicate-policy :multiset
                   :dynamic-shape? true
                   :out-elems output-elements
                   :materialized-intermediates []
                   :complexity :quadratic-in-edges-and-head-components}})))

(defn- score-reuse-subgroup-size
  [desc]
  (long (or (:subgroup-size desc) 16)))

(defn- score-reuse-source
  [plan fields subgroup-size name]
  (let [dtype (:accumulator-dtype plan)
        ctype (c-type dtype)
        zero (fp-literal dtype 0.0)
        one (fp-literal dtype 1.0)
        bound (fp-literal dtype (double (get-in plan [:score :arguments 2 :value])))
        epsilon (fp-literal dtype (double (get-in plan [:normalization :epsilon])))
        scalar-signature (->> fields
                              (map (fn [{:keys [c-name]}] (str "    long " c-name)))
                              (str/join ",\n"))]
    (str (when (= :double dtype)
           "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n")
         "__attribute__((intel_reqd_sub_group_size(" subgroup-size ")))\n"
         "__kernel void " name "(\n"
         "    __global const " ctype "* q,\n"
         "    __global const " ctype "* k,\n"
         "    __global const " ctype "* v,\n"
         "    __global const long* destination_indices,\n"
         "    __global const long* source_indices,\n"
         "    __global " ctype "* output,\n"
         scalar-signature ") {\n"
         "  const long lane = (long)get_sub_group_local_id();\n"
         "  const long component_tile = (long)get_group_id(0);\n"
         "  const long head = (long)get_group_id(1);\n"
         "  const long destination = (long)get_group_id(2);\n"
         "  const long component = component_tile * " subgroup-size "L + lane;\n"
         "  const long feature = head * n_components + component;\n"
         "  const int active = component < n_components && feature < total_dim;\n"
         "  " ctype " numerator = " zero ";\n"
         "  " ctype " denominator = " zero ";\n"
         "  int invalid = n_edges < 0L || n_heads <= 0L || n_components <= 0L\n"
         "        || n_heads > total_dim / n_components;\n"
         "  for (long edge = 0L; edge < n_edges; ++edge) {\n"
         "    const long edge_destination = destination_indices[edge];\n"
         "    const long source = source_indices[edge];\n"
         "    if (edge_destination < 0L || edge_destination >= n_entities\n"
         "        || source < 0L || source >= n_entities) invalid = 1;\n"
         "    const int visible = !invalid && edge_destination == destination;\n"
         "    " ctype " partial_dot = " zero ";\n"
         "    if (visible) {\n"
         "      const long q_base = destination * total_dim + head * n_components;\n"
         "      const long k_base = source * total_dim + head * n_components;\n"
         "      for (long x = lane; x < n_components; x += " subgroup-size "L)\n"
         "        partial_dot += q[q_base + x] * k[k_base + x];\n"
         "    }\n"
         "    const " ctype " dot = sub_group_reduce_add(partial_dot);\n"
         "    " ctype " weight = " zero ";\n"
         "    if (lane == 0L && visible) {\n"
         "        const " ctype " scaled = dot * ((" ctype ")" one
         " / sqrt((" ctype ")n_components));\n"
         "        const " ctype " score = isnan(scaled) ? scaled\n"
         "            : fmin((" ctype ")" bound ", fmax(-(" ctype ")" bound ", scaled));\n"
         "        weight = exp(score);\n"
         "    }\n"
         "    weight = sub_group_broadcast(weight, 0);\n"
         "    denominator += weight;\n"
         "    if (active && visible)\n"
         "      numerator += weight * v[source * total_dim + feature];\n"
         "  }\n"
         "  if (active) {\n"
         "    const long output_index = destination * total_dim + feature;\n"
         "    output[output_index] = invalid ? (" ctype ")NAN\n"
         "        : (denominator == " zero " ? " zero
         " : numerator / (denominator + (" ctype ")" epsilon "));\n"
         "  }\n"
         "  if (head == 0L && component_tile == 0L) {\n"
         "    const long active_width = n_heads * n_components;\n"
         "    for (long tail = active_width + lane; tail < total_dim; tail += "
         subgroup-size "L)\n"
         "      output[destination * total_dim + tail] = invalid ? (" ctype ")NAN : " zero ";\n"
         "  }\n"
         "}\n")))

(defn emit-dynamic-score-reuse
  "Emit a 3-D destination/head/component-tile schedule. One hardware subgroup cooperatively
   reduces each edge score and broadcasts its weight across component lanes. This retains the
   edge-list ABI while removing the reference leaf's per-output score recomputation."
  [plan desc]
  (let [plan (indexed-attention-plan! plan)
        fields (dynamic-fields plan)
        values (mapv :value fields)
        [entities _ _ heads components output-elements] values
        subgroup-size (score-reuse-subgroup-size desc)
        name (kernel-name plan {:schedule :destination-score-reuse
                                :subgroup-size subgroup-size})
        inputs (swr/ordered-input-ids plan)
        output (get-in plan [:output :id])
        dtype (:accumulator-dtype plan)
        abi (dynamic-abi plan fields)
        arguments (into (conj inputs output) values)]
    (kart/make
     {:kernel-name name
      :source (score-reuse-source plan fields subgroup-size name)
      :abi abi
      :arguments arguments
      :launch (klaunch/spec
               {:workgroup-size [subgroup-size 1 1]
                :group-count [(klaunch/ceil-div components subgroup-size)
                              (klaunch/runtime-value heads)
                              (klaunch/runtime-value entities)]})
      :effects {:kind :segmented-weighted-reduction :reads inputs :writes [output]}
      :provenance {:operation-id (get-in plan [:provenance :operation-id])
                   :semantic-op (get-in plan [:provenance :semantic-op])
                   :algebra-plan-id (:id plan)
                   :lowering :destination-score-reuse}
      :attributes {:strategy :indexed-segmented-reduction-subgroup-score-reuse
                   :optimization-tier :subgroup
                   :algebra :segmented-weighted-reduction
                   :algebra-key (swr/algebra-key plan)
                   :storage-dtype dtype :accumulator-dtype dtype
                   :membership :edge-list-by-destination
                   :duplicate-policy :multiset
                   :dynamic-shape? true
                   :out-elems output-elements
                   :score-reuse-width subgroup-size
                   :materialized-intermediates []
                   :complexity :destination-head-edge-dot-plus-value}})))

(defn emit-reference
  "Emit the direct indexed-attention correctness schedule for a resolved shape environment."
  [plan shape-env desc]
  (let [plan (indexed-attention-plan! plan)
        shape (checked-shape! plan shape-env)
        {:keys [entities total-dim]} shape
        workgroup-x (reference-workgroup-x plan shape-env desc)
        name (kernel-name plan shape)
        inputs (swr/ordered-input-ids plan)
        output (get-in plan [:output :id])]
    (kart/make
     {:kernel-name name
      :source (source plan shape name)
      :abi (ordered-abi plan)
      :arguments (conj inputs output)
      :launch (klaunch/spec
               {:workgroup-size [workgroup-x 1]
                :group-count [(long (quot (+ total-dim (dec workgroup-x)) workgroup-x))
                              entities]})
      :effects {:kind :segmented-weighted-reduction :reads inputs :writes [output]}
      :provenance {:operation-id (get-in plan [:provenance :operation-id])
                   :semantic-op (get-in plan [:provenance :semantic-op])
                   :algebra-plan-id (:id plan)
                   :lowering :direct-reference}
      :attributes {:strategy :indexed-segmented-reduction-reference
                   :optimization-tier :reference
                   :algebra :segmented-weighted-reduction
                   :algebra-key (swr/algebra-key plan)
                   :storage-dtype (:accumulator-dtype plan)
                   :accumulator-dtype (:accumulator-dtype plan)
                   :membership :edge-list-by-destination
                   :duplicate-policy :multiset
                   :shape shape
                   :materialized-intermediates []
                   :complexity :quadratic-in-edges-and-head-components}})))

(defn kernel-graph
  "Wrap the direct leaf in a one-node graph with resolved external buffer extents."
  [plan shape-env artifact]
  (let [plan (indexed-attention-plan! plan)
        {:keys [entities edges total-dim]} (checked-shape! plan shape-env)
        artifact (kart/validate! artifact)
        inputs (swr/ordered-input-ids plan)
        output (get-in plan [:output :id])
        descriptors (into {} (map (juxt :id identity))
                          (conj (:operands plan) (:output plan)))
        elements (fn [id]
                   (if (contains? #{(get-in plan [:membership :destination-indices])
                                    (get-in plan [:membership :source-indices])} id)
                     edges
                     (* entities total-dim)))
        graph-buffer (fn [id role]
                       (kgraph/buffer id (:dtype (get descriptors id))
                                      (elements id) :device role))
        uses (vec (concat (map #(kgraph/->ValueUse % :read) inputs)
                          [(kgraph/->ValueUse output :write)]))]
    (kgraph/make
     {:inputs (mapv #(graph-buffer % :input) inputs)
      :outputs [(graph-buffer output :output)]
      :nodes [(kgraph/->ScheduledKernel
               [:segmented-weighted-reduction (:id plan) :indexed-reference] artifact uses [])]
      :effects {:kind :segmented-weighted-reduction :materialized-intermediates []}
      :provenance {:operation-id (get-in plan [:provenance :operation-id])
                   :semantic-op (get-in plan [:provenance :semantic-op])}
      :attributes {:strategy :indexed-segmented-reduction-reference :reference? true}})))

(defn dynamic-kernel-graph
  "Wrap a dynamic artifact while retaining runtime-resolvable external buffer ranges."
  [plan artifact]
  (let [plan (indexed-attention-plan! plan)
        artifact (kart/validate! artifact)
        inputs (swr/ordered-input-ids plan)
        output (get-in plan [:output :id])
        descriptors (into {} (map (juxt :id identity))
                          (conj (:operands plan) (:output plan)))
        edge-ids #{(get-in plan [:membership :destination-indices])
                   (get-in plan [:membership :source-indices])}
        edge-elements (:edges (:membership plan))
        value-elements (:elements (:output plan))
        graph-buffer (fn [id role]
                       (kgraph/buffer
                        id (:dtype (get descriptors id))
                        (klaunch/runtime-value
                         (if (contains? edge-ids id) edge-elements value-elements))
                        :device role))
        uses (vec (concat (map #(kgraph/->ValueUse % :read) inputs)
                          [(kgraph/->ValueUse output :write)]))
        strategy (get-in artifact [:attributes :strategy])
        reference? (= :reference (get-in artifact [:attributes :optimization-tier]))]
    (kgraph/make
     {:inputs (mapv #(graph-buffer % :input) inputs)
      :outputs [(graph-buffer output :output)]
      :nodes [(kgraph/->ScheduledKernel
               [:segmented-weighted-reduction (:id plan) :indexed-dynamic-reference]
               artifact uses [])]
      :effects {:kind :segmented-weighted-reduction :materialized-intermediates []}
      :provenance {:operation-id (get-in plan [:provenance :operation-id])
                   :semantic-op (get-in plan [:provenance :semantic-op])}
      :attributes {:strategy strategy :reference? reference? :dynamic-shape? true}})))
