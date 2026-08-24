(ns raster.compiler.backend.gpu.indexed-attention
  "Direct correctness lowering for recognized indexed graph attention.

   One work-item owns one destination/feature output and scans the edge list. The schedule is
   intentionally simple, but it is genuinely fused: scores, clamped exponential weights,
   denominator and weighted values remain private scalars and no edge-sized intermediates are
   materialized."
  (:require [raster.compiler.ir.kernel-abi :as kabi]
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
     (and (= :indexed-graph-attention (:semantic-op provenance))
          (= :recognized-indexed-attention-chain (:lowering provenance))
          (= [:destination :head] (mapv :name segment-axes))
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
      (fail "indexed-attention leaf cannot preserve this reduction plan exactly"
            :indexed-attention-reference-plan-unsupported
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

(defn- kernel-name
  [plan shape]
  (let [identity {:schedule (swr/schedule-key plan) :shape shape}]
    (format "raster_indexed_attention_ref_%08x"
            (bit-and 0xffffffff (long (hash identity))))))

(defn- source
  [plan shape name]
  (let [{:keys [entities heads edges components total-dim]} shape
        dtype (:accumulator-dtype plan)
        ctype (c-type dtype)
        bound (double (get-in plan [:score :arguments 2 :value]))
        epsilon (double (get-in plan [:normalization :epsilon]))
        scale (/ 1.0 (Math/sqrt (double components)))
        zero (fp-literal dtype 0.0)]
    (str (when (= :double dtype)
           "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n")
         "__kernel void " name "(\n"
         "    __global const " ctype "* q,\n"
         "    __global const " ctype "* k,\n"
         "    __global const " ctype "* v,\n"
         "    __global const long* destination_indices,\n"
         "    __global const long* source_indices,\n"
         "    __global " ctype "* output) {\n"
         "  const long feature = (long)get_global_id(0);\n"
         "  const long destination = (long)get_global_id(1);\n"
         "  if (feature >= " total-dim "L || destination >= " entities "L) return;\n"
         "  const long output_index = destination * " total-dim "L + feature;\n"
         "  if (feature >= " (* heads components) "L) {\n"
         "    output[output_index] = " zero ";\n"
         "    return;\n"
         "  }\n"
         "  const long head = feature / " components "L;\n"
         "  const long component = feature - head * " components "L;\n"
         "  " ctype " numerator = " zero ";\n"
         "  " ctype " denominator = " zero ";\n"
         "  for (long edge = 0; edge < " edges "L; ++edge) {\n"
         "    const long edge_destination = destination_indices[edge];\n"
         "    const long source = source_indices[edge];\n"
         "    if (edge_destination < 0L || edge_destination >= " entities
         "L || source < 0L || source >= " entities "L) {\n"
         "      output[output_index] = (" ctype ")NAN;\n"
         "      return;\n"
         "    }\n"
         "    if (edge_destination != destination) continue;\n"
         "    const long q_base = destination * " total-dim "L + head * " components "L;\n"
         "    const long kv_base = source * " total-dim "L + head * " components "L;\n"
         "    " ctype " dot = " zero ";\n"
         "    for (long x = 0; x < " components "L; ++x)\n"
         "      dot += q[q_base + x] * k[kv_base + x];\n"
         "    const " ctype " scaled = dot * (" ctype ")" (fp-literal dtype scale) ";\n"
         ;; Java Math/min and Math/max propagate NaN; OpenCL fmin/fmax select the numeric operand.
         ;; Preserve the recognized source semantics explicitly instead of inheriting that drift.
         "    const " ctype " score = isnan(scaled) ? scaled\n"
         "        : fmin((" ctype ")" (fp-literal dtype bound)
         ", fmax((" ctype ")" (fp-literal dtype (- bound)) ", scaled));\n"
         "    const " ctype " weight = exp(score);\n"
         "    numerator += weight * v[kv_base + component];\n"
         "    denominator += weight;\n"
         "  }\n"
         "  output[output_index] = denominator == " zero " ? " zero
         " : numerator / (denominator + (" ctype ")" (fp-literal dtype epsilon) ");\n"
         "}\n")))

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
      :effects {:kind :indexed-attention :reads inputs :writes [output]}
      :provenance {:operation-id (get-in plan [:provenance :operation-id])
                   :semantic-op :indexed-graph-attention
                   :algebra-plan-id (:id plan)
                   :lowering :direct-reference}
      :attributes {:strategy :indexed-attention-reference
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
               [:indexed-attention (:id plan) :direct-reference] artifact uses [])]
      :effects {:kind :indexed-attention :materialized-intermediates []}
      :provenance {:operation-id (get-in plan [:provenance :operation-id])
                   :semantic-op :indexed-graph-attention}
      :attributes {:strategy :indexed-attention-reference :reference? true}})))
