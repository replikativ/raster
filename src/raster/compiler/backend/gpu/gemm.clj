(ns raster.compiler.backend.gpu.gemm
  "Compiler-owned executable schedules for dense GEMM.

   The public call is uniformly `(A B C M N K)` over f32 resident buffers. A schedule may be one
   scalar kernel or a graph containing conversion, layout conversion, matrix contraction, and
   split-K combination. All mixed-precision scratch and derived scheduling scalars are private to
   the graph; callers never bind them and runtimes never reconstruct the algorithm from `:gemm`."
  (:require [clojure.string :as str]
            [raster.compiler.backend.gpu.kernel-body-opencl :as kernel-body-opencl]
            [raster.compiler.backend.gpu.opencl-codegen :as codegen]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.passes.parallel.contraction-schedule :as contraction-schedule]))

(def ^:private default-min-split-chunk 1024)
(def ^:private default-max-splits 64)

(defn- identifier
  [value]
  (let [text (str/replace (str value) #"[^A-Za-z0-9_]" "_")]
    (if (re-find #"^[A-Za-z_]" text) text (str "k_" text))))

(defn- graph-buffer
  [id dtype elements role]
  (kgraph/buffer id dtype elements :device role))

(defn- value-use
  [buffer access]
  (kgraph/->ValueUse buffer access))

(defn- node
  [id operation uses dependencies]
  (kgraph/->ScheduledKernel id operation (vec uses) (vec dependencies)))

(defn- outer-interface
  [{:keys [a b c m n k]}]
  {:abi [(kabi/slot a :input :float :c-name "A" :role :lhs)
         (kabi/slot b :input :float :c-name "B" :role :rhs)
         (kabi/slot c :output :float :c-name "C" :role :result)
         (kabi/slot m :scalar :int :c-name "M" :role :extent)
         (kabi/slot n :scalar :int :c-name "N" :role :extent)
         (kabi/slot k :scalar :int :c-name "K" :role :extent)]
   :arguments [a b c m n k]})

(defn- extents
  [{:keys [m n k variant]}]
  {:a-elements (if (= :tn variant) (klaunch/product k m) (klaunch/product m k))
   :b-elements (if (= :nt variant) (klaunch/product n k) (klaunch/product k n))
   :c-elements (klaunch/product m n)})

(defn- effects
  [{:keys [a b c]}]
  {:kind :tensor-contraction :reads [a b] :writes [c]})

(defn- artifact
  [kernel-name source abi arguments launch phase attributes]
  (kart/make
   {:kernel-name kernel-name
    :source source
    :abi abi
    :arguments arguments
    :launch launch
    :effects {:kind :tensor-contraction-stage}
    :provenance {:semantic-op :contraction :lowering :gemm-graph :phase phase}
    :attributes (merge {:strategy phase} attributes)}))

(defn- scalar-graph
  [{:keys [id a b c m n k variant] :as spec}]
  (let [{:keys [abi arguments]} (outer-interface spec)
        {:keys [a-elements b-elements c-elements]} (extents spec)
        prefix (identifier (str id "_scalar"))
        kernel-name (str prefix "_gemm")
        gemm (artifact
              kernel-name
              (codegen/emit-gemm-scalar-kernel kernel-name :variant variant)
              [(kabi/slot a :input :float :c-name "A")
               (kabi/slot b :input :float :c-name "B")
               (kabi/slot c :output :float :c-name "C")
               (kabi/slot m :scalar :int :c-name "m")
               (kabi/slot n :scalar :int :c-name "n")
               (kabi/slot k :scalar :int :c-name "k")]
              [a b c m n k]
              (klaunch/spec {:workgroup-size [256]
                             :group-count [(klaunch/ceil-div c-elements 256)]})
              :scalar-gemm {:variant variant})]
    (kgraph/make
     {:inputs [(graph-buffer a :float a-elements :input)
               (graph-buffer b :float b-elements :input)]
      :outputs [(graph-buffer c :float c-elements :output)]
      :nodes [(node [:gemm id :scalar] gemm
                    [(value-use a :read) (value-use b :read) (value-use c :write)] [])]
      :abi abi :arguments arguments
      :effects (effects spec)
      :provenance {:semantic-op :contraction :variant variant :lowering :scalar-gemm}
      :attributes {:strategy :f32-scalar :variant variant :precision :f32}})))

(defn- convert-artifact
  [kernel-name in out elements vector-width phase]
  (artifact
   kernel-name (codegen/emit-f32-to-f16-kernel kernel-name vector-width)
   [(kabi/slot in :input :float :c-name "in")
    (kabi/slot out :output :half :c-name "out")
    (kabi/slot :elements :scalar :int :c-name "n")]
   [in out elements]
   (klaunch/spec
    {:workgroup-size [256]
     :group-count [(klaunch/ceil-div
                    (klaunch/ceil-div (klaunch/runtime-value elements) vector-width)
                    256)]})
   phase {:vector-width vector-width :from :float :to :half
          :cacheable-transform? true}))

(defn- transpose-artifact
  [kernel-name in out rows cols phase]
  (let [elements (klaunch/product rows cols)]
    (artifact
     kernel-name (codegen/emit-transpose-kernel kernel-name :dtype :half)
     [(kabi/slot in :input :half :c-name "in")
      (kabi/slot out :output :half :c-name "out")
      (kabi/slot :rows :scalar :int :c-name "rows")
      (kabi/slot :cols :scalar :int :c-name "cols")]
     [in out rows cols]
     (klaunch/spec {:workgroup-size [256]
                    :group-count [(klaunch/ceil-div elements 256)]})
     phase {:layout :transpose :dtype :half :cacheable-transform? true})))

(defn- matrix-workgroup-size
  [{:keys [block-m block-n sg-m sg-n matrix]}]
  (* (quot block-m sg-m) (quot block-n sg-n) (:subgroup matrix 16)))

(defn emit-scheduled-matrix-kernel
  "Build and directly lower one canonical f16 matrix contraction.

  This is the shared compiler entry for graph-owned, legacy-plan, and resident direct/tiled GEMM
  front doors.  Caller identities remain the KernelBody ABI identities; OpenCL M/N/K spelling is
  solely a target concern.  Split-K, batched grid-Z, and opaque source epilogues are not accepted
  here until their scheduling operations are explicit in KernelBody."
  [{:keys [kernel-name id a b c m n k tile result-dtype provenance]
    :or {result-dtype :float provenance {}}}]
  (let [kernel-body
        (contraction-schedule/matrix-body
         {:id (or id [:gemm kernel-name])
          :row a :col b :out c
          :dimensions [m n k]
          :dimension-parameters [m n k]
          :axis-symbols ['i 'j 'l]
          :tile tile
          :bindings {:row a :col b}
          :result-dtype result-dtype
          :provenance (merge {:dialect :gemm :lowering :scheduled-matrix} provenance)})]
    {:source (kernel-body-opencl/emit-matrix-kernel kernel-name kernel-body nil)
     :kernel-body kernel-body
     :workgroup-size (get-in kernel-body [:launch :workgroup-size])}))

(defn- gemm-artifact
  [{:keys [id m n k tile]} kernel-name a b c split-k? kc splits phase]
  (let [{:keys [block-m block-n]} tile
        source-args (concat [:c-dtype :float :split-k? split-k?
                             :schedule-splits-arg? split-k?]
                            (mapcat identity
                                    (select-keys tile
                                                 [:block-m :block-n :sg-m :sg-n :block-k
                                                  :matrix :num-stages])))
        source-args (vec (mapcat (fn [[key value]]
                                   [(if (= :num-stages key) :prefetch key) value])
                                 (partition 2 source-args)))
        abi (cond-> [(kabi/slot a :input :half :c-name "A")
                     (kabi/slot b :input :half :c-name "B")
                     (kabi/slot c :output :float :c-name "C")
                     (kabi/slot m :scalar :int :c-name "M")
                     (kabi/slot n :scalar :int :c-name "N")
                     (kabi/slot k :scalar :int :c-name "K")]
              split-k? (conj (kabi/slot :k-chunk :scalar :int :c-name "KC")
                             (kabi/slot :splits :scalar :int :c-name "splits")))
        arguments (cond-> [a b c m n k] split-k? (conj kc splits))
        group-count (cond-> [(klaunch/ceil-div n block-n)
                             (klaunch/ceil-div m block-m)]
                      split-k? (conj (klaunch/runtime-value splits)))
        scheduled (when-not split-k?
                    (emit-scheduled-matrix-kernel
                     {:kernel-name kernel-name
                      :id [:gemm id phase]
                      :a a :b b :c c :m m :n n :k k
                      :tile tile :result-dtype :float
                      :provenance {:operation-id id :phase phase}}))]
    (artifact
     kernel-name (if scheduled
                   (:source scheduled)
                   (apply codegen/emit-gemm-tiled kernel-name source-args))
     abi arguments
     (klaunch/spec {:workgroup-size (if scheduled
                                      (:workgroup-size scheduled)
                                      [(matrix-workgroup-size tile) 1 1])
                    :group-count group-count})
     phase (cond-> {:tile tile :split-k? split-k? :accumulator-dtype :float}
             scheduled (assoc :kernel-body (:kernel-body scheduled))))))

(defn- combine-artifact
  [kernel-name partials c mn splits]
  (artifact
   kernel-name (codegen/emit-gemm-splitk-reduce-kernel kernel-name)
   [(kabi/slot partials :input :float :c-name "partials")
    (kabi/slot c :output :float :c-name "C")
    (kabi/slot :mn :scalar :int :c-name "mn")
    (kabi/slot :splits :scalar :int :c-name "splits")]
   [partials c mn splits]
   (klaunch/spec {:workgroup-size [256]
                  :group-count [(klaunch/ceil-div (klaunch/runtime-value mn) 256)]})
   :split-k-combine {:accumulator-dtype :float}))

(defn- xmx-graph
  [{:keys [id a b c m n k variant tile vector-width requested-splits split-k?] :as spec}]
  (let [{:keys [abi arguments]} (outer-interface spec)
        {:keys [a-elements b-elements c-elements]} (extents spec)
        strategy (if split-k? :xmx-split-k :xmx-direct)
        prefix (identifier (str id "_" (name strategy)))
        a16 [:gemm id strategy :a16]
        b16 [:gemm id strategy :b16]
        at16 [:gemm id strategy :at16]
        bt16 [:gemm id strategy :bt16]
        partials [:gemm id strategy :partials]
        final-a (if (= :tn variant) at16 a16)
        final-b (if (= :nt variant) bt16 b16)
        kc (when split-k?
             (klaunch/align-up (klaunch/ceil-div k requested-splits) (:block-k tile)))
        splits (when split-k? (klaunch/ceil-div k kc))
        partial-elements (when split-k? (klaunch/product splits m n))
        convert-a-id [:gemm id strategy :convert-a]
        convert-b-id [:gemm id strategy :convert-b]
        transpose-a-id [:gemm id strategy :transpose-a]
        transpose-b-id [:gemm id strategy :transpose-b]
        contract-id [:gemm id strategy :contract]
        convert-a (convert-artifact (str prefix "_convert_a") a a16 a-elements vector-width
                                    :convert-a)
        convert-b (convert-artifact (str prefix "_convert_b") b b16 b-elements vector-width
                                    :convert-b)
        transpose-a (when (= :tn variant)
                      (transpose-artifact (str prefix "_transpose_a") a16 at16 k m
                                          :transpose-a))
        transpose-b (when (= :nt variant)
                      (transpose-artifact (str prefix "_transpose_b") b16 bt16 n k
                                          :transpose-b))
        contract-output (if split-k? partials c)
        contract (gemm-artifact spec (str prefix "_contract") final-a final-b contract-output
                                split-k? kc splits :matrix-contract)
        combine (when split-k?
                  (combine-artifact (str prefix "_combine") partials c c-elements splits))
        nodes (cond->
               [(node convert-a-id convert-a [(value-use a :read) (value-use a16 :write)] [])
                (node convert-b-id convert-b [(value-use b :read) (value-use b16 :write)] [])]
                transpose-a
                (conj (node transpose-a-id transpose-a
                            [(value-use a16 :read) (value-use at16 :write)] [convert-a-id]))
                transpose-b
                (conj (node transpose-b-id transpose-b
                            [(value-use b16 :read) (value-use bt16 :write)] [convert-b-id]))
                true
                (conj (node contract-id contract
                            [(value-use final-a :read) (value-use final-b :read)
                             (value-use contract-output :write)]
                            [(if transpose-a transpose-a-id convert-a-id)
                             (if transpose-b transpose-b-id convert-b-id)]))
                combine
                (conj (node [:gemm id strategy :combine] combine
                            [(value-use partials :read) (value-use c :write)] [contract-id])))
        temporaries (cond-> [(graph-buffer a16 :half a-elements :temporary)
                             (graph-buffer b16 :half b-elements :temporary)]
                      transpose-a (conj (graph-buffer at16 :half a-elements :temporary))
                      transpose-b (conj (graph-buffer bt16 :half b-elements :temporary))
                      split-k? (conj (graph-buffer partials :float partial-elements :temporary)))]
    (kgraph/make
     {:inputs [(graph-buffer a :float a-elements :input)
               (graph-buffer b :float b-elements :input)]
      :outputs [(graph-buffer c :float c-elements :output)]
      :temporaries temporaries
      :nodes nodes
      :abi abi :arguments arguments
      :effects (effects spec)
      :provenance {:semantic-op :contraction :variant variant :lowering :xmx-gemm}
      :attributes {:strategy strategy :variant variant :precision :mixed-f16-f32
                   :tile tile :requested-splits requested-splits}})))

(defn requested-splits
  "Build the generic occupancy expression used by both selection and split-K storage/launches."
  [{:keys [m n k tile fill-workgroups target-fill-multiple min-split-chunk max-splits]
    :or {min-split-chunk default-min-split-chunk
         max-splits default-max-splits
         target-fill-multiple 4}}]
  (let [{:keys [block-m block-n]} tile
        output-workgroups (klaunch/product (klaunch/ceil-div m block-m)
                                           (klaunch/ceil-div n block-n))
        target-workgroups (* target-fill-multiple fill-workgroups)
        ;; Exactly zero once the unsplit output grid fills the machine, one otherwise. This keeps
        ;; the historic "never split a filling GEMM" legality/policy gate inside the same checked
        ;; arithmetic IR without introducing an opaque conditional callback.
        starved (klaunch/minimum 1 (klaunch/floor-div (dec fill-workgroups)
                                                      output-workgroups))]
    (klaunch/product
     starved
     (klaunch/minimum (klaunch/ceil-div target-workgroups output-workgroups)
                      (klaunch/floor-div k min-split-chunk)
                      max-splits))))

(defn emit-executable
  "Emit the resident GEMM schedule as one graph or a checked runtime dispatch.

   Required keys: :id, :a/:b/:c, :m/:n/:k compiler values, :variant, :precision, :tile,
   :fill-workgroups. The mixed-precision dispatch handles the XMX pitch gate and low-occupancy
   split-K choice entirely through generic expression cases."
  [{:keys [id a b c m n k variant precision tile fill-workgroups vector-width]
    :or {vector-width 4}
    :as spec}]
  (when-not (contains? #{:nn :nt :tn} variant)
    (throw (ex-info "GEMM executable requires :nn, :nt, or :tn variant"
                    {:id id :variant variant})))
  (doseq [[field value] [[:id id] [:a a] [:b b] [:c c] [:m m] [:n n] [:k k]
                         [:precision precision] [:tile tile] [:fill-workgroups fill-workgroups]]]
    (when (nil? value)
      (throw (ex-info "GEMM executable is missing a required field" {:field field :spec spec}))))
  (let [spec (assoc spec :vector-width vector-width)
        scalar (scalar-graph spec)]
    (case precision
      :f32-scalar scalar
      :f16-xmx
      (let [split-expression (requested-splits spec)
            xmx-spec (assoc spec :requested-splits split-expression)
            direct (xmx-graph (assoc xmx-spec :split-k? false))
            split (xmx-graph (assoc xmx-spec :split-k? true))]
        (kdispatch/make
         {:id (str id)
          :alternatives [scalar direct split]
          :default-strategy :xmx-direct
          :selector
          {:kind :runtime-expression-cases
           :cases [{:expression n :op :< :value 8 :strategy :f32-scalar}
                   {:expression k :op :< :value 8 :strategy :f32-scalar}
                   {:expression split-expression :op :>= :value 2 :strategy :xmx-split-k}]
           :default :xmx-direct}
          :provenance {:semantic-op :contraction :variant variant :lowering :gemm-schedule}
          :attributes {:tile tile :precision precision :hardware-aware? true}}))
      (throw (ex-info "unsupported GEMM precision" {:id id :precision precision})))))
