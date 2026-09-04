(ns raster.compiler.backend.gpu.gemm
  "Compiler-owned executable schedules for dense GEMM.

   The public call is uniformly `(A B C M N K)` over f32 resident buffers. A schedule may be one
   scalar kernel or a graph containing conversion, layout conversion, matrix contraction, and
   split-K combination. All mixed-precision scratch and derived scheduling scalars are private to
   the graph; callers never bind them and runtimes never reconstruct the algorithm from `:gemm`."
  (:require [clojure.string :as str]
            [raster.compiler.backend.gpu.kernel-body-opencl :as kernel-body-opencl]
            [raster.compiler.backend.gpu.layout-transform :as layout-emitter]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-body :as kbody]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.ir.contraction-facts :as contraction-facts]
            [raster.compiler.passes.parallel.contract-lower :as contract-lower]
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
  {:a-elements (if (contains? #{:tn :tt} variant)
                 (klaunch/product k m) (klaunch/product m k))
   :b-elements (if (contains? #{:nt :tt} variant)
                 (klaunch/product n k) (klaunch/product k n))
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

(defn- scalar-contraction-form
  [variant]
  (case variant
    :nn '(raster.par/contract C [[i m] [j n]] [[l k]]
                               (* (aget A (+ (* i k) l))
                                  (aget B (+ (* l n) j))))
    :nt '(raster.par/contract C [[i m] [j n]] [[l k]]
                               (* (aget A (+ (* i k) l))
                                  (aget B (+ (* j k) l))))
    :tn '(raster.par/contract C [[i m] [j n]] [[l k]]
                               (* (aget A (+ (* l m) i))
                                  (aget B (+ (* l n) j))))
    :tt '(raster.par/contract C [[i m] [j n]] [[l k]]
                               (* (aget A (+ (* l m) i))
                                  (aget B (+ (* j k) l))))))

(defn emit-portable-scalar-matrix-kernel
  "Lower a dynamic f32 NN/NT/TN/TT matrix product through the portable contraction schedule."
  ([kernel-name variant]
   (emit-portable-scalar-matrix-kernel kernel-name variant :opencl-intel))
  ([kernel-name variant target-dialect]
   (let [form (scalar-contraction-form variant)
         facts (contraction-facts/contraction-facts form :dtype :float)
         operation (contract-lower/contract-form->segred form :dtype :float :facts facts)
         planned (contraction-schedule/plan-portable-body
                  facts operation {}
                  {:array-types {'A :float 'B :float 'C :float}
                   :scalar-types {'m :int 'n :int 'k :int}})
         _ (when-not (:ok planned)
             (throw (ex-info "matrix product did not admit the portable contraction schedule"
                             {:reason :raster/bug :variant variant :plan planned})))
         kernel-body (:body planned)]
     {:source
      (kernel-body-opencl/emit-scalar-kernel
       kernel-name kernel-body
       {:target-dialect target-dialect
        :parameter-names {'A "A" 'B "B" 'C "C" 'k "k" 'm "m" 'n "n"
                          '_nseg "_nseg"}})
      :kernel-body kernel-body
      :workgroup-size 256})))

(defn- scalar-graph
  [{:keys [id a b c m n k variant] :as spec}]
  (let [{:keys [abi arguments]} (outer-interface spec)
        {:keys [a-elements b-elements c-elements]} (extents spec)
        prefix (identifier (str id "_scalar"))
        kernel-name (str prefix "_gemm")
        emitted (emit-portable-scalar-matrix-kernel kernel-name variant)
        gemm (artifact
              kernel-name
              (:source emitted)
              [(kabi/slot 'A :input :float :c-name "A" :role :operand
                          :aliasing :no-write-alias)
               (kabi/slot 'B :input :float :c-name "B" :role :operand
                          :aliasing :no-write-alias)
               (kabi/slot 'C :output :float :c-name "C" :role :result)
               (kabi/slot 'k :scalar :int :c-name "k" :role :extent)
               (kabi/slot 'm :scalar :int :c-name "m" :role :extent)
               (kabi/slot 'n :scalar :int :c-name "n" :role :extent)
               (kabi/slot '_nseg :scalar :int :c-name "_nseg" :role :bound)]
              [a b c k m n c-elements]
              (klaunch/spec {:workgroup-size [256]
                             :group-count [(klaunch/ceil-div c-elements 256)]})
              :scalar-gemm {:variant variant :kernel-body (:kernel-body emitted)
                            :semantic-op :contraction})]
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
  (let [{:keys [source kernel-body]}
        (layout-emitter/emit-cast-kernel
         {:kernel-name kernel-name :input in :output out
          :source-dtype :float :destination-dtype :half :vector-width vector-width
          :rounding :nearest-even :overflow :ieee})]
    (artifact
     kernel-name source
     [(kabi/slot in :input :float :c-name "in")
      (kabi/slot out :output :half :c-name "out")
      (kabi/slot :layout-elements :scalar :int :c-name "n")]
     [in out elements]
     (klaunch/spec
      {:workgroup-size [256]
       :group-count [(klaunch/ceil-div
                      (klaunch/ceil-div (klaunch/runtime-value elements) vector-width)
                      256)]})
     phase {:vector-width vector-width :from :float :to :half
            :rounding :nearest-even :overflow :ieee
            :kernel-body kernel-body :cacheable-transform? true})))

(defn- transpose-artifact
  [kernel-name in out rows cols phase]
  (let [{:keys [source kernel-body]}
        (layout-emitter/emit-transpose-kernel
         {:kernel-name kernel-name :input in :output out :element-dtype :half})]
    (artifact
     kernel-name source
     [(kabi/slot in :input :half :c-name "in")
      (kabi/slot out :output :half :c-name "out")
      (kabi/slot :layout-rows :scalar :int :c-name "rows")
      (kabi/slot :layout-cols :scalar :int :c-name "cols")]
     [in out rows cols]
     (klaunch/spec {:workgroup-size [256]
                    :group-count [(klaunch/ceil-div (klaunch/product rows cols) 256)]})
     phase {:layout :transpose :dtype :half
            :kernel-body kernel-body :cacheable-transform? true})))

(defn emit-scheduled-matrix-kernel
  "Build and directly lower one canonical f16 matrix contraction.

  This is the shared compiler entry for graph-owned, legacy-plan, and resident direct/tiled GEMM
  front doors. Caller identities remain the KernelBody ABI identities; OpenCL parameter spelling
  is solely a target concern. Optional views, hardware indices, and K bounds are explicit schedule
  values used by the split-K and batched wrappers below. An optional epilogue becomes a typed
  ScalarRegion on every store and is lowered as part of the body."
  [{:keys [kernel-name id a b c m n k tile result-dtype provenance
           additional-parameters additional-indices buffer-shapes buffer-views operation-buffers
           k-range launch-group-count attributes parameter-names epilogue]
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
          :epilogue epilogue
          :result-dtype result-dtype
          :additional-parameters additional-parameters
          :additional-indices additional-indices
          :buffer-shapes buffer-shapes
          :buffer-views buffer-views
          :operation-buffers operation-buffers
          :k-range k-range
          :launch-group-count launch-group-count
          :attributes attributes
          :provenance (merge {:dialect :gemm :lowering :scheduled-matrix} provenance)})]
    {:source (kernel-body-opencl/emit-matrix-kernel
              kernel-name kernel-body {:parameter-names parameter-names})
     :kernel-body kernel-body
     :workgroup-size (get-in kernel-body [:launch :workgroup-size])}))

(defn emit-scheduled-split-k-kernel
  "Lower a grid-Z partition of the K reduction into disjoint f32 output views."
  [{:keys [kernel-name id a b c m n k kc splits tile provenance]}]
  (let [z 'k-slice
        c-view 'split-result-view
        k-lower (kbody/expression :mul z kc)
        k-upper (kbody/expression :min (kbody/expression :add k-lower kc) k)]
    (emit-scheduled-matrix-kernel
     {:kernel-name kernel-name :id id :a a :b b :c c :m m :n n :k k
      :tile tile :result-dtype :float :provenance provenance
      :additional-parameters [(kbody/->KernelParameter kc :scalar :int [] nil nil :schedule)
                              (kbody/->KernelParameter splits :scalar :int [] nil nil :schedule)]
      :additional-indices [(kbody/->IndexBinding z :group 2)]
      :buffer-shapes {c [splits m n]}
      :buffer-views [{:id c-view :buffer c
                      :element-offset (kbody/expression :mul z m n)
                      :shape [m n]}]
      :operation-buffers {c c-view}
      :k-range [k-lower k-upper]
      :launch-group-count [(klaunch/ceil-div (klaunch/runtime-value n) (:block-n tile))
                           (klaunch/ceil-div (klaunch/runtime-value m) (:block-m tile))
                           (klaunch/runtime-value splits)]
      :attributes {:grid-z {:index z :extent splits :purpose :reduction-partition}}
      :parameter-names {kc "KC" splits "splits"}})))

(defn emit-scheduled-batched-matrix-kernel
  "Lower independent dense matrix slabs as grid-Z-selected contiguous buffer views."
  [{:keys [kernel-name id a b c m n k batch tile provenance]}]
  (let [z 'slab
        a-view 'batch-lhs-view
        b-view 'batch-rhs-view
        c-view 'batch-result-view]
    (emit-scheduled-matrix-kernel
     {:kernel-name kernel-name :id id :a a :b b :c c :m m :n n :k k
      :tile tile :result-dtype :float :provenance provenance
      :additional-parameters [(kbody/->KernelParameter batch :scalar :int [] nil nil :schedule)]
      :additional-indices [(kbody/->IndexBinding z :group 2)]
      :buffer-shapes {a [batch m k] b [batch k n] c [batch m n]}
      :buffer-views [{:id a-view :buffer a
                      :element-offset (kbody/expression :mul z m k) :shape [m k]}
                     {:id b-view :buffer b
                      :element-offset (kbody/expression :mul z k n) :shape [k n]}
                     {:id c-view :buffer c
                      :element-offset (kbody/expression :mul z m n) :shape [m n]}]
      :operation-buffers {a a-view b b-view c c-view}
      :launch-group-count [(klaunch/ceil-div (klaunch/runtime-value n) (:block-n tile))
                           (klaunch/ceil-div (klaunch/runtime-value m) (:block-m tile))
                           (klaunch/runtime-value batch)]
      :attributes {:grid-z {:index z :extent batch :purpose :independent-slices}}
      :parameter-names {batch "batch"}})))

(defn- gemm-artifact
  [{:keys [id m n k tile]} kernel-name a b c split-k? kc splits phase]
  (let [{:keys [block-m block-n]} tile
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
        emit-args {:kernel-name kernel-name
                   :id [:gemm id phase]
                   :a a :b b :c c :m m :n n :k k
                   :tile tile :result-dtype :float
                   :provenance {:operation-id id :phase phase}}
        scheduled (if split-k?
                    (emit-scheduled-split-k-kernel
                     (assoc emit-args :kc :k-chunk :splits :splits))
                    (emit-scheduled-matrix-kernel emit-args))]
    (artifact
     kernel-name (:source scheduled)
     abi arguments
     (klaunch/spec {:workgroup-size (:workgroup-size scheduled)
                    :group-count group-count})
     phase (cond-> {:tile tile :split-k? split-k? :accumulator-dtype :float}
             scheduled (assoc :kernel-body (:kernel-body scheduled))))))

(defn emit-split-k-combine-kernel
  "Lower C[i] = sum_s partials[s, i] through the generic portable contraction schedule."
  ([kernel-name] (emit-split-k-combine-kernel kernel-name :opencl-intel))
  ([kernel-name target-dialect]
   (let [form '(raster.par/contract C [[i mn]] [[s splits]]
                                     (clojure.core/aget
                                      partials (clojure.core/+ (clojure.core/* s mn) i)))
        facts (contraction-facts/contraction-facts form :dtype :float)
        operation (contract-lower/contract-form->segred form :dtype :float :facts facts)
        planned (contraction-schedule/plan-portable-body
                 facts operation {}
                 {:array-types {'partials :float 'C :float}
                  :scalar-types {'mn :int 'splits :int}})
        _ (when-not (:ok planned)
            (throw (ex-info "split-K combination did not admit the portable contraction schedule"
                            {:reason :raster/bug :plan planned})))
        kernel-body (:body planned)
        source (kernel-body-opencl/emit-scalar-kernel
                kernel-name kernel-body
                {:target-dialect target-dialect
                 :parameter-names {'partials "partials" 'C "C"
                                   'mn "mn" 'splits "splits" '_nseg "_nseg"}})]
     {:source source :kernel-body kernel-body :workgroup-size 256})))

(defn- combine-artifact
  [kernel-name partials c mn splits]
  (let [{:keys [source kernel-body]} (emit-split-k-combine-kernel kernel-name)]
    (artifact
     kernel-name source
     [(kabi/slot 'partials :input :float :c-name "partials" :role :operand
                 :aliasing :no-write-alias)
      (kabi/slot 'C :output :float :c-name "C" :role :result)
      (kabi/slot 'mn :scalar :int :c-name "mn" :role :extent)
      (kabi/slot 'splits :scalar :int :c-name "splits" :role :extent)
      (kabi/slot '_nseg :scalar :int :c-name "_nseg" :role :bound)]
     [partials c mn splits mn]
     (klaunch/spec {:workgroup-size [256]
                    :group-count [(klaunch/ceil-div (klaunch/runtime-value mn) 256)]})
     :split-k-combine {:accumulator-dtype :float :kernel-body kernel-body
                       :semantic-op :contraction})))

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
        final-a (if (contains? #{:tn :tt} variant) at16 a16)
        final-b (if (contains? #{:nt :tt} variant) bt16 b16)
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
        transpose-a (when (contains? #{:tn :tt} variant)
                      (transpose-artifact (str prefix "_transpose_a") a16 at16 k m
                                          :transpose-a))
        transpose-b (when (contains? #{:nt :tt} variant)
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

(defn mixed-dpas-schedule
  "Return the target-derived schedule facts for the current mixed f16×f16→f32 DPAS graph.

   A nil result is an honest target decline.  CUDA MMA and HIP MFMA will be separate target
   lowering rows over the same typed contraction; an arbitrary `:matrix` capability must never be
   emitted with Intel DPAS source."
  [desc requested-tile]
  (let [{:keys [family m n k subgroup]} (:matrix desc)
        backend (:backend desc)]
    (when (and (contains? #{:ze :opencl} backend)
               (= :dpas family) (= [8 16 16] [m n k])
               (contains? #{8 16} subgroup)
               (contains? (hardware/supported-subgroup-sizes desc) (long subgroup)))
      (let [tile (or requested-tile (hardware/gemm-tile-for desc))
            workgroup-size (* (quot (:block-m tile) (:sg-m tile))
                              (quot (:block-n tile) (:sg-n tile))
                              subgroup)]
        {:tile tile
         :fill-workgroups (hardware/fill-workgroups desc workgroup-size)
         :matrix (:matrix desc)}))))

(defn emit-executable
  "Emit the resident GEMM schedule as one graph or a checked runtime dispatch.

   Required keys: :id, :a/:b/:c, :m/:n/:k compiler values, :variant, :precision, :tile,
   :fill-workgroups. The mixed-precision dispatch handles the XMX pitch gate and low-occupancy
   split-K choice entirely through generic expression cases."
  [{:keys [id a b c m n k variant precision tile fill-workgroups vector-width]
    :or {vector-width 4}
    :as spec}]
  (when-not (contains? #{:nn :nt :tn :tt} variant)
    (throw (ex-info "GEMM executable requires :nn, :nt, :tn, or :tt variant"
                    {:id id :variant variant})))
  (doseq [[field value] [[:id id] [:a a] [:b b] [:c c] [:m m] [:n n] [:k k]
                         [:precision precision] [:tile tile] [:fill-workgroups fill-workgroups]]]
    (when (nil? value)
      (throw (ex-info "GEMM executable is missing a required field" {:field field :spec spec}))))
  (let [spec (assoc spec :vector-width vector-width)
        scalar (scalar-graph spec)]
    (case precision
      :f32-scalar scalar
      :mixed-f16-f32
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
                   {:expression (kbody/expression :mod n 8)
                    :op :> :value 0 :strategy :f32-scalar}
                   {:expression (kbody/expression :mod k (get-in tile [:matrix :k]))
                    :op :> :value 0 :strategy :f32-scalar}
                   {:expression split-expression :op :>= :value 2 :strategy :xmx-split-k}]
           :default :xmx-direct}
          :provenance {:semantic-op :contraction :variant variant :lowering :gemm-schedule}
          :attributes {:tile tile :precision precision :hardware-aware? true}}))
      (throw (ex-info "unsupported GEMM precision" {:id id :precision precision})))))
