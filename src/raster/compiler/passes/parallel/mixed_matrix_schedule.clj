(ns raster.compiler.passes.parallel.mixed-matrix-schedule
  "Target-neutral construction of the existing mixed-precision contraction graph.

   Representation conversion, layout, matrix tiling, fusion and split-K combination live here.
   Target admission and artifact emission remain outside this pass. A structural refinement is
   retained producer evidence, not by itself an algorithm or complete-write proof."
  (:require [clojure.walk :as walk]
            [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.contraction-facts :as contraction-facts]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-body :as kbody]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.ir.layout-stage :as layout-stage]
            [raster.compiler.ir.matrix-stage :as matrix-stage]
            [raster.compiler.ir.scheduled-graph-refinement :as graph-refinement]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.contract-lower :as contract-lower]
            [raster.compiler.passes.parallel.contraction-schedule :as contraction-schedule]
            [raster.compiler.passes.parallel.matrix-input-fusion :as input-fusion]
            [raster.compiler.passes.parallel.map-read-requirements :as read-requirements]
            [raster.compiler.passes.parallel.typed-contraction-context :as typed-context]))

(defn ^:no-doc graph-buffer
  [id dtype elements role]
  (kgraph/buffer id dtype elements :device role))

(defn- ordered-public-buffers
  "Preserve the supplied public argument order, independently of matrix lhs/rhs order.
   Only reorder locally derived buffer descriptions; do not copy facts from the source witness."
  [buffers arguments]
  (let [positions (zipmap arguments (range))]
    (when-not (every? #(contains? positions (:id %)) buffers)
      (throw (ex-info "matrix schedule buffer is absent from its public interface"
                      {:reason :gemm-public-buffer-interface
                       :buffers buffers :arguments arguments})))
    (vec (sort-by #(get positions (:id %)) buffers))))

(defn ^:no-doc value-use
  [buffer access]
  (kgraph/->ValueUse buffer access))

(defn- make-stage-graph
  "Keep an independently supplied semantic boundary, checking its storage minima.
   Logical matrix volumes size private stages, not a replacement callable interface."
  [description source]
  (if-not source
    (kgraph/make description)
    (let [boundary (kgraph/boundary-contract source)
          expected (vec (concat (:inputs description) (:outputs description)))
          actual (into {} (map (juxt :id identity))
                       (concat (:inputs boundary) (:outputs boundary)))]
      (when-not (and (= (set (map :id expected)) (set (keys actual)))
                     (set/subset? (set (map :id (:scalars description)))
                                  (set (map :id (:scalars boundary)))))
        (throw (ex-info "matrix stage graph changed its semantic boundary identities"
                        {:reason :mixed-matrix-public-boundary})))
      (doseq [required expected
              :let [id (:id required) provided (get actual id)]]
        (when-not (and (= (select-keys required [:dtype :memory-space :role])
                         (select-keys provided [:dtype :memory-space :role]))
                       (read-requirements/graph-capacity-covers?
                        (:elements provided) (:elements required) (:preconditions boundary)))
          (throw (ex-info "semantic storage does not cover the matrix stage domain"
                          {:reason :mixed-matrix-public-storage
                           :value id :required required :provided provided}))))
      (kgraph/make (merge description boundary)))))

(defn ^:no-doc stage-node
  [id operation uses scalar-values dependencies]
  (kgraph/->ScheduledKernel
   id operation (vec uses)
   (reduce into #{} (map klaunch/expression-references scalar-values))
   (vec dependencies)))

(defn- epilogue-interface
  [epilogue]
  (vec
   (concat
    (for [{:keys [sym dtype] :or {dtype :float}} (:operands epilogue)]
      [(kabi/slot sym :input dtype :c-name (name sym) :role :epilogue)
       sym])
    (for [{:keys [sym dtype] :or {dtype :float}} (:scalars epilogue)]
      [(kabi/slot sym :scalar dtype :c-name (name sym) :role :epilogue)
       sym]))))

(defn- epilogue-buffer-specs
  [epilogue]
  (mapv (fn [{:keys [sym dtype map] :or {dtype :float}}]
          (let [shape (axis-map/shape map)
                elements (if (seq shape) (apply klaunch/product shape) 1)]
            {:id sym :dtype dtype :elements elements}))
        (:operands epilogue)))

(defn- outer-interface
  [{:keys [a b c m n k epilogue]}]
  (let [base [[(kabi/slot a :input :float :c-name "A" :role :lhs) a]
              [(kabi/slot b :input :float :c-name "B" :role :rhs) b]
              [(kabi/slot c :output :float :c-name "C" :role :result) c]
              [(kabi/slot m :scalar :int :c-name "M" :role :extent) m]
              [(kabi/slot n :scalar :int :c-name "N" :role :extent) n]
              [(kabi/slot k :scalar :int :c-name "K" :role :extent) k]]
        interface (into base (epilogue-interface epilogue))]
    {:abi (mapv first interface)
     :arguments (mapv second interface)}))

(defn- batched-outer-interface
  [{:keys [a b c batch m n k epilogue]}]
  (let [base [[(kabi/slot a :input :float :c-name "A" :role :lhs) a]
              [(kabi/slot b :input :float :c-name "B" :role :rhs) b]
              [(kabi/slot c :output :float :c-name "C" :role :result) c]
              [(kabi/slot batch :scalar :int :c-name "batch" :role :extent) batch]
              [(kabi/slot m :scalar :int :c-name "M" :role :extent) m]
              [(kabi/slot n :scalar :int :c-name "N" :role :extent) n]
              [(kabi/slot k :scalar :int :c-name "K" :role :extent) k]]
        interface (into base (epilogue-interface epilogue))]
    {:abi (mapv first interface)
     :arguments (mapv second interface)}))

(defn ^:no-doc public-outer-interface
  [spec]
  (let [{:keys [abi arguments]} (outer-interface spec)]
    (kgraph/public-interface abi arguments)))

(defn- public-batched-outer-interface
  [spec]
  (let [{:keys [abi arguments]} (batched-outer-interface spec)]
    (kgraph/public-interface abi arguments)))

(defn ^:no-doc extents
  [{:keys [m n k variant]}]
  {:a-elements (if (contains? #{:tn :tt} variant)
                 (klaunch/product k m) (klaunch/product m k))
   :b-elements (if (contains? #{:nt :tt} variant)
                 (klaunch/product n k) (klaunch/product k n))
   :c-elements (klaunch/product m n)})

(defn ^:no-doc effects
  [{:keys [a b c epilogue]}]
  {:kind :tensor-contraction
   :reads (into [a b] (map :sym) (:operands epilogue))
   :writes [c]})

(defn- convert-stage
  [stage-id in out elements vector-width]
  (layout-stage/make
   {:id stage-id :operation :cast :input in :output out
    :input-shape [elements] :output-shape [elements]
    :input-dtype :float :output-dtype :half
    :policy {:rounding :nearest-even :overflow :ieee
             :vector-width vector-width}}))

(defn- transpose-stage
  [stage-id in out rows cols]
  (layout-stage/make
   {:id stage-id :operation :transpose :input in :output out
    :input-shape [rows cols] :output-shape [cols rows]
    :input-dtype :half :output-dtype :half
    :policy {:permutation [1 0]}}))

(defn- matrix-stage-for
  [{:keys [m n k axis-symbols epilogue tile]} stage-id a b c split-k? kc splits]
  (let [reduction (if split-k?
                    (let [slice 'k-slice
                          lower (kbody/expression :mul slice kc)]
                      {:kind :split-k :slice slice :chunk kc :partitions splits
                       :range [lower (kbody/expression
                                      :min (kbody/expression :add lower kc) k)]})
                    {:kind :full :range [0 k]})]
    (matrix-stage/make
     {:id stage-id
      :lhs a :rhs b :result c :dimensions [m n k]
      :axis-symbols (or axis-symbols ['i 'j 'l])
      :reduction reduction
      :result-shape (if split-k? [splits m n] [m n])
      :epilogue (when-not split-k? epilogue)
      :schedule {:kind :matrix-instruction-tiling :tile tile}})))

(defn ^:no-doc split-k-combine-plan
  ([stage-id] (split-k-combine-plan stage-id {'mn :int 'splits :int}))
  ([stage-id scalar-types]
  (let [facts (contraction-facts/from-components
               {:out 'C :free-axes '[[i mn]] :contract-axes '[[s splits]] :dtype :float
                :local-identities {:accumulator '__raster_split_combine_acc}
                :body '(clojure.core/aget
                        partials (clojure.core/+ (clojure.core/* s mn) i))})
        operation (contract-lower/contraction-facts->segred
                   facts :id stage-id :flat-idx '__raster_split_combine_tid)
        planned (contraction-schedule/plan-portable-body
                 facts operation {}
                 {:array-types {'partials :float 'C :float}
                  :scalar-types scalar-types})
        _ (when-not (:ok planned)
            (throw (ex-info "split-K combination did not admit the portable contraction schedule"
                            {:reason :raster/bug :plan planned})))]
    {:operation operation :body (:body planned) :plan planned})))

(defn- refinement-numerics
  [{:keys [epilogue]} split-k?]
  (cond-> {:mode :bounded-error
           :policy :f32-input-f16-matrix-f32-output
           :rounding :nearest-even
           :accumulator-dtype :float
           :error-model {:kind :composed-mixed-precision-stages
                         :operand-conversion {:from :float :to :half
                                              :rounding :nearest-even :overflow :ieee}
                         :reduction-order
                         (if split-k?
                           {:kind :split-k
                            :within-partition :tiled
                            :partial-combine :ordered-sequential}
                           {:kind :tiled})}}
    (seq epilogue)
    (assoc :result-transform
           {:kind :typed-scalar-region
            :policy :same-typed-ssa-evaluation-order
            :input-dtype :float :result-dtype :float})))

(defn- make-refinement
  [stage-graph source-operation source-graph
   {:keys [strategy variant tile vector-width requested-splits split-k? batch batching
           fuse-tile-inputs? fuse-lhs-cast?] :as spec}]
  (when source-operation
    (when-not source-graph
      (throw (ex-info "typed mixed-precision scheduling requires its independent source graph"
                      {:reason :gemm-refinement-source-graph
                       :operation (:id source-operation)})))
    (let [source-graph (kgraph/validate! source-graph)]
      (when-not (identical? source-operation (-> source-graph :nodes first :operation))
        (throw (ex-info "mixed-precision refinement source graph lost exact SegRed identity"
                        {:reason :gemm-refinement-source
                         :operation (:id source-operation)})))
      (graph-refinement/make
       {:source source-graph
        :graph stage-graph
        :schedule {:kind :mixed-precision-contraction :version 1
                   :strategy strategy :variant variant :tile tile
                   :vector-width vector-width :split-k? (boolean split-k?)
                   :requested-splits requested-splits
                   :batched? (some? batch)
                   ;; Reconstruction consumes explicit physical choices, never guesses
                   ;; fusion or shared operands from a strategy label or emitted graph.
                   :input-fusion (cond
                                   (some? batch) :batched-tile-casts
                                   fuse-tile-inputs? :tile-inputs
                                   fuse-lhs-cast? :lhs-cast
                                   :else :materialized)
                   :batching (when (some? batch)
                               {:extent batch :lhs (get batching :row true)
                                :rhs (get batching :col true)})}
        :numerics (refinement-numerics spec split-k?)
        :provenance {:operation-id (:id source-operation)
                     :source-dialect :typed-soac}
        :attributes {:compiler-stage :gemm-graph-schedule}}))))

(defn plan
  "Construct the existing mixed-precision stage graph without target emission.

   Returns {:graph :refinement}, or nil when a requested input fusion declines.
   This is schedule construction, not proof of the complete-write or numerical law."
  [{:keys [id a b c m n k variant tile vector-width requested-splits split-k? epilogue
           strategy source-operation source-graph external-interface]
    :as spec}]
  (let [{:keys [abi arguments effects]}
        (or (when (:abi external-interface) external-interface)
            (assoc (public-outer-interface spec) :effects (effects spec)))
        {:keys [a-elements b-elements c-elements]} (extents spec)
        epilogue-buffers (epilogue-buffer-specs epilogue)
        strategy (or strategy (if split-k? :xmx-split-k :xmx-direct))
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
        convert-a (convert-stage convert-a-id a a16 a-elements vector-width)
        convert-b (convert-stage convert-b-id b b16 b-elements vector-width)
        transpose-a (when (contains? #{:tn :tt} variant)
                      (transpose-stage transpose-a-id a16 at16 k m))
        transpose-b (when (contains? #{:nt :tt} variant)
                      (transpose-stage transpose-b-id b16 bt16 n k))
        contract-output (if split-k? partials c)
        contract (matrix-stage-for spec contract-id final-a final-b contract-output
                                   split-k? kc splits)
        combine (when split-k?
                  (walk/postwalk-replace
                   {'partials partials 'C c 'mn c-elements 'splits splits}
                   (:operation (split-k-combine-plan [:gemm id strategy :combine]))))
        nodes (cond->
               [(stage-node convert-a-id convert-a
                            [(value-use a :read) (value-use a16 :write)] [a-elements] [])
                (stage-node convert-b-id convert-b
                            [(value-use b :read) (value-use b16 :write)] [b-elements] [])]
                transpose-a
                (conj (stage-node transpose-a-id transpose-a
                                  [(value-use a16 :read) (value-use at16 :write)]
                                  [k m] [convert-a-id]))
                transpose-b
                (conj (stage-node transpose-b-id transpose-b
                                  [(value-use b16 :read) (value-use bt16 :write)]
                                  [n k] [convert-b-id]))
                true
                (conj (stage-node
                       contract-id contract
                       (into [(value-use final-a :read) (value-use final-b :read)
                              (value-use contract-output :write)]
                             (map #(value-use (:id %) :read)) epilogue-buffers)
                       (vec (concat [m n k] (when split-k? [kc splits])
                                    (map :sym (:scalars epilogue))))
                       [(if transpose-a transpose-a-id convert-a-id)
                        (if transpose-b transpose-b-id convert-b-id)]))
                combine
                (conj (stage-node [:gemm id strategy :combine] combine
                                  [(value-use partials :read) (value-use c :write)]
                                  (vec (segop/operation-scalars combine)) [contract-id])))
        temporaries (cond-> [(graph-buffer a16 :half a-elements :temporary)
                             (graph-buffer b16 :half b-elements :temporary)]
                      transpose-a (conj (graph-buffer at16 :half a-elements :temporary))
                      transpose-b (conj (graph-buffer bt16 :half b-elements :temporary))
                      split-k? (conj (graph-buffer partials :float partial-elements :temporary)))]
    (let [stage-graph
          (make-stage-graph
           {:inputs (ordered-public-buffers
                     (into [(graph-buffer a :float a-elements :input)
                            (graph-buffer b :float b-elements :input)]
                           (map #(graph-buffer (:id %) (:dtype %) (:elements %) :input))
                           epilogue-buffers)
                     arguments)
            :outputs [(graph-buffer c :float c-elements :output)]
            :temporaries temporaries
            :scalars (kgraph/interface-scalars abi arguments)
            :nodes nodes
            :abi abi :arguments arguments
            :effects effects
            :provenance {:semantic-op :contraction :variant variant
                         :lowering :xmx-gemm-schedule}
            :attributes {:strategy strategy :variant variant :precision :mixed-f16-f32
                         :tile tile :vector-width vector-width
                         :requested-splits requested-splits}} source-graph)
          stage-graph
          (cond
            (:fuse-tile-inputs? spec)
            (some-> (case variant
                      :nn stage-graph
                      :nt (input-fusion/fuse-input-transpose
                           stage-graph transpose-b-id contract-id :rhs))
                    (input-fusion/fuse-lhs-cast convert-a-id contract-id)
                    (input-fusion/fuse-rhs-cast convert-b-id contract-id))

            (:fuse-lhs-cast? spec)
            (input-fusion/fuse-lhs-cast stage-graph convert-a-id contract-id)

            :else stage-graph)]
      (when stage-graph
        (let [emit-spec (assoc spec :strategy strategy)
              refinement (make-refinement stage-graph source-operation source-graph emit-spec)]
          {:graph stage-graph :refinement refinement})))))

(defn plan-batched
  "Construct the existing batched NN/NT mixed-matrix stage graph without emission."
  [{:keys [id a b c batch m n k variant tile vector-width batching epilogue
           source-operation source-graph external-interface]
    :or {vector-width 4 batching {:row true :col true}}
    :as spec}]
  (when-not (contains? #{:nn :nt} variant)
    (throw (ex-info "batched matrix schedule requires NN or NT storage"
                    {:reason :batched-matrix-layout-not-lowered
                     :id id :variant variant})))
  (doseq [[field value] [[:id id] [:a a] [:b b] [:c c] [:batch batch]
                         [:m m] [:n n] [:k k] [:tile tile]]]
    (when (nil? value)
      (throw (ex-info "batched matrix schedule is missing a required field"
                      {:reason :raster/bug :field field :spec spec}))))
  (let [{:keys [abi arguments effects]}
        (or (when (:abi external-interface) external-interface)
            (assoc (public-batched-outer-interface spec) :effects (effects spec)))
        a-elements (if (get batching :row true)
                     (klaunch/product batch m k)
                     (klaunch/product m k))
        b-elements (if (get batching :col true)
                     (klaunch/product batch k n)
                     (klaunch/product k n))
        c-elements (klaunch/product batch m n)
        epilogue-buffers (epilogue-buffer-specs epilogue)
        a16 [:gemm id :xmx-batched :a16]
        b16 [:gemm id :xmx-batched :b16]
        convert-a-id [:gemm id :xmx-batched :convert-a]
        convert-b-id [:gemm id :xmx-batched :convert-b]
        contract-id [:gemm id :xmx-batched :contract]
        convert-a (convert-stage convert-a-id a a16 a-elements vector-width)
        convert-b (convert-stage convert-b-id b b16 b-elements vector-width)
        contract (matrix-stage/make
                  {:id contract-id
                   :lhs a16 :rhs b16 :result c :dimensions [m n k]
                   :axis-symbols (or (:axis-symbols spec) ['i 'j 'l])
                   :batching {:extent batch
                              :lhs (get batching :row true)
                              :rhs (get batching :col true)}
                   :reduction {:kind :full :range [0 k]}
                   :result-shape [batch m n]
                   :epilogue epilogue
                   :schedule {:kind :matrix-instruction-tiling :tile tile}
                   :input-layouts
                   (cond-> {}
                     (= :nt variant)
                     (assoc b16 (layout/transpose-layout
                                 (layout/row-major [k n] :half))))})
        stage-graph
        (make-stage-graph
         {:inputs (ordered-public-buffers
                   (into [(graph-buffer a :float a-elements :input)
                          (graph-buffer b :float b-elements :input)]
                         (map #(graph-buffer (:id %) (:dtype %) (:elements %) :input))
                         epilogue-buffers)
                   arguments)
          :outputs [(graph-buffer c :float c-elements :output)]
          :temporaries [(graph-buffer a16 :half a-elements :temporary)
                        (graph-buffer b16 :half b-elements :temporary)]
          :scalars (kgraph/interface-scalars abi arguments)
          :nodes [(stage-node convert-a-id convert-a
                              [(value-use a :read) (value-use a16 :write)]
                              [a-elements] [])
                  (stage-node convert-b-id convert-b
                              [(value-use b :read) (value-use b16 :write)]
                              [b-elements] [])
                  (stage-node contract-id contract
                              (into [(value-use a16 :read) (value-use b16 :read)
                                     (value-use c :write)]
                                    (map #(value-use (:id %) :read)) epilogue-buffers)
                              (vec (concat [batch m n k]
                                           (map :sym (:scalars epilogue))))
                              [convert-a-id convert-b-id])]
          :abi abi :arguments arguments
          :effects effects
          :provenance {:semantic-op :contraction
                       :variant variant
                       :lowering :batched-xmx-gemm}
          :attributes {:strategy :xmx-batched
                       :variant variant
                       :batched? true
                       :batching batching
                       :result-transform? (boolean (seq epilogue))
                       :precision :mixed-f16-f32
                       :vector-width vector-width
                       :tile tile}} source-graph)
        stage-graph
        (or (some-> stage-graph
                    (input-fusion/fuse-lhs-cast convert-a-id contract-id)
                    (input-fusion/fuse-rhs-cast convert-b-id contract-id))
            (throw (ex-info "batched matrix input casts could not be fused into tile loads"
                            {:reason :batched-matrix-input-fusion
                             :id id :batching batching})))
        emit-spec (assoc spec :strategy :xmx-batched
                         :vector-width vector-width :batching batching)
        refinement (make-refinement stage-graph source-operation source-graph emit-spec)]
    {:graph stage-graph :refinement refinement}))

(defn reconstruct-refinement
  "Rebuild a mixed-matrix plan from a retained typed equation and independent source graph.

   Only closed physical choices come from the refinement recipe. Operand identities, dimensions,
   layout, batching, axis symbols and epilogue come from the typed algorithm. The source graph
   must be freshly derived by the caller, not borrowed from the candidate witness.

   Returns a reference plan, not validation authority: its graph and generated terminal bodies
   still need comparison with the candidate. In particular, a matching boundary is not a
   complete-write proof. Split-combine local identities are allocated by the collision-checked
   contraction constructor, so repeated reference plans compare exactly."
  [algorithm independent-source refinement]
  (let [source (kgraph/validate! independent-source)
        refinement (graph-refinement/validate-against! refinement source)
        _ (when-not (= 1 (count (:nodes source)))
            (throw (ex-info "mixed matrix reconstruction requires one semantic contraction"
                            {:reason :mixed-matrix-reconstruction-source})))
        operation (get-in source [:nodes 0 :operation])
        {:keys [facts operation-id dtype]} (typed-context/validate! algorithm operation)
        view (contraction-facts/dense-matrix-view facts)
        _ (when-not (and (= :float (dtype/canon dtype)) (:ok view))
            (throw (ex-info "typed equation does not admit the mixed matrix representation"
                            {:reason :mixed-matrix-reconstruction-algorithm
                             :dtype dtype :view view})))
        _ (when (some #(= :inout (:kind %)) (:abi source))
            (throw (ex-info "mixed matrix reconstruction requires a write-only result"
                            {:reason :mixed-matrix-reconstruction-inout})))
        recipe (:schedule refinement)
        fusion (:input-fusion recipe)
        _ (when-not (and (= :mixed-precision-contraction (:kind recipe))
                          (= 1 (:version recipe))
                          (= (:batched? view) (:batched? recipe))
                          (boolean? (:split-k? recipe))
                          (if (:batched? view)
                            (and (= :batched-tile-casts fusion) (not (:split-k? recipe)))
                            (and (contains? #{:materialized :lhs-cast :tile-inputs} fusion)
                                 (or (not (:split-k? recipe)) (= :materialized fusion)))))
            (throw (ex-info "mixed matrix recipe has an unsupported physical policy"
                            {:reason :mixed-matrix-reconstruction-policy :recipe recipe})))
        [m n k] (:dimensions view)
        {:keys [row col]} (:bindings view)
        spec (cond-> {:id [:typed-contraction operation-id]
                      :a row :b col :c (:out facts) :m m :n n :k k
                      :axis-symbols (vec (concat (map first (take-last 2 (:free-axes facts)))
                                                (map first (:contract-axes facts))))
                      :variant (:variant view) :epilogue (:epilogue view)
                      :strategy (:strategy recipe) :tile (:tile recipe)
                      :vector-width (:vector-width recipe)
                      :split-k? (:split-k? recipe)
                      :requested-splits (:requested-splits recipe)
                      :source-operation operation :source-graph source
                      :external-interface (select-keys source [:abi :arguments :effects])}
               (= :lhs-cast fusion) (assoc :fuse-lhs-cast? true)
               (= :tile-inputs fusion) (assoc :fuse-tile-inputs? true)
               (:batched? view) (assoc :batch (:batch view) :batching (:batching view)))
        planned ((if (:batched? view) plan-batched plan) spec)]
    (when-not (and planned (= recipe (get-in planned [:refinement :schedule]))
                   (= (:numerics refinement) (get-in planned [:refinement :numerics])))
      (throw (ex-info "mixed matrix recipe or numerics disagree with typed reconstruction"
                      {:reason :mixed-matrix-reconstruction-description})))
    planned))
