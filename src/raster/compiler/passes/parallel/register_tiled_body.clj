(ns raster.compiler.passes.parallel.register-tiled-body
  "Lower a verified dense contraction to a cooperative register-tiled KernelBody.

   This is a schedule, not a GEMM operation: contraction facts retain the additive reduction,
   operand maps and result transform. The body makes workgroup staging, barriers, the scalar
   microtile and 2-D launch explicit so every C-family target consumes the same scheduled value."
  (:require [clojure.walk :as walk]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.passes.parallel.contraction-body :as contraction-body]
            [raster.compiler.passes.parallel.scalar-region-lower :as scalar-region-lower]))

(def default-tile
  {:block-m 64 :block-n 64 :block-k 16 :thread-m 4 :thread-n 4})

(def ^:private tile-candidates
  [default-tile
   {:block-m 32 :block-n 32 :block-k 16 :thread-m 4 :thread-n 4}
   {:block-m 16 :block-n 16 :block-k 8 :thread-m 4 :thread-n 4}
   {:block-m 8 :block-n 8 :block-k 8 :thread-m 2 :thread-n 2}])

(defn- decline!
  [rule message data]
  (throw (ex-info message (assoc data
                                 :reason :register-tiled-kernel-body-declined
                                 :missing-rule rule))))

(defn declined?
  [exception]
  (or (= :register-tiled-kernel-body-declined (:reason (ex-data exception)))
      (scalar-region-lower/declined? exception)))

(defn- tile-resources
  [tile storage-dtype]
  (let [{:keys [block-m block-n block-k thread-m thread-n]} tile]
    {:workgroup-size (* (quot block-m thread-m) (quot block-n thread-n))
     :shared-memory-bytes (* block-k (+ block-m block-n)
                             (dtype/bytes-of storage-dtype))}))

(defn- valid-tile?
  [tile]
  (let [{:keys [block-m block-n block-k thread-m thread-n]} tile]
    (and (every? pos-int? [block-m block-n block-k thread-m thread-n])
         (zero? (mod block-m thread-m))
         (zero? (mod block-n thread-n)))))

(defn- resource-limits
  [descriptor]
  {:workgroup-size (long (or (get-in descriptor [:execution :max-workgroup-size])
                             (:max-workgroup-size descriptor) Long/MAX_VALUE))
   :shared-memory-bytes (long (or (:shared-local-memory descriptor)
                                  (get-in descriptor [:memory :shared-local-memory])
                                  (get-in descriptor [:memory :local-bytes])
                                  Long/MAX_VALUE))})

(defn- feasible-tile?
  [tile storage-dtype limits]
  (let [resources (tile-resources tile storage-dtype)]
    (and (<= (:workgroup-size resources) (:workgroup-size limits))
         (<= (:shared-memory-bytes resources) (:shared-memory-bytes limits)))))

(defn- select-tile
  [requested descriptor storage-dtype]
  (let [limits (resource-limits descriptor)]
    (if requested
      (cond
        (not (valid-tile? requested))
        (throw (ex-info "register-tiled schedule has invalid tile divisibility"
                        {:reason :raster/bug :tile requested}))

        (feasible-tile? requested storage-dtype limits)
        requested

        :else
        (decline! :target-resources
                  "register-tiled schedule exceeds target workgroup or shared-memory limits"
                  {:tile requested :resources (tile-resources requested storage-dtype)
                   :limits limits}))
      (or (first (filter #(feasible-tile? % storage-dtype limits) tile-candidates))
          (decline! :target-resources
                    "target cannot host any register-tiled schedule candidate"
                    {:candidates (mapv #(assoc % :resources
                                               (tile-resources % storage-dtype))
                                       tile-candidates)
                     :limits limits})))))

(defn- additive?
  [combine]
  (contains? '#{+ clojure.core/+ raster.numeric/+} combine))

(defn- identifier
  [prefix & parts]
  (symbol (apply str prefix (map #(str "-" %) parts))))

(defn- add [& values] (apply body/expression :add values))
(defn- mul [& values] (apply body/expression :mul values))
(defn- div [left right] (body/expression :floor-div left right))
(defn- modulo [left right] (body/expression :mod left right))

(defn- barrier
  []
  (body/->WorkgroupBarrier
   :workgroup #{:workgroup} :acquire-release (body/full-participation)))

(defn- transform-parameters
  [result-transform base-parameters]
  (let [base-ids (set (map :id base-parameters))]
    (vec
     (concat
      (for [{:keys [sym dtype map] :or {dtype :float}} (:operands result-transform)
            :when (not (contains? base-ids sym))]
        (let [shape (axis-map/shape map)]
          (body/->KernelParameter sym :input (dtype/canon dtype) shape :global
                                  (layout/row-major shape dtype) :epilogue)))
      (for [{:keys [sym dtype] :or {dtype :float}} (:scalars result-transform)
            :when (not (contains? base-ids sym))]
        (body/->KernelParameter sym :scalar (dtype/canon dtype) [] nil nil :epilogue))))))

(defn- staging-loop
  [{:keys [id buffer allocation dtype elements thread-id workgroup-width
           row-coordinate col-coordinate source-coordinates valid-mask]}]
  (let [index (identifier id "index")
        value (identifier id "value")]
    (body/->ForLoop
     (body/value index :int) thread-id elements workgroup-width []
     [(body/->ScalarLoad (body/value value dtype) buffer
                         (source-coordinates index) valid-mask
                         (body/literal 0 dtype) :cached)
      (body/->ScalarStore allocation
                          [(row-coordinate index) (col-coordinate index)] value nil)
      (body/->Yield [])]
     [] {:unroll false})))

(defn lower
  "Apply one static cooperative register-tiled schedule to verified contraction facts."
  [contract-facts {:keys [tile descriptor operation-id]}]
  (when-not (facts/facts? contract-facts)
    (throw (ex-info "register-tiled scheduling requires verified contraction facts"
                    {:reason :raster/bug :facts contract-facts})))
  (let [{:keys [dtype free-axes contract-axes epilogue out]} contract-facts
        dtype (dtype/canon dtype)
        _ (when-not (dtype/known? dtype)
            (decline! :storage-dtype
                      "register-tiled schedule requires a known storage dtype"
                      {:dtype dtype}))
        ;; Matrix arithmetic is semantic value algebra, unlike the schedule's bounded indices.
        ;; Until an integer accumulation algebra is carried by contraction facts, do not choose
        ;; wrap, trap, or saturation implicitly for this target-specific schedule.
        _ (when (dtype/integral? dtype)
            (decline! :integral-overflow-contract
                      "register-tiled integer contraction requires an explicit accumulation overflow algebra"
                      {:dtype dtype :operation-id operation-id}))
        tile (select-tile tile descriptor dtype)
        {:keys [combine neutral]} (facts/scalar-reduction-view contract-facts)
        {:keys [block-m block-n block-k thread-m thread-n]} tile
        _ (when-not (= [2 1] [(count free-axes) (count contract-axes)])
            (decline! :not-2-free
                      "register-tiled schedule requires exactly two free axes and one reduction axis"
                      {:free-axes free-axes :contract-axes contract-axes}))
        _ (when-not (every? number? (concat (map second free-axes)
                                            (map second contract-axes)))
            (decline! :symbolic-dims
                      "register-tiled schedule requires literal dims"
                      {:free-axes free-axes :contract-axes contract-axes}))
        _ (when-not (additive? combine)
            (decline! :non-plus-combine
                      "register-tiled contraction combine must be +"
                      {:combine combine}))
        _ (when-not (and (number? neutral) (zero? neutral))
            (decline! :non-zero-matrix-init
                      "register-tiled contraction requires a zero reduction identity"
                      {:neutral neutral}))
        layout-verdict (facts/check-layout contract-facts (:dpas facts/leaf-layouts))
        _ (when-not (:ok layout-verdict)
            (decline! :dense-row-major-operands
                      "register-tiled schedule requires dense A(i,k) and B(k,j) operands"
                      layout-verdict))
        {:keys [row col]} (:bindings layout-verdict)
        [[i M] [j N]] free-axes
        [[k K]] contract-axes
        row-shape [M K]
        col-shape [K N]
        out-shape [M N]
        row-layout (layout/row-major row-shape dtype)
        col-layout (layout/row-major col-shape dtype)
        out-layout (layout/row-major out-shape dtype)
        ;; A result transform that reads the destination reads the element this thread stores;
        ;; the destination is then one read-write parameter.
        destination-read? (boolean (some #(= out (:sym %)) (:operands epilogue)))
        base-parameters
        [(body/->KernelParameter row :input dtype row-shape :global row-layout :lhs)
         (body/->KernelParameter col :input dtype col-shape :global col-layout :rhs)
         (body/->KernelParameter out (if destination-read? :inout :output) dtype out-shape
                                 :global out-layout :result)]
        parameters (into (vec base-parameters) (transform-parameters epilogue base-parameters))
        parameter-map (into {} (map (juxt :id identity)) parameters)
        row-allocation 'register-tile-a
        col-allocation 'register-tile-b
        allocations
        [(body/->WorkgroupAllocation row-allocation dtype [block-m block-k]
                                     (layout/row-major [block-m block-k] dtype)
                                     (dtype/bytes-of dtype))
         (body/->WorkgroupAllocation col-allocation dtype [block-k block-n]
                                     (layout/row-major [block-k block-n] dtype)
                                     (dtype/bytes-of dtype))]
        workgroup-by-row (quot block-m thread-m)
        workgroup-by-col (quot block-n thread-n)
        workgroup-width (* workgroup-by-row workgroup-by-col)
        local-col 'register-local-col
        local-row 'register-local-row
        group-col 'register-group-col
        group-row 'register-group-row
        thread-id 'register-thread-id
        block-row 'register-block-row
        block-col 'register-block-col
        indices
        [(body/->IndexBinding local-col :local 0)
         (body/->IndexBinding local-row :local 1)
         (body/->IndexBinding group-col :group 0)
         (body/->IndexBinding group-row :group 1)
         (body/->IndexCompute thread-id (add (mul local-row workgroup-by-col) local-col))
         (body/->IndexCompute block-row (mul group-row block-m))
         (body/->IndexCompute block-col (mul group-col block-n))]
        row-stage-index 'register-a-index
        col-stage-index 'register-b-index
        row-stage-row #(div % block-k)
        row-stage-col #(modulo % block-k)
        col-stage-row #(div % block-n)
        col-stage-col #(modulo % block-n)
        k-block 'register-k-block
        row-valid-mask :register-a-valid
        col-valid-mask :register-b-valid
        masks
        (vec
         (concat
          [(body/->Mask
            row-valid-mask
            [(body/predicate :lt (add block-row (row-stage-row row-stage-index)) M)
             (body/predicate :lt (add k-block (row-stage-col row-stage-index)) K)])
           (body/->Mask
            col-valid-mask
            [(body/predicate :lt (add k-block (col-stage-row col-stage-index)) K)
             (body/predicate :lt (add block-col (col-stage-col col-stage-index)) N)])]
          (for [mm (range thread-m) nn (range thread-n)]
            (body/->Mask
             (keyword (str "register-store-" mm "-" nn))
             [(body/predicate :lt (add block-row (mul local-row thread-m) mm) M)
              (body/predicate :lt (add block-col (mul local-col thread-n) nn) N)]))))
        outer-accumulator-bindings
        (vec (for [mm (range thread-m) nn (range thread-n)]
               (identifier "register-outer-acc" mm nn)))
        inner-accumulator-bindings
        (vec (for [mm (range thread-m) nn (range thread-n)]
               (identifier "register-inner-acc" mm nn)))
        accumulator-results
        (vec (for [mm (range thread-m) nn (range thread-n)]
               (identifier "register-result" mm nn)))
        inner-results
        (vec (for [mm (range thread-m) nn (range thread-n)]
               (identifier "register-inner-result" mm nn)))
        inner-index 'register-k-inner
        inner-operations
        (vec
         (concat
          (mapcat
           (fn [mm]
             (let [loaded (identifier "register-a" mm)]
               [(body/->ScalarLoad
                 (body/value loaded dtype) row-allocation
                 [(add (mul local-row thread-m) mm) inner-index] nil nil :cached)]))
           (range thread-m))
          (mapcat
           (fn [nn]
             (let [loaded (identifier "register-b" nn)]
               [(body/->ScalarLoad
                 (body/value loaded dtype) col-allocation
                 [inner-index (add (mul local-col thread-n) nn)] nil nil :cached)]))
           (range thread-n))
          (mapcat
           (fn [[mm nn]]
             (let [product (identifier "register-product" mm nn)
                   next-accumulator (identifier "register-next" mm nn)
                   accumulator (identifier "register-inner-acc" mm nn)]
               [(body/->ScalarCompute
                 (body/value product dtype)
                 (body/scalar-expression :* dtype
                                         [(identifier "register-a" mm)
                                          (identifier "register-b" nn)]))
                (body/->ScalarCompute
                 (body/value next-accumulator dtype)
                 (body/scalar-expression :+ dtype [accumulator product]))]))
           (for [mm (range thread-m) nn (range thread-n)] [mm nn]))
          [(body/->Yield
            (vec (for [mm (range thread-m) nn (range thread-n)]
                   (identifier "register-next" mm nn))))]))
        inner-loop
        (body/->ForLoop
         (body/value inner-index :int) 0 block-k 1
         (mapv (fn [binding initial]
                 (body/->LoopArg (body/value binding dtype) initial))
               inner-accumulator-bindings outer-accumulator-bindings)
         inner-operations
         (mapv #(body/value % dtype) inner-results)
         {:unroll true})
        stage-row
        (staging-loop
         {:id "register-a" :buffer row :allocation row-allocation :dtype dtype
          :elements (* block-m block-k) :thread-id thread-id
          :workgroup-width workgroup-width
          :row-coordinate row-stage-row :col-coordinate row-stage-col
          :source-coordinates
          (fn [index]
            [(add block-row (row-stage-row index))
             (add k-block (row-stage-col index))])
          :valid-mask row-valid-mask})
        stage-col
        (staging-loop
         {:id "register-b" :buffer col :allocation col-allocation :dtype dtype
          :elements (* block-k block-n) :thread-id thread-id
          :workgroup-width workgroup-width
          :row-coordinate col-stage-row :col-coordinate col-stage-col
          :source-coordinates
          (fn [index]
            [(add k-block (col-stage-row index))
             (add block-col (col-stage-col index))])
          :valid-mask col-valid-mask})
        outer-loop
        (body/->ForLoop
         (body/value k-block :int) 0 K block-k
         (mapv (fn [binding]
                 (body/->LoopArg (body/value binding dtype) (body/literal 0 dtype)))
               outer-accumulator-bindings)
         [stage-row stage-col (barrier) inner-loop (barrier)
          (body/->Yield inner-results)]
         (mapv #(body/value % dtype) accumulator-results)
         {})
        semantic-region (scalar-region-lower/make-region epilogue)
        stores
        (vec
         (mapcat
          (fn [mm]
            (mapcat
             (fn [nn]
               (let [position (+ (* mm thread-n) nn)
                     accumulator (nth accumulator-results position)
                     store-mask (keyword (str "register-store-" mm "-" nn))
                     row-source (list '+ block-row (list '* local-row thread-m) mm)
                     col-source (list '+ block-col (list '* local-col thread-n) nn)
                     coordinates [(contraction-body/lower-index
                                   row-source #{block-row block-col local-row local-col})
                                  (contraction-body/lower-index
                                   col-source #{block-row block-col local-row local-col})]
                     lowered
                     (when semantic-region
                       (scalar-region-lower/lower
                        semantic-region
                        {:accumulator accumulator
                         :accumulator-dtype dtype
                         :store-dtype dtype
                         :parameters parameter-map
                         :id-prefix (str "register-store-" mm "-" nn)
                         :coordinate-lower
                         #(mapv (fn [coordinate]
                                  (contraction-body/lower-index
                                   (walk/postwalk-replace
                                    {i row-source j col-source} coordinate)
                                   #{block-row block-col local-row local-col}))
                                (axis-map/coordinate-exprs %))
                         :predicate store-mask}))]
                 (concat (:operations lowered)
                         [(body/->ScalarStore out coordinates
                                              (or (:result lowered) accumulator)
                                              store-mask)])))
             (range thread-n)))
          (range thread-m)))
        memory-plan (body/workgroup-memory-plan allocations)]
    {:kernel-body
     (body/make
      {:id [:contraction (or operation-id out) :register-tiled]
       :parameters parameters
       :stable-reads (mapv body/stable-read
                           (distinct (remove #{out}
                                             (concat [row col] (map :sym (:operands epilogue))))))
       :allocations allocations
       :indices indices
       :masks masks
       :operations (into [outer-loop] stores)
       :schedule (assoc tile :strategy :register-tiled)
       :launch (launch/spec
                {:workgroup-size [workgroup-by-col workgroup-by-row]
                 :group-count [(launch/ceil-div N block-n)
                               (launch/ceil-div M block-m)]
                 :shared-memory-bytes (:bytes memory-plan)})
       :provenance {:dialect :kernel-body :operation-id operation-id}
       :attributes {:kind :register-tiled-contraction
                    :dims [M N K]
                    :axis-symbols [i j k]
                    :bindings (:bindings layout-verdict)
                    :result-transform epilogue}})
     :bindings (:bindings layout-verdict)
     :dims [M N K]
     :tile tile}))
