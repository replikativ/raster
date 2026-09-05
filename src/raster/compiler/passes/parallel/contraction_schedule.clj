(ns raster.compiler.passes.parallel.contraction-schedule
  "Apply a hardware schedule to verified contraction facts.

  This pass chooses no target syntax.  Its successful result is a KernelBody whose matrix
  instruction, named operand/accumulator layouts, hardware indices, masks, K loop and stores are
  all inspectable compiler values.  A backend may decline an instruction family it cannot lower,
  but it must consume this body rather than reconstructing the schedule from the source form."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.passes.parallel.contraction-body :as contraction-body]
            [raster.compiler.passes.parallel.scalar-region-lower :as scalar-region-lower]))

(defn- decline [reason & [data]]
  (merge {:ok false :reason reason} data))

(defn- additive? [combine]
  (contains? '#{+ clojure.core/+ raster.numeric/+} combine))

(defn- compiler-id? [value]
  (or (symbol? value) (keyword? value)))

(defn- lowered-dpas-instruction?
  [{:keys [family m n k subgroup]}]
  (and (= :dpas family) (= 8 m) (= 16 k)
       (contains? #{8 16} subgroup) (= n subgroup)))

(defn- tile-valid?
  [{:keys [block-m block-n block-k sg-m sg-n matrix num-stages]}]
  (let [{:keys [m n k subgroup]} matrix
        num-stages (or num-stages 3)]
    (and (every? #(and (integer? %) (pos? %))
                 [block-m block-n block-k sg-m sg-n m n k subgroup num-stages])
         (zero? (mod block-m sg-m))
         (zero? (mod block-n sg-n))
         (zero? (mod sg-m m))
         (zero? (mod sg-n n))
         (zero? (mod block-k k)))))

(defn- matrix-parameters
  [row col out M N K dimension-parameters row-layout col-layout out-layout result-dtype epilogue
   buffer-shapes additional-parameters]
  (let [[m-parameter n-parameter k-parameter] dimension-parameters
        base-parameters
        (vec
         (concat
          [(body/->KernelParameter row :input :half (get buffer-shapes row [M K])
                                   :global row-layout :lhs)
           (body/->KernelParameter col :input :half (get buffer-shapes col [K N])
                                   :global col-layout :rhs)
           (body/->KernelParameter out :output result-dtype (get buffer-shapes out [M N])
                                   :global out-layout :result)
           (body/->KernelParameter m-parameter :scalar :int [] nil nil :dimension)
           (body/->KernelParameter n-parameter :scalar :int [] nil nil :dimension)
           (body/->KernelParameter k-parameter :scalar :int [] nil nil :dimension)]
          additional-parameters))
        base-ids (set (map :id base-parameters))]
    (vec
     (concat
      base-parameters
      (for [{:keys [sym dtype map] :or {dtype :float}} (:operands epilogue)
            :when (not (contains? base-ids sym))]
        (let [shape (axis-map/shape map)]
          (body/->KernelParameter sym :input dtype shape :global
                                  (layout/row-major shape dtype) :epilogue)))
      (for [{:keys [sym dtype] :or {dtype :float}} (:scalars epilogue)
            :when (not (contains? base-ids sym))]
        (body/->KernelParameter sym :scalar dtype [] nil nil :epilogue))))))

(defn- fragment-id [prefix a & [b]]
  (keyword (str prefix "-" a (when (some? b) (str "-" b)))))

(defn matrix-body
  "Construct one target-neutral scheduled matrix body from an already chosen canonical layout.

  `dimensions` are semantic tensor extents while `dimension-parameters` are the ordered compiler
  identities bound to the emitted M/N/K ABI.  Keeping both makes static contraction facts and
  symbolic resident graphs share one body without renaming caller values.  The current matrix
  instruction is f16×f16→f32; `result-dtype` controls only the final store representation."
  [{:keys [id row col out dimensions dimension-parameters axis-symbols tile bindings epilogue
           result-dtype provenance additional-parameters additional-indices buffer-shapes
           buffer-views operation-buffers k-range launch-group-count attributes]
    :or {dimension-parameters ['M 'N 'K]
         axis-symbols ['i 'j 'k]
         result-dtype :half
         provenance {}
         additional-parameters []
         additional-indices []
         buffer-shapes {}
         buffer-views []
         operation-buffers {}
         attributes {}}}]
  (let [tile (assoc tile :num-stages (or (:num-stages tile) 3))
        _ (when-not (and (= 3 (count dimension-parameters))
                         (every? compiler-id? dimension-parameters)
                         (= 3 (count (set dimension-parameters))))
            (throw (ex-info "matrix body requires three distinct compiler dimension identities"
                            {:reason :raster/bug
                             :dimension-parameters dimension-parameters})))
        [M N K] dimensions
        [m-parameter n-parameter k-parameter] dimension-parameters
        [k-lower k-upper] (or k-range [0 k-parameter])
        row-buffer (get operation-buffers row row)
        col-buffer (get operation-buffers col col)
        out-buffer (get operation-buffers out out)
        [i j k-sym] axis-symbols
        matrix (:matrix tile)
        desc {:matrix matrix :subgroup-size (:subgroup matrix)}
        acc-layout (layout/derive-layout :mma-acc :float desc)
        ;; Dot operands inherit the ACCUMULATOR distribution; their own dtype only controls the
        ;; packed K width.  Calling derive-layout independently at :half would create a different
        ;; half-typed parent and make the three fragments look layout-incompatible.
        k-width (max 1 (quot 32 (layout/dtype-bits :half)))
        row-layout (layout/dot-operand 0 acc-layout k-width :half)
        col-layout (layout/dot-operand 1 acc-layout k-width :half)
        row-storage-layout (layout/row-major (get buffer-shapes row [M K]) :half)
        col-storage-layout (layout/row-major (get buffer-shapes col [K N]) :half)
        out-layout (layout/row-major (get buffer-shapes out [M N]) result-dtype)
        buffer-layouts {row row-storage-layout col col-storage-layout out out-layout}
        buffer-views (mapv (fn [view]
                             (if (instance? raster.compiler.ir.kernel_body.BufferView view)
                               view
                               (let [parent-layout (get buffer-layouts (:buffer view))]
                                 (body/->BufferView
                                  (:id view) (:buffer view) (:element-offset view) (:shape view)
                                  (or (:layout view)
                                      (layout/row-major (:shape view)
                                                        (:dtype parent-layout)))))))
                           buffer-views)
        {:keys [block-m block-n block-k sg-m sg-n num-stages]} tile
        {matrix-m :m matrix-n :n matrix-k :k subgroup :subgroup} matrix
        subgroup-rows (quot block-m sg-m)
        subgroup-cols (quot block-n sg-n)
        m-fragments (quot sg-m matrix-m)
        n-fragments (quot sg-n matrix-n)
        group-m 'group-m
        group-n 'group-n
        subgroup-id 'subgroup-id
        lane-id 'lane-id
        subgroup-row 'subgroup-row
        subgroup-col 'subgroup-col
        m-base 'm-base
        n-base (fn [nn] (symbol (str "n-base-" nn)))
        add (fn [& xs] (apply body/expression :add xs))
        mul (fn [& xs] (apply body/expression :mul xs))
        floor-div (fn [x y] (body/expression :floor-div x y))
        rem (fn [x y] (body/expression :mod x y))
        indices
        (vec
         (concat
          [(body/->IndexBinding group-n :group 0)
           (body/->IndexBinding group-m :group 1)
           (body/->IndexBinding subgroup-id :subgroup 0)
           (body/->IndexBinding lane-id :lane 0)
           (body/->IndexCompute subgroup-row (floor-div subgroup-id subgroup-cols))
           (body/->IndexCompute subgroup-col (rem subgroup-id subgroup-cols))
           (body/->IndexCompute m-base (add (mul group-m block-m)
                                            (mul subgroup-row sg-m)))]
          (for [nn (range n-fragments)]
            (body/->IndexCompute
             (n-base nn)
             (add (mul group-n block-n) (mul subgroup-col sg-n) (* nn matrix-n))))))
        masks
        (vec
         (concat
          [(body/->Mask :tile-active
                        (cond-> [(body/predicate :lt m-base m-parameter)
                                 (body/predicate :lt (n-base 0) n-parameter)]
                          (not= [0 k-parameter] [k-lower k-upper])
                          (conj (body/predicate :lt k-lower k-upper))))
           (body/->Mask :k-active [(body/predicate :lt 'k-fragment k-upper)])
           (body/->Mask :prefetch-active
                        [(body/predicate :lt
                                         (add 'k-fragment (* num-stages matrix-k)) k-upper)])]
          (for [mm (range m-fragments) nn (range n-fragments)]
            (body/->Mask
             (fragment-id "store" mm nn)
             [(body/predicate :lt (add m-base (* mm matrix-m)) m-parameter)
              (body/predicate :lt (add (n-base nn) lane-id) n-parameter)]))))
        accumulator-ids (vec (for [mm (range m-fragments) nn (range n-fragments)]
                               (fragment-id "acc" mm nn)))
        row-fragment-ids (vec (for [mm (range m-fragments)] (fragment-id "lhs" mm)))
        col-fragment-ids (vec (for [nn (range n-fragments)] (fragment-id "rhs" nn)))
        fragments
        (vec
         (concat
          (for [acc accumulator-ids]
            (body/->Fragment acc :float [matrix-m matrix-n] acc-layout))
          (for [lhs row-fragment-ids]
            (body/->Fragment lhs :half [matrix-m matrix-k] row-layout))
          (for [rhs col-fragment-ids]
            (body/->Fragment rhs :half [matrix-k matrix-n] col-layout))))
        k-fragment 'k-fragment
        one-k-step
        (vec
         (concat
          (for [mm (range m-fragments)]
            (body/->TilePrefetch
             row-buffer [(add m-base (* mm matrix-m))
                         (add k-fragment (* num-stages matrix-k))]
             [matrix-m matrix-k] row-layout :prefetch-active num-stages))
          (for [nn (range n-fragments)]
            (body/->TileLoad (fragment-id "rhs" nn) col-buffer
                             [k-fragment (n-base nn)] :k-active :cached))
          (for [mm (range m-fragments)]
            (body/->TileLoad (fragment-id "lhs" mm) row-buffer
                             [(add m-base (* mm matrix-m)) k-fragment] :k-active :cached))
          (for [mm (range m-fragments) nn (range n-fragments)]
            (body/->MatrixMad (fragment-id "acc" mm nn)
                              (fragment-id "lhs" mm)
                              (fragment-id "rhs" nn)
                              matrix))))
        init-ops (mapv #(body/->FragmentInit % 0.0) accumulator-ids)
        fragment-loop (body/->Loop k-fragment 'k-block
                                   (body/expression :min (add 'k-block block-k) k-upper)
                                   matrix-k one-k-step
                                   {:unroll true :matrix-step matrix-k})
        k-loop (body/->Loop 'k-block k-lower k-upper block-k [fragment-loop]
                            {:unrolled-by (quot block-k matrix-k)
                             :matrix-step matrix-k
                             :pipeline-depth num-stages})
        parameters (matrix-parameters row col out M N K dimension-parameters
                                      row-storage-layout col-storage-layout out-layout
                                      result-dtype epilogue
                                      buffer-shapes additional-parameters)
        semantic-region (scalar-region-lower/make-region epilogue)
        region
        (when semantic-region
          (scalar-region-lower/lower-region
           semantic-region
           {:accumulator (first (:parameters semantic-region))
            :accumulator-dtype :float
            :store-dtype result-dtype
            :indices [i j]
            :parameters (into {} (map (juxt :id identity)) parameters)
            :coordinate-lower
            #(mapv (fn [coordinate]
                     (contraction-body/lower-index
                      coordinate (set (concat axis-symbols dimension-parameters))))
                   (axis-map/coordinate-exprs %))
            :predicate nil}))
        stores (vec
                (for [mm (range m-fragments) nn (range n-fragments)]
                  (body/->TileStore
                   out-buffer (fragment-id "acc" mm nn)
                   [(add m-base (* mm matrix-m)) (n-base nn)]
                   (fragment-id "store" mm nn) region)))
        workgroup-size (* subgroup-rows subgroup-cols subgroup)
        group-count (or launch-group-count
                        [(launch/ceil-div (launch/runtime-value n-parameter) block-n)
                         (launch/ceil-div (launch/runtime-value m-parameter) block-m)])]
    (body/make
     {:id id
      :parameters parameters
      :views (vec buffer-views)
      :stable-reads (mapv body/stable-read
                          (map :id (filter #(= :input (:kind %)) parameters)))
      :indices (into (vec additional-indices) indices)
      :masks masks
      :fragments fragments
      :operations [(body/->Guard :tile-active (vec (concat init-ops [k-loop] stores)))]
      :schedule tile
      :launch (launch/spec
               {:workgroup-size (into [workgroup-size] (repeat (dec (count group-count)) 1))
                :group-count group-count})
      :provenance provenance
      :attributes (merge
                   {:kind :matrix-contraction
                    :instruction-family (:family matrix)
                    :dims [M N K]
                    :dimension-parameters {:m m-parameter :n n-parameter :k k-parameter}
                    :dimension-values {m-parameter M n-parameter N k-parameter K}
                    :axis-symbols [i j k-sym]
                    :bindings bindings
                    :operation-buffers {:row row-buffer :col col-buffer :out out-buffer}
                    :iteration-range {:k [k-lower k-upper]}
                    :boundary-policy {:m :masked :n :masked
                                      :k (if (= [0 k-parameter] [k-lower k-upper])
                                           :exact-fragments :sliced-fragments)}
                    :epilogue epilogue}
                   attributes)})))

(defn plan-matrix-body
  "Apply `tile` to verified contraction facts.

  Returns `{:ok true :body KernelBody :bindings …}` or a structured decline.  The initial family
  is Intel DPAS f16×f16→f32.  Other families become additional target-lowering rows over this same
  body vocabulary; they are not new contraction IRs."
  ([contract-facts desc tile]
   (plan-matrix-body contract-facts desc tile {}))
  ([contract-facts desc tile {:keys [operation-id]}]
   (when-not (facts/facts? contract-facts)
     (throw (ex-info "contraction scheduling requires verified contraction facts"
                     {:reason :raster/bug :facts contract-facts})))
   (let [{:keys [dtype free-axes contract-axes operands epilogue]}
         contract-facts
         {:keys [element combine neutral]} (facts/scalar-reduction-view contract-facts)
         authoritative-desc? (and desc (or (:backend desc) (:execution desc)))
         matrix-capability-unavailable? (and authoritative-desc? (nil? (:matrix desc)))
         raw-tile (when-not matrix-capability-unavailable?
                    (or tile (hardware/derive-gemm-tile (or desc {}))))
         matrix (when raw-tile
                  (assoc (:matrix raw-tile)
                         :family (or (get-in raw-tile [:matrix :family]) :dpas)))
         tile (when raw-tile
                (assoc raw-tile :matrix matrix :num-stages (or (:num-stages raw-tile) 3)))
         layout-verdict (when (and (= 2 (count free-axes)) (= 1 (count contract-axes)))
                          (facts/check-layout contract-facts (:dpas facts/leaf-layouts)))
         bindings (:bindings layout-verdict)
         M (second (first free-axes))
         N (second (second free-axes))
         K (second (first contract-axes))
         matrix-k (:k matrix)]
     (cond
       matrix-capability-unavailable?
       (decline :matrix-capability-unavailable
                {:backend (:backend desc)
                 :supported-subgroup-sizes
                 (hardware/supported-subgroup-sizes desc)})

       (or (not (dtype/known? dtype)) (not= :half (dtype/canon dtype)))
       (decline :dtype-not-dpas {:dtype dtype})

       (not= [2 1] [(count free-axes) (count contract-axes)])
       (decline :not-2-free)

       (not (additive? combine))
       (decline :non-plus-combine {:combine combine})

       (not (constant/zero-value? neutral))
       (decline :non-zero-matrix-init {:init neutral})

       (nil? (facts/body-product-of element (map :sym operands)))
       (decline :body-has-unmodeled-terms)

       (not (:ok layout-verdict))
       (decline :non-canonical-orientation (dissoc layout-verdict :ok :reason))

       (not (every? number? [M N K]))
       (decline :symbolic-dims)

       (not= :dpas (:family matrix))
       (decline :matrix-family-not-lowered {:family (:family matrix)})

       (not (lowered-dpas-instruction? matrix))
       (decline :matrix-instruction-not-lowered {:matrix matrix})

       (not (tile-valid? tile))
       (throw (ex-info "matrix schedule tile is not divisible or contains an invalid extent"
                       {:reason :raster/bug :tile tile}))

       (not (zero? (mod (* (long N) 2) 16)))
       (decline :n-pitch-unaligned {:N N})

       (not (zero? (mod (* (long K) 2) 16)))
       (decline :k-pitch-unaligned {:K K})

      ;; A matrix instruction consumes K values as one indivisible fragment.  The old DPAS gate
      ;; admitted K%8=0 although the emitted instruction is K16, so K=24 could issue an out-of-
      ;; bounds final fragment.  Until a zero-filled fragment load exists, refuse it loudly.
       (not (zero? (mod (long K) (long matrix-k))))
       (decline :partial-matrix-k-fragment {:K K :matrix-k matrix-k})

      ;; Helper strings are target source pasted above a kernel and have no KernelBody meaning.
      ;; Refuse them so callers express the computation in the typed scalar expression instead.
       (seq (:helpers epilogue))
       (decline :opaque-epilogue-helper)

       :else
       {:ok true
        :bindings bindings
        :tile tile
        :body (matrix-body
               {:id [:contraction (or operation-id (:out contract-facts))]
                :row (:row bindings)
                :col (:col bindings)
                :out (:out contract-facts)
                :dimensions [M N K]
                :axis-symbols (vec (concat (map first free-axes)
                                           (map first contract-axes)))
                :tile tile
                :bindings bindings
                :epilogue epilogue
                :provenance {:dialect :segcontract :operation-id operation-id}})}))))

(defn- portable-workgroup-size
  [desc]
  (let [limit (long (or (get-in desc [:execution :max-workgroup-size]) 256))
        limit (max 1 (min 256 limit))]
    (loop [width 1]
      (if (<= (* 2 width) limit) (recur (* 2 width)) width))))

(defn plan-portable-body
  "Apply the portable sequential-segment schedule to a verified contraction SegRed.

   The descriptor affects only legal launch width. Unsupported semantic/indexing cases return a
   structured decline so the established source emitter remains available during migration."
  ([contract-facts segred desc]
   (plan-portable-body contract-facts segred desc {}))
  ([contract-facts segred desc {:keys [array-types scalar-types]
                                :or {array-types {} scalar-types {}}}]
   (try
     (let [workgroup-size (portable-workgroup-size desc)
           lowered (contraction-body/lower
                    contract-facts segred
                    {:workgroup-size workgroup-size
                     :array-types array-types :scalar-types scalar-types})]
       {:ok true
        :body (:kernel-body lowered)
        :workgroup-size workgroup-size
        :arrays (:arrays lowered)
        :scalars (:scalars lowered)
        :segment-count (:segment-count lowered)})
     (catch clojure.lang.ExceptionInfo exception
       (if (contraction-body/declined? exception)
         {:ok false
          :reason (:missing-rule (ex-data exception))
          :detail (ex-data exception)}
         (throw exception))))))
