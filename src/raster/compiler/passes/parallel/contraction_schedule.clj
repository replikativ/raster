(ns raster.compiler.passes.parallel.contraction-schedule
  "Apply a hardware schedule to verified contraction facts.

  This pass chooses no target syntax.  Its successful result is a KernelBody whose matrix
  instruction, named operand/accumulator layouts, hardware indices, masks, K loop and stores are
  all inspectable compiler values.  A backend may decline an instruction family it cannot lower,
  but it must consume this body rather than reconstructing the schedule from the source form."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.contraction-facts :as facts]
            [raster.compiler.ir.kernel-body :as body]))

(defn- decline [reason & [data]]
  (merge {:ok false :reason reason} data))

(defn- additive? [combine]
  (contains? '#{+ clojure.core/+ raster.numeric/+} combine))

(defn- zero-init? [init]
  (and (number? init) (zero? init)))

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
  [row col out M N K row-layout col-layout out-layout epilogue]
  (vec
   (concat
    [(body/->KernelParameter row :input :half [M K] :global row-layout :lhs)
     (body/->KernelParameter col :input :half [K N] :global col-layout :rhs)
     (body/->KernelParameter out :output :half [M N] :global out-layout :result)
     (body/->KernelParameter 'M :scalar :int [] nil nil :dimension)
     (body/->KernelParameter 'N :scalar :int [] nil nil :dimension)
     (body/->KernelParameter 'K :scalar :int [] nil nil :dimension)]
    (for [{:keys [sym dtype map] :or {dtype :float}} (:operands epilogue)]
      (let [shape (axis-map/shape map)]
        (body/->KernelParameter sym :input dtype shape :global
                                (layout/row-major shape dtype) :epilogue)))
    (for [{:keys [sym dtype] :or {dtype :float}} (:scalars epilogue)]
      (body/->KernelParameter sym :scalar dtype [] nil nil :epilogue)))))

(defn- scalar-region [epilogue]
  (when epilogue
    (body/->ScalarRegion
     (vec (concat [(:acc epilogue)]
                  (map :sym (:operands epilogue))
                  (map :sym (:scalars epilogue))))
     (:expr epilogue)
     (vec (:operands epilogue))
     (get epilogue :dtype :float))))

(defn- fragment-id [prefix a & [b]]
  (keyword (str prefix "-" a (when (some? b) (str "-" b)))))

(defn- build-matrix-body
  [contract-facts tile bindings epilogue operation-id]
  (let [{:keys [out free-axes contract-axes]} contract-facts
        [[i M] [j N]] free-axes
        [[k-sym K]] contract-axes
        row (:row bindings)
        col (:col bindings)
        matrix (:matrix tile)
        desc {:matrix matrix :subgroup-size (:subgroup matrix)}
        acc-layout (layout/derive-layout :mma-acc :float desc)
        ;; Dot operands inherit the ACCUMULATOR distribution; their own dtype only controls the
        ;; packed K width.  Calling derive-layout independently at :half would create a different
        ;; half-typed parent and make the three fragments look layout-incompatible.
        k-width (max 1 (quot 32 (layout/dtype-bits :half)))
        row-layout (layout/dot-operand 0 acc-layout k-width :half)
        col-layout (layout/dot-operand 1 acc-layout k-width :half)
        out-layout (layout/row-major [M N] :half)
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
                        [(body/predicate :lt m-base 'M)
                         (body/predicate :lt (n-base 0) 'N)])
           (body/->Mask :k-active [(body/predicate :lt 'k-fragment 'K)])
           (body/->Mask :prefetch-active
                        [(body/predicate :lt
                                         (add 'k-fragment (* num-stages matrix-k)) 'K)])]
          (for [mm (range m-fragments) nn (range n-fragments)]
            (body/->Mask
             (fragment-id "store" mm nn)
             [(body/predicate :lt (add m-base (* mm matrix-m)) 'M)
              (body/predicate :lt (add (n-base nn) lane-id) 'N)]))))
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
             row [(add m-base (* mm matrix-m))
                  (add k-fragment (* num-stages matrix-k))]
             [matrix-m matrix-k] row-layout :prefetch-active num-stages))
          (for [nn (range n-fragments)]
            (body/->TileLoad (fragment-id "rhs" nn) col
                             [k-fragment (n-base nn)] :k-active :cached))
          (for [mm (range m-fragments)]
            (body/->TileLoad (fragment-id "lhs" mm) row
                             [(add m-base (* mm matrix-m)) k-fragment] :k-active :cached))
          (for [mm (range m-fragments) nn (range n-fragments)]
            (body/->MatrixMad (fragment-id "acc" mm nn)
                              (fragment-id "lhs" mm)
                              (fragment-id "rhs" nn)
                              matrix))))
        init-ops (mapv #(body/->FragmentInit % 0.0) accumulator-ids)
        fragment-loop (body/->Loop k-fragment 'k-block
                                   (body/expression :min (add 'k-block block-k) 'K)
                                   matrix-k one-k-step
                                   {:unroll true :matrix-step matrix-k})
        k-loop (body/->Loop 'k-block 0 'K block-k [fragment-loop]
                            {:unrolled-by (quot block-k matrix-k)
                             :matrix-step matrix-k
                             :pipeline-depth num-stages})
        region (scalar-region epilogue)
        stores (vec
                (for [mm (range m-fragments) nn (range n-fragments)]
                  (body/->TileStore
                   out (fragment-id "acc" mm nn)
                   [(add m-base (* mm matrix-m)) (n-base nn)]
                   (fragment-id "store" mm nn) region)))
        workgroup-size (* subgroup-rows subgroup-cols subgroup)]
    (body/make
     {:id [:contraction (or operation-id out)]
      :parameters (matrix-parameters row col out M N K row-layout col-layout out-layout epilogue)
      :indices indices
      :masks masks
      :fragments fragments
      :operations [(body/->Guard :tile-active (vec (concat init-ops [k-loop] stores)))]
      :schedule tile
      :launch {:workgroup-size [workgroup-size 1]
               :group-count [(body/expression :ceil-div 'N block-n)
                             (body/expression :ceil-div 'M block-m)]}
      :provenance {:dialect :segcontract :operation-id operation-id}
      :attributes {:kind :matrix-contraction
                   :instruction-family (:family matrix)
                   :dims [M N K]
                   :axis-symbols [i j k-sym]
                   :bindings bindings
                   :boundary-policy {:m :masked :n :masked :k :exact-fragments}
                   :epilogue epilogue}})))

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
   (let [{:keys [dtype free-axes contract-axes body combine init operands epilogue]}
         contract-facts
         raw-tile (or tile (hardware/derive-gemm-tile (or desc {})))
         matrix (assoc (:matrix raw-tile) :family (or (get-in raw-tile [:matrix :family]) :dpas))
         tile (assoc raw-tile :matrix matrix :num-stages (or (:num-stages raw-tile) 3))
         layout-verdict (when (and (= 2 (count free-axes)) (= 1 (count contract-axes)))
                          (facts/check-layout contract-facts (:dpas facts/leaf-layouts)))
         bindings (:bindings layout-verdict)
         M (second (first free-axes))
         N (second (second free-axes))
         K (second (first contract-axes))
         matrix-k (:k matrix)]
     (cond
       (or (not (dtype/known? dtype)) (not= :half (dtype/canon dtype)))
       (decline :dtype-not-dpas {:dtype dtype})

       (not= [2 1] [(count free-axes) (count contract-axes)])
       (decline :not-2-free)

       (not (additive? combine))
       (decline :non-plus-combine {:combine combine})

       (not (zero-init? init))
       (decline :non-zero-matrix-init {:init init})

       (nil? (facts/body-product-of body (map :sym operands)))
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

      ;; Helpers are target source pasted above the kernel.  Keeping them out is what makes the
      ;; KernelBody store region typed rather than an opaque C escape hatch.  Existing emission
      ;; remains available while helper expressions are promoted into the scalar expression IR.
       (seq (:helpers epilogue))
       (decline :opaque-epilogue-helper)

       :else
       {:ok true
        :bindings bindings
        :tile tile
        :body (build-matrix-body contract-facts tile bindings epilogue operation-id)}))))
