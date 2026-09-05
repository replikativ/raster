(ns raster.compiler.passes.parallel.segscan-body
  "Portable KernelBody schedules for certified one-dimensional SegScan graphs.

   The functional scan algebra is already certified before this pass.  This pass makes the
   selected Hillis-Steele schedule explicit: workgroup storage, full-participation barriers,
   block totals, the ordered block-total carry, and inclusive/exclusive result placement.  It
   emits no target syntax and performs no scalar type inference."
  (:require [clojure.set :as set]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.index-expression :as index-expression]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-expression]))

(defn- decline!
  [rule message data]
  (throw (ex-info message (assoc data :reason :segscan-kernel-body-declined
                                 :missing-rule rule :fallback :none))))

(defn declined?
  [exception]
  (= :segscan-kernel-body-declined (:reason (ex-data exception))))

(defn- barrier []
  (body/->WorkgroupBarrier
   :workgroup #{:workgroup} :acquire-release (body/full-participation)))

(defn- identity-value
  [value result-type]
  (let [value (if (and (seq? value) (= 2 (count value)) (number? (second value)))
                (second value)
                value)
        value (if (symbol? value)
                (case (str value)
                  "Double/POSITIVE_INFINITY" Double/POSITIVE_INFINITY
                  "Double/NEGATIVE_INFINITY" Double/NEGATIVE_INFINITY
                  "Float/POSITIVE_INFINITY" Float/POSITIVE_INFINITY
                  "Float/NEGATIVE_INFINITY" Float/NEGATIVE_INFINITY
                  "Integer/MAX_VALUE" Integer/MAX_VALUE
                  "Integer/MIN_VALUE" Integer/MIN_VALUE
                  "Long/MAX_VALUE" Long/MAX_VALUE
                  "Long/MIN_VALUE" Long/MIN_VALUE
                  value)
                value)]
    (when-not (number? value)
      (decline! :literal-identity
                "portable scan requires a scalar numerical identity"
                {:identity value :dtype result-type}))
    (body/literal value result-type)))

(defn- offset-mask [offset]
  (keyword (str "scan-offset-" offset)))

(defn- scan-stages
  [scratch result-type operator identity workgroup-size]
  (mapcat
   (fn [offset]
     (let [self (symbol (str "scan-self-" offset))
           left (symbol (str "scan-left-" offset))
           combined (symbol (str "scan-combined-" offset))]
       [(body/->ScalarLoad (body/value self result-type) scratch ['scan-lane]
                           nil nil :cached)
        (body/->ScalarLoad
         (body/value left result-type) scratch
         [(body/expression :sub 'scan-lane offset)]
         (offset-mask offset) identity :cached)
        (barrier)
        (body/->ScalarCompute
         (body/value combined result-type)
         (body/scalar-expression operator result-type [left self]))
        (body/->ScalarStore scratch ['scan-lane] combined (offset-mask offset))
        (barrier)]))
   (take-while #(< % workgroup-size) (iterate #(* 2 %) 1))))

(defn- parameter
  [buffer access spec role]
  (let [buffer-type (dtype/canon (:dtype spec))
        elements (:elements spec)]
    (when-not (and buffer-type elements)
      (decline! :buffer-contract
                "scan graph buffer requires an explicit dtype and element extent"
                {:buffer buffer :spec spec}))
    (body/->KernelParameter
     buffer (if (= :read access) :input :output) buffer-type [elements] :global
     (layout/row-major [elements] buffer-type) role)))

(defn- common-context
  [operation {:keys [uses buffers temporary-ids output-ids scalar-types scan-algebra
                     scan-mode scan-workgroup]}]
  (let [phase (or (:phase operation) :carry-in)
        result-type (dtype/canon (:dtype scan-algebra))
        workgroup-size (or (get-in operation [:grid :block-size]) scan-workgroup)
        bound (:bound (segop/seg-space-reduced-dim (:space operation)))
        pointer-ids (mapv :buffer uses)
        scalar-ids (vec (sort-by name (:scalars operation)))
        scalar-type (fn [id]
                      (dtype/canon (or (get scalar-types id)
                                       (get scalar-types (symbol (name id)))
                                       result-type)))
        roles (fn [id access]
                (cond
                  (contains? temporary-ids id) :temporary
                  (contains? output-ids id) (if (= :read-write access) :inout :result)
                  :else :operand))
        parameters (vec
                    (concat
                     (map (fn [{:keys [buffer access]}]
                            (parameter buffer access (get buffers buffer)
                                       (roles buffer access)))
                          uses)
                     (map #(body/->KernelParameter % :scalar (scalar-type %) [] nil nil
                                                   :parameter)
                          scalar-ids)
                     [(body/->KernelParameter '_n_bound :scalar :int [] nil nil :bound)]))]
    ;; Scan's combine is semantic value arithmetic.  Its association certification does not prove
    ;; a finite-width overflow algebra, so a schedule may not label it `:no-overflow`.
    (when (dtype/integral? result-type)
      (decline! :integral-overflow-algebra
                "portable integer scan requires an explicit combine overflow contract"
                {:phase phase :mode scan-mode :algebra scan-algebra
                 :result-type result-type}))
    (when-not (and (contains? #{:single :intra-block :block-scan :carry-in} phase)
                   (contains? #{:inclusive :exclusive} scan-mode)
                   (scan/associative-scan? scan-algebra)
                   (= result-type (dtype/canon (:dtype operation)))
                   (integer? workgroup-size) (pos? workgroup-size)
                   (zero? (bit-and workgroup-size (dec workgroup-size))))
      (decline! :schedule-contract
                "portable scan requires a certified scalar scan and a power-of-two workgroup"
                {:phase phase :mode scan-mode :algebra scan-algebra
                 :operation-dtype (:dtype operation) :workgroup-size workgroup-size}))
    {:phase phase :result-type result-type :workgroup-size workgroup-size :bound bound
     :pointer-ids pointer-ids :scalar-ids scalar-ids :scalar-type scalar-type
     :parameters parameters
     :stable-reads (mapv (comp body/stable-read :buffer)
                         (filter #(= :read (:access %)) uses))}))

(defn- element-operations
  [operation context]
  (let [{:keys [result-type pointer-ids scalar-ids scalar-type scan-algebra]} context
        source-index (:name (segop/seg-space-reduced-dim (:space operation)))
        expression (util/subst-syms {source-index 'scan-index} (:element scan-algebra))
        array-types (into {} (map (fn [id] [id (get-in context [:buffers id :dtype])]))
                          pointer-ids)
        scalar-types (into {} (map (juxt identity scalar-type)) scalar-ids)
        index-scope (conj (set scalar-ids) 'scan-index)
        lower-index
        (fn lower-index
          ([form] (lower-index form #{}))
          ([form extra-scope]
           (index-expression/lower form (set/union index-scope extra-scope) decline!)))
        lowerer (scalar-expression/make-lowerer
                 {:array-types array-types :scalar-types scalar-types
                  :arrays (set pointer-ids) :index-scope index-scope
                  :lower-index lower-index :predicate nil
                  :id-prefix "scan-element" :decline! decline!})
        lowered ((:lower lowerer) expression result-type
                                    (assoc scalar-types 'scan-index :int))]
    [(body/->ScalarCompute
      (body/value 'scan-element-active :predicate)
      (body/scalar-expression :lt :predicate ['scan-index '_n_bound]))
     (body/->IfRegion
      'scan-element-active
      (conj (vec (:operations lowered)) (body/->Yield [(:result lowered)]))
      [(body/->Yield [(identity-value (:identity scan-algebra) result-type)])]
      [(body/value 'scan-element result-type)])]))

(defn- scan-node-body
  [operation context]
  (let [{:keys [phase result-type workgroup-size bound parameters stable-reads scan-algebra
                scan-mode temporary-ids output-ids]} context
        operator (intrinsics/canonical (:combine scan-algebra))
        identity (identity-value (:identity scan-algebra) result-type)
        scratch 'scan-workgroup-values
        output (first (filter output-ids (map :buffer (:uses context))))
        totals (first (filter temporary-ids (map :buffer (:uses context))))
        result-index (if (= :exclusive scan-mode)
                       (body/expression :add 'scan-index 1)
                       'scan-index)
        offsets (take-while #(< % workgroup-size) (iterate #(* 2 %) 1))
        masks (vec
               (concat
                [(body/->Mask :scan-active [(body/predicate :lt 'scan-index '_n_bound)])
                 (body/->Mask :scan-lane-zero [(body/predicate :eq 'scan-lane 0)])
                 (body/->Mask :scan-first-lane
                              [(body/predicate :eq 'scan-group 0)
                               (body/predicate :eq 'scan-lane 0)])]
                (map #(body/->Mask (offset-mask %)
                                   [(body/predicate :lte % 'scan-lane)])
                     offsets)))
        indices (vec
                 (concat
                  [(body/->IndexBinding 'scan-group :group 0)
                   (body/->IndexBinding 'scan-lane :local 0)
                   (body/->IndexCompute
                    'scan-index
                    (body/expression :add
                                     (body/expression :mul 'scan-group workgroup-size)
                                     'scan-lane))]
                  (when totals
                    [(body/->IndexCompute
                      'scan-block-base (body/expression :mul 'scan-group workgroup-size))
                     (body/->IndexCompute
                      'scan-valid-count
                      (body/expression :min workgroup-size
                                       (body/expression :sub '_n_bound
                                                        'scan-block-base)))])))
        initial (element-operations operation context)
        stages (scan-stages scratch result-type operator identity workgroup-size)
        tail (vec
              (concat
               [(body/->ScalarLoad (body/value 'scan-result result-type) scratch ['scan-lane]
                                   nil nil :cached)
                (body/->ScalarStore output [result-index] 'scan-result :scan-active)]
               (when totals
                 [(body/->ScalarLoad
                   (body/value 'scan-block-total result-type) scratch
                   [(body/expression :sub 'scan-valid-count 1)]
                   :scan-lane-zero identity :cached)
                  (body/->ScalarStore totals ['scan-group] 'scan-block-total :scan-lane-zero)])
               (when (= :exclusive scan-mode)
                 [(body/->ScalarStore output [0]
                                      identity
                                      :scan-first-lane)])))
        group-count (if (= :single phase) 1 (get-in operation [:grid :num-blocks]))]
    (body/make
     {:id [:segscan (:id operation) phase :portable-workgroup]
      :parameters parameters :stable-reads stable-reads
      :allocations [(body/->WorkgroupAllocation
                     scratch result-type [workgroup-size]
                     (layout/row-major [workgroup-size] result-type)
                     (dtype/bytes-of result-type))]
      :indices indices :masks masks
      :operations (vec (concat initial
                               [(body/->ScalarStore scratch ['scan-lane]
                                                    'scan-element nil)
                                (barrier)]
                               stages tail))
      :schedule {:strategy :workgroup-hillis-steele :association :certified
                 :phase phase :workgroup-size workgroup-size :operator operator}
      :launch (launch/spec
               {:workgroup-size [workgroup-size]
                :group-count [group-count]
                :shared-memory-bytes (* workgroup-size (dtype/bytes-of result-type))})
      :provenance {:dialect :kernel-body :source-dialect :segscan
                   :segop-id (:id operation)}
      :attributes {:kind :portable-segscan :phase phase :scan-mode scan-mode
                   :operator operator :identity (:identity scan-algebra)}})))

(defn- block-scan-body
  [operation context]
  (let [{:keys [result-type workgroup-size bound parameters scan-algebra temporary-ids]} context
        totals (first (filter temporary-ids (map :buffer (:uses context))))
        operator (intrinsics/canonical (:combine scan-algebra))
        identity (identity-value (:identity scan-algebra) result-type)
        scratch 'scan-workgroup-values
        carry 'scan-workgroup-carry
        ;; Do not treat the ABI's `:bound` role as a proof.  The loop owns this clamp, so the
        ;; verifier can replay the non-negative extent fact used by its schedule arithmetic.
        safe-bound 'scan-safe-bound
        offsets (take-while #(< % workgroup-size) (iterate #(* 2 %) 1))
        masks (vec
               (concat
                [(body/->Mask :scan-lane-zero [(body/predicate :eq 'scan-lane 0)])
                 (body/->Mask :scan-loop-active
                              [(body/predicate
                                :lt (body/expression :add 'scan-base 'scan-lane)
                                safe-bound)])]
                (map #(body/->Mask (offset-mask %)
                                   [(body/predicate :lte % 'scan-lane)])
                     offsets)))
        loop-body
        (vec
         (concat
          [(body/->ScalarLoad (body/value 'scan-chunk-value result-type) totals
                              [(body/expression :add 'scan-base 'scan-lane)]
                              :scan-loop-active identity :cached)
           (body/->ScalarStore scratch ['scan-lane] 'scan-chunk-value nil)
           (barrier)]
          (scan-stages scratch result-type operator identity workgroup-size)
          [(body/->ScalarLoad (body/value 'scan-prefix result-type) carry [0]
                              nil nil :cached)
           (body/->ScalarLoad (body/value 'scan-chunk-prefix result-type) scratch
                              ['scan-lane] nil nil :cached)
           (body/->ScalarCompute
            (body/value 'scan-total-prefix result-type)
            (body/scalar-expression operator result-type
                                    ['scan-prefix 'scan-chunk-prefix]))
           (body/->ScalarStore totals [(body/expression :add 'scan-base 'scan-lane)]
                                'scan-total-prefix :scan-loop-active)
           (barrier)
           (body/->ScalarCompute
            (body/value 'scan-chunk-remaining :int)
            (body/scalar-expression :min :int
                                    [(body/literal workgroup-size :int)
                                     (body/scalar-expression :- :int
                                                             [safe-bound 'scan-base]
                                                             {:overflow :no-overflow})]))
           (body/->ScalarLoad
            (body/value 'scan-chunk-last result-type) scratch
            [(body/expression :sub 'scan-chunk-remaining 1)]
            :scan-lane-zero identity :cached)
           (body/->ScalarCompute
            (body/value 'scan-next-carry result-type)
            (body/scalar-expression operator result-type
                                    ['scan-prefix 'scan-chunk-last]))
           (body/->ScalarStore carry [0] 'scan-next-carry :scan-lane-zero)
           (barrier)
           (body/->Yield [])]))]
    (when-not totals
      (decline! :block-total-buffer
                "block scan requires its graph-owned totals buffer"
                {:operation (:id operation) :uses (:uses context)}))
    (body/make
     {:id [:segscan (:id operation) :block-scan :portable-workgroup]
      :parameters parameters
      :allocations [(body/->WorkgroupAllocation
                     scratch result-type [workgroup-size]
                     (layout/row-major [workgroup-size] result-type)
                     (dtype/bytes-of result-type))
                    (body/->WorkgroupAllocation
                     carry result-type [1] (layout/row-major [1] result-type)
                     (dtype/bytes-of result-type))]
      :indices [(body/->IndexBinding 'scan-lane :local 0)]
      :masks masks
      :operations [(body/->ScalarCompute
                    (body/value safe-bound :int)
                    (body/scalar-expression
                     :min :int
                     [(body/scalar-expression :max :int
                                              ['_n_bound (body/literal 0 :int)])
                      (body/literal Integer/MAX_VALUE :int)]))
                   (body/->ScalarStore carry [0] identity :scan-lane-zero)
                   (barrier)
                   (body/->ForLoop
                    (body/value 'scan-base :int) 0 safe-bound workgroup-size []
                    loop-body [] {:association :ordered
                                  :uniform-iter-args #{}})]
      :schedule {:strategy :ordered-workgroup-chunks :association :certified
                 :phase :block-scan :workgroup-size workgroup-size :operator operator}
      :launch (launch/spec
               {:workgroup-size [workgroup-size] :group-count [1]
                :shared-memory-bytes (* (inc workgroup-size)
                                        (dtype/bytes-of result-type))})
      :provenance {:dialect :kernel-body :source-dialect :segscan
                   :segop-id (:id operation)}
      :attributes {:kind :portable-segscan :phase :block-scan
                   :scan-mode (:scan-mode context)
                   :operator operator :identity (:identity scan-algebra)}})))

(defn- carry-body
  [operation context]
  (let [{:keys [result-type workgroup-size scan-workgroup bound parameters stable-reads
                scan-algebra scan-mode temporary-ids output-ids]} context
        totals (first (filter temporary-ids (map :buffer (:uses context))))
        output (first (filter output-ids (map :buffer (:uses context))))
        operator (intrinsics/canonical (:combine scan-algebra))
        identity (identity-value (:identity scan-algebra) result-type)
        result-index (if (= :exclusive scan-mode)
                       (body/expression :add 'scan-index 1)
                       'scan-index)]
    (when-not (and totals output)
      (decline! :carry-buffers
                "scan carry propagation requires totals and result buffers"
                {:operation (:id operation) :uses (:uses context)}))
    (body/make
     {:id [:segscan (:id operation) :carry-in :portable-map]
      :parameters parameters :stable-reads stable-reads
      :indices [(body/->IndexBinding 'scan-group :group 0)
                (body/->IndexBinding 'scan-lane :local 0)
                (body/->IndexCompute
                 'scan-index
                 (body/expression :add
                                  (body/expression :mul 'scan-group workgroup-size)
                                  'scan-lane))
                (body/->IndexCompute
                 'scan-block (body/expression :floor-div 'scan-index scan-workgroup))]
      :masks [(body/->Mask :scan-active [(body/predicate :lt 'scan-index '_n_bound)])
              (body/->Mask :scan-has-carry
                           [(body/predicate :lt 'scan-index '_n_bound)
                            (body/predicate :lte 1 'scan-block)])]
      :operations [(body/->ScalarLoad (body/value 'scan-current result-type) output
                                      [result-index] :scan-active identity :cached)
                   (body/->ScalarLoad
                    (body/value 'scan-carry result-type) totals
                    [(body/expression :sub 'scan-block 1)]
                    :scan-has-carry identity :cached)
                   (body/->ScalarCompute
                    (body/value 'scan-with-carry result-type)
                    (body/scalar-expression operator result-type
                                            ['scan-carry 'scan-current]))
                   (body/->ScalarStore output [result-index]
                                        'scan-with-carry :scan-active)]
      :schedule {:strategy :one-work-item-per-element :association :independent
                 :phase :carry-in :workgroup-size workgroup-size
                 :scan-block-size scan-workgroup :operator operator}
      :launch (launch/spec
               {:workgroup-size [workgroup-size]
                :group-count [(launch/ceil-div bound workgroup-size)]})
      :provenance {:dialect :kernel-body :source-dialect :segscan
                   :segop-id (:id operation)}
      :attributes {:kind :portable-segscan :phase :carry-in :scan-mode scan-mode
                   :operator operator :identity (:identity scan-algebra)}})))

(defn lower
  "Lower one scheduled node of a certified scan graph to a portable KernelBody."
  [operation options]
  (let [context (merge (common-context operation options)
                       (select-keys options [:uses :buffers :temporary-ids :output-ids
                                             :scan-algebra :scan-mode :scan-workgroup]))
        kernel-body (case (:phase context)
                      (:single :intra-block) (scan-node-body operation context)
                      :block-scan (block-scan-body operation context)
                      :carry-in (carry-body operation context))]
    {:kernel-body kernel-body
     :phase (:phase context) :bound (:bound context)
     :pointer-ids (:pointer-ids context) :scalar-ids (:scalar-ids context)}))
