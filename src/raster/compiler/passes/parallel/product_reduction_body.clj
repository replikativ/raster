(ns raster.compiler.passes.parallel.product-reduction-body
  "Candidate KernelBody refinement of row-segmented product reductions.

   This applies the existing lane-strided/fixed-tree schedule, not a new reduction algebra.
   It is not yet selected by production routing. Storage extents and region binding types are
   explicit caller evidence; unsupported source/index contracts decline without a fallback."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.index-expression :as index]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar]))

(defn- decline! [rule message data]
  (throw (ex-info message (assoc data :reason :product-kernel-body-declined
                                 :missing-rule rule :fallback :none))))

(defn- region-binding-types
  "Read retained TypedSOAC local tags. Explicit candidate facts may fill omissions, not override."
  [region supplied]
  (reduce (fn [facts id]
            (if-let [retained (some-> (types/sym-type-tag id) dtype/dtype-for-scalar-tag)]
              (do
                (when (and (get supplied id) (not= retained (get supplied id)))
                  (decline! :binding-dtype-conflict
                            "candidate binding dtype disagrees with its retained source type"
                            {:binding id :retained retained :supplied (get supplied id)}))
                (assoc facts id retained))
              facts))
          (or supplied {}) (take-nth 2 (:bindings region))))

(defn lower
  "Lower one row-segmented SegRed with retained input shapes and scalar/region binding dtypes.
   Address arithmetic must already carry the long-width facts required by lower-typed."
  [segred {:keys [array-types array-shapes scalar-types
                 element-binding-types combine-binding-types]}]
  (when-not (instance? raster.compiler.ir.segop.SegRed segred)
    (decline! :source "product body requires a retained SegRed" {:source segred}))
  (let [operator (reduction/validate! (:reduction segred))
        source-schedule (reduction/validate-product-tree! operator (:schedule segred))
        segments (segop/seg-space-segment-dims (:space segred))
        _ (when-not (= 1 (count segments))
            (decline! :segment-rank "initial product body requires one row segment axis"
                      {:segments segments}))
        {row :name rows :bound} (first segments)
        {column :name width :bound} (segop/seg-space-reduced-dim (:space segred))
        components (:components operator)
        element-binding-types (region-binding-types (reduction/element-region operator)
                                                     element-binding-types)
        combine-binding-types (region-binding-types (reduction/combine-region operator)
                                                     combine-binding-types)
        types (mapv :dtype components)
        wg (:workgroup-size source-schedule)
        inputs (vec (sort-by name (:inputs segred)))
        outputs (vec (keep :result components))
        scalars (vec (sort-by name (set/union (set (:scalars segred))
                                             (util/free-syms rows) (util/free-syms width))))
        _ (doseq [id scalars]
            (when-not (get scalar-types id)
              (decline! :scalar-dtype "product scalar requires retained dtype" {:scalar id})))
        _ (doseq [bound [rows width]]
            (when-not (and (or (integer? bound) (symbol? bound))
                           (= :int (launch/typed-expression-dtype bound scalar-types))
                           (or (symbol? bound) (not (neg? bound))))
              (decline! :bound-dtype
                        "initial product body requires nonnegative int32 row/column bounds"
                        {:bound bound})))
        _ (doseq [id inputs]
            (when-not (and (get array-types id) (seq (get array-shapes id)))
              (decline! :input-storage "product input requires retained dtype and extent"
                        {:input id})))
        parameters
        (vec (concat
              (map (fn [id]
                     (body/->KernelParameter id :input (get array-types id)
                                              (get array-shapes id) :global
                                              (layout/row-major (get array-shapes id)
                                                                (get array-types id)) :operand))
                   inputs)
              (keep (fn [{:keys [result dtype]}]
                      (when result
                        (body/->KernelParameter result :output dtype ['_n_bound] :global
                                                 (layout/row-major ['_n_bound] dtype) :effect)))
                    components)
              (map #(body/->KernelParameter % :scalar (get scalar-types %) [] nil nil :parameter)
                   scalars)
              [(body/->KernelParameter '_n_bound :scalar :int [] nil nil :bound)]))
        index-types (assoc scalar-types row :int column :long)
        lower-index (fn [expression locals]
                      (index/lower-typed expression (set/union (set (keys index-types)) locals)
                                         index-types :long decline!))
        lowerer (scalar/make-lowerer
                 {:arrays (set inputs) :array-types array-types :scalar-types scalar-types
                  :lower-index lower-index :id-prefix "product" :decline! decline!})
        neutrals (mapv (fn [{:keys [neutral dtype]}]
                         (if-let [evidence (constant/value neutral)]
                           (body/literal (:value evidence) dtype)
                           (decline! :neutral "product neutral requires checked literal evidence"
                                     {:neutral neutral :dtype dtype}))) components)
        ids (fn [prefix] (mapv #(symbol (str prefix "-" %)) (range (count components))))
        carries (ids "product-carry")
        lane-results (ids "product-lane-result")
        scratches (ids "product-scratch")
        combine-region (reduction/combine-region operator)
        combine
        (fn [left right]
          (let [replacements (into {} (mapcat (fn [[l r] lv rv] [[l lv] [r rv]])
                                              (:parameters combine-region) left right))
                region (-> combine-region
                           (update :bindings #(util/subst-syms replacements %))
                           (update :results #(util/subst-syms replacements %)))
                env (into {} (concat (map vector left types) (map vector right types)))]
            ((:lower-region lowerer) region types combine-binding-types env)))
        element ((:lower-region lowerer) (reduction/element-region operator) types
                 element-binding-types index-types)
        lane-update (combine carries (:results element))
        barrier #(body/->WorkgroupBarrier :workgroup #{:workgroup} :acquire-release
                                          (body/full-participation))
        strides (vec (take-while pos? (iterate #(quot % 2) (quot wg 2))))
        mask-id #(keyword (str "product-tree-" %))
        tree
        (mapcat
         (fn [stride]
           (let [left (ids (str "product-left-" stride))
                 right (ids (str "product-right-" stride))
                 merged (combine left right)]
             [(body/->ScalarCompute (body/value (mask-id stride) :predicate)
                                    (body/scalar-expression :lt :predicate
                                                            ['product-lane (body/literal stride :int)]))
              (body/->IfRegion
               (mask-id stride)
               (vec (concat
                     (mapcat (fn [l r scratch type]
                               [(body/->ScalarLoad (body/value l type) scratch ['product-lane]
                                                   nil nil :cached)
                                (body/->ScalarLoad (body/value r type) scratch
                                                    [(body/expression :add 'product-lane stride)]
                                                    nil nil :cached)]) left right scratches types)
                     (:operations merged)
                     ;; No store precedes any component's evaluation: coupled updates see the
                     ;; same old tuple on both sides of the combine.
                     (map #(body/->ScalarStore %1 ['product-lane] %2 nil)
                          scratches (:results merged))
                     [(body/->Yield [])]))
               [(body/->Yield [])] [])
              (barrier)])) strides)
        kernel-body
        (body/make
         {:id [:product-reduction (:id segred)] :parameters parameters
          :stable-reads (mapv body/stable-read inputs)
          :allocations (mapv #(body/->WorkgroupAllocation %1 %2 [wg]
                                                         (layout/row-major [wg] %2)
                                                         (dtype/bytes-of %2)) scratches types)
          :indices [(body/->IndexBinding row :group 0)
                    (body/->IndexBinding 'product-lane :local 0)]
          :operations
          (vec (concat
                [(body/->ForLoop
                  (body/value column :long) (body/index-cast 'product-lane :long :exact)
                  (lower-index width #{}) wg
                  (mapv #(body/->LoopArg (body/value %1 %2) %3) carries types neutrals)
                  (vec (concat (:operations element) (:operations lane-update)
                               [(body/->Yield (:results lane-update))]))
                  (mapv body/value lane-results types) {})]
                (map #(body/->ScalarStore %1 ['product-lane] %2 nil) scratches lane-results)
                [(barrier)] tree
                [(body/->ScalarCompute (body/value :product-writer :predicate)
                                       (body/scalar-expression :eq :predicate
                                                               ['product-lane (body/literal 0 :int)]))
                 (body/->IfRegion
                  :product-writer
                  (conj (vec (mapcat
                        (fn [{:keys [result dtype]} scratch value]
                          (when result
                            [(body/->ScalarLoad (body/value value dtype) scratch [0] nil nil :cached)
                             (body/->ScalarStore result [row] value nil)]))
                        components scratches (ids "product-final"))) (body/->Yield []))
                  [(body/->Yield [])] [])]))
          :schedule {:strategy :product-workgroup-tree :workgroup-size wg}
          :launch (launch/spec {:workgroup-size [wg] :group-count [(launch/runtime-value '_n_bound)]
                                :shared-memory-bytes (* wg (reduce + (map dtype/bytes-of types)))})
          :provenance {:dialect :kernel-body :source-dialect :segred :segop-id (:id segred)}
          :attributes {:kind :product-reduction :component-dtypes types}})]
    {:kernel-body kernel-body :rows rows :inputs inputs :outputs outputs :scalars scalars}))

(defn schedule [segred options]
  (let [{:keys [kernel-body rows]} (lower segred options)
        arguments (mapv #(if (= '_n_bound (:id %)) rows (:id %)) (:parameters kernel-body))]
    (scheduled/make
     {:source segred :body kernel-body :arguments arguments
      :scalar-bindings (scheduled/derive-scalar-bindings kernel-body arguments)
      :effects {:kind :product-reduction :uses (scheduled/derive-uses kernel-body arguments)}
      :legality {:kind :product-workgroup-tree :algebra (:algebra (:reduction segred))
                 :source-schedule (:schedule segred) :aliasing :no-write-alias}
      :numerics {:mode :reassociated :policy :declared-product-tree
                 :accumulators (mapv (fn [{:keys [id dtype]}]
                                       {:value id :dtype dtype
                                        :rounding (if (dtype/fp-dtype? dtype) :nearest-even :exact)
                                        :policy :declared-product-tree})
                                     (:components (:reduction segred)))}
      :provenance {:dialect :kernel-body :source-dialect :segred :segop-id (:id segred)}
      :attributes {:candidate-only true :source-storage-certified? false}})))
