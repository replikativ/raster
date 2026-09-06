(ns raster.compiler.passes.parallel.product-reduction-regions
  "Shared typed scalar lowering for product scheduling and source-derived read requirements.
   No physical shape is invented to lower a region; storage admission is a separate obligation."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.index-expression :as index]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar]))

(defn- binding-types [region supplied decline!]
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
  "Lower element once and provide a shared combine lowerer with deterministic fresh SSA IDs.
   The caller supplies its decline mechanism and retained array/scalar/local dtype facts."
  [segred {:keys [array-types scalar-types element-binding-types combine-binding-types]} decline!]
  (let [operator (reduction/validate! (:reduction segred))
        segments (segop/seg-space-segment-dims (:space segred))
        _ (when-not (= 1 (count segments))
            (decline! :segment-rank "product region requires one row axis" {:segments segments}))
        {row :name rows :bound} (first segments)
        {column :name width :bound} (segop/seg-space-reduced-dim (:space segred))
        types (mapv :dtype (:components operator))
        index-types (assoc scalar-types row :int column :long)
        lower-index (fn [expression locals]
                      (index/lower-typed expression (set/union (set (keys index-types)) locals)
                                         index-types :long decline!))
        lowerer (scalar/make-lowerer
                 {:arrays (set (:inputs segred)) :array-types array-types :scalar-types scalar-types
                  :lower-index lower-index :id-prefix "product" :decline! decline!})
        element-region (reduction/element-region operator)
        combine-region (reduction/combine-region operator)
        element-types (binding-types element-region element-binding-types decline!)
        combine-types (binding-types combine-region combine-binding-types decline!)
        combine (fn [left right]
                  (let [replacements (into {} (mapcat (fn [[l r] lv rv] [[l lv] [r rv]])
                                                      (:parameters combine-region) left right))
                        region (-> combine-region
                                   (update :bindings #(util/subst-syms replacements %))
                                   (update :results #(util/subst-syms replacements %)))
                        env (into {} (concat (map vector left types) (map vector right types)))]
                    ((:lower-region lowerer) region types combine-types env)))]
    {:element ((:lower-region lowerer) element-region types element-types index-types)
     :combine combine :lower-index lower-index :index-types index-types
     :axes [[row rows] [column width]]}))

(defn dense-read-requirements
  "Derive minimum flat capacities from the actual typed element loads, or decline.

   Initially every input load must implement the full row/column AxisMap. Gather, narrower
   broadcasts, computed SSA coordinates and combine-time reads are not silently called dense.
   This returns required capacities, not actual allocation sizes or an execution certificate.
   The row guard and column loop must establish active domains when the body is scheduled."
  [segred options decline!]
  (let [{:keys [element combine index-types axes]} (lower segred options decline!)
        layout (axis-map/of-axes axes)
        loads (fn [operations]
                (filter #(= "raster.compiler.ir.kernel_body.ScalarLoad" (some-> % class .getName))
                        (tree-seq coll? seq operations)))
        element-loads (loads (:operations element))
        ;; The tree combine also executes when width=0, so element-domain premises do not
        ;; justify any of its external reads. Inspect its lowered operations independently.
        arity (count (get-in segred [:reduction :components]))
        combine-region (combine (mapv #(symbol (str "read-left-" %)) (range arity))
                                (mapv #(symbol (str "read-right-" %)) (range arity)))
        _ (when (seq (loads (:operations combine-region)))
            (decline! :combine-read "dense element bounds do not justify combine reads" {}))
        _ (doseq [{:keys [buffer coordinates]} element-loads]
            (when-not (and (= 1 (count coordinates))
                           (axis-map/bounded-typed-index-matches?
                            layout (first coordinates) index-types))
              (decline! :dense-read-index
                        "product input does not have a bounded typed dense index"
                        {:input buffer :coordinates coordinates})))
        extent (apply launch/product (map second axes))]
    (into {} (map (fn [{:keys [buffer]}] [buffer extent])) element-loads)))
