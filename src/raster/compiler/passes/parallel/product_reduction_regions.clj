(ns raster.compiler.passes.parallel.product-reduction-regions
  "Shared typed scalar lowering for product scheduling and source-derived read requirements.
   No physical shape is invented to lower a region; storage admission is a separate obligation."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.extent-expression :as extent-expression]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.scalar-range :as scalar-range]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.index-expression :as index]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar]))

(defn- ordered-axis-selections
  "All non-empty ordered subsets of a small semantic axis set. The caller bounds axis count."
  [axes]
  (letfn [(walk [prefix remaining]
            (concat (when (seq prefix) [prefix])
                    (mapcat (fn [axis]
                              (walk (conj prefix axis)
                                    (vec (remove #(= axis %) remaining))))
                            remaining)))]
    (walk [] (vec axes))))

(defn- matching-indexing-map
  [axes coordinate index-types]
  (some (fn [selection]
          (let [candidate (axis-map/of-groups [selection])]
            (when (axis-map/bounded-typed-index-matches?
                   candidate coordinate index-types)
              [candidate (apply launch/product (map second selection))])))
        (ordered-axis-selections axes)))

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
        _ (when (empty? segments)
            (decline! :segment-rank "product region requires at least one segment axis"
                      {:segments segments}))
        {column :name width :bound} (segop/seg-space-reduced-dim (:space segred))
        types (mapv :dtype (:components operator))
        index-types (into (assoc scalar-types column :long)
                          (map (fn [{:keys [name]}]
                                 [name (if (= 1 (count segments)) :int :long)]))
                          segments)
        width-type (launch/typed-expression-dtype width scalar-types)
        width-range (if (integer? width)
                      (scalar-range/literal width width-type)
                      (scalar-range/for-dtype width-type))
        ;; The element executes only at entries of `column = product-lane; column < width;
        ;; column += workgroup-size`. This owner proof is intentionally absent for a full-width
        ;; long bound: the exiting increment may wrap and a checked `(int column)` may really trap.
        column-range (scalar-range/exclusive-positive-loop-entry-range
                      (scalar-range/for-dtype :int) width-range
                      (get-in segred [:schedule :workgroup-size]) :long)
        lower-index (fn [expression locals]
                      (index/lower-typed expression (set/union (set (keys index-types)) locals)
                                         index-types :long decline!))
        lowerer (scalar/make-lowerer
                 {:arrays (set (:inputs segred)) :array-types array-types :scalar-types index-types
                  :scalar-ranges (cond-> {} column-range (assoc column column-range))
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
     :axes (conj (mapv (juxt :name :bound) segments) [column width])}))

(defn dense-read-requirements
  "Derive exact flat capacities from verified operand indexing maps, or decline.

   A load may use any permutation/subset of the semantic segment and reduction axes, which covers
   dense tensors, transposes and broadcasts. Its actual coordinate must equal the selected
   AxisMap and that map's volume must equal retained physical storage. Gather, computed SSA
   coordinates and combine-time reads are not silently admitted. This returns required capacities,
   not an independent execution certificate; the segment guard and lane loop establish the active
   domains when the body is scheduled."
  [segred options decline!]
  (let [{:keys [element combine index-types axes]} (lower segred options decline!)
        _ (when (> (count axes) 6)
            (decline! :indexing-map-rank
                      "product indexing-map search is bounded to six semantic axes"
                      {:axes axes}))
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
        contracts
        (mapv (fn [{:keys [buffer coordinates]}]
                (let [[indexing-map extent]
                      (when (= 1 (count coordinates))
                        (matching-indexing-map axes (first coordinates) index-types))]
                  (when-not indexing-map
                    (decline! :dense-read-index
                              "product input has no verified permutation/broadcast indexing map"
                              {:input buffer :coordinates coordinates :axes axes}))
                  [buffer extent]))
              element-loads)]
    (reduce (fn [requirements [buffer extent]]
              (if-let [prior (get requirements buffer)]
                (if (extent-expression/equivalent? prior extent)
                  requirements
                  (decline! :input-indexing-map
                            "one product input is read through incompatible physical maps"
                            {:input buffer :left prior :right extent}))
                (assoc requirements buffer extent)))
            {} contracts)))
