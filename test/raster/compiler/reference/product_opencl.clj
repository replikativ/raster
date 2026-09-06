(ns raster.compiler.reference.product-opencl
  "Frozen migration oracle for the former handwritten product workgroup tree.
   Test-only: production emits ScheduledKernelBody through the common C-family graph boundary.
   Keep this independent algorithm for differential evidence, not as a fallback or schedule.
   Its 32-bit loop/bound assumptions limit comparisons to the tested small domains."
  (:require [clojure.set]
            [clojure.string :as str]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [raster.compiler.backend.gpu.opencl-codegen :as codegen]
            [raster.compiler.core.dtype :as dt]
            [raster.compiler.core.numeric-constant :as constant]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.segop :as segop]))

(defn- product-neutral-c
  [neutral dtype]
  (let [dtype (dt/canon dtype)
        original neutral
        neutral (constant/literal-or-original neutral)]
    (cond
      (and (= :int dtype) (= neutral Integer/MAX_VALUE)) "INT_MAX"
      (and (= :int dtype) (= neutral Integer/MIN_VALUE)) "INT_MIN"
      (and (= :long dtype) (= neutral Long/MAX_VALUE)) "LONG_MAX"
      (and (= :long dtype) (= neutral Long/MIN_VALUE)) "LONG_MIN"
      (and (number? neutral) (Double/isInfinite (double neutral)))
      (if (pos? (double neutral)) "INFINITY" "-INFINITY")
      (number? neutral) (case dtype
                          :float (str (float neutral) "f")
                          :double (str (double neutral))
                          (str (long neutral)))
      :else (throw (ex-info "product reduction neutral has no OpenCL literal"
                            {:reason :product-reduction-neutral-not-emittable
                             :neutral original :dtype dtype})))))

(defn generate-product-reduction-kernel
  "Lower a typed segmented ProductReduction to one deterministic workgroup tree per segment.

   This is the frozen pre-KernelBody test oracle. Every lane folds a strided subset, then the declared
   closed combine region merges lane products in a fixed tree. Mixed component dtypes share no
   representation assumptions; each gets its own typed local array."
  [segred & {:keys [kernel-name-prefix scalar-types array-types]
             :or {kernel-name-prefix "product_reduce" scalar-types {} array-types {}}}]
  (let [operator (reduction/validate! (:reduction segred))
        schedule (reduction/validate-product-tree! operator (:schedule segred))
        components (:components operator)
        element (reduction/element-region operator)
        combine (reduction/combine-region operator)
        space (:space segred)
        segment-dims (segop/seg-space-segment-dims space)
        reduced-dim (segop/seg-space-reduced-dim space)
        idx (:name reduced-dim)
        bound (:bound reduced-dim)
        block-size (:workgroup-size schedule)
        _ (when-not (and (pos-int? block-size) (zero? (bit-and block-size (dec block-size))))
            (throw (ex-info "product reduction workgroup size must be a positive power of two"
                            {:reason :product-reduction-workgroup :block-size block-size})))
        num-segments (segop/seg-space-num-segments-expr space)
        launch-segments (klaunch/maximum
                         1 (let [bounds (mapv :bound segment-dims)]
                             (case (count bounds)
                               0 1
                               1 (klaunch/runtime-value (first bounds))
                               (apply klaunch/product bounds))))
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        input-params (vec (sort-by name (:inputs segred)))
        output-params (vec (keep :result components))
        output-dtypes (into {} (keep (fn [{:keys [result dtype]}]
                                       (when result [result dtype]))) components)
        bound-syms (reduce clojure.set/union #{}
                           (map (comp util/free-syms :bound) (segop/seg-space-dims space)))
        scalar-params (vec (sort-by name (clojure.set/union (:scalars segred) bound-syms)))
        scalar-types (merge (into {} (map (fn [s] [s :int]) bound-syms)) scalar-types)
        default-dtype (or (:dtype segred) :float)
        default-ctype (dt/ctype :opencl default-dtype)
        scalar-dtype #(ce/scalar-parameter-dtype % scalar-types default-dtype)
        scalar-ctype #(dt/ctype :opencl (scalar-dtype %))
        scalar-var-types (into {} (map (juxt identity scalar-ctype)) scalar-params)
        input-dtype #(get array-types % (get array-types (symbol (name %)) (:dtype segred)))
        input-ctype #(dt/ctype :opencl (input-dtype %))
        component-ctype #(dt/ctype :opencl (:dtype %))
        pointer-params
        (concat
         (map #(str "__global const " (input-ctype %) "* restrict " (ce/c-symbol %)) input-params)
         (map #(str "__global " (dt/ctype :opencl (get output-dtypes %)) "* restrict "
                    (ce/c-symbol %)) output-params))
        scalar-param-decls (map #(str (scalar-ctype %) " " (ce/c-symbol %)) scalar-params)
        all-params (str/join ", " (concat pointer-params scalar-param-decls ["int _n_bound"]))
        array-syms (set (concat input-params output-params))
        int-vars (into #{idx} (concat (map :name segment-dims)
                                      (filter #(contains? #{:int :long} (scalar-dtype %)) scalar-params)))
        emit-index-expr
        (fn [expr]
          (binding [ce/*emit-config* ce/opencl-config
                    ce/*scalar-type* default-ctype
                    ce/*scalar-var-types* scalar-var-types
                    ce/*idx-sym* idx
                    ce/*int-vars* (into ce/*int-vars* int-vars)]
            (ce/emit-expr expr idx array-syms (ce/c-symbol idx))))
        bound-c (emit-index-expr bound)
        emit-result
        (fn [region result dtype substitutions]
          (let [form (list 'let* (:bindings region) result)
                form (util/subst-syms substitutions (ce/normalize-array-prims form))]
            (binding [ce/*emit-config* ce/opencl-config
                      ce/*scalar-type* (dt/ctype :opencl dtype)
                      ce/*scalar-var-types* scalar-var-types
                      ce/*idx-sym* idx
                      ce/*int-vars* (into ce/*int-vars* int-vars)]
              (ce/emit-expr (ce/adapt-casts-for-dtype form dtype) idx array-syms (ce/c-symbol idx)))))
        acc-names (mapv #(str "acc_" %) (range (count components)))
        elem-names (mapv #(str "elem_" %) (range (count components)))
        next-names (mapv #(str "next_" %) (range (count components)))
        shared-names (mapv #(str "shared_" %) (range (count components)))
        element-lines
        (mapv (fn [component result elem-name]
                (str (component-ctype component) " " elem-name " = "
                     (emit-result element result (:dtype component) {}) ";"))
              components (:results element) elem-names)
        combine-lines
        (fn [left-exprs right-exprs destination-names]
          (let [substitutions
                (into {}
                      (mapcat (fn [[[left right] left-expr right-expr]]
                                [[left left-expr] [right right-expr]])
                              (map vector (:parameters combine) left-exprs right-exprs)))]
            (mapv (fn [component result destination]
                    (str (component-ctype component) " " destination " = "
                         (emit-result combine result (:dtype component) substitutions) ";"))
                  components (:results combine) destination-names)))
        segment-decls
        (loop [dims (reverse segment-dims) remaining "segment" lines []]
          (if-let [{:keys [name bound]} (first dims)]
            (let [cname (ce/c-symbol name)
                  cbound (emit-index-expr bound)]
              (recur (next dims) (str "(" remaining " / " cbound ")")
                     (conj lines (str "int " cname " = " remaining " % " cbound ";"))))
            (str/join "\n    " (reverse lines))))
        initial-lines (mapv (fn [component acc-name]
                              (str (component-ctype component) " " acc-name " = "
                                   (product-neutral-c (:neutral component) (:dtype component)) ";"))
                            components acc-names)
        first-combine (combine-lines (mapv symbol acc-names)
                                     (mapv symbol elem-names)
                                     next-names)
        assign-next (mapv #(str % " = " %2 ";") acc-names next-names)
        shared-decls (mapv (fn [component shared]
                             (str "__local " (component-ctype component) " " shared
                                  "[" block-size "];"))
                           components shared-names)
        shared-store (mapv #(str % "[lid] = " %2 ";") shared-names acc-names)
        tree-left (mapv #(list 'clojure.core/aget (symbol %) 'lid) shared-names)
        tree-right (mapv #(list 'clojure.core/aget (symbol %)
                                (list 'clojure.core/+ 'lid 'stride))
                         shared-names)
        tree-next (mapv #(str "tree_next_" %) (range (count components)))
        tree-combine (combine-lines tree-left tree-right tree-next)
        tree-store (mapv #(str % "[lid] = " %2 ";") shared-names tree-next)
        final-stores (mapv (fn [result shared]
                             (str (ce/c-symbol result) "[segment] = " shared "[0];"))
                           output-params
                           (keep (fn [[component shared]] (when (:result component) shared))
                                 (map vector components shared-names)))
        source (str (apply codegen/extension-pragmas (distinct (concat (map :dtype components)
                                                                       (map input-dtype input-params))))
                    "__kernel void " kernel-name "(" all-params ") {\n"
                    "    int segment = get_group_id(0);\n"
                    "    if (segment >= _n_bound) return;\n"
                    "    int lid = get_local_id(0);\n    " segment-decls "\n    "
                    (str/join "\n    " shared-decls) "\n    "
                    (str/join "\n    " initial-lines) "\n"
                    "    for (int " (ce/c-symbol idx) " = lid; " (ce/c-symbol idx) " < "
                    bound-c "; " (ce/c-symbol idx) " += " block-size ") {\n        "
                    (str/join "\n        " element-lines) "\n        "
                    (str/join "\n        " first-combine) "\n        "
                    (str/join "\n        " assign-next) "\n    }\n    "
                    (str/join "\n    " shared-store) "\n"
                    "    barrier(CLK_LOCAL_MEM_FENCE);\n"
                    "    for (int stride = " (/ block-size 2) "; stride > 0; stride >>= 1) {\n"
                    "        if (lid < stride) {\n            "
                    (str/join "\n            " tree-combine) "\n            "
                    (str/join "\n            " tree-store) "\n        }\n"
                    "        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n"
                    "    if (lid == 0) {\n        " (str/join "\n        " final-stores)
                    "\n    }\n}\n")
        abi (kabi/validate!
             (vec (concat
                   (map #(kabi/slot % :input (input-dtype %)
                                    :c-name (ce/c-symbol %) :role :operand)
                        input-params)
                   (map #(kabi/slot % :output (get output-dtypes %)
                                    :c-name (ce/c-symbol %) :role :effect)
                        output-params)
                   (map #(kabi/slot % :scalar (scalar-dtype %)
                                    :c-name (ce/c-symbol %) :role :parameter)
                        scalar-params)
                   [(kabi/slot '_n_bound :scalar :int :c-name "_n_bound" :role :bound)])))
        arguments (vec (concat input-params output-params scalar-params [num-segments]))]
    (kart/make
     {:kernel-name kernel-name :source source :abi abi :arguments arguments
      :launch (klaunch/spec {:workgroup-size [block-size] :group-count [launch-segments]})
      :temporaries [] :effects {:kind :product-reduction}
      :provenance {:dialect :segred :segop-id (:id segred)}
      :attributes {:array-params (vec (concat input-params output-params))
                   :scalar-params scalar-params :written-arrays (set output-params)
                   :component-dtypes (reduction/dtypes operator)
                   :schedule schedule}})))
