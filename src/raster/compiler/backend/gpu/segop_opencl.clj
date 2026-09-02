(ns raster.compiler.backend.gpu.segop-opencl
  "C-family kernel generation from scheduled SegOp IR.

   Portable schedules lower through typed KernelBody and emit OpenCL, CUDA, or HIP source. The
   remaining specialized OpenCL leaves use the same pre-computed inputs/outputs/scalars from
   SegOp lowering instead of re-analyzing par forms.

   This is the GPU counterpart to segop_simd.clj — both consume the
   same SegOp IR but produce different target code."
  (:require [raster.compiler.backend.gpu.kernel-body-opencl :as kernel-body-opencl]
            [raster.compiler.backend.gpu.kernel-body-c-dialect :as kernel-body-c-dialect]
            [raster.compiler.backend.gpu.opencl-codegen :as codegen]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.core.dtype :as dt]
            [raster.compiler.core.hardware :as hw]
            [raster.compiler.ir.axis-map :as am]
            [raster.compiler.ir.contract-stages :as cstage]
            [raster.compiler.ir.contraction-facts :as cf]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-body :as kbody]
            [raster.compiler.ir.kernel-body-abi :as body-abi]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.passes.parallel.register-tiled-body :as register-tiled-body]
            [raster.compiler.passes.parallel.contraction-schedule :as contraction-schedule]
            [raster.compiler.passes.parallel.segred-body :as segred-body]
            [raster.compiler.passes.parallel.segfoldmap-body :as segfoldmap-body]
            [raster.compiler.passes.parallel.segmap-body :as segmap-body]
            [raster.compiler.passes.parallel.segscan-body :as segscan-body]
            [raster.compiler.passes.parallel.segstencil-body :as segstencil-body]
            [clojure.walk :as walk]
            [clojure.set]
            [raster.compiler.ir.segop :as segop]
            [clojure.string :as str]))

;; ================================================================
;; SegOp field accessors
;; ================================================================

;; Innermost (reduced/mapped) dim — `first` for the 1-D case (unchanged).
;; The N-D segmented GPU kernel (generate-segmented-reduce-kernel) iterates outer segment dims.
(defn- seg-idx [segop] (:name (segop/seg-space-reduced-dim (:space segop))))
(defn- seg-bound [segop] (:bound (segop/seg-space-reduced-dim (:space segop))))

(defn- fresh-c-local
  "Choose a deterministic kernel-local C identifier that cannot shadow an ABI parameter."
  [base symbols]
  (let [occupied (set (map ce/c-symbol symbols))]
    (first (remove occupied
                   (cons base (map #(str base "_" %) (range)))))))

;; ================================================================
;; SegMap → OpenCL kernel
;; ================================================================

(defn generate-explicit-segmap-kernel
  "Emit an explicit-store SegMap without reconstructing a source `map-void!` form.

   Unique-destination scatter schedules use this shape: the scalar region owns its indexed aset
   statements, while SegMap owns the iteration geometry and the typed input/output boundary."
  [segmap & {:keys [dtype kernel-name-prefix scalar-types array-types]
             :or {dtype :float kernel-name-prefix "segmap_effect"
                  scalar-types {} array-types {}}}]
  (let [idx (seg-idx segmap)
        bound (seg-bound segmap)
        body (ce/normalize-array-prims (:lambda segmap))
        inputs (set (:inputs segmap))
        outputs (set (:outputs segmap))
        pointers (vec (sort-by name (clojure.set/union inputs outputs)))
        scalars (vec (sort-by name (:scalars segmap)))
        default-dtype (or (:dtype segmap) dtype)
        default-ctype (dt/ctype :opencl default-dtype)
        meta-types (ce/collect-array-types-from-meta body)
        array-types (merge meta-types array-types)
        pointer-dtype (fn [symbol]
                        (or (get array-types symbol)
                            (get array-types (clojure.core/symbol (name symbol)))
                            default-dtype))
        pointer-ctype #(dt/ctype :opencl (pointer-dtype %))
        written? #(contains? outputs %)
        read? #(contains? inputs %)
        pointer-kind (fn [symbol]
                       (cond
                         (and (read? symbol) (written? symbol)) :inout
                         (written? symbol) :output
                         :else :input))
        scalar-dtype #(ce/scalar-parameter-dtype % scalar-types default-dtype)
        scalar-ctype #(dt/ctype :opencl (scalar-dtype %))
        scalar-var-types (into {} (map (juxt identity scalar-ctype)) scalars)
        pointer-params
        (str/join ", "
                  (map (fn [symbol]
                         (str "__global "
                              (when-not (written? symbol) "const ")
                              (pointer-ctype symbol) "* "
                              (when-not (written? symbol) "restrict ")
                              (ce/c-symbol symbol)))
                       pointers))
        scalar-params (str/join ", "
                                (map #(str (scalar-ctype %) " " (ce/c-symbol %)) scalars))
        parameter-list (str/join ", "
                                 (remove empty? [pointer-params scalar-params "int _n_bound"]))
        local-index (fresh-c-local "idx" (concat pointers scalars ['_n_bound]))
        array-symbols (set (map #(clojure.core/symbol (name %)) pointers))
        int-scalars (into #{idx} (filter #(= "int" (scalar-ctype %)) scalars))
        adapted-body (ce/adapt-casts-for-dtype body default-dtype)
        body-source (binding [ce/*emit-config* ce/opencl-config
                              ce/*scalar-type* default-ctype
                              ce/*scalar-var-types* scalar-var-types
                              ce/*array-dtypes* array-types
                              ce/*idx-sym* idx
                              ce/*int-vars* (into ce/*int-vars* int-scalars)]
                      (ce/emit-stmt adapted-body idx array-symbols local-index))
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        abi (kabi/validate!
             (vec (concat
                   (map (fn [symbol]
                          (kabi/slot symbol (pointer-kind symbol) (pointer-dtype symbol)
                                     :c-name (ce/c-symbol symbol)
                                     :role (if (written? symbol) :effect :operand)))
                        pointers)
                   (map (fn [symbol]
                          (kabi/slot symbol :scalar (scalar-dtype symbol)
                                     :c-name (ce/c-symbol symbol) :role :parameter))
                        scalars)
                   [(kabi/slot '_n_bound :scalar :int :role :bound)])))
        source (str (apply codegen/extension-pragmas
                           default-dtype (map pointer-dtype pointers))
                    (when (and (= :reduce (:write-conflict segmap))
                               (= :float default-dtype))
                      "#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable\n")
                    (ce/helper-sources body-source)
                    "__kernel void " kernel-name "(" parameter-list ") {\n"
                    "    for (int " local-index " = get_global_id(0); " local-index
                    " < _n_bound; " local-index " += get_global_size(0)) {\n"
                    "        " body-source "\n"
                    "    }\n}\n")]
    (kabi/validate-source-signature! kernel-name source abi)
    (kart/make
     {:kernel-name kernel-name
      :source source
      :abi abi
      :arguments (vec (concat pointers scalars [bound]))
      :launch (klaunch/spec {:workgroup-size [256]
                             :group-count [(klaunch/ceil-div bound 256)]})
      :temporaries []
      :effects {:kind (if (= :reduce (:write-conflict segmap))
                        :reducing-scatter :side-effect-map)
                :write-conflict (or (:write-conflict segmap) :unique)}
      :provenance {:dialect :segmap :segop-id (:id segmap)}
      :attributes {:array-params pointers
                   :scalar-params scalars
                   :written-arrays (vec (sort-by name outputs))
                   :array-types array-types
                   :dtype default-dtype
                   :conflict-contract (:conflict-contract segmap)
                   :explicit-stores true}})))

(defn generate-segmap-kernel
  "Generate an OpenCL C kernel from a SegMap record.

   Mirrors the legacy par-map-void generator's array handling so a fused pure
   par/map (possibly composed of several maps the SOAC fuser collapsed) emits
   correct C: (1) normalize the devirtualized array prims (.invk aget_m_T-impl …)
   back to aget heads so array detection + per-element typing + emit recognize
   them (else the array is mis-classified scalar and aget becomes a broken helper
   call); (2) type each INPUT array by its declared element type (array-types,
   merged with the body's :tag metadata) — a float input read through a double*
   param silently miscompiles; (3) type the OUTPUT by the map's computed element
   dtype (:dtype segmap), which may differ from the inputs (e.g. a float input
   promoted to double by a double literal).

   Returns a verified KernelArtifact whose ABI, compiler arguments and launch contract have one
   authoritative order."
  [segmap out-sym & {:keys [dtype kernel-name-prefix scalar-types array-types]
                     :or {dtype :double kernel-name-prefix "par_map"
                          scalar-types {} array-types {}}}]
  (let [idx (seg-idx segmap)
        ;; (1) normalize .invk array prims → aget/aset heads
        body (ce/normalize-array-prims (:lambda segmap))
        cast-fn (:cast-fn segmap)
        out-dtype (or (:dtype segmap) dtype)
        default-ctype (dt/ctype :opencl dtype)
        out-ctype (dt/ctype :opencl out-dtype)
        ;; (2) per-array element types: declared (array-types) ∪ body :tag metadata
        meta-types (ce/collect-array-types-from-meta body)
        array-types (merge meta-types array-types)
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        workgroup-size 256
        ;; Use pre-computed inputs/outputs/scalars from SegOp.
        ;; :outputs may carry SECONDARY outputs beyond `out-sym` — the side-effect
        ;; aset targets of a horizontally-fused multi-output map. Those are array
        ;; params too (declared NON-const, appended after the inputs so the invoke's
        ;; positional arg order matches the C signature); an input that is also
        ;; written (read+write buffer) likewise loses const.
        written (set (map #(symbol (name %)) (:outputs segmap)))
        out-name (when out-sym (symbol (name out-sym)))
        all-input-params (vec (sort-by name (:inputs segmap)))
        input-name-set (set (map #(symbol (name %)) all-input-params))
        primary-inout? (contains? input-name-set out-name)
        ;; The mapped result may also be read pointwise. Keep that physical value in the dedicated
        ;; result position and rewrite its scalar-region reads to the emitted `out` parameter;
        ;; passing it once as an input and again as the result violates both ABI identity and C
        ;; restrict aliasing.
        input-params (if primary-inout?
                       (filterv #(not= out-name (symbol (name %))) all-input-params)
                       all-input-params)
        result-c-name (if primary-inout? "inout_result" "out")
        result-symbol (symbol result-c-name)
        body (if primary-inout? (util/subst-syms {out-sym result-symbol} body) body)
        extra-outs (vec (sort-by name
                                 (remove #(or (= % out-name)
                                              (contains? input-name-set %))
                                         written)))
        arr-params (into input-params extra-outs)
        scl-params (vec (sort-by name (:scalars segmap)))
        arr-dtype (fn [s] (get array-types s (get array-types (symbol (name s)) dtype)))
        arr-type (fn [s] (get codegen/opencl-type-map (arr-dtype s) default-ctype))
        written-params (filterv #(contains? written (symbol (name %))) arr-params)
        arr-param-str (str/join ", "
                                (map (fn [s] (str "__global "
                                                  (when-not (contains? written (symbol (name s)))
                                                    "const ")
                                                  (arr-type s) "* restrict "
                                                  (ce/c-symbol s)))
                                     arr-params))
        scl-dtype #(ce/scalar-parameter-dtype % scalar-types dtype)
        scl-type #(dt/ctype :opencl (scl-dtype %))
        scalar-var-types (into {} (map (juxt identity scl-type)) scl-params)
        int-scalar-syms (set (filter #(dt/integral? (scl-dtype %)) scl-params))
        scl-param-str (str/join ", "
                                (map (fn [s] (str (scl-type s) " " (ce/c-symbol s)))
                                     scl-params))
        out-param (str "__global " out-ctype "* restrict " result-c-name)
        all-params (str/join ", "
                             (remove empty?
                                     [arr-param-str out-param scl-param-str "int _n_bound"]))
        local-index (fresh-c-local "idx"
                                   (concat arr-params [out-sym] scl-params ['_n_bound]))
        ;; Emit body as C expression
        adapted-body (ce/adapt-casts-for-dtype body out-dtype)
        arr-sym-set (cond-> (set (map #(symbol (name %)) arr-params))
                      primary-inout? (conj result-symbol))
        body-str (binding [ce/*emit-config* ce/opencl-config
                           ce/*scalar-type* out-ctype
                           ce/*scalar-var-types* scalar-var-types
                           ce/*idx-sym* idx
                           ce/*int-vars* (into ce/*int-vars* int-scalar-syms)]
                   (ce/emit-expr adapted-body idx arr-sym-set local-index))
        cast-str (if cast-fn (str "(" (name cast-fn) ")(" body-str ")") body-str)
        scalar-body-str (str result-c-name "[" local-index "] = " cast-str ";")
        ;; Affine-index vectorization (shared c_emit): a SegMap store is `out[idx] = f(..)`,
        ;; expressed here as the synthetic aset the vectorizer analyzes. The store target
        ;; is the literal `out` param (not c-symbol-mangled), so pass :store-name. nil ⇒
        ;; scalar loop.
        loop-region (binding [ce/*emit-config* ce/opencl-config
                              ce/*scalar-type* out-ctype
                              ce/*scalar-var-types* scalar-var-types
                              ce/*idx-sym* idx
                              ce/*int-vars* (into ce/*int-vars* int-scalar-syms)]
                      (ce/emit-vectorized-elementwise-loop
                       (list 'aset result-symbol idx
                             (if cast-fn (list cast-fn adapted-body) adapted-body))
                       idx (conj arr-sym-set result-symbol) local-index scalar-body-str
                       {:n-bound "_n_bound" :store-name result-c-name}))
        ;; pragmas cover the output dtype AND every input array's dtype
        scalar-dtype scl-dtype
        abi (kabi/validate!
             (vec (concat
                   (map (fn [s]
                          (let [written? (contains? written (symbol (name s)))
                                extra-output? (and written? (not (contains? input-name-set
                                                                            (symbol (name s)))))]
                            (kabi/slot s (cond extra-output? :output
                                               written? :inout
                                               :else :input)
                                       (arr-dtype s)
                                       :c-name (ce/c-symbol s)
                                       :role (cond written? :secondary-result
                                                   :else :operand))))
                        arr-params)
                   [(kabi/slot out-sym (if primary-inout? :inout :output) out-dtype
                               :c-name result-c-name :role :result)]
                   (map #(kabi/slot % :scalar (scalar-dtype %)
                                    :c-name (ce/c-symbol %) :role :parameter)
                        scl-params)
                   [(kabi/slot '_n_bound :scalar :int :role :bound)])))
        source (str (apply codegen/extension-pragmas out-dtype (map arr-dtype arr-params))
                    ;; registry intrinsics this body calls (e.g. rstr_dp4a) must be DEFINED
                    (ce/helper-sources scalar-body-str)
                    "__kernel void " kernel-name
                    "(" all-params ") {\n"
                    "    "
                    (or loop-region
                        (str "for (int " local-index " = get_global_id(0); " local-index
                             " < _n_bound; " local-index " += get_global_size(0)) {\n"
                             "        " scalar-body-str "\n"
                             "    }"))
                    "\n}\n")
        bound (seg-bound segmap)]
    (kart/make
     {:kernel-name kernel-name
      :source source
      :abi abi
      :arguments (vec (concat arr-params [out-sym] scl-params [bound]))
      :launch (klaunch/spec
               {:workgroup-size [workgroup-size]
                :group-count [(klaunch/ceil-div bound workgroup-size)]})
      :temporaries []
      :effects {:kind :elementwise-map}
      :provenance {:dialect :segmap :segop-id (:id segmap)}
      :attributes
      {:array-params arr-params
       :scalar-params scl-params
       ;; Array params (by signature name) the kernel WRITES — secondary fused outputs and
       ;; read+write inputs. The staging invoke copies these back to their JVM arrays after
       ;; launch; the resident role derivation marks written parameters :output.
       :written-arrays written-params
       :out-param out-param
       :dtype out-dtype}})))

(defn generate-segfoldmap-kernel
  "Schedule and emit one ordered SegFoldMap through the shared scalar KernelBody dialect."
  [segfoldmap & {:keys [kernel-name-prefix target-dialect workgroup-size scalar-types array-types]
                 :or {kernel-name-prefix "segmented_fold_map"
                      target-dialect :opencl-intel workgroup-size 256
                      scalar-types {} array-types {}}}]
  (let [{:keys [kernel-body segment-count]}
        (segfoldmap-body/lower segfoldmap
                               {:workgroup-size workgroup-size
                                :scalar-types scalar-types :array-types array-types})
        parameters (:parameters kernel-body)
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        parameter-names (into {}
                              (map (fn [parameter]
                                     [(:id parameter) (ce/c-symbol (:id parameter))]))
                              parameters)
        source (kernel-body-opencl/emit-scalar-kernel
                kernel-name kernel-body
                {:target-dialect target-dialect :parameter-names parameter-names})
        abi (body-abi/project-contracts
             (mapv (fn [{:keys [id kind dtype role]}]
                     (kabi/slot id kind dtype :c-name (get parameter-names id)
                                :role (if (= id '_nseg) :bound role)))
                   parameters)
             kernel-body)
        arguments (mapv (fn [parameter]
                          (if (= '_nseg (:id parameter)) segment-count (:id parameter)))
                        parameters)]
    (kart/make
     {:kernel-name kernel-name
      :source source
      :abi abi
      :arguments arguments
      :launch (:launch kernel-body)
      :temporaries []
      :effects {:kind :segmented-fold-map :association :ordered}
      :target (kernel-body-c-dialect/target
               (kernel-body-c-dialect/resolve! target-dialect))
      :provenance {:dialect :kernel-body :source-dialect :segfoldmap
                   :segop-id (:id segfoldmap)}
      :attributes {:kernel-body kernel-body
                   :array-params (vec (concat (:inputs segfoldmap) (:outputs segfoldmap)))
                   :scalar-params (vec (:scalars segfoldmap))
                   :dtype (first (:dtypes segfoldmap))
                   :aliasing :no-write-alias}})))

(defn generate-segmap-kernel-body
  "Schedule and emit one typed SegMap through the shared scalar KernelBody dialect."
  [segmap & {:keys [kernel-name-prefix target-dialect workgroup-size scalar-types array-types]
             :or {kernel-name-prefix "segmap"
                  target-dialect :opencl-intel workgroup-size 256
                  scalar-types {} array-types {}}}]
  (let [{:keys [kernel-body bound inputs outputs scalars]}
        (segmap-body/lower segmap
                           {:workgroup-size workgroup-size
                            :scalar-types scalar-types :array-types array-types})
        parameters (:parameters kernel-body)
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        parameter-names (into {}
                              (map (fn [parameter]
                                     [(:id parameter) (ce/c-symbol (:id parameter))]))
                              parameters)
        source (kernel-body-opencl/emit-scalar-kernel
                kernel-name kernel-body
                {:target-dialect target-dialect :parameter-names parameter-names})
        abi (body-abi/project-contracts
             (mapv (fn [{:keys [id kind dtype role]}]
                     (kabi/slot id kind dtype :c-name (get parameter-names id)
                                :role (if (= id '_n_bound) :bound role)))
                   parameters)
             kernel-body)
        arguments (mapv (fn [parameter]
                          (if (= '_n_bound (:id parameter)) bound (:id parameter)))
                        parameters)]
    (kart/make
     {:kernel-name kernel-name
      :source source
      :abi abi
      :arguments arguments
      :launch (:launch kernel-body)
      :temporaries []
      :effects {:kind (case (:write-conflict segmap)
                        :reduce :reducing-scatter
                        (if (:out-sym segmap) :elementwise-map :side-effect-map))
                :write-conflict (or (:write-conflict segmap) :unique)
                :conflict-contract (:conflict-contract segmap)}
      :target (kernel-body-c-dialect/target
               (kernel-body-c-dialect/resolve! target-dialect))
      :provenance {:dialect :kernel-body :source-dialect :segmap
                   :segop-id (:id segmap)}
      :attributes {:kernel-body kernel-body
                   :emission-route :kernel-body
                   :array-params (vec (concat inputs outputs))
                   :scalar-params scalars
                   :dtype (:dtype segmap)
                   :aliasing :no-write-alias}})))

(defn generate-segstencil-kernel-body
  "Schedule and emit one certified SegStencil through portable KernelBody."
  [stencil & {:keys [kernel-name-prefix target-dialect workgroup-size scalar-types array-types]
              :or {kernel-name-prefix "segstencil"
                   target-dialect :opencl-intel
                   scalar-types {} array-types {}}}]
  (let [workgroup-size (or workgroup-size (get-in stencil [:grid :block-size]) 256)
        {:keys [kernel-body bound inputs outputs scalars]}
        (segstencil-body/lower
         stencil {:workgroup-size workgroup-size
                  :scalar-types scalar-types :array-types array-types})
        parameters (:parameters kernel-body)
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        parameter-names (into {}
                              (map (fn [parameter]
                                     [(:id parameter) (ce/c-symbol (:id parameter))]))
                              parameters)
        source (kernel-body-opencl/emit-scalar-kernel
                kernel-name kernel-body
                {:target-dialect target-dialect :parameter-names parameter-names})
        abi (body-abi/project-contracts
             (mapv (fn [{:keys [id kind dtype role]}]
                     (kabi/slot id kind dtype :c-name (get parameter-names id)
                                :role (if (= id '_n_bound) :bound role)))
                   parameters)
             kernel-body)
        arguments (mapv (fn [parameter]
                          (if (= '_n_bound (:id parameter)) bound (:id parameter)))
                        parameters)]
    (kart/make
     {:kernel-name kernel-name :source source :abi abi :arguments arguments
      :launch (:launch kernel-body) :temporaries []
      :effects {:kind :stencil :boundary (:boundary stencil) :radius (:radius stencil)}
      :target (kernel-body-c-dialect/target
               (kernel-body-c-dialect/resolve! target-dialect))
      :provenance {:dialect :kernel-body :source-dialect :segstencil
                   :segop-id (:id stencil)}
      :attributes {:kernel-body kernel-body
                   :array-params (vec (concat inputs outputs))
                   :scalar-params scalars :dtype (:dtype stencil)
                   :boundary (:boundary stencil) :radius (:radius stencil)
                   :aliasing :no-write-alias}})))

;; ================================================================
;; SegRed → OpenCL kernel (two-phase reduction)
;; ================================================================

(defn- product-neutral-c
  [neutral dtype]
  (let [dtype (dt/canon dtype)
        field (when (symbol? neutral) (name neutral))]
    (cond
      (and (contains? #{:float :double :half} dtype) (= "POSITIVE_INFINITY" field)) "INFINITY"
      (and (contains? #{:float :double :half} dtype) (= "NEGATIVE_INFINITY" field)) "-INFINITY"
      (and (= :int dtype) (= "MAX_VALUE" field)) "INT_MAX"
      (and (= :int dtype) (= "MIN_VALUE" field)) "INT_MIN"
      (and (= :long dtype) (= "MAX_VALUE" field)) "LONG_MAX"
      (and (= :long dtype) (= "MIN_VALUE" field)) "LONG_MIN"
      (and (number? neutral) (Double/isInfinite (double neutral)))
      (if (pos? (double neutral)) "INFINITY" "-INFINITY")
      (number? neutral) (case dtype
                          :float (str (float neutral) "f")
                          :double (str (double neutral))
                          (str (long neutral)))
      (and (seq? neutral) (= 2 (count neutral)))
      (product-neutral-c (second neutral) dtype)
      :else (throw (ex-info "product reduction neutral has no OpenCL literal"
                            {:reason :product-reduction-neutral-not-emittable
                             :neutral neutral :dtype dtype})))))

(defn generate-product-reduction-kernel
  "Lower a typed segmented ProductReduction to one deterministic workgroup tree per segment.

   This is the portable scheduled body. Every lane folds a strided subset, then the declared
   closed combine region merges lane products in a fixed tree. Mixed component dtypes share no
   representation assumptions; each gets its own typed local array."
  [segred & {:keys [kernel-name-prefix scalar-types array-types]
             :or {kernel-name-prefix "product_reduce" scalar-types {} array-types {}}}]
  (let [operator (reduction/validate! (:reduction segred))
        schedule (reduction/validate-schedule! (:schedule segred))
        _ (when-not (= :segmented-workgroup-tree (:strategy schedule))
            (throw (ex-info "product reduction schedule has no OpenCL lowering"
                            {:reason :product-reduction-schedule-not-emittable
                             :strategy (:strategy schedule)})))
        _ (when-not (true? (get-in operator [:algebra :associative?]))
            (throw (ex-info "parallel product schedule requires an explicit associative algebra"
                            {:reason :product-reduction-not-associative
                             :algebra (:algebra operator)})))
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
        launch-segments (let [bounds (mapv :bound segment-dims)]
                          (case (count bounds)
                            0 1
                            1 (klaunch/runtime-value (first bounds))
                            (apply klaunch/product bounds)))
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

(defn generate-segred-kernel-body
  "Lower an eligible scalar SegRed through verified KernelBody and a thin C-family dialect.

   This function is public so CUDA/HIP compiler fixtures can validate the exact same scheduled
   body without a device. It throws only structured `:segred-kernel-body-declined` exceptions for
   unsupported scalar regions; verified-body or emitter failures remain compiler errors."
  [segred out-sym & {:keys [dtype kernel-name-prefix scalar-types array-types target-dialect]
                     :or {dtype :double kernel-name-prefix "par_reduce" scalar-types {}
                          array-types {} target-dialect :opencl-intel}}]
  (let [{:keys [kernel-body operator identity arrays scalars output bound]}
        (segred-body/lower segred out-sym :dtype dtype :array-types array-types
                           :scalar-types scalar-types)
        dtype (or (:dtype segred) dtype)
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        parameter-names (into {output "output" '_n_bound "_n_bound"}
                              (map (fn [id] [id (ce/c-symbol id)]))
                              (concat arrays scalars))
        source (kernel-body-opencl/emit-scalar-kernel
                kernel-name kernel-body
                {:target-dialect target-dialect :parameter-names parameter-names})
        scalar-dtype (fn [id]
                       (or (get scalar-types id)
                           (get scalar-types (symbol (name id)))
                           dtype))
        result-name (or out-sym 'output)
        abi (kabi/validate!
             (vec (concat
                   (map #(kabi/slot % :input dtype :c-name (ce/c-symbol %) :role :operand
                                    :aliasing :no-write-alias)
                        arrays)
                   [(kabi/slot result-name :output dtype :c-name "output" :role :result)]
                   (map #(kabi/slot % :scalar (scalar-dtype %)
                                    :c-name (ce/c-symbol %) :role :parameter)
                        scalars)
                   [(kabi/slot '_n_bound :scalar :int :role :bound)])))
        c-op ({:+ "+" :* "*" :min "fmin" :max "fmax"} operator)]
    (kart/make
     {:kernel-name kernel-name
      :target (kernel-body-c-dialect/target (kernel-body-c-dialect/resolve! target-dialect))
      :source source
      :abi abi
      :arguments (vec (concat arrays [out-sym] scalars [bound]))
      :launch (:launch kernel-body)
      :temporaries []
      :effects {:kind :pure-reduction}
      :provenance {:dialect :segred :segop-id (:id segred)}
      :attributes {:array-params arrays
                   :scalar-params scalars
                   :dtype dtype
                   :n-phases 2
                   :identity-val identity
                   :c-op c-op
                   :kernel-body kernel-body
                   :emission-route :kernel-body
                   :target-dialect target-dialect}})))

(defn generate-scheduled-segmap-kernel
  "Emit one scheduled SegMap through the single KernelBody-first C-family boundary.

   Portable targets never consume source-shaped OpenCL fallback code. OpenCL retains the verified
   compatibility emitter for scalar regions and memory effects that KernelBody cannot represent
   yet, with the structured decline attached to the artifact so the remaining debt is observable."
  [operation & {:keys [dtype kernel-name-prefix scalar-types array-types target-dialect]
                :or {kernel-name-prefix "segmap" scalar-types {} array-types {}
                     target-dialect :opencl-intel}}]
  (let [dtype (or (:dtype operation) dtype :double)
        target (kernel-body-c-dialect/resolve! target-dialect)]
    (kart/certify-scheduled-operation
     (try
       (generate-segmap-kernel-body
        operation :dtype dtype :scalar-types scalar-types :array-types array-types
        :target-dialect target-dialect :kernel-name-prefix kernel-name-prefix)
       (catch clojure.lang.ExceptionInfo exception
         (if (and (kernel-body-c-dialect/opencl? target)
                  (segmap-body/declined? exception))
           (-> (if (:out-sym operation)
                 (generate-segmap-kernel
                  operation (:out-sym operation)
                  :dtype dtype :scalar-types scalar-types :array-types array-types
                  :kernel-name-prefix kernel-name-prefix)
                 (generate-explicit-segmap-kernel
                  operation :dtype dtype :scalar-types scalar-types :array-types array-types
                  :kernel-name-prefix (str kernel-name-prefix "_effect")))
               (assoc-in [:attributes :emission-route] :verified-segmap-opencl)
               (assoc-in [:attributes :kernel-body-decline]
                         (assoc (ex-data exception) :fallback :verified-segmap-opencl)))
           (throw exception))))
     operation)))

(defn generate-segred-kernel
  "Emit a scheduled full reduction exclusively through target-neutral KernelBody.

   The artifact carries the complete ordered ABI, symbolic launch, workgroup scratch and two-phase
   reduction protocol. Unsupported scalar regions fail with their structured KernelBody decline;
   no target may recover semantics by reparsing a source-shaped lambda."
  [segred out-sym & {:keys [dtype kernel-name-prefix scalar-types array-types target-dialect]
                     :or {dtype :double kernel-name-prefix "par_reduce" scalar-types {}
                          array-types {} target-dialect :opencl-intel}}]
  (when (seq (segop/seg-space-segment-dims (:space segred)))
    (throw (ex-info
            "segmented reductions require a verified contraction schedule"
            {:reason :segmented-reduction-requires-contraction-schedule
             :operation (:id segred) :fallback :none})))
  (try
    (generate-segred-kernel-body
     segred out-sym :dtype dtype :kernel-name-prefix kernel-name-prefix
     :scalar-types scalar-types :array-types array-types :target-dialect target-dialect)
    (catch clojure.lang.ExceptionInfo exception
      (when-not (segred-body/declined? exception) (throw exception))
      (let [decline (assoc (ex-data exception) :fallback :none)]
        (if (get-in segred [:reduction :attributes :result-region])
          (throw (ex-info
                  "completed-reduction transforms require verified KernelBody lowering"
                  (assoc decline :reason :segred-result-transform-lowering
                         :kernel-body-decline (ex-data exception))
                  exception))
          (throw (ex-info
                  "scheduled reduction is outside the portable KernelBody C-family subset"
                  {:reason :kernel-graph-target-lowering-missing
                   :target-dialect target-dialect
                   :operation (:id segred)
                   :kernel-body-decline decline
                   :fallback :none}
                  exception)))))))
;; ================================================================
;; KernelGraph scan → OpenCL KernelArtifacts
;; ================================================================

(defn- finalize-emitted-graph
  "Give an artifact-valued graph its one ordered external ABI.

   Node artifacts remain free to order their own physical parameters. The graph boundary groups
   equal symbolic scalars, proves their emitted dtypes agree, and exposes every external buffer
   exactly once."
  [emitted target-dialect scalar-types]
  (let [external-buffers (vec (distinct (concat (:inputs emitted) (:outputs emitted))))
        scalar-pairs (->> (:nodes emitted)
                          (mapcat (fn [node]
                                    (map vector (get-in node [:operation :abi])
                                         (get-in node [:operation :arguments]))))
                          (filter (fn [[slot argument]]
                                    (and (= :scalar (:kind slot)) (symbol? argument))))
                          vec)
        scalar-groups (group-by second scalar-pairs)
        _ (doseq [[argument pairs] scalar-groups]
            (when-not (apply = (map (comp :kernel-dtype first) pairs))
              (throw (ex-info "kernel graph scalar has inconsistent emitted ABI dtypes"
                              {:reason :kernel-graph-scalar-dtype
                               :argument argument :slots (mapv first pairs)}))))
        pointer-abi (mapv (fn [{:keys [id dtype role]}]
                            (kabi/slot id (if (= :input role) :input :output) dtype
                                       :role (case role
                                               :input :operand
                                               :output :result
                                               :inout :inout)))
                          external-buffers)
        scalar-arguments (vec (sort-by name (keys scalar-groups)))
        scalar-abi (mapv (fn [argument]
                           (let [slots (mapv first (get scalar-groups argument))
                                 kernel-dtype (:kernel-dtype (first slots))
                                 logical-dtype (or (get scalar-types argument) kernel-dtype)
                                 role (if (some #(= :bound (:role %)) slots)
                                        :bound :parameter)]
                             (kabi/slot argument :scalar logical-dtype
                                        :kernel-dtype kernel-dtype :role role)))
                         scalar-arguments)]
    (-> emitted
        (assoc :abi (vec (concat pointer-abi scalar-abi))
               :arguments (vec (concat (map :id external-buffers) scalar-arguments)))
        (assoc-in [:provenance :target-dialect] target-dialect)
        (assoc-in [:attributes :emitted?] true)
        kgraph/validate!)))
(defn- graph-array-types
  [graph array-types]
  (let [declared-array-types
        (into {}
              (map (juxt :id :dtype))
              (distinct (concat (:inputs graph) (:outputs graph) (:temporaries graph))))
        _ (doseq [[id declared] declared-array-types
                  :let [supplied (or (get array-types id)
                                     (when (symbol? id)
                                       (get array-types (symbol (name id)))))]
                  :when (and supplied (not= supplied declared))]
            (throw (ex-info "kernel graph buffer dtype differs from target emission facts"
                            {:reason :kernel-graph-buffer-dtype
                             :buffer id :declared declared :supplied supplied})))]
    (merge array-types declared-array-types)))

(defn generate-scan-kernel-graph
  "Lower a certified scheduled scan graph through portable KernelBody data.

   Every node retains the graph's exact ordered ABI and dependency structure.  OpenCL, CUDA and
   HIP consume the same scheduled bodies; this boundary contains no source-template fallback."
  [graph & {:keys [kernel-name-prefix scalar-types array-types target-dialect]
            :or {kernel-name-prefix "segscan" scalar-types {} array-types {}
                 target-dialect :opencl-intel}}]
  (let [graph (kgraph/validate! graph)
        algebra (get-in graph [:attributes :scan-algebra])
        scan-mode (or (get-in graph [:attributes :scan-mode]) :inclusive)
        _ (when-not (scan/associative-scan? algebra)
            (throw (ex-info "scan KernelGraph lacks certified associative algebra"
                            {:reason :uncertified-scan-graph :algebra algebra})))
        _ (when-not (contains? #{:inclusive :exclusive} scan-mode)
            (throw (ex-info "scan KernelGraph has an unsupported result mode"
                            {:reason :scan-mode-not-emittable :mode scan-mode})))
        _ (when-not (= 1 (count (:outputs graph)))
            (throw (ex-info "scan artifact lowering requires exactly one graph output"
                            {:reason :scan-multi-output-unimplemented
                             :outputs (mapv :id (:outputs graph))})))
        target (kernel-body-c-dialect/resolve! target-dialect)
        target-module (kernel-body-c-dialect/target target)
        array-types (graph-array-types graph array-types)
        buffers (into {} (map (juxt :id identity))
                      (concat (:inputs graph) (:outputs graph) (:temporaries graph)))
        temporary-ids (set (map :id (:temporaries graph)))
        output-ids (set (map :id (:outputs graph)))
        first-scan (some #(when (instance? raster.compiler.ir.segop.SegScan (:operation %))
                            (:operation %))
                         (:nodes graph))
        scan-workgroup (or (get-in first-scan [:grid :block-size]) 256)
        scalar-dtype (fn [id]
                       (or (get scalar-types id)
                           (get scalar-types (symbol (name id)))
                           (:dtype algebra)))
        emitted
        (kgraph/map-operations
         graph
         (fn [{:keys [id operation uses]}]
           (let [{:keys [kernel-body phase bound pointer-ids scalar-ids]}
                 (segscan-body/lower
                  operation
                  {:uses uses :buffers buffers
                   :temporary-ids temporary-ids :output-ids output-ids
                   :scan-algebra algebra :scan-mode scan-mode
                   :scan-workgroup scan-workgroup
                   :scalar-types scalar-types :array-types array-types})
                 parameters (:parameters kernel-body)
                 kernel-name (str kernel-name-prefix "_" (str/replace (name phase) "-" "_")
                                  "_" (gensym ""))
                 parameter-names (into {}
                                       (map (fn [parameter]
                                              [(:id parameter) (ce/c-symbol (:id parameter))]))
                                       parameters)
                 source (kernel-body-opencl/emit-scalar-kernel
                         kernel-name kernel-body
                         {:target-dialect target-dialect
                          :parameter-names parameter-names})
                 pointer-slots
                 (mapv
                  (fn [{:keys [buffer access]}]
                    (let [spec (get buffers buffer)
                          output? (contains? #{:write :read-write} access)]
                      (kabi/slot
                       buffer
                       (case access :read :input :write :output :read-write :inout)
                       (:dtype spec) :c-name (get parameter-names buffer)
                       :role (cond
                               (contains? temporary-ids buffer) :temporary
                               (and output? (contains? output-ids buffer)) :result
                               :else :operand))))
                  uses)
                 scalar-slots
                 (mapv #(kabi/slot % :scalar (scalar-dtype %)
                                   :c-name (get parameter-names %) :role :parameter)
                       scalar-ids)
                 bound-slot (kabi/slot '_n_bound :scalar :int
                                       :c-name (get parameter-names '_n_bound) :role :bound)
                 abi (body-abi/project-contracts
                      (vec (concat pointer-slots scalar-slots [bound-slot])) kernel-body)
                 artifact
                 (kart/make
                  {:kernel-name kernel-name :target target-module :source source :abi abi
                   :arguments (vec (concat pointer-ids scalar-ids [bound]))
                   :launch (:launch kernel-body) :temporaries []
                   :effects {:kind :scan-stage :phase phase}
                   :provenance {:dialect :kernel-body :source-dialect :segscan
                                :segop-id (:id operation) :graph-node id}
                   :attributes {:phase phase :dtype (:dtype algebra) :scan-mode scan-mode
                                :scan-workgroup scan-workgroup :kernel-body kernel-body
                                :emission-route :kernel-body
                                :target-dialect target-dialect}})]
             (kart/certify-scheduled-operation artifact operation))))]
    (finalize-emitted-graph emitted target-module scalar-types)))

(defn- generate-elementwise-kernel-graph
  [graph {:keys [scalar-types array-types target-dialect]
          :or {scalar-types {} array-types {} target-dialect :opencl-intel}}]
  (let [target (kernel-body-c-dialect/resolve! target-dialect)
        array-types (graph-array-types graph array-types)
        emitted
        (kgraph/map-operations
         graph
         (fn [{:keys [id operation]}]
           (kart/certify-scheduled-operation
            (cond
              (instance? raster.compiler.ir.segop.SegMap operation)
              (generate-scheduled-segmap-kernel
               operation :dtype (:dtype operation)
               :scalar-types scalar-types :array-types array-types
               :target-dialect target-dialect
               :kernel-name-prefix "graph_segmap")

              (instance? raster.compiler.ir.segop.SegStencil operation)
              (generate-segstencil-kernel-body
               operation :scalar-types scalar-types :array-types array-types
               :target-dialect target-dialect
               :kernel-name-prefix "graph_segstencil")

              :else
              (throw (ex-info "OpenCL elementwise graph has an unsupported scheduled node"
                              {:reason :kernel-graph-node-target-lowering-missing
                               :target :opencl-c :node id :operation operation})))
            operation)))]
    (finalize-emitted-graph emitted (kernel-body-c-dialect/target target) scalar-types)))

(declare generate-contraction-kernel-artifact)

(defn- generate-reduction-kernel-graph
  [graph {:keys [scalar-types array-types target-dialect target-device contraction-facts]
          :or {scalar-types {} array-types {} target-dialect :opencl-intel
               contraction-facts {}}}]
  (let [array-types (graph-array-types graph array-types)
        emitted
        (kgraph/map-operations
         graph
         (fn [{:keys [id operation]}]
           (let [outputs (vec (:outputs operation))]
             (when-not (= 1 (count outputs))
               (throw (ex-info "scalar SegRed graph node requires exactly one scheduled output"
                               {:reason :kernel-graph-reduction-output
                                :target :opencl-c :node id :outputs outputs})))
             (kart/certify-scheduled-operation
              (if (seq (segop/seg-space-segment-dims (:space operation)))
                (let [facts (or (get contraction-facts (:id operation))
                                (throw (ex-info
                                        "segmented reduction lacks verified contraction facts"
                                        {:reason :kernel-graph-segmented-reduction-facts
                                         :operation (:id operation) :fallback :none})))
                      descriptor (hw/descriptor-for target-device)
                      planned (contraction-schedule/plan-portable-body
                               facts operation descriptor
                               {:array-types array-types :scalar-types scalar-types})]
                  (when-not (:ok planned)
                    (throw (ex-info
                            "segmented reduction has no portable KernelBody schedule"
                            {:reason :kernel-graph-segmented-reduction-body
                             :operation (:id operation) :schedule-decline planned
                             :fallback :none})))
                  (generate-contraction-kernel-artifact
                   (:body planned) :target-dialect target-dialect
                   :kernel-name-prefix "graph_contraction"))
                (generate-segred-kernel
                 operation (first outputs)
                 :dtype (:dtype operation)
                 :scalar-types scalar-types :array-types array-types
                 :target-dialect target-dialect))
              operation))))]
    (finalize-emitted-graph
     emitted
     (kernel-body-c-dialect/target (kernel-body-c-dialect/resolve! target-dialect))
     scalar-types)))

(defn- generate-fold-map-kernel-graph
  [graph {:keys [scalar-types array-types target-dialect]
          :or {scalar-types {} array-types {} target-dialect :opencl-intel}}]
  (let [target (kernel-body-c-dialect/resolve! target-dialect)
        array-types (graph-array-types graph array-types)
        emitted
        (kgraph/map-operations
         graph
         (fn [{:keys [operation]}]
           (try
             (kart/certify-scheduled-operation
              (generate-segfoldmap-kernel
               operation :scalar-types scalar-types :array-types array-types
               :target-dialect target-dialect
               :kernel-name-prefix "graph_segmented_fold_map")
              operation)
             (catch clojure.lang.ExceptionInfo exception
               (if (and (not (kernel-body-c-dialect/opencl? target))
                        (segfoldmap-body/declined? exception))
                 (throw (ex-info
                         "scheduled fold-map is outside the portable KernelBody C-family subset"
                         {:reason :kernel-graph-target-lowering-missing
                          :target-dialect target-dialect
                          :operation (:id operation)
                          :kernel-body-decline (ex-data exception)
                          :fallback :none}
                         exception))
                 (throw exception))))))]
    (finalize-emitted-graph
     emitted
     (kernel-body-c-dialect/target target)
     scalar-types)))

(defn generate-kernel-graph
  "Target-lower one scheduled KernelGraph through the backend's single graph-emission boundary.

   Operation-family recognition is confined here and must be proved by graph attributes established
   by scheduling. Unsupported graph families fail loudly; callers never fall back to reconstructing
   an operation from source spelling or node names."
  [graph & {:as opts}]
  (let [graph (kgraph/validate! graph)
        target-dialect (get opts :target-dialect :opencl-intel)
        opencl? (kernel-body-c-dialect/opencl?
                 (kernel-body-c-dialect/resolve! target-dialect))]
    (cond
      (scan/associative-scan? (get-in graph [:attributes :scan-algebra]))
      (apply generate-scan-kernel-graph graph (mapcat identity opts))

      (and (seq (:nodes graph))
           (every? #(or (instance? raster.compiler.ir.segop.SegMap (:operation %))
                        (instance? raster.compiler.ir.segop.SegStencil (:operation %)))
                   (:nodes graph)))
      (try
        (generate-elementwise-kernel-graph graph opts)
        (catch clojure.lang.ExceptionInfo exception
          (if (and (not opencl?) (segmap-body/declined? exception))
            (throw (ex-info "scheduled map is outside the portable KernelBody C-family subset"
                            {:reason :kernel-graph-target-lowering-missing
                             :target-dialect target-dialect
                             :kernel-body-decline (ex-data exception)
                             :fallback :none}
                            exception))
            (throw exception))))

      (and (seq (:nodes graph))
           (every? #(instance? raster.compiler.ir.segop.SegRed (:operation %))
                   (:nodes graph)))
      (generate-reduction-kernel-graph graph opts)

      (and (seq (:nodes graph))
           (every? #(instance? raster.compiler.ir.segop.SegFoldMap (:operation %))
                   (:nodes graph)))
      (generate-fold-map-kernel-graph graph opts)

      :else
      (throw (ex-info "C-family backend has no target lowering for scheduled KernelGraph"
                      {:reason :kernel-graph-target-lowering-missing
                       :target-dialect target-dialect
                       :fallback :none
                       :provenance (:provenance graph)
                       :attributes (:attributes graph)})))))

;; ================================================================
;; Segmented reduction (contraction) → OpenCL — the multi-axis SegSpace path
;; ================================================================

(defn generate-contraction-kernel-body
  "Emit one scheduled portable contraction KernelBody through the shared C-family boundary."
  [kernel-body & {:keys [kernel-name-prefix target-dialect]
                  :or {kernel-name-prefix "contract" target-dialect :opencl-intel}}]
  (let [kernel-body (kbody/validate! kernel-body)
        parameters (:parameters kernel-body)
        inputs (vec (map :id (filter #(and (= :input (:kind %))
                                           (not= :epilogue (:role %))) parameters)))
        epilogue-operands
        (vec (map :id (filter #(and (= :input (:kind %))
                                    (= :epilogue (:role %))) parameters)))
        output-parameter (first (filter #(= :output (:kind %)) parameters))
        output (:id output-parameter)
        scalar-parameters (vec (filter #(and (= :scalar (:kind %))
                                             (not= :epilogue (:role %))) parameters))
        epilogue-scalars
        (vec (map :id (filter #(and (= :scalar (:kind %))
                                    (= :epilogue (:role %))) parameters)))
        scalars (vec (map :id (remove #(= '_nseg (:id %)) scalar-parameters)))
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        parameter-names (into {output "out" '_nseg "_nseg"}
                              (map (fn [parameter]
                                     [(:id parameter) (ce/c-symbol (:id parameter))]))
                              parameters)
        source (kernel-body-opencl/emit-scalar-kernel
                kernel-name kernel-body
                {:target-dialect target-dialect :parameter-names parameter-names})
        abi (kabi/validate!
             (mapv (fn [parameter]
                     (let [{:keys [id kind dtype role]} parameter]
                       (kabi/slot
                        id kind dtype
                        :c-name (get parameter-names id)
                        :role (if (= id '_nseg) :bound role)
                        :aliasing (when (= kind :input) :no-write-alias))))
                   parameters))]
    {:kernel-name kernel-name
     :target (kernel-body-c-dialect/target
              (kernel-body-c-dialect/resolve! target-dialect))
     :source source
     :abi abi
     :array-params inputs
     :scalar-params scalars
     :epilogue-operands epilogue-operands
     :epilogue-scalars epilogue-scalars
     :output output
     :kernel-body kernel-body
     :launch (:launch kernel-body)}))

(defn generate-contraction-kernel-artifact
  "Emit one portable contraction KernelBody as a complete executable compiler value."
  [kernel-body & {:keys [kernel-name-prefix target-dialect]
                  :or {kernel-name-prefix "contract" target-dialect :opencl-intel}}]
  (let [kernel-body (kbody/validate! kernel-body)
        emitted (generate-contraction-kernel-body
                 kernel-body :kernel-name-prefix kernel-name-prefix
                 :target-dialect target-dialect)
        launch-segment-count (get-in kernel-body [:attributes :launch-segment-count])
        arguments (mapv (fn [parameter]
                          (if (= '_nseg (:id parameter)) launch-segment-count (:id parameter)))
                        (:parameters kernel-body))]
    (kart/make
     {:kernel-name (:kernel-name emitted)
      :target (:target emitted)
      :source (:source emitted)
      :abi (:abi emitted)
      :arguments arguments
      :launch (:launch emitted)
      :effects {:kind :pure-contraction}
      :provenance {:dialect :kernel-body
                   :source-dialect :segcontract
                   :segop-id (get-in kernel-body [:provenance :segop-id])}
      :attributes {:array-params (:array-params emitted)
                   :scalar-params (:scalar-params emitted)
                   :epilogue-operands (:epilogue-operands emitted)
                   :epilogue-scalars (:epilogue-scalars emitted)
                   :written-arrays #{(:output emitted)}
                   :kernel-body kernel-body
                   :strategy (get-in kernel-body [:schedule :strategy])
                   :out-elems launch-segment-count
                   :emission-route :kernel-body
                   :target-dialect target-dialect}})))

(defn generate-segmented-reduce-kernel
  "NAIVE segmented reduction (a contraction) → OpenCL. One work-item per SEGMENT (free-axis
   tuple); each sequentially folds over the reduced (innermost) axis. segment-dims = the
   free/parallel axes, reduced-dim = the contracted axis (Futhark innermost-reduced
   convention). Emits: decompose the flat segment id into the free indices (row-major via
   suffix products), loop over the reduced axis accumulating the combine's element (the
   product), store out[seg]. This is the multi-axis SegSpace emit path the 1-D emitters
   never handled; NAIVE (one thread/output) — the substrate the BlkRegTiling-style tiling
   pass optimizes. Combine op/element detection mirrors generate-segred-kernel (a shared
   helper is a later dedup). Bounds must be symbols or int literals in this prototype."
  [segred out-sym & {:keys [dtype kernel-name-prefix]
                     :or {dtype :double kernel-name-prefix "contract"}}]
  (let [space    (:space segred)
        seg-dims (segop/seg-space-segment-dims space)   ; free (parallel) axes, outer→inner
        red-dim  (segop/seg-space-reduced-dim space)    ; contracted axis (innermost)
        _ (when (empty? seg-dims)
            (throw (ex-info "segmented-reduce: no segment dims — use generate-segred-kernel for a full reduction"
                            {:space space})))
        dtype (or (:dtype segred) dtype)
        ctype (dt/ctype :opencl dtype)
        {:keys [acc init lambda]} (segop/scalar-reduce-op segred)
        lambda (ce/normalize-array-prims lambda)
        ;; combine op + element detection (mirrors generate-segred-kernel)
        op-sym (when (seq? lambda) (descriptor/semantic-op lambda))
        normalized-op (get {'+ '+ 'clojure.core/+ '+ 'raster.numeric/+ '+
                            '* '* 'clojure.core/* '* 'raster.numeric/* '*
                            'max 'max 'clojure.core/max 'max 'Math/max 'max 'raster.numeric/max 'max
                            'min 'min 'clojure.core/min 'min 'Math/min 'min 'raster.numeric/min 'min}
                           op-sym op-sym)
        c-op (condp = normalized-op '+ "+" '* "*" 'max "fmax" 'min "fmin"
                    (throw (ex-info (str "segmented-reduce: unsupported combine op " op-sym
                                         " — need an associative op (+ * max min)")
                                    {:op op-sym :lambda lambda})))
        c-combine (fn [a b] (if (#{"fmax" "fmin"} c-op)
                              (str c-op "(" a ", " b ")")
                              (str "(" a " " c-op " " b ")")))
        op-args (vec (when (seq? lambda) (descriptor/call-args lambda)))
        acc-at? (fn [a] (or (= a acc) (and (seq? a) (= 'double (first a)) (= acc (second a)))))
        elem-expr (when (>= (count op-args) 2)
                    (let [a0 (nth op-args 0) a1 (nth op-args 1)]
                      (cond (acc-at? a0) a1 (acc-at? a1) a0 :else nil)))
        _ (when (nil? elem-expr)
            (throw (ex-info "segmented-reduce: could not isolate the element (non-acc) operand"
                            {:lambda lambda :acc acc})))
        ;; params
        arr-params (vec (sort-by name (:inputs segred)))
        scl-params (vec (sort-by name (:scalars segred)))
        kernel-name (str kernel-name-prefix "_" (gensym ""))
        bound-c (fn [b] (cond (symbol? b) (ce/c-symbol b)
                              (number? b) (str b)
                              :else (throw (ex-info "segmented-reduce: bound must be a symbol or int in this prototype"
                                                    {:bound b}))))
        arr-param-str (str/join ", " (map (fn [s] (str "__global const " ctype "* restrict " (ce/c-symbol s))) arr-params))
        scl-param-str (str/join ", " (map (fn [s] (str "int " (ce/c-symbol s))) scl-params))
        ;; Trailing `int _nseg` (= number of segments = launch count) matches the generic
        ;; emitter convention (generate-segmap-kernel's `int _n_bound`) and the arg order
        ;; invoke-registered-kernel builds (inputs, output, scalars, count).
        all-params (str/join ", " (remove empty?
                                          [arr-param-str (str "__global " ctype "* restrict out")
                                           scl-param-str "int _nseg"]))
        seg-bound-cs (mapv (fn [d] (bound-c (:bound d))) seg-dims)
        ;; row-major decompose: idx_p = (seg / product(bounds after p)) % bound_p
        decomp (str/join "\n"
                         (map-indexed
                          (fn [p d]
                            (let [after (drop (inc p) seg-bound-cs)
                                  div (if (seq after) (str "(seg / (" (str/join " * " after) "))") "seg")
                                  rhs (if (seq after) (str div " % " (bound-c (:bound d))) (str "seg % " (bound-c (:bound d))))]
                              (str "    int " (ce/c-symbol (:name d)) " = " rhs ";")))
                          seg-dims))
        ;; ce/emit-expr renders the primary index sym as the C var "idx" (its convention);
        ;; the reduced-axis loop var MUST therefore be named "idx" so the body's references
        ;; to the reduced index resolve. Free indices render by their own c-symbol names.
        red-bound-c (bound-c (:bound red-dim))
        int-vars (into #{} (map #(symbol (name %)))
                       (concat (map :name seg-dims) [(:name red-dim)] scl-params))
        arr-sym-set (set (map #(symbol (name %)) arr-params))
        elem-str (binding [ce/*emit-config* ce/opencl-config
                           ce/*scalar-type* ctype
                           ce/*int-vars* (into ce/*int-vars* int-vars)]
                   (ce/emit-expr (ce/adapt-casts-for-dtype elem-expr dtype) (:name red-dim) arr-sym-set))
        source (str (codegen/extension-pragmas dtype)
                    "__kernel void " kernel-name "(" all-params ") {\n"
                    "    int seg = get_global_id(0);\n"
                    "    if (seg >= _nseg) return;\n"
                    decomp "\n"
                    "    " ctype " acc = " (str init) ";\n"
                    "    for (int idx = 0; idx < " red-bound-c "; idx++) {\n"
                    "        acc = " (c-combine "acc" elem-str) ";\n"
                    "    }\n"
                    "    out[seg] = acc;\n"
                    "}\n")]
    {:kernel-name kernel-name
     :source source
     :array-params arr-params
     :scalar-params scl-params
     :abi (kabi/validate!
           (vec (concat
                 (map #(kabi/slot % :input dtype :c-name (ce/c-symbol %) :role :operand) arr-params)
                 [(kabi/slot out-sym :output dtype :c-name "out" :role :result)]
                 (map #(kabi/slot % :scalar :int :c-name (ce/c-symbol %) :role :parameter) scl-params)
                 [(kabi/slot '_nseg :scalar :int :role :bound)])))
     :dtype dtype
     :c-op c-op}))

(defn generate-segmap-nd-kernel
  "N-D pure map → OpenCL: an OUTER PRODUCT / broadcast / elementwise contraction with ZERO
   contract axes. One work-item per output element; decompose the flat index into the free
   indices (row-major, outer→inner) and write out[seg] = body. This is the empty-reduce
   projection of a contraction — the SegMap counterpart of generate-segmented-reduce-kernel.
   The SegMap's space dims are ALL free/output axes (no reduced dim). Trailing `int _nseg`
   count param (matches the generic emitter convention)."
  [segmap out-sym & {:keys [dtype] :or {dtype :double}}]
  (let [dims (segop/seg-space-dims (:space segmap))   ; all free (no reduced dim)
        dtype (or (:dtype segmap) dtype)
        ctype (dt/ctype :opencl dtype)
        body (ce/normalize-array-prims (:lambda segmap))
        arr-params (vec (sort-by name (:inputs segmap)))
        arr-sym-set (set (map #(symbol (name %)) arr-params))
        kernel-name (str "segmap_nd_" (gensym ""))
        bound-c (fn [b] (cond (symbol? b) (ce/c-symbol b)
                              (number? b) (str b)
                              :else (throw (ex-info "segmap-nd: bound must be symbol or int" {:bound b}))))
        dim-cs (mapv (fn [d] (bound-c (:bound d))) dims)
        decomp (str/join "\n"
                         (map-indexed
                          (fn [p d]
                            (let [after (drop (inc p) dim-cs)
                                  rhs (if (seq after)
                                        (str "(seg / (" (str/join " * " after) ")) % " (bound-c (:bound d)))
                                        (str "seg % " (bound-c (:bound d))))]
                              (str "    int " (ce/c-symbol (:name d)) " = " rhs ";")))
                          dims))
        int-vars (into #{} (map #(symbol (name (:name %)))) dims)
        dummy (gensym "z__")
        body-str (binding [ce/*emit-config* ce/opencl-config
                           ce/*scalar-type* ctype
                           ce/*int-vars* (into ce/*int-vars* int-vars)]
                   (ce/emit-expr (ce/adapt-casts-for-dtype body dtype) dummy arr-sym-set))
        arr-param-str (str/join ", " (map (fn [s] (str "__global const " ctype "* restrict " (ce/c-symbol s))) arr-params))
        ;; SYMBOLIC axis bounds must be DECLARED as int params, exactly as the segmented-reduce
        ;; sibling does — the decompose above emits their names, so without this the kernel
        ;; references undeclared identifiers and fails to compile.
        scl-params (vec (sort-by name (:scalars segmap)))
        scl-param-str (str/join "" (map (fn [s] (str ", int " (ce/c-symbol s))) scl-params))
        src (str (codegen/extension-pragmas dtype)
                 "__kernel void " kernel-name "(" arr-param-str ", __global " ctype "* restrict out"
                 scl-param-str ", int _nseg) {\n"
                 "    int seg = get_global_id(0);\n"
                 "    if (seg >= _nseg) return;\n"
                 decomp "\n"
                 "    out[seg] = " body-str ";\n"
                 "}\n")]
    {:kernel-name kernel-name :source src :array-params arr-params
     :scalar-params scl-params
     :abi (kabi/validate!
           (vec (concat
                 (map #(kabi/slot % :input dtype :c-name (ce/c-symbol %) :role :operand) arr-params)
                 [(kabi/slot out-sym :output dtype :c-name "out" :role :result)]
                 (map #(kabi/slot % :scalar :int :c-name (ce/c-symbol %) :role :parameter) scl-params)
                 [(kabi/slot '_nseg :scalar :int :role :bound)])))
     :dtype dtype}))

;; ================================================================
;; Block-tiled + __local-staged contraction (BlkRegTiling, block-tile level)
;; ================================================================

(defn- syms-in [expr] (set (filter symbol? (tree-seq coll? seq expr))))

;; Compatibility DPAS source gates still consume this structural analysis while their callers
;; migrate to contraction facts and scheduled KernelBody values.
(declare analyze-contraction)

(defn- analyze-contraction
  "Shared structural analysis for the tiled contraction emitters. Prototype scope: 2 free
   axes + 1 contract, LITERAL dims, sum-of-two-agets element. Returns dims (M N L), axis
   syms, dtype/ctype, init, array params, and the row/col operand LOAD strings (row uses C
   vars i-sym,l-sym; col uses l-sym,j-sym — the caller declares them with the right values).
   Operands are assigned by DECLARED-axis dependence (no recognition). (generate-tiled-
   contraction-kernel predates this and still inlines the same logic — dedup TODO.)"
  [segred dtype]
  (let [space (:space segred)
        seg-dims (segop/seg-space-segment-dims space)
        red-dim  (segop/seg-space-reduced-dim space)
        ;; Structural preconditions of the TENSORIZE leaves. These are ex-info (not assert) so
        ;; the legality gate catches them and the router FALLS BACK to the general naive leaf,
        ;; instead of an AssertionError escaping and hard-failing a legal contraction.
        _ (when-not (= 2 (count seg-dims))
            (throw (ex-info "tensorize: needs exactly 2 free axes" {:reason :not-2-free :n-free (count seg-dims)})))
        [fi fj] seg-dims
        M (:bound fi) N (:bound fj) L (:bound red-dim)
        _ (when-not (every? number? [M N L])
            (throw (ex-info "tensorize: needs literal dims" {:reason :symbolic-dims :dims [M N L]})))
        i-sym (:name fi) j-sym (:name fj) l-sym (:name red-dim)
        dtype (or (:dtype segred) dtype)
        ctype (dt/ctype :opencl dtype)
        {:keys [init lambda]} (segop/scalar-reduce-op segred)
        lambda (ce/normalize-array-prims lambda)
        _ (when-not (#{'+ 'clojure.core/+ 'raster.numeric/+} (descriptor/semantic-op lambda))
            (throw (ex-info "tensorize: combine must be +" {:reason :non-plus-combine :op (descriptor/semantic-op lambda)})))
        acc-sym (:acc (segop/scalar-reduce-op segred))
        acc-at? (fn [a] (or (= a acc-sym) (and (seq? a) (= 'double (first a)) (= acc-sym (second a)))))
        op-args (vec (descriptor/call-args lambda))
        elem (let [a0 (nth op-args 0) a1 (nth op-args 1)] (if (acc-at? a0) a1 a0))
        _ (when-not (and (seq? elem) (#{'* 'clojure.core/* 'raster.numeric/*} (descriptor/semantic-op elem)))
            (throw (ex-info "tensorize: element must be a product of two agets" {:reason :non-product-element})))
        ;; Registry-classified, not `(= 'aget (first e))`: the walker emits `clojure.core/aget`, so
        ;; the literal match threw here for EVERY compiled deftm — and `route-2free-1contract`'s
        ;; `(catch ExceptionInfo _ nil)` swallowed the message, silently demoting a canonical matmul
        ;; to :naive-segred. The throw itself is right; what was wrong was which forms reached it.
        parts (fn [e] (let [e (ce/normalize-array-prims e)]
                        (when-not (descriptor/aget-call? e)
                          (throw (ex-info "tensorize: operand must be an aget"
                                          {:reason :non-aget-operand :operand e})))
                        {:arr (descriptor/aget-array-sym e)
                         ;; canonicalized so the same contraction emits one kernel text whichever
                         ;; way its body was spelled — the walked dialect's `(long 640)` extents
                         ;; would otherwise survive into the C for verbatim-index leaves
                         :idx (descriptor/canonicalize-index (descriptor/aget-index e))}))
        [pa pb] (mapv parts (descriptor/call-args elem))
        dep? (fn [idx s] (contains? (syms-in idx) s))
        rc (fn [x y] (when (and (dep? (:idx x) i-sym) (dep? (:idx x) l-sym) (not (dep? (:idx x) j-sym))
                                (dep? (:idx y) l-sym) (dep? (:idx y) j-sym) (not (dep? (:idx y) i-sym)))
                       [x y]))
        [rowop colop] (or (rc pa pb) (rc pb pa)
                          (throw (ex-info "tiled: operands don't match A(i,l)·B(l,j) variance" {:pa pa :pb pb})))
        arr-params (vec (sort-by name (:inputs segred)))
        arr-sym-set (set (map #(symbol (name %)) arr-params))
        dummy (gensym "z__")
        emit-load (fn [{:keys [arr idx]}]
                    (binding [ce/*emit-config* ce/opencl-config
                              ce/*scalar-type* ctype
                              ce/*int-vars* (into ce/*int-vars* (map #(symbol (name %)) [i-sym j-sym l-sym]))]
                      (ce/emit-expr (list 'aget arr idx) dummy arr-sym-set)))]
    {:M M :N N :L L :i-sym i-sym :j-sym j-sym :l-sym l-sym
     :dtype dtype :ctype ctype :init init :arr-params arr-params
     :row-load (emit-load rowop) :col-load (emit-load colop)
     ;; operand arrays + index exprs (for orientation analysis, e.g. DPAS tensorize):
     ;; rowop is the A(i,l) operand (→ A slot), colop the B(l,j) operand (→ B slot).
     :row-arr (:arr rowop) :row-idx (:idx rowop)
     :col-arr (:arr colop) :col-idx (:idx colop)}))

(defn generate-register-tiled-kernel-body
  "Emit the verified cooperative register-tiled schedule through the shared C-family backend.

   It consumes contraction facts, preserves a typed result transform and derives its ABI from
   KernelBody; no target source is reconstructed from a compatibility SegRed."
  [contract-facts out-sym & {:keys [tile descriptor operation-id]}]
  (let [{:keys [kernel-body bindings dims tile]}
        (register-tiled-body/lower
         contract-facts (cond-> {:operation-id operation-id :descriptor descriptor}
                          tile (assoc :tile tile)))
        row (:row bindings)
        col (:col bindings)
        kernel-name (str "regtiled_contract_" (gensym ""))
        parameter-names (merge (into {} (map (fn [parameter]
                                               [(:id parameter)
                                                (ce/c-symbol (:id parameter))]))
                                     (:parameters kernel-body))
                               {row "A" col "B" out-sym "out"})
        source (kernel-body-opencl/emit-scalar-kernel
                kernel-name kernel-body {:parameter-names parameter-names})
        abi
        (body-abi/project-contracts
         (mapv (fn [{:keys [id kind dtype role]}]
                 (kabi/slot id kind dtype
                            :c-name (or (get parameter-names id) (ce/c-symbol id))
                            :role (if (contains? #{:lhs :rhs} role) :operand role)))
               (:parameters kernel-body))
         kernel-body)
        epilogue-slots (filterv #(= :epilogue (:role %)) abi)]
    {:kernel-name kernel-name
     :source source
     :array-params [row col]
     :abi abi
     :dtype (:dtype contract-facts)
     :block [(:block-m tile) (:block-n tile) (:block-k tile)]
     :micro [(:thread-m tile) (:thread-n tile)]
     :workgroup (get-in kernel-body [:launch :workgroup-size])
     :dims dims
     :kernel-body kernel-body
     :epilogue-operands (mapv :name (filter #(= :input (:kind %)) epilogue-slots))
     :epilogue-scalars (mapv :name (filter #(= :scalar (:kind %)) epilogue-slots))}))

;; ================================================================
;; DPAS/XMX-tensorized contraction (PEAK) — raster's edge over Futhark
;; ================================================================

(defn- canonical-rowmajor?
  "Does affine index `idx` equal `(+ (* outer stride) inner)` (row-major, leading dim `stride`)?

   ONE RELATION. This used to be a hand-rolled pattern match that was +/*-order-agnostic but could
   only see a literal 3-element sum, while `am/index-matches?` flattened nested sums but compared
   terms positionally. The two accepted DIFFERENT languages, so a legal :nt operand written
   `(+ l (* j K))` passed one gate and was rejected by the other — reaching no leaf at all. Both
   now normalize to the same affine form, which accepts every arithmetically-identical spelling."
  [idx outer stride inner]
  (am/index-matches? (am/of-axes [[outer 1] [inner stride]])
                     (ce/normalize-array-prims idx)))

(defn dpas-contraction-legal?
  "The tensorize LEGALITY GATE: is `segred` a contraction that lowers to the DPAS/XMX
   matmul body? This is where the recognizer's affine-invariance core rehomes — but stated
   as a *legality check on already-declared axes*, not a recognition. Returns {:ok true …}
   with the extracted operands/dims, or {:ok false :reason kw} so the caller falls back to
   the portable register-tiled kernel.

   Conditions (Arc DPAS): exactly 2 free + 1 contract axis, element = product of two agets,
   BOTH operands in canonical row-major orientation (A[i,l]=i·L+l ⇒ [M,L]; B[l,j]=l·N+j ⇒
   [L,N]) so the golden 2D-block reads address them correctly, dtype ∈ DPAS types (Arc:
   half; bf16/int8 are future variants), and the operand PITCHES are 16-byte-aligned —
   N·2 and K·2 multiples of 16 ⇔ N%8==0 and K%8==0 for f16. The pitch condition is a HARD
   hardware constraint of intel_sub_group_2d_block_read: a mis-aligned pitch (e.g. N=70,
   K=124) SILENTLY MISCOMPILES ~80% of outputs (device-verified) — production GEMM shapes
   (N∈{640,1024,2048}) are all N%8==0 so never hit it, but a general contraction can, and
   the gate must reject it so the caller falls back to the register-tiled kernel (which
   handles arbitrary dims). M (the block-read HEIGHT) is unconstrained. Transposed operands
   (:tn/:nt) and a batch axis are legal extensions the golden body already has flags for —
   rejected here as :non-canonical-orientation / :not-a-contraction until wired."
  [segred dtype]
  (let [dtype (or (:dtype segred) dtype)
        pitch-ok? (fn [d] (and (number? d) (zero? (mod (long d) 8))))]
    (if-not (and (dt/known? dtype) (= :half (dt/canon dtype)))
      {:ok false :reason :dtype-not-dpas :dtype dtype}
      (try
        (let [{:keys [M N L i-sym j-sym l-sym row-idx col-idx row-arr col-arr arr-params]}
              (analyze-contraction segred dtype)]
          (cond
            (not (canonical-rowmajor? row-idx i-sym L l-sym))
            {:ok false :reason :non-canonical-orientation :operand :row :idx row-idx}
            (not (canonical-rowmajor? col-idx l-sym N j-sym))
            {:ok false :reason :non-canonical-orientation :operand :col :idx col-idx}
            (not (pitch-ok? N))   ; B pitch = N·2 bytes must be 16-byte aligned
            {:ok false :reason :n-pitch-unaligned :N N}
            (not (pitch-ok? L))   ; A pitch = K·2 bytes must be 16-byte aligned
            {:ok false :reason :k-pitch-unaligned :L L}
            :else
            ;; i-sym/j-sym are the FREE axes; an epilogue binds them to the store slot's row/col
            {:ok true :M M :N N :L L :i-sym i-sym :j-sym j-sym
             :row-arr row-arr :col-arr col-arr :arr-params arr-params}))
        (catch clojure.lang.ExceptionInfo e
          {:ok false :reason :not-a-contraction :msg (.getMessage e)})))))

(defn epilogue-legal?
  "Legality of splicing `expr` into the contraction's STORE slot. Returns {:ok true} or
   {:ok false :reason kw}. Distilled from Halide / XLA / Triton / MLIR-linalg:

     • the epilogue is emitted strictly AFTER the reduction loop closes. Placing it inside forces
       accumulator multi-buffering under software pipelining — a silent 2x on the most expensive
       resource. Our splice is in the store slot, so this holds by construction; it is asserted
       here so a future change cannot quietly break it.
     • no op that forces a layout/distribution change: scan, non-associative reduce, permuting
       reshape, atomic. A reduction in the epilogue is a re-tiling decision (redistribute the
       accumulator so the reduce is warp-local) — legal in principle, unimplemented, so refused
       with :reduction-in-epilogue rather than silently miscompiled.
     • the accumulator must appear exactly ONCE: more than one use duplicates the reduction result
       through the epilogue expression, which is the recompute case the ceilings guard against."
  [epilogue]
  (kbody/scalar-region-legal? epilogue))

(defn epilogue-cost
  "Byte-traffic model for splicing an epilogue into the contraction's store, and the register
   pressure it adds. Returns {:fused-bytes :unfused-bytes :saved-bytes :profitable :acc-regs}.

   NOTE ON A TEST THAT DOES NOT TRANSFER. Halide's `is_func_trivial_to_inline`
   (1 + sizeof(out) >= arith + bytes) is a PRODUCER-INLINING test: should I duplicate a producer's
   work at each consumer, paying recompute? Applied to an epilogue it gives the wrong answer — it
   scores a bias-add as unprofitable (call 3 vs inline 6 at f16) because it charges the bias LOAD
   without crediting the eliminated round-trip. Epilogue (output) fusion is the opposite direction:
   it REMOVES traffic rather than duplicating work.

       unfused:  write C  +  read C  +  read operands  +  write C   ~ 3·M·N + operands
       fused:    read operands                                     ~ operands

   So an epilogue is profitable whenever it is legal and fits in registers — which is why Triton
   and XLA fuse epilogues unconditionally and gate on legality + register pressure instead. The
   recompute ceilings (8x CPU / 10x GPU) and the multi-consumer refusal belong to the PRODUCER
   direction; the analogue here is `epilogue-legal?`'s accumulator-used-once rule.

   Register estimate follows Triton's closed form: elems-per-thread × threads-per-warp ×
   warps-per-CTA × elem-bytes / 4 registers for the accumulator; the epilogue's live values must
   fit in the remaining budget (Triton clamps accumulator traffic at maxnreg/2 to avoid spilling)."
  [{:keys [operands]} out-dtype [M N] tile]
  (let [bytes-of (fn [d] (long (dt/bytes-of d)))   ; one registry, not a private table
        cb (bytes-of out-dtype)
        c-elems (* (long M) (long N))
        operand-bytes (reduce + 0 (for [{:keys [dtype] :or {dtype :float}} operands]
                                    (* (long N) (bytes-of dtype))))
        unfused (+ (* 3 c-elems cb) operand-bytes)   ; write + read + write, plus operand reads
        fused   operand-bytes
        ;; accumulator registers per lane: (block-m·block-n / (subgroups·subgroup)) f32 values
        {:keys [block-m block-n sg-m sg-n]} tile
        sg (long (get-in tile [:matrix :subgroup] 16))
        acc-regs (when (and block-m block-n sg-m sg-n)
                   (quot (* (long sg-m) (long sg-n)) sg))]
    {:fused-bytes fused :unfused-bytes unfused
     :saved-bytes (- unfused fused)
     :profitable (< fused unfused)
     :acc-regs acc-regs}))

(defn epilogue-splice
  "Build the OpenCL scalar-expression spelling consumed by a matrix store region from a
   DOMAIN-AGNOSTIC spec. The compiler learns no op names: the spec is
   just an EXPRESSION over the accumulator and some operands, each operand carrying its own
   axis-map, so bias / activation / residual / dequant-scale are all the same mechanism and
   COMPOSE by nesting (one bigger expression), per CLAUDE.md's domain-agnostic-passes rule.

     {:acc  acc            ;; symbol standing for the contraction's accumulator
      :expr <s-expr>       ;; e.g. (raster.numeric/* (raster.numeric/+ acc (aget bias j)) s)
      :operands [{:sym bias :map <axis-map over the FREE axes> :dtype :float}]
      :scalars  [{:sym s :dtype :float}]}      ;; optional uniform scalars

   `free-syms` are the contraction's two free-axis symbols, bound to the store slot's `row`/`col`
   C variables — so an operand's axis-map generates its index exactly as in the kernel body.
   Returns {:epilogue fn :epilogue-params str}. Raw target helper source is not an IR."
  [{:keys [acc expr operands scalars helpers] :as spec} [i-sym j-sym] dtype]
  (when (seq helpers)
    (throw (ex-info "epilogue helper source has no typed scalar-region representation"
                    {:reason :opaque-epilogue-helper})))
  (let [legal (epilogue-legal? spec)
        _ (when-not (:ok legal)
            (throw (ex-info (str "epilogue-splice: illegal epilogue (" (:reason legal) ")")
                            (assoc legal :expr expr))))
        ctype (dt/ctype :opencl dtype)
        arr-syms (set (map (comp #(symbol (name %)) :sym) operands))
        int-vars (into #{} (map #(symbol (name %))) [i-sym j-sym])
        ;; substitute each operand's aget index from its declared map (maps, not pattern-matching)
        idx-of (into {} (for [{:keys [sym map]} operands] [sym (am/index-expr map)]))
        expr' (descriptor/rewrite-aget-indices expr idx-of)
        acc-token (str "__acc_" (name (gensym "")))
        ;; emit once with a distinctive token standing in for the accumulator, then splice the
        ;; real acc C-expression in at call time (the hook supplies it per store slot)
        emitted (binding [ce/*emit-config* ce/opencl-config
                          ce/*scalar-type* ctype
                          ce/*int-vars* (into ce/*int-vars* int-vars)]
                  (ce/emit-expr (walk/postwalk-replace {acc (symbol acc-token)} expr')
                                (gensym "z__") arr-syms))
        params (apply str
                      (concat
                       (for [{:keys [sym dtype] :or {dtype :float}} operands]
                         (str ", __global const " (dt/ctype :opencl dtype)
                              "* restrict " (ce/c-symbol sym)))
                       (for [{:keys [sym dtype] :or {dtype :float}} scalars]
                         (str ", " (dt/ctype :opencl dtype) " " (ce/c-symbol sym)))))]
    {:epilogue (fn [acc-expr row col]
                 (-> emitted
                     (str/replace acc-token (str "(" acc-expr ")"))
                     (str/replace (re-pattern (str "\\b" (ce/c-symbol i-sym) "\\b")) row)
                     (str/replace (re-pattern (str "\\b" (ce/c-symbol j-sym) "\\b")) col)))
     :epilogue-params params}))

(defn- emit-dpas-plan
  "Lower one already-checked DPAS plan. A KernelBody takes the direct operation lowerer; nil is
  reserved for the independent source oracle."
  [kernel-name row-arr col-arr out-sym [M N L] [i-sym j-sym] tile epilogue kernel-body]
  (let [effective-tile (if kernel-body (:schedule kernel-body) tile)
        result-dtype (if kernel-body
                       (:dtype (first (filter #(= :result (:role %))
                                              (:parameters kernel-body))))
                       :half)
        sg (long (get-in effective-tile [:matrix :subgroup] 16))
        ep (if kernel-body
             (kernel-body-opencl/lower-store-region kernel-body)
             (when epilogue
               (epilogue-splice epilogue [i-sym j-sym] (get epilogue :dtype :float))))
        effective-epilogue (or epilogue (get-in kernel-body [:attributes :epilogue]))
        matrix-abi
        (when kernel-body
          (let [dimensions (filterv #(= :dimension (:role %)) (:parameters kernel-body))
                dimension-names (zipmap (map :id dimensions) ["M" "N" "K"])
                c-name (fn [{:keys [id role]}]
                         (case role
                           :lhs "A"
                           :rhs "B"
                           :result "C"
                           (or (get dimension-names id) (ce/c-symbol id))))]
            (body-abi/project-contracts
             (mapv (fn [{:keys [id kind dtype role] :as parameter}]
                     (kabi/slot id kind dtype
                                :c-name (c-name parameter)
                                :role (if (contains? #{:lhs :rhs} role) :operand role)))
                   (:parameters kernel-body))
             kernel-body)))
        source (if kernel-body
                 (kernel-body-opencl/emit-matrix-kernel kernel-name kernel-body)
                 (apply codegen/emit-gemm-tiled kernel-name
                        (concat [:c-dtype :half
                                 :block-m (:block-m effective-tile)
                                 :block-n (:block-n effective-tile)
                                 :sg-m (:sg-m effective-tile) :sg-n (:sg-n effective-tile)
                                 :block-k (:block-k effective-tile) :matrix (:matrix effective-tile)
                                 :prefetch (:num-stages effective-tile 3)]
                                (when ep [:epilogue (:epilogue ep)
                                          :epilogue-params (:epilogue-params ep)]))))]
    (cond->
     {:kernel-name kernel-name
      :source source
      :array-params [row-arr col-arr]
      :abi (or matrix-abi
               (kabi/validate!
                (vec (concat
                      [(kabi/slot row-arr :input :half :c-name "A" :role :operand)
                       (kabi/slot col-arr :input :half :c-name "B" :role :operand)
                       (kabi/slot out-sym :output result-dtype :c-name "C" :role :result)
                       (kabi/slot 'M :scalar :int :role :dimension)
                       (kabi/slot 'N :scalar :int :role :dimension)
                       (kabi/slot 'K :scalar :int :role :dimension)]
                      (for [{:keys [sym dtype] :or {dtype :float}} (:operands effective-epilogue)]
                        (kabi/slot sym :input dtype :c-name (ce/c-symbol sym) :role :epilogue))
                      (for [{:keys [sym dtype] :or {dtype :float}} (:scalars effective-epilogue)]
                        (kabi/slot sym :scalar dtype :c-name (ce/c-symbol sym) :role :epilogue))))))
      :dims [M N L]
      :dtype result-dtype
      :tile effective-tile
      :epilogue-params (when ep (:epilogue-params ep))
      :epilogue-operands (when ep (mapv :sym (:operands effective-epilogue)))
      :epilogue-scalars (when ep (mapv :sym (:scalars effective-epilogue)))
      :workgroup (if kernel-body
                   (get-in kernel-body [:launch :workgroup-size])
                   [(* (quot (:block-m effective-tile) (:sg-m effective-tile))
                       (quot (:block-n effective-tile) (:sg-n effective-tile))
                       sg) 1])
      :tensorized true}
      kernel-body (assoc :kernel-body kernel-body))))

(defn generate-dpas-kernel-body
  "Lower a verified, scheduled matrix KernelBody to the Intel OpenCL target.

   This is deliberately a separate boundary from contraction scheduling.  The production path
   walks explicit body operations and layouts directly.  The legacy source generator remains only
   behind `generate-dpas-contraction-kernel` as an independent equivalence oracle."
  [kernel-body out-sym]
  (let [kernel-body (kbody/validate! kernel-body)
        {:keys [kind instruction-family dims bindings epilogue axis-symbols]}
        (:attributes kernel-body)]
    (when-not (and (= :matrix-contraction kind) (= :dpas instruction-family))
      (throw (ex-info "OpenCL DPAS lowering requires a DPAS matrix KernelBody"
                      {:kind kind :instruction-family instruction-family})))
    (emit-dpas-plan (str "dpas_contract_" (gensym ""))
                    (:row bindings) (:col bindings) out-sym dims
                    (subvec (vec axis-symbols) 0 2)
                    (:schedule kernel-body) epilogue kernel-body)))

(defn generate-dpas-contraction-kernel
  "Legacy direct entry to the proven DPAS/XMX OpenCL emitter.

   Production canonical f16 contractions now travel through ContractionFacts and a scheduled
   KernelBody before reaching `generate-dpas-kernel-body`. This entry remains only as the
   independent source oracle for the direct operation lowerer.
   Its legality analysis determines operand orientation and launch dimensions; the emitted body is
   f16 input, f32 accumulation and f16 output with a tile-parametric K16 matrix instruction.

   Returns {:kernel-name :source :array-params [row-arr col-arr] :dims [M N L] :dtype :half
            :tensorized true}  — NB: :array-params is in [row col] BINDING order (row's
   buffer → A slot, col's → B slot, out → C), NOT sorted-by-name. Returns
   {:tensorized false :reason …} when the gate rejects (caller falls back to regtiled)."
  [segred out-sym & {:keys [dtype desc tile epilogue] :or {dtype :half}}]
  (let [gate (dpas-contraction-legal? segred dtype)]
    (if-not (:ok gate)
      {:tensorized false :reason (:reason gate) :detail gate}
      (let [{:keys [M N L row-arr col-arr]} gate
            kernel-name (str "dpas_contract_" (gensym ""))
            ;; TILE GEOMETRY IS DERIVED FROM THE HARDWARE DESCRIPTOR, never hardcoded: the
            ;; per-subgroup accumulator tile is GRF-bound and rounded to the matrix (DPAS)
            ;; fragment granularity, so a part with a different GRF budget / subgroup size /
            ;; matrix shape gets a correctly rescaled tile from the same rule. An explicit
            ;; `tile` (e.g. an autotune result via hw/gemm-tile-candidates) overrides.
            ;; hw/derive-gemm-tile's own defaults reproduce the Arc 140V config, so passing no
            ;; descriptor is equivalent to the previous literal — with zero magic numbers here.
            tile (or tile (hw/derive-gemm-tile (or desc {})))]
        (emit-dpas-plan kernel-name row-arr col-arr out-sym [M N L]
                        [(:i-sym gate) (:j-sym gate)] tile epilogue nil)))))

;; ================================================================
;; QUANT (int8) contraction — the SAME skeleton, WIDENING facet
;; ================================================================

(defn- flat-decompose-c
  "C declarations recovering each free index from the flat segment id `seg`, row-major:
   idx_p = (seg / Π bounds-after-p) % bound_p, with the innermost simplifying to `seg % bound`.
   ONE source for the row-major decompose. (Debt: three older copies of this arithmetic remain
   inline — generate-segmented-reduce-kernel, generate-segmap-nd-kernel, and the two quant
   kernels — and should collapse onto this helper; not done here to keep their emitted text
   provably byte-identical.)"
  [free-axes]
  (let [v (vec free-axes) n (count v)]
    (str/join "\n"
              (map-indexed
               (fn [p [sym bound]]
                 ;; bounds go through c-symbol too: a symbolic bound named `n-cols` was emitted
                 ;; raw as `(seg / (n-cols))` — valid C that computes `n - cols`.
                 (let [bc (fn [b] (if (symbol? b) (ce/c-symbol b) b))
                       after (map (comp bc second) (subvec v (inc p) n))
                       div (if (seq after) (str "(seg / (" (str/join " * " after) "))") "seg")]
                   (str "    int " (ce/c-symbol sym) " = " div " % " (bc bound) ";")))
               v))))

(defn staged-inner-dp4a-legal?
  "Can the INNERMOST stage of a staged contraction be tensorized with dp4a (4 int8 MACs into an
   int32 in one op)? Returns {:ok true :packed-maps {sym amap} :packed-extent n} or
   {:ok false :reason kw}.

   This is the int8 PEAK leaf for block-quant, and it is the same structure llama.cpp hand-writes:
   the inner stage is already an exact int32 accumulation over a short K-contiguous run, which is
   precisely dp4a's shape. Because the stage list says which axis the inner accumulation runs over,
   there is nothing to recognize — the gate only has to CHECK.

   Required, and the failure each check prevents:
     • declared operand axis-maps. Tensorizing needs to know each operand's innermost axis; per the
       compiler's declare-don't-pattern-match rule that is data, not inference.
     • each map must VERIFIABLY equal the operand's actual index in the body (am/index-matches?).
       Assuming a layout while having checked only the axis symbols is how a transpose rewrite
       silently miscompiled before; a leaf may only assume what it has proved.
     • the inner stage's axis must be the INNERMOST axis of both maps — dp4a packs 4 consecutive
       elements along the contraction, so both operands must be contiguous in it (the :nt layout).
     • int8 operands, integral inner accumulator (widening as a dtype pair).
     • the inner extent must be a literal multiple of 4, else the packed load is mis-strided."
  [{:keys [stages body operands dtype]}]
  (let [inner-stage (peek (vec stages))
        int-acc? (contains? #{:int :long :int32} (:dtype inner-stage))
        agets (into {} (map (juxt :sym :idx)) (descriptor/aget-reads body))
        ;; THE BODY IS DISCARDED when this leaf fires — the whole summand is replaced by one
        ;; rstr_dp4a call — so the gate must account for EVERY term first. The requirement lives in
        ;; ir/contraction-facts as `body-product-of`, shared with every other body-replacing leaf,
        ;; rather than re-derived per gate.
        exact-product? (some? (cf/body-product-of body (map :sym operands)))]
    (cond
      (not= 2 (count operands)) {:ok false :reason :dp4a-needs-two-declared-operands}
      (not (every? :map operands)) {:ok false :reason :operand-without-a-declared-map}
      (not (and (dt/known? dtype) (= :byte (dt/canon dtype))))
      {:ok false :reason :dp4a-needs-int8-operands :dtype dtype}
      ;; A :decode is a LOAD-LAMBDA applied by substituting into the body — and this leaf discards
      ;; the body, replacing the whole summand with one rstr_dp4a call. So a zero-point would be
      ;; silently dropped: Σ a·b instead of Σ(a−za)(b−zb), wrong by a constant-plus-linear term with
      ;; no diagnostic. Load-bearing, since q4_0/q8_0 carry zero-points 8 and 128. Same underlying
      ;; reason as :body-has-unmodeled-terms below — the body is not evaluated here.
      (some :decode operands)
      {:ok false :reason :decode-on-a-body-replacing-leaf
       :detail "this leaf replaces the body with a single hardware op, so a per-operand :decode (e.g. a zero-point) would be silently dropped"
       :decoded (mapv :sym (filter :decode operands))}

      (not exact-product?)
      {:ok false :reason :body-has-unmodeled-terms
       :detail "this leaf replaces the body with a single hardware op; any term beyond the two declared operands would be silently dropped"
       :body body :declared (mapv :sym operands)}
      (not int-acc?) {:ok false :reason :inner-stage-accumulator-not-integral
                      :dtype (:dtype inner-stage)}
      :else
      (or
       ;; the declared map must PROVABLY be the operand's actual index expression
       (first (keep (fn [{:keys [sym map]}]
                      (let [idx (get agets sym)]
                        (cond
                          (nil? idx) {:ok false :reason :declared-operand-not-read-by-the-body :sym sym}
                          (not (am/index-matches? map idx))
                          {:ok false :reason :declared-map-does-not-match-the-body-index
                           :sym sym :declared (am/index-expr map) :actual idx}
                          (not= (:axis inner-stage) (am/innermost-axis map))
                          {:ok false :reason :inner-stage-axis-is-not-contiguous
                           :sym sym :innermost (am/innermost-axis map) :axis (:axis inner-stage)}
                          :else nil)))
                    operands))
       (let [p-sym (gensym "p__")
             packed (into {} (for [{:keys [sym map]} operands]
                               [sym (am/pack-innermost map 4 p-sym)]))]
         (if (some nil? (vals packed))
           {:ok false :reason :inner-extent-not-a-multiple-of-4 :extent (:extent inner-stage)}
           {:ok true :packed-maps packed :p-sym p-sym
            :packed-extent (quot (long (:extent inner-stage)) 4)}))))))

(defn generate-staged-contraction-kernel
  "STAGED contraction → OpenCL: a reduction accumulating in N levels, each with its own
   accumulator dtype, with a `lift` splicing each level's partial sum into the level above.
   This is the shape every block-quantized format needs and the flat leaves cannot express —
   an int32 MAC inside the block, a float accumulate across blocks (see ir/contract-stages).
   2 stages = q8_0/q4_0, 3 = k-quants, 1 = the flat contraction (so this emitter subsumes it).

   The emitted nest is exactly the schedule `backend/cpu/quant.clj` already uses on CPU and
   llama.cpp hand-writes per format — here it comes from the stage list, not from a kernel per
   format. Domain-agnostic: no scale/zero-point/format concept appears below; a stage is an
   accumulator dtype plus a lift expression, and the lift's operand arrays are indexed by their
   DECLARED axis-maps (never by inferred strides).

   Spec:
     {:free-axes [[i M] [j N]]      ;; output axes, outer→inner (any rank ≥ 1)
      :stages    [outer … inner]    ;; see ir/contract-stages for the stage shape
      :body      <expr>             ;; the summand, over the free + stage axes
      :inputs    [a b]              ;; the body's operand arrays
      :dtype     :byte              ;; the body operands' element dtype
      :out-dtype :float}

   Returns {:kernel-name :source :array-params :dtype :out-dtype :dims :stages :out-elems
            :lift-operands}. :array-params is the body's inputs; :lift-operands are the EXTRA
   scale arrays, bound after them — surfaced because omitting them from a launch descriptor is
   an arity bug (the signature has them either way)."
  [{:keys [free-axes stages body inputs dtype out-dtype operands tensorize-inner? contract-axes
           epilogue]
    :or {dtype :float out-dtype :float} :as spec} out-sym]
  (let [;; The stage list is checked against the axes the FORM DECLARED, never against axes derived
        ;; from the stages themselves. Deriving them made the span rule unfireable, and a stage list
        ;; that under-covers the contract space emitted a kernel summing a FRACTION of the terms
        ;; while par/contract's CPU path summed all of them — a silent divergence between two
        ;; consumers of one form. A gate must not validate its own arguments.
        _ (when (nil? contract-axes)
            (throw (ex-info "staged contraction: :contract-axes is required — the stage list can only be validated against the axes the form declared"
                            {:reason :missing-declared-contract-axes :stages stages})))
        ;; This emitter interpolates extents into the loop bounds and the decompose but declares no
        ;; scalar params for them, so a symbolic bound emits an UNDECLARED identifier — and
        ;; validate-descriptor passes it (the counts agree), so it fails only at device build, i.e.
        ;; never in CI. Its segmented-reduce sibling declares them; until this one does, refuse.
        _ (let [syms (remove number? (concat (map second free-axes) (map :extent stages)))]
            (when (seq syms)
              (throw (ex-info (str "staged contraction: symbolic bounds are not supported by this "
                                   "emitter (it declares no scalar params for them) — "
                                   (pr-str (vec syms)))
                              {:reason :symbolic-bounds-unsupported :bounds (vec syms)}))))
        legal (cstage/stages-legal? stages contract-axes)
        _ (when-not (:ok legal)
            (throw (ex-info (str "staged contraction: illegal stages (" (:reason legal) ")")
                            (assoc legal :stages stages :contract-axes contract-axes))))
        stages (vec stages)
        ;; TENSORIZE THE INNER STAGE: the innermost accumulation is already an exact int32 sum over
        ;; a short K-contiguous run, which is exactly dp4a's shape. Requested explicitly (a schedule
        ;; choice), then GATED — a rejection falls back to the scalar nest, never to a wrong kernel.
        tz (when tensorize-inner?
             (let [g (staged-inner-dp4a-legal? spec)]
               (when-not (:ok g)
                 (throw (ex-info (str "staged contraction: cannot tensorize inner stage ("
                                      (:reason g) ")") g)))
               g))
        ;; packed operands are READ as int32 words (4 int8 each); the buffer bytes are unchanged
        op-ctype  (get codegen/opencl-type-map (if tz :int dtype) "float")
        out-ctype (dt/ctype :opencl out-dtype)
        lift-ops (cstage/lift-operands stages)
        idx-of (cstage/stage-index-exprs stages)
        ;; every axis in scope is an int loop/decompose variable
        int-vars (into #{} (map (comp symbol name))
                       (concat (map first free-axes) (map :axis stages)
                               (when tz [(:p-sym tz)])))
        arr-syms (into #{} (map (comp symbol name)) (concat inputs (map :sym lift-ops)))
        emit (fn [expr]
               (binding [ce/*emit-config* ce/opencl-config
                         ce/*scalar-type* out-ctype
                         ce/*int-vars* (into ce/*int-vars* int-vars)]
                 (ce/emit-expr expr (gensym "z__") arr-syms)))
        acc-name (fn [d] (str "acc_" d))
        ;; PER-OPERAND DECODE — the load-lambda. `:decode` is an expression in `x`, standing for the
        ;; raw load, applied to that operand's every read. This is where a zero-point subtraction
        ;; belongs: `Σ(a-za)(b-zb)` is exact on the load path and needs no correction reductions,
        ;; whereas a per-tensor SCALE factors out of the sum entirely and belongs in the epilogue.
        ;; WIDENING is deliberately NOT expressed here — it is the dtype PAIR (operand dtype +
        ;; accumulator dtype), and hiding it in a lambda would blind the tensorize gate to the very
        ;; pair it dispatches on.
        decodes (into {} (keep (fn [{:keys [sym decode]}] (when decode [sym decode]))) operands)
        ;; `x` in a :decode lambda stands for the RAW LOAD. Substituted capture-avoidingly, and the
        ;; read is matched via the registry — the literal `(= 'aget (first f))` here meant a decode
        ;; was SILENTLY DROPPED for every walker-spelled operand, emitting `a[l]*b[l]` where the
        ;; semantics are `(a[l]-zp)*b[l]`. A wrong answer, not a slow one.
        body (if (empty? decodes)
               body
               (descriptor/rewrite-aget-reads
                body
                (fn [f] (let [arr (descriptor/aget-array-sym f)]
                          (when (contains? decodes arr)
                            (util/subst-syms {'x f} (get decodes arr)))))))
        ;; EPILOGUE — the store splice. An epilogue is a lift on a virtual outermost level of
        ;; extent 1, which is why it needs no linearity (nothing to distribute over one iteration)
        ;; while a real stage lift does. Gives this emitter the dequant scale that the two quant
        ;; leaves hardwired into their store lines.
        ep (when epilogue
             (when-not (= 2 (count free-axes))
               (throw (ex-info "staged contraction: an epilogue needs exactly 2 free axes (the store slot binds row/col)"
                               {:reason :epilogue-needs-2-free :n-free (count free-axes)})))
             (epilogue-splice epilogue (mapv first free-axes) (or (:dtype epilogue) out-dtype)))
        ;; innermost accumulates the body; each outer accumulates its lift with `inner` bound to
        ;; the accumulator one level down. Built inside-out.
        n (count stages)
        ;; ONE description of the innermost loop, used at both nesting depths.
        inner-loop
        (fn [indent acc]
          (let [{:keys [dtype init axis extent]} (peek stages)
                t (dt/ctype :opencl dtype)
                [loop-var bound step]
                (if tz [(:p-sym tz) (:packed-extent tz) "dp4a"] [axis extent "scalar"])
                pad (apply str (repeat indent " "))]
            (str pad t " " acc " = " (or init 0) ";\n"
                 pad "for (int " (ce/c-symbol loop-var) " = 0; "
                 (ce/c-symbol loop-var) " < " bound "; " (ce/c-symbol loop-var) "++) {\n"
                 pad "    " acc
                 (if (= "dp4a" step)
                   ;; 4 int8 MACs into the int32 accumulator per op; indices come from the PACKED
                   ;; maps, so the stride rescaling is the axis-map algebra's, not hand-written.
                   (str " = rstr_dp4a("
                        (str/join ", " (for [{:keys [sym]} operands]
                                         (str (ce/c-symbol sym) "["
                                              (emit (am/index-expr (get (:packed-maps tz) sym)))
                                              "]")))
                        ", " acc ")")
                   (str " += " (emit body)))
                 ";\n"
                 pad "}\n")))
        inner-most (inner-loop 8 (acc-name (dec n)))
        nest (reduce
              (fn [inner-src d]
                (let [{:keys [dtype init axis extent lift]} (nth stages d)
                      t (dt/ctype :opencl dtype)
                      lift' (cstage/substitute-operand-indices lift idx-of)
                      ;; splice the level-below accumulator in for `inner`
                      lift'' (walk/postwalk-replace {'inner (symbol (acc-name (inc d)))} lift')]
                  (str "    " t " " (acc-name d) " = " (or init 0) ";\n"
                       "    for (int " (ce/c-symbol axis) " = 0; "
                       (ce/c-symbol axis) " < " extent "; " (ce/c-symbol axis) "++) {\n"
                       inner-src
                       "        " (acc-name d) " += " (emit lift'') ";\n"
                       "    }\n")))
              inner-most
              (reverse (range (dec n))))
        ;; a single stage has no lift, so its accumulator is the whole nest, unindented
        nest (if (= 1 n) (inner-loop 4 (acc-name 0)) nest)
        n-out (am/n-elements (am/of-axes (vec free-axes)))
        kernel-name (str "staged_contract_" (gensym ""))
        params (str (str/join ", " (for [a inputs]
                                     (str "__global const " op-ctype "* restrict " (ce/c-symbol a))))
                    (apply str (for [{:keys [sym dtype] :or {dtype :float}} lift-ops]
                                 (str ", __global const " (dt/ctype :opencl dtype)
                                      "* restrict " (ce/c-symbol sym))))
                    ", __global " out-ctype "* restrict out"
                    (when ep (:epilogue-params ep))
                    ", int _nseg")
        src (str
                 ;; fp64/fp16 need their OpenCL extension enabled. All three sibling emitters emit
                 ;; this; the staged one did not, so a :double staged contraction — reachable from
                 ;; opencl_pass, whose default dtype IS :double — emitted `double` with no pragma.
             (codegen/extension-pragmas out-dtype dtype)
                 ;; registry-driven, not `(intrinsics/descriptor 'dp4a)`: define every intrinsic
                 ;; helper this body actually calls — the same scan every other emitter uses
             (ce/helper-sources nest)
             "__kernel void " kernel-name "(" params ") {\n"
             "    int seg = get_global_id(0);\n"
             "    if (seg >= _nseg) return;\n"
             (flat-decompose-c free-axes) "\n"
             nest
             "    out[seg] = "
             (let [acc (str "(" out-ctype ")" (acc-name 0))]
               (if ep
                 ((:epilogue ep) acc
                                 (ce/c-symbol (first (first free-axes)))
                                 (ce/c-symbol (first (second free-axes))))
                 acc))
             ";\n"
             "}\n")]
    {:kernel-name kernel-name :source src
     :array-params (vec inputs)
     :abi (kabi/validate!
           (vec (concat
                 (for [a inputs]
                   (kabi/slot a :input dtype :c-name (ce/c-symbol a)
                              :kernel-dtype (if tz :int dtype)
                              :role :operand))
                 (for [{:keys [sym dtype] :or {dtype :float}} lift-ops]
                   (kabi/slot sym :input dtype :c-name (ce/c-symbol sym) :role :lift))
                 [(kabi/slot out-sym :output out-dtype :c-name "out" :role :result)]
                 (for [{:keys [sym dtype] :or {dtype :float}} (:operands epilogue)]
                   (kabi/slot sym :input dtype :c-name (ce/c-symbol sym) :role :epilogue))
                 (for [{:keys [sym dtype] :or {dtype :float}} (:scalars epilogue)]
                   (kabi/slot sym :scalar dtype :c-name (ce/c-symbol sym) :role :epilogue))
                 [(kabi/slot '_nseg :scalar :int :role :bound)])))
     :lift-operands (mapv :sym lift-ops)
     :epilogue-operands (when ep (mapv :sym (:operands epilogue)))
     :epilogue-scalars (when ep (mapv :sym (:scalars epilogue)))
     :dtype dtype :out-dtype out-dtype
     :stages stages
     ;; the operand buffers are BOUND unchanged; only the kernel's view of them widens
     :tensorized (boolean tz)
     :packed (when tz :int8x4)
     :out-elems n-out
     :dims (mapv second free-axes)}))
