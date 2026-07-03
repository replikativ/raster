(ns raster.compiler.core.dtype
  "Dtype-driven type tag remapping helpers for compiler lowering.")

;; ── Single faceted source of truth per numeric dtype ───────────────────────
;; Add a dtype = add ONE row here; backends and the typing seam derive their
;; per-dtype facts from this map instead of keeping private copies. Per-backend
;; native-name differences are LEGITIMATE facets, not inconsistencies: GLSL has
;; no 64-bit int (long→"int"), OpenCL `long` is 64-bit, C uses `long long`.
;;
;; Facets:
;;   :native       {backend → native type name} (:c / :opencl / :glsl)
;;   :scalar-tag   primitive dispatch tag for a scalar of this dtype, or nil
;;                 when the dtype has no JVM primitive (e.g. :half)
;;   :array-tag    dispatch tag for a primitive array of this dtype
;;   :vt           wasm / vector value-type keyword
;;   :needs-pragma OpenCL extension pragma this type requires, or nil
;;   :fp?          floating-point?
(def dtype-info
  {:double {:native {:c "double" :opencl "double" :glsl "double"}
            :scalar-tag 'double :array-tag 'doubles :vt :f64 :needs-pragma :cl_khr_fp64 :fp? true}
   :float  {:native {:c "float" :opencl "float" :glsl "float"}
            :scalar-tag 'float :array-tag 'floats :vt :f32 :needs-pragma nil :fp? true}
   :half   {:native {:c "_Float16" :opencl "half" :glsl "float16_t"}
            :scalar-tag nil :array-tag 'shorts :vt :f16 :needs-pragma :cl_khr_fp16 :fp? true}
   :int    {:native {:c "int" :opencl "int" :glsl "int"}
            :scalar-tag 'int :array-tag 'ints :vt :i32 :needs-pragma nil :fp? false}
   :long   {:native {:c "long long" :opencl "long" :glsl "int"}
            :scalar-tag 'long :array-tag 'longs :vt :i64 :needs-pragma nil :fp? false}
   :byte   {:native {:c "int8_t" :opencl "char" :glsl "int"}
            :scalar-tag 'byte :array-tag 'bytes :vt :i32 :needs-pragma nil :fp? false}})

(def native-types
  "Scalar dtype keyword → {backend → native type name}. Derived from dtype-info."
  (into {} (map (fn [[k v]] [k (:native v)])) dtype-info))

(defn backend-types
  "The {dtype → native-type-string} map for one backend (:c / :opencl / :glsl),
   derived from `native-types`."
  [backend]
  (into {} (map (fn [[dt facets]] [dt (get facets backend)])) native-types))

(defn scalar-tag-for-dtype
  "Scalar primitive type tag for a dtype (e.g. :float → 'float). Throws on an
   unknown or non-JVM-primitive dtype (e.g. :half has no primitive scalar)."
  [dtype]
  (or (:scalar-tag (get dtype-info dtype))
      (throw (ex-info (str "scalar-tag-for-dtype: no JVM-primitive scalar for dtype `"
                           dtype "`. Expected one of :double :float :int :long :byte.")
                      {:dtype dtype}))))

(defn array-tag-for-dtype
  "Primitive-array dispatch tag for a dtype (e.g. :float → 'floats). Throws on
   an unknown dtype."
  [dtype]
  (or (:array-tag (get dtype-info dtype))
      (throw (ex-info (str "array-tag-for-dtype: unknown dtype `" dtype "`.")
                      {:dtype dtype}))))

(defn needs-pragma-for
  "The OpenCL extension pragma keyword this dtype requires (e.g. :half →
   :cl_khr_fp16, :double → :cl_khr_fp64), or nil."
  [dtype]
  (:needs-pragma (get dtype-info dtype)))

(defn fp-dtype?
  "True when `dtype` is a floating-point dtype."
  [dtype]
  (boolean (:fp? (get dtype-info dtype))))

(def ^:private dtype-type-remap
  "Type tag remapping per dtype. Only FP types are remapped."
  {:float {'double 'float, 'doubles 'floats}})

(defn- remap-tag
  "Remap a type tag according to dtype. Returns tag unchanged if no remap."
  [tag dtype]
  (if-let [remap (get dtype-type-remap dtype)]
    (get remap tag tag)
    tag))

(defn remap-env
  "Remap all FP type tags in an env according to dtype."
  [env dtype]
  (if (or (nil? dtype) (= dtype :double))
    env
    (into {} (map (fn [[k v]] [k (remap-tag v dtype)])) env)))
