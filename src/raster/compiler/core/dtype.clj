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
;;   :bytes        width of one element
;;   :aliases      other spellings that MEAN this dtype (:int8 and :byte are one type, not two)
;;
;; ALIASES EXIST BECAUSE THE ABSENCE OF THEM WAS A BUG. Gates accepted `:int8`/`:int32`/`:float16`
;; while this map had no such rows, so `(get opencl-type-map :int32 "double")` handed back "double"
;; — emitting `__global const double*` over an int8 buffer and turning an int32 accumulator into a
;; float, losing the exactness that block-quant staging exists for. Hence the two access rules
;; below: `ctype`/`bytes-of`/`canon` THROW on an unknown dtype (an emitter must fail loud), while
;; `known?` RETURNS false (a legality gate must return a verdict, never throw).
(def dtype-info
  {:double {:native {:c "double" :opencl "double" :glsl "double"}
            :scalar-tag 'double :array-tag 'doubles :vt :f64 :needs-pragma :cl_khr_fp64 :fp? true
            :bytes 8 :aliases #{:float64 :f64}}
   :float  {:native {:c "float" :opencl "float" :glsl "float"}
            :scalar-tag 'float :array-tag 'floats :vt :f32 :needs-pragma nil :fp? true
            :bytes 4 :aliases #{:float32 :f32}}
   :half   {:native {:c "_Float16" :opencl "half" :glsl "float16_t"}
            :scalar-tag nil :array-tag 'shorts :vt :f16 :needs-pragma :cl_khr_fp16 :fp? true
            :bytes 2 :aliases #{:float16 :f16}}
   :int    {:native {:c "int" :opencl "int" :glsl "int"}
            :scalar-tag 'int :array-tag 'ints :vt :i32 :needs-pragma nil :fp? false
            :bytes 4 :aliases #{:int32 :i32}}
   :long   {:native {:c "long long" :opencl "long" :glsl "int"}
            :scalar-tag 'long :array-tag 'longs :vt :i64 :needs-pragma nil :fp? false
            :bytes 8 :aliases #{:int64 :i64}}
   :byte   {:native {:c "int8_t" :opencl "char" :glsl "int"}
            :scalar-tag 'byte :array-tag 'bytes :vt :i32 :needs-pragma nil :fp? false
            :bytes 1 :aliases #{:int8 :i8}}})

(def ^:private alias->canonical
  (into {} (for [[k v] dtype-info, a (cons k (:aliases v))] [a k])))

(defn known?
  "Is `dt` a dtype this compiler can spell (canonical or alias)? Returns a BOOLEAN — this is the
   accessor a legality gate uses, because a gate must return a verdict rather than throw."
  [dt]
  (contains? alias->canonical dt))

(defn canon
  "Alias → canonical dtype (`:int8`/`:i8` → `:byte`, `:float16` → `:half`, …). THROWS on an unknown
   dtype: an emitter that cannot spell a type must fail loudly, never silently substitute one."
  [dt]
  (or (alias->canonical dt)
      (throw (ex-info (str "unknown dtype `" dt "`. Known: "
                           (pr-str (sort (keys alias->canonical))))
                      {:reason :unknown-dtype :dtype dt}))))

(defn info "Facet row for a dtype, resolving aliases. Throws on unknown." [dt]
  (get dtype-info (canon dt)))

(defn ctype
  "Native type name for `backend` (:c / :opencl / :glsl). THROWS on an unknown dtype — the silent
   string default this replaces is what emitted `double*` over int8 buffers."
  [backend dt]
  (or (get-in dtype-info [(canon dt) :native backend])
      (throw (ex-info (str "dtype `" dt "` has no " backend " native type")
                      {:reason :no-native-type :dtype dt :backend backend}))))

(defn bytes-of "Width of one element in bytes. Throws on unknown." [dt] (:bytes (info dt)))

(defn integral? "True for integer dtypes (the widening-accumulator side of a dtype pair)."
  [dt] (not (:fp? (info dt))))

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
  "True when `dtype` is a floating-point dtype. Alias-tolerant; false for an unknown dtype
   (callers wanting a loud failure use `canon`/`info`)."
  [dtype]
  (boolean (when (known? dtype) (:fp? (info dtype)))))

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

(defn infer-dtype-from-tags
  "Element dtype of a body given its parameter dispatch tags — the effective-
   dtype rule shared by ALL monomorphized walk seams (compile-aot get-walked-body,
   lazy-JIT jit-walk-with-tc, specialize-fn!): array tags first (floats/doubles),
   then scalar tags (float/double). :float, :double, or nil (integer-only /
   generic). A float-tagged body walked WITHOUT this dtype types its bare 0.0
   accumulator inits double — the f64-in-f32 tier-divergence class."
  [tags]
  (when tags
    (cond
      (some #{'floats} tags) :float
      (some #{'doubles} tags) :double
      (some #{'float} tags) :float
      (some #{'double} tags) :double
      :else nil)))
