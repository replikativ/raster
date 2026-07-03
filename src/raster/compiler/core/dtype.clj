(ns raster.compiler.core.dtype
  "Dtype-driven type tag remapping helpers for compiler lowering.")

;; ── Single faceted source for scalar dtype → per-backend native type ────────
;; Previously each textual backend kept its own {:double "double" …} map, which
;; drifted independently. Per-backend value differences are LEGITIMATE facets,
;; not inconsistencies: GLSL has no 64-bit int (long→"int"), OpenCL `long` is
;; 64-bit, C uses `long long`. Add a new dtype = add ONE row here; backends
;; derive their map via `backend-types`.
(def native-types
  "Scalar dtype keyword → {backend → native type name}."
  {:double {:c "double"    :opencl "double" :glsl "double"}
   :float  {:c "float"     :opencl "float"  :glsl "float"}
   :int    {:c "int"       :opencl "int"    :glsl "int"}
   :long   {:c "long long" :opencl "long"   :glsl "int"}})

(defn backend-types
  "The {dtype → native-type-string} map for one backend (:c / :opencl / :glsl),
   derived from `native-types`. Replaces the per-backend literal type maps."
  [backend]
  (into {} (map (fn [[dt facets]] [dt (get facets backend)])) native-types))

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

(defn scalar-tag-for-dtype
  "Scalar type tag for a dtype. Throws on nil/unknown."
  [dtype]
  (case dtype
    :float 'float
    :double 'double
    (throw (ex-info (str "scalar-tag-for-dtype: unknown dtype `" dtype
                         "`. Expected :double or :float.")
                    {:dtype dtype}))))