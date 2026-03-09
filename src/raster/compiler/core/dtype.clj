(ns raster.compiler.core.dtype
  "Dtype-driven type tag remapping helpers for compiler lowering.")

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