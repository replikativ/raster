(ns raster.compiler.support.mangled
  "Shared helpers for compiler-facing deftm and impl symbol mangling.")

(defn strip-impl-suffix
  "Strip the -impl suffix from a function name string."
  [name-str]
  (if (.endsWith ^String name-str "-impl")
    (.substring ^String name-str 0 (- (count name-str) 5))
    name-str))

(defn strip-deftm-mangling
  "Strip deftm mangling suffixes from a function name string.
	Example: dense_m_Object_Object -> dense"
  [name-str]
  (let [impl-free (strip-impl-suffix name-str)
        mangled-idx (.indexOf ^String impl-free "_m_")]
    (if (neg? mangled-idx)
      impl-free
      (.substring ^String impl-free 0 mangled-idx))))

(defn unqualified-base-name
  "Return the unqualified base name for a possibly mangled symbol."
  [sym]
  (when (symbol? sym)
    (strip-deftm-mangling (name sym))))

(defn extract-deftm-base
  "Extract the base symbol from a possibly deftm-mangled symbol.
	Preserves the original namespace when present."
  [sym]
  (when (symbol? sym)
    (let [base-name (unqualified-base-name sym)
          ns-str (namespace sym)]
      (if ns-str
        (symbol ns-str base-name)
        (symbol base-name)))))