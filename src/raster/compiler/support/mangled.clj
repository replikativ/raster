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

(defn impl->op
  "Recover a qualified source operation from a deftm implementation identity."
  [sym]
  (when (and (symbol? sym) (namespace sym)
             (pos? (.indexOf ^String (name sym) "_m_")))
    (let [base (unqualified-base-name sym)
          decoded (reduce (fn [s [encoded original]]
                            (.replace ^String s ^CharSequence encoded ^CharSequence original))
                          base [["_plus_" "+"] ["_minus_" "-"] ["_star_" "*"]
                                ["_div_" "/"] ["_lt_" "<"] ["_gt_" ">"]
                                ["_lteq_" "<="] ["_gteq_" ">="] ["_eq_" "="]
                                ["_bang" "!"] ["_qmark" "?"]])]
      (symbol (namespace sym) decoded))))
