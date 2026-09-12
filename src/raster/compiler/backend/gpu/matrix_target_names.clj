(ns raster.compiler.backend.gpu.matrix-target-names
  "Shared ordered ABI spelling and generated-local collision checks for matrix targets."
  (:require [raster.compiler.backend.gpu.c-emit :as c-emit]))

(defn parameter-names
  [body overrides]
  (let [{:keys [m n k]} (get-in body [:attributes :dimension-parameters])]
    (merge
     (into {}
           (map (fn [{:keys [id role]}]
                  [id (case role
                        :lhs "A"
                        :rhs "B"
                        :result "C"
                        (c-emit/c-symbol id))]))
           (:parameters body))
     {m "M" n "N" k "K"}
     overrides)))

(def ^:private matrix-local-names
  #{"warpId" "warp_row" "warp_col" "sg_id" "sg_lid" "sg_row" "sg_col"
    "m_base" "n_base" "k" "pk" "pr" "row" "col" "k_begin" "k_end"
    "a_wb" "a_pb" "b_wb" "b_pb" "rstr_epilogue_element"})

(defn- generated-matrix-local?
  [generated-locals c-name]
  (or (contains? generated-locals c-name)
      (boolean (re-matches #"(?:a|b|sa|bp|acc|n_base)[0-9]+(?:_[0-9]+)?" c-name))))

(defn- matrix-generated-locals
  [body]
  (into matrix-local-names
        (comp (filter #(and (= :group (:source %)) (= 2 (:axis %))))
              (map #(c-emit/c-symbol (:id %))))
        (:indices body)))

(defn validate!
  [kernel-name body names]
  (when-not (c-emit/c-identifier? kernel-name)
    (throw (ex-info "matrix kernel entry point is not a portable C-family identifier"
                    {:reason :matrix-target-entry-point :kernel-name kernel-name})))
  (let [ordered-names (mapv #(get names (:id %)) (:parameters body))
        generated-locals (matrix-generated-locals body)]
    (doseq [[parameter c-name] (map vector (:parameters body) ordered-names)]
      (when-not (c-emit/c-identifier? c-name)
        (throw (ex-info "matrix ABI parameter is not a portable C-family identifier"
                        {:reason :matrix-target-parameter-name
                         :parameter parameter :c-name c-name})))
      (when (generated-matrix-local? generated-locals c-name)
        (throw (ex-info "matrix ABI parameter collides with a generated kernel local"
                        {:reason :matrix-target-name-collision
                         :parameter parameter :c-name c-name}))))
    (when-not (= (count ordered-names) (count (set ordered-names)))
      (throw (ex-info "matrix ABI parameter names are not unique after target spelling"
                      {:reason :matrix-target-name-collision
                       :parameters (:parameters body) :c-names ordered-names})))
    names))

