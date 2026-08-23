(ns raster.compiler.ir.kernel-abi
  "Ordered, typed kernel argument ABI.

   An ABI is a vector because kernel arguments are positional.  `:name` identifies the
   compiler value, `:c-name` identifies the emitted parameter, `:kind` controls binding,
   `:dtype` is the storage dtype, and `:kernel-dtype` is the kernel's view.  The last two
   deliberately differ for packed inputs (for example byte storage viewed as int32 by dp4a)."
  (:require [raster.compiler.core.dtype :as dt]))

(def ^:private kinds #{:input :output :scalar})

(defn slot
  "Construct one ABI slot.  Options: :c-name, :kernel-dtype and :role."
  [nm kind dtype & {:keys [c-name kernel-dtype role]}]
  (cond-> {:name nm
           :c-name (or c-name (name nm))
           :kind kind
           :dtype (dt/canon dtype)
           :kernel-dtype (dt/canon (or kernel-dtype dtype))}
    role (assoc :role role)))

(defn validate!
  "Validate an ordered ABI and return it unchanged.  Contraction ABIs have exactly one output."
  [abi]
  (when-not (vector? abi)
    (throw (ex-info "kernel ABI must be a vector (argument order is semantic)" {:abi abi})))
  (doseq [[i s] (map-indexed vector abi)]
    (when-not (map? s)
      (throw (ex-info "kernel ABI slot must be a map" {:index i :slot s :abi abi})))
    (when-not (contains? kinds (:kind s))
      (throw (ex-info "kernel ABI slot has an invalid :kind"
                      {:index i :slot s :allowed kinds})))
    (doseq [k [:name :c-name :dtype :kernel-dtype]]
      (when-not (some? (get s k))
        (throw (ex-info (str "kernel ABI slot is missing " k) {:index i :slot s}))))
    (doseq [k [:dtype :kernel-dtype]]
      (when-not (dt/known? (get s k))
        (throw (ex-info (str "kernel ABI slot has an unknown " k)
                        {:index i :slot s :dtype (get s k)})))))
  (when-not (= (count abi) (count (distinct (map :c-name abi))))
    (throw (ex-info "kernel ABI parameter names must be unique" {:abi abi})))
  (when-not (= 1 (count (filter #(= :output (:kind %)) abi)))
    (throw (ex-info "contraction kernel ABI must contain exactly one output" {:abi abi})))
  abi)

(defn signature-shape
  "The structural portion checked against emitted C."
  [abi]
  (mapv (fn [{:keys [c-name kind]}]
          {:c-name c-name :pointer? (not= :scalar kind)})
        (validate! abi)))
