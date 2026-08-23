(ns raster.compiler.ir.kernel-abi
  "Ordered, typed kernel argument ABI.

   An ABI is a vector because kernel arguments are positional.  `:name` identifies the
   compiler value, `:c-name` identifies the emitted parameter, `:kind` controls binding,
   `:dtype` is the storage dtype, and `:kernel-dtype` is the kernel's view.  The last two
   deliberately differ for packed inputs (for example byte storage viewed as int32 by dp4a)."
  (:require [clojure.string :as str]
            [raster.compiler.core.dtype :as dt]))

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
  "Validate an ordered ABI and return it unchanged.  A kernel has at least one output;
   multi-output maps represent every independently written result in signature order."
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
  (when-not (some #(= :output (:kind %)) abi)
    (throw (ex-info "kernel ABI must contain at least one output" {:abi abi})))
  abi)

(defn pointer-slots
  "Pointer-valued ABI slots, in kernel signature order."
  [abi]
  (filterv #(not= :scalar (:kind %)) (validate! abi)))

(defn scalar-slots
  "Scalar ABI slots, in kernel signature order."
  [abi]
  (filterv #(= :scalar (:kind %)) (validate! abi)))

(defn validate-arguments!
  "Check that `arguments` supplies exactly one value per ordered ABI slot."
  [abi arguments]
  (let [abi (validate! abi)]
    (when-not (vector? arguments)
      (throw (ex-info "kernel ABI arguments must be a vector (argument order is semantic)"
                      {:arguments arguments :abi abi})))
    (when-not (= (count abi) (count arguments))
      (throw (ex-info "kernel ABI argument count mismatch"
                      {:expected (count abi) :actual (count arguments) :abi abi})))
    arguments))

(defn validate-split-binding!
  "Validate the compatibility binder shape `(pointers, user-scalars, implicit-bound)` against an
   ordered ABI. Pre-typed scalar maps must also agree with the ABI's kernel view dtype."
  [abi pointers scalars]
  (let [pointer-slots (pointer-slots abi)
        all-scalars (scalar-slots abi)
        bound-slots (filterv #(= :bound (:role %)) all-scalars)
        user-slots (filterv #(not= :bound (:role %)) all-scalars)]
    (when-not (= 1 (count bound-slots))
      (throw (ex-info "split map binding requires exactly one implicit :bound ABI slot"
                      {:abi abi :bound-slots bound-slots})))
    (when-not (= (count pointer-slots) (count pointers))
      (throw (ex-info "kernel ABI pointer count does not match binding"
                      {:expected (count pointer-slots) :actual (count pointers) :abi abi})))
    (when-not (= (count user-slots) (count scalars))
      (throw (ex-info "kernel ABI scalar count does not match binding"
                      {:expected (count user-slots) :actual (count scalars) :abi abi})))
    (doseq [[slot value] (map vector user-slots scalars)
            :when (map? value)]
      (when-not (= (:kernel-dtype slot) (:type value))
        (throw (ex-info "kernel ABI scalar dtype does not match binding"
                        {:slot slot :expected (:kernel-dtype slot) :actual (:type value)}))))
    {:pointer-slots pointer-slots :scalar-slots user-slots :bound-slot (first bound-slots)}))

(defn signature-shape
  "The structural portion checked against emitted C."
  [abi]
  (mapv (fn [{:keys [c-name kind]}]
          {:c-name c-name :pointer? (not= :scalar kind)})
        (validate! abi)))

(defn source-signature-shape
  "Extract ordered C parameter names and pointer/scalar shape for one OpenCL kernel."
  [kernel-name source]
  (let [pattern (re-pattern
                 (str "(?s)__kernel\\s+void\\s+" (java.util.regex.Pattern/quote kernel-name)
                      "\\s*\\(([^)]*)\\)"))
        params (second (re-find pattern source))]
    (when-not params
      (throw (ex-info "kernel ABI could not find emitted OpenCL signature"
                      {:kernel-name kernel-name})))
    (if (str/blank? params)
      []
      (mapv (fn [param]
              (let [param (str/trim param)
                    c-name (second (re-find #"([A-Za-z_][A-Za-z0-9_]*)\s*$" param))]
                (when-not c-name
                  (throw (ex-info "kernel ABI could not parse emitted OpenCL parameter"
                                  {:kernel-name kernel-name :parameter param})))
                {:c-name c-name :pointer? (str/includes? param "*")}))
            (str/split params #",")))))

(defn validate-source-signature!
  "Compare emitted OpenCL parameter order/shape with the ABI before registration."
  [kernel-name source abi]
  (let [expected (signature-shape abi)
        actual (source-signature-shape kernel-name source)]
    (when-not (= expected actual)
      (throw (ex-info "emitted OpenCL signature does not match kernel ABI"
                      {:kernel-name kernel-name :expected expected :actual actual :abi abi})))
    abi))
