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
  "Construct one ABI slot. Options: :c-name, :kernel-dtype, :role, and :binding.

   `:binding` names the logical caller value that supplies a physical pointer slot. It is
   normally absent (and therefore identical to `:name`), but an SoA argument has one physical
   slot per field and all of those slots share the same logical binding."
  [nm kind dtype & {:keys [c-name kernel-dtype role binding]}]
  (cond-> {:name nm
           :c-name (or c-name (name nm))
           :kind kind
           :dtype (dt/canon dtype)
           :kernel-dtype (dt/canon (or kernel-dtype dtype))}
    role (assoc :role role)
    binding (assoc :binding binding)))

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
                        {:index i :slot s :dtype (get s k)}))))
    (when (and (= :scalar (:kind s)) (:binding s))
      (throw (ex-info "kernel ABI scalar slot cannot have a logical pointer :binding"
                      {:index i :slot s :abi abi}))))
  (when-not (= (count abi) (count (distinct (map :c-name abi))))
    (throw (ex-info "kernel ABI parameter names must be unique" {:abi abi})))
  (let [pointers (vec (remove #(= :scalar (:kind %)) abi))]
    (doseq [binding (distinct (keep :binding pointers))]
      (let [indexes (keep-indexed (fn [i slot] (when (= binding (:binding slot)) i)) pointers)]
        (when-not (= (count indexes) (inc (- (apply max indexes) (apply min indexes))))
          (throw (ex-info "kernel ABI physical slots for one logical binding must be contiguous"
                          {:binding binding :indexes (vec indexes) :abi abi}))))))
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

(defn pointer-binding-names
  "Logical caller pointer values, in first physical-signature occurrence order.

   Usually this is simply `(mapv :name (pointer-slots abi))`. An SoA occupies several physical
   C pointer slots but is supplied by one logical GpuSoA value; those slots carry a shared
   `:binding` and collapse to one name here."
  [abi]
  (:names
   (reduce (fn [{:keys [seen] :as acc} slot]
             (if-let [binding (:binding slot)]
               (if (contains? seen binding)
                 acc
                 (-> acc (update :names conj binding) (update :seen conj binding)))
               (update acc :names conj (:name slot))))
           {:names [] :seen #{}} (pointer-slots abi))))

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
        pointer-bindings (pointer-binding-names abi)
        all-scalars (scalar-slots abi)
        bound-slots (filterv #(= :bound (:role %)) all-scalars)
        user-slots (filterv #(not= :bound (:role %)) all-scalars)]
    (when-not (= 1 (count bound-slots))
      (throw (ex-info "split map binding requires exactly one implicit :bound ABI slot"
                      {:abi abi :bound-slots bound-slots})))
    (when-not (= (count pointer-bindings) (count pointers))
      (throw (ex-info "kernel ABI pointer count does not match binding"
                      {:expected (count pointer-bindings) :actual (count pointers) :abi abi})))
    (when-not (= (count user-slots) (count scalars))
      (throw (ex-info "kernel ABI scalar count does not match binding"
                      {:expected (count user-slots) :actual (count scalars) :abi abi})))
    (doseq [[slot value] (map vector user-slots scalars)
            :when (map? value)]
      (when-not (= (:kernel-dtype slot) (:type value))
        (throw (ex-info "kernel ABI scalar dtype does not match binding"
                        {:slot slot :expected (:kernel-dtype slot) :actual (:type value)}))))
    {:pointer-slots pointer-slots :pointer-bindings pointer-bindings
     :scalar-slots user-slots :bound-slot (first bound-slots)}))

(defn validate-physical-pointer-dtypes!
  "Check the storage dtype of every PHYSICAL pointer value after logical values (notably SoAs)
   have been expanded. `actual-dtypes` must follow the emitted C signature order. This is kept
   separate from `validate-split-binding!` because expansion is a runtime representation concern,
   while the ABI remains a backend-neutral physical signature."
  [abi actual-dtypes]
  (let [slots (pointer-slots abi)]
    (when-not (= (count slots) (count actual-dtypes))
      (throw (ex-info "kernel ABI physical pointer count does not match expanded binding"
                      {:expected (count slots) :actual (count actual-dtypes) :abi abi})))
    (doseq [[slot actual] (map vector slots actual-dtypes)]
      (when-not (or (= :opaque actual) (and actual (dt/known? actual)))
        (throw (ex-info "kernel ABI pointer value has no supported storage dtype"
                        {:slot slot :actual actual :abi abi})))
      (when (and (not= :opaque actual) (not= (:dtype slot) (dt/canon actual)))
        (throw (ex-info "kernel ABI storage dtype does not match binding"
                        {:slot slot :expected (:dtype slot) :actual (dt/canon actual)}))))
    actual-dtypes))

(defn validate-reduction-arguments!
  "Validate a complete ordered reduction binding and expose its semantic projections without
   reconstructing positional conventions.  A reduction owns exactly one pointer-valued :result
   and exactly one scalar :bound; captured scalars and operands may otherwise be arbitrary ABI
   slots.  Returns the ordered slot/value pairs plus role-derived projections."
  [abi arguments]
  (let [arguments (validate-arguments! abi arguments)
        pairs (mapv vector abi arguments)
        result-pairs (filterv (fn [[slot _]] (= :result (:role slot))) pairs)
        bound-pairs (filterv (fn [[slot _]] (= :bound (:role slot))) pairs)]
    (when-not (= 1 (count result-pairs))
      (throw (ex-info "reduction ABI must identify exactly one :result slot"
                      {:abi abi :result-slots (mapv first result-pairs)})))
    (when-not (= :output (:kind (ffirst result-pairs)))
      (throw (ex-info "reduction ABI :result slot must be an output pointer"
                      {:abi abi :result-slot (ffirst result-pairs)})))
    (when-not (= 1 (count bound-pairs))
      (throw (ex-info "reduction ABI must identify exactly one :bound slot"
                      {:abi abi :bound-slots (mapv first bound-pairs)})))
    (when-not (= :scalar (:kind (ffirst bound-pairs)))
      (throw (ex-info "reduction ABI :bound slot must be scalar"
                      {:abi abi :bound-slot (ffirst bound-pairs)})))
    {:pairs pairs
     :pointer-pairs (filterv (fn [[slot _]] (not= :scalar (:kind slot))) pairs)
     :scalar-pairs (filterv (fn [[slot _]] (and (= :scalar (:kind slot))
                                                 (not= :bound (:role slot)))) pairs)
     :result-pair (first result-pairs)
     :bound-pair (first bound-pairs)}))

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
                    ;; Clojure gensyms in emitted kernels may retain Unicode letters (for example
                    ;; the AD pipeline uses α). Capture the complete final non-whitespace token;
                    ;; an ASCII-only C-identifier regex silently reduced `y_α_42` to `_42`.
                    c-name (second (re-find #"([^\s*]+)\s*$" param))]
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
