(ns raster.compiler.ir.kernel-dispatch
  "A verified runtime choice among ABI-compatible emitted kernels.

   KernelArtifact remains one entry point. KernelDispatch is the scheduling value above it: every
   alternative implements the same logical call and differs only in emitted schedule/geometry.
   Selection is pure data evaluated after symbolic ABI scalars become concrete and before a
   backend binder sees a KernelCall."
  (:require [raster.compiler.ir.kernel-artifact :as kart]))

(defrecord KernelDispatch
           [id
            alternatives
            default-strategy
            selector
            provenance
            attributes])

(defn kernel-dispatch?
  "Recognize dispatches across Typed Clojure child classloaders."
  [x]
  (and x (= "raster.compiler.ir.kernel_dispatch.KernelDispatch"
            (.getName (class x)))))

(defn artifact-strategy
  [artifact]
  (get-in (kart/validate! artifact) [:attributes :strategy]))

(defn- strategy-map
  [alternatives]
  (into {} (map (juxt artifact-strategy identity)) alternatives))

(defn- finite-number?
  [value]
  (and (number? value) (Double/isFinite (double value))))

(defn- validate-selector-strategy!
  [dispatch strategies branch strategy]
  (when-not (contains? strategies strategy)
    (throw (ex-info "kernel dispatch selector names an absent strategy"
                    {:id (:id dispatch) :branch branch :strategy strategy
                     :strategies (set (keys strategies))}))))

(defn- validate-selector!
  [dispatch strategies common-arguments]
  (let [{:keys [kind argument threshold at-least otherwise ranges below]}
        (:selector dispatch)]
    (when-not (some #(= argument %) common-arguments)
      (throw (ex-info "kernel dispatch selector argument is absent from the common ABI values"
                      {:id (:id dispatch) :argument argument
                       :arguments common-arguments})))
    (case kind
      :runtime-scalar-threshold
      (do
        (when-not (and (finite-number? threshold) (pos? (double threshold)))
          (throw (ex-info "kernel dispatch threshold must be finite and positive"
                          {:id (:id dispatch) :threshold threshold})))
        (doseq [[branch strategy] [[:at-least at-least] [:otherwise otherwise]]]
          (validate-selector-strategy! dispatch strategies branch strategy)))

      :runtime-scalar-ranges
      (do
        (validate-selector-strategy! dispatch strategies :below below)
        (when-not (vector? ranges)
          (throw (ex-info "kernel dispatch ranges must be an ordered vector"
                          {:id (:id dispatch) :ranges ranges})))
        (doseq [[index {:keys [at-least strategy] :as range}] (map-indexed vector ranges)]
          (when-not (= #{:at-least :strategy} (set (keys range)))
            (throw (ex-info "kernel dispatch range requires exactly :at-least and :strategy"
                            {:id (:id dispatch) :index index :range range})))
          (when-not (finite-number? at-least)
            (throw (ex-info "kernel dispatch range boundary must be finite"
                            {:id (:id dispatch) :index index :at-least at-least})))
          (validate-selector-strategy! dispatch strategies [:ranges index] strategy))
        (when-not (or (< (count ranges) 2)
                      (apply < (map :at-least ranges)))
          (throw (ex-info "kernel dispatch range boundaries must be strictly increasing"
                          {:id (:id dispatch) :ranges ranges}))))

      (throw (ex-info "kernel dispatch has an unsupported selector"
                      {:id (:id dispatch) :selector (:selector dispatch)})))))

(defn validate!
  "Validate a dispatch and return it unchanged.

   Alternatives must have unique strategy identities and identical target, ABI, compiler argument
   order, temporaries and effects. Their source and launch geometry may differ—that is the point."
  [dispatch]
  (when-not (kernel-dispatch? dispatch)
    (throw (ex-info "kernel dispatch must be a KernelDispatch value"
                    {:dispatch dispatch :actual (type dispatch)})))
  (let [{:keys [id alternatives default-strategy provenance attributes]} dispatch]
    (when-not (and (string? id) (not-empty id))
      (throw (ex-info "kernel dispatch requires a non-empty string id" {:id id})))
    (when-not (and (vector? alternatives) (seq alternatives))
      (throw (ex-info "kernel dispatch requires an ordered non-empty alternative vector"
                      {:id id :alternatives alternatives})))
    (doseq [artifact alternatives] (kart/validate! artifact))
    (let [strategies (mapv artifact-strategy alternatives)
          strategy-set (set strategies)
          common (first alternatives)
          common-view (select-keys common [:target :abi :arguments :temporaries :effects])]
      (when-not (every? keyword? strategies)
        (throw (ex-info "every kernel dispatch alternative requires a keyword :strategy"
                        {:id id :strategies strategies})))
      (when-not (= (count strategies) (count strategy-set))
        (throw (ex-info "kernel dispatch alternative strategies must be unique"
                        {:id id :strategies strategies})))
      (doseq [artifact (rest alternatives)]
        (when-not (= common-view
                     (select-keys artifact [:target :abi :arguments :temporaries :effects]))
          (throw (ex-info "kernel dispatch alternatives must share target, ABI and logical effects"
                          {:id id
                           :common-kernel (:kernel-name common)
                           :different-kernel (:kernel-name artifact)}))))
      (when-not (contains? strategy-set default-strategy)
        (throw (ex-info "kernel dispatch default strategy is absent"
                        {:id id :default default-strategy :strategies strategy-set})))
      (validate-selector! dispatch (strategy-map alternatives) (:arguments common)))
    (doseq [[field value] [[:provenance provenance] [:attributes attributes]]]
      (when-not (map? value)
        (throw (ex-info "kernel dispatch metadata must be maps"
                        {:id id :field field :value value}))))
    dispatch))

(defn make
  [{:keys [id alternatives default-strategy selector provenance attributes]
    :or {provenance {} attributes {}}}]
  (validate!
   (->KernelDispatch id alternatives default-strategy selector provenance attributes)))

(defn artifact
  "Return the artifact implementing `strategy`, or throw with the legal strategy set."
  [dispatch strategy]
  (let [dispatch (validate! dispatch)
        alternatives (strategy-map (:alternatives dispatch))]
    (or (get alternatives strategy)
        (throw (ex-info "kernel dispatch strategy is unavailable"
                        {:id (:id dispatch) :strategy strategy
                         :strategies (set (keys alternatives))})))))

(defn default-artifact
  [dispatch]
  (let [dispatch (validate! dispatch)]
    (artifact dispatch (:default-strategy dispatch))))

(defn with-selector
  "Return `dispatch` with a replacement selector, revalidating it against the common ABI and
   available strategies. Used to bake an offline tuning result into otherwise identical IR."
  [dispatch selector]
  (validate! (assoc (validate! dispatch) :selector selector)))

(defn- runtime-number
  [value]
  (if (and (map? value) (contains? value :value)) (:value value) value))

(defn select-artifact
  "Select an artifact from concrete ABI-ordered values.

   `override` is nil/:auto for data-driven selection or an explicit alternative strategy. The
   runtime threshold selector is intentionally generic: it knows only a compiler argument and a
   numeric crossover, not which model operation produced them."
  ([dispatch runtime-arguments] (select-artifact dispatch runtime-arguments nil))
  ([dispatch runtime-arguments override]
   (let [dispatch (validate! dispatch)]
     (if (and override (not= :auto override))
       (artifact dispatch override)
       (let [{:keys [kind argument threshold at-least otherwise ranges below]}
             (:selector dispatch)
             common-arguments (:arguments (default-artifact dispatch))
             indexes (keep-indexed (fn [index value] (when (= argument value) index))
                                   common-arguments)]
         (when-not (= 1 (count indexes))
           (throw (ex-info "kernel dispatch selector argument must have one runtime position"
                           {:id (:id dispatch) :argument argument
                            :indexes (vec indexes)})))
         (when-not (= (count common-arguments) (count runtime-arguments))
           (throw (ex-info "kernel dispatch runtime argument count mismatch"
                           {:id (:id dispatch) :expected (count common-arguments)
                            :actual (count runtime-arguments)})))
         (let [value (runtime-number (nth runtime-arguments (first indexes)))]
           (when-not (number? value)
             (throw (ex-info "kernel dispatch selector requires a numeric runtime scalar"
                             {:id (:id dispatch) :argument argument :value value})))
           (artifact
            dispatch
            (case kind
              :runtime-scalar-threshold
              (if (>= (double value) (double threshold)) at-least otherwise)

              :runtime-scalar-ranges
              (reduce (fn [strategy range]
                        (if (>= (double value) (double (:at-least range)))
                          (:strategy range)
                          (reduced strategy)))
                      below ranges)))))))))
