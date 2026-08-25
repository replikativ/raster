(ns raster.compiler.ir.kernel-dispatch
  "A verified runtime choice among ABI-compatible kernel executables.

   An alternative may be one KernelArtifact or an emitted KernelGraph. KernelDispatch is the
   scheduling value above both: every alternative implements the same logical call and differs
   only in its emitted schedule. Selection is pure data evaluated after symbolic ABI scalars
   become concrete and before a backend binder sees the selected executable."
  (:require [raster.compiler.ir.kernel-executable :as kexec]
            [raster.compiler.ir.kernel-launch :as klaunch]))

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

(defn alternative-strategy
  [alternative]
  (kexec/strategy alternative))

(defn- strategy-map
  [alternatives]
  (into {} (map (juxt alternative-strategy identity)) alternatives))

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
  [dispatch strategies common]
  (let [{:keys [kind argument expression threshold at-least otherwise ranges below]}
        (:selector dispatch)
        common-arguments (:arguments common)
        scalar-arguments (->> (map vector (:abi common) common-arguments)
                              (keep (fn [[slot compiler-value]]
                                      (when (= :scalar (:kind slot)) compiler-value)))
                              set)
        validate-threshold! (fn []
                              (when-not (and (finite-number? threshold)
                                             (pos? (double threshold)))
                                (throw (ex-info "kernel dispatch threshold must be finite and positive"
                                                {:id (:id dispatch) :threshold threshold})))
                              (doseq [[branch strategy] [[:at-least at-least]
                                                         [:otherwise otherwise]]]
                                (validate-selector-strategy! dispatch strategies branch strategy)))
        validate-expression! (fn [owner expression]
                               (when-not (klaunch/expression? expression)
                                 (throw (ex-info
                                         "kernel dispatch selector requires checked integer expression IR"
                                         {:id (:id dispatch) :owner owner
                                          :expression expression})))
                               (let [references (klaunch/expression-references expression)]
                                 (when-not (every? scalar-arguments references)
                                   (throw (ex-info
                                           "kernel dispatch expression reads values outside its scalar ABI"
                                           {:id (:id dispatch) :owner owner
                                            :references references
                                            :scalar-arguments scalar-arguments})))))
        validate-argument! (fn []
                             (let [indexes (keep-indexed
                                            (fn [index value] (when (= argument value) index))
                                            common-arguments)]
                               (when-not (= 1 (count indexes))
                                 (throw (ex-info
                                         "kernel dispatch selector argument must have one common ABI position"
                                         {:id (:id dispatch) :argument argument
                                          :arguments common-arguments :indexes (vec indexes)})))
                               (let [slot (nth (:abi common) (first indexes))]
                                 (when-not (= :scalar (:kind slot))
                                   (throw (ex-info
                                           "kernel dispatch selector argument must name a scalar ABI slot"
                                           {:id (:id dispatch) :argument argument :slot slot}))))))]
    (case kind
      :runtime-scalar-threshold
      (do
        (validate-argument!)
        (validate-threshold!))

      :runtime-expression-threshold
      (do
        (validate-expression! :expression expression)
        (validate-threshold!))

      :runtime-expression-cases
      (let [cases (get-in dispatch [:selector :cases])
            default (get-in dispatch [:selector :default])]
        (when-not (and (vector? cases) (seq cases))
          (throw (ex-info "kernel dispatch expression cases must be a non-empty ordered vector"
                          {:id (:id dispatch) :cases cases})))
        (validate-selector-strategy! dispatch strategies :default default)
        (doseq [[index rule] (map-indexed vector cases)]
          (when-not (= #{:expression :op :value :strategy} (set (keys rule)))
            (throw (ex-info "kernel dispatch expression case has invalid fields"
                            {:id (:id dispatch) :index index :rule rule})))
          (validate-expression! [:cases index] (:expression rule))
          (when-not (contains? #{:< :<= := :>= :>} (:op rule))
            (throw (ex-info "kernel dispatch expression case has an invalid comparison"
                            {:id (:id dispatch) :index index :op (:op rule)})))
          (when-not (finite-number? (:value rule))
            (throw (ex-info "kernel dispatch expression case requires a finite comparison value"
                            {:id (:id dispatch) :index index :value (:value rule)})))
          (validate-selector-strategy! dispatch strategies [:cases index] (:strategy rule))))

      :runtime-scalar-ranges
      (do
        (validate-argument!)
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
   order, and logical effects. Their entry points, launch geometry, graph topology, and private
   temporaries may differ—that is the point."
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
    (doseq [alternative alternatives] (kexec/validate! alternative))
    (let [strategies (mapv alternative-strategy alternatives)
          strategy-set (set strategies)
          common (first alternatives)
          common-view (kexec/common-view common)
          named-artifacts (mapcat (fn [alternative]
                                    (map (juxt :kernel-name identity)
                                         (kexec/artifacts alternative)))
                                  alternatives)]
      (when-not (every? keyword? strategies)
        (throw (ex-info "every kernel dispatch alternative requires a keyword :strategy"
                        {:id id :strategies strategies})))
      (when-not (= (count strategies) (count strategy-set))
        (throw (ex-info "kernel dispatch alternative strategies must be unique"
                        {:id id :strategies strategies})))
      (doseq [alternative (rest alternatives)]
        (when-not (= common-view (kexec/common-view alternative))
          (throw (ex-info "kernel dispatch alternatives must share target, ABI and logical effects"
                          {:id id
                           :common-strategy (alternative-strategy common)
                           :different-strategy (alternative-strategy alternative)}))))
      ;; Graph schedules may deliberately share entry points. Reuse is legal only when the name
      ;; denotes the same emitted module/signature; otherwise backend registration is ambiguous.
      (doseq [[kernel-name entries] (group-by first named-artifacts)]
        (let [implementations (set (map (fn [[_ artifact]]
                                          (select-keys artifact [:target :source :abi :arguments]))
                                        entries))]
          (when-not (= 1 (count implementations))
            (throw (ex-info "kernel dispatch reuses an entry point for conflicting modules"
                            {:id id :kernel-name kernel-name})))))
      (when-not (contains? strategy-set default-strategy)
        (throw (ex-info "kernel dispatch default strategy is absent"
                        {:id id :default default-strategy :strategies strategy-set})))
      (validate-selector! dispatch (strategy-map alternatives) common))
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

(defn alternative
  "Return the executable implementing `strategy`, or throw with the legal strategy set."
  [dispatch strategy]
  (let [dispatch (validate! dispatch)
        alternatives (strategy-map (:alternatives dispatch))]
    (or (get alternatives strategy)
        (throw (ex-info "kernel dispatch strategy is unavailable"
                        {:id (:id dispatch) :strategy strategy
                         :strategies (set (keys alternatives))})))))

(defn default-alternative
  [dispatch]
  (let [dispatch (validate! dispatch)]
    (alternative dispatch (:default-strategy dispatch))))

(defn with-selector
  "Return `dispatch` with a replacement selector, revalidating it against the common ABI and
   available strategies. Used to bake an offline tuning result into otherwise identical IR."
  [dispatch selector]
  (validate! (assoc (validate! dispatch) :selector selector)))

(defn- runtime-number
  [value]
  (if (and (map? value) (contains? value :value)) (:value value) value))

(defn- compare-value?
  [op actual expected]
  (case op
    :< (< actual expected)
    :<= (<= actual expected)
    := (= actual expected)
    :>= (>= actual expected)
    :> (> actual expected)))

(defn select-alternative
  "Select an executable alternative from concrete ABI-ordered values.

   `override` is nil/:auto for data-driven selection or an explicit alternative strategy. The
   runtime threshold selector is intentionally generic: it knows only a compiler argument and a
   numeric crossover, not which model operation produced them."
  ([dispatch runtime-arguments] (select-alternative dispatch runtime-arguments nil))
  ([dispatch runtime-arguments override]
   (let [dispatch (validate! dispatch)]
     (if (and override (not= :auto override))
       (alternative dispatch override)
       (let [{:keys [kind argument expression threshold at-least otherwise ranges below]}
             (:selector dispatch)
             common-arguments (kexec/arguments (default-alternative dispatch))
             indexes (keep-indexed (fn [index value] (when (= argument value) index))
                                   common-arguments)]
         (when-not (= (count common-arguments) (count runtime-arguments))
           (throw (ex-info "kernel dispatch runtime argument count mismatch"
                           {:id (:id dispatch) :expected (count common-arguments)
                            :actual (count runtime-arguments)})))
         (when (contains? #{:runtime-scalar-threshold :runtime-scalar-ranges} kind)
           (when-not (= 1 (count indexes))
             (throw (ex-info "kernel dispatch selector argument must have one runtime position"
                             {:id (:id dispatch) :argument argument
                              :indexes (vec indexes)}))))
         (let [environment (zipmap common-arguments (mapv runtime-number runtime-arguments))
               value (case kind
                       :runtime-expression-threshold
                       (klaunch/resolve-expression environment expression)
                       :runtime-expression-cases nil
                       (runtime-number (nth runtime-arguments (first indexes))))]
           (when (and (not= :runtime-expression-cases kind) (not (number? value)))
             (throw (ex-info "kernel dispatch selector requires a numeric runtime scalar"
                             {:id (:id dispatch) :argument argument :value value})))
           (alternative
            dispatch
            (case kind
              :runtime-scalar-threshold
              (if (>= (double value) (double threshold)) at-least otherwise)

              :runtime-expression-threshold
              (if (>= (double value) (double threshold)) at-least otherwise)

              :runtime-expression-cases
              (or (some (fn [{:keys [expression op value strategy]}]
                          (when (compare-value?
                                 op
                                 (klaunch/resolve-expression environment expression)
                                 value)
                            strategy))
                        (get-in dispatch [:selector :cases]))
                  (get-in dispatch [:selector :default]))

              :runtime-scalar-ranges
              (reduce (fn [strategy range]
                        (if (>= (double value) (double (:at-least range)))
                          (:strategy range)
                          (reduced strategy)))
                      below ranges)))))))))
