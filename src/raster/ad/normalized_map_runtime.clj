(ns raster.ad.normalized-map-runtime
  "Opt-in CPU execution of checked maps, ordered map→sum and indexed-map AD bundles.
  This owns no compiler cache, derivative rules, source importer or GPU session."
  (:require [raster.ad.normalized-scalar :as normalized]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.scalar.effects :as effects]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(defn- fail! [reason data]
  (throw (ex-info "Arguments are outside the checked map AD runtime boundary"
                  (assoc data :reason reason))))

(defn- compile-projection [projection source-ns]
  (assoc projection :execute
           (binding [*ns* source-ns]
             (eval (list 'fn (:parameters projection) (:body projection))))))

(defn- invoke [executable values]
  (apply (:execute executable) (mapv values (concat (:inputs executable) (:physical-inputs executable)))))

(defn- array! [value dt extent role]
  (when-not (and (case dt :float (instance? float/1 value)
                            :double (instance? double/1 value)
                            :int (instance? int/1 value) :long (instance? long/1 value) false)
                 (= extent (count value)))
    (fail! :ad-map-runtime-array {:role role :dtype dt :extent extent
                                 :actual-class (some-> value class .getName)}))
  value)

(defn- scalar! [value dt role]
  (when-not (case dt :float (instance? Float value) :double (instance? Double value) false)
    (fail! :ad-map-runtime-scalar {:role role :dtype dt :actual-class (some-> value class .getName)}))
  value)

(defn- integral-control! [value dt role]
  (when-not (case dt :int (instance? Integer value) :long (instance? Long value) false)
    (fail! :ad-map-runtime-extent-type
           {:extent value :role role :dtype dt :actual-class (some-> value class .getName)}))
  value)

(defn- direct-dimension-id
  "Return the retained value ID for a direct dimension, nil for a literal, or refuse.
   This deliberately does not interpret general shape/index expressions."
  [dimension role]
  (cond
    (integer? dimension) nil
    (symbol? dimension) dimension
    (and (seq? dimension) (= 'value (first dimension)) (= 2 (count dimension)))
    (second dimension)
    :else (fail! :ad-map-runtime-shape {:role role :dimension dimension})))

(defn- direct-dimension!
  [dimension environment source-values role]
  (if (integer? dimension)
    (do
      (when-not (<= 0 dimension Integer/MAX_VALUE)
        (fail! :ad-map-runtime-extent {:extent dimension :role role}))
      dimension)
    (let [id (direct-dimension-id dimension role)
          abstract (get source-values id)]
      (when-not (and (map? abstract) (= [] (:shape abstract))
                     (contains? #{:int :long} (:dtype abstract))
                     (contains? environment id))
        (fail! :ad-map-runtime-shape
               {:role role :dimension dimension :value-id id :abstract-value abstract}))
      (let [value (integral-control! (get environment id) (:dtype abstract) id)]
        (when-not (<= 0 value Integer/MAX_VALUE)
          (fail! :ad-map-runtime-extent {:extent value :role role}))
        value))))

(defn- source-value!
  [value abstract-value environment source-values role]
  (let [shape (:shape abstract-value) dt (:dtype abstract-value)]
    (when-not (vector? shape)
      (fail! :ad-map-runtime-shape {:role role :abstract-value abstract-value}))
    (cond
      (= [] shape)
      (case dt
        (:float :double) (scalar! value dt role)
        (:int :long) (integral-control! value dt role)
        (fail! :ad-map-runtime-shape {:role role :abstract-value abstract-value}))

      (= 1 (count shape))
      (array! value dt (direct-dimension! (first shape) environment source-values role) role)

      :else (fail! :ad-map-runtime-shape {:role role :abstract-value abstract-value}))))

(defn- differential-value!
  [value abstract-value environment source-values role]
  (when-not (contains? #{:float :double} (:dtype abstract-value))
    (fail! :ad-map-runtime-shape {:role role :abstract-value abstract-value}))
  (source-value! value abstract-value environment source-values role))

(defn- prepare-validated-owner
  "Compile a freshly validated canonical owner's host projections once.

  Returns {:inputs ordered-source-ids :vjp fn :jvp fn}. (:vjp prepared) takes
  an ordered argument vector and returns [output pullback]; pullback takes an
  explicit dy array (map) or exactly typed scalar seed (sum), and returns positional cotangents
  (nil for excluded arrays and scalar inputs).
  (:jvp prepared) takes the argument vector and, for one active input, its explicit tangent.
  Multiple active map inputs require an exact ordered vector of tangents in active-port order.
  It returns
  [output tangent], both scalar for a sum. No data or index array may be mutated concurrently
  with primal execution.
  Pullbacks return positional source-input cotangents and privately own fresh coefficient snapshots,
  including required excluded-array
  elements, not any mutable source array. Scalar-capture
  cotangents use the source's ordered sum rather than a new reduction policy.
  Indexed maps privately snapshot Int/Long indices after checking every bound in its original
  width. Their explicit data tangent has data capacity, their seed has output extent, and each
  pullback owns a new typed-zero data gradient. No source data, indices or returned output is saved.
  This is not implicit-seed value+grad, device execution, or general loop AD."
  [owner]
  (let [expected (normalized/revisions)
        indexed? (= :checked-scalar-indexed-map (get-in owner [:derivation :kind]))
        {:keys [source execution-source primal linear transpose interface]} owner
        execution-source (or execution-source source)
        seed-id (get-in interface [:cotangent-input :id])
        inputs (mapv :id (:inputs interface))
        values (:values (soac/facts source))
        input-ports (:inputs interface)
        ;; Only the indexed adapter still needs the checked operation's physical
        ;; index/output geometry. Nonindexed geometry comes from canonical ports.
        parts (when indexed? (mapv soac/operation-parts (soac/equations execution-source)))
        operand-arrays (set (mapcat :arrays parts))
        arrays (filterv operand-arrays inputs)
        {:keys [lambda]} (first parts)
        active-inputs (vec (:active-inputs interface))
        active (first active-inputs)
        dt (:dtype (get values active))
        capacity-id (when indexed? (first (:shape (get values active))))
        gradient-ids (mapv :id (:cotangent-results interface))
        projections (mapv route/materialize-host-program [primal linear transpose])
        executable-inputs (set (:inputs (first projections)))
        executable-ports (filterv #(contains? executable-inputs (:id %)) input-ports)
        array-ports (filterv #(= 1 (count (get-in % [:value :shape]))) executable-ports)
        capture-ports (filterv #(and (= [] (get-in % [:value :shape]))
                                    (contains? #{:float :double} (get-in % [:value :dtype])))
                               executable-ports)
        ;; This dimension preserves extent-before-array diagnostics only. It is
        ;; never a common extent for the independently checked array ports.
        primary-dimension (when-not indexed? (first (get-in (first array-ports) [:value :shape])))
        extent-id (if indexed? (get-in (first parts) [:attributes :extent])
                      (if (integer? primary-dimension) primary-dimension
                          (direct-dimension-id primary-dimension (:id (first array-ports)))))
        source-dependencies
        (cond-> (normalized/execution-sources (get-in owner [:derivation :bundle]))
          indexed?
          ;; Admission requires a single direct read and EMPTY locals. The scalar identity lane
          ;; omits this executable read/alias, so capture its original checked lambda in addition
          ;; to all three raw scalar programs, not through another importer.
          (conj (let [{:keys [parameters body-results]} (soac/lambda-parts lambda)]
                  {:parameters parameters :body body-results})))
        guard! (normalized/execution-guard (:source-ns owner) expected
                                          (into projections source-dependencies))
        ;; Bind the captured effective callees to the existing selected-call proof,
        ;; rather than blessing a mutation between adaptation and identity capture.
        residual-plan (canonical/execution-residual-plan owner)
        owned-residuals (:owned-arrays residual-plan)
        physical-bindings (:physical-bindings residual-plan)
        shape-ports (concat input-ports (:tangent-inputs interface)
                            [(:primal-result interface) (:tangent-result interface)
                             (:cotangent-input interface)]
                            (:cotangent-results interface)
                            (vals owned-residuals))
        shape-value-ids
        (into #{}
              (keep (fn [port]
                      (let [shape (get-in port [:value :shape])]
                        (when-not (vector? shape)
                          (fail! :ad-map-runtime-shape
                                 {:role (:id port) :abstract-value (:value port)}))
                        (case (count shape)
                          0 nil
                          1 (direct-dimension-id (first shape) (:id port))
                          (fail! :ad-map-runtime-shape
                                 {:role (:id port) :abstract-value (:value port)})))))
              shape-ports)
        retained-source-ids
        (into shape-value-ids
              (map #(get-in % [:producer :input]) (vals (:immutable-scalars residual-plan))))
        index-snapshot (when indexed? (:id (get owned-residuals (first arrays))))
        required-zero-gradient (when indexed?
                                 (some (fn [[_ binding]]
                                         (when (= (first gradient-ids) (get-in binding [:consumer :result]))
                                           (:requirement binding)))
                                       physical-bindings))
        _ (guard!)
        compiled (mapv #(compile-projection % (:source-ns owner)) projections)
        _ (guard!)
        [primal push pull] compiled
        run (fn [executable environment]
              (guard!)
              (let [result (invoke executable environment)]
                (guard!)
                result))
        context
        (fn [arguments]
          (guard!)
          (when-not (and (vector? arguments) (= (count inputs) (count arguments)))
            (fail! :ad-map-runtime-arity {:inputs inputs}))
          (let [environment (zipmap inputs arguments)
                extent (if (integer? extent-id) extent-id (get environment extent-id))]
            (when-not (and (integer? extent) (<= 0 extent Integer/MAX_VALUE))
              (fail! :ad-map-runtime-extent {:extent extent}))
            (when (and (not (integer? extent-id))
                       (not (case (:dtype (get values extent-id))
                              :long (instance? Long extent)
                              :int (instance? Integer extent) false)))
              (fail! :ad-map-runtime-extent-type {:extent extent}))
            (if indexed?
              (let [data (get environment active)
                    _ (when-not (case dt :float (instance? float/1 data) :double (instance? double/1 data) false)
                        (fail! :ad-map-runtime-array {:role active :dtype dt}))
                    capacity (count data)
                    expected (if (integer? capacity-id) capacity-id
                                 (if (contains? environment capacity-id) (get environment capacity-id) capacity))
                    _ (when-not (= capacity expected)
                        (fail! :ad-map-runtime-capacity {:capacity capacity :expected expected}))
                    index-id (first arrays) it (:dtype (get values index-id))
                    indices (array! (get environment index-id) it extent index-id)]
                ;; Inspect the original integral width before any cast or primal allocation.
                (dotimes [i extent]
                  (let [index (if (= :int it) (aget ^ints indices i) (aget ^longs indices i))]
                    (when-not (< -1 index capacity)
                      (fail! :ad-map-runtime-index {:position i :index index :capacity capacity}))))
                (let [environment (if (integer? capacity-id) environment
                                    (assoc environment capacity-id
                                           (case (:dtype (get values capacity-id))
                                             :int (int capacity) :long (long capacity))))]
                  (doseq [{:keys [id value]} input-ports]
                    (source-value! (get environment id) value environment values id))
                  {:environment environment
                   :extent extent :capacity capacity}))
              (do
                (doseq [{:keys [id value]} array-ports]
                  (source-value! (get environment id) value environment values id))
                (doseq [{:keys [id value]} capture-ports]
                  (when-not (case (:dtype value)
                              :float (instance? Float (get environment id))
                              :double (instance? Double (get environment id)) false)
                    (fail! :ad-map-runtime-capture {:input id :dtype (:dtype value)})))
                (doseq [{:keys [id value]} input-ports]
                  (source-value! (get environment id) value environment values id))
                {:environment environment :extent extent}))))
        acquire
        (fn [environment]
          (let [results (run primal environment)
                output-ids (:outputs primal)
                _ (when-not (= (count output-ids) (count results))
                    (fail! :ad-map-runtime-results
                           {:phase :primal :outputs output-ids :result-count (count results)}))
                produced (zipmap output-ids results)
                _ (source-value! (first results) (get-in interface [:primal-result :value])
                                 environment values :primal-output)
                _ (doseq [[_ {:keys [id value]}] owned-residuals]
                    (source-value! (get produced id) value environment values id))
                ;; Keep only immutable scalar captures/extent and freshly owned
                ;; residual outputs. No mutable source array or returned y is a pullback input.
                residual-ids (map :id (vals owned-residuals))
                saved (merge (select-keys environment retained-source-ids)
                             (select-keys produced residual-ids))]
            (when indexed?
              (let [extent (if (integer? extent-id) extent-id (get environment extent-id))]
                (array! (first results) dt extent :primal-output)
                (array! (get saved index-snapshot)
                        (:dtype (get values (first arrays))) extent :index-snapshot)))
            [(first results) saved]))]
    {:inputs inputs
     :vjp
     (fn [arguments]
       (let [{:keys [environment capacity]} (context arguments)
             [output saved] (acquire environment)]
         [output
          (fn [dy]
            (guard!)
            (differential-value! dy (get-in interface [:cotangent-input :value])
                                 saved values :cotangent)
            (let [seeded (assoc saved seed-id dy)
                  seeded (if indexed?
                           (assoc seeded (:value required-zero-gradient)
                                  (case dt :float (float-array capacity) :double (double-array capacity)))
                           seeded)
                  gradients (vec (run pull seeded))]
              (when-not (= (count active-inputs) (count (:cotangent-results interface))
                           (count gradients))
                (fail! :ad-map-runtime-gradients
                       {:active-inputs active-inputs
                        :cotangent-results (mapv :id (:cotangent-results interface))
                        :gradient-count (count gradients)}))
              (doseq [[active gradient port] (map vector active-inputs gradients
                                                   (:cotangent-results interface))]
                (differential-value! gradient (:value port) seeded values active))
              (when indexed?
                (array! (first gradients) dt capacity :gradient)
                (when-not (identical? (first gradients) (get seeded (:value required-zero-gradient)))
                  (fail! :ad-map-runtime-physical-owner {:output (first gradient-ids)})))
              (guard!)
              (let [by-source (zipmap active-inputs gradients)]
                (mapv by-source inputs))))]))
     :jvp
     (fn [arguments dx]
       (let [{:keys [environment]} (context arguments)
             tangents (if (= 1 (count active-inputs)) [dx] dx)]
         (when-not (and (vector? tangents)
                        (= (count active-inputs) (count (:tangent-inputs interface))
                           (count tangents)))
           (fail! :ad-map-runtime-tangents
                  {:active-inputs active-inputs :tangent-count (when (vector? tangents)
                                                                 (count tangents))}))
         (doseq [[active tangent port] (map vector active-inputs tangents
                                            (:tangent-inputs interface))]
           (differential-value! tangent (:value port) environment values active))
         (let [[output saved] (acquire environment)
               push-environment (into saved (map (fn [port tangent]
                                                   [(:id port) tangent])
                                                 (:tangent-inputs interface) tangents))
               tangent-results (vec (run push push-environment))
               _ (when-not (= 1 (count tangent-results))
                   (fail! :ad-map-runtime-results
                          {:phase :linear :result-count (count tangent-results)}))
               tangent (first tangent-results)]
           (differential-value! tangent (get-in interface [:tangent-result :value])
                                push-environment values :tangent-output)
           [output tangent])))}))

(defn prepare-owner
  "Validate a canonical owner and prepare CPU execution through the common private consumer.
   Returns ordered :inputs, :vjp and :jvp closures. Tangents/seeds use their retained scalar
   or array spaces; excluded gradient slots are nil. Saved pullbacks own coefficient/index
   snapshots, not caller arrays. Concurrent input mutation is forbidden. Indexed gradients
   receive fresh exclusive typed-zero owners. Epoch and executable identities are fenced.
   No default AD migration, implicit seed, device execution or general equation composition."
  [owner]
  (effects/init-raster-context!)
  (prepare-validated-owner (canonical/validate! owner)))

(defn prepare
  "Compatibility entry for checked derivation evidence. Canonical adaptation owns family
   admission; prepare-owner owns host execution. See prepare-owner for value/lifetime contracts."
  [bundle]
  (effects/init-raster-context!)
  ;; Adaptation has just performed independent canonical validation. Reuse the private
  ;; consumer, not a skip flag: both public entries retain the post-capture proof/fence.
  (prepare-validated-owner (canonical/adapt bundle)))
