(ns raster.ad.normalized-map-runtime
  "Opt-in CPU execution of checked maps, ordered map→sum and indexed-map AD bundles.
  This owns no compiler cache, derivative rules, source importer or GPU session."
  (:require [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(defn- fail! [reason data]
  (throw (ex-info "Arguments are outside the checked map AD runtime boundary"
                  (assoc data :reason reason))))

(defn- executable [program source-ns indexed?]
  (let [projection ((if indexed? route/materialize-host-indexed-map
                       (if (= 2 (count (soac/equations program)))
                         route/materialize-host-map-reduce route/materialize-host-map)) program)]
    (assoc projection :execute
           (binding [*ns* source-ns]
             (eval (list 'fn (:parameters projection) (:body projection)))))))

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

(defn prepare
  "Compile existing primal/linear/transpose host projections once.

  Returns {:inputs ordered-source-ids :vjp fn :jvp fn}. (:vjp prepared) takes
  an ordered argument vector and returns [output pullback]; pullback takes an
  explicit dy array (map) or exactly typed scalar seed (sum), and returns positional cotangents
  (nil for excluded arrays and scalar inputs).
  (:jvp prepared) takes the argument vector and an explicit dx array and returns
  [output tangent], both scalar for a sum. No data or index array may be mutated concurrently
  with primal execution.
  Pullbacks privately own fresh coefficient snapshots, including required excluded-array
  elements, not any mutable source array. Exactly one source array is active.
  Indexed maps privately snapshot Int/Long indices after checking every bound in its original
  width. Their explicit data tangent has data capacity, their seed has output extent, and each
  pullback owns a new typed-zero data gradient. No source data, indices or returned output is saved.
  This is not implicit-seed value+grad, device execution, or general loop AD."
  [bundle]
  (when (and (:gather-programs bundle)
             (or (contains? bundle :map-programs) (contains? bundle :sum-programs)))
    (fail! :ad-map-runtime-composition {:facets (set (keys (select-keys bundle [:map-programs :sum-programs :gather-programs])))}))
  (normalized/validate! bundle)
  (when-not (or (:map-programs bundle) (:gather-programs bundle))
    (fail! :ad-map-runtime-bundle {}))
  (let [indexed? (contains? bundle :gather-programs)
        sum? (contains? bundle :sum-programs)
        programs (cond indexed? (:gather-programs bundle) sum? (:sum-programs bundle) :else (:map-programs bundle))
        {:keys [interface]} (if indexed? programs (:map-programs bundle))
        {:keys [source primal linear transpose required-zero-gradient]} programs
        seed-id (if sum? (get-in bundle [:sum-programs :interface :seed]) (:cotangent-input interface))
        {:keys [arrays captures attributes]} (soac/operation-parts (first (soac/equations source)))
        inputs (:inputs (soac/facts source))
        values (:values (soac/facts source))
        active (if indexed? (first captures) (:active-array interface))
        dt (:dtype (get values active))
        extent-id (:extent attributes)
        capacity-id (when indexed? (first (:shape (get values active))))
        primal (executable primal (:source-ns bundle) indexed?)
        push (executable linear (:source-ns bundle) indexed?)
        pull (executable transpose (:source-ns bundle) indexed?)
        context
        (fn [arguments]
          (normalized/validate! bundle)
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
                {:environment (if (integer? capacity-id) environment
                                  (assoc environment capacity-id
                                         (case (:dtype (get values capacity-id))
                                           :int (int capacity) :long (long capacity))))
                 :extent extent :capacity capacity})
              (do
                (doseq [id arrays] (array! (get environment id) dt extent id))
                (doseq [id captures]
                  (when-not (case (:dtype (get values id))
                              :float (instance? Float (get environment id))
                              :double (instance? Double (get environment id)) false)
                    (fail! :ad-map-runtime-capture {:input id :dtype (:dtype (get values id))})))
                {:environment environment :extent extent}))))
        acquire
        (fn [environment]
          (let [results (invoke primal environment)
                ;; Keep only immutable scalar captures/extent and freshly owned
                ;; residual outputs. No mutable source array or returned y is a pullback input.
                residual-ids (if indexed? [(:index-snapshot interface)] (vals (:residual-arrays interface)))
                saved (merge (apply dissoc environment (if indexed? (concat arrays captures) arrays))
                             (select-keys (zipmap (:outputs primal) results) residual-ids))]
            (when indexed?
              (let [extent (if (integer? extent-id) extent-id (get environment extent-id))]
                (array! (first results) dt extent :primal-output)
                (array! (get saved (:index-snapshot interface))
                        (:dtype (get values (first arrays))) extent :index-snapshot)))
            [(first results) saved]))]
    {:inputs inputs
     :vjp
     (fn [arguments]
       (let [{:keys [environment extent capacity]} (context arguments)
             [output saved] (acquire environment)]
         [output
          (fn [dy]
            (normalized/validate! bundle)
            (if sum? (scalar! dy dt :cotangent) (array! dy dt extent :cotangent))
            (let [seeded (assoc saved seed-id dy)
                  seeded (if indexed?
                           (assoc seeded (:value required-zero-gradient)
                                  (case dt :float (float-array capacity) :double (double-array capacity)))
                           seeded)
                  gradient (first (invoke pull seeded))]
              (when indexed?
                (array! gradient dt capacity :gradient)
                (when-not (identical? gradient (get seeded (:value required-zero-gradient)))
                  (fail! :ad-map-runtime-physical-owner {:output (:gradient-output interface)})))
              (mapv #(when (= active %) gradient) inputs)))]))
     :jvp
     (fn [arguments dx]
       (let [{:keys [environment extent capacity]} (context arguments)]
         (array! dx dt (if indexed? capacity extent) :tangent)
         (let [[output saved] (acquire environment)]
           [output (first (invoke push (assoc saved (:tangent-input interface) dx)))])))}))
