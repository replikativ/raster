(ns raster.ad.normalized-map-runtime
  "Opt-in CPU execution of an already checked, single homogeneous map AD bundle.
  This owns no compiler cache, derivative rules, source importer or GPU session."
  (:require [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(defn- fail! [reason data]
  (throw (ex-info "Arguments are outside the checked map AD runtime boundary"
                  (assoc data :reason reason))))

(defn- executable [program source-ns]
  (let [projection (route/materialize-host-map program)]
    (assoc projection :execute
           (binding [*ns* source-ns]
             (eval (list 'fn (:parameters projection) (:body projection)))))))

(defn- invoke [executable values]
  (apply (:execute executable) (mapv values (:inputs executable))))

(defn- array! [value dt extent role]
  (when-not (and (case dt :float (instance? float/1 value)
                            :double (instance? double/1 value) false)
                 (= extent (count value)))
    (fail! :ad-map-runtime-array {:role role :dtype dt :extent extent
                                 :actual-class (some-> value class .getName)}))
  value)

(defn prepare
  "Compile existing primal/linear/transpose host projections once.

  Returns {:inputs ordered-source-ids :vjp fn :jvp fn}. (:vjp prepared) takes
  an ordered argument vector and returns [output pullback]; pullback takes an
  explicit dy array and returns positional cotangents (nil for excluded arrays and scalar inputs).
  (:jvp prepared) takes the argument vector and an explicit dx array and returns
  [output tangent]. Data must not be mutated concurrently with primal execution.
  Pullbacks privately own fresh coefficient snapshots, including required excluded-array
  elements, not any mutable source array. Exactly one source array is active.
  This is not implicit-seed value+grad, device execution, or general loop AD."
  [bundle]
  (normalized/validate! bundle)
  (when-not (:map-programs bundle)
    (fail! :ad-map-runtime-bundle {}))
  (let [{:keys [source interface primal linear transpose]} (:map-programs bundle)
        {:keys [arrays captures attributes]} (soac/operation-parts (first (soac/equations source)))
        inputs (:inputs (soac/facts source))
        values (:values (soac/facts source))
        active (:active-array interface)
        dt (:dtype (get values active))
        extent-id (:extent attributes)
        primal (executable primal (:source-ns bundle))
        push (executable linear (:source-ns bundle))
        pull (executable transpose (:source-ns bundle))
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
            (doseq [id arrays]
              (array! (get environment id) dt extent id))
            (doseq [id captures]
              (when-not (case (:dtype (get values id))
                          :float (instance? Float (get environment id))
                          :double (instance? Double (get environment id)) false)
                (fail! :ad-map-runtime-capture {:input id :dtype (:dtype (get values id))})))
            {:environment environment :extent extent}))
        acquire
        (fn [environment]
          (let [results (invoke primal environment)
                ;; Keep only immutable scalar captures/extent and freshly owned
                ;; residual outputs. No mutable source array or returned y is a pullback input.
                residual-ids (vals (:residual-arrays interface))
                saved (merge (apply dissoc environment arrays)
                             (select-keys (zipmap (:outputs primal) results) residual-ids))]
            [(first results) saved]))]
    {:inputs inputs
     :vjp
     (fn [arguments]
       (let [{:keys [environment extent]} (context arguments)
             [output saved] (acquire environment)]
         [output
          (fn [dy]
            (normalized/validate! bundle)
            (array! dy dt extent :cotangent)
            (let [gradient (first (invoke pull (assoc saved (:cotangent-input interface) dy)))]
              (mapv #(when (= active %) gradient) inputs)))]))
     :jvp
     (fn [arguments dx]
       (let [{:keys [environment extent]} (context arguments)]
         (array! dx dt extent :tangent)
         (let [[output saved] (acquire environment)]
           [output (first (invoke push (assoc saved (:tangent-input interface) dx)))])))}))
