(ns raster.compiler.passes.parallel.scheduled-equation-graph
  "Derive one target-neutral KernelGraph from a scheduled TypedSOAC program.

   This is the shared graph boundary for an ordinary equation and for one iteration of structured
   control. It consumes only the retained functional algorithm, ordered SegOps, and AbstractValue
   contracts; source spelling and operation-family names play no role."
  (:require [clojure.set :as set]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as soac]))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :pass :scheduled-equation-graph))))

(defn- value-elements
  [value]
  (let [dimension-value (fn [dimension]
                          ;; AbstractValue uses `(value id)` to distinguish a compound stable
                          ;; value ID from shape syntax. KernelGraph owns explicit integer
                          ;; expressions, so remove only that marker at this physical boundary.
                          (if (and (seq? dimension)
                                   (= 'value (first dimension))
                                   (= 2 (count dimension)))
                            (second dimension)
                            dimension))
        shape (mapv dimension-value (:shape value))]
    (cond
      (empty? shape) 1
      (= 1 (count shape)) (first shape)
      :else (apply launch/product shape))))

(defn- physical-outputs
  [algorithm]
  (let [facts (soac/facts algorithm)
        producers (into {}
                        (mapcat (fn [equation]
                                  (map vector (nth equation 2)
                                       (soac/physical-results facts equation))))
                        (soac/equations algorithm))]
    (mapv (fn [result]
            (or (get producers result)
                (fail! :scheduled-equation-result-storage
                       "a scheduled result has no physical storage identity"
                       {:result result})))
          (soac/outputs algorithm))))

(defn- external-inputs
  [operations]
  (:inputs
   (reduce (fn [{:keys [initialized] :as state} operation]
             (let [reads (segop/operation-inputs operation)
                   writes (segop/operation-outputs operation)]
               (-> state
                   (update :inputs set/union (set/difference reads initialized))
                   (update :initialized set/union writes))))
           {:inputs #{} :initialized #{}}
           operations)))

(defn- storage-spec
  [values id]
  (let [value (get values id)]
    (when-not value
      (fail! :scheduled-equation-storage-value
             "scheduled storage lacks an AbstractValue"
             {:value id}))
    {:dtype (:dtype value)
     :elements (value-elements value)
     :memory-space (or (:memory-space value) :device)}))

(defn- algorithm-boundary?
  [equation algorithm]
  (and (= algorithm (soac/validate! algorithm))
       (= (:operands equation) (:inputs (soac/facts algorithm)))
       (= (:results equation) (soac/outputs algorithm))))

(defn- ordered-distinct
  [values]
  (reduce (fn [result value]
            (if (some #(= value %) result) result (conj result value)))
          [] values))

(defn- public-scalars
  [scheduled operations]
  (let [required (reduce set/union #{} (map segop/operation-scalars operations))
        ordered (ordered-distinct
                 (concat (filter required (:inputs scheduled))
                         (sort-by pr-str (remove (set (:inputs scheduled)) required))))
        values (:values scheduled)]
    (mapv (fn [id]
            (let [value (get values id)]
              (when-not value
                (fail! :scheduled-equation-scalar-value
                       "scheduled scalar lacks an AbstractValue"
                       {:value id}))
              (when-not (empty? (:shape value))
                (fail! :scheduled-equation-scalar-shape
                       "scheduled scalar dependency must have scalar shape"
                       {:value id :shape (:shape value)}))
              (graph/scalar id (:dtype value))))
          ordered)))

(defn make
  "Build a verified graph from `algorithm` and its fully scheduled SegOp ParallelProgram.

   Optional facts describe the enclosing control/equation context without changing dataflow."
  ([algorithm scheduled] (make algorithm scheduled {}))
  ([algorithm scheduled {:keys [effects provenance attributes]
                         :or {provenance {} attributes {}}}]
   (let [algorithm (soac/validate! algorithm)
         scheduled (program/validate! scheduled segop/segop-node? algorithm-boundary?)
         _ (when-not (= :segop (:dialect scheduled))
             (fail! :scheduled-equation-dialect
                    "KernelGraph derivation requires a fully scheduled :segop program"
                    {:dialect (:dialect scheduled)}))
         retained-equations (vec (mapcat (comp soac/equations :algorithm)
                                         (:equations scheduled)))
         _ (when-not (and (= (soac/equations algorithm) retained-equations)
                          (= (:inputs (soac/facts algorithm)) (:inputs scheduled))
                          (= (soac/outputs algorithm) (:outputs scheduled)))
             (fail! :scheduled-equation-algorithm
                    "scheduled equations or program boundary differ from the retained algorithm"
                    {:algorithm-inputs (:inputs (soac/facts algorithm))
                     :scheduled-inputs (:inputs scheduled)
                     :algorithm-outputs (soac/outputs algorithm)
                     :scheduled-outputs (:outputs scheduled)}))
         operations (vec (mapcat :operations (:equations scheduled)))
         _ (when (empty? operations)
             (fail! :scheduled-equation-empty
                    "a KernelGraph requires at least one scheduled operation" {}))
         _ (when (some #(get-in % [:attributes :host-only]) (:equations scheduled))
             (fail! :scheduled-equation-host-scalar
                    "host scalar equations require an explicit host schedule" {}))
         inputs (external-inputs operations)
         outputs (set (physical-outputs algorithm))
         operation-values (reduce set/union #{}
                                  (map #(set/union (segop/operation-inputs %)
                                                   (segop/operation-outputs %))
                                       operations))
         temporary-ids (set/difference operation-values inputs outputs)
         values (:values scheduled)
         scalars (public-scalars scheduled operations)
         buffer-specs (into {}
                            (map (fn [id] [id (storage-spec values id)]))
                            (set/union inputs outputs temporary-ids))]
     (graph/from-segops
      operations
      {:inputs inputs
       :outputs outputs
       :temporaries (select-keys buffer-specs temporary-ids)
       :scalars scalars
       :buffer-specs buffer-specs
       :dtype (:dtype (first (vals buffer-specs)))
       :effects (or effects {:semantic (:effects (soac/facts algorithm))})
       :provenance (merge {:source-dialect :typed-soac
                           :algorithm-dialect :typed-soac
                           :schedule-dialect :segop}
                          provenance)
       :attributes attributes}))))
