(ns raster.compiler.passes.parallel.typed-soac-initialization
  "Schedule fresh-storage initialization as ordinary typed equations.

   Allocation contracts come from the frontend, not an allocator registry here. This pass is
   for resident execution, where allocation itself does not implement the language's zeros and
   a replay must reestablish them. Native fresh-array execution needs no such schedule."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.extent-proof :as extent-proof]
            [raster.compiler.ir.soac-dialect :as dialect]))

(defn- fail! [message data]
  (throw (ex-info message (assoc data :reason :typed-soac-initialization-contract))))

(defn- physical-id [facts value]
  (let [aliases (into {} (mapcat :aliases) (vals (:equations facts)))]
    (loop [id value seen #{}]
      (if-let [target (let [target (get aliases id)] (when (not= target id) target))]
        (do (when (contains? seen id)
              (fail! "initialization encountered a cyclic physical alias" {:value id}))
            (recur target (conj seen id)))
        id))))

(defn- physical-inputs [facts equation]
  (set (map #(physical-id facts %) (dialect/operation-inputs equation))))

(defn- touches? [facts destination equation]
  (or (contains? (physical-inputs facts equation) destination)
      (some #(= destination (physical-id facts (:destination %)))
            (dialect/result-storage facts (second equation)))))

(defn- plain-storage? [value]
  (and (= {:kind :plain} (:representation value)) (nil? (:logical-layout value))))

(defn- full-overwrite? [facts extent-environment {:keys [destination extent]} equation]
  ;; These functional operations produce their entire validated logical result shape.
  ;; Indexed/guarded effect maps and scatter do not have that guarantee.
  (and (contains? '#{map stencil contract segmented-reduce product-reduce
                     segmented-fold-map scan} (dialect/operation-kind equation))
       (not (contains? (physical-inputs facts equation) destination))
       (some (fn [[result storage]]
               (and (= destination (physical-id facts (:destination storage)))
                    (= :write (:access storage))
                    (plain-storage? (get-in facts [:values result]))
                    (plain-storage? (get-in facts [:values destination]))
                    (extent-proof/same-volume? extent-environment extent
                                              (get-in facts [:values result :shape]))))
             (map vector (nth equation 2) (dialect/result-storage facts (second equation))))))

(defn- fresh-symbol [used prefix]
  (first (remove used (map #(symbol (str prefix %)) (range)))))

(defn- initialization-site! [facts {:keys [destination source-binding-id] :as allocation} consumer]
  (let [consumer-facts (get-in facts [:equations (second consumer)])
        placements (or (seq (keys (get-in consumer-facts [:attributes :fusion/constituents])))
                       [(get-in consumer-facts [:provenance :source-binding-id])])]
    (when-not (every? integer? placements)
      (fail! "initialization requires an ordered analyzed-source consumer"
             {:allocation allocation :consumer (second consumer) :placements placements}))
    (let [placement (apply max placements)
          reads (filter #(and (some (fn [value] (= destination (physical-id facts value)))
                                   (:values %))
                              (< source-binding-id (:source-binding-id %) placement))
                        (get-in facts [:attributes :host-read-sites]))
          host-sites (set (get-in facts [:attributes :host-binding-ids]))
          native? (and (seq reads)
                       (every? #(and (contains? host-sites (:source-binding-id %))
                                     (< (:source-binding-id %) (apply min placements))) reads))]
      (when-not (< source-binding-id placement)
        (fail! "initialization consumer must follow allocation"
               {:allocation allocation :consumer (second consumer) :placement placement}))
      (doseq [read reads :when (not native?)]
        (fail! "a host observation precedes the first resident initialization site"
               {:allocation allocation :host-read read :placement placement}))
      ;; Host-controlled consumers retain the native allocator and its initial contents.
      ;; Resident extraction rejects these buffer-reading host bindings; staging uploads the
      ;; resulting host array. A later fill would erase the host's writes. An observation
      ;; between fused constituents is not a native-first use and still fails closed.
      {:placement placement :native? (boolean native?)})))

(defn- integral-extent? [facts extent]
  (or (and (integer? extent) (not (neg? extent)))
      (let [value (get-in facts [:values extent])]
        (and (dialect/value-id? extent) (= :tensor (:kind value))
             (= [] (:shape value)) (contains? #{:int :long} (:dtype value))))))

(defn- initializer [facts {:keys [destination extent dtype source-binding-id] :as allocation}
                    placement]
  (when-not (integral-extent? facts extent)
    (fail! "resident initialization requires a typed allocation extent and scalar element type"
           {:allocation allocation}))
  (let [scalar-tag (try (dtype/scalar-tag-for-dtype dtype)
                       (catch clojure.lang.ExceptionInfo e
                         (fail! "resident initialization requires a scalar-emittable element type"
                                {:allocation allocation :cause (ex-data e)})))
        used (set/union (set (keys (:values facts))) (set (keys (:equations facts)))
                        (set (get-in facts [:attributes :source-bindings])))
        result (fresh-symbol used "rstr_initialized_")
        id (fresh-symbol (conj used result) "rstr_initialization_equation_")
        index (fresh-symbol (conj used result) "rstr_zero_index_")
        equation (list '= id [result]
                       (list 'map {:index index :extent extent} [] []
                             (dialect/lambda-form [] []
                                                  [(list (symbol "clojure.core" (name scalar-tag)) 0)])))
        equation-facts
        (-> (dialect/default-equation-facts
             {:source-binding-id placement :allocation-source-binding-id source-binding-id})
            (assoc :effects #{:memory/write} :aliases {result destination})
            (assoc :attributes {:initialization :zero
                                :result-storage [{:destination destination :access :write
                                                  :host-return :effect}]}))]
    {:equations [equation]
     :facts (-> facts
                (assoc-in [:values result]
                          (av/tensor {:dtype dtype :shape [extent]
                                      :representation {:kind :plain}}))
                (assoc-in [:equations id] equation-facts)
                (update :effects conj :memory/write))}))

(defn- allocation-contracts! [facts]
  (let [allocations (get-in facts [:attributes :allocations] [])]
    (when-not (and (vector? allocations)
                   (every? #(and (map? %) (dialect/value-id? (:destination %))
                                 (integer? (:source-binding-id %))
                                 (not (neg? (:source-binding-id %)))
                                 (contains? #{:zero :copy :unspecified} (:initialization %)))
                           allocations)
                   (= (count allocations) (count (distinct (map :destination allocations))))
                   (= (count allocations) (count (distinct (map :source-binding-id allocations)))))
      (fail! "allocation facts must identify distinct ordered storage contracts"
             {:allocations allocations}))
    (doseq [{:keys [destination dtype] :as allocation} allocations
            :let [value (get-in facts [:values destination])]
            :when value]
      (when-not (and (= :tensor (:kind value)) (seq (:shape value))
                     (dtype/known? dtype) (= dtype (dtype/canon dtype))
                     (= dtype (:dtype value)))
        (fail! "allocation and destination must agree on the retained element type"
               {:allocation allocation :value value}))
      (when (and (= :zero (:initialization allocation))
                 (not (integral-extent? facts (:extent allocation))))
        (fail! "zero allocation requires a canonical integral scalar extent"
               {:allocation allocation}))
      (when (and (= :zero (:initialization allocation))
                 (not (plain-storage? value)))
        (fail! "fresh zero allocation requires plain dense storage"
               {:allocation allocation :value value}))
      ;; AbstractValue shape can be a consumer's logical access domain, not the allocator's
      ;; physical capacity (e.g. an AD gradient consumed over alength(weights)). The constructor
      ;; extent is authoritative for initialization. Never guess a symbolic equality here;
      ;; concrete graph/LinkPlan binding checks access capacity. Reject a known short allocation.
      (when (and (= :zero (:initialization allocation))
                 (integer? (:extent allocation)) (every? integer? (:shape value))
                 (> (reduce *' 1 (:shape value)) (:extent allocation)))
        (fail! "fresh zero allocation is smaller than its logical access domain"
               {:allocation allocation :value value})))
    allocations))

(defn materialize
  "Return [program stats], inserting zero maps before the first physical use of fresh storage.
   Unspecified/copy allocations retain their own contracts. Elision needs exact dense coverage,
   never merely an ABI :write permission. Run after fusion and before ownership certification."
  [program]
  (dialect/validate! program)
  (let [original-facts (dialect/facts program)
        extent-environment (extent-proof/environment program)
        allocations (filter #(and (= :zero (:initialization %))
                                  (contains? (:values original-facts) (:destination %)))
                            (allocation-contracts! original-facts))
        {:keys [facts equations pending fills elided native]}
        (reduce
         (fn [{:keys [facts pending] :as state} equation]
           (let [active (filterv #(touches? facts (:destination %) equation) pending)
                 state (reduce
                        (fn [{:keys [facts] :as state} allocation]
                          (let [{:keys [placement native?]}
                                (initialization-site! facts allocation equation)]
                            (cond
                              native? (-> state (update :native inc)
                                          (update-in [:facts :attributes :native-initialization-providers]
                                                     #(vec (distinct (conj (or % [])
                                                                          (:destination allocation))))))
                              (full-overwrite? facts extent-environment allocation equation)
                              (update state :elided inc)
                              :else
                              (let [{:keys [equations facts]} (initializer facts allocation placement)]
                                (-> state (assoc :facts facts)
                                    (update :equations into equations) (update :fills inc))))))
                        state active)]
             (-> state
                 (assoc :pending (vec (remove (set active) pending)))
                 (update :equations conj equation))))
         {:facts original-facts :equations [] :pending (vec allocations) :fills 0 :elided 0 :native 0}
         (dialect/equations program))
        _ (when (seq pending)
            (fail! "live zero storage has no typed initialization site"
                   {:allocations pending}))
        definitions (set (mapcat #(nth % 2) equations))
        references (set (mapcat #(into (dialect/operation-inputs %)
                                      (filter dialect/value-id? (dialect/operation-extents %)))
                               equations))
        facts (assoc facts :inputs (vec (sort-by pr-str (set/difference references definitions))))]
    [(dialect/make facts equations (dialect/outputs program))
     {:initialization-fills fills :initialization-full-overwrites elided
      :initialization-native-providers native}]))
