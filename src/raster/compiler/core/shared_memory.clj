(ns raster.compiler.core.shared-memory
  "Static resource and bank-conflict analysis for verified shared-memory layouts.

  This is a schedule cost model, not a target rewrite.  Layouts determine physical addresses;
  hardware descriptors determine bank topology; an access sample determines conflicts.  When a
  target does not expose a topology the model abstains explicitly instead of borrowing another
  vendor's constants."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.layout :as layout]))

(defn- bank-model!
  [model]
  (when-not (and (map? model)
                 (pos-int? (:count model))
                 (pos-int? (:word-bytes model))
                 (keyword? (:provenance model)))
    (throw (ex-info "shared-memory bank topology is incomplete"
                    {:reason :shared-memory-bank-model-invalid :model model})))
  model)

(defn- physical-byte-address
  [descriptor coordinates]
  (* (long (layout/layout->offset descriptor coordinates))
     (long (dtype/bytes-of (:dtype descriptor)))))

(defn analyze-access
  "Analyze one scalar access per active lane.

   `coordinates` is an ordered vector of numeric logical coordinates, one entry per lane.
   Exact-address reads may broadcast; distinct element addresses remain distinct transactions even
   when they occupy one bank word, which is a conservative cross-vendor rule. Elements wider than
   one bank word contribute a transaction to every word/bank they touch. Writes also retain exact
   address identity because overlapping writes are not a portable broadcast. The result reports
   the maximum number of distinct transactions serialized by any bank.
   A nil topology returns a structured abstention."
  ([bank-model descriptor coordinates]
   (analyze-access bank-model descriptor coordinates :read))
  ([bank-model descriptor coordinates access]
   (layout/validate-shared-memory! descriptor)
   (when-not (contains? #{:read :write} access)
     (throw (ex-info "shared-memory access model requires :read or :write"
                     {:reason :shared-memory-access-kind-invalid :access access})))
   (when-not (and (vector? coordinates)
                  (every? #(and (vector? %)
                                (= (:rank descriptor) (count %))
                                (every? nat-int? %)
                                (every? true? (map < % (:shape descriptor))))
                          coordinates))
     (throw (ex-info "shared-memory conflict analysis requires numeric per-lane coordinates"
                     {:reason :shared-memory-access-coordinates-invalid
                      :layout descriptor :coordinates coordinates})))
   (if-not bank-model
     {:status :unavailable
      :reason :shared-memory-bank-topology-unavailable
      :lanes (count coordinates)
      :swizzle (:swizzle descriptor)}
     (let [{:keys [word-bytes] :as bank-model} (bank-model! bank-model)
           bank-count (:count bank-model)
           element-bytes (dtype/bytes-of (:dtype descriptor))
           byte-addresses (mapv #(physical-byte-address descriptor %) coordinates)
           word-addresses-per-lane
           (mapv (fn [byte-address]
                   (vec (range (quot byte-address word-bytes)
                               (inc (quot (+ byte-address (dec element-bytes)) word-bytes)))))
                 byte-addresses)
           banks-per-lane (mapv #(mapv (fn [word] (mod word bank-count)) %)
                                word-addresses-per-lane)
           bank-transactions
           (mapcat (fn [byte-address word-addresses]
                     (map (fn [word-address]
                            [(mod word-address bank-count) [word-address byte-address]])
                          word-addresses))
                   byte-addresses word-addresses-per-lane)
           transactions-by-bank
           (reduce (fn [result [bank transaction]]
                     (update result bank (fnil conj #{}) transaction))
                   {} bank-transactions)
           conflict-degrees (into {} (map (fn [[bank addresses]] [bank (count addresses)]))
                                  transactions-by-bank)
           broadcasts (->> byte-addresses frequencies (filter (fn [[_ lanes]] (> lanes 1)))
                           (into {}))]
       {:status :modeled
        :access access
        :lanes (count coordinates)
        :swizzle (:swizzle descriptor)
        :bank-model bank-model
        :physical-byte-addresses byte-addresses
        :banks-per-lane banks-per-lane
        :banks (when (every? #(= 1 (count %)) banks-per-lane)
                 (mapv first banks-per-lane))
        :transactions-by-bank conflict-degrees
        :max-conflict-degree (reduce max 0 (vals conflict-degrees))
        :broadcast-address-groups broadcasts}))))

(defn resource-model
  "Combine exact allocation charge with an optional bank-access estimate.

   XOR members are in-place bijections, so logical and physical bytes are identical. `capacity`
   may be the descriptor's per-workgroup scratchpad limit; feasibility is omitted when unknown."
  [bank-model descriptor coordinates & {:keys [access capacity] :or {access :read}}]
  (layout/validate-shared-memory! descriptor)
  (let [elements (reduce * (:shape descriptor))
        bytes (* elements (dtype/bytes-of (:dtype descriptor)))]
    (cond-> {:layout descriptor
             :elements elements
             :logical-bytes bytes
             :physical-bytes bytes
             :padding-bytes 0
             :access (analyze-access bank-model descriptor coordinates access)}
      (some? capacity) (assoc :capacity-bytes capacity :feasible? (<= bytes capacity)))))
