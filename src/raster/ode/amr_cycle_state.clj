(ns raster.ode.amr-cycle-state
  "Capture a checked bounded AMR cycle through the existing numerical content contracts.
   Owns no provider, cache or device session. Capture is not metadata publication."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.buffer-view :as view]
            [raster.compiler.ir.numerical-state :as state]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.compiler.ir.validate :refer [fail! exact-keys!]]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.distributed :as distributed]
            [raster.ode.amr-cycle-execution :as execution]
            [raster.runtime.resident-state :as resident]))

(defn capture!
  "Capture both complete fields after the exact certified distributed cycle finishes.
   Revalidates the compiler certificate and exact original runtime plan, then retains the
   output read scope through hashing and synchronous ingestion. facts maps physical targets
   to original measured Double representation evidence. opts supplies :id and
   :logical-coordinate; both input and output coordinates contain exactly nonnegative :step
   and synchronized :phase. Physical time and other coordinate facets require an explicit
   derived extension, not arbitrary caller overrides.

   Field geometry, policy and lineage come from the checked workload. Geometry and initial
   numerical meaning remain producer-attested, not authenticated by downloading bytes.
   All fields and the manifest are validated before provider writes. Failure returns no state
   but may leave orphan content. Independently finalize-state-availability! before publishing;
   its callback owns metadata atomicity and parent existence. Direct session mutation and
   borrowed-view escape remain outside the distributed read-scope contract."
  [certified executable facts provider target-tier opts]
  (execution/verify! certified)
  (when-not (map? opts)
    (fail! "cycle capture requires options" :amr-cycle-capture-options {}))
  (exact-keys! "cycle capture options" :amr-cycle-capture-options opts #{:id :logical-coordinate})
  (let [plan (get-in certified [:workload :plan :distributed-plan :plan])
        input (get-in certified [:workload :plan :state :manifest])
        coordinate (:logical-coordinate opts)
        step (get-in input [:logical-coordinate :step])]
    (when-not (and (distributed/original-executable? executable) (= plan (:plan executable)))
      (fail! "cycle capture requires its exact original distributed execution"
             :amr-cycle-capture-owner {}))
    (when-not (and (integer? step) (not (neg? step)) (map? coordinate)
                   (= #{:step :phase} (set (keys (:logical-coordinate input))))
                   (= #{:step :phase} (set (keys coordinate)))
                   (= :synchronized (:phase coordinate)) (= (inc step) (:step coordinate))
                   (some? (:id opts)) (not= (:id input) (:id opts)))
      (fail! "cycle capture must advance to a distinct synchronized state"
             :amr-cycle-capture-coordinate {}))
    (let [completion (get-in certified [:certificate :completion])
          prepared (:prepared certified)
          worker (:device (first (:steps plan)))
          target (get-in plan [:device-plans worker :target])
          interface (compiled/producer-interface prepared)
          fields-by-id (into {} (map (juxt :id identity)) (:fields input))
          _ (when-not (= (set (keys fields-by-id))
                         (set (map :field (vals (get-in certified [:certificate :fields])))))
              (fail! "cycle capture requires exactly its two retained state fields"
                     :amr-cycle-capture-fields {}))
          specs
          (mapv
           (fn [role]
             (let [{:keys [field node domain]} (get-in certified [:certificate :fields role])
                   physical (get-in (compiled/plan prepared) [:nodes node :view])
                   ports (filterv #(= node (:node %)) (:outputs interface))
                   schema (get fields-by-id field)
                   shape (get-in schema [:value :shape])
                   width (dtype/bytes-of :double)
                   bytes (reduce #(Math/multiplyExact (long %1) (long %2)) (long width) shape)]
               (when-not (and (= 1 (count ports)) schema (= :double (:dtype physical))
                              (= :double (get-in schema [:value :dtype]))
                              (= {:kind :plain} (get-in schema [:value :representation]))
                              (nil? (get-in schema [:value :logical-layout]))
                              (view/contiguous? physical) (= shape (:shape domain))
                              ;; A verified dense patch may flatten rank, not ordering or coverage.
                              (= (reduce *' shape) (reduce *' (:shape physical)))
                              (= bytes (:byte-length physical)))
                 (fail! "cycle capture requires its exact whole dense output field"
                        :amr-cycle-capture-field {:role role :field field}))
               {:id field :step completion :key (:key (first ports))
                :value (:value schema) :coordinate-space (:coordinate-space schema)
                :attributes (:attributes schema)}))
           [:coarse :fine])]
      (resident/capture-distributed-with-manifest!
       executable {[worker (get-in certified [:certificate :entry])] prepared}
       {[target :double] (get facts target)} provider target-tier
       {:id (:id opts) :parents [(:id input)] :logical-coordinate coordinate
        :fields specs :numerical-contract (:numerical-contract input)}
       (fn [{:keys [fields provenance]}]
         (state/manifest
          (assoc (into {} input)
                 :id (:id opts) :parents [(:id input)] :logical-coordinate coordinate
                 :fields fields
                 :provenance {:scope :exact-completed-distributed-amr-cycle
                              :semantic-authority :producer-attested
                              :program-fingerprint (get-in certified [:certificate :program :fingerprint])
                              :program (get-in certified [:certificate :program])
                              :temporal (get-in certified [:certificate :temporal])
                              :completion completion
                              :representations (:representations provenance)})))))))

(defn verify-restore!
  "Check a captured cycle against independent source compiler and complete target semantics.
   Invoke before localization, decoding or upload. source-certified is the checked producing
   cycle, not a guessed continuation consumer. expected-semantics independently names all
   fields/coordinates, the logical coordinate and numerical policy, using the existing fixed
   numerical-state restore schema. The captured producer, temporal stages, completion, lineage,
   field order and complete raw FP64 encoding must match the retained source contract.

   This verifies consistency, not authentication of serialized claims, content integrity,
   durable parent existence or PDE equivalence. Representation snapshots remain historical
   audit data; fresh execution must measure its own target and independently decode byte order.
   No provider, compiler cache, device session or upload is contacted here."
  [captured source-certified expected-semantics]
  (execution/verify! source-certified)
  (state/verify-restore-semantics! captured expected-semantics)
  (let [manifest (:manifest captured)
        source (get-in source-certified [:workload :plan :state :manifest])
        certificate (:certificate source-certified)
        provenance (:provenance manifest)
        source-fields (into {} (map (juxt :id identity)) (:fields source))
        ordered (mapv #(get-in certificate [:fields % :field]) [:coarse :fine])
        plan (get-in source-certified [:workload :plan :distributed-plan :plan])
        target (get-in plan [:device-plans (:device (first (:steps plan))) :target])]
    (exact-keys! "captured cycle provenance" :amr-cycle-restore-provenance provenance
                 #{:scope :semantic-authority :program-fingerprint :program :temporal :completion :representations})
    (when-not (and (= :exact-completed-distributed-amr-cycle (:scope provenance))
                   (= :producer-attested (:semantic-authority provenance))
                   (= (get-in certificate [:program :fingerprint]) (:program-fingerprint provenance))
                   (fingerprint/equivalent? (:program certificate) (:program provenance))
                   (fingerprint/equivalent? (:temporal certificate) (:temporal provenance))
                   (= (:completion certificate) (:completion provenance)))
      (fail! "captured cycle differs from the independently checked source producer"
             :amr-cycle-restore-producer {}))
    (when-not (and (= #{:step :phase} (set (keys (:logical-coordinate source))))
                   (= #{:step :phase} (set (keys (:logical-coordinate manifest))))
                   (integer? (get-in source [:logical-coordinate :step]))
                   (not (neg? (get-in source [:logical-coordinate :step])))
                   (= {:step (inc (get-in source [:logical-coordinate :step])) :phase :synchronized}
                      (:logical-coordinate manifest))
                   (= [(:id source)] (:parents manifest))
                   (fingerprint/equivalent? (:numerical-contract source) (:numerical-contract manifest))
                   (not= (:id source) (:id manifest)))
      (fail! "captured cycle coordinate or parent differs from its source state"
             :amr-cycle-restore-lineage {}))
    (when-not (and (= (set ordered) (set (keys source-fields)))
                   (= ordered (mapv :id (:fields manifest)))
                   (vector? (:representations provenance)) (= 2 (count (:representations provenance))))
      (fail! "captured cycle must retain its exact ordered whole-field producer bindings"
             :amr-cycle-restore-fields {}))
    (doseq [[field representation] (map vector (:fields manifest) (:representations provenance))]
      (let [original (get source-fields (:id field))
            shape (get-in field [:value :shape])
            chunks (:chunks field) chunk (first chunks)
            byte-order (:byte-order representation)]
        (when-not (and (fingerprint/equivalent? (select-keys original [:value :coordinate-space :attributes])
                                               (select-keys field [:value :coordinate-space :attributes]))
                       (= :double (get-in field [:value :dtype]))
                       (= {:kind :plain} (get-in field [:value :representation]))
                       (nil? (get-in field [:value :logical-layout]))
                       (= :raster.distributed/resident-representation-v1 (:kind representation))
                       (= :double (:dtype representation)) (= target (:target representation))
                       (contains? state/byte-orders byte-order)
                       (= 1 (count chunks)) (= 0 (:id chunk))
                       (= shape (:chunk-shape field) (:shape chunk))
                       (= (vec (repeat (count shape) 0)) (:offsets chunk))
                       (= {:format :raw-array :byte-order byte-order} (:storage chunk))
                       (= (:logical-byte-length chunk) (:stored-byte-length chunk)))
          (fail! "captured field differs from its complete measured source encoding"
                 :amr-cycle-restore-storage {:field (:id field)}))))
    captured))
