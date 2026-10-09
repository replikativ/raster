(ns raster.ode.amr-cycle-state
  "Capture a checked bounded AMR cycle through the existing numerical content contracts.
   Owns no provider, cache or device session. Capture is not metadata publication."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.buffer-view :as view]
            [raster.compiler.ir.numerical-state :as state]
            [raster.compiler.ir.validate :refer [fail! exact-keys!]]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.distributed :as distributed]
            [raster.ode.amr-cycle-execution :as execution]
            [raster.runtime.numerical-content :as content]))

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
    (distributed/with-output-values!
     executable
     (fn [outputs]
       (let [completion (get-in certified [:certificate :completion])
             fields-by-id (into {} (map (juxt :id identity)) (:fields input))
             _ (when-not (= (set (keys fields-by-id))
                            (set (map :field (vals (get-in certified [:certificate :fields])))))
                 (fail! "cycle capture requires exactly its two retained state fields"
                        :amr-cycle-capture-fields {}))
             entries
             (mapv
              (fn [role]
                (let [{:keys [field value node domain]} (get-in certified [:certificate :fields role])
                      physical (get-in (compiled/plan (:prepared certified)) [:nodes node :view])
                      worker (:device (first (:steps plan)))
                      target (get-in plan [:device-plans worker :target])
                      validate-storage! #(distributed/storage-representation-description
                                          executable target :double (get facts target))
                      representation (validate-storage!)
                      resident (get-in outputs [completion value])
                      schema (get fields-by-id field)
                      shape (get-in schema [:value :shape])
                      width (dtype/bytes-of :double)
                      bytes (reduce #(Math/multiplyExact (long %1) (long %2)) (long width) shape)]
                  (when-not (and resident schema (= :double (:dtype physical))
                                 (= :double (get-in schema [:value :dtype]))
                                 (= {:kind :plain} (get-in schema [:value :representation]))
                                 (nil? (get-in schema [:value :logical-layout]))
                                 (view/contiguous? physical) (= shape (:shape domain))
                                 ;; The verified local-domain binding may flatten a dense patch.
                                 ;; This changes rank, not ordering, coverage or byte extent.
                                 (= (reduce *' shape) (reduce *' (:shape physical)))
                                 (= bytes (:byte-length physical)))
                    (fail! "cycle capture requires its exact whole dense output field"
                           :amr-cycle-capture-field {:role role :field field}))
                  (let [reader (content/element-byte-reader
                                bytes width
                                (fn [start elements destination]
                                  (validate-storage!)
                                  (gpu/download-range! (get (:sessions executable) target)
                                                       resident destination
                                                       {:src-element start :elements elements})))
                        address (content/content-address-from-reader bytes reader)
                        chunk (state/chunk {:id 0 :offsets (vec (repeat (count shape) 0))
                                            :shape shape :logical-byte-length bytes :stored-byte-length bytes
                                            :content address :storage {:format :raw-array
                                                                       :byte-order (:byte-order representation)}})]
                    {:field (state/field (assoc (into {} schema) :chunk-shape shape :chunks [chunk]))
                     :reader reader :bytes bytes :address address :representation representation
                     :validate-storage! validate-storage!})))
              [:coarse :fine])
             captured (state/certify
                       (state/manifest
                        (assoc (into {} input)
                               :id (:id opts) :parents [(:id input)] :logical-coordinate coordinate
                               :fields (mapv :field entries)
                               :provenance {:scope :exact-completed-distributed-amr-cycle
                                            :semantic-authority :producer-attested
                                            :program-fingerprint (get-in certified [:certificate :program :fingerprint])
                                            :program (get-in certified [:certificate :program])
                                            :temporal (get-in certified [:certificate :temporal])
                                            :completion completion
                                            :representations (mapv :representation entries)})))]
         (let [placements
               (mapv (fn [{:keys [field reader bytes address]}]
                       {:field-id (:id field) :chunk-id 0
                        :placement (content/ingest-content! provider address target-tier bytes reader)})
                     entries)]
           ;; Provider callbacks may run between downloads. Never return a successful capture
           ;; against a changed session/device snapshot, including after the final callback.
           (doseq [{:keys [validate-storage!]} entries] (validate-storage!))
           {:state captured :placements placements}))))))
