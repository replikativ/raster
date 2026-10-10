(ns raster.runtime.distributed-state-test
  "Hardware-free consistency rejection; synthesized manifests are not completion authority."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.build-manifest :as build]
            [raster.compiler.fixtures.distributed-capture :as fixture]
            [raster.compiler.ir.numerical-state :as state]
            [raster.compiler.ir.distributed-plan :as distributed-plan]
            [raster.compiler.ir.semantic-fingerprint :as fingerprint]
            [raster.gpu.completed-evidence-device-test :as identity]
            [raster.gpu.core :as gpu]
            [raster.gpu.distributed :as distributed]
            [raster.runtime.numerical-content :as content]
            [raster.runtime.resident-state :as resident])
  (:import [java.lang.foreign MemorySegment]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- fixture-state []
  (let [{:keys [plan prepared-by-entry capture-options expected]} (fixture/fixture :ocl:0
                                                                               {:twice [1.0 2.0] :increment [3.0 4.0]})
        sources (#'resident/distributed-producers plan prepared-by-entry)
        programs (#'resident/distributed-programs plan sources)
        specs (mapv #(select-keys % [:id :step :key]) (:fields capture-options))
        bindings (mapv #(#'resident/distributed-field-source plan sources %) specs)
        addresses (mapv #(content/content-address-of (MemorySegment/ofArray (float-array (get expected (:id %))))) specs)
        fields (mapv (fn [spec address]
                       (state/field {:id (:id spec) :value (:value spec) :coordinate-space {}
                                     :chunk-shape [2]
                                     :chunks [(state/chunk {:id 0 :offsets [0] :shape [2]
                                                            :logical-byte-length 8 :stored-byte-length 8
                                                            :content address
                                                            :storage {:format :raw-array :byte-order :little-endian}})]}))
                     (:fields capture-options) addresses)
        certified (state/certify
                   (state/manifest
                    (assoc (dissoc capture-options :fields) :fields fields
                           :provenance {:scope :completed-distributed-entry-outputs
                                        :semantic-authority :application-declared
                                        :programs programs :program-fingerprint (fingerprint/fingerprint programs)
                                        :field-producers (mapv #(assoc (:producer %1) :content %2) bindings addresses)
                                        :representations (mapv (fn [_] {:kind :raster.distributed/resident-representation-v1
                                                                         :target :ocl:0 :dtype :float
                                                                         :byte-order :little-endian}) fields)})))
        semantics {:fields (mapv #(select-keys % [:id :value :coordinate-space]) fields)
                   :logical-coordinate (:logical-coordinate capture-options)
                   :numerical-contract (:numerical-contract capture-options)}]
    {:state certified :semantics semantics :plan plan :prepared prepared-by-entry :specs specs}))

(deftest distributed-restore-rejects-inconsistent-producers-before-storage
  (with-redefs [build/current-identity #'identity/test-build
                gpu/make-session (fn [& _] (throw (AssertionError. "pure restore opened a session")))
                gpu/alloc! (fn [& _] (throw (AssertionError. "pure restore allocated")))
                content/with-local-content (fn [& _] (throw (AssertionError. "restore opened storage")))]
    (let [{:keys [state semantics plan prepared specs]} (fixture-state)
          verify! #(resident/verify-distributed-restore! % semantics plan prepared specs)
          mutate (fn [f] (state/certify (state/manifest (f (into {} (:manifest state))))))]
      (is (identical? state (verify! state)))
      (doseq [change [#(update-in % [:provenance :programs] (comp vec reverse))
                      #(assoc-in % [:provenance :program-fingerprint] "wrong")
                      #(assoc-in % [:provenance :field-producers 0 :step] :increment)
                      #(assoc-in % [:provenance :field-producers 0 :key] :increment)
                      #(assoc-in % [:provenance :field-producers 0 :value] :wrong)
                      #(assoc-in % [:provenance :field-producers 0 :node] :wrong)
                      #(assoc-in % [:provenance :field-producers 0 :physical-shape] [1 2])
                      #(assoc-in % [:provenance :field-producers 0 :content]
                                 (get-in % [:provenance :field-producers 1 :content]))
                      #(assoc-in % [:provenance :representations 0 :dtype] :double)
                      #(assoc-in % [:provenance :representations 0 :target] :ze:0)
                      #(assoc-in % [:provenance :representations 0 :byte-order] :big-endian)
                      #(update % :fields (comp vec reverse))
                      #(update-in % [:provenance :representations] pop)
                      #(assoc-in % [:fields 0 :chunks 0 :id] 1)]]
        (is (some? (reason #(verify! (mutate change))))))
      (is (= :resident-state-producer-field-keys
             (reason #(resident/verify-distributed-restore! state semantics plan prepared
                                                            (assoc-in specs [0 :node] :caller-guess)))))
      (is (= :resident-state-distributed-bindings
             (reason #(resident/verify-distributed-restore! state semantics plan
                                                            (assoc prepared [:extra :entry] (first (vals prepared))) specs))))
      (is (= :resident-state-distributed-producer
             (reason #(resident/verify-distributed-restore! state semantics plan
                                                            (assoc prepared [:worker :twice] (get prepared [:worker :increment])) specs))))
      (let [overwritten (-> plan
                            (update :steps conj (distributed-plan/compute-step
                                                 {:id :repeat :device :worker :duration-ns 1 :dependencies [:twice]}))
                            (update :outputs conj :repeat)
                            (assoc-in [:device-plans :worker :steps :repeat]
                                      (get-in plan [:device-plans :worker :steps :twice])))]
        (is (= :distributed-runtime-output-overwritten
               (reason #(distributed-plan/check-retained-output-readiness overwritten))))
        (is (= :distributed-runtime-output-overwritten
               (reason #(distributed/instantiate! overwritten {:device-capacities {:ocl:0 1048576}}))))
        (is (= :distributed-runtime-output-overwritten
               (reason #(resident/verify-distributed-restore! state semantics overwritten prepared specs))))))))
