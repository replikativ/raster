(ns raster.ode.amr-cycle-restore-test
  "Pure restore consistency checks; synthetic compiler authority is explicit in the fixture."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.numerical-state :as state]
            [raster.gpu.distributed :as distributed]
            [raster.ode.amr-cycle-state :as cycle]
            [raster.ode.amr-cycle-state-test :as fixture]
            [raster.runtime.resident-state-test :as provider-fixture]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- semantics [manifest]
  {:fields (mapv #(select-keys % [:id :value :coordinate-space]) (:fields manifest))
   :logical-coordinate (:logical-coordinate manifest)
   :numerical-contract (:numerical-contract manifest)})

(defn- with-captured [f]
  (#'fixture/fixture
   (fn [source owner _ _ _]
     (with-redefs [distributed/storage-representation-description
                   (fn [& _] {:kind :raster.distributed/resident-representation-v1
                              :dtype :double :target :unit :byte-order :little-endian})]
       (let [{:keys [provider]} (#'provider-fixture/provider (fn [& _]))
             captured (:state (cycle/capture! source owner {} provider :local
                                              {:id :next :logical-coordinate {:step 1 :phase :synchronized}}))]
         (f source captured (semantics (:manifest captured))))))))

(deftest exact-source-and-independent-semantics-are-both-required
  (with-captured
    (fn [source captured expected]
      (is (identical? captured (cycle/verify-restore! captured source expected)))
      (is (= :numerical-state-restore-incompatible
             (reason #(cycle/verify-restore! captured source (assoc-in expected [:logical-coordinate :step] 2)))))
      (doseq [path [[:program-fingerprint] [:completion] [:temporal]]]
        (let [bad (state/certify (update-in (:manifest captured) (into [:provenance] path) (constantly "other")))]
          (is (= :amr-cycle-restore-producer (reason #(cycle/verify-restore! bad source expected)))))))))

(deftest recertified-claims-do-not-override-the-source-boundary
  (with-captured
    (fn [source captured _]
      (doseq [[change expected-reason]
              [[#(assoc % :parents [:other]) :amr-cycle-restore-lineage]
               [#(assoc-in % [:logical-coordinate :step] 2) :amr-cycle-restore-lineage]
               [#(assoc-in % [:numerical-contract :compatibility-id] "other") :amr-cycle-restore-lineage]
               [#(update % :fields (comp vec reverse)) :amr-cycle-restore-fields]
               [#(assoc-in % [:provenance :representations 0 :target] :other) :amr-cycle-restore-storage]
               [#(assoc-in % [:provenance :representations 0 :byte-order] :big-endian) :amr-cycle-restore-storage]]]
        (let [manifest (change (:manifest captured))
              bad (state/certify manifest)]
          ;; Even independently supplied semantics agreeing with the changed claim cannot
          ;; replace the checked source lineage, producer bindings or physical encoding.
          (is (= expected-reason (reason #(cycle/verify-restore! bad source (semantics manifest))))))))))
