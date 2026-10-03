(ns raster.gpu.ze-registry-test
  "Registry resource-retention checks without driver initialization or native allocation."
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.ze-runtime :as ze]
            [raster.gpu.resource-cleanup :as cleanup])
  (:import [java.lang.foreign MemorySegment]))

(defn- loaded []
  {:target :opencl-c :source "unit source" :attributes {} :arena-id :unit-arena
   ::ze/registration-payload-identity nil
   ;; Heap-backed metadata fixtures have no driver allocations; their explicit owner governs
   ;; registry lifetime. Native acquisition/destruction is swept in registry-cleanup-test.
   ::cleanup/owner (cleanup/owner [])
   ::ze/registration {:context (volatile! nil) :staging (atom {})}
   :module (MemorySegment/ofArray (byte-array 1))
   :kernel-handle (MemorySegment/ofArray (byte-array 1))
   :staging (MemorySegment/ofArray (byte-array 4))
   :entry-name "canonical_entry" :spv-bytes (byte-array [1 2 3])})

(deftest identical-registration-retains-loaded-resources-and-refreshes-metadata
  (doseq [explicit? [true false]]
    (let [registry (atom {})
          incoming (cond-> {:target :opencl-c :source "unit source" :attributes {}}
                     explicit? (assoc :spv-bytes (byte-array [1 2 3])))]
      (with-redefs [ze/kernel-registry registry]
        (ze/register-kernel! "unit" incoming :unit-arena)
        ;; Simulate lazy loading preserving registration-time payload provenance.
        (swap! registry update "unit" merge
               (dissoc (loaded) ::ze/registration-payload-identity ::cleanup/owner ::ze/registration))
        (let [prior (get @registry "unit")
              incoming (cond-> (assoc incoming :attributes {:strategy :new})
                         explicit? (assoc :spv-bytes (byte-array [1 2 3])))]
          (ze/register-kernel! "unit" incoming :unit-arena)
          (let [entry (get @registry "unit")]
            (doseq [key [:module :kernel-handle :staging :spv-bytes]]
              (is (identical? (get prior key) (get entry key))))
            (is (= "canonical_entry" (:entry-name entry)))
            (is (= {:strategy :new} (:attributes entry)))))))))

(deftest explicit-payload-and-source-only-registrations-never-alias
  (doseq [prior-explicit? [true false]]
    (with-redefs [ze/kernel-registry (atom {})]
      (ze/register-kernel! "unit"
                           (cond-> {:target :opencl-c :source "unit source" :attributes {}}
                             prior-explicit? (assoc :spv-bytes (byte-array [1 2 3]))) :unit-arena)
      (swap! ze/kernel-registry update "unit" merge
             (dissoc (loaded) ::ze/registration-payload-identity ::cleanup/owner ::ze/registration))
      (ze/register-kernel! "unit"
                           (cond-> {:target :opencl-c :source "unit source" :attributes {}}
                             (not prior-explicit?) (assoc :spv-bytes (byte-array [1 2 3]))) :unit-arena)
      (is (nil? (:kernel-handle (get @ze/kernel-registry "unit")))))))

(deftest explicit-registration-snapshots-caller-owned-payload
  (let [payload (byte-array [1 2 3])]
    (with-redefs [ze/kernel-registry (atom {})]
      (ze/register-kernel! "unit" {:target :opencl-c :source "unit source"
                                   :attributes {} :spv-bytes payload} :unit-arena)
      (aset-byte payload 0 (byte 9))
      (is (= [1 2 3] (vec (:spv-bytes (ze/kernel-registry-entry "unit"))))))))

(deftest changed-target-or-unknown-registration-provenance-does-not-reuse
  (doseq [prior [(assoc (loaded) :target :hip-cpp)
                 (dissoc (loaded) ::ze/registration-payload-identity)]]
    (with-redefs [ze/kernel-registry (atom {"unit" prior})]
      (ze/register-kernel! "unit" {:target :opencl-c :source "unit source" :attributes {}} :unit-arena)
      (is (nil? (:kernel-handle (get @ze/kernel-registry "unit")))))))

(deftest changed-source-arena-or-precompiled-payload-does-not-reuse-a-loaded-kernel
  (doseq [[source arena payload] [["different source" :unit-arena nil]
                                  ["unit source" :other-arena nil]
                                  ["unit source" :unit-arena (byte-array [9 2 3])]]]
    (with-redefs [ze/kernel-registry (atom {"unit" (loaded)})]
      (ze/register-kernel! "unit" (cond-> {:target :opencl-c :source source :attributes {}}
                                    payload (assoc :spv-bytes payload)) arena)
      (is (nil? (:kernel-handle (get @ze/kernel-registry "unit"))))
      (is (nil? (:staging (get @ze/kernel-registry "unit")))))))

(deftest unsupported-compilation-requirements-cannot-overwrite-live-registration
  (let [prior (loaded)]
    (with-redefs [ze/kernel-registry (atom {"unit" prior})]
      (is (thrown? clojure.lang.ExceptionInfo
                   (ze/register-kernel!
                    "unit" {:target :opencl-c :source "unit source"
                            :attributes {:compilation {:language-standard "CL1.2" :extensions #{}}}}
                    :unit-arena)))
      (is (identical? prior (get @ze/kernel-registry "unit"))))))
