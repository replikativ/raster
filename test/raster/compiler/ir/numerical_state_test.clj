(ns raster.compiler.ir.numerical-state-test
  (:refer-clojure :exclude [chunk])
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as abstract-value]
            [raster.compiler.ir.numerical-state :as state]))

(defn- content
  [n]
  (state/content-address :sha-256 (format "%064x" n)))

(defn- chunk
  [id offsets shape logical-bytes stored-bytes digest]
  (state/chunk
   {:id id
    :offsets offsets
    :shape shape
    :logical-byte-length logical-bytes
    :stored-byte-length stored-bytes
    :content (content digest)
    :storage {:format :boring-rfc8746 :byte-order :little-endian}
    :attributes {}}))

(defn- temperature-field
  []
  (state/field
   {:id :temperature
    :value (abstract-value/tensor
            {:dtype :float
             :shape [5 4]
             :logical-layout {:order :row-major}
             :representation {:kind :plain}
             :ownership :owned})
    :chunk-shape [3 3]
    :coordinate-space
    {:level 0
     :axes [{:name :y :origin 0.0 :spacing 0.25}
            {:name :x :origin 0.0 :spacing 0.5}]}
    :chunks [(chunk :t-0-0 [0 0] [3 3] 36 48 1)
             (chunk :t-0-3 [0 3] [3 1] 12 24 2)
             (chunk :t-3-0 [3 0] [2 3] 24 36 3)
             (chunk :t-3-3 [3 3] [2 1] 8 20 4)]}))

(defn- checkpoint
  ([] (checkpoint [(temperature-field)]))
  ([fields]
   (state/manifest
    {:id :weather/step-40
     :parents [:weather/step-32]
     :logical-coordinate {:step 40 :time-seconds 10.0 :stage :accepted}
     :fields fields
     :numerical-contract
     {:mode :ieee-mixed-precision
      :determinism :reproducible-order
      :compatibility-id "shallow-water-v3:f32-storage:f32-accumulate"}
     :provenance
     {:program-fingerprint "sha256:compiler-and-solver-fingerprint"
      :parameters {:cfl 0.45}}
     :attributes {:scenario :coastal-probe}})))

(defn- thrown-data
  [f]
  (try
    (f)
    nil
    (catch clojure.lang.ExceptionInfo exception
      (ex-data exception))))

(deftest complete-dense-state-is-certified-with-lineage-and-costs
  (let [certified (state/certify (checkpoint))
        certificate (:certificate certified)]
    (is (state/certified-state? (state/verify! certified)))
    (is (= :weather/step-40 (:state-id certificate)))
    (is (= [:weather/step-32] (:parents certificate)))
    (is (= {:step 40 :time-seconds 10.0 :stage :accepted}
           (:logical-coordinate certificate)))
    (is (= 80 (:logical-byte-length certificate)))
    (is (= 128 (:stored-byte-length certificate)))
    (is (= (mapv (comp :digest :content) (:chunks (temperature-field)))
           (mapv :digest (:content-addresses certificate))))
    (is (= :reproducible-order (get-in certificate [:numerical-contract :determinism])))
    (is (= "sha256:compiler-and-solver-fingerprint"
           (get-in certificate [:provenance :program-fingerprint])))))

(deftest storage-placement-is-not-part-of-the-compiler-manifest
  (let [candidate (first (:chunks (temperature-field)))]
    (is (= #{:id :offsets :shape :logical-byte-length :stored-byte-length
             :content :storage :attributes}
           (set (keys candidate))))
    (is (not-any? #(contains? candidate %)
                  [:path :bucket :object-key :store :mmap :segment :buffer :device :queue]))
    (is (= :numerical-state-chunk-fields
           (:reason
            (thrown-data
             #(state/certify
               (checkpoint
                [(assoc-in (temperature-field) [:chunks 0 :object-key]
                           "s3://bucket/chunk")]))))))))

(deftest regular-grid-coverage-shape-and-byte-count-are-load-bearing
  (let [field (temperature-field)]
    (testing "a missing cell is a coverage gap"
      (is (= :numerical-state-chunk-coverage
             (:reason
              (thrown-data
               #(state/certify
                 (checkpoint [(assoc field :chunks (pop (:chunks field)))])))))))
    (testing "edge cells must be clipped to the global shape"
      (is (= :numerical-state-chunk-grid-shape
             (:reason
              (thrown-data
               #(state/certify
                 (checkpoint
                  [(assoc-in field [:chunks 3 :shape] [3 1])])))))))
    (testing "logical byte counts are derived from dtype and shape"
      (is (= :numerical-state-chunk-logical-bytes
             (:reason
              (thrown-data
               #(state/certify
                 (checkpoint
                  [(assoc-in field [:chunks 0 :logical-byte-length] 35)])))))))
    (testing "native byte order is not a durable cross-machine contract"
      (is (= :numerical-state-chunk-byte-order
             (:reason
              (thrown-data
               #(state/certify
                 (checkpoint
                  [(assoc-in field [:chunks 0 :storage :byte-order] :native)])))))))
    (testing "canonical grid order makes publication and certification deterministic"
      (is (= :numerical-state-chunk-order
             (:reason
              (thrown-data
               #(state/certify
                 (checkpoint
                  [(assoc field :chunks (vec (reverse (:chunks field))))])))))))))

(deftest certificates-detect-manifest-drift
  (let [certified (state/certify (checkpoint))
        changed-coordinate (assoc-in certified [:manifest :logical-coordinate :step] 41)
        changed-content (assoc-in certified
                                  [:manifest :fields 0 :chunks 0 :content]
                                  (content 99))]
    (is (= :numerical-state-certificate
           (:reason (thrown-data #(state/verify! changed-coordinate)))))
    (is (= :numerical-state-certificate
           (:reason (thrown-data #(state/verify! changed-content)))))))

(deftest numerical-compatibility-and-provenance-are-explicit
  (testing "restore compatibility is not inferred from dtype and shape"
    (is (= :numerical-state-compatibility-id
           (:reason
            (thrown-data
             #(state/certify
               (assoc (checkpoint)
                      :numerical-contract
                      {:mode :ieee-mixed-precision
                       :determinism :reproducible-order})))))))
  (testing "the producing program is identified independently from state and chunk identities"
    (is (= :numerical-state-program-fingerprint
           (:reason
            (thrown-data
             #(state/certify
               (assoc (checkpoint) :provenance {:parameters {:cfl 0.45}}))))))))

(deftest manifests-may-reuse-immutable-chunks-across-branches
  (let [parent-field (temperature-field)
        reused (get-in parent-field [:chunks 0 :content])
        child-field (assoc-in parent-field [:chunks 1 :content] (content 20))
        certified (state/certify
                   (state/manifest
                    (assoc (into {} (checkpoint [child-field]))
                           :id :weather/branch-b
                           :parents [:weather/step-40]
                           :logical-coordinate {:step 48 :branch :higher-viscosity})))]
    (is (= reused (get-in certified [:certificate :fields 0 :chunks 0 :content])))
    (is (= (content 20) (get-in certified [:certificate :fields 0 :chunks 1 :content])))))
