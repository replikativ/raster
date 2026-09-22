(ns raster.compiler.equation-artifact-test
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [raster.compiler.equation-artifact :as artifact]
            [raster.compiler.equation-first :as equation-first]
            [raster.core :refer [deftm]]
            [raster.runtime.hardware :as hardware]))

(def ^:private target :cuda:equation-artifact-test)

(use-fixtures
  :once
  (fn [run]
    (hardware/register-target-device!
     target
     {:type :cuda
      :name "Synthetic artifact codec target"
      :capabilities {:compute-capability [8 0]
                     :warp-size 32
                     :subgroup-sizes [32]
                     :max-workgroup-size 1024
                     :shared-local-memory 65536
                     :total-eus 108}})
    (run)))

(deftm artifact-map
  [input :- (Array float) n :- Long] :- (Array float)
  (let [output (float-array n)]
    (raster.par/map! output index n float
                     (raster.numeric/* (float 2.0)
                                       (raster.arrays/aget input index)))))

(def ^:private identity
  {:semantic-request-fingerprint "semantic-request"
   :compiler-build-fingerprint "compiler-build"
   :source-dependency-fingerprint "source-dependencies"
   :target-descriptor-fingerprint "target-descriptor"})

(defonce ^:private compilation
  (delay (equation-first/compile #'artifact-map {:target target :dtype :float})))

(defn- reason-of [thunk]
  (try (thunk) nil
       (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest equation-compilation-round-trips-and-remains-lowerable
  (let [original @compilation
        envelope (artifact/seal identity original)
        transport (artifact/encode envelope)
        restored (artifact/open identity (artifact/decode transport))
        arguments [(float-array [1.0 2.0 3.0 4.0]) 4]]
    (is (= original restored))
    (is (= (equation-first/lower original arguments)
           (equation-first/lower restored arguments)))
    (is (= (:payload-fingerprint envelope)
           (:payload-fingerprint (artifact/decode transport))))))

(deftest envelope-authentication-fails-before-record-reconstruction
  (let [envelope (artifact/seal identity @compilation)]
    (testing "a different current compiler/request identity cannot open an artifact"
      (is (= :equation-artifact-identity-mismatch
             (reason-of #(artifact/open
                          (assoc identity :compiler-build-fingerprint "different-build")
                          envelope)))))
    (testing "payload modification is detected"
      (let [changed (aclone ^bytes (:payload envelope))]
        (aset-byte changed 0 (unchecked-byte (bit-xor 0xff (aget changed 0))))
        (is (= :equation-artifact-integrity
               (reason-of #(artifact/open identity (assoc envelope :payload changed)))))))
    (testing "unknown schemas fail closed"
      (is (= :equation-artifact-schema
             (reason-of #(artifact/open identity
                                        (assoc envelope :schema-version 2))))))))

(deftest unsupported-runtime-values-never-enter-artifacts
  (is (= :semantic-fingerprint-unsupported
         (reason-of #(artifact/seal identity
                                    (assoc @compilation :options {:callback (fn [] nil)}))))))
