(ns raster.compiler.equation-artifact-test
  (:require [boring.core :as boring]
            [clojure.test :refer [deftest is testing use-fixtures]]
            [raster.compiler.equation-artifact :as artifact]
            [raster.compiler.equation-artifact-store :as store]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.core :refer [deftm]]
            [raster.gpu.compiled :as compiled]
            [raster.runtime.hardware :as hardware])
  (:import [java.nio.file Files Path]
           [java.util Comparator]))

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

(deftest equation-first-cache-normalizes-only-equivalent-schedule-spellings
  (let [args [(float-array 4) 4]
        base {:compiler :equation-first :target target :dtype :float}
        count-compiles (atom 0)
        original equation-first/compile]
    ;; Settle generated specializations before measuring template acquisitions.
    (compiled/lower #'artifact-map args base)
    (compiled/clear-compilation-cache!)
    (try
      (with-redefs [equation-first/compile
                    (fn [& xs] (swap! count-compiles inc) (apply original xs))]
        (let [old (compiled/lower #'artifact-map args
                                  (assoc base :schedule {:gemm-precision :f32-scalar}))
              canonical (compiled/lower #'artifact-map args
                                        (assoc base :schedule {:precision :f32-scalar}))]
          (is (= 1 @count-compiles))
          (is (= (:schedule old) (:schedule canonical)))
          (is (true? (get-in (compiled/preparation-report canonical) [:template :cache-hit?])))
          (is (= (get-in (compiled/preparation-report old) [:template :semantic-fingerprint])
                 (get-in (compiled/preparation-report canonical) [:template :semantic-fingerprint])))
          (is (thrown? clojure.lang.ExceptionInfo
                       (compiled/lower #'artifact-map args
                                       (assoc base :schedule {:gemm-precision :mixed-f16-f32
                                                              :precision :f32-scalar}))))
          (is (= 1 @count-compiles) "conflicting sugar fails before acquiring a template")
          (compiled/lower #'artifact-map args (assoc base :schedule {:precision :mixed-f16-f32}))
          (is (= 2 @count-compiles) "distinct precision policies remain distinct")
          (let [with-meta (compiled/lower #'artifact-map args
                                          (assoc base :schedule {:precision :mixed-f16-f32
                                                                 :meta {:ignored true}}))]
            (is (= 2 @count-compiles))
            (is (true? (get-in (compiled/preparation-report with-meta) [:template :cache-hit?]))))
          (compiled/lower #'artifact-map args base)
          (is (= 3 @count-compiles) "default and explicitly pinned policy provenance remain distinct")
          (is (thrown? clojure.lang.ExceptionInfo
                       (compiled/lower #'artifact-map args
                                       (assoc base :gemm-precision :unknown
                                              :schedule {:precision :f32-scalar}))))))
      (finally (compiled/clear-compilation-cache!)))))

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

(defn- with-temporary-directory [f]
  (let [directory (Files/createTempDirectory "raster-equation-artifacts-"
                                             (make-array java.nio.file.attribute.FileAttribute 0))]
    (try
      (f directory)
      (finally
        (with-open [paths (Files/walk directory (make-array java.nio.file.FileVisitOption 0))]
          (doseq [^Path path (iterator-seq (.iterator (.sorted paths (Comparator/reverseOrder))))]
            (Files/deleteIfExists path)))))))

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

(deftest scheduled-preconditions-round-trip-through-the-compiler-record-codec
  (let [original (get-in @compilation [:kernels 0 :provenance :scheduled-operation])
        parameter (first (filter #(and (= :scalar (:kind %))
                                       (contains? #{:int :long} (:dtype %)))
                                 (get-in original [:body :parameters])))
        condition {:expression (:id parameter) :op :> :value 0}
        conditioned (scheduled-body/validate!
                     (assoc original :preconditions [condition]))
        options {:profile :archival
                 :registry (var-get #'artifact/compiler-record-registry)
                 :on-unknown-record :error}
        restored (#'artifact/restore-sequences
                  (boring/decode
                   (boring/encode (#'artifact/prepare-sequences conditioned) options)
                   options))]
    (is (some? parameter) "the fixture must retain its integral launch bound")
    (is (= conditioned restored))
    (is (= [condition] (:preconditions restored)))
    (is (= restored (scheduled-body/validate! restored)))))

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

(deftest atomic-store-round-trips-bounds-and-reports-corruption
  (with-temporary-directory
    (fn [directory]
      (let [cache (store/make-store {:root (.toFile directory) :max-entries 1
                                     :max-bytes (* 1024 1024)})
            first-id identity
            second-id (assoc identity :semantic-request-fingerprint "second-request")]
        (is (= :stored (:status (store/store-artifact!
                                 cache "semantic-request" first-id @compilation))))
        (is (= @compilation
               (:value (store/load-artifact cache "semantic-request" first-id))))
        (store/store-artifact! cache "second-request" second-id @compilation)
        (is (= 1 (count (filter #(.endsWith (.getName ^java.io.File %) ".cbor")
                                (.listFiles (.toFile directory))))))
        (let [entry (store/entry-file cache "second-request")]
          (Files/write (.toPath entry) (byte-array [0 1 2])
                       (make-array java.nio.file.OpenOption 0))
          (is (= {:status :miss :reason :invalid-entry}
                 (select-keys (store/load-artifact cache "second-request" second-id)
                              [:status :reason]))))))))

(deftest equation-template-cache-reuses-a-persistent-artifact-after-process-clear
  (with-temporary-directory
    (fn [directory]
      (let [cache (store/make-store {:root (.toFile directory) :max-entries 4
                                     :max-bytes (* 4 1024 1024)})
            key {:kind ::persistent-fixture
                 :semantic-fingerprint "semantic-request"
                 :persistent-cache-eligible? true
                 :persistence-blockers #{}
                 :semantic-request
                 {:compiler-build-fingerprint "compiler-build"
                  :source {:source-dependency-fingerprint "source-dependencies"}
                  :target {:descriptor-fingerprint "target-descriptor"}}}
            compiles (atom 0)
            report (atom nil)
            expected-identity {:semantic-request-fingerprint "semantic-request"
                               :compiler-build-fingerprint "compiler-build"
                               :source-dependency-fingerprint "source-dependencies"
                               :target-descriptor-fingerprint "target-descriptor"}]
        (compiled/clear-compilation-cache!)
        (try
          (binding [compiled/*equation-artifact-store* cache
                    compiled/*compilation-template-observer* #(reset! report %)]
            (#'compiled/cached-compilation-template
             key :equation-first #(do (swap! compiles inc) @compilation)))
          (is (= expected-identity (:persistent-artifact-identity @report)))
          (compiled/clear-compilation-cache!)
          (binding [compiled/*equation-artifact-store* cache
                    compiled/*compilation-template-observer* #(reset! report %)]
            (is (= @compilation
                   (#'compiled/cached-compilation-template
                    key :equation-first #(throw (ex-info "must not compile" {}))))))
          (is (= 1 @compiles))
          (is (= :hit (get-in @report [:persistent-artifact :status])))
          (is (= expected-identity (:persistent-artifact-identity @report)))
          (binding [compiled/*equation-artifact-store* cache
                    compiled/*compilation-template-observer* #(reset! report %)]
            (#'compiled/cached-compilation-template
             key :equation-first #(throw (ex-info "must reuse process entry" {}))))
          (is (true? (:cache-hit? @report)))
          (is (= :process-cache-hit (get-in @report [:persistent-artifact :reason])))
          (is (= expected-identity (:persistent-artifact-identity @report)))
          (let [development-key (-> key
                                    (assoc :semantic-fingerprint "development-request"
                                           :persistent-cache-eligible? false
                                           :persistence-blockers #{:compiler-build-fingerprint})
                                    (assoc-in [:semantic-request :compiler-build-fingerprint] nil))]
            (binding [compiled/*equation-artifact-store* cache
                      compiled/*compilation-template-observer* #(reset! report %)]
              (#'compiled/cached-compilation-template development-key :equation-first
               (fn [] @compilation)))
            (is (false? (:persistent-cache-eligible? @report)))
            (is (= #{:compiler-build-fingerprint} (:persistence-blockers @report)))
            (is (nil? (get-in @report [:persistent-artifact-identity :compiler-build-fingerprint])))
            (is (= :ineligible (get-in @report [:persistent-artifact :status]))))
          (finally
            (compiled/clear-compilation-cache!)))))))
