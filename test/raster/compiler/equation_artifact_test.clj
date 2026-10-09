(ns raster.compiler.equation-artifact-test
  (:require [boring.core :as boring]
            [clojure.test :refer [deftest is testing use-fixtures]]
            [raster.compiler.build-manifest :as build-manifest]
            [raster.compiler.equation-artifact :as artifact]
            [raster.compiler.equation-artifact-store :as store]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.backend.gpu.segop-opencl :as segop-opencl]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.soac-lower :as soac-lower]
            [raster.compiler.passes.parallel.segscan-body :as segscan-body]
            [raster.compiler.ir.kernel-body :as kernel-body]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.core :refer [deftm]]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.link :as gpu-link]
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

(deftm artifact-scale
  [input :- (Array float) factor :- Float n :- Long] :- (Array float)
  (let [output (float-array n)]
    (raster.par/map! output index n float
                     (raster.numeric/* factor (raster.arrays/aget input index)))))

(deftm artifact-state!
  [state :- (Array float) n :- Long] :- (Array float)
  (raster.par/map-void! index n
                        (raster.arrays/aset state index
                                            (raster.numeric/+ (raster.arrays/aget state index) (float 1.0))))
  state)

(defn- reason-of [thunk]
  (try (thunk) nil
       (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest completed-frontier-keeps-effect-event-prologue-density-decline-order
  (let [c {:executable {:owns-session? true
                        :plan {:outputs [:a]
                               :nodes {:a {:view {:allocation {:ownership :owned}
                                                  :shape [4] :strides [2]}}}}}
           :lowering {:certificate {:effect-evidence
                                    {:initialization {:requires #{:a} :initializers #{} :writes #{}}}}}}
        checks (atom 0)
        event-error (ex-info "pending event" {:reason :unit-pending-event})
        run #(reason-of (fn [] (#'compiled/completed-frontier! c)))]
    (with-redefs-fn {#'link-plan/retained-effect-evidence? (constantly false)
                     #'compiled/require-no-evidence-events! (fn [_] (swap! checks inc) (throw event-error))}
      #(is (= :compiled-evidence-initialization (run))))
    (is (zero? @checks))
    (with-redefs-fn {#'link-plan/retained-effect-evidence? (constantly true)
                     #'compiled/require-no-evidence-events! (fn [_] (throw event-error))}
      #(is (= :unit-pending-event (run))))
    (with-redefs-fn {#'link-plan/retained-effect-evidence? (constantly true)
                     #'compiled/require-no-evidence-events! (fn [_])
                     #'gpu-link/execution-order (constantly {:record-time-prologue [:unit]})}
      #(is (= :compiled-evidence-record-time-prologue (run))))
    (with-redefs-fn {#'link-plan/retained-effect-evidence? (constantly true)
                     #'compiled/require-no-evidence-events! (fn [_])
                     #'gpu-link/execution-order (constantly {})}
      #(is (= :compiled-evidence-noncontiguous (run))))))

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

(defn- with-temporary-directory [f]
  (let [directory (Files/createTempDirectory "raster-equation-artifacts-"
                                             (make-array java.nio.file.attribute.FileAttribute 0))]
    (try
      (f directory)
      (finally
        (with-open [paths (Files/walk directory (make-array java.nio.file.FileVisitOption 0))]
          (doseq [^Path path (iterator-seq (.iterator (.sorted paths (Comparator/reverseOrder))))]
            (Files/deleteIfExists path)))))))

(deftest selected-artifact-requires-independent-math-request
  (let [request {:scalar-math {:overrides {[:tanh :float] :f64-target-library-rte-f32}}}
        original (equation-first/compile #'artifact-map (merge {:target target :dtype :float} request))
        envelope (artifact/seal identity original request)
        restored (artifact/open identity (artifact/decode (artifact/encode envelope)) request)
        decodes (atom 0)]
    (is (= original restored))
    (is (identical? (:emitted restored) (emitted-program/validate! (:emitted restored) request)))
    (is (thrown? clojure.lang.ExceptionInfo (artifact/seal identity original)))
    (is (thrown? clojure.lang.ExceptionInfo (artifact/open identity envelope))
        "serialized options cannot grant their own math permission")
    (with-redefs [boring/decode (fn [& _] (swap! decodes inc)
                                (throw (ex-info "unexpected payload decode" {})))]
      (is (= :scalar-math-policy
             (reason-of #(artifact/open identity envelope
                                        {:scalar-math {:overrides {} :unknown true}})))))
    (is (zero? @decodes))))

(deftest selected-template-survives-process-cache-clear-under-same-request
  (with-temporary-directory
    (fn [directory]
      (let [request {:scalar-math {:overrides {[:tanh :float] :f64-target-library-rte-f32}}}
            original (equation-first/compile #'artifact-map (merge {:target target :dtype :float} request))
            cache (store/make-store {:root (.toFile directory)})
            key {:semantic-fingerprint (:semantic-request-fingerprint identity)
                 :persistent-cache-eligible? true
                 :semantic-request {:compiler-build-fingerprint (:compiler-build-fingerprint identity)
                                    :source {:source-dependency-fingerprint (:source-dependency-fingerprint identity)}
                                    :target {:descriptor-fingerprint (:target-descriptor-fingerprint identity)}}}
            report (atom nil)
            calls (atom 0)
            resolve! (fn []
                       (binding [compiled/*equation-artifact-store* cache
                                 compiled/*compilation-template-observer* #(reset! report %)]
                         (#'compiled/cached-compilation-template
                          key :equation-first #(do (swap! calls inc) original) request)))]
        (compiled/clear-compilation-cache!)
        (try
          (is (= original (resolve!)))
          (is (= :stored (get-in @report [:persistent-artifact :status])))
          (is (= :miss (:status (store/load-artifact cache (:semantic-request-fingerprint identity) identity))))
          (is (= :hit (:status (store/load-artifact cache (:semantic-request-fingerprint identity) identity request))))
          (compiled/clear-compilation-cache!)
          (is (= original (resolve!)))
          (is (= :hit (get-in @report [:persistent-artifact :status])))
          (is (= 1 @calls) "same-request durable reload must not recompile")
          (finally (compiled/clear-compilation-cache!)))))))

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

(defn- compiler-record-round-trip [value]
  (let [options {:profile :archival
                 :registry (var-get #'artifact/compiler-record-registry)
                 :on-unknown-record :error}]
    (#'artifact/restore-sequences
     (boring/decode
      (boring/encode (#'artifact/prepare-sequences value) options)
      options))))

(deftest scan-certificates-round-trip-through-the-existing-compiler-record-codec
  (let [source (soac/par-form->soac 'result
                                  '(raster.par/scan out acc 0.0 i n double
                                                    (+ acc (aget input i)))
                                  908 :dtype :double)
        operations (soac-lower/lower-scan source nil :dtype :double)
        graph (soac-lower/scan-kernel-graph source operations {})
        emitted (segop-opencl/generate-scan-kernel-graph graph :target-dialect :cuda)]
    (doseq [[node emitted-node] (map vector (:nodes graph) (:nodes emitted))
            :let [original (:operation emitted-node)
                  restored (compiler-record-round-trip original)
                  certificate (get-in restored [:provenance :scheduled-operation])]]
      (is (= original restored))
      (is (= certificate (segscan-body/validate-against-node! certificate node graph)))
      (is (= restored (scheduled-body/validate-artifact-projection! certificate restored))))))

(deftest elementary-math-realization-round-trips-through-the-compiler-record-codec
  (doseq [op [:exp :tanh :log :sin] dt [:float :double]]
    (let [expression (kernel-body/scalar-expression op dt [(kernel-body/literal 0.0 dt)])
          restored (compiler-record-round-trip expression)]
      (is (= expression restored))
      (is (= {:kind :target-library :accuracy :implementation-defined}
             (get-in restored [:options :math-realization]))))))

(deftest scheduled-preconditions-round-trip-through-the-compiler-record-codec
  (let [original (get-in @compilation [:kernels 0 :provenance :scheduled-operation])
        parameter (first (filter #(and (= :scalar (:kind %))
                                       (contains? #{:int :long} (:dtype %)))
                                 (get-in original [:body :parameters])))
        condition {:expression (:id parameter) :op :> :value 0}
        conditioned (scheduled-body/validate!
                     (assoc original :preconditions [condition]))
        restored (compiler-record-round-trip conditioned)]
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
                              [:status :reason])))
          (is (not (contains? (store/load-artifact cache "second-request" second-id)
                             :compilation-fingerprint)))
          (is (not (contains? (store/load-artifact cache "second-request" second-id)
                             :payload-fingerprint))))))))

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
            retained-artifact (atom nil)
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
          (reset! retained-artifact (:retained-artifact @report))
          (is (= #{:compilation-fingerprint :payload-fingerprint}
                 (set (keys @retained-artifact))))
          (is (every? string? (vals @retained-artifact)))
          (compiled/clear-compilation-cache!)
          (binding [compiled/*equation-artifact-store* cache
                    compiled/*compilation-template-observer* #(reset! report %)]
            (is (= @compilation
                   (#'compiled/cached-compilation-template
                    key :equation-first #(throw (ex-info "must not compile" {}))))))
          (is (= 1 @compiles))
          (is (= :hit (get-in @report [:persistent-artifact :status])))
          (is (= expected-identity (:persistent-artifact-identity @report)))
          (is (= @retained-artifact (:retained-artifact @report)))
          (binding [compiled/*equation-artifact-store* cache
                    compiled/*compilation-template-observer* #(reset! report %)]
            (#'compiled/cached-compilation-template
             key :equation-first #(throw (ex-info "must reuse process entry" {}))))
          (is (true? (:cache-hit? @report)))
          (is (= :process-cache-hit (get-in @report [:persistent-artifact :reason])))
          (is (= expected-identity (:persistent-artifact-identity @report)))
          (is (= @retained-artifact (:retained-artifact @report)))
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
            (is (nil? (:persistent-artifact-identity @report)))
            (is (= :ineligible (get-in @report [:persistent-artifact :status])))
            (is (= {} (:retained-artifact @report))))
          (with-redefs [store/store-artifact!
                        (fn [& _] (throw (ex-info "simulated unavailable artifact store" {})))]
            (binding [compiled/*equation-artifact-store* cache
                      compiled/*compilation-template-observer* #(reset! report %)]
              (#'compiled/cached-compilation-template
               (assoc key :semantic-fingerprint "failed-store-request") :equation-first
               (fn [] @compilation)))
            (is (true? (:persistent-cache-eligible? @report)))
            (is (= :write-failed (get-in @report [:persistent-artifact :status])))
            (is (= {} (:retained-artifact @report))))
          (finally
            (compiled/clear-compilation-cache!)))))))

(deftest exact-bound-program-evidence-does-not-attest-input-bytes
  ;; Synthetic packaged-build evidence exercises the identity boundary, not release provenance.
  (let [build (build-manifest/manifest-identity
               {:schema-version build-manifest/schema-version :library 'raster/raster
                :version "identity-test-only" :revision "synthetic-identity-test"
                :runtime {:java-version (System/getProperty "java.version")
                          :clojure-version (clojure-version)}
                :source-namespaces '[raster.arrays raster.numeric raster.par]
                :dependencies {}})
        options {:compiler :equation-first :target target :dtype :float}]
    (compiled/clear-compilation-cache!)
    (try
      (with-temporary-directory
        (fn [directory]
          (binding [compiled/*equation-artifact-store*
                    (store/make-store {:root (.toFile directory) :max-entries 4
                                       :max-bytes (* 4 1024 1024)})]
            (with-redefs [build-manifest/current-identity (constantly build)]
              (let [p (compiled/lower #'artifact-map [(float-array [1 2 3 4]) 4] options)
                    q (compiled/lower #'artifact-map [(float-array [9 8 7 6]) 4] options)
                    a (compiled/execution-identity p)
                    b (compiled/execution-identity q)
                    input (:key (first (:in-tree p)))
                    output (:key (first (:out-tree p)))
                    composition (fn [connections]
                                  (compiled/compose
                                   {:id :identity/twice
                                    :components [{:id :a :program p} {:id :b :program q}]
                                    :connections connections
                                    :outputs [{:key :result :from [:b output]}]}))
                    connected (composition [{:from [:a output] :to [:b input]}])
                    separate (composition [])]
                (is (= :exact-bound-program (:scope a)))
                (is (string? (:fingerprint a)))
                (is (= a b) "host array identity and contents are deliberately not attested")
                (let [interface (compiled/producer-interface p)
                      updated (compiled/lower #'artifact-state! [(:default (first (:in-tree p))) 4]
                                              (assoc options :donate '[state]))
                      state-interface (compiled/producer-interface updated)
                      shared (compiled/compose
                              {:id :producer/shared-state
                               :components [{:id :update :program updated} {:id :forward :program p}]
                               :mutable-shares [{:owner [:update :state] :borrowers [[:forward input]]
                                                 :output [:update :state']}]
                               :outputs [{:key :updated :from [:update :state']}
                                         {:key :prediction :from [:forward output]}]})
                      shared-interface (compiled/producer-interface shared)]
                  (is (= :raster.compiled/producer-interface-v1 (:kind interface)))
                  (is (= (select-keys a [:scope :fingerprint]) (:program interface)))
                  (is (= interface (compiled/producer-interface q)))
                  (is (= [(select-keys (first (:out-tree p)) [:key :node :dtype :shape])]
                         (:outputs interface)))
                  (is (empty? (:post-state interface)) "read-only input roots are not mutable state")
                  (is (= [(select-keys (first (:in-tree updated)) [:key :node :dtype :shape])]
                         (:post-state state-interface)))
                  (is (= [:updated :prediction] (mapv :key (:outputs shared-interface))))
                  (is (= [[:update :state]] (mapv :key (:post-state shared-interface))))
                  (is (not-any? #(= [:forward input] (:key %)) (:post-state shared-interface))
                      "mutable borrowers do not become independent public post-state ports")
                  (is (= :compiled-execution-identity-owner
                         (reason-of #(compiled/producer-interface (assoc p :target (:target p))))))
                  (is (= :compiled-producer-prepared
                         (reason-of #(compiled/producer-interface {})))))
                (let [interface (compiled/producer-interface connected)]
                  (is (= [(select-keys (first (:out-tree connected)) [:key :node :dtype :shape])]
                         (:outputs interface)))
                  (is (= [:result] (mapv :key (:outputs interface))))
                  (is (empty? (:post-state interface)))
                  (is (not-any? #(= [:a output] (:key %)) (:outputs interface))
                      "connected component outputs are not public composition exports"))
                ;; The real composition boundary already rejects duplicate output nodes.
                ;; Restore independently rejects aliases too, rather than relying on this
                ;; upstream invariant for every future public-port source.
                (is (= :link-output-duplicates
                       (reason-of #(compiled/compose
                                    {:id :producer/aliased
                                     :components [{:id :a :program p}]
                                     :outputs [{:key :first :from [:a output]}
                                               {:key :second :from [:a output]}]}))))
                (is (false? (:attests-input-bytes? a)))
                (with-redefs [gpu-link/instantiate-certified! (fn [& _] ::bound-executable)
                              gpu-link/instantiate! (fn [& _] ::independently-bound-executable)]
                  (let [c (compiled/instantiate! p)]
                    (is (identical? p (:prepared c)))
                    (is (= ::bound-executable (:executable c)))
                    (is (= a (compiled/execution-identity c)))
                    (with-redefs [build-manifest/current-identity (constantly nil)]
                      (is (= a (compiled/execution-identity c))
                          "inspection retains the bound artifact rather than current build/cache state"))
                    (doseq [changed [(assoc c :executable ::foreign-executable)
                                     (assoc c :in-tree [])
                                     (assoc c :prepared q)]]
                      (is (= :compiled-execution-identity-owner
                             (reason-of #(compiled/execution-identity changed)))))
                    (is (= :compiled-execution-identity-owner
                           (reason-of #(compiled/execution-identity
                                        (compiled/instantiate! (assoc p :target (:target p))))))))
                  (with-redefs [compiled/execution-identity
                                (fn [& _] (throw (ex-info "unexpected hot-path identity hashing" {})))]
                    (is (compiled/compiled? (compiled/instantiate! p)))))
                (is (= [input] (mapv :key (:data-slots a))))
                (is (not= (:fingerprint a)
                          (:fingerprint (compiled/execution-identity
                                         (compiled/lower #'artifact-map [(float-array 8) 8] options)))))
                (is (not= (:fingerprint a)
                          (:fingerprint (compiled/execution-identity
                                         (compiled/lower #'artifact-map [(float-array 4) 4]
                                                         (assoc options :constants '[input]))))))
                (let [scale-identity (fn [factor]
                                       (:fingerprint
                                        (compiled/execution-identity
                                         (compiled/lower #'artifact-scale [(float-array 4) factor 4]
                                                         options))))
                      nan (fn [bits] (Float/intBitsToFloat (int bits)))]
                  (is (not= (scale-identity (float 2.0)) (scale-identity (float 3.0))))
                  (is (not= (scale-identity (float 0.0)) (scale-identity (float -0.0))))
                  (is (= (scale-identity (nan 0x7fc00012)) (scale-identity (nan 0x7fc00012))))
                  (is (not= (scale-identity (nan 0x7fc00012)) (scale-identity (nan 0x7fc00013)))))
                (is (= :compiled-execution-identity-owner
                       (reason-of #(compiled/execution-identity (assoc p :target :changed)))))
                (is (= 1 (count (:data-slots (compiled/execution-identity connected)))))
                (is (= 2 (count (:data-slots (compiled/execution-identity separate)))))
                (is (not= (:fingerprint (compiled/execution-identity connected))
                          (:fingerprint (compiled/execution-identity separate))))
                (let [nested (compiled/compose
                              {:id :identity/nested
                               :components [{:id :inner :program connected}]
                               :outputs [{:key :result :from [:inner :result]}]})]
                  (is (= 1 (count (:data-slots (compiled/execution-identity nested))))))
                (compiled/clear-compilation-cache!)
                (is (= a (compiled/execution-identity
                          (compiled/lower #'artifact-map [(float-array 4) 4] options)))
                    "persistent loads retain the exact compilation's generated IDs"))))))
      (with-redefs [build-manifest/current-identity (constantly nil)]
        (is (= :compiled-execution-identity-incomplete
               (reason-of #(compiled/execution-identity
                            (compiled/lower #'artifact-map [(float-array 4) 4] options))))))
      (finally (compiled/clear-compilation-cache!)))))
