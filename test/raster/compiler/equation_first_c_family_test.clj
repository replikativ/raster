(ns raster.compiler.equation-first-c-family-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing use-fixtures]]
            [raster.arrays]
            [raster.compiler.compatibility-ledger-test :as ledger]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.backend.gpu.parallel-program-c-family :as program-c-family]
            [raster.compiler.core.hardware :as compiler-hardware]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.equation-artifact :as equation-artifact]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.compiler.fixtures.checked-casts :as checked-casts]
            [raster.compiler.fixtures.extrema :as extrema]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.fixtures.contractions :as contractions]
            [raster.compiler.fixtures.scalar-helpers :as scalar-helpers]
            [raster.compiler.fixtures.symbolic-storage :as symbolic-storage]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-equation-dispatch :as equation-dispatch]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.kernel-graph-call :as graph-call]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.semantic-fingerprint :as semantic-fingerprint]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
            [raster.compiler.passes.parallel.contraction-schedule :as contraction-schedule]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.typed-contraction-context :as contraction-context]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.passes.parallel.structured-control-route :as structured-route]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.parallel-program :as program-runtime]
            [raster.core :refer [deftm]]
            [raster.dl.attention :as attention]
            [raster.dl.array-ops :as array-ops]
            [raster.dl.diffusion :as diffusion]
            [raster.dl.loss :as loss]
            [raster.dl.nn :as dl-nn]
            [raster.numeric]
            [raster.nn :as nn]
            [raster.ode.pde :as pde]
            [raster.ode.multilevel :as multilevel]
            [raster.par]
            [raster.runtime.hardware :as hardware]))

(def ^:private cuda-target :cuda:equation-first-source-test)
(def ^:private hip-target :hip:equation-first-source-test)
(def ^:private hip-matrix-target :hip:equation-first-matrix-test)
(def ^:private ocl-target :ocl:equation-first-source-test)
(def ^:private intel-matrix-target :ocl:equation-first-intel-matrix-test)

(deftest equation-template-owns-static-proof-with-fresh-final-validation
  (compiled/clear-compilation-cache!)
  (try
    ;; Resolve the existing fixture at test execution, after its deftm has been defined.
    (let [source (ns-resolve 'raster.compiler.equation-first-c-family-test 'c-family-elementwise)
          checks (atom 0)
          original emitted-program/validate-with-physical-results!
          prepare (fn [] (compiled/lower source [(float-array 8) 8]
                                         {:compiler :equation-first :target cuda-target :dtype :float}))]
      (with-redefs [emitted-program/validate-with-physical-results!
                    (fn [program] (swap! checks inc) (original program))]
        (let [first-prepared (prepare)
              cold-checks @checks
              second-prepared (prepare)
              warm-checks (- @checks cold-checks)
              [key entry] (first @(var-get #'compiled/compilation-template-cache))
              compilation @(:value entry)
              owner {:key key :entry entry}
              epoch (dispatch/compiler-definition-revision)]
          (is (= 1 cold-checks) "cold preparation derives the owner static program proof once")
          (is (= 0 warm-checks) "warm construction reuses only that exact static program proof")
          (is (emitted-program/retained-validation?
                (:emitted compilation) (#'compiled/owned-emitted-validation owner compilation)))
          (is (not (contains? compilation :emitted-validation))
              "process-local proof is not added to the ordinary persisted compilation value")
          (is (= [false true] (mapv #(get-in (compiled/preparation-report %) [:template :cache-hit?])
                                   [first-prepared second-prepared])))
          (is (not (identical? (get-in first-prepared [:lowering :plan :nodes
                                                     (get-in first-prepared [:in-tree 0 :node]) :source])
                               (get-in second-prepared [:lowering :plan :nodes
                                                      (get-in second-prepared [:in-tree 0 :node]) :source]))))
          (invocation-link/verify! (:lowering second-prepared))
          (is (= (+ cold-checks warm-checks 1) @checks) "public verification remains independent")
          (with-redefs [dispatch/compiler-definition-revision (constantly (inc epoch))]
            (is (nil? (#'compiled/owned-emitted-validation owner compilation))))
          (with-redefs [equation-first/compile (fn [& _] nil)]
            (is (nil? (#'compiled/owned-emitted-validation owner compilation))))
          (with-redefs [emitted-program/validate-with-physical-results! original]
            (is (nil? (#'compiled/owned-emitted-validation owner compilation))))
          (is (nil? (#'compiled/owned-emitted-validation owner (assoc compilation :stats {})))))))
    (finally (compiled/clear-compilation-cache!))))

(deftest final-plan-static-proof-is-scoped-after-projection-and-by-program-identity
  (let [source (ns-resolve 'raster.compiler.equation-first-c-family-test 'c-family-elementwise)
        prepared (compiled/lower source [(float-array 8) 8]
                                 {:compiler :equation-first :target cuda-target :dtype :float})
        plan (get-in prepared [:lowering :plan])
        instance (first (:instances plan))
        call (:call instance)
        evidence (emitted-program/validate-with-physical-results! (:program call))
        original emitted-program/validate-with-physical-results!
        checks (atom 0)
        scope (ns-resolve 'raster.compiler.ir.link-plan '*retained-program-validations*)
        input-node (get-in prepared [:in-tree 0 :node])
        output-node (first (:outputs plan))
        reason (fn [check value]
                 (try (check value) nil
                      (catch clojure.lang.ExceptionInfo error
                        [(.getMessage error) (ex-data error)])))]
    (with-redefs [emitted-program/validate-with-physical-results!
                  (fn [program] (swap! checks inc) (original program))]
      (let [result (link-plan/make-with-final-projection
                    plan (fn [candidate]
                           (is (nil? @scope) "projection callback cannot inherit static authority")
                           (program-call/validate! call)
                           {:plan candidate :projection :checked}) evidence)]
        (is (= 1 @checks) "only the independent callback validation rederives the program")
        (is (= :checked (:projection result)))
        (is (= (:nodes plan) (get-in result [:plan :nodes])))
        (is (not (contains? (:plan result) :retained-validation)))
        (link-plan/validate! (:plan result))
        (is (= 2 @checks) "later public plan validation independently checks the program"))
      (reset! checks 0)
      (let [other (assoc instance :id ::other
                         :call (assoc call :program (with-meta (:program call) {:other true})))
            mixed (assoc plan :instances [instance other])]
        (link-plan/validate-with-effect-evidence! mixed evidence)
        (is (= 1 @checks) "a structurally equal but different program validates independently")))
    (doseq [[label invalid]
            [[:roles (assoc-in plan [:instances 0 :roles (first (keys (:outputs call)))] :constant)]
             [:range (assoc-in plan [:nodes output-node :view :shape] [1])]
             [:aliases (assoc plan :aliases #{#{::missing output-node}})]
             [:initialization (-> plan
                                  (assoc-in [:nodes input-node :source] nil)
                                  (assoc-in [:nodes input-node :role] :internal))]]]
      (testing (name label)
        (let [independent (reason link-plan/validate! invalid)]
          (is (some? independent))
          (is (= independent
                 (reason #(link-plan/validate-with-effect-evidence! % evidence) invalid))))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (link-plan/validate-with-effect-evidence!
                  plan (with-meta evidence (assoc (meta evidence) :copied true)))))
    (is (nil? @scope) "failed validation restores its construction scope")
    (is (link-plan/link-plan? (link-plan/validate! plan)))))

(deftest storage-projection-reuses-only-owned-static-evidence
  (compiled/clear-compilation-cache!)
  (try
    (let [source (ns-resolve 'raster.compiler.equation-first-c-family-test 'c-family-elementwise)
          args [(float-array 8) 8]
          _ (compiled/lower source args
                            {:compiler :equation-first :target cuda-target :dtype :float})
          [key entry] (first @(var-get #'compiled/compilation-template-cache))
          compilation @(:value entry)
          evidence (#'compiled/owned-emitted-validation {:key key :entry entry} compilation)
          original emitted-equation/validate-with-physical-results
          projections (atom 0)]
      (with-redefs [emitted-equation/validate-with-physical-results
                    (fn [boundary] (swap! projections inc) (original boundary))]
        (let [fresh (equation-first/lower compilation args)
              independent-count @projections
              retained (:plan (equation-first/lower compilation args (fn [plan] {:plan plan})
                                                    evidence))]
          (is (= 1 independent-count)
              "ordinary lowering independently proves the boundary and derives storage together")
          (is (= independent-count @projections) "owned projection avoids only static re-derivation")
          (is (= (:values fresh) (:values retained)))
          (is (= (:outputs fresh) (:outputs retained)))
          (is (= (:aliases fresh) (:aliases retained))))))
    (finally (compiled/clear-compilation-cache!))))

(deftest static-proof-failure-evicts-the-owner-and-allows-retry
  (compiled/clear-compilation-cache!)
  (try
    (let [source (ns-resolve 'raster.compiler.equation-first-c-family-test 'c-family-elementwise)
          prepare #(compiled/lower source [(float-array 8) 8]
                                   {:compiler :equation-first :target cuda-target :dtype :float})
          original emitted-program/validate-with-physical-results!
          fail? (atom true)
          failure (ex-info "static validation failed" {:reason ::static-proof-probe})]
      (with-redefs [emitted-program/validate-with-physical-results!
                    (fn [program]
                      (if (compare-and-set! fail? true false) (throw failure) (original program)))]
        (is (identical? failure (try (prepare) nil (catch clojure.lang.ExceptionInfo error error))))
        (is (zero? (:entries (compiled/compilation-cache-stats))))
        (is (compiled/prepared? (prepare)))
        (is (= 1 (:entries (compiled/compilation-cache-stats))))))
    (finally (compiled/clear-compilation-cache!))))

(deftest delayed-static-proof-rechecks-owner-guards-after-resolution
  (compiled/clear-compilation-cache!)
  (try
    (let [source (ns-resolve 'raster.compiler.equation-first-c-family-test 'c-family-elementwise)
          _ (compiled/lower source [(float-array 8) 8]
                            {:compiler :equation-first :target cuda-target :dtype :float})
          cache (var-get #'compiled/compilation-template-cache)
          [key original-entry] (first @cache)
          compilation @(:value original-entry)
          validated @(:emitted-validation original-entry)
          epoch (dispatch/compiler-definition-revision)
          pipeline-root @#'equation-first/compile
          validator-root @#'emitted-program/validate-with-physical-results!]
      (doseq [change [:epoch :pipeline :validator :owner]]
        (let [revision (atom epoch)
              resolutions (atom 0)
              entry (assoc original-entry :emitted-validation
                           (delay
                             (swap! resolutions inc)
                             (case change
                               :epoch (swap! revision inc)
                               :pipeline (alter-var-root #'equation-first/compile
                                                         (constantly (fn [& args] (apply pipeline-root args))))
                               :validator (alter-var-root #'emitted-program/validate-with-physical-results!
                                                          (constantly (fn [program] (validator-root program))))
                               :owner (swap! cache assoc key (assoc original-entry :replacement true)))
                             validated))]
          (swap! cache assoc key entry)
          (with-redefs [dispatch/compiler-definition-revision #(deref revision)
                        equation-first/compile pipeline-root
                        emitted-program/validate-with-physical-results! validator-root]
            (is (nil? (#'compiled/owned-emitted-validation {:key key :entry entry} compilation))
                (str change " drift during resolution cannot authorize reuse"))
            (is (= 1 @resolutions)))
          (swap! cache assoc key original-entry)))
      (let [replacement (assoc original-entry :replacement true)
            failure (ex-info "failed after replacement" {:reason ::owner-replaced})
            entry (assoc original-entry :emitted-validation
                         (delay (swap! cache assoc key replacement) (throw failure)))]
        (swap! cache assoc key entry)
        (is (identical? failure
                        (try (#'compiled/owned-emitted-validation {:key key :entry entry} compilation)
                             nil (catch clojure.lang.ExceptionInfo error error))))
        (is (identical? replacement (get @cache key))
            "failed proof must not evict a concurrent replacement owner")))
    (finally (compiled/clear-compilation-cache!))))

(deftest symbolic-storage-prefixes-are-range-checked-before-allocation
  (doseq [target [ocl-target cuda-target hip-target]]
    (is (map? (compiled/lower #'symbolic-storage/prefix-map [(float-array 6) 6 4]
                             {:compiler :equation-first :target target :dtype :float})))
    (is (map? (compiled/lower #'symbolic-storage/prefix-map [(float-array 6) 4 4]
                             {:compiler :equation-first :target target :dtype :float})))
    (is (= :program-link-graph-range
           (try
             (compiled/lower #'symbolic-storage/prefix-map [(float-array 3) 4 4]
                             {:compiler :equation-first :target target :dtype :float})
             nil
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= :program-link-graph-range
           (try
             (compiled/lower #'symbolic-storage/prefix-map [(float-array 6) 6 7]
                             {:compiler :equation-first :target target :dtype :float})
             nil
             (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (thrown? clojure.lang.ExceptionInfo
                 (compiled/lower #'symbolic-storage/prefix-map [(float-array 3) 4 4]
                                 {:compiler :equation-first :target target :dtype :float
                                  :values {'x (av/tensor {:dtype :float :shape [3]})}}))
        "a supplied static capacity cannot erase the reclassified pointwise minimum")))

(deftest public-effect-storage-is-capacity-not-traversal
  (doseq [target [ocl-target cuda-target hip-target]]
    (let [lower (fn [nx nb no full owned]
                  (compiled/lower #'symbolic-storage/fill-and-read-prefix!
                                  [(float-array nx) (float-array nb) (float-array no) full owned]
                                  {:compiler :equation-first :target target :dtype :float
                                   :outputs '[out]}))]
      (is (map? (lower 32 32 16 32 16)))
      (doseq [[nx nb no full owned reason]
              [[31 32 16 32 16 :program-link-graph-range] ; input shorter than fill reads
               [32 31 16 32 16 :program-link-value-contract] ; destination shorter than fill writes
               [16 16 17 16 17 :program-link-graph-range] ; later read exceeds boundary storage
               [32 32 15 32 16 :program-link-value-contract]]] ; prefix destination too small
        (is (= reason
               (try (lower nx nb no full owned) nil
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))))

(deftest indexed-captures-retain-capacity-without-claiming-a-dense-read-proof
  (doseq [target [ocl-target cuda-target hip-target]
          [source arguments capacity]
          [[#'symbolic-storage/shifted-map [(float-array 5) 4] 5]
           [#'symbolic-storage/indirect-map [(float-array 6) (long-array [0 2 4 5]) 4] 6]
           [#'symbolic-storage/shifted-scan [(float-array 5) 4] 5]
           [#'symbolic-storage/indirect-scan [(float-array 6) (long-array [0 2 4 5]) 4] 6]]]
    (is (map? (compiled/lower source arguments
                             {:compiler :equation-first :target target :dtype :float
                              :values {'x (av/tensor {:dtype :float :shape [capacity]})
                                       'indices (av/tensor {:dtype :long :shape [4]})}}))
        "indexed captures retain caller bounds obligations, not a dense traversal certificate")))

(deftest fused-result-transform-retains-its-independent-operand-minimum
  (doseq [target [ocl-target cuda-target hip-target]]
    (let [compilation (equation-first/compile #'symbolic-storage/fused-bias-contract!
                                             {:target target :dtype :float})
          arguments [(float-array 64) (float-array 128) (float-array 8) (float-array 32)]]
      (is (= 1 (count (:kernels compilation))) "initializer and contraction are fused")
      (is (map? (equation-first/lower compilation arguments)))
      (is (thrown? clojure.lang.ExceptionInfo
                   (equation-first/lower compilation (assoc arguments 2 (float-array 7))))
          "a fused epilogue cannot erase the bias capacity requirement"))))

(use-fixtures
  :once
  (fn [f]
    (hardware/register-target-device!
     cuda-target
     {:type :cuda
      :name "Synthetic NVIDIA equation-first source target"
      :capabilities {:compute-capability [8 0]
                     :warp-size 32
                     :subgroup-sizes [32]
                     :max-workgroup-size 1024
                     :shared-local-memory 65536
                     :total-eus 108}})
    (hardware/register-target-device!
     hip-target
     {:type :hip
      :name "Synthetic AMD equation-first source target"
      :capabilities {:gfx-arch :gfx1100
                     :warp-size 32
                     :subgroup-sizes [32]
                     :max-workgroup-size 1024
                     :shared-local-memory 65536
                     :total-eus 60}})
    (hardware/register-target-device!
     hip-matrix-target
     {:type :hip
      :name "Synthetic gfx90a equation-first matrix target"
      :capabilities {:gfx-arch :gfx90a
                     :wavefront-size 64
                     :subgroup-sizes [64]
                     :max-workgroup-size 1024
                     :shared-local-memory 65536
                     :total-eus 110}})
    (hardware/register-target-device!
     ocl-target
     {:type :ocl
      :name "Synthetic portable OpenCL equation-first source target"
      :capabilities {:warp-size 32
                     :subgroup-sizes [16 32]
                     :max-workgroup-size 1024
                     :shared-local-memory 65536
                     :total-eus 32}})
    (hardware/register-target-device!
     intel-matrix-target
     {:type :ocl :vendor "Intel" :name "Synthetic Intel DPAS public dispatch"
      :capabilities {:warp-size 16 :subgroup-sizes [16 32]
                     :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}
                     :max-workgroup-size 1024 :shared-local-memory 131072 :total-eus 32}})
    (f)))

(deftest typed-source-extrema-use-shared-semantic-realization
  (doseq [target [ocl-target cuda-target hip-target]
          [source dtype _array-fn _bits op] extrema/cases
          :let [compilation (equation-first/compile source {:target target :dtype dtype})
                kernel-source (:source (first (:kernels compilation)))
                helper-name (intrinsics/c-floating-extremum-name op dtype)]]
    (is (= 1 (count (:kernels compilation))))
    (is (= 2 (count (re-seq (re-pattern (str helper-name "\\(")) kernel-source)))
        "one helper definition and one typed call, not raw target fmin/fmax")
    (is (str/includes? kernel-source (str (if (= target ocl-target)
                                          "inline " "__device__ __forceinline__ ")
                                        (name dtype) " " helper-name)))
    (is (str/includes? kernel-source "isnan(a)"))
    (is (str/includes? kernel-source "signbit(a)"))))

(deftm c-family-dot
  "A public TypedSOAC reduction compiled without a CUDA/HIP runtime or physical GPU."
  (All [T] [left :- (Array T) right :- (Array T) n :- Long] :- Double
       (raster.par/reduce
        accumulator 0.0 index n
        (raster.numeric/+
         accumulator
         (raster.numeric/* (raster.arrays/aget left index)
                           (raster.arrays/aget right index))))))

(deftm c-family-elementwise
  [input :- (Array float) n :- Long] :- (Array float)
  (let [output (float-array n)]
    (raster.par/map! output index n float
                     (raster.numeric/* (float 2.0)
                                       (raster.arrays/aget input index)))))

(deftest program-emission-retains-independent-selected-math-intent
  (let [request {:scalar-math {:overrides {[:tanh :float] :f64-target-library-rte-f32}}}
        tensor (av/tensor {:dtype :float :shape '[n]})
        source (soac/make
                (soac/default-program-facts
                 {:values {'n (av/tensor {:dtype :long :shape []}) 'input tensor 'output tensor}
                  :inputs '[n input]
                  :equations {'activation (soac/default-equation-facts)}})
                [(list '= 'activation '[output]
                       (list 'map {:index 'index :extent 'n} '[input] '[]
                             (soac/lambda-form '[x] '[(Math/tanh x)])))] '[output])
        ;; This source-free compiler-generated algorithm exercises emission, not public
        ;; invocation promotion; schedule-program independently validates the typed union.
        semantic (assoc (typed-route/program-envelope source) :dialect :typed-parallel)]
    (doseq [[target dialect] [[ocl-target :opencl-portable] [cuda-target :cuda] [hip-target :hip]]]
      (let [options {:target-device target :target-descriptor (compiler-hardware/descriptor-for target)
                     :target-dialect dialect :dtype :float
                     :array-types {'input :float 'output :float} :scalar-types {'n :long}}
            scheduled (structured-route/schedule-program semantic options)
            ordinary (program-c-family/emit-program scheduled options)
            selected (program-c-family/emit-program scheduled (merge options request))
            emitted (:program selected)]
        (is (identical? (:program ordinary) (emitted-program/validate! (:program ordinary))))
        (is (identical? emitted (emitted-program/validate! emitted request)))
        (is (thrown? clojure.lang.ExceptionInfo (emitted-program/validate! emitted)))
        (is (= (:values semantic) (:values emitted)))
        (is (some #(str/includes? (:source %) "double") (:kernels selected)))))))

(deftm c-family-four-layers
  "A multi-stage contraction fixture whose named middle value is a distinct resident tap."
  [w :- (Array double) b :- (Array double) x :- (Array double)] :- (Array double)
  (let [a (nn/dense w x b)
        middle (nn/dense w a b)
        c (nn/dense w middle b)]
    (nn/dense w c b)))

(def ^:private register-tiled-schedule
  {:typed-contraction {:strategy :register-tiled}})

(deftm c-family-mixed-matmul
  [left :- (Array float) right :- (Array float)] :- (Array float)
  (let [output (float-array 4096)]
    (raster.par/contract output [[i 64] [j 64]] [[p 64]]
                         (raster.numeric/*
                          (raster.arrays/aget left (+ (* i 64) p))
                          (raster.arrays/aget right (+ (* p 64) j)))
                         :init (float 0.0) :combine raster.numeric/+)
    output))

(deftest public-mixed-matrix-dispatch-requires-explicit-consent
  (let [options {:target intel-matrix-target :dtype :float
                 :schedule {:precision :mixed-f16-f32
                            :typed-contraction {:strategy :dispatch-mixed-matrix}}}
        compilation (equation-first/compile #'c-family-mixed-matmul options)
        operation (-> compilation :emitted :equations last :operations first)
        arguments [(float-array 4096) (float-array 4096)]
        linked (equation-first/lower compilation arguments)]
    (is (equation-dispatch/emitted-equation-dispatch? operation))
    (is (= :sequential-segments (get-in operation [:dispatch :default-strategy])))
    (is (= #{:exact :approximate-model}
           (get-in operation [:numerical-policy :permitted-modes])))
    (is (= 1 (count (get-in operation [:numerical-policy :permitted-models]))))
    (is (= :xmx-direct-tile-inputs
           (executable/strategy (-> linked :instances first :call :steps last :graph))))
    (is (= 2 (count (:kernels compilation))))
    (is (= 1 (get-in compilation [:stats :emission :contraction-dispatches])))
    (let [identity {:semantic-request-fingerprint "mixed-public-request"
                    :compiler-build-fingerprint "test-build"
                    :source-dependency-fingerprint "mixed-matmul"
                    :target-descriptor-fingerprint "synthetic-intel"}
          restored (equation-artifact/open
                    identity (equation-artifact/decode
                              (equation-artifact/encode
                               (equation-artifact/seal identity compilation))))]
      (is (semantic-fingerprint/equivalent? compilation restored))
      (is (= linked (equation-first/lower restored arguments))))
    (is (= :equation-first-matrix-consent
           (try (equation-first/compile
                 #'c-family-mixed-matmul (update options :schedule dissoc :precision))
                :accepted
                (catch clojure.lang.ExceptionInfo exception (:reason (ex-data exception))))))
    (doseq [[key requested] [[:matrix-tiles :finite] [:split-factors [2]]]]
      (is (= :equation-first-matrix-candidate-space
             (try (equation-first/compile
                   #'c-family-mixed-matmul
                   (assoc-in options [:schedule :typed-contraction key] requested))
                  :accepted
                  (catch clojure.lang.ExceptionInfo exception (:reason (ex-data exception)))))))
    (let [choice (:dispatch operation)
          fallback (equation-first/compile
                    #'c-family-mixed-matmul
                    (assoc-in options [:schedule :typed-contraction :measured-selectors]
                              {(:id choice) {:kind :fixed-strategy :strategy :sequential-segments}}))
          fallback-link (equation-first/lower fallback [(float-array 4096) (float-array 4096)])]
      (is (= :sequential-segments
             (executable/strategy (-> fallback-link :instances first :call :steps last :graph)))))
    (doseq [target [cuda-target hip-target ocl-target]]
      (let [declined (equation-first/compile #'c-family-mixed-matmul (assoc options :target target))]
        (is (not (equation-dispatch/emitted-equation-dispatch?
                  (-> declined :emitted :equations last :operations first))))
        (is (seq (get-in declined [:stats :emission :contraction-candidate-declines])))))))

(deftm c-family-dynamic-mixed-matmul
  [left :- (Array float) right :- (Array float)
   m :- Integer n :- Integer k :- Integer] :- (Array float)
  (let [output (float-array (* m n))]
    (raster.par/contract output [[i m] [j n]] [[p k]]
                         (raster.numeric/*
                          (raster.arrays/aget left (+ (* i k) p))
                          (raster.arrays/aget right (+ (* p n) j)))
                         :init (float 0.0) :combine raster.numeric/+)
    output))

(deftest public-mixed-binding-falls-back-on-physical-shape-preconditions
  (doseq [[source shape-value width] [[#'c-family-dynamic-mixed-matmul int :int]
                                     [#'contractions/dynamic-matmul long :long]]]
   (let [compilation (equation-first/compile
                     source
                     {:target intel-matrix-target :dtype :float
                      :schedule {:precision :mixed-f16-f32
                                 :typed-contraction {:strategy :dispatch-mixed-matrix}}})
        operation (-> compilation :emitted :equations last :operations first)]
    (is (equation-dispatch/emitted-equation-dispatch? operation))
    (doseq [candidate (:alternatives operation)]
      (is (= #{[width width]}
             (into #{} (comp (filter #(= :scalar (:kind %)))
                            (map (juxt :dtype :kernel-dtype)))
                   (executable/abi (:graph candidate))))
          "schedule choice cannot change the declared integral public carrier"))
    (doseq [[k strategy] [[64 :xmx-direct-tile-inputs] [63 :sequential-segments]]]
      (let [linked (equation-first/lower compilation
                                         [(float-array (* 64 k)) (float-array (* k 64))
                                          (shape-value 64) (shape-value 64) (shape-value k)])
            selected (-> linked :instances first :call :steps last :graph)]
        (is (= strategy (executable/strategy selected)))
        (is (= linked (link-plan/validate! linked))))))))

(deftest explicit-matrix-schedule-emits-typed-equations-on-vendor-targets
  (let [source
        '(let* [step (raster.par/contract C [[i 128] [j 128]] [[k 64]]
                                          (raster.numeric/*
                                           (clojure.core/aget A (+ (* i 64) k))
                                           (clojure.core/aget B (+ (* k 128) j)))
                                          :out-dtype :float)]
               step)
        schedule {:precision :mixed-f16-f32
                  :typed-contraction {:strategy :matrix}}
        array-types {'A :half 'B :half 'C :float}]
    (doseq [[target dialect instruction]
            [[cuda-target :cuda "wmma::mma_sync"]
             [hip-matrix-target :hip "rocwmma::mma_sync"]]]
      (let [descriptor (compiler-hardware/descriptor-for target)
            {:keys [form stats]}
            (pipeline/schedule-parallel-form
             source {:target-device target :target-descriptor descriptor
                     :dtype :half :array-types array-types})
            emitted (program-c-family/emit-program
                     (assoc form :dialect :scheduled-parallel)
                     {:target-device target :target-descriptor descriptor
                      :target-dialect dialect :array-types array-types
                      :schedule schedule})
            artifact (first (:kernels emitted))
            certificate (get-in artifact [:provenance :scheduled-operation])]
        (is (= :typed-soac (:source-dialect stats)))
        (is (= :matrix (get-in artifact [:attributes :strategy])))
        (is (str/includes? (:source artifact) instruction))
        (is (= artifact (scheduled-body/validate-artifact-projection!
                         certificate artifact)))))
    (let [descriptor (compiler-hardware/descriptor-for hip-target)
          {:keys [form]} (pipeline/schedule-parallel-form
                          source {:target-device hip-target :target-descriptor descriptor
                                  :dtype :half :array-types array-types})]
      (doseq [[policy expected]
              [[:mixed-f16-f32 :no-legal-matrix-tile]
               [:f32-scalar :matrix-numerical-policy]]]
        (try
          (program-c-family/emit-program
           (assoc form :dialect :scheduled-parallel)
           {:target-device hip-target :target-descriptor descriptor
            :target-dialect :hip :array-types array-types
            :schedule {:precision policy :typed-contraction {:strategy :matrix}}})
          (is false "an ineligible explicit matrix schedule must fail before target emission")
          (catch clojure.lang.ExceptionInfo exception
            (is (= :kernel-graph-contraction-schedule (:reason (ex-data exception))))
            (is (= expected (get-in (ex-data exception) [:schedule-decline :reason])))
            (is (= :none (:fallback (ex-data exception))))))))))

(defn- fixed-contraction-site
  [target]
  (let [compilation (equation-first/compile
                     #'contractions/fixed-matmul {:target target :dtype :float})
        scheduled (:scheduled compilation)
        equation (first (filter #(some (fn [operation]
                                         (= :contraction (:phase operation)))
                                       (:operations %))
                                (:equations scheduled)))
        operation (first (filter #(= :contraction (:phase %)) (:operations equation)))
        graph (:graph (equation-graph/make-for-equation scheduled equation))
        node (first (filter #(= operation (:operation %)) (:nodes graph)))
        facts (:facts (contraction-context/validate! (:algorithm equation) operation))]
    {:node node :graph graph :facts facts :algorithm (:algorithm equation)
     :descriptor (compiler-hardware/descriptor-for target)
     :options (select-keys (:options compilation) [:array-types :scalar-types])}))

(deftest register-arithmetic-choice-closes-the-complete-write-proof
  (let [{:keys [node graph facts descriptor options algorithm]} (fixed-contraction-site ocl-target)]
    (doseq [policy [:decomposed :fused]]
      (let [planned (contraction-schedule/plan-register-tiled-for-node
                     node graph facts descriptor
                     (assoc options :precision :mixed-f16-f32 :multiply-add policy))
            certificate (:scheduled planned)
            opposite (if (= policy :fused) :decomposed :fused)
            proof #(contraction-schedule/complete-write-domain algorithm node graph %)]
        (is (:ok planned))
        (is (= policy (get-in certificate [:legality :multiply-add])))
        (is (seq (proof certificate)))
        (is (nil? (proof (update certificate :legality dissoc :multiply-add))))
        (is (nil? (proof (assoc-in certificate [:legality :multiply-add] opposite))))
        (is (nil? (proof (assoc-in certificate [:body :schedule :multiply-add] opposite))))
        (is (nil? (proof (assoc-in certificate [:numerics :policy] :unproven-policy))))))))

(deftest fixed-register-tile-retains-the-equation-graph-certificate
  (doseq [target [ocl-target cuda-target hip-target] policy [:decomposed :fused]]
    (let [compilation (equation-first/compile
                       #'contractions/fixed-matmul
                       {:target target :dtype :float
                        :schedule (assoc-in register-tiled-schedule
                                            [:typed-contraction :multiply-add] policy)})]
      (let [identity {:semantic-request-fingerprint "fixed-tile-request"
                      :compiler-build-fingerprint "test-build"
                      :source-dependency-fingerprint "fixed-matmul"
                      :target-descriptor-fingerprint (str target)}
            restored (equation-artifact/open
                      identity (equation-artifact/decode
                                (equation-artifact/encode
                                 (equation-artifact/seal identity compilation))))
            arguments [(float-array 15) (float-array 21)]]
        (is (= compilation restored))
        (is (= (equation-first/lower compilation arguments)
               (equation-first/lower restored arguments))))
      (let [artifact (first (:kernels compilation))
            certificate (get-in artifact [:provenance :scheduled-operation])]
        (is (= :register-tiled (get-in artifact [:attributes :strategy])))
        (is (= (if (= policy :fused) :ordered-k-fused-multiply-add
                                    :ordered-k-decomposed-multiply-add)
               (get-in certificate [:numerics :policy])))
        (is (= artifact (scheduled-body/validate-artifact-projection! certificate artifact)))
        (is (= 2 (count (get-in artifact [:launch :workgroup-size]))))))))

(deftest public-contraction-dispatch-emits-both-certified-c-family-alternatives
  (doseq [target [ocl-target cuda-target hip-target] policy [:decomposed :fused]]
    (let [compilation (equation-first/compile
                       #'contractions/fixed-matmul
                       {:target target :dtype :float
                        :schedule {:typed-contraction {:strategy :dispatch-register-tiled
                                                       :multiply-add policy}}})
          operation (-> compilation :emitted :equations last :operations first)
          linked (equation-first/lower compilation [(float-array 15) (float-array 21)])]
      (is (equation-dispatch/emitted-equation-dispatch? operation))
      (is (= 2 (count (:kernels compilation))))
      (is (= :register-tiled
             (executable/strategy (-> linked :instances first :call :steps last :graph))))
      (let [choice (:dispatch operation)
            _ (is (= policy (get-in choice [:attributes :tuning :numerical-mode :multiply-add])))
            selected (equation-first/compile
                      #'contractions/fixed-matmul
                      {:target target :dtype :float
                       :schedule {:typed-contraction
                                  {:strategy :dispatch-register-tiled
                                   :multiply-add policy
                                   :measured-selectors
                                   {(:id choice) {:kind :fixed-strategy
                                                  :strategy :sequential-segments}}}}})
            selected-link (equation-first/lower selected [(float-array 15) (float-array 21)])]
        (is (= 2 (count (:kernels selected))))
        (is (= :sequential-segments
               (-> selected :emitted :equations last :operations first :dispatch :default-strategy)))
        (is (= :sequential-segments
               (executable/strategy (-> selected-link :instances first :call :steps last :graph)))))
      (doseq [artifact (:kernels compilation)]
        (is (= artifact
               (scheduled-body/validate-artifact-projection!
                (get-in artifact [:provenance :scheduled-operation]) artifact)))))))

(deftest fixed-register-tile-candidate-declines-before-target-emission
  (let [{:keys [node graph facts descriptor options]} (fixed-contraction-site ocl-target)]
    (doseq [[expected candidate-facts candidate-options candidate-descriptor]
            [[:register-tiled-numerical-policy facts
              (assoc options :precision :f32-scalar) descriptor]
             [:symbolic-dims (assoc-in facts [:free-axes 0 1] '(+ m 1))
              (assoc options :precision :mixed-f16-f32) descriptor]
             [:register-tiled-fp32-candidate (assoc facts :dtype :double)
              (assoc options :precision :mixed-f16-f32) descriptor]
             [:register-tiled-static-capacity
              (assoc-in facts [:free-axes 0 1] Integer/MAX_VALUE)
              (assoc options :precision :mixed-f16-f32) descriptor]
             [:register-tiled-static-capacity
              (-> facts
                  (assoc-in [:free-axes 0 1] 65536)
                  (assoc-in [:free-axes 1 1] 65536)
                  (assoc-in [:contract-axes 0 1] 'depth))
              (assoc options :precision :mixed-f16-f32) descriptor]
             [:target-resources facts
              (assoc options :precision :mixed-f16-f32)
              (assoc-in descriptor [:execution :max-workgroup-size] 1)]]]
      (let [decline (contraction-schedule/plan-register-tiled-for-node
                     node graph candidate-facts candidate-descriptor candidate-options)]
        (is (false? (:ok decline)))
        (is (= expected (:reason decline)))))))

(deftest public-register-tile-request-does-not-weaken-strict-arithmetic
  (let [error (try
                (equation-first/compile
                 #'contractions/fixed-matmul
                 {:target ocl-target :dtype :float
                  :schedule {:precision :f32-scalar
                             :typed-contraction {:strategy :register-tiled}}})
                nil
                (catch clojure.lang.ExceptionInfo exception exception))]
    (is (some? error))
    (is (= :register-tiled-numerical-policy
           (get-in (ex-data error) [:schedule-decline :reason])))
    (is (= :none (:fallback (ex-data error))))))

(deftest public-contraction-measured-selectors-cannot-be-silently-ignored
  (doseq [target [ocl-target cuda-target hip-target]]
    (let [error (with-redefs [pipeline/get-walked-body
                             (fn [& _] (throw (AssertionError. "unsupported selector reached lowering")))]
                  (try
                    (equation-first/compile
                     #'contractions/fixed-matmul
                     {:target target :dtype :float
                      :schedule {:typed-contraction
                                 {:measured-selectors {"recorded-contraction" {:kind :fixed-strategy}}}}})
                    nil
                    (catch clojure.lang.ExceptionInfo exception exception)))]
      (is (= :equation-first-contraction-selector-unsupported (:reason (ex-data error))))
      (is (= [:typed-contraction :measured-selectors] (:schedule-path (ex-data error))))
      (is (= :none (:fallback (ex-data error)))))))

(deftest public-contraction-selection-preserves-default
  (let [compile-fixed (fn [strategy]
                        (equation-first/compile
                         #'contractions/fixed-matmul
                         (cond-> {:target ocl-target :dtype :float}
                           strategy (assoc :schedule {:typed-contraction {:strategy strategy}}))))
        automatic (compile-fixed nil)
        portable (compile-fixed :portable)
        certificates (fn [compilation]
                       (mapv #(select-keys (get-in % [:provenance :scheduled-operation])
                                           [:numerics :legality :attributes])
                             (:kernels compilation)))]
    (is (= (certificates automatic) (certificates portable)))
    (is (= :same-typed-ssa-evaluation-order
           (get-in (first (certificates automatic)) [:numerics :policy])))))

(deftest dynamic-register-tiles-preserve-preallocation-shape-obligations
  (doseq [target [ocl-target cuda-target hip-target]]
    (let [compilation (equation-first/compile
                       #'contractions/dynamic-matmul
                       {:target target :dtype :float :schedule register-tiled-schedule})
          artifact (first (:kernels compilation))
          certificate (get-in artifact [:provenance :scheduled-operation])
          graph (some (fn [equation]
                        (some (fn [operation]
                                (when (some #(= artifact (:operation %))
                                            (get-in operation [:graph :nodes]))
                                  (:graph operation)))
                              (:operations equation)))
                      (get-in compilation [:emitted :equations]))
          scalars (fn [m n k]
                    (into {} (map (fn [[id value]] [id {:type :long :value value}]))
                          {'m m 'n n 'k k}))]
      (is (= :register-tiled (get-in artifact [:attributes :strategy])))
      (is (seq (:preconditions certificate)))
      (is (= (:preconditions certificate) (:preconditions artifact)))
      (is (= artifact (scheduled-body/validate-artifact-projection! certificate artifact)))
      (doseq [shape [[1 3 2] [3 5 7] [65 17 67]]]
        (is (= graph (graph-call/preflight! graph (apply scalars shape)))))
      ;; No huge buffer allocations: rejected entirely by scalar preflight.
      (doseq [shape [[0 3 2] [1 -1 2] [1 3 0]
                     [65536 1 65536] [65536 65536 1] [1 65536 65536]
                     [Long/MAX_VALUE 1 1]]]
        (is (thrown? clojure.lang.ExceptionInfo
                     (graph-call/preflight! graph (apply scalars shape))))))))

(deftest public-transposed-projection-retains-generated-schedule-certificate
  (doseq [target [ocl-target cuda-target hip-target]]
    (let [compilation (equation-first/compile
                       #'dl-nn/linear-nb
                       {:target target :dtype :float :schedule register-tiled-schedule})
          artifact (first (:kernels compilation))
          certificate (get-in artifact [:provenance :scheduled-operation])]
      (is (= :register-tiled (get-in artifact [:attributes :strategy])))
      (is (= :nt (get-in certificate [:legality :variant])))
      (is (= :nt (get-in certificate [:attributes :variant])))
      (is (= (:preconditions certificate) (:preconditions artifact)))
      (is (= artifact (scheduled-body/validate-artifact-projection! certificate artifact))))))

(deftest typed-contraction-strategy-participates-in-the-public-template-identity
  (compiled/clear-compilation-cache!)
  (try
    (let [arguments [(float-array 15) (float-array 21)]
          prepare (fn [strategy]
                    (compiled/lower
                     #'contractions/fixed-matmul arguments
                     (cond-> {:compiler :equation-first :target cuda-target :dtype :float}
                       strategy
                       (assoc :schedule {:typed-contraction {:strategy strategy}}))))
          automatic (prepare nil)
          register-tiled (prepare :register-tiled)
          register-tiled-again (prepare :register-tiled)]
      (is (= [:auto :register-tiled :register-tiled]
             (mapv #(get-in % [:schedule :typed-contraction :strategy])
                   [automatic register-tiled register-tiled-again])))
      (is (= [false false true]
             (mapv #(get-in (compiled/preparation-report %) [:template :cache-hit?])
                   [automatic register-tiled register-tiled-again])))
      (doseq [prepared [automatic register-tiled register-tiled-again]
              :let [report (compiled/preparation-report prepared)
                    phases (:phases-ns report)
                    lowering (:lowering prepared)]]
        (is (link-plan/retained-effect-evidence?
             (:plan lowering) (get-in lowering [:certificate :effect-evidence])))
        (is (= #{:equation-lower :role-projection :invocation-certification}
               (set (keys phases))))
        (is (every? #(and (integer? %) (not (neg? %))) (vals phases)))
        (is (<= (reduce + (vals phases)) (:link-plan-lowering-ns report)))
        (let [equation-phases (:equation-lower-phases-ns report)]
          (is (= #{:materialization-ns :link-plan-construction-ns}
                 (set (keys equation-phases))))
          (is (every? #(and (integer? %) (not (neg? %)))
                      (vals equation-phases)))
          (is (<= (reduce + (vals equation-phases))
                  (:equation-lower phases)))))
      (is (= 2 (:entries (compiled/compilation-cache-stats)))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest equation-first-retains-and-validates-the-public-numerical-policy
  (doseq [target [ocl-target cuda-target hip-target]
          [operation arguments] [[#'c-family-elementwise [(float-array [1 2]) 2]]
                                 [#'dl-nn/linear-nb [(float-array 6) (float-array 6) 2 3 2]]]]
    (let [options {:target target :dtype :float}
          via-sugar (equation-first/compile operation
                                           (assoc options :gemm-precision :f32-scalar))
          via-schedule (equation-first/compile operation
                                              (assoc options :schedule {:precision :f32-scalar}))
          ;; The generated entry name is intentionally fresh, not an arithmetic difference.
          sources (fn [compilation]
                    (mapv #(str/replace (:source %) (:kernel-name %) "test_entry")
                          (:kernels compilation)))
          buffer-parameters (filter #(contains? #{:input :output :inout} (:kind %))
                                    (mapcat #(get-in % [:attributes :kernel-body :parameters])
                                            (:kernels via-sugar)))
          prepared (compiled/lower operation arguments
                                   (assoc options :compiler :equation-first
                                                  :gemm-precision :f32-scalar))]
      (is (= :f32-scalar (get-in via-sugar [:options :schedule :precision])
             (get-in via-schedule [:options :schedule :precision])
             (get-in prepared [:schedule :precision])))
      (is (not (contains? (:options via-sugar) :gemm-precision)))
      (is (seq buffer-parameters))
      (is (every? #(= :float (:dtype %)) buffer-parameters)
          "strict policy does not introduce narrowed operand storage")
      (is (= (sources via-sugar) (sources via-schedule))
          "equivalent policies preserve emitted arithmetic on every source target")
      (when (= operation #'dl-nn/linear-nb)
        (let [contractions (filter #(= :contraction
                                       (get-in % [:provenance :scheduled-operation :source :phase]))
                                   (:kernels via-sugar))]
          (is (seq contractions) "the contraction uses the common scheduled-body certificate")
          (doseq [artifact contractions
                  :let [certificate (get-in artifact [:provenance :scheduled-operation])]]
            (is (scheduled-body/scheduled-kernel-body? certificate))
            (is (= artifact (scheduled-body/validate-artifact-projection! certificate artifact)))
            (is (= :same-typed-ssa-evaluation-order (get-in certificate [:numerics :policy])))
            (is (thrown? clojure.lang.ExceptionInfo
                         (scheduled-body/validate-artifact-projection!
                          certificate (assoc artifact :arguments (vec (reverse (:arguments artifact))))))))))
      (with-redefs [pipeline/get-walked-body
                    (fn [& _] (throw (ex-info "invalid policy reached semantic compilation" {})))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"unknown compilation"
                             (equation-first/compile #'c-family-elementwise
                                                     (assoc options :gemm-precision :typo))))
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"stage"
                             (equation-first/compile #'c-family-elementwise
                                                     (assoc options :schedule {:stage {:space :typo}}))))))))

(deftm c-family-scan
  "A public certified scan lowered as a three-stage portable KernelGraph."
  [input :- (Array float) n :- Long] :- (Array float)
  (let [output (float-array n)]
    (raster.par/scan output accumulator (float 0.0) index n float
                     (raster.numeric/+ accumulator
                                       (raster.arrays/aget input index)))))

(deftm c-family-effect-map!
  [input :- (Array float) left :- (Array float) right :- (Array long) n :- Long] :- Void
  (raster.par/map-void!
   index n
   (do (raster.arrays/aset left index
                           (float (raster.numeric/* (float 2.0)
                                                    (raster.arrays/aget input index))))
       (raster.arrays/aset right index (long index)))))

(deftm c-family-case-map!
  [choices :- (Array int) output :- (Array float) n :- Long] :- Void
  (raster.par/map-void!
   index n
   (case (raster.arrays/aget choices index)
     0 (raster.arrays/aset output index (float 10.0))
     1 (raster.arrays/aset output index (float 20.0))
     nil)))

(deftm c-family-shifted-inout!
  [output :- (Array float) n :- Long] :- Void
  (raster.par/map-void!
   index n
   (raster.arrays/aset
    output index
    (raster.arrays/aget output
                        (if (< (inc index) n) (inc index) (long 0))))))

(deftm c-family-lane-owned-inout!
  [output :- (Array float) n :- Long] :- Void
  (raster.par/map-void!
   index n
   (raster.arrays/aset output index
                       (raster.numeric/+ (raster.arrays/aget output index)
                                         (float 1.0)))))

(deftm c-family-stencil
  [input :- (Array float) n :- Long] :- (Array float)
  (let [output (float-array n)]
    (raster.par/stencil!
     output [input] 1 :dirichlet float index n
     (raster.numeric/+
      (raster.arrays/aget input (dec index))
      (raster.arrays/aget input (inc index))))))

(deftm c-family-segment-sum!
  [output :- (Array float) segment-count :- Long width :- Long] :- Void
  (let [effect
        (raster.par/segmented-fold-map!
         [output] [[segment segment-count]] index width
         [[sum 0.0 :float width sum]]
         [(float sum)])]
    effect))

(deftm c-family-broadcast-product-map!
  "A short tuple reduction followed by an ordered epilogue over compiler-owned intermediates."
  [input :- (Array int), weights :- (Array int), output :- (Array int), rows :- Long] :- Void
  (let [segments (* rows 8)
        partials (int-array segments)]
    (raster.par/product-reduce!
     [partials]
     [[sum 0 :int]]
     [[row rows] [lane 8]]
     chunk 8
     [value (unchecked-add-int
             (raster.arrays/aget input (+ (* (+ (* row 8) lane) 8) chunk))
             ;; Reduction-major weights are shared across rows. This is the same verified
             ;; permutation+broadcast shape used by packed matrix/vector products.
             (raster.arrays/aget weights (+ (* chunk 8) lane)))]
     [value]
     [[left right]]
     []
     [(unchecked-add-int left right)]
     {:associative? true :commutative? true
      :overflow :wrap :order :implementation-defined})
    (raster.par/map-void!
     row rows
     (let [base (* row 8)
           total (loop [lane 0 sum 0]
                   (if (< lane 8)
                     (recur (inc lane)
                            (unchecked-add-int
                             sum (raster.arrays/aget partials (+ base lane))))
                     sum))]
       (raster.arrays/aset output row total)))))

(deftest equation-first-captures-one-target-description-for-all-stages
  (doseq [target [ocl-target cuda-target hip-target]
          [operation options]
          [[#'contractions/fixed-matmul {:dtype :float :schedule register-tiled-schedule}]
           [#'c-family-elementwise {:dtype :float}]
           [#'c-family-effect-map! {:dtype :float}]
           [#'c-family-dot {:dtype :double}]
           [#'c-family-stencil {:dtype :float}]
           [#'c-family-segment-sum! {:dtype :float}]
           [#'c-family-scan {:dtype :float}]
           [#'c-family-broadcast-product-map!
            {:dtype :int
             :values {'input (av/tensor {:dtype :int :shape [64]})
                      'weights (av/tensor {:dtype :int :shape [64]})
                      'output (av/tensor {:dtype :int :shape [1]})}}]]]
    (let [descriptor (compiler-hardware/descriptor-for target)
          calls (atom [])
          compilation
          (with-redefs [compiler-hardware/descriptor-for
                        (fn [requested]
                          (swap! calls conj requested)
                          (when (< 1 (count @calls))
                            (throw (ex-info "compilation reread mutable target facts"
                                            {:target requested})))
                          descriptor)]
            (equation-first/compile operation (assoc options :target target)))]
      (is (= [target] @calls) (str operation " " target))
      (is (identical? descriptor (get-in compilation [:options :target-descriptor])))
      (is (seq (:kernels compilation))))))

(deftest equation-template-identity-and-emission-share-the-target-snapshot
  (compiled/clear-compilation-cache!)
  (try
    (let [base (compiler-hardware/descriptor-for cuda-target)
          first-target (assoc base :calibration-version 11)
          second-target (assoc base :calibration-version 12)
          descriptions [first-target second-target first-target]
          reads (atom 0)
          built (atom [])
          original equation-first/compile]
      (with-redefs [compiler-hardware/descriptor-for
                    (fn [_] (nth descriptions (dec (swap! reads inc))))
                    equation-first/compile
                    (fn [operation options descriptor]
                      (let [result (original operation options descriptor)]
                        (swap! built conj (get-in result [:options :target-descriptor]))
                        result))]
        (let [prepared (mapv (fn [_]
                               (compiled/lower
                                #'c-family-elementwise [(float-array 8) 8]
                                {:compiler :equation-first :target cuda-target :dtype :float}))
                             (range 3))
              reports (mapv #(get-in (compiled/preparation-report %) [:template]) prepared)
              identities (mapv :semantic-fingerprint reports)]
          (is (= 3 @reads) "one capture per request, including cache hits")
          (is (= [first-target second-target] @built))
          (is (= [false false true] (mapv :cache-hit? reports)))
          (is (= (first identities) (last identities)))
          (is (not= (first identities) (second identities)))
          (is (= 2 (:entries (compiled/compilation-cache-stats)))))))
    (finally (compiled/clear-compilation-cache!))))

(deftest captured-target-must-match-the-requested-device
  (let [descriptor (compiler-hardware/descriptor-for cuda-target)
        error (try
                (equation-first/compile #'c-family-elementwise
                                        {:target hip-target :dtype :float} descriptor)
                nil
                (catch clojure.lang.ExceptionInfo exception exception))]
    (is (= :equation-first-target-description (:reason (ex-data error))))
    (is (= cuda-target (:descriptor-target (ex-data error))))))

(deftest invalid-target-snapshots-cannot-hit-a-warm-template
  (compiled/clear-compilation-cache!)
  (try
    (let [descriptor (compiler-hardware/descriptor-for cuda-target)
          current (atom descriptor)
          prepare #(compiled/lower #'c-family-elementwise [(float-array 8) 8]
                                   {:compiler :equation-first :target cuda-target :dtype :float})]
      (with-redefs [compiler-hardware/descriptor-for (fn [_] @current)]
        (prepare)
        (doseq [[invalid expected]
                [[(assoc descriptor :device-id hip-target) :equation-first-target-description]
                 [(assoc descriptor :non-data-probe (fn [] nil)) :semantic-fingerprint-unsupported]]]
          (reset! current invalid)
          (let [error (try (prepare) nil
                           (catch clojure.lang.ExceptionInfo exception exception))]
            (is (= expected (:reason (ex-data error))))
            (is (= 1 (:entries (compiled/compilation-cache-stats))))))))
    (finally (compiled/clear-compilation-cache!))))

(defn- reason-of
  [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo exception
      (ex-data exception))))

(defn- assert-certified-output-boundary
  [prepared]
  (let [escaped (mapv :node (:out-tree prepared))
        plan (compiled/plan prepared)
        certificate (compiled/certificate prepared)
        witness (invocation-link/memory-witness (:lowering prepared))
        memory (:memory witness)]
    (is (= escaped (:outputs plan))
        "every escaped DeviceArray is part of the validated LinkPlan boundary")
    (is (= escaped (:outputs certificate))
        "the certificate records the same ordered physical boundary")
    (is (= (set escaped) (:outputs (link-plan/initialization-contract plan)))
        "effect validation covers every retained output")
    (is (every? #(true? (get-in memory [:nodes % :output?])) escaped)
        "liveness sees the same physical escape boundary")
    (is (every? (fn [{:keys [storage public-output?]}]
                  (= (contains? (set escaped) storage) public-output?))
                (vals (:compiler-values witness)))
        "compiler value bindings agree with physical storage escape")
    (is (every? #(true? (get-in % [:retention :physical-storage-escaped?]))
                (filter #(and (:public-output? %)
                              (not= :storage-only (get-in % [:definition :kind])))
                        (vals (:compiler-values witness))))
        "physical output retention does not pretend to identify an earlier logical version")
    (is (every? #(contains? (:boundary-reasons %) :physical-output)
                (keep (get-in witness [:value-versions :storage]) escaped)))))

(deftest equation-first-rejects-explicit-host-orchestration-before-lowering
  (is (= :equation-first-host-only
         (:reason (reason-of #(equation-first/compile
                              #'pde/solve-fixed-step
                              {:target cuda-target :dtype :double}))))))

(defn- nested-operations
  [operations]
  (mapcat (fn [operation]
            (cons operation
                  (concat (nested-operations (or (:operations operation) []))
                          (nested-operations (or (:then-operations operation) []))
                          (nested-operations (or (:else-operations operation) [])))))
          operations))

(deftest public-equation-first-compilation-emits-cuda-and-hip-source
  (doseq [[target program-dialect module-target]
          [[cuda-target :cuda-parallel :cuda-c]
           [hip-target :hip-parallel :hip-cpp]]]
    (testing (name target)
      (let [compilation (equation-first/compile
                         #'c-family-dot {:target target :dtype :float})
            linked (equation-first/lower
                    compilation [(float-array 8) (float-array 8) 8])
            kernels (:kernels compilation)]
        (is (= program-dialect (get-in compilation [:emitted :dialect])))
        (is (= :none (get-in compilation [:stats :fallback])))
        (is (= 2 (count kernels)) "the scheduled reduction owns both emitted phases")
        (is (every? #(= module-target (:target %)) kernels))
        (is (every? #(get-in % [:attributes :kernel-body]) kernels))
        (is (every? #(str/includes? (:source %) "__global__ void") kernels))
        (is (not-any? #(re-find #"__kernel|get_global_id|get_local_id" (:source %)) kernels))
        (is (= 0 (get-in linked [:attributes :driver-allocations])))
        (is (= #{'left 'right}
               (set (keys (get-in linked [:attributes :public-buffer-bindings])))))
        (is (= {'left :input 'right :input}
               (get-in linked [:attributes :public-buffer-roles])))
        (is (= 1 (count (:outputs linked))))
        (is (= (set (:outputs linked))
               (:complete-writes (link-plan/initialization-contract linked)))
            "a scalar reduction establishes its complete one-element result")
        (is (= (:emitted compilation)
               (emitted-program/validate! (:emitted compilation))))))))

(deftest equation-first-link-retains-public-output-and-state-roles
  (let [output (float-array 8)
        effect-plan
        (equation-first/lower
         (equation-first/compile #'c-family-effect-map!
                                 {:target cuda-target :dtype :float})
         [(float-array 8) output (long-array 8) 8])
        state-plan
        (equation-first/lower
         (equation-first/compile #'c-family-lane-owned-inout!
                                 {:target cuda-target :dtype :float})
         [output 8])]
    (is (= {'input :input 'left :output 'right :output}
           (get-in effect-plan [:attributes :public-buffer-roles])))
    (is (= {'output :state}
           (get-in state-plan [:attributes :public-buffer-roles])))
    (is (= (get-in effect-plan [:attributes :public-buffer-bindings 'left])
           (some (fn [[id node]] (when (identical? output (:source node)) id))
                 (:nodes effect-plan))))))

(deftest compiled-artifact-consumes-the-equation-first-link-contract
  (let [input (float-array 8)
        output (float-array 8)
        functional (compiled/lower #'c-family-elementwise [input 8]
                                   {:compiler :equation-first
                                    :target cuda-target :dtype :float})
        effect (compiled/lower #'c-family-effect-map!
                               [input output (long-array 8) 8]
                               {:compiler :equation-first
                                :target cuda-target :dtype :float
                                :outputs '[left right]})
        state (compiled/lower #'c-family-lane-owned-inout! [output 8]
                              {:compiler :equation-first
                               :target cuda-target :dtype :float})]
    (is (every? compiled/prepared? [functional effect state]))
    (is (every? #(true? (get-in % [:descriptor :equation-first?]))
                [functional effect state]))
    (is (= [[:input :input]]
           (mapv (juxt :key :role) (:in-tree functional))))
    (is (= [:result] (mapv :key (:out-tree functional))))
    (is (= [[:input :input] [:left :output] [:right :output]]
           (mapv (juxt :key :role) (:in-tree effect))))
    (is (= [:left :right] (mapv :key (:out-tree effect))))
    (is (= [[:output :state]]
           (mapv (juxt :key :role) (:in-tree state))))
    (is (every? link-plan/link-plan?
                (map compiled/plan [functional effect state])))
    (is (invocation-link/certificate? (compiled/certificate functional)))
    (let [{:keys [compiler-buffer-bindings compiler-values memory value-versions]}
          (invocation-link/memory-witness (:lowering functional))]
      (is (= :semantic-equation-read-before-write (:ordering value-versions)))
      (is (= :unproven (:completion value-versions)))
      (is (= :witnessed (:status value-versions)))
      (is (every? #(not= :storage-only
                         (get-in compiler-values [% :definition :kind]))
                  (mapcat :versions (vals (:storage value-versions)))))
      (is (every? #(map? (:retention %))
                  (filter #(not= :storage-only (get-in % [:definition :kind]))
                          (vals compiler-values))))
      (is (= (set (keys compiler-buffer-bindings)) (set (keys compiler-values))))
      (is (every? #(contains? (:values memory) %)
                  (vals compiler-buffer-bindings)))
      (is (every? #(= (:storage (val %)) (get compiler-buffer-bindings (key %)))
                  compiler-values))
      (is (every? #(contains? (:definition %) :kind) (vals compiler-values)))
      (is (some #(= :equation (get-in % [:definition :kind]))
                (vals compiler-values)))
      (is (some #(= :storage-only (get-in % [:definition :kind]))
                (vals compiler-values))
          "a physical result destination is not a second semantic SSA version")
      (is (seq (:accesses memory)))
      (is (= [:unproven :unproven]
             ((juxt :reuse :completion) memory))))
    (is (= :invocation-link-certificate
           (:reason
            (reason-of
             #(invocation-link/verify!
               (assoc-in (:lowering functional)
                         [:plan :attributes :public-buffer-roles 'input]
                         :state))))))
    (is (every? #(zero? (get-in (compiled/certificate %)
                                [:driver-allocations]))
                [functional effect state]))))

(deftest compiled-equation-first-certifies-every-escaped-output
  (let [input (float-array 8)
        explicit (compiled/lower #'c-family-effect-map!
                                 [input (float-array 8) (long-array 8) 8]
                                 {:compiler :equation-first
                                  :target cuda-target :dtype :float
                                  :outputs '[left right]})
        donated (compiled/lower #'c-family-lane-owned-inout! [(float-array 8) 8]
                                {:compiler :equation-first
                                 :target cuda-target :dtype :float
                                 :donate '[output]})
        tapped (compiled/lower #'c-family-four-layers
                               [(double-array [1.0 0.25 -0.5 1.0])
                                (double-array [0.25 -0.125])
                                (double-array [1.0 -2.0])]
                               {:compiler :equation-first
                                :target cuda-target :dtype :double
                                :taps '[middle]})]
    (is (= [:output :output] (mapv :from (:out-tree explicit))))
    (is (= [:donated] (mapv :from (:out-tree donated))))
    (is (= [:result :tap] (mapv :from (:out-tree tapped))))
    (is (= 2 (count (distinct (map :node (:out-tree tapped)))))
        "the retained middle contraction is not the final semantic result")
    (doseq [prepared [explicit donated tapped]]
      (assert-certified-output-boundary prepared))
    (let [plan (compiled/plan tapped)]
      (is (= :invocation-link-output-boundary
             (:reason (reason-of #(invocation-link/certify plan (pop (:outputs plan))))))
          "recertification cannot silently drop an already retained output")
      (is (= :link-outputs
             (:reason (reason-of #(invocation-link/certify
                                   plan (conj (:outputs plan) ::unknown-output)))))
          "an escaped identity must name storage in the validated plan"))))

(deftest equation-first-final-boundary-has-one-link-proof
  (let [arguments [(float-array 8) 8]
        opts {:compiler :equation-first :target cuda-target :dtype :float}
        original link-plan/validate-with-effect-evidence!
        validate-call! program-call/validate!
        validate-retained-call! program-call/validate-with-retained-program!
        validate-boundary! emitted-equation/validate!
        lower-storage @#'invocation-link/lower-equation-storage
        checks (atom 0)
        call-checks (atom 0)
        boundary-checks (atom 0)
        prepared (with-redefs [link-plan/validate-with-effect-evidence!
                               (fn [& arguments]
                                 (swap! checks inc)
                                 (apply original arguments))
                               program-call/validate!
                               (fn [call]
                                 (swap! call-checks inc)
                                 (validate-call! call))
                               program-call/validate-with-retained-program!
                               (fn [call evidence]
                                 (swap! call-checks inc)
                                 (validate-retained-call! call evidence))
                               invocation-link/lower-equation-storage
                               (fn [& arguments]
                                 (with-redefs [emitted-equation/validate!
                                               (fn [boundary]
                                                 (swap! boundary-checks inc)
                                                 (validate-boundary! boundary))]
                                   (apply lower-storage arguments)))]
                   (compiled/lower #'c-family-elementwise arguments opts))
        plan (compiled/plan prepared)
        evidence (get-in prepared [:lowering :certificate :effect-evidence])]
    (is (= 1 @checks) "the projected public boundary is proved once, not twice")
    (is (= 1 @call-checks)
        "only the projected instance is checked; its pre-projection candidate does not escape")
    (is (= 0 @boundary-checks)
        "storage projection consumes the exact owner proof; final call validation remains fresh")
    (is (link-plan/retained-effect-evidence? plan evidence))
    (is (false? (link-plan/retained-effect-evidence?
                 (assoc plan :outputs []) evidence)))
    (is (= :link-outputs
           (:reason (reason-of
                     #(link-plan/make-with-final-projection
                       plan (fn [candidate]
                              {:plan (assoc candidate :outputs [::unknown])}))))))
    (is (= :emitted-parallel-program-target
           (:reason (reason-of
                     #(link-plan/make-with-final-projection
                       plan (fn [candidate]
                              {:plan (assoc-in candidate
                                               [:instances 0 :call :program :dialect]
                                               :hip-parallel)}))))))))

(deftest retention-requires-complete-call-binding-coverage
  (let [prepared (compiled/lower #'c-family-elementwise [(float-array 8) 8]
                                 {:compiler :equation-first :target cuda-target :dtype :float})
        original (compiled/plan prepared)
        bindings (get-in original [:attributes :compiler-buffer-bindings])]
    (is (seq bindings))
    (doseq [incomplete [{} (dissoc bindings (first (keys bindings)))]]
      (let [changed (assoc-in original [:attributes :compiler-buffer-bindings] incomplete)
            report (invocation-link/memory-witness (invocation-link/certify changed))]
        (is (= :typed-invocation (get-in changed [:attributes :source])))
        (is (= :unknown (get-in report [:value-versions :status])))
        (is (some #(and (= :incomplete-compiler-buffer-bindings (:reason %))
                        (seq (:missing %)))
                  (get-in report [:value-versions :unknown])))))))

(deftest trusted-equation-first-construction-derives-its-link-witness-once
  (compiled/clear-compilation-cache!)
  (try
    (let [derive-var (ns-resolve 'raster.compiler.ir.invocation-link 'derive-certificate)
          derive @derive-var
          calls (atom 0)]
      (with-redefs-fn
        {derive-var (fn [plan effect-evidence]
                      (swap! calls inc)
                      (derive plan effect-evidence))}
        (fn []
          (let [prepared (compiled/lower #'c-family-elementwise [(float-array 8) 8]
                                         {:compiler :equation-first
                                          :target cuda-target :dtype :float})
                lowering (:lowering prepared)]
            (is (= 1 @calls))
            (is (identical? lowering (invocation-link/verify! lowering)))
            (is (= 2 @calls))))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest equation-first-compiled-artifacts-compose-before-allocation
  (compiled/clear-compilation-cache!)
  (try
    (let [prepare #(compiled/lower #'c-family-elementwise [(float-array 8) 8]
                                   {:compiler :equation-first
                                    :target cuda-target :dtype :float})
          composite (compiled/compose
                     {:id :equation-first-pipeline
                      :components [{:id :first :program (prepare)}
                                   {:id :second :program (prepare)}]
                      :connections [{:from [:first :result]
                                     :to [:second :input]}]
                      :outputs [{:key :result :from [:second :result]}]})]
      (is (compiled/prepared? composite))
      (is (= [[:first :input]] (mapv :key (:in-tree composite))))
      (is (= [:result] (mapv :key (:out-tree composite))))
      (is (= 2 (count (:instances (compiled/plan composite)))))
      (is (= 0 (get-in (compiled/certificate composite) [:driver-allocations] 0)))
      (is (= {:hits 1 :misses 1 :misses-by-reason {:compulsory 1}
              :compilations 1 :failures 0
              :entries 1 :entries-by-compiler {:equation-first 1}}
             (dissoc (compiled/compilation-cache-stats) :compile-nanos)))
      (let [producer (compiled/lower #'c-family-effect-map!
                                     [(float-array 8) (float-array (repeat 8 -317.0)) (long-array 8) 8]
                                     {:compiler :equation-first :target cuda-target :dtype :float
                                      :outputs '[left]})
            consumer (prepare)
            source-node (get-in producer [:out-tree 0 :node])
            composite (compiled/compose
                       {:id :write-only-initializer-elimination
                        :components [{:id :producer :program producer} {:id :consumer :program consumer}]
                        :connections [{:from [:producer :left] :to [:consumer :input]}]
                        :outputs [{:key :result :from [:consumer :result]}]})
            mapped (get-in composite [:lowering :certificate :node-mapping [:producer source-node]])]
        (is (some? (get-in (compiled/plan producer) [:nodes source-node :source]))
            "the standalone producer has a captured caller output initializer")
        (is (contains? (get-in producer [:lowering :certificate :effect-evidence :initialization :complete-writes])
                       source-node))
        (is (not (contains? (get-in producer [:lowering :certificate :effect-evidence :initialization :reads])
                            source-node)))
        (is (nil? (get-in (compiled/plan composite) [:nodes mapped :source]))
            "certified write-only connected outputs need no upload even without device execution")))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest public-softmax-backward-keeps-the-reduction-resident
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile
                       #'nn/softmax-backward {:target target :dtype :double})
          linked (equation-first/lower
                  compilation [(double-array [1.0 2.0 3.0])
                               (double-array [0.2 0.3 0.5])])
          values (vals (get-in compilation [:semantic :values]))]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 1 (count (filter #(= :resident-scalar-buffer
                                 (get-in % [:representation :kind])) values))))
      (is (= 3 (count (:kernels compilation))))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation)))
      (is (= 0 (get-in linked [:attributes :driver-allocations])))
      (is (= 1 (count (:outputs linked)))))))

(deftest public-softmax-is-a-complete-typed-soac-program
  (doseq [target [cuda-target hip-target ocl-target]]
    (let [compilation (equation-first/compile #'nn/softmax {:target target :dtype :float})
          semantic (:semantic compilation)
          linked (equation-first/lower compilation [(float-array [1.0 2.0 3.0])])]
      (is (= :typed-parallel (:dialect semantic)))
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (seq (get-in semantic [:attributes :invocation-plan :steps])))
      (is (link-plan/link-plan? linked))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation)))
      (is (some #(and (str/includes? (:source %) "isnan(")
                      (str/includes? (:source %) "rstr_source_max_f32("))
                (:kernels compilation))
          "the max tree preserves Math/Raster NaN and signed-zero semantics"))))

(deftest public-loss-composition-keeps-device-results-resident
  (doseq [target [cuda-target hip-target]
          [operation arguments expected-kernels expected-outputs]
          [[#'nn/cross-entropy
            [(float-array [0.1 0.7 0.2]) (float-array [0.0 1.0 0.0])] 2 1]
           [#'nn/softmax-cross-entropy
            [(float-array [1.0 2.0 3.0]) (float-array [0.0 1.0 0.0])] 8 2]
           [#'nn/loss-fn
            [(float-array [0.1 0.2 0.3 0.4 0.5 0.6]) (float-array [0.0 0.0])
             (float-array [0.1 0.2 0.3 0.4]) (float-array [0.0 0.0])
             (float-array [1.0 2.0 3.0]) (float-array [0.0 1.0])]
            9 1]]]
    (let [compilation (equation-first/compile operation {:target target :dtype :float})
          linked (equation-first/lower compilation arguments)]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= expected-kernels (count (:kernels compilation))))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation)))
      (is (= expected-outputs (count (:outputs linked))))
      (is (= 0 (get-in linked [:attributes :driver-allocations]))))))

(deftest public-scalar-cast-helper-uses-the-common-gpu-frontend
  (doseq [target [ocl-target cuda-target hip-target]]
    (let [compilation (equation-first/compile
                       #'scalar-helpers/map-cast-helper {:target target :dtype :float})
          linked (equation-first/lower compilation [(float-array [0.25 -2.5 8.0]) 3])]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 1 (count (:kernels compilation))))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation)))
      (is (= 1 (count (:outputs linked))))
      (is (= 0 (get-in linked [:attributes :driver-allocations]))))))

(deftest descriptor-and-diagnostic-entries-expand-the-same-scalar-helper
  ;; The compatibility descriptor entry remains OpenCL/Level Zero-only. Public C-family
  ;; equation-first emission above is the CUDA/HIP source boundary; do not conflate them.
  (doseq [target [ocl-target]
          diagnostic? [false true]]
    (let [descriptor (pipeline/compile-gpu-program
                      #'scalar-helpers/map-cast-helper target
                      :dtype :float :compiler-report? diagnostic? :on-non-resident :throw)]
      (is (= 1 (count (:steps descriptor))))
      (is (= [:map-void] (mapv :convention (:steps descriptor))))
      (is (= diagnostic? (boolean (:compiler-report descriptor)))))))

(deftest staged-aot-and-diagnostics-share-gpu-helper-policy
  (let [stages (pipeline/show-pipeline #'scalar-helpers/map-cast-helper
                                      :dtype :float :target-device ocl-target)]
    (is (= :typed-soac (get-in stages [:soac-fused-stats :route])))
    (is (= 0 (get-in stages [:backend-applied-stats :fallback])))
    (is (= 1 (count (:kernels stages)))))
  ;; Force only construction, not the returned staged GPU invocation or a driver session.
  (with-redefs-fn {(requiring-resolve 'raster.gpu.ze-runtime/make-gpu-fn)
                  (fn [compile!] (compile!))}
    #(is (ifn? (pipeline/compile-aot #'scalar-helpers/map-cast-helper
                                    :dtype :float :target-device ocl-target)))))

(deftest scalar-helper-admission-preserves-public-softmax-reduction-boundaries
  (doseq [target [ocl-target cuda-target hip-target]]
    (let [compilation (equation-first/compile #'nn/softmax {:target target :dtype :float})]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 1 (count (:outputs (equation-first/lower
                                compilation [(float-array [1.0 2.0 3.0])])))))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation))))))

(deftest ordinary-and-diagnostic-runners-preserve-non-gpu-inline-policy
  (let [seen (atom [])
        diagnostic (var-get (ns-resolve 'raster.compiler.pipeline 'run-passes-diagnostic))]
    (with-redefs [pipeline/pass-specs
                  {:probe {:from :walked :to :walked
                           :fn (fn [form _]
                                 (swap! seen conj inline/*inline-scalar-bodies?*)
                                 {:form form})}}]
      (doseq [runner [pipeline/run-passes diagnostic]]
        (binding [inline/*inline-scalar-bodies?* false]
          (runner 'x [:probe] {})
          (runner 'x [:probe] {:target-device :cpu:0})
          (runner 'x [:probe] {:target-device ocl-target}))
        (binding [inline/*inline-scalar-bodies?* true]
          (runner 'x [:probe] {}))))
    (is (= [false false true true false false true true] @seen))))

(deftest public-array-clone-is-a-generated-identity-map
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile
                       #'nn/dense-backward-db {:target target :dtype :float})
          linked (equation-first/lower compilation [(float-array [1.0 2.0 3.0])])]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 1 (count (:kernels compilation))))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation)))
      (is (= 1 (count (:outputs linked))))
      (is (= 0 (get-in linked [:attributes :driver-allocations]))))))

(deftest link-plan-validates-an-exact-program-instance-once-per-check
  (let [compilation (equation-first/compile
                     #'nn/dense-backward-db {:target cuda-target :dtype :float})
        plan (equation-first/lower compilation [(float-array [1.0 2.0 3.0])])
        validate-call! program-call/validate!
        calls (atom 0)]
    (with-redefs [program-call/validate!
                  (fn [call]
                    (swap! calls inc)
                    (validate-call! call))]
      (is (link-plan/link-plan? (link-plan/validate! plan)))
      (is (= 1 @calls) "structure and effect derivation share one exact-object check")
      (reset! calls 0)
      (is (link-plan/link-plan? (link-plan/validate! plan)))
      (is (= 1 @calls) "a new public validation independently checks the program"))
    (let [forged (assoc-in plan [:instances 0 :call :program :dialect] :hip-parallel)
          reason (try (link-plan/validate! forged)
                      nil
                      (catch clojure.lang.ExceptionInfo exception
                        (:reason (ex-data exception))))]
      (is (= :emitted-parallel-program-target reason)
          "a changed embedded program cannot reuse a previous validation"))))

(deftest allocating-dense-weight-gradient-elides-its-dead-zero-fill
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile
                       #'nn/dense-backward-dW {:target target :dtype :float})
          linked (equation-first/lower
                  compilation [(float-array [1.0 2.0]) (float-array [3.0 4.0 5.0])])]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 1 (count (:kernels compilation))))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation)))
      (is (= 1 (count (:outputs linked))))
      (is (= 0 (get-in linked [:attributes :driver-allocations]))))))

(deftest in-place-convolution-gradients-project-blas-effects-to-contractions
  ;; These helpers return their caller-owned destination.  The BLAS call must remain an explicit
  ;; effect in the block so the common frontend can project NT/TN layouts to typed contractions;
  ;; returning the call directly used to leave both public deftms on the scalar route.
  (doseq [target [cuda-target hip-target ocl-target]
          operation [#'dl-nn/conv2d-backward-dW-into!
                     #'dl-nn/conv2d-backward-dcols-into!]]
    (let [compilation (equation-first/compile operation {:target target :dtype :float})]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 1 (count (:kernels compilation))))
      (is (get-in compilation [:kernels 0 :attributes :kernel-body]))
      (is (= 1 (get-in compilation [:stats :emission :typed-equations-emitted]))))))

(deftest public-huber-loss-shares-typed-conditional-reduction-lowering
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile
                       #'loss/huber-loss {:target target :dtype :float})]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 2 (count (:kernels compilation))))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation)))
      (is (some #(str/includes? (:source %) "if (") (:kernels compilation))
          "the mixed Float/Double value conditional is emitted from shared KernelBody control"))))

(deftest public-loss-gradients-use-portable-predicate-and-signum-ssa
  (doseq [target [cuda-target hip-target]
          operation [#'loss/huber-loss-backward #'loss/l1-loss-backward]]
    (let [compilation (equation-first/compile operation {:target target :dtype :float})]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 1 (count (:kernels compilation))))
      (is (get-in compilation [:kernels 0 :attributes :kernel-body])))))

(deftest public-dense-input-gradient-retains-shape-only-array-input
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile
                       #'nn/dense-backward-dx {:target target :dtype :double})
          linked (equation-first/lower
                  compilation [(double-array [2.0 -1.0])
                               (double-array [1.0 2.0 3.0 4.0 5.0 6.0])
                               (double-array 3)])]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (seq (:kernels compilation)))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation)))
      (is (= 0 (get-in linked [:attributes :driver-allocations])))
      (is (= 1 (count (:outputs linked)))))))

(deftest public-staged-contraction-preserves-independent-types-on-cuda-and-hip
  (doseq [[target module-target] [[cuda-target :cuda-c] [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile
                       #'ledger/staged-byte-float-contract! {:target target :dtype :float})
          linked (equation-first/lower compilation
                                       [(byte-array 8) (byte-array 16)
                                        (float-array 2) (float-array 4) (float-array 2)])
          kernels (:kernels compilation)]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 1 (count kernels)))
      (is (= module-target (:target (first kernels))))
      (is (every? #(get-in % [:attributes :kernel-body]) kernels))
      (is (str/includes? (:source (first kernels)) "rstr_dp4a"))
      (is (= 0 (get-in linked [:attributes :driver-allocations])))
      (is (empty? (:outputs linked)) "the public Void destination remains a state buffer")
      (let [call (:call (first (:instances linked)))
            equation (get-in call [:program :equations 0])
            algorithm (get-in equation [:operations 0 :algorithm])
            result (first (:results equation))
            physical (first (soac/physical-results algorithm (first (soac/equations algorithm))))
            make-call #(program-call/make (:program call)
                                          (assoc (:buffers call) result :unrelated-result)
                                          (:scalar-values call) {} nil %)
            checked-step program-call/validate-equation-call!
            checked-boundary emitted-equation/validate-with-result-contracts
            checks (atom 0)
            forged (with-redefs [emitted-equation/validate-with-result-contracts
                                 (fn [boundary]
                                   (swap! checks inc)
                                   (checked-boundary boundary))]
                     (make-call {result physical}))
            binds (atom 0)
            executor {:bind! (fn [& _] (swap! binds inc)) :run! identity :release! identity}
            reason (fn [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
        (is (= 1 @checks)
            "constructor shares the exact program-boundary projection with equation preparation")
        (with-redefs [emitted-equation/validate-with-result-contracts
                      (fn [boundary] (swap! checks inc) (checked-boundary boundary))]
          (program-call/validate! forged))
        (is (= 2 @checks) "later public validation independently rechecks the boundary")
        (reset! checks 0)
        (with-redefs [emitted-equation/validate-with-result-contracts
                      (fn [boundary] (swap! checks inc) (checked-boundary boundary))]
          (program-call/make (:program call) (:buffers call) (:scalar-values call) {} nil))
        (is (= 1 @checks) "empty result views do not cause extra schedule rederivation")
        (reset! checks 0)
        (with-redefs [program-call/validate-equation-call!
                      (fn [step] (swap! checks inc) (checked-step step))]
          (program-call/validate! forged))
        (is (= 1 @checks) "public call validation independently rechecks the step")
        (let [step (first (:steps forged))
              graph (:graph step)
              expected (get-in forged [:program :equations 0 :operations 0 :graph])
              copied (assoc graph :attributes (into {} (:attributes graph)))
              copied-step (assoc step :graph copied)
              equivalent? semantic-fingerprint/equivalent?
              equivalence-checks (atom 0)]
          (reset! checks 0)
          (with-redefs [emitted-equation/validate-with-result-contracts
                        (fn [boundary] (swap! checks inc) (checked-boundary boundary))]
            (program-call/validate-equation-call! step))
          (is (= 1 @checks) "public step validation independently rederives its boundary once")
          (is (= :emitted-program-result-view
                 (reason #(program-call/validate-equation-call!
                           (assoc step :result-views {result :wrong-destination})))))
          (is (= :scheduled-kernel-body-artifact-projection
                 (reason #(program-call/validate-equation-call!
                           (assoc-in step [:equation :operations 0 :graph :nodes 0
                                           :operation :target]
                                     (if (= module-target :cuda-c) :hip-cpp :cuda-c)))))
              "a modified boundary cannot reuse the constructor's proof")
          (is (identical? expected graph) "selected graph is the validated alternative")
          (is (not (identical? graph copied)))
          (with-redefs [semantic-fingerprint/equivalent?
                        (fn [& arguments]
                          (swap! equivalence-checks inc)
                          (apply equivalent? arguments))]
            (is (= step (program-call/validate-equation-call! step)))
            (is (zero? @equivalence-checks)
                "the already-validated graph needs no canonical comparison")
            (is (= copied-step (program-call/validate-equation-call! copied-step))
                "a reconstructed equivalent graph still takes the canonical fallback")
            (is (= 1 @equivalence-checks))))
        (is (= :emitted-program-result-views (reason #(make-call {:not-a-result physical}))))
        (is (= :parallel-program-result-view-resolver
               (reason #(program-runtime/prepare-with! forged executor))))
        (is (= :emitted-program-result-view-bindings
               (reason #(program-call/validate!
                         (assoc-in forged [:steps 0 :buffers physical] :wrong-kernel-buffer)))))
        (is (= :emitted-program-result-view-bindings
               (reason #(program-call/validate!
                         (assoc-in forged [:steps 0 :outputs result] :wrong-logical-buffer)))))
        (is (= :scheduled-kernel-body-artifact-projection
               (reason #(program-call/validate!
                         (assoc-in forged
                                   [:program :equations 0 :operations 0 :graph :nodes 0
                                    :operation :target]
                                   (if (= module-target :cuda-c) :hip-cpp :cuda-c)))))
            "public call validation still independently checks its embedded program")
        (is (= :parallel-program-result-view-shape
               (reason #(program-runtime/prepare-with!
                         forged (assoc executor :buffer-view
                                       (fn [token]
                                         (bview/view
                                          (bview/allocation {:id :shared :byte-size 8 :device target
                                                             :memory-space :device :ownership :owned})
                                          {:dtype :float :shape [(if (= token :unrelated-result) 1 2)]})))))))
        (is (= :parallel-program-result-view
               (reason #(program-runtime/prepare-with!
                         forged (assoc executor :buffer-view
                                       (fn [token]
                                         (bview/view
                                          (bview/allocation {:id token :byte-size 8 :device target
                                                             :memory-space :device :ownership :owned})
                                          {:dtype :float :shape [2]})))))))
        (is (zero? @binds) "unrelated result views must fail before staging any graph"))
      (doseq [[slot short-buffer] [[0 (byte-array 7)] [2 (float-array 1)]
                                  [4 (float-array 1)]]]
        (is (thrown? clojure.lang.ExceptionInfo
                     (equation-first/lower
                      compilation
                      (assoc [(byte-array 8) (byte-array 16) (float-array 2)
                              (float-array 4) (float-array 2)] slot short-buffer)))
            "core, scale and destination capacities are checked before allocation"))
      (is (= (:emitted compilation) (emitted-program/validate! (:emitted compilation)))))))

(deftest staged-result-views-initialize-only-the-written-prefix
  (let [output (float-array 4)
        compilation (equation-first/compile
                     #'ledger/staged-byte-float-contract!
                     {:target cuda-target :dtype :float
                      :values {'out (av/tensor {:dtype :float :shape [4]})}})
        linked (equation-first/lower compilation [(byte-array 8) (byte-array 16)
                                                  (float-array 2) (float-array 4) output])
        call (:call (first (:instances linked)))
        result (first (get-in call [:steps 0 :equation :results]))
        prefix (get (:buffers call) result)
        base (get (:buffers call) 'out)
        fresh (-> linked
                  (assoc-in [:nodes base :source] nil)
                  (assoc-in [:nodes base :role] :internal)
                  (assoc :outputs [prefix]))
        reason (fn [plan] (try (link-plan/validate! plan) nil
                              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (is (not= base prefix))
    (is (= (get-in linked [:nodes base :view :allocation])
           (get-in linked [:nodes prefix :view :allocation]))
        "result views share storage without another allocation")
    (is (nil? (get-in linked [:nodes prefix :source])))
    (is (nil? (reason fresh)) "the contraction produces its entire logical prefix")
    (let [contract (link-plan/initialization-contract fresh)]
      (is (contains? (:produces contract) prefix))
      (is (not (contains? (:produces contract) base)))
      (is (not (contains? (:requires contract) base))
          "a write-only prefix does not demand an initialized backing tail")
      (is (contains? (:writes contract) base)
          "the ABI pointer is written, but only the certified prefix is initialized"))
    (is (= :link-unproduced-output (reason (assoc fresh :outputs [base])))
        "writing a prefix cannot initialize or export the untouched tail")
    (is (= :program-link-result-view
           (reason (assoc-in fresh [:nodes prefix :view :byte-offset] 4)))
        "a claimed prefix may not be shifted into another part of the allocation")))

(deftest public-elementwise-map-uses-portable-kernel-body
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile
                       #'c-family-elementwise {:target target :dtype :float})
          kernel (first (:kernels compilation))]
      (is (= [module-target] (mapv :target (:kernels compilation))))
      (is (= :portable-segmap
             (get-in kernel [:attributes :kernel-body :attributes :kind])))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-scan-uses-one-portable-kernel-body-graph
  (doseq [[target program-dialect module-target]
          [[cuda-target :cuda-parallel :cuda-c]
           [hip-target :hip-parallel :hip-cpp]]]
    (let [compilation (equation-first/compile
                       #'c-family-scan {:target target :dtype :float})
          linked (equation-first/lower
                  compilation [(float-array [1 2 3 4]) 4])
          kernels (:kernels compilation)]
      (is (= program-dialect (get-in compilation [:emitted :dialect])))
      (is (= [:intra-block :block-scan :carry-in]
             (mapv #(get-in % [:attributes :phase]) kernels)))
      (is (every? #(= module-target (:target %)) kernels))
      (is (every? #(= :portable-segscan
                      (get-in % [:attributes :kernel-body :attributes :kind]))
                  kernels))
      (is (not-any? #(re-find #"__kernel|get_global_id|get_local_id" (:source %)) kernels))
      (is (= 0 (get-in linked [:attributes :driver-allocations])))
      (is (= 1 (count (:outputs linked))))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-effect-map-preserves-typed-multi-output-storage
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile
                       #'c-family-effect-map! {:target target :dtype :float})
          linked (equation-first/lower
                  compilation [(float-array 8) (float-array 8) (long-array 8) 8])
          kernel (first (:kernels compilation))]
      (is (= module-target (:target kernel)))
      (is (= #{'left 'right}
             (into #{}
                   (comp (filter #(= :output (:kind %))) (map :id))
                   (get-in kernel [:attributes :kernel-body :parameters]))))
      (is (empty? (:outputs linked)) "Void host semantics remain effect-only")
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-integer-case-map-lowers-through-portable-kernel-control
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile
                       #'c-family-case-map! {:target target :dtype :float})
          kernel (first (:kernels compilation))
          operations (nested-operations
                      (get-in kernel [:attributes :kernel-body :operations]))]
      (is (= module-target (:target kernel)))
      (is (= :portable-segmap
             (get-in kernel [:attributes :kernel-body :attributes :kind])))
      (is (some #(= "IfRegion" (some-> % class .getSimpleName)) operations))
      (is (not (str/includes? (:source kernel) "case*")))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-reassociated-rmsnorm-is-one-cooperative-c-family-artifact
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile
                       #'dl-nn/rms-norm-reassociated! {:target target :dtype :float})
          linked (equation-first/lower
                  compilation
                  [(float-array 640) (float-array 640) (float-array 640)
                   1 640 1.0e-6 1.0])
          kernel (first (:kernels compilation))]
      (is (= 1 (count (:kernels compilation))))
      (is (= module-target (:target kernel)))
      (is (= :cooperative-segmented-fold-map
             (get-in kernel [:attributes :kernel-body :attributes :kind])))
      (is (= 0 (get-in linked [:attributes :driver-allocations])))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest portable-inout-map-refuses-a-cross-lane-read
  (let [reason (reason-of #(equation-first/compile
                            #'c-family-shifted-inout!
                            {:target cuda-target :dtype :float}))]
    (is (= :kernel-graph-target-lowering-missing (:reason reason)))
    (is (= :inout-storage
           (get-in reason [:kernel-body-decline :missing-rule])))))

(deftest public-gqa-composition-emits-only-portable-c-family-kernels
  (doseq [[target program-dialect module-target]
          [[cuda-target :cuda-parallel :cuda-c]
           [hip-target :hip-parallel :hip-cpp]]]
    (let [compilation (equation-first/compile
                       #'attention/gqa-causal-mha {:target target :dtype :float})
          linked (equation-first/lower
                  compilation
                  [(float-array [1 0 0 1 1 1 1 -1])
                   (float-array [1 0 0 1])
                   (float-array [1 2 3 4])
                   1 2 2 1 2])
          kernels (:kernels compilation)
          loops (for [kernel kernels
                      operation (nested-operations
                                 (get-in kernel [:attributes :kernel-body :operations]))
                      :when (= "ForLoop" (some-> operation class .getSimpleName))]
                  operation)]
      (is (= program-dialect (get-in compilation [:emitted :dialect])))
      (is (= 7 (count kernels)))
      (is (every? #(= module-target (:target %)) kernels))
      (is (every? #(get-in % [:attributes :kernel-body]) kernels))
      (is (not-any? #(re-find #"__kernel|get_global_id|get_local_id" (:source %)) kernels))
      (is (seq loops))
      (is (every? #(= :ordered (get-in % [:attributes :association]))
                  (remove #(= :segment-grid-stride (get-in % [:attributes :role])) loops)))
      (is (every? #(= :independent (get-in % [:attributes :association]))
                  (filter #(= :segment-grid-stride (get-in % [:attributes :role])) loops)))
      (is (= 0 (get-in linked [:attributes :driver-allocations])))
      (is (= 1 (count (:outputs linked))))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-gqa-jvp-is-the-same-generated-c-family-composition
  (doseq [[target module-target] [[cuda-target :cuda-c]
                                  [hip-target :hip-cpp]
                                  [ocl-target :opencl-c]]]
    (let [compilation (equation-first/compile
                       #'attention/gqa-causal-mha-jvp {:target target :dtype :float})
          kernels (:kernels compilation)
          certified (filter #(get-in % [:attributes :kernel-body :attributes
                                        :address-projection]) kernels)]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 11 (count kernels)))
      (is (every? #(= module-target (:target %)) kernels))
      (is (every? #(get-in % [:attributes :kernel-body]) kernels))
      (is (seq certified))
      (is (not-any?
           (fn [kernel]
             (some #(and (= "ScalarCompute" (some-> % class .getSimpleName))
                         (= :trap (get-in % [:expression :options :overflow])))
                   (nested-operations
                    (get-in kernel [:attributes :kernel-body :operations]))))
           certified)
          "graph-certified addresses lower as IndexExpr, not trapping scalar arithmetic")
      (is (= {:kernel-body 11}
             (get-in compilation [:stats :emission :emission-routes]))))))

(deftest public-stencil-uses-portable-kernel-body
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile
                       #'c-family-stencil {:target target :dtype :float})
          kernel (first (:kernels compilation))]
      (is (= [module-target] (mapv :target (:kernels compilation))))
      (is (= :portable-segstencil
             (get-in kernel [:attributes :kernel-body :attributes :kind])))
      (is (not (re-find #"__kernel|get_global_id|get_local_id" (:source kernel))))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-segmented-fold-map-uses-the-same-c-family-boundary
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile
                       #'c-family-segment-sum! {:target target :dtype :float})
          linked (equation-first/lower
                  compilation [(float-array 8) 2 4])]
      (is (= [module-target] (mapv :target (:kernels compilation))))
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 0 (get-in linked [:attributes :driver-allocations]))))))

(deftest public-broadcast-gradient-is-one-portable-segmented-reduction
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile
                       #'array-ops/broadcast-add-dt {:target target :dtype :double})]
      (is (= [module-target] (mapv :target (:kernels compilation))))
      (is (= 1 (count (:kernels compilation))))
      (is (= :kernel-body
             (get-in compilation [:kernels 0 :attributes :emission-route])))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-row-dots-are-portable-segmented-reductions
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]
          kernel [#'array-ops/dot-rows #'array-ops/dot-rows-dW]]
    (let [compilation (equation-first/compile kernel {:target target :dtype :double})]
      (is (= [module-target] (mapv :target (:kernels compilation))))
      (is (= 1 (count (:kernels compilation))))
      (is (= :kernel-body
             (get-in compilation [:kernels 0 :attributes :emission-route])))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-axis-reduction-is-a-portable-segmented-reduction
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile #'array-ops/reduce-axis
                                              {:target target :dtype :double})]
      (is (= [module-target] (mapv :target (:kernels compilation))))
      (is (= 1 (count (:kernels compilation))))
      (is (= :kernel-body
             (get-in compilation [:kernels 0 :attributes :emission-route])))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-indexed-dot-is-a-portable-segmented-reduction
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile #'array-ops/indexed-dot
                                              {:target target :dtype :double})]
      (is (= [module-target] (mapv :target (:kernels compilation))))
      (is (= 1 (count (:kernels compilation))))
      (is (= :kernel-body
             (get-in compilation [:kernels 0 :attributes :emission-route])))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-segment-div-z-adjoint-is-a-portable-segmented-reduction
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile #'array-ops/segment-div-dZ
                                              {:target target :dtype :double})]
      (is (= [module-target] (mapv :target (:kernels compilation))))
      (is (= 1 (count (:kernels compilation))))
      (is (= :kernel-body
             (get-in compilation [:kernels 0 :attributes :emission-route])))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-output-owned-adjoints-are-portable-segmented-reductions
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]
          kernel [#'array-ops/scatter-mul-add-d-coeffs
                  #'array-ops/flat-embed-d-values
                  #'array-ops/flat-embed-dWe
                  #'array-ops/flat-embed-dbe]]
    (let [compilation (equation-first/compile kernel {:target target :dtype :double})]
      (is (= [module-target] (mapv :target (:kernels compilation))))
      (is (= 1 (count (:kernels compilation))))
      (is (= :kernel-body
             (get-in compilation [:kernels 0 :attributes :emission-route])))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-diffusion-cumulative-product-is-a-portable-scan
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile #'diffusion/compute-alphas-cumprod
                                              {:target target :dtype :float :return-tag 'void})]
      (is (= 1 (count (get-in compilation [:semantic :outputs])))
          "a caller option cannot replace the physical deftm's non-Void return contract")
      (is (= 3 (count (:kernels compilation))))
      (is (every? #(= module-target (:target %)) (:kernels compilation)))
      (is (every? #(= :kernel-body (get-in % [:attributes :emission-route]))
                  (:kernels compilation)))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-embedding-table-adjoints-use-portable-additive-scatters
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]
          kernel [#'array-ops/flat-embed-d-space-emb
                  #'array-ops/flat-embed-d-state-emb]]
    (let [compilation (equation-first/compile kernel {:target target :dtype :float})]
      (is (seq (:kernels compilation)))
      (is (every? #(= module-target (:target %)) (:kernels compilation)))
      (is (every? #(= :kernel-body (get-in % [:attributes :emission-route]))
                  (:kernels compilation)))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest product-reduction-composes-with-an-ordered-epilogue
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile
                       #'c-family-broadcast-product-map!
                       {:target target :dtype :int
                        :values {'input (av/tensor {:dtype :int :shape [64]})
                                 'weights (av/tensor {:dtype :int :shape [64]})
                                 'output (av/tensor {:dtype :int :shape [1]})}})
          linked (equation-first/lower
                  compilation [(int-array 64) (int-array 64) (int-array 1) 1])]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 2 (count (:kernels compilation))))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation)))
      (is (= [8] (get-in compilation
                         [:kernels 0 :attributes :kernel-body :launch :workgroup-size])))
      (is (= 0 (get-in linked [:attributes :driver-allocations]))))))

(deftest counted-softmax-initializers-use-the-public-c-family-boundary
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile #'attention/softmax-rows!
                                            {:target target :dtype :float})
          plan (equation-first/lower compilation [(float-array 6) 2 3])]
      (is (= 4 (count (:kernels compilation))))
      (is (every? #(< (count (:source %)) 32768) (:kernels compilation))
          "the polynomial's shared scalar spine must not expand into megabytes of source")
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 0 (get-in plan [:attributes :driver-allocations]))))))

(deftest counted-mixed-precision-stores-use-explicit-destination-conversion
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile #'array-ops/reduce-axis-backward
                                            {:target target :dtype :float})]
      (is (seq (:kernels compilation)))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest heat-2d-counted-stores-use-the-public-c-family-boundary
  (doseq [[target module-target] [[cuda-target :cuda-c] [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile #'pde/heat-rhs-2d!
                                            {:target target :dtype :double})
          plan (equation-first/lower compilation
                                     [(double-array 35) (double-array 35)
                                      5 7 0.25 4.0 9.0])]
      (is (seq (:kernels compilation)))
      (is (every? #(= module-target (:target %)) (:kernels compilation)))
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 0 (get-in plan [:attributes :driver-allocations]))))))

(deftest coarse-fine-operators-use-the-public-c-family-boundary
  (doseq [[target module-target] [[cuda-target :cuda-c] [hip-target :hip-cpp]]
          operator [#'multilevel/prolong-constant-2d! #'multilevel/restrict-average-2d!]]
    (let [prolong? (= operator #'multilevel/prolong-constant-2d!)
          compilation (equation-first/compile operator {:target target :dtype :double})
          plan (equation-first/lower compilation
                                     [(double-array (if prolong? 60 15))
                                      (double-array (if prolong? 15 60)) 3 5])]
      (is (seq (:kernels compilation)))
      (is (every? #(= module-target (:target %)) (:kernels compilation)))
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 0 (get-in plan [:attributes :driver-allocations]))))))

(deftest declared-map-result-cast-preserves-jvm-source-semantics
  (let [output (int-array 2)]
    (checked-casts/declared-narrow-rows!
     (long-array [Integer/MIN_VALUE Integer/MAX_VALUE]) output 2)
    (is (= [Integer/MIN_VALUE Integer/MAX_VALUE] (vec output))))
  (doseq [value [(inc (long Integer/MAX_VALUE)) (dec (long Integer/MIN_VALUE))]]
    (let [input (long-array [value]) output (int-array [-7])]
      (is (thrown? ArithmeticException
                   (raster.par/map! output index 1 int (aget input index))))
      (is (= [-7] (vec output)))
      (is (thrown? ArithmeticException
                   (checked-casts/declared-narrow-rows! input output 1)))
      (is (= [-7] (vec output))))))

(deftest checked-source-narrowing-reaches-public-c-family-kernels
  (doseq [target [cuda-target hip-target]
          operation [#'checked-casts/narrow-rows!
                     #'checked-casts/declared-narrow-rows!
                     #'checked-casts/narrow-stores!
                     #'checked-casts/unused-narrow-rows!
                     #'checked-casts/annihilated-narrow-rows!]]
    (let [compilation (equation-first/compile operation
                                            {:target target :dtype :long})
          plan (equation-first/lower compilation [(long-array [-2147483648 2147483647])
                                                 (int-array 2) 2])
          nodes (mapcat #(tree-seq coll? seq (get-in % [:attributes :kernel-body :operations]))
                        (:kernels compilation))
          input-loads (set (keep #(when (and (map? %) (= 'input (:buffer %))
                                             (= :long (get-in % [:result :type])))
                                    (get-in % [:result :id])) nodes))]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (some #(str/includes? (:source %) "rstr_trap_cast_i64_i32")
                (:kernels compilation)))
      (is (some #(and (map? %) (= :cast (:op %)) (= :int (:result-type %))
                       (= {:rounding :exact :overflow :trap} (:options %))
                       (contains? input-loads (first (:arguments %)))) nodes)
          "the input load, not merely a launch-bound scalar, must feed a checked cast")
      (is (= 0 (get-in plan [:attributes :driver-allocations]))))))

(deftest unused-checked-prefix-remains-observable-before-device-work
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile #'checked-casts/checked-prefix-rows!
                                            {:target target :dtype :long})
          input (long-array [7 9])
          output (int-array [-1 -1])]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 0 (get-in (equation-first/lower compilation [input output 2 2147483647])
                       [:attributes :driver-allocations])))
      (is (thrown? ArithmeticException
                   (equation-first/lower compilation [input output 2 2147483648])))
      (is (= [-1 -1] (vec output))))))

(deftest checked-scalars-after-device-work-do-not-become-preparation-checks
  (doseq [target [cuda-target hip-target]]
    (is (= :equation-first-coverage
           (try
             (equation-first/compile #'checked-casts/checked-after-write!
                                    {:target target :dtype :int})
             :accepted
             (catch clojure.lang.ExceptionInfo exception
               (:reason (ex-data exception))))))))

(deftest padded-map-lanes-do-not-evaluate-checked-conversions
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile #'checked-casts/narrow-index!
                                            {:target target :dtype :byte})
          kernel (first (:kernels compilation))
          operations (get-in kernel [:attributes :kernel-body :operations])
          guarded (first operations)
          nodes (tree-seq coll? seq (:operations guarded))]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 1 (count operations)))
      (is (= :map-active (:mask guarded))
          "the complete per-element region is inactive for padded work-items")
      (is (some #(and (map? %) (= :cast (:op %)) (= :byte (:result-type %))
                       (= :trap (get-in % [:options :overflow]))) nodes))
      (is (= 0 (get-in (equation-first/lower compilation [(byte-array 1) 1])
                       [:attributes :driver-allocations]))))))

(deftest emitted-program-rejects-a-mixed-target-module
  (let [compilation (equation-first/compile
                     #'c-family-dot {:target cuda-target :dtype :float})
        mixed (assoc-in (:emitted compilation)
                        [:equations 0 :operations 0 :graph :nodes 0 :operation :target]
                        :hip-cpp)
        failure (reason-of #(emitted-program/validate! mixed))]
    (is (= :kernel-graph-executable-targets (:reason failure))
        "the executable graph rejects a mixed target before its enclosing program must")
    (is (= #{:cuda-c :hip-cpp} (:targets failure)))))
