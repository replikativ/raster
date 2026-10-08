(ns raster.compiler.ir.emitted-equation-dispatch-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-artifact :as artifact]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.fixtures.contractions :as contractions]
            [raster.compiler.fixtures.attention-projection :as attention-projection]
            [raster.compiler.backend.gpu.parallel-program-c-family :as program-target]
            [raster.compiler.backend.gpu.kernel-body-target :as body-target]
            [raster.compiler.ir.emitted-equation-dispatch :as equation-dispatch]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.semantic-fingerprint :as semantic-fingerprint]
            [raster.gpu.indexed-attention-device-test :as indexed-fixture]
            [raster.gpu.dispatch-benchmark :as benchmark]
            [raster.gpu.dispatch-tuning :as tuning]
            [raster.gpu.measurement :as measurement]
            [raster.runtime.hardware :as hardware]))

(def ^:private target :ocl:certified-equation-dispatch-test)

(def ^:private artifact-identity
  {:semantic-request-fingerprint "equation-dispatch-request"
   :compiler-build-fingerprint "equation-dispatch-build"
   :source-dependency-fingerprint "equation-dispatch-source"
   :target-descriptor-fingerprint "equation-dispatch-target"})

(defn- register-target!
  []
  (hardware/register-target-device!
   target
   {:type :ocl :name "Synthetic Intel certified equation dispatch"
    :vendor "Intel"
    :capabilities {:warp-size 16 :subgroup-sizes [16]
                   :max-workgroup-size 256 :shared-local-memory 65536 :total-eus 32}}))

(defn- compilation
  [strategy]
  (equation-first/compile
   #'indexed-fixture/resident-indexed-attention-probe
   {:target target :dtype :float
    :schedule {:segmented-weighted-reduction {:strategy strategy}}}))

(defn- reason
  [thunk]
  (try (thunk) nil
       (catch clojure.lang.ExceptionInfo exception
         (:reason (ex-data exception)))))

(deftest approximate-modes-require-paired-reconstructed-models
  (let [check (ns-resolve 'raster.compiler.ir.emitted-equation-dispatch
                         'validate-model-pair!)
        model {:kind :mixed-matrix-operational-model :version 1
               :mode :approximate-model}]
    (doseq [mode [:exact :reassociated]]
      (is (nil? (check mode {})))
      (is (= :equation-dispatch-numerics
             (reason #(check mode {:numerical-model model})))))
    (is (nil? (check :approximate-model {:numerical-model model})))
    (doseq [report [{} {:numerical-model (assoc model :mode :exact)}
                       {:numerical-model (assoc model :version 2)}]]
      (is (= :equation-dispatch-numerics
             (reason #(check :approximate-model report)))))))

(defn- contraction-candidates
  [reference-compilation]
  ;; Both emissions refine one retained semantic spine. Re-running the frontend produces fresh
  ;; SSA identities and cannot establish this correspondence merely by sharing a public ABI.
  (let [register-emission
        (program-target/emit-program
         (:scheduled reference-compilation)
         (assoc-in (:options reference-compilation)
                   [:schedule :typed-contraction :strategy] :register-tiled))]
    (mapv (fn [program]
            (let [candidate (-> program :equations last :operations first)]
              (assoc-in candidate [:graph :attributes :strategy]
                        (get-in candidate [:graph :nodes 0 :operation :attributes :strategy]))))
          [(:emitted reference-compilation) (:program register-emission)])))

(defn- contraction-selection [alternatives]
  (dispatch/make
   {:id "certified-fp32-contraction"
    :alternatives (mapv :graph alternatives)
    :default-strategy :sequential-segments
    :selector {:kind :fixed-strategy :strategy :register-tiled}}))

(defn- check-candidate-proof-reuse! [candidate certified write-domains]
  (let [expected-graph-var (ns-resolve 'raster.compiler.ir.emitted-parallel-equation
                                     'expected-graph)
        original @expected-graph-var
        reconstructions (atom 0)
        expected-domain (write-domains candidate)]
    (let [report (emitted-equation/validate-with-result-contracts candidate)]
      (is (identical? candidate (:boundary report)))
      (is (= expected-domain (:complete-write-domains report)))
      (is (= (emitted-equation/physical-results candidate) (:physical-results report)))
      (is (not (contains? report :source-graph)) "the temporary reconstructed graph does not escape"))
    (with-redefs-fn
      {expected-graph-var (fn [algorithm body]
                            (when (and (identical? algorithm (:algorithm candidate))
                                       (identical? body (:body candidate)))
                              (swap! reconstructions inc))
                            (original algorithm body))}
      (fn []
        (is (= expected-domain (write-domains candidate)))
        (is (= 1 @reconstructions)
            "complete-write projection reuses the source graph from its boundary proof")
        (is (= expected-domain (write-domains candidate)))
        (is (= 2 @reconstructions) "a later public query reconstructs its own source")
        (reset! reconstructions 0)
        (is (identical? certified (equation-dispatch/validate! certified)))
        (is (= 1 @reconstructions)
            "dispatch derives storage and full-write facts from one candidate proof")
        (is (identical? certified (equation-dispatch/validate! certified)))
        (is (= 2 @reconstructions) "a later public dispatch validation remains independent")))
    (is (thrown? clojure.lang.ExceptionInfo
                 (write-domains (assoc-in candidate [:graph :nodes 0 :operation :source] nil)))
        "a previous successful query cannot hide a malformed candidate")))

(deftest generated-contraction-candidates-share-a-certified-dispatch-boundary
  (register-target!)
  (let [reference (equation-first/compile #'contractions/fixed-matmul
                                         {:target target :dtype :float})
        alternatives (contraction-candidates reference)
        selection (contraction-selection alternatives)
        policy {:permitted-modes #{:exact :reassociated}}
        certified (equation-dispatch/make alternatives selection policy)
        program (update (:emitted reference) :equations
                        #(update % (dec (count %)) assoc :operations [certified]))
        linked (equation-first/lower (assoc reference :emitted program)
                                    [(float-array 15) (float-array 21)])]
    (check-candidate-proof-reuse! (first alternatives) certified
                                 emitted-equation/contraction-write-domains)
    (is (= program (emitted-program/validate! program)))
    (let [reversed (equation-dispatch/make (vec (reverse alternatives))
                                         (contraction-selection (vec (reverse alternatives)))
                                         policy)
          reversed-program (update program :equations
                                   #(update % (dec (count %)) assoc :operations [reversed]))
          evidence (emitted-program/validate-with-physical-results! reversed-program)
          projection (emitted-program/operation-projection (:projections evidence) reversed)]
      (is (identical? (first alternatives) (:boundary projection))
          "the certified exact fallback need not be the first alternative")
      (is (= (emitted-equation/physical-results (first alternatives))
             (:physical-results projection)))
      (is (= (vec (reverse alternatives)) (:candidates projection)))
      (is (= (equation-dispatch/complete-write-domains reversed)
             (:complete-write-domains projection))
          "retained extents are jointly proved symbolic domains, not runtime completeness")
      (let [plain (first alternatives)
            projection (emitted-equation/validate-with-physical-results plain)
            input (:id (first (get-in plain [:graph :inputs])))
            prove (fn [input-storage]
                    (#'invocation-link/complete-write?
                     plain 'output 35 {} {'output :destination input input-storage} {} projection))]
        (is (seq (:complete-write-domains projection)))
        (is (true? (boolean (prove :independent))))
        (is (false? (boolean (prove :destination)))
            "a retained contraction extent must not bypass ordinary SOAC's fresh alias check"))
      (doseq [changed [(assoc reversed :numerical-policy {:permitted-modes #{:exact}})
                       (assoc-in reversed [:dispatch :default-strategy] :register-tiled)
                       (assoc reversed :alternatives [(second alternatives)])]]
        (let [changed-program (update reversed-program :equations
                                      #(update % (dec (count %)) assoc :operations [changed]))]
          (is (= :emitted-parallel-program-retained-validation
                 (reason #(emitted-program/checked-retained-validation!
                           changed-program evidence))))))
      (is (= :emitted-parallel-program-operation-projection
             (reason #(emitted-program/operation-projection
                       (:projections evidence) (with-meta reversed {:copy true})))))
      (let [calls (atom 0)
            candidate-checks (atom 0)
            original equation-dispatch/validate-with-boundary
            original-candidate emitted-equation/validate-with-result-contracts
            call (-> linked :instances first :call)
            baseline-effects (:step-facts (:effect-evidence
                                          (link-plan/validate-with-effect-evidence! linked)))
            host-results (into {} (keep #(when (program-call/evaluated-host-equation? %)
                                          [(:id (:equation %)) (:results %)])) (:steps call))]
        (with-redefs [equation-dispatch/validate-with-boundary
                      (fn [operation] (swap! calls inc) (original operation))
                      emitted-equation/validate-with-result-contracts
                      (fn [candidate]
                        (swap! candidate-checks inc)
                        (original-candidate candidate))]
          (let [evidence (emitted-program/validate-with-physical-results! program)
                boundaries (#'invocation-link/equation-boundaries program (:projections evidence))
                retained (program-call/make program (:buffers call) (:scalar-values call) {}
                                            (fn [equation _] (get host-results (:id equation)))
                                            {} evidence)]
            (is (= 1 @calls) "one program proof supplies all synchronous dispatch projections")
            (is (= 2 @candidate-checks) "each candidate is independently reconstructed once")
            (is (some #(identical? certified (:operation %)) boundaries))
            (is (= (:graph (last (:steps call))) (:graph (last (:steps retained)))))
            (program-call/validate-with-retained-program! retained evidence)
            (is (= 1 @calls) "retained concrete validation does not reconstruct static candidates")
            (is (= 2 @candidate-checks) "candidate-level reconstruction is not hidden elsewhere")
            (let [projection (emitted-program/operation-projection (:projections evidence) certified)]
              (doseq [[capacity expected] [[34 false] [35 true] [36 false]]]
                (is (= expected
                       (#'invocation-link/complete-write? certified 'output capacity {} {} {}
                                                        projection))
                    "complete-write admission resolves current capacity, not a retained boolean")))
            (let [effects (link-plan/validate-with-effect-evidence! linked evidence)]
              (is (= baseline-effects (:step-facts (:effect-evidence effects)))
                  "independent and retained paths derive identical fresh effect facts"))
            (is (= 2 @candidate-checks)
                "retained invocation coverage and final LinkPlan proofs do not reconstruct candidates")
            (program-call/validate! retained)
            (is (= 2 @calls) "later public validation independently reconstructs every candidate")
            (is (= 4 @candidate-checks) "later public proof independently reconstructs both candidates")
            (let [copied (assoc-in retained
                                   [:steps (dec (count (:steps retained))) :equation :operations 0]
                                   (with-meta certified {:copy true}))]
              (is (program-call/emitted-equation-call?
                   (program-call/validate-equation-call! (last (:steps copied))))
                  "independent equation validation reconstructs an equal copied operation")
              (is (= :emitted-parallel-program-operation-projection
                     (reason #(program-call/validate! copied)))
                  "whole-program validation requires the step's exact owning operation too")
              (is (= :emitted-parallel-program-operation-projection
                     (reason #(program-call/validate-with-retained-program! copied evidence)))
                  "an equal copied step remains independently valid, but inherits no owner's proof"))))))
    (is (= :register-tiled
           (executable/strategy (-> linked :instances first :call :steps last :graph))))
    (let [outputs (filter #(= :output (:role %)) (vals (:nodes linked)))]
      (is (= 1 (count outputs)))
      (is (nil? (:source (first outputs)))
          "fresh zero storage may become source-free only after all candidates prove overwrite"))
    (is (= :sequential-segments
           (get-in (equation-dispatch/default-equation certified)
                   [:graph :attributes :strategy])))
    (is (= {'output 35}
           (update-vals (equation-dispatch/complete-write-domains certified)
                        #(launch/resolve-expression (constantly nil) %))))
    (is (= :equation-dispatch-numerics
           (reason #(equation-dispatch/make alternatives selection
                                            {:permitted-modes #{:exact}}))))
    (is (= :equation-dispatch-default-numerics
           (reason #(equation-dispatch/make
                     alternatives (assoc selection :default-strategy :register-tiled) policy))))
    (let [candidate (second alternatives)
          artifact (get-in candidate [:graph :nodes 0 :operation])
          scheduled (assoc-in (get-in artifact [:provenance :scheduled-operation])
                              [:numerics :mode] :exact)
          replacement (body-target/emit-artifact
                       (:kernel-name artifact) scheduled
                       (get-in artifact [:provenance :target-dialect])
                       {:attributes (:attributes artifact)})
          forged (assoc-in candidate [:graph :nodes 0 :operation] replacement)
          forged-alternatives [(first alternatives) forged]]
      (is (= forged (emitted-equation/validate! forged)))
      (is (= :equation-dispatch-complete-write
             (reason #(equation-dispatch/make
                       forged-alternatives (contraction-selection forged-alternatives)
                       policy)))
          "a forged exact label fails certificate rederivation before numerical permission"))
    ;; Re-emit a valid artifact with a truncated store body: projection and ABI remain valid,
    ;; but neither target emission nor a shared write label proves complete initialization.
    (let [candidate (first alternatives)
          artifact (get-in candidate [:graph :nodes 0 :operation])
          scheduled (get-in artifact [:provenance :scheduled-operation])
          truncated (update-in scheduled [:body :operations] pop)
          replacement (body-target/emit-artifact
                       (:kernel-name artifact) truncated
                       (get-in artifact [:provenance :target-dialect])
                       {:attributes (:attributes artifact)})
          partial (assoc-in candidate [:graph :nodes 0 :operation] replacement)
          partial-alternatives [partial (second alternatives)]]
      (is (= partial (emitted-equation/validate! partial)))
      (is (nil? (emitted-equation/contraction-write-domains partial)))
      (is (= :equation-dispatch-complete-write
             (reason #(equation-dispatch/make
                       partial-alternatives (contraction-selection partial-alternatives)
                       policy)))))))

(deftest public-contraction-dispatch-is-opt-in-and-retains-runtime-admission
  (register-target!)
  (let [options {:target target :dtype :float
                 :schedule {:typed-contraction {:strategy :dispatch-register-tiled}}}
        fixed (equation-first/compile #'contractions/fixed-matmul options)
        dynamic (equation-first/compile #'contractions/dynamic-matmul options)
        fixed-link (equation-first/lower fixed [(float-array 15) (float-array 21)])
        dynamic-link (equation-first/lower dynamic
                                           [(float-array 15) (float-array 21) 5 7 3])
        empty-link (equation-first/lower dynamic
                                         [(float-array 0) (float-array 21) 0 7 3])]
    (is (= 2 (count (:kernels fixed))))
    (is (= {:kernel-body 2} (get-in fixed [:stats :emission :emission-routes])))
    (is (= 1 (get-in fixed [:stats :emission :contraction-dispatches])))
    (is (= :dispatch-register-tiled
           (get-in fixed [:options :schedule :typed-contraction :strategy])))
    (is (= :register-tiled
           (executable/strategy (-> fixed-link :instances first :call :steps last :graph))))
    (is (= :register-tiled
           (executable/strategy (-> dynamic-link :instances first :call :steps last :graph))))
    (is (= :sequential-segments
           (executable/strategy (-> empty-link :instances first :call :steps last :graph)))
        "a failed runtime register precondition selects the exact admitted fallback")
    (with-redefs [dispatch/admit-alternative
                  (fn [& _] (throw (AssertionError. "public validation reselected dispatch")))]
      (doseq [plan [fixed-link dynamic-link empty-link]
              :let [call (-> plan :instances first :call)]]
        (is (identical? call (program-call/validate! call))
            "source-ordered binding validation checks the retained admitted alternative, not selection")))
    (let [default (equation-first/compile #'contractions/fixed-matmul
                                          {:target target :dtype :float})]
      (is (not (equation-dispatch/emitted-equation-dispatch?
                (-> default :emitted :equations last :operations first)))))
    (let [strict (equation-first/compile
                  #'contractions/fixed-matmul
                  (assoc-in options [:schedule :precision] :f32-scalar))]
      (is (= 0 (get-in strict [:stats :emission :contraction-dispatches])))
      (is (= 1 (get-in strict [:stats :emission :contraction-candidate-declines
                              :register-tiled-numerical-policy])))
      (is (= 1 (count (:kernels strict)))))
    (is (= :equation-first-contraction-selector-unconsumed
           (reason #(equation-first/compile
          #'contractions/fixed-matmul
          (assoc-in options [:schedule :typed-contraction :measured-selectors]
                    {"unsupported-selector" {:kind :fixed-strategy
                                             :strategy :register-tiled}}))))
        "unconsumed measured selectors remain a loud decline, not ignored evidence")
    (is (= :contraction-equation-dispatch-requires-equation-first
           (reason #(pipeline/compile-gpu-program
                     #'contractions/fixed-matmul target :dtype :float
                     :schedule (:schedule options)))))))

(deftest public-contraction-selectors-use-the-existing-tuning-contract
  (register-target!)
  (let [options {:target target :dtype :float
                 :schedule {:typed-contraction {:strategy :dispatch-register-tiled}}}
        original (equation-first/compile #'contractions/fixed-matmul options)
        choice (-> original :emitted :equations last :operations first :dispatch)
        contract (get-in choice [:attributes :tuning])
        descriptor (get-in original [:options :target-descriptor])
        recorded (with-redefs [tuning/cache-put! identity]
                   (tuning/tune-fixed!
                    choice descriptor
                    (fn [candidate]
                      ;; Synthetic evidence validates the transport contract, not GPU performance.
                      (let [signature (tuning/executable-signature candidate)
                            duration (if (= :sequential-segments (:strategy signature)) 10.0 20.0)]
                        {:measurement (measurement/summarize [duration duration duration])
                         :validation {:passed? true :oracle-hash "synthetic-contraction-oracle"
                                      :candidate-hash (:source-hash signature)}}))
                    :force? true :numerical-mode (:numerical-mode contract) :layout (:layout contract)))
        override (benchmark/tuning-schedule-override choice recorded descriptor
                                                    (:numerical-mode contract) (:layout contract))
        selected-options (assoc-in options [:schedule :typed-contraction :measured-selectors]
                                   (get-in override [:typed-contraction :measured-selectors]))
        recompiled (equation-first/compile #'contractions/fixed-matmul selected-options)
        selected (-> recompiled :emitted :equations last :operations first :dispatch)
        linked (equation-first/lower recompiled [(float-array 15) (float-array 21)])]
    (is (= [:typed-contraction :measured-selectors] (:schedule-path contract)))
    (is (= (:id choice) (:schedule-key contract)))
    (is (= :sequential-segments (get-in selected [:selector :strategy])))
    (is (= :supplied-selector (get-in selected [:attributes :selection])))
    (is (= :sequential-segments (:default-strategy selected)))
    (is (= 2 (count (:alternatives selected))))
    (is (= :sequential-segments
           (executable/strategy (-> linked :instances first :call :steps last :graph))))
    (is (= :equation-first-contraction-selector-unconsumed
           (reason #(equation-first/compile
                     #'contractions/fixed-matmul
                     (assoc-in selected-options
                               [:schedule :typed-contraction :measured-selectors "extra-stale-id"]
                               {:kind :fixed-strategy :strategy :sequential-segments})))))
    (is (= :equation-first-contraction-selector-unconsumed
           (reason #(equation-first/compile
                     #'contractions/fixed-matmul
                     (assoc-in selected-options [:schedule :precision] :f32-scalar)))))
    (is (= :equation-first-contraction-pinning-unsupported
           (reason #(equation-first/compile
                     #'contractions/fixed-matmul
                     (assoc-in selected-options
                               [:schedule :typed-contraction :measured-selectors (:id choice) :fallback]
                               :none)))))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"absent strategy"
         (equation-first/compile
          #'contractions/fixed-matmul
          (assoc-in selected-options
                    [:schedule :typed-contraction :measured-selectors (:id choice) :strategy]
                    :unavailable))))))

(deftest reduction-dispatch-preserves-a-conservatively-declined-contraction
  (register-target!)
  (let [compilation (equation-first/compile
                     #'attention-projection/attention-projection
                     {:target target :dtype :float
                      :schedule {:typed-contraction {:strategy :dispatch-register-tiled}
                                 :segmented-weighted-reduction
                                 {:strategy :dispatch-reassociated}}})
        operations (keep (comp first :operations) (get-in compilation [:emitted :equations]))
        dispatched (filter equation-dispatch/emitted-equation-dispatch? operations)]
    (is (= 1 (count dispatched)))
    (is (= 1 (get-in compilation [:stats :emission :reduction-dispatches])))
    (is (= 0 (get-in compilation [:stats :emission :contraction-dispatches])))
    (is (= 1 (get-in compilation [:stats :emission :contraction-candidate-declines
                                 :body-has-unmodeled-terms])))
    (is (= 3 (count (:kernels compilation))))
    (is (= {:kernel-body 3} (get-in compilation [:stats :emission :emission-routes])))
    (is (= 3 (count (set (map :kernel-name (:kernels compilation))))))
    (is (= #{:indexed-segmented-reduction-reference}
           (set (map #(get-in % [:dispatch :default-strategy]) dispatched))))
    (is (= :sequential-segments
           (->> operations (filter emitted-equation/emitted-equation?) first
                :graph :nodes first :operation :attributes :strategy))
        "a double-product/float-result summand stays ordered until conversion is proved")))

(deftest independently-certified-equations-require-numerical-permission
  ;; Source generation is hardware-free; no driver or runtime session is opened.
  (register-target!)
  (let [reference-compilation (compilation :reference)
        subgroup-compilation (compilation :subgroup-score-reuse)
        reference-program (:emitted reference-compilation)
        subgroup-program (:emitted subgroup-compilation)
        reference (-> reference-program :equations last :operations first)
        subgroup (-> subgroup-program :equations last :operations first)
        alternatives [reference subgroup]
        selector (dispatch/make
                  {:id "certified-indexed-equation"
                   :alternatives (mapv :graph alternatives)
                   :default-strategy :indexed-segmented-reduction-reference
                   :selector {:kind :fixed-strategy
                              :strategy :indexed-segmented-reduction-subgroup-score-reuse}})
        exact-only {:permitted-modes #{:exact}}
        reassociation {:permitted-modes #{:exact :reassociated}}
        certified (equation-dispatch/make alternatives selector reassociation)
        program (update reference-program :equations
                        (fn [equations]
                          (update equations (dec (count equations)) assoc
                                  :operations [certified])))]
    (check-candidate-proof-reuse! reference certified emitted-equation/complete-write-domains)
    (is (equation-dispatch/emitted-equation-dispatch? certified))
    (is (= program (emitted-program/validate! program)))
    (let [arguments [(float-array 15) (float-array 15) (float-array 15)
                     (long-array [0 0 2 2]) (long-array [1 1 0 2]) 3 4 5 2]
          linked (equation-first/lower (assoc reference-compilation :emitted program)
                                      arguments)
          selected (-> linked :instances first :call :steps last :graph)]
      (is (= :indexed-segmented-reduction-subgroup-score-reuse
             (executable/strategy selected))))
    (is (= [:exact :reassociated]
           (mapv #(get-in (last (get-in % [:body :equations]))
                          [:operations 0 :numerics :mode]) alternatives)))
    (is (= :equation-dispatch-numerics
           (reason #(equation-dispatch/make alternatives selector exact-only))))
    (is (= :equation-dispatch-numerical-policy
           (reason #(equation-dispatch/make alternatives selector {}))))
    (is (= :equation-dispatch-executables
           (reason #(equation-dispatch/make
                     alternatives
                     (assoc selector :alternatives (vec (reverse (:alternatives selector))))
                     reassociation))))
    (is (= :equation-dispatch-default-numerics
           (reason #(equation-dispatch/make
                     alternatives
                     (assoc selector :default-strategy
                            :indexed-segmented-reduction-subgroup-score-reuse)
                     reassociation))))
    (is (= :emitted-reduction-artifact-refinement
           (reason #(equation-dispatch/validate!
                     (assoc certified :alternatives
                            [reference (assoc-in subgroup
                                                 [:graph :nodes 0 :operation :provenance
                                                  :scheduled-operation :numerics :mode]
                                                 :exact)])))))))

(deftest public-compilation-retains-numerical-dispatch-policy
  (register-target!)
  (let [auto (compilation :auto)
        dispatched (compilation :dispatch-reassociated)
        operation (-> dispatched :emitted :equations last :operations first)
        arguments [(float-array 15) (float-array 15) (float-array 15)
                   (long-array [0 0 2 2]) (long-array [1 1 0 2]) 3 4 5 2]
        wide-arguments [(float-array (* 3 1024)) (float-array (* 3 1024))
                        (float-array (* 3 1024))
                        (long-array [0 0 2 2]) (long-array [1 1 0 2]) 3 4 1024 2]
        linked (equation-first/lower dispatched arguments)
        wide-linked (equation-first/lower dispatched wide-arguments)
        selection (:dispatch operation)]
    (is (not (equation-dispatch/emitted-equation-dispatch?
              (-> auto :emitted :equations last :operations first)))
        "existing :auto retains exact fixed-reference numerics")
    (is (equation-dispatch/emitted-equation-dispatch? operation))
    (is (= :dispatch-reassociated
           (get-in dispatched [:options :schedule :segmented-weighted-reduction :strategy])))
    (is (= 2 (count (:kernels dispatched))))
    (is (= :runtime-scalar-threshold (get-in selection [:selector :kind])))
    (is (= 256 (get-in selection [:selector :threshold])))
    (is (= #{:exact :reassociated}
           (get-in selection [:attributes :tuning :numerical-mode :permitted-modes])))
    (is (= :indexed-segmented-reduction-reference
           (executable/strategy (-> linked :instances first :call :steps last :graph))))
    (is (= :indexed-segmented-reduction-subgroup-score-reuse
           (executable/strategy (-> wide-linked :instances first :call :steps last :graph))))))

(deftest public-dispatch-consumes-measured-selector-with-stable-identity
  (register-target!)
  (let [baseline (compilation :dispatch-reassociated)
        dispatch-id (-> baseline :emitted :equations last :operations first :dispatch :id)
        selector {:kind :fixed-strategy
                  :strategy :indexed-segmented-reduction-subgroup-score-reuse}
        measured (equation-first/compile
                  #'indexed-fixture/resident-indexed-attention-probe
                  {:target target :dtype :float
                   :schedule {:segmented-weighted-reduction
                              {:strategy :dispatch-reassociated
                               :measured-selectors {dispatch-id selector}}}})
        arguments [(float-array 15) (float-array 15) (float-array 15)
                   (long-array [0 0 2 2]) (long-array [1 1 0 2]) 3 4 5 2]
        linked (equation-first/lower measured arguments)]
    (is (= dispatch-id
           (-> measured :emitted :equations last :operations first :dispatch :id)))
    (is (= :measured-runtime-shape
           (-> measured :emitted :equations last :operations first
               :dispatch :attributes :selection)))
    (is (= :indexed-segmented-reduction-subgroup-score-reuse
           (executable/strategy (-> linked :instances first :call :steps last :graph))))))

(deftest certified-dispatch-survives-persistent-artifact-round-trip
  (register-target!)
  (let [original (compilation :dispatch-reassociated)
        restored (artifact/open artifact-identity
                                (artifact/decode
                                 (artifact/encode
                                  (artifact/seal artifact-identity original))))
        arguments [(float-array 15) (float-array 15) (float-array 15)
                   (long-array [0 0 2 2]) (long-array [1 1 0 2]) 3 4 5 2]]
    (is (semantic-fingerprint/equivalent? original restored))
    (is (equation-dispatch/emitted-equation-dispatch?
         (-> restored :emitted :equations last :operations first)))
    (is (= (executable/strategy
            (-> (equation-first/lower original arguments)
                :instances first :call :steps last :graph))
           (executable/strategy
            (-> (equation-first/lower restored arguments)
                :instances first :call :steps last :graph))))))

(deftest reassociated-dispatch-declines-an-unproved-target
  (let [target-id :ocl:unproved-equation-dispatch-test]
    (hardware/register-target-device!
     target-id
     {:type :ocl :name "Synthetic NVIDIA portable equation target"
      :vendor "NVIDIA"
      :capabilities {:warp-size 32 :subgroup-sizes [32]
                     :max-workgroup-size 256 :shared-local-memory 65536 :total-eus 32}})
    (is (= :score-reuse-requires-intel-subgroup-dialect
           (reason #(equation-first/compile
                     #'indexed-fixture/resident-indexed-attention-probe
                     {:target target-id :dtype :float
                      :schedule {:segmented-weighted-reduction
                                 {:strategy :dispatch-reassociated}}}))))))

(deftest legacy-resident-entry-declines-equation-only-dispatch-mode
  (register-target!)
  (is (= :reassociated-equation-dispatch-requires-equation-first
         (reason #(pipeline/compile-gpu-program
                   #'indexed-fixture/resident-indexed-attention-probe target
                   :dtype :float :on-non-resident :nil
                   :schedule {:segmented-weighted-reduction
                              {:strategy :dispatch-reassociated}})))))
