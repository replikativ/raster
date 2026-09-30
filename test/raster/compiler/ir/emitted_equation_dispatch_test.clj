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
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.semantic-fingerprint :as semantic-fingerprint]
            [raster.gpu.indexed-attention-device-test :as indexed-fixture]
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
    (is (= program (emitted-program/validate! program)))
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
      (is (= :equation-dispatch-numerics
             (reason #(equation-dispatch/make
                       forged-alternatives (contraction-selection forged-alternatives)
                       policy)))
          "a register schedule cannot self-label as exact to evade numerical permission"))
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
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"measured typed contraction selectors require"
         (equation-first/compile
          #'contractions/fixed-matmul
          (assoc-in options [:schedule :typed-contraction :measured-selectors]
                    {"unsupported-selector" {:kind :fixed-strategy
                                             :strategy :register-tiled}})))
        "unconsumed measured selectors remain a loud decline, not ignored evidence")
    (is (= :contraction-equation-dispatch-requires-equation-first
           (reason #(pipeline/compile-gpu-program
                     #'contractions/fixed-matmul target :dtype :float
                     :schedule (:schedule options)))))))

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
