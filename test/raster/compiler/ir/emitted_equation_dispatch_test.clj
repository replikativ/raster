(ns raster.compiler.ir.emitted-equation-dispatch-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-artifact :as artifact]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.ir.emitted-equation-dispatch :as equation-dispatch]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
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
