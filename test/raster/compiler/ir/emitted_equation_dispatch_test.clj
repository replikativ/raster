(ns raster.compiler.ir.emitted-equation-dispatch-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.emitted-equation-dispatch :as equation-dispatch]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.kernel-dispatch :as dispatch]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.gpu.indexed-attention-device-test :as indexed-fixture]
            [raster.runtime.hardware :as hardware]))

(def ^:private target :ocl:certified-equation-dispatch-test)

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
  (hardware/register-target-device!
   target
   {:type :ocl :name "Synthetic Intel certified equation dispatch"
    :vendor "Intel"
    :capabilities {:warp-size 16 :subgroup-sizes [16]
                   :max-workgroup-size 256 :shared-local-memory 65536 :total-eus 32}})
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
