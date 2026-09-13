(ns raster.compiler.passes.parallel.typed-soac-initialization-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.kernel-body-compile-fixtures :as fixtures]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-initialization :as initialization]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(defn- source [allocation extent]
  (list 'let* ['output allocation
               'written (list 'raster.par/map! 'output 'i extent nil '(aget input i))]
        'written))

(defn- program [source]
  (frontend/form->program source {:dtype :double :array-types {'input :float 'output :float}}))

(deftest zero-initialization-needs-full-coverage-not-write-permission
  (doseq [[extent fills elided] [[4 1 0] [8 0 1]]]
    (let [source (source '(float-array 8) extent)
          [scheduled stats] (initialization/materialize (program source))]
      (is (= fills (:initialization-fills stats)))
      (is (= elided (:initialization-full-overwrites stats)))
      (is (= (inc fills) (count (dialect/equations scheduled))))
      (when (pos? fills)
        (let [fill (first (dialect/equations scheduled))
              result (first (nth fill 2))
              facts (dialect/facts scheduled)
              realized (#'route/realize-source source scheduled)
              pairs (mapv vec (partition 2 (second (:source realized))))]
          (is (= 'map (dialect/operation-kind fill)))
          (is (= :float (get-in facts [:values result :dtype])))
          (is (= [8] (get-in facts [:values result :shape])))
          (is (= '[output (float-array 8)] (first pairs)) "allocation survives placement")
          (is (= 3 (count pairs)) "fill precedes the prefix write")
          (is (= scheduled (first (initialization/materialize scheduled))) "idempotent"))))))

(deftest unspecified-and-copy-storage-do-not-acquire-zero-fills
  (doseq [allocation ['(raster.math/alloc-like input 8) '(aclone input)]]
    (let [program (program (source allocation 4))
          [scheduled stats] (initialization/materialize program)]
      (is (= program scheduled))
      (is (zero? (:initialization-fills stats))))))

(defn- with-facts [program f]
  (dialect/make (f (dialect/facts program))
                (dialect/equations program) (dialect/outputs program)))

(defn- reason [program]
  (try (initialization/materialize program) nil
       (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest initialization-contracts-fail-closed
  (let [program (program (source '(float-array 8) 4))]
    (doseq [change [#(update-in % [:attributes :allocations] into
                               (get-in % [:attributes :allocations]))
                    #(assoc-in % [:attributes :allocations 0 :dtype] :double)
                    #(assoc-in % [:attributes :allocations 0 :initialization] :invented)
                    #(assoc-in % [:attributes :allocations 0 :extent] -1)
                    #(assoc-in % [:attributes :allocations 0 :extent] 'input)
                    #(assoc-in % [:attributes :allocations 0 :extent] '(do (mutate!) 8))
                    #(assoc-in % [:attributes :allocations 0 :source-binding-id] nil)
                    #(update-in % [:equations 1 :provenance] dissoc :source-binding-id)
                    #(assoc-in % [:attributes :allocations 0 :source-binding-id] 2)]]
      (is (= :typed-soac-initialization-contract (reason (with-facts program change)))))))

(deftest generated-initializers-avoid-host-symbols
  (let [program (with-facts (program (source '(float-array 8) 4))
                  #(update-in % [:attributes :source-bindings] conj
                              'rstr_initialized_0 'rstr_initialization_equation_0))
        [scheduled] (initialization/materialize program)
        fill (first (dialect/equations scheduled))]
    (is (not= 'rstr_initialized_0 (first (nth fill 2))))
    (is (not= 'rstr_initialization_equation_0 (second fill)))))

(deftest aliases-cannot-turn-a-read-into-an-overwrite-proof
  (let [program (with-facts (program (source '(float-array 8) 8))
                  #(update-in % [:equations 1 :aliases] assoc 'input 'middle 'middle 'output))
        [_ stats] (initialization/materialize program)]
    (is (= 1 (:initialization-fills stats)))
    (is (zero? (:initialization-full-overwrites stats)))
    (is (= :typed-soac-initialization-contract
           (reason (with-facts program #(assoc-in % [:equations 1 :aliases 'middle] 'input)))))))

(deftest host-observations-prevent-late-initialization-and-elision
  (doseq [extent [4 8]]
    (let [program (with-facts (program (source '(float-array 8) extent))
                    #(-> %
                         (assoc-in [:equations 1 :provenance :source-binding-id] 2)
                         (assoc-in [:attributes :host-read-sites]
                                   [{:source-binding-id 1 :values #{'output}}])))]
      (is (= :typed-soac-initialization-contract (reason program))))))

(deftest public-initialization-cross-compiles-without-a-device
  (doseq [target [:cuda:0 :hip:0]
          [function arguments] [[#'fixtures/public-c-family-scatter
                                 [(float-array [1 2 3]) (int-array [0 0 2]) 3]]
                                [#'fixtures/public-c-family-strided-scatter
                                 [(float-array [1 2 3 4 5 6]) (int-array [0 0 2]) 3 2]]]]
    (let [compilation (equation-first/compile function {:target target :dtype :float})
          plan (equation-first/lower compilation arguments)]
      (is (= [:kernel-body :kernel-body]
             (mapv #(get-in % [:attributes :emission-route]) (:kernels compilation))))
      (is (link-plan/link-plan? plan))
      (is (= 0 (get-in plan [:attributes :driver-allocations]))))))

(deftest allocation-facts-compose-with-value-remapping
  (let [original (frontend/form->program
                  (source '(float-array n) 'n)
                  {:dtype :float :array-types {'input :float 'output :float}
                   :scalar-types {'n :long}})
        original (with-facts original
                   #(assoc-in % [:attributes :host-read-sites]
                              [{:source-binding-id 2 :values #{'output}}]))
        remapped (dialect/remap-values original {'output 'storage 'n 'length})
        [scheduled stats] (initialization/materialize remapped)]
    (is (= 'storage (get-in (dialect/facts scheduled) [:attributes :allocations 0 :destination])))
    (is (= 'length (get-in (dialect/facts scheduled) [:attributes :allocations 0 :extent])))
    (is (= #{'storage} (get-in (dialect/facts scheduled) [:attributes :host-read-sites 0 :values])))
    (is (= '[output written] (get-in (dialect/facts scheduled) [:attributes :source-bindings])))
    (is (= 1 (:initialization-full-overwrites stats)))))

(deftest allocation-extent-normalization-is-hygienic-and-precedes-allocation
  (let [source '(let* [output (float-array (* n width))
                       written (raster.par/map! output i (* n width) nil
                                               (+ (aget input i) rstr_extent_0))]
                 written)
        normalized (frontend/normalize-source
                    source {:array-types {'input :float 'output :float}
                            :scalar-types {'n :long 'width :long 'rstr_extent_0 :long}})
        [[extent expression] [output allocation] [_ consumer]]
        (partition 2 (second normalized))]
    (is (not= 'rstr_extent_0 extent))
    (is (= '(* n width) expression))
    (is (= 'output output))
    (is (= (list 'float-array extent) allocation))
    (is (= extent (nth consumer 3)))
    (is (= '(+ (aget input i) rstr_extent_0) (last consumer)))
    (is (= normalized (frontend/normalize-source
                      normalized {:array-types {'input :float 'output :float}
                                  :scalar-types {'n :long 'width :long 'rstr_extent_0 :long}})))))
