(ns raster.compiler.passes.parallel.structured-control-route-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.parallel-program :as program]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.ir.structured-control :as control]
            [raster.compiler.ir.structured-control-schedule :as schedule]
            [raster.compiler.passes.parallel.structured-control-route :as route]))

(defn- loop-decomposition
  []
  (let [extent (av/tensor {:dtype :int :shape []})
        index (av/tensor {:dtype :long :shape []})
        tensor (av/tensor {:dtype :float :shape '[n]})
        inner (av/tensor {:dtype :float :shape '[n-in]})
        equation (list '= 'advance '[u-next]
                       (list 'map {:index 'i :extent 'n-in}
                             '[u-in] '[]
                             (soac/lambda-form '[value] '[(+ value 1.0)])))
        body (soac/make
              (soac/default-program-facts
               {:values {'iteration index 'n-in extent 'u-in inner 'u-next inner}
                :inputs '[n-in u-in]
                :equations {'advance (soac/default-equation-facts)}})
              [equation] '[u-next])
        loop (control/make
              {:id 'time-loop :effects #{} :provenance {:source :test}
               :attributes {:association :sequential}}
              '[iteration steps]
              [{:outer 'n :parameter 'n-in}]
              [{:initial 'u :parameter 'u-in :result 'u-next :output 'u-final}]
              body
              {'steps index 'n extent 'u tensor 'u-final tensor})]
    {:loop loop :loop-binding 'time-loop :source '(let* [] nil)}))

(deftest structured-control-uses-the-common-program-equation-spine
  (let [typed (route/program-envelope (loop-decomposition))
        scheduled (route/schedule-program typed {:target-device :cpu:0 :dtype :float})
        typed-equation (first (:equations typed))
        scheduled-equation (first (:equations scheduled))]
    (is (program/parallel-program? typed))
    (is (= :typed-parallel (:dialect typed)))
    (is (= '[steps n u] (:inputs typed)))
    (is (= '[u-final] (:outputs typed)))
    (is (= (:algorithm typed-equation) (:algorithm scheduled-equation)))
    (is (= :structured-control-schedule (:dialect scheduled)))
    (is (schedule/scheduled-loop? (first (:operations scheduled-equation))))
    (testing "the scheduled equation retains the certified one-iteration graph"
      (is (= :kernel-graph
             (get-in scheduled-equation [:attributes :graph-dialect])))
      (is (= '[u-in]
             (mapv :id (get-in scheduled-equation [:operations 0 :graph :inputs])))))))

(deftest repeated-outer-operands-do-not-break-program-ssa
  (let [{:keys [loop] :as decomposition} (loop-decomposition)
        loop (control/make
              (control/facts loop)
              (assoc (control/loop-index loop) 1 'n)
              (control/invariants loop)
              (control/carried loop)
              (control/body loop)
              (control/outer-values loop))
        typed (route/program-envelope (assoc decomposition :loop loop))]
    (is (= '[n u] (:inputs typed)))
    (is (= '[n u] (:operands (first (:equations typed)))))))
