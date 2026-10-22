(ns raster.compiler.passes.parallel.ordered-scalar-fold-fixture
  "Pure native-schedule fixture shared by structural, device and offline compiler checks.
   It is not a public deftm/source-admission qualification."
  (:require [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.passes.parallel.segred-body :as fold]))

(defn operation [dtype seed bound operator]
  (let [step (with-meta (list operator 'acc '(clojure.core/aget a i))
               {:raster.type/tag (if (= :float dtype) 'float 'double)})
        r (reduction/make
           {:components [{:id :value :accumulator 'acc :neutral seed :dtype dtype :result 'result}]
            :index 'i :step (reduction/->ReductionRegion [] [step] {})
            :attributes {:physical-phase :single}})]
    (segop/->SegRed 301 (segop/make-seg-space 'i bound) (segop/->SegLevel :block :none)
                    r nil #{'a} #{'result} #{} (segop/->KernelGrid 1 1 0) :single
                    (fold/scalar-ordered-seeded-fold-schedule r) dtype)))
