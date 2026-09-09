(ns raster.compiler.passes.parallel.carried-effect-loop-fixture
  "Scheduled-region fixture: source/dialect admission is intentionally not claimed."
  (:require [raster.compiler.backend.gpu.segop-opencl :as emit]
            [raster.compiler.ir.segop :as segop]))

(defn scheduled-loop [trips]
  (segop/map->SegMap
   {:id :carried-effect-loop :space (segop/make-seg-space 'i 'rows)
    :dtype :float :inputs #{'x} :outputs #{'words 'totals} :scalars #{'rows}
    :write-conflict :ordered-effects :write-conflicts {'words :unique 'totals :unique}
    :scalar-region
    {:locals [{:id 'seed :dtype :float :init 0.25}]
     :iteration-order :independent
     :effects [{:loop {:index 'k :lower 0 :extent trips
                      :locals [{:id 'loaded :dtype :float :init '(aget x (+ (* i 8) k))}]
                      :effects [{:destination 'words :dtype :float :conflict :unique
                                 :destination-index '(+ (* i 8) k) :predicate true :value 'loaded}]
                      :carry {:parameter 'acc :result 'sum :dtype :float
                              :init 'seed :update (with-meta '(+ acc loaded) {:raster.type/tag 'double})}}}
               {:destination 'totals :dtype :float :conflict :unique
                :destination-index 'i :predicate true :value 'sum}]}}))

(defn artifact [operation target]
  (emit/generate-scheduled-segmap-kernel
   operation :dtype :float :target-dialect target
   :array-types {'x :float 'words :float 'totals :float} :scalar-types {'rows :long}))
