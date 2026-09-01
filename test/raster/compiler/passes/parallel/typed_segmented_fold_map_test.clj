(ns raster.compiler.passes.parallel.typed-segmented-fold-map-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.segop-opencl :as segop-opencl]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.segfoldmap-body :as fold-body]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.par :as par]))

(def ^:private source
  '(let* [effect
          (raster.par/segmented-fold-map!
           [out] [[segment nsegments]] index width
           [[maximum -1.0E38 :float width
             (clojure.core/max
              maximum
              (clojure.core/aget values
                                 (clojure.core/+ (clojure.core/* segment width) index)))]
            [denominator 0.0 :float width
             (clojure.core/+
              denominator
              (Math/exp
               (clojure.core/-
                (clojure.core/aget values
                                   (clojure.core/+ (clojure.core/* segment width) index))
                maximum)))]]
           [(clojure.core//
             (Math/exp
              (clojure.core/-
               (clojure.core/aget values
                                  (clojure.core/+ (clojure.core/* segment width) index))
               maximum))
             denominator)])]
         effect))

(defn- scheduled-operation []
  (let [program (:program
                 (route/attempt source :float {'values :float 'out :float}
                                {:scalar-types {'nsegments :int 'width :int}}))
        scheduled (:form
                   (segop-lower/segop-lower-pass
                    program {:dtype :float :target-device :ocl:0
                             :array-types {'values :float 'out :float}
                             :scalar-types {'nsegments :int 'width :int}}))]
    (first (:operations (first (:equations scheduled))))))

(deftest interpreted-segmented-fold-map-preserves-dependent-fold-order
  (let [values (float-array [1.0 3.0, 2.0 2.0])
        out (float-array 4)]
    (par/segmented-fold-map!
     [out] [[segment 2]] index 2
     [[maximum -1.0E38 :float 2
       (max maximum (aget values (+ (* segment 2) index)))]
      [denominator 0.0 :float 2
       (+ denominator (Math/exp (- (aget values (+ (* segment 2) index)) maximum)))]]
     [(/ (Math/exp (- (aget values (+ (* segment 2) index)) maximum)) denominator)])
    (is (< (Math/abs (- (aget out 0) 0.11920292)) 1.0e-6))
    (is (< (Math/abs (- (aget out 1) 0.8807971)) 1.0e-6))
    (is (< (Math/abs (- (aget out 2) 0.5)) 1.0e-6))
    (is (< (Math/abs (- (aget out 3) 0.5)) 1.0e-6))))

(deftest dependent-folds-are-first-class-typed-soac
  (let [program (frontend/form->program
                 source {:dtype :float :array-types {'values :float 'out :float}
                         :scalar-types {'nsegments :int 'width :int}})
        equation (first (dialect/equations program))
        operation (dialect/operation-parts equation)]
    (is (= 'segmented-fold-map (:kind operation)))
    (is (= 2 (count (:folds operation))))
    (is (= '[nsegments values width] (:inputs (dialect/facts program))))
    (is (= '[out] (dialect/physical-results program equation)))
    (is (= :int (:dtype (get-in (dialect/facts program) [:values 'width])))
        "launch dimensions retain their declared ABI representation")))

(deftest portable-schedule-is-three-explicit-ordered-loops
  (let [operation (scheduled-operation)
        {:keys [kernel-body]} (fold-body/lower
                               operation {:workgroup-size 64
                                          :array-types {'values :float 'out :float}
                                          :scalar-types {'nsegments :int 'width :int}})
        loops (filterv #(instance? raster.compiler.ir.kernel_body.ForLoop %)
                       (:operations kernel-body))]
    (is (= 3 (count loops)) "maximum, denominator, then the final dense map")
    (is (= [:ordered :ordered :ordered]
           (mapv #(get-in % [:attributes :association]) loops)))
    (is (str/includes? (pr-str (second loops)) "maximum")
        "the second fold consumes the completed first fold")
    (is (empty? (:iter-args (last loops))))
    (is (= :final-map (get-in (last loops) [:attributes :role])))
    (is (= ['values] (mapv :buffer (:stable-reads kernel-body))))))

(deftest one-certified-schedule-emits-for-all-c-family-targets
  (let [operation (scheduled-operation)]
    (doseq [[target expected]
            [[:opencl-portable :opencl-c]
             [:cuda :cuda-c]
             [:hip :hip-cpp]]]
      (testing (name target)
        (let [kernel (segop-opencl/generate-segfoldmap-kernel
                      operation :target-dialect target
                      :array-types {'values :float 'out :float}
                      :scalar-types {'nsegments :int 'width :int})]
          (is (artifact/kernel-artifact? kernel))
          (is (= expected (:target kernel)))
          (is (= :no-write-alias (get-in kernel [:abi 0 :aliasing])))
          (is (every? #(<= (int %) 127) (:source kernel))))))))

(deftest scalar-captures-retain-their-typed-abi
  (let [scaled-source
        '(let* [effect
                (raster.par/segmented-fold-map!
                 [out] [[segment nsegments]] index width
                 [[sum 0.0 :float width
                   (clojure.core/+ sum
                                   (clojure.core/aget values
                                                      (clojure.core/+ (clojure.core/* segment width)
                                                                      index)))]]
                 [(clojure.core/* scale sum)])]
               effect)
        program (:program
                 (route/attempt scaled-source :float {'values :float 'out :float}
                                {:scalar-types {'nsegments :int 'width :int 'scale :float}}))
        operation (-> (segop-lower/segop-lower-pass
                       program {:dtype :float :target-device :ocl:0
                                :array-types {'values :float 'out :float}
                                :scalar-types {'nsegments :int 'width :int 'scale :float}})
                      :form :equations first :operations first)
        kernel (segop-opencl/generate-segfoldmap-kernel
                operation :target-dialect :opencl-portable
                :array-types {'values :float 'out :float}
                :scalar-types {'nsegments :int 'width :int 'scale :float})
        scale-parameter (some #(when (= 'scale (:name %)) %) (:abi kernel))]
    (is (= :float (:dtype scale-parameter)))
    (is (str/includes? (:source kernel) "float scale"))))

(deftest a-fold-cannot-reference-a-future-accumulator
  (let [bad
        '(let* [effect
                (raster.par/segmented-fold-map!
                 [out] [[segment nsegments]] index width
                 [[first 0.0 :float width (clojure.core/+ first future)]
                  [future 0.0 :float width (clojure.core/+ future 1.0)]]
                 [first])]
               effect)]
    (try
      (frontend/form->program
       bad {:dtype :float :array-types {'out :float}
            :scalar-types {'nsegments :int 'width :int}})
      (is false "future fold dependencies must fail validation")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :typed-soac-segmented-fold-map-fold-closure
               (:reason (ex-data exception))))))))
