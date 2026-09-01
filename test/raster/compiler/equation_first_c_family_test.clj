(ns raster.compiler.equation-first-c-family-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing use-fixtures]]
            [raster.arrays]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.core :refer [deftm]]
            [raster.dl.attention :as attention]
            [raster.numeric]
            [raster.par]
            [raster.runtime.hardware :as hardware]))

(def ^:private cuda-target :cuda:equation-first-source-test)
(def ^:private hip-target :hip:equation-first-source-test)

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
                     :subgroup-sizes [32 64]
                     :max-workgroup-size 1024
                     :shared-local-memory 65536
                     :total-eus 60}})
    (f)))

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

(deftm c-family-effect-map!
  [input :- (Array float) left :- (Array float) right :- (Array long) n :- Long] :- Void
  (raster.par/map-void!
   index n
   (do (raster.arrays/aset left index
                           (float (raster.numeric/* (float 2.0)
                                                    (raster.arrays/aget input index))))
       (raster.arrays/aset right index (long index)))))

(deftm uncovered-stencil
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

(defn- reason-of
  [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo exception
      (ex-data exception))))

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
        (is (= 1 (count (:outputs linked))))
        (is (= (:emitted compilation)
               (emitted-program/validate! (:emitted compilation))))))))

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
      (is (every? #(= :ordered (get-in % [:attributes :association])) loops))
      (is (= 0 (get-in linked [:attributes :driver-allocations])))
      (is (= 1 (count (:outputs linked))))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest uncovered-c-family-route-never-borrows-opencl-source
  (doseq [target [cuda-target hip-target]]
    (let [failure (reason-of #(equation-first/compile
                               #'uncovered-stencil {:target target :dtype :float}))]
      (is (contains? #{:equation-first-coverage :kernel-graph-target-lowering-missing}
                     (:reason failure)))
      (is (= :none (:fallback failure))))))

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

(deftest emitted-program-rejects-a-mixed-target-module
  (let [compilation (equation-first/compile
                     #'c-family-dot {:target cuda-target :dtype :float})
        mixed (assoc-in (:emitted compilation)
                        [:equations 0 :operations 0 :graph :nodes 0 :operation :target]
                        :hip-cpp)
        failure (reason-of #(emitted-program/validate! mixed))]
    (is (= :emitted-parallel-program-target (:reason failure)))
    (is (= :cuda-c (:expected-target failure)))))
