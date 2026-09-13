(ns raster.compiler.equation-first-c-family-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing use-fixtures]]
            [raster.arrays]
            [raster.compiler.compatibility-ledger-test :as ledger]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.gpu.parallel-program :as program-runtime]
            [raster.core :refer [deftm]]
            [raster.dl.attention :as attention]
            [raster.dl.array-ops :as array-ops]
            [raster.numeric]
            [raster.ode.pde :as pde]
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

(deftm c-family-scan
  "A public certified scan lowered as a three-stage portable KernelGraph."
  [input :- (Array float) n :- Long] :- (Array float)
  (let [output (float-array n)]
    (raster.par/scan output accumulator (float 0.0) index n float
                     (raster.numeric/+ accumulator
                                       (raster.arrays/aget input index)))))

(deftm c-family-effect-map!
  [input :- (Array float) left :- (Array float) right :- (Array long) n :- Long] :- Void
  (raster.par/map-void!
   index n
   (do (raster.arrays/aset left index
                           (float (raster.numeric/* (float 2.0)
                                                    (raster.arrays/aget input index))))
       (raster.arrays/aset right index (long index)))))

(deftm c-family-case-map!
  [choices :- (Array int) output :- (Array float) n :- Long] :- Void
  (raster.par/map-void!
   index n
   (case (raster.arrays/aget choices index)
     0 (raster.arrays/aset output index (float 10.0))
     1 (raster.arrays/aset output index (float 20.0))
     nil)))

(deftm c-family-shifted-inout!
  [output :- (Array float) n :- Long] :- Void
  (raster.par/map-void!
   index n
   (raster.arrays/aset
    output index
    (raster.arrays/aget output
                        (if (< (inc index) n) (inc index) (long 0))))))

(deftm c-family-stencil
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

(deftest public-staged-contraction-preserves-independent-types-on-cuda-and-hip
  (doseq [[target module-target] [[cuda-target :cuda-c] [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile
                       #'ledger/staged-byte-float-contract! {:target target :dtype :float})
          linked (equation-first/lower compilation
                                       [(byte-array 8) (byte-array 16)
                                        (float-array 2) (float-array 4) (float-array 2)])
          kernels (:kernels compilation)]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 1 (count kernels)))
      (is (= module-target (:target (first kernels))))
      (is (every? #(get-in % [:attributes :kernel-body]) kernels))
      (is (str/includes? (:source (first kernels)) "rstr_dp4a"))
      (is (= 0 (get-in linked [:attributes :driver-allocations])))
      (is (empty? (:outputs linked)) "the public Void destination remains a state buffer")
      (let [call (:call (first (:instances linked)))
            equation (get-in call [:program :equations 0])
            algorithm (get-in equation [:operations 0 :algorithm])
            result (first (:results equation))
            physical (first (soac/physical-results algorithm (first (soac/equations algorithm))))
            make-call #(program-call/make (:program call)
                                          (assoc (:buffers call) result :unrelated-result)
                                          (:scalar-values call) {} nil %)
            forged (make-call {result physical})
            binds (atom 0)
            executor {:bind! (fn [& _] (swap! binds inc)) :run! identity :release! identity}
            reason (fn [f] (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
        (is (= :emitted-program-result-views (reason #(make-call {:not-a-result physical}))))
        (is (= :parallel-program-result-view-resolver
               (reason #(program-runtime/prepare-with! forged executor))))
        (is (= :emitted-program-result-view-bindings
               (reason #(program-call/validate!
                         (assoc-in forged [:steps 0 :buffers physical] :wrong-kernel-buffer)))))
        (is (= :emitted-program-result-view-bindings
               (reason #(program-call/validate!
                         (assoc-in forged [:steps 0 :outputs result] :wrong-logical-buffer)))))
        (is (= :parallel-program-result-view-shape
               (reason #(program-runtime/prepare-with!
                         forged (assoc executor :buffer-view
                                       (fn [token]
                                         (bview/view
                                          (bview/allocation {:id :shared :byte-size 8 :device target
                                                             :memory-space :device :ownership :owned})
                                          {:dtype :float :shape [(if (= token :unrelated-result) 1 2)]})))))))
        (is (= :parallel-program-result-view
               (reason #(program-runtime/prepare-with!
                         forged (assoc executor :buffer-view
                                       (fn [token]
                                         (bview/view
                                          (bview/allocation {:id token :byte-size 8 :device target
                                                             :memory-space :device :ownership :owned})
                                          {:dtype :float :shape [2]})))))))
        (is (zero? @binds) "unrelated result views must fail before staging any graph"))
      (doseq [[slot short-buffer] [[0 (byte-array 7)] [2 (float-array 1)]
                                  [4 (float-array 1)]]]
        (is (thrown? clojure.lang.ExceptionInfo
                     (equation-first/lower
                      compilation
                      (assoc [(byte-array 8) (byte-array 16) (float-array 2)
                              (float-array 4) (float-array 2)] slot short-buffer)))
            "core, scale and destination capacities are checked before allocation"))
      (is (= (:emitted compilation) (emitted-program/validate! (:emitted compilation)))))))

(deftest staged-result-views-initialize-only-the-written-prefix
  (let [output (float-array 4)
        compilation (equation-first/compile
                     #'ledger/staged-byte-float-contract!
                     {:target cuda-target :dtype :float
                      :values {'out (av/tensor {:dtype :float :shape [4]})}})
        linked (equation-first/lower compilation [(byte-array 8) (byte-array 16)
                                                  (float-array 2) (float-array 4) output])
        call (:call (first (:instances linked)))
        result (first (get-in call [:steps 0 :equation :results]))
        prefix (get (:buffers call) result)
        base (get (:buffers call) 'out)
        fresh (-> linked
                  (assoc-in [:nodes base :source] nil)
                  (assoc-in [:nodes base :role] :internal)
                  (assoc :outputs [prefix]))
        reason (fn [plan] (try (link-plan/validate! plan) nil
                              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (is (not= base prefix))
    (is (= (get-in linked [:nodes base :view :allocation])
           (get-in linked [:nodes prefix :view :allocation]))
        "result views share storage without another allocation")
    (is (nil? (get-in linked [:nodes prefix :source])))
    (is (nil? (reason fresh)) "the contraction produces its entire logical prefix")
    (is (= :link-unproduced-output (reason (assoc fresh :outputs [base])))
        "writing a prefix cannot initialize or export the untouched tail")
    (is (= :program-link-result-view
           (reason (assoc-in fresh [:nodes prefix :view :byte-offset] 4)))
        "a claimed prefix may not be shifted into another part of the allocation")))

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

(deftest public-scan-uses-one-portable-kernel-body-graph
  (doseq [[target program-dialect module-target]
          [[cuda-target :cuda-parallel :cuda-c]
           [hip-target :hip-parallel :hip-cpp]]]
    (let [compilation (equation-first/compile
                       #'c-family-scan {:target target :dtype :float})
          linked (equation-first/lower
                  compilation [(float-array [1 2 3 4]) 4])
          kernels (:kernels compilation)]
      (is (= program-dialect (get-in compilation [:emitted :dialect])))
      (is (= [:intra-block :block-scan :carry-in]
             (mapv #(get-in % [:attributes :phase]) kernels)))
      (is (every? #(= module-target (:target %)) kernels))
      (is (every? #(= :portable-segscan
                      (get-in % [:attributes :kernel-body :attributes :kind]))
                  kernels))
      (is (not-any? #(re-find #"__kernel|get_global_id|get_local_id" (:source %)) kernels))
      (is (= 0 (get-in linked [:attributes :driver-allocations])))
      (is (= 1 (count (:outputs linked))))
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

(deftest public-integer-case-map-lowers-through-portable-kernel-control
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile
                       #'c-family-case-map! {:target target :dtype :float})
          kernel (first (:kernels compilation))
          operations (nested-operations
                      (get-in kernel [:attributes :kernel-body :operations]))]
      (is (= module-target (:target kernel)))
      (is (= :portable-segmap
             (get-in kernel [:attributes :kernel-body :attributes :kind])))
      (is (some #(= "IfRegion" (some-> % class .getSimpleName)) operations))
      (is (not (str/includes? (:source kernel) "case*")))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest portable-inout-map-refuses-a-cross-lane-read
  (let [reason (reason-of #(equation-first/compile
                            #'c-family-shifted-inout!
                            {:target cuda-target :dtype :float}))]
    (is (= :kernel-graph-target-lowering-missing (:reason reason)))
    (is (= :inout-storage
           (get-in reason [:kernel-body-decline :missing-rule])))))

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
      (is (every? #(= :ordered (get-in % [:attributes :association]))
                  (remove #(= :segment-grid-stride (get-in % [:attributes :role])) loops)))
      (is (every? #(= :independent (get-in % [:attributes :association]))
                  (filter #(= :segment-grid-stride (get-in % [:attributes :role])) loops)))
      (is (= 0 (get-in linked [:attributes :driver-allocations])))
      (is (= 1 (count (:outputs linked))))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-stencil-uses-portable-kernel-body
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile
                       #'c-family-stencil {:target target :dtype :float})
          kernel (first (:kernels compilation))]
      (is (= [module-target] (mapv :target (:kernels compilation))))
      (is (= :portable-segstencil
             (get-in kernel [:attributes :kernel-body :attributes :kind])))
      (is (not (re-find #"__kernel|get_global_id|get_local_id" (:source kernel))))
      (is (= :none (get-in compilation [:stats :fallback]))))))

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

(deftest counted-mixed-precision-stores-use-explicit-destination-conversion
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile #'array-ops/reduce-axis-backward
                                            {:target target :dtype :float})]
      (is (seq (:kernels compilation)))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest heat-2d-counted-stores-use-the-public-c-family-boundary
  (doseq [[target module-target] [[cuda-target :cuda-c] [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile #'pde/heat-rhs-2d!
                                            {:target target :dtype :double})
          plan (equation-first/lower compilation
                                     [(double-array 35) (double-array 35)
                                      5 7 0.25 4.0 9.0])]
      (is (seq (:kernels compilation)))
      (is (every? #(= module-target (:target %)) (:kernels compilation)))
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 0 (get-in plan [:attributes :driver-allocations]))))))

(deftest emitted-program-rejects-a-mixed-target-module
  (let [compilation (equation-first/compile
                     #'c-family-dot {:target cuda-target :dtype :float})
        mixed (assoc-in (:emitted compilation)
                        [:equations 0 :operations 0 :graph :nodes 0 :operation :target]
                        :hip-cpp)
        failure (reason-of #(emitted-program/validate! mixed))]
    (is (= :kernel-graph-executable-targets (:reason failure))
        "the executable graph rejects a mixed target before its enclosing program must")
    (is (= #{:cuda-c :hip-cpp} (:targets failure)))))
