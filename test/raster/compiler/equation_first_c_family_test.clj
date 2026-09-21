(ns raster.compiler.equation-first-c-family-test
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing use-fixtures]]
            [raster.arrays]
            [raster.compiler.compatibility-ledger-test :as ledger]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.fixtures.checked-casts :as checked-casts]
            [raster.compiler.ir.emitted-parallel-program :as emitted-program]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.parallel-program :as program-runtime]
            [raster.core :refer [deftm]]
            [raster.dl.attention :as attention]
            [raster.dl.array-ops :as array-ops]
            [raster.dl.diffusion :as diffusion]
            [raster.dl.loss :as loss]
            [raster.dl.nn :as dl-nn]
            [raster.numeric]
            [raster.nn :as nn]
            [raster.ode.pde :as pde]
            [raster.ode.multilevel :as multilevel]
            [raster.par]
            [raster.runtime.hardware :as hardware]))

(def ^:private cuda-target :cuda:equation-first-source-test)
(def ^:private hip-target :hip:equation-first-source-test)
(def ^:private ocl-target :ocl:equation-first-source-test)

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
    (hardware/register-target-device!
     ocl-target
     {:type :ocl
      :name "Synthetic portable OpenCL equation-first source target"
      :capabilities {:warp-size 32
                     :subgroup-sizes [16 32]
                     :max-workgroup-size 1024
                     :shared-local-memory 65536
                     :total-eus 32}})
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

(deftm c-family-lane-owned-inout!
  [output :- (Array float) n :- Long] :- Void
  (raster.par/map-void!
   index n
   (raster.arrays/aset output index
                       (raster.numeric/+ (raster.arrays/aget output index)
                                         (float 1.0)))))

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

(deftm c-family-broadcast-product-map!
  "A short tuple reduction followed by an ordered epilogue over compiler-owned intermediates."
  [input :- (Array int), weights :- (Array int), output :- (Array int), rows :- Long] :- Void
  (let [segments (* rows 8)
        partials (int-array segments)]
    (raster.par/product-reduce!
     [partials]
     [[sum 0 :int]]
     [[row rows] [lane 8]]
     chunk 8
     [value (unchecked-add-int
             (raster.arrays/aget input (+ (* (+ (* row 8) lane) 8) chunk))
             ;; Reduction-major weights are shared across rows. This is the same verified
             ;; permutation+broadcast shape used by packed matrix/vector products.
             (raster.arrays/aget weights (+ (* chunk 8) lane)))]
     [value]
     [[left right]]
     []
     [(unchecked-add-int left right)]
     {:associative? true :commutative? true
      :overflow :wrap :order :implementation-defined})
    (raster.par/map-void!
     row rows
     (let [base (* row 8)
           total (loop [lane 0 sum 0]
                   (if (< lane 8)
                     (recur (inc lane)
                            (unchecked-add-int
                             sum (raster.arrays/aget partials (+ base lane))))
                     sum))]
       (raster.arrays/aset output row total)))))

(defn- reason-of
  [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo exception
      (ex-data exception))))

(deftest equation-first-rejects-explicit-host-orchestration-before-lowering
  (is (= :equation-first-host-only
         (:reason (reason-of #(equation-first/compile
                              #'pde/solve-fixed-step
                              {:target cuda-target :dtype :double}))))))

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
        (is (= #{'left 'right}
               (set (keys (get-in linked [:attributes :public-buffer-bindings])))))
        (is (= {'left :input 'right :input}
               (get-in linked [:attributes :public-buffer-roles])))
        (is (= 1 (count (:outputs linked))))
        (is (= (set (:outputs linked))
               (:complete-writes (link-plan/initialization-contract linked)))
            "a scalar reduction establishes its complete one-element result")
        (is (= (:emitted compilation)
               (emitted-program/validate! (:emitted compilation))))))))

(deftest equation-first-link-retains-public-output-and-state-roles
  (let [output (float-array 8)
        effect-plan
        (equation-first/lower
         (equation-first/compile #'c-family-effect-map!
                                 {:target cuda-target :dtype :float})
         [(float-array 8) output (long-array 8) 8])
        state-plan
        (equation-first/lower
         (equation-first/compile #'c-family-lane-owned-inout!
                                 {:target cuda-target :dtype :float})
         [output 8])]
    (is (= {'input :input 'left :output 'right :output}
           (get-in effect-plan [:attributes :public-buffer-roles])))
    (is (= {'output :state}
           (get-in state-plan [:attributes :public-buffer-roles])))
    (is (= (get-in effect-plan [:attributes :public-buffer-bindings 'left])
           (some (fn [[id node]] (when (identical? output (:source node)) id))
                 (:nodes effect-plan))))))

(deftest compiled-artifact-consumes-the-equation-first-link-contract
  (let [input (float-array 8)
        output (float-array 8)
        functional (compiled/lower #'c-family-elementwise [input 8]
                                   {:compiler :equation-first
                                    :target cuda-target :dtype :float})
        effect (compiled/lower #'c-family-effect-map!
                               [input output (long-array 8) 8]
                               {:compiler :equation-first
                                :target cuda-target :dtype :float
                                :outputs '[left right]})
        state (compiled/lower #'c-family-lane-owned-inout! [output 8]
                              {:compiler :equation-first
                               :target cuda-target :dtype :float})]
    (is (every? compiled/prepared? [functional effect state]))
    (is (every? #(true? (get-in % [:descriptor :equation-first?]))
                [functional effect state]))
    (is (= [[:input :input]]
           (mapv (juxt :key :role) (:in-tree functional))))
    (is (= [:result] (mapv :key (:out-tree functional))))
    (is (= [[:input :input] [:left :output] [:right :output]]
           (mapv (juxt :key :role) (:in-tree effect))))
    (is (= [:left :right] (mapv :key (:out-tree effect))))
    (is (= [[:output :state]]
           (mapv (juxt :key :role) (:in-tree state))))
    (is (every? link-plan/link-plan?
                (map compiled/plan [functional effect state])))
    (is (invocation-link/certificate? (compiled/certificate functional)))
    (is (= :invocation-link-certificate
           (:reason
            (reason-of
             #(invocation-link/verify!
               (assoc-in (:lowering functional)
                         [:plan :attributes :public-buffer-roles 'input]
                         :state))))))
    (is (every? #(zero? (get-in (compiled/certificate %)
                                [:driver-allocations]))
                [functional effect state]))))

(deftest trusted-equation-first-construction-derives-its-link-witness-once
  (compiled/clear-compilation-cache!)
  (try
    (let [derive-var (ns-resolve 'raster.compiler.ir.invocation-link 'derive-certificate)
          derive @derive-var
          calls (atom 0)]
      (with-redefs-fn
        {derive-var (fn [plan effect-evidence]
                      (swap! calls inc)
                      (derive plan effect-evidence))}
        (fn []
          (let [prepared (compiled/lower #'c-family-elementwise [(float-array 8) 8]
                                         {:compiler :equation-first
                                          :target cuda-target :dtype :float})
                lowering (:lowering prepared)]
            (is (= 1 @calls))
            (is (identical? lowering (invocation-link/verify! lowering)))
            (is (= 2 @calls))))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest equation-first-compiled-artifacts-compose-before-allocation
  (compiled/clear-compilation-cache!)
  (try
    (let [prepare #(compiled/lower #'c-family-elementwise [(float-array 8) 8]
                                   {:compiler :equation-first
                                    :target cuda-target :dtype :float})
          composite (compiled/compose
                     {:id :equation-first-pipeline
                      :components [{:id :first :program (prepare)}
                                   {:id :second :program (prepare)}]
                      :connections [{:from [:first :result]
                                     :to [:second :input]}]
                      :outputs [{:key :result :from [:second :result]}]})]
      (is (compiled/prepared? composite))
      (is (= [[:first :input]] (mapv :key (:in-tree composite))))
      (is (= [:result] (mapv :key (:out-tree composite))))
      (is (= 2 (count (:instances (compiled/plan composite)))))
      (is (= 0 (get-in (compiled/certificate composite) [:driver-allocations] 0)))
      (is (= {:hits 1 :misses 1 :compilations 1 :failures 0
              :entries 1 :entries-by-compiler {:equation-first 1}}
             (dissoc (compiled/compilation-cache-stats) :compile-nanos))))
    (finally
      (compiled/clear-compilation-cache!))))

(deftest public-softmax-backward-keeps-the-reduction-resident
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile
                       #'nn/softmax-backward {:target target :dtype :double})
          linked (equation-first/lower
                  compilation [(double-array [1.0 2.0 3.0])
                               (double-array [0.2 0.3 0.5])])
          values (vals (get-in compilation [:semantic :values]))]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 1 (count (filter #(= :resident-scalar-buffer
                                 (get-in % [:representation :kind])) values))))
      (is (= 3 (count (:kernels compilation))))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation)))
      (is (= 0 (get-in linked [:attributes :driver-allocations])))
      (is (= 1 (count (:outputs linked)))))))

(deftest public-softmax-is-a-complete-typed-soac-program
  (doseq [target [cuda-target hip-target ocl-target]]
    (let [compilation (equation-first/compile #'nn/softmax {:target target :dtype :float})
          semantic (:semantic compilation)
          linked (equation-first/lower compilation [(float-array [1.0 2.0 3.0])])]
      (is (= :typed-parallel (:dialect semantic)))
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (seq (get-in semantic [:attributes :invocation-plan :steps])))
      (is (link-plan/link-plan? linked))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation)))
      (is (some #(and (str/includes? (:source %) "isnan(")
                      (str/includes? (:source %) "fmax"))
                (:kernels compilation))
          "the max tree preserves Math/Raster NaN and signed-zero semantics"))))

(deftest public-loss-composition-keeps-device-results-resident
  (doseq [target [cuda-target hip-target]
          [operation arguments expected-kernels expected-outputs]
          [[#'nn/cross-entropy
            [(float-array [0.1 0.7 0.2]) (float-array [0.0 1.0 0.0])] 2 1]
           [#'nn/softmax-cross-entropy
            [(float-array [1.0 2.0 3.0]) (float-array [0.0 1.0 0.0])] 8 2]
           [#'nn/loss-fn
            [(float-array [0.1 0.2 0.3 0.4 0.5 0.6]) (float-array [0.0 0.0])
             (float-array [0.1 0.2 0.3 0.4]) (float-array [0.0 0.0])
             (float-array [1.0 2.0 3.0]) (float-array [0.0 1.0])]
            9 1]]]
    (let [compilation (equation-first/compile operation {:target target :dtype :float})
          linked (equation-first/lower compilation arguments)]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= expected-kernels (count (:kernels compilation))))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation)))
      (is (= expected-outputs (count (:outputs linked))))
      (is (= 0 (get-in linked [:attributes :driver-allocations]))))))

(deftest public-array-clone-is-a-generated-identity-map
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile
                       #'nn/dense-backward-db {:target target :dtype :float})
          linked (equation-first/lower compilation [(float-array [1.0 2.0 3.0])])]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 1 (count (:kernels compilation))))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation)))
      (is (= 1 (count (:outputs linked))))
      (is (= 0 (get-in linked [:attributes :driver-allocations]))))))

(deftest allocating-dense-weight-gradient-elides-its-dead-zero-fill
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile
                       #'nn/dense-backward-dW {:target target :dtype :float})
          linked (equation-first/lower
                  compilation [(float-array [1.0 2.0]) (float-array [3.0 4.0 5.0])])]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 1 (count (:kernels compilation))))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation)))
      (is (= 1 (count (:outputs linked))))
      (is (= 0 (get-in linked [:attributes :driver-allocations]))))))

(deftest public-huber-loss-shares-typed-conditional-reduction-lowering
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile
                       #'loss/huber-loss {:target target :dtype :float})]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 2 (count (:kernels compilation))))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation)))
      (is (some #(str/includes? (:source %) "if (") (:kernels compilation))
          "the mixed Float/Double value conditional is emitted from shared KernelBody control"))))

(deftest public-loss-gradients-use-portable-predicate-and-signum-ssa
  (doseq [target [cuda-target hip-target]
          operation [#'loss/huber-loss-backward #'loss/l1-loss-backward]]
    (let [compilation (equation-first/compile operation {:target target :dtype :float})]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 1 (count (:kernels compilation))))
      (is (get-in compilation [:kernels 0 :attributes :kernel-body])))))

(deftest public-dense-input-gradient-retains-shape-only-array-input
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile
                       #'nn/dense-backward-dx {:target target :dtype :double})
          linked (equation-first/lower
                  compilation [(double-array [2.0 -1.0])
                               (double-array [1.0 2.0 3.0 4.0 5.0 6.0])
                               (double-array 3)])]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (seq (:kernels compilation)))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation)))
      (is (= 0 (get-in linked [:attributes :driver-allocations])))
      (is (= 1 (count (:outputs linked)))))))

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
    (let [contract (link-plan/initialization-contract fresh)]
      (is (contains? (:produces contract) prefix))
      (is (not (contains? (:produces contract) base)))
      (is (not (contains? (:requires contract) base))
          "a write-only prefix does not demand an initialized backing tail")
      (is (contains? (:writes contract) base)
          "the ABI pointer is written, but only the certified prefix is initialized"))
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

(deftest public-reassociated-rmsnorm-is-one-cooperative-c-family-artifact
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile
                       #'dl-nn/rms-norm-reassociated! {:target target :dtype :float})
          linked (equation-first/lower
                  compilation
                  [(float-array 640) (float-array 640) (float-array 640)
                   1 640 1.0e-6 1.0])
          kernel (first (:kernels compilation))]
      (is (= 1 (count (:kernels compilation))))
      (is (= module-target (:target kernel)))
      (is (= :cooperative-segmented-fold-map
             (get-in kernel [:attributes :kernel-body :attributes :kind])))
      (is (= 0 (get-in linked [:attributes :driver-allocations])))
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

(deftest public-broadcast-gradient-is-one-portable-segmented-reduction
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile
                       #'array-ops/broadcast-add-dt {:target target :dtype :double})]
      (is (= [module-target] (mapv :target (:kernels compilation))))
      (is (= 1 (count (:kernels compilation))))
      (is (= :kernel-body
             (get-in compilation [:kernels 0 :attributes :emission-route])))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-row-dots-are-portable-segmented-reductions
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]
          kernel [#'array-ops/dot-rows #'array-ops/dot-rows-dW]]
    (let [compilation (equation-first/compile kernel {:target target :dtype :double})]
      (is (= [module-target] (mapv :target (:kernels compilation))))
      (is (= 1 (count (:kernels compilation))))
      (is (= :kernel-body
             (get-in compilation [:kernels 0 :attributes :emission-route])))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-axis-reduction-is-a-portable-segmented-reduction
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile #'array-ops/reduce-axis
                                              {:target target :dtype :double})]
      (is (= [module-target] (mapv :target (:kernels compilation))))
      (is (= 1 (count (:kernels compilation))))
      (is (= :kernel-body
             (get-in compilation [:kernels 0 :attributes :emission-route])))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-indexed-dot-is-a-portable-segmented-reduction
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile #'array-ops/indexed-dot
                                              {:target target :dtype :double})]
      (is (= [module-target] (mapv :target (:kernels compilation))))
      (is (= 1 (count (:kernels compilation))))
      (is (= :kernel-body
             (get-in compilation [:kernels 0 :attributes :emission-route])))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-segment-div-z-adjoint-is-a-portable-segmented-reduction
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile #'array-ops/segment-div-dZ
                                              {:target target :dtype :double})]
      (is (= [module-target] (mapv :target (:kernels compilation))))
      (is (= 1 (count (:kernels compilation))))
      (is (= :kernel-body
             (get-in compilation [:kernels 0 :attributes :emission-route])))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-output-owned-adjoints-are-portable-segmented-reductions
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]
          kernel [#'array-ops/scatter-mul-add-d-coeffs
                  #'array-ops/flat-embed-d-values
                  #'array-ops/flat-embed-dWe
                  #'array-ops/flat-embed-dbe]]
    (let [compilation (equation-first/compile kernel {:target target :dtype :double})]
      (is (= [module-target] (mapv :target (:kernels compilation))))
      (is (= 1 (count (:kernels compilation))))
      (is (= :kernel-body
             (get-in compilation [:kernels 0 :attributes :emission-route])))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-diffusion-cumulative-product-is-a-portable-scan
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]]
    (let [compilation (equation-first/compile #'diffusion/compute-alphas-cumprod
                                              {:target target :dtype :float})]
      (is (= 3 (count (:kernels compilation))))
      (is (every? #(= module-target (:target %)) (:kernels compilation)))
      (is (every? #(= :kernel-body (get-in % [:attributes :emission-route]))
                  (:kernels compilation)))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest public-embedding-table-adjoints-use-portable-additive-scatters
  (doseq [[target module-target]
          [[cuda-target :cuda-c]
           [hip-target :hip-cpp]]
          kernel [#'array-ops/flat-embed-d-space-emb
                  #'array-ops/flat-embed-d-state-emb]]
    (let [compilation (equation-first/compile kernel {:target target :dtype :float})]
      (is (seq (:kernels compilation)))
      (is (every? #(= module-target (:target %)) (:kernels compilation)))
      (is (every? #(= :kernel-body (get-in % [:attributes :emission-route]))
                  (:kernels compilation)))
      (is (= :none (get-in compilation [:stats :fallback]))))))

(deftest product-reduction-composes-with-an-ordered-epilogue
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile
                       #'c-family-broadcast-product-map!
                       {:target target :dtype :int
                        :values {'input (av/tensor {:dtype :int :shape [64]})
                                 'weights (av/tensor {:dtype :int :shape [64]})
                                 'output (av/tensor {:dtype :int :shape [1]})}})
          linked (equation-first/lower
                  compilation [(int-array 64) (int-array 64) (int-array 1) 1])]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 2 (count (:kernels compilation))))
      (is (every? #(get-in % [:attributes :kernel-body]) (:kernels compilation)))
      (is (= [8] (get-in compilation
                         [:kernels 0 :attributes :kernel-body :launch :workgroup-size])))
      (is (= 0 (get-in linked [:attributes :driver-allocations]))))))

(deftest counted-softmax-initializers-use-the-public-c-family-boundary
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile #'attention/softmax-rows!
                                            {:target target :dtype :float})
          plan (equation-first/lower compilation [(float-array 6) 2 3])]
      (is (= 4 (count (:kernels compilation))))
      (is (every? #(< (count (:source %)) 32768) (:kernels compilation))
          "the polynomial's shared scalar spine must not expand into megabytes of source")
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 0 (get-in plan [:attributes :driver-allocations]))))))

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

(deftest coarse-fine-operators-use-the-public-c-family-boundary
  (doseq [[target module-target] [[cuda-target :cuda-c] [hip-target :hip-cpp]]
          operator [#'multilevel/prolong-constant-2d! #'multilevel/restrict-average-2d!]]
    (let [prolong? (= operator #'multilevel/prolong-constant-2d!)
          compilation (equation-first/compile operator {:target target :dtype :double})
          plan (equation-first/lower compilation
                                     [(double-array (if prolong? 60 15))
                                      (double-array (if prolong? 15 60)) 3 5])]
      (is (seq (:kernels compilation)))
      (is (every? #(= module-target (:target %)) (:kernels compilation)))
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 0 (get-in plan [:attributes :driver-allocations]))))))

(deftest declared-map-result-cast-preserves-jvm-source-semantics
  (let [output (int-array 2)]
    (checked-casts/declared-narrow-rows!
     (long-array [Integer/MIN_VALUE Integer/MAX_VALUE]) output 2)
    (is (= [Integer/MIN_VALUE Integer/MAX_VALUE] (vec output))))
  (doseq [value [(inc (long Integer/MAX_VALUE)) (dec (long Integer/MIN_VALUE))]]
    (let [input (long-array [value]) output (int-array [-7])]
      (is (thrown? ArithmeticException
                   (raster.par/map! output index 1 int (aget input index))))
      (is (= [-7] (vec output)))
      (is (thrown? ArithmeticException
                   (checked-casts/declared-narrow-rows! input output 1)))
      (is (= [-7] (vec output))))))

(deftest checked-source-narrowing-reaches-public-c-family-kernels
  (doseq [target [cuda-target hip-target]
          operation [#'checked-casts/narrow-rows!
                     #'checked-casts/declared-narrow-rows!
                     #'checked-casts/narrow-stores!
                     #'checked-casts/unused-narrow-rows!
                     #'checked-casts/annihilated-narrow-rows!]]
    (let [compilation (equation-first/compile operation
                                            {:target target :dtype :long})
          plan (equation-first/lower compilation [(long-array [-2147483648 2147483647])
                                                 (int-array 2) 2])
          nodes (mapcat #(tree-seq coll? seq (get-in % [:attributes :kernel-body :operations]))
                        (:kernels compilation))
          input-loads (set (keep #(when (and (map? %) (= 'input (:buffer %))
                                             (= :long (get-in % [:result :type])))
                                    (get-in % [:result :id])) nodes))]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (some #(str/includes? (:source %) "rstr_trap_cast_i64_i32")
                (:kernels compilation)))
      (is (some #(and (map? %) (= :cast (:op %)) (= :int (:result-type %))
                       (= {:rounding :exact :overflow :trap} (:options %))
                       (contains? input-loads (first (:arguments %)))) nodes)
          "the input load, not merely a launch-bound scalar, must feed a checked cast")
      (is (= 0 (get-in plan [:attributes :driver-allocations]))))))

(deftest unused-checked-prefix-remains-observable-before-device-work
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile #'checked-casts/checked-prefix-rows!
                                            {:target target :dtype :long})
          input (long-array [7 9])
          output (int-array [-1 -1])]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 0 (get-in (equation-first/lower compilation [input output 2 2147483647])
                       [:attributes :driver-allocations])))
      (is (thrown? ArithmeticException
                   (equation-first/lower compilation [input output 2 2147483648])))
      (is (= [-1 -1] (vec output))))))

(deftest checked-scalars-after-device-work-do-not-become-preparation-checks
  (doseq [target [cuda-target hip-target]]
    (is (= :equation-first-coverage
           (try
             (equation-first/compile #'checked-casts/checked-after-write!
                                    {:target target :dtype :int})
             :accepted
             (catch clojure.lang.ExceptionInfo exception
               (:reason (ex-data exception))))))))

(deftest padded-map-lanes-do-not-evaluate-checked-conversions
  (doseq [target [cuda-target hip-target]]
    (let [compilation (equation-first/compile #'checked-casts/narrow-index!
                                            {:target target :dtype :byte})
          kernel (first (:kernels compilation))
          operations (get-in kernel [:attributes :kernel-body :operations])
          guarded (first operations)
          nodes (tree-seq coll? seq (:operations guarded))]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 1 (count operations)))
      (is (= :map-active (:mask guarded))
          "the complete per-element region is inactive for padded work-items")
      (is (some #(and (map? %) (= :cast (:op %)) (= :byte (:result-type %))
                       (= :trap (get-in % [:options :overflow]))) nodes))
      (is (= 0 (get-in (equation-first/lower compilation [(byte-array 1) 1])
                       [:attributes :driver-allocations]))))))

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
