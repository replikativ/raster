(ns raster.compiler.passes.parallel.matrix-body-plan-test
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.walk :as walk]
            [raster.compiler.backend.gpu.kernel-body-opencl :as opencl]
            [raster.compiler.passes.parallel.matrix-body-plan :as matrix-plan]
            [raster.compiler.backend.gpu.matrix-target :as matrix-target]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.passes.parallel.mixed-matrix-body :as mixed-body]
            [raster.compiler.passes.parallel.contraction-schedule :as schedule]))

(defn- matrix-body
  [family]
  (let [matrix (case family
                 :mma {:family :mma :m 16 :n 16 :k 16 :subgroup 32}
                 {:family family :m 8 :n 16 :k 16 :subgroup 16})]
    (schedule/matrix-body
     {:id [:matrix-plan-test family]
      :row 'a :col 'b :out 'c
      :dimensions [64 64 64]
      :tile (assoc (hardware/derive-gemm-tile {}) :matrix matrix)})))

(deftest matrix-targets-cannot-ignore-a-typed-input-transformation
  (let [region (body/->ScalarSSARegion
                ['element] [] [] :half
                [(body/->ScalarCompute (body/value 'twice :half)
                                       (body/scalar-expression :+ :half ['element 'element]))]
                'twice :half)
        kernel (walk/postwalk
                (fn [node]
                  (if (instance? raster.compiler.ir.kernel_body.TileLoad node)
                    (assoc node :value-region region) node))
                (matrix-body :dpas))]
    (is (= kernel (body/validate! kernel)))
    (is (nil? (opencl/lower-uniform-store-region kernel {} :cuda))
        "an input transformation is not a store epilogue")
    (doseq [target [:opencl-intel :cuda :hip]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"transformed tile input"
                            (matrix-target/emit-matrix-kernel "transformed_input" kernel target))))))

(deftest matrix-plan-is-neutral-over-instruction-families
  (let [plan (matrix-plan/analyze (matrix-body :mma))]
    (is (= {:family :mma :m 16 :n 16 :k 16 :subgroup 32}
           (:instruction plan)))
    (is (= [128 128 32 32]
           ((juxt :block-m :block-n :sg-m :sg-n) plan)))
    (is (= [16 16 16 32]
           ((juxt :mi :ni :ki :subgroup) plan)))))

(deftest matrix-k-arithmetic-requires-widening-before-computation
  (let [kernel (matrix-body :dpas)
        narrow (walk/postwalk
                (fn [node]
                  (cond
                    (instance? raster.compiler.ir.kernel_body.IndexCast node) (:argument node)
                    (instance? raster.compiler.ir.kernel_body.ForLoop node)
                    (assoc-in node [:index :type] :int)
                    :else node)) kernel)
        late (update-in kernel [:operations 0 :operations]
                        (fn [operations]
                          (mapv (fn [op]
                                  (if (instance? raster.compiler.ir.kernel_body.ForLoop op)
                                    (assoc op :lower
                                           (body/index-cast (body/expression :add 0 0) :long :exact))
                                    op)) operations)))]
    (is (= :long (:index-dtype (matrix-plan/analyze kernel))))
    (is (body/kernel-body? (body/validate! narrow)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"requires long induction"
                          (matrix-plan/analyze narrow)))
    (is (body/kernel-body? (body/validate! late)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"widen leaves before arithmetic"
                          (matrix-plan/analyze late)))))

(deftest matrix-plan-rejects-structure-hidden-by-set-comparisons
  (let [kernel (matrix-body :dpas)]
    (testing "duplicate matrix operations cannot replace a missing pair"
      (let [bad
            (update-in
             kernel [:operations 0 :operations]
             (fn [operations]
               (mapv
                (fn [operation]
                  (if (instance? raster.compiler.ir.kernel_body.ForLoop operation)
                    (update-in
                     operation [:operations 0 :operations]
                     (fn [inner]
                       (let [mad (first (filter #(instance?
                                                 raster.compiler.ir.kernel_body.MatrixMad %)
                                               inner))]
                         (conj (pop inner) mad (peek inner)))))
                    operation))
                operations)))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"fragment product"
                              (matrix-plan/analyze bad)))))
    (testing "launch geometry is a checked consequence of the body topology"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"launch does not match"
                            (matrix-plan/analyze
                             (assoc-in kernel [:launch :workgroup-size 0] 1)))))
    (testing "an emitter rejects a physical operand permutation it cannot lower"
      (let [shape (get-in kernel [:parameters 0 :shape])]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"dense operand permutations and row-major results"
                              (matrix-plan/analyze
                               (assoc-in kernel [:parameters 0 :layout]
                                         (layout/col-major shape :half)))))))
    (testing "runtime dimension identities are specialized to the storage contract"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"body parameter or exact specialization"
                            (matrix-plan/analyze
                             (assoc-in kernel [:attributes :dimension-values 'M] 63)))))
    (testing "restrict-qualified matrix inputs require stable read contracts"
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"stable no-write-alias"
                            (matrix-plan/analyze (assoc kernel :stable-reads [])))))))

(deftest intel-opencl-retains-its-own-instruction-legality
  (testing "neutral MMA analysis does not make an Intel DPAS emitter accept MMA"
    (try
      (opencl/emit-matrix-kernel "mma_is_not_dpas" (matrix-body :mma))
      (is false "Intel OpenCL lowering unexpectedly accepted an MMA instruction")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :kernel-body-opencl-unimplemented (:reason (ex-data exception))))
        (is (= :mma (get-in (ex-data exception) [:instruction :family])))))))

(deftest direct-matrix-write-coverage-checks-the-complete-launch-and-storage
  (let [make-kernel (fn [dimensions]
                      (schedule/matrix-body
                       {:id :matrix-write-domain :row 'a :col 'b :out 'c
                        :dimensions dimensions :result-dtype :float
                        :tile (hardware/derive-gemm-tile {})}))
        kernel (make-kernel '[m n k])]
    (is (thrown? clojure.lang.ExceptionInfo (make-kernel [0 0 0]))
        "static empty storage remains outside the existing KernelBody contract")
    (doseq [dimensions [[65 79 64] '[m n k]]]
      (is (= {'c (subvec (vec dimensions) 0 2)}
             (matrix-plan/dense-result-write-domain (make-kernel dimensions)))
          "partial tiles retain logical coverage, not rounded buffer capacity"))
    (doseq [counts [[1 1]
                    (vec (reverse (get-in kernel [:launch :group-count])))
                    [(launch/ceil-div (launch/runtime-value 'N) 256)
                     (second (get-in kernel [:launch :group-count]))]]]
      (is (nil? (matrix-plan/dense-result-write-domain
                 (assoc-in kernel [:launch :group-count] counts)))
          "workgroup topology alone does not prove the complete grid"))
    (let [result (first (filter #(= :result (:role %)) (:parameters kernel)))
          view (body/->BufferView 'c-view 'c (body/index-cast 0 :long :exact)
                                 (:shape result) (:layout result))]
      (is (thrown? clojure.lang.ExceptionInfo (matrix-plan/dense-result-write-domain
                 (-> kernel
                     (assoc :views [view])
                     (assoc-in [:attributes :operation-buffers :out] 'c-view)
                     (update :operations
                             #(walk/postwalk (fn [node]
                                               (if (instance? raster.compiler.ir.kernel_body.TileStore node)
                                                 (assoc node :buffer 'c-view) node)) %)))))
          "even zero-offset result views are outside this proof"))
    (is (thrown? clojure.lang.ExceptionInfo (matrix-plan/dense-result-write-domain
               (update kernel :parameters
                       (fn [parameters]
                         (mapv #(if (= :result (:role %)) (assoc % :kind :inout) %) parameters)))))
        "read/write result permissions are not a write-only proof")
    (doseq [field [:coordinates :mask]]
      (is (thrown? clojure.lang.ExceptionInfo
                   (matrix-plan/dense-result-write-domain
                    (walk/postwalk
                     (fn [node]
                       (if (instance? raster.compiler.ir.kernel_body.TileStore node)
                         (assoc node field (if (= field :coordinates) [0 0] :tile-active)) node))
                     kernel)))
          "actual stores and masks must preserve the complete fragment partition"))))

(deftest leading-batch-writes-compose-the-verified-slice-partition
  (let [spec {:id :batch-write-domain :a 'a :b 'b :c 'c
              :m 'm :n 'n :k 'k :batch 'batch :tile (hardware/derive-gemm-tile {})}
        make-kernel (fn [batching transposed?]
                      (mixed-body/scheduled-matrix-body
                       (mixed-body/batched-matrix-spec
                        (cond-> (assoc spec :batching batching)
                          transposed? (assoc :input-layouts
                                             {'b (assoc (layout/row-major '[K N] :half)
                                                        :perm [1 0])})))))
        kernel (make-kernel {:row true :col true} false)]
    (doseq [batching [{:row true :col true} {:row false :col true} {:row true :col false}]
            transposed? [false true]]
      (is (= {'c '[batch m n]}
             (matrix-plan/leading-batch-result-write-domain (make-kernel batching transposed?)))
          "sharing or transposing inputs does not alter parent result coverage"))
    (is (nil? (matrix-plan/dense-result-write-domain kernel))
        "the ordinary 2D query cannot erase a leading batch")
    (is (thrown? clojure.lang.ExceptionInfo
                 (matrix-plan/leading-batch-result-write-domain
                  (assoc-in kernel [:launch :group-count 2] 1)))
        "the slice extent must agree with the complete group-z launch")
    (is (thrown? clojure.lang.ExceptionInfo
                 (matrix-plan/leading-batch-result-write-domain
                  (update kernel :views
                          (fn [views]
                            (mapv #(if (= 'c (:buffer %))
                                     (update % :element-offset
                                             (fn [offset] (body/expression :add offset 1))) %)
                                  views)))))
        "a shifted result slice cannot borrow the contiguous partition proof")
    (is (nil? (matrix-plan/leading-batch-result-write-domain
               (update kernel :parameters
                       (fn [parameters]
                         (mapv #(if (= :result (:role %))
                                  (assoc-in % [:layout :perm] [0 2 1]) %) parameters)))))
        "the parent result must retain its proved row-major layout")
    (let [split (mixed-body/scheduled-matrix-body
                 (mixed-body/split-k-matrix-spec (assoc spec :kc :kc :splits :splits)))]
      (is (nil? (matrix-plan/leading-batch-result-write-domain split))
          "a reduction partition is not an independent full-K batch"))))
