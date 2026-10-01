(ns raster.compiler.public-mixed-aggregate-test
  (:require [clojure.test :refer [deftest is]]
            [raster.linalg.sparse :as sparse]
            [raster.compiler.equation-first :as equation]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.passes.scalar.soa-lower :as soa]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.value :as value]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]))

(defn- matrix [entries]
  (sparse/->CSRMatrix (int-array [0 2 3]) (int-array [0 2 1])
                      (double-array entries) 2 3 3))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest mixed-record-projection-retains-scalar-and-buffer-contracts
  (let [compilation (equation/compile #'sparse/spmv {:target :cuda:0 :dtype :double})
        plan (get-in compilation [:semantic :attributes :invocation-plan])
        projection (get-in plan [:attributes :parameter-projection])
        A (matrix [3 -1 4]) x (double-array [1 2 3]) y (double-array [5 6])
        args (materialization/parameter-arguments plan [A x y 2.0 0.5])
        representation (equation/parameter-representation #'sparse/spmv :double)]
    (is (= '[A x y alpha beta] (:public-parameters projection)))
    (is (= [:rowptr :colidx :values :nrows :ncols :nnz]
           (mapv :field (take 6 (:physical-parameters projection)))))
    (is (identical? (.-rowptr A) (first args)))
    (is (identical? (.-values A) (nth args 2)))
    (is (= [2 3 3] (subvec args 3 6)))
    (is (identical? x (nth args 6)))
    (is (= :none (get-in compilation [:stats :fallback])))
    (is (= :kernel-body (first (keys (get-in compilation [:stats :emission :emission-routes])))))
    (is (nil? (get (soa/soa-param-env (:params representation)) 'A))
        "the old resident binder does not silently acquire mixed-record semantics")
    (is (= :invocation-parameter-projection
           (reason #(invocation/validate!
                     (assoc-in plan [:attributes :parameter-projection :physical-parameters 3 :tag]
                               'double)))))
    (is (= :invocation-materialization-aggregate-class
           (reason #(materialization/parameter-arguments plan [{} x y 2.0 0.5]))))))

(defn- run-device [target]
  (let [A (matrix [3 -1 4]) x (double-array [1 2 3]) y (double-array [5 6])
        oracle (vec (sparse/spmv A x (aclone y) 2.0 0.5))
        prepared (compiled/lower #'sparse/spmv [A x y 2.0 0.5]
                                 {:target target :dtype :double :compiler :equation-first
                                  :donate '[y]})
        program (compiled/instantiate! prepared)]
    (try
      (is (= 0 (get-in (compiled/plan prepared) [:attributes :driver-allocations])))
      (is (= #{[:A :rowptr] [:A :colidx] [:A :values] :x :y}
             (set (map :key (:in-tree prepared)))))
      (is (= oracle (vec (value/->host (get (program {}) :y')))))
      (doseq [[field changed]
              [[:nrows (sparse/->CSRMatrix (.-rowptr A) (.-colidx A) (.-values A) 3 3 3)]
               [:ncols (sparse/->CSRMatrix (.-rowptr A) (.-colidx A) (.-values A) 2 4 3)]
               [:nnz (sparse/->CSRMatrix (.-rowptr A) (.-colidx A) (.-values A) 2 3 4)]]]
        (is (= :compiled-aggregate-scalar-change (reason #(program {:A changed}))) (str field)))
      ;; The rejected calls above must neither run the kernel nor change state.
      (let [changed (matrix [1 2 3])
            next-oracle (vec (sparse/spmv changed x (double-array oracle) 2.0 0.5))]
        (is (= next-oracle (vec (value/->host (get (program {:A changed}) :y'))))))
      (let [larger (sparse/->CSRMatrix (int-array [0 1 2 3]) (int-array [0 2 1])
                                      (double-array [3 -1 4]) 3 3 3)
            initial (double-array [5 6 7])
            expected (vec (sparse/spmv larger x (aclone initial) 2.0 0.5))
            fresh (compiled/instantiate!
                   (compiled/lower #'sparse/spmv [larger x initial 2.0 0.5]
                                   {:target target :dtype :double :compiler :equation-first
                                    :donate '[y]}))]
        (try
          (is (= expected (vec (value/->host (get (fresh {}) :y'))))
              "preparing again admits changed scalar dimensions and the new launch shape")
          (finally (compiled/close! fresh))))
      (finally (compiled/close! program)))))

(deftest mixed-record-on-opencl
  (if @opencl/opencl-available? (run-device :ocl:0)
      (opencl/opencl-skip! "mixed-record-opencl")))

(deftest mixed-record-on-level-zero
  (if @ze/gpu-available? (run-device :ze:0)
      (ze/gpu-skip! "mixed-record-level-zero")))
