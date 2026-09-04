(ns raster.compiler.backend.gpu.matrix-target-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.gemm :as gemm]
            [raster.compiler.backend.gpu.matrix-target :as matrix-target]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.passes.parallel.contraction-schedule :as schedule]))

(def ^:private mma
  {:family :mma :m 16 :n 16 :k 16 :subgroup 32})

(defn- mma-tile []
  (assoc (hardware/derive-gemm-tile
          {:device-type :gpu :backend :cuda :compute-capability [8 0]
           :subgroup-size 32 :grf-bytes-per-lane 256 :matrix mma})
         :matrix mma))

(defn- mma-body []
  (schedule/matrix-body
   {:id :matrix-target-test :row 'a :col 'b :out 'c
    :dimensions [128 128 64] :dimension-parameters ['m 'n 'k]
    :tile (mma-tile) :result-dtype :float}))

(deftest target-lowering-forks-after-one-verified-body
  (let [body (mma-body)
        cuda (matrix-target/emit-matrix-kernel "matrix_target_cuda" body :cuda)]
    (is (= :cuda-c (:target cuda)))
    (is (= :cuda (:target-dialect cuda)))
    (is (identical? body (:kernel-body cuda)))
    (is (re-find #"wmma::mma_sync" (:source cuda)))
    (testing "an instruction cannot leak into an unrelated target spelling"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"no builtin"
           (matrix-target/emit-matrix-kernel "mma_as_dpas" body :opencl-intel))))
    (testing "unimplemented families decline at the single target boundary"
      (try
        (matrix-target/emit-matrix-kernel "mma_as_mfma" body :hip)
        (is false "HIP matrix target unexpectedly accepted an unimplemented MFMA row")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :matrix-target-dialect-not-lowered (:reason (ex-data exception)))))))))

(deftest scheduled-matrix-entry-no-longer-implies-opencl
  (let [emitted
        (gemm/emit-scheduled-matrix-kernel
         {:kernel-name "scheduled_cuda_mma"
          :a 'a :b 'b :c 'c
          :m 128 :n 128 :k 64
          :dimension-parameters ['m 'n 'k]
          :tile (mma-tile)
          :result-dtype :float
          :target-dialect :cuda})]
    (is (= :cuda-c (:target emitted)))
    (is (= :cuda (:target-dialect emitted)))
    (is (= ['a 'b 'c 'm 'n 'k]
           (mapv :id (get-in emitted [:kernel-body :parameters]))))
    (is (re-find #"extern \"C\" __global__" (:source emitted)))
    (is (not (re-find #"__kernel" (:source emitted))))))
