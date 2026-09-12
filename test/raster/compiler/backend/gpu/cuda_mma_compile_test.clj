(ns raster.compiler.backend.gpu.cuda-mma-compile-test
  "T5 (NVIDIA :mma fork): the CUDA WMMA GEMM generator COMPILES and SELECTS Tensor Cores. Gated on a
   present nvcc toolchain (compiles CUDA→SASS with NO GPU needed). This is the strongest structural
   gate available without an NVIDIA GPU — it catches the 'garbage kernel' failure mode (invalid code
   / wrong instruction selection). NUMERICAL correctness needs a real GPU and is deferred (the doc in
   .internal/cross_vendor_matrix_fork.md tracks the on-device plan + the AMD MFMA / WGMMA families)."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.java.shell :as sh]
            [clojure.java.io :as io]
            [clojure.walk :as walk]
            [raster.compiler.core.hardware :as hw]
            [raster.compiler.backend.gpu.cuda-codegen :as cuda]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.passes.parallel.contraction-schedule :as schedule]))

(defn- nvcc-available? []
  (try (zero? (:exit (sh/sh "bash" "-c" "command -v nvcc"))) (catch Exception _ false)))

(defn- compile-cu
  "Compile CUDA source to a cubin for sm_80, then dump SASS. Returns {:compiled? :sass}."
  [src]
  (let [dir (io/file (System/getProperty "java.io.tmpdir") (str "raster-cuda-" (System/nanoTime)))]
    (.mkdirs dir)
    (let [cu (io/file dir "k.cu") cubin (io/file dir "k.cubin")]
      (spit cu src)
      (let [c (sh/sh "nvcc" "-arch=sm_80" "-cubin" "-o" (.getPath cubin) (.getPath cu))]
        (if-not (zero? (:exit c))
          {:compiled? false :err (:err c)}
          (let [s (sh/sh "cuobjdump" "-sass" (.getPath cubin))]
            {:compiled? true :sass (:out s)}))))))

(defn- matrix-body
  [id tile dimensions]
  (schedule/matrix-body
   {:id id :row 'a :col 'b :out 'c
    :dimensions dimensions :tile tile :result-dtype :float}))

(deftest cuda-wmma-derives-and-compiles-with-tensor-cores
  (if-not (nvcc-available?)
    (println "SKIP cuda-mma: no nvcc toolchain")
    (let [a100 {:device-type :gpu :device-id :cuda:0 :compute-capability [8 0]
                :grf-bytes-per-lane 256 :subgroup-size 32
                :matrix {:family :mma :m 16 :n 16 :k 16 :subgroup 32}}]
      (testing "an NVIDIA descriptor derives a WMMA-shaped tile"
        (let [tile (hw/derive-gemm-tile a100)]
          (is (= :mma (get-in tile [:matrix :family])))
          (is (zero? (mod (:sg-m tile) 16)) "warp-tile M is a 16-fragment multiple")
          (is (zero? (mod (:sg-n tile) 16)) "warp-tile N is a 16-fragment multiple")))
      (testing "the derived tile emits CUDA that nvcc compiles + lowers to HMMA Tensor Cores"
        (let [tile (hw/derive-gemm-tile a100)
              dimensions [(:block-m tile) (:block-n tile) (* 2 (:block-k tile))]
              src (cuda/emit-matrix-kernel
                   "gemm_mma_derived" (matrix-body :gemm-mma-derived tile dimensions))
              {:keys [compiled? sass err]} (compile-cu src)]
          (is compiled? (str "nvcc must accept the emitted kernel; stderr:\n" err))
          (is (re-find #"if \(M != [0-9]+ \|\| N != [0-9]+ \|\| K != [0-9]+\)"
                       src)
              "runtime dimensions are guarded by the checked body specialization")
          (is (re-find #"HMMA" (or sass "")) "SASS must contain HMMA (Tensor Core) instructions")))
      (testing "a different tile also compiles + selects Tensor Cores (parametricity holds through nvcc)"
        (let [tile {:block-m 128 :block-n 128 :sg-m 64 :sg-n 64 :block-k 32
                    :num-stages 3 :matrix (:matrix a100)}
              src (cuda/emit-matrix-kernel
                   "gemm_mma_big" (matrix-body :gemm-mma-big tile [128 128 64]))
              {:keys [compiled? sass err]} (compile-cu src)]
          (is compiled? (str "big tile must compile; stderr:\n" err))
          (is (re-find #"HMMA" (or sass "")) "big tile SASS has HMMA")))
      (testing "the emitter rejects a schedule that could overread its final K block"
        (let [tile {:block-m 64 :block-n 64 :sg-m 32 :sg-n 32 :block-k 32
                    :num-stages 3 :matrix (:matrix a100)}]
          (try
            (cuda/emit-matrix-kernel "bad_k" (matrix-body :bad-k tile [64 64 16]))
            (is false "CUDA matrix lowering admitted K smaller than its scheduled K block")
            (catch clojure.lang.ExceptionInfo exception
              (is (= :cuda-mma-requires-aligned-static-dimensions
                     (:reason (ex-data exception))))))))
      (testing "the emitter rejects result-store semantics it cannot preserve"
        (let [tile (hw/derive-gemm-tile a100)
              dimensions [(:block-m tile) (:block-n tile) (:block-k tile)]
              half-body (schedule/matrix-body
                         {:id :half-result :row 'a :col 'b :out 'c
                          :dimensions dimensions :tile tile :result-dtype :half})]
          (try
            (cuda/emit-matrix-kernel "half_result" half-body)
            (is false "CUDA matrix lowering silently narrowed an unsupported result")
            (catch clojure.lang.ExceptionInfo exception
              (is (= :cuda-mma-result-dtype-unsupported
                     (:reason (ex-data exception)))))))))))

(deftest cuda-uniform-epilogue-uses-the-existing-store-region
  (let [tile {:block-m 64 :block-n 64 :sg-m 32 :sg-n 32 :block-k 32
              :num-stages 3 :matrix {:family :mma :m 16 :n 16 :k 16 :subgroup 32}}
        body (schedule/matrix-body
              {:id :uniform-epilogue :row 'a :col 'b :out 'c
               :dimensions [64 64 64] :tile tile :result-dtype :float
               :epilogue {:acc 'acc
                          :expr '(raster.numeric/max (raster.numeric/* acc alpha) (float 0.0))
                          :scalars [{:sym 'alpha :dtype :float}]}})
        source (cuda/emit-matrix-kernel "uniform_epilogue" body)]
    (is (re-find #"float alpha" source))
    (is (re-find #"\.x\[rstr_epilogue_element\] = .*alpha" source))
    (is (re-find #"fmax\(" source))
    (is (not (re-find #"__shared__" source))
        "a coordinate-free transform needs no hidden scratch")
    (doseq [id ['rstr_epilogue_element 'n_base 'warpId]]
      (try
        (cuda/emit-matrix-kernel "collision"
                                (walk/postwalk-replace {'alpha id} body))
        (is false "direct CUDA emission must reject scalar/generated-name collisions")
        (catch clojure.lang.ExceptionInfo e
          (is (= :matrix-target-name-collision (:reason (ex-data e)))))))
    (when (nvcc-available?)
      (let [{:keys [compiled? sass err]} (compile-cu source)]
        (is compiled? err)
        (is (re-find #"HMMA" (or sass "")))))))

(deftest cuda-opaque-fragments-reject-coordinate-dependent-epilogues
  (let [tile {:block-m 64 :block-n 64 :sg-m 32 :sg-n 32 :block-k 32
              :num-stages 3 :matrix {:family :mma :m 16 :n 16 :k 16 :subgroup 32}}
        body (schedule/matrix-body
              {:id :coordinate-epilogue :row 'a :col 'b :out 'c
               :dimensions [64 64 64] :tile tile :result-dtype :float
               :epilogue {:acc 'acc
                          :expr '(raster.numeric/+ acc (clojure.core/aget bias j))
                          :operands [{:sym 'bias :dtype :float
                                      :map (axis-map/of-axes [['j 64]])}]}})]
    (try
      (cuda/emit-matrix-kernel "coordinate_epilogue" body)
      (is false "logical row identity cannot be recovered from opaque fragment slots")
      (catch clojure.lang.ExceptionInfo e
        (is (= :matrix-store-region-requires-coordinate-layout (:reason (ex-data e))))))))

(deftest cuda-signature-follows-the-ordered-kernel-body-abi
  (let [matrix {:family :mma :m 16 :n 16 :k 16 :subgroup 32}
        tile {:block-m 64 :block-n 64 :sg-m 32 :sg-n 32 :block-k 32
              :num-stages 3 :matrix matrix}
        kernel (matrix-body :ordered-abi tile [64 64 64])
        [a b & tail] (:parameters kernel)
        reordered (assoc kernel :parameters (vec (concat [b a] tail)))
        source (cuda/emit-matrix-kernel "ordered_abi" reordered)
        signature (subs source (.indexOf source "extern \"C\"") (.indexOf source ") {"))]
    (is (< (.indexOf signature " B") (.indexOf signature " A"))
        "target parameter declarations preserve KernelBody ABI order")))
