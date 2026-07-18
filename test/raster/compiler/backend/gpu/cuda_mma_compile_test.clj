(ns raster.compiler.backend.gpu.cuda-mma-compile-test
  "T5 (NVIDIA :mma fork): the CUDA WMMA GEMM generator COMPILES and SELECTS Tensor Cores. Gated on a
   present nvcc toolchain (compiles CUDA→SASS with NO GPU needed). This is the strongest structural
   gate available without an NVIDIA GPU — it catches the 'garbage kernel' failure mode (invalid code
   / wrong instruction selection). NUMERICAL correctness needs a real GPU and is deferred (the doc in
   .internal/cross_vendor_matrix_fork.md tracks the on-device plan + the AMD MFMA / WGMMA families)."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.java.shell :as sh]
            [clojure.java.io :as io]
            [raster.compiler.core.hardware :as hw]
            [raster.compiler.backend.gpu.cuda-codegen :as cuda]))

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
              src  (apply cuda/emit-gemm-tiled-cuda "gemm_mma_derived"
                          (mapcat identity (select-keys tile [:block-m :block-n :sg-m :sg-n :block-k :matrix])))
              {:keys [compiled? sass err]} (compile-cu src)]
          (is compiled? (str "nvcc must accept the emitted kernel; stderr:\n" err))
          (is (re-find #"HMMA" (or sass "")) "SASS must contain HMMA (Tensor Core) instructions")))
      (testing "a different tile also compiles + selects Tensor Cores (parametricity holds through nvcc)"
        (let [src (cuda/emit-gemm-tiled-cuda "gemm_mma_big"
                                             :block-m 128 :block-n 128 :sg-m 64 :sg-n 64 :block-k 32)
              {:keys [compiled? sass err]} (compile-cu src)]
          (is compiled? (str "big tile must compile; stderr:\n" err))
          (is (re-find #"HMMA" (or sass "")) "big tile SASS has HMMA")))
      (testing "the generator rejects a non-divisible tile (fail-loud, like the Intel emitter)"
        (is (thrown? clojure.lang.ExceptionInfo
                     (cuda/emit-gemm-tiled-cuda "bad" :sg-m 24)))))))
