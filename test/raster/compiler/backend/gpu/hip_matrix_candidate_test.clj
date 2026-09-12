(ns raster.compiler.backend.gpu.hip-matrix-candidate-test
  "Source-only MFMA candidate coverage; mandatory architecture compilation runs in CI."
  (:require [clojure.test :refer [deftest is]]
            [clojure.java.io :as io]
            [clojure.java.shell :as sh]
            [raster.compiler.backend.gpu.matrix-fragment-source :as fragment]
            [raster.compiler.backend.gpu.matrix-target :as target]
            [raster.compiler.passes.parallel.contraction-schedule :as schedule]))

(defn candidate-body
  "The same scheduled candidate is emitted by the mandatory HIP matrix compile fixture."
  ([] (candidate-body {:family :mfma :m 16 :n 16 :k 16 :subgroup 64}))
  ([instruction]
   (schedule/matrix-body
    {:id :hip-mfma-candidate :row 'a :col 'b :out 'c
     :dimensions [64 64 64] :result-dtype :float
     :tile {:block-m 64 :block-n 64 :sg-m 32 :sg-n 32 :block-k 32
            :num-stages 3 :matrix instruction}
     :epilogue {:acc 'acc
                :expr '(raster.numeric/max (raster.numeric/* acc alpha) (float 0.0))
                :scalars [{:sym 'alpha :dtype :float}]}})))

(deftest mfma-candidate-retains-the-typed-body-and-uniform-epilogue
  (let [body (candidate-body)
        source (fragment/emit-matrix-kernel "mfma_candidate" body :hip)]
    (is (re-find #"rocwmma::mma_sync" source))
    (is (re-find #"threadIdx.x / 64" source))
    (is (re-find #"const rocwmma::float16_t\* __restrict__ A" source))
    (is (re-find #"float alpha" source))
    (is (re-find #"\.x\[rstr_epilogue_element\] = .*alpha" source))
    (is (re-find #"fmax\(" source))
    (is (re-find #"!defined\(__gfx90a__\)" source))
    (is (not (re-find #"__shared__|nvcuda|wmma::fragment<wmma::" source)))
    (try
      (target/emit-matrix-kernel "not_runtime_ready" body :hip)
      (is false "a candidate requiring external headers must not become a runtime artifact")
      (catch clojure.lang.ExceptionInfo e
        (is (= :matrix-target-dialect-not-lowered (:reason (ex-data e))))))
    (doseq [instruction [{:family :mma :m 16 :n 16 :k 16 :subgroup 32}
                         {:family :mfma :m 16 :n 16 :k 16 :subgroup 32}]]
      (try
        (fragment/emit-matrix-kernel "bad_instruction" (candidate-body instruction) :hip)
        (is false "target family and wave width must match the admitted MFMA instruction")
        (catch clojure.lang.ExceptionInfo e
          (is (= :hip-mfma-instruction-unsupported (:reason (ex-data e)))))))))

(deftest pinned-mfma-candidate-compiles-without-amd-hardware
  (if-let [include (System/getenv "RASTER_ROCWMMA_INCLUDE")]
    (let [dir (.toFile (java.nio.file.Files/createTempDirectory
                       "raster-mfma-" (make-array java.nio.file.attribute.FileAttribute 0)))
          source (io/file dir "candidate.hip")
          output (io/file dir "candidate.hsaco")]
      (try
        (spit source (fragment/emit-matrix-kernel "mfma_candidate" (candidate-body) :hip))
        (let [compile! (fn [arch]
                         (sh/sh "hipcc" "-D__HIP_PLATFORM_AMD__" "-std=c++17"
                                (str "--offload-arch=" arch) "--genco" "-I" include
                                (.getPath source) "-o" (.getPath output)))
              accepted (compile! "gfx90a")
              rejected (compile! "gfx1100")]
          (is (zero? (:exit accepted)) (:err accepted))
          (is (not (zero? (:exit rejected))))
          (is (re-find #"Raster MFMA candidate requires gfx90a" (:err rejected))))
        (finally
          (doseq [file (reverse (file-seq dir))] (io/delete-file file)))))
    (println "HIP matrix local compile not run: set RASTER_ROCWMMA_INCLUDE; mandatory CI covers it.")))
