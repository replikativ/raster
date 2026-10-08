(ns raster.compiler.backend.gpu.hip-matrix-candidate-test
  "Verified MFMA artifact coverage; mandatory architecture compilation runs in CI."
  (:require [clojure.test :refer [deftest is]]
            [clojure.java.io :as io]
            [clojure.java.shell :as sh]
            [raster.compiler.backend.gpu.kernel-body-target :as body-target]
            [raster.compiler.backend.gpu.matrix-fragment-source :as fragment]
            [raster.compiler.backend.gpu.matrix-target :as target]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]
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
    (is (re-find #"__device__ __forceinline__ float rstr_mul_f32_noncontract" source))
    (is (re-find #"__asm__\(\"\" : \"\+v\"\(product\)\)" source)
        "the shared typed scalar epilogue retains its product rounding boundary")
    (is (re-find #"!defined\(__gfx90a__\)" source))
    (is (not (re-find #"__shared__|nvcuda|wmma::fragment<wmma::" source)))
    (let [emitted (target/emit-matrix-kernel "mfma_candidate" body :hip)]
      (is (= :hip-cpp (:target emitted)))
      (is (= source (:source emitted)))
      (is (= :gfx90a (get-in emitted [:target-facts :required-gfx-arch])))
      (is (= 16 (get-in emitted [:parameter-alignments 'a]))))
    (doseq [instruction [{:family :mma :m 16 :n 16 :k 16 :subgroup 32}
                         {:family :mfma :m 16 :n 16 :k 16 :subgroup 32}]]
      (try
        (fragment/emit-matrix-kernel "bad_instruction" (candidate-body instruction) :hip)
        (is false "target family and wave width must match the admitted MFMA instruction")
        (catch clojure.lang.ExceptionInfo e
          (is (= :hip-mfma-instruction-unsupported (:reason (ex-data e)))))))))

(deftest mfma-source-is-a-verified-scheduled-artifact
  (let [body (candidate-body)
        arguments (mapv :id (:parameters body))
        scheduled (scheduled-body/make
                   {:source :hip-mfma-candidate :body body :arguments arguments
                    :effects {:kind :tensor-contraction-stage
                              :uses (scheduled-body/derive-uses body arguments)}
                    :legality {:kind :matrix-instruction-tiling}
                    :numerics {:mode :reassociated :policy :tiled-contraction
                               :rounding :nearest-even :accumulator-dtype :float}})
        emitted (body-target/emit-artifact "mfma_artifact" scheduled :hip)]
    (is (artifact/kernel-artifact? emitted))
    (is (= :hip-cpp (:target emitted)))
    (is (= :gfx90a (get-in emitted [:attributes :target-facts :required-gfx-arch])))
    (is (= [16 16 16] (mapv :alignment (take 3 (:abi emitted)))))
    (is (= ['a 'b 'c] (mapv :id (take 3 (:parameters body)))))
    (is (re-find #"rocwmma::mma_sync" (:source emitted)))
    (let [downgraded (-> emitted
                         (update :abi #(mapv (fn [slot] (dissoc slot :alignment)) %))
                         (update-in [:attributes :target-facts]
                                    dissoc :pointer-alignment :required-gfx-arch)
                         (update-in [:provenance :target-facts]
                                    dissoc :pointer-alignment :required-gfx-arch))]
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"requires gfx90a"
           (scheduled-body/validate-artifact-projection! scheduled downgraded))))))

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
