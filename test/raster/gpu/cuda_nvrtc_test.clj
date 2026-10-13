(ns raster.gpu.cuda-nvrtc-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-abi :as abi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.gpu.cuda-nvrtc :as nvrtc]))

(def ^:private smoke-artifact
  (artifact/make
   {:kernel-name "nvrtc_probe"
    :target :cuda-c
    :source (str "#include <stdint.h>\n#include <math.h>\n"
                 "extern \"C\" __global__ void nvrtc_probe(uint8_t* out) {"
                 " out[0] = (uint8_t)1; }")
    :abi [(abi/slot 'out :output :byte :role :result)] :arguments '[out]
    :launch (launch/spec {:workgroup-size [1] :group-count [1]})}))

(deftest compile-request-rejects-invalid-virtual-architectures-before-library-access
  (doseq [architecture ["sm_80" "compute_80 --bad-option" ""]]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"invalid NVRTC compilation request"
                          (nvrtc/compile-ptx "source" "kernel" architecture)))))

(deftest compile-request-rejects-malformed-includes-before-library-access
  (doseq [includes ["/include" [nil] [""] [23]]]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"invalid NVRTC compilation request"
                          (nvrtc/compile-ptx "source" "kernel" "compute_80"
                                             :include-paths includes)))))

(deftest installed-nvrtc-compiles-cuda-preamble-without-a-device
  (if (nvrtc/available?)
    (let [{:keys [ptx nvrtc-version target virtual-header-version]}
          (nvrtc/compile-artifact-ptx smoke-artifact "compute_80")]
      (is (pos? (alength ptx)))
      (is (= :cuda-c target))
      (is (= nvrtc/virtual-header-version virtual-header-version))
      (is (= (nvrtc/version) nvrtc-version))
      (is (re-find #"nvrtc_probe" (String. ^bytes ptx java.nio.charset.StandardCharsets/UTF_8)))
      (let [error (try (nvrtc/compile-ptx "not valid CUDA C++" "bad_probe" "compute_80")
                       nil
                       (catch clojure.lang.ExceptionInfo e e))]
        (is (= :nvrtc-compilation (:reason (ex-data error))))
        (is (seq (:log (ex-data error))))))
    (if (= "1" (System/getenv "RASTER_REQUIRE_NVRTC"))
      (is false "NVRTC is required for this test run but is not installed")
      (println "[NVRTC SKIP] CUDA runtime compilation library is not installed"))))

(deftest runtime-facing-entry-rejects-non-cuda-artifacts
  (let [error (try (nvrtc/compile-artifact-ptx
                    (assoc smoke-artifact :target :opencl-c
                           :source "__kernel void nvrtc_probe(__global char* out) { out[0] = 1; }")
                    "compute_80")
                   nil
                   (catch clojure.lang.ExceptionInfo e e))]
    (is (= :nvrtc-artifact-target (:reason (ex-data error))))))

(deftest executable-options-retain-scalar-rounding-and-complete-identity
  (if (nvrtc/available?)
    (let [expected ["--gpu-architecture=compute_80" "--std=c++17" "--fmad=false"]
          raw (nvrtc/compile-ptx (:source smoke-artifact) "nvrtc_probe" "compute_80")
          exact (nvrtc/compile-artifact-ptx
                 (assoc-in smoke-artifact [:attributes :numerics]
                           {:mode :exact :policy :same-scalar-evaluation-order}) "compute_80")
          reassociated (nvrtc/compile-artifact-ptx
                        (assoc-in smoke-artifact [:attributes :numerics]
                                  {:mode :reassociated :policy :ordered-k-target-contraction
                                   :rounding :nearest-even :accumulator-dtype :float
                                   :result-transform {:kind :typed-scalar-region
                                                      :policy :same-typed-ssa-evaluation-order
                                                      :input-dtype :float :result-dtype :float}})
                        "compute_80")
          bounded (nvrtc/compile-artifact-ptx
                   (assoc-in smoke-artifact [:attributes :numerics]
                             {:mode :bounded-error :policy :matrix-approximation
                              :rounding :nearest-even :accumulator-dtype :float
                              :error-model {:kind :test-fixture}
                              :result-transform {:kind :typed-scalar-region
                                                 :policy :same-typed-ssa-evaluation-order
                                                 :input-dtype :float :result-dtype :float}})
                   "compute_80")]
      (doseq [result [raw exact reassociated bounded]]
        (is (= expected (:compiler-options result)))
        (is (= [] (:include-paths result)))
        (is (= nvrtc/virtual-header-version (:virtual-header-version result))))
      (let [included (nvrtc/compile-artifact-ptx smoke-artifact "compute_80"
                                               :include-paths ["/reference/include"])]
        (is (= (conj expected "--include-path=/reference/include") (:compiler-options included)))
        (is (not= (:compiler-options raw) (:compiler-options included)))))
    (when (= "1" (System/getenv "RASTER_REQUIRE_NVRTC"))
      (is false "NVRTC is required for executable option qualification"))))

(deftest implicit-contraction-is-disabled-without-disabling-explicit-fma
  (if (nvrtc/available?)
    (let [source (str "extern \"C\" __global__ void rounding_probe(float* out, const float* x) {"
                      " out[0] = x[0]*x[1]+x[2];"
                      " out[1] = __fmaf_rn(x[0],x[1],x[2]); }")
          result (nvrtc/compile-ptx source "rounding_probe" "compute_80")
          ptx (String. ^bytes (:ptx result) java.nio.charset.StandardCharsets/UTF_8)]
      ;; Input loads keep this independent of constant folding. Check the compiler's
      ;; actual output, not just its option list. This remains a source-only oracle.
      (is (= 1 (count (re-seq #"\bmul\.rn\.f32\b" ptx))))
      (is (= 1 (count (re-seq #"\badd\.rn\.f32\b" ptx))))
      (is (= 1 (count (re-seq #"\bfma\.rn\.f32\b" ptx)))))
    (when (= "1" (System/getenv "RASTER_REQUIRE_NVRTC"))
      (is false "NVRTC is required for instruction-level rounding qualification"))))
