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

(deftest installed-nvrtc-compiles-cuda-preamble-without-a-device
  (if (nvrtc/available?)
    (let [{:keys [ptx nvrtc-version target]} (nvrtc/compile-artifact-ptx
                                             smoke-artifact "compute_80")]
      (is (pos? (alength ptx)))
      (is (= :cuda-c target))
      (is (= (nvrtc/version) nvrtc-version))
      (is (re-find #"nvrtc_probe" (String. ^bytes ptx java.nio.charset.StandardCharsets/UTF_8)))
      (let [error (try (nvrtc/compile-ptx "not valid CUDA C++" "bad_probe" "compute_80")
                       nil
                       (catch clojure.lang.ExceptionInfo e e))]
        (is (= :nvrtc-compilation (:reason (ex-data error))))
        (is (seq (:log (ex-data error))))))
    (println "[NVRTC SKIP] CUDA runtime compilation library is not installed")))

(deftest runtime-facing-entry-rejects-non-cuda-artifacts
  (let [error (try (nvrtc/compile-artifact-ptx
                    (assoc smoke-artifact :target :opencl-c
                           :source "__kernel void nvrtc_probe(__global char* out) { out[0] = 1; }")
                    "compute_80")
                   nil
                   (catch clojure.lang.ExceptionInfo e e))]
    (is (= :nvrtc-artifact-target (:reason (ex-data error))))))
