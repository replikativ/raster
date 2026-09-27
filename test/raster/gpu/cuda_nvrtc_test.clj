(ns raster.gpu.cuda-nvrtc-test
  (:require [clojure.test :refer [deftest is]]
            [raster.gpu.cuda-nvrtc :as nvrtc]))

(deftest compile-request-rejects-invalid-virtual-architectures-before-library-access
  (doseq [architecture ["sm_80" "compute_80 --bad-option" ""]]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"invalid NVRTC compilation request"
                          (nvrtc/compile-ptx "source" "kernel" architecture)))))

(deftest installed-nvrtc-compiles-cuda-preamble-without-a-device
  (if (nvrtc/available?)
    (let [source (str "#include <stdint.h>\n#include <math.h>\n"
                      "extern \"C\" __global__ void nvrtc_probe(uint8_t* out) {"
                      " out[0] = (uint8_t)1; }")
          {:keys [ptx nvrtc-version]} (nvrtc/compile-ptx source "nvrtc_probe" "compute_80")]
      (is (pos? (alength ptx)))
      (is (= (nvrtc/version) nvrtc-version))
      (is (re-find #"nvrtc_probe" (String. ^bytes ptx java.nio.charset.StandardCharsets/UTF_8)))
      (let [error (try (nvrtc/compile-ptx "not valid CUDA C++" "bad_probe" "compute_80")
                       nil
                       (catch clojure.lang.ExceptionInfo e e))]
        (is (= :nvrtc-compilation (:reason (ex-data error))))
        (is (seq (:log (ex-data error))))))
    (println "[NVRTC SKIP] CUDA runtime compilation library is not installed")))
