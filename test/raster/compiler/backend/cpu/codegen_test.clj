(ns raster.compiler.backend.cpu.codegen-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.cpu.codegen :as cpu]))

(deftest native-source-cache-key-test
  (let [key-fn (ns-resolve 'raster.compiler.backend.cpu.codegen 'source-cache-key)
        identity-var (ns-resolve 'raster.compiler.backend.cpu.codegen 'compiler-identity)
        source "void k(float *x) { x[0] = 1.0f; }"
        key (key-fn source)]
    (testing "keys are deterministic full SHA-256 values and source-sensitive"
      (is (= key (key-fn source)))
      (is (re-matches #"[0-9a-f]{64}" key))
      (is (not= key (key-fn (str source "\n")))))
    (testing "toolchain identity participates in the artifact key"
      (is (not= key
                (with-redefs-fn {identity-var (delay {:schema :test/different-toolchain})}
                  #(key-fn source)))))))
