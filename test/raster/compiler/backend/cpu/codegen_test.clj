(ns raster.compiler.backend.cpu.codegen-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.cpu.codegen :as cpu]
            [raster.compiler.backend.intrinsics :as intrinsics]))

(deftest native-extrema-evaluate-operands-once
  (doseq [op [:min :max]
          :let [name (get-in intrinsics/table [op :native-c :fn])
                body (str "void extrema_once(float *x, float *out) {\n"
                          "out[0] = " name "(x[0]++, x[1]++);\n"
                          "out[1] = x[0]; out[2] = x[1];\n}\n")
                source (str "#include <math.h>\n"
                            (intrinsics/native-c-helper-sources body) body)
                native (cpu/load-kernel (cpu/compile-source! source) "extrema_once" 2 [])
                x (float-array [-2.0 -3.0])
                out (float-array 3)]]
    (native x out)
    (is (= [(if (= :min op) -3.0 -2.0) -1.0 -2.0] (vec out)))))

(deftest native-extrema-preserve-integral-operands
  (doseq [op ['Math/min 'Math/max]
          :let [native (cpu/compile-elementwise "extrema_long" :long (list op 'x 'y) 2)
                a (long-array [Long/MAX_VALUE Long/MIN_VALUE 9007199254740993])
                b (long-array [(dec Long/MAX_VALUE) (inc Long/MIN_VALUE) 9007199254740992])
                out (long-array 3)]]
    (native a b out 3)
    (is (= (mapv (if (= op 'Math/min) min max) a b) (vec out)))))

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

(deftest native-default-preserves-floating-point-operations
  ;; Runtime inputs prevent constant folding from hiding permissive compiler
  ;; flags. This exercises the actual compile/load boundary, not a flags regex.
  (let [source (str "#include <math.h>\n"
                    "void fp_semantics(const float *x, float *out) {\n"
                    " out[0] = isnan(x[0]) ? 1.0f : 0.0f;\n"
                    " out[1] = x[1] * 0.0f;\n"
                    " out[2] = x[2] * x[2] + x[3];\n"
                    "}\n")
        native (cpu/load-kernel (cpu/compile-source! source) "fp_semantics" 2 [])
        x (float-array [Float/NaN -1.0 1.0000001192092896 -1.000000238418579])
        out (float-array 3)]
    (native x out)
    (testing "NaN tests remain observable"
      (is (= 1.0 (aget out 0))))
    (testing "negative zero remains observable"
      (is (= (Float/floatToRawIntBits -0.0)
             (Float/floatToRawIntBits (aget out 1)))))
    (testing "separate multiply and add do not become an implicit FMA"
      (is (= 0.0 (aget out 2)))
      (is (not= (double (aget out 2))
                (double (Math/fma (aget x 2) (aget x 2) (aget x 3))))))))
