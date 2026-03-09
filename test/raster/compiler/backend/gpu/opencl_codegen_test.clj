(ns raster.compiler.backend.gpu.opencl-codegen-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.opencl-codegen :as codegen]
            [raster.compiler.backend.gpu.c-emit :as ce]
            [clojure.string :as str]))

;; ================================================================
;; Shared kernel-codegen tests
;; ================================================================

(deftest c-symbol-test
  (testing "Dash to underscore"
    (is (= "my_var" (ce/c-symbol 'my-var))))
  (testing "Question mark to _p"
    (is (= "nil_p" (ce/c-symbol 'nil?))))
  (testing "Combined"
    (is (= "has_value_p" (ce/c-symbol 'has-value?)))))

(deftest shared-emit-expr-test
  (testing "Number literals"
    (is (= "42" (codegen/emit-expr 42)))
    (is (= "3.14" (codegen/emit-expr 3.14))))
  (testing "Symbol mangling"
    (is (= "my_var" (codegen/emit-expr 'my-var))))
  (testing "Unary function"
    (is (= "sin(x)" (codegen/emit-expr '(Math/sin x)))))
  (testing "Binary infix"
    (is (= "(a + b)" (codegen/emit-expr '(+ a b)))))
  (testing "Binary function"
    (is (= "pow(x, 2.0)" (codegen/emit-expr '(Math/pow x 2.0)))))
  (testing "Ternary function"
    (is (= "fma(a, b, c)" (codegen/emit-expr '(Math/fma a b c))))))

(deftest normalize-identity-val-test
  (testing "Java infinities normalized to C"
    (is (= "-INFINITY" (ce/normalize-identity-val Double/NEGATIVE_INFINITY)))
    (is (= "INFINITY" (ce/normalize-identity-val Double/POSITIVE_INFINITY)))
    (is (= "NAN" (ce/normalize-identity-val Double/NaN))))
  (testing "Normal values pass through"
    (is (= "0.0" (ce/normalize-identity-val 0.0)))
    (is (= "1.0" (ce/normalize-identity-val 1.0)))))

;; ================================================================
;; OpenCL elementwise kernel
;; ================================================================

(deftest emit-elementwise-kernel-test
  (testing "Simple relu-like kernel"
    (let [src (codegen/emit-elementwise-kernel "relu_kernel" :double
                                               '(Math/max 0.0 x))]
      (is (string? src))
      (is (str/includes? src "__kernel void relu_kernel"))
      (is (str/includes? src "__global const double* restrict arr0"))
      (is (str/includes? src "__global double* restrict out"))
      (is (str/includes? src "int n"))
      (is (str/includes? src "fmax(0.0, x)"))
      (is (str/includes? src "get_global_id(0)"))
      (is (str/includes? src "get_global_size(0)"))
      ;; fp64 pragma for double
      (is (str/includes? src "cl_khr_fp64"))
      ;; Must NOT have CUDA syntax
      (is (not (str/includes? src "blockIdx")))
      (is (not (str/includes? src "__global__")))
      (is (not (str/includes? src "extern \"C\"")))))

  (testing "Float type — no fp64 pragma"
    (let [src (codegen/emit-elementwise-kernel "exp_kernel" :float
                                               '(Math/exp x))]
      (is (str/includes? src "__global const float* restrict arr0"))
      (is (str/includes? src "__global float* restrict out"))
      (is (str/includes? src "exp(x)"))
      (is (not (str/includes? src "cl_khr_fp64")))))

  (testing "Multi-array kernel"
    (let [src (codegen/emit-elementwise-kernel "add_kernel" :double
                                               '(+ x y) :n-arrays 2)]
      (is (str/includes? src "arr0"))
      (is (str/includes? src "arr1")))))

;; ================================================================
;; OpenCL reduction kernel
;; ================================================================

(deftest emit-reduction-kernel-test
  (testing "Sum reduction with local memory"
    (let [src (codegen/emit-reduction-kernel "sum_reduce" :double '+ 0.0)]
      (is (str/includes? src "__kernel void sum_reduce"))
      (is (str/includes? src "__local double sdata[256]"))
      (is (str/includes? src "barrier(CLK_LOCAL_MEM_FENCE)"))
      (is (str/includes? src "(sdata[tid] + sdata[tid + s])"))
      (is (str/includes? src "output[get_group_id(0)] = sdata[0]"))
      ;; 4x unrolled grid-stride
      (is (str/includes? src "i + 3 * stride < n"))
      ;; Must NOT have CUDA syntax
      (is (not (str/includes? src "__syncthreads()")))
      (is (not (str/includes? src "extern __shared__")))))

  (testing "Max reduction uses fmax"
    (let [src (codegen/emit-reduction-kernel "max_reduce" :double 'Math/max
                                             Double/NEGATIVE_INFINITY)]
      (is (str/includes? src "fmax(sdata[tid], sdata[tid + s])"))
      (is (str/includes? src "-INFINITY"))))

  (testing "Custom workgroup size"
    (let [src (codegen/emit-reduction-kernel "sum_512" :double '+ 0.0
                                             :workgroup-size 512)]
      (is (str/includes? src "__local double sdata[512]")))))

;; ================================================================
;; OpenCL par/map! kernel
;; ================================================================

(deftest emit-par-map-kernel-test
  (testing "Map kernel with arrays and scalars"
    (let [src (codegen/emit-par-map-kernel "par_map_1" :double
                                           '(+ a b) '[a b] '[alpha])]
      (is (str/includes? src "__kernel void par_map_1"))
      (is (str/includes? src "__global const double* restrict a"))
      (is (str/includes? src "__global const double* restrict b"))
      (is (str/includes? src "__global double* restrict out"))
      (is (str/includes? src "double alpha"))
      (is (str/includes? src "get_global_id(0)")))))

;; ================================================================
;; Kernel launch config
;; ================================================================

(deftest kernel-launch-config-test
  (testing "Below threshold returns nil"
    (is (nil? (codegen/kernel-launch-config 100)))
    (is (nil? (codegen/kernel-launch-config 4095))))

  (testing "At threshold returns config"
    (let [cfg (codegen/kernel-launch-config 4096)]
      (is (some? cfg))
      (is (pos? (:workgroup-size cfg)))
      (is (pos? (:group-count cfg)))))

  (testing "Large array config"
    (let [cfg (codegen/kernel-launch-config 1000000)]
      (is (pos? (:workgroup-size cfg)))
      (is (pos? (:group-count cfg))))))
