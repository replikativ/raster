(ns raster.compiler.backend.gpu.par-opencl-test
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [raster.compiler.backend.gpu.par-opencl :as par-opencl]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.passes.parallel.device :as device]
            [raster.compiler.support.spirv-cache :as spirv-cache]
            [raster.runtime.hardware :as hw]
            [clojure.string :as str]))

;; Set up target device for tests (no actual GPU required)
(use-fixtures :each
  (fn [f]
    (hw/reset-hardware!)
    (hw/init!)
    (hw/register-target-device! :ze:0
                                {:name "Intel(R) Arc(TM) Graphics"
                                 :capabilities {:total-eus 64
                                                :threads-per-eu 8
                                                :simd-width 16
                                                :subgroup-sizes [16 32]
                                                :max-workgroup-size 1024
                                                :shared-local-memory 131072}})
    (f)))

;; ================================================================
;; Kernel generation: par/map!
;; ================================================================

(defn- emitted-map
  [form & opts]
  (first (:kernels (apply opencl-pass/opencl-pass form
                          (concat [:min-elements 0] opts)))))

(deftest generate-segmap-kernel-artifact-simple-test
  (testing "Simple element-wise add with OpenCL syntax"
    (let [form '(raster.par/map! out i n double (+ (aget a i) (aget b i)))
          kernel (emitted-map form)]
      (is (kart/kernel-artifact? kernel))
      (is (string? (:kernel-name kernel)))
      (is (string? (:source kernel)))
      ;; OpenCL-specific syntax
      (is (str/includes? (:source kernel) "__kernel void"))
      (is (str/includes? (:source kernel) "get_global_id(0)"))
      (is (str/includes? (:source kernel) "get_global_size(0)"))
      (is (str/includes? (:source kernel) "__global const double* restrict"))
      (is (str/includes? (:source kernel) "__global double* restrict out"))
      ;; Must NOT have CUDA syntax
      (is (not (str/includes? (:source kernel) "blockIdx")))
      (is (not (str/includes? (:source kernel) "__global__")))
      (is (not (str/includes? (:source kernel) "extern \"C\"")))
      ;; Check parameter lists
      (is (= 2 (count (kart/attribute kernel :array-params))))
      (is (= '[a b out _n_bound] (mapv :name (:abi kernel))))
      (is (= [:input :input :output :scalar] (mapv :kind (:abi kernel))))
      (is (= '[a b out n] (:arguments kernel)))
      (is (= :double (kart/attribute kernel :dtype)))
      (is (= :segmap (get-in kernel [:provenance :dialect])))
      (is (= [256] (get-in kernel [:launch :workgroup-size]))))))

(deftest generate-segmap-kernel-scalar-test
  (testing "Map with scalar parameter"
    (let [form '(raster.par/map! out i n double (* alpha (aget a i)))
          kernel (emitted-map form)]
      (is (kart/kernel-artifact? kernel))
      (is (str/includes? (:source kernel) "alpha"))
      (is (= 1 (count (kart/attribute kernel :array-params))))
      (is (= '[a out alpha _n_bound] (mapv :name (:abi kernel))))
      (is (= [:double :double :double :int] (mapv :dtype (:abi kernel)))))))

(deftest generate-segmap-kernel-math-test
  (testing "Map with math operations"
    (let [form '(raster.par/map! out i n double (Math/sin (aget a i)))
          kernel (emitted-map form)]
      (is (some? kernel))
      (is (str/includes? (:source kernel) "sin(")))))

(deftest generate-segmap-kernel-float-test
  (testing "Float dtype kernel — no fp64 pragma"
    (let [form '(raster.par/map! out i n float (+ (aget a i) (aget b i)))
          kernel (emitted-map form :dtype :float)]
      (is (some? kernel))
      (is (str/includes? (:source kernel) "float"))
      (is (not (str/includes? (:source kernel) "cl_khr_fp64")))
      (is (= :float (kart/attribute kernel :dtype))))))

;; ================================================================
;; Kernel generation: par/map-void!
;; ================================================================

(deftest generate-par-map-void-ordered-abi-test
  (testing "multi-write/inout map-void ABI follows the emitted signature and preserves dtypes"
    (let [form '(raster.par/map-void! i n
                  (do (aset y i (* scale (aget x i)))
                      (aset state i (+ (aget state i) limit))))
          k (par-opencl/generate-par-map-void-kernel
             form :dtype :float
             :array-types {'state :int 'x :float 'y :float}
             :scalar-types {'limit :int 'scale :float})]
      (is (kart/kernel-artifact? k))
      (is (= '[state x y limit scale _n_bound] (mapv :name (:abi k))))
      (is (= [:output :input :output :scalar :scalar :scalar] (mapv :kind (:abi k))))
      (is (= [:int :float :float :int :float :int] (mapv :dtype (:abi k))))
      (is (= [:inout :operand :effect :parameter :parameter :bound]
             (mapv :role (:abi k))))
      (is (= '[state x y] (kart/attribute k :array-params)))
      (is (= '[state x y limit scale n] (:arguments k))))))

(deftest generate-gather-kernel-is-a-side-effect-artifact
  (let [k (par-opencl/generate-par-gather-kernel
           '(raster.par/gather out src index n stride) :dtype :float)]
    (is (kart/kernel-artifact? k))
    (is (= '[out src index stride _n_bound] (mapv :name (:abi k))))
    (is (= '[out src index stride n] (:arguments k)))
    (is (= '[out src index stride n] (kcall/logical-arguments k)))
    (is (= :side-effect-map (get-in k [:effects :kind])))
    (is (= [256] (get-in k [:launch :workgroup-size])))))

;; ================================================================
;; Kernel generation: par/reduce
;; ================================================================

(deftest generate-segred-kernel-artifact-test
  (testing "Sum reduction reaches one verified SegRed kernel artifact"
    (let [form '(raster.par/reduce acc 0.0 i n (+ acc (aget a i)))
          kernel (first (:kernels (opencl-pass/opencl-pass form :min-elements 0)))]
      (is (kart/kernel-artifact? kernel))
      (is (string? (:source kernel)))
      ;; OpenCL-specific syntax
      (is (str/includes? (:source kernel) "__kernel void"))
      (is (str/includes? (:source kernel) "__local double sdata"))
      (is (str/includes? (:source kernel) "barrier(CLK_LOCAL_MEM_FENCE)"))
      (is (str/includes? (:source kernel) "get_group_id(0)"))
      ;; Subgroup extension
      (is (str/includes? (:source kernel) "cl_intel_subgroups"))
      ;; Must NOT have CUDA
      (is (not (str/includes? (:source kernel) "__syncthreads()")))
      (is (= :segred (get-in kernel [:provenance :dialect])))
      (is (= [256] (get-in kernel [:launch :workgroup-size]))))))

;; ================================================================
;; Pipeline pass: opencl-pass
;; ================================================================

(deftest opencl-pass-map-test
  (testing "par/map! gets replaced with ze invoke-kernel marker"
    (let [form '(let* [out (double-array n)]
                      (raster.par/map! out i n double (+ (aget a i) (aget b i)))
                      out)
          result (opencl-pass/opencl-pass form :device-id :ze:0)]
      (is (map? result))
      (is (some? (:form result)))
      (is (map? (:stats result)))
      (is (= 1 (:ze-maps (:stats result))))
      (is (= 1 (count (:kernels result))))
      (is (kart/kernel-artifact? (first (:kernels result))))
      ;; Check the marker form
      (let [transformed (:form result)
            body (nth transformed 2)] ;; body of let*
        (is (and (seq? body)
                 (= 'raster.gpu.ze-runtime/invoke-registered-kernel (first body))))))))

(deftest opencl-pass-map-void-marker-follows-abi-test
  (testing "the compatibility marker is projected from the ordered ABI"
    (let [form '(raster.par/map-void! i n
                  (do (aset y i (* scale (aget x i)))
                      (aset state i (+ (aget state i) limit))))
          result (opencl-pass/opencl-pass
                  form :device-id :ze:0 :dtype :float
                  :array-types {'state :int 'x :float 'y :float}
                  :scalar-types {'limit :int 'scale :float})
          marker (:form result)
          abi (:abi (first (:kernels result)))]
      (is (= 'raster.gpu.ze-runtime/invoke-registered-map-void-kernel (first marker)))
      (is (= (kabi/pointer-binding-names abi) (nth marker 2)))
      (is (= '[limit scale] (nth marker 3)))
      (is (= 'n (nth marker 4))))))

(deftest opencl-pass-fallback-test
  (testing "Small arrays fall back to scalar expansion"
    (let [form '(raster.par/map! out i 100 double (+ (aget a i) (aget b i)))
          result (opencl-pass/opencl-pass form :device-id :ze:0 :min-elements 4096)]
      (is (= 1 (:fallback (:stats result))))
      (is (= 0 (:ze-maps (:stats result)))))))

(deftest opencl-pass-reduce-test
  (testing "par/reduce gets replaced with the ordered registered reduction marker"
    (let [form '(raster.par/reduce acc 0.0 i n (+ acc (* scale (aget a i))))
          result (opencl-pass/opencl-pass form :device-id :ze:0 :dtype :float
                                          :scalar-types {'scale :float 'n :int})]
      (is (= 1 (:ze-reduces (:stats result))))
      (is (= 1 (count (:kernels result))))
      (is (= 'raster.gpu.ze-runtime/invoke-registered-reduction-kernel
             (first (:form result))))
      (is (= '[a nil scale n] (nth (:form result) 2)))
      (is (= '[a output scale _n_bound]
             (mapv :name (:abi (first (:kernels result))))))))
  (testing "reduce-into supplies its resident result at the same ordered ABI slot"
    (let [form '(raster.par/reduce-into obuf acc 0.0 i n (+ acc (* scale (aget a i))))
          result (opencl-pass/opencl-pass form :device-id :ze:0 :dtype :float
                                          :scalar-types {'scale :float 'n :int})]
      (is (= 'raster.gpu.ze-runtime/invoke-registered-reduction-kernel
             (first (:form result))))
      (is (= '[a obuf scale n] (nth (:form result) 2)))
      (is (= '[a obuf scale _n_bound]
             (mapv :name (:abi (first (:kernels result)))))))))

(deftest opencl-pass-full-contraction-reduction-uses-ordered-marker
  (let [form '(raster.par/contract O [] [[i 8]] (* (aget A i) (aget B i)))
        result (opencl-pass/opencl-pass form :device-id :ze:0 :dtype :float)
        kernel (first (:kernels result))]
    (is (= 'raster.gpu.ze-runtime/invoke-registered-reduction-kernel
           (first (:form result))))
    (is (= '[A B nil 8] (nth (:form result) 2)))
    (is (= '[A B O _n_bound] (mapv :name (:abi kernel))))
    (is (= [:operand :operand :result :bound] (mapv :role (:abi kernel))))))

(deftest opencl-pass-nested-let-test
  (testing "Nested let* forms are traversed"
    (let [form '(let* [tmp (double-array n)
                       _ (raster.par/map! tmp i n double (Math/sin (aget a i)))
                       out (double-array n)
                       _ (raster.par/map! out i n double (* 2.0 (aget tmp i)))]
                      out)
          result (opencl-pass/opencl-pass form :device-id :ze:0)]
      (is (= 2 (:ze-maps (:stats result))))
      (is (= 2 (count (:kernels result)))))))

;; ================================================================
;; SPIR-V cache
;; ================================================================

(deftest spirv-cache-test
  (testing "Cache create and stats"
    (let [cache (spirv-cache/make-cache
                 :dir (str (System/getProperty "java.io.tmpdir")
                           "/raster-test-spirv-" (System/nanoTime)))]
      (is (= {:hits 0 :misses 0 :compiles 0}
             (spirv-cache/cache-stats cache)))
      ;; Put and get
      (let [src "__kernel void test(int n) {}"
            spv (byte-array [0x07 0x23 0x02 0x03])] ;; fake SPIR-V magic
        (spirv-cache/put-cache! cache src spv)
        (is (= 1 (:compiles (spirv-cache/cache-stats cache))))
        (let [cached (spirv-cache/get-cached cache src)]
          (is (some? cached))
          (is (= (seq spv) (seq cached)))
          (is (= 1 (:hits (spirv-cache/cache-stats cache))))))
      ;; Cleanup
      (spirv-cache/clear-cache! cache))))

;; ================================================================
;; Compound kernel codegen: aget index handling
;; ================================================================

(deftest compound-kernel-stencil-indexing-test
  (testing "Stencil aget emits full index expressions, not just idx"
    (let [metadata {:execution {:kind :compound :strategy :local :parallel-bound 64 :phase-count 1 :phase-kinds [:fused]}
                    :trip-count-sym '_step
                    :trip-count-bound 'nsteps
                    :inputs []
                    :outputs ['u]
                    :scratch ['k1]
                    :scalars ['alpha 'inv-dx2]
                    :phases [{:type :stencil :out 'k1
                              :inputs ['u] :idx 'i :bound 64
                              :body '(* alpha (* inv-dx2
                                                 (+ (clojure.core/aget u (clojure.core/- i 1))
                                                    (* -2.0 (clojure.core/aget u i))
                                                    (clojure.core/aget u (clojure.core/+ i 1)))))}
                             {:type :map :out 'u
                              :inputs ['u 'k1] :idx 'i :bound 64
                              :body '(+ (clojure.core/aget u i) (clojure.core/aget k1 i))}]}
          kernel (par-opencl/generate-compound-local-kernel metadata
                                                            :dtype :double)]
      ;; Stencil must use offset indexing
      (is (str/includes? (:source kernel) "u[(i - 1)]"))
      (is (str/includes? (:source kernel) "u[(i + 1)]"))
      ;; Not just u[i] for all accesses
      (is (> (count (re-seq #"u\[\(i [+-]" (:source kernel))) 0)))))

(deftest compound-kernel-local-array-size-test
  (testing "__local arrays use fixed max size, not runtime n"
    (let [metadata {:execution {:kind :compound :strategy :local
                                :parallel-bound '(clojure.core/alength k1)
                                :phase-count 1 :phase-kinds [:fused]}
                    :trip-count-sym '_step
                    :trip-count-bound 'nsteps
                    :inputs []
                    :outputs ['u]
                    :scratch ['k1]
                    :scalars []
                    :phases [{:type :map :out 'k1
                              :inputs ['u] :idx 'i :bound 'n
                              :body '(clojure.core/aget u i)}
                             {:type :map :out 'u
                              :inputs ['u 'k1] :idx 'i :bound 'n
                              :body '(+ (clojure.core/aget u i) (clojure.core/aget k1 i))}]}
          kernel (par-opencl/generate-compound-local-kernel metadata
                                                            :dtype :double)]
      ;; Must use a numeric constant, not "(clojure.core/alength k1)"
      (is (not (str/includes? (:source kernel) "clojure")))
      (is (not (str/includes? (:source kernel) "alength")))
      ;; Should have a numeric size like [1024]
      (is (re-find #"__local double \w+\[\d+\]" (:source kernel))))))

(deftest compound-kernel-fp64-pragma-test
  (testing "Double dtype includes fp64 pragma, float does not"
    (let [metadata {:execution {:kind :compound :strategy :local :parallel-bound 32 :phase-count 1 :phase-kinds [:fused]}
                    :trip-count-sym '_s :trip-count-bound 'ns
                    :inputs [] :outputs ['u] :scratch ['k]
                    :scalars []
                    :phases [{:type :map :out 'k :inputs ['u] :idx 'i :bound 32
                              :body '(clojure.core/aget u i)}
                             {:type :map :out 'u :inputs ['k] :idx 'i :bound 32
                              :body '(clojure.core/aget k i)}]}
          k-dbl (par-opencl/generate-compound-local-kernel metadata :dtype :double)
          k-flt (par-opencl/generate-compound-local-kernel metadata :dtype :float)]
      (is (str/includes? (:source k-dbl) "cl_khr_fp64"))
      (is (not (str/includes? (:source k-flt) "cl_khr_fp64")))
      (is (str/includes? (:source k-dbl) "__local double"))
      (is (str/includes? (:source k-flt) "__local float")))))

(deftest compound-kernel-output-copy-in-test
  (testing "Output arrays (read+write) are copied from __global to __local"
    (let [metadata {:execution {:kind :compound :strategy :local :parallel-bound 32 :phase-count 1 :phase-kinds [:fused]}
                    :trip-count-sym '_s :trip-count-bound 'ns
                    :inputs [] :outputs ['u] :scratch ['k]
                    :scalars []
                    :phases [{:type :map :out 'k :inputs ['u] :idx 'i :bound 32
                              :body '(clojure.core/aget u i)}
                             {:type :map :out 'u :inputs ['k] :idx 'i :bound 32
                              :body '(clojure.core/aget k i)}]}
          kernel (par-opencl/generate-compound-local-kernel metadata :dtype :double)]
      ;; u must be copied in (read+write)
      (is (str/includes? (:source kernel) "u[i] = u_global[i]"))
      ;; u must be copied out
      (is (str/includes? (:source kernel) "u_global[i] = u[i]")))))

;; ================================================================
;; Device integration
;; ================================================================

(deftest device-type-test
  (testing "Level Zero device type detection"
    (is (= :ze (device/device-type :ze:0)))
    (is (= :ze (device/device-type :ze:1)))))

(deftest select-backend-test
  (testing "Level Zero device selects :opencl backend"
    (is (= :opencl (device/select-backend :ze:0 nil)))
    (is (= :opencl (device/select-backend :ze:0 100000)))))
