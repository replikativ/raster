(ns raster.compiler.backend.gpu.kernel-body-opencl-test
  (:require [clojure.java.shell :as shell]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.kernel-body-fixtures :as fixtures]
            [raster.compiler.backend.gpu.kernel-body-opencl :as opencl]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.passes.parallel.scalar-expression-body :as scalar-expression]))

(defn- scalar-kernel-body []
  (let [group 'query-row
        lane 'lane
        x-value 'x-value
        x-float 'x-float
        bias-value 'bias-value
        scaled 'scaled
        biased 'biased
        invalid? 'invalid?
        clean 'clean
        iteration 'iteration
        accumulator 'accumulator
        next-accumulator 'next-accumulator
        loop-result 'loop-result
        subgroup-sum 'subgroup-sum
        shared-sum 'shared-sum
        output-value 'output-value]
    (body/make
     {:id :scalar-opencl-test
      :parameters [(body/->KernelParameter
                    'x :input :half [2 16] :global
                    (layout/row-major [2 16] :half) :input)
                   (body/->KernelParameter
                    'bias :input :float [2] :global
                    (layout/row-major [2] :float) :bias)
                   (body/->KernelParameter
                    'y :output :half [2] :global
                    (layout/row-major [2] :half) :result)
                   (body/->KernelParameter 'scale :scalar :float [] nil nil :scale)]
      :views [(body/->BufferView
               'x-row 'x (body/expression :mul group 16) [16]
               (layout/row-major [16] :half))]
      :stable-reads [(body/stable-read 'x) (body/stable-read 'bias)]
      :indices [(body/->IndexBinding group :group 0)
                (body/->IndexBinding lane :lane 0)
                (body/->IndexCompute 'linear (body/expression :add
                                                              (body/expression :mul group 16)
                                                              lane))]
      :masks [(body/->Mask :active [(body/predicate :lt lane 16)])
              (body/->Mask :lane-zero [(body/predicate :eq lane 0)])]
      :operations
      [(body/->ScalarLoad (body/value x-value :half) 'x-row [lane] :active
                          (body/literal 0.0 :half) :cached)
       (body/->ScalarCompute (body/value x-float :float)
                             (body/cast-expression x-value :float :exact :exact))
       (body/->ScalarLoad (body/value bias-value :float) 'bias [group]
                          nil nil :cached)
       (body/->ScalarCompute (body/value scaled :float)
                             (body/scalar-expression :* :float [x-float 'scale]))
       (body/->ScalarCompute (body/value biased :float)
                             (body/scalar-expression :+ :float [scaled bias-value]))
       (body/->ScalarCompute (body/value invalid? :predicate)
                             (body/scalar-expression :isnan :predicate [biased]))
       (body/->IfRegion invalid?
                        [(body/->Yield [(body/literal 0.0 :float)])]
                        [(body/->Yield [biased])]
                        [(body/value clean :float)])
       (body/->ForLoop
        (body/value iteration :int) 0 2 1
        [(body/->LoopArg (body/value accumulator :float) clean)]
        [(body/->ScalarCompute
          (body/value next-accumulator :float)
          (body/scalar-expression :+ :float [accumulator 'scale]))
         (body/->Yield [next-accumulator])]
        [(body/value loop-result :float)]
        {:unroll true})
       (body/->Collective
        (body/value subgroup-sum :float) :reduce :subgroup 16 loop-result :+ nil
        (body/full-participation) :implementation-defined)
       (body/->Collective
        (body/value shared-sum :float) :broadcast :subgroup 16 subgroup-sum nil 0
        (body/full-participation) nil)
       (body/->ScalarCompute
        (body/value output-value :half)
        (body/cast-expression shared-sum :half :nearest-even :ieee))
       (body/->ScalarStore 'y [group] output-value :lane-zero)]
      :schedule {:subgroup-size 16}
      :launch (launch/spec {:workgroup-size [16] :group-count [2]})
      :provenance {:dialect :test}
      :attributes {:kind :scalar}})))

(defn workgroup-kernel-body
  "Small production-shaped fixture shared with the hardware-free vendor compiler gates."
  []
  (fixtures/workgroup-memory-body 16))

(defn swizzled-workgroup-kernel-body []
  (fixtures/swizzled-workgroup-memory-body 32))

(defn async-staging-kernel-body
  ([] (async-staging-kernel-body :preferred))
  ([overlap] (fixtures/async-staging-body 32 overlap)))

(defn pipelined-staging-kernel-body
  ([] (pipelined-staging-kernel-body :preferred))
  ([overlap] (fixtures/pipelined-staging-body 32 overlap)))

(defn- dp4a-kernel-body []
  (body/make
   {:id :dp4a-helper-test
    :parameters [(body/->KernelParameter 'a :scalar :int [] nil nil :a)
                 (body/->KernelParameter 'b :scalar :int [] nil nil :b)
                 (body/->KernelParameter 'out :output :int [1] :global
                                         (layout/row-major [1] :int) :result)]
    :operations [(body/->ScalarCompute
                  (body/value 'dot :int)
                  (body/scalar-expression :dp4a :int
                                          ['a 'b (body/literal 0 :int)]))
                 (body/->ScalarStore 'out [0] 'dot nil)]
    :launch (launch/spec {:workgroup-size [1] :group-count [1]})
    :provenance {:dialect :test}
    :attributes {:kind :scalar}}))

(defn- atomic-add-kernel-body []
  (body/make
   {:id :atomic-add-test
    :parameters [(body/->KernelParameter
                  'out :inout :float [16] :global
                  (layout/row-major [16] :float) :result)
                 (body/->KernelParameter 'contribution :scalar :float [] nil nil :value)]
    :indices [(body/->IndexBinding 'lane :local 0)]
    :operations [(body/->AtomicRMW 'out ['lane] 'contribution :+ nil)]
    :launch (launch/spec {:workgroup-size [16] :group-count [1]})
    :provenance {:dialect :test}
    :attributes {:kind :scalar}}))

(defn- integer-arithmetic-kernel-body
  ([overflow]
   (integer-arithmetic-kernel-body overflow :+ :long))
  ([overflow operation type]
   (body/make
    {:id [:integer-arithmetic-test overflow operation type]
     :parameters [(body/->KernelParameter 'a :scalar type [] nil nil :left)
                  (body/->KernelParameter 'b :scalar type [] nil nil :right)
                  (body/->KernelParameter 'out :output type [1] :global
                                          (layout/row-major [1] type) :result)]
     :operations [(body/->ScalarCompute
                   (body/value 'result type)
                   (body/scalar-expression operation type ['a 'b] {:overflow overflow}))
                  (body/->ScalarStore 'out [0] 'result nil)]
     :launch (launch/spec {:workgroup-size [1] :group-count [1]})
     :provenance {:dialect :test}
     :attributes {:kind :scalar}})))

(defn- wrapping-arithmetic-kernel-body []
  (integer-arithmetic-kernel-body :wrap))

(defn- bounded-byte-add-kernel-body []
  (let [decline! (fn [rule message data]
                   (throw (ex-info message (assoc data :rule rule))))
        lowerer (scalar-expression/make-lowerer
                 {:array-types {'q :byte} :scalar-types {'i :long} :arrays #{'q}
                  :index-scope #{'i} :lower-index (fn [value _] value)
                  :decline! decline!})
        lowered ((:lower lowerer) '(clojure.core/+ (int (clojure.core/aget q i)) 7)
                 :int {'i :long})]
    (body/make
     {:id :bounded-byte-add
      :parameters [(body/->KernelParameter 'q :input :byte [16] :global
                                            (layout/row-major [16] :byte) :input)
                   (body/->KernelParameter 'i :scalar :long [] nil nil :index)
                   (body/->KernelParameter 'out :output :int [1] :global
                                            (layout/row-major [1] :int) :result)]
      :stable-reads [(body/stable-read 'q)]
      :operations (conj (vec (:operations lowered))
                        (body/->ScalarStore 'out [0] (:result lowered) nil))
      :launch (launch/spec {:workgroup-size [1] :group-count [1]})
      :provenance {:dialect :test}
      :attributes {:kind :scalar}})))

(defn- forged-no-overflow-kernel-body []
  (body/make
   {:id :forged-no-overflow
    :parameters [(body/->KernelParameter 'a :scalar :int [] nil nil :left)
                 (body/->KernelParameter 'b :scalar :int [] nil nil :right)
                 (body/->KernelParameter 'out :output :int [1] :global
                                          (layout/row-major [1] :int) :result)]
    :operations [(body/->ScalarCompute
                  (body/value 'result :int)
                  (body/scalar-expression
                   :+ :int ['a 'b]
                   {:overflow :no-overflow
                    ;; This contains the verifier's full operand result interval, but cannot
                    ;; make that interval fit in int.  Evidence may describe a derivation; it
                    ;; cannot turn overflow-prone arithmetic into `:no-overflow`.
                    :proof {:kind :typed-scalar-range
                            :lower Integer/MIN_VALUE :upper Integer/MAX_VALUE}}))
                 (body/->ScalarStore 'out [0] 'result nil)]
    :launch (launch/spec {:workgroup-size [1] :group-count [1]})
    :provenance {:dialect :test}
    :attributes {:kind :scalar}}))

(deftest scalar-kernel-body-lowers-without-recovering-a-schedule
  (let [source (opencl/emit-scalar-kernel
                "scheduled_scalar"
                (scalar-kernel-body)
                {:parameter-names {'x "input_rows" 'bias "row_bias"
                                   'y "output_rows" 'scale "scale"}})]
    (is (str/includes? source "__global const half* restrict input_rows"))
    (is (str/includes? source
                       "input_rows[((long)((rstr_query_row * 16)) + ((long)(rstr_lane)"))
    (is (str/includes? source "((rstr_lane < 16)) ? input_rows"))
    (is (str/includes? source "float rstr_loop_result = rstr_clean;"))
    (is (str/includes? source "sub_group_reduce_add(rstr_loop_result)"))
    (is (str/includes? source "sub_group_broadcast(rstr_subgroup_sum, 0)"))
    (is (str/includes? source "convert_half_rte(rstr_shared_sum)"))
    (is (str/includes? source "if (((rstr_lane == 0)))"))
    (is (not (str/includes? source "__global half* restrict output_rows"))
        "only body-proven stable reads acquire a target no-alias qualifier")
    (testing "the emitted target program is valid OpenCL C"
      (if-not (zero? (:exit (shell/sh "sh" "-c" "command -v clang")))
        (is true "clang unavailable")
        (let [result (shell/sh "clang" "-x" "cl" "-cl-std=CL2.0"
                               "-fsyntax-only" "-" :in source)]
          (is (zero? (:exit result)) (:err result)))))))

(deftest target-lowering-refuses-unrepresentable-numerical-contracts
  (let [kernel (scalar-kernel-body)
        trapping-cast
        (body/->ScalarCompute
         (body/value 'trapping :int)
         (body/cast-expression 'shared-sum :int :toward-zero :trap))
        kernel (-> kernel
                   (update :operations #(vec (concat (subvec % 0 10)
                                                     [trapping-cast]
                                                     (subvec % 10))))
                   body/validate!)]
    (testing "target conversion syntax may not weaken explicit cast policy"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"cannot preserve this KernelBody cast policy"
           (opencl/emit-scalar-kernel "unsupported_cast" kernel))))
    (testing "the target must have an actual collective spelling"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"no matching subgroup reduction builtin"
           (opencl/emit-scalar-kernel
            "unsupported_collective"
            (-> (scalar-kernel-body)
                (assoc-in [:operations 8 :operator] :*)
                body/validate!)))))
    (testing "an explicit numerical association cannot become a target builtin tree"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"cannot preserve an explicit reduction tree"
           (opencl/emit-scalar-kernel
            "explicit_tree"
            (-> (scalar-kernel-body)
                (assoc-in [:operations 8 :association]
                          {:kind :shuffle-down-tree :distances [8 4 2 1]})
                body/validate!)))))
    (testing "physical layout is not guessed from an unsupported descriptor"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"requires a dense strided layout"
           (opencl/emit-scalar-kernel
            "unsupported_layout"
            (-> (scalar-kernel-body)
                (assoc-in [:views 0 :layout :kind] :blocked)
                body/validate!)))))
    (testing "external ABI spelling cannot collide with generated SSA names"
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"names collide"
           (opencl/emit-scalar-kernel
            "colliding_names" (scalar-kernel-body)
            {:parameter-names {'x "same" 'bias "same"}}))))))

(defn- command-available? [command]
  (zero? (:exit (shell/sh "sh" "-c" (str "command -v " command)))))

(defn- near-signed-limit-loop-body []
  (body/make
   {:id :near-signed-limit-loop
    :operations
    [(body/->ForLoop
      (body/value 'edge-iteration :int)
      2147483646 2147483647 2 []
      [(body/->Yield [])] []
      {:association :ordered})]
    :launch (launch/spec {:workgroup-size [1] :group-count [1]})
    :provenance {:dialect :test}
    :attributes {:kind :scalar}}))

(deftest positive-loop-steps-cannot-overflow-the-signed-induction-variable
  (let [kernel (near-signed-limit-loop-body)
        sources (mapv (fn [target]
                        (opencl/emit-scalar-kernel
                         "near_signed_limit" kernel {:target-dialect target}))
                      [:opencl-portable :cuda :hip])]
    (testing "all C-family targets guard the update in same-width unsigned arithmetic"
      (is (str/includes? (first sources)
                         "(uint)(2147483647) - (uint)(rstr_edge_iteration) <= (uint)(2)"))
      (doseq [source (rest sources)]
        (is (str/includes?
             source
             (str "(unsigned int)(2147483647) - (unsigned int)(rstr_edge_iteration)"
                  " <= (unsigned int)(2)"))))
      (doseq [source sources]
        (is (str/includes? source "if ("))
        (is (str/includes? source "rstr_edge_iteration += 2;"))))
    (testing "the guarded boundary case remains valid OpenCL C"
      (when (command-available? "clang")
        (let [{:keys [exit err]} (shell/sh "clang" "-x" "cl" "-cl-std=CL2.0"
                                           "-fsyntax-only" "-" :in (first sources))]
          (is (zero? exit) err))))))

(defn- compile-c-family-source
  [target source]
  (let [suffix (case target :cuda ".cu" :hip ".hip")
        source-file (java.io.File/createTempFile "raster-kernel-body-" suffix)
        output-file (str (.getAbsolutePath source-file) ".ptx")]
    (spit source-file source)
    (case target
      :cuda (shell/sh "nvcc" "-ptx" "-arch=sm_80" "-o" output-file
                      (.getAbsolutePath source-file))
      :hip (shell/sh "hipcc" "--offload-arch=gfx1100" "--genco"
                     "-o" (str (.getAbsolutePath source-file) ".hsaco")
                     (.getAbsolutePath source-file)))))

(deftest one-scheduled-body-lowers-to-cuda-and-hip
  (doseq [[target compiler shuffle broadcast]
          [[:cuda "nvcc" "__shfl_down_sync(__activemask()"
            "__shfl_sync(__activemask()"]
           [:hip "hipcc" "__shfl_down(" "__shfl("]]]
    (testing (str (name target) " uses only its thin target spelling")
      (let [source (opencl/emit-scalar-kernel
                    "scheduled_scalar" (scalar-kernel-body)
                    {:target-dialect target
                     :parameter-names {'x "input_rows" 'bias "row_bias"
                                       'y "output_rows" 'scale "scale"}})]
        (is (str/includes? source "extern \"C\" __global__ void scheduled_scalar"))
        (is (str/includes? source "blockIdx.x"))
        (is (str/includes? source "threadIdx.x"))
        (is (str/includes? source shuffle))
        (is (str/includes? source broadcast))
        (is (str/includes? source "__half2float("))
        (is (str/includes? source "__float2half_rn("))
        (is (not (str/includes? source "get_group_id")))
        (is (not (str/includes? source "sub_group_reduce")))
        (testing "the installed host toolchain accepts the emitted device source"
          (if-not (command-available? compiler)
            (is true (str compiler " unavailable; source structure remains covered"))
            (let [{:keys [exit err]} (compile-c-family-source target source)]
              (is (zero? exit) err))))))))

(deftest one-atomic-update-contract-has-thin-target-spellings
  (doseq [[target spelling]
          [[:opencl-portable "atomic_add_float(rstr_out_ +"]
           [:cuda "atomicAdd(rstr_out_ +"]
           [:hip "atomicAdd(rstr_out_ +"]]]
    (let [source (opencl/emit-scalar-kernel
                  "atomic_add_test" (atomic-add-kernel-body)
                  {:target-dialect target})]
      (is (str/includes? source spelling) (name target))
      (is (not (str/includes? source "const float* out")) (name target))))
  (testing "the OpenCL helper and pointer spelling type-check together"
    (when (command-available? "clang")
      (let [source (opencl/emit-scalar-kernel
                    "atomic_add_test" (atomic-add-kernel-body)
                    {:target-dialect :opencl-portable})
            {:keys [exit err]} (shell/sh "clang" "-x" "cl" "-cl-std=CL2.0"
                                         "-fsyntax-only" "-" :in source)]
        (is (zero? exit) err)))))

(deftest source-integral-arithmetic-retains-its-overflow-semantics
  (let [decline! (fn [rule message data]
                   (throw (ex-info message (assoc data :rule rule))))
        lowerer (scalar-expression/make-lowerer
                 {:array-types {} :scalar-types {'a :long 'b :long} :arrays #{}
                  :index-scope #{} :lower-index (fn [value _] value)
                  :decline! decline!})
        checked (mapv #((:lower lowerer) % :long {'a :long 'b :long})
                      ['(clojure.core/+ a b)
                       '(clojure.core/- a b)
                       '(clojure.core/* a b)])
        lowered ((:lower lowerer) '(unchecked-add a b) :long
                 {'a :long 'b :long})]
    (is (every? #(= {:overflow :trap}
                     (get-in % [:operations 0 :expression :options]))
                checked)
        "ordinary Clojure integer arithmetic is checked, never target-signed overflow")
    (is (= {:overflow :wrap}
           (get-in lowered [:operations 0 :expression :options])))
    (is (= :+ (get-in lowered [:operations 0 :expression :op]))))
  (let [decline! (fn [rule message data]
                   (throw (ex-info message (assoc data :rule rule))))
        lowerer (scalar-expression/make-lowerer
                 {:array-types {'q :byte} :scalar-types {'i :long} :arrays #{'q}
                  :index-scope #{'i} :lower-index (fn [value _] value)
                  :decline! decline!})
        bounded ((:lower lowerer) '(clojure.core/+ (int (clojure.core/aget q i)) 7)
                 :int {'i :long})
        unknown ((:lower lowerer) '(clojure.core/+ a i) :long {'a :long 'i :long})
        proved-kernel (bounded-byte-add-kernel-body)
        opencl-source (opencl/emit-scalar-kernel
                       "proved_byte_add" proved-kernel {:target-dialect :opencl-portable})]
    (is (= {:overflow :no-overflow
            :proof {:kind :typed-scalar-range :lower -121 :upper 134}}
           (get-in (peek (:operations bounded)) [:expression :options])))
    (is (= {:overflow :trap}
           (get-in unknown [:operations 0 :expression :options])))
    (is (= :+ (get-in (peek (:operations bounded)) [:expression :op]))
        "byte storage, exact widening, and a literal prove this OpenCL-safe operation")
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo
         #"not derivable from operand ranges"
         (integer-arithmetic-kernel-body :no-overflow :+ :int))
        "a producer cannot certify arbitrary scalar parameters with a bare no-overflow tag")
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo
         #"integral arithmetic"
         (forged-no-overflow-kernel-body))
        "a producer proof map is evidence to check, never an authority to forge")
    (is (str/includes? opencl-source "+ 7")
        "a range-certified semantic operation remains legal portable OpenCL C"))
  (doseq [[target unsigned-type]
          [[:opencl-portable "ulong"]
           [:cuda "unsigned long long"]
           [:hip "unsigned long long"]]]
    (testing (name target)
      (let [source (opencl/emit-scalar-kernel
                    "wrapping_arithmetic" (wrapping-arithmetic-kernel-body)
                    {:target-dialect target})]
        (is (str/includes? source
                           (str "(" unsigned-type ")(rstr_a) + (" unsigned-type ")(rstr_b)")))
        (is (not (str/includes? source "rstr_a + rstr_b"))
            "signed target arithmetic must not weaken modulo-2^N semantics")))))

(deftest explicit-signed-overflow-contracts-reach-the-target-boundary
  (doseq [target [:opencl-portable :cuda :hip]]
    (testing (name target)
      (let [source (opencl/emit-scalar-kernel
                    "proved_arithmetic"
                    (bounded-byte-add-kernel-body)
                    {:target-dialect target})]
        (is (str/includes? source "+ 7")
            "a proved in-range operation may use the target's signed instruction"))))
  (testing "portable OpenCL declines a contract for which the language has no standard trap primitive"
    (try
      (opencl/emit-scalar-kernel
       "trapping_arithmetic" (integer-arithmetic-kernel-body :trap)
       {:target-dialect :opencl-portable})
      (is false "trapping arithmetic must not silently become signed target overflow")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :kernel-body-c-trap-unsupported (:reason (ex-data exception)))))))
  (testing "Intel OpenCL uses its explicit vendor trap contract"
    (let [source (opencl/emit-scalar-kernel
                  "trapping_arithmetic" (integer-arithmetic-kernel-body :trap)
                  {:target-dialect :opencl-intel})]
      (is (str/includes? source "rstr_trap_add_i64(rstr_a, rstr_b)"))
      (is (str/includes? source "__builtin_trap();"))))
  (doseq [[target compiler trap-spelling]
          [[:cuda "nvcc" "asm volatile(\"trap;\")"]
           [:hip "hipcc" "__builtin_trap()"]]]
    (testing (str (name target) " checked helpers")
      (doseq [[operation operation-name guard]
              [[:+ "add" "~(ua ^ ub) & (ua ^ ur)"]
               [:- "sub" "(ua ^ ub) & (ua ^ ur)"]
               [:* "mul" "ma > (limit / mb)"]]
              type [:byte :int :long]]
        (let [source (opencl/emit-scalar-kernel
                      "trapping_arithmetic"
                      (integer-arithmetic-kernel-body :trap operation type)
                      {:target-dialect target})
              helper-name (str "rstr_trap_" operation-name "_"
                               ({:byte "i8" :int "i32" :long "i64"} type))]
          (is (str/includes? source (str helper-name "(rstr_a, rstr_b)")))
          (is (str/includes? source guard))
          (is (str/includes? source trap-spelling))
          (is (= 2 (count (re-seq (re-pattern (str helper-name "\\(")) source)))
              "one checked definition accompanies one typed call")))
      ;; `trapping-arithmetic-body` puts an add expression directly in ScalarStore.value.  This
      ;; keeps helper discovery honest for legal non-ScalarCompute expression placements.
      (let [source (opencl/emit-scalar-kernel
                    "trapping_arithmetic" (fixtures/trapping-arithmetic-body)
                    {:target-dialect target})]
        (is (= 3 (count (re-seq #"rstr_trap_add_i32\(" source)))
            "direct ScalarStore and nested LoopArg trap expressions receive their CUDA/HIP helper"))
      (when (command-available? compiler)
        (let [source (opencl/emit-scalar-kernel
                      "trapping_arithmetic" (fixtures/trapping-arithmetic-body)
                      {:target-dialect target})
              {:keys [exit err]} (compile-c-family-source target source)]
          (is (zero? exit) err))))))

(deftest registry-intrinsic-helpers-follow-the-c-family-target
  (doseq [[target qualifier physical-op compiler]
          [[:opencl-portable "inline int rstr_dp4a" "a0*b0" nil]
           [:cuda "__device__ __forceinline__ int rstr_dp4a" "__dp4a(a, b, acc)" "nvcc"]
           [:hip "__device__ __forceinline__ int rstr_dp4a"
            "__ockl_sdot4(a, b, acc, false)" "hipcc"]]]
    (testing (name target)
      (let [source (opencl/emit-scalar-kernel
                    "dp4a_helper" (dp4a-kernel-body) {:target-dialect target})]
        (is (str/includes? source qualifier))
        (is (str/includes? source physical-op))
        (is (= 2 (count (re-seq #"rstr_dp4a\(" source)))
            "one registry-owned definition accompanies one typed call")
        (when (and compiler (command-available? compiler))
          (let [{:keys [exit err]} (compile-c-family-source target source)]
            (is (zero? exit) err)))))))

(deftest portable-opencl-keeps-the-scheduled-subgroup-width
  (let [source (opencl/emit-scalar-kernel
                "portable_scalar" (scalar-kernel-body)
                {:target-dialect :opencl-portable})]
    (is (str/includes? source "__attribute__((reqd_sub_group_size(16)))"))
    (is (not (str/includes? source "intel_reqd_sub_group_size")))
    (if-not (command-available? "clang")
      (is true "clang unavailable; source structure remains covered")
      (let [{:keys [exit err]} (shell/sh "clang" "-x" "cl" "-cl-std=CL2.0"
                                         "-fsyntax-only" "-" :in source)]
        (is (zero? exit) err)))))

(deftest workgroup-memory-and-barriers-share-one-c-family-lowering
  (doseq [[target arena barrier pointer]
          [[:opencl-intel "__local uint4 rstr_workgroup_memory[4]"
            "barrier(CLK_LOCAL_MEM_FENCE);" "__local float* rstr_scratch"]
           [:cuda "__shared__ __align__(16) unsigned char rstr_workgroup_memory[64]"
            "__syncthreads();" "float* rstr_scratch = reinterpret_cast<float*>"]
           [:hip "__shared__ __attribute__((aligned(16))) unsigned char rstr_workgroup_memory[64]"
            "__syncthreads();" "float* rstr_scratch = reinterpret_cast<float*>"]]]
    (testing (name target)
      (let [source (opencl/emit-scalar-kernel
                    "workgroup_memory" (workgroup-kernel-body)
                    {:target-dialect target})]
        (is (str/includes? source arena))
        (is (str/includes? source pointer))
        (is (str/includes? source barrier))
        (is (str/includes? source "rstr_scratch[(((long)(rstr_lane) * (long)(1)))]")))))
  (testing "the portable OpenCL source remains valid OpenCL C"
    (let [source (opencl/emit-scalar-kernel
                  "workgroup_memory" (workgroup-kernel-body)
                  {:target-dialect :opencl-portable})]
      (if-not (command-available? "clang")
        (is true "clang unavailable; source structure remains covered")
        (let [{:keys [exit err]} (shell/sh "clang" "-x" "cl" "-cl-std=CL2.0"
                                           "-fsyntax-only" "-" :in source)]
          (is (zero? exit) err))))))

(deftest verified-xor-layout-shares-one-c-family-address-transform
  (doseq [target [:opencl-portable :cuda :hip]]
    (let [source (opencl/emit-scalar-kernel
                  "swizzled_workgroup_memory" (swizzled-workgroup-kernel-body)
                  {:target-dialect target})]
      (is (str/includes? source
                         "((long)(rstr_lane) * (long)(32) + ((long)(0) ^ ((long)(rstr_lane) & 31)))")
          (name target))))
  (testing "the portable source remains valid OpenCL C"
    (let [source (opencl/emit-scalar-kernel
                  "swizzled_workgroup_memory" (swizzled-workgroup-kernel-body)
                  {:target-dialect :opencl-portable})]
      (if-not (command-available? "clang")
        (is true "clang unavailable; source structure remains covered")
        (let [{:keys [exit err]} (shell/sh "clang" "-x" "cl" "-cl-std=CL2.0"
                                           "-fsyntax-only" "-" :in source)]
          (is (zero? exit) err))))))

(deftest async-staging-retains-one-dependency-contract-across-targets
  (doseq [[target copy commit wait]
          [[:opencl-portable "async_work_group_copy" "OpenCL event group"
            "wait_group_events(1, &rstr_stage_x)"]
           [:cuda "cp.async.ca.shared.global" "cp.async.commit_group"
            "cp.async.wait_group 0"]
           [:hip "rstr_stage_x_element_offset" "synchronous async group"
            "synchronous cooperative copies are complete"]]]
    (testing (name target)
      (let [source (opencl/emit-scalar-kernel
                    "async_staging" (async-staging-kernel-body)
                    (cond-> {:target-dialect target}
                      (= :cuda target)
                      (assoc :target-features {:compute-capability [8 0]})))]
        (is (str/includes? source copy))
        (is (str/includes? source commit))
        (is (str/includes? source wait))
        (is (str/includes? source (if (= target :opencl-portable)
                                    "barrier(CLK_LOCAL_MEM_FENCE)"
                                    "__syncthreads()"))))))
  (testing "required overlap fails honestly on a synchronous-only target"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"cannot preserve required async-copy overlap"
         (opencl/emit-scalar-kernel
          "async_required" (async-staging-kernel-body :required)
          {:target-dialect :hip}))))
  (testing "CUDA async instructions are gated by the concrete architecture"
    (let [source (opencl/emit-scalar-kernel
                  "async_pre_ampere" (async-staging-kernel-body)
                  {:target-dialect :cuda
                   :target-features {:compute-capability [7 5]}})]
      (is (not (str/includes? source "cp.async")))
      (is (str/includes? source "synchronous cooperative copies are complete")))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"cannot preserve required async-copy overlap"
         (opencl/emit-scalar-kernel
          "async_required_unknown_cuda" (async-staging-kernel-body :required)
          {:target-dialect :cuda}))))
  (testing "the native event lowering remains valid OpenCL C"
    (when (command-available? "clang")
      (let [source (opencl/emit-scalar-kernel
                    "async_staging" (async-staging-kernel-body)
                    {:target-dialect :opencl-portable})
            {:keys [exit err]} (shell/sh "clang" "-x" "cl" "-cl-std=CL2.0"
                                         "-fsyntax-only" "-" :in source)]
        (is (zero? exit) err)))))

(deftest pipelined-for-lowers-loop-carried-async-groups-across-targets
  (let [opencl-source (opencl/emit-scalar-kernel
                       "pipelined_staging" (pipelined-staging-kernel-body)
                       {:target-dialect :opencl-portable})
        cuda-source (opencl/emit-scalar-kernel
                     "pipelined_staging" (pipelined-staging-kernel-body)
                     {:target-dialect :cuda
                      :target-features {:compute-capability [8 0]}})
        hip-source (opencl/emit-scalar-kernel
                    "pipelined_staging" (pipelined-staging-kernel-body)
                    {:target-dialect :hip})]
    (testing "OpenCL materializes event-valued loop carries and parallel backedge copies"
      (is (str/includes? opencl-source
                         "event_t rstr_carry_a_event_0 = rstr_warm_a"))
      (is (str/includes? opencl-source
                         "event_t rstr_carry_a_next_event_0 = rstr_refill_a"))
      (is (str/includes? opencl-source
                         "rstr_carry_a_event_0 = rstr_carry_a_next_event_0"))
      (is (str/includes? opencl-source
                         "wait_group_events(1, &rstr_carry_a_event_0)")))
    (testing "CUDA retains the same steady-state loop through native cp.async groups"
      (is (str/includes? cuda-source "for (int rstr_pipeline_iteration = 0"))
      (is (str/includes? cuda-source "cp.async.commit_group"))
      (is (str/includes? cuda-source "cp.async.wait_group 1")))
    (testing "HIP preserves correctness with an honest synchronous pipeline spelling"
      (is (str/includes? hip-source "for (int rstr_pipeline_iteration = 0"))
      (is (str/includes? hip-source "synchronous cooperative copies are complete"))
      (is (not (str/includes? hip-source "cp.async"))))
    (testing "the native event pipeline remains valid OpenCL C"
      (when (command-available? "clang")
        (let [{:keys [exit err]} (shell/sh "clang" "-x" "cl" "-cl-std=CL2.0"
                                           "-fsyntax-only" "-" :in opencl-source)]
          (is (zero? exit) err))))))
