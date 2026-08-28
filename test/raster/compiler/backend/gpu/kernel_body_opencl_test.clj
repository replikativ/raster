(ns raster.compiler.backend.gpu.kernel-body-opencl-test
  (:require [clojure.java.shell :as shell]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.kernel-body-opencl :as opencl]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]))

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

(deftest scalar-kernel-body-lowers-without-recovering-a-schedule
  (let [source (opencl/emit-scalar-kernel
                "scheduled_scalar"
                (scalar-kernel-body)
                {:parameter-names {'x "input_rows" 'bias "row_bias"
                                   'y "output_rows" 'scale "scale"}})]
    (is (str/includes? source "__global const half* input_rows"))
    (is (str/includes? source
                       "input_rows[((long)((rstr_query_row * 16)) + ((long)(rstr_lane)"))
    (is (str/includes? source "((rstr_lane < 16)) ? input_rows"))
    (is (str/includes? source "float rstr_loop_result = rstr_clean;"))
    (is (str/includes? source "sub_group_reduce_add(rstr_loop_result)"))
    (is (str/includes? source "sub_group_broadcast(rstr_subgroup_sum, 0)"))
    (is (str/includes? source "convert_half_rte(rstr_shared_sum)"))
    (is (str/includes? source "if (((rstr_lane == 0)))"))
    (is (not (str/includes? source "restrict")))
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
