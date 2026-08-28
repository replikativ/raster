(ns raster.compiler.backend.gpu.kernel-body-workgroup-device-test
  "Numerical OpenCL oracle for verified workgroup storage and synchronization."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.kernel-body-fixtures :as fixtures]
            [raster.compiler.backend.gpu.kernel-body-opencl :as body-emit]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-body-abi :as body-abi]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.gpu.device-probe :as device-probe]))

(defn- artifact
  [width]
  (let [kernel-body (fixtures/workgroup-memory-body width)
        abi (body-abi/project-contracts
             [(kabi/slot 'x :input :float :c-name "rstr_x" :role :input)
              (kabi/slot 'y :output :float :c-name "rstr_y" :role :result)]
             kernel-body)]
    (kart/make
     {:kernel-name "kernel_body_workgroup_reverse"
      :target :opencl-c
      :source (body-emit/emit-scalar-kernel
               "kernel_body_workgroup_reverse" kernel-body
               {:target-dialect :opencl-portable})
      :abi abi
      :arguments '[x y]
      :launch (:launch kernel-body)
      :effects {:kind :workgroup-permutation}
      :provenance {:kernel-body (:id kernel-body)}
      :attributes {:workgroup-memory-bytes
                   (get-in kernel-body [:launch :shared-memory-bytes])}})))

(defn- async-artifact
  [width]
  (let [kernel-body (fixtures/async-staging-body width :preferred)
        abi (body-abi/project-contracts
             [(kabi/slot 'x :input :float :c-name "rstr_x" :role :input)
              (kabi/slot 'y :output :float :c-name "rstr_y" :role :result)]
             kernel-body)]
    (kart/make
     {:kernel-name "kernel_body_async_stage"
      :target :opencl-c
      :source (body-emit/emit-scalar-kernel
               "kernel_body_async_stage" kernel-body
               {:target-dialect :opencl-portable})
      :abi abi
      :arguments '[x y]
      :launch (:launch kernel-body)
      :effects {:kind :async-workgroup-stage}
      :provenance {:kernel-body (:id kernel-body)}
      :attributes {:async-copy-lowering :native-events
                   :workgroup-memory-bytes
                   (get-in kernel-body [:launch :shared-memory-bytes])}})))

(defn- pipelined-artifact
  [width]
  (let [kernel-body (fixtures/pipelined-staging-body width :preferred)
        abi (body-abi/project-contracts
             [(kabi/slot 'x :input :float :c-name "rstr_x" :role :input)
              (kabi/slot 'y :output :float :c-name "rstr_y" :role :result)]
             kernel-body)]
    (kart/make
     {:kernel-name "kernel_body_pipelined_stage"
      :target :opencl-c
      :source (body-emit/emit-scalar-kernel
               "kernel_body_pipelined_stage" kernel-body
               {:target-dialect :opencl-portable})
      :abi abi
      :arguments '[x y]
      :launch (:launch kernel-body)
      :effects {:kind :pipelined-workgroup-stage}
      :provenance {:kernel-body (:id kernel-body)}
      :attributes {:async-copy-lowering :native-events
                   :pipeline-depth 2
                   :workgroup-memory-bytes
                   (get-in kernel-body [:launch :shared-memory-bytes])}})))

(deftest workgroup-local-roundtrip-is-correct-on-opencl
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "KernelBody workgroup-local storage and barrier")
    (let [ocl (find-ns 'raster.gpu.ocl-runtime)
          register! (ns-resolve ocl 'register-kernel!)
          buffer-of-array (ns-resolve ocl 'buffer-of-array)
          make-buffer (ns-resolve ocl 'make-buffer)
          bind-call (ns-resolve ocl 'bind-kernel-call)
          launch! (ns-resolve ocl 'launch-registered-bound!)
          buffer->array (ns-resolve ocl 'buffer->array)
          free! (ns-resolve ocl 'free-buffer!)
          width 16
          input (float-array (map float (range width)))
          compiled (artifact width)
          x (buffer-of-array input :float)
          y (make-buffer width :float)]
      (try
        (register! (:kernel-name compiled) compiled)
        (launch! (bind-call (kcall/make compiled [x y])))
        (is (= (vec (reverse input)) (vec (buffer->array y))))
        (finally
          (free! y)
          (free! x))))))

(deftest async-workgroup-stage-is-correct-on-opencl
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "KernelBody async workgroup staging")
    (let [ocl (find-ns 'raster.gpu.ocl-runtime)
          register! (ns-resolve ocl 'register-kernel!)
          buffer-of-array (ns-resolve ocl 'buffer-of-array)
          make-buffer (ns-resolve ocl 'make-buffer)
          bind-call (ns-resolve ocl 'bind-kernel-call)
          launch! (ns-resolve ocl 'launch-registered-bound!)
          buffer->array (ns-resolve ocl 'buffer->array)
          free! (ns-resolve ocl 'free-buffer!)
          width 32
          input (float-array (map float (range width)))
          compiled (async-artifact width)
          x (buffer-of-array input :float)
          y (make-buffer width :float)]
      (try
        (register! (:kernel-name compiled) compiled)
        (launch! (bind-call (kcall/make compiled [x y])))
        (is (= (vec input) (vec (buffer->array y))))
        (finally
          (free! y)
          (free! x))))))

(deftest pipelined-workgroup-stage-is-correct-on-opencl
  (if-not @device-probe/opencl-available?
    (device-probe/opencl-skip! "KernelBody loop-carried async workgroup staging")
    (let [ocl (find-ns 'raster.gpu.ocl-runtime)
          register! (ns-resolve ocl 'register-kernel!)
          buffer-of-array (ns-resolve ocl 'buffer-of-array)
          make-buffer (ns-resolve ocl 'make-buffer)
          bind-call (ns-resolve ocl 'bind-kernel-call)
          launch! (ns-resolve ocl 'launch-registered-bound!)
          buffer->array (ns-resolve ocl 'buffer->array)
          free! (ns-resolve ocl 'free-buffer!)
          width 32
          input (float-array (map float (range width)))
          compiled (pipelined-artifact width)
          x (buffer-of-array input :float)
          y (make-buffer width :float)]
      (try
        (register! (:kernel-name compiled) compiled)
        (launch! (bind-call (kcall/make compiled [x y])))
        (is (= (mapv #(* 2.0 %) input) (vec (buffer->array y))))
        (finally
          (free! y)
          (free! x))))))
