(ns raster.gpu.distributed-training-device-test
  "Actual AD → collective → SGD execution using fresh checked LinkPlan instances.
   Co-location is not fabric evidence; unavailable devices retain explicit skips."
  (:require [clojure.test :refer [deftest]]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.compiler.fixtures.distributed-training
             :refer [check-multi-step-training! check-distributed-scalar-loss!
                     check-training!]]))

(deftest finite-multi-step-ad-all-reduce-sgd-on-local-devices
  (if @opencl/opencl-available?
    (check-multi-step-training! :ocl:0)
    (opencl/opencl-skip! "finite multi-step distributed training"))
  (if @ze/gpu-available?
    (check-multi-step-training! :ze:0)
    (ze/gpu-skip! "finite multi-step distributed training")))

(deftest scalar-loss-as-an-explicit-rank-zero-distributed-output
  (if @opencl/opencl-available?
    (check-distributed-scalar-loss! :ocl:0)
    (opencl/opencl-skip! "rank-zero distributed scalar loss"))
  (if @ze/gpu-available?
    (check-distributed-scalar-loss! :ze:0)
    (ze/gpu-skip! "rank-zero distributed scalar loss")))

(deftest actual-ad-all-reduce-sgd-on-colocated-opencl-workers
  (if @opencl/opencl-available?
    (check-training! :ocl:0)
    (opencl/opencl-skip! "actual AD/all-reduce/SGD with unequal local batches")))

(deftest actual-ad-all-reduce-sgd-on-colocated-level-zero-workers
  (if @ze/gpu-available?
    (check-training! :ze:0)
    (ze/gpu-skip! "actual AD/all-reduce/SGD with unequal local batches")))
