(ns raster.compiler.passes.parallel.staged-load-transform-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.fixtures.staged-contracts :as fixtures]
            [raster.gpu.device-probe :as probe]
            [raster.gpu.link :as link]
            [raster.runtime.hardware :as hardware]))

(deftest public-decoded-stages-use-the-checked-generated-vertical
  (doseq [[device kind] [[:cuda:decoded-stage-test :cuda] [:hip:decoded-stage-test :hip]]]
    (hardware/register-target-device! device
                                     {:type kind :capabilities {:compute-capability [8 0]
                                                                :warp-size 32 :subgroup-sizes [32]
                                                                :max-workgroup-size 1024}})
    (let [compilation (equation-first/compile #'fixtures/floating-decoded-stages!
                                              {:target device :dtype :float})
          plan (equation-first/lower compilation [(float-array 16) (float-array 8)
                                                  (float-array 2) (float 1) (float 0.5)])]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= :staged-scalar (get-in compilation [:kernels 0 :attributes :kernel-body :schedule :strategy])))
      (is (= 0 (get-in plan [:attributes :driver-allocations]))))))

(deftest public-decoded-stages-match-host-through-resident-link-plan
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "public decoded staged contraction")
    (let [a (float-array (range 16)) b (float-array (repeat 8 2.0))
          expected (float-array 2) output (float-array (repeat 2 -555.0))
          _ (fixtures/floating-decoded-stages! a b expected (float 1) (float 0.5))
          compilation (equation-first/compile #'fixtures/floating-decoded-stages! {:target :ocl:0 :dtype :float})
          plan (equation-first/lower compilation [a b output (float 1) (float 0.5)])
          output-id (some (fn [[id node]] (when (identical? output (:source node)) id)) (:nodes plan))
          executable (link/instantiate! plan)]
      (try
        (link/run! executable)
        (is (= [20.0 84.0] (vec expected)))
        (is (= (vec expected) (vec (link/download executable output-id))))
        (finally (link/close! executable))))))

(deftest widening-decode-retypes-the-enclosing-product
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "widening decoded staged contraction")
    (let [a (float-array [1.0]) b (float-array [1.5]) expected (float-array 1)
          output (float-array [-555.0]) epsilon (double 6.0e-8)
          _ (fixtures/widening-decoded-stages! a b expected epsilon)
          compilation (equation-first/compile #'fixtures/widening-decoded-stages! {:target :ocl:0 :dtype :float})
          plan (equation-first/lower compilation [a b output epsilon])
          output-id (some (fn [[id node]] (when (identical? output (:source node)) id)) (:nodes plan))
          executable (link/instantiate! plan)]
      (try
        (link/run! executable)
        (is (= :none (get-in compilation [:stats :fallback])))
        (is (= [(float (* (+ 1.0 epsilon) 1.5))] (vec expected)))
        (is (not= [(float (* (float (+ 1.0 epsilon)) 1.5))] (vec expected)))
        (is (= (vec expected) (vec (link/download executable output-id))))
        (finally (link/close! executable))))))
