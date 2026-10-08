(ns raster.gpu.collective-combine-device-test
  "Local monoid combines only: these checks do not claim distributed all-reduce execution."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.passes.parallel.collective-combine :as combine]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]))

(defn- check-combines! [device]
  (let [n 37]
    (gpu/with-gpu-session [session device]
      (gpu/alloc! session {:left [:float n nil] :right [:float n nil] :result [:float n nil]})
      (doseq [[operator identity oracle] [['+ 0.0 +] ['* 1.0 *]]]
        (let [algebra (scan/certify-reassociation
                       {:acc 'acc :init identity :lambda (list operator 'acc 'element)} :float)
              emitted (combine/emit algebra n {:target-device device
                                               :target-descriptor (hardware/descriptor-for device)})
              _ (combine/validate! algebra n emitted)
              handle (gpu/bind-kernel-graph!
                       session operator (:graph emitted)
                       {'left :left 'right :right 'result :result} {})]
          (try
            (doseq [replay [0 1]]
              (let [left (float-array (map #(* 0.25 (- % 19 (* replay 3))) (range n)))
                    right (float-array (map #(* 0.5 (+ 1 % (* replay 2))) (range n)))
                    expected (mapv #(float (oracle (double %1) (double %2))) left right)]
                (gpu/upload! session :left left)
                (gpu/upload! session :right right)
                (gpu/run-kernel-graph! session handle)
                (is (= expected (vec (gpu/download session :result)))
                    "independent scalar CPU oracle and changed inputs on the same resident graph")))
            (finally (gpu/release-kernel-graph! session handle))))))))

(deftest ordinary-typed-collective-combines-on-opencl
  (if @opencl/opencl-available?
    (check-combines! :ocl:0)
    (opencl/opencl-skip! "ordinary typed collective combines")))

(deftest ordinary-typed-collective-combines-on-level-zero
  (if @ze/gpu-available?
    (check-combines! :ze:0)
    (ze/gpu-skip! "ordinary typed collective combines")))
