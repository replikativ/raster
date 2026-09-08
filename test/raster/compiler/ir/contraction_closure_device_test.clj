(ns raster.compiler.ir.contraction-closure-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.gpu.segop-opencl :as emitter]
            [raster.compiler.compatibility-ledger-test :as ledger]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.contraction-closure-test :as fixture]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.device-probe :as probe]))

(deftest public-staged-contraction-compiles-links-and-runs
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "public typed staged contraction")
    (let [output (float-array (repeat 2 -555.0))
          compilation (equation-first/compile
                       #'ledger/staged-byte-float-contract! {:target :ocl:0 :dtype :float})
          plan (equation-first/lower
                compilation [(byte-array (repeat 8 1))
                             (byte-array (concat (repeat 8 2) (repeat 8 3)))
                             (float-array (repeat 2 0.5))
                             (float-array (repeat 4 0.25)) output])
          output-id (some (fn [[id node]] (when (identical? output (:source node)) id))
                          (:nodes plan))]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 0 (get-in plan [:attributes :driver-allocations])))
      (is (some? output-id) "resolve the state buffer by source identity, not kernel ABI spelling")
      (let [executable (link/instantiate! plan)]
        (try
          (link/run! executable)
          (is (= [2.0 3.0] (vec (link/download executable output-id))))
          (finally (link/close! executable)))))))

(deftest retained-contraction-runs-through-the-resident-graph-binder
  (if-not @probe/opencl-available?
    (probe/opencl-skip! "retained typed staged contraction graph")
    (let [{:keys [algorithm body graph]}
          (fixture/production-graph {'a [:arg 0] 'b [:arg 1] 'da [:arg 2]
                                     'db [:arg 3] 'out [:output 0] 'result [:result 0]})
          emitted (emitter/generate-kernel-graph
                   graph :target-dialect :opencl-portable
                   :scheduled-equation-algorithm algorithm :scheduled-equation-body body)
          expected (vec (for [i (range 1 4) j (range 1 6)] (float (* 12 i j))))]
      (gpu/with-gpu-session [sess :ocl:0]
        (gpu/alloc! sess {:a [:byte 288 (byte-array (mapcat #(repeat 96 %) (range 1 4)))]
                          :b [:byte 480 (byte-array (mapcat #(repeat 96 %) (range 1 6)))]
                          :da [:float 9 (float-array (repeat 9 0.5))]
                          :db [:float 15 (float-array (repeat 15 0.25))]
                          :out [:float 15 (float-array (repeat 15 -555.0))]})
        (let [handle (gpu/bind-kernel-graph!
                      sess :typed-contraction emitted
                      {[:arg 0] :a [:arg 1] :b [:arg 2] :da [:arg 3] :db [:output 0] :out} {})]
          (try
            (let [event (gpu/submit-kernel-graph! sess handle)]
              (try
                (gpu/await-event! sess event)
                (is (= expected (vec (gpu/download sess :out))))
                (finally (gpu/release-event! sess event))))
            (finally (gpu/release-kernel-graph! sess handle))))))))
