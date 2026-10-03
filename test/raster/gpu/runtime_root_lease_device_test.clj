(ns raster.gpu.runtime-root-lease-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.dl.gpu-grad-parity :as ze-probe]
            [raster.gpu.device-probe :as ocl-probe]
            [raster.compiler.backend.gpu.storage-representation :as probe]
            [raster.compiler.backend.gpu.kernel-body-opencl :as emitter]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-call :as call]
            [raster.gpu.runtime-root :as root]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.ze-runtime :as ze]))

(defn- run-buffer-lease-case! [backend]
  (let [namespace (if (= :ocl backend) 'raster.gpu.ocl-runtime 'raster.gpu.ze-runtime)
        v #(ns-resolve namespace %)]
    ((v 'ensure-init!))
    (let [state @(v 'state) before (root/lease-count state)
          buffer ((v 'buffer-of-array) (float-array [0 1 2 3 4 5 6 7]))
          view-slot (volatile! nil)]
      (try
        (let [view ((v 'slice-buffer) buffer 0 16 :float)]
          (vreset! view-slot view)
          (is (= (+ before (if (= :ocl backend) 2 1)) (root/lease-count state)))
          (when (= :ocl backend)
            ((v 'free-buffer!) buffer)
            (is (= (inc before) (root/lease-count state))))
          (is (= [0.0 1.0 2.0 3.0]
                 (vec ((v (if (= :ocl backend) 'buffer->array 'buffer->float-array)) view))))
          (when (= :ze backend)
            (is (identical? (cleanup/lifetime-owner buffer) (cleanup/lifetime-owner view)))))
        (finally
          (when-let [owner (::cleanup/owner @view-slot)] (cleanup/release! owner))
          (cleanup/release! (::cleanup/owner buffer))))
      (is (= before (root/lease-count state))))))

(deftest opencl-canonical-buffer-and-sub-buffer-pin-the-root
  (if @ocl-probe/opencl-available?
    (run-buffer-lease-case! :ocl)
    (ocl-probe/opencl-skip! "canonical root buffer/sub-buffer leases")))

(deftest level-zero-pointer-views-share-one-root-lease
  (if @ze-probe/gpu-available?
    (run-buffer-lease-case! :ze)
    (ze-probe/gpu-skip! "canonical root buffer/view leases")))

(defn- run-prepared-lease-case!
  ([backend] (run-prepared-lease-case! backend nil))
  ([backend recording-options]
   (let [namespace (if (= :ocl backend) 'raster.gpu.ocl-runtime 'raster.gpu.ze-runtime)
         v #(ns-resolve namespace %)]
     ((v 'ensure-init!))
     (let [state @(v 'state) before (root/lease-count state)
           name (str "rstr_prepared_lease_" (gensym))
           module (emitter/emit-scalar-module name (probe/kernel-body :float)
                                              {:target-dialect :opencl-portable
                                               :parameter-names {'out "rstr_out"}})
           artifact (artifact/make (assoc (probe/emit-artifact :float :opencl-portable)
                                          :kernel-name name :source (:source module)))
           arena-id ((v 'make-kernel-arena!))
           buffer ((v 'make-buffer) 2 :float)
           prepared-slot (volatile! nil)
           graph-slot (volatile! nil)]
       (try
         ((v 'register-kernel!) (:kernel-name artifact) artifact arena-id)
         (let [prepared ((v 'bind-kernel-call) (call/make artifact [buffer]))]
           (vreset! prepared-slot prepared)
           (when recording-options
             (vreset! graph-slot ((v 'record-graph!) [prepared] recording-options)))
           ((v 'close-kernel-arena!) arena-id)
          ;; Only the output allocation and the independent prepared kernel remain.
           (is (= (+ before (if recording-options 3 2)) (root/lease-count state)))
           (if recording-options
             ((v 'replay-graph!) @graph-slot)
             ((v 'launch-registered-bound!) prepared))
           (is (= (mapv #(Float/intBitsToFloat (int %)) [0x3fabcdef 0x40234567])
                  (vec ((v (if (= :ocl backend) 'buffer->array 'buffer->float-array)) buffer))))
           (when-let [graph @graph-slot]
             ((v 'destroy-graph!) graph)
             (is (= (+ before 2) (root/lease-count state))))
           ((v 'destroy-prepared!) prepared)
           (is (= (inc before) (root/lease-count state))))
         (finally
           (when-let [graph @graph-slot] ((v 'destroy-graph!) graph))
           (when-let [prepared @prepared-slot] ((v 'destroy-prepared!) prepared))
           ((v 'close-kernel-arena!) arena-id)
           (cleanup/release! (::cleanup/owner buffer))))
       (is (= before (root/lease-count state)))))))

(deftest opencl-prepared-kernel-survives-base-registration-retirement
  (if @ocl-probe/opencl-available?
    (run-prepared-lease-case! :ocl)
    (ocl-probe/opencl-skip! "independent prepared kernel root lease")))

(deftest level-zero-prepared-kernel-survives-base-registration-retirement
  (if @ze-probe/gpu-available?
    (run-prepared-lease-case! :ze)
    (ze-probe/gpu-skip! "independent prepared kernel root lease")))

(deftest opencl-recordings-balance-independent-root-pins
  (if @ocl-probe/opencl-available?
    (doseq [profile? [false true]]
      (run-prepared-lease-case! :ocl {:profile? profile?}))
    (ocl-probe/opencl-skip! "recording root leases")))

(deftest level-zero-recordings-balance-independent-root-pins
  (if @ze-probe/gpu-available?
    (doseq [profile? [false true]]
      (run-prepared-lease-case! :ze {:profile? profile?}))
    (ze-probe/gpu-skip! "recording root leases")))
