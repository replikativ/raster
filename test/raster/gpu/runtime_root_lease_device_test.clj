(ns raster.gpu.runtime-root-lease-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.dl.gpu-grad-parity :as ze-probe]
            [raster.gpu.device-probe :as ocl-probe]
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
