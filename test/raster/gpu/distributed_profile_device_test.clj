(ns raster.gpu.distributed-profile-device-test
  "Real local cross-context staging; not network/fabric performance evidence."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.buffer-view :as view]
            [raster.gpu.core :as gpu]
            [raster.gpu.distributed :as runtime]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.test-lifecycle :as lifecycle]))

(defn- check-staged! [device timing-source]
  (let [source-session (gpu/make-session device)]
    (try
      (let [target-session (gpu/make-session device)]
        (try
          (gpu/alloc! source-session {:source [:float 9 nil]})
          (gpu/alloc! target-session {:target [:float 9 nil]})
          (let [source (float-array (range 9)) target (float-array 9)
                staging (atom nil)
                owner (lifecycle/distributed-executable
                       {:state (atom :ready) :staging staging
                        :sessions {:source-worker source-session :target-worker target-session}})
                make-view (fn [id worker]
                            (view/view
                             (view/allocation {:id id :device worker :byte-size 36 :memory-space :device})
                             {:dtype :float :shape [9]}))]
            (let [context (#'runtime/device-observations owner)]
              (is (= #{:source-worker :target-worker} (set (keys context))))
              (is (not= (get-in context [:source-worker :session-id])
                        (get-in context [:target-worker :session-id])))
              (doseq [worker [:source-worker :target-worker]]
                (is (= device (get-in context [worker :hardware-evidence :device-id]))))
              (is (= (get-in context [:source-worker :hardware-evidence])
                     (get-in context [:target-worker :hardware-evidence])))
              (is (= context (#'runtime/device-observations owner))))
            (gpu/upload-range! source-session :source source {:elements 9})
            (let [legs (#'runtime/transfer-host-staged!
                        {:source-worker source-session :target-worker target-session}
                        (make-view :source :source-worker) (make-view :target :target-worker) 16 true staging)]
              (is (= 6 (count legs)))
              (is (= [16 16 16 16 4 4] (mapv :bytes legs)))
              (doseq [leg legs]
                (is (= timing-source (get-in leg [:measurement :timing-source])))
                (is (= (:bytes leg) (get-in leg [:measurement :bytes])))
                (is (<= 0 (get-in leg [:measurement :elapsed-ns])))))
            (gpu/download-range! target-session :target target {:elements 9})
            (is (= (vec source) (vec target)))
            (runtime/close! owner))
          (finally (gpu/close-session! target-session))))
      (finally (gpu/close-session! source-session)))))

(deftest profiled-bounded-staging-between-real-local-contexts
  (if @opencl/opencl-available?
    (check-staged! :ocl:0 :device-event)
    (opencl/opencl-skip! "profiled distributed staging on OpenCL"))
  (if @ze/gpu-available?
    (check-staged! :ze:0 :host-monotonic)
    (ze/gpu-skip! "profiled distributed staging on Level Zero")))
