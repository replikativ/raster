(ns raster.gpu.compiled-output-lease-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.arrays :as arrays]
            [raster.core :refer [deftm]]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.value :as value]
            [raster.par :as par]))

(deftm leased-copy!
  [x :- (Array float) out :- (Array float) n :- Long] :- Void
  (par/map-void! i n
    (arrays/aset out i (arrays/aget x i))))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest typed-compiled-invocation-pins-its-real-opencl-output
  (if-not @opencl/opencl-available?
    (opencl/opencl-skip! "typed Compiled output lease")
    (let [first-input (float-array [1.0 2.0 3.0 4.0])
          second-input (float-array [5.0 6.0 7.0 8.0])
          artifact (compiled/compile
                    #'leased-copy! [first-input (float-array 4) 4]
                    {:target :ocl:0 :compiler :equation-first
                     :outputs '[out] :on-non-resident :throw})]
      (try
        (let [lease (compiled/invoke-leased artifact {:x first-input})
              output (first (vals @lease))]
          (try
            (is (value/live? output))
            (is (= [1.0 2.0 3.0 4.0] (vec (value/->host output))))
            (is (= :link-output-lease-active
                   (reason #(artifact {:x second-input}))))
            (is (= :link-output-lease-active
                   (reason #(compiled/close! artifact))))
            (is (= [1.0 2.0 3.0 4.0] (vec (value/->host output))))
            (finally (.close ^java.io.Closeable lease)))
          (is (not (value/live? output)))
          (is (= :compiled-output-lease-released (reason #(deref lease)))))
        (with-open [lease (compiled/invoke-leased artifact {:x second-input})]
          (is (= [5.0 6.0 7.0 8.0]
                 (vec (value/->host (first (vals @lease)))))))
        (finally (compiled/close! artifact))))))
