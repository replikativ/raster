(ns raster.gpu.invocation-profile-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.arrays :as arrays]
            [raster.core :refer [deftm]]
            [raster.dl.gpu-grad-parity :as ze]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.core :as gpu]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.link :as link]
            [raster.gpu.value :as value]
            [raster.par :as par]))

(deftm profile-copy!
  [x :- (Array float) out :- (Array float) n :- Long] :- Void
  (par/map-void! i n (arrays/aset out i (arrays/aget x i))))

(deftest host-profile-preserves-generated-resident-copy-on-local-backends
  (doseq [[target available skip]
          [[:ocl:0 opencl/opencl-available? opencl/opencl-skip!]
           [:ze:0 ze/gpu-available? ze/gpu-skip!]]]
    (if-not @available
      (skip "ordinary invocation host profile")
      (let [first-input (float-array [1 2 3 4])
            next-input (float-array [5 6 7 8])
            expected (float-array 4)
            _ (profile-copy! first-input expected 4)
            artifact (compiled/compile #'profile-copy! [first-input (float-array 4) 4]
                                       {:target target :compiler :equation-first :outputs '[out]
                                        :on-non-resident :throw})
            replay link/run!
            calls (atom 0)]
        (try
          (let [{:keys [outputs report]}
                (with-redefs [gpu/alloc! (fn [& _] (throw (AssertionError. "invocation must not allocate")))
                              gpu/download (fn [& _] (throw (AssertionError. "invocation must not download")))
                              gpu/download-range! (fn [& _] (throw (AssertionError. "invocation must not download")))
                              link/run! (fn [& args] (swap! calls inc) (apply replay args))]
                  (compiled/invoke-profiled artifact {:x first-input}))]
            (is (= 1 @calls))
            (is (= (vec expected) (vec (value/->host (first (vals outputs))))))
            (is (= {:host-upload {:operations 1 :logical-bytes 16}} (:input-refresh report)))
            (is (= #{:graph-resolution :synchronous-backend-replay}
                   (set (keys (:replay-detail-ns report)))))
            (is (<= (reduce + 0 (vals (:replay-detail-ns report)))
                    (get-in report [:phases-ns :replay-host])))
            (is (= (:total-ns report)
                   (+ (:unpartitioned-ns report) (reduce + 0 (vals (:phases-ns report))))))
            (let [normal (artifact {:x next-input})]
              (is (= [5.0 6.0 7.0 8.0] (vec (value/->host (first (vals normal))))))
              (is (not (value/live? (first (vals outputs)))))))
          (finally (compiled/close! artifact)))))))
