(ns raster.compiler.passes.parallel.cooperative-segfoldmap-device-test
  "Device acceptance for the generic cooperative per-segment fold-map schedule."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :refer [deftm]]
            [raster.gpu.core :as gpu]
            [raster.gpu.descriptor-fixture :as fixture]
            [raster.gpu.device-probe :as opencl-probe]
            [raster.par]))

(deftm cooperative-row-sum-scale!
  [values :- (Array float), out :- (Array float), rows :- Long, width :- Long] :- Void
  (raster.par/segmented-fold-map!
   [out] [[row rows]] index width
   [[sum 0.0 :float width
     (+ sum (aget values (+ (* row width) index)))
     {:association :implementation-defined}]]
   [(float (* sum (aget values (+ (* row width) index))))]))

(deftest public-cooperative-fold-map-runs-one-workgroup-per-row
  (if @opencl-probe/opencl-available?
    (let [descriptor (pipeline/compile-gpu-program
                      #'cooperative-row-sum-scale! :ocl:0 :dtype :float)]
      (is (= 1 (count (:steps descriptor))))
      (is (= :one-workgroup-per-segment
             (get-in descriptor [:steps 0 :artifact :attributes
                                 :kernel-body :schedule :strategy])))
      (gpu/with-gpu-session [session :ocl:0]
        (doseq [[rows width] [[1 1] [1 17] [1 513] [3 17]]]
          (let [values (float-array
                        (map #(float (/ (- (mod % 13) 6) 7.0))
                             (range (* rows width))))
                output (float-array (* rows width))
                expected
                (vec
                 (mapcat
                  (fn [row]
                    (let [base (* row width)
                          sum (reduce + 0.0
                                      (map #(double (aget values (+ base %)))
                                           (range width)))]
                      (map #(float (* sum (double (aget values (+ base %)))))
                           (range width))))
                  (range rows)))
                arguments [values output (long rows) (long width)]
                program (fixture/instantiate!
                         session descriptor arguments {'values :input 'out :output})]
            (try
              (let [actual (vec (get (fixture/run! program arguments) 'out))]
                (is (= (count expected) (count actual)))
                (is (every? #(< (Math/abs (double %)) 1.0e-4)
                            (map - expected actual))))
              (finally
                (fixture/close! program)))))))
    (opencl-probe/opencl-skip! "cooperative segmented fold-map")))
