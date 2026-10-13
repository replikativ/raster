(ns raster.compiler.public-hinted-array-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.arrays :as arrays]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.compiler.equation-first :as equation]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.value :as value]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]))

;; AD-generated arrays carry these same retained primitive hints. The public
;; compiler must realize the operation even when the receiver is already typed.
(deftm scale-hinted-array!
  [x :- (Array float), out :- (Array float), n :- Long, scale :- Float] :- (Array float)
  (par/map-void! i n
    (arrays/aset ^floats out i (n/* (arrays/aget ^floats x i) scale)))
  out)

(deftest hinted-array-operations-enter-the-public-typed-source-vertical
  (doseq [target [:cuda:0 :hip:0]]
    (let [compilation (equation/compile #'scale-hinted-array! {:target target :dtype :float})]
      (is (= :none (get-in compilation [:stats :fallback])))
      (is (= 1 (count (:kernels compilation)))))))

(defn- check-device! [target]
  (let [scale (float 0.5)
        initial (float-array [1.0 -2.0 3.0])
        prepared (compiled/lower #'scale-hinted-array! [initial (float-array 3) 3 scale]
                                 {:compiler :equation-first :target target :dtype :float})
        program (compiled/instantiate! prepared)]
    (try
      (doseq [input [initial (float-array [-7.0 0.0 0.125])]]
        (let [expected (mapv #(float (* (double %) (double scale))) input)
              cpu (scale-hinted-array! input (float-array 3) 3 scale)
              actual (value/->host (:result (program {:x input})))]
          (is (= expected (vec cpu)))
          (is (= expected (vec actual)))))
      (finally (compiled/close! program)))))

(deftest hinted-array-opencl-replay-matches-independent-oracle
  (if @opencl/opencl-available?
    (check-device! :ocl:0)
    (opencl/opencl-skip! "public hinted primitive arrays")))

(deftest hinted-array-level-zero-replay-matches-independent-oracle
  (if @ze/gpu-available?
    (check-device! :ze:0)
    (ze/gpu-skip! "public hinted primitive arrays")))
