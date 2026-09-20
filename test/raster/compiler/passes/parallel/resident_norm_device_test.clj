(ns raster.compiler.passes.parallel.resident-norm-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.equation-first :as equation-first]
            [raster.nn :as numerical-nn]
            [raster.dl.nn :as nn]
            [raster.dl.gpu-grad-parity :as gpu-probe]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as link]
            [raster.gpu.descriptor-fixture :as fixture]
            [raster.gpu.device-probe :as opencl-probe]))

(deftest public-rmsnorm-is-one-cooperative-executable
  (doseq [target [:ocl:0 :ze:0]]
    (doseq [norm-var [#'nn/rms-norm-reassociated! #'nn/rms-norm-1row!]]
      (let [descriptor (pipeline/compile-gpu-program norm-var target :dtype :float)]
        (is (= [:executable] (mapv :convention (:steps descriptor))))
        (is (= :one-workgroup-per-segment
               (get-in descriptor [:steps 0 :artifact :attributes
                                   :kernel-body :schedule :strategy])))
        (is (empty? (:allocs descriptor)) "the completed fold stays inside one kernel")))))

(deftest public-softmax-backward-composes-resident-reduction-and-map
  (if @opencl-probe/opencl-available?
    (let [compilation (equation-first/compile
                       #'numerical-nn/softmax-backward {:target :ocl:0 :dtype :double})]
      (doseq [width [1 17 513]]
        (let [dy (double-array (map #(/ (- (mod % 11) 5) 7.0) (range width)))
              weights (mapv #(double (inc (mod % 7))) (range width))
              denominator (reduce + weights)
              s (double-array (map #(/ % denominator) weights))
              dot (reduce + (map * (vec s) (vec dy)))
              expected (mapv #(* %1 (- %2 dot)) (vec s) (vec dy))
              plan (equation-first/lower compilation [dy s])]
          (is (= 0 (get-in plan [:attributes :driver-allocations])))
          (with-open [live (link/instantiate! plan)]
            (dotimes [_ 2]
              (link/run! live)
              (let [actual (vec (link/download live (first (:outputs plan))))]
                (is (= width (count actual)))
                (is (every? #(< (Math/abs (double %)) 1.0e-10)
                            (map - expected actual)))))))))
    (opencl-probe/opencl-skip! "public resident softmax backward composition")))

(deftest public-dense-input-gradient-uses-the-shape-not-the-witness-data
  (if @opencl-probe/opencl-available?
    (let [compilation (equation-first/compile
                       #'numerical-nn/dense-backward-dx {:target :ocl:0 :dtype :double})]
      (doseq [[rows cols] [[2 3] [3 5]]]
        (let [dy (double-array (map #(- (double %) 1.5) (range rows)))
              weights (double-array (map #(/ (- (mod % 7) 3) 8.0) (range (* rows cols))))
              witness (double-array (repeat cols Double/NaN))
              expected (mapv (fn [j]
                               (reduce + (for [i (range rows)]
                                           (* (aget weights (+ (* i cols) j)) (aget dy i)))))
                             (range cols))
              plan (equation-first/lower compilation [dy weights witness])]
          (with-open [live (link/instantiate! plan)]
            (link/run! live)
            (let [actual (vec (link/download live (first (:outputs plan))))]
              (is (= cols (count actual)))
              (is (every? #(< (Math/abs (double %)) 1.0e-10)
                          (map - expected actual))))))))
    (opencl-probe/opencl-skip! "public dense input gradient shape-only input")))

(defn- run-norm! [target]
  (let [descriptor (pipeline/compile-gpu-program
                    #'nn/rms-norm-reassociated! target :dtype :float)]
    (gpu/with-gpu-session [session target]
      (doseq [[rows width] [[1 1] [1 17] [1 513] [3 17]]]
        (let [x (float-array (map #(float (/ (- (mod % 11) 5) 7.0))
                                  (range (* rows width))))
              weights (float-array (map #(float (/ (inc (mod % 7)) 9.0)) (range width)))
              output (float-array (* rows width))
              eps 0.0001
              gain 1.0
              arguments [x weights output (long rows) (long width) eps gain]
              expected
              (vec
               (mapcat
                (fn [row]
                  (let [base (* row width)
                        inverse (/ 1.0
                                   (Math/sqrt
                                    (+ eps
                                       (/ (reduce +
                                                  (map (fn [i]
                                                         (let [v (double (aget x (+ base i)))]
                                                           (* v v)))
                                                       (range width)))
                                          width))))]
                    (mapv (fn [i]
                            (* (aget x (+ base i)) inverse (+ gain (aget weights i))))
                          (range width))))
                (range rows)))
              program (fixture/instantiate! session descriptor arguments
                                            {'x :input 'weight :input 'out :output})]
          (try
            (dotimes [_ 2]
              (let [actual (get (fixture/run! program arguments) 'out)]
                (is (= (* rows width) (count actual)))
                (doseq [[want got] (map vector expected actual)]
                  (is (< (Math/abs (- want got)) 0.00001)))))
            (finally (fixture/close! program))))))))

(deftest opencl-resident-rmsnorm-matches-reference
  (if @opencl-probe/opencl-available?
    (run-norm! :ocl:0)
    (opencl-probe/opencl-skip! "resident RMSNorm result transform")))

(deftest level-zero-resident-rmsnorm-matches-reference
  (if @gpu-probe/gpu-available?
    (run-norm! :ze:0)
    (gpu-probe/gpu-skip! "resident RMSNorm result transform")))
