(ns raster.sci.keyed-sampling-integration-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.jvp :as jvp]
            [raster.ad.reverse :as rev]
            [raster.compiler.ir.kernel-body :as kernel-body]
            [raster.compiler.pipeline :as pipeline]
            [raster.core :refer [deftm]]
            [raster.gpu.core :as gpu]
            [raster.gpu.descriptor-fixture :as fixture]
            [raster.gpu.device-probe :as opencl-probe]
            [raster.par :as par]
            [raster.sci.distributions :as dist]))

(deftm keyed-normal-block
  [mu :- Double, sigma :- Double, seed :- Long, counter :- Long] :- Double
  (dist/sample (dist/->Normal mu sigma) seed counter))

(deftm keyed-uniform-fill!
  [out :- (Array double), n :- Long, seed :- Long] :- (Array double)
  (par/map! out i n double (par/uniform-open01 seed (long i))))

(deftest keyed-normal-is-reparameterizable
  (let [seed 42 counter 17
        mu 0.3 sigma 1.2
        z (dist/sample (dist/->Normal 0.0 1.0) seed counter)
        [value dmu dsigma dseed dcounter]
        ((rev/value+grad #'keyed-normal-block) mu sigma seed counter)
        [_ tangent] ((jvp/jvp #'keyed-normal-block)
                     mu sigma seed counter 2.0 3.0)]
    (is (< (Math/abs (- value (+ mu (* sigma z)))) 1e-12))
    (is (< (Math/abs (- dmu 1.0)) 1e-12))
    (is (< (Math/abs (- dsigma z)) 1e-12))
    (is (nil? dseed))
    (is (nil? dcounter))
    (is (< (Math/abs (- tangent (+ 2.0 (* 3.0 z)))) 1e-12))))

(deftest keyed-uniform-reaches-typed-gpu-pipeline
  (let [descriptor (pipeline/compile-gpu-program
                    #'keyed-uniform-fill! :ocl:0 :dtype :double)]
    (is (= [:map] (mapv :convention (:steps descriptor))))
    (is (kernel-body/kernel-body?
         (get-in descriptor [:steps 0 :artifact :attributes :kernel-body])))
    (is (empty? (:allocs descriptor)))))

(deftest keyed-uniform-opencl-matches-jvm
  (if @opencl-probe/opencl-available?
    (let [n 32 seed 42
          out (double-array n)
          arguments [out (long n) (long seed)]
          descriptor (pipeline/compile-gpu-program
                      #'keyed-uniform-fill! :ocl:0 :dtype :double)
          expected (mapv #(par/uniform-open01 seed %) (range n))]
      (gpu/with-gpu-session [session :ocl:0]
        (let [program (fixture/instantiate!
                       session descriptor arguments {'out :output})]
          (try
            (let [actual (get (fixture/run! program arguments) 'out)]
              (is (= expected (vec actual))))
            (finally (fixture/close! program))))))
    (opencl-probe/opencl-skip! "keyed uniform map")))
