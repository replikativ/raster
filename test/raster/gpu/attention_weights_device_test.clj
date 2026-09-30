(ns raster.gpu.attention-weights-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.dl.attention :as attention]
            [raster.dl.attention-reference :as reference]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.value :as value]))

(defn- near? [expected actual tolerance]
  (and (= (count expected) (count actual))
       (every? true? (map (fn [x y] (<= (Math/abs (- (double x) (double y)))
                                      (* tolerance (max 1.0 (Math/abs (double x))))))
                         expected actual))))

(defn- run-case [target]
  (doseq [dtype [:float :double]
          [nq nkv n hd materialization?] [[4 4 5 2 false] [4 2 5 2 false]
                                        [2 1 0 2 false] [3 1 1 2 false]
                                        [3 1 1 1 true]]]
    (let [make-array (if (= dtype :float) float-array double-array)
          tolerance (if (= dtype :float) 1e-5 1e-9)
          rng (java.util.Random. 11)
          random-array (fn [length] (make-array (repeatedly length #(.nextGaussian rng))))
          q (if materialization? (make-array (* nq hd)) (random-array (* nq hd)))
          k (if materialization? (make-array (* n nkv hd)) (random-array (* n nkv hd)))
          v (if materialization? (make-array (repeat (* n nkv hd) 7.0))
                (random-array (* n nkv hd)))
          initial (make-array (concat (repeat n (if materialization? 8388608.0 0.25))
                                     [17.0 19.0]))
          expected-sink (aclone initial)
          function (core/resolve-deftm-var #'attention/gqa-decode-attention-weights-resident!
                                          {:dtype dtype})
          artifact (compiled/compile
                    function [q k v (make-array (repeat (* nq hd) Double/NaN))
                              (make-array (repeat (* nq n) Double/NaN))
                              (double-array (repeat nq Double/NaN)) initial n nq nkv hd 0.5]
                    {:compiler :equation-first :target target :dtype :double :inline? true
                     :preserve-declared-array-storage? true :donate '[wsink] :outputs '[out]})]
      (try
        (dotimes [_ 2]
          (let [expected (reference/gqa-decode-with-weights q k v n nq nkv hd 0.5 expected-sink)
                result (artifact {})]
            (is (near? (vec expected) (vec (value/->host (:out result))) tolerance))
            (is (near? (vec expected-sink) (vec (value/->host (:wsink' result))) tolerance))
            (when (and materialization? (= dtype :float))
              (is (= 8388608.0 (double (first (value/->host (:wsink' result)))))
                  "each head's Float sink store rounds separately"))
            (is (= [17.0 19.0] (subvec (vec (value/->host (:wsink' result))) n)))))
        (finally (compiled/close! artifact))))))

(deftest level-zero-resident-alignment-matches-the-host
  (if @gp/gpu-available?
    (run-case :ze:0)
    (gp/gpu-skip! "caller-owned ASR alignment on Level Zero")))

(deftest opencl-resident-alignment-matches-the-host
  (if @opencl/opencl-available?
    (run-case :ocl:0)
    (opencl/opencl-skip! "caller-owned ASR alignment on OpenCL")))
