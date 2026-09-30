(ns raster.gpu.declared-array-storage-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.fixtures.mixed-storage :as storage]
            [raster.dl.gpu-grad-parity :as gp]
            [raster.gpu.compiled :as compiled]
            [raster.gpu.device-probe :as opencl]
            [raster.gpu.value :as value]))

(defn- run-case [target]
  (doseq [compiler [nil :equation-first]]
    (let [artifact (compiled/compile
                    #'storage/mixed-storage!
                    [(float-array 4) (double-array 4) (double-array 4) 4]
                    (cond-> (merge storage/policy
                                   {:target target :dtype :double :outputs '[out]})
                      compiler (assoc :compiler compiler)))]
      (try
        (doseq [weights [[1.25 -2.5 0.125 8.0] [16.0 32.0 -4.0 0.5]]]
          (let [x (float-array weights)
                state (double-array [0.0000000001 1.0 4.0 -2.0])
                expected (double-array 4)]
            (storage/mixed-storage! x state expected 4)
            (is (= (vec expected)
                   (vec (value/->host (:out (artifact {:weights x :state state}))))))))
        (finally (compiled/close! artifact))))))

(defn- run-fold-case [target]
  (doseq [compiler [nil :equation-first]]
    (let [initial (float-array [16777216.0 1.0 1.0 8.0 0.25 0.5 19.0])
          expected (aclone initial)
          artifact (compiled/compile
                    #'storage/mixed-fold-storage! [initial 2 3]
                    (cond-> (merge storage/policy
                                   {:target target :dtype :double :donate '[storage]})
                      compiler (assoc :compiler compiler)))]
      (try
        (dotimes [_ 2]
          (storage/mixed-fold-storage! expected 2 3)
          (is (= (vec expected)
                 (vec (value/->host (:storage' (artifact {})))))
              "Double recurrence precedes each Float store, preserving source order and the tail"))
        (finally (compiled/close! artifact))))))

(defn- run-scale-case [target]
  (let [alpha (+ 1.0 (Math/scalb 1.0 (int -24)))
        x (float-array [1.5 -1.5 0.0])
        expected (mapv #(float (* alpha (double %))) x)
        artifact (compiled/compile
                  #'storage/mixed-scale [alpha x]
                  (merge storage/policy
                         {:compiler :equation-first :target target :dtype :double}))]
    (try
      (is (= expected (vec (storage/mixed-scale alpha x))))
      (dotimes [_ 2]
        (let [outputs (artifact {})
              result (value/->host (:result outputs))]
          (is (= #{:result} (set (keys outputs))))
          (is (= (class x) (class result)))
          (is (= expected (vec result)))))
      (finally (compiled/close! artifact)))))

(deftest mixed-scale-public-jvm-device-storage-boundary
  (if @gp/gpu-available?
    (run-scale-case :ze:0)
    (gp/gpu-skip! "mixed coefficient scale on Level Zero"))
  (if @opencl/opencl-available?
    (run-scale-case :ocl:0)
    (opencl/opencl-skip! "mixed coefficient scale on OpenCL")))

(defn- run-completed-conversion-case [target]
  (doseq [n [0 3 2049]]
    (let [x (double-array (take n (cycle [16777216.0 1.0 1.0])))
          expected (storage/double-reduction-float-result x)
          artifact (compiled/compile
                    #'storage/double-reduction-float-result [x]
                    (merge storage/policy
                           {:compiler :equation-first :target target :dtype :double}))]
      (try
        (dotimes [_ 2]
          (let [outputs (artifact {})
                result (value/->host (:result outputs))]
            (is (= #{:result} (set (keys outputs))))
            (is (= (class (float-array 0)) (class result)))
            (is (= [expected] (vec result)))))
        (finally (compiled/close! artifact))))))

(deftest completed-reduction-conversion-matches-jvm-on-both-backends
  (if @gp/gpu-available?
    (run-completed-conversion-case :ze:0)
    (gp/gpu-skip! "completed reduction conversion on Level Zero"))
  (if @opencl/opencl-available?
    (run-completed-conversion-case :ocl:0)
    (opencl/opencl-skip! "completed reduction conversion on OpenCL")))

(deftest level-zero-mixed-storage-public-compilers-match-jvm
  (if @gp/gpu-available?
    (do (run-case :ze:0) (run-fold-case :ze:0))
    (gp/gpu-skip! "mixed array storage on Level Zero")))

(deftest opencl-mixed-storage-public-compilers-match-jvm
  (if @opencl/opencl-available?
    (do (run-case :ocl:0) (run-fold-case :ocl:0))
    (opencl/opencl-skip! "mixed array storage on OpenCL")))
