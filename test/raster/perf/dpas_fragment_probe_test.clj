(ns raster.perf.dpas-fragment-probe-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.perf.dpas-fragment-probe :as probe]))

(deftest physical-oracle-has-a-checked-artifact-and-distinguishable-coordinates
  (let [a (probe/probe-artifact)
        coordinates (:coordinates (probe/inputs))]
    (is (= a (artifact/validate! a)))
    (is (= '[A H bits] (:arguments a)))
    (is (= [:float :half :int] (mapv :dtype (:abi a))))
    (is (= [16] (get-in a [:launch :workgroup-size])))
    (is (= [4] (get-in a [:launch :group-count])))
    (is (= 512 (count (set coordinates))))
    (is (every? #(= (float %) (Float/float16ToFloat (Float/floatToFloat16 (float %))))
                coordinates))))

(deftest rounding-oracle-does-not-confuse-poison-zero-or-infinity-with-nan
  (is (probe/half-bits-match? 0x7e00 0xfe01))
  (is (probe/half-bits-match? 0x8000 0x8000))
  (doseq [[expected actual] [[0 0x8000] [0x8000 0] [0x7c00 0xfc00]
                            [0x7e00 0x7c00] [0x7e00 -1] [0x7e00 0x17e00]
                            [1 0] [0x7c00 0x7e00]]]
    (is (not (probe/half-bits-match? expected actual)))))

(deftest device-probe-does-not-swallow-failure-and-releases-allocated-buffers
  (doseq [failure [:upload :launch :download]]
   (let [freed (atom []) allocations (atom 0)
         fail! #(throw (ex-info "injected device failure" {}))]
    (with-redefs [ocl/init! (constantly nil)
                  ocl/register-kernel! (fn [& _])
                  ocl/selected-device-info (constantly {})
                  ocl/make-buffer (fn [& _] (swap! allocations inc))
                  ocl/array->buffer! (fn [& _] (when (= failure :upload) (fail!)))
                  ocl/bind-kernel-call identity
                  ocl/launch-registered-bound! (fn [& _] (when (= failure :launch) (fail!)))
                  ocl/buffer->array (fn [& _] (fail!))
                  ocl/free-buffer! #(swap! freed conj %)]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"injected device failure" (probe/run!)))
      (is (= (if (= failure :upload) [1] [3 2 1]) @freed))))))
