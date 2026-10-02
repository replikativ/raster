(ns raster.compiler.backend.gpu.storage-representation-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.storage-representation :as probe]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-body :as body]))

(deftest explicit-oracles-and-rejection
  (is (= [0x78 0x56 0x34 0x12 0x19 0x67 0x45 0x23]
         (probe/expected-bytes :int :little-endian)))
  (is (= [0x3d 0x55 0x42 0x5b] (probe/expected-bytes :half :big-endian)))
  (doseq [dt (keys dtype/dtype-info)]
    (testing (str dt)
      (doseq [order [:little-endian :big-endian]]
        (let [bytes (probe/expected-bytes dt order)]
          (is (= (* 2 (dtype/bytes-of dt)) (count bytes)))
          (is (= (if (= :byte dt) :order-invariant order)
                 (probe/classify-bytes dt bytes)))
          ;; Signed byte observations are normalized, never numerically converted.
          (is (= (probe/classify-bytes dt bytes)
                 (probe/classify-bytes dt (map unchecked-byte bytes))))
          (is (thrown? clojure.lang.ExceptionInfo (probe/classify-bytes dt (pop bytes))))
          (is (thrown? clojure.lang.ExceptionInfo
                       (probe/classify-bytes dt (assoc bytes 0 (bit-xor 1 (first bytes))))))
          (is (thrown? clojure.lang.ExceptionInfo
                       (probe/classify-bytes dt (vec (reverse bytes)))))))))
  (is (= (probe/expected-bytes :float :big-endian)
         (probe/expected-bytes :f32 :big-endian)))
  (is (thrown? clojure.lang.ExceptionInfo (probe/expected-bytes :bogus :big-endian)))
  (is (thrown? clojure.lang.ExceptionInfo (probe/expected-bytes :int :native))))

(deftest observations-cannot-truncate-or-wrap-nonbytes
  (doseq [not-byte [376 -136 120.9 120.0 nil "120" 18446744073709551736N]]
    (is (thrown? clojure.lang.ExceptionInfo
                 (probe/classify-bytes :int
                   (assoc (probe/expected-bytes :int :little-endian) 0 not-byte))))))

(deftest finite-literals-preserve-the-independent-bit-oracles
  (let [values (fn [dt] (mapv #(get-in % [:value :value])
                             (:operations (probe/kernel-body dt))))]
    (is (= [0x3d55 0x425b] (mapv #(bit-and 65535 (Float/floatToFloat16 %)) (values :half))))
    (is (= [0x3fabcdef 0x40234567] (mapv #(Float/floatToRawIntBits %) (values :float))))
    (is (= [0x3ff123456789abcd 0x40023456789abcde]
           (mapv #(Double/doubleToRawLongBits %) (values :double))))))

(deftest probes-use-the-common-typed-emitter
  (doseq [dt (keys dtype/dtype-info)]
    (is (body/kernel-body? (body/validate! (probe/kernel-body dt))))
    (doseq [dialect [:opencl-portable :cuda :hip]]
      (let [a (probe/emit-artifact dt dialect)]
        (is (artifact/kernel-artifact? (artifact/validate! a)))
        (is (= dt (:dtype (first (:abi a)))))
        (is (= '[out] (:arguments a)))))))
