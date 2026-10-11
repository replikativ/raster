(ns raster.compiler.backend.cpu.native-marshalling-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.cpu.codegen :as cpu]))

(deftest primitive-storage-conversion-retains-width-and-alias
  (let [convert (ns-resolve 'raster.compiler.backend.cpu.codegen 'array-segment)]
    (doseq [[array width] [[(byte-array 3) 1] [(short-array 3) 2]
                           [(char-array 3) 2] [(int-array 3) 4]
                           [(long-array 3) 8] [(float-array 3) 4]
                           [(double-array 3) 8]]]
      (let [segment ^java.lang.foreign.MemorySegment (convert array)]
        (is (= (* 3 width) (.byteSize segment)))
        (is (identical? array (.orElseThrow (.heapBase segment))))))
    (doseq [invalid [nil (boolean-array 1) (object-array 1) [1.0]]]
      (is (= :unsupported-native-pointer
             (try (convert invalid) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest planned-native-call-preserves-mutating-and-mixed-scalar-abi
  (let [source "double plan_probe(double *x, double *y, int i, long long l, float f, double d) { x[0] += i; y[0] += l; return x[0] + y[0] + f + d; }"
        native (cpu/load-kernel (cpu/compile-source! source) "plan_probe"
                                2 [:int :long :float :double] :double)
        x (double-array [1.0]) y (double-array [2.0])]
    (is (= 21.0 (native x y 3 4 5.0 6.0)))
    (is (= [4.0] (vec x)))
    (is (= [6.0] (vec y)))
    (is (= 20.0 (native x x 1 2 0.0 6.0)))
    (is (= [7.0] (vec x)))
    (is (= 4294967310.375 (native x y 0 4294967297 0.25 0.125)))
    (doseq [arguments [[x] [x y 1 2 3.0 4.0 5]]]
      (is (= :native-call-arity
             (try (apply native arguments) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (is (= :unsupported-native-pointer
           (try (native x (boolean-array 1) 1 2 3.0 4.0) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (thrown? ClassCastException (native x y :not-a-number 2 3.0 4.0)))
    (is (= [7.0] (vec x)) "Rejected calls never execute the native mutation")
    (is (= [4294967303.0] (vec y)))))
