(ns raster.linalg.native-library-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.linalg.native-library :as native-library]
            [raster.linalg.blas :as blas]
            [raster.linalg.lapack :as lapack])
  (:import [java.lang.foreign Arena FunctionDescriptor Linker MemoryLayout
            MemorySegment SymbolLookup ValueLayout]
           [java.lang.invoke MethodHandles]
           [java.util Optional]))

(deftest integer-abi-admission
  ;; No numerical downcall or installed BLAS is required for these gates.
  (let [lookup (Object.)]
    (doseq [config [nil "OpenBLAS 0.3.32 DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64"]]
      (with-redefs [native-library/openblas-config (constantly config)]
        (is (identical? lookup (native-library/require-lp64! lookup)))))
    (doseq [config ["OpenBLAS 0.3.32 USE64BITINT DYNAMIC_ARCH"
                    "USE64BITINT" "OpenBLAS 0.3.32 USE64BITINT"]]
      (with-redefs [native-library/openblas-config (constantly config)]
        (let [e (try (native-library/require-lp64! lookup)
                     (catch clojure.lang.ExceptionInfo e e))]
          (is (= {:reason :native-integer-abi-mismatch
                  :expected :lp64 :actual :ilp64
                  :provider :openblas :configuration config}
                 (ex-data e))))))
    (testing "a substring is not the documented build flag"
      (with-redefs [native-library/openblas-config (constantly "NOT_USE64BITINT")]
        (is (identical? lookup (native-library/require-lp64! lookup)))))))

(deftest optional-provider-metadata
  (is (nil? (native-library/openblas-config nil)))
  (is (nil? (native-library/require-lp64! nil)))
  (is (nil? (native-library/openblas-config
             (reify SymbolLookup
               (find [_ _] (Optional/empty)))))))

(deftest availability-does-not-hide-admission-errors
  (doseq [[namespace state available] [['raster.linalg.blas 'blas-state blas/available?]
                                      ['raster.linalg.lapack 'openblas lapack/available?]]]
    (let [error (ex-info "incompatible provider" {:reason :native-integer-abi-mismatch})]
      (with-redefs-fn {(ns-resolve namespace state) (delay (throw error))}
        #(is (identical? error
                         (try (available)
                              (catch clojure.lang.ExceptionInfo e e))))))))

(deftest configuration-downcall
  ;; Exercise the actual address-returning ABI using a native upcall stub,
  ;; without loading a BLAS provider or compiling a fixture library.
  (with-open [arena (Arena/ofConfined)]
    (doseq [[config expected] [["OpenBLAS 0.3.32 DYNAMIC_ARCH" "OpenBLAS 0.3.32 DYNAMIC_ARCH"]
                               ["OpenBLAS 0.3.32 USE64BITINT" "OpenBLAS 0.3.32 USE64BITINT"]
                               [nil :invalid-native-library-metadata]]]
      (let [address (if config (.allocateFrom arena ^String config) MemorySegment/NULL)
            handle (MethodHandles/constant MemorySegment address)
            descriptor (FunctionDescriptor/of ValueLayout/ADDRESS
                                              (into-array MemoryLayout []))
            stub (.upcallStub (Linker/nativeLinker) handle descriptor arena
                              (into-array java.lang.foreign.Linker$Option []))
            lookup (reify SymbolLookup
                     (find [_ name]
                       (if (= name "openblas_get_config")
                         (Optional/of stub) (Optional/empty))))]
        (if config
          (is (= expected (native-library/openblas-config lookup)))
          (is (= expected
                 (:reason (ex-data (try (native-library/openblas-config lookup)
                                       (catch clojure.lang.ExceptionInfo e e)))))))))))
