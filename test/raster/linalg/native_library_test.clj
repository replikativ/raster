(ns raster.linalg.native-library-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.linalg.native-library :as native-library]
            [raster.linalg.blas :as blas]
            [raster.linalg.lapack :as lapack])
  (:import [java.lang.foreign Arena FunctionDescriptor Linker MemoryLayout
            MemorySegment SymbolLookup ValueLayout]
           [java.lang.invoke MethodHandles MethodType]
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

(deftest ci-native-dependencies-are-functional
  ;; This namespace has no optional-provider fixture: CI must not skip the
  ;; tests merely because the packages it explicitly installs are unusable.
  ;; Laptop and compiler-only jobs do not require an installed numerical stack.
  (when (= "1" (System/getenv "RASTER_EXPECT_NATIVE_LIBRARIES"))
    (is (boolean (blas/available?)) "CI requires functional LP64 BLAS")
    (is (true? (lapack/available?)) "CI requires functional LP64 LAPACK")))

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

(defn- lookup-with-symbols [symbols]
  (reify SymbolLookup
    (find [_ name]
      (if (contains? symbols name) (Optional/of MemorySegment/NULL) (Optional/empty)))))

(deftest explicit-provider-never-falls-back
  (let [calls (atom [])
        symbols #{"cblas_dgemm" "dgesdd_" "LAPACKE_dgeqrf"}
        lookup (lookup-with-symbols symbols)
        load-var (ns-resolve 'raster.linalg.native-library 'load-library)
        loader-var (ns-resolve 'raster.linalg.native-library 'loader-lookup)]
    (with-redefs [native-library/explicit-path (constantly "/pinned/openblas.so")
                  native-library/openblas-config (constantly "OpenBLAS 0.3.32 DYNAMIC_ARCH")]
      (with-redefs-fn
        {load-var (fn [path] (swap! calls conj path) lookup)
         loader-var (fn [] (throw (AssertionError. "explicit pin queried preloaded libraries")))}
        #(do
           (doseq [symbol symbols]
             (is (identical? lookup (native-library/find-library symbol ["/fallback.so"]))))
           (is (= (repeat 3 "/pinned/openblas.so") @calls))
           (is (= :explicit-native-symbol-unavailable
                  (:reason (ex-data
                             (try (native-library/find-library "absent" ["/fallback.so"])
                                  (catch clojure.lang.ExceptionInfo e e)))))))))
    (doseq [[configuration reason] [[nil :explicit-native-provider-unverified]
                                    ["" :explicit-native-provider-unverified]
                                    ["OpenBLAS 0.3.32 USE64BITINT" :native-integer-abi-mismatch]]]
      (with-redefs [native-library/explicit-path (constantly "/pinned/openblas.so")
                    native-library/openblas-config (constantly configuration)]
        (with-redefs-fn {load-var (constantly lookup)}
          #(is (= reason (:reason (ex-data
                                    (try (native-library/find-library "cblas_dgemm" ["/fallback.so"])
                                         (catch clojure.lang.ExceptionInfo e e)))))))))
    (reset! calls [])
    (with-redefs [native-library/explicit-path (constantly "/missing.so")]
      (with-redefs-fn {load-var (fn [path] (swap! calls conj path)
                                 (throw (IllegalArgumentException. "not found")))}
        #(do
           (is (= :explicit-native-library-unavailable
                  (:reason (ex-data
                             (try (native-library/find-library "cblas_dgemm" ["/fallback.so"])
                                  (catch clojure.lang.ExceptionInfo e e))))))
           (is (= ["/missing.so"] @calls)))))))

(deftest default-discovery-keeps-preloaded-precedence
  (let [lookup (lookup-with-symbols #{"cblas_dgemm"})]
    (with-redefs [native-library/explicit-path (constantly nil)]
      (with-redefs-fn
        {(ns-resolve 'raster.linalg.native-library 'loader-lookup) (constantly lookup)
         (ns-resolve 'raster.linalg.native-library 'load-library)
         (fn [_] (throw (AssertionError. "preloaded provider should win")))}
        #(is (identical? lookup (native-library/find-library "cblas_dgemm" ["/fallback.so"])))))))

(deftest default-discovery-skips-unusable-candidates
  (let [lookup (lookup-with-symbols #{"cblas_dgemm"})
        empty-lookup (lookup-with-symbols #{}) calls (atom [])]
    (with-redefs [native-library/explicit-path (constantly nil)]
      (with-redefs-fn
        {(ns-resolve 'raster.linalg.native-library 'loader-lookup) (constantly empty-lookup)
         (ns-resolve 'raster.linalg.native-library 'load-library)
         (fn [path] (swap! calls conj path)
           (case path "missing" (throw (IllegalArgumentException. "not found"))
                 "wrong-symbol" empty-lookup "valid" lookup))}
        #(do
           (is (identical? lookup (native-library/find-library "cblas_dgemm"
                                                              ["missing" "wrong-symbol" "valid"])))
           (is (= ["missing" "wrong-symbol" "valid"] @calls))
           (is (nil? (native-library/find-library "absent" ["wrong-symbol"]))))))))

(deftest explicit-provider-bypasses-mkl-preference
  (let [lookup (lookup-with-symbols #{"cblas_dgemm"}) calls (atom [])]
    (with-redefs [native-library/explicit-path (constantly "/pinned/openblas.so")
                  native-library/select-library (fn [symbol _ path]
                                                  (swap! calls conj [symbol path])
                                                  {:lookup lookup :source :explicit :path path})]
      (with-redefs-fn
        {(ns-resolve 'raster.linalg.blas 'try-load-mkl)
         (fn [] (throw (AssertionError. "explicit OpenBLAS must bypass MKL")))}
        #(let [[library backend evidence] ((ns-resolve 'raster.linalg.blas 'find-blas))]
           (is (= [lookup :openblas] [library backend]))
           (is (= {:source :explicit :path "/pinned/openblas.so"} evidence))))
      (is (= [["cblas_dgemm" "/pinned/openblas.so"]] @calls)))))

(deftest retained-selection-evidence
  (let [lookup (lookup-with-symbols #{"cblas_dgemm"})]
    (with-redefs [native-library/openblas-config (constantly "OpenBLAS 0.3.32 DYNAMIC_ARCH")]
      (let [selection (native-library/selected-library lookup :searched "/selected.so" "cblas_dgemm")]
        (is (identical? lookup (:lookup selection)))
        (is (= {:status :selected :source :searched :path "/selected.so"
                :required-symbol "cblas_dgemm" :advertised-openblas-integer-abi :lp64
                :configuration "OpenBLAS 0.3.32 DYNAMIC_ARCH" :abi-evidence :openblas-build-config}
               (native-library/selection-status (constantly selection))))))
    (with-redefs [native-library/openblas-config (constantly nil)]
      (let [info (native-library/selection-status
                   #(native-library/selected-library lookup :preloaded nil "cblas_dgemm"))]
        (is (= :unknown (:advertised-openblas-integer-abi info)))
        (is (not (contains? info :path)))
        (is (not (contains? info :configuration)))))))

(deftest selection-status-distinguishes-absence-and-failure
  (is (= {:status :absent} (native-library/selection-status (constantly nil))))
  (let [data {:reason :native-integer-abi-mismatch :expected :lp64 :actual :ilp64}]
    (is (= (assoc data :status :rejected :message "bad ABI")
           (native-library/selection-status #(throw (ex-info "bad ABI" data))))))
  (is (= {:status :error :error-class "java.lang.IllegalStateException" :message "discovery failed"}
         (native-library/selection-status #(throw (IllegalStateException. "discovery failed"))))))

(deftest public-info-reports-retained-provider-not-current-property
  (let [info {:status :selected :source :explicit :path "/original.so"
              :advertised-openblas-integer-abi :lp64 :configuration "OpenBLAS test" :backend :openblas}]
    (with-redefs [native-library/explicit-path (constantly "/changed.so")]
      (with-redefs-fn {(ns-resolve 'raster.linalg.blas 'blas-state) (delay [(Object.) :openblas info])}
        #(is (= info (blas/library-info))))))
  (let [fortran {:status :selected :lookup (Object.) :source :searched :advertised-openblas-integer-abi :unknown}
        rejected {:reason :explicit-native-symbol-unavailable :symbol "LAPACKE_dgeqrf"}]
    (with-redefs-fn {(ns-resolve 'raster.linalg.lapack 'openblas) (delay fortran)
                    (ns-resolve 'raster.linalg.lapack 'lapacke) (delay (throw (ex-info "missing QR" rejected)))
                    #'lapack/set-num-threads! (fn [_] (throw (AssertionError. "diagnostics must not configure threads")))
                    #'lapack/dsyevd! (fn [& _] (throw (AssertionError. "diagnostics must not run a solver")))}
      #(is (= {:fortran (dissoc fortran :lookup)
               :lapacke (assoc rejected :status :rejected :message "missing QR")}
              (lapack/library-info))))))

(deftest lp64-integer-arguments-do-not-wrap
  (doseq [value [2147483648 -2147483649]]
    (is (thrown? ArithmeticException ((ns-resolve 'raster.linalg.lapack 'int-seg) value))))
  ;; A harmless zero-argument handle replaces native GEMM. Overflow must throw
  ;; before invocation; a wrapped value would instead reach a wrong-arity call.
  (let [handle (MethodHandles/empty (MethodType/methodType Void/TYPE))]
    (with-redefs-fn {(ns-resolve 'raster.linalg.blas 'dgemm-mh) (delay handle)}
      #(is (thrown? ArithmeticException
                    (blas/dgemm! (double-array 1) (double-array 1) (double-array 1)
                                 2147483648 1 1 1.0 0.0))))))
