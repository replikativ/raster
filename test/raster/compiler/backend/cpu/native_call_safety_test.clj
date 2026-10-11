(ns raster.compiler.backend.cpu.native-call-safety-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.cpu.codegen :as cpu]
            [raster.compiler.backend.cpu.aot :as aot]
            [raster.compiler.backend.gpu.c-emit :as c-emit]
            [raster.runtime.native-call :as native-call])
  (:import [java.lang.foreign Arena MemorySegment ValueLayout]
           [java.lang.invoke MethodHandles MethodType]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest critical-permission-is-explicit-and-validated-before-loading
  (doseq [options [{:call-policy :unknown}
                   {:call-policy :critical}
                   {:call-policy :critical :critical-contract {:extremely-short? true}}
                   {:call-policy :critical :critical-contract {:no-upcalls? true}}]]
    (is (= :unsupported-native-call-policy
           (reason #(cpu/load-kernel "nonexistent.so" "probe" 0 [] :double options)))))
  (is (= :unsupported-critical-no-alias
         (reason #(cpu/load-kernel "nonexistent.so" "probe" 0 [] :double
                                  {:call-policy :critical :no-alias? true
                                   :critical-contract {:extremely-short? true :no-upcalls? true}})))))

(deftest ordinary-staging-and-resident-storage-preserve-results
  (let [library (cpu/compile-source!
                 "double safe_probe(double *x, double *y) { x[0] += y[0]; return x[0]; }")
        native (cpu/load-kernel library "safe_probe" 2 [] :double
                                {:pointer-access [:read-write :read] :no-alias? true})
        x (double-array [2.0]) y (double-array [3.0])]
    (is (= 5.0 (native x y)))
    (is (= [5.0] (vec x)))
    (is (= [3.0] (vec y)))
    (is (= :native-pointer-alias (reason #(native x x))))
    (is (= [5.0] (vec x)))
    (let [aliasing (cpu/load-kernel library "safe_probe" 2 [] :double)]
      (is (= 10.0 (aliasing x x)))
      (is (= [10.0] (vec x))))
    (is (= 13.0 (native (MemorySegment/ofArray x) y)))
    (is (= [13.0] (vec x)))
    (with-open [arena (Arena/ofConfined)]
      (let [storage (.allocate arena 16 8)
            first-row (.asSlice storage 0 8)
            second-row (.asSlice storage 8 8)]
        (.set first-row ValueLayout/JAVA_DOUBLE 0 (double 2.0))
        (.set second-row ValueLayout/JAVA_DOUBLE 0 (double 3.0))
        (is (= 5.0 (native first-row second-row)))
        (is (= 5.0 (.get first-row ValueLayout/JAVA_DOUBLE 0)))
        (is (= :native-pointer-alias (reason #(native first-row (.asSlice storage 0 16)))))
        (is (= :inaccessible-native-pointer (reason #(native (.asReadOnly first-row) second-row))))
        (is (= :inaccessible-native-pointer
               @(future (reason #(native first-row second-row)))))))
    (let [arena (Arena/ofConfined) closed (.allocate arena 8 8)]
      (.close arena)
      (is (= :inaccessible-native-pointer (reason #(native closed y)))))))

(deftest readonly-repeated-storage-and-explicit-short-critical-call
  (let [library (cpu/compile-source! "double read_pair(const double *x, const double *y) { return x[0]+y[0]; }")
        x (double-array [2.0])
        ordinary (cpu/load-kernel library "read_pair" 2 [] :double
                                 {:pointer-access [:read :read] :no-alias? true})
        critical (cpu/load-kernel library "read_pair" 2 [] :double
                                 {:call-policy :critical
                                  :critical-contract {:extremely-short? true :no-upcalls? true}})]
    (is (= 4.0 (ordinary x x)))
    (is (= 4.0 (critical x x)))
    (is (= [2.0] (vec x)))))

(deftest failed-call-closes-owned-staging-without-publishing-writes
  (let [captured (atom nil)
        callback (fn [^MemorySegment segment]
                   (reset! captured segment)
                   (.set segment ValueLayout/JAVA_DOUBLE 0 (double 99.0))
                   (throw (IllegalStateException. "test failure")))
        handle (.bindTo (.findVirtual (MethodHandles/lookup) clojure.lang.IFn "invoke"
                                     (MethodType/methodType Object (into-array Class [Object])))
                        callback)
        x (double-array [2.0])
        invoke native-call/invoke-ordinary!]
    (is (thrown? IllegalStateException
                 (invoke handle (object-array [(MemorySegment/ofArray x)])
                         1 [:read-write] false)))
    (is (= [2.0] (vec x)))
    (is (false? (.isAlive (.scope ^MemorySegment @captured))))))

(deftest default-ordinary-call-allows-general-work-without-a-critical-claim
  (let [library (cpu/compile-source!
                 "long long ordinary_loop(long long n) { volatile long long sum=0; for(long long i=0;i<n;i++) sum+=i; return sum; }")
        native (cpu/load-kernel library "ordinary_loop" 0 [:long] :long)]
    (is (= 124999750000 (native 500000)))
    (is (= 0 (native 0)))))

(deftest opaque-helper-writes-are-not-silently-readonly
  (let [writes (ns-resolve 'raster.compiler.backend.cpu.aot 'written-pointer-syms)]
    (with-redefs [c-emit/collect-gpu-fn-calls (constantly [])]
      (is (= #{'out} (writes '(aset out 0 1.0) nil ['input 'out]))))
    (with-redefs [c-emit/collect-gpu-fn-calls (constantly [{:sym 'opaque}])]
      (is (= #{'input 'out} (writes '(opaque input out) nil ['input 'out]))))))
