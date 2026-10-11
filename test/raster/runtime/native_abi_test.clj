(ns raster.runtime.native-abi-test
  (:require [clojure.test :refer [deftest is]]
            [raster.runtime.native-call :as native]
            [raster.compiler.backend.cpu.codegen :as cpu])
  (:import [java.lang.foreign Arena FunctionDescriptor Linker Linker$Option
            MemoryLayout MemorySegment SymbolLookup ValueLayout SegmentAllocator]
           [java.lang.invoke MethodHandles MethodType]))

(def ^:private fixture
  (delay
    (SymbolLookup/libraryLookup
     ^String (cpu/compile-source!
              "int interleaved(int n,const double*x,double alpha,double*y,long long bias){y[0]=alpha*x[0]+bias;return n;}
               void references(const int*n,const double*x,double*work,int*info){for(int i=0;i<*n;i++)work[i]=2*x[i];*info=7;}
               double scalar_only(double x,int n){return x+n;}")
     (Arena/global))))

(defn- handle [name result parameters]
  (let [layouts (into-array MemoryLayout parameters)
        descriptor (if result (FunctionDescriptor/of result layouts)
                       (FunctionDescriptor/ofVoid layouts))]
    (.downcallHandle (Linker/nativeLinker) (.orElseThrow (.find ^SymbolLookup @fixture name))
                     descriptor (into-array Linker$Option []))))

(defn- interleaved-handle []
  (handle "interleaved" ValueLayout/JAVA_INT
          [ValueLayout/JAVA_INT ValueLayout/ADDRESS ValueLayout/JAVA_DOUBLE
           ValueLayout/ADDRESS ValueLayout/JAVA_LONG]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest interleaved-abi-retains-slot-order-and-shared-view-state
  (let [call (native/prepare-call (interleaved-handle) [:read :write])
        a (double-array [2.0 99.0 77.0])
        backing (MemorySegment/ofArray a)]
    (is (= 3 (call [(int 3) (.asSlice backing 0 8) (double 4.0)
                   (.asSlice backing 8 8) (long 1)])))
    (is (= [2.0 9.0 77.0] (vec a)))
    (is (= 4 (call [(int 4) a (double 2.0) a (long 3)])))
    (is (= [7.0 9.0 77.0] (vec a)))
    (with-open [arena (Arena/ofConfined)]
      (let [output (.allocate arena 8 8)]
        (is (= 5 (call [(int 5) a (double 3.0) output (long 2)])))
        (is (= 23.0 (.get output ValueLayout/JAVA_DOUBLE 0)))))))

(deftest reference-abi-publishes-workspace-and-nonzero-info
  (let [call (native/prepare-call
              (handle "references" nil (repeat 4 ValueLayout/ADDRESS))
              [:read :read :write :write])
        n (int-array [2]) x (double-array [1.0 3.0])
        workspace (double-array [0.0 0.0 99.0]) info (int-array [0])]
    (call [n x workspace info])
    (is (= [2.0 6.0 99.0] (vec workspace)))
    (is (= [7] (vec info)) "Native status is interpreted after successful copyback")
    (is (= [2] (vec n)))
    (is (= [1.0 3.0] (vec x)))))

(deftest scalar-only-handle-needs-no-staging
  (let [call (native/prepare-call
              (handle "scalar_only" ValueLayout/JAVA_DOUBLE
                      [ValueLayout/JAVA_DOUBLE ValueLayout/JAVA_INT]) [])]
    (is (= 5.5 (call [(double 2.5) (int 3)])))
    (is (= :native-call-arity (reason #(call [(double 2.5)]))))))

(deftest malformed-plans-and-invocation-arity-refuse-before-execution
  (let [mh (interleaved-handle)
        call (native/prepare-call mh [:read :write])
        x (double-array [1.0]) y (double-array [99.0])]
    (doseq [access [[] [:read] [:read :unknown] [:read :write :read]]]
      (is (= :invalid-native-pointer-access (reason #(native/prepare-call mh access)))))
    (doseq [arguments [[(int 2) x (double 3.0) y]
                       [(int 2) x (double 3.0) y (long 1) :extra]]]
      (is (= :native-call-arity (reason #(call arguments)))))
    (doseq [slots [[1 1] [-1] [5] [:pointer] 6 -1]]
      (is (= :invalid-native-pointer-slots
             (reason #(native/invoke-ordinary! mh (object-array [(int 2) x (double 3.0) y (long 1)])
                                               slots (repeat (if (vector? slots) (count slots) 0) :read)
                                               false)))))
    (is (= [99.0] (vec y)))
    (doseq [signature [(MethodType/methodType MemorySegment)
                       (MethodType/methodType Void/TYPE (into-array Class [SegmentAllocator]))
                       (MethodType/methodType Void/TYPE (into-array Class [Object]))]]
      (is (= :unsupported-native-handle-shape
             (reason #(native/prepare-call (MethodHandles/empty signature) [])))))))
