(ns raster.runtime.native-call-test
  (:require [clojure.test :refer [deftest is]]
            [raster.runtime.native-call :as native]
            [raster.compiler.backend.cpu.codegen :as cpu])
  (:import [java.lang.foreign MemorySegment ValueLayout]
           [java.lang.invoke MethodHandles MethodType]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(defn- callback-handle [callback arity]
  (.bindTo (.findVirtual (MethodHandles/lookup) clojure.lang.IFn "invoke"
                        (MethodType/methodType Object (into-array Class (repeat arity Object))))
           callback))

(deftest overlapping-views-share-one-staged-backing
  (let [library (cpu/compile-source!
                 "void range_alias(double*x,double*y){x[0]+=1;y[0]+=x[1];x[1]+=y[0];}")
        call (cpu/load-kernel library "range_alias" 2 [])
        a (double-array [10.0 20.0 30.0 40.0])
        backing (MemorySegment/ofArray a)
        x (.asSlice backing 8 16) y (.asSlice backing 16 8)]
    (call x y)
    (is (= [10.0 21.0 120.0 40.0] (vec a)))
    (let [restricted (cpu/load-kernel library "range_alias" 2 [] :void {:no-alias? true})]
      (is (= :native-pointer-alias (reason #(restricted x y))))
      (is (= [10.0 21.0 120.0 40.0] (vec a))))))

(deftest disjoint-views-of-one-backing-satisfy-no-alias
  (let [library (cpu/compile-source!
                 "void range_copy(const double*x,double*y){y[0]=x[0]+x[1];y[1]=x[0]-x[1];}")
        call (cpu/load-kernel library "range_copy" 2 [] :void
                              {:pointer-access [:read :write] :no-alias? true})
        a (double-array [2.0 3.0 77.0 88.0 99.0])
        backing (MemorySegment/ofArray a)]
    (call (.asSlice backing 0 16) (.asSlice backing 16 16))
    (is (= [2.0 3.0 5.0 -1.0 99.0] (vec a)))
    (is (= :unsupported-native-heap-view
           (reason #(call (.asSlice backing 0 16)
                          (.asReadOnly (.asSlice backing 16 16))))))))

(deftest readonly-heap-backing-is-not-guessed
  (let [view (.asReadOnly (MemorySegment/ofArray (double-array [1.0])))
        called (atom false)
        handle (callback-handle (fn [_] (reset! called true)) 1)]
    (is (= :unsupported-native-heap-view (reason #(native/pointer-segment view))))
    (is (= :unsupported-native-heap-view
           (reason #(native/invoke-ordinary! handle (object-array [view]) 1 [:read] false))))
    (is (false? @called))))

(deftest copyback-is-confined-to-declared-writable-ranges
  (let [a (double-array [10.0 20.0 30.0 40.0])
        backing (MemorySegment/ofArray a)
        handle (callback-handle
                (fn [^MemorySegment read ^MemorySegment write]
                  ;; Independent host state outside the writable view must not
                  ;; be overwritten by copying the entire staged backing back.
                  (aset a 0 99.0)
                  (.set read ValueLayout/JAVA_DOUBLE 0 (double 777.0))
                  (.set write ValueLayout/JAVA_DOUBLE 0 (double 55.0))
                  nil) 2)]
    (native/invoke-ordinary! handle
                            (object-array [(.asSlice backing 8 8) (.asSlice backing 16 8)])
                            2 [:read :write] true)
    (is (= [99.0 20.0 55.0 40.0] (vec a)))))

(deftest failed-view-call-does-not-publish-and-closes-staging
  (let [a (double-array [1.0 2.0 3.0])
        backing (MemorySegment/ofArray a)
        captured (atom nil)
        handle (callback-handle
                (fn [^MemorySegment x ^MemorySegment y]
                  (reset! captured [x y])
                  (is (= 8 (- (.address y) (.address x))))
                  (.set x ValueLayout/JAVA_DOUBLE 0 (double 88.0))
                  (throw (IllegalStateException. "view failure"))) 2)]
    (is (thrown? IllegalStateException
                 (native/invoke-ordinary! handle
                                          (object-array [(.asSlice backing 0 16)
                                                         (.asSlice backing 8 16)])
                                          2 [:read-write :read-write] false)))
    (is (= [1.0 2.0 3.0] (vec a)))
    (is (every? #(not (.isAlive (.scope ^MemorySegment %))) @captured))))

(deftest byte-ranges-preserve-all-primitive-carriers-and-empty-views
  (let [library (cpu/compile-source!
                 "void range_bytes(unsigned char*x,int n){for(int i=0;i<n;i++)x[i]=0x5a;}")
        call (cpu/load-kernel library "range_bytes" 1 [:int] :void)]
    (doseq [a [(byte-array 4) (short-array 4) (char-array 4) (int-array 4)
               (long-array 4) (float-array 4) (double-array 4)]]
      (let [backing (native/array-segment a)
            size (.byteSize backing)]
        (call (.asSlice backing 1 2) 2)
        (is (= [0 90 90 0] (mapv #(.get backing ValueLayout/JAVA_BYTE (long %)) (range 4))))
        (call (.asSlice backing size 0) 0)
        (is (= 0 (.byteSize (native/pointer-segment (.asSlice backing size 0)))))))
    (let [empty (MemorySegment/ofArray (double-array 0))]
      (call empty 0)
      (is (= 0 (.byteSize empty))))))
