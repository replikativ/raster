(ns raster.compiler.backend.cpu.native-alias-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.string :as str]
            [raster.core :refer [deftm]]
            [raster.dl.nn :as nn]
            [raster.runtime.native-call :as native-call]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.backend.cpu.aot :as aot])
  (:import [java.lang.foreign Arena MemorySegment ValueLayout]))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftm mixed-storage! [x :- (Array float) out :- (Array double) n :- Long] :- Void
  (raster.par/map-void! i n (aset out i (double (aget x i)))))

(deftm conditional-pointwise! [x :- (Array double) y :- (Array double)
                              out :- (Array double) n :- Long] :- Void
  (raster.par/map-void! i n
    (let [a (aget x i) b (aget y i)]
      (if (< a b) (aset out i (+ a b)) (aset out i (- a b))))))

(deftest pointwise-admission-is-not-specific-to-gelu
  (let [compiled (aot/compile-aot-c #'conditional-pointwise! :double :no-alias? false)]
    (doseq [alias [:first :second :both]]
      (let [x (double-array [-2 3 -4])
            y (if (= alias :both) x (double-array [1 -5 6]))
            out (if (= alias :second) y x)
            expected (double-array 3)]
        (conditional-pointwise! x y expected 3)
        (compiled x y out 3)
        (is (= (vec expected) (vec out)))))))

(deftest mixed-storage-cannot-admit-an-exact-byte-alias
  (with-open [arena (Arena/ofConfined)]
    (let [region (.allocate arena 16 8)
          before (byte-array 16)]
      (.fill region (byte 42))
      ;; Compilation must fail before this native region can enter a mixed-layout map.
      (is (= :unsupported-native-alias-scope
             (reason #(let [compiled (aot/compile-aot-c #'mixed-storage! :double :no-alias? false)]
                        (compiled region region 2)))))
      (MemorySegment/copy region 0 (MemorySegment/ofArray before) 0 16)
      (is (= (repeat 16 (byte 42)) (seq before))))))

(deftest gelu-exact-aliases-have-a-consistent-native-contract
  (doseq [[dtype array-fn layout] [[:float float-array ValueLayout/JAVA_FLOAT]
                                 [:double double-array ValueLayout/JAVA_DOUBLE]]]
    (let [compiled (pipeline/compile-aot #'nn/gelu-erf! :target :c :dtype dtype :no-alias? false)
          reference (aot/compile-aot-c #'nn/gelu-erf! dtype)
          bytes (.byteSize layout)]
      (is (= :same-region-pointwise (:native-alias-policy (meta compiled))))
      (is (not (str/includes? (:c-source (meta compiled)) "restrict")))
      (doseq [values [[-3.0 -1.0 0.0 1.0 3.0] [2.0 -2.0 0.5 -0.5 4.0]]]
        (let [input (array-fn values) expected (array-fn 5) in-place (array-fn values)
              disjoint (array-fn 5)]
          (reference input expected 5)
          (compiled input disjoint 5)
          (is (= (vec expected) (vec disjoint)))
          (compiled in-place in-place 5)
          (is (= (vec expected) (vec in-place)))
          (is (= :native-pointer-alias (reason #(reference input input 5))))
          (with-open [arena (Arena/ofConfined)]
            (let [native (.allocate arena (* 5 bytes) bytes)
                  ;; ofArray's primitive overload is chosen through the existing boundary.
                  source (native-call/array-segment input)]
              (MemorySegment/copy source 0 native 0 (* 5 bytes))
              (compiled native (.asSlice native 0 (* 5 bytes)) 5)
              (let [actual (array-fn 5)]
                (MemorySegment/copy native 0 (native-call/array-segment actual) 0 (* 5 bytes))
                (is (= (vec expected) (vec actual)))))))))))

(deftest shifted-overlap-is-rejected-before-mutation
  (let [compiled (aot/compile-aot-c #'nn/gelu-erf! :double :no-alias? false)
        values (double-array [1 2 3 4 5 6])
        backing (MemorySegment/ofArray values)]
    (is (= :native-pointer-alias
           (reason #(compiled (.asSlice backing 0 40) (.asSlice backing 8 40) 5))))
    (is (= [1.0 2.0 3.0 4.0 5.0 6.0] (vec values)))
    (with-open [arena (Arena/ofConfined)]
      (let [native (.allocate arena 48 8)]
        (MemorySegment/copy backing 0 native 0 48)
        (is (= :native-pointer-alias
               (reason #(compiled (.asSlice native 0 40) (.asSlice native 8 40) 5))))
        (let [actual (double-array 6)]
          (MemorySegment/copy native 0 (MemorySegment/ofArray actual) 0 48)
          (is (= (vec values) (vec actual))))))))

(deftest alias-admission-precedes-disjoint-storage-rewrites
  (let [verify @#'aot/verify-pointwise-alias-scope!
        env {'x 'doubles 'y 'doubles 'n 'long}]
    (doseq [source '[(System/arraycopy x 0 y 1 n)
                     (raster.par/map-void! i n (clojure.core/aset y i (clojure.core/aget x (+ i 1))))
                     (raster.par/map-void! i n (let* [v (clojure.core/aset y i 2.0)] (clojure.core/aget x i)))
                     (raster.par/map-void! i n (opaque-helper x y i))]]
      (is (= :unsupported-native-alias-scope (reason #(verify source '[x y n] env)))))
    (is (some? (verify '(raster.par/map-void! i n
                         (clojure.core/aset y i (clojure.core/aget x i))) '[x y n] env)))
    (is (= :unsupported-native-alias-scope
           (reason #(verify '(raster.par/map-void! i n
                               (clojure.core/aset y i (clojure.core/aget x i)))
                            '[x y n] (assoc env 'x 'floats)))))))

(deftest alias-policy-is-not-a-truthiness-switch
  (doseq [policy [nil :unsafe 0]]
    (is (= :invalid-native-alias-policy
           (reason #(aot/compile-aot-c #'nn/gelu-erf! :float :no-alias? policy))))))
