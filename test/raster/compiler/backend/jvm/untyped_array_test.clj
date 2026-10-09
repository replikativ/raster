(ns raster.compiler.backend.jvm.untyped-array-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.compiler.pipeline :as pipeline]))

(deftm unknown-read [array :- Object index :- Long] :- Object
  (clojure.core/aget array index))

(deftm unknown-write! [array :- Object index :- Long value :- Object] :- Object
  (clojure.core/aset array index value))

(deftm unknown-write-void! [array :- Object index :- Long value :- Object] :- Object
  (clojure.core/aset array index value)
  (clojure.core/aget array index))

(deftm unknown-read-effects [array-fn :- Object index-fn :- Object] :- Object
  (clojure.core/aget (array-fn) (index-fn)))

(deftm unknown-write-effects [array-fn :- Object index-fn :- Object value-fn :- Object] :- Object
  (clojure.core/aset (array-fn) (index-fn) (value-fn)))

(deftm typed-read [array :- (Array float) index :- Long] :- Float
  (clojure.core/aget array index))

(deftm typed-write! [array :- (Array float) index :- Long value :- Float] :- Float
  (clojure.core/aset array index value))

(deftm nested-read [array :- (Array Object) outer :- Long inner :- Long] :- Object
  (clojure.core/aget array outer inner))

(deftm nested-write! [array :- (Array Object) outer :- Long inner :- Long value :- Object] :- Object
  (clojure.core/aset array outer inner value))

(defn- observations [value]
  {:class (class value)
   :value (cond
            (instance? Float value) (Float/floatToRawIntBits value)
            (instance? Double value) (Double/doubleToRawLongBits value)
            :else value)})

(defn- cases []
  [{:array (float-array [1.0 2.0]) :value (Float/intBitsToFloat (unchecked-int 0x80000000))}
   {:array (double-array [1.0 2.0]) :value (Double/longBitsToDouble Long/MIN_VALUE)}
   {:array (float-array [1.0 2.0]) :value (Float/intBitsToFloat 0x7fc01234)}
   {:array (double-array [1.0 2.0]) :value (Double/longBitsToDouble 0x7ff8000000001234)}
   {:array (long-array [1 2]) :value Long/MIN_VALUE}
   {:array (int-array [1 2]) :value Integer/MIN_VALUE}
   {:array (short-array [1 2]) :value (short -3)}
   {:array (byte-array [1 2]) :value (byte -4)}
   {:array (char-array [\a \b]) :value \z}
   {:array (boolean-array [true false]) :value true}
   {:array (object-array [:a :b]) :value :replacement}
   {:array (into-array String ["a" "b"]) :value "replacement"}])

(defn- error-class [f]
  (try (f) nil (catch Throwable error (class error))))

(deftest unknown-storage-read-write-matches-clojure-runtime-dispatch
  (doseq [tier [:jit :aot]]
    (let [read-fn (if (= :jit tier) unknown-read (pipeline/compile-aot #'unknown-read))
          write-fn (if (= :jit tier) unknown-write! (pipeline/compile-aot #'unknown-write!))
          void-fn (if (= :jit tier) unknown-write-void! (pipeline/compile-aot #'unknown-write-void!))
          runtime-read (deref #'clojure.core/aget)
          runtime-write (deref #'clojure.core/aset)]
      (doseq [{:keys [array value]} (cases)]
        (testing (str tier " " (class array))
          (is (= (observations (runtime-read array 1)) (observations (read-fn array 1))))
          (let [clone-array (deref #'clojure.core/aclone)
                expected-array (clone-array array)
                actual-array (clone-array array)
                void-array (clone-array array)
                expected (runtime-write expected-array 1 value)]
            (is (= (observations expected) (observations (write-fn actual-array 1 value))))
            (is (= (observations (runtime-read expected-array 1))
                   (observations (runtime-read actual-array 1))))
            (is (= (observations expected) (observations (void-fn void-array 1 value))))
            (is (= (observations (runtime-read expected-array 1))
                   (observations (runtime-read void-array 1)))))
          (doseq [index [-1 2]]
            (is (= (error-class #(runtime-read array index)) (error-class #(read-fn array index))))
            (is (= (error-class #(runtime-write array index value))
                   (error-class #(write-fn array index value)))))))
      (doseq [array [nil :not-an-array (float-array 2)]]
        (is (= (error-class #(runtime-write array 0 :not-a-float))
               (error-class #(write-fn array 0 :not-a-float))))))))

(deftest unknown-storage-operands-evaluate-once-in-source-order
  (doseq [tier [:jit :aot]]
    (let [read-fn (if (= :jit tier) unknown-read-effects
                    (pipeline/compile-aot #'unknown-read-effects))
          write-fn (if (= :jit tier) unknown-write-effects
                     (pipeline/compile-aot #'unknown-write-effects))
          trace (atom [])
          array (float-array [1.0 2.0])
          array-fn #(do (swap! trace conj :array) array)
          index-fn #(do (swap! trace conj :index) 1)
          value-fn #(do (swap! trace conj :value) (float 3.0))]
      (is (= (float 2.0) (read-fn array-fn index-fn)))
      (is (= [:array :index] @trace))
      (reset! trace [])
      (is (= (float 3.0) (write-fn array-fn index-fn value-fn)))
      (is (= [:array :index :value] @trace))
      (reset! trace [])
      (is (some? (error-class #(write-fn (fn [] (swap! trace conj :array) :not-an-array)
                                       index-fn value-fn))))
      (is (= [:array :index :value] @trace)))))

(deftest retained-array-types-keep-direct-jvm-load-store-paths
  (doseq [tier [:jit :aot]]
    (let [read-fn (if (= :jit tier) typed-read (pipeline/compile-aot #'typed-read :dtype :float))
          write-fn (if (= :jit tier) typed-write! (pipeline/compile-aot #'typed-write! :dtype :float))
          array (float-array [1.0 2.0])]
      ;; Finish lazy compilation before replacing runtime array dispatch with tripwires.
      (read-fn array 0)
      (write-fn array 0 (float 1.0))
      (with-redefs [clojure.core/aget (fn [& _] (throw (ex-info "typed read used boxed dispatch" {})))
                    clojure.core/aset (fn [& _] (throw (ex-info "typed write used boxed dispatch" {})))]
        (is (= (float 2.0) (read-fn array 1)))
        (is (= (float 4.0) (write-fn array 1 (float 4.0))))
        (is (= (float 4.0) (read-fn array 1)))))))

(deftest nested-array-arity-does-not-drop-inner-indices-or-values
  (doseq [tier [:jit :aot]]
    (let [read-fn (if (= :jit tier) nested-read (pipeline/compile-aot #'nested-read))
          write-fn (if (= :jit tier) nested-write! (pipeline/compile-aot #'nested-write!))
          array (object-array [(float-array [1.0 2.0]) (float-array [3.0 4.0])])]
      (is (= (float 4.0) (read-fn array 1 1)))
      (is (= (float 5.0) (write-fn array 1 1 (float 5.0))))
      (is (= (float 5.0) (read-fn array 1 1)))
      (is (= (float 3.0) (read-fn array 1 0)))
      (is (= (float 2.0) (read-fn array 0 1))))))
