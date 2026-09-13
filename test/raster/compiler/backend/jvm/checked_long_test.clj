(ns raster.compiler.backend.jvm.checked-long-test
  "Small source-versus-bytecode boundary checks; no GPU or full compiler corpus needed."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.compiler.fixtures.checked-casts :as checked-casts]
            [raster.compiler.pipeline :as pipeline])
  (:import (java.lang.reflect InvocationTargetException)))

(def ^:private cases
  [{:name 'add :types '[long long] :body '(clojure.core/+ x y)}
   {:name 'subtract :types '[long long] :body '(clojure.core/- x y)}
   {:name 'multiply :types '[long long] :body '(clojure.core/* x y)}
   {:name 'negate :types '[long] :body '(clojure.core/- x)}
   {:name 'increment :types '[long] :body '(clojure.core/inc x)}
   {:name 'decrement :types '[long] :body '(clojure.core/dec x)}
   {:name 'mixed-add :types '[int long] :body '(clojure.core/+ x y)}
   {:name 'multiply-then-zero :types '[long long] :body '(clojure.core/* x y (long 0))}
   {:name 'wrap-add :types '[long long] :body '(clojure.core/unchecked-add x y)}
   {:name 'wrap-subtract :types '[long long] :body '(clojure.core/unchecked-subtract x y)}
   {:name 'wrap-multiply :types '[long long] :body '(clojure.core/unchecked-multiply x y)}
   {:name 'wrap-negate :types '[long] :body '(clojure.core/unchecked-negate x)}
   {:name 'wrap-increment :types '[long] :body '(clojure.core/unchecked-inc x)}
   {:name 'wrap-decrement :types '[long] :body '(clojure.core/unchecked-dec x)}])

(defn- parameters [types]
  (mapv #(with-meta %1 {:tag %2}) '[x y z] types))

(defn- outcome [f args]
  (try
    {:value (apply f args)}
    (catch Throwable error
      {:exception (class (if (instance? InvocationTargetException error)
                           (.getCause error) error))})))

(defn- map-outcome
  [f value & trailing]
  (let [output (int-array [17])
        result (outcome f (into [(long-array [value]) output (long 1)] trailing))]
    (assoc (dissoc result :value) :output (vec output))))

(deftest checked-and-explicitly-wrapping-long-arithmetic-match-clojure
  (let [specs (mapv (fn [{:keys [name types body]}]
                      {:name name :params (parameters types) :walked-body [body]
                       :source-ns (the-ns 'raster.compiler.backend.jvm.checked-long-test)
                       :return-tag 'long :element-type 'long}) cases)
        compiled (bytecode/compile-specialized-class! (str "raster.test.CheckedLong" (gensym)) specs)]
    (doseq [{:keys [name types body]} cases
            :let [method (get-in compiled [:methods name])
                  invoke #(.invoke method nil (object-array %&))
                  ;; Clojure fn primitive parameters support long/double, not int. Keep the
                  ;; oracle boxed; Integer inputs still exercise exact mixed-width promotion.
                  oracle (eval (list 'fn (vec (take (count types) '[x y z])) body))]
            args (if (= 1 (count types))
                   [[Long/MIN_VALUE] [-1] [0] [1] [Long/MAX_VALUE]]
                   (if (= 'int (first types))
                     [[(int 1) Long/MAX_VALUE] [(int -1) Long/MIN_VALUE]
                      [(int -7) (long 3)]]
                     [[Long/MAX_VALUE (long 2)] [Long/MIN_VALUE (long -1)]
                      [Long/MAX_VALUE (long 1)] [Long/MIN_VALUE (long 1)]
                      [(long -7) (long 3)] [(long 0) (long 0)]]))]
      (is (= (outcome oracle args) (outcome invoke args)) (pr-str [name args])))))

(deftest checked-product-fails-before-a-following-store
  (let [compiled (bytecode/compile-specialized-class!
                  (str "raster.test.CheckedExtentStore" (gensym))
                  [{:name 'store :params [(with-meta 'x {:tag 'long})
                                          (with-meta 'y {:tag 'long})
                                          (with-meta 'out {:tag 'longs})]
                    :source-ns (the-ns 'raster.compiler.backend.jvm.checked-long-test)
                    :return-tag 'long :element-type 'long
                    :walked-body ['(let* [size (clojure.core/* x y)]
                                     (clojure.core/aset out 0 size)
                                     size)]}])
        method (get-in compiled [:methods 'store])
        invoke #(.invoke method nil (object-array %&))
        output (long-array [17])]
    (is (= {:exception ArithmeticException} (outcome invoke [Long/MAX_VALUE (long 2) output])))
    (is (= [17] (vec output)))
    (is (= {:value 12} (outcome invoke [(long 3) (long 4) output])))
    (is (= [12] (vec output)))))

(deftest explicit-long-to-int-casts-are-checked-in-jit-and-aot-bytecode
  (let [jit checked-casts/declared-narrow-rows!
        aot (pipeline/compile-aot #'checked-casts/declared-narrow-rows! :simd? false)]
    (doseq [[tier f] [[:jit jit] [:aot aot]]]
      (testing (name tier)
        (is (= {:output [Integer/MIN_VALUE]} (map-outcome f Integer/MIN_VALUE)))
        (is (= {:output [Integer/MAX_VALUE]} (map-outcome f Integer/MAX_VALUE)))
        (is (= {:exception ArithmeticException :output [17]}
               (map-outcome f (inc (long Integer/MAX_VALUE)))))
        (is (= {:exception ArithmeticException :output [17]}
               (map-outcome f (dec (long Integer/MIN_VALUE)))))))))

(deftest explicit-unchecked-int-remains-wrapping-in-jit-and-aot-bytecode
  (let [jit checked-casts/checked-prefix-rows!
        aot (pipeline/compile-aot #'checked-casts/checked-prefix-rows! :simd? false)]
    (doseq [[tier f] [[:jit jit] [:aot aot]]]
      (testing (name tier)
        (is (= {:output [Integer/MIN_VALUE]}
               (map-outcome f (inc (long Integer/MAX_VALUE)) (long 0))))
        (is (= {:output [Integer/MAX_VALUE]}
               (map-outcome f (dec (long Integer/MIN_VALUE)) (long 0))))))))
