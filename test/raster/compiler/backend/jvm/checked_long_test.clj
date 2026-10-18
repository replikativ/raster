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

(deftest reference-carried-numbers-are-not-rounded-through-double
  (let [operations '[clojure.core/+ clojure.core/- clojure.core/* clojure.core//]
        bodies (mapv #(list % 'x 'y) operations)
        artifact (bytecode/compile-specialized-class!
                  (str "raster.test.ReferenceNumbers" (gensym))
                  (mapv (fn [index body]
                          {:name (symbol (str "run" index)) :params (parameters '[Object Object])
                           :walked-body [body] :source-ns *ns* :return-tag 'Object})
                        (range) bodies))]
    (doseq [[index body] (map-indexed vector bodies)
            :let [method (get-in artifact [:methods (symbol (str "run" index))])
                  run #(.invoke method nil (object-array %&))
                  oracle (eval (list 'fn '[x y] body))]
            arguments [[9007199254740995 1] [Long/MAX_VALUE 1] [Long/MIN_VALUE -1]
                       [3 2] [1/3 2] [100000000000000000000N 3]
                       [-0.0 2.0] [(float 1.5) (float 2.0)] [nil 1]]]
      (let [expected (outcome oracle arguments) actual (outcome run arguments)]
        (is (= expected actual) (pr-str [body arguments]))
        (when (contains? expected :value)
          (is (= (class (:value expected)) (class (:value actual))))
          (when (double? (:value expected))
            (is (= (Double/doubleToRawLongBits (:value expected))
                   (Double/doubleToRawLongBits (:value actual))))))))))

(deftest direct-arithmetic-and-explicit-calls-preserve-their-own-argument-order
  (doseq [body ['(clojure.core/+ x y (do (clojure.core/aset z 0 99) 1))
                '(clojure.core/+ (clojure.core/long x) (clojure.core/long y)
                                 (clojure.core/long (do (clojure.core/aset z 0 99) 1)))
                '(let [f clojure.core/+]
                   (f x y (do (clojure.core/aset z 0 99) 1)))]]
    (let [method (get-in (bytecode/compile-specialized-class!
                          (str "raster.test.ReferenceArgumentOrder" (gensym))
                          [{:name 'run :params (parameters '[Object Object ints])
                            :walked-body [body] :source-ns *ns* :return-tag 'Object}]) [:methods 'run])
          oracle (eval (list 'fn '[x y z] body))
          expected-effects (int-array [0])
          actual-effects (int-array [0])]
      (is (= (outcome oracle [Long/MAX_VALUE 1 expected-effects])
             (outcome #(.invoke method nil (object-array %&))
                      [Long/MAX_VALUE 1 actual-effects])))
      (is (= (vec expected-effects) (vec actual-effects)) (pr-str body)))))

(deftest source-cast-descriptors-do-not-prove-emitted-primitive-carriers
  (let [body '(clojure.core/+ (clojure.core/unchecked-int 1) 9007199254740995)
        method (get-in (bytecode/compile-specialized-class!
                       (str "raster.test.FallbackCastCarrier" (gensym))
                       [{:name 'run :params [] :walked-body [body]
                         :source-ns *ns* :return-tag 'Object}]) [:methods 'run])]
    (is (= 9007199254740996 (.invoke method nil (object-array []))))))

(deftest mixed-reference-arithmetic-selects-the-actual-numbers-overload
  (doseq [tags '[[long Object] [Object long] [double Object] [Object double]]
          operator '[clojure.core/+ clojure.core/- clojure.core/* clojure.core//]]
    (let [body (list operator 'x 'y)
          params (parameters tags)
          oracle (eval (list 'fn params body))
          method (get-in (bytecode/compile-specialized-class!
                         (str "raster.test.MixedNumberCarriers" (gensym))
                         [{:name 'run :params params :walked-body [body]
                           :source-ns *ns* :return-tag 'Object}]) [:methods 'run])]
      (doseq [arguments
              (concat
               (map #(mapv (fn [tag value] (if (= tag 'double) (double value) value)) tags %)
                    [[9007199254740995 1] [3 2] [-7 3] [1 0] [0 0]])
               (for [value [nil 1/3 100000000000000000000N Double/NaN (float 1.5)]]
                 (mapv #(case % Object value double 3.0 long 3) tags)))
              :let [expected (outcome oracle arguments)
                    actual (outcome #(.invoke method nil (object-array %&)) arguments)]]
        (if (and (contains? expected :value) (double? (:value expected)))
          (is (= (Double/doubleToRawLongBits (:value expected))
                 (Double/doubleToRawLongBits (double (:value actual))))
              (pr-str [tags body arguments expected actual]))
          (is (= expected actual) (pr-str [tags body arguments expected actual])))))))

(deftest compound-logical-tags-do-not-prove-primitive-operands
  (let [nullable (with-meta '(if flag x nil) {:raster.type/tag 'long})
        bodies [(list 'clojure.core/- nullable 1)
                '(clojure.core/+ (clojure.core/+) x)
                '(clojure.core/* (clojure.core/*) x)]
        artifact (bytecode/compile-specialized-class!
                  (str "raster.test.CompoundNumberCarriers" (gensym))
                  (mapv (fn [index body]
                          {:name (symbol (str "run" index))
                           :params [(with-meta 'x {:tag 'Object}) (with-meta 'flag {:tag 'boolean})]
                           :walked-body [body] :source-ns *ns* :return-tag 'Object})
                        (range) bodies))]
    (doseq [[index body] (map-indexed vector bodies)
            :let [method (get-in artifact [:methods (symbol (str "run" index))])
                  run #(.invoke method nil (object-array %&))
                  oracle (eval (list 'fn '[x flag] body))]
            flag [true false]]
      (is (= (outcome oracle [9007199254740995 flag])
             (outcome run [9007199254740995 flag]))))))

(def ^:private reference-number-root nil)

(deftest global-logical-tags-do-not-prove-local-primitive-carriers
  (with-redefs [reference-number-root nil]
    (let [reference (with-meta
                      'raster.compiler.backend.jvm.checked-long-test/reference-number-root
                      {:raster.type/tag 'long})
          body (list 'clojure.core/- reference 1)
          method (get-in (bytecode/compile-specialized-class!
                         (str "raster.test.GlobalNumberCarrier" (gensym))
                         [{:name 'run :params [] :walked-body [body]
                           :source-ns *ns* :return-tag 'Object}]) [:methods 'run])]
      (alter-var-root #'reference-number-root (constantly 9007199254740995))
      (is (= 9007199254740994 (.invoke method nil (object-array [])))))))

(deftest explicit-primitive-input-contexts-preserve-ieee-division
  ;; An explicit double cast selects primitive IEEE arithmetic in source too.
  ;; This differs from generic Number division, whose zero check must remain.
  (doseq [body ['(clojure.core// (clojure.core/double x) (clojure.core/double y))
                '(clojure.core// (double x) (double y))
                '(clojure.core// (clojure.core/double x) y)
                '(clojure.core// x (clojure.core/double y))]]
    (let [method (get-in (bytecode/compile-specialized-class!
                         (str "raster.test.PrimitiveDivisionInputs" (gensym))
                         [{:name 'run :params (parameters '[Object Object])
                           :walked-body [body] :source-ns *ns* :return-tag 'double}]) [:methods 'run])
          oracle (eval (list 'fn '[x y] body))]
      (doseq [[x y] [[1.0 0.0] [-1.0 0.0] [0.0 0.0] [1.0 -0.0] [3.0 2.0]]]
        (is (= (Double/doubleToRawLongBits (double (oracle x y)))
               (Double/doubleToRawLongBits (double (.invoke method nil (object-array [x y]))))))))))

(deftest variadic-primitive-arithmetic-uses-actual-intermediate-carriers
  (let [body '(clojure.core// (clojure.core/double x) (clojure.core/double y)
                            (clojure.core/double (do (clojure.core/aset z 0 99) 0.0)))
        oracle (eval (list 'fn '[x y z] body))
        method (get-in (bytecode/compile-specialized-class!
                       (str "raster.test.VariadicDivisionCarriers" (gensym))
                       [{:name 'run :params (parameters '[Object Object ints])
                         :walked-body [body] :source-ns *ns* :return-tag 'Object}]) [:methods 'run])]
    (doseq [[x y] [[1.0 0.0] [0.0 0.0] [1.0 2.0]]]
      (let [expected-effects (int-array [0]) actual-effects (int-array [0])
            expected (oracle x y expected-effects)
            actual (.invoke method nil (object-array [x y actual-effects]))]
        (is (= (Double/doubleToRawLongBits (double expected))
               (Double/doubleToRawLongBits (double actual))))
        (is (= (vec expected-effects) (vec actual-effects)))))))

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
