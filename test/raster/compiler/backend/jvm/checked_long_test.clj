(ns raster.compiler.backend.jvm.checked-long-test
  "Small source-versus-bytecode boundary checks; no GPU or full compiler corpus needed."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.bytecode :as bytecode])
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
