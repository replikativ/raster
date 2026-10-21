(ns raster.compiler.backend.jvm.source-integral-cast-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.jvm.bytecode :as bytecode]
            [raster.par :as par])
  (:import [java.lang.reflect InvocationTargetException Method]))

(defn- outcome [run]
  (try {:value (run)}
       (catch Throwable error
         {:exception (class (if (instance? InvocationTargetException error)
                              (.getCause error) error))})))

(defn- compile-expression [parameters expression]
  (get-in (bytecode/compile-specialized-class!
           (str "raster.compiled.SourceIntegralCast" (clojure.lang.RT/nextID))
           [{:name 'run :params (mapv (fn [{:keys [sym tag]}] (with-meta sym {:tag tag})) parameters)
             :return-tag 'long
             :walked-body [expression] :source-ns *ns*}])
          [:methods 'run]))

(defn- invoke [^Method method arguments]
  (.invoke method nil (object-array arguments)))

(deftest checked-source-casts-use-the-actual-runtime-overload
  ;; RT is an independent oracle for the source cast, not an implementation of
  ;; its range/NaN checks in the test. Primitive and Object overloads deliberately
  ;; have different failure behavior; a retained numeric hint cannot choose one.
  (doseq [[tag carrier values]
          [['int Integer/TYPE [Integer/MIN_VALUE (int -1) (int 0) Integer/MAX_VALUE]]
           ['long Long/TYPE [Long/MIN_VALUE (dec (long Integer/MIN_VALUE))
                            (long -1) (long 0) (long Integer/MAX_VALUE)
                            (inc (long Integer/MAX_VALUE)) Long/MAX_VALUE]]
           ['float Float/TYPE [Float/NEGATIVE_INFINITY (float (- Float/MAX_VALUE))
                              (float -1.9) (float -0.0) (float 0.0) (float 1.9)
                              Float/MAX_VALUE Float/POSITIVE_INFINITY Float/NaN]]
           ['double Double/TYPE [Double/NEGATIVE_INFINITY (- Double/MAX_VALUE)
                                -2147483649.0 -1.9 -0.0 0.0 1.9 2147483648.0
                                Double/MAX_VALUE Double/POSITIVE_INFINITY Double/NaN]]
           ['Object Object [nil true false \a (long 0) 1.9 Double/MAX_VALUE
                            Double/POSITIVE_INFINITY Double/NaN 1/3 1M]]
           ['boolean Boolean/TYPE [true false]]]
          cast '[clojure.core/int clojure.core/long]]
    (testing (str tag " -> " cast)
      (let [oracle (.getMethod clojure.lang.RT (str (name cast) "Cast")
                               (into-array Class [(if (= Boolean/TYPE carrier) Object carrier)]))
            compiled (compile-expression [{:sym 'x :tag tag}]
                                         (list 'clojure.core/long (list cast 'x)))
            wrapper (get {Integer/TYPE Integer Long/TYPE Long Float/TYPE Float
                          Double/TYPE Double Boolean/TYPE Boolean Object Object} carrier)]
        (is (= [carrier] (vec (.getParameterTypes ^Method compiled))))
        (is (every? #(or (and (= Object carrier) (nil? %)) (.isInstance ^Class wrapper %)) values)
            "the oracle and compiled method must not both fail on invalid reflection arguments")
        (doseq [value values]
          (is (= (outcome #(long (invoke oracle [value])))
                 (outcome #(invoke compiled [value])))
              (pr-str [tag cast value])))))))

(deftest source-cast-evaluates-its-operand-once-even-when-result-is-discarded
  (doseq [cast '[clojure.core/int clojure.core/long]]
    (let [compiled (compile-expression
                    [{:sym 'ticks :tag 'ints} {:sym 'x :tag 'double}]
                    (list 'do
                          (list cast '(do (clojure.core/aset ticks 0
                                                            (clojure.core/inc (clojure.core/aget ticks 0)))
                                          x))
                          (long 17)))]
      (doseq [[value expected] [[1.9 {:value 17}]
                                [Double/POSITIVE_INFINITY {:exception IllegalArgumentException}]]]
        (let [ticks (int-array 1)]
          (is (= expected (outcome #(invoke compiled [ticks value]))))
          (is (= [1] (vec ticks))))))))

(deftest void-and-divergent-operands-keep-their-source-failures
  (doseq [cast '[clojure.core/int clojure.core/long]]
    (let [compiled (compile-expression
                    [{:sym 'ticks :tag 'ints}]
                    (list 'clojure.core/long
                          (list cast '(java.util.Arrays/fill ticks (int 9)))))
          ticks (int-array 1)]
      (is (= {:exception NullPointerException} (outcome #(invoke compiled [ticks]))))
      (is (= [9] (vec ticks))))
    (let [compiled (compile-expression [{:sym 'failure :tag 'Object}]
                                       (list cast '(throw failure)))]
      (is (= {:exception IllegalStateException}
             (outcome #(invoke compiled [(IllegalStateException. "operand")])))))))

(deftest checked-and-unchecked-long-narrowing-remain-distinct
  (let [checked (compile-expression [{:sym 'x :tag 'long}] '(clojure.core/int x))
        wrapping (compile-expression [{:sym 'x :tag 'long}] '(clojure.core/unchecked-int x))
        input (inc (long Integer/MAX_VALUE))]
    (is (= {:exception ArithmeticException} (outcome #(invoke checked [input]))))
    (is (= {:value (long Integer/MIN_VALUE)} (outcome #(invoke wrapping [input]))))))

(deftest reduction-count-conversion-precedes-initializer-and-array-reads
  (let [compiled (compile-expression
                  [{:sym 'bound :tag 'double} {:sym 'ticks :tag 'ints} {:sym 'xs :tag 'longs}]
                  '(raster.par/reduce acc
                     (do (clojure.core/aset ticks 0 (clojure.core/inc (clojure.core/aget ticks 0)))
                         (long 0))
                     i bound (clojure.core/+ acc (clojure.core/aget xs i))))
        reference (fn [^double bound ^ints ticks ^longs xs]
                    (par/reduce acc
                      (do (aset ticks 0 (inc (aget ticks 0))) (long 0))
                      i bound (+ acc (aget xs i))))]
    (doseq [bound [-3.5 -0.0 0.0 1.9 2.0 3.0 Double/NaN
                   2147483648.0 Double/MAX_VALUE Double/POSITIVE_INFINITY]]
      (let [expected-ticks (int-array 1) actual-ticks (int-array 1)
            xs (long-array [3 7])]
        (is (= (outcome #(reference bound expected-ticks xs))
               (outcome #(invoke compiled [bound actual-ticks xs])))
            (pr-str bound))
        (is (= (vec expected-ticks) (vec actual-ticks)))
        (when (or (Double/isInfinite bound) (> bound Integer/MAX_VALUE))
          (is (= [0] (vec actual-ticks)) "invalid counts fail before initializer effects"))))))
