(ns raster.compiler.backend.cpu.scalar-return-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.java.shell :as shell]
            [raster.core :refer [deftm]]
            [raster.compiler.backend.cpu.aot :as aot]
            [raster.compiler.backend.cpu.codegen :as codegen]
            [raster.compiler.fixtures.mixed-storage :as storage]
            [raster.sci.distributions :as dist]))

(defn- compiler-available? []
  (try (zero? (:exit (shell/sh (or (System/getenv "RASTER_CC") "clang") "--version")))
       (catch java.io.IOException _ false)))

(deftm double-return [x :- Double] :- Double x)
(deftm float-return [x :- Float] :- Float x)
(deftm long-return [x :- Long] :- Long x)
(deftm int-return [x :- Integer] :- Integer x)
(deftm boolean-return [x :- Boolean] :- Boolean x)
(deftm conditional-return [x :- Double] :- Double
  (let [y (+ x 1.0)] (if (< y 0.0) (- y) y)))
(deftm mutated-return [out :- (Array double) x :- Double] :- Double
  (aset out 0 (+ (aget out 0) 1.0))
  (+ x (aget out 0)))
(deftm normal-lp [x :- Double s :- Double] :- Double
  (dist/logpdf (dist/->Normal 0.0 s) x))
(deftm loop-exit-return [out :- (Array double) n :- Long] :- Long
  (loop [i 0]
    (if (< i n)
      (do (aset out i (+ (aget out i) 1.0)) (recur (inc i)))
      i)))

(deftest numeric-native-return-abis
  (when (compiler-available?)
    (doseq [[v dtype value expected-class]
            [[#'double-return :float 1.234567890123 Double]
             [#'float-return :double (float 1.234567890123) Float]
             [#'long-return :double 4294967303 Long]
             [#'int-return :double 2147483647 Integer]]]
      (let [f (aot/compile-aot-c v dtype)
            actual (f value)]
        (is (= value actual))
        (is (= expected-class (class actual)))))))

(deftest native-value-tail-is-evaluated-once-after-effects
  (when (compiler-available?)
    (let [f (aot/compile-aot-c #'mutated-return :double)
          out (double-array [2.0])]
      (is (= 7.0 (f out 4.0)))
      (is (= 3.0 (aget out 0)))
      (is (= 8.0 (f out 4.0)))
      (is (= 4.0 (aget out 0))))
    (let [f (aot/compile-aot-c #'conditional-return :double)]
      (is (= 2.0 (f -3.0)))
      (is (= 5.0 (f 4.0))))))

(deftest reduction-return-uses-declared-precision
  (when (compiler-available?)
    (let [f (aot/compile-aot-c #'storage/double-reduction-float-result :double)
          values (double-array [0.1 0.2 0.3])]
      (is (= Float (class (f values))))
      (is (= (float 0.6) (f values))))))

(deftest counted-store-loop-retains-its-scalar-exit
  (when (compiler-available?)
    (let [f (aot/compile-aot-c #'loop-exit-return :double)
          out (double-array [2.0 3.0])]
      (is (= 2 (f out 2)))
      (is (= [3.0 4.0] (vec out)))
      (is (= 0 (f out 0)))
      (is (= [3.0 4.0] (vec out))))))

(deftest constructed-normal-native-scalar-return
  (when (compiler-available?)
    (let [f (aot/compile-aot-c #'normal-lp :double)]
      (doseq [[x s] [[0.5 2.0] [-0.75 0.4]]]
        (let [expected (* -0.5 (+ (Math/log (* 2.0 Math/PI)) (* 2.0 (Math/log s))
                                  (* (/ x s) (/ x s))))]
          (is (< (Math/abs (- expected (f x s))) 1.0e-12)))))))

(deftest unsupported-return-is-rejected-before-library-loading
  (is (= :unsupported-native-return
         (try (codegen/load-kernel "nonexistent.so" "not_loaded" 0 [] :object)
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
  (with-redefs [codegen/compile-source! (fn [& _] (throw (ex-info "must not compile" {})))]
    (is (= :unsupported-native-return
           (try (aot/compile-aot-c #'boolean-return :double)
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
