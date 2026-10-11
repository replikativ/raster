(ns raster.compiler.backend.cpu.native-call-plan-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.cpu.aot :as aot]))

(defn- wrapper [native reuse?]
  (#'aot/native-call-wrapper native '[x n eps]
    '[[x "double"]] '[[n "long"] [eps "double"]] '[[x x_length]]
    '[[out extent :double]]
    '[[irrelevant (Math/sqrt eps)] [base (+ n x_length)] [extent (int base)]]
    :vector '[out x] reuse?))

(deftest precomputed-native-abi-and-size-dependencies
  (let [calls (atom [])
        resolver aot/resolve-int-expr
        evaluated (atom [])
        f (wrapper (fn [& args] (swap! calls conj (vec args))) false)
        x (double-array 3)]
    (with-redefs [aot/resolve-int-expr
                  (fn [expr env] (swap! evaluated conj expr) (resolver expr env))]
      (let [[out original] (f x 2 0.25)]
        (is (= 5 (alength ^doubles out)))
        (is (identical? original x))
        (is (identical? out (second (first @calls))))
        (is (= [2 0.25 3] (subvec (first @calls) 2)))
      (is (not-any? #{'(Math/sqrt eps)} @evaluated))))))

(deftest mixed-storage-and-scalar-abi-order
  (let [received (atom nil)
        x (float-array 2)
        ids (int-array [1 0])
        f (#'aot/native-call-wrapper
           (fn [& args] (reset! received (vec args))) '[scale ids x n]
           '[[ids "int"] [x "float"]] '[[scale "double"] [n "long"]]
           '[[ids ids_length] [x x_length]] [] [] :scalar '[x] false)]
    (is (identical? x (f 0.5 ids x 7)))
    (is (identical? ids (first @received)))
    (is (identical? x (second @received)))
    (is (= [0.5 7 2 2] (subvec @received 2)))))

(deftest reuse-keys-all-resolved-extents-not-only-input-lengths
  (let [f (wrapper (fn [& _]) true)
        x (double-array 3)
        a (first (f x 2 0.25))
        b (first (f x 2 0.5))
        c (first (f x 4 0.25))]
    (is (identical? a b))
    (is (not (identical? a c)))
    (is (= 7 (alength ^doubles c)))))

(deftest fresh-buffers-and-zero-extents
  (let [f (wrapper (fn [& _]) false)
        x (double-array 0)
        a (first (f x 0 1.0))
        b (first (f x 0 1.0))]
    (is (= 0 (alength ^doubles a)))
    (is (not (identical? a b)))
    (is (thrown? NegativeArraySizeException (f x -1 1.0)))))

(deftest required-unsupported-size-expressions-fail-before-native-call
  (let [called (atom false)
        f (#'aot/native-call-wrapper
           (fn [& _] (reset! called true)) '[n] [] '[[n "long"]] []
           '[[out extent :double]] '[[extent (unknown-size n)]] :scalar '[out] false)]
    (is (thrown? clojure.lang.ExceptionInfo (f 2)))
    (is (false? @called))))

(deftest native-failure-is-not-swallowed
  (let [failure (ex-info "native failure" {:marker :test})
        f (wrapper (fn [& _] (throw failure)) false)]
    (is (identical? failure
                    (try (f (double-array 1) 1 1.0)
                         (catch clojure.lang.ExceptionInfo e e))))))

(deftest concurrent-default-calls-own-distinct-results
  (let [f (wrapper (fn [& _]) false)
        x (double-array 2)
        a (future (first (f x 3 1.0)))
        b (future (first (f x 3 1.0)))]
    (is (= 5 (alength ^doubles @a)))
    (is (= 5 (alength ^doubles @b)))
    (is (not (identical? @a @b)))))
