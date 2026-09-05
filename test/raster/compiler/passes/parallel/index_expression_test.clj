(ns raster.compiler.passes.parallel.index-expression-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.passes.parallel.index-expression :as index-expression]))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason))))

(deftest typed-launch-projection-widens-operands-before-arithmetic
  (let [dtypes {'rows :int 'width :int}
        lowered (index-expression/lower-typed
                 (with-meta '(clojure.core/* rows width) {:raster.type/tag 'long})
                 #{'rows 'width} dtypes :long fail!)
        projected (index-expression/to-launch-expression lowered fail!)]
    (is (= :mul (:op lowered)))
    (is (= [:long :long] (mapv :dtype (:arguments lowered))))
    (is (every? #(= :exact (:overflow %)) (:arguments lowered)))
    (is (= :long (launch/typed-expression-dtype projected dtypes)))
    (is (= #{'rows 'width} (launch/expression-references projected)))))

(deftest typed-launch-projection-declines-int-overflow-semantics
  (let [reason (try
                 (index-expression/lower-typed
                  (with-meta '(clojure.core/* rows width) {:raster.type/tag 'int})
                  #{'rows 'width} {'rows :long 'width :long} :int fail!)
                 nil
                 (catch clojure.lang.ExceptionInfo error
                   (:reason (ex-data error))))]
    (is (= :index-expression-overflow reason))))

(deftest typed-launch-projection-requires-authoritative-compound-width
  (let [reason (try
                 (index-expression/lower-typed
                  '(clojure.core/* rows width)
                  #{'rows 'width} {'rows :int 'width :int} :long fail!)
                 nil
                 (catch clojure.lang.ExceptionInfo error
                   (:reason (ex-data error))))]
    (is (= :index-expression reason))))

(deftest wrapping-raster-int-arithmetic-is-not-exact-launch-algebra
  (let [reason (try
                 (index-expression/lower-typed
                  (with-meta '(raster.numeric/* rows width) {:raster.type/tag 'int})
                  #{'rows 'width} {'rows :int 'width :int} :int fail!)
                 nil
                 (catch clojure.lang.ExceptionInfo error
                   (:reason (ex-data error))))]
    (is (= :index-expression-overflow reason))))
