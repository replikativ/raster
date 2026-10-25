(ns raster.compiler.backend.jvm.array-index-completion-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.jvm.bytecode :as bytecode])
  (:import [java.lang.reflect InvocationTargetException]))

(defn- invoke-method [method arguments]
  (try (.invoke ^java.lang.reflect.Method method nil (object-array arguments))
       (catch InvocationTargetException error (throw (.getCause error)))))

(defn- observe [invoke index]
  (let [array (long-array [3]) effects (long-array 1)
        outcome (try [:value (invoke array index effects)]
                     (catch Throwable error [:throw (class error)]))]
    {:outcome outcome :array (vec array) :effects (vec effects)}))

(defn- store-body [loop?]
  `(clojure.core/aset ~'array ~'index
     (do (clojure.core/aset ~'effects 0 9)
         ~(if loop?
            '(loop* [iteration (clojure.core/int 0)]
               (if (clojure.core/< iteration 2)
                 (recur (clojure.core/unchecked-inc-int iteration)) 7))
            7))))

(deftest core-index-completion-matches-clojure-before-value-effects
  (doseq [policy [false true :warn-on-boxed]
          [tag inputs] [['long [0 1 -1 4294967296 Long/MIN_VALUE Long/MAX_VALUE]]
                        ['double [0.0 0.75 1.0 4294967296.0 Double/NaN
                                  Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY]]
                        ['Object [0 4294967296 0.75 \u0000 nil true (Object.)]]]
          loop? [false true]]
    (let [parameters [(with-meta 'array {:tag 'longs})
                      (with-meta 'index {:tag tag})
                      (with-meta 'effects {:tag 'longs})]
          body (store-body loop?)
          source (binding [*unchecked-math* policy] (eval (list 'fn parameters body)))
          artifact (bytecode/compile-specialized-class!
                    "raster.compiled.ArrayIndexCompletion"
                    [{:name 'run :params parameters :return-tag 'Object
                      :walked-body [body] :source-ns *ns*
                      :source-policy {:unchecked-math policy}}])
          method (get-in artifact [:methods 'run])
          compiled (fn [& arguments] (invoke-method method arguments))]
      (doseq [index inputs]
        (testing (str "policy=" policy " carrier=" tag " loop=" loop? " index=" (pr-str index))
          (is (= (observe source index) (observe compiled index))))))))

(deftest explicit-integral-casts-share-the-source-policy-and-actual-carrier-rule
  (doseq [policy [false true :warn-on-boxed]
          cast ['clojure.core/int 'clojure.core/long]
          [tag inputs] [['long [0 4294967296 Long/MAX_VALUE]]
                        ['double [0.75 4294967296.0 Double/NaN
                                  Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY]]
                        ['Object [0 4294967296 0.75 \u0000 nil true (Object.)]]]]
    (let [parameters [(with-meta 'value {:tag tag})]
          body (list cast 'value)
          source (binding [*unchecked-math* policy] (eval (list 'fn parameters body)))
          artifact (bytecode/compile-specialized-class!
                    "raster.compiled.SourceIntegralCompletion"
                    [{:name 'run :params parameters :return-tag 'Object
                      :walked-body [body] :source-ns *ns*
                      :source-policy {:unchecked-math policy}}])
          method (get-in artifact [:methods 'run])
          outcome (fn [f value]
                    (try (let [result (f value)] [:value (class result) result])
                         (catch Throwable error [:throw (class error)])))]
      (doseq [value inputs]
        (is (= (outcome source value)
               (outcome #(invoke-method method [%]) value))
            (str policy " " cast " " tag " " (pr-str value)))))))

