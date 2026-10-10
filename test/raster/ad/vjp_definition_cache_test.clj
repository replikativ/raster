(ns raster.ad.vjp-definition-cache-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.core :as core]))

(defn- with-model [f]
  (let [namespace (create-ns (gensym "raster.vjp-definition-model."))]
    (try
      (binding [*ns* namespace]
        (refer 'clojure.core)
        (alias 'n 'raster.numeric)
        (eval '(raster.core/deftm child [x :- Double] :- Double (n/* x x)))
        (eval '(raster.core/deftm parent [x :- Double] :- Double
                 (n/+ (child x) 1.0)))
        (f namespace (ns-resolve namespace 'parent)))
      (finally (remove-ns (ns-name namespace))))))

(defn- replace-child! []
  (eval '(raster.core/deftm child [x :- Double] :- Double (n/* 3.0 x))))

(deftest nested-definition-change-invalidates-a-fresh-vjp-acquisition
  (with-model
    (fn [_ parent]
      (let [[before old-pullback] (reverse/vjp parent 2.0)
            resolved (core/resolve-deftm-var parent)
            walked (core/ensure-walked-body! resolved)
            revision (dispatch/compiler-definition-revision)]
        (is (= 5.0 before))
        (is (= [4.0] (old-pullback 1.0)))
        (replace-child!)
        (is (< revision (dispatch/compiler-definition-revision)))
        (is (= walked (core/ensure-walked-body! resolved)))
        (let [[after new-pullback] (reverse/vjp parent 2.0)]
          (is (= 7.0 after))
          (is (= [3.0] (new-pullback 1.0)))
          (is (= [4.0] (old-pullback 1.0))
              "an already acquired pullback retains its captured residual"))))))

(deftest unchanged-definitions-still-reuse-the-vjp
  (with-model
    (fn [_ parent]
      (let [transform reverse/transform-body
            calls (atom 0)]
        (with-redefs [reverse/transform-body
                      (fn [& args] (swap! calls inc) (apply transform args))]
          (reverse/vjp parent 2.0)
          (let [initial @calls]
            (is (pos? initial))
            (let [[value pullback] (reverse/vjp parent 3.0)]
              (is (= 10.0 value))
              (is (= [6.0] (pullback 1.0)))
              (is (= initial @calls)))))))))

(deftest mutation-during-transformation-discards-the-old-definition-snapshot
  (with-model
    (fn [namespace parent]
      (let [transform reverse/transform-body
            replaced? (atom false)]
        (with-redefs [reverse/transform-body
                      (fn [& args]
                        (let [result (apply transform args)]
                          (when (compare-and-set! replaced? false true)
                            (binding [*ns* namespace] (replace-child!)))
                          result))]
          (let [[value pullback] (reverse/vjp parent 2.0)]
            (is @replaced?)
            (is (= 7.0 value))
            (is (= [3.0] (pullback 1.0)))))))))

(deftest mutation-during-walk-does-not-return-an-old-cache-hit
  (with-model
    (fn [namespace parent]
      (reverse/vjp parent 2.0)
      (let [walk core/ensure-walked-body!
            resolved (core/resolve-deftm-var parent)
            replaced? (atom false)]
        (with-redefs [core/ensure-walked-body!
                      (fn [v]
                        (let [result (walk v)]
                          (when (and (identical? v resolved)
                                     (compare-and-set! replaced? false true))
                            (binding [*ns* namespace] (replace-child!)))
                          result))]
          (let [[value pullback] (reverse/vjp parent 2.0)]
            (is @replaced?)
            (is (= 7.0 value))
            (is (= [3.0] (pullback 1.0)))))))))
