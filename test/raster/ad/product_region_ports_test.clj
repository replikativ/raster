(ns raster.ad.product-region-ports-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]))

(defn- typed [id tag] (with-meta id {:raster.type/tag tag}))

(defn- compile-program [params bindings result active]
  (let [policy {:unchecked-math false}]
    (#'reverse/make-runtime-value+grad-fn
     {:kind :generated-product-bindings :source-policy policy
      :bindings bindings :result result :active-inputs active}
     params {:generated-source-policy policy})))

(deftest scalar-branches-import-product-captures-and-export-field-cotangents
  (let [x (typed 'x 'double) flag (typed 'flag 'boolean)
        square (typed 'square 'double) result (typed 'result 'double)
        f (compile-program
           [x flag]
           [square '(raster.numeric/* x x)
            'p ['x 'square nil false]
            result '(if flag
                      (let* [a (clojure.core/first p)
                             b (clojure.core/second p)]
                        (raster.numeric/* a b))
                      (let* [a (clojure.core/first p)
                             b (clojure.core/second p)]
                        (raster.numeric/+ a b)))] result ['x])]
    (doseq [x [-2.0 0.0 3.0] flag [true false]]
      (let [[value pb] (f x flag)]
        (is (= (if flag (* x x x) (+ x (* x x))) value))
        (is (= [(if flag (* 3.0 x x) (+ 1.0 (* 2.0 x)))] (pb 1.0)))))))

(deftest unused-singular-product-fields-remain-absent-across-branches
  (let [x (typed 'x 'double) flag (typed 'flag 'boolean)
        singular (typed 'singular 'double) result (typed 'result 'double)
        f (compile-program
           [x flag]
           [singular '(raster.numeric/sqrt x)
            'p ['x 'singular]
            result '(if flag
                      (let* [a (clojure.core/first p)] (raster.numeric/+ a a))
                      (let* [a (clojure.core/first p)] a))] result ['x])]
    (doseq [flag [true false]]
      (let [[value pb] (f 0.0 flag)]
        (is (= 0.0 value))
        (is (= [(if flag 2.0 1.0)] (pb 1.0)))))))

(deftest field-demand-follows-the-selected-arm-not-the-union-of-arms
  (let [x (typed 'x 'double) flag (typed 'flag 'boolean)
        singular (typed 'singular 'double) result (typed 'result 'double)
        f (compile-program
           [x flag]
           [singular '(raster.numeric/sqrt x)
            'p ['x 'singular]
            result '(if flag
                      (let* [a (clojure.core/first p)] a)
                      (let* [b (clojure.core/second p)] b))] result ['x])]
    (let [[value pb] (f 0.0 true)]
      (is (= 0.0 value))
      (is (= [1.0] (pb 1.0))))
    (let [[value pb] (f 0.0 false)]
      (is (= 0.0 value))
      (is (= [Double/POSITIVE_INFINITY] (pb 1.0)))
      ;; Explicit numeric zero is not an absent edge: preserve 0 * infinity.
      (is (Double/isNaN (double (first (pb 0.0))))))))

(deftest array-fields-and-aliases-cross-only-the-selected-branch
  (let [xs (typed 'xs 'doubles) flag (typed 'flag 'boolean)
        result (typed 'result 'double)
        f (compile-program
           [xs flag]
           ['p ['xs 'xs]
            result '(if flag
                      (let* [a (clojure.core/first p)] (raster.arrays/aget a 0))
                      (let* [b (clojure.core/second p)
                             value (raster.arrays/aget b 1)]
                        (raster.numeric/* value value)))] result ['xs])
        input (double-array [2.0 3.0])]
    (doseq [flag [true false]]
      (let [[value pb] (f input flag)
            gradient (first (pb 1.0))
            doubled (first (pb 2.0))]
        (is (= (if flag 2.0 9.0) value))
        (is (= (if flag [1.0 0.0] [0.0 6.0]) (vec gradient)))
        (is (= (if flag [2.0 0.0] [0.0 12.0]) (vec doubled)))
        (is (not (identical? gradient doubled)))
        (is (= [2.0 3.0] (vec input)))))))

(deftest mixed-field-declarations-remain-in-their-own-tangent-spaces
  (let [x (typed 'x 'float) y (typed 'y 'double) flag (typed 'flag 'boolean)
        result (typed 'result 'double)
        f (compile-program
           [x y flag]
           ['p ['x 'y]
            result '(if flag
                      (let* [a (clojure.core/first p) b (clojure.core/second p)]
                        (raster.numeric/+ a b))
                      (let* [a (clojure.core/first p) b (clojure.core/second p)]
                        (raster.numeric/- b a)))] result ['x 'y])]
    (doseq [flag [true false]]
      (let [[value pb] (f (float 2.0) 3.0 flag)
            [dx dy] (pb 1.0)]
        (is (= (if flag 5.0 1.0) value))
        (is (instance? Float dx))
        (is (instance? Double dy))
        (is (= (float (if flag 1.0 -1.0)) dx))
        (is (= 1.0 dy))))))

(deftest nested-branches-preserve-their-own-field-demand
  (let [x (typed 'x 'double) outer (typed 'outer 'boolean) inner (typed 'inner 'boolean)
        singular (typed 'singular 'double) result (typed 'result 'double)
        f (compile-program
           [x outer inner]
           [singular '(raster.numeric/sqrt x) 'p ['x 'singular]
            result '(if outer
                      (let* [a (clojure.core/first p)
                             b (clojure.core/second p)]
                        (if inner a b))
                      (let* [a (clojure.core/first p)] (raster.numeric/+ a a)))] result ['x])]
    (doseq [[outer inner expected] [[true true 1.0] [false true 2.0]
                                   [false false 2.0] [true false Double/POSITIVE_INFINITY]]]
      (let [[value pb] (f 0.0 outer inner)]
        (is (= 0.0 value))
        (is (= [expected] (pb 1.0)))))))

(deftest a-branch-cannot-hide-mutation-or-an-unclosed-product-result
  (let [x (typed 'x 'double) xs (typed 'xs 'doubles) flag (typed 'flag 'boolean)
        result (typed 'result 'double)
        reason (fn [bindings]
                 (try (compile-program [x xs flag] bindings result ['x]) nil
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (is (= :ad-generated-product-region
           (reason [result '(if flag
                              (let* [written (raster.arrays/aset xs 0 x)] x)
                              x)])))
    (is (= :ad-generated-product-result
           (reason ['p ['x] result '(if flag p p)])))))

(deftest unused-scalar-captures-do-not-create-present-zero-edges
  (let [x (typed 'x 'double) flag (typed 'flag 'boolean)
        singular (typed 'singular 'double) result (typed 'result 'double)]
    (doseq [unused ['singular '(clojure.core/> singular 0.0)]]
      (let [f (compile-program
               [x flag]
               [singular '(raster.numeric/sqrt x)
                result (list 'if 'flag (list 'let* ['unused unused] 'x) 'x)] result ['x])]
        (doseq [flag [true false]]
          (let [[value pb] (f 0.0 flag)]
            (is (= 0.0 value))
            (is (= [1.0] (pb 1.0)))))))))

(deftest final-public-gradients-materialize-zero-only-at-the-boundary
  (let [x (typed 'x 'double) flag (typed 'flag 'boolean)
        result (typed 'result 'double)
        f (compile-program [x flag]
                           ['p ['x]
                            result '(if flag
                                      (let* [a (clojure.core/first p)] a)
                                      0.0)] result ['x])]
    (doseq [flag [true false]]
      (let [[value pb] (f 2.0 flag)]
        (is (= (if flag 2.0 0.0) value))
        (is (= [(if flag 1.0 0.0)] (pb 1.0))))))
  (doseq [[tag input expected] [['float (float 2.0) (float 0.0)]
                               ['double 2.0 0.0]
                               ['doubles (double-array [2.0 3.0]) [0.0 0.0]]]]
    (let [x (typed 'x tag) flag (typed 'flag 'boolean) result (typed 'result 'double)
          f (compile-program [x flag]
                             [result '(if flag (let* [unused x] 1.0) 1.0)] result ['x])]
      (doseq [flag [true false] seed [0.0 1.0]]
        (let [[value pb] (f input flag)
              gradient (first (pb seed))]
          (is (= 1.0 value))
          (if (= tag 'doubles)
            (is (= expected (vec gradient)))
            (do (is (= expected gradient))
                (is (= (class expected) (class gradient))))))))))
