(ns raster.ad.source-product-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]
            [raster.compiler.passes.scalar.local-products :as local-products]))

(defn- typed [id tag] (with-meta id {:raster.type/tag tag}))

(defn- compile-source [parameters bindings result active]
  (let [source (list 'let* bindings result)
        ;; Exercise actual structural records, not successful scalar replacement
        ;; or a fabricated generated-call/source-policy owner.
        transformed (with-redefs [local-products/lower-local-products identity]
                      (with-bindings {#'reverse/*source-memory-context*
                                      {:values parameters :value-facts {}
                                       :replayed {} :unprojectable {} :aliases {}}}
                        (reverse/transform-body source active)))]
    {:source (eval (list 'fn parameters source))
     :paired (eval (list 'fn parameters transformed))
     :form transformed}))

(deftest source-products-retain-nested-constructors-and-field-derivatives
  (let [x (typed 'x 'double) square (typed 'square 'double)
        a (typed 'a 'double) b (typed 'b 'double) result (typed 'result 'double)
        {:keys [source paired form]}
        (compile-source [x]
                        [square '(raster.numeric/* x x)
                         'p [['square 'x]] 'inner '(clojure.core/nth p 0)
                         a '(clojure.core/nth inner 0) b '(clojure.core/nth inner 1)
                         result '(raster.numeric/* a b)] result [x])]
    (is (some #{[['square 'x]]} (tree-seq coll? seq form))
        "structural AD keeps the original constructor in the primal")
    (doseq [x [-2.5 -0.25 0.0 0.75 3.0]]
      (let [[value pb] (paired x) h 1.0e-5
            fd (/ (- (source (+ x h)) (source (- x h))) (* 2.0 h))]
        (is (= (* x x x) value))
        (is (< (Math/abs (- fd (first (pb 1.0)))) 1.0e-7))))))

(deftest source-product-branch-demand-is-sparse-not-an-inactive-numeric-zero
  (let [x (typed 'x 'double) flag (typed 'flag 'boolean)
        singular (typed 'singular 'double) result (typed 'result 'double)
        {:keys [source paired]}
        (compile-source [x flag]
                        [singular '(raster.numeric/sqrt x)
                         'p ['x 'singular]
                         result '(if flag
                                   (let* [a (clojure.core/nth p 0)] a)
                                   (let* [b (clojure.core/nth p 1)] b))]
                        result [x])]
    (doseq [flag [true false]]
      (let [[value pb] (paired 0.0 flag)]
        (is (= (source 0.0 flag) value))
        (is (= [(if flag 1.0 Double/POSITIVE_INFINITY)] (pb 1.0)))))
    (let [[_ pb] (paired 0.0 false)]
      (is (Double/isNaN (double (first (pb 0.0))))
          "an explicit numeric zero still executes the singular derivative"))))

(deftest active-compound-product-fields-never-become-inactive-constants
  (let [x (typed 'x 'double) result (typed 'result 'double)]
    (is (= :ad-product-unclosed-field
           (try (compile-source [x] ['p ['(raster.numeric/* x x)]
                                     result '(clojure.core/nth p 0)] result [x])
                :incorrectly-accepted
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest source-branch-discrete-captures-do-not-materialize-internal-zero-edges
  (let [x (typed 'x 'double) flag (typed 'flag 'boolean)
        singular (typed 'singular 'double) result (typed 'result 'double)
        {:keys [paired]}
        (compile-source [x flag]
                        [singular '(raster.numeric/sqrt x)
                         result '(if flag
                                   (let* [ignored (clojure.core/> singular 0.0)] x)
                                   x)] result [x])]
    (doseq [flag [true false]]
      (let [[value pb] (paired 0.0 flag)]
        (is (= 0.0 value))
        (is (= [1.0] (pb 1.0)))))))

(deftest source-structural-forward-retains-product-identity-and-metadata
  (let [x (typed 'x 'double) flag (typed 'flag 'boolean)
        token 'token result (typed 'result 'double)
        p (with-meta [token token] {:origin :source-product})
        {:keys [source paired]}
        (compile-source [x token flag]
                        ['p p 'q ['p 'p]
                         result '(if flag
                                   (let* [left (clojure.core/nth q 0)
                                          right (clojure.core/nth q 1)
                                          same (clojure.core/identical? left right)
                                          original (clojure.core/identical? (clojure.core/nth left 1) token)]
                                     (if (and same original) x 99.0))
                                   (let* [left (clojure.core/nth q 0)
                                          marker (:origin (meta left))]
                                     (if (= marker :source-product) x 88.0)))]
                        result [x])
        object (Object.)]
    (doseq [flag [true false] value [-2.0 3.0]]
      (let [[primal pb] (paired value object flag)]
        (is (= value (source value object flag) primal))
        (is (= [1.0] (pb 1.0)))))))

(deftest unknown-observers-of-active-products-refuse-rather-than-reconstruct-identity
  (let [x (typed 'x 'double) result (typed 'result 'double)]
    (is (= :ad-unsupported-product-use
           (try (compile-source [x]
                                ['p [x] 'q ['p 'p]
                                 'a '(clojure.core/nth q 0) 'b '(clojure.core/nth q 1)
                                 'same '(clojure.core/identical? a b)
                                 result '(if same x 0.0)] result [x])
                :incorrectly-accepted
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))

(deftest source-product-array-captures-route-and-accumulate-only-selected-reads
  (let [xs (typed 'xs 'doubles) flag (typed 'flag 'boolean)
        result (typed 'result 'double)
        {:keys [paired]}
        (compile-source [xs flag]
                        ['p [xs xs]
                         result '(if flag
                                   (let* [a (clojure.core/nth p 0)] (raster.arrays/aget a 0))
                                   (let* [b (clojure.core/nth p 1)
                                          value (raster.arrays/aget b 1)]
                                     (raster.numeric/* value value)))] result [xs])
        input (double-array [2.0 3.0])]
    (doseq [flag [true false]]
      (let [[value pb] (paired input flag)
            gradient (first (pb 1.0)) twice (first (pb 2.0))]
        (is (= (if flag 2.0 9.0) value))
        (is (= (if flag [1.0 0.0] [0.0 6.0]) (vec gradient)))
        (is (= (if flag [2.0 0.0] [0.0 12.0]) (vec twice)))
        (is (not (identical? gradient twice)))
        (is (= [2.0 3.0] (vec input)))))))

(deftest source-product-mixed-field-precision-remains-per-field
  (let [x (typed 'x 'float) y (typed 'y 'double) flag (typed 'flag 'boolean)
        result (typed 'result 'double)
        {:keys [paired]}
        (compile-source [x y flag]
                        ['p [x y]
                         result '(if flag
                                   (let* [a (clojure.core/nth p 0) b (clojure.core/nth p 1)]
                                     (raster.numeric/+ a b))
                                   (let* [a (clojure.core/nth p 0) b (clojure.core/nth p 1)]
                                     (raster.numeric/- b a)))] result [x y])]
    (doseq [flag [true false]]
      (let [[value pb] (paired (float 2.0) 3.0 flag)
            [dx dy] (pb 1.0)]
        (is (= (if flag 5.0 1.0) value))
        (is (instance? Float dx))
        (is (instance? Double dy))
        (is (= (float (if flag 1.0 -1.0)) dx))
        (is (= 1.0 dy))))))
