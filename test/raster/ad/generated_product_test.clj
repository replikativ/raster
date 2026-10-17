(ns raster.ad.generated-product-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]
            [raster.compiler.core.value-completion :as completion]))

(def policy {:unchecked-math false})
(defn- typed [id tag] (with-meta id {:raster.type/tag tag}))

(defn- compile-program [params bindings result active]
  (#'reverse/make-runtime-value+grad-fn
   {:kind :generated-product-bindings :source-policy policy
    :bindings bindings :result result :active-inputs active}
   params {:generated-source-policy policy}))

(defn- failure-reason [f]
  (try (f) nil
       (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))
       (catch ClassCastException _ :class-cast)))

(deftest numeric-input-and-seed-carriers-are-not-implicitly-adapted
  (let [x (typed 'x 'double) result (typed 'result 'double)
        f (compile-program [x] [result '(raster.numeric/* x x)] result ['x])]
    (is (= :ad-generated-input-carrier (failure-reason #(f nil))))
    (doseq [input [(float 2.0) 2]]
      (is (= :class-cast (failure-reason #(f input)))))
    (let [[value pb] (f 2.0)]
      (is (= 4.0 value))
      (doseq [seed [nil (float 1.0) 1]]
        (is (= :ad-generated-seed-carrier (failure-reason #(pb seed)))))
      ;; A present zero is a valid cotangent, not absence of demand.
      (is (= [0.0] (pb 0.0))))))

(deftest generated-data-scope-and-effects-must-be-closed
  (let [x (typed 'x 'double) xs (typed 'xs 'doubles)
        result (typed 'result 'double)]
    (is (= :ad-generated-open-value
           (failure-reason #(compile-program [x]
                              [result '(raster.numeric/* x coefficient)] result ['x]))))
    (is (= :ad-generated-open-value
           (failure-reason #(compile-program [x]
                              ['p ['later] 'later 'x result '(clojure.core/first p)]
                              result ['x]))))
    (is (= :ad-generated-product-region
           (failure-reason #(compile-program [xs]
                              ['write '(raster.arrays/aset xs 0 1.0)
                               result '(raster.arrays/aget xs 0)] result ['xs]))))))

(deftest structural-pullback-agrees-with-independent-finite-differences
  (let [x (typed 'x 'double) square (typed 'square 'double)
        result (typed 'result 'double)
        f (compile-program [x]
                           [square '(raster.numeric/* x x)
                            'p [['square 'x]]
                            'inner '(clojure.core/first p)
                            'a '(clojure.core/first inner)
                            'b '(clojure.core/second inner)
                            result '(raster.numeric/* a b)] result ['x])
        oracle (fn [x] (* x x x))
        h 1.0e-5]
    (doseq [input [-2.5 -0.25 0.0 0.75 3.0]]
      (let [[value pb] (f input)
            fd (/ (- (oracle (+ input h)) (oracle (- input h))) (* 2.0 h))]
        (is (= (oracle input) value))
        (is (< (Math/abs (- fd (first (pb 1.0)))) 1.0e-7))))))

(deftest scalar-aliases-transport-existing-field-declarations
  (let [x (typed 'x 'double) result (typed 'result 'double)
        f (compile-program [x] ['alias 'x 'p ['alias]
                                result '(clojure.core/first p)] result ['x])
        [value pb] (f 3.0)]
    (is (= 3.0 value))
    (is (= [1.0] (pb 1.0)))))

(deftest generated-products-reuse-the-existing-rule-staleness-fence
  (let [x (typed 'x 'double) result (typed 'result 'double)
        f (compile-program [x] ['p ['x] 'a '(clojure.core/first p)
                                result '(raster.numeric/* a a)] result ['x])
        [_ pb] (f 3.0)
        revisions (ns-resolve 'raster.ad.normalized-scalar 'revisions)
        current (revisions)]
    (with-redefs-fn {revisions (constantly (update current :template-revision inc))}
      (fn []
        (is (= :ad-normalized-stale-rules (failure-reason #(f 3.0))))
        (is (= :ad-normalized-stale-rules (failure-reason #(pb 1.0))))))
    (is (= [6.0] (pb 1.0)))))

(deftest standalone-data-aliases-must-also-be-visible
  (let [x (typed 'x 'double) result (typed 'result 'double)]
    (doseq [bindings [[result 'raster.ad.generated-product-test/policy]
                      [result 'later 'later 'x]
                      [result result]]]
      (is (= :ad-generated-open-value
             (failure-reason #(compile-program [x] bindings result ['x])))))))

(deftest construction-cannot-bless-a-derivative-from-an-earlier-rule-epoch
  (let [x (typed 'x 'double) result (typed 'result 'double)
        revisions (ns-resolve 'raster.ad.normalized-scalar 'revisions)
        epoch (atom (revisions))
        derive (ns-resolve 'raster.ad.reverse 'generated-product-form)
        original @derive]
    (with-redefs-fn {revisions #(deref epoch)
                    derive (fn [& args]
                             (let [program (apply original args)]
                               (swap! epoch update :template-revision inc)
                               program))}
      (fn []
        (is (= :ad-normalized-stale-rules
               (failure-reason #(compile-program [x]
                                  [result '(raster.numeric/* x x)] result ['x]))))))))

(deftest retained-result-declarations-do-not-coerce-the-primal
  (let [x (typed 'x 'double)
        result (typed 'result 'float)
        f (compile-program [x] [result 'x] result ['x])]
    (is (= :ad-generated-result-carrier (failure-reason #(f 2.0)))))
  (let [x (typed 'x 'double) result (typed 'result 'double)]
    (doseq [initializer [nil 2]]
      (let [f (compile-program [x] [result initializer] result ['x])]
        (is (= :ad-generated-result-carrier (failure-reason #(f 2.0)))))))
  (let [x (typed 'x 'float) result (typed 'result 'double)
        f (compile-program [x] [result 'x] result ['x])]
    (is (= :ad-generated-result-carrier (failure-reason #(f (float 2.0)))))))

(deftest constructor-projections-and-whole-use-share-all-cotangents
  (let [x (typed 'x 'double)
        square (typed 'square 'double)
        result (typed 'result 'double)
        f (compile-program [x]
                           [square '(raster.numeric/* x x)
                            'p ['square]
                            'selected '(clojure.core/nth p 0)
                            'observed '(clojure.core/first p)
                            result '(raster.numeric/+ selected observed)] result ['x])]
    (doseq [x [-2.0 0.0 3.0]]
      (let [[value pb] (f x)]
        (is (= (* 2.0 x x) value))
        (is (= [(* 4.0 x)] (pb 1.0)))))))

(deftest nested-products-and-aliases-rebase-owned-ports
  (let [x (typed 'x 'double)
        square (typed 'square 'double)
        result (typed 'result 'double)
        f (compile-program [x]
                           [square '(raster.numeric/* x x)
                            'outer [['x 'square] nil]
                            'copy 'outer
                            'inner '(clojure.core/nth copy 0)
                            'a '(clojure.core/first inner)
                            'b '(clojure.core/second inner)
                            result '(raster.numeric/+ a b)] result ['x])]
    (doseq [x [-2.0 0.0 3.0]]
      (let [[value pb] (f x)]
        (is (= (+ x (* x x)) value))
        (is (= [(+ 1.0 (* 2.0 x))] (pb 1.0)))))))

(deftest object-completion-has-a-structural-identity-record
  (let [x (typed 'x 'double)
        materialized (typed 'materialized 'double)
        square (typed 'square 'double)
        result (typed 'result 'double)
        f (compile-program [x]
                           [materialized (completion/complete-value completion/object-input-contract 'x)
                            square '(raster.numeric/* materialized materialized)
                            'p ['materialized 'square]
                            'a '(clojure.core/first p)
                            'b '(clojure.core/second p)
                            result '(raster.numeric/+ a b)] result ['x])]
    (let [[value pb] (f 3.0)]
      (is (= 12.0 value))
      (is (= [7.0] (pb 1.0))))))

(deftest arrays-in-products-remain-borrowed-and-cotangents-are-fieldwise
  (let [xs (typed 'xs 'doubles)
        selected (typed 'selected 'doubles)
        value (typed 'value 'double)
        result (typed 'result 'double)
        f (compile-program [xs]
                           ['p ['xs]
                            selected '(clojure.core/first p)
                            value '(raster.arrays/aget selected 1)
                            result '(raster.numeric/* value value)] result ['xs])
        input (double-array [2.0 3.0])
        [primal pb] (f input)]
    (is (= 9.0 primal))
    (is (= [2.0 3.0] (vec input)))
    (is (= [0.0 6.0] (vec (first (pb 1.0)))))))

(deftest repeated-array-fields-accumulate-with-independent-pullbacks
  (let [xs (typed 'xs 'doubles)
        a (typed 'a 'doubles) b (typed 'b 'doubles)
        av (typed 'av 'double) bv (typed 'bv 'double)
        result (typed 'result 'double)
        f (compile-program [xs]
                           ['p ['xs 'xs]
                            a '(clojure.core/first p) b '(clojure.core/second p)
                            av '(raster.arrays/aget a 0) bv '(raster.arrays/aget b 0)
                            result '(raster.numeric/+ av bv)] result ['xs])
        input (double-array [2.0 3.0])
        [value pb] (f input)
        first-gradient (first (pb 1.0))
        second-gradient (first (pb 2.0))]
    (is (= 4.0 value))
    (is (= [2.0 0.0] (vec first-gradient)))
    (is (= [4.0 0.0] (vec second-gradient)))
    (is (= [2.0 0.0] (vec first-gradient)))
    (is (not (identical? first-gradient second-gradient)))
    (is (= [2.0 3.0] (vec input)))))

(deftest an-unused-singular-field-has-no-cotangent-demand
  (let [x (typed 'x 'double) singular (typed 'singular 'double)
        result (typed 'result 'double)
        f (compile-program [x]
                           [singular '(raster.numeric/sqrt x)
                            'p ['x 'singular]
                            result '(clojure.core/first p)] result ['x])
        [value pb] (f 0.0)]
    (is (= 0.0 value))
    (is (= [1.0] (pb 1.0)))))

(deftest float-field-fanout-keeps-the-declared-tangent-carrier
  (let [x (typed 'x 'float)
        square (typed 'square 'float)
        a (typed 'a 'float) b (typed 'b 'float)
        result (typed 'result 'float)
        f (compile-program [x] [square '(raster.numeric/* x x)
                                'p ['square]
                                a '(clojure.core/first p)
                                b '(clojure.core/first p)
                                result '(raster.numeric/+ a b)] result ['x])
        [value pb] (f (float 3.0))
        gradient (first (pb (float 1.0)))]
    (is (= (float 18.0) value))
    (is (instance? Float gradient))
    (is (= (float 12.0) gradient))))

(deftest generated-product-ids-do-not-capture-existing-source-names
  (let [x (typed 'generated_seed__1 'double)
        square (typed 'square 'double)
        result (typed 'result 'double)
        f (compile-program [x] [square '(raster.numeric/* generated_seed__1 generated_seed__1)
                                'p ['square]
                                result '(clojure.core/first p)] result ['generated_seed__1])
        [value pb] (f 2.0)]
    (is (= 4.0 value))
    (is (= [4.0] (pb 1.0)))))

(deftest unsupported-structured-uses-refuse-instead-of-dropping-activity
  (let [x (typed 'x 'double)
        reason (fn [bindings result]
                 (try (compile-program [x] bindings result ['x]) nil
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (is (= :ad-product-unclosed-field
           (reason ['p ['(raster.numeric/* x x)]] 'p)))
    (is (= :ad-unsupported-product-use
           (reason ['p ['x] 'r '(clojure.core/count p)] 'r)))
    (is (= :ad-generated-product-result (reason ['p ['x]] 'p)))
    (is (= :ad-generated-product-region
           (reason ['p '(if true [x] [x])] 'p)))))
