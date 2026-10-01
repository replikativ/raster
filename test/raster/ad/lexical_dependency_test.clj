(ns raster.ad.lexical-dependency-test
  (:require [clojure.test :refer [deftest is]]
            [raster.ad.reverse :as reverse]
            [raster.ad.jvp :as jvp]))

(deftest forward-dependence-respects-binders-and-quoted-data
  (let [tangents {'x 'dx}]
    (doseq [expression ['(let* [x 7.0] x)
                        '(loop* [x 7.0] x)
                        '(fn* [x] x)
                        '(quote x)]]
      (is (false? (#'jvp/any-active? tangents expression)) (str expression)))
    (doseq [expression ['(let* [local x] local)
                        '(loop* [local x] local)
                        '(fn* [local] (+ local x))]]
      (is (true? (#'jvp/any-active? tangents expression)) (str expression)))
    (is (true? (#'jvp/any-active? {'count 'dcount} '(+ count 1.0))))))

(deftest reverse-soac-dependence-excludes-local-shadowing
  (let [activity {'x true}]
    (doseq [[body expected] [['(let* [x 7.0] x) false]
                            ['(let* [local x] local) true]
                            ['(quote x) false]]]
      (is (= expected
             (#'reverse/init-active? (list 'raster.par/map! 'out 'i 'n 'double body)
                                     activity)))
      (is (= expected
             (#'reverse/init-active? (list 'raster.par/reduce 'acc 0.0 'i 'n body)
                                     activity))))))

(deftest reverse-loop-initializers-see-prior-local-binders
  (is (false? (#'reverse/init-active? '(loop* [x 7.0 y x] y) {'x true})))
  (is (true? (#'reverse/init-active? '(loop* [local x y local] y) {'x true}))))

(deftest reverse-loop-carry-dependence-respects-lexical-scope
  (doseq [[expression expected]
          [['(let* [x 7.0] x) []]
           ['(quote x) []]
           ['(let* [local x] local) ['acc]]
           ['(loop* [local x] local) ['acc]]
           ['(fn* [local] (+ local x)) ['acc]]]]
    (is (= expected (#'reverse/loop-var-activity
                      [['acc expression]] ['acc] ['x])) (str expression)))
  (is (= ['acc] (#'reverse/loop-var-activity
                  [['acc 0.0]] ['next-value] ['x]
                  [['next-value '(let* [local x] local)]])))
  (is (= ['count 'sum] (#'reverse/loop-var-activity
                        [['count 'x] ['sum 0.0]]
                        ['count '(+ sum count)] ['x])))
  (is (= ['acc] (#'reverse/loop-var-activity
                  [['acc 0.0]] ['count] ['x] [['count 'x]]))))

(deftest reified-residual-captures-use-the-pullbacks-lexical-boundary
  ;; Isolate capture projection from derivative rules. The forward value `saved` occurs
  ;; only as quoted data or a nested local; `needed` and core-named `count` are genuine reads.
  (with-redefs-fn
    {#'reverse/process-bindings
     (fn [& _]
       {:fwd-bindings '[saved 3.0 needed 4.0 count 5.0]
        :rev-ctx {:bindings '[local (let* [saved 9.0] saved)
                             quoted (quote saved)
                             result (clojure.core/* needed count dy__rad)]}
        :param-adj-syms '[result]})}
    (fn []
      (let [pullback (reverse/reify-pullback 'x ['x])]
        (is (= '[needed count] (:captured-syms pullback)))
        (is (= '[dy__rad needed count] (:pullback-params pullback)))))))
