(ns raster.compiler.backend.jvm.ordered-comparison-test
  "Compare emitted predicates with independent source behavior, including unordered operands."
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.backend.jvm.bytecode :as bytecode]))

(deftest floating-predicates-preserve-ordered-source-comparisons
  (let [operators [["<" <] ["<=" <=] [">" >] [">=" >=] ["==" ==]]
        carriers [['float 'float] ['double 'double] ['float 'double] ['double 'float]]
        specs (vec (for [[operation oracle] operators
                         [left right] carriers
                         branch? [false true]
                         :let [predicate (list (symbol "clojure.core" operation) 'x 'y)]]
                     {:name (gensym "comparison") :branch? branch?
                      :params [(with-meta 'x {:tag left}) (with-meta 'y {:tag right})]
                      :walked-body [(if branch? (list 'if predicate 17 -23) predicate)]
                      :source-ns *ns* :return-tag 'Object
                      :oracle oracle :carriers [left right] :operation operation}))
        artifact (bytecode/compile-specialized-class!
                  (str "raster.test.OrderedFloatingPredicates" (gensym)) specs)
        values [Double/NEGATIVE_INFINITY -1.0 -0.0 0.0 1.0 Double/POSITIVE_INFINITY Double/NaN]
        ;; Build actual Float infinities/NaNs, not a checked source float cast.
        convert (fn [carrier value] (if (= 'float carrier) (.floatValue ^Number value) (double value)))]
    (doseq [{:keys [name oracle carriers operation branch?]} specs
            x values y values
            :let [[left right] (mapv convert carriers [x y])
                  method (get-in artifact [:methods name])]]
      (is (= (if branch? (if (oracle left right) 17 -23) (oracle left right))
             (.invoke method nil (object-array [left right])))
          (pr-str [operation carriers branch? left right])))))

(deftest unary-floating-predicates-preserve-source-special-values
  (let [operators [["pos?" pos?] ["neg?" neg?] ["zero?" zero?]]
        specs (vec (for [[operation oracle] operators carrier ['float 'double 'Object]
                         branch? [false true]
                         :let [predicate (list (symbol "clojure.core" operation) 'x)]]
                     {:name (gensym "sign") :params [(with-meta 'x {:tag carrier})]
                      :walked-body [(if branch? (list 'if predicate 17 -23) predicate)]
                      :source-ns *ns* :return-tag 'Object
                      :oracle oracle :carrier carrier :operation operation :branch? branch?}))
        artifact (bytecode/compile-specialized-class!
                  (str "raster.test.FloatingSignPredicates" (gensym)) specs)]
    (doseq [{:keys [name oracle carrier operation branch?]} specs
            value [Double/NEGATIVE_INFINITY -1.0 -0.0 0.0 1.0 Double/POSITIVE_INFINITY Double/NaN]
            :let [value (if (= 'float carrier) (.floatValue ^Number value) value)
                  method (get-in artifact [:methods name])]]
      (is (= (if branch? (if (oracle value) 17 -23) (oracle value))
             (.invoke method nil (object-array [value])))
          (pr-str [operation carrier branch? value])))))
