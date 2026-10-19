(ns raster.compiler.core.scalar-conversion-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.core.scalar-conversion :as conversion]))

(defn- outcome [f x]
  (try
    [:value (Float/floatToRawIntBits (f x))]
    (catch IllegalArgumentException e [:exception (class e)])))

(deftest source-cast-and-numeric-conversion-have-distinct-overflow-contracts
  (is (= [:nearest-even :ieee] (conversion/policy :double :float)))
  (is (= [:nearest-even :trap]
         (conversion/source-policy :double :float 'clojure.core/float)))
  (is (= [:nearest-even :ieee]
         (conversion/source-policy :double :float 'clojure.core/unchecked-float)))
  (is (= [:exact :exact]
         (conversion/source-policy :float :float 'clojure.core/float)))
  (is (nil? (conversion/source-policy :double :float 'clojure.core/double)))
  (let [checked {:source-dtype :double :target-dtype :float
                 :rounding :nearest-even :overflow :trap :source-op 'clojure.core/float}
        unchecked (assoc checked :overflow :ieee :source-op 'clojure.core/unchecked-float)]
    (is (conversion/canonical-attributes? checked))
    (is (conversion/canonical-attributes? unchecked))
    (doseq [attributes [(assoc checked :overflow :ieee)
                        (assoc unchecked :overflow :trap)]]
      (is (not (conversion/canonical-attributes? attributes)))
      (is (thrown-with-msg?
           clojure.lang.ExceptionInfo #"invalid canonical scalar conversion"
           (conversion/project-canonical-to-source
            (list conversion/scalar-convert-symbol attributes 'value)))))
    (doseq [[attributes oracle]
            [[checked (fn [^double value] (float value))]
             [unchecked (fn [^double value] (unchecked-float value))]]]
      (let [parameter (with-meta 'value {:tag 'double})
            term (list conversion/scalar-convert-symbol attributes parameter)
            execute (eval (list 'fn [parameter] (conversion/project-canonical-to-source term)))]
        (doseq [value [0.0 -0.0 1.0000000596046448 Double/MIN_VALUE
                       Double/MAX_VALUE (- Double/MAX_VALUE)
                       Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY Double/NaN]]
          (is (= (outcome oracle value) (outcome execute value))
              (str (:source-op attributes) " value=" value)))))))

(deftest float-infinity-from-an-unchecked-cast-is-not-double-infinity
  (let [inner (list conversion/scalar-convert-symbol
                    {:source-dtype :double :target-dtype :float
                     :rounding :nearest-even :overflow :ieee
                     :source-op 'clojure.core/unchecked-float} 'value)
        outer (list conversion/scalar-convert-symbol
                    {:source-dtype :float :target-dtype :float
                     :rounding :exact :overflow :exact :source-op 'clojure.core/float} inner)
        parameter (with-meta 'value {:tag 'double})
        execute (eval (list 'fn [parameter] (conversion/project-canonical-to-source outer)))
        oracle (fn [^double value] (float (unchecked-float value)))]
    (doseq [value [Double/MAX_VALUE Double/POSITIVE_INFINITY Double/NEGATIVE_INFINITY Double/NaN]]
      (is (= (outcome oracle value) (outcome execute value))))))
