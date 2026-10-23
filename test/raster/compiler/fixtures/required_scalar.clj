(ns raster.compiler.fixtures.required-scalar
  "Reduction, unused required quotient, and a later independent observable map."
  (:require [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac-dialect :as soac]))

(defn algorithm []
  (let [scalar (av/tensor {:dtype :double :shape []})
        array (av/tensor {:dtype :double :shape '[n]})
        step '(clojure.core/+ acc (clojure.core/* element element))
        obligation {:source-binding-id 1 :flags #{:preserve-source-evaluation}
                    :ordering :source}]
    (soac/make
     (soac/default-program-facts
      {:values {'x array 'n (av/tensor {:dtype :long :shape []})
                'divisor scalar 'total scalar 'unused-loss scalar 'out array}
       :inputs '[x n divisor]
       :equations {0 (soac/default-equation-facts {:source-binding-id 0})
                   1 (assoc-in (soac/default-equation-facts {:source-binding-id 1})
                               [:attributes :required-evaluation] obligation)
                   2 (soac/default-equation-facts {:source-binding-id 2})}
       :attributes {:required-evaluations {1 obligation}}})
     [(list '= 0 '[total]
            (list 'reduce {:index 'i :extent 'n :accumulators '[acc]
                           :identities [0.0] :dtypes [:double]
                           :algebra [(scan/certify-reassociation
                                      {:acc 'acc :init 0.0 :lambda step} :double)]}
                  '[x] [] (soac/lambda-form '[acc element] [step])))
      '(= 1 [unused-loss]
          (scalar {:dtypes [:double]} [total divisor]
                  (lambda [sum denominator] (region [] [(clojure.core// sum denominator)]))))
      '(= 2 [out]
          (map {:index i :extent n} [x] []
               (lambda [element] (region [] [(clojure.core/+ element 1.0)]))))]
     '[out])))
