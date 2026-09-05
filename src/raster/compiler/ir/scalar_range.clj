(ns raster.compiler.ir.scalar-range
  "Small reusable interval facts for typed scalar lowering.

   These facts are proof-only: they are derived from retained scalar/storage dtypes, explicit
   literals and exact widening casts. Unknown values retain their full declared range, so the
   analysis can certify `:no-overflow` only when a canonical scalar operation is representable
   for every runtime value admitted by the typed contract."
  (:require [raster.compiler.core.dtype :as dtype]))

(def ^:private integral-bounds
  {:byte [Byte/MIN_VALUE Byte/MAX_VALUE]
   :int [Integer/MIN_VALUE Integer/MAX_VALUE]
   :long [Long/MIN_VALUE Long/MAX_VALUE]})

(defn for-dtype
  "The complete representable interval for an integral dtype, else nil."
  [type]
  (when-let [[lower upper] (get integral-bounds (dtype/canon type))]
    {:lower lower :upper upper}))

(defn contained-in-dtype?
  "Whether every value in `range` is representable in `type`."
  [range type]
  (when-let [{minimum :lower maximum :upper} (for-dtype type)]
    (and range (<= minimum (:lower range)) (<= (:upper range) maximum))))

(defn literal
  "An exact integral literal interval when representable in `type`, else nil."
  [value type]
  (let [range {:lower value :upper value}]
    (when (contained-in-dtype? range type) range)))

(defn arithmetic
  "Exact interval transfer for a canonical binary integral `operator`.

  `+'`, `-'`, and `*'` keep the proof calculation unbounded, preventing host arithmetic from
  wrapping into a false certificate. Unknown input ranges produce nil."
  [operator operands]
  (when (every? some? operands)
    (let [lowers (mapv :lower operands)
          uppers (mapv :upper operands)]
      (case operator
        :+ {:lower (reduce +' lowers) :upper (reduce +' uppers)}
        :- (let [[left right] operands]
             {:lower (-' (:lower left) (:upper right))
              :upper (-' (:upper left) (:lower right))})
        :* (let [[left right] operands
                 products (map #(*' %1 %2)
                               [(:lower left) (:lower left) (:upper left) (:upper left)]
                               [(:lower right) (:upper right) (:lower right) (:upper right)])]
             {:lower (reduce min products) :upper (reduce max products)})
        nil))))
