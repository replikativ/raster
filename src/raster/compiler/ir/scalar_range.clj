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
  ;; Predicate values are deliberately outside `dtype/canon`: a predicate is a control
  ;; value, not a numeric scalar.  Range analysis is optional evidence, so it must be
  ;; harmless when asked about one.
  (when-let [[lower upper] (get integral-bounds (when (not= :predicate type)
                                                   (dtype/canon type)))]
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

(defn hull
  "The least interval containing every non-nil input interval, or nil when an input is
  unknown.  It is used for control-flow joins rather than as an assertion mechanism."
  [ranges]
  (when (every? some? ranges)
    {:lower (reduce min (map :lower ranges))
     :upper (reduce max (map :upper ranges))}))

(defn minmax
  "Sound interval transfer for canonical integral min/max."
  [operator operands]
  (when (every? some? operands)
    (case operator
      :min {:lower (reduce min (map :lower operands))
            :upper (reduce min (map :upper operands))}
      :max {:lower (reduce max (map :lower operands))
            :upper (reduce max (map :upper operands))}
      nil)))

(defn quotient
  "Interval transfer for truncating quotient by one statically positive divisor.

  Truncation toward zero is monotone for a positive divisor, including negative numerators.  We
  intentionally require an exact divisor literal/range: a zero or sign-varying denominator has
  distinct exceptional semantics and carries no schedule proof here."
  [operands]
  (when (every? some? operands)
    (let [[numerator divisor] operands
          divisor-value (:lower divisor)]
      (when (and (= divisor-value (:upper divisor)) (pos? divisor-value))
        {:lower (quot (:lower numerator) divisor-value)
         :upper (quot (:upper numerator) divisor-value)}))))
