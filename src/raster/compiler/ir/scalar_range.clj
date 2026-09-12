(ns raster.compiler.ir.scalar-range
  "Small reusable interval facts for typed scalar lowering.

   These facts are proof-only: they are derived from retained scalar/storage dtypes, explicit
   literals and exact widening casts. Unknown values retain their full declared range, so the
   analysis can certify `:no-overflow` only when a canonical scalar operation is representable
   for every runtime value admitted by the typed contract."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.kernel-launch :as launch]))


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

(defn counted-loop-trips
  "Exact trip count for static positive-step loops; unknown bounds decline. Proof arithmetic is
   unbounded, including an inclusive endpoint at the maximum finite-width integer."
  ([lower upper step]
   (counted-loop-trips lower upper step :exclusive))
  ([lower upper step upper-bound]
   (when (and (integer? lower) (integer? upper) (integer? step) (pos? step)
              (contains? #{:exclusive :inclusive} upper-bound))
     (case upper-bound
       :exclusive (quot (+' (max 0 (-' upper lower)) (dec step)) step)
       :inclusive (if (> lower upper)
                    0
                    (inc (quot (-' upper lower) step)))))))

(defn counted-loop-index-range
  "Static positive-step counted-loop indices, INCLUDING the increment which exits the loop.
   Unknown bounds/steps return nil. Exact host arithmetic avoids overflow in the proof itself.
   Empty loops include only the initializer; no increment is evaluated. This is an index proof,
   not a proof about loop-carried accumulators or whether runtime scalar bounds fit their ABI."
  [lower upper step]
  (when-let [trips (counted-loop-trips lower upper step)]
    {:lower lower :upper (+' lower (*' trips step))}))

(defn accumulation-prefixes
  "Enclose every prefix of up to `count` additions of values in `term`, starting at `initial`.
   Unknown or nonintegral counts decline; proof arithmetic remains unbounded."
  [initial term count]
  (when (and (integer? count) (not (neg? count)))
    (arithmetic :+ [initial (arithmetic :* [term {:lower 0 :upper count}])])))

(defn additive-loop-ranges
  "Enclose carry values on body entry and after a static additive loop.
   The caller must independently prove the term range and its independence from evolving carries.
   Zero trips preserve the initializer without requiring a term range. Body entry excludes the
   final backedge, avoiding a spurious N+1 iteration. This does not select an overflow policy."
  [initial term trips]
  (when (and initial (integer? trips) (not (neg? trips)))
    (if (zero? trips)
      {:entry nil :result initial}
      (when term
        {:entry (accumulation-prefixes initial term (dec trips))
         :result (accumulation-prefixes initial term trips)}))))

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

(defn typed-index-range
  "Conditional interval proof for a bounded typed index tree.
   Leaf domains must hold at the access. Every intermediate must fit its declared integer
   width; exact widening, add/subtract/multiply and nonnegative div/rem by a positive constant
   are supported. Unknowns or a tree exceeding 128 nodes return nil, never an assumed bound."
  [expression leaf-types leaf-ranges]
  (try
    (let [remaining (volatile! 128)
          check! (fn check! [x]
                   (when (neg? (vswap! remaining dec))
                     (throw (ex-info "index proof budget" {})))
                   (cond
                     (launch/index-expr? x) (doseq [a (:arguments x)] (check! a))
                     (launch/index-cast? x) (check! (:argument x))
                     (or (integer? x) (symbol? x)) nil
                     :else (throw (ex-info "unsupported index proof leaf" {}))))]
      (check! expression)
      (launch/typed-expression-dtype expression leaf-types)
      (letfn [(visit [x]
                (let [type (launch/typed-expression-dtype x leaf-types)
                      result
                      (cond
                        (integer? x) (literal x type)
                        (symbol? x) (get leaf-ranges x)
                        (launch/index-cast? x) (visit (:argument x))
                        (launch/index-expr? x)
                        (let [args (mapv visit (:arguments x))
                              [a b] args]
                          (when (every? some? args)
                            (case (:op x)
                              (:add :mul)
                              (reduce (fn [a b]
                                        (let [r (arithmetic (if (= :add (:op x)) :+ :*) [a b])]
                                          (when (contained-in-dtype? r type) r)))
                                      args)
                              :sub (when (= 2 (count args)) (arithmetic :- args))
                              (:floor-div :mod)
                              (when (and (= 2 (count args)) (<= 0 (:lower a))
                                         (= (:lower b) (:upper b)) (pos? (:lower b)))
                                (if (= :floor-div (:op x))
                                  (quotient args)
                                  {:lower 0 :upper (min (:upper a) (dec (:lower b)))}))
                              nil)))
                        :else nil)]
                  (when (and result (<= (:lower result) (:upper result))
                             (contained-in-dtype? result type))
                    result)))]
        (visit expression)))
    (catch clojure.lang.ExceptionInfo _ nil)))
