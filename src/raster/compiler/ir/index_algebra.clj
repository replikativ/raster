(ns raster.compiler.ir.index-algebra
  "Mixed-radix index forms and the injectivity facts a store contract needs.

   A map index `idx` over `[0, N)` is decomposed into digits by a `quot`/`rem` chain and
   re-assembled into a store index by an affine form over those digits. `index-form` recovers
   that structure from region locals; `injective?` proves the form maps distinct work items to
   distinct addresses, and `disjoint-offsets?` proves several stores of one region never
   collide. Every proof is a sufficient condition over symbolic extents: it either succeeds or
   declines, it never guesses.

   Extents are compared as monomials: an integer constant times a multiset of non-negative
   symbolic factors. `(quot e 2)` becomes an opaque factor that remembers `2·factor ≤ e`, which
   is exactly the fact a half-dimension layout needs.

   See .internal/design/index_algebra.md for the derivation."
  (:require [clojure.set :as set]
            [clojure.walk]))

;; ---------------------------------------------------------------------------------------------
;; Monomials over symbolic extents

(defn- strip-cast
  [expression]
  (if (and (seq? expression) (= 2 (count expression))
           (contains? '#{long int clojure.core/long clojure.core/int} (first expression)))
    (recur (second expression))
    expression))

(defn monomial
  "Normalize an extent expression to `{:const n :factors [sym …]}` (factors sorted, repeated
   for powers), or nil when the expression is not a product of non-negative factors."
  [expression]
  (let [expression (strip-cast expression)]
    (cond
      (integer? expression) (when (<= 0 expression) {:const (long expression) :factors []})
      (symbol? expression) {:const 1 :factors [expression]}
      (and (seq? expression) (contains? '#{* clojure.core/*} (first expression)))
      (reduce (fn [product operand]
                (when-let [m (monomial operand)]
                  (when product
                    {:const (* (:const product) (:const m))
                     :factors (vec (sort (concat (:factors product) (:factors m))))})))
              {:const 1 :factors []}
              (rest expression))
      :else nil)))

(defn- multiset
  [factors]
  (frequencies factors))

(defn product
  [& monomials]
  (reduce (fn [acc m]
            (when (and acc m)
              {:const (* (:const acc) (:const m))
               :factors (vec (sort (concat (:factors acc) (:factors m))))}))
          {:const 1 :factors []}
          monomials))

(defn add
  "The sum of two monomials when it is again a monomial (equal factor multisets), else nil."
  [a b]
  (when (and a b (= (:factors a) (:factors b)))
    {:const (+ (:const a) (:const b)) :factors (:factors a)}))

(defn- dominates-directly?
  [a b]
  (and a b
       (>= (:const a) (:const b))
       (every? (fn [[factor n]] (>= (get (multiset (:factors a)) factor 0) n))
               (multiset (:factors b)))))

(defn- scale
  [m n]
  (update m :const * n))

(defn- replace-factor
  [m factor replacement]
  (let [factors (:factors m)
        position (.indexOf ^java.util.List factors factor)]
    (when (<= 0 position)
      {:const (* (:const m) (:const replacement))
       :factors (vec (sort (concat (subvec factors 0 position)
                                   (subvec factors (inc position))
                                   (:factors replacement))))})))

(defn dominates?
  "Whether monomial `a` is at least monomial `b` for every non-negative assignment of the
   factors: `a`'s factors contain `b`'s as a multiset and its constant is at least `b`'s.

   `facts` are `{:factor f :times m :le e}` bounds (`m·f ≤ e`, from `(quot e m)`): when `b`
   mentions such an `f`, `a ≥ b` follows from `m·a ≥ b[f := e]`. Sufficient, not complete."
  ([a b] (dominates? a b []))
  ([a b facts]
   (boolean
    (and a b
         (or (dominates-directly? a b)
             (some (fn [{:keys [factor times le]}]
                     (when (some #{factor} (:factors b))
                       (dominates? (product a times)
                                   (replace-factor b factor le)
                                   (remove #(= factor (:factor %)) facts))))
                   facts))))))

;; ---------------------------------------------------------------------------------------------
;; Digits

(defn- quot-form?
  [expression]
  (and (seq? expression) (= 3 (count expression))
       (contains? '#{quot clojure.core/quot} (first expression))))

(defn- rem-form?
  [expression]
  (and (seq? expression) (= 3 (count expression))
       (contains? '#{rem clojure.core/rem mod clojure.core/mod} (first expression))))

(defn- quotient
  "The exact monomial quotient `a / b`, or nil when `b` does not divide `a` syntactically."
  [a b]
  (when (and a b (pos? (:const b)) (zero? (mod (:const a) (:const b))))
    (let [remaining (reduce (fn [factors f]
                              (let [position (.indexOf ^java.util.List factors f)]
                                (if (and factors (<= 0 position))
                                  (vec (concat (subvec factors 0 position)
                                               (subvec factors (inc position))))
                                  (reduced nil))))
                            (:factors a) (:factors b))]
      (when remaining
        {:const (quot (:const a) (:const b)) :factors remaining}))))

(defn digits
  "Recover the mixed-radix digits of `index` (extent `extent`) from ordered region locals.

   A quantity `x` of extent `E` is decomposed only by a matched pair `(quot x d)` / `(rem x d)`
   with the same divisor `d` dividing `E` exactly: the remainder digit has radix `d`, the
   quotient digit radix `E/d`, and `x` stops being a leaf only then (a lone `quot` or `rem` is a
   lossy projection, so `x` stays a leaf and a store using only the projection is incomplete).
   A scalar `(quot e c)` local becomes an opaque factor with the fact `c·factor ≤ e`. Locals
   that are products or sums are substituted into later expressions; any other local stays an
   unresolved symbol that `index-form` refuses."
  [index extent locals]
  (let [initial {:quantities {index {:extent nil :raw extent}}
                 :digits {index {:radix nil :order 0}}
                 :leaves #{index}
                 :parents {}
                 :pending {}
                 :locals #{}
                 :substitutions {}
                 :quot-expressions {}
                 :quot-facts []
                 :order 1}
        ;; a radix or divisor is an extent: it may not mention the index, a digit, or a region
        ;; local the algebra did not resolve (`y = (- 8 i)` varies with the item). A
        ;; `(quot e c)` inside it is the scalar local that computed it, when one exists.
        resolve (fn [{:keys [substitutions quot-expressions digits] unresolved :locals} form]
                  (let [m (monomial (->> (strip-cast form)
                                         (clojure.walk/postwalk-replace substitutions)
                                         (clojure.walk/postwalk strip-cast)
                                         (clojure.walk/postwalk-replace quot-expressions)))
                        factors (set (:factors m))]
                    (when (and m
                               (empty? (set/intersection factors (set (keys digits))))
                               (empty? (set/intersection factors
                                                         (set (remove (set (keys substitutions))
                                                                      unresolved)))))
                      m)))
        ;; the map extent may mention a quot the locals define later; resolve it on demand
        quantity-extent (fn [state quantity]
                          (or (get-in state [:quantities quantity :extent])
                              (when-let [raw (get-in state [:quantities quantity :raw])]
                                (resolve state raw))))]
    (reduce
     (fn [{:keys [quantities] :as state} {:keys [id init]}]
       (let [init (strip-cast init)
             source (when (seq? init) (strip-cast (second init)))
             state (update state :locals conj id)
             ;; a quot/rem digit of a known quantity: registered once its divisor divides the
             ;; quantity's extent exactly; the source leaves the leaf set only when both digits
             ;; of the same divisor exist
             decompose
             (fn [kind]
               (let [divisor (resolve state (nth init 2))
                     source-extent (quantity-extent state source)
                     radix (when (and divisor source-extent)
                             (case kind
                               :rem divisor
                               :quot (quotient source-extent divisor)))]
                 (if (and radix (pos? (:const radix)))
                   (let [state (-> state
                                   (assoc-in [:digits id] {:radix radix :order (:order state)})
                                   (assoc-in [:quantities id] {:extent radix})
                                   (assoc-in [:parents id] source)
                                   (assoc-in [:pending source divisor kind] id)
                                   (update :order inc))
                         pair (get-in state [:pending source divisor])]
                     (if (and (:quot pair) (:rem pair))
                       (update state :leaves #(-> % (disj source) (conj (:quot pair) (:rem pair))))
                       state))
                   state)))]
         (cond
           (and (rem-form? init) (contains? quantities source))
           (decompose :rem)

           (and (quot-form? init) (contains? quantities source))
           (decompose :quot)

           ;; a scalar quotient of an extent (`hdim2 = (quot head-dim 2)`): an opaque factor
           ;; with its bound, usable in later radices and coefficients
           (quot-form? init)
           (if-let [[divisor bound] (let [d (resolve state (nth init 2))
                                          b (resolve state (second init))]
                                      (when (and d b (pos? (:const d))) [d b]))]
             (-> state
                 (assoc-in [:substitutions id] id)
                 (assoc-in [:quot-expressions (clojure.walk/postwalk strip-cast init)] id)
                 (update :quot-facts conj {:factor id :times divisor :le bound}))
             state)

           ;; an affine or product local (`per-row = heads*hdim2`, `base = t*width + h`):
           ;; substitute it into later expressions
           (and (seq? init)
                (contains? '#{* + clojure.core/* clojure.core/+} (first init)))
           (assoc-in state [:substitutions id]
                     (clojure.walk/postwalk-replace (:substitutions state) init))

           :else state)))
     initial
     locals)))

(defn- complete-digits
  "Resolve the map index's own radix (its extent) once every scalar local is known, under the
   same guard as any radix: no digit and no unresolved local among its factors."
  [{:keys [digits substitutions quot-expressions] unresolved :locals :as state} index]
  (let [extent (or (get-in digits [index :radix])
                   (let [raw (get-in state [:quantities index :raw])
                         m (monomial (->> (strip-cast raw)
                                          (clojure.walk/postwalk-replace substitutions)
                                          (clojure.walk/postwalk strip-cast)
                                          (clojure.walk/postwalk-replace quot-expressions)))
                         factors (set (:factors m))]
                     (when (and m
                                (empty? (set/intersection factors (set (keys digits))))
                                (empty? (set/intersection
                                         factors (set (remove (set (keys substitutions))
                                                              unresolved)))))
                       m)))]
    (assoc-in state [:digits index :radix] extent)))

;; ---------------------------------------------------------------------------------------------
;; Affine forms over digits

(defn- monomial-form
  [{:keys [const factors]}]
  (apply list 'clojure.core/* const factors))

(defn- affine
  "Affine normal form `{:terms {sym monomial-coefficient} :const n}` of an index expression over
   the given digit symbols, or nil when it is not affine in them. A product of one affine
   factor by invariant monomials distributes: `(i·a + h)·b = i·(a·b) + h·b`."
  [expression digit-set]
  (let [expression (strip-cast expression)]
    (cond
      (integer? expression) {:terms {} :const (long expression)}
      (symbol? expression) (if (contains? digit-set expression)
                             {:terms {expression {:const 1 :factors []}} :const 0}
                             ;; a free symbolic scalar is a symbolic constant offset
                             {:terms {} :const 0 :symbolic [expression]})
      (and (seq? expression) (contains? '#{+ clojure.core/+} (first expression)))
      (reduce (fn [sum operand]
                (when-let [a (affine operand digit-set)]
                  (when sum
                    (let [terms (merge-with add (:terms sum) (:terms a))]
                      (when (every? some? (vals terms))
                        {:terms terms
                         :const (+ (:const sum) (:const a))
                         :symbolic (vec (concat (:symbolic sum) (:symbolic a)))})))))
              {:terms {} :const 0 :symbolic []}
              (rest expression))
      (and (seq? expression) (contains? '#{* clojure.core/*} (first expression)))
      (let [operands (rest expression)
            digit-operands (filter #(contains? digit-set (strip-cast %)) operands)
            others (remove #(contains? digit-set (strip-cast %)) operands)]
        (cond
          (and (= 1 (count digit-operands)) (every? monomial others))
          {:terms {(strip-cast (first digit-operands)) (apply product (map monomial others))}
           :const 0 :symbolic []}
          ;; a product without any digit is a symbolic offset
          (and (empty? digit-operands) (monomial expression))
          {:terms {} :const 0 :symbolic [expression]}
          ;; one non-monomial factor (a sum, or a nested product carrying digits) times
          ;; invariant monomials: distribute the scale over its terms and offsets
          :else
          (let [invariant? (fn [operand]
                             (and (monomial operand)
                                  (not (contains? digit-set (strip-cast operand)))))
                [inner :as affine-operands] (remove invariant? operands)
                scale (map monomial (filter invariant? operands))]
            (when (and (= 1 (count affine-operands)) (seq scale))
              (when-let [inner (affine inner digit-set)]
                (let [m (apply product scale)]
                  {:terms (into {} (map (fn [[digit coefficient]]
                                          [digit (product coefficient m)]))
                                (:terms inner))
                   :const 0
                   :symbolic (vec (concat (map (fn [symbolic]
                                                 (list 'clojure.core/* symbolic (monomial-form m)))
                                               (:symbolic inner))
                                          (when-not (zero? (:const inner))
                                            [(list 'clojure.core/* (:const inner)
                                                   (monomial-form m))])))}))))))
      :else nil)))

(defn index-form
  "The affine form of `expression` over the digits of `index`, with each term's radix, or nil.

   `loop-indices` is a map of loop index symbol → extent for counted loops inside the region;
   they are digits too, with their extent as radix."
  [expression index extent locals loop-indices]
  (let [{:keys [digits leaves parents substitutions quot-facts quot-expressions]
         unresolved :locals :as state}
        (complete-digits (digits index extent locals) index)
        ;; a loop extent is a radix: an extent monomial over resolved, digit-free factors
        invariant-extent (fn [form]
                           (let [m (monomial (->> (strip-cast form)
                                                  (clojure.walk/postwalk-replace substitutions)
                                                  (clojure.walk/postwalk strip-cast)
                                                  (clojure.walk/postwalk-replace quot-expressions)))
                                 factors (set (:factors m))]
                             (when (and m (pos? (:const m))
                                        (empty? (set/intersection factors (set (keys digits))))
                                        (empty? (set/intersection
                                                 factors
                                                 (set (remove (set (keys substitutions))
                                                              unresolved)))))
                               m)))
        loop-digits (reduce-kv (fn [acc loop-index loop-extent]
                                 (when acc
                                   (if-let [radix (invariant-extent loop-extent)]
                                     (assoc acc loop-index {:radix radix :order (count acc)})
                                     ;; a triangular or data-dependent loop extent is not a
                                     ;; radix; the loop index cannot be a digit, so the form is
                                     ;; undecidable rather than the index an invariant offset
                                     (reduced nil))))
                               {} loop-indices)
        digits (merge digits loop-digits)
        leaves (into leaves (keys loop-digits))
        digit-set (set (keys digits))
        expression (clojure.walk/postwalk-replace substitutions (strip-cast expression))
        ;; every local the index references must have become a digit or been substituted:
        ;; an unresolved local (a subtraction, a load, a cast of a float) is not invariant
        resolved (set/union digit-set (set (keys substitutions)))
        unresolved-locals (set/intersection (set (remove resolved unresolved))
                                            (set (filter symbol? (tree-seq coll? seq expression))))
        form (when (and loop-digits (empty? unresolved-locals)) (affine expression digit-set))
        ;; the constant offset: an integer, or the sum of the symbolic operands when that sum
        ;; is one monomial (`d + d = 2d`); a mixed or non-monomial sum is undecidable
        offset (when form
                 (let [symbolic (map monomial (:symbolic form))]
                   (cond
                     (empty? symbolic) {:const (:const form) :factors []}
                     (and (every? some? symbolic) (zero? (:const form)))
                     (reduce add symbolic)
                     :else nil)))
        ancestors (fn ancestors [digit]
                    (when-let [parent (get parents digit)]
                      (cons parent (ancestors parent))))
        term-set (set (keys (:terms form)))]
    (when (and form offset (seq term-set)
               ;; a quantity used whole may not appear beside a digit derived from it
               (not-any? (fn [digit] (some term-set (ancestors digit))) term-set)
               (every? #(some? (get-in digits [% :radix])) term-set)
               ;; coefficients and the offset are constant across the digits
               (every? (fn [[_ coefficient]]
                         (empty? (set/intersection (set (:factors coefficient)) digit-set)))
                       (:terms form))
               (empty? (set/intersection (set (:factors offset)) digit-set)))
      {:terms (into {} (map (fn [[digit coefficient]]
                              [digit {:coefficient coefficient
                                      :radix (get-in digits [digit :radix])}]))
                    (:terms form))
       :offset offset
       :leaves leaves
       :parents parents
       :quot-facts quot-facts})))

;; ---------------------------------------------------------------------------------------------
;; Proofs

(defn complete?
  "Every leaf digit of the map index is covered by a term: itself, or an ancestor used whole.
   A dropped digit means two work items collide."
  [{:keys [terms leaves parents]}]
  (let [term-set (set (keys terms))
        covered? (fn covered? [digit]
                   (or (contains? term-set digit)
                       (when-let [parent (get parents digit)] (covered? parent))))]
    (every? covered? leaves)))

(defn- supported-factor?
  "A symbolic factor of a coefficient is admissible only when some lower digit's radix vouches
   for it: it is a factor of that radix, or a quot fact bounds such a factor by it. Then a zero
   value of the factor empties the lower digit and no item stores at all, so `a·f ≥ a` may be
   used; an arbitrary captured scalar (`stride`) could be zero and collapse every item."
  [factor lower-radices quot-facts]
  (or (some #(some #{factor} (:factors %)) lower-radices)
      (some (fn [{:keys [factor* le]}]
              (and (some #(some #{factor*} (:factors %)) lower-radices)
                   (some #{factor} (:factors le))))
            (map #(set/rename-keys % {:factor :factor*}) quot-facts))))

(defn injective?
  "Sufficient condition that the index form maps distinct digit tuples to distinct addresses:
   with terms ordered by coefficient dominance, every coefficient dominates the product of the
   radices of all lower terms (the row-major carry condition), every coefficient's symbolic
   factors are vouched for by a lower radix, and the form is complete."
  [{:keys [terms quot-facts] :as form}]
  (boolean
   (and form
        (complete? form)
        (let [entries (vec (vals terms))
              ;; sort by dominance: insertion sort with the fact-aware comparison
              ordered (reduce (fn [sorted entry]
                                (let [position (count (take-while
                                                       #(dominates? (:coefficient %)
                                                                    (:coefficient entry)
                                                                    quot-facts)
                                                       sorted))]
                                  (vec (concat (subvec sorted 0 position) [entry]
                                               (subvec sorted position)))))
                              [] entries)]
          (and
           (every? (fn [[a b]] (dominates? (:coefficient a) (:coefficient b) quot-facts))
                   (partition 2 1 ordered))
           ;; Nested row-major carry: each coefficient covers the whole span of the next
           ;; level, `c_k ≥ c_{k+1}·r_{k+1}`, and the innermost stride is at least one. By
           ;; induction `c_k ≥ Σ_{j>k} c_j (r_j − 1) + 1`, so distinct digit tuples never meet.
           (every? (fn [k]
                     (let [{:keys [coefficient]} (nth ordered k)
                           lower (subvec ordered (inc k))
                           required (if-let [next-term (first lower)]
                                      (product (:coefficient next-term) (:radix next-term))
                                      {:const 1 :factors []})]
                       (and (pos? (:const coefficient))
                            (every? #(supported-factor? % (map :radix lower) quot-facts)
                                    (:factors coefficient))
                            (dominates? coefficient required quot-facts))))
                   (range (count ordered))))))))

(defn- offset-multiple
  "The integer `q` with `offset = q·stride` for monomials, or nil."
  [offset stride]
  (when (and offset stride
             (= (:factors offset) (:factors stride))
             (pos? (:const stride))
             (zero? (mod (:const offset) (:const stride))))
    (quot (:const offset) (:const stride))))

(defn disjoint-offsets?
  "Sufficient condition that stores sharing one index form and differing only by constant
   offsets never collide within one work item or across items.

   Offsets are extent expressions (`0`, `hdim2`, `(* 2 hdim2)`). They must be multiples
   `q·d` of one stride monomial `d` with `0 ≤ q < m`; the store ordinal then becomes one more
   digit of radix `m` and coefficient `d`, and the row-major carry condition decides. For RoPE
   (`+ i` and `+ i + hdim2` under `h·head-dim`) the fact `2·hdim2 ≤ head-dim` closes it."
  [{:keys [terms quot-facts] :as form} offsets]
  (let [monomials (mapv #(if (map? %) % (monomial %)) offsets)]
    (boolean
     (and form
          (every? some? monomials)
          (or (<= (count (distinct monomials)) 1)
              (let [zero {:const 0 :factors []}
                    nonzero (remove #(= zero %) (distinct monomials))
                    ;; the smallest non-zero offset is the candidate stride
                    stride (reduce (fn [a b] (if (dominates? a b quot-facts) b a)) nonzero)
                    multiples (mapv #(if (= zero %) 0 (offset-multiple % stride)) (distinct monomials))]
                (and (every? some? multiples)
                     (= (count multiples) (count (distinct multiples)))
                     (let [radix {:const (inc (long (apply max multiples))) :factors []}
                           ordinal (symbol "rstr_store_ordinal")]
                       (injective?
                        (-> form
                            (assoc-in [:terms ordinal] {:coefficient stride :radix radix})
                            (update :leaves conj ordinal)))))))))))
