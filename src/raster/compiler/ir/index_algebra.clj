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

(defn digits
  "Recover the mixed-radix digits of `index` (extent `extent`) from ordered region locals.

   Returns `{:digits {sym {:radix monomial :order n}} :quot-facts [{:factor sym :times m :le e}]}`.
   A `rem`-digit `(rem x m)` of an already known quantity `x` has radix `m`; a `quot`-digit
   `(quot x m)` has radix `extent(x)/m`, recorded as an opaque factor with the fact
   `m·factor ≤ extent(x)`. Locals that are neither are ignored (they may be arithmetic over
   digits, which `index-form` handles)."
  [index extent locals]
  (let [initial {:quantities {index {:extent nil :raw extent}}
                 :digits {index {:radix nil :order 0}}
                 :leaves #{index}
                 :parents {}
                 :substitutions {}
                 :quot-expressions {}
                 :quot-facts []
                 :order 1}
        ;; a radix or divisor is an extent: it may not mention the index or a digit. A
        ;; `(quot e c)` inside it is the scalar local that computed it, when one exists.
        resolve (fn [{:keys [substitutions quot-expressions digits]} form]
                  (let [m (monomial (->> (strip-cast form)
                                         (clojure.walk/postwalk-replace substitutions)
                                         (clojure.walk/postwalk strip-cast)
                                         (clojure.walk/postwalk-replace quot-expressions)))]
                    (when (and m (empty? (set/intersection (set (:factors m))
                                                           (set (keys digits)))))
                      m)))
        ;; the map extent may mention a quot the locals define later; resolve it on demand
        quantity-extent (fn [state quantity]
                          (or (get-in state [:quantities quantity :extent])
                              (when-let [raw (get-in state [:quantities quantity :raw])]
                                (resolve state raw))))]
    (reduce
     (fn [{:keys [quantities substitutions] :as state} {:keys [id init]}]
       (let [init (strip-cast init)
             source (when (seq? init) (strip-cast (second init)))]
         (cond
           ;; a digit: the remainder of a known quantity
           (and (rem-form? init) (contains? quantities source))
           (if-let [radix (resolve state (nth init 2))]
             (-> state
                 (assoc-in [:digits id] {:radix radix :order (:order state)})
                 (assoc-in [:quantities id] {:extent radix})
                 (assoc-in [:parents id] source)
                 (update :leaves #(-> % (disj source) (conj id)))
                 (update :order inc))
             state)

           ;; a digit: the quotient of a known quantity, with radix extent/divisor as an
           ;; opaque factor bounded by `divisor·factor ≤ extent`
           (and (quot-form? init) (contains? quantities source))
           (let [divisor (resolve state (nth init 2))
                 source-extent (quantity-extent state source)]
             (if (and divisor source-extent)
               (let [factor (symbol (str "rstr_quot_" (name id)))
                     radix {:const 1 :factors [factor]}]
                 (-> state
                     (assoc-in [:digits id] {:radix radix :order (:order state)})
                     (assoc-in [:quantities id] {:extent radix})
                     (assoc-in [:parents id] source)
                     (update :leaves #(-> % (disj source) (conj id)))
                     (update :quot-facts conj {:factor factor :times divisor :le source-extent})
                     (update :order inc)))
               state))

           ;; a scalar quotient of an extent (`hdim2 = (quot head-dim 2)`): an opaque factor
           ;; with its bound, usable in later radices and coefficients
           (quot-form? init)
           (if-let [[divisor bound] (let [d (resolve state (nth init 2))
                                          b (resolve state (second init))]
                                      (when (and d b) [d b]))]
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
                     (clojure.walk/postwalk-replace substitutions init))

           :else state)))
     initial
     locals)))

(defn- complete-digits
  "Resolve the map index's own radix (its extent) once every scalar local is known."
  [{:keys [digits] :as state} index]
  (let [extent (or (get-in digits [index :radix])
                   (let [{:keys [substitutions quot-expressions]} state
                         raw (get-in state [:quantities index :raw])]
                     (monomial (->> (strip-cast raw)
                                    (clojure.walk/postwalk-replace substitutions)
                                    (clojure.walk/postwalk strip-cast)
                                    (clojure.walk/postwalk-replace quot-expressions)))))]
    (assoc-in state [:digits index :radix] extent)))

;; ---------------------------------------------------------------------------------------------
;; Affine forms over digits

(defn- affine
  "Affine normal form `{:terms {sym monomial-coefficient} :const n}` of an index expression over
   the given digit symbols, or nil when it is not affine in them."
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
                    {:terms (merge-with (fn [x y] (product x y)) (:terms sum) (:terms a))
                     :const (+ (:const sum) (:const a))
                     :symbolic (vec (concat (:symbolic sum) (:symbolic a)))})))
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
          :else nil))
      :else nil)))

(defn index-form
  "The affine form of `expression` over the digits of `index`, with each term's radix, or nil.

   `loop-indices` is a map of loop index symbol → extent for counted loops inside the region;
   they are digits too, with their extent as radix."
  [expression index extent locals loop-indices]
  (let [{:keys [digits leaves parents substitutions quot-facts]}
        (complete-digits (digits index extent locals) index)
        loop-digits (reduce-kv (fn [acc loop-index loop-extent]
                                 (if-let [radix (monomial loop-extent)]
                                   (assoc acc loop-index {:radix radix :order (count acc)})
                                   acc))
                               {} loop-indices)
        digits (merge digits loop-digits)
        leaves (into leaves (keys loop-digits))
        digit-set (set (keys digits))
        expression (clojure.walk/postwalk-replace substitutions (strip-cast expression))
        form (affine expression digit-set)
        ;; the constant offset: an integer, or one symbolic extent monomial (not both)
        offset (when form
                 (let [symbolic (distinct (:symbolic form))]
                   (cond
                     (empty? symbolic) {:const (:const form) :factors []}
                     (and (= 1 (count symbolic)) (zero? (:const form)))
                     (monomial (first symbolic))
                     :else nil)))
        ancestors (fn ancestors [digit]
                    (when-let [parent (get parents digit)]
                      (cons parent (ancestors parent))))
        term-set (set (keys (:terms form)))]
    (when (and form offset (seq term-set)
               ;; a quantity used whole may not appear beside a digit derived from it
               (not-any? (fn [digit] (some term-set (ancestors digit))) term-set)
               (every? #(some? (get-in digits [% :radix])) term-set))
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

(defn- span
  "The monomial bound on Σ c_j (r_j − 1) + 1 over `terms`: Σ c_j·r_j is a sound over-approximation."
  [terms]
  (reduce (fn [acc {:keys [coefficient radix]}]
            (let [term (product coefficient radix)]
              (when (and acc term)
                ;; monomials do not add; a sound bound is the dominant term times the count
                (if (dominates? acc term) acc (if (dominates? term acc) term nil)))))
          {:const 0 :factors []}
          terms))

(defn complete?
  "Every leaf digit of the map index is covered by a term: itself, or an ancestor used whole.
   A dropped digit means two work items collide."
  [{:keys [terms leaves parents]}]
  (let [term-set (set (keys terms))
        covered? (fn covered? [digit]
                   (or (contains? term-set digit)
                       (when-let [parent (get parents digit)] (covered? parent))))]
    (every? covered? leaves)))

(defn injective?
  "Sufficient condition that the index form maps distinct digit tuples to distinct addresses:
   with terms ordered by coefficient dominance, every coefficient dominates the product of the
   radices of all lower terms (the row-major carry condition), and the form is complete."
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
                           required (if-let [next-term (nth ordered (inc k) nil)]
                                      (product (:coefficient next-term) (:radix next-term))
                                      {:const 1 :factors []})]
                       (dominates? coefficient required quot-facts)))
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
