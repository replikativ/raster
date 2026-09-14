(ns raster.compiler.passes.parallel.typed-soac-ownership
  "Prove cross-work-item ownership from canonical TypedSOAC effects.

   This pass never recognizes Clojure loop syntax. It consumes explicit Fold/effect-loop scopes
   and mixed-radix index forms, upgrading a sequential effect-map only when every access to each
   written destination has one injective, possibly translated mixed-radix address family.
   Distinct written destinations remain a binding precondition: KernelGraph call validation rejects
   overlapping writable resident views before private allocation or launch."
  (:require [clojure.set :as set]
            [clojure.walk :as walk]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.index-algebra :as index-algebra]
            [raster.compiler.ir.soac-dialect :as dialect]))

(declare expression-accesses)

(defn- expressions-accesses
  [expressions destination-parameters locals loops]
  (vec (mapcat #(expression-accesses % destination-parameters locals loops) expressions)))

(defn- local-accesses
  [local-values destination-parameters inherited-locals loops]
  (loop [remaining local-values, locals (vec inherited-locals), accesses []]
    (if-let [local (first remaining)]
      (recur (next remaining) (conj locals local)
             (into accesses
                   (expression-accesses (:init local) destination-parameters locals loops)))
      {:locals locals :accesses accesses})))

(defn- expression-accesses
  [expression destination-parameters locals loops]
  (cond
    (dialect/scalar-fold-form? expression)
    (let [{:keys [attributes lambda]} (dialect/scalar-fold-parts expression)
          {fold-locals :locals body-results :body-results} (dialect/lambda-parts lambda)
          loop {:index (:index attributes) :extent (:extent attributes) :lower 0 :kind :fold}
          prefix (expressions-accesses [(:identity attributes) (:extent attributes)]
                                       destination-parameters locals loops)
          scoped (local-accesses fold-locals destination-parameters locals (conj loops loop))]
      (into prefix
            (concat (:accesses scoped)
                    (expressions-accesses body-results destination-parameters
                                          (:locals scoped) (conj loops loop)))))

    (descriptor/aget-call? expression)
    (let [array (descriptor/aget-array-sym expression)
          arguments (descriptor/call-args expression)
          indices (vec (rest arguments))]
      (into (if (and (contains? destination-parameters array) (= 1 (count indices)))
              [{:kind :read :destination array :index (first indices)
                :locals locals :loops loops}]
              (if (contains? destination-parameters array)
                [{:kind :unsupported :reason :non-scalar-destination-read
                  :destination array :expression expression}]
                []))
            (expressions-accesses indices destination-parameters locals loops)))

    (contains? destination-parameters expression)
    [{:kind :unsupported :reason :opaque-destination-use :destination expression}]

    (and (seq? expression) (= 'quote (first expression))) []
    (coll? expression)
    (expressions-accesses (if (map? expression) (mapcat identity expression) expression)
                          destination-parameters locals loops)
    :else []))

(declare effect-accesses)

(defn- region-accesses
  [region destination-parameters inherited-locals loops]
  (let [scoped (local-accesses (:locals region) destination-parameters inherited-locals loops)]
    (into (:accesses scoped)
          (mapcat #(effect-accesses % destination-parameters (:locals scoped) loops)
                  (:body-results region)))))

(defn- effect-accesses
  [effect destination-parameters locals loops]
  (let [{:keys [region loop index lower upper-bound extent lambda carry destination destination-index
                predicate value conflict]}
        (dialect/effect-parts effect)]
    (cond
      region (region-accesses region destination-parameters locals loops)

      loop
      (let [loop-scope {:index index :extent extent :lower lower
                        :upper-bound upper-bound :kind :effect-loop}
            prefix (expressions-accesses (cond-> [lower extent] carry (conj (:init carry)))
                                         destination-parameters locals loops)
            body (dialect/lambda-parts lambda)
            scoped (local-accesses (:locals body) destination-parameters locals
                                   (conj loops loop-scope))]
        (into prefix
              (concat (:accesses scoped)
                      (mapcat #(effect-accesses % destination-parameters (:locals scoped)
                                               (conj loops loop-scope))
                              (:body-results body))
                      (when-let [result (:effect-result body)]
                        (expression-accesses result destination-parameters (:locals scoped)
                                             (conj loops loop-scope))))))

      destination
      (into (cond-> [{:kind :write :destination destination :index destination-index
                      :locals locals :loops loops}]
              (not (contains? #{true 1} predicate))
              (conj {:kind :unsupported :reason :guarded-destination-write
                     :destination destination :predicate predicate})
              (not (contains? #{:unique :ordered} conflict))
              (conj {:kind :unsupported :reason :unsupported-effect-conflict
                     :destination destination :conflict conflict}))
            (expressions-accesses [destination-index predicate value]
                                  destination-parameters locals loops))

      :else [{:kind :unsupported :reason :unknown-effect :effect effect}])))

(def ^:private minimum-operators
  "Canonical minimum spellings accepted only as an explicit executable bound. Devirtualized calls
   retain their original operator through op-descriptor metadata, so no implementation name is
   parsed here."
  '#{min clojure.core/min raster.numeric/min Math/min})

(defn- unwrap-bound-expression
  [expression local-initializers seen]
  (cond
    (and (symbol? expression) (contains? local-initializers expression)
         (not (contains? seen expression)))
    (recur (get local-initializers expression) local-initializers (conj seen expression))

    (and (seq? expression) (= 2 (count expression))
         (descriptor/cast-op? (first expression)))
    (recur (second expression) local-initializers seen)

    :else expression))

(defn- loop-radix-candidates
  "Return exact or explicitly dominating radices for a loop extent.

   `i < min(a,b)` proves that either `a` or `b` can safely be the physical row radix. This is an
   executable bound, unlike an ABI comment or a guessed relation between unrelated scalars. The
   later index algebra still has to prove that a candidate is invariant and is the actual address
   stride."
  [extent local-initializers]
  (let [resolved (unwrap-bound-expression extent local-initializers #{})
        operator (descriptor/semantic-op resolved)]
    (vec (distinct
          (cons extent
                (when (contains? minimum-operators operator)
                  (descriptor/call-args resolved)))))))

(defn- loop-radix-environments
  [loops local-initializers]
  (reduce
   (fn [environments {:keys [index extent]}]
     (vec (for [environment environments
                candidate (loop-radix-candidates extent local-initializers)]
            (assoc environment index candidate))))
   [{}] loops))

(defn- ownership-signature
  [{:keys [index locals loops]} outer-index outer-extent forbidden-index-symbols
   capture-substitutions]
  (let [local-initializers (into {} (map (juxt :id :init)) locals)
        address-dependencies
        (loop [symbols (util/free-syms index)]
          (let [expanded (into symbols
                               (mapcat #(some-> (get local-initializers %)
                                               util/free-syms))
                               symbols)]
            (if (= symbols expanded) symbols (recur expanded))))
        relevant-loops (filterv #(contains? address-dependencies (:index %)) loops)
        loop-indices (loop-radix-environments relevant-loops local-initializers)
        inner-indices (mapv :index relevant-loops)
        ;; A nested sequential loop may evaluate an address invariant in one of its indices
        ;; (for example every output component rereads the same per-head score row). Ownership is
        ;; across outer work items: require every digit which actually reaches the address to be
        ;; canonical and bounded, without demanding irrelevant lexical indices in the formula.
        expected-digits (into #{outer-index} inner-indices)
        canonical
        (when (every? #(and (= 0 (:lower %))
                            (not= :inclusive (:upper-bound %)))
                      relevant-loops)
          (some
           (fn [loop-environment]
             (let [form (index-algebra/index-form
                         index outer-index outer-extent locals loop-environment)
                   form-symbols (set (filter symbol? (tree-seq coll? seq form)))]
               (when (and form (index-algebra/injective? form)
                          (contains? (:terms form) outer-index)
                          (empty? (set/intersection forbidden-index-symbols form-symbols))
                          (= expected-digits (set (keys (:terms form))))
                          (= expected-digits (:leaves form))
                          (empty? (:parents form))
                          (nil? (:fixed-leaves form)))
                 (clojure.walk/postwalk-replace capture-substitutions form))))
           loop-indices))]
    (when canonical
      {:owner (get-in canonical [:terms outer-index])
       :inner (mapv #(get-in canonical [:terms %]) inner-indices)
       :offset (:offset canonical)
       :offset-terms (:offset-terms canonical)
       :form canonical
       :quot-facts (:quot-facts canonical)})))

(defn- carried-symbols
  "Values that evolve with an inner iteration or are derived from one. They are not invariant
   address factors even if an affine parser could otherwise treat a free symbol as a scalar."
  [lambda]
  (into #{}
        (mapcat (fn [form]
                  (cond
                    (dialect/scalar-fold-form? form)
                    (let [{:keys [attributes]} (dialect/scalar-fold-parts form)]
                      [(:accumulator attributes)])

                    (dialect/effect-loop-form? form)
                    (when-let [carry (:carry (dialect/effect-parts form))]
                      [(:parameter carry) (:result carry)]))))
        (filter #(or (dialect/scalar-fold-form? %)
                     (dialect/effect-loop-form? %))
                (tree-seq coll? seq lambda))))

(defn- equation-ownership
  [program equation]
  (let [{:keys [kind attributes arrays captures destinations lambda]}
        (dialect/operation-parts equation)
        layout (when (= 'effect-map kind) (dialect/parameter-layout equation))
        destination-parameters (set (:destination-parameters layout))
        capture-substitutions (zipmap (:capture-parameters layout) captures)
        values (:values (dialect/facts program))]
    (when (and (= 'effect-map kind)
               (= :sequential (:iteration-order attributes))
               (empty? arrays)
               ;; Read-only tensor captures do not affect cross-item ownership. If one reaches a
               ;; destination address, index-form cannot resolve that data-dependent expression
               ;; and the access still declines below.
               (every? #(contains? values %) captures)
               (seq destinations)
               (= (count destinations) (count destination-parameters)))
      (let [region (dialect/lambda-parts lambda)
            forbidden-index-symbols (carried-symbols lambda)
            accesses (region-accesses region destination-parameters [] [])
            unsupported (filter #(= :unsupported (:kind %)) accesses)
            memory (filter #(contains? #{:read :write} (:kind %)) accesses)
            grouped (group-by :destination memory)
            proofs
            (into {}
                  (keep (fn [[destination destination-accesses]]
                          (let [signatures (mapv #(ownership-signature % (:index attributes)
                                                                     (:extent attributes)
                                                                     forbidden-index-symbols
                                                                     capture-substitutions)
                                                 destination-accesses)]
                            (when (and (some #(= :write (:kind %)) destination-accesses)
                                       (every? some? signatures)
                                       ;; All accesses belong to one translated mixed-radix
                                       ;; address family. Repeated reads/writes at an address are
                                       ;; allowed within an item; distinct slabs must be proved
                                       ;; disjoint after cancelling their common translation.
                                       (apply = (map #(dissoc % :offset :offset-terms :form)
                                                     signatures))
                                       (or (= 1 (count (distinct
                                                        (map :offset-terms signatures))))
                                           (index-algebra/disjoint-translated-forms?
                                            (mapv :form signatures))))
                              [destination {:accesses (count destination-accesses)
                                            :signature (dissoc (first signatures) :form)
                                            :offsets (vec (distinct
                                                           (map :offset-terms signatures)))}]))))
                  grouped)]
        (when (and (empty? unsupported)
                   (= destination-parameters (set (keys grouped)) (set (keys proofs))))
          {:kind :outer-item-owned
           :equation (second equation)
           :destinations proofs})))))

(defn- apply-ownership
  [equation proof]
  (let [[equals equation-id results operation] equation
        [kind attributes arrays captures destinations lambda] operation
        lambda (walk/postwalk
                (fn [form]
                  (if (and (seq? form) (= 'effect (first form)) (= :ordered (nth form 2 nil)))
                    (util/remake form 'effect (second form) :unique
                                 (nth form 3) (nth form 4) (nth form 5))
                    form))
                lambda)
        attributes (-> attributes
                       (assoc :iteration-order :independent)
                       (update :attributes assoc :ownership-proof proof))]
    (util/remake equation equals equation-id results
                 (util/remake operation kind attributes arrays captures destinations lambda))))

(defn prove
  "Recompute canonical outer-item ownership and return `[program stats]`.

   A proof is deliberately local to this exact program version. Any later rewrite of accesses,
   aliases, bounds or storage must rerun this pass rather than carrying the certificate forward."
  [program]
  (let [program (dialect/validate! program)
        [equations proofs]
        (reduce (fn [[equations proofs] equation]
                  (if-let [proof (equation-ownership program equation)]
                    [(conj equations (apply-ownership equation proof)) (conj proofs proof)]
                    [(conj equations equation) proofs]))
                [[] []] (dialect/equations program))
        result (dialect/make (dialect/facts program) equations (dialect/outputs program))]
    [result {:effect-row-ownership-proofs (count proofs)
             :effect-row-ownership-equations (mapv :equation proofs)}]))
