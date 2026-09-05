(ns raster.compiler.passes.parallel.typed-soac-frontend
  "Direct analyzed-source to TypedSOAC construction for the closed map/scalar/reduce/certified-scan subset.

   This is the production front door for TypedSOAC.  It consumes the retained source form and
   walker type metadata directly; it never constructs the older record graph.  Unsupported
   parallel forms return nil so the pipeline can use the explicitly separate compatibility route.
   Once an operation is accepted, all value, effect and provenance facts are made explicit before
   fusion or scheduling sees it."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.contraction-facts :as contraction-facts]
            [raster.compiler.ir.index-algebra :as index-algebra]
            [raster.compiler.ir.form :as form]
            [raster.compiler.ir.par :as par]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.fusion-support :as fusion-support]
            [raster.compiler.passes.scalar.effects :as effects]))

(defn- fail!
  [reason message data]
  (throw (ex-info message (assoc data :reason reason :front-end :analyzed-source))))

(def source-decline-reasons
  "Structured analysis failures that mean source lies outside the closed TypedSOAC subset."
  #{:unsupported-scalar-binding
    :source-value-conflict
    :scan-dtype-unsupported
    :scan-not-associative
    :scan-not-elementwise
    :scan-element-impure-or-unknown
    :scan-nonidentity-init
    :reduction-dtype-unsupported
    :reduction-not-associative
    :reduction-not-elementwise
    :reduction-element-impure-or-unknown
    :reduction-nonidentity-init
    :typed-soac-production-subset
    :typed-soac-stable-read-alias
    :typed-soac-syntax
    :typed-soac-unbound-scalar
    :typed-soac-unknown-value
    :unique-index-not-provable})

(defn source-decline?
  [exception]
  (contains? source-decline-reasons (:reason (ex-data exception))))

(defn- extract-io
  [body index outputs & {:keys [accumulator]}]
  (let [inputs (par/collect-aget-arrays body)
        written (par/collect-aset-arrays body)
        excluded (set/union inputs (set outputs) written #{index}
                            (if accumulator #{accumulator} #{})
                            descriptor/aget-ops descriptor/aset-ops
                            #{'do 'let 'let* 'if 'double 'float 'int 'long})]
    {:inputs inputs
     :outputs (into (set outputs) written)
     :scalars (set/difference (util/free-syms body) excluded)}))

(defn- strip-index-cast
  [expression]
  (if (and (seq? expression)
           (contains? #{'long 'int 'clojure.core/long 'clojure.core/int}
                      (first expression))
           (= 2 (count expression)))
    (second expression)
    expression))

(def ^:private unique-index-ops
  '#{raster.par/unique-index unique-index})

(def ^:private atomic-add-ops
  '#{raster.par/atomic-add! atomic-add!})

(defn- unique-index-expression
  "Return the inner destination expression when `expression` carries Raster's explicit
   uniqueness contract. The marker may be a direct source call or a walker-devirtualized call;
   semantic-op/call-args are the only sanctioned way to look through the latter."
  [expression]
  (when (and (seq? expression)
             (contains? unique-index-ops (descriptor/semantic-op expression))
             (= 1 (count (descriptor/call-args expression))))
    (first (descriptor/call-args expression))))

(defn- atomic-add-call?
  [expression]
  (and (seq? expression)
       (contains? atomic-add-ops (descriptor/semantic-op expression))
       (= 3 (count (descriptor/call-args expression)))))

(defn- same-symbol?
  [left right]
  (if (and (symbol? left) (symbol? right))
    (= (name left) (name right))
    (= left right)))

(defn- additive-update-contribution
  "Return the contribution in destination[index] + contribution.

   The contribution must not read the destination again: such a read would occur outside the
   eventual atomic update and would therefore be racy. This recognizes algebra, not a source
   primitive, so generic effect maps and indexed operations share one conflict proof."
  [destination destination-index value]
  (when (and (seq? value)
             (contains? #{'+ 'clojure.core/+} (descriptor/semantic-op value)))
    (let [arguments (vec (descriptor/call-args value))
          accumulator-read?
          (fn [expression]
            (and (descriptor/aget-call? expression)
                 (same-symbol? destination (descriptor/aget-array-sym expression))
                 (= (strip-index-cast destination-index)
                    (strip-index-cast (descriptor/aget-index expression)))))]
      (when (= 2 (count arguments))
        (let [[left right] arguments
              contribution (cond
                             (accumulator-read? left) right
                             (accumulator-read? right) left)]
          (when (and contribution
                     (not-any? #(same-symbol? destination %)
                               (par/collect-aget-arrays contribution)))
            contribution))))))

(defn- retained-local-dtype
  [binding init]
  (let [init-tag (when (instance? clojure.lang.IObj init)
                   (or (:raster.type/tag (meta init)) (:tag (meta init))))]
    (or (dtype/dtype-for-scalar-tag (types/sym-type-tag binding))
        (dtype/dtype-for-scalar-tag init-tag))))

(defn- typed-region-locals
  "Translate a flat source binding vector into explicit typed lexical SSA.

   This is deliberately metadata-only: a missing walker/TypedClojure fact declines the direct
   route instead of introducing another local type inference registry."
  [bindings]
  (when (and (vector? bindings) (even? (count bindings)))
    (let [locals (mapv (fn [[binding init]]
                         {:id binding :dtype (retained-local-dtype binding init) :init init})
                       (partition 2 bindings))]
      (when (and (= (count locals) (count (distinct (map :id locals))))
                 (every? :dtype locals))
        locals))))

(defn- integer-case-chain
  "Project one closed-core integer `case*` into the conditional scalar vocabulary.

   Clojure has already bound the tested expression before producing `case*`, so accepting only a
   symbolic test preserves its evaluate-once semantics.  The clause map's dispatch hashes are an
   implementation detail; exact integer test values are retained in its `[test-value result]`
   entries.  Other case representations decline instead of leaking hash/keyword interpretation
   into a target emitter."
  [expression]
  (when (and (seq? expression) (= 'case* (first expression)))
    (let [[_ test _shift _mask default clauses _switch-type test-type] expression
          ordered-clauses (when (and (map? clauses) (every? integer? (keys clauses)))
                            (->> clauses
                                 (sort-by key)
                                 (mapv val)))]
      (when (and (symbol? test)
                 (= :int test-type)
                 (map? clauses)
                 (every? integer? (keys clauses))
                 (every? #(and (vector? %) (= 2 (count %))
                                (integer? (first %)))
                         ordered-clauses))
        (reduce (fn [otherwise [test-value result]]
                  (list 'if (list 'clojure.core/== test test-value)
                        result otherwise))
                default (reverse ordered-clauses))))))

(defn- substitute-store
  [substitutions store]
  (reduce (fn [store field]
            (update store field #(util/subst-syms substitutions %)))
          store [:index :predicate :value]))

(defn- substitute-loop
  "Apply `substitutions` inside a counted store loop: its extent, its body locals' inits and its
   stores. The loop index itself is bound by the loop and is never substituted."
  [substitutions {:keys [index] :as loop}]
  (let [substitutions (dissoc substitutions index)]
    (-> loop
        (update :extent #(util/subst-syms substitutions %))
        (update :locals (fn [locals]
                          (mapv #(update % :init (fn [init] (util/subst-syms substitutions init)))
                                locals)))
        (update :stores (fn [stores] (mapv #(substitute-store substitutions %) stores))))))

(defn- region-order
  "The source order of a region's effects as `[:store i]` / `[:loop j]` entries into its
   `:stores` and `:loops` vectors. Regions without loops are their stores in order."
  [{:keys [stores loops order]}]
  (when (and (nil? order) (seq loops))
    (throw (ex-info "a region with store loops must carry its source order"
                    {:reason :raster/bug :loops (count loops) :stores (count stores)})))
  (or order (mapv (fn [ordinal] [:store ordinal]) (range (count stores)))))

(defn- rename-loop-index
  "Give a counted store loop the index name `fresh`, renaming its body locals' inits and its
   stores. Sibling loops that reuse one source index name (`i`) would otherwise share a
   KernelBody SSA value."
  [{:keys [index] :as loop} fresh]
  (let [renames {index fresh}]
    (-> loop
        (assoc :index fresh)
        (update :locals (fn [locals]
                          (mapv #(update % :init (fn [init] (util/subst-syms renames init)))
                                locals)))
        (update :stores (fn [stores] (mapv #(substitute-store renames %) stores))))))

(defn- rebase-region-locals
  "Move a recursively recognized region behind `offset` lexical SSA values.

   Every nested scope initially numbers its locals from zero.  Rebasing before composing scopes
   gives the combined region one deterministic, collision-free SSA namespace. Counted store
   loops keep their own body locals; only their references to the enclosing locals are renamed."
  [{:keys [locals stores loops] :as region} offset]
  (let [renames (into {}
                      (map-indexed
                       (fn [ordinal {:keys [id]}]
                         [id (symbol (str "rstr_local_" (+ offset ordinal)))])
                       locals))]
    {:locals (mapv (fn [{:keys [id] :as local}]
                     (-> local
                         (assoc :id (get renames id))
                         (update :init #(util/subst-syms renames %))))
                   locals)
     :stores (mapv #(substitute-store renames %) stores)
     :loops (mapv #(substitute-loop renames %) (or loops []))
     :order (region-order region)}))

(defn- strip-trailing-recur
  "Remove a counted loop's `(recur (inc i))` from the tail of its body, returning the remaining
   statements form, or nil when the tail is not that exact step."
  [form index]
  (let [step? (fn [x]
                (and (seq? x) (= 'recur (first x)) (= 2 (count x))
                     (let [update (strip-index-cast (second x))]
                       (or (and (seq? update) (= 2 (count update))
                                (contains? '#{inc clojure.core/inc} (first update))
                                (= index (strip-index-cast (second update))))
                           (and (seq? update) (= 3 (count update))
                                (contains? '#{+ clojure.core/+} (first update))
                                (= index (strip-index-cast (second update)))
                                (= 1 (strip-index-cast (nth update 2))))))))]
    (cond
      (step? form) '(do)
      (and (seq? form) (= 'do (first form)) (step? (last form)))
      (list* 'do (butlast (rest form)))
      ;; (let* [...] stmt… (recur …)) — the closed-core spelling puts the statements directly
      ;; in the let body; normalize them into one `do` before the recursive recognition.
      (and (seq? form) (symbol? (first form)) (form/let-head? (first form)) (<= 3 (count form)))
      (let [[head bindings & statements] form
            inner (if (= 1 (count statements))
                    (first statements)
                    (list* 'do statements))]
        (when-let [stripped (strip-trailing-recur inner index)]
          (list head bindings stripped)))
      :else nil)))

(declare store-region)

(defn- counted-store-loop
  "Recognize a counted loop of stores inside an effect-map body.

   Accepted shapes are the unexpanded `(dotimes [i n] …)` and its closed-core form
   `(loop* [i 0] (if (< i n) (do … (recur (inc i)))))`. The body is recognized by `store-region`
   with the map index as its context; nested store loops decline. The loop's body locals live in
   their own SSA namespace so they cannot collide with the enclosing region's locals."
  [form index]
  (let [[head bindings & body] (when (seq? form) form)
        counted (cond
                  (and (contains? '#{dotimes clojure.core/dotimes} head)
                       (vector? bindings) (= 2 (count bindings)) (symbol? (first bindings)))
                  {:index (first bindings) :lower 0 :extent (strip-index-cast (second bindings))
                   :body (list* 'do body)}

                  (and (symbol? head) (form/loop-head? head)
                       (vector? bindings) (= 2 (count bindings))
                       (symbol? (first bindings))
                       (integer? (second bindings)) (<= 0 (second bindings))
                       (= 1 (count body)))
                  (let [[loop-index lower] bindings
                        conditional (first body)]
                    (when (and (seq? conditional)
                               (contains? '#{if clojure.core/if} (first conditional))
                               (<= 3 (count conditional) 4)
                               (nil? (nth conditional 3 nil)))
                      (let [[_ test then] conditional]
                        (when (and (seq? test) (= 3 (count test))
                                   (contains? '#{< clojure.core/<} (first test))
                                   (= loop-index (strip-index-cast (second test))))
                          (when-let [statements (strip-trailing-recur then loop-index)]
                            {:index loop-index :lower lower
                             :extent (strip-index-cast (nth test 2))
                             :body statements}))))))]
    (when (and counted
               (not= (:index counted) index)
               (not (contains? (util/free-syms (:extent counted)) (:index counted))))
      (when-let [region (store-region (:body counted) index)]
        (when (and (seq (:stores region)) (empty? (:loops region)))
          ;; The loop index and the body locals get their own names: a source loop index may
          ;; shadow a captured scalar of the enclosing region, and the region's SSA namespace
          ;; must not confuse the two.
          (let [loop-index (symbol (str "rstr_loop_index_" (name (:index counted))))
                renames (into {(:index counted) loop-index}
                              (map (fn [{:keys [id]}]
                                     [id (symbol (str "rstr_loop_" (name id)))])
                                   (:locals region)))]
            {:locals []
             :stores []
             :loops [{:index loop-index
                      :lower (:lower counted)
                      :extent (:extent counted)
                      :locals (mapv (fn [{:keys [id] :as local}]
                                      (-> local (assoc :id (get renames id))
                                          (update :init #(util/subst-syms renames %))))
                                    (:locals region))
                      :stores (mapv #(substitute-store renames %) (:stores region))}]
             :order [[:loop 0]]}))))))

(defn- store-region
  "Recognize an ordered, pure local-SSA spine ending exclusively in certified effects.

   Local types come only from retained walker/TypedClojure facts. Nested lexical scopes are
   alpha-renamed into one ordered SSA spine; missing local types decline because guessing them in
   this source recognizer or a C emitter would make the region only nominally typed. Direct stores
   and atomic additions become data here; cross-work-item legality is checked separately after
   local dependencies are expanded for analysis."
  [body index]
  (cond
    (and (seq? body) (form/let-head? (first body)))
    (let [[_ bindings & nested-body] body]
      (when (and (even? (count bindings))
                 (seq nested-body)
                 (not-any? util/effectful? (take-nth 2 (rest bindings))))
        (when-let [nested-region (store-region (list* 'do nested-body) index)]
          (let [pairs (vec (partition 2 bindings))
                typed (mapv (fn [[binding init]]
                              [binding init (retained-local-dtype binding init)])
                            pairs)]
            (when (every? (comp some? #(nth % 2)) typed)
              (let [{:keys [locals substitutions]}
                    (reduce (fn [{:keys [locals substitutions]} [binding init local-dtype]]
                              (let [id (symbol (str "rstr_local_" (count locals)))]
                                {:locals (conj locals
                                               {:id id :dtype local-dtype
                                                :init (util/subst-syms substitutions init)})
                                 :substitutions (assoc substitutions binding id)}))
                            {:locals [] :substitutions {}} typed)
                    nested (rebase-region-locals nested-region (count locals))]
                {:locals (into locals
                               (map #(update % :init
                                             (fn [init]
                                               (util/subst-syms substitutions init))))
                               (:locals nested))
                 :stores (mapv #(substitute-store substitutions %)
                               (:stores nested))
                 :loops (mapv #(substitute-loop substitutions %) (:loops nested))
                 :order (region-order nested)}))))))

    (and (seq? body) (= 'do (first body)))
    (let [expressions (vec (rest body))]
      (if (= 1 (count expressions))
        (store-region (first expressions) index)
        (let [groups (mapv #(store-region % index) expressions)]
          (when (and (seq groups) (every? some? groups)
                     (every? (comp empty? :locals) groups))
            (reduce (fn [{:keys [stores loops order]} group]
                      (let [store-offset (count stores)
                            loop-offset (count loops)]
                        {:locals []
                         :stores (into stores (:stores group))
                         :loops (into loops (:loops group))
                         :order (into order
                                      (map (fn [[kind ordinal]]
                                             [kind (+ ordinal (if (= :store kind)
                                                                store-offset
                                                                loop-offset))]))
                                      (region-order group))}))
                    {:locals [] :stores [] :loops [] :order []}
                    groups)))))

    (and (seq? body)
         (contains? #{'if 'clojure.core/if} (first body))
         (<= 3 (count body) 4))
    (let [[_ predicate then-expression else-expression] body
          then-region (store-region then-expression index)
          else-region (when else-expression (store-region else-expression index))
          aligned? (and else-region
                        (= (mapv (juxt :out :index) (:stores then-region))
                           (mapv (juxt :out :index) (:stores else-region))))
          merged-predicate (fn [then-predicate else-predicate]
                             ;; both branches store unconditionally: the merged store is a
                             ;; full write, not a partial one that would force an in/out read
                             (if (and (contains? #{true 1} then-predicate)
                                      (contains? #{true 1} else-predicate))
                               true
                               (list 'if predicate then-predicate else-predicate)))
          branch-value (fn [{:keys [locals stores]}]
                         ;; A branch that computes its own locals before one store is the
                         ;; store of a scoped value: keep the locals lexically inside the
                         ;; branch instead of executing them unconditionally.
                         (let [bindings (vec (mapcat (fn [{:keys [id init]}] [id init]) locals))
                               value (:value (first stores))]
                           (if (seq bindings) (list 'let* bindings value) value)))]
      (cond
        ;; Store loops under a branch would need predicated loop regions; decline for now.
        (or (seq (:loops then-region)) (seq (:loops else-region)))
        nil

        ;; Both branches store once to the same destination and at least one branch owns
        ;; locals: one predicated store of a value-if over the two scoped branch values.
        (and then-region else-region aligned?
             (= 1 (count (:stores then-region)) (count (:stores else-region)))
             (or (seq (:locals then-region)) (seq (:locals else-region))))
        (let [then-store (first (:stores then-region))
              else-store (first (:stores else-region))]
          {:locals []
           :stores [(assoc then-store
                           :value (list 'if predicate
                                        (branch-value then-region)
                                        (branch-value else-region))
                           :predicate (merged-predicate (:predicate then-store)
                                                        (:predicate else-store)))]})

        (and then-region
             (empty? (:locals then-region))
             (or (nil? else-expression)
                 (and else-region (empty? (:locals else-region)))))
        {:locals []
         :stores
         (if aligned?
           (mapv (fn [ordinal then-store]
                   (let [else-store (nth (:stores else-region) ordinal)]
                     (assoc then-store
                            :value (list 'if predicate (:value then-store) (:value else-store))
                            :predicate (merged-predicate (:predicate then-store)
                                                         (:predicate else-store)))))
                 (range) (:stores then-region))
           (vec
            (concat
             (map (fn [store]
                    (update store :predicate
                            #(if (contains? #{true 1} %)
                               predicate
                               (list 'if predicate % 0))))
                  (:stores then-region))
             (map (fn [store]
                    (update store :predicate
                            #(let [else-predicate (list 'if predicate 0 1)]
                               (if (contains? #{true 1} %)
                                 else-predicate
                                 (list 'if else-predicate % 0)))))
                  (:stores else-region)))))}))

    (and (seq? body) (= 'case* (first body)))
    (when-let [conditional (integer-case-chain body)]
      (store-region conditional index))

    (and (seq? body) (symbol? (first body))
         (or (form/loop-head? (first body))
             (contains? '#{dotimes clojure.core/dotimes} (first body))))
    (counted-store-loop body index)

    (descriptor/aset-call? body)
    (let [arguments (vec (descriptor/call-args body))]
      (when (= 3 (count arguments))
        (let [raw-index (nth arguments 1)
              unique-index (unique-index-expression raw-index)
              value (nth arguments 2)
              ;; A dense destination[i] update is still an ordinary pointwise map: its old-value
              ;; read belongs in the scalar expression and needs no cross-work-item conflict
              ;; algebra. Extract an atomic contribution only for a genuinely indirect index.
              contribution (when (and (not unique-index)
                                      (not= (strip-index-cast index)
                                            (strip-index-cast raw-index)))
                             (additive-update-contribution
                              (descriptor/aset-array-sym body) raw-index value))
              cast? (and (seq? value)
                         (contains? #{'float 'double 'int 'long
                                      'clojure.core/float 'clojure.core/double
                                      'clojure.core/int 'clojure.core/long}
                                    (first value))
                         (= 2 (count value)))]
          {:locals []
           :stores [{:out (descriptor/aset-array-sym body)
                     :index (strip-index-cast (or unique-index raw-index))
                     :conflict (when unique-index :unique)
                     :reduction-op (when contribution '+)
                     :predicate 1
                     :value (if contribution
                              contribution
                              (if cast? (second value) value))
                     :cast (when (and cast? (not contribution)) (first value))}]})))

    (atomic-add-call? body)
    (let [[destination raw-index raw-value] (descriptor/call-args body)
          cast? (and (seq? raw-value)
                     (contains? #{'float 'double 'int 'long
                                  'clojure.core/float 'clojure.core/double
                                  'clojure.core/int 'clojure.core/long}
                                (first raw-value))
                     (= 2 (count raw-value)))]
      {:locals []
       :stores [{:out destination
                 :index (strip-index-cast raw-index)
                 :reduction-op '+
                 :predicate 1
                 :value (if cast? (second raw-value) raw-value)
                 :cast (when cast? (first raw-value))}]})

    :else nil))

(defn- independent-stores?
  [_locals stores]
  (let [destinations (mapv :out stores)
        destination-set (set destinations)]
    (and (= (count destinations) (count destination-set))
         (every? (fn [{:keys [out index predicate value]}]
                   ;; Region locals are lexical SSA snapshots evaluated before the
                   ;; store sequence.  A local may therefore read any destination
                   ;; without introducing an ordering dependency between tuple
                   ;; results.  Only a DIRECT sibling-destination read in a store
                   ;; expression observes an earlier write and must decline the
                   ;; functional effect-map representation.
                   (empty? (disj (set/intersection destination-set
                                                   (par/collect-aget-arrays
                                                    (list 'do index predicate value)))
                                 out)))
                 stores))))

(defn- ordered-effects-safe?
  "An ordered effect may not observe a sibling destination directly after stores begin.

   Typed locals are evaluated before the effect sequence and may deliberately snapshot destination
   state. Direct reads in coordinates, predicates, or values would instead make cross-work-item
   ordering observable and require a stronger schedule than an effect map."
  [locals stores & [loop-expressions]]
  (let [destinations (set (map :out stores))]
    (and (seq stores)
         (every? (fn [{:keys [index predicate value]}]
                   (empty? (set/intersection
                            destinations
                            (par/collect-aget-arrays (list 'do index predicate value)))))
                 stores)
         ;; A region local is a snapshot only within its own work item: another item's store
         ;; may already be visible when it is taken, so a local that reads a destination makes
         ;; the cross-item order observable. Loop-body locals and extents are re-evaluated
         ;; inside the loop and are checked for the same reason.
         (empty? (set/intersection destinations
                                   (par/collect-aget-arrays
                                    (list* 'do (concat (map :init locals) loop-expressions)))))
         ;; Locals are an ordered pre-effect SSA spine, so their reads are snapshots rather than
         ;; sibling effect dependencies. Keep the argument explicit for this legality boundary.
         (vector? locals))))

(defn- binder-array-types
  "Element dtypes of the arrays a source `let` binds, read from the walker's array tags on the
   binders (`floats`, `doubles`, …). A use-site symbol carries no metadata, so an internal
   allocation such as an attention output would otherwise have no declared dtype and every map
   writing it would decline. Declared `array-types` take precedence.

   Float-family tags follow the kernel dtype exactly as `derive-param-types` maps the
   parameters: a kernel compiled at `:float` reads and writes float buffers throughout, so a
   double-declared allocation inside it is a float buffer of that kernel, not a second precision."
  [pairs array-types dtype]
  (let [kernel-dtype (some-> dtype dtype/canon)
        policy (fn [element]
                 (if (and kernel-dtype (dtype/fp-dtype? kernel-dtype)
                          (contains? #{:float :double} element))
                   kernel-dtype
                   element))]
    (merge
     (into {}
           (keep (fn [[binder init]]
                   (when (symbol? binder)
                     (when-let [element (some-> (types/sym-type-tag binder)
                                                dtype/dtype-for-array-tag dtype/canon)]
                       ;; Only an allocation that does not name its element type (`alloc-like`,
                       ;; `zeros-like`, `similar`) follows the policy. `float-array` states its
                       ;; element type, and a parallel form's result binder carries the form's
                       ;; own explicit dtype (`pmap … double`); retyping either would contradict
                       ;; a declared fact.
                       [binder (let [operation (when (seq? init) (descriptor/semantic-op init))
                                     named-element?
                                     (or (contains? descriptor/alloc-sym->array-tag operation)
                                         (contains? descriptor/alloc-sym->array-tag
                                                    (some-> operation name clojure.core/symbol)))]
                                 (if (and operation (descriptor/alloc-op? operation)
                                          (not named-element?))
                                   (policy element)
                                   element))]))))
           pairs)
     (or array-types {}))))

(defn- destination-dtype
  [array-types destination]
  (or (get array-types destination)
      (when (symbol? destination)
        (or (get array-types (symbol (name destination)))
            (dtype/dtype-for-scalar-tag (types/sym-type-tag destination))))))

(defn- effect-result-id
  [equation-id ordinal]
  [:effect-map equation-id ordinal])

(defn- typed-result-transform
  "Close a source epilogue descriptor over lexical scalar-region parameters.

   Program value IDs remain on the boundary records; the expression names only lambda parameters
   and segment indices. This is what makes later ParallelProgram SSA remapping alpha-stable."
  [epilogue]
  (when epilogue
    (dialect/make-result-transform
     {:accumulator (:acc epilogue)
      :expression (:expr epilogue)
      :operands (mapv #(-> % (assoc :value (:sym %)) (dissoc :sym)) (:operands epilogue))
      :scalars (mapv #(-> % (assoc :value (:sym %)) (dissoc :sym)) (:scalars epilogue))
      :result-dtype (or (:dtype epilogue) :float)})))

(defn- strip-index-casts
  [expression]
  (util/postwalk-preserving-meta strip-index-cast expression))

(def ^:dynamic ^:private *scalar-definitions*
  "Host scalar bindings preceding the operation being described, as `{id expression}` for the
   pure product/sum definitions (the normalizer's `rstr_extent_n` ids among them). The index
   algebra needs an extent's product structure, which its SSA id hides."
  {})

(defn- expand-scalar-definitions
  [expression]
  (loop [expression expression remaining (inc (count *scalar-definitions*))]
    (let [expanded (util/subst-syms *scalar-definitions* expression)]
      (if (or (= expanded expression) (zero? remaining))
        expanded
        (recur expanded (dec remaining))))))

(defn- invariant-read-atoms
  "Replace every array read in `form` that is uniform across the map with an atom symbol: a
   read of an array the region does not write, at an index free of the map index, the loop
   indices and the region locals (`(aget posbuf 0)`, a position established before the map).
   Such a read is a scalar the algebra can treat as an invariant factor. `atoms` memoizes one
   symbol per distinct read so equal reads stay equal."
  [form destinations varying atoms]
  (descriptor/rewrite-aget-reads
   form
   (fn [read]
     (let [array (descriptor/aget-array-sym read)
           index (descriptor/aget-index read)]
       (when (and (symbol? array)
                  (not (contains? destinations array))
                  (empty? (set/intersection varying (util/free-syms index))))
         (let [key (list 'clojure.core/aget array index)]
           (or (get @atoms key)
               (let [atom-symbol (clojure.core/symbol (str "rstr_read_" (count @atoms)))]
                 (swap! atoms assoc key atom-symbol)
                 atom-symbol))))))))

(defn- store-index-form
  "The mixed-radix index form of a store's destination index over the map index (extent
   `extent`), the region locals and, for a loop store, its loop's locals and index. Host
   scalar definitions are expanded first so extents and strides show their factors, and
   uniform array reads become invariant atoms (`destinations` are the region's written arrays)."
  [store index extent locals loops destinations]
  (let [loop (when (:loop store) (nth loops (:loop store)))
        varying (into #{index}
                      (concat (map :id locals) (map :id (:locals loop))
                              (map :index loops)))
        atoms (atom {})
        expand (fn [form]
                 (invariant-read-atoms (expand-scalar-definitions form) destinations varying atoms))]
    (index-algebra/index-form (expand (:index store)) index (expand extent)
                              (mapv #(update % :init expand) (concat locals (:locals loop)))
                              (if loop {(:index loop) (expand (:extent loop))} {}))))

(defn- proven-unique-stores
  "The stores whose destination writes are provably injective across work items: each store's
   index form is injective, and stores sharing a destination share one form and write at
   provably disjoint offsets. Returns the set of store ordinals."
  [stores index extent locals loops]
  (let [destinations (set (map :out stores))
        forms (mapv #(store-index-form % index extent locals loops destinations) stores)
        by-destination (group-by (fn [ordinal] (:out (nth stores ordinal)))
                                 (range (count stores)))]
    (into #{}
          (mapcat (fn [[_ ordinals]]
                    (let [group-forms (map #(nth forms %) ordinals)]
                      (when (and (every? some? group-forms)
                                 (every? index-algebra/injective? group-forms)
                                 (apply = (map :terms group-forms))
                                 ;; stores at one address (the arms of a conditional) are the
                                 ;; same element of one work item; distinct offsets must be
                                 ;; disjoint across work items
                                 (index-algebra/disjoint-offsets?
                                  (first group-forms) (distinct (map :offset group-forms))))
                        ordinals))))
          by-destination)))

(defn- effect-leaves
  "The store effects of a description, descending into store loops."
  [effects]
  (vec (mapcat (fn [effect]
                 (if-let [loop (:loop effect)]
                   (effect-leaves (:effects loop))
                   [effect]))
               effects)))

(defn- write-region-description
  [id symbol index extent {:keys [locals stores loops] :as region} elem-type
   & {:keys [host-return array-types] :or {host-return :effect array-types {}}}]
  (let [order (region-order region)
        loops (vec (map-indexed (fn [ordinal loop]
                                  (rename-loop-index
                                   loop (clojure.core/symbol (str "rstr_loop_index_" ordinal))))
                                (or loops [])))]
    (when (and (or (seq stores) (seq loops))
               (or (= :effect host-return) (and (empty? loops) (= 1 (count stores))))
               ;; A destination written both directly and inside a store loop would lose the work
               ;; item's source order between the two kinds of write; such regions stay unsupported.
               (empty? (set/intersection (set (map :out stores))
                                         (set (mapcat #(map :out (:stores %)) loops)))))
      (let [stores (mapv #(merge {:index index :predicate 1} %) stores)
            loop-stores
            (vec (mapcat (fn [ordinal {loop-store-list :stores}]
                           (map (fn [store]
                                  (assoc (merge {:index index :predicate 1} store)
                                         :loop ordinal))
                                loop-store-list))
                         (range) loops))
            dense-pointwise? (and (empty? loops)
                                  (every? #(and (= index (:index %)) (nil? (:reduction-op %)))
                                          stores))
            pointwise? (and dense-pointwise? (independent-stores? locals stores))
            stores (if pointwise?
                     (mapv (fn [{:keys [out predicate value] :as store}]
                             (assoc store :value
                                    (if (contains? #{true 1} predicate)
                                      value
                                      ;; A guarded dense write preserves its caller-owned
                                      ;; destination. Making that read explicit yields an ordinary
                                      ;; inout map instead of a hidden conditional effect.
                                      (list 'if predicate value
                                            (list 'clojure.core/aget out index)))))
                           stores)
                     stores)
            candidate-stores (into stores loop-stores)
            ;; Uniqueness is a fact the index algebra proves. A source `unique-index` marker is
            ;; an author's claim: it is redundant when the proof succeeds, it is the only
            ;; certificate when the index depends on data the algebra cannot see (an indirect
            ;; index through another array, established at runtime by validate-block-indices!),
            ;; and it is an error when the index is in the algebra's reach and the proof fails.
            proven (proven-unique-stores (vec (remove :reduction-op candidate-stores))
                                         index extent locals loops)
            proven? (let [indices (vec (remove :reduction-op candidate-stores))]
                      (fn [store] (contains? proven (.indexOf ^java.util.List indices store))))
            ;; A marker is honoured only for an index outside the algebra's reach: the index
            ;; expression or a local it depends on (transitively) reads an array. Unrelated
            ;; locals may not authorize a claim, so only the index's dependency slice counts.
            data-dependent? (fn [{:keys [index] :as store}]
                              (let [scope (concat locals
                                                  (:locals (when (:loop store)
                                                             (nth loops (:loop store)))))
                                    inits (into {} (map (juxt :id :init)) scope)
                                    slice (loop [pending (set (filter symbol?
                                                                      (tree-seq coll? seq index)))
                                                 seen #{}
                                                 expressions [index]]
                                            (let [next (set/difference pending seen)]
                                              (if (empty? next)
                                                expressions
                                                (let [found (keep inits next)]
                                                  (recur (set (filter symbol?
                                                                      (mapcat #(tree-seq coll? seq %)
                                                                              found)))
                                                         (set/union seen next)
                                                         (into expressions found))))))]
                                (seq (par/collect-aget-arrays (list* 'do slice)))))
            all-stores (mapv (fn [{:keys [out reduction-op conflict] :as store}]
                               (let [destination-type (some-> (destination-dtype array-types out)
                                                             dtype/canon)
                                     uniqueness (cond
                                                  reduction-op nil
                                                  (proven? store) :proven
                                                  (and (= :unique conflict) (data-dependent? store))
                                                  :claimed
                                                  (= :unique conflict)
                                                  (fail! :unique-index-not-provable
                                                         "a unique-index claim on an index the algebra can decide must be provable; the proof is a sufficient condition, so the form declines to the compatibility route"
                                                         {:binding id :destination out
                                                          :index (:index store)})
                                                  :else nil)
                                     contract (cond
                                                reduction-op
                                                (when destination-type
                                                  (dialect/reducing-scatter-conflict
                                                   reduction-op destination-type))
                                                uniqueness :unique
                                                (and (nil? (:loop store)) (= index (:index store)))
                                                :unique
                                                :else :ordered)]
                                 (assoc store :effect-conflict contract
                                              :destination-dtype destination-type
                                              :uniqueness uniqueness)))
                             candidate-stores)
            ;; One physical destination has one cross-item contract. If any ordinary store may
            ;; collide, its otherwise-unique siblings must join the same sequential contract. A
            ;; reduction mixed with an ordered overwrite remains unsupported: those are distinct
            ;; operations and must not be blurred into one conflict certificate.
            all-stores (reduce (fn [current [_ grouped]]
                                 (let [contracts (set (map :effect-conflict grouped))]
                                   (if (and (contains? contracts :ordered)
                                            (set/subset? contracts #{:unique :ordered}))
                                     (mapv (fn [store]
                                             (if (= (:out (first grouped)) (:out store))
                                               (assoc store :effect-conflict :ordered)
                                               store))
                                           current)
                                     current)))
                               all-stores (group-by :out all-stores))
            stores (vec (remove :loop all-stores))
            effect-contracts (mapv :effect-conflict all-stores)
            uniform-conflict (when (= 1 (count (set effect-contracts)))
                               (first effect-contracts))
            scatter? (and (not pointwise?)
                          (empty? loops)
                          (independent-stores? locals stores)
                          (or (= :unique uniform-conflict)
                              (dialect/reducing-scatter-conflict? uniform-conflict)))
            ordered? (and (not dense-pointwise?) (not scatter?) (= :effect host-return)
                          (every? some? effect-contracts)
                          (every? (fn [[_ grouped]]
                                    (= 1 (count (set (map :effect-conflict grouped)))))
                                  (group-by :out all-stores)))
            all-locals (vec (concat locals (mapcat :locals loops)))
            iteration-order (when ordered?
                              (if (or (some #(= :ordered (:effect-conflict %)) all-stores)
                                      (not (ordered-effects-safe?
                                            locals all-stores
                                            (concat (map :extent loops)
                                                    (map :init (mapcat :locals loops))))))
                                :sequential
                                :independent))
            destinations (if ordered?
                           (vec (distinct (map :out all-stores)))
                           (mapv :out stores))
            values (mapv :value all-stores)
            write-indices (mapv :index all-stores)
            predicates (mapv :predicate all-stores)
            analysis-values (concat (map :init all-locals) (map :extent loops)
                                    write-indices predicates values)
            io (update (extract-io (list* 'do analysis-values) index destinations)
                       :scalars set/difference (set (map :id all-locals))
                       (set (map :index loops)))
            results (if (= :buffer host-return)
                      [symbol]
                      (mapv #(effect-result-id id %) (range (count destinations))))
            result-dtypes (mapv #(some-> (destination-dtype array-types %) dtype/canon)
                                destinations)
            store-effect (fn [store]
                           (select-keys store [:out :index :predicate :value :cast
                                               :effect-conflict]))
            loop-effects (mapv (fn [ordinal {loop-index :index loop-locals :locals
                                             :keys [lower extent]}]
                                 {:loop {:index loop-index :lower lower :extent extent
                                         :locals loop-locals
                                         :effects (mapv store-effect
                                                        (filter #(= ordinal (:loop %))
                                                                all-stores))}})
                               (range) loops)
            ;; Effects keep the region's source order across direct stores and loops.
            ordered-effects (mapv (fn [[kind ordinal]]
                                    (if (= :store kind)
                                      (store-effect (nth stores ordinal))
                                      (nth loop-effects ordinal)))
                                  order)]
        (when (or pointwise? scatter? (and ordered? (every? some? result-dtypes)))
          (merge {:kind (cond pointwise? :map scatter? :scatter :else :effect-map)
                  :id id :sym symbol :index index :extent extent
                  :results results :locals locals :bodies values :casts (mapv :cast all-stores)
                  :write-indices write-indices :predicates predicates
                  :conflict (when scatter? uniform-conflict)
                  :effects (when ordered? ordered-effects)
                  :iteration-order iteration-order
                  :result-dtypes (when ordered? result-dtypes)
                  :effect-only? (= :effect host-return)
                  :host-binding symbol :elem-type elem-type
                  :result-storage
                  (mapv (fn [destination]
                          {:destination destination
                           :access (if (or scatter?
                                           (and ordered?
                                                (some #(and (= destination (:out %))
                                                            (dialect/reducing-scatter-conflict?
                                                             (:effect-conflict %)))
                                                      all-stores))
                                           (contains? (:inputs io) destination))
                                     :read-write :write)
                           :host-return host-return})
                        destinations)}
                 io))))))

(defn- reducing-scatter-description
  [id symbol out values indices extent operator default-dtype]
  (let [index (clojure.core/symbol (str "scatter_i__" id))
        contribution (list 'clojure.core/aget values index)
        destination-index (list 'clojure.core/aget indices index)
        elem-type (dtype/canon default-dtype)
        io (extract-io (list 'do contribution destination-index) index [out])]
    (merge {:kind :scatter :id id :sym symbol :index index :extent extent
            :results [symbol] :locals [] :casts [nil] :bodies [contribution]
            :write-indices [destination-index] :predicates [1]
            :conflict (dialect/reducing-scatter-conflict operator elem-type)
            :effect-only? false :host-binding symbol :elem-type elem-type
            :result-storage [{:destination out :access :read-write
                              :host-return :buffer}]
            :source-operation :reducing-scatter}
           io)))

(defn- operation-description
  [id symbol expression default-dtype array-types]
  (cond
    ;; SplitMix64 is pointwise scalar algebra, not a semantic parallel primitive. Preserve the
    ;; public convenience operation's fixed-width ABI here, then expose an ordinary typed map to
    ;; fusion and scheduling. Wrapping arithmetic remains explicit in the scalar source and is
    ;; retained by scalar-expression-body when a schedule becomes KernelBody SSA.
    (par/par-rng-fill-form? expression)
    (let [{:keys [seeds n base-seed]} (par/extract-par-rng-fill-info expression)
          typed-symbol (fn [value tag]
                         (if (symbol? value)
                           (with-meta value (merge (meta value)
                                                  {:tag tag :raster.type/tag tag}))
                           value))
          extent (typed-symbol (descriptor/unwrap-int-cast n) 'int)
          base (typed-symbol (descriptor/unwrap-int-cast base-seed) 'long)
          index (with-meta (clojure.core/symbol (str "rstr_rng_index_" id))
                  {:tag 'long :raster.type/tag 'long})
          local (fn [name]
                  (with-meta (clojure.core/symbol (str "rstr_rng_" name "_" id))
                    {:tag 'long :raster.type/tag 'long}))
          state (local "state")
          s1 (local "s1")
          s2 (local "s2")
          s3 (local "s3")
          s4 (local "s4")
          s5 (local "s5")
          locals [{:id state :dtype :long
                   :init (list 'unchecked-add base
                               (list 'unchecked-multiply (list 'long index) par/SM-GAMMA))}
                  {:id s1 :dtype :long
                   :init (list 'bit-xor state
                               (list 'unsigned-bit-shift-right state 30))}
                  {:id s2 :dtype :long :init (list 'unchecked-multiply s1 par/SM-MIX1)}
                  {:id s3 :dtype :long
                   :init (list 'bit-xor s2 (list 'unsigned-bit-shift-right s2 27))}
                  {:id s4 :dtype :long :init (list 'unchecked-multiply s3 par/SM-MIX2)}
                  {:id s5 :dtype :long
                   :init (list 'bit-xor s4 (list 'unsigned-bit-shift-right s4 31))}]]
      {:kind :map :id id :sym symbol :results [symbol]
       :index index :extent extent :locals locals :casts [nil] :bodies [s5]
       :inputs #{} :outputs #{seeds} :scalars (set (filter symbol? [base]))
       :result-storage [{:destination seeds :access :write :host-return :buffer}]
       :host-binding symbol :elem-type :long :source-operation :raster.par/rng-fill!})

    (par/par-scatter-form? expression)
    (let [{:keys [out src index n stride]} (par/extract-par-scatter-info expression)]
      (when-not stride
        (reducing-scatter-description id symbol out src index n '+ default-dtype)))

    (par/par-reduce-by-key-form? expression)
    (let [{:keys [out keys vals n op]} (par/extract-par-reduce-by-key-info expression)]
      (when (contains? #{'+ 'clojure.core/+} op)
        (reducing-scatter-description id symbol out vals keys n op default-dtype)))

    ;; A flat gather is semantically an ordinary dense map with one pointwise index input and one
    ;; stable, indirectly-read data input. Keep it in that small functional vocabulary; the JVM
    ;; scheduler may later select hardware vgather from the typed scalar region, while GPU targets
    ;; consume the same SegMap. normalize-source has already expressed a strided gather as the
    ;; corresponding flattened map with one hoisted product extent.
    (par/par-gather-form? expression)
    (let [{:keys [out src index n stride]} (par/extract-par-gather-info expression)]
      (when-not stride
        (let [idx (clojure.core/symbol (str "gather_i__" id))
              body (list 'clojure.core/aget src
                         (list 'clojure.core/aget index idx))
              elem-type (dtype/canon default-dtype)
              io (extract-io body idx [out])]
          (merge {:kind :map :id id :sym symbol :results [symbol]
                  :index idx :extent n :locals [] :casts [nil] :bodies [body]
                  :result-storage [{:destination out :access :write
                                    :host-return :buffer}]
                  :host-binding symbol :elem-type elem-type
                  :source-operation :raster.par/gather}
                 io))))

    (par/par-map-pure-form? expression)
    (let [{:keys [idx bound cast body elem-type]} (par/extract-par-map-pure-info expression)
          elem-type (dtype/canon (or elem-type
                                     (dtype/dtype-for-scalar-tag cast)
                                     default-dtype))
          io (extract-io body idx [symbol])]
      (merge {:kind :map :id id :sym symbol :results [symbol]
              :index idx :extent bound :locals [] :casts [cast] :bodies [body]
              :pure? true :elem-type elem-type}
             io))

    (par/par-map-form? expression)
    (let [{:keys [out idx bound cast body elem-type offset]}
          (par/extract-par-map-info expression)
          ;; The map's element dtype is a fact before it is a policy: the form's own element
          ;; type, its store cast, then the destination's declared dtype (a lifted copy into a
          ;; `double-array` under a float policy stores doubles). The program dtype is the last
          ;; resort when nothing declares it.
          elem-type (dtype/canon (or elem-type
                                     (dtype/dtype-for-scalar-tag cast)
                                     (when (symbol? out) (get array-types out))
                                     default-dtype))
          write-index (when offset (list 'clojure.core/+ offset idx))
          io (extract-io (if offset (list 'do write-index body) body) idx [out])]
      ;; A binder with the same spelling as the caller-owned destination needs distinct value/view
      ;; identity before it can be SSA. Every other offset map is an injective partial write:
      ;; destination[base+i] is a typed unique scatter, not a map carrying an emitter-only offset.
      ;; The destination is read/write because elements outside the slice remain observable.
      (when-not (= symbol out)
        (if offset
          (merge {:kind :scatter :id id :sym symbol :results [symbol]
                  :index idx :extent bound :locals [] :casts [cast] :bodies [body]
                  :write-indices [write-index] :predicates [1] :conflict :unique
                  :result-storage [{:destination out :access :read-write
                                    :host-return :buffer}]
                  :host-binding symbol :elem-type elem-type
                  :source-operation :raster.par/map-offset}
                 io)
          (merge {:kind :map :id id :sym symbol :results [symbol]
                  :index idx :extent bound :locals [] :casts [cast] :bodies [body]
                  :result-storage [{:destination out
                                    :access (if (contains? (:inputs io) out) :read-write :write)
                                    :host-return :buffer}]
                  :host-binding symbol
                  :elem-type elem-type}
                 io))))

    (par/par-stencil-form? expression)
    (let [{:keys [out in-arrays radius boundary cast idx bound body elem-type]}
          (par/extract-par-stencil-info expression)
          io (extract-io body idx [out])
          stencil-inputs (set/union (set in-arrays) (:inputs io))
          result-dtype (or elem-type
                           (dtype/dtype-for-scalar-tag cast)
                           default-dtype)]
      (when (and (symbol? out)
                 (every? symbol? in-arrays)
                 (integer? radius) (pos? radius)
                 (= :dirichlet boundary)
                 (empty? (par/collect-aset-arrays body)))
        (merge {:kind :stencil :id id :sym symbol :results [symbol]
                :index idx :extent bound :radius radius :boundary boundary
                :casts [cast] :bodies [body] :inputs stencil-inputs
                :outputs #{out} :result-dtype (dtype/canon result-dtype)
                :result-storage [{:destination out :access :write
                                  :host-return :buffer}]
                :host-binding symbol}
               (dissoc io :inputs :outputs))))

    (par/par-map2-form? expression)
    (let [{:keys [out1 out2 idx bound cast body1 body2 elem-type]}
          (par/extract-par-map2-info expression)]
      (write-region-description id symbol idx bound
                              {:locals []
                               :stores [{:out out1 :value body1 :cast cast}
                                        {:out out2 :value body2 :cast cast}]}
                              elem-type :array-types array-types))

    (par/par-product-reduce-form? expression)
    (let [{:keys [outputs components segment-axes idx bound element-bindings element-results
                  combine-parameters combine-bindings combine-results algebra]}
          (par/extract-par-product-reduce-info expression)
          materialized (vec (keep-indexed (fn [ordinal output]
                                            (when (symbol? output) [ordinal output]))
                                          outputs))
          element-locals (typed-region-locals element-bindings)
          combine-locals (typed-region-locals combine-bindings)
          element-form (list 'let* element-bindings (vec element-results))
          combine-form (list 'let* combine-bindings (vec combine-results))
          component-records
          (mapv (fn [ordinal [accumulator neutral component-dtype] output]
                  {:id (keyword (str "component-" ordinal))
                   :accumulator accumulator :neutral neutral
                   :dtype (dtype/canon component-dtype) :result output})
                (range) components outputs)
          product (when (and (seq materialized)
                             (true? (:associative? algebra))
                             (some? element-locals) (some? combine-locals))
                    (reduction/make
                     {:components component-records :index idx
                      :element-bindings element-bindings :element-results element-results
                      :combine-parameters combine-parameters
                      :combine-bindings combine-bindings :combine-results combine-results
                      :algebra algebra
                      :attributes {:source :raster.par/product-reduce!}}))
          inputs (par/collect-aget-arrays element-form)
          binders (set (concat (map first segment-axes) [idx]
                               (map first components)
                               (take-nth 2 element-bindings)
                               (mapcat identity combine-parameters)
                               (take-nth 2 combine-bindings)))
          output-set (set (map second materialized))
          scalar-uses (set/union
                       (util/free-syms element-form)
                       (util/free-syms combine-form)
                       (reduce set/union #{}
                               (map (comp util/free-syms second) components))
                       (reduce set/union #{} (map (comp util/free-syms second) segment-axes))
                       (util/free-syms bound))
          scalars (set/difference scalar-uses inputs output-set binders
                                  descriptor/aget-ops descriptor/aset-ops
                                  #{'do 'let 'let* 'if 'double 'float 'int 'long})
          results (mapv #(effect-result-id id (first %)) materialized)
          storage (mapv (fn [[_ output]]
                          {:destination output :access :write :host-return :effect})
                        materialized)]
      (when product
        {:kind :product-reduce :id id :sym symbol
         :segment-axes segment-axes :reduce-index idx :reduce-extent bound
         :results results :result-components (mapv first materialized)
         :product product :inputs inputs :outputs output-set :scalars scalars
         :element-locals element-locals :combine-locals combine-locals
         :effect-only? true :host-binding symbol :result-storage storage}))

    (par/par-segmented-fold-map-form? expression)
    (let [{:keys [outputs segment-axes idx map-extent folds map-results elem-type]}
          (par/extract-par-segmented-fold-map-info expression)
          fold-dtype (fn [declared]
                       (dtype/canon (if (= :element declared)
                                      (or elem-type default-dtype :double)
                                      declared)))
          normalized-folds
          (mapv (fn [[accumulator identity declared-dtype extent step]]
                  {:accumulator accumulator :identity identity
                   :dtype (fold-dtype declared-dtype) :extent extent :step step})
                folds)
          all-expressions (vec (concat (map :step normalized-folds) map-results))
          inputs (reduce set/union #{} (map par/collect-aget-arrays all-expressions))
          binders (set (concat (map first segment-axes) [idx]
                               (map :accumulator normalized-folds)))
          output-set (set outputs)
          scalar-uses
          (reduce set/union #{}
                  (map util/free-syms
                       (concat (map second segment-axes) [map-extent]
                               (map :extent normalized-folds)
                               (map :identity normalized-folds)
                               all-expressions)))
          scalars (set/difference scalar-uses inputs output-set binders
                                  descriptor/aget-ops descriptor/aset-ops
                                  #{'do 'let 'let* 'if 'double 'float 'int 'long})
          results (mapv #(effect-result-id id %) (range (count outputs)))
          result-dtype (dtype/canon (or elem-type default-dtype :double))]
      (when (and (seq outputs) (= (count outputs) (count map-results))
                 (every? symbol? outputs) (seq segment-axes) (seq normalized-folds)
                 (every? #(and (dtype/known? (:dtype %))
                               (= (:dtype %) (dtype/canon (:dtype %))))
                         normalized-folds))
        {:kind :segmented-fold-map :id id :sym symbol
         :segment-axes segment-axes :index idx :extent map-extent
         :folds normalized-folds :map-results map-results
         :results results :result-dtypes (vec (repeat (count outputs) result-dtype))
         :inputs inputs :outputs output-set :scalars scalars
         :effect-only? true :host-binding symbol
         :result-storage (mapv (fn [output]
                                 {:destination output :access :write :host-return :effect})
                               outputs)}))

    (par/par-reduce-form? expression)
    (let [{:keys [acc init idx bound body elem-type]} (par/extract-par-reduce-info expression)
          io (extract-io body idx [symbol] :accumulator acc)
          identity-scalars
          (set/difference (util/free-syms init) (:inputs io)
                          (set [symbol acc idx]) descriptor/aget-ops descriptor/aset-ops)]
      (merge {:kind :reduce :id id :sym symbol :index idx :extent bound
              :product (reduction/scalar
                        ;; The walker stamps contextual FP narrowing on the par form.  Without
                        ;; that retained fact, Clojure's scalar reduction semantics are double;
                        ;; the target's preferred array dtype is not permission to narrow it.
                        {:accumulator acc :neutral init :dtype (or elem-type :double)
                         :result symbol :index idx :step-result body
                         :attributes {:source :raster.par/reduce}})}
             (update io :scalars set/union identity-scalars)))

    (and (seq? expression) (= 'raster.par/contract (first expression)))
    (let [facts (contraction-facts/contraction-facts
                 expression
                 ;; A contraction's element dtype is the declared dtype of the array it writes.
                 ;; The form's elem-type stamp is the compile's dtype policy, which a hard-typed
                 ;; double kernel compiled under a float policy does not follow; the program-wide
                 ;; dtype is only the last resort when nothing declares it.
                 :dtype (or (some-> (get array-types (second expression)) dtype/canon)
                            (:raster.type/elem-type (meta expression))
                            default-dtype :double))
          {:keys [free-axes contract-axes out opts]} facts
          contraction-dtype (dtype/canon (:dtype facts))
          ;; A result transform that reads the destination reads the storage the contraction
          ;; writes: that operand and the transform result have the contraction's dtype (a
          ;; double-spelled GEMM compiled under the float policy accumulates into a float
          ;; buffer), whatever tag the source call carried.
          epilogue (let [epilogue (:epilogue facts)]
                     (if (some #(= out (:sym %)) (:operands epilogue))
                       (-> epilogue
                           (assoc :dtype contraction-dtype)
                           (update :operands
                                   (fn [operands]
                                     (mapv #(cond-> % (= out (:sym %))
                                              (assoc :dtype contraction-dtype))
                                           operands))))
                       epilogue))
          result-transform (typed-result-transform epilogue)
          epilogue-arrays (set (map :value (:operands result-transform)))
          epilogue-scalar-ids (set (map :value (:scalars result-transform)))]
      ;; The direct slice is scalar segmented-reduction algebra plus an optional closed typed
      ;; result transform. Staged quantization carries additional schedule/load contracts and must
      ;; not be admitted by dropping those facts.
      (when (and (seq free-axes) (seq contract-axes)
                 ;; Decode lambdas, declared physical maps, staged accumulators and output
                 ;; conversions are not scalar fold syntax. Keep them on the certified
                 ;; contraction route until TypedSOAC represents those facts explicitly.
                 (empty? (apply dissoc opts [:init :combine :algebra :epilogue]))
                 ;; The contraction's element dtype is the dtype of its operands. A double
                 ;; product over arrays declared float (a hard-typed double function compiled
                 ;; under a float policy) has no single typed kernel; it stays a host call.
                 (every? (fn [{:keys [sym]}]
                           (let [declared (some-> (get array-types sym) dtype/canon)]
                             (or (nil? declared) (= declared (dtype/canon (:dtype facts))))))
                         (:operands facts))
                 (or (nil? epilogue) (dialect/result-transform? result-transform))
                 (reduction/scalar? (:reduction facts)))
        (let [product (:reduction facts)
              component (first (:components product))
              step-result (first (:results (reduction/fold-region product)))
              arrays (set/union (set (map :sym (:operands facts))) epilogue-arrays)
              axis-symbols (set (concat (map first free-axes) [(:index product)]))
              bound-symbols (reduce set/union #{}
                                    (map util/free-syms
                                         (concat (map second free-axes)
                                                 (map second contract-axes))))
              scalars (set/difference
                       (set/union bound-symbols (util/free-syms step-result))
                       arrays axis-symbols epilogue-scalar-ids #{(:accumulator component) out})]
          {:kind :segmented-reduce :id id :sym symbol
           :segment-axes free-axes
           :reduce-index (:index product)
           :reduce-extent (second (:flat-contract-axis facts))
           :results [(effect-result-id id 0)]
           :product product :inputs arrays :outputs #{out}
           :scalars (set/union scalars epilogue-scalar-ids)
           :result-transform result-transform
           :effect-only? true :host-binding symbol
           ;; Contract is effectful at the source spelling because it writes `out`, but its
           ;; TypedSOAC equation denotes the mathematical output tensor. A terminal reference to
           ;; that physical destination therefore returns the logical value; generic map-void
           ;; destinations remain `:effect` storage and retain host nil semantics. A result
           ;; transform that reads the destination (an accumulating GEMM) makes it read-write.
           :result-storage [{:destination out
                             :access (if (contains? epilogue-arrays out) :read-write :write)
                             :host-return :buffer}]})))

    (or (par/par-scan-form? expression)
        (par/par-scan-exclusive-form? expression))
    (let [mode (if (par/par-scan-exclusive-form? expression) :exclusive :inclusive)
          {:keys [out acc init idx bound cast body elem-type]}
          ((if (= :exclusive mode)
             par/extract-par-scan-exclusive-info
             par/extract-par-scan-info)
           expression)
          scan-dtype (or elem-type
                         (dtype/dtype-for-scalar-tag cast)
                         :double)]
      ;; A source binder with the same spelling as the caller-owned destination needs an explicit
      ;; value/view identity before it can be SSA. Keep that uncommon compatibility form outside
      ;; the typed route instead of pretending the destination is both an input and a definition.
      (when-not (= symbol out)
        (let [algebra (scan/certify {:acc acc :init init :lambda body :out out} scan-dtype)
              io (extract-io body idx [out] :accumulator acc)
              identity-scalars
              (set/difference (util/free-syms init) (:inputs io)
                              (set [symbol out acc idx])
                              descriptor/aget-ops descriptor/aset-ops)]
          (merge {:kind :scan :id id :sym symbol :results [symbol]
                  :index idx :extent bound :mode mode
                  :primary-out out :accumulator acc :identity init :dtype scan-dtype
                  :algebra algebra :body body
                  :result-storage [{:destination out :access :write
                                    :host-return :buffer}]
                  :host-binding symbol}
                 (update io :scalars set/union identity-scalars)))))

    (par/par-map-void-form? expression)
    (let [{:keys [idx bound body elem-type]}
          (par/extract-par-map-void-info expression)]
      (when-let [region (store-region body idx)]
        (write-region-description id symbol idx bound region
                                (dtype/canon (or elem-type default-dtype))
                                :host-return (or (::host-return (meta expression)) :effect)
                                :array-types array-types)))

    :else nil))

(defn- provably-pure-scalar?
  [expression]
  (or (effects/removable-expr? expression)
      (and (descriptor/alength-op? (descriptor/semantic-op expression))
           (= 1 (count (descriptor/call-args expression)))
           (symbol? (first (descriptor/call-args expression))))))

(defn- parallel-extent
  [expression]
  (cond
    (par/par-rng-fill-form? expression) (:n (par/extract-par-rng-fill-info expression))
    (par/par-map-pure-form? expression) (nth expression 2)
    (par/par-map-form? expression) (nth expression 3)
    (par/par-map2-form? expression) (nth expression 4)
    (par/par-reduce-form? expression) (nth expression 4)
    (or (par/par-scan-form? expression) (par/par-scan-exclusive-form? expression))
    (nth expression 5)
    (par/par-stencil-form? expression) (nth expression 7)
    (par/par-segmented-fold-map-form? expression) (nth expression 4)
    (par/par-map-void-form? expression) (nth expression 2)
    :else nil))

(defn- replace-parallel-extent
  [expression extent]
  (let [position (cond
                   (par/par-rng-fill-form? expression) 2
                   (par/par-map-pure-form? expression) 2
                   (par/par-map-form? expression) 3
                   (par/par-map2-form? expression) 4
                   (par/par-reduce-form? expression) 4
                   (or (par/par-scan-form? expression)
                       (par/par-scan-exclusive-form? expression)) 5
                   (par/par-stencil-form? expression) 7
                   (par/par-segmented-fold-map-form? expression) 4
                   (par/par-map-void-form? expression) 2)]
    (when position
      (with-meta (apply list (assoc (vec expression) position extent)) (meta expression)))))

(defn- canonicalize-strided-indexed-operation
  "Express a strided gather/scatter in the existing one-dimensional functional algebra.

   The flattened coordinate is split into row/component indices inside the scalar region. The
   ordinary compound-extent normalization below then gives n*stride one stable scalar SSA value.
   Gather becomes a dense map. Scatter becomes an additive effect region that generic store
   analysis certifies as a reducing scatter. Storage layout remains in index expressions instead
   of introducing operation-specific semantic nodes."
  [ordinal expression]
  (let [gather (when (par/par-gather-form? expression)
                 (par/extract-par-gather-info expression))
        scatter (when (par/par-scatter-form? expression)
                  (par/extract-par-scatter-info expression))
        {:keys [out src index n stride]} (or gather scatter)]
    (if stride
      (let [flat-index (clojure.core/symbol (str "rstr_indexed_flat_" ordinal))
            row-index (list 'clojure.core/quot flat-index stride)
            component (list 'clojure.core/rem flat-index stride)
            routed-index (list 'clojure.core/+
                               (list 'clojure.core/*
                                     (list 'clojure.core/aget index row-index)
                                     stride)
                               component)
            extent (list 'clojure.core/* n stride)]
        (with-meta
          (if gather
            (list 'raster.par/map! out flat-index extent nil
                  (list 'clojure.core/aget src routed-index))
            (list 'raster.par/map-void! flat-index extent
                  (list 'clojure.core/aset out routed-index
                        (list 'clojure.core/+
                              (list 'clojure.core/aget out routed-index)
                              (list 'clojure.core/aget src flat-index)))))
          (cond-> (meta expression) scatter (assoc ::host-return :buffer))))
      expression)))

(defn- canonicalize-blas-gemm
  "Express a devirtualized BLAS GEMM call as the explicit contraction it computes.

   `(dgemm! A B C m k n alpha beta)` is `C[m,n] = alpha·A·B + beta·C` over row-major operands;
   the `-tn!`/`-nt!` variants store `A` as `[k,m]` / `B` as `[n,k]`. With `beta = 0` the call
   is the pure contraction `(raster.par/contract C [[i m] [j n]] [[l k]] (* A[i,l] B[l,j]))`
   scaled by `alpha`, and the typed route can schedule it like any contraction instead of
   leaving the whole block on the host because one binding is an opaque BLAS effect. A non-zero
   `beta` (a literal or a scalar value) is the same contraction with a result transform that
   reads the destination element it overwrites: `C[i,j] := acc + beta·C[i,j]`. The destination
   is then read-write storage of one kernel, which the KernelBody builders express as a single
   `:inout` parameter. A `beta` that is neither a literal nor a scalar value id, or a call whose
   element type is unknown, stays a host call."
  [ordinal expression]
  (let [variant (when (and (seq? expression) (= '.invk (first expression)))
                  (get descriptor/blas-gemm-ops (:raster.op/original (meta expression))))
        arguments (when variant (vec (drop 2 expression)))
        [A B C m k n alpha beta] arguments
        beta-literal (when variant (descriptor/gemm-scalar-literal beta))
        alpha-literal (when variant (descriptor/gemm-scalar-literal alpha))
        ;; `alpha`/`beta` arrive as `(oftype witness value)` from the source spelling; the scalar
        ;; factor is its value, not the type witness array.
        scalar-value (fn [argument literal]
                       (cond
                         ;; A literal factor (`-1`, `(float 2.0)`) is the element-typed number
                         ;; it denotes, not an integer literal that would mistype the product.
                         (some? literal) literal
                         (and (seq? argument)
                              (= 'raster.numeric/oftype (descriptor/semantic-op argument))
                              (= 2 (count (descriptor/call-args argument))))
                         (second (descriptor/call-args argument))
                         :else argument))
        beta-value (when variant (scalar-value beta beta-literal))
        accumulate? (not= 0.0 beta-literal)
        elem-type (some-> (or (:raster.type/tag (meta expression)) (:tag (meta expression)))
                          dtype/dtype-for-array-tag dtype/canon)]
    (if (and variant (= 8 (count arguments))
             (symbol? A) (symbol? B) (symbol? C)
             (or (not accumulate?)
                 (and elem-type (or (number? beta-value) (symbol? beta-value)))))
      (let [i (clojure.core/symbol (str "rstr_gemm_i_" ordinal))
            j (clojure.core/symbol (str "rstr_gemm_j_" ordinal))
            l (clojure.core/symbol (str "rstr_gemm_l_" ordinal))
            [m k n] (map descriptor/unwrap-int-cast [m k n])
            a-index (case variant
                      (:nn :nt) (list 'clojure.core/+ (list 'clojure.core/* i k) l)
                      :tn (list 'clojure.core/+ (list 'clojure.core/* l m) i))
            b-index (case variant
                      (:nn :tn) (list 'clojure.core/+ (list 'clojure.core/* l n) j)
                      :nt (list 'clojure.core/+ (list 'clojure.core/* j k) l))
            ;; The walker stamps every arithmetic form with its result dtype; the emitted loads
            ;; and product carry the same stamp so scalar lowering reads the type instead of
            ;; guessing it.
            scalar-tag (some-> elem-type dtype/scalar-tag-for-dtype)
            typed (fn [form]
                    (if scalar-tag
                      (with-meta form {:raster.type/tag scalar-tag :tag scalar-tag})
                      form))
            product (typed (list 'clojure.core/*
                                 (typed (list 'clojure.core/aget A a-index))
                                 (typed (list 'clojure.core/aget B b-index))))
            alpha-value (scalar-value alpha alpha-literal)
            body (if (= 1.0 alpha-literal)
                   product
                   (typed (list 'clojure.core/* alpha-value product)))
            ;; `beta·C[i,j]` is read at the store coordinates of the element being produced;
            ;; the operand map declares that coordinate, the index inside the read is its
            ;; row-major spelling for the host expansion.
            epilogue
            (when accumulate?
              (let [acc (clojure.core/symbol (str "rstr_gemm_acc_" ordinal))
                    destination (typed (list 'clojure.core/aget C
                                             (list 'clojure.core/+ (list 'clojure.core/* i n) j)))
                    scaled (if (= 1.0 beta-literal)
                             destination
                             (typed (list 'clojure.core/* beta-value destination)))]
                {:acc acc
                 :expr (typed (list 'clojure.core/+ acc scaled))
                 :operands [{:sym C :map (axis-map/of-axes [[i m] [j n]]) :dtype elem-type}]
                 :scalars (if (symbol? beta-value) [{:sym beta-value :dtype elem-type}] [])
                 :dtype elem-type}))]
        (with-meta (cond-> (list 'raster.par/contract C [[i m] [j n]] [[l k]] body)
                     epilogue (concat [:epilogue epilogue])
                     true (->> (apply list)))
          (cond-> {:raster.op/original (:raster.op/original (meta expression))}
            elem-type (assoc :raster.type/elem-type elem-type))))
      expression)))

(defn- source-shadowing-locals
  "The symbols that are locals of the analyzed source even when they collide with a
   `clojure.core` name: the let's own binders and the declared parameters. `util/free-syms`
   would otherwise read a parameter named `seq` or `count` as the core function and drop it
   from every capture set, leaving a kernel that references an unbound scalar."
  [source & type-maps]
  (let [[_ bindings] (when (and (seq? source) (contains? #{'let 'let*} (first source)))
                       source)]
    (into (set (filter symbol? (take-nth 2 bindings)))
          (mapcat keys type-maps))))

(declare alength-array)

(def ^:dynamic ^:private *declared-kinds*
  "Declared value kinds visible to `normalize-source`: `{:arrays #{sym} :scalars #{sym}}`."
  {:arrays #{} :scalars #{}})

(defn- length-expression?
  "A syntactic array-length operand: a scalar value id, an integer, or integer arithmetic over
   them. A bare symbol counts only when it is positively a scalar (declared, or carrying an
   integral scalar tag): `(float-array x)` with an array `x` is a copy constructor, and a symbol
   of unknown kind is not a length either."
  [expression]
  (or (and (symbol? expression)
           (not (contains? (:arrays *declared-kinds*) expression))
           (or (contains? (:scalars *declared-kinds*) expression)
               (contains? #{:long :int}
                          (some-> (types/sym-type-tag expression) dtype/dtype-for-scalar-tag
                                  dtype/canon))))
      (integer? expression)
      (and (seq? expression)
           (contains? '#{* + - quot clojure.core/* clojure.core/+ clojure.core/- clojure.core/quot
                         long int clojure.core/long clojure.core/int}
                      (first expression))
           (every? length-expression? (rest expression)))))

(defn- allocation-length
  "The length expression an allocation binding declares, or nil.

   `(float-array n)` and `(zeros-like a n)` allocate `n` elements; `(zeros-like a)` allocates
   `(alength a)`. Recording this lets `(alength y)` over a local allocation resolve to the same
   extent id as every other use of `n`, instead of each `alength` minting a fresh compound
   extent that then contradicts the buffer's shape elsewhere in the program."
  [expression]
  (when (and (seq? expression) (descriptor/alloc-op? (descriptor/semantic-op expression)))
    (let [operation (descriptor/semantic-op expression)
          arguments (vec (descriptor/call-args expression))
          constructor? (or (contains? descriptor/alloc-sym->array-tag operation)
                           (contains? descriptor/alloc-sym->array-tag
                                      (some-> operation name clojure.core/symbol)))]
      (cond
        (and constructor? (= 1 (count arguments)) (length-expression? (first arguments)))
        (first arguments)

        (and (not constructor?) (= 2 (count arguments)) (symbol? (first arguments))
             (length-expression? (second arguments)))
        (second arguments)

        (and (not constructor?) (= 1 (count arguments)) (symbol? (first arguments)))
        (list 'clojure.core/alength (first arguments))

        :else nil))))

(defn- normalize-source*
  [source]
  ;; Direct backend entry may see source before the ordinary pipeline's SSA cleanup. Clojure
  ;; permits sequential rebinding (most commonly repeated `_` effect binders), while TypedSOAC
  ;; deliberately requires one logical definition per value. Use the shared scope-aware
  ;; alpha-renamer so later references keep their lexical meaning; inventing identities only in
  ;; operation-description would disconnect host materialization from the semantic equation.
  (let [source (util/uniquify-rebindings source)]
    (if (and (seq? source) (contains? #{'let 'let*} (first source)))
      (let [[head bindings & body] source
            pairs (vec (partition 2 bindings))
            {:keys [normalized]}
            (reduce
             (fn [{:keys [compound-extents allocation-lengths scalar-aliases pure-scalar-ids]
                   :as state}
                  [ordinal [symbol expression]]]
               (let [expression (->> expression
                                     (canonicalize-strided-indexed-operation ordinal)
                                     (canonicalize-blas-gemm ordinal))
                     ;; Host scalar identities: a binding that renames a scalar, or recomputes
                     ;; a pure scalar expression an earlier binding already computed, denotes
                     ;; that earlier value. Extents are compared in these canonical names so
                     ;; `(* seq dff)` and `(* n1 n2)` with `n1 = seq`, `n2 = dff` are one size.
                     canonical-scalar (fn [form]
                                        (util/subst-syms scalar-aliases (strip-index-casts form)))
                     ;; `(long x)` of a floating scalar is a conversion, not a renaming.
                     floating-cast? (and (seq? expression) (= 2 (count expression))
                                         (contains? '#{long int clojure.core/long clojure.core/int}
                                                    (first expression))
                                         (symbol? (second expression))
                                         (some-> (types/sym-type-tag (second expression))
                                                 dtype/dtype-for-scalar-tag dtype/fp-dtype?))
                     scalar-form (when-not (or (parallel-extent expression)
                                               (allocation-length expression)
                                               (par/par-form? expression)
                                               floating-cast?)
                                   (canonical-scalar expression))
                     ;; A recomputed scalar denotes an earlier one only when nothing between
                     ;; them can have changed its value: it reads no array (a kernel in between
                     ;; may write one), or it is an array length, which no kernel changes.
                     stable-scalar? (and (seq? scalar-form)
                                         (provably-pure-scalar? scalar-form)
                                         (or (empty? (par/collect-aget-arrays scalar-form))
                                             (some? (alength-array scalar-form))))
                     state (cond
                             (and (symbol? scalar-form) (not= scalar-form symbol))
                             (do
                               ;; the alias inherits its target's declared kind
                               (when (contains? (:scalars *declared-kinds*) scalar-form)
                                 (set! *declared-kinds*
                                       (update *declared-kinds* :scalars conj symbol)))
                               (when (contains? (:arrays *declared-kinds*) scalar-form)
                                 (set! *declared-kinds*
                                       (update *declared-kinds* :arrays conj symbol)))
                               (assoc-in state [:scalar-aliases symbol] scalar-form))

                             stable-scalar?
                             (if-let [earlier (get pure-scalar-ids scalar-form)]
                               (assoc-in state [:scalar-aliases symbol] earlier)
                               (assoc-in state [:pure-scalar-ids scalar-form] symbol))

                             :else state)
                     scalar-aliases (:scalar-aliases state)
                     extent (parallel-extent expression)
                     ;; `(alength y)` over a local allocation is the allocation's declared
                     ;; length; follow renamed allocations to their length as well.
                     resolve-length (fn resolve-length [extent seen]
                                      (let [array (alength-array extent)]
                                        (if (and array (contains? allocation-lengths array)
                                                 (not (contains? seen array)))
                                          (resolve-length (get allocation-lengths array)
                                                          (conj seen array))
                                          extent)))
                     canonical-extent (some-> extent
                                              (resolve-length #{})
                                              (->> (util/subst-syms scalar-aliases))
                                              descriptor/unwrap-int-cast)
                     state (if-let [length (allocation-length expression)]
                             (assoc-in state [:allocation-lengths symbol]
                                       (->> (resolve-length length #{})
                                            (util/subst-syms scalar-aliases)
                                            descriptor/unwrap-int-cast))
                             (if (and (symbol? expression)
                                      (contains? allocation-lengths expression))
                               (assoc-in state [:allocation-lengths symbol]
                                         (get allocation-lengths expression))
                               state))]
                 (cond
                   (nil? extent)
                   (update state :normalized conj [symbol expression])

                   ;; Integral casts are representation noise, not new semantic dimensions.
                   ;; Keeping their underlying value identity also prevents one array from
                   ;; acquiring incompatible [n] and [(long n)] shapes across operations.
                   (dialect/extent? canonical-extent)
                   (update state :normalized conj
                           [symbol (replace-parallel-extent expression canonical-extent)])

                   (provably-pure-scalar? canonical-extent)
                   (if-let [extent-id (get compound-extents canonical-extent)]
                     ;; Equal pure extent expressions are one SSA value. Besides avoiding
                     ;; redundant scalar work, this exposes equal launch geometry to the
                     ;; general horizontal-fusion rule (for example two same-shaped views).
                     (update state :normalized conj
                             [symbol (replace-parallel-extent expression extent-id)])
                     (let [extent-id (with-meta
                                       (clojure.core/symbol (str "rstr_extent_" ordinal))
                                       {:tag 'long :raster.type/tag 'long
                                        ;; This SSA value is introduced by the frontend as the
                                        ;; canonical identity of compound launch/storage algebra.
                                        ;; Preserve that provenance without relying on its name.
                                        :raster.compiler/normalized-extent true})]
                       (-> state
                           (assoc-in [:compound-extents canonical-extent] extent-id)
                           (update :normalized into
                                   [[extent-id canonical-extent]
                                    [symbol (replace-parallel-extent expression extent-id)]]))))

                   :else
                   (update state :normalized conj [symbol expression]))))
             {:normalized [] :compound-extents {} :allocation-lengths {}
              :scalar-aliases {} :pure-scalar-ids {}}
             (map-indexed vector pairs))]
        (with-meta (list* head (vec (mapcat identity normalized)) body) (meta source)))
      source)))

(defn- alength-array
  [expression]
  (when (and (seq? expression)
             (descriptor/alength-op? (descriptor/semantic-op expression))
             (= 1 (count (descriptor/call-args expression))))
    (first (descriptor/call-args expression))))

(defn- source-descriptions
  [pairs default-dtype array-types]
  ;; Earlier local allocations are authoritative array-type facts for later effects. Thread those
  ;; facts in source order instead of falling back to the program-wide arithmetic dtype: a local
  ;; float-array reduced by a strided scatter remains FP32 even in a mixed-precision program.
  (:descriptions
   (reduce
    (fn [{:keys [array-types scalar-definitions] :as state} [id [symbol expression]]]
      (let [description
            (or (binding [*scalar-definitions* scalar-definitions]
                  (operation-description id symbol expression default-dtype array-types))
                (if (par/par-form? expression)
                  {:kind :unsupported :id id :sym symbol :expr expression}
                  {:kind :scalar :id id :sym symbol :expr expression}))
            ;; a pure product/sum of scalars is a definition later index algebra may expand
            scalar-definition
            (when (and (= :scalar (:kind description)) (symbol? symbol)
                       (seq? expression)
                       (contains? '#{* + clojure.core/* clojure.core/+ quot clojure.core/quot}
                                  (descriptor/semantic-op expression))
                       (provably-pure-scalar? expression)
                       ;; a definition that reads an array is not an invariant extent
                       (empty? (par/collect-aget-arrays expression)))
              expression)
            allocation-dtype
            (when (and (= :scalar (:kind description))
                       (seq? expression)
                       (descriptor/alloc-op? (descriptor/semantic-op expression)))
              (let [operation (descriptor/semantic-op expression)
                    array-tag (or (get descriptor/alloc-sym->array-tag operation)
                                  (get descriptor/alloc-sym->array-tag
                                       (some-> operation name symbol)))]
                (some-> array-tag dtype/dtype-for-array-tag dtype/canon)))]
        (cond-> (update state :descriptions conj description)
          allocation-dtype (assoc-in [:array-types symbol] allocation-dtype)
          scalar-definition (assoc-in [:scalar-definitions symbol] scalar-definition))))
    {:descriptions [] :array-types array-types :scalar-definitions {}}
    (map-indexed vector pairs))))

(defn- canonical-extent
  [equalities values extent]
  (loop [extent extent seen #{}]
    (let [wrapped? (and (seq? extent) (= 'value (first extent)) (= 2 (count extent)))
          inner (if wrapped? (second extent) extent)
          normalized (descriptor/unwrap-int-cast inner)
          array (alength-array normalized)
          proved-shape (when array (:shape (get values array)))
          proved-extent (when (= 1 (count proved-shape)) (first proved-shape))
          next (or proved-extent
                   (get equalities normalized)
                   (get equalities inner)
                   (when-not (= inner normalized) normalized)
                   extent)]
      (cond
        (or (= next extent) (contains? seen next)) extent
        ;; `(value compound-id)` is an unambiguous dimension reference. Once relational facts
        ;; prove that compound value equal to a proper dimension, the wrapper has served its
        ;; purpose and must not survive as a distinct shape.
        (and wrapped? (= next inner)) extent
        :else (recur next (conj seen extent))))))

(defn- normalize-extents
  [descriptions shape-equalities values]
  (:descriptions
   (reduce
    (fn [{:keys [extents scalar-representatives] :as state} description]
      (case (:kind description)
        :scalar
        (let [expression (:expr description)
              array (alength-array expression)
              proved-extent (canonical-extent shape-equalities values expression)
              representative (cond
                               (and array (contains? extents array)) (get extents array)
                               (symbol? expression)
                               (get scalar-representatives expression expression)
                               (integer? expression) expression
                               (not= proved-extent expression) proved-extent
                               :else (:sym description))]
          (-> state
              (update :descriptions conj
                      (assoc description :expr (if (= representative (:sym description))
                                                 expression representative)))
              (assoc-in [:scalar-representatives (:sym description)] representative)))

        (:map :scatter :effect-map :stencil :reduce :scan)
        (let [extent (descriptor/unwrap-int-cast (:extent description))
              array (alength-array extent)
              proved-extent (canonical-extent shape-equalities values extent)
              extent' (cond
                        (and array (contains? extents array)) (get extents array)
                        (symbol? extent) (get scalar-representatives extent extent)
                        (not= proved-extent extent) proved-extent
                        :else extent)
              description' (assoc description :extent extent')]
          (cond-> (update state :descriptions conj description')
            (contains? #{:map :scatter :effect-map :stencil :scan} (:kind description'))
            (assoc-in [:extents (:sym description')] extent')))

        (update state :descriptions conj description)))
    {:descriptions [] :extents {} :scalar-representatives {}}
    descriptions)))

(defn- generated-scaffolding?
  [description physical-outputs]
  (and (= :scalar (:kind description))
       (let [expression (:expr description)]
         (or (and (symbol? expression) (contains? physical-outputs expression))
             (and (contains? physical-outputs (:sym description))
                  (seq? expression)
                  (descriptor/alloc-op? (descriptor/semantic-op expression)))))))

(defn- physical-output-symbols
  "Every buffer some operation writes, closed under host renamings: a binding that merely
   renames a buffer (`out y`) shares its physical identity, so a write through the alias is a
   write to the allocation it names and that allocation is generated scaffolding as well."
  [descriptions]
  (let [written (reduce set/union #{}
                        (map #(if (contains? #{:map :scatter :effect-map :stencil :reduce
                                               :segmented-reduce :product-reduce
                                               :segmented-fold-map :scan} (:kind %))
                                (:outputs %) #{})
                             descriptions))
        renamings (keep #(when (and (= :scalar (:kind %)) (symbol? (:expr %)))
                           [(:sym %) (:expr %)])
                        descriptions)]
    (loop [outputs written]
      (let [closed (into outputs (keep (fn [[alias source]]
                                         (when (contains? outputs alias) source))
                                       renamings))]
        (if (= closed outputs) outputs (recur closed))))))

(defn- contains-parallel-form?
  [expression]
  (boolean (some par/par-form? (tree-seq coll? seq expression))))

(defn- supported-description?
  [physical-outputs description]
  (case (:kind description)
    :scalar (and (not (contains-parallel-form? (:expr description)))
                 (or (provably-pure-scalar? (:expr description))
                     (generated-scaffolding? description physical-outputs)))
    :map (or (:pure? description)
             (and (seq (:result-storage description))
                  (every? (comp symbol? :destination) (:result-storage description))))
    :scatter (and (seq (:result-storage description))
                  (or (= :unique (:conflict description))
                      (dialect/reducing-scatter-conflict? (:conflict description)))
                  (every? (comp symbol? :destination) (:result-storage description)))
    :effect-map (and (seq (:effects description))
                     (seq (:result-storage description))
                     (= (count (:result-storage description))
                        (count (:result-dtypes description)))
                     (every? (comp symbol? :destination) (:result-storage description))
                     (every? (fn [{:keys [effect-conflict]}]
                               (or (contains? #{:unique :ordered} effect-conflict)
                                   (dialect/reducing-scatter-conflict? effect-conflict)))
                             (effect-leaves (:effects description))))
    :stencil (and (= 1 (count (:result-storage description)))
                  (symbol? (get-in description [:result-storage 0 :destination])))
    :reduce true
    :segmented-reduce (and (seq (:segment-axes description))
                           (symbol? (get-in description [:result-storage 0 :destination])))
    :product-reduce (and (seq (:segment-axes description))
                         (seq (:result-components description))
                         (every? (comp symbol? :destination) (:result-storage description)))
    :segmented-fold-map
    (and (seq (:segment-axes description)) (seq (:folds description))
         (every? (comp symbol? :destination) (:result-storage description)))
    :scan (and (= 1 (count (:result-storage description)))
               (= (:primary-out description)
                  (get-in description [:result-storage 0 :destination])))
    false))

(defn- supported-descriptions?
  [descriptions]
  (let [physical-outputs (physical-output-symbols descriptions)]
    ;; Ordinary host scalar bindings are opaque control/dataflow boundaries around TypedSOAC
    ;; islands. Unsupported parallel operations still decline: executing those through a second
    ;; lowering route inside one program would duplicate semantics.
    (every? #(or (= :scalar (:kind %))
                 (supported-description? physical-outputs %))
            descriptions)))

(defn normalize-source
  "Give pure compound parallel extents stable scalar SSA identities before dialect construction.

   TypedSOAC operations name extents; executable host expressions never leak into schedule fields.
   This normalization inserts an ordinary typed scalar binding immediately before its operation,
   so the same expression remains visible to JVM materialization and runtime specialization."
  ([source] (normalize-source source {}))
  ([source {:keys [array-types scalar-types]}]
   ;; The let's own binders declare kinds too: a `^long n` binder is a scalar, a `^floats y`
   ;; binder an array, exactly as the walker stamps them.
   (let [binders (when (and (seq? source) (contains? #{'let 'let*} (first source)))
                   (filter symbol? (take-nth 2 (second source))))
         tagged (fn [kind?]
                  (set (filter #(kind? (types/sym-type-tag %)) binders)))]
     (binding [util/*shadowing-locals* (source-shadowing-locals source)
               *declared-kinds*
               {:arrays (into (set (keys array-types))
                              (tagged #(some? (dtype/dtype-for-array-tag %))))
                :scalars (into (set (keys scalar-types))
                               (tagged #(contains? #{:long :int}
                                                   (some-> % dtype/dtype-for-scalar-tag
                                                           dtype/canon))))}]
       (normalize-source* source)))))

(declare coverage-decline*)

(defn coverage-decline
  "Describe the exact source bindings that prevent admission to the closed TypedSOAC subset.

   This is diagnostic evidence only: it uses the same descriptions and admission predicate as
   form->program, so reporting cannot become a second legality implementation."
  [source {:keys [array-types scalar-types values] :as options}]
  (binding [util/*shadowing-locals* (source-shadowing-locals source array-types scalar-types
                                                             values)]
    (coverage-decline* source options)))

(defn- coverage-decline*
  [source {:keys [dtype array-types values shape-equalities]
           :or {dtype :double array-types {} values {} shape-equalities {}}}]
  (when (and (seq? source) (contains? #{'let 'let*} (first source)))
    (let [[_ bindings] source
          pairs (vec (partition 2 bindings))
          array-types (binder-array-types pairs array-types dtype)
          descriptions (normalize-extents (source-descriptions pairs dtype array-types)
                                          shape-equalities values)
          physical-outputs (physical-output-symbols descriptions)
          declined (remove #(supported-description? physical-outputs %) descriptions)
          parallel? (some #(not= :scalar (:kind %)) descriptions)]
      (when (and parallel? (seq declined))
        {:reason :typed-soac-source-coverage
         :bindings
         (mapv (fn [{:keys [id sym kind expr]}]
                 {:id id
                  :binding sym
                  :kind kind
                  :operation (when (seq? expr) (descriptor/semantic-op expr))
                  :reason (case kind
                            :unsupported :unsupported-parallel-operation
                            :scalar :uncertified-host-scalar
                            :map :unverified-map-effects
                            :scatter :unverified-scatter-conflict
                            :effect-map :unverified-ordered-effects
                            :stencil :unverified-stencil-storage
                            :segmented-reduce :unverified-segmented-reduction
                            :product-reduce :unverified-product-reduction
                            :segmented-fold-map :unverified-segmented-fold-map
                            :scan :unverified-scan-storage
                            :unverified-operation-contract)})
               declined)}))))

(defn- element-symbols [n] (mapv #(symbol (str "%element" %)) (range n)))
(defn- capture-symbols [n] (mapv #(symbol (str "%capture" %)) (range n)))

(defn- pointwise-input?
  [expressions array index]
  (let [reads (filter (fn [{:keys [sym]}]
                        (and (symbol? sym) (= (name array) (name sym))))
                      (mapcat descriptor/aget-reads expressions))]
    (and (seq reads)
         (every? #(= index (descriptor/unwrap-int-cast (:idx %))) reads))))

(defn- elementize
  [expressions arrays parameters index]
  (mapv (fn [expression]
          (reduce (fn [body [array parameter]]
                    (fusion-support/substitute-aget body array index index parameter))
                  expression (map vector arrays parameters)))
        expressions))

(defn- canonicalize-scalar-folds
  [expression default-dtype]
  (util/postwalk-preserving-meta
   (fn [form]
     (if (par/par-reduce-form? form)
       (let [{:keys [acc init idx bound body elem-type]}
             (par/extract-par-reduce-info form)
             fold-dtype (dtype/canon (or elem-type default-dtype :double))
             algebra (when (and (symbol? acc) (symbol? idx))
                       (try
                         (scan/certify {:acc acc :init init :lambda body} fold-dtype)
                         (catch clojure.lang.ExceptionInfo _ nil)))]
         (if (and (symbol? acc) (symbol? idx) (dialect/scalar-literal? init))
           (util/remake
            form
            'fold
            (cond-> {:accumulator acc :index idx :identity init :dtype fold-dtype
                     :extent bound
                     :association (if algebra :implementation-defined :ordered)}
              algebra (assoc :algebra algebra))
            (dialect/lambda-form [acc idx] [body]))
           form))
       form))
   expression))

(defn- declare-result-conversion
  "Wrap a map body in the explicit cast to its result element dtype when the body's retained
   dtype differs.

   Both dtypes are facts: the result buffer's element dtype and the walker's stamp on the body.
   A double body written to a float buffer (a double-typed schedule compiled under the float
   policy) is a stated narrowing, not a silent one; KernelBody would otherwise refuse the store."
  [expression elem-type]
  (let [tag (when (instance? clojure.lang.IObj expression)
              (or (:raster.type/tag (meta expression)) (:tag (meta expression))))
        source (some-> tag dtype/dtype-for-scalar-tag dtype/canon)
        target (some-> elem-type dtype/canon)]
    (if (and source target (not= source target)
             (dtype/fp-dtype? source) (dtype/fp-dtype? target))
      (list (dtype/scalar-tag-for-dtype target) expression)
      expression)))

(defn- map-equation
  [description]
  (let [{:keys [id index extent locals casts bodies inputs results elem-type]} description
        fold-dtype (or elem-type (:result-dtype description) :double)
        expressions (mapv (fn [cast body]
                            (if cast
                              (list cast body)
                              (declare-result-conversion body fold-dtype)))
                          casts bodies)
        all-expressions (into (mapv :init locals) expressions)
        [pointwise stable] ((juxt filter remove) #(pointwise-input? all-expressions % index) inputs)
        arrays (vec (sort-by pr-str pointwise))
        captures (vec (sort-by pr-str (distinct (concat stable (:scalars description)))))
        parameters (element-symbols (count arrays))
        capture-parameters (capture-symbols (count captures))
        substitutions (zipmap captures capture-parameters)
        local-forms
        (->> locals
             (mapv (fn [{:keys [id dtype init]}]
                     (dialect/local-value
                      id dtype
                      (canonicalize-scalar-folds
                       (util/subst-syms
                        substitutions
                        (first (elementize [init] arrays parameters index)))
                       fold-dtype)))))
        body-results (mapv #(canonicalize-scalar-folds
                             (util/subst-syms (zipmap captures capture-parameters) %)
                             fold-dtype)
                           (elementize expressions arrays parameters index))]
    (list '= id results
          (list 'map {:index index :extent extent
                      :attributes {:stable-array-captures (vec (sort-by pr-str stable))}}
                arrays captures
                (dialect/lambda-form (vec (concat parameters capture-parameters))
                                     local-forms body-results)))))

(defn- scatter-equation
  [description]
  (let [{:keys [id index extent locals casts bodies inputs results result-storage
                write-indices predicates conflict]} description
        values (mapv (fn [cast body] (if cast (list cast body) body)) casts bodies)
        destinations (mapv :destination result-storage)
        semantic-inputs (into (set inputs) destinations)
        all-expressions (vec (concat (map :init locals) write-indices predicates values))
        [pointwise stable]
        ((juxt filter remove) #(pointwise-input? all-expressions % index) semantic-inputs)
        arrays (vec (sort-by pr-str pointwise))
        captures (vec (sort-by pr-str (distinct (concat stable (:scalars description)))))
        parameters (element-symbols (count arrays))
        capture-parameters (capture-symbols (count captures))
        substitutions (zipmap captures capture-parameters)
        transform (fn [expression]
                    (util/subst-syms
                     substitutions
                     (first (elementize [expression] arrays parameters index))))
        local-forms (mapv (fn [{:keys [id dtype init]}]
                            (dialect/local-value id dtype (transform init)))
                          locals)
        writes (mapv (fn [destination-index predicate value]
                       (list 'write (transform destination-index)
                             (transform predicate) (transform value)))
                     write-indices predicates values)]
    (list '= id results
          (list 'scatter {:index index :extent extent :conflict conflict
                          :attributes {:stable-array-captures
                                       (vec (sort-by pr-str stable))}}
                arrays captures
                (dialect/lambda-form (vec (concat parameters capture-parameters))
                                     local-forms writes)))))

(defn- effect-expressions
  "Every expression an effect evaluates, descending into store loops."
  [effect]
  (if-let [{:keys [extent locals effects]} (:loop effect)]
    (concat [extent] (map :init locals) (mapcat effect-expressions effects))
    [(:index effect) (:predicate effect) (:value effect)]))

(defn- effect-map-equation
  [{:keys [id index extent iteration-order locals inputs scalars results result-storage effects
           result-dtypes]}]
  (let [destinations (mapv :destination result-storage)
        destination-set (set destinations)
        all-expressions (vec (concat (map :init locals)
                                     (mapcat effect-expressions effects)))
        semantic-inputs (set/difference (set inputs) destination-set)
        [pointwise stable]
        ((juxt filter remove) #(pointwise-input? all-expressions % index) semantic-inputs)
        arrays (vec (sort-by pr-str pointwise))
        captures (vec (sort-by pr-str (distinct (concat stable scalars))))
        element-parameters (element-symbols (count arrays))
        capture-parameters (capture-symbols (count captures))
        destination-parameters (mapv #(symbol (str "%destination" %))
                                     (range (count destinations)))
        destination-substitutions (zipmap destinations destination-parameters)
        substitutions (into (zipmap captures capture-parameters)
                            destination-substitutions)
        transform (fn [expression]
                    (util/subst-syms
                     substitutions
                     (first (elementize [expression] arrays element-parameters index))))
        local-forms (mapv (fn [{:keys [id dtype init]}]
                            (dialect/local-value id dtype (transform init)))
                          locals)
        effect-form
        (fn effect-form [{:keys [out index predicate value cast effect-conflict] :as effect}]
          (if-let [{loop-index :index loop-locals :locals loop-effects :effects
                    :keys [lower extent]} (:loop effect)]
            (list 'effect-loop {:index loop-index :lower lower} (transform extent)
                  (list 'lambda [loop-index]
                        (dialect/effect-lambda-region
                         (mapv (fn [{:keys [id dtype init]}]
                                 (dialect/local-value id dtype (transform init)))
                               loop-locals)
                         (mapv effect-form loop-effects))))
            (list 'effect (get destination-substitutions out)
                  effect-conflict (transform index) (transform predicate)
                  (transform (if cast (list cast value) value)))))
        effect-forms (mapv effect-form effects)]
    (list '= id results
          (list 'effect-map
                {:index index :extent extent :dtypes result-dtypes
                 :iteration-order iteration-order
                 :attributes {:stable-array-captures (vec (sort-by pr-str stable))}}
                arrays captures destinations
                (dialect/effect-lambda-form
                 (vec (concat element-parameters capture-parameters destination-parameters))
                 local-forms effect-forms)))))

(defn- stencil-equation
  [{:keys [id index extent radius boundary inputs scalars results result-dtype casts bodies]}]
  (let [stable (vec (sort-by pr-str inputs))
        captures (vec (sort-by pr-str (distinct (concat stable scalars))))
        parameters (capture-symbols (count captures))
        substitutions (zipmap captures parameters)
        body (util/subst-syms substitutions (first bodies))
        cast (first casts)]
    (list '= id results
          (list 'stencil
                {:index index :extent extent :radius radius :boundary boundary
                 :dtypes [result-dtype]
                 :attributes {:stable-array-captures stable
                              :source-operation :raster.par/stencil!}}
                [] captures
                (dialect/lambda-form parameters [(if cast (list cast body) body)])))))

(defn- reduce-equation
  [{:keys [id extent inputs scalars product]}]
  (let [component (first (:components product))
        expressions (:results (reduction/fold-region product))
        index (:index product)
        [pointwise stable] ((juxt filter remove) #(pointwise-input? expressions % index) inputs)
        arrays (vec (sort-by pr-str pointwise))
        captures (vec (sort-by pr-str (distinct (concat stable scalars))))
        elements (element-symbols (count arrays))
        capture-parameters (capture-symbols (count captures))
        results (->> (elementize expressions arrays elements index)
                     (mapv #(util/subst-syms (zipmap captures capture-parameters) %)))
        algebra (scan/certify-reassociation
                 {:acc (:accumulator component)
                  :init (:neutral component)
                  :lambda (first results)}
                 (:dtype component))]
    (list '= id (vec (filter some? (reduction/results product)))
          (list 'reduce {:index index :extent extent
                         :attributes {:stable-array-captures (vec (sort-by pr-str stable))}
                         :accumulators [(:accumulator component)]
                         :identities [(:neutral component)]
                         :dtypes [(:dtype component)]
                         :algebra [algebra]}
                arrays captures
                (dialect/lambda-form
                 (vec (concat [(:accumulator component)] elements capture-parameters))
                 results)))))

(defn- segmented-reduce-equation
  [{:keys [id segment-axes reduce-index reduce-extent inputs scalars product results
           result-transform]}]
  (let [component (first (:components product))
        stable (vec (sort-by pr-str inputs))
        captures (vec (sort-by pr-str (distinct (concat stable scalars))))
        capture-parameters (capture-symbols (count captures))
        step-results (mapv #(util/subst-syms (zipmap captures capture-parameters) %)
                           (:results (reduction/fold-region product)))
        algebra (scan/certify-reassociation
                 {:acc (:accumulator component)
                  :init (:neutral component)
                  :lambda (first step-results)}
                 (:dtype component))]
    (list '= id results
          (list 'segmented-reduce
                {:segment-axes segment-axes
                 :index reduce-index :extent reduce-extent
                 :attributes {:stable-array-captures stable
                              :source-operation :raster.par/contract}
                 :accumulators [(:accumulator component)]
                 :identities [(:neutral component)]
                 :dtypes [(:dtype component)]
                 :algebra [algebra]
                 :result-transform result-transform}
                [] captures
                (dialect/lambda-form
                 (vec (concat [(:accumulator component)] capture-parameters))
                 step-results)))))

(defn- product-reduce-equation
  [{:keys [id segment-axes reduce-index reduce-extent inputs scalars product results
           result-components element-locals combine-locals]}]
  (let [captures (vec (sort-by pr-str (distinct (concat inputs scalars))))
        capture-parameters (capture-symbols (count captures))
        substitutions (zipmap captures capture-parameters)
        element-region (reduction/element-region product)
        combine-region (reduction/combine-region product)
        element-locals
        (mapv #(update % :init (fn [init] (util/subst-syms substitutions init)))
              element-locals)
        element-results (mapv #(util/subst-syms substitutions %)
                              (:results element-region))]
    (list '= id results
          (list 'product-reduce
                {:segment-axes segment-axes :index reduce-index :extent reduce-extent
                 :component-ids (mapv :id (:components product))
                 :accumulators (reduction/accumulators product)
                 :identities (reduction/neutrals product)
                 :dtypes (reduction/dtypes product)
                 :result-components result-components
                 :algebra (:algebra product)
                 :attributes {:stable-array-captures (vec (sort-by pr-str inputs))
                              :source-operation :raster.par/product-reduce!}}
                [] captures
                (dialect/lambda-form capture-parameters
                                     (dialect/emit-locals element-locals)
                                     element-results)
                (dialect/lambda-form
                 (vec (mapcat identity (:parameters combine-region)))
                 (dialect/emit-locals combine-locals)
                 (:results combine-region))))))

(defn- segmented-fold-map-equation
  [{:keys [id segment-axes index extent folds map-results inputs scalars results
           result-dtypes]}]
  (let [accumulator-set (set (map :accumulator folds))
        stable (vec (sort-by pr-str inputs))
        ;; Accumulator names are lexical across the complete ordered fold chain,
        ;; never implicit external captures. Removing every accumulator here makes
        ;; a reference to a later fold remain unbound and lets dialect validation
        ;; reject it instead of silently changing its meaning into a scalar input.
        captures (vec (sort-by pr-str
                               (remove accumulator-set
                                       (distinct (concat stable scalars)))))
        capture-parameters (capture-symbols (count captures))
        substitutions (zipmap captures capture-parameters)
        accumulators (mapv :accumulator folds)
        fold-forms
        (mapv (fn [ordinal {:keys [accumulator identity dtype extent step]}]
                (list 'fold
                      {:accumulator accumulator :identity identity :dtype dtype
                       :extent extent :association :ordered}
                      (dialect/lambda-form
                       (vec (concat [accumulator] (subvec accumulators 0 ordinal)
                                    capture-parameters))
                       [(util/subst-syms substitutions step)])))
              (range) folds)
        map-lambda
        (dialect/lambda-form
         (vec (concat accumulators capture-parameters))
         (mapv #(util/subst-syms substitutions %) map-results))]
    (list '= id results
          (list 'segmented-fold-map
                {:segment-axes segment-axes :index index :extent extent
                 :association :ordered :dtypes result-dtypes
                 :attributes {:stable-array-captures stable
                              :source-operation :raster.par/segmented-fold-map!}}
                [] captures fold-forms map-lambda))))

(defn- scan-equation
  [{:keys [id sym index extent mode inputs scalars primary-out accumulator identity dtype body]}]
  (let [[pointwise stable] ((juxt filter remove) #(pointwise-input? [body] % index) inputs)
        arrays (vec (sort-by pr-str pointwise))
        ;; The caller-owned destination is physical result storage, not a value read by the
        ;; functional scan. Keeping it out of captures prevents a false read/alias contract.
        stable (set stable)
        captures (vec (sort-by pr-str (distinct (concat stable scalars))))
        elements (element-symbols (count arrays))
        capture-parameters (capture-symbols (count captures))
        result (util/subst-syms
                (zipmap captures capture-parameters)
                (first (elementize [body] arrays elements index)))
        algebra (scan/certify {:acc accumulator :init identity :lambda result
                               :out primary-out} dtype)]
    (list '= id [sym]
          (list 'scan {:mode mode :index index :extent extent
                       :attributes {:stable-array-captures (vec (sort-by pr-str stable))}
                       :accumulators [accumulator]
                       :identities [identity]
                       :dtypes [dtype]
                       :algebra [algebra]}
                arrays captures
                (dialect/lambda-form
                 (vec (concat [accumulator] elements capture-parameters))
                 [result])))))

(defn- scalar-dtype
  [{:keys [sym expr]} scalar-dtypes scalar-types]
  (let [expression-tag (when (instance? clojure.lang.IObj expr)
                         (or (:raster.type/tag (meta expr)) (:tag (meta expr))))]
    (or (get scalar-types sym)
        (when (symbol? expr) (get scalar-dtypes expr))
        (dtype/dtype-for-scalar-tag (types/sym-type-tag sym))
        (dtype/dtype-for-scalar-tag expression-tag))))

(defn- scalar-equation
  [description scalar-dtypes scalar-types]
  (let [{:keys [id sym expr]} description
        captures (vec (sort-by pr-str (util/free-syms expr)))
        parameters (capture-symbols (count captures))
        result-dtype (scalar-dtype description scalar-dtypes scalar-types)]
    (when-not result-dtype
      (fail! :unsupported-scalar-binding
             "typed scalar equations require a retained result dtype"
             {:binding id :symbol sym :expression expr}))
    (list '= id [sym]
          (list 'scalar {:dtypes [result-dtype]} captures
                (dialect/lambda-form
                 parameters
                 [(util/subst-syms (zipmap captures parameters) expr)])))))

(defn- terminal-results
  [descriptions body]
  (let [physical-outputs (physical-output-symbols descriptions)
        operations (filter #(contains? #{:map :scatter :effect-map :stencil :reduce :segmented-reduce
                                         :product-reduce :segmented-fold-map :scan} (:kind %))
                           descriptions)
        operation-definitions (set (mapcat #(case (:kind %)
                                              (:map :scatter :effect-map :stencil) (:results %)
                                              :scan [(:sym %)]
                                              :segmented-reduce (:results %)
                                              :product-reduce (:results %)
                                              :segmented-fold-map (:results %)
                                              (:outputs %))
                                           operations))
        terminal-operation-definitions
        (set (mapcat #(case (:kind %)
                        (:map :scatter :effect-map :stencil)
                        (if (:effect-only? %) [] (:results %))
                        :scan [(:sym %)]
                        :segmented-reduce (if (:effect-only? %) [] (:results %))
                        :product-reduce (if (:effect-only? %) [] (:results %))
                        :segmented-fold-map (if (:effect-only? %) [] (:results %))
                        (:outputs %))
                     operations))
        typed-scalars (filter #(and (= :scalar (:kind %))
                                    (supported-description? physical-outputs %))
                              descriptions)
        host-scalars (filter #(and (= :scalar (:kind %))
                                   (not (supported-description? physical-outputs %)))
                             descriptions)
        scalar-definitions
        (set (keep #(when (and (= :scalar (:kind %))
                               (supported-description? physical-outputs %)
                               (not (generated-scaffolding? % physical-outputs)))
                      (:sym %))
                   descriptions))
        all-definitions (set/union operation-definitions scalar-definitions)
        operation-uses (set (concat (mapcat #(concat (:inputs %) (:scalars %)) operations)
                                    (mapcat #(util/free-syms (:expr %)) typed-scalars)))
        host-uses (set (mapcat #(util/free-syms (:expr %)) host-scalars))
        body-uses (set (mapcat util/free-syms body))
        ;; Destination-writing source forms return the destination buffer, while TypedSOAC names
        ;; the fresh logical result produced by that write. Preserve the public return by
        ;; projecting a returned physical destination to its corresponding logical SSA result.
        ;; This is the same result-storage relation later consumed by scheduling and linking; no
        ;; operation-specific knowledge is introduced here.
        destination-results
        (into {}
              (mapcat (fn [description]
                        (keep (fn [[result {:keys [destination host-return]}]]
                                ;; Effect-only operations expose their destinations through the
                                ;; reconstructed host form, not as numerical TypedSOAC results.
                                ;; Only value-returning storage contracts may rewrite a terminal
                                ;; physical destination to its logical SSA result.
                                (when (= :buffer host-return)
                                  [destination result]))
                              (map vector (:results description)
                                   (:result-storage description)))))
              descriptions)
        returned-destination-results
        (into #{} (keep destination-results) body-uses)]
    (vec (sort-by pr-str
                  (set/union (set/difference terminal-operation-definitions operation-uses)
                             (set/intersection operation-definitions host-uses)
                             (set/intersection all-definitions body-uses)
                             returned-destination-results)))))

(defn- scalar-dependency-closure
  [by-symbol roots]
  (loop [needed (set/intersection (set roots) (set (keys by-symbol)))]
    (let [dependencies (set (mapcat #(util/free-syms (:expr (get by-symbol %))) needed))
          needed' (set/union needed (set/intersection dependencies (set (keys by-symbol))))]
      (if (= needed needed') needed (recur needed')))))

(defn- allocation-capacity-scalars
  [descriptions physical-outputs by-symbol]
  ;; An allocation binding is generated storage scaffolding, but the typed scalar algebra which
  ;; computes its capacity is executable proof data.  Keep the transitive scalar definitions and
  ;; mark them below; no generated-name convention is involved.
  (scalar-dependency-closure
   by-symbol
   (mapcat (comp util/free-syms :expr)
           (filter #(generated-scaffolding? % physical-outputs) descriptions))))

(defn- selected-scalars
  [descriptions operation-equations outputs]
  (let [physical-outputs (physical-output-symbols descriptions)
        by-symbol (into {}
                        (keep #(when (and (= :scalar (:kind %))
                                          (supported-description? physical-outputs %)
                                          (not (generated-scaffolding? % physical-outputs)))
                                 [(:sym %) %]))
                        descriptions)
        ;; Allocation bindings themselves are generated storage scaffolding and must not become
        ;; executable scalar steps.  Their capacity algebra is different: it is the authoritative
        ;; definition of an AbstractValue shape that a later KernelGraph allocates.  Retain just
        ;; the scalar dependencies of that algebra, so a normalised `rstr_extent_n` (or a user
        ;; scalar size) remains an ordered, typed host equation rather than an opaque shape name.
        ;; The ordinary dependency closure below keeps this independent of allocation spelling.
        allocation-capacity-roots
        (allocation-capacity-scalars descriptions physical-outputs by-symbol)
        roots (set (concat outputs
                           allocation-capacity-roots
                           (mapcat (fn [equation]
                                     (into (dialect/operation-inputs equation)
                                           (filter dialect/value-id?
                                                   (dialect/operation-extents equation))))
                                   operation-equations)))]
    (scalar-dependency-closure by-symbol roots)))

(defn- tensor-value [dtype shape]
  (av/tensor {:dtype dtype :shape shape :representation {:kind :plain}}))

(defn- value-dtype [id default-dtype array-types]
  (or (get array-types id)
      (when (symbol? id) (get array-types (symbol (name id))))
      default-dtype :double))

(defn- declared-type
  [types id]
  (or (get types id)
      (when (symbol? id) (get types (clojure.core/symbol (name id))))))

(defn- equation-values
  [equation default-dtype array-types scalar-types known-values]
  (let [[_ _ results] equation
        {:keys [kind attributes arrays captures]} (dialect/operation-parts equation)
        extent (:extent attributes)
        dimension-ids (set (filter dialect/value-id? (dialect/operation-extents equation)))
        dimension-value
        (fn [id]
          ;; A shape dimension is integral, but its representation is an ABI fact:
          ;; GPU launch/scalar parameters are commonly :int while JVM-derived extents
          ;; may be :long.  Preserve the already-known or declared scalar contract and
          ;; use :long only when no upstream type information exists.
          (or (get known-values id)
              (when-let [declared (declared-type scalar-types id)]
                (tensor-value declared []))
              (when-let [declared (and (symbol? id)
                                       (dtype/dtype-for-scalar-tag
                                        (types/sym-type-tag id)))]
                (tensor-value declared []))
              (tensor-value :long [])))
        stable (set (get-in attributes [:attributes :stable-array-captures]))
        result-dtypes (case kind
                        scalar (:dtypes attributes)
                        stencil (:dtypes attributes)
                        reduce (:dtypes attributes)
                        segmented-reduce (:dtypes attributes)
                        product-reduce (mapv #((:dtypes attributes) %)
                                             (:result-components attributes))
                        segmented-fold-map (:dtypes attributes)
                        scan (:dtypes attributes)
                        effect-map (:dtypes attributes)
                        (map scatter) (map #(value-dtype % default-dtype array-types) results))]
    (merge
     (into {} (map (fn [id] [id (dimension-value id)])) dimension-ids)
     (into {} (map (fn [id] [id (tensor-value (value-dtype id default-dtype array-types)
                                              (dialect/extent-shape extent))]) arrays))
     (if (= 'scalar kind)
       (into {} (keep (fn [id]
                        (when-let [value (or (get known-values id)
                                             (when-let [declared (declared-type scalar-types id)]
                                               (tensor-value declared []))
                                             (when-let [declared (and (symbol? id)
                                                                      (dtype/dtype-for-scalar-tag
                                                                       (types/sym-type-tag id)))]
                                               (tensor-value declared [])))]
                          [id value]))) captures)
       (into {} (map (fn [id]
                       [id (or (when (contains? dimension-ids id)
                                 (dimension-value id))
                               (get known-values id)
                               (if (or (contains? stable id)
                                       (contains? array-types id)
                                       (and (symbol? id)
                                            (contains? array-types (symbol (name id)))))
                                 (tensor-value (value-dtype id default-dtype array-types)
                                               [(list 'unknown-dimension id)])
                                 (tensor-value (or (declared-type scalar-types id)
                                                   (value-dtype id default-dtype array-types))
                                               [])))])
                     captures)))
     (into {} (map (fn [id result-dtype]
                     [id (tensor-value (or result-dtype default-dtype :double)
                                       (case kind
                                         (scalar reduce) []
                                         segmented-reduce
                                         (dialect/segmented-reduce-result-shape attributes)
                                         product-reduce
                                         (dialect/segmented-reduce-result-shape attributes)
                                         segmented-fold-map
                                         (dialect/segmented-fold-map-result-shape attributes)
                                         scan (dialect/scan-result-shape attributes)
                                         (scatter effect-map) [(list 'unknown-dimension id)]
                                         (dialect/extent-shape extent)))])
                   results result-dtypes)))))

(defn- merge-value
  ([values id contract] (merge-value values id contract {}))
  ([values id contract shape-equalities]
   (if-let [prior (get values id)]
     (let [unknown-shape? (fn [value]
                            (= [(list 'unknown-dimension id)] (:shape value)))
           same-nonshape-contract? (= (dissoc prior :shape) (dissoc contract :shape))
           equivalent-shape?
           (= (mapv #(canonical-extent shape-equalities values %) (:shape prior))
              (mapv #(canonical-extent shape-equalities values %) (:shape contract)))]
       (cond
         (= prior contract) values
         (and same-nonshape-contract? equivalent-shape?) (assoc values id contract)
         (and same-nonshape-contract? (unknown-shape? prior)) (assoc values id contract)
         (and same-nonshape-contract? (unknown-shape? contract)) values
         :else
         (fail! :source-value-conflict
                "source bindings imply incompatible AbstractValues for one logical value"
                {:id id :first prior :second contract})))
     (assoc values id contract))))

(declare form->program*)

(defn form->program
  "Construct and validate TypedSOAC islands directly from a let form.

   Opaque scalar host bindings are retained as ordered barriers around the functional equations.
   Returns nil when a parallel binding is outside the certified source subset. Type/effect/value
   contradictions throw ExceptionInfo because falling through after accepting them would hide a
   compiler correctness defect."
  [source {:keys [array-types scalar-types values] :as options}]
  (binding [util/*shadowing-locals* (source-shadowing-locals source array-types scalar-types
                                                             values)]
    (form->program* source options)))

(defn- form->program*
  [source {:keys [dtype array-types scalar-types values shape-equalities]
           :or {dtype :double array-types {} scalar-types {} values {} shape-equalities {}}}]
  (when (and (seq? source) (contains? #{'let 'let*} (first source)))
    (let [[_ bindings & body] source
          pairs (vec (partition 2 bindings))
          array-types (binder-array-types pairs array-types dtype)
          descriptions (normalize-extents (source-descriptions pairs dtype array-types)
                                          shape-equalities values)]
      (when (and (even? (count bindings))
                 (seq descriptions)
                 (some #(contains? #{:map :scatter :effect-map :stencil :reduce :segmented-reduce
                                     :product-reduce :segmented-fold-map :scan}
                                   (:kind %))
                       descriptions)
                 (supported-descriptions? descriptions))
        (let [operation-descriptions
              (filterv #(contains? #{:map :scatter :effect-map :stencil :reduce :segmented-reduce
                                     :product-reduce :segmented-fold-map :scan} (:kind %))
                       descriptions)
              physical-outputs (physical-output-symbols descriptions)
              operation-equations (mapv #(case (:kind %) :map (map-equation %)
                                               :scatter (scatter-equation %)
                                               :effect-map (effect-map-equation %)
                                               :stencil (stencil-equation %)
                                               :reduce (reduce-equation %)
                                               :segmented-reduce (segmented-reduce-equation %)
                                               :product-reduce (product-reduce-equation %)
                                               :segmented-fold-map (segmented-fold-map-equation %)
                                               :scan (scan-equation %))
                                        operation-descriptions)
              outputs (terminal-results descriptions body)
              required-scalars (selected-scalars descriptions operation-equations outputs)
              scalar-descriptions
              (into {}
                    (keep #(when (and (= :scalar (:kind %))
                                      (supported-description? physical-outputs %)
                                      (not (generated-scaffolding? % physical-outputs)))
                             [(:sym %) %]))
                    descriptions)
              allocation-capacity-scalar-ids
              (allocation-capacity-scalars descriptions physical-outputs scalar-descriptions)
              normalized-extent-scalar-ids
              (into #{} (keep #(when (and (= :scalar (:kind %))
                                          (true? (:raster.compiler/normalized-extent
                                                  (meta (:sym %)))))
                                 (:sym %)))
                    descriptions)
              graph-shape-scalar-ids
              (set/union allocation-capacity-scalar-ids normalized-extent-scalar-ids)
              {:keys [equations equation-descriptions]}
              (reduce (fn [{:keys [scalar-dtypes] :as state} description]
                        (case (:kind description)
                          :scalar
                          (if (contains? required-scalars (:sym description))
                            (let [equation (scalar-equation description scalar-dtypes scalar-types)
                                  result-dtype (first (:dtypes (second (nth equation 3))))]
                              (-> state
                                  (update :equations conj equation)
                                  (update :equation-descriptions conj description)
                                  (assoc-in [:scalar-dtypes (:sym description)] result-dtype)))
                            state)
                          (:map :scatter :effect-map :stencil :reduce :segmented-reduce
                                :product-reduce :segmented-fold-map :scan)
                          (-> state
                              (update :equations conj
                                      (case (:kind description)
                                        :map (map-equation description)
                                        :scatter (scatter-equation description)
                                        :effect-map (effect-map-equation description)
                                        :stencil (stencil-equation description)
                                        :reduce (reduce-equation description)
                                        :segmented-reduce (segmented-reduce-equation description)
                                        :product-reduce (product-reduce-equation description)
                                        :segmented-fold-map
                                        (segmented-fold-map-equation description)
                                        :scan (scan-equation description)))
                              (update :equation-descriptions conj description))))
                      {:equations [] :equation-descriptions [] :scalar-dtypes {}}
                      descriptions)
              equation-info (mapv (fn [equation]
                                    {:results (nth equation 2)
                                     :inputs (dialect/operation-inputs equation)
                                     :extents (dialect/operation-extents equation)})
                                  equations)
              definitions (set (mapcat :results equation-info))
              references (set (mapcat #(into (:inputs %)
                                             (filter dialect/value-id? (:extents %)))
                                      equation-info))
              inputs (vec (sort-by pr-str (set/difference references definitions)))
              logical-result-types
              (into {}
                    (mapcat (fn [description]
                              (map (fn [result storage]
                                     [result (value-dtype (:destination storage)
                                                          dtype array-types)])
                                   (:results description)
                                   (:result-storage description))))
                    (filter :result-storage equation-descriptions))
              array-types' (merge array-types logical-result-types)
              destination-values
              (into {}
                    (mapcat (fn [description]
                              (map (fn [{:keys [destination]}]
                                     [destination
                                      (tensor-value
                                       (value-dtype destination dtype array-types)
                                       [(list 'unknown-dimension destination)])])
                                   (:result-storage description))))
                    equation-descriptions)
              inferred-values (reduce (fn [contracts equation]
                                        (reduce-kv #(merge-value %1 %2 %3 shape-equalities) contracts
                                                   (equation-values equation dtype array-types'
                                                                    scalar-types
                                                                    contracts)))
                                      destination-values equations)
              values (reduce-kv #(merge-value %1 %2 %3 shape-equalities)
                                inferred-values values)
              equation-facts
              (into {}
                    (map (fn [description]
                           (let [storage (:result-storage description)]
                             [(:id description)
                              (cond-> (dialect/default-equation-facts
                                       {:front-end :analyzed-source
                                        :source-binding-id (:id description)})
                                (contains? graph-shape-scalar-ids (:sym description))
                                (update :attributes assoc :graph-shape-definition true)
                                storage
                                (-> (assoc :effects #{:memory/write}
                                           :aliases (into {}
                                                          (map (fn [result {:keys [destination]}]
                                                                 [result destination])
                                                               (:results description) storage)))
                                    (update :attributes merge
                                            {:result-storage storage
                                             :host-binding (:host-binding description)})))]))
                         equation-descriptions))
              total-effects (reduce set/union #{} (map :effects (vals equation-facts)))
              host-descriptions
              (filterv #(and (= :scalar (:kind %))
                             (not (supported-description? physical-outputs %)))
                       descriptions)
              host-binding-ids (mapv :id host-descriptions)
              ;; Program values a host-controlled binding reads are uses of those values: a
              ;; producer read by the host must stay materialized, whatever fusion does with
              ;; its typed consumers.
              host-read-values
              (->> host-descriptions
                   (mapcat #(util/free-syms (:expr %)))
                   (filter #(contains? values %))
                   distinct
                   vec)
              facts (dialect/default-program-facts
                     {:values values :inputs inputs :equations equation-facts
                      :effects total-effects
                      :provenance {:front-end :analyzed-source}
                      :attributes {:source-dialect :closed-clojure
                                   :host-binding-ids host-binding-ids
                                   :host-read-values host-read-values}})]
          (dialect/make facts equations outputs))))))
