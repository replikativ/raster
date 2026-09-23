(ns raster.compiler.passes.parallel.map-read-requirements
  "Minimum flat storage derived from actual typed map loads, not from output size."
  (:require [clojure.set :as set]
            [clojure.walk :as walk]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.extent-expression :as extent]
            [raster.compiler.ir.index-algebra :as algebra]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.scalar-range :as ranges]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.patterns :as patterns]
            [raster.compiler.passes.parallel.segmap-body :as map-body]))

(defn- span-expression
  [{:keys [const factors]}]
  (let [operands (cond-> (vec factors) (not= 1 const) (conj const))]
    (case (count operands)
      0 1
      1 (first operands)
      (apply launch/product operands))))

(declare address-substitutions)
(defn- local-dependencies
  [locals expression]
  (:dependencies
   (reduce (fn [{:keys [live dependencies]} {:keys [id init]}]
             (if (contains? live id)
               {:live (into (disj live id) (util/free-syms init))
                :dependencies (conj dependencies id)}
               {:live live :dependencies dependencies}))
           {:live (util/free-syms expression) :dependencies #{}}
           (reverse locals))))

(defn- strip-index-cast
  [expression]
  (if (and (seq? expression) (= 2 (count expression))
           (contains? '#{long int clojure.core/long clojure.core/int} (first expression)))
    (recur (second expression))
    expression))

(defn- conservative-loop-extent
  [index extent locals bound inclusive?]
  (let [bound (strip-index-cast bound)]
    (or (when (symbol? bound) (algebra/digit-radix index extent locals bound))
        (if inclusive? (list 'clojure.core/+ bound 1) bound))))

(defn- lexical-read-sites
  "Collect array reads with the exact lexical locals and counted-loop domains at each site."
  [expression index extent initial-locals]
  (letfn [(collect-sequential [bindings body locals loops]
            (loop [pairs (seq (partition 2 bindings)) locals locals result []]
              (if-let [[id init] (first pairs)]
                (recur (next pairs) (conj locals {:id id :init init})
                       (into result (collect init locals loops)))
                (into result (mapcat #(collect % locals loops) body)))))
          (collect-fold [expression locals loops]
            (let [{:keys [attributes lambda]} (dialect/scalar-fold-parts expression)
                  {parameters :parameters fold-locals :locals body :body-results}
                  (dialect/lambda-parts lambda)
                  [_acc loop-index] parameters
                  lower (:lower attributes 0)
                  inclusive? (= :inclusive (:upper-bound attributes))
                  loop-extent (when (= 0 lower)
                                (conservative-loop-extent
                                 index extent locals (:extent attributes) inclusive?))
                  loops (cond-> loops loop-extent (assoc loop-index loop-extent))]
              (into (collect (:identity attributes) locals loops)
                    (loop [remaining fold-locals locals locals result []]
                      (if-let [local (first remaining)]
                        (recur (next remaining) (conj locals local)
                               (into result (collect (:init local) locals loops)))
                        (into result (mapcat #(collect % locals loops) body)))))))
          (collect [form locals loops]
            (cond
              (descriptor/aget-call? form)
              (into [{:read {:sym (descriptor/aget-array-sym form)
                             :idx (descriptor/aget-index form) :form form}
                      :locals locals :loop-indices loops}]
                    (collect (descriptor/aget-index form) locals loops))

              (and (seq? form) (contains? #{'let 'let* 'clojure.core/let} (first form))
                   (vector? (second form)) (even? (count (second form))))
              (collect-sequential (second form) (drop 2 form) locals loops)

              (dialect/scalar-fold-form? form)
              (collect-fold form locals loops)

              (and (seq? form) (contains? #{'loop 'loop*} (first form)))
              (if-let [{:keys [index-sym index-init bound-expr bound-mode acc-init
                               scoped-update-expr else-expr]}
                       (patterns/match-ordered-reduce-loop form)]
                (let [loop-extent (when (= 0 index-init)
                                    (conservative-loop-extent
                                     index extent locals bound-expr (= :inclusive bound-mode)))
                      loops (cond-> loops loop-extent (assoc index-sym loop-extent))]
                  (into (collect acc-init locals loops)
                        (concat (collect scoped-update-expr locals loops)
                                (collect else-expr locals loops))))
                [])

              (seq? form) (mapcat #(collect % locals loops) (rest form))
              (vector? form) (mapcat #(collect % locals loops) form)
              (map? form) (mapcat #(collect % locals loops) (apply concat form))
              :else []))]
    (vec (collect expression initial-locals {}))))
(defn symbolic-read-certificate
  "Certify exact symbolic spans for the flat reads of a typed one-dimensional map.

   The proof is structural: the map index is decomposed into mixed-radix digits and each address
   must enumerate a zero-based dense interval over the digits it represents. Omitted digits are
   permitted because a broadcast may repeat an input interval; padding, translation, indirect
   reads, and undecidable arithmetic decline. `scalar-definitions` contains already checked
   KernelLaunch definitions from the host prefix. Only monomial definitions are substituted, so
   quotient relations are never invented here."
  [operation {:keys [scalar-definitions] :or {scalar-definitions {}}}]
  (when (and (segop/seg-map? operation)
             (= 1 (count (get-in operation [:space :dims])))
             (empty? (set/intersection (set (:inputs operation)) (set (:outputs operation))))
             (not (seq (get-in operation [:scalar-region :effects]))))
    (let [{index :name bound :bound} (first (get-in operation [:space :dims]))
          {source-locals :locals result :result} (:scalar-region operation)
          monomial-definitions (into {} (filter (comp algebra/monomial val)) scalar-definitions)
          expand #(walk/postwalk-replace monomial-definitions %)
          bound (expand bound)
          locals (mapv #(update % :init (comp algebra/canonical-arithmetic expand)) source-locals)
          expressions (concat (map :init source-locals) [result])
          source-reads (->> expressions
                            (mapcat descriptor/aget-reads)
                            (filter #(contains? (:inputs operation) (:sym %)))
                            vec)
          read-sites (->> expressions
                          (mapcat #(lexical-read-sites % index bound source-locals))
                          (filter #(contains? (:inputs operation) (get-in % [:read :sym])))
                          vec)
          read-facts
          (mapv (fn [{{:keys [sym idx]} :read site-locals :locals
                      loop-indices :loop-indices}]
                  (let [coordinate (algebra/canonical-arithmetic (expand idx))
                        projected-coordinate
                        (algebra/canonical-arithmetic
                         (util/subst-syms (address-substitutions site-locals) idx))
                        form (algebra/index-form
                              coordinate
                              index bound
                              (mapv #(update % :init (comp algebra/canonical-arithmetic expand))
                                    site-locals)
                              loop-indices)]
                    (when-let [span (algebra/zero-based-dense-span form)]
                       {:buffer sym :source-coordinate idx
                       :coordinate coordinate
                       :projected-coordinate projected-coordinate
                       :address-locals (local-dependencies site-locals idx)
                       :loop-indices loop-indices :form form
                       :span (span-expression span)})))
                read-sites)]
      (when (and (seq source-reads)
                 (= (count source-reads) (count read-sites))
                 (every? some? read-facts))
        {:kind :zero-based-dense-read-spans
         :index index :extent bound
         ;; Retain both sides of the proof. `:source-locals` and
         ;; `:scalar-definitions` let a later graph-bound target projection recompute this
         ;; certificate from the exact SegMap instead of trusting attached metadata. `:locals`
         ;; is the canonical expanded region used by the mixed-radix proof.
         :source-locals source-locals
         :scalar-definitions monomial-definitions
         :locals locals :reads read-facts
         :requirements
         (reduce (fn [result {:keys [buffer span]}]
                  (update result buffer
                          (fn [prior]
                            (cond
                              (nil? prior) span
                              (= prior span) prior
                              :else (launch/maximum prior span)))))
                 {} read-facts)}))))

(defn symbolic-read-requirements
  "Return only the buffer-capacity projection of `symbolic-read-certificate`."
  [operation options]
  (:requirements (symbolic-read-certificate operation options)))

(defn- capacity-covers?
  [capacity required]
  (or (extent/equivalent? capacity required)
      (and (integer? capacity) (integer? required) (<= required capacity))
      (and (= "raster.compiler.ir.kernel_launch.Maximum" (some-> capacity class .getName))
           (some #(capacity-covers? % required) (:values capacity)))))

(defn- address-substitutions
  [locals]
  (reduce (fn [substitutions {:keys [id init]}]
            (assoc substitutions id (util/subst-syms substitutions init)))
          {} locals))

(defn- retain-live-locals
  [locals result removable]
  (:locals
   (reduce
    (fn [{:keys [live locals]} {:keys [id init] :as local}]
      (if (or (contains? live id) (not (contains? removable id)))
        {:live (into (disj live id) (util/free-syms init))
         :locals (into [local] locals)}
        {:live live :locals locals}))
    {:live (util/free-syms result) :locals []}
    (reverse locals))))

(defn- prune-certified-address-lets
  [expression removable]
  (letfn [(prune [form]
            (cond
              (and (seq? form) (contains? #{'let 'let* 'clojure.core/let} (first form))
                   (vector? (second form)) (even? (count (second form))))
              (let [head (first form)
                    pairs (mapv (fn [[id init]] [id (prune init)])
                                (partition 2 (second form)))
                    body (mapv prune (drop 2 form))
                    state
                    (reduce (fn [{:keys [live pairs]} [id init :as pair]]
                              (if (or (contains? live id) (not (contains? removable id)))
                                {:live (into (disj live id) (util/free-syms init))
                                 :pairs (into [pair] pairs)}
                                {:live live :pairs pairs}))
                            {:live (apply set/union #{} (map util/free-syms body)) :pairs []}
                            (reverse pairs))]
                (with-meta (list* head (vec (mapcat identity (:pairs state))) body) (meta form)))

              (seq? form) (with-meta (apply list (map prune form)) (meta form))
              (vector? form) (with-meta (mapv prune form) (meta form))
              ;; Compiler IR records are map-like, but rebuilding them through `empty` is neither
              ;; supported nor desirable: address-let pruning only owns source collection forms.
              (and (map? form) (not (record? form)))
              (with-meta (into (empty form)
                               (map (fn [[k v]] [(prune k) (prune v)])) form)
                         (meta form))
              :else form))]
    (prune expression)))

(defn validate-and-project-addresses
  "Project certified map load coordinates out of scalar SSA and into the index dialect.

   This transformation is deliberately graph-bound. It recomputes the attached structural
   certificate from the exact SegMap, checks that the exact graph node is present, and verifies
   that every certified minimum is enforced by the graph buffer. Only then are lexical address
   locals substituted into load coordinates; locals still used numerically remain intact.
   Direct schedule callers therefore retain checked scalar arithmetic and its traps."
  [operation node kernel-graph]
  (graph/validate! kernel-graph)
  (when-not (and (= operation (:operation node))
                 (some #(= node %) (:nodes kernel-graph)))
    (throw (ex-info "map address projection requires its exact graph node"
                    {:reason :map-address-certificate-node})))
  (let [attached (:read-capacity-certificate node)
        _ (when-not (= :zero-based-dense-read-spans (:kind attached))
            (throw (ex-info "map address projection requires a structural read certificate"
                            {:reason :map-address-certificate-missing})))
        recomputed (symbolic-read-certificate
                    operation
                    {:scalar-definitions (:scalar-definitions attached)})
        _ (when-not (= attached recomputed)
            (throw (ex-info "map address certificate does not match its source operation"
                            {:reason :map-address-certificate-mismatch
                             :attached attached :recomputed recomputed})))
        buffers (into {} (map (juxt :id identity))
                      (concat (:inputs kernel-graph) (:outputs kernel-graph)
                              (:temporaries kernel-graph)))
        _ (doseq [[id required] (:requirements attached)]
            (let [capacity (:elements (get buffers id))]
              (when-not (capacity-covers? capacity required)
                (throw (ex-info "graph buffer does not enforce the certified map read span"
                                {:reason :map-address-certificate-capacity
                                 :buffer id :required required :capacity capacity})))))
        remaining (atom (:reads attached))
        rewrite
        (fn [expression]
          (descriptor/rewrite-aget-reads
           expression
           (fn [read]
             (when (contains? (:inputs operation) (descriptor/aget-array-sym read))
               (let [fact (first @remaining)]
                 (when-not (and fact
                                (= (descriptor/aget-array-sym read) (:buffer fact))
                                (= (descriptor/aget-index read) (:source-coordinate fact)))
                   (throw (ex-info "map read order differs from its recomputed certificate"
                                   {:reason :map-address-certificate-read
                                    :read read :fact fact})))
                 (swap! remaining subvec 1)
                 (descriptor/rewrite-aget-index read (:projected-coordinate fact)))))))
        removable (apply set/union #{} (map :address-locals (:reads attached)))
        region (:scalar-region operation)
        locals (mapv #(update % :init
                              (fn [init]
                                (prune-certified-address-lets (rewrite init) removable)))
                     (:locals region))
        result (prune-certified-address-lets (rewrite (:result region)) removable)
        locals (retain-live-locals locals result removable)
        _ (when (seq @remaining)
            (throw (ex-info "map address certificate contains unprojected reads"
                            {:reason :map-address-certificate-read
                             :remaining @remaining})))]
    (-> operation
        (assoc :scalar-region (assoc region :locals locals :result result))
        (assoc :address-projection
               {:kind :certified-index-expression
                :certificate-kind (:kind attached)
                :requirements (:requirements attached)}))))

(defn static-read-requirements
  "Optional all-load proof for a plain, positive static, one-dimensional result map.
   The map schedule establishes 0<=index<bound at each :map-active load. Indirect/local-SSA
   coordinates, effects and unknown domains decline. Returned counts are minimum capacities."
  [operation options]
  (when (and (segop/seg-map? operation)
             (:out-sym operation)
             (= 1 (count (get-in operation [:space :dims])))
             (empty? (set/intersection (set (:inputs operation)) (set (:outputs operation))))
             (not (seq (get-in operation [:scalar-region :effects]))))
    (let [{index :name bound :bound} (first (get-in operation [:space :dims]))]
      (when (and (integer? bound) (<= 1 bound Integer/MAX_VALUE))
        (try
          (let [lowered (map-body/lower operation options)
                operations (tree-seq coll? seq (get-in lowered [:kernel-body :operations]))
                loads (filter #(= "raster.compiler.ir.kernel_body.ScalarLoad"
                                  (some-> % class .getName)) operations)
                ;; A lexical map local may be retained as typed SSA rather than beta-expanded
                ;; into the load coordinate.  Reuse only the range certificates carried by its
                ;; validated ScalarCompute; otherwise a harmless `(let* [k (+ i j)] (aget a k))`
                ;; loses the same capacity proof as its inlined spelling.
                computed-ranges
                (reduce
                 (fn [known node]
                   (if (= "raster.compiler.ir.kernel_body.ScalarCompute"
                          (some-> node class .getName))
                     (let [id (get-in node [:result :id])
                           type (get-in node [:result :type])
                           expression (:expression node)
                           proof (get-in expression [:options :proof])
                           argument-range
                           (fn [argument]
                             (cond
                               (symbol? argument) (get-in known [argument :range])
                               (and (integer? (:value argument)) (:type argument))
                               (ranges/literal (:value argument) (:type argument))
                               :else nil))
                           arguments (mapv argument-range (:arguments expression))
                           derived
                           (or (when (and (= :typed-scalar-range (:kind proof))
                                          (ranges/contained-in-dtype? proof type))
                                 (select-keys proof [:lower :upper]))
                               (when (every? some? arguments)
                                 (case (:op expression)
                                   (:+ :- :*)
                                   (let [range (ranges/arithmetic (:op expression) arguments)]
                                     (when (ranges/contained-in-dtype? range type) range))
                                   :quot
                                   (let [[numerator divisor] arguments
                                         range (when (and (<= 0 (:lower numerator))
                                                          (= (:lower divisor) (:upper divisor))
                                                          (pos? (:lower divisor)))
                                                 (ranges/quotient arguments))]
                                     (when (ranges/contained-in-dtype? range type) range))
                                   :rem
                                   (let [[numerator divisor] arguments
                                         d (:lower divisor)
                                         range (when (and (<= 0 (:lower numerator))
                                                          (= d (:upper divisor)) (pos? d))
                                                 {:lower 0
                                                  :upper (min (:upper numerator) (dec d))})]
                                     (when (ranges/contained-in-dtype? range type) range))
                                   :cast
                                   (let [range (first arguments)]
                                     (when (and (= :exact (get-in expression [:options :overflow]))
                                                (ranges/contained-in-dtype? range type))
                                       range))
                                   nil)))]
                       (if (and (symbol? id) derived)
                         (assoc known id {:type type :range derived})
                         known))
                     known))
                 {index {:type :long :range {:lower 0 :upper (dec bound)}}}
                 operations)
                leaf-types (merge (:scalar-types options)
                                  {index :long}
                                  (into {} (map (fn [[id fact]] [id (:type fact)]))
                                        computed-ranges))
                leaf-ranges (merge {index {:lower 0 :upper (dec bound)}}
                                   (into {} (map (fn [[id fact]] [id (:range fact)]))
                                         computed-ranges))
                requirements
                (mapv (fn [{:keys [buffer coordinates predicate]}]
                        (when (and (= :map-active predicate) (= 1 (count coordinates)))
                          (when-let [range (ranges/typed-index-range
                                           (first coordinates)
                                           leaf-types leaf-ranges)]
                            (when (<= 0 (:lower range))
                              [buffer (inc' (:upper range))]))))
                      loads)]
            (when (and (seq loads) (every? some? requirements))
              (reduce (fn [result [id extent]] (update result id (fnil max 0) extent))
                      {} requirements)))
          (catch clojure.lang.ExceptionInfo _ nil))))))
