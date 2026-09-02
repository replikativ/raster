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
            [raster.compiler.core.types :as types]
            [raster.compiler.core.util :as util]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.contraction-facts :as contraction-facts]
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
    :typed-soac-unknown-value})

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

(defn- unique-index-expression
  "Return the inner destination expression when `expression` carries Raster's explicit
   uniqueness contract. The marker may be a direct source call or a walker-devirtualized call;
   semantic-op/call-args are the only sanctioned way to look through the latter."
  [expression]
  (when (and (seq? expression)
             (contains? unique-index-ops (descriptor/semantic-op expression))
             (= 1 (count (descriptor/call-args expression))))
    (first (descriptor/call-args expression))))

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

(defn- pointwise-region
  "Recognize an ordered, pure local-SSA spine ending exclusively in indexed stores.

   Local types come only from retained walker/TypedClojure facts. Nested local scopes and missing
   local types decline; guessing them in this source recognizer or a C emitter would make the
   region only nominally typed. Store legality is checked separately after local dependencies are
   expanded for analysis."
  [body index]
  (cond
    (and (seq? body) (form/let-head? (first body)))
    (let [[_ bindings & nested-body] body]
      (when (and (even? (count bindings))
                 (seq nested-body)
                 (not-any? util/effectful? (take-nth 2 (rest bindings))))
        (when-let [{nested-locals :locals stores :stores}
                   (pointwise-region (list* 'do nested-body) index)]
          ;; Flattening nested lexical scopes needs alpha-renaming across each scope. Keep this
          ;; first production slice to the ANF shape the walker emits: one local spine around the
          ;; store sequence.
          (when (empty? nested-locals)
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
                              {:locals [] :substitutions {}} typed)]
                  {:locals locals
                   :stores (mapv (fn [store]
                                   (reduce (fn [store field]
                                             (update store field
                                                     #(util/subst-syms substitutions %)))
                                           store [:index :predicate :value]))
                                 stores)})))))))

    (and (seq? body) (= 'do (first body)))
    (let [groups (mapv #(pointwise-region % index) (rest body))]
      (when (and (seq groups) (every? some? groups)
                 (every? (comp empty? :locals) groups))
        {:locals [] :stores (vec (mapcat :stores groups))}))

    (and (seq? body)
         (contains? #{'if 'clojure.core/if} (first body))
         (<= 3 (count body) 4))
    (let [[_ predicate then-expression else-expression] body
          then-region (pointwise-region then-expression index)
          else-region (when else-expression (pointwise-region else-expression index))]
      (when (and then-region
                 (empty? (:locals then-region))
                 (or (nil? else-expression)
                     (and else-region (empty? (:locals else-region))))
                 (or (nil? else-region)
                     (= (mapv (juxt :out :index) (:stores then-region))
                        (mapv (juxt :out :index) (:stores else-region)))))
        {:locals []
         :stores
         (mapv (fn [ordinal then-store]
                 (let [else-store (when else-region (nth (:stores else-region) ordinal))]
                   (if else-store
                     (assoc then-store
                            :value (list 'if predicate (:value then-store) (:value else-store))
                            :predicate (list 'if predicate
                                             (:predicate then-store)
                                             (:predicate else-store)))
                     (update then-store :predicate
                             #(if (contains? #{true 1} %)
                                predicate
                                (list 'if predicate % 0))))))
               (range) (:stores then-region))}))

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
                                      'clojure.core/float 'clojure.core/double}
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

(defn- effect-map-description
  [id symbol index extent {:keys [locals stores]} elem-type
   & {:keys [host-return] :or {host-return :effect}}]
  (when (and (seq stores) (independent-stores? locals stores)
             (or (= :effect host-return) (= 1 (count stores))))
    (let [stores (mapv #(merge {:index index :predicate 1} %) stores)
          pointwise? (every? #(and (= index (:index %)) (nil? (:reduction-op %))) stores)
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
          reduction-ops (set (keep :reduction-op stores))
          conflict (cond
                     (every? #(= :unique (:conflict %)) stores) :unique
                     (and (= 1 (count reduction-ops))
                          (every? :reduction-op stores))
                     (dialect/reducing-scatter-conflict (first reduction-ops) elem-type))
          destinations (mapv :out stores)
          values (mapv :value stores)
          write-indices (mapv :index stores)
          predicates (mapv :predicate stores)
          analysis-values (concat (map :init locals) write-indices predicates values)
          io (update (extract-io (list* 'do analysis-values) index destinations)
                     :scalars set/difference (set (map :id locals)))
          results (if (= :buffer host-return)
                    [symbol]
                    (mapv #(effect-result-id id %) (range (count stores))))]
      (when (or pointwise? (= :unique conflict)
                (dialect/reducing-scatter-conflict? conflict))
        (merge {:kind (if pointwise? :map :scatter)
                :id id :sym symbol :index index :extent extent
                :results results :locals locals :bodies values :casts (mapv :cast stores)
                :write-indices write-indices :predicates predicates
                :conflict (when-not pointwise? conflict)
                :effect-only? (= :effect host-return)
                :host-binding symbol :elem-type elem-type
                :result-storage
                (mapv (fn [destination]
                        {:destination destination
                         :access (if (or (not pointwise?)
                                         (contains? (:inputs io) destination))
                                   :read-write :write)
                         :host-return host-return})
                      destinations)}
               io)))))

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
  [id symbol expression default-dtype]
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
          elem-type (dtype/canon (or elem-type
                                     (dtype/dtype-for-scalar-tag cast)
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
      (effect-map-description id symbol idx bound
                              {:locals []
                               :stores [{:out out1 :value body1 :cast cast}
                                        {:out out2 :value body2 :cast cast}]}
                              elem-type))

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
                 expression :dtype (or (:raster.type/elem-type (meta expression))
                                       default-dtype :double))
          {:keys [free-axes contract-axes out opts]} facts
          epilogue (:epilogue facts)
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
           ;; destinations remain `:effect` storage and retain host nil semantics.
           :result-storage [{:destination out :access :write :host-return :buffer}]})))

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
      (when-let [region (pointwise-region body idx)]
        (effect-map-description id symbol idx bound region
                                (dtype/canon (or elem-type default-dtype))
                                :host-return (or (::host-return (meta expression)) :effect))))

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

(defn normalize-source
  "Give pure compound parallel extents stable scalar SSA identities before dialect construction.

   TypedSOAC operations name extents; executable host expressions never leak into schedule fields.
   This normalization inserts an ordinary typed scalar binding immediately before its operation,
   so the same expression remains visible to JVM materialization and runtime specialization."
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
             (fn [{:keys [compound-extents] :as state} [ordinal [symbol expression]]]
               (let [expression (canonicalize-strided-indexed-operation ordinal expression)
                     extent (parallel-extent expression)
                     canonical-extent (some-> extent descriptor/unwrap-int-cast)]
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
                                       {:tag 'long :raster.type/tag 'long})]
                       (-> state
                           (assoc-in [:compound-extents canonical-extent] extent-id)
                           (update :normalized into
                                   [[extent-id canonical-extent]
                                    [symbol (replace-parallel-extent expression extent-id)]]))))

                   :else
                   (update state :normalized conj [symbol expression]))))
             {:normalized [] :compound-extents {}}
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
  [pairs default-dtype]
  (mapv (fn [id [symbol expression]]
          (or (operation-description id symbol expression default-dtype)
              (if (par/par-form? expression)
                {:kind :unsupported :id id :sym symbol :expr expression}
                {:kind :scalar :id id :sym symbol :expr expression})))
        (range) pairs))

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

        (:map :scatter :stencil :reduce :scan)
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
            (contains? #{:map :scatter :stencil :scan} (:kind description'))
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
  [descriptions]
  (reduce set/union #{}
          (map #(if (contains? #{:map :scatter :stencil :reduce :segmented-reduce
                                 :product-reduce :segmented-fold-map :scan} (:kind %))
                  (:outputs %) #{})
               descriptions)))

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

(defn coverage-decline
  "Describe the exact source bindings that prevent admission to the closed TypedSOAC subset.

   This is diagnostic evidence only: it uses the same descriptions and admission predicate as
   form->program, so reporting cannot become a second legality implementation."
  [source {:keys [dtype values shape-equalities]
           :or {dtype :double values {} shape-equalities {}}}]
  (when (and (seq? source) (contains? #{'let 'let*} (first source)))
    (let [[_ bindings] source
          pairs (vec (partition 2 bindings))
          descriptions (normalize-extents (source-descriptions pairs dtype)
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

(defn- map-equation
  [description]
  (let [{:keys [id index extent locals casts bodies inputs results elem-type]} description
        fold-dtype (or elem-type (:result-dtype description) :double)
        expressions (mapv (fn [cast body] (if cast (list cast body) body)) casts bodies)
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
        operations (filter #(contains? #{:map :scatter :stencil :reduce :segmented-reduce
                                         :product-reduce :segmented-fold-map :scan} (:kind %))
                           descriptions)
        operation-definitions (set (mapcat #(case (:kind %)
                                              (:map :scatter :stencil) (:results %)
                                              :scan [(:sym %)]
                                              :segmented-reduce (:results %)
                                              :product-reduce (:results %)
                                              :segmented-fold-map (:results %)
                                              (:outputs %))
                                           operations))
        terminal-operation-definitions
        (set (mapcat #(case (:kind %)
                        (:map :scatter :stencil) (if (:effect-only? %) [] (:results %))
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

(defn- selected-scalars
  [descriptions operation-equations outputs]
  (let [physical-outputs (physical-output-symbols descriptions)
        by-symbol (into {}
                        (keep #(when (and (= :scalar (:kind %))
                                          (supported-description? physical-outputs %)
                                          (not (generated-scaffolding? % physical-outputs)))
                                 [(:sym %) %]))
                        descriptions)
        roots (set (concat outputs
                           (mapcat (fn [equation]
                                     (into (dialect/operation-inputs equation)
                                           (filter dialect/value-id?
                                                   (dialect/operation-extents equation))))
                                   operation-equations)))]
    (loop [needed (set/intersection roots (set (keys by-symbol)))]
      (let [dependencies (set (mapcat #(util/free-syms (:expr (get by-symbol %))) needed))
            needed' (set/union needed (set/intersection dependencies (set (keys by-symbol))))]
        (if (= needed needed') needed (recur needed'))))))

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
                                         scatter [(list 'unknown-dimension id)]
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

(defn form->program
  "Construct and validate TypedSOAC islands directly from a let form.

   Opaque scalar host bindings are retained as ordered barriers around the functional equations.
   Returns nil when a parallel binding is outside the certified source subset. Type/effect/value
   contradictions throw ExceptionInfo because falling through after accepting them would hide a
   compiler correctness defect."
  [source {:keys [dtype array-types scalar-types values shape-equalities]
           :or {dtype :double array-types {} scalar-types {} values {} shape-equalities {}}}]
  (when (and (seq? source) (contains? #{'let 'let*} (first source)))
    (let [[_ bindings & body] source
          pairs (vec (partition 2 bindings))
          descriptions (normalize-extents (source-descriptions pairs dtype)
                                          shape-equalities values)]
      (when (and (even? (count bindings))
                 (seq descriptions)
                 (some #(contains? #{:map :scatter :stencil :reduce :segmented-reduce
                                     :product-reduce :segmented-fold-map :scan}
                                   (:kind %))
                       descriptions)
                 (supported-descriptions? descriptions))
        (let [operation-descriptions
              (filterv #(contains? #{:map :scatter :stencil :reduce :segmented-reduce
                                     :product-reduce :segmented-fold-map :scan} (:kind %))
                       descriptions)
              physical-outputs (physical-output-symbols descriptions)
              operation-equations (mapv #(case (:kind %) :map (map-equation %)
                                               :scatter (scatter-equation %)
                                               :stencil (stencil-equation %)
                                               :reduce (reduce-equation %)
                                               :segmented-reduce (segmented-reduce-equation %)
                                               :product-reduce (product-reduce-equation %)
                                               :segmented-fold-map (segmented-fold-map-equation %)
                                               :scan (scan-equation %))
                                        operation-descriptions)
              outputs (terminal-results descriptions body)
              required-scalars (selected-scalars descriptions operation-equations outputs)
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
                          (:map :scatter :stencil :reduce :segmented-reduce
                                :product-reduce :segmented-fold-map :scan)
                          (-> state
                              (update :equations conj
                                      (case (:kind description)
                                        :map (map-equation description)
                                        :scatter (scatter-equation description)
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
                                storage
                                (assoc :effects #{:memory/write}
                                       :aliases (into {}
                                                      (map (fn [result {:keys [destination]}]
                                                             [result destination])
                                                           (:results description) storage))
                                       :attributes {:result-storage storage
                                                    :host-binding
                                                    (:host-binding description)}))]))
                         equation-descriptions))
              total-effects (reduce set/union #{} (map :effects (vals equation-facts)))
              host-binding-ids
              (->> descriptions
                   (keep #(when (and (= :scalar (:kind %))
                                     (not (supported-description?
                                           physical-outputs %)))
                            (:id %)))
                   vec)
              facts (dialect/default-program-facts
                     {:values values :inputs inputs :equations equation-facts
                      :effects total-effects
                      :provenance {:front-end :analyzed-source}
                      :attributes {:source-dialect :closed-clojure
                                   :host-binding-ids host-binding-ids}})]
          (dialect/make facts equations outputs))))))
