(ns raster.compiler.passes.parallel.typed-soac-route
  "Production route for the closed TypedSOAC map/scalar/reduction subset.

   Supported programs are represented and fused in the typed dialect whether or not a fusion
   fires. The resulting functional equations are mechanically projected into the surrounding host
   control expression and retained as first-class algorithms in a ParallelProgram."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-fusion :as fusion]
            [raster.compiler.passes.parallel.typed-soac-projection :as projection]
            [raster.compiler.passes.parallel.typed-soac-resident :as resident]
            [raster.compiler.core.util :as util]))

(def ^:private dtype->allocation
  {:float ['clojure.core/float-array 'floats 'float]
   :double ['clojure.core/double-array 'doubles 'double]
   :int ['clojure.core/int-array 'ints 'int]
   :long ['clojure.core/long-array 'longs 'long]
   :byte ['clojure.core/byte-array 'bytes 'byte]})

(defn- equation-subprogram
  [program equation]
  (let [equation-id (second equation)
        results (vec (nth equation 2))
        inputs (vec (distinct
                     (into (dialect/operation-inputs equation)
                           (filter dialect/value-id? (dialect/operation-extents equation)))))
        source-facts (dialect/facts program)
        equation-facts (get-in source-facts [:equations equation-id])
        value-ids (set (concat inputs results (dialect/physical-results source-facts equation)))
        facts (-> source-facts
                  (assoc :values (select-keys (:values source-facts) value-ids)
                         :inputs (vec inputs)
                         :equations {equation-id equation-facts}
                         :effects (:effects equation-facts)))]
    (dialect/make facts [equation] results)))

(defn- scalar-region
  [equation]
  (let [{:keys [kind attributes arrays captures lambda element-lambda]}
        (dialect/operation-parts equation)]
    (if (= 'segmented-fold-map kind)
      {:locals [] :bodies []}
      (let [{:keys [locals body-results]}
            (dialect/lambda-parts (or lambda element-lambda))
            {:keys [elements capture-parameters]} (dialect/parameter-layout equation)
            substitutions
            (into (zipmap capture-parameters captures)
                  (map (fn [parameter array]
                         [parameter (list 'clojure.core/aget array (:index attributes))])
                       elements arrays))]
        {:locals (mapv #(update % :init (fn [init]
                                          (projection/scalar-folds->source
                                           (util/subst-syms substitutions init))))
                       locals)
         :bodies (mapv #(projection/scalar-folds->source
                         (util/subst-syms substitutions %))
                       body-results)}))))

(defn- materialize-region
  [locals body]
  (if (seq locals)
    (list 'let*
          (vec
           (mapcat (fn [{:keys [id dtype init]}]
                     (let [tag (dtype/scalar-tag-for-dtype dtype)]
                       [(with-meta id {:raster.type/tag tag})
                        (list tag init)]))
                   locals))
          body)
    body))

(defn- allocation-pair
  [values result extent]
  (let [dtype (:dtype (get values result))
        [allocation tag] (or (get dtype->allocation dtype)
                             (throw (ex-info "TypedSOAC materialization has no array allocation"
                                             {:reason :typed-soac-materialization-dtype
                                              :result result :dtype dtype})))
        result' (with-meta result {:tag tag :raster.type/tag tag
                                   :raster.buffer/hoistable true})]
    [result' (list allocation extent)]))

(defn- realize-equation
  [program equation]
  (let [[_ equation-id results] equation
        {:keys [kind attributes]} (dialect/operation-parts equation)
        values (:values (dialect/facts program))
        {region-locals :locals bodies :bodies} (scalar-region equation)
        placement-facts (get-in (dialect/facts program) [:equations equation-id])
        constituent-ids (or (some-> placement-facts :attributes :fusion/constituents keys set)
                            #{(or (get-in placement-facts [:provenance :source-binding-id])
                                  (get-in placement-facts [:provenance :legacy-soac-id])
                                  equation-id)})
        placement (apply max constituent-ids)]
    (case kind
      scalar
      (let [_ (when-not (= 1 (count results))
                (throw (ex-info "the production scalar route requires one scalar result"
                                {:reason :typed-soac-production-subset
                                 :equation equation-id :results results})))
            result (first results)
            source (materialize-region region-locals (first bodies))]
        {:equation-id equation-id
         :placement placement
         :pairs [[result source]]
         :site [:binding result]
         :source source})

      map
      (do
        (when-not (and (seq results) (= (count results) (count bodies)))
          (throw (ex-info "the production TypedSOAC route requires one body per map result"
                          {:reason :typed-soac-production-subset
                           :equation equation-id :results results})))
        (let [result (first results)
              result-dtypes (mapv #(:dtype (get values %)) results)
              casts (mapv #(nth (get dtype->allocation %) 2 nil) result-dtypes)
              _ (when (some nil? casts)
                  (throw (ex-info "TypedSOAC materialization has no scalar cast for map output"
                                  {:reason :typed-soac-materialization-dtype
                                   :results results :dtypes result-dtypes})))
              cast (first casts)
              storage (dialect/result-storage (dialect/facts program) equation-id)
              physical-results (dialect/physical-results (dialect/facts program) equation)
              host-binding (or (get-in placement-facts [:attributes :host-binding]) result)
              host-bindings (or (get-in placement-facts [:attributes :host-bindings])
                                {result host-binding})
              host-returns (set (map :host-return storage))
              _ (when (and storage
                           (not (or (= #{:effect} host-returns)
                                    (= #{:buffer} host-returns))))
                  (throw (ex-info "stored map results require one coherent host return contract"
                                  {:reason :typed-soac-production-subset
                                   :equation equation-id :storage storage})))
              secondary-stores
              (mapv (fn [secondary secondary-cast body]
                      (list 'clojure.core/aset secondary (:index attributes)
                            (list secondary-cast body)))
                    (rest physical-results) (rest casts) (rest bodies))
              body (materialize-region
                    region-locals
                    (if (seq secondary-stores)
                      (list* 'do (concat secondary-stores [(first bodies)]))
                      (first bodies)))
              effect (gensym (str "typed_soac_map_" equation-id "__"))
              source (with-meta
                       (cond
                         (= #{:effect} host-returns)
                         (list 'raster.par/map-void! (:index attributes) (:extent attributes)
                               (materialize-region
                                region-locals
                                (list* 'do
                                       (map (fn [destination result-cast result-body]
                                              (list 'clojure.core/aset destination
                                                    (:index attributes)
                                                    (list result-cast result-body)))
                                            physical-results casts bodies))))

                         (= #{:buffer} host-returns)
                         (list 'raster.par/map! (first physical-results) (:index attributes)
                               (:extent attributes) cast
                               (materialize-region region-locals (first bodies)))

                         :else
                         (list 'raster.par/map! result (:index attributes) (:extent attributes)
                               cast body))
                       {:raster.type/elem-type (first result-dtypes)})]
          {:equation-id equation-id
           :placement placement
           :pairs (if storage
                    (into [[host-binding source]]
                          (keep (fn [[logical destination]]
                                  (let [binding (get host-bindings logical)]
                                    (when (and binding (not= binding host-binding))
                                      [binding destination])))
                                (map vector results physical-results)))
                    (conj (mapv #(allocation-pair values % (:extent attributes)) results)
                          [effect source]))
           :site [:binding (if storage host-binding effect)]
           :source source}))

      scatter
      (let [storage (dialect/result-storage (dialect/facts program) equation-id)
            physical-results (dialect/physical-results (dialect/facts program) equation)
            host-binding (or (get-in placement-facts [:attributes :host-binding])
                             (first results))
            result-dtypes (mapv #(:dtype (get values %)) results)
            casts (mapv #(nth (get dtype->allocation %) 2 nil) result-dtypes)
            writes (mapv dialect/write-parts bodies)
            conflict (:conflict attributes)
            reducing? (dialect/reducing-scatter-conflict? conflict)
            host-returns (mapv :host-return storage)
            _ (when-not (and (= (count results) (count writes) (count physical-results))
                             (every? some? writes)
                             (every? some? casts)
                             (every? #{:effect :buffer} host-returns))
                (throw (ex-info "the production scatter route requires typed effect writes"
                                {:reason :typed-soac-production-subset
                                 :equation equation-id :results results
                                 :storage storage :writes writes :dtypes result-dtypes})))
            statements
            (mapv (fn [destination result-dtype cast
                       {:keys [destination-index predicate value]}]
                    (let [destination (with-meta destination
                                        {:raster.type/tag
                                         (dtype/scalar-tag-for-dtype result-dtype)
                                         :tag (dtype/scalar-tag-for-dtype result-dtype)})
                          typed-value (list cast value)
                          store (if reducing?
                                  (list 'raster.par/atomic-add!
                                        destination destination-index typed-value)
                                  (list 'clojure.core/aset destination destination-index
                                        typed-value))]
                      (if (contains? #{true 1} predicate) store (list 'if predicate store))))
                  physical-results result-dtypes casts writes)
            effect-source (with-meta
                            (list 'raster.par/map-void! (:index attributes)
                                  (:extent attributes)
                                  (materialize-region region-locals (list* 'do statements)))
                            {:raster.type/elem-type (first result-dtypes)})
            effect-binding (clojure.core/symbol (str "scatter_effect__" equation-id))
            buffer-return? (= [:buffer] host-returns)]
        {:equation-id equation-id
         :placement placement
         :pairs (if buffer-return?
                  [[effect-binding effect-source]
                   [host-binding (first physical-results)]]
                  [[host-binding effect-source]])
         :site [:binding (if buffer-return? effect-binding host-binding)]
         :source effect-source})

      stencil
      (let [host-binding (or (get-in placement-facts [:attributes :host-binding])
                             (first results))
            source (projection/stencil-form program equation)]
        {:equation-id equation-id
         :placement placement
         :pairs [[host-binding source]]
         :site [:binding host-binding]
         :source source})

      reduce
      (let [result (first results)
            accumulator (first (:accumulators attributes))
            resident? (resident/resident-scalar-value? (get values result))
            body (materialize-region region-locals (first bodies))
            source (if resident?
                     ;; reduce-into owns its explicit one-element destination first. The
                     ;; functional reduction algorithm itself remains unchanged.
                     (list 'raster.par/reduce-into result accumulator
                           (first (:identities attributes)) (:index attributes)
                           (:extent attributes) body)
                     (list 'raster.par/reduce accumulator (first (:identities attributes))
                           (:index attributes) (:extent attributes) body))]
        (if resident?
          (let [effect (gensym (str "typed_soac_reduce_" equation-id "__"))]
            {:equation-id equation-id
             :placement placement
             :pairs [(allocation-pair values result 1) [effect source]]
             :site [:binding effect]
             :source source})
          {:equation-id equation-id
           :placement placement
           :pairs [[result source]]
           :site [:binding result]
           :source source}))

      segmented-reduce
      (let [host-binding (or (get-in placement-facts [:attributes :host-binding])
                             (first results))
            source (projection/segmented-reduce-contract-form program equation)]
        {:equation-id equation-id
         :placement placement
         :pairs [[host-binding source]]
         :site [:binding host-binding]
         :source source})

      product-reduce
      (let [host-binding (or (get-in placement-facts [:attributes :host-binding])
                             (first results))
            source (projection/product-reduce-form program equation)]
        {:equation-id equation-id
         :placement placement
         :pairs [[host-binding source]]
         :site [:binding host-binding]
         :source source})

      segmented-fold-map
      (let [host-binding (or (get-in placement-facts [:attributes :host-binding])
                             (first results))
            source (projection/segmented-fold-map-form program equation)]
        {:equation-id equation-id
         :placement placement
         :pairs [[host-binding source]]
         :site [:binding host-binding]
         :source source})

      scan
      (let [result (first results)
            mode (:mode attributes)
            ;; Scans share the same logical-result/physical-storage contract as maps and
            ;; segmented reductions. Materialization must not recover a second, scan-only
            ;; destination convention from equation attributes.
            storage (get-in placement-facts [:attributes :result-storage])
            destination (:destination (first storage))
            _ (when-not (and (= 1 (count results)) destination)
                (throw (ex-info "typed scan requires one caller-owned destination"
                                {:reason :typed-soac-production-subset
                                 :equation equation-id :results results
                                 :result-storage storage})))
            source (with-meta
                     (list (case mode
                             :inclusive 'raster.par/scan
                             :exclusive 'raster.par/scan-exclusive)
                           destination
                           (first (:accumulators attributes))
                           (first (:identities attributes))
                           (:index attributes) (:extent attributes)
                           (nth (get dtype->allocation (first (:dtypes attributes))) 2 nil)
                           (materialize-region region-locals (first bodies)))
                     {:raster.type/elem-type (first (:dtypes attributes))})]
        {:equation-id equation-id
         :placement placement
         :pairs [[result source]]
         :site [:binding result]
         :source source}))))

(defn- required-host-bindings
  "Return the transitive source bindings needed by host materialization.

   Typed equations own parallel/scalar computation, but a caller-owned physical destination may
   retain an ordinary allocation expression. That expression can depend on pure scalar bindings
   introduced by inlining (for example `size = m*n`). Preserve that dependency closure instead of
   leaving an alpha-renamed callee parameter free in the reconstructed JVM/C host form. `pairs`
   contains only bindings without a realized typed equation, so dependency traversal stops at the
   new semantic boundary instead of resurrecting a fused-away producer."
  [pairs roots]
  (let [definitions (into {} (map (fn [[symbol expression]] [symbol expression])) pairs)
        definition-symbols (set (keys definitions))]
    (loop [required (set/intersection definition-symbols (set roots))]
      (let [dependencies
            (->> required
                 (mapcat #(util/free-syms (get definitions %)))
                 set
                 (set/intersection definition-symbols))
            required' (set/union required dependencies)]
        (if (= required required') required (recur required'))))))

(defn- realize-source
  [form program]
  (let [[let-head bindings & body] form
        original-pairs (vec (partition 2 bindings))
        realized (mapv #(realize-equation program %) (dialect/equations program))
        by-placement (group-by :placement realized)
        realized-host-symbols
        (into #{} (mapcat (fn [equation]
                            (keep (fn [[symbol _]] (when (symbol? symbol) symbol))
                                  (:pairs equation))))
              realized)
        physical-destinations
        (into #{}
              (mapcat (fn [[_ equation-facts]]
                        (concat
                         (keep identity [(get-in equation-facts [:attributes :destination])])
                         (map :destination
                              (get-in equation-facts [:attributes :result-storage])))))
              (:equations (dialect/facts program)))
        host-bindings
        (required-host-bindings
         (keep-indexed (fn [_ [symbol :as pair]]
                         ;; A fused equation may be placed at a later constituent while
                         ;; defining a host symbol whose original source pair occurred earlier.
                         ;; That symbol is globally realized; retaining its old pair would execute
                         ;; the producer twice and bypass the fused schedule.
                         (when-not (contains? realized-host-symbols symbol) pair))
                       original-pairs)
         (set/union physical-destinations
                    (into #{} (mapcat util/free-syms) body)))
        pairs
        (vec
         (mapcat
          (fn [id]
            (let [[symbol :as original-pair] (nth original-pairs id)]
              (concat
               ;; A destination may be a caller-owned input or a locally allocated buffer. The
               ;; typed algorithm records the physical write boundary but does not own host-side
               ;; allocation. Preserve a defining source binding when one exists; dropping it
               ;; leaves CPU-C/JVM materialization with an unbound destination and also loses a
               ;; resident scratch allocation before GPU extraction.
               (when (and (contains? host-bindings symbol)
                          (not (contains? realized-host-symbols symbol))
                          (empty? (get by-placement id)))
                 [original-pair])
               (mapcat :pairs (get by-placement id)))))
          (range (count original-pairs))))
        source (with-meta (list* let-head (vec (mapcat identity pairs)) body) (meta form))]
    {:source source :realized (into {} (map (juxt :equation-id identity)) realized)}))

(defn- equation-operation?
  [operation]
  (boolean (fusion/equation-info operation)))

(defn- envelope
  [typed-program source realized]
  (let [facts (dialect/facts typed-program)
        equations
        (mapv
         (fn [equation]
           (let [equation-id (second equation)
                 subprogram (equation-subprogram typed-program equation)
                 {:keys [site source]} (get realized equation-id)
                 equation-facts (get-in facts [:equations equation-id])]
             (parallel-program/->ProgramEquation
              equation-id site source
              (:inputs (dialect/facts subprogram)) (dialect/outputs subprogram)
              subprogram [equation] (:effects equation-facts)
              (assoc (:provenance equation-facts) :pass :typed-soac-fusion)
              (assoc (:attributes equation-facts) :algorithm-dialect :typed-soac))))
         (dialect/equations typed-program))]
    (parallel-program/make
     {:dialect :typed-soac
      :source source
      :values (:values facts)
      :inputs (:inputs facts)
      :equations equations
      :outputs (dialect/outputs typed-program)
      :effects (:effects facts)
      :diagnostics (:diagnostics facts)
      :provenance (assoc (:provenance facts) :pass :typed-soac-fusion)
      :attributes (assoc (:attributes facts) :host-control :typed-soac-materialization)
      :operation? equation-operation?
      :algorithm? (fn [equation algorithm]
                    (and (= algorithm (dialect/validate! algorithm))
                         (= (:operands equation) (:inputs (dialect/facts algorithm)))
                         (= (:results equation) (dialect/outputs algorithm))))})))

(defn program-envelope
  "Wrap an already constructed TypedSOAC program in the same ParallelProgram boundary used by
   the analyzed-source production route.

   This source-free entry is for compiler-generated algorithms such as structured-control bodies.
   Each equation keeps its closed TypedSOAC subprogram and stable value/equation facts; no host
   expression is reconstructed merely to enter scheduling."
  [typed-program]
  (let [typed-program (dialect/validate! typed-program)
        realized (into {}
                       (map (fn [equation]
                              (let [equation-id (second equation)]
                                [equation-id {:site [:algorithm equation-id] :source nil}])))
                       (dialect/equations typed-program))]
    (-> (envelope typed-program nil realized)
        (assoc :attributes (assoc (:attributes (dialect/facts typed-program))
                                  :host-control :explicit-typed-algorithm))
        (parallel-program/validate!
         equation-operation?
         (fn [equation algorithm]
           (and (= algorithm (dialect/validate! algorithm))
                (= (:operands equation) (:inputs (dialect/facts algorithm)))
                (= (:results equation) (dialect/outputs algorithm))))))))

(defn attempt
  "Return a typed production ParallelProgram result, an explicit `:declined` result, or nil when
   the form is outside this closed subset."
  ([form dtype]
   (attempt form dtype {}))
  ([form dtype array-types]
   (attempt form dtype array-types {}))
  ([form dtype array-types {:keys [resident-reductions? scalar-types values abstract-machine]
                            :or {resident-reductions? false}}]
   (when (and (seq? form) (contains? #{'let 'let*} (first form)))
     (let [form (frontend/normalize-source form)]
       (try
         (let [frontend-options {:dtype dtype :array-types array-types
                                 :scalar-types scalar-types :values values}
               typed-input (frontend/form->program form frontend-options)]
           (if-not typed-input
             (when-let [decline (frontend/coverage-decline form frontend-options)]
               {:declined decline})
             (let [[typed-result typed-stats]
                   (fusion/fusion-fixpoint typed-input abstract-machine)
                   [typed-result resident-stats]
                   (if resident-reductions?
                     (resident/realize typed-result)
                     [typed-result {:resident-reductions 0 :inlined-scalars 0}])]
               (if (not-any? #(contains? #{:map :scatter :stencil :reduce
                                           :segmented-reduce :product-reduce
                                           :segmented-fold-map :scan}
                                         (:kind (fusion/equation-info %)))
                             (dialect/equations typed-result))
                 {:declined {:reason :no-certified-parallel-equation
                             :stats (merge typed-stats resident-stats)}}
                 (let [{:keys [source realized]} (realize-source form typed-result)]
                   {:program (envelope typed-result source realized)
                    :stats (merge typed-stats resident-stats
                                  {:route :typed-soac :typed-validated true
                                   :front-end :analyzed-source})})))))
         (catch clojure.lang.ExceptionInfo exception
           (when (contains? #{:raster/fatal :raster/bug} (:reason (ex-data exception)))
             (throw exception))
           {:declined {:reason (or (:reason (ex-data exception)) :typed-soac-route-declined)
                       :message (.getMessage exception)
                       :details (ex-data exception)}}))))))
