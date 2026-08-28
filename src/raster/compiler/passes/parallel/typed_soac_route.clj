(ns raster.compiler.passes.parallel.typed-soac-route
  "Production bridge for the first closed TypedSOAC fusion subset.

   Pure maps and scalar/full reductions are fused in the typed dialect.  The resulting functional
   equations are mechanically materialized into the compatibility host expression and retained as
   first-class algorithms in a ParallelProgram.  A shadow comparison with the established graph
   implementation is a temporary migration certificate: disagreement declines to the old route."
  (:require [clojure.set :as set]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.reduction :as reduction]
            [raster.compiler.ir.soac :as legacy]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.soac-dialect-adapter :as adapter]
            [raster.compiler.passes.parallel.soac-graph :as graph]
            [raster.compiler.passes.parallel.typed-soac-fusion :as fusion]
            [raster.compiler.core.util :as util]))

(def ^:private dtype->allocation
  {:float ['clojure.core/float-array 'floats 'float]
   :double ['clojure.core/double-array 'doubles 'double]
   :int ['clojure.core/int-array 'ints 'int]
   :long ['clojure.core/long-array 'longs 'long]
   :byte ['clojure.core/byte-array 'bytes 'byte]})

(defn- provably-pure-host-binding?
  [expression]
  (or (not (seq? expression))
      (and (descriptor/alength-op? (descriptor/semantic-op expression))
           (= 1 (count (descriptor/call-args expression)))
           (symbol? (first (descriptor/call-args expression))))))

(defn- supported-node?
  [node]
  ;; Unknown calls are not evidence of purity. This first route carries atomic bindings and the
  ;; compiler's semantically identified array-length query; richer scalar equations belong in the
  ;; typed program itself.
  (or (and (legacy/scalar-binding? node) (provably-pure-host-binding? (:expr node)))
      (and (legacy/soac-map? node) (:pure? node))
      (and (legacy/soac-reduce? node)
           (empty? (:segment-axes node))
           (reduction/scalar? (:reduction node)))))

(defn- alength-array
  [expression]
  (when (and (seq? expression)
             (descriptor/alength-op? (descriptor/semantic-op expression))
             (= 1 (count (descriptor/call-args expression))))
    (first (descriptor/call-args expression))))

(defn- normalize-node-extents
  "Value-number pure scalar extents and replace `alength` of a known result with its certified
   producer extent. Scalar names remain distinct values; only their operation uses are canonical."
  [nodes]
  (:nodes
   (reduce
    (fn [{:keys [extents scalar-representatives] :as state} node]
      (cond
        (legacy/scalar-binding? node)
        (let [expression (:expr node)
              array (alength-array expression)
              representative (cond
                               (and array (contains? extents array)) (get extents array)
                               (symbol? expression)
                               (get scalar-representatives expression expression)
                               (integer? expression) expression
                               :else (:sym node))
              expression' (if (= representative (:sym node)) expression representative)]
          (-> state
              (update :nodes conj (assoc node :expr expression'))
              (assoc-in [:scalar-representatives (:sym node)] representative)))

        (or (legacy/soac-map? node) (legacy/soac-reduce? node))
        (let [bound (:bound node)
              array (alength-array bound)
              bound' (cond
                       (and array (contains? extents array)) (get extents array)
                       (symbol? bound) (get scalar-representatives bound bound)
                       :else bound)
              node' (assoc node :bound bound')]
          (cond-> (update state :nodes conj node')
            (legacy/soac-map? node') (assoc-in [:extents (:sym node')] bound')))

        :else
        (update state :nodes conj node)))
    {:nodes [] :extents {} :scalar-representatives {}}
    nodes)))

(defn- terminal-results
  [nodes body]
  (let [operations (filterv #(or (legacy/soac-map? %) (legacy/soac-reduce? %)) nodes)
        operation-definitions (set (mapcat (fn [node]
                                             (if (legacy/soac-map? node)
                                               [(:sym node)] (:outputs node)))
                                           operations))
        scalar-definitions (set (keep #(when (legacy/scalar-binding? %) (:sym %)) nodes))
        all-definitions (set/union operation-definitions scalar-definitions)
        operation-uses (set (concat
                             (mapcat #(concat (or (:inputs %) #{}) (or (:scalars %) #{}))
                                     operations)
                             (mapcat #(when (legacy/scalar-binding? %)
                                        (util/free-syms (:expr %)))
                                     nodes)))
        body-uses (set (mapcat util/free-syms body))]
    (vec (sort-by pr-str
                  (set/union (set/difference operation-definitions operation-uses)
                             (set/intersection all-definitions body-uses))))))

(defn- equation-subprogram
  [program equation]
  (let [equation-id (second equation)
        results (vec (nth equation 2))
        extent (dialect/operation-extent equation)
        inputs (cond-> (dialect/operation-inputs equation)
                 (dialect/value-id? extent) (conj extent))
        source-facts (dialect/facts program)
        equation-facts (get-in source-facts [:equations equation-id])
        value-ids (set (concat inputs results))
        facts (-> source-facts
                  (assoc :values (select-keys (:values source-facts) value-ids)
                         :inputs (vec inputs)
                         :equations {equation-id equation-facts}
                         :effects (:effects equation-facts)))]
    (dialect/make facts [equation] results)))

(defn- scalar-region
  [equation]
  (let [{:keys [attributes arrays captures lambda]} (dialect/operation-parts equation)
        [_ _ body-results] lambda
        {:keys [elements capture-parameters]} (dialect/parameter-layout equation)
        substitutions
        (into (zipmap capture-parameters captures)
              (map (fn [parameter array]
                     [parameter (list 'clojure.core/aget array (:index attributes))])
                   elements arrays))]
    (mapv #(util/subst-syms substitutions %) body-results)))

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
        bodies (scalar-region equation)
        placement-facts (get-in (dialect/facts program) [:equations equation-id])
        constituent-ids (or (some-> placement-facts :attributes :fusion/constituents keys set)
                            #{(or (get-in placement-facts [:provenance :legacy-soac-id]) equation-id)})
        placement (apply max constituent-ids)]
    (case kind
      scalar
      (let [_ (when-not (= 1 (count results))
                (throw (ex-info "the production scalar route requires one scalar result"
                                {:reason :typed-soac-production-subset
                                 :equation equation-id :results results})))
            result (first results)
            source (first bodies)]
        {:equation-id equation-id
         :placement placement
         :pairs [[result source]]
         :site [:binding result]
         :source source})

      map
      (do
        (when-not (= 1 (count results))
          (throw (ex-info "the first production TypedSOAC route requires one map result"
                          {:reason :typed-soac-production-subset
                           :equation equation-id :results results})))
        (let [result (first results)
              dtype (:dtype (get values result))
              [_ _ cast] (get dtype->allocation dtype)
              effect (gensym (str "typed_soac_map_" equation-id "__"))
              source (with-meta
                       (list 'raster.par/map! result (:index attributes) (:extent attributes)
                             cast (first bodies))
                       {:raster.type/elem-type dtype})]
          {:equation-id equation-id
           :placement placement
           :pairs [(allocation-pair values result (:extent attributes)) [effect source]]
           :site [:binding effect]
           :source source}))

      reduce
      (let [result (first results)
            accumulator (first (:accumulators attributes))
            source (list 'raster.par/reduce accumulator (first (:identities attributes))
                         (:index attributes) (:extent attributes) (first bodies))]
        {:equation-id equation-id
         :placement placement
         :pairs [[result source]]
         :site [:binding result]
         :source source}))))

(defn- realize-source
  [form program]
  (let [[let-head bindings & body] form
        realized (mapv #(realize-equation program %) (dialect/equations program))
        by-placement (group-by :placement realized)
        pairs
        (vec
         (mapcat
          (fn [id]
            (mapcat :pairs (get by-placement id)))
          (range (count (partition 2 bindings)))))
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
      :attributes {:host-control :typed-soac-materialization}
      :operation? equation-operation?
      :algorithm? (fn [equation algorithm]
                    (and (= algorithm (dialect/validate! algorithm))
                         (= (:operands equation) (:inputs (dialect/facts algorithm)))
                         (= (:results equation) (dialect/outputs algorithm))))})))

(defn attempt
  "Return a typed production ParallelProgram result, an explicit `:declined` result, or nil when
   the form is outside this closed subset. The legacy graph is used only as a temporary
   differential oracle during migration."
  ([form abstract-machine dtype]
   (attempt form abstract-machine dtype {}))
  ([form abstract-machine dtype array-types]
   (when (and (seq? form) (contains? #{'let 'let*} (first form)))
     (try
       (let [[_ bindings & body] form
             pairs (vec (partition 2 bindings))
             source-nodes (legacy/let-bindings->nodes pairs)
             nodes (normalize-node-extents source-nodes)]
         (when (and (seq nodes) (every? supported-node? nodes))
           (let [outputs (terminal-results nodes body)
                 typed-input (adapter/legacy-nodes->program
                              nodes {:outputs outputs :dtype dtype
                                     :array-types array-types
                                     :include-scalar-bindings? true})
                 [typed-result typed-stats] (fusion/fusion-fixpoint typed-input)
                 source-graph (graph/build-fusion-graph nodes)
                 [legacy-result _] (graph/fusion-fixpoint source-graph abstract-machine dtype)
                 shadow (adapter/legacy-nodes->program
                         (:nodes legacy-result) {:outputs outputs :dtype dtype
                                                 :array-types array-types
                                                 :include-scalar-bindings? true})]
             (cond
               (not (pos? (:vertical typed-stats)))
               {:declined {:reason :no-certified-vertical-fusion :stats typed-stats}}

               (pos? (:horizontal typed-stats))
               {:declined {:reason :typed-horizontal-materialization-pending :stats typed-stats}}

               (not= (dialect/equations shadow) (dialect/equations typed-result))
               {:declined {:reason :fusion-shadow-disagreement :stats typed-stats}}

               :else
               (let [{:keys [source realized]} (realize-source form typed-result)]
                 {:program (envelope typed-result source realized)
                  :stats (assoc typed-stats :route :typed-soac
                                :shadow-certified true)})))))
       (catch clojure.lang.ExceptionInfo exception
         (when (contains? #{:raster/fatal :raster/bug} (:reason (ex-data exception)))
           (throw exception))
         {:declined {:reason (or (:reason (ex-data exception)) :typed-soac-route-declined)
                     :message (.getMessage exception)
                     :details (ex-data exception)}})))))
