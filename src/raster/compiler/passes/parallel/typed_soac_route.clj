(ns raster.compiler.passes.parallel.typed-soac-route
  "Production route for the closed TypedSOAC map/scalar/reduction subset.

   Supported programs are represented and fused in the typed dialect whether or not a fusion
   fires. The resulting functional equations are mechanically projected into the surrounding host
   control expression and retained as first-class algorithms in a ParallelProgram."
  (:require [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-fusion :as fusion]
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
            source (first bodies)]
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
              destination (get-in placement-facts [:attributes :destination])
              _ (when (and destination (not= 1 (count results)))
                  (throw (ex-info "an effectful map destination cannot have fused logical outputs"
                                  {:reason :typed-soac-production-subset
                                   :equation equation-id :results results
                                   :destination destination})))
              secondary-stores
              (mapv (fn [secondary secondary-cast body]
                      (list 'clojure.core/aset secondary (:index attributes)
                            (list secondary-cast body)))
                    (rest results) (rest casts) (rest bodies))
              body (if (seq secondary-stores)
                     (list* 'do (concat secondary-stores [(first bodies)]))
                     (first bodies))
              effect (gensym (str "typed_soac_map_" equation-id "__"))
              source (with-meta
                       (if destination
                         (list 'raster.par/map-void! (:index attributes) (:extent attributes)
                               (list 'clojure.core/aset destination (:index attributes)
                                     (list cast body)))
                         (list 'raster.par/map! result (:index attributes) (:extent attributes)
                               cast body))
                       {:raster.type/elem-type (first result-dtypes)})]
          {:equation-id equation-id
           :placement placement
           :pairs (if destination
                    [[result source]]
                    (conj (mapv #(allocation-pair values % (:extent attributes)) results)
                          [effect source]))
           :site [:binding (if destination result effect)]
           :source source}))

      reduce
      (let [result (first results)
            accumulator (first (:accumulators attributes))
            resident? (resident/resident-scalar-value? (get values result))
            source (if resident?
                     ;; reduce-into owns its explicit one-element destination first. The
                     ;; functional reduction algorithm itself remains unchanged.
                     (list 'raster.par/reduce-into result accumulator
                           (first (:identities attributes)) (:index attributes)
                           (:extent attributes) (first bodies))
                     (list 'raster.par/reduce accumulator (first (:identities attributes))
                           (:index attributes) (:extent attributes) (first bodies)))]
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
           :source source})))))

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
   the form is outside this closed subset."
  ([form dtype]
   (attempt form dtype {}))
  ([form dtype array-types]
   (attempt form dtype array-types {}))
  ([form dtype array-types {:keys [resident-reductions?]
                            :or {resident-reductions? false}}]
   (when (and (seq? form) (contains? #{'let 'let*} (first form)))
     (try
       (when-let [typed-input (frontend/form->program
                               form {:dtype dtype :array-types array-types})]
         (let [[typed-result typed-stats] (fusion/fusion-fixpoint typed-input)
               [typed-result resident-stats]
               (if resident-reductions?
                 (resident/realize typed-result)
                 [typed-result {:resident-reductions 0 :inlined-scalars 0}])]
           (if (not-any? #(contains? #{:map :reduce}
                                     (:kind (fusion/equation-info %)))
                         (dialect/equations typed-result))
             {:declined {:reason :no-certified-parallel-equation
                         :stats (merge typed-stats resident-stats)}}
             (let [{:keys [source realized]} (realize-source form typed-result)]
               {:program (envelope typed-result source realized)
                :stats (merge typed-stats resident-stats
                              {:route :typed-soac :typed-validated true
                               :front-end :analyzed-source})}))))
       (catch clojure.lang.ExceptionInfo exception
         (when (contains? #{:raster/fatal :raster/bug} (:reason (ex-data exception)))
           (throw exception))
         {:declined {:reason (or (:reason (ex-data exception)) :typed-soac-route-declined)
                     :message (.getMessage exception)
                     :details (ex-data exception)}})))))
