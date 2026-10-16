(ns raster.ad.normalized-device
  "Opt-in lease-oriented device execution for one canonical normalized map owner.

   This surface is intentionally not the CPU-compatible normalized AD runtime: results remain
   resident behind opaque entry leases and callers supply exact primitive host input buffers for
   each invocation. Preparation fixes scalar values, shapes, target and numerical policy."
  (:require [clojure.set :as set]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.invocation-materialization :as materialization]
            [raster.compiler.ir.invocation-plan :as invocation]
            [raster.compiler.ir.link-module-plan :as module-plan]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.gpu.link-module :as link-module]))

(defn- fail! [reason message data]
  (throw (ex-info message (assoc data :reason reason :ad :normalized-device))))

(def ^:private option-keys #{:target :schedule :scalar-math :profile?})
(def ^:private prepared-keys #{:ports :primal! :transpose! :close!})

(defn- caller-options [options]
  (when (contains? options :scalar-math) (select-keys options [:scalar-math])))

(defn- authority [owner expected-kind]
  (let [owner (canonical/validate! owner)
        derivation (:derivation owner)
        bundle (:bundle derivation)]
    (when-not (= expected-kind (:kind derivation))
      (fail! :normalized-device-family
             "device preparation received a different canonical normalized family"
             {:expected expected-kind :kind (:kind derivation)}))
    (let [guard! (normalized/execution-guard
                  (:source-ns owner) (normalized/revisions)
                  (normalized/execution-sources bundle))]
      {:owner owner
       :guard! guard!
       :check! (fn [] (guard!) (canonical/validate! owner))})))

(defn- compile-phase! [{:keys [owner check!]} phase options]
  (check!)
  (let [compiled (equation-first/compile-typed (get owner phase) options)]
    (check!)
    compiled))

(defn- compile-program! [{:keys [check!]} program options]
  (check!)
  (let [compiled (equation-first/compile-typed program options)]
    (check!)
    compiled))

(defn- singleton-derivative-ports! [interface family]
  (let [active-inputs (vec (:active-inputs interface))
        tangent-inputs (vec (:tangent-inputs interface))
        cotangent-results (vec (:cotangent-results interface))]
    ;; Canonical validation has already established exact port correspondence. Device lowering is
    ;; deliberately still the singleton slice, and must refuse before compilation/materialization.
    (when-not (= 1 (count active-inputs) (count tangent-inputs) (count cotangent-results))
      (fail! :normalized-device-multi-active
             "normalized device execution does not yet support multiple active derivative ports"
             {:family family :active-inputs active-inputs
              :tangent-inputs (mapv :id tangent-inputs)
              :cotangent-results (mapv :id cotangent-results)}))
    (let [inputs (:inputs interface)
          active-id (first active-inputs)
          active (some #(when (= active-id (:id %)) %) inputs)
          tangent (first tangent-inputs)
          gradient (first cotangent-results)
          primal (:primal-result interface)
          seed (:cotangent-input interface)]
      ;; Select the retained canonical descriptors once. Families below restrict
      ;; execution capabilities; they do not reconstruct derivative associations.
      (when-not (and active
                     (= active-id (:source tangent) (:source gradient))
                     (= (:id primal) (:source seed)))
        (fail! :normalized-device-interface
               "device derivative ports must retain their canonical source associations"
               {:family family :active active-id :tangent-source (:source tangent)
                :gradient-source (:source gradient) :primal (:id primal)
                :seed-source (:source seed)}))
      (doseq [[relation left right]
              [[:seed (:value primal) (:value seed)]
               [:gradient (:value active) (:value gradient)]]]
        (when-not (and (av/storage-contract-compatible? left right)
                       (= (:shape left) (:shape right)))
          (fail! :normalized-device-interface-contract
                 "canonical derivative ports must preserve exact symbolic storage geometry"
                 {:relation relation :left left :right right})))
      {:inputs inputs :active-id active-id :active active :tangent tangent
       :primal primal :seed seed :gradient gradient})))

(defn- static-positive-vector? [value]
  (and (= 1 (count (:shape value)))
       (integer? (first (:shape value)))
       (pos? (first (:shape value)))
       (<= (first (:shape value)) Integer/MAX_VALUE)))

(defn- map-interface! [owner]
  (let [interface (:interface owner)
        {:keys [inputs active-id active tangent primal seed gradient]}
        (singleton-derivative-ports! interface :map)
        arrays (filterv #(seq (get-in % [:value :shape])) inputs)]
    (when-not (and (seq arrays) (some #{active} arrays)
                   (every? #{'map} (map soac/operation-kind (soac/equations (:source owner))))
                   (or (= 1 (count arrays))
                       (every? (comp static-positive-vector? :value)
                               (concat arrays [primal seed gradient]))))
      (fail! :normalized-device-interface
             "device preparation requires pure maps, one active array and static multi-input geometry"
             {:inputs (mapv :id inputs) :active active-id
              :primal (:id primal) :seed-source (:source seed)
              :gradient-source (:source gradient)}))
    {:inputs inputs :arrays arrays :active active :tangent tangent
     :primal primal :seed seed :gradient gradient}))

(defn- sum-interface! [owner]
  (let [kinds (mapv soac/operation-kind (soac/equations (:source owner)))
        _ (when-not (= '[map reduce] kinds)
            (fail! :normalized-device-interface
                   "resident sum preparation currently requires the verified single-map reduction slice"
                   {:operations kinds}))
        interface (:interface owner)
        {:keys [inputs active-id active tangent primal seed gradient]}
        (singleton-derivative-ports! interface :map-sum)
        active-ids (vec (:active-inputs interface))
        arrays (filterv #(seq (get-in % [:value :shape])) inputs)
        scalars (filterv #(empty? (get-in % [:value :shape])) inputs)]
    (when-not (and (= 1 (count active-ids)) active
                   (some #(= active-id (:id %)) arrays)
                   (= (count inputs) (+ (count arrays) (count scalars)))
                   (empty? (get-in primal [:value :shape]))
                   (empty? (get-in seed [:value :shape])))
      (fail! :normalized-device-interface
             "map-sum preparation requires one active array and exact canonical scalar ports"
             {:inputs (mapv :id inputs) :arrays (mapv :id arrays)
              :active active-ids :primal (:id primal) :seed (:id seed)
              :gradient-source (:source gradient)}))
    {:inputs inputs :arrays arrays :scalars scalars :active active :tangent tangent
     :primal primal :seed seed :gradient gradient}))

(defn- indexed-interface! [owner residual-plan]
  (let [interface (:interface owner)
        {:keys [inputs active-id active tangent primal seed gradient]}
        (singleton-derivative-ports! interface :indexed-map)
        arrays (filterv #(seq (get-in % [:value :shape])) inputs)
        residuals (vals (:owned-arrays residual-plan))
        physicals (:physical-bindings residual-plan)
        [physical-id physical] (first physicals)
        index-id (get-in (first residuals) [:origin :input])
        index (some #(when (= index-id (:id %)) %) inputs)
        index-snapshot (:id (first residuals))
        zero-requirement (:requirement physical)
        zero-value (:value physical)
        consumer (:consumer physical)
        gradient-id (:id gradient)
        index-values (map :value [index primal seed])
        dynamic-values (map :value [active gradient])
        dynamic-values (conj (vec dynamic-values) zero-value)
        dynamic-shape (:shape (first dynamic-values))
        dimension (when (and (= 1 (count dynamic-shape))
                             (symbol? (first dynamic-shape)))
                    (first dynamic-shape))
        dimension-value (when dimension
                          (get-in (soac/facts (:transpose owner)) [:values dimension]))
        dynamic-geometry?
        (and dimension
             (every? #(= [dimension] (:shape %)) dynamic-values)
             (= :tensor (:kind dimension-value))
             (empty? (:shape dimension-value))
             (contains? #{:int :long} (dtype/canon (:dtype dimension-value)))
             (not (some #{dimension} (map :id inputs))))
        static-geometry? (every? static-positive-vector? dynamic-values)]
    (when-not
     (and (= 2 (count inputs)) (= 2 (count arrays)) (empty? (remove #(seq (get-in % [:value :shape])) inputs))
          (= 1 (count residuals)) (= 1 (count physicals)) active index (not= active index)
          (= #{index-id active-id} (set (map :id inputs)))
          (contains? #{:int :long} (get-in index [:value :dtype]))
          (contains? #{:float :double} (get-in active [:value :dtype]))
          (contains? #{nil :owned} (get-in index [:value :ownership]))
          (contains? #{nil :owned} (get-in active [:value :ownership]))
          (every? static-positive-vector? index-values)
          (or static-geometry? dynamic-geometry?)
          (= (:shape (get-in index [:value])) (:shape (get-in primal [:value])))
          (= :borrowed (get-in seed [:value :ownership]))
          (= :owned (get-in gradient [:value :ownership]))
          (= :owned (:ownership zero-value))
          (= index-snapshot (get-in (first residuals) [:producer :result]))
          (= zero-value (get-in (soac/facts (:transpose owner)) [:values physical-id]))
          (= {:value physical-id :dtype (:dtype zero-value) :shape (:shape zero-value)
              :ownership :owned :initialization :typed-zero :exclusive true}
             zero-requirement)
          (= {:equation (get-in consumer [:equation]) :destination physical-id
              :result gradient-id :alias physical-id}
             consumer)
          (= physical-id (get-in (soac/facts (:transpose owner))
                                 [:equations (:equation consumer) :aliases gradient-id]))
          (= (:dtype zero-value) (get-in active [:value :dtype])
             (get-in gradient [:value :dtype]))
          (= (:shape zero-value) (get-in active [:value :shape])
             (get-in gradient [:value :shape])))
      (fail! :normalized-device-indexed-interface
             "indexed preparation requires a static index space and one exact gradient geometry"
             {:inputs (mapv :id inputs) :active active-id :index index-id
              :residuals (mapv :id residuals) :physical-bindings (set (keys physicals))}))
    {:inputs inputs :arrays arrays :active active :tangent tangent :index index
     :index-snapshot index-snapshot :primal primal :seed seed :gradient gradient
     :zero-id physical-id :zero-value zero-value :zero-requirement zero-requirement
     :external-dimension (when dynamic-geometry?
                           {:id dimension :dtype (dtype/canon (:dtype dimension-value))})}))

(defn- materialize-layout-with! [compilation input-descriptors arguments caller-options admit-steps!]
  (let [invocation-plan (get-in compilation [:semantic :attributes :invocation-plan])
        arguments (vec arguments)
        semantic-ids (mapv :id input-descriptors)
        invocation-ids (mapv :symbol (:parameters invocation-plan))
        _ (when-not (and (= (count semantic-ids) (count (set semantic-ids)))
                         (= (count invocation-ids) (count (set invocation-ids)))
                         (= (count semantic-ids) (count arguments))
                         (= (set semantic-ids) (set invocation-ids)))
            (fail! :normalized-device-arguments
                   "canonical and compiled public input identities must correspond exactly"
                   {:semantic semantic-ids :invocation invocation-ids
                    :arguments (count arguments)}))
        argument-by-id (zipmap semantic-ids arguments)
        invocation-arguments (mapv argument-by-id invocation-ids)
        _ (admit-steps! invocation-plan)
        materialized
        (materialization/materialize
         invocation-plan invocation-arguments
         (fn [& _]
           (fail! :normalized-device-specialization
                  "device preparation cannot evaluate a host scalar prefix" {})))]
    (invocation-link/project-resident-layout
     materialized (:emitted compilation) (:target compilation) caller-options)))

(defn- materialize-layout! [compilation input-descriptors arguments caller-options]
  (materialize-layout-with!
   compilation input-descriptors arguments caller-options
   (fn [plan]
     (when (seq (:steps plan))
       (fail! :normalized-device-specialization
              "the first device consumer accepts only direct array/scalar specialization"
              {:steps (mapv :id (:steps plan))})))))

(defn- materialize-indexed-layout!
  [compilation input-descriptors arguments caller-options {:keys [external-dimension active]}]
  (materialize-layout-with!
   compilation input-descriptors arguments caller-options
   (fn [plan]
     (let [steps (:steps plan)
           marker (get-in plan [:attributes :source-free-shape-projections])
           expected (when external-dimension
                      [{:dimension (:id external-dimension) :source (:id active) :axis 0}])]
       (when-not (if external-dimension
                   (and (= expected marker) (= 1 (count steps))
                        (invocation/shape-projection? (first steps))
                        (= (:id external-dimension) (:symbol (first steps))))
                   (and (empty? steps) (nil? marker)))
         (fail! :normalized-device-indexed-shape-projection
                "indexed preparation requires exactly its canonical self-shape projection"
                {:expected expected :marker marker
                 :steps (mapv :id steps)}))))))

(defn- public-array-boundary!
  "Match canonical array descriptors to a projected resident layout by compiler identity.

   Canonical source order and the generated invocation ABI may differ.  This boundary preserves
   canonical order for the public device API while proving an exact bijection to the layout's
   ordered public inputs and compiler bindings."
  [descriptors layout]
  (let [canonical-ids (mapv :id descriptors)
        public-inputs (:public-inputs layout)
        public-ids (mapv :symbol public-inputs)
        compiler-bindings (:buffer-bindings layout)
        canonical-values (mapv #(get compiler-bindings %) canonical-ids)]
    (when-not
     (and (vector? descriptors)
          (every? #(and (map? %) (symbol? (:id %))) descriptors)
          (= (count canonical-ids) (count (distinct canonical-ids)))
          (vector? public-inputs)
          (map? compiler-bindings)
          (every? #(and (map? %) (symbol? (:symbol %)) (some? (:value %))) public-inputs)
          (= (count public-ids) (count (distinct public-ids)))
          (= (set canonical-ids) (set public-ids))
          (= (count canonical-values) (count (distinct canonical-values)))
          (every? some? canonical-values)
          (every? (fn [{:keys [symbol value]}]
                    (= value (get compiler-bindings symbol)))
                  public-inputs))
      (fail! :normalized-device-array-boundary
             "canonical arrays must correspond bijectively to resident public inputs"
             {:canonical canonical-ids :public-inputs public-inputs
              :canonical-values canonical-values}))
    {:ids canonical-ids :values canonical-values}))

(defn- compile-device-phases!
  "Compile a canonical primal/transpose pair; callers retain family-specific phase policy."
  [checked options resident-outputs]
  (let [target (:target options)
        compile-options (select-keys options [:target :schedule :scalar-math])
        caller-options (caller-options options)
        primal-options (cond-> compile-options
                         (seq resident-outputs) (assoc :resident-outputs resident-outputs))
        primal-compilation (compile-phase! checked :primal primal-options)
        transpose-compilation (compile-phase! checked :transpose compile-options)]
    {:target target
     :compile-options compile-options
     :caller-options caller-options
     :primal-compilation primal-compilation
     :transpose-compilation transpose-compilation}))

(defn- fresh-id [used base]
  (first (remove used (map #(vector ::resident base %) (range)))))

(defn- fresh-layout-ids [layout bases]
  (loop [used (into (set (keys (:nodes layout)))
                    (concat (keys (:values layout))
                            (map #(get-in % [:view :allocation :id])
                                 (vals (:nodes layout)))))
         bases bases
         result []]
    (if-let [base (first bases)]
      (let [id (fresh-id used base)]
        (recur (conj used id) (next bases) (conj result id)))
      result)))

(defn- clone-seed-layout! [layout interface primal-value]
  (let [[allocation-id node-id value-id]
        (fresh-layout-ids layout [[:seed :allocation] [:seed :node] [:seed :value]])
        seed-id (get-in interface [:seed :id])
        cloned
        (invocation-link/clone-resident-value-layout
         layout {:template-value primal-value
                 :template-compiler-value (get-in interface [:primal :id])
                 :template-abstract (get-in interface [:primal :value])
                 :compiler-value seed-id :abstract (get-in interface [:seed :value])
                 :value-id value-id :node-id node-id :allocation-id allocation-id
                 :role :input :project-ownership? true})]
    {:layout (-> layout
                 (assoc-in [:nodes node-id] (:node cloned))
                 (assoc-in [:values value-id] (:value cloned)))
     :value value-id}))

(defn- clone-gradient-layout! [layout interface active-value]
  (let [[allocation-id node-id value-id]
        (fresh-layout-ids layout [[:gradient :allocation] [:gradient :node]
                                  [:gradient :value]])
        gradient-id (get-in interface [:gradient :id])
        cloned
        (invocation-link/clone-resident-value-layout
         layout {:template-value active-value
                 :template-compiler-value (get-in interface [:active :id])
                 :template-abstract (get-in interface [:active :value])
                 :compiler-value gradient-id :abstract (get-in interface [:gradient :value])
                 :value-id value-id :node-id node-id :allocation-id allocation-id
                 :role :output :project-ownership? false})]
    {:layout (-> layout
                 (assoc-in [:nodes node-id] (:node cloned))
                 (assoc-in [:values value-id] (:value cloned)))
     :value value-id}))

(defn- close-preserving! [primary closeable]
  (try
    (.close ^java.io.Closeable closeable)
    (catch Throwable cleanup-error
      (when-not (identical? primary cleanup-error)
        (.addSuppressed ^Throwable primary cleanup-error))))
  (throw primary))

(defn- guarded-run! [guard! invoke]
  (guard!)
  (let [lease (invoke)]
    (try
      (guard!)
      lease
      (catch Throwable primary (close-preserving! primary lease)))))

(defn- publish-result! [lease build validate!]
  (try
    (let [result (build)]
      (validate! result)
      result)
    (catch Throwable primary
      (close-preserving! primary lease))))

(defn- ordered-residual-frontiers! [owner residual-plan compiler-bindings]
  (let [residual-bindings (vals (:owned-arrays residual-plan))
        residual-by-id (into {} (map (juxt :id identity)) residual-bindings)
        residual-ids (into [] (filter #(contains? residual-by-id %))
                           (soac/outputs (:primal owner)))]
    (when-not (= (set residual-ids) (set (keys residual-by-id)))
      (fail! :normalized-device-residual-order
             "canonical primal outputs must order every owned residual exactly once"
             {:outputs (soac/outputs (:primal owner))
              :residuals (set (keys residual-by-id))}))
    (doseq [residual-id residual-ids
            :let [binding (get residual-by-id residual-id)]]
      (when-not (and (= residual-id (get-in binding [:producer :result]))
                     (= (:value binding)
                        (get-in (soac/facts (:primal owner)) [:values residual-id]))
                     (contains? compiler-bindings residual-id))
        (fail! :normalized-device-residual-correspondence
               "resident residual differs from its exact canonical primal producer"
               {:residual residual-id :binding binding})))
    {:ids residual-ids :values (mapv compiler-bindings residual-ids)}))

(defn- typed-zero-program
  [owner {:keys [zero-id zero-value zero-requirement]} concrete-shape]
  (let [used (into (set (keys (:values (soac/facts (:transpose owner)))))
                   (map second (soac/equations (:transpose owner))))
        fresh (fn [prefix]
                (first (remove used (map #(symbol (str prefix %)) (range)))))
        equation-id (fresh "normalized-device-zero-fill-")
        index (fresh "normalized-device-zero-index-")
        dtype (:dtype zero-value)
        realized-zero (assoc zero-value :shape concrete-shape)
        extent (first concrete-shape)
        cast (dtype/scalar-tag-for-dtype dtype)
        equation (list '= equation-id [zero-id]
                       (list 'map {:index index :extent extent} [] []
                             (soac/lambda-form [] [] [(list cast 0.0)])))
        program
        (soac/make
         (soac/default-program-facts
          {:values {zero-id realized-zero}
           :inputs [] :equations {equation-id (soac/default-equation-facts)}})
         [equation] [zero-id])]
    (when-not (and (= :typed-zero (:initialization zero-requirement))
                   (= :owned (:ownership zero-requirement))
                   (true? (:exclusive zero-requirement))
                   (= zero-id (:value zero-requirement))
                   (= (dissoc zero-value :shape) (dissoc realized-zero :shape))
                   (= concrete-shape (:shape realized-zero))
                   (= realized-zero (get-in (soac/facts program) [:values zero-id]))
                   (= 1 (count concrete-shape))
                   (integer? extent) (pos? extent) (<= extent Integer/MAX_VALUE)
                   (empty? (:inputs (soac/facts program))))
      (fail! :normalized-device-zero-fill
             "zero fill must derive exactly from the canonical physical obligation"
             {:zero zero-id :requirement zero-requirement}))
    program))

(defn- resident-value-shape!
  [nodes values value-id]
  (let [value (get values value-id)
        leaves (:leaves value)
        node-id (get-in leaves [0 :node])
        node (get nodes node-id)
        shape (get-in node [:view :shape])]
    (when-not (and value (= 1 (count leaves)) node
                   (= 1 (count shape)) (integer? (first shape))
                   (pos? (first shape)) (<= (first shape) Integer/MAX_VALUE))
      (fail! :normalized-device-indexed-geometry
             "indexed execution requires one positive concrete resident vector"
             {:value value-id :leaves leaves :node node-id :shape shape}))
    shape))

(defn- bind-module!
  [check! target caller-options nodes values primal-phase transpose-phases residual-values]
  (let [bind (fn [id {:keys [compilation buffer-bindings scalar-values]}]
               (let [request {:id id :target target :nodes nodes :values values
                              :buffer-bindings buffer-bindings :scalar-values scalar-values}]
                 (if caller-options
                   (invocation-link/bind-resident-program
                    (:emitted compilation) request caller-options)
                   (invocation-link/bind-resident-program (:emitted compilation) request))))
        primal-instance (bind ::primal primal-phase)
        transpose-instances
        (mapv (fn [ordinal phase]
                (bind (or (:instance-id phase) [::transpose ordinal]) phase))
              (range) transpose-phases)
        transpose-ids (mapv :id transpose-instances)
        spine-request {:id ::spine :target target :nodes nodes :values values
                       :instances (into [primal-instance] transpose-instances) :outputs []}
        spine (if caller-options
                (link-plan/make spine-request caller-options)
                (link-plan/make spine-request))
        module-request
        {:id ::module :spine spine
         :entries [{:name :primal :instances [::primal] :interface (:interface primal-phase)}
                   {:name :transpose :instances transpose-ids
                    :interface (:interface (last transpose-phases))}]}
        module (if caller-options
                 (module-plan/make module-request caller-options)
                 (module-plan/make module-request))
        frontier-values (mapv :value (filterv #(= :transpose (:consumer %))
                                              (:frontiers module)))]
    ;; The module orders frontiers by first use, independently of canonical primal outputs.
    ;; Dependencies are value-ID keyed: prove a bijection, then preserve canonical publication
    ;; order without rewriting the module's independently checked producer/lifetime evidence.
    (when-not (and (= (count residual-values) (count (distinct residual-values)))
                   (= (count frontier-values) (count (distinct frontier-values)))
                   (= (count residual-values) (count frontier-values))
                   (= (set residual-values) (set frontier-values)))
      (fail! :normalized-device-frontiers
             "module frontiers must correspond bijectively to canonical residual values"
             {:expected residual-values :actual frontier-values}))
    (check!)
    {:module module :frontier-values residual-values}))

(defn- functional-seed-binding!
  "Project the admitted seed transport without reconstructing derivative associations.
   Array seeds own fresh storage; fixed scalar seeds close from the canonical phase inputs."
  [owner interface layout primal-value {:keys [transport value]}]
  (let [seed-id (get-in interface [:seed :id])]
    (case transport
      :array
      (let [cloned (clone-seed-layout! layout interface primal-value)
            seed-value (:value cloned)]
        {:layout (:layout cloned) :buffers {seed-id seed-value}
         :scalars (:scalar-values layout) :inputs [seed-value]
         :ports {:seed-input seed-value}})

      :scalar
      (let [typed-seed (materialization/checked-scalar-value seed-id (get-in interface [:seed :value]) value)
            facts (soac/facts (:transpose owner))
            scalar-ids (filterv #(empty? (get-in facts [:values % :shape])) (:inputs facts))
            scalars (into {} (map (fn [id]
                                   [id (if (= id seed-id) typed-seed (get (:scalar-values layout) id))]))
                          scalar-ids)]
        (when-not (and (some #{seed-id} scalar-ids) (every? some? (vals scalars)))
          (fail! :normalized-device-specialization
                 "map-sum transpose scalars must close from fixed primal specialization and seed"
                 {:seed seed-id :inputs (:inputs facts)
                  :scalars (set (keys scalars)) :available (set (keys (:scalar-values layout)))}))
        {:layout layout :buffers {} :scalars scalars :inputs []
         :ports {:fixed-seed {:id seed-id :value typed-seed}}}))))

(defn- assemble-functional!
  "Bind admitted pure canonical phases through one layout/frontier/module construction.
   Callers retain admission and public compatibility projections. Seed transport determines
   only physical scalar/array binding, never derivative rules or operation admission."
  [{:keys [owner check!] :as checked} interface residual-plan arguments options seed-request]
  (let [scalar-seed? (case (:transport seed-request) :array false :scalar true)
        active-id (get-in interface [:active :id])
        primal-id (get-in interface [:primal :id])
        gradient-id (get-in interface [:gradient :id])
        {:keys [target caller-options primal-compilation transpose-compilation]}
        (compile-device-phases! checked options (when scalar-seed? [primal-id]))
        layout (materialize-layout! primal-compilation (:inputs interface) arguments caller-options)
        compiler-bindings (:buffer-bindings layout)
        array-boundary (public-array-boundary! (:arrays interface) layout)
        active-value (get compiler-bindings active-id)
        primal-value (get compiler-bindings primal-id)
        _ (if scalar-seed?
            (when-not (and active-value primal-value
                           (= :resident-scalar-buffer
                              (get-in layout [:values primal-value :abstract :representation :kind]))
                           (= [] (get-in layout [:values primal-value :abstract :shape])))
              (fail! :normalized-device-layout-interface
                     "map-sum resident layout must expose every canonical array and scalar loss"
                     {:expected-arrays (:ids array-boundary) :public-inputs (:public-inputs layout)
                      :active active-id :active-value active-value :primal primal-id :primal-value primal-value
                      :primal-abstract (get-in layout [:values primal-value :abstract])}))
            (when-not (and active-value primal-value
                           (some #{active-id} (:ids array-boundary))
                           (some #{active-value} (:values array-boundary)))
              (fail! :normalized-device-layout-interface
                     "resident layout must expose the canonical arrays and selected active input"
                     {:active active-id :active-value active-value
                      :primal primal-id :primal-value primal-value
                      :public-inputs (:public-inputs layout)})))
        residuals (ordered-residual-frontiers! owner residual-plan compiler-bindings)
        residual-ids (:ids residuals)
        seed (functional-seed-binding! owner interface layout primal-value seed-request)
        gradient-clone (clone-gradient-layout! (:layout seed) interface active-value)
        gradient-value (:value gradient-clone)
        nodes (get-in gradient-clone [:layout :nodes])
        values (get-in gradient-clone [:layout :values])
        residual-values (:values residuals)
        transpose-bindings
        (into (assoc (:buffers seed) gradient-id gradient-value)
              (map (fn [id] [id (get compiler-bindings id)])) residual-ids)
        transpose-request {:id ::transpose :target target :nodes nodes :values values
                           :buffer-bindings transpose-bindings :scalar-values (:scalars seed)}
        transpose-layout
        (if caller-options
          (invocation-link/complete-resident-buffer-layout
           (:emitted transpose-compilation) transpose-request caller-options)
          (invocation-link/complete-resident-buffer-layout
           (:emitted transpose-compilation) transpose-request))
        bound (bind-module!
               check! target caller-options (:nodes transpose-layout) (:values transpose-layout)
               {:compilation primal-compilation :buffer-bindings compiler-bindings
                :scalar-values (:scalar-values layout)
                :interface {:inputs (:values array-boundary) :state [] :outputs [primal-value]}}
               [{:instance-id ::transpose
                 :compilation transpose-compilation :buffer-bindings (:buffer-bindings transpose-layout)
                 :scalar-values (:scalars seed)
                 :interface {:inputs (:inputs seed) :state [] :outputs [gradient-value]}}]
               residual-values)]
    {:module (:module bound) :frontier-values (:frontier-values bound)
     :ports (merge {:inputs (mapv (fn [id value] {:id id :value value})
                                 (:ids array-boundary) (:values array-boundary))
                    :primal-output primal-value :gradient-output gradient-value
                    :residuals (:frontier-values bound)}
                   (:ports seed))}))

(defn- assemble-map!
  [{:keys [owner] :as checked} arguments options]
  (let [interface (map-interface! owner)
        residual-plan (canonical/execution-residual-plan owner)]
    (when (or (seq (:immutable-scalars residual-plan)) (seq (:physical-bindings residual-plan)))
      (fail! :normalized-device-residuals
             "device preparation currently requires array-only residual storage"
             {:immutable-scalars (set (keys (:immutable-scalars residual-plan)))
              :physical-bindings (set (keys (:physical-bindings residual-plan)))}))
    (assemble-functional! checked interface residual-plan arguments options {:transport :array})))

(defn- assemble-sum!
  [{:keys [owner] :as checked} arguments fixed-seed options]
  (let [interface (sum-interface! owner)
        residual-plan (canonical/execution-residual-plan owner)
        _ (when (or (seq (:immutable-scalars residual-plan))
                    (seq (:physical-bindings residual-plan)))
            (fail! :normalized-device-residuals
                   "the first map-sum device slice requires array-only residual storage"
                   {:immutable-scalars (set (keys (:immutable-scalars residual-plan)))
                    :physical-bindings (set (keys (:physical-bindings residual-plan)))}))]
    (assemble-functional! checked interface residual-plan arguments options
                          {:transport :scalar :value fixed-seed})))

(defn- assemble-indexed!
  [{:keys [owner check!] :as checked} arguments options]
  (let [residual-plan (canonical/execution-residual-plan owner)
        interface (indexed-interface! owner residual-plan)
        input-values (into {} (map (juxt :id :value)) (:inputs interface))
        {:keys [target compile-options caller-options
                primal-compilation transpose-compilation]}
        (compile-device-phases! checked options nil)
        layout (materialize-indexed-layout!
                primal-compilation (:inputs interface) arguments caller-options interface)
        compiler-bindings (:buffer-bindings layout)
        array-boundary (public-array-boundary! (:arrays interface) layout)
        canonical-array-ids (:ids array-boundary)
        public-array-values (:values array-boundary)
        index-id (get-in interface [:index :id])
        active-id (get-in interface [:active :id])
        primal-id (get-in interface [:primal :id])
        seed-id (get-in interface [:seed :id])
        gradient-id (get-in interface [:gradient :id])
        zero-id (:zero-id interface)
        index-value (get compiler-bindings index-id)
        active-value (get compiler-bindings active-id)
        primal-value (get compiler-bindings primal-id)
        _ (when-not (and (= #{index-id active-id} (set canonical-array-ids))
                         (every? some? [index-value active-value primal-value]))
            (fail! :normalized-device-layout-interface
                   "indexed resident layout must expose exact index and active arrays"
                   {:expected canonical-array-ids :public-inputs (:public-inputs layout)
                    :primal primal-id :primal-value primal-value}))
        residuals (ordered-residual-frontiers! owner residual-plan compiler-bindings)
        residual-ids (:ids residuals)
        residual-values (:values residuals)
        _ (when-not (= [(:index-snapshot interface)] residual-ids)
            (fail! :normalized-device-indexed-residual
                   "indexed execution requires the exact canonical index snapshot"
                   {:expected (:index-snapshot interface) :actual residual-ids}))
        seed-clone (clone-seed-layout! layout interface primal-value)
        seed-value (:value seed-clone)
        gradient-clone (clone-gradient-layout! (:layout seed-clone) interface active-value)
        gradient-value (:value gradient-clone)
        nodes (get-in gradient-clone [:layout :nodes])
        values (get-in gradient-clone [:layout :values])
        index-shape (resident-value-shape! nodes values index-value)
        active-shape (resident-value-shape! nodes values active-value)
        primal-shape (resident-value-shape! nodes values primal-value)
        seed-shape (resident-value-shape! nodes values seed-value)
        gradient-shape (resident-value-shape! nodes values gradient-value)
        external-dimension (:external-dimension interface)
        transpose-plan (get-in transpose-compilation
                               [:semantic :attributes :invocation-plan])
        declared-dimensions (get-in transpose-plan
                                    [:attributes :external-dimension-values])
        typed-dimension (when external-dimension
                          (get (:scalar-values layout) (:id external-dimension)))
        transpose-scalars
        (if external-dimension {(:id external-dimension) typed-dimension} {})
        _ (when-not
           (and (= index-shape primal-shape seed-shape)
                (= active-shape gradient-shape)
                (if external-dimension
                  (and (= [external-dimension] declared-dimensions)
                       (= #{(:id external-dimension)}
                          (set (keys (:scalar-values layout))))
                       (= {:type (:dtype external-dimension)
                           :value (first gradient-shape)}
                          typed-dimension)
                       (= [(:id external-dimension)] (:shape (:zero-value interface)))
                       (= [(:id external-dimension)]
                          (get-in interface [:gradient :value :shape]))
                       (= [(:id external-dimension)]
                          (get-in interface [:active :value :shape])))
                  (and (nil? declared-dimensions)
                       (empty? (:scalar-values layout))
                       (= active-shape (:shape (:zero-value interface))))))
          (fail! :normalized-device-indexed-geometry
                 "indexed symbolic and resident geometry must correspond exactly"
                 {:external-dimension external-dimension
                  :declared-dimensions declared-dimensions
                  :typed-dimension typed-dimension
                  :index-shape index-shape :primal-shape primal-shape
                  :seed-shape seed-shape :active-shape active-shape
                  :gradient-shape gradient-shape}))
        zero-program (typed-zero-program owner interface gradient-shape)
        zero-compilation (compile-program! checked zero-program compile-options)
        zero-bindings {zero-id gradient-value}
        transpose-bindings
        (into {seed-id seed-value zero-id gradient-value gradient-id gradient-value}
              (map (fn [id] [id (get compiler-bindings id)])) residual-ids)
        bound
        (bind-module!
         check! target caller-options nodes values
         {:compilation primal-compilation :buffer-bindings compiler-bindings
          :scalar-values (:scalar-values layout)
          :interface {:inputs public-array-values :state [] :outputs [primal-value]}}
         [{:instance-id ::zero-fill :compilation zero-compilation
           :buffer-bindings zero-bindings :scalar-values {}
           :interface {:inputs [] :state [] :outputs [gradient-value]}}
          {:instance-id ::transpose-effect :compilation transpose-compilation
           :buffer-bindings transpose-bindings :scalar-values transpose-scalars
           :interface {:inputs [seed-value] :state [] :outputs [gradient-value]}}]
         residual-values)
        index-count (first index-shape)
        gradient-count (first gradient-shape)]
    {:module (:module bound) :frontier-values (:frontier-values bound)
     :ports {:inputs (mapv (fn [id value]
                             {:id id :value value
                              :dtype (:dtype (get input-values id))
                              :elements (first (resident-value-shape!
                                                nodes values value))})
                           canonical-array-ids public-array-values)
             :index-input index-value :index-id index-id
             :index-dtype (get-in interface [:index :value :dtype])
             :index-count index-count :gradient-count gradient-count
             :primal-output primal-value :seed-input seed-value
             :seed-dtype (get-in interface [:seed :value :dtype])
             :seed-elements (first seed-shape)
             :gradient-output gradient-value :residuals residual-values
             :zero-destination zero-id :logical-gradient gradient-id}}))

(defn- run-result!
  [guard! runtime entry request output-id frontier-values publish-dependencies?]
  (let [lease (guarded-run! guard! #(link-module/run-and-lease! runtime entry request))]
    (publish-result!
     lease
     #(cond-> {:lease lease :output-id output-id}
        publish-dependencies? (assoc :dependencies
                                     (zipmap frontier-values (repeat lease))))
     (fn [result]
       (let [expected-keys (cond-> #{:lease :output-id}
                             publish-dependencies? (conj :dependencies))]
         (when-not
          (and (= expected-keys (set (keys result)))
               (identical? lease (:lease result))
               (= output-id (:output-id result))
               (or (not publish-dependencies?)
                   (and (= (set frontier-values) (set (keys (:dependencies result))))
                        (= (count frontier-values) (count (:dependencies result)))
                        (every? #(identical? lease %) (vals (:dependencies result))))))
          (fail! :normalized-device-result
                 "device result must publish the exact issued lease, output and frontiers"
                 {:entry entry :output (:output-id result)
                  :dependencies (some-> result :dependencies keys set)})))))))

(defn- checked-primitive-array
  [id expected-dtype expected-elements source copy?]
  (let [actual-dtype (dtype/dtype-for-jvm-array source)
        actual-elements (when actual-dtype (java.lang.reflect.Array/getLength source))]
    (when-not (and (= expected-dtype actual-dtype)
                   (= expected-elements actual-elements))
      (fail! :normalized-device-array
             "device entry inputs require exact primitive dtype and capacity"
             {:input id :expected-dtype expected-dtype :actual-dtype actual-dtype
              :expected-elements expected-elements :actual-elements actual-elements}))
    (if copy? (aclone source) source)))

(defn- checked-indexed-inputs!
  [ports inputs]
  (let [input-ports (:inputs ports)
        expected (mapv :id input-ports)]
    (when-not (and (map? inputs) (= (set expected) (set (keys inputs)))
                   (= (count expected) (count inputs)))
      (fail! :normalized-device-inputs
             "indexed primal inputs must exactly cover canonical array IDs"
             {:expected expected :actual (when (map? inputs) (set (keys inputs)))}))
    (let [checked
          (into {}
                (map (fn [{:keys [id value dtype elements]}]
                       [value (checked-primitive-array
                               id dtype elements (get inputs id)
                               (= value (:index-input ports)))]))
                input-ports)
          indices (get checked (:index-input ports))
          limit (long (:gradient-count ports))]
      (dotimes [position (:index-count ports)]
        (let [index (long (case (:index-dtype ports)
                            :int (aget ^ints indices position)
                            :long (aget ^longs indices position)))]
          (when-not (< -1 index limit)
            (fail! :normalized-device-index-bounds
                   "indexed primal indices must be in the active array's exact extent"
                   {:input (:index-id ports) :position position
                    :index index :extent limit}))))
      checked)))

(defn- canonical-array-request!
  "Project only canonical public array IDs. Prepared scalar arguments cannot be overridden."
  [ports inputs]
  (let [input-ports (:inputs ports)
        expected (mapv :id input-ports)]
    (when-not (and (map? inputs) (= (set expected) (set (keys inputs)))
                   (= (count expected) (count inputs)))
      (fail! :normalized-device-inputs
             "primal inputs must exactly cover canonical array IDs"
             {:expected expected :actual (when (map? inputs) (set (keys inputs)))}))
    {:inputs (into {} (map (fn [{:keys [id value]}] [value (get inputs id)])) input-ports)}))

(defn- instantiate-prepared!
  [{:keys [check! guard!]} {:keys [module frontier-values ports]} options family]
  (let [runtime-options (select-keys options [:scalar-math :profile?])
        runtime (link-module/instantiate! module runtime-options)]
    (try
      (check!)
      (let [pure-primal! (fn [inputs]
                          (run-result! guard! runtime :primal
                                       (canonical-array-request! ports inputs)
                                       (:primal-output ports) frontier-values true))
            [primal! transpose!]
            (case family
              :map
              [pure-primal!
               (fn [seed dependencies]
                 (run-result! guard! runtime :transpose
                              {:inputs {(:seed-input ports) seed}
                               :dependencies dependencies}
                              (:gradient-output ports) frontier-values false))]

              :map-sum
              [pure-primal!
               (fn [dependencies]
                 (run-result! guard! runtime :transpose {:dependencies dependencies}
                              (:gradient-output ports) frontier-values false))]

              :indexed-map
              [(fn [inputs]
                 (guard!)
                 (let [checked (checked-indexed-inputs! ports inputs)]
                   (guard!)
                   (run-result! guard! runtime :primal {:inputs checked}
                                (:primal-output ports) frontier-values true)))
               (fn [seed dependencies]
                 (guard!)
                 (let [seed (checked-primitive-array
                             (:seed-input ports) (:seed-dtype ports)
                             (:seed-elements ports) seed false)]
                   (guard!)
                   (run-result! guard! runtime :transpose
                                {:inputs {(:seed-input ports) seed}
                                 :dependencies dependencies}
                                (:gradient-output ports) frontier-values false)))]

              (fail! :normalized-device-family "unknown prepared device family"
                     {:family family}))
            prepared {:ports ports :primal! primal! :transpose! transpose!
                      :close! #(.close ^java.io.Closeable runtime)}]
        (when-not (= prepared-keys (set (keys prepared)))
          (fail! :normalized-device-prepared "invalid prepared owner projection" {}))
        prepared)
      (catch Throwable primary (close-preserving! primary runtime)))))

(defn prepare-owner
  "Prepare one canonical checked-map owner for explicit resident primal/transpose execution.

   This is an opt-in lease API, not a CPU-compatible function. `arguments` fix the canonical
   public scalar values and array geometry for the lifetime of the returned owner. `:primal!`
   takes an exact map from canonical array IDs to primitive buffers; :ports exposes ordered
   :inputs descriptors. `:transpose!` takes an array seed and the primal dependency map.
   One derivative input is active; multiple public arrays require positive static vector
   geometry and an admitted pure-map source. Outputs remain resident behind opaque leases.
   Callers must close every lease before invoking `:close!`.

   No default AD engine, cache, host readback, mutable state, map-sum or indexed-map path is
   activated by this boundary."
  [owner arguments {:keys [target] :as options}]
   (when-not (and (map? options) (set/subset? (set (keys options)) option-keys))
     (fail! :normalized-device-options
            "normalized device preparation accepts only target, schedule, scalar-math and profile"
            {:keys (when (map? options) (set (keys options)))}))
   (when-not (keyword? target)
     (fail! :normalized-device-target
            "normalized device preparation requires one explicit target keyword"
            {:target target}))
   (let [checked (authority owner :checked-scalar-map)
         assembled (assemble-map! checked arguments options)]
     (instantiate-prepared! checked assembled options :map)))

(defn prepare-sum-owner
  "Prepare one canonical map→ordered-sum owner with a fixed typed cotangent seed.

   `arguments` specialize all canonical arrays/scalars and are discarded after resident layout
   projection. `:primal!` accepts an exact map from every canonical array ID to a primitive host
   buffer. `:transpose!` accepts only the dependency map returned by `:primal!`; the seed is fixed
   for this prepared owner and is never a per-run scalar override. Results are opaque resident
   leases and every lease must be closed before `:close!`.

   This opt-in boundary does not enable mutable state, indexed maps, arbitrary DAGs, caching,
   host readback, or a default AD engine."
  [owner arguments fixed-seed {:keys [target] :as options}]
  (when-not (and (map? options) (set/subset? (set (keys options)) option-keys))
    (fail! :normalized-device-options
           "normalized device preparation accepts only target, schedule, scalar-math and profile"
           {:keys (when (map? options) (set (keys options)))}))
  (when-not (keyword? target)
    (fail! :normalized-device-target
           "normalized device preparation requires one explicit target keyword"
           {:target target}))
  (let [checked (authority owner :checked-scalar-map-sum)
        assembled (assemble-sum! checked arguments fixed-seed options)]
    (instantiate-prepared! checked assembled options :map-sum)))

(defn prepare-indexed-owner
  "Prepare one canonical fixed-geometry indexed map for resident gather/transpose execution.

   `arguments` fix two positive one-dimensional arrays in canonical interface order: excluded
   Int/Long indices and active Float/Double data. The index/cotangent extent is static; the data
   extent may also be one exact private dimension projected from the prepared data array and then
   remains fixed. `:primal!` accepts an exact ID-to-array map, snapshots a checked clone of the
   indices on device, and returns its dependency lease. `:transpose!` accepts a cotangent array plus
   that exact dependency map; one private complete zero-fill precedes the canonical sequential
   ordered update in the same named entry.

   This opt-in lease boundary supports neither empty or per-call dynamic geometry nor general scatter,
   mutable state, caching, host readback, or default AD activation. Every issued lease must close
   before replaying an overlapping entry or calling `:close!`."
  [owner arguments {:keys [target] :as options}]
  (when-not (and (map? options) (set/subset? (set (keys options)) option-keys))
    (fail! :normalized-device-options
           "normalized device preparation accepts only target, schedule, scalar-math and profile"
           {:keys (when (map? options) (set (keys options)))}))
  (when-not (keyword? target)
    (fail! :normalized-device-target
           "normalized device preparation requires one explicit target keyword"
           {:target target}))
  (let [checked (authority owner :checked-scalar-indexed-map)
        assembled (assemble-indexed! checked arguments options)]
    (instantiate-prepared! checked assembled options :indexed-map)))
