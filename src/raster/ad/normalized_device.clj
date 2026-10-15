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

(defn- map-interface! [owner]
  (let [interface (:interface owner)
        inputs (:inputs interface)
        active-id (get-in interface [:cotangent-result :source])
        active (some #(when (= active-id (:id %)) %) inputs)
        arrays (filterv #(seq (get-in % [:value :shape])) inputs)
        primal (:primal-result interface)
        seed (:cotangent-input interface)
        gradient (:cotangent-result interface)]
    (when-not (and (= 1 (count arrays)) (= active (first arrays))
                   (= (:id primal) (:source seed))
                   (= active-id (:source gradient)))
      (fail! :normalized-device-interface
             "device preparation requires one active array and exact canonical derivative ports"
             {:inputs (mapv :id inputs) :active active-id
              :primal (:id primal) :seed-source (:source seed)
              :gradient-source (:source gradient)}))
    (doseq [[relation left right]
            [[:seed (:value primal) (:value seed)]
             [:gradient (:value active) (:value gradient)]]]
      (when-not (and (av/storage-contract-compatible? left right)
                     (= (:shape left) (:shape right)))
        (fail! :normalized-device-interface-contract
               "canonical derivative ports must preserve exact symbolic storage geometry"
               {:relation relation :left left :right right})))
    {:inputs inputs :active active
     :primal primal :seed seed :gradient gradient}))

(defn- sum-interface! [owner]
  (let [interface (:interface owner)
        inputs (:inputs interface)
        active-ids (vec (:active-inputs interface))
        active-id (first active-ids)
        active (some #(when (= active-id (:id %)) %) inputs)
        arrays (filterv #(seq (get-in % [:value :shape])) inputs)
        scalars (filterv #(empty? (get-in % [:value :shape])) inputs)
        primal (:primal-result interface)
        seed (:cotangent-input interface)
        gradient (:cotangent-result interface)]
    (when-not (and (= 1 (count active-ids)) active
                   (some #(= active-id (:id %)) arrays)
                   (= (count inputs) (+ (count arrays) (count scalars)))
                   (= (:id primal) (:source seed))
                   (= active-id (:source gradient))
                   (empty? (get-in primal [:value :shape]))
                   (empty? (get-in seed [:value :shape])))
      (fail! :normalized-device-interface
             "map-sum preparation requires one active array and exact canonical scalar ports"
             {:inputs (mapv :id inputs) :arrays (mapv :id arrays)
              :active active-ids :primal (:id primal) :seed (:id seed)
              :gradient-source (:source gradient)}))
    (doseq [[relation left right]
            [[:seed (:value primal) (:value seed)]
             [:gradient (:value active) (:value gradient)]]]
      (when-not (and (av/storage-contract-compatible? left right)
                     (= (:shape left) (:shape right)))
        (fail! :normalized-device-interface-contract
               "canonical derivative ports must preserve exact symbolic storage geometry"
               {:relation relation :left left :right right})))
    {:inputs inputs :arrays arrays :scalars scalars :active active
     :primal primal :seed seed :gradient gradient}))

(defn- static-positive-vector? [value]
  (and (= 1 (count (:shape value)))
       (integer? (first (:shape value)))
       (pos? (first (:shape value)))
       (<= (first (:shape value)) Integer/MAX_VALUE)))

(defn- indexed-interface! [owner residual-plan]
  (let [interface (:interface owner)
        inputs (:inputs interface)
        active-id (get-in interface [:cotangent-result :source])
        active (some #(when (= active-id (:id %)) %) inputs)
        arrays (filterv #(seq (get-in % [:value :shape])) inputs)
        primal (:primal-result interface)
        seed (:cotangent-input interface)
        gradient (:cotangent-result interface)
        residuals (vals (:owned-arrays residual-plan))
        physicals (:physical-bindings residual-plan)
        [physical-id physical] (first physicals)
        index-id (get-in (first residuals) [:origin :input])
        index (some #(when (= index-id (:id %)) %) inputs)
        index-snapshot (:id (first residuals))
        zero-requirement (:requirement physical)
        zero-value (:value physical)
        consumer (:consumer physical)
        gradient-id (:id gradient)]
    (when-not
     (and (= 2 (count inputs)) (= 2 (count arrays)) (empty? (remove #(seq (get-in % [:value :shape])) inputs))
          (= 1 (count residuals)) (= 1 (count physicals)) active index (not= active index)
          (= #{index-id active-id} (set (map :id inputs)))
          (contains? #{:int :long} (get-in index [:value :dtype]))
          (contains? #{:float :double} (get-in active [:value :dtype]))
          (contains? #{nil :owned} (get-in index [:value :ownership]))
          (contains? #{nil :owned} (get-in active [:value :ownership]))
          (every? static-positive-vector? (map :value [index active primal seed gradient]))
          (= (:shape (get-in index [:value])) (:shape (get-in primal [:value])))
          (= (:shape (get-in primal [:value])) (:shape (get-in seed [:value])))
          (= (:shape (get-in active [:value])) (:shape (get-in gradient [:value])))
          (av/storage-contract-compatible? (:value primal) (:value seed))
          (av/storage-contract-compatible? (:value active) (:value gradient))
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
             "indexed preparation requires two static arrays and one exact typed-zero gradient"
             {:inputs (mapv :id inputs) :active active-id :index index-id
              :residuals (mapv :id residuals) :physical-bindings (set (keys physicals))}))
    {:inputs inputs :arrays arrays :active active :index index
     :index-snapshot index-snapshot :primal primal :seed seed :gradient gradient
     :zero-id physical-id :zero-value zero-value :zero-requirement zero-requirement}))

(defn- materialize-layout! [compilation input-descriptors arguments caller-options]
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
        _ (when (seq (:steps invocation-plan))
            (fail! :normalized-device-specialization
                   "the first device consumer accepts only direct array/scalar specialization"
                   {:steps (mapv :id (:steps invocation-plan))}))
        materialized
        (materialization/materialize
         invocation-plan invocation-arguments
         (fn [& _]
           (fail! :normalized-device-specialization
                  "device preparation cannot evaluate a host scalar prefix" {})))]
    (invocation-link/project-resident-layout
     materialized (:emitted compilation) (:target compilation) caller-options)))

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

(defn- typed-zero-program [owner {:keys [zero-id zero-value zero-requirement]}]
  (let [used (into (set (keys (:values (soac/facts (:transpose owner)))))
                   (map second (soac/equations (:transpose owner))))
        fresh (fn [prefix]
                (first (remove used (map #(symbol (str prefix %)) (range)))))
        equation-id (fresh "normalized-device-zero-fill-")
        index (fresh "normalized-device-zero-index-")
        dtype (:dtype zero-value)
        extent (first (:shape zero-value))
        cast (dtype/scalar-tag-for-dtype dtype)
        equation (list '= equation-id [zero-id]
                       (list 'map {:index index :extent extent} [] []
                             (soac/lambda-form [] [] [(list cast 0.0)])))
        program
        (soac/make
         (soac/default-program-facts
          {:values {zero-id zero-value}
           :inputs [] :equations {equation-id (soac/default-equation-facts)}})
         [equation] [zero-id])]
    (when-not (and (= :typed-zero (:initialization zero-requirement))
                   (= :owned (:ownership zero-requirement))
                   (true? (:exclusive zero-requirement))
                   (= zero-id (:value zero-requirement))
                   (= zero-value (get-in (soac/facts program) [:values zero-id]))
                   (empty? (:inputs (soac/facts program))))
      (fail! :normalized-device-zero-fill
             "zero fill must derive exactly from the canonical physical obligation"
             {:zero zero-id :requirement zero-requirement}))
    program))

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
    (when-not (= residual-values frontier-values)
      (fail! :normalized-device-frontiers
             "module frontiers must exactly preserve canonical residual order"
             {:expected residual-values :actual frontier-values}))
    (check!)
    {:module module :frontier-values frontier-values}))

(defn- assemble-map!
  [{:keys [owner check!] :as checked} arguments options]
  (let [interface (map-interface! owner)
        residual-plan (canonical/execution-residual-plan owner)
        _ (when (or (seq (:immutable-scalars residual-plan))
                    (seq (:physical-bindings residual-plan)))
            (fail! :normalized-device-residuals
                   "device preparation currently requires array-only residual storage"
                   {:immutable-scalars (set (keys (:immutable-scalars residual-plan)))
                    :physical-bindings (set (keys (:physical-bindings residual-plan)))}))
        target (:target options)
        compile-options (select-keys options [:target :schedule :scalar-math])
        caller-options (caller-options options)
        primal-compilation (compile-phase! checked :primal compile-options)
        transpose-compilation (compile-phase! checked :transpose compile-options)
        layout (materialize-layout! primal-compilation (:inputs interface) arguments caller-options)
        active-id (get-in interface [:active :id])
        primal-id (get-in interface [:primal :id])
        seed-id (get-in interface [:seed :id])
        gradient-id (get-in interface [:gradient :id])
        compiler-bindings (:buffer-bindings layout)
        array-boundary (public-array-boundary! [(:active interface)] layout)
        active-value (get compiler-bindings active-id)
        primal-value (get compiler-bindings primal-id)
        _ (when-not (and active-value primal-value
                         (= [active-id] (:ids array-boundary))
                         (= [active-value] (:values array-boundary)))
            (fail! :normalized-device-layout-interface
                   "resident layout must expose exactly the canonical active array input"
                   {:active active-id :active-value active-value
                    :primal primal-id :primal-value primal-value
                    :public-inputs (:public-inputs layout)}))
        residuals (ordered-residual-frontiers! owner residual-plan compiler-bindings)
        residual-ids (:ids residuals)
        [seed-allocation seed-node seed-value
         gradient-allocation gradient-node gradient-value]
        (fresh-layout-ids layout [[:seed :allocation] [:seed :node] [:seed :value]
                                  [:gradient :allocation] [:gradient :node]
                                  [:gradient :value]])
        seed-layout
        (invocation-link/clone-resident-value-layout
         layout {:template-value primal-value :template-compiler-value primal-id
                 :template-abstract (get-in interface [:primal :value])
                 :compiler-value seed-id :abstract (get-in interface [:seed :value])
                 :value-id seed-value :node-id seed-node :allocation-id seed-allocation
                 :role :input :project-ownership? true})
        gradient-layout
        (invocation-link/clone-resident-value-layout
         layout {:template-value active-value :template-compiler-value active-id
                 :template-abstract (get-in interface [:active :value])
                 :compiler-value gradient-id :abstract (get-in interface [:gradient :value])
                 :value-id gradient-value :node-id gradient-node
                 :allocation-id gradient-allocation :role :output
                 :project-ownership? false})
        nodes (assoc (:nodes layout)
                     seed-node (:node seed-layout)
                     gradient-node (:node gradient-layout))
        values (assoc (:values layout)
                      seed-value (:value seed-layout)
                      gradient-value (:value gradient-layout))
        residual-values (:values residuals)
        transpose-bindings
        (into {seed-id seed-value gradient-id gradient-value}
              (map (fn [id] [id (get compiler-bindings id)])) residual-ids)
        bound (bind-module!
               check! target caller-options nodes values
               {:compilation primal-compilation :buffer-bindings compiler-bindings
                :scalar-values (:scalar-values layout)
                :interface {:inputs [active-value] :state [] :outputs [primal-value]}}
               [{:instance-id ::transpose
                 :compilation transpose-compilation :buffer-bindings transpose-bindings
                 :scalar-values (:scalar-values layout)
                 :interface {:inputs [seed-value] :state [] :outputs [gradient-value]}}]
               residual-values)]
    {:module (:module bound) :frontier-values (:frontier-values bound)
     :ports {:input active-value :primal-output primal-value
             :seed-input seed-value :gradient-output gradient-value
             :residuals (:frontier-values bound)}}))

(defn- assemble-sum!
  [{:keys [owner check!] :as checked} arguments fixed-seed options]
  (let [interface (sum-interface! owner)
        residual-plan (canonical/execution-residual-plan owner)
        _ (when (or (seq (:immutable-scalars residual-plan))
                    (seq (:physical-bindings residual-plan)))
            (fail! :normalized-device-residuals
                   "the first map-sum device slice requires array-only residual storage"
                   {:immutable-scalars (set (keys (:immutable-scalars residual-plan)))
                    :physical-bindings (set (keys (:physical-bindings residual-plan)))}))
        target (:target options)
        compile-options (select-keys options [:target :schedule :scalar-math])
        caller-options (caller-options options)
        active-id (get-in interface [:active :id])
        primal-id (get-in interface [:primal :id])
        seed-id (get-in interface [:seed :id])
        gradient-id (get-in interface [:gradient :id])
        primal-compilation
        (compile-phase! checked :primal (assoc compile-options :resident-outputs [primal-id]))
        transpose-compilation (compile-phase! checked :transpose compile-options)
        layout (materialize-layout! primal-compilation (:inputs interface) arguments caller-options)
        compiler-bindings (:buffer-bindings layout)
        active-value (get compiler-bindings active-id)
        primal-value (get compiler-bindings primal-id)
        array-boundary (public-array-boundary! (:arrays interface) layout)
        canonical-array-ids (:ids array-boundary)
        public-array-values (:values array-boundary)
        _ (when-not (and active-value primal-value
                         (= :resident-scalar-buffer
                            (get-in layout [:values primal-value :abstract
                                            :representation :kind]))
                         (= [] (get-in layout [:values primal-value :abstract :shape])))
            (fail! :normalized-device-layout-interface
                   "map-sum resident layout must expose every canonical array and scalar loss"
                   {:expected-arrays canonical-array-ids :public-inputs (:public-inputs layout)
                    :active active-id :active-value active-value
                    :primal primal-id :primal-value primal-value
                    :primal-abstract (get-in layout [:values primal-value :abstract])}))
        residuals (ordered-residual-frontiers! owner residual-plan compiler-bindings)
        residual-ids (:ids residuals)
        residual-values (:values residuals)
        typed-seed (materialization/checked-scalar-value
                    seed-id (get-in interface [:seed :value]) fixed-seed)
        transpose-facts (soac/facts (:transpose owner))
        transpose-inputs (:inputs transpose-facts)
        transpose-values (:values transpose-facts)
        transpose-scalar-ids
        (filterv #(empty? (get-in transpose-values [% :shape])) transpose-inputs)
        transpose-scalars
        (into {}
              (map (fn [id]
                     [id (if (= id seed-id) typed-seed (get (:scalar-values layout) id))]))
              transpose-scalar-ids)
        _ (when-not (and (some #{seed-id} transpose-scalar-ids)
                         (every? some? (vals transpose-scalars)))
            (fail! :normalized-device-specialization
                   "map-sum transpose scalars must close from fixed primal specialization and seed"
                   {:seed seed-id :inputs transpose-inputs
                    :scalars (set (keys transpose-scalars))
                    :available (set (keys (:scalar-values layout)))}))
        [gradient-allocation gradient-node gradient-value]
        (fresh-layout-ids layout [[:gradient :allocation] [:gradient :node]
                                  [:gradient :value]])
        gradient-layout
        (invocation-link/clone-resident-value-layout
         layout {:template-value active-value :template-compiler-value active-id
                 :template-abstract (get-in interface [:active :value])
                 :compiler-value gradient-id :abstract (get-in interface [:gradient :value])
                 :value-id gradient-value :node-id gradient-node
                 :allocation-id gradient-allocation :role :output
                 :project-ownership? false})
        nodes (assoc (:nodes layout) gradient-node (:node gradient-layout))
        values (assoc (:values layout) gradient-value (:value gradient-layout))
        transpose-bindings
        (into {gradient-id gradient-value}
              (map (fn [id] [id (get compiler-bindings id)])) residual-ids)
        bound
        (bind-module!
         check! target caller-options nodes values
         {:compilation primal-compilation :buffer-bindings compiler-bindings
          :scalar-values (:scalar-values layout)
          :interface {:inputs public-array-values :state [] :outputs [primal-value]}}
         [{:instance-id ::transpose
           :compilation transpose-compilation :buffer-bindings transpose-bindings
           :scalar-values transpose-scalars
           :interface {:inputs [] :state [] :outputs [gradient-value]}}]
         residual-values)]
    {:module (:module bound) :frontier-values (:frontier-values bound)
     :ports {:inputs (mapv (fn [id value] {:id id :value value})
                           canonical-array-ids public-array-values)
             :primal-output primal-value :gradient-output gradient-value
             :residuals (:frontier-values bound)
             :fixed-seed {:id seed-id :value typed-seed}}}))

(defn- assemble-indexed!
  [{:keys [owner check!] :as checked} arguments options]
  (let [residual-plan (canonical/execution-residual-plan owner)
        interface (indexed-interface! owner residual-plan)
        target (:target options)
        compile-options (select-keys options [:target :schedule :scalar-math])
        caller-options (caller-options options)
        primal-compilation (compile-phase! checked :primal compile-options)
        transpose-compilation (compile-phase! checked :transpose compile-options)
        zero-program (typed-zero-program owner interface)
        zero-compilation (compile-program! checked zero-program compile-options)
        layout (materialize-layout! primal-compilation (:inputs interface) arguments caller-options)
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
        [seed-allocation seed-node seed-value
         gradient-allocation gradient-node gradient-value]
        (fresh-layout-ids layout [[:seed :allocation] [:seed :node] [:seed :value]
                                  [:gradient :allocation] [:gradient :node]
                                  [:gradient :value]])
        seed-layout
        (invocation-link/clone-resident-value-layout
         layout {:template-value primal-value :template-compiler-value primal-id
                 :template-abstract (get-in interface [:primal :value])
                 :compiler-value seed-id :abstract (get-in interface [:seed :value])
                 :value-id seed-value :node-id seed-node :allocation-id seed-allocation
                 :role :input :project-ownership? true})
        gradient-layout
        (invocation-link/clone-resident-value-layout
         layout {:template-value active-value :template-compiler-value active-id
                 :template-abstract (get-in interface [:active :value])
                 :compiler-value gradient-id :abstract (get-in interface [:gradient :value])
                 :value-id gradient-value :node-id gradient-node
                 :allocation-id gradient-allocation :role :output
                 :project-ownership? false})
        nodes (assoc (:nodes layout)
                     seed-node (:node seed-layout)
                     gradient-node (:node gradient-layout))
        values (assoc (:values layout)
                      seed-value (:value seed-layout)
                      gradient-value (:value gradient-layout))
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
           :buffer-bindings transpose-bindings :scalar-values (:scalar-values layout)
           :interface {:inputs [seed-value] :state [] :outputs [gradient-value]}}]
         residual-values)
        index-count (first (get-in interface [:index :value :shape]))
        gradient-count (first (get-in interface [:active :value :shape]))]
    {:module (:module bound) :frontier-values (:frontier-values bound)
     :ports {:inputs (mapv (fn [id value]
                             {:id id :value value
                              :dtype (get-in (soac/facts (:primal owner)) [:values id :dtype])
                              :elements (first (get-in (soac/facts (:primal owner))
                                                       [:values id :shape]))})
                           canonical-array-ids public-array-values)
             :index-input index-value :index-id index-id
             :index-dtype (get-in interface [:index :value :dtype])
             :index-count index-count :gradient-count gradient-count
             :primal-output primal-value :seed-input seed-value
             :seed-dtype (get-in interface [:seed :value :dtype])
             :seed-elements (first (get-in interface [:seed :value :shape]))
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

(defn- instantiate-prepared!
  [{:keys [check! guard!]} {:keys [module frontier-values ports]} options family]
  (let [runtime-options (select-keys options [:scalar-math :profile?])
        runtime (link-module/instantiate! module runtime-options)]
    (try
      (check!)
      (let [[primal! transpose!]
            (case family
              :map
              [(fn [input]
                 (run-result! guard! runtime :primal
                              {:inputs {(:input ports) input}}
                              (:primal-output ports) frontier-values true))
               (fn [seed dependencies]
                 (run-result! guard! runtime :transpose
                              {:inputs {(:seed-input ports) seed}
                               :dependencies dependencies}
                              (:gradient-output ports) frontier-values false))]

              :map-sum
              [(fn [inputs]
                 (let [input-ports (:inputs ports)
                       expected (mapv :id input-ports)]
                   (when-not (and (map? inputs) (= (set expected) (set (keys inputs)))
                                  (= (count expected) (count inputs)))
                     (fail! :normalized-device-inputs
                            "map-sum primal inputs must exactly cover canonical array IDs"
                            {:expected expected
                             :actual (when (map? inputs) (set (keys inputs)))}))
                   (run-result!
                    guard! runtime :primal
                    {:inputs (into {} (map (fn [{:keys [id value]}]
                                             [value (get inputs id)])) input-ports)}
                    (:primal-output ports) frontier-values true)))
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
   and `:transpose!` accept new exact primitive buffers, returning opaque leases whose outputs
   remain device-resident. Callers must close every lease before invoking `:close!`.

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
  "Prepare one canonical static indexed map for explicit resident gather/transpose execution.

   `arguments` fix two positive one-dimensional arrays in canonical interface order: excluded
   Int/Long indices and active Float/Double data. `:primal!` accepts an exact ID-to-array map, snapshots a
   checked clone of the indices on device, and returns its dependency lease. `:transpose!` accepts
   a cotangent array plus that exact dependency map; one private complete zero-fill precedes the
   canonical sequential ordered update in the same named entry.

   This opt-in lease boundary supports neither dynamic/empty geometry nor general scatter,
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
