(ns raster.ad.normalized-device
  "Opt-in lease-oriented device execution for one canonical normalized map owner.

   This surface is intentionally not the CPU-compatible normalized AD runtime: results remain
   resident behind opaque entry leases and callers supply exact primitive host input buffers for
   each invocation. Preparation fixes scalar values, shapes, target and numerical policy."
  (:require [clojure.set :as set]
            [raster.ad.normalized-program-bundle :as canonical]
            [raster.ad.normalized-scalar :as normalized]
            [raster.compiler.equation-first :as equation-first]
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

(defn- bind-module!
  [check! target caller-options nodes values primal-phase transpose-phase residual-values]
  (let [bind (fn [id {:keys [compilation buffer-bindings scalar-values]}]
               (let [request {:id id :target target :nodes nodes :values values
                              :buffer-bindings buffer-bindings :scalar-values scalar-values}]
                 (if caller-options
                   (invocation-link/bind-resident-program
                    (:emitted compilation) request caller-options)
                   (invocation-link/bind-resident-program (:emitted compilation) request))))
        primal-instance (bind ::primal primal-phase)
        transpose-instance (bind ::transpose transpose-phase)
        spine-request {:id ::spine :target target :nodes nodes :values values
                       :instances [primal-instance transpose-instance] :outputs []}
        spine (if caller-options
                (link-plan/make spine-request caller-options)
                (link-plan/make spine-request))
        module-request
        {:id ::module :spine spine
         :entries [{:name :primal :instances [::primal] :interface (:interface primal-phase)}
                   {:name :transpose :instances [::transpose]
                    :interface (:interface transpose-phase)}]}
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
        active-value (get compiler-bindings active-id)
        primal-value (get compiler-bindings primal-id)
        _ (when-not (and active-value primal-value
                         (= [{:symbol active-id :value active-value}]
                            (:public-inputs layout)))
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
               {:compilation transpose-compilation :buffer-bindings transpose-bindings
                :scalar-values (:scalar-values layout)
                :interface {:inputs [seed-value] :state [] :outputs [gradient-value]}}
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
        canonical-array-ids (mapv :id (:arrays interface))
        public-inputs (:public-inputs layout)
        public-array-ids (mapv :symbol public-inputs)
        public-array-values (mapv :value public-inputs)
        _ (when-not (and active-value primal-value
                         (= canonical-array-ids public-array-ids)
                         (= (count public-array-values) (count (set public-array-values)))
                         (= :resident-scalar-buffer
                            (get-in layout [:values primal-value :abstract
                                            :representation :kind]))
                         (= [] (get-in layout [:values primal-value :abstract :shape])))
            (fail! :normalized-device-layout-interface
                   "map-sum resident layout must expose every canonical array and scalar loss"
                   {:expected-arrays canonical-array-ids :public-inputs public-inputs
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
         {:compilation transpose-compilation :buffer-bindings transpose-bindings
          :scalar-values transpose-scalars
          :interface {:inputs [] :state [] :outputs [gradient-value]}}
         residual-values)]
    {:module (:module bound) :frontier-values (:frontier-values bound)
     :ports {:inputs (mapv (fn [id value] {:id id :value value})
                           canonical-array-ids public-array-values)
             :primal-output primal-value :gradient-output gradient-value
             :residuals (:frontier-values bound)
             :fixed-seed {:id seed-id :value typed-seed}}}))

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
