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

(defn- authority [owner]
  (let [owner (canonical/validate! owner)
        derivation (:derivation owner)
        bundle (:bundle derivation)]
    (when-not (= :checked-scalar-map (:kind derivation))
      (fail! :normalized-device-family
             "device preparation currently admits only canonical checked maps and linear spines"
             {:kind (:kind derivation)}))
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

(defn- interface! [owner]
  (let [interface (:interface owner)
        inputs (:inputs interface)
        active-id (get-in interface [:cotangent-result :source])
        active (some #(when (= active-id (:id %)) %) inputs)
        arrays (filterv #(seq (get-in % [:value :shape])) inputs)
        scalars (filterv #(empty? (get-in % [:value :shape])) inputs)
        primal (:primal-result interface)
        seed (:cotangent-input interface)
        gradient (:cotangent-result interface)]
    (when-not (and (= 1 (count arrays)) (= active (first arrays))
                   (= (count inputs) (+ (count arrays) (count scalars)))
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
    {:inputs inputs :active active :scalars scalars
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

(defn- close-preserving! [primary closeable]
  (try
    (.close ^java.io.Closeable closeable)
    (catch Throwable cleanup-error
      (when-not (identical? primary cleanup-error)
        (.addSuppressed ^Throwable primary cleanup-error))))
  (throw primary))

(defn- guarded-run! [check! invoke]
  (check!)
  (let [lease (invoke)]
    (try
      (check!)
      lease
      (catch Throwable primary (close-preserving! primary lease)))))

(defn- publish-result! [lease build validate!]
  (try
    (let [result (build)]
      (validate! result)
      result)
    (catch Throwable primary
      (close-preserving! primary lease))))

(defn- assemble!
  [{:keys [owner check!] :as checked} arguments options]
  (let [interface (interface! owner)
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
        residual-bindings (vals (:owned-arrays residual-plan))
        residual-by-id (into {} (map (juxt :id identity)) residual-bindings)
        residual-ids (into [] (filter #(contains? residual-by-id %))
                           (soac/outputs (:primal owner)))
        _ (when-not (= (set residual-ids) (set (keys residual-by-id)))
            (fail! :normalized-device-residual-order
                   "canonical primal outputs must order every owned residual exactly once"
                   {:outputs (soac/outputs (:primal owner))
                    :residuals (set (keys residual-by-id))}))
        _ (doseq [residual-id residual-ids
                  :let [binding (get residual-by-id residual-id)]]
            (when-not (and (= residual-id (get-in binding [:producer :result]))
                           (= (:value binding)
                              (get-in (soac/facts (:primal owner)) [:values residual-id]))
                           (contains? compiler-bindings residual-id))
              (fail! :normalized-device-residual-correspondence
                     "resident residual differs from its exact canonical primal producer"
                     {:residual residual-id :binding binding})))
        used (into (set (keys (:nodes layout)))
                   (concat (keys (:values layout))
                           (map #(get-in % [:view :allocation :id]) (vals (:nodes layout)))))
        seed-allocation (fresh-id used [:seed :allocation])
        used (conj used seed-allocation)
        seed-node (fresh-id used [:seed :node])
        used (conj used seed-node)
        seed-value (fresh-id used [:seed :value])
        used (conj used seed-value)
        gradient-allocation (fresh-id used [:gradient :allocation])
        used (conj used gradient-allocation)
        gradient-node (fresh-id used [:gradient :node])
        used (conj used gradient-node)
        gradient-value (fresh-id used [:gradient :value])
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
        residual-values (mapv compiler-bindings residual-ids)
        transpose-bindings
        (into {seed-id seed-value gradient-id gradient-value}
              (map (fn [id] [id (get compiler-bindings id)])) residual-ids)
        bind (fn [id compilation bindings]
               (let [request {:id id :target target :nodes nodes :values values
                              :buffer-bindings bindings
                              :scalar-values (:scalar-values layout)}]
                 (if caller-options
                   (invocation-link/bind-resident-program
                    (:emitted compilation) request caller-options)
                   (invocation-link/bind-resident-program (:emitted compilation) request))))
        primal-instance (bind ::primal primal-compilation compiler-bindings)
        transpose-instance (bind ::transpose transpose-compilation transpose-bindings)
        spine-request {:id ::spine :target target :nodes nodes :values values
                       :instances [primal-instance transpose-instance] :outputs []}
        spine (if caller-options
                (link-plan/make spine-request caller-options)
                (link-plan/make spine-request))
        module-request
        {:id ::module :spine spine
         :entries [{:name :primal :instances [::primal]
                    :interface {:inputs [active-value] :state [] :outputs [primal-value]}}
                   {:name :transpose :instances [::transpose]
                    :interface {:inputs [seed-value] :state [] :outputs [gradient-value]}}]}
        module (if caller-options
                 (module-plan/make module-request caller-options)
                 (module-plan/make module-request))
        frontiers (filterv #(= :transpose (:consumer %)) (:frontiers module))
        frontier-values (mapv :value frontiers)]
    (when-not (= residual-values frontier-values)
      (fail! :normalized-device-frontiers
             "module frontiers must exactly preserve canonical residual order"
             {:expected residual-values :actual frontier-values}))
    (check!)
    {:module module :frontier-values frontier-values
     :ports {:input active-value :primal-output primal-value
             :seed-input seed-value :gradient-output gradient-value
             :residuals frontier-values}}))

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
   (let [options options
         checked (authority owner)
         {:keys [module frontier-values ports]} (assemble! checked arguments options)
         check! (:check! checked)
         runtime-options (select-keys options [:scalar-math :profile?])
         runtime (link-module/instantiate! module runtime-options)]
     (try
       (check!)
       (let [primal!
             (fn [input]
               (let [lease (guarded-run!
                            check!
                            #(link-module/run-and-lease!
                              runtime :primal {:inputs {(:input ports) input}}))]
                 (publish-result!
                  lease
                  #(hash-map :lease lease :output-id (:primal-output ports)
                             :dependencies (zipmap frontier-values (repeat lease)))
                  (fn [result]
                    (when-not (and (= #{:lease :output-id :dependencies}
                                      (set (keys result)))
                                   (identical? lease (:lease result))
                                   (= (:primal-output ports) (:output-id result))
                                   (= (set frontier-values)
                                      (set (keys (:dependencies result))))
                                   (= (count frontier-values)
                                      (count (:dependencies result)))
                                   (every? #(identical? lease %)
                                           (vals (:dependencies result))))
                      (fail! :normalized-device-result
                             "primal result must publish the exact issued lease and frontiers"
                             {:output (:output-id result)
                              :dependencies (set (keys (:dependencies result)))}))))))
             transpose!
             (fn [seed dependencies]
               (let [lease (guarded-run!
                            check!
                            #(link-module/run-and-lease!
                              runtime :transpose
                              {:inputs {(:seed-input ports) seed}
                               :dependencies dependencies}))]
                 (publish-result!
                  lease
                  #(hash-map :lease lease :output-id (:gradient-output ports))
                  (fn [result]
                    (when-not (and (= #{:lease :output-id} (set (keys result)))
                                   (identical? lease (:lease result))
                                   (= (:gradient-output ports) (:output-id result)))
                      (fail! :normalized-device-result
                             "transpose result must publish the exact issued lease and output"
                             {:output (:output-id result)}))))))
             prepared {:ports ports :primal! primal! :transpose! transpose!
                       :close! #(.close ^java.io.Closeable runtime)}]
         (when-not (= prepared-keys (set (keys prepared)))
           (fail! :normalized-device-prepared "invalid prepared owner projection" {}))
         prepared)
       (catch Throwable primary (close-preserving! primary runtime)))))
