(ns raster.compiler.ir.kernel-graph-call
  "A backend-neutral executable call of an emitted KernelGraph.

   KernelGraph owns stable buffers, node uses and dependencies. KernelGraphCall supplies one
   resident value for every graph buffer and turns each emitted node into a checked KernelCall.
   Driver allocation, registration, recording and events remain runtime concerns."
  (:require [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-executable :as executable]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-launch :as klaunch]))

(defrecord ScheduledKernelCall [id call dependencies])
(defrecord KernelGraphCall [graph buffers scalar-values nodes])

(defn scheduled-kernel-call? [x]
  (and x (= "raster.compiler.ir.kernel_graph_call.ScheduledKernelCall"
            (.getName (class x)))))

(defn kernel-graph-call? [x]
  (and x (= "raster.compiler.ir.kernel_graph_call.KernelGraphCall"
            (.getName (class x)))))

(defn- graph-buffers
  [graph]
  (concat (:inputs graph) (:outputs graph) (:temporaries graph)))

(defn- declared-buffer-ids
  [graph]
  (set (map :id (graph-buffers graph))))

(defn- scalar-interface
  [graph]
  (filterv (fn [[slot _]] (= :scalar (:kind slot)))
           (mapv vector (:abi graph) (:arguments graph))))

(defn- validate-scalar-values!
  [graph scalar-values]
  (when-not (map? scalar-values)
    (throw (ex-info "kernel graph call scalar values must be a map"
                    {:scalar-values scalar-values})))
  (let [interface (scalar-interface graph)
        expected (set (map second interface))
        actual (set (keys scalar-values))]
    (when-not (= expected actual)
      (throw (ex-info "kernel graph call requires exactly every public scalar"
                      {:reason :kernel-graph-call-scalars
                       :expected expected :bound actual})))
    (doseq [[slot argument] interface
            :let [value (get scalar-values argument)]]
      (when-not (and (map? value) (contains? value :type) (contains? value :value))
        (throw (ex-info "graph symbolic scalar requires an explicitly typed runtime value"
                        {:reason :kernel-graph-call-scalar-type
                         :argument argument :slot slot :value value})))
      (when-not (= (:kernel-dtype slot) (:type value))
        (throw (ex-info "kernel graph scalar argument has the wrong ABI dtype"
                        {:reason :kernel-graph-call-scalar-type
                         :argument argument :slot slot :value value}))))
    scalar-values))

(defn- scalar-number
  [scalar-values value]
  (let [resolved (if (contains? scalar-values value)
                   (get scalar-values value)
                   value)
        resolved (if (and (map? resolved) (contains? resolved :value))
                   (:value resolved)
                   resolved)]
    (when-not (integer? resolved)
      (throw (ex-info "graph extent expression did not resolve to an integer"
                      {:expression value :resolved resolved})))
    resolved))

(defn resolve-integer
  "Resolve a graph extent or derived bound without evaluating arbitrary Clojure forms."
  [scalar-values expression]
  (klaunch/resolve-expression #(scalar-number scalar-values %) expression))

(defn temporary-specs
  "Resolve graph-owned temporary storage to core allocation specs: `{id [dtype elements nil]}`."
  [graph scalar-values]
  (let [graph (kgraph/validate! graph)]
    (into {}
          (map (fn [{:keys [id dtype elements]}]
                 (let [n (resolve-integer scalar-values elements)]
                   (when (neg? n)
                     (throw (ex-info "graph temporary extent must be non-negative"
                                     {:buffer id :elements elements :resolved n})))
                   [id [dtype n nil]])))
          (:temporaries graph))))

(defn- cast-scalar
  [dtype value]
  (case dtype
    :int (Math/toIntExact (long value))
    :long (long value)
    (throw (ex-info "derived graph scalar is only defined for integer ABI slots"
                    {:dtype dtype :value value}))))

(defn- scalar-argument
  [scalar-values slot compiler-value]
  (if (contains? scalar-values compiler-value)
    (let [value (get scalar-values compiler-value)]
      (when-not (and (map? value) (contains? value :type) (contains? value :value))
        (throw (ex-info "graph symbolic scalar requires an explicitly typed runtime value"
                        {:compiler-value compiler-value :slot slot :value value})))
      (if (= (:type value) (:kernel-dtype slot))
        value
        (if (every? #{:int :long} [(:type value) (:kernel-dtype slot)])
          (executable/physical-runtime-scalar slot (:value value))
          (throw (ex-info "graph scalar requires an unsupported physical conversion"
                          {:reason :kernel-graph-call-scalar-conversion
                           :slot slot :value value})))))
    (let [value (resolve-integer scalar-values compiler-value)]
      {:type (:kernel-dtype slot)
       :value (cast-scalar (:kernel-dtype slot) value)})))

(defn validate!
  "Validate and return a KernelGraphCall."
  [graph-call]
  (when-not (kernel-graph-call? graph-call)
    (throw (ex-info "kernel graph call must be a KernelGraphCall value"
                    {:call graph-call :actual (type graph-call)})))
  (let [{:keys [graph buffers scalar-values nodes]} graph-call
        graph (kgraph/validate! graph)
        declared (declared-buffer-ids graph)]
    (when-not (map? buffers)
      (throw (ex-info "kernel graph call buffers must be a map" {:buffers buffers})))
    (when-not (= declared (set (keys buffers)))
      (throw (ex-info "kernel graph call buffer bindings differ from graph declarations"
                      {:declared declared :bound (set (keys buffers))})))
    (when (some nil? (vals buffers))
      (throw (ex-info "kernel graph call buffer cannot be nil" {})))
    (validate-scalar-values! graph scalar-values)
    (when-not (vector? nodes)
      (throw (ex-info "kernel graph call nodes must be an ordered vector" {:nodes nodes})))
    (when-not (= (count (:nodes graph)) (count nodes))
      (throw (ex-info "kernel graph call node count differs from its graph"
                      {:expected (count (:nodes graph)) :actual (count nodes)})))
    (doseq [[scheduled called] (map vector (:nodes graph) nodes)]
      (when-not (scheduled-kernel-call? called)
        (throw (ex-info "kernel graph call contains a non-node call" {:node called})))
      (when-not (= (:id scheduled) (:id called))
        (throw (ex-info "kernel graph node call identity differs from its schedule"
                        {:scheduled (:id scheduled) :called (:id called)})))
      (when-not (= (:dependencies scheduled) (:dependencies called))
        (throw (ex-info "kernel graph node call dependencies differ from its schedule"
                        {:node (:id scheduled)
                         :scheduled (:dependencies scheduled)
                         :called (:dependencies called)})))
      (let [call (kcall/validate! (:call called))]
        (when-not (= (:operation scheduled) (:artifact call))
          (throw (ex-info "kernel graph node call uses a different artifact"
                          {:node (:id scheduled)
                           :scheduled (:operation scheduled)
                           :called (:artifact call)})))))
    graph-call))

(defn make
  "Construct a checked graph call from an emitted graph, complete resident buffer map, and typed
   symbolic scalar values. Derived integer arguments such as a block-count CeilDiv are resolved
   from the typed scalar environment and receive the ABI slot's integer type."
  [graph buffers scalar-values]
  (let [graph (executable/validate! graph)
        declared (declared-buffer-ids graph)
        scalar-values (validate-scalar-values! graph (or scalar-values {}))]
    (when-not (= declared (set (keys buffers)))
      (throw (ex-info "kernel graph call requires exactly every declared graph buffer"
                      {:declared declared :bound (set (keys buffers))})))
    (let [nodes
          (mapv
           (fn [{:keys [id operation dependencies]}]
             (let [artifact (kart/validate! operation)
                   arguments
                   (mapv (fn [slot compiler-value]
                           (if (= :scalar (:kind slot))
                             (scalar-argument scalar-values slot compiler-value)
                             (or (get buffers compiler-value)
                                 (throw (ex-info "kernel pointer argument is not a graph buffer"
                                                 {:node id :compiler-value compiler-value
                                                  :declared declared})))))
                         (:abi artifact) (:arguments artifact))]
               ;; A node may bind a private scalar expression (for example M*K) while its launch
               ;; still names the public leaves M and K. Resolve launch algebra from the graph's
               ;; typed scalar environment, not by searching only the node's narrowed ABI.
               (->ScheduledKernelCall
                id
                (kcall/make artifact arguments
                            {:resolve-value #(resolve-integer scalar-values %)})
                dependencies)))
           (:nodes graph))]
      (validate! (->KernelGraphCall graph buffers scalar-values nodes)))))
