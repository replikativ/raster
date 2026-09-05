(ns raster.compiler.ir.kernel-executable
  "The common executable boundary for one emitted kernel or a scheduled emitted kernel graph.

   This namespace does not erase the distinction between the two values. It exposes only the
   properties that dispatch selection and offline tuning may compare: target, ordered external
   ABI, compiler arguments, logical effects, strategy, and concrete entry-point artifacts. Graph
   temporaries, dependencies, and derived scalars remain graph-owned schedule details."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-launch :as klaunch]))

(defn kernel-executable?
  [value]
  (or (kart/kernel-artifact? value) (kgraph/kernel-graph? value)))

(defn kind
  [executable]
  (cond
    (kart/kernel-artifact? executable) :kernel-artifact
    (kgraph/kernel-graph? executable) :kernel-graph
    :else nil))

(defn artifacts
  "Return every concrete entry-point artifact in execution order."
  [executable]
  (case (kind executable)
    :kernel-artifact [(kart/validate! executable)]
    :kernel-graph (mapv (comp kart/validate! :operation)
                        (:nodes (kgraph/validate! executable)))
    (throw (ex-info "kernel executable must be an artifact or graph"
                    {:executable executable :actual (type executable)}))))

(defn- access-flags
  [access]
  (case access
    :read #{:read}
    :write #{:write}
    :read-write #{:read :write}))

(defn- flags-access
  [flags]
  (case flags
    #{:read} :read
    #{:write} :write
    #{:read :write} :read-write))

(defn- validate-node-artifact-bindings!
  "Bind every emitted pointer slot back to the exact scheduled ValueUse it implements."
  [graph]
  (let [buffers (into {} (map (juxt :id identity))
                      (concat (:inputs graph) (:outputs graph) (:temporaries graph)))
        public-scalars
        (into {}
              (keep (fn [[slot argument]]
                      (when (= :scalar (:kind slot)) [argument slot])))
              (map vector (:abi graph) (:arguments graph)))]
    (doseq [{:keys [id operation uses scalar-uses]} (:nodes graph)]
      (let [artifact (kart/validate! operation)
            pointer-pairs (filterv (fn [[slot _]] (not= :scalar (:kind slot)))
                                   (mapv vector (:abi artifact) (:arguments artifact)))
            undeclared (filterv (fn [[_ argument]] (not (contains? buffers argument)))
                                pointer-pairs)]
        (when (seq undeclared)
          (throw (ex-info "kernel artifact pointer argument is not a declared graph buffer"
                          {:reason :kernel-graph-artifact-buffer
                           :node id :arguments (mapv second undeclared)
                           :declared (set (keys buffers))})))
        (doseq [[slot argument] pointer-pairs
                :let [buffer (get buffers argument)]]
          (when-not (= (:dtype slot) (:dtype buffer))
            (throw (ex-info "kernel artifact pointer dtype differs from its graph buffer"
                            {:reason :kernel-graph-artifact-buffer-dtype
                             :node id :argument argument :slot slot :buffer buffer}))))
        (let [actual
              (into {}
                    (map (fn [[argument pairs]]
                           [argument
                            (flags-access
                             (reduce into #{}
                                     (map (comp access-flags kabi/slot-access first) pairs)))]))
                    (group-by second pointer-pairs))
              expected (into {} (map (juxt :buffer :access)) uses)]
          (when-not (= expected actual)
            (throw (ex-info "kernel artifact pointer ABI differs from scheduled graph uses"
                            {:reason :kernel-graph-artifact-uses
                             :node id :expected expected :actual actual}))))
        (doseq [[slot argument] (filterv (fn [[slot _]] (= :scalar (:kind slot)))
                                         (mapv vector (:abi artifact) (:arguments artifact)))]
          (if-let [public-slot (get public-scalars argument)]
            (when-not (= (select-keys slot [:dtype :kernel-dtype])
                         (select-keys public-slot [:dtype :kernel-dtype]))
              (throw (ex-info "kernel artifact scalar dtype differs from its graph interface"
                              {:reason :kernel-graph-artifact-scalar-dtype
                               :node id :argument argument
                               :artifact-slot slot :graph-slot public-slot})))
            (let [references (klaunch/expression-references argument)]
              (try
                (when-not (contains? #{:int :long} (:kernel-dtype slot))
                  (throw (ex-info "target-private scalar representation must be integral"
                                  {:kernel-dtype (:kernel-dtype slot)})))
                (klaunch/validate-typed-expression!
                 argument #(some-> (get public-scalars %) :dtype))
                (catch clojure.lang.ExceptionInfo exception
                  (throw (ex-info "kernel artifact scalar is not closed over checked graph scalars"
                                  {:reason :kernel-graph-artifact-scalar
                                   :node id :argument argument :slot slot
                                   :references references
                                   :public-scalars (set (keys public-scalars))
                                   :expression-error (ex-data exception)}
                                  exception)))))))
        (let [actual-scalar-uses
              (kgraph/scalar-argument-uses (:abi artifact) (:arguments artifact))]
          (when (not= scalar-uses actual-scalar-uses)
            (throw (ex-info "kernel artifact scalar arguments differ from scheduled graph uses"
                            {:reason :kernel-graph-artifact-scalar-uses
                             :node id :expected scalar-uses :actual actual-scalar-uses}))))))
    graph))

(defn validate!
  "Validate an emitted executable and return it unchanged."
  [executable]
  (case (kind executable)
    :kernel-artifact (kart/validate! executable)
    :kernel-graph
    (let [graph (kgraph/validate! executable)]
      (when-not (kgraph/has-interface? graph)
        (throw (ex-info "emitted kernel graph requires an ordered external ABI"
                        {:reason :kernel-graph-executable-interface :graph graph})))
      (when-not (seq (:nodes graph))
        (throw (ex-info "emitted kernel graph requires at least one kernel node"
                        {:reason :kernel-graph-executable-empty :graph graph})))
      (when (or (nil? (:scalars graph))
                (some (comp nil? :scalar-uses) (:nodes graph)))
        (throw (ex-info "emitted kernel graph requires explicit scalar dependencies"
                        {:reason :kernel-graph-executable-scalars
                         :scalars (:scalars graph)
                         :node-scalar-uses (mapv (juxt :id :scalar-uses) (:nodes graph))})))
      (validate-node-artifact-bindings! graph)
      (let [targets (set (map :target (artifacts graph)))]
        (when-not (= 1 (count targets))
          (throw (ex-info "kernel graph nodes must share one target dialect"
                          {:reason :kernel-graph-executable-targets :targets targets}))))
      graph)
    (throw (ex-info "kernel executable must be an artifact or graph"
                    {:executable executable :actual (type executable)}))))

(defn target [executable]
  (:target (first (artifacts (validate! executable)))))

(defn abi [executable]
  (:abi (validate! executable)))

(defn arguments [executable]
  (:arguments (validate! executable)))

(defn effects [executable]
  (:effects (validate! executable)))

(defn attributes [executable]
  (:attributes (validate! executable)))

(defn strategy
  [executable]
  (get (attributes executable) :strategy))

(defn entry-points
  [executable]
  (mapv :kernel-name (artifacts (validate! executable))))

(defn- cast-runtime-scalar
  [dtype value]
  (case dtype
    :int (if (integer? value)
           (Math/toIntExact (long value))
           (throw (ex-info "kernel integral scalar requires an integer value"
                           {:reason :kernel-executable-integral-scalar
                            :dtype dtype :value value})))
    :long (if (integer? value)
            (long value)
            (throw (ex-info "kernel integral scalar requires an integer value"
                            {:reason :kernel-executable-integral-scalar
                             :dtype dtype :value value})))
    :float (float value)
    :double (double value)
    (throw (ex-info "kernel executable has no runtime scalar representation for ABI dtype"
                    {:dtype dtype :value value}))))

(defn physical-runtime-scalar
  "Normalize one raw or already-physical scalar for an ABI slot.

   This is the fail-before-driver boundary used by compatibility binders that already split
   pointers, user scalars, and the implicit bound. In particular, an `:int` slot uses
   `Math/toIntExact`; Clojure's unchecked `(int value)` is never an ABI conversion proof."
  [slot value]
  (when-not (= :scalar (:kind slot))
    (throw (ex-info "physical runtime scalar requires a scalar ABI slot"
                    {:reason :kernel-executable-scalar-slot :slot slot :value value})))
  (let [kernel-dtype (:kernel-dtype slot)
        raw (if (and (map? value) (contains? value :type) (contains? value :value))
              (do
                (when-not (= (dtype/canon kernel-dtype) (dtype/canon (:type value)))
                  (throw (ex-info "physical runtime scalar has the wrong kernel ABI dtype"
                                  {:reason :kernel-executable-scalar-type
                                   :slot slot :expected kernel-dtype
                                   :actual (:type value) :value value})))
                (:value value))
              value)
        physical (cast-runtime-scalar kernel-dtype raw)]
    (when (and (= :bound (:role slot)) (neg? physical))
      (throw (ex-info "kernel extent bound must be non-negative"
                      {:reason :kernel-bound-range :slot slot :value raw})))
    {:type kernel-dtype :value physical}))

(defn typed-runtime-arguments
  "Normalize raw caller scalars to the explicit values required by KernelCall/KernelGraphCall.

   Pointer values remain opaque. A caller may also supply an already typed scalar map; its type
   must agree with the executable ABI. This is the single normalization seam used before runtime
   dispatch selection and binding."
  [executable runtime-arguments]
  (let [executable (validate! executable)
        abi (:abi executable)
        runtime-arguments (kabi/validate-arguments! abi runtime-arguments)]
    (mapv (fn [slot value]
            (if (= :scalar (:kind slot))
              (if (and (map? value) (contains? value :type) (contains? value :value))
                (do
                  (when-not (= (dtype/canon (:dtype slot))
                               (dtype/canon (:type value)))
                    (throw (ex-info "kernel executable scalar argument has the wrong logical ABI dtype"
                                    {:slot slot :expected (:dtype slot)
                                     :actual (:type value) :value value})))
                  (physical-runtime-scalar slot (:value value)))
                (physical-runtime-scalar slot value))
              value))
          abi runtime-arguments)))

(defn common-view
  "Return the logical interface that every dispatch alternative must preserve."
  [executable]
  (let [executable (validate! executable)]
    {:target (target executable)
     :abi (:abi executable)
     :arguments (:arguments executable)
     :effects (:effects executable)}))

(defn graph-bindings
  "Project ABI-ordered runtime arguments into the maps consumed by KernelGraphCall.

   Pointer values remain opaque resident keys/views. External scalar values must already be typed;
   graph-internal derived integer expressions are resolved later from this scalar environment."
  [graph runtime-arguments]
  (let [graph (validate! graph)]
    (when-not (= :kernel-graph (kind graph))
      (throw (ex-info "graph bindings require a KernelGraph executable"
                      {:executable graph :kind (kind graph)})))
    (kabi/validate-arguments! (:abi graph) runtime-arguments)
    (reduce (fn [{:keys [buffers scalar-values] :as bindings} [slot argument value]]
              (if (= :scalar (:kind slot))
                (do
                  (when-not (and (map? value) (contains? value :type) (contains? value :value))
                    (throw (ex-info "kernel graph scalar argument must be explicitly typed"
                                    {:slot slot :argument argument :value value})))
                  (when-not (= (:kernel-dtype slot) (:type value))
                    (throw (ex-info "kernel graph scalar argument has the wrong ABI dtype"
                                    {:slot slot :argument argument :value value})))
                  (assoc bindings :scalar-values (assoc scalar-values argument value)))
                (assoc bindings :buffers (assoc buffers argument value))))
            {:buffers {} :scalar-values {}}
            (map vector (:abi graph) (:arguments graph) runtime-arguments))))
