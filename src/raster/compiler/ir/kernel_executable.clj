(ns raster.compiler.ir.kernel-executable
  "The common executable boundary for one emitted kernel or a scheduled emitted kernel graph.

   This namespace does not erase the distinction between the two values. It exposes only the
   properties that dispatch selection and offline tuning may compare: target, ordered external
   ABI, compiler arguments, logical effects, strategy, and concrete entry-point artifacts. Graph
   temporaries, dependencies, and derived scalars remain graph-owned schedule details."
  (:require [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-graph :as kgraph]))

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

(defn validate!
  "Validate an emitted executable and return it unchanged."
  [executable]
  (case (kind executable)
    :kernel-artifact (kart/validate! executable)
    :kernel-graph
    (let [graph (kgraph/validate! executable)]
      (when-not (kgraph/has-interface? graph)
        (throw (ex-info "emitted kernel graph requires an ordered external ABI"
                        {:graph graph})))
      (when-not (seq (:nodes graph))
        (throw (ex-info "emitted kernel graph requires at least one kernel node" {:graph graph})))
      (let [targets (set (map :target (artifacts graph)))]
        (when-not (= 1 (count targets))
          (throw (ex-info "kernel graph nodes must share one target dialect"
                          {:targets targets}))))
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
           (int value))
    :long (long value)
    :float (float value)
    :double (double value)
    (throw (ex-info "kernel executable has no runtime scalar representation for ABI dtype"
                    {:dtype dtype :value value}))))

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
                  {:type (:kernel-dtype slot)
                   :value (cast-runtime-scalar (:kernel-dtype slot) (:value value))})
                {:type (:kernel-dtype slot)
                 :value (cast-runtime-scalar (:kernel-dtype slot) value)})
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
