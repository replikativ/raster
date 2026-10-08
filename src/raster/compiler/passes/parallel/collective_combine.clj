(ns raster.compiler.passes.parallel.collective-combine
  "Project certified collective arithmetic to ordinary functional TypedSOAC.

   This is a local binary combine, not an all-reduce implementation or transport. Distributed
   refinement still owns contribution ancestry, placement, copies and numerical reassociation.
   Scheduling, target emission and executable validation use the existing compiler boundaries."
  (:require [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.buffer-view :as view]
            [raster.compiler.ir.emitted-parallel-equation :as emitted-equation]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.ir.scan :as scan]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.typed-soac-route :as typed-route]
            [raster.compiler.passes.parallel.structured-control-route :as structured-route]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.backend.gpu.parallel-program-c-family :as c-family]
            [raster.compiler.backend.gpu.target :as target]))

(defn algorithm
  "Derive one dense elementwise combine with explicit scalar types and immutable result.
   The retained certificate supplies the operator; there is no collective operator registry.
   The certificate's original element expression is not copied: operands are the two arrays."
  [algebra elements]
  (scan/validate! algebra)
  (when-not (and (integer? elements) (pos? elements) (<= elements Long/MAX_VALUE))
    (throw (ex-info "collective combine requires a positive representable element count"
                    {:reason :collective-combine-extent :elements elements})))
  (let [value (av/tensor {:dtype (:dtype algebra) :shape [elements]})
        equation (list '= 'combine '[result]
                       (list 'map {:index 'i :extent elements} '[left right] []
                             (soac/lambda-form
                              '[lhs rhs]
                              [(soac/local-value 'combined (:dtype algebra)
                                                 (list (:combine algebra) 'lhs 'rhs))]
                              '[combined])))]
    (soac/make
     (soac/default-program-facts
      {:values {'left value 'right value 'result value}
       :inputs '[left right]
       :equations {'combine (soac/default-equation-facts)}})
     [equation] '[result])))

(defn- target-options! [{:keys [target-device target-descriptor target-dialect]}]
  (let [dialect (target/kernel-body-c-dialect target-descriptor)]
    (when-not (and (keyword? target-device)
                   (= target-device (:device-id target-descriptor))
                   dialect (or (nil? target-dialect) (= dialect target-dialect)))
      (throw (ex-info "collective combine needs one matching frozen GPU target snapshot"
                      {:reason :collective-combine-target :target-device target-device
                       :descriptor-device (:device-id target-descriptor)
                       :declared-dialect target-dialect :derived-dialect dialect})))
    {:target-device target-device :target-descriptor target-descriptor
     :target-dialect dialect}))

(defn- schedule [algorithm options]
  (structured-route/schedule-program
   (assoc (typed-route/program-envelope algorithm) :dialect :typed-parallel) options))

(defn emit
  "Emit a local combine through the ordinary TypedSOAC/SegMap/KernelBody vertical.
   Returns the existing checked EmittedParallelEquation, not a new executable convention.
   Target descriptor/device/dialect are supplied by the enclosing compiler's resolved options."
  [algebra elements options]
  (let [options (assoc (target-options! options) :dtype (:dtype algebra))
        scheduled (schedule (algorithm algebra elements) options)
        emitted (get-in (c-family/emit-program scheduled options)
                        [:program :equations 0 :operations 0])]
    (assoc-in emitted [:attributes :collective-target] (dissoc options :dtype))))

(defn validate!
  "Bind an emitted local combine to its exact declared algebra and extent.
   Generic emitted-equation validation alone proves its own algorithm, not the caller's monoid."
  [algebra elements emitted]
  (let [expected (algorithm algebra elements)]
    (emitted-equation/validate! emitted)
    (when-not (= expected (:algorithm emitted))
      (throw (ex-info "emitted combine differs from its declared typed monoid"
                      {:reason :collective-combine-algorithm
                       :expected expected :actual (:algorithm emitted)})))
    ;; Generic emission validates the retained schedule and artifact against each other. Bind
    ;; that schedule back to the caller's algebra too: a valid product schedule is not a sum
    ;; merely because both have the same memory interface. Reuse the exact frozen target facts.
    (let [options (assoc (target-options! (get-in emitted [:attributes :collective-target]))
                         :dtype (:dtype algebra))
          scheduled (schedule expected options)
          body (:body (equation-graph/make-for-equation scheduled (first (:equations scheduled))))
          expected-operations (mapcat :operations (:equations body))
          actual-operations (mapcat :operations (get-in emitted [:body :equations]))]
      (when-not (and (= (:target-dialect options) (get-in emitted [:provenance :target-dialect]))
                     (every? #(= (:target-dialect options)
                                  (get-in % [:operation :provenance :target-dialect]))
                             (get-in emitted [:graph :nodes])))
        (throw (ex-info "combine target emission differs from its retained target snapshot"
                        {:reason :collective-combine-target})))
      ;; SegSpace gives its private flattened index a fresh name. Rebind only that generated
      ;; binder; all logical dimensions, scalar regions, operand sets and launch grids must agree.
      (when-not (and (= (count expected-operations) (count actual-operations))
                     (every? true?
                             (map (fn [expected actual]
                                    (= (assoc-in expected [:space :flat-idx]
                                                 (get-in actual [:space :flat-idx])) actual))
                                  expected-operations actual-operations)))
        (throw (ex-info "combine schedule differs from the declared typed scalar algorithm"
                        {:reason :collective-combine-schedule}))))
    emitted))

(defn- local-plan [algebra elements emitted {:keys [id nodes] :as request}]
  (when-not (and (map? request) (= #{:id :nodes} (set (keys request))) (some? id)
                 (map? nodes) (= #{:left :right :result} (set (keys nodes)))
                 (every? link/link-node? (vals nodes))
                 (= 3 (count (distinct (map :id (vals nodes))))))
    (throw (ex-info "local combine needs a closed three-node storage contract"
                    {:reason :collective-combine-storage})))
  (doseq [[role node] nodes]
    (link/validate-node! node)
    (when-not (and (= (:dtype algebra) (dtype/canon (get-in node [:view :dtype])))
                   (view/contiguous? (:view node))
                   (= elements (reduce *' 1 (get-in node [:view :shape])))
                   (or (not= :result role) (nil? (:source node))))
      (throw (ex-info "combine storage requires exact dense typed extents and a fresh output"
                      {:reason :collective-combine-storage :role role :node (:id node)}))))
  (doseq [input [:left :right]]
    (when (view/overlaps? (get-in nodes [input :view]) (get-in nodes [:result :view]))
      (throw (ex-info "combine output cannot alias an immutable operand"
                      {:reason :collective-combine-storage :operand input}))))
  (link/make
   {:id id :target (get-in emitted [:attributes :collective-target :target-device])
    :nodes (mapv (fn [[role node]]
                   (assoc node :role (if (= :result role) :output :input))) nodes)
    :values (mapv (fn [[_ node]]
                    (link/value {:id (:id node)
                                 :abstract (av/tensor {:dtype (:dtype algebra)
                                                       :shape (get-in node [:view :shape])})
                                 :leaves [{:name :value :node (:id node)}]})) nodes)
    :instances [(link/graph-instance
                 {:id :combine :graph (:graph emitted)
                  :bindings (into {} (map (fn [[role node]] [(symbol (name role)) (:id node)])) nodes)})]
    :outputs [(:id (:result nodes))]}))

(defn bind-local
  "Emit a certified combine and bind its immutable operands/fresh result to ordinary LinkNodes.
   Storage identity and shape come from the enclosing compiler, not from ABI-name discovery.
   Dense tensors may retain their rank: the generated map addresses their ordered flat cells.
   Returns existing compiler boundaries; no allocation, upload or new runtime convention."
  [algebra elements options request]
  (let [emitted (validate! algebra elements (emit algebra elements options))]
    {:emitted emitted :link-plan (local-plan algebra elements emitted request)}))

(defn validate-local!
  "Independently bind an emitted combine and LinkPlan to the requested semantic storage roles.
   The caller must supply its original storage request, not infer roles from the candidate plan."
  [algebra elements request {:keys [emitted link-plan] :as linked}]
  (when-not (and (map? linked) (= #{:emitted :link-plan} (set (keys linked))))
    (throw (ex-info "expected closed emitted-combine and local-plan boundaries"
                    {:reason :collective-combine-local-plan})))
  (validate! algebra elements emitted)
  (link/validate! link-plan)
  (when-not (= (local-plan algebra elements emitted request) link-plan)
    (throw (ex-info "combine LinkPlan differs from the semantic storage request"
                    {:reason :collective-combine-local-plan})))
  linked)
