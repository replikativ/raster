(ns raster.compiler.ir.link-plan
  "Pure, backend-neutral composition of compiled resident program descriptors.

   A LinkPlan names storage with stable LinkNodes and binds each descriptor's compiler symbols to
   those node identities. Validation closes shapes, dtypes, aliases, scalar environments and
   ordered read/write effects before a runtime session or driver object exists. Runtime allocation,
   executable selection and graph recording deliberately live in raster.gpu.link."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-executable :as kexec]
            [raster.compiler.ir.kernel-graph-call :as kgcall]))

(def node-roles #{:input :constant :state :output :internal :scratch})
(def binder-roles #{:input :constant :state :output :scratch})

(defrecord LinkNode [id view role source])
(defrecord LinkInstance [id descriptor bindings scalars schedule roles])
(defrecord LinkPlan [id target nodes instances outputs aliases attributes])

(defn link-node? [x]
  (and x (= "raster.compiler.ir.link_plan.LinkNode" (.getName (class x)))))

(defn link-instance? [x]
  (and x (= "raster.compiler.ir.link_plan.LinkInstance" (.getName (class x)))))

(defn link-plan? [x]
  (and x (= "raster.compiler.ir.link_plan.LinkPlan" (.getName (class x)))))

(defn- shape-elements [shape]
  (reduce * 1 shape))

(defn- primitive-array-dtype [value]
  (when (and value (.isArray (class value)))
    (case (.getName (.getComponentType (class value)))
      "byte" :byte
      "short" :half
      "int" :int
      "long" :long
      "float" :float
      "double" :double
      nil)))

(defn- memory-segment? [value]
  (instance? java.lang.foreign.MemorySegment value))

(defn- validate-source! [{:keys [id view source] :as node}]
  (when source
    (when-not (bview/contiguous? view)
      (throw (ex-info "a link node initializer requires a contiguous view"
                      {:reason :link-noncontiguous-initializer :node id :view (:id view)})))
    (cond
      (memory-segment? source)
      (when-not (= (long (.byteSize ^java.lang.foreign.MemorySegment source))
                   (long (:byte-length view)))
        (throw (ex-info "a link node MemorySegment initializer has the wrong byte length"
                        {:reason :link-initializer-size :node id
                         :expected (:byte-length view)
                         :actual (.byteSize ^java.lang.foreign.MemorySegment source)})))

      (.isArray (class source))
      (let [source-dtype (primitive-array-dtype source)
            expected-dtype (dtype/canon (:dtype view))
            expected-elements (shape-elements (:shape view))
            actual-elements (java.lang.reflect.Array/getLength source)]
        (when-not source-dtype
          (throw (ex-info "a link node initializer must be a primitive array or MemorySegment"
                          {:reason :link-initializer-type :node id :actual (type source)})))
        (when-not (= expected-dtype source-dtype)
          (throw (ex-info "a link node initializer dtype differs from its view"
                          {:reason :link-initializer-dtype :node id
                           :expected expected-dtype :actual source-dtype})))
        (when-not (= expected-elements actual-elements)
          (throw (ex-info "a link node initializer length differs from its logical shape"
                          {:reason :link-initializer-size :node id
                           :expected expected-elements :actual actual-elements}))))

      :else
      (throw (ex-info "a link node initializer must be a primitive array or MemorySegment"
                      {:reason :link-initializer-type :node id :actual (type source)}))))
  node)

(defn validate-node!
  [node]
  (when-not (link-node? node)
    (throw (ex-info "link plan nodes must be LinkNode values"
                    {:reason :link-node-type :node node :actual (type node)})))
  (when (nil? (:id node))
    (throw (ex-info "a link node requires a stable identity" {:reason :link-node-id})))
  (bview/validate-view! (:view node))
  (when-not (contains? node-roles (:role node))
    (throw (ex-info "a link node has an invalid dataflow role"
                    {:reason :link-node-role :node (:id node) :role (:role node)
                     :allowed node-roles})))
  (when (and (:source node)
             (contains? #{:borrowed :external} (get-in node [:view :allocation :ownership])))
    (throw (ex-info "caller-owned link allocations cannot also carry host initializers"
                    {:reason :link-external-initializer :node (:id node)
                     :ownership (get-in node [:view :allocation :ownership])})))
  (validate-source! node))

(defn validate-node-source!
  "Validate a prospective runtime upload against a LinkNode without changing its ownership or
   initializer contract. Returns `source`."
  [node source]
  (validate-node! node)
  (validate-source! (assoc node :source source))
  source)

(defn node
  "Construct one typed, shaped link node.

   Pass an existing checked `:view`, or describe an allocation with `:dtype`, `:shape`, optional
   `:strides`/`:byte-offset`/`:byte-size`, `:device`, `:memory-space`, `:ownership`, and
   `:allocation-id`. Nodes may share an allocation only through views carrying the same allocation
   identity. `:source` is an optional exact primitive-array/MemorySegment initializer."
  [{:keys [id view role source dtype shape strides byte-offset byte-size device memory-space
           ownership allocation-id alignment coherence]
    :or {role :internal byte-offset 0 memory-space :device ownership :owned}}]
  (let [view (or view
                 (let [shape (vec shape)
                       strides (vec (or strides (bview/dense-strides shape)))
                       span (bview/required-byte-span dtype shape strides)
                       allocation (bview/allocation
                                   {:id (or allocation-id id)
                                    :byte-size (or byte-size (+ byte-offset span))
                                    :memory-space memory-space
                                    :device device
                                    :alignment (or alignment 1)
                                    :coherence (or coherence :device-only)
                                    :ownership ownership})]
                   (bview/view allocation {:id id :byte-offset byte-offset :dtype dtype
                                           :shape shape :strides strides})))]
    (validate-node! (->LinkNode id view role source))))

(defn validate-instance!
  [instance]
  (when-not (link-instance? instance)
    (throw (ex-info "link plan instances must be LinkInstance values"
                    {:reason :link-instance-type :instance instance :actual (type instance)})))
  (let [{:keys [id descriptor bindings scalars schedule roles]} instance]
    (when (nil? id)
      (throw (ex-info "a link instance requires a stable identity" {:reason :link-instance-id})))
    (when-not (and (map? descriptor) (vector? (:steps descriptor))
                   (vector? (:all-params descriptor)) (vector? (:array-params descriptor))
                   (vector? (:scalar-params descriptor)))
      (throw (ex-info "a link instance requires a compiled resident program descriptor"
                      {:reason :link-descriptor :instance id :descriptor descriptor})))
    (when-not (map? bindings)
      (throw (ex-info "link instance bindings must map compiler symbols to node identities"
                      {:reason :link-bindings-type :instance id :bindings bindings})))
    (when-not (map? scalars)
      (throw (ex-info "link instance scalars must be a symbol-to-value map"
                      {:reason :link-scalars-type :instance id :scalars scalars})))
    (when-not (= (set (:scalar-params descriptor)) (set (keys scalars)))
      (throw (ex-info "link instance scalars differ from the descriptor scalar parameters"
                      {:reason :link-scalars :instance id
                       :expected (set (:scalar-params descriptor)) :actual (set (keys scalars))})))
    (when-not (or (nil? schedule) (map? schedule))
      (throw (ex-info "link instance schedule must be nil or resolved schedule data"
                      {:reason :link-schedule :instance id :schedule schedule})))
    (when-not (and (map? roles) (every? binder-roles (vals roles)))
      (throw (ex-info "link instance roles must map compiler symbols to resident roles"
                      {:reason :link-instance-roles :instance id :roles roles
                       :allowed binder-roles}))))
  instance)

(defn instance
  "Construct one descriptor instance. `:bindings` is data—not a name-decoding function—and maps
   every pointer-valued compiler symbol used by the descriptor to a LinkNode identity. `:scalars`
   supplies exactly the descriptor's scalar parameters."
  [{:keys [id descriptor bindings scalars schedule roles] :or {scalars {} roles {}}}]
  (validate-instance! (->LinkInstance id descriptor bindings scalars schedule roles)))

(defn instance-arguments
  "Build the descriptor's ordered runtime argument vector. Array positions are intentionally nil:
   resident binding uses LinkNodes, while scalar/shape closures read their named scalar positions."
  [instance]
  (let [{:keys [descriptor scalars]} (validate-instance! instance)
        array-params (set (:array-params descriptor))]
    (mapv (fn [parameter]
            (if (contains? array-params parameter) nil (get scalars parameter)))
          (:all-params descriptor))))

(defn- pointer-symbols [descriptor]
  (into #{}
        (mapcat (fn [step]
                  (if (= :scatter (:convention step))
                    (:arrays step)
                    (keep (fn [{:keys [kind sym]}]
                            (when (not= :scalar kind) sym))
                          (:argument-specs step)))))
        (:steps descriptor)))

(defn- step-interface [step]
  (or (:artifact step)
      (some-> (:dispatch step) kdispatch/default-alternative)
      (throw (ex-info "a linkable descriptor step has no executable interface"
                      {:reason :link-step-interface :phase (:phase step)
                       :convention (:convention step)}))))

(defn- slot-access [{:keys [kind role]}]
  (cond
    (= :inout role) :read-write
    (= :input kind) :read
    (= :output kind) :write
    :else nil))

(defn- merge-access [left right]
  (if (= left right) left
      (if (or (= :read-write left) (= :read-write right)
              (and left right))
        :read-write
        (or left right))))

(defn- step-selection-override [step schedule]
  (if-let [{:keys [path mapping default]} (:strategy-selection step)]
    (get mapping (get-in schedule path) default)
    (when (= :reduce (:convention step))
      (get-in schedule [:segmented-weighted-reduction :strategy] :auto))))

(defn- abi-step-facts
  [nodes instance step-index step]
  (let [{:keys [id descriptor bindings schedule]} instance
        args (instance-arguments instance)
        interface (kexec/validate! (step-interface step))
        pointer-specs (filterv #(not= :scalar (:kind %)) (:argument-specs step))
        pointer-slots (kabi/pointer-slots (kexec/abi interface))
        logical-names (kabi/pointer-binding-names (kexec/abi interface))
        spec-syms (mapv :sym pointer-specs)]
    (when-not (= (count spec-syms) (count logical-names))
      (throw (ex-info "link descriptor step pointer plan differs from its executable ABI"
                      {:reason :link-step-abi :instance id :step step-index
                       :phase (:phase step) :argument-symbols spec-syms
                       :abi-bindings logical-names})))
    (let [slots-by-logical
          (group-by #(or (:binding %) (:name %)) pointer-slots)
          facts
          (mapv (fn [sym logical-name]
                  (let [node-id (get bindings sym ::missing)
                        node (get nodes node-id)
                        slots (get slots-by-logical logical-name)
                        slot-dtypes (set (map (comp dtype/canon :dtype) slots))]
                    (when (= ::missing node-id)
                      (throw (ex-info "link instance omits a descriptor pointer binding"
                                      {:reason :link-missing-binding :instance id :symbol sym
                                       :phase (:phase step)})))
                    (when-not node
                      (throw (ex-info "link instance binding names an absent node"
                                      {:reason :link-absent-node :instance id :symbol sym
                                       :node node-id :phase (:phase step)})))
                    (when-not (bview/contiguous? (:view node))
                      (throw (ex-info "linked kernel ABI bindings currently require contiguous views"
                                      {:reason :link-noncontiguous-binding :instance id
                                       :symbol sym :node node-id :phase (:phase step)
                                       :shape (get-in node [:view :shape])
                                       :strides (get-in node [:view :strides])})))
                    (when-not (= 1 (count slots))
                      (throw (ex-info "one link node cannot represent a multi-slot composite ABI"
                                      {:reason :link-composite-binding-required :instance id
                                       :symbol sym :node node-id :slots slots
                                       :dtypes slot-dtypes})))
                    (when-not (= (dtype/canon (get-in node [:view :dtype])) (first slot-dtypes))
                      (throw (ex-info "link node dtype differs from the executable ABI"
                                      {:reason :link-node-dtype :instance id :symbol sym :node node-id
                                       :expected (first slot-dtypes)
                                       :actual (get-in node [:view :dtype])})))
                    {:symbol sym :node node-id
                     :access (reduce merge-access nil (map slot-access slots))}))
                spec-syms logical-names)
          runtime-arguments
          (mapv (fn [{:keys [kind type value-fn sym]}]
                  (if (= :scalar kind)
                    {:type type :value (value-fn args)}
                    (get bindings sym)))
                (:argument-specs step))
          selected (if-let [dispatch (:dispatch step)]
                     (kdispatch/select-alternative
                      dispatch runtime-arguments
                      (step-selection-override step (or schedule (:schedule descriptor))))
                     interface)
          scalar-values
          (into {}
                (keep (fn [[slot argument value]]
                        (when (= :scalar (:kind slot)) [argument value])))
                (map vector (kexec/abi selected) (kexec/arguments selected) runtime-arguments))
          node-by-executable-buffer
          (into {}
                (keep (fn [[slot argument value]]
                        (when (not= :scalar (:kind slot)) [argument value])))
                (map vector (kexec/abi selected) (kexec/arguments selected) runtime-arguments))]
      (when (= :kernel-graph (kexec/kind selected))
        (doseq [buffer (concat (:inputs selected) (:outputs selected))
                :let [node-id (get node-by-executable-buffer (:id buffer))
                      node (get nodes node-id)
                      expected (when (some? (:elements buffer))
                                 (kgcall/resolve-integer scalar-values (:elements buffer)))
                      capacity (quot (get-in node [:view :byte-length])
                                     (dtype/bytes-of (get-in node [:view :dtype])))]
                :when expected]
          (when (> (long expected) (long capacity))
            (throw (ex-info "selected kernel graph extent exceeds its linked node view"
                            {:reason :link-node-range :instance id :step step-index
                             :phase (:phase step) :buffer (:id buffer) :node node-id
                             :expected expected :capacity capacity})))))
      {:instance id :step step-index :phase (:phase step) :facts facts})))

(defn- scatter-step-facts
  [nodes instance step-index step]
  (let [{:keys [id bindings]} instance
        [out-sym src-sym index-sym :as symbols] (:arrays step)
        args (instance-arguments instance)
        n (try (long ((:n-fn step) args))
               (catch Exception e
                 (throw (ex-info "link scatter bound did not resolve from its scalar environment"
                                 {:reason :link-scatter-bound :instance id :step step-index
                                  :phase (:phase step)}
                                 e))))
        _ (when-not (= 3 (count symbols))
            (throw (ex-info "a linked scatter step requires output, source and index buffers"
                            {:reason :link-scatter-abi :instance id :step step-index
                             :arrays symbols})))
        node-for
        (fn [symbol]
          (let [node-id (get bindings symbol ::missing)
                link-node (get nodes node-id)]
            (when (= ::missing node-id)
              (throw (ex-info "link instance omits a scatter buffer binding"
                              {:reason :link-missing-binding :instance id :symbol symbol
                               :phase (:phase step)})))
            (when-not link-node
              (throw (ex-info "link scatter binding names an absent node"
                              {:reason :link-absent-node :instance id :symbol symbol
                               :node node-id :phase (:phase step)})))
            (when-not (bview/contiguous? (:view link-node))
              (throw (ex-info "linked scatter bindings require contiguous views"
                              {:reason :link-noncontiguous-binding :instance id :symbol symbol
                               :node node-id :phase (:phase step)})))
            [node-id link-node]))
        [[out-id out] [src-id src] [index-id index]] (mapv node-for symbols)
        capacity (fn [link-node]
                   (quot (get-in link-node [:view :byte-length])
                         (dtype/bytes-of (get-in link-node [:view :dtype]))))]
    (when-not (= (dtype/canon (get-in out [:view :dtype]))
                 (dtype/canon (get-in src [:view :dtype])))
      (throw (ex-info "linked scatter source and output dtypes differ"
                      {:reason :link-node-dtype :instance id :step step-index
                       :output out-id :source src-id
                       :output-dtype (get-in out [:view :dtype])
                       :source-dtype (get-in src [:view :dtype])})))
    (when-not (= :int (dtype/canon (get-in index [:view :dtype])))
      (throw (ex-info "linked scatter index storage must be int32"
                      {:reason :link-node-dtype :instance id :step step-index
                       :index index-id :actual (get-in index [:view :dtype])})))
    (doseq [[node-id link-node] [[src-id src] [index-id index]]]
      (when (> n (capacity link-node))
        (throw (ex-info "linked scatter bound exceeds a source view"
                        {:reason :link-node-range :instance id :step step-index
                         :node node-id :expected n :capacity (capacity link-node)}))))
    {:instance id :step step-index :phase (:phase step)
     :facts [{:symbol out-sym :node out-id :access :write}
             {:symbol src-sym :node src-id :access :read}
             {:symbol index-sym :node index-id :access :read}]}))

(defn- instance-step-facts
  [nodes instance step-index step]
  (if (= :scatter (:convention step))
    (scatter-step-facts nodes instance step-index step)
    (abi-step-facts nodes instance step-index step)))

(defn- canonical-alias-pair [pair]
  (let [pair (set pair)]
    (when-not (= 2 (count pair))
      (throw (ex-info "a link alias declaration must name exactly two distinct nodes"
                      {:reason :link-alias-pair :alias pair})))
    pair))

(defn- validate-plan-structure! [plan]
  (when-not (link-plan? plan)
    (throw (ex-info "expected a LinkPlan value"
                    {:reason :link-plan-type :plan plan :actual (type plan)})))
  (let [{:keys [id target nodes instances outputs aliases attributes]} plan]
    (when (nil? id)
      (throw (ex-info "a link plan requires a stable identity" {:reason :link-plan-id})))
    (when-not (keyword? target)
      (throw (ex-info "a link plan requires a target device keyword"
                      {:reason :link-target :target target})))
    (when-not (and (map? nodes) (seq nodes))
      (throw (ex-info "a link plan requires a non-empty node map"
                      {:reason :link-nodes :nodes nodes})))
    (doseq [[node-id link-node] nodes]
      (validate-node! link-node)
      (when-not (= node-id (:id link-node))
        (throw (ex-info "link node map key differs from the node identity"
                        {:reason :link-node-map-key :key node-id :node (:id link-node)})))
      (let [device (get-in link-node [:view :allocation :device])]
        (when-not (or (nil? device) (= target device))
          (throw (ex-info "link node allocation belongs to a different device"
                          {:reason :link-node-device :node node-id
                           :target target :device device})))))
    (when-not (and (vector? instances) (seq instances))
      (throw (ex-info "a link plan requires an ordered non-empty instance vector"
                      {:reason :link-instances :instances instances})))
    (doseq [link-instance instances] (validate-instance! link-instance))
    (when-not (= (count instances) (count (distinct (map :id instances))))
      (throw (ex-info "link instance identities must be unique"
                      {:reason :link-instance-ids :ids (mapv :id instances)})))
    (when-not (and (vector? outputs) (every? #(contains? nodes %) outputs))
      (throw (ex-info "link plan outputs must be an ordered vector of existing node identities"
                      {:reason :link-outputs :outputs outputs :nodes (set (keys nodes))})))
    (when-not (= (count outputs) (count (distinct outputs)))
      (throw (ex-info "link plan output identities must be unique"
                      {:reason :link-output-duplicates :outputs outputs})))
    (when-not (and (set? aliases) (every? set? aliases))
      (throw (ex-info "link plan aliases must be a set of two-node sets"
                      {:reason :link-aliases :aliases aliases})))
    (when-not (map? attributes)
      (throw (ex-info "link plan attributes must be a map"
                      {:reason :link-attributes :attributes attributes}))))
  plan)

(defn- validate-allocations-and-aliases! [{:keys [nodes aliases] :as plan}]
  (let [by-allocation (group-by #(get-in % [:view :allocation :id]) (vals nodes))]
    (doseq [[allocation-id allocation-nodes] by-allocation]
      (let [contracts (set (map #(get-in % [:view :allocation]) allocation-nodes))
            dtypes (set (map #(dtype/canon (get-in % [:view :dtype])) allocation-nodes))]
        (when-not (= 1 (count contracts))
          (throw (ex-info "one link allocation identity has conflicting contracts"
                          {:reason :link-allocation-contract :allocation allocation-id})))
        (when-not (= 1 (count dtypes))
          (throw (ex-info "runtime link allocations cannot be reinterpreted across dtypes"
                          {:reason :link-allocation-reinterpret :allocation allocation-id
                           :dtypes dtypes})))))
    (doseq [[left-id left] nodes
            [right-id right] nodes
            :when (neg? (compare (pr-str left-id) (pr-str right-id)))
            :when (bview/overlaps? (:view left) (:view right))]
      (let [pair #{left-id right-id}]
        (when-not (contains? aliases pair)
          (throw (ex-info "overlapping link nodes require an explicit alias declaration"
                          {:reason :link-undeclared-alias :nodes pair})))
        (when (and (:source left) (:source right))
          (throw (ex-info "overlapping link nodes cannot carry competing initializers"
                          {:reason :link-alias-initializers :nodes pair})))))
    (doseq [pair aliases
            :let [[left-id right-id] (vec (canonical-alias-pair pair))
                  left (get nodes left-id) right (get nodes right-id)]]
      (when-not (and left right)
        (throw (ex-info "link alias declaration names an absent node"
                        {:reason :link-alias-node :alias pair :nodes (set (keys nodes))})))
      (when-not (bview/overlaps? (:view left) (:view right))
        (throw (ex-info "link alias declaration names disjoint views"
                        {:reason :link-spurious-alias :alias pair}))))
    plan))

(defn- validate-instance-bindings! [nodes instance]
  (let [{:keys [id descriptor bindings roles]} instance
        expected (pointer-symbols descriptor)
        actual (set (keys bindings))]
    (when-not (= expected actual)
      (throw (ex-info "link instance bindings differ from descriptor pointer symbols"
                      {:reason :link-bindings :instance id :expected expected :actual actual
                       :missing (set/difference expected actual)
                       :extra (set/difference actual expected)})))
    (when-not (set/subset? (set (keys roles)) expected)
      (throw (ex-info "link instance roles name symbols outside its pointer bindings"
                      {:reason :link-role-symbols :instance id
                       :extra (set/difference (set (keys roles)) expected)})))
    (doseq [[sym node-id] bindings]
      (when-not (contains? nodes node-id)
        (throw (ex-info "link instance binding names an absent node"
                        {:reason :link-absent-node :instance id :symbol sym :node node-id})))
      (when (and (= :constant (get roles sym))
                 (not= :constant (get-in nodes [node-id :role])))
        (throw (ex-info "a constant instance binding requires a constant LinkNode"
                        {:reason :link-role-mismatch :instance id :symbol sym :node node-id
                         :node-role (get-in nodes [node-id :role])}))))
    (let [args (instance-arguments instance)]
      (doseq [{:keys [sym dtype size-fn]} (:allocs descriptor)]
        (let [node-id (get bindings sym)
              link-node (get nodes node-id)
              expected-elements
              (try (long (size-fn args))
                   (catch Exception e
                     (throw (ex-info "link descriptor allocation extent did not resolve from scalars"
                                     {:reason :link-allocation-shape :instance id :symbol sym}
                                     e))))
              actual-elements (shape-elements (get-in link-node [:view :shape]))]
          (when-not (= (dtype/canon dtype) (dtype/canon (get-in link-node [:view :dtype])))
            (throw (ex-info "link internal node dtype differs from its descriptor allocation"
                            {:reason :link-allocation-dtype :instance id :symbol sym :node node-id
                             :expected dtype :actual (get-in link-node [:view :dtype])})))
          (when-not (= expected-elements actual-elements)
            (throw (ex-info "link internal node shape differs from its descriptor allocation"
                            {:reason :link-allocation-shape :instance id :symbol sym :node node-id
                             :expected-elements expected-elements
                             :actual-elements actual-elements})))))
      (mapv (fn [[step-index step]]
              (instance-step-facts nodes instance step-index step))
            (map-indexed vector (:steps descriptor))))))

(defn- validate-effects! [{:keys [target nodes instances outputs aliases] :as plan}]
  (when (and (not (let [target-name (name target)]
                    (or (= "ze" target-name) (.startsWith target-name "ze:"))))
             (some #(= :scatter (:convention %))
                   (mapcat (comp :steps :descriptor) instances)))
    (throw (ex-info "linked scatter execution is not available on this target backend"
                    {:reason :link-target-convention :target target :convention :scatter})))
  (let [step-facts (mapcat #(validate-instance-bindings! nodes %) instances)
        initialized (volatile! (into #{}
                                     (keep (fn [[id {:keys [role source view]}]]
                                             (when (or source
                                                       (contains? #{:input :constant :state} role)
                                                       (contains? #{:borrowed :external}
                                                                  (get-in view [:allocation :ownership])))
                                               id)))
                                     nodes))
        written (volatile! #{})]
    (doseq [{:keys [instance step phase facts]} step-facts]
      (let [by-node (reduce (fn [m {:keys [node access]}]
                              (update m node merge-access access)) {} facts)]
        (doseq [[node-id access] by-node
                :let [role (get-in nodes [node-id :role])]]
          (when (and (contains? #{:read :read-write} access)
                     (not (contains? @initialized node-id)))
            (throw (ex-info "link plan reads an internal node before an ordered producer writes it"
                            {:reason :link-read-before-write :instance instance :step step
                             :phase phase :node node-id})))
          (when (and (contains? #{:write :read-write} access)
                     (contains? #{:input :constant} role))
            (throw (ex-info "link plan writes a read-only node"
                            {:reason :link-write-read-only :instance instance :step step
                             :phase phase :node node-id :role role})))
          (when (contains? #{:write :read-write} access)
            (vswap! initialized conj node-id)
            (vswap! written conj node-id)))
        (doseq [[left-id left-access] by-node
                [right-id right-access] by-node
                :when (neg? (compare (pr-str left-id) (pr-str right-id)))
                :when (contains? aliases #{left-id right-id})
                :when (or (contains? #{:write :read-write} left-access)
                          (contains? #{:write :read-write} right-access))]
          (throw (ex-info "one linked kernel step cannot access overlapping aliases when either writes"
                          {:reason :link-same-step-alias-hazard :instance instance :step step
                           :phase phase :nodes #{left-id right-id}})))))
    (doseq [node-id outputs]
      (when-not (or (contains? @written node-id)
                    (contains? @initialized node-id))
        (throw (ex-info "link plan exports a node with no value"
                        {:reason :link-unproduced-output :node node-id}))))
    plan))

(defn validate!
  "Validate a LinkPlan without allocating storage, registering kernels, or contacting a driver."
  [plan]
  (-> plan validate-plan-structure! validate-allocations-and-aliases! validate-effects!))

(defn make
  "Construct and purely validate a LinkPlan.

   `:nodes` may be a node-id map or an ordered collection of LinkNodes. `:aliases` is an explicit
   collection of two-node collections for overlapping views; undeclared physical overlap fails.
   Instance order and each descriptor's step order define the current serial dependency schedule."
  [{:keys [id target nodes instances outputs aliases attributes]
    :or {outputs [] aliases #{} attributes {}}}]
  (let [nodes (if (map? nodes) nodes (into {} (map (juxt :id identity)) nodes))
        aliases (into #{} (map canonical-alias-pair) aliases)]
    (validate! (->LinkPlan id target nodes (vec instances) (vec outputs) aliases attributes))))

(defn instance-roles
  "Resolve compiler-symbol roles for the runtime binder. Only `:constant` affects executable
   prologue hoisting today; the complete role map remains data for residency/lifetime policy."
  [plan instance]
  (when-not (link-plan? plan)
    (throw (ex-info "instance-roles requires a LinkPlan" {:actual (type plan)})))
  (validate-instance! instance)
  (let [nodes (:nodes plan)]
    (merge
     (into {}
           (map (fn [[sym node-id]]
                  [sym (case (get-in nodes [node-id :role])
                         :internal :scratch
                         (get-in nodes [node-id :role]))]))
           (:bindings instance))
     (:roles instance))))
