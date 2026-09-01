(ns raster.compiler.ir.link-plan
  "Pure, backend-neutral composition of compiled resident descriptors and emitted programs.

   A LinkPlan names storage with stable LinkNodes and binds each descriptor's compiler symbols to
   those node identities. Validation closes shapes, dtypes, aliases, scalar environments and
   ordered read/write effects before a runtime session or driver object exists. Runtime allocation,
   executable selection and graph recording deliberately live in raster.gpu.link."
  (:require [clojure.set :as set]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.ir.abstract-value :as av]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.emitted-parallel-program-call :as program-call]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-executable :as kexec]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-graph-call :as kgcall]
            [raster.compiler.ir.structured-loop-call :as loop-call]))

(def node-roles #{:input :constant :state :output :internal :scratch})
(def binder-roles #{:input :constant :state :output :scratch})

(defrecord LinkNode [id view role source])
(defrecord LinkValue [id abstract physical-layout leaves])
(defrecord LinkInstance [id descriptor bindings scalars schedule roles arguments])
(defrecord ProgramLinkInstance [id call roles attributes])
(defrecord LinkPlan [id target nodes values instances outputs aliases attributes])

(defn link-node? [x]
  (and x (= "raster.compiler.ir.link_plan.LinkNode" (.getName (class x)))))

(defn link-instance? [x]
  (and x (= "raster.compiler.ir.link_plan.LinkInstance" (.getName (class x)))))

(defn program-link-instance? [x]
  (and x (= "raster.compiler.ir.link_plan.ProgramLinkInstance" (.getName (class x)))))

(defn plan-instance? [x]
  (or (link-instance? x) (program-link-instance? x)))

(defn link-plan? [x]
  (and x (= "raster.compiler.ir.link_plan.LinkPlan" (.getName (class x)))))

(defn link-value? [x]
  (and x (= "raster.compiler.ir.link_plan.LinkValue" (.getName (class x)))))

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

(defn value
  "Construct one logical value realized by ordered physical LinkNode leaves.

   Each leaf is `{:name field-id :node node-id}`. The order is the physical ABI order; names are
   independently checked against KernelABI `:field` identities so equal-dtype fields cannot be
   silently swapped. `:physical-layout` identifies the realized packing independently of the
   AbstractValue's logical layout and numerical representation."
  [{:keys [id abstract physical-layout leaves]}]
  (when (nil? id)
    (throw (ex-info "a link value requires a stable identity" {:reason :link-value-id})))
  (av/validate! abstract)
  (when-not (or (nil? physical-layout) (map? physical-layout))
    (throw (ex-info "link value physical layout must be a descriptor map or nil"
                    {:reason :link-value-physical-layout :value id
                     :physical-layout physical-layout})))
  (when-not (and (vector? leaves) (seq leaves)
                 (every? #(and (map? %) (:name %) (contains? % :node)) leaves))
    (throw (ex-info "a link value requires ordered named physical leaves"
                    {:reason :link-value-leaves :value id :leaves leaves})))
  (let [names (mapv :name leaves)
        nodes (mapv :node leaves)]
    (when-not (= (count names) (count (distinct names)))
      (throw (ex-info "link value leaf names must be unique"
                      {:reason :link-value-leaf-names :value id :names names})))
    (when-not (= (count nodes) (count (distinct nodes)))
      (throw (ex-info "one link value cannot repeat a physical node"
                      {:reason :link-value-leaf-nodes :value id :nodes nodes}))))
  (->LinkValue id abstract
               (or physical-layout {:kind (if (= 1 (count leaves)) :dense :ordered-fields)})
               leaves))

(defn- node-abstract-value
  [node]
  (let [view (bview/validate-view! (:view node))
        allocation (:allocation view)]
    (av/tensor {:dtype (:dtype view)
                :shape (:shape view)
                :logical-layout {:strides (:strides view)}
                :memory-space (:memory-space allocation)
                :placement {:device (:device allocation)}
                :ownership (:ownership allocation)})))

(defn- implicit-value
  [node]
  (value {:id (:id node)
          :abstract (node-abstract-value node)
          :leaves [{:name :value :node (:id node)}]}))

(defn- normalize-values
  [nodes values]
  (let [explicit (cond
                   (nil? values) {}
                   (map? values) values
                   :else (into {} (map (juxt :id identity)) values))
        _ (doseq [[value-id link-value] explicit]
            (when-not (link-value? link-value)
              (throw (ex-info "link plan values must be LinkValue records"
                              {:reason :link-value-type :value value-id
                               :actual (type link-value)})))
            (when-not (= value-id (:id link-value))
              (throw (ex-info "link value map key differs from the value identity"
                              {:reason :link-value-map-key :key value-id
                               :value (:id link-value)})))
            (value link-value))
        claimed (mapcat (fn [[_ link-value]] (map :node (:leaves link-value))) explicit)
        frequencies (frequencies claimed)]
    (when-let [[node-id count] (first (filter (fn [[_ count]] (< 1 count)) frequencies))]
      (throw (ex-info "one physical node cannot belong to multiple logical values"
                      {:reason :link-value-overlap :node node-id :claims count})))
    (merge explicit
           (into {}
                 (keep (fn [[node-id link-node]]
                         (when-not (contains? frequencies node-id)
                           [node-id (implicit-value link-node)])))
                 nodes))))

(defn value-node-ids
  "Ordered physical LinkNode identities realizing `value-id`."
  [plan value-id]
  (let [link-value (get (:values plan) value-id)]
    (when-not link-value
      (throw (ex-info "link value identity is absent"
                      {:reason :link-absent-value :value value-id
                       :values (set (keys (:values plan)))})))
    (mapv :node (:leaves link-value))))

(defn output-value-ids
  "Return public logical value identities in physical output order. LinkPlan validation guarantees
   that each selected value contributes all of its leaves contiguously and in field order."
  [plan]
  (let [positions (zipmap (:outputs plan) (range))]
    (->> (:values plan)
         (keep (fn [[value-id value]]
                 (let [leaf-positions (mapv positions (map :node (:leaves value)))]
                   (when (every? some? leaf-positions)
                     [value-id (first leaf-positions)]))))
         (sort-by second)
         (mapv first))))

(defn validate-instance!
  [instance]
  (when-not (link-instance? instance)
    (throw (ex-info "link plan instances must be LinkInstance values"
                    {:reason :link-instance-type :instance instance :actual (type instance)})))
  (let [{:keys [id descriptor bindings scalars schedule roles arguments]} instance]
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
    (when arguments
      (when-not (and (vector? arguments) (= (count (:all-params descriptor)) (count arguments)))
        (throw (ex-info "link instance specialization arguments must follow descriptor order"
                        {:reason :link-instance-arguments :instance id
                         :parameters (:all-params descriptor)
                         :expected (count (:all-params descriptor))
                         :actual (when (sequential? arguments) (count arguments))})))
      (let [environment (zipmap (:all-params descriptor) arguments)]
        (when-not (= scalars (select-keys environment (:scalar-params descriptor)))
          (throw (ex-info "link instance scalars differ from its specialization arguments"
                          {:reason :link-instance-argument-scalars :instance id
                           :scalars scalars
                           :arguments (select-keys environment (:scalar-params descriptor))})))))
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
   every pointer-valued compiler symbol used by the descriptor to a LinkValue identity. `:scalars`
   supplies exactly the descriptor's scalar parameters. Optional ordered `:arguments` retains the
   complete compile-time specialization environment for descriptor closures whose shapes depend on
   array parameters; runtime pointer binding still comes exclusively from `:bindings`."
  [{:keys [id descriptor bindings scalars schedule roles arguments] :or {scalars {} roles {}}}]
  (validate-instance! (->LinkInstance id descriptor bindings scalars schedule roles arguments)))

(defn validate-program-instance!
  [instance]
  (when-not (program-link-instance? instance)
    (throw (ex-info "expected a ProgramLinkInstance"
                    {:reason :program-link-instance-type
                     :instance instance :actual (type instance)})))
  (let [{:keys [id call roles attributes]} instance
        call (program-call/validate! call)
        buffer-values (set (keys (:buffers call)))]
    (when (nil? id)
      (throw (ex-info "a program link instance requires a stable identity"
                      {:reason :program-link-instance-id})))
    (when-not (and (map? roles) (every? binder-roles (vals roles)))
      (throw (ex-info "program link roles must map compiler values to resident roles"
                      {:reason :program-link-instance-roles :instance id
                       :roles roles :allowed binder-roles})))
    (when-not (set/subset? (set (keys roles)) buffer-values)
      (throw (ex-info "program link roles name values outside its resident buffer boundary"
                      {:reason :program-link-instance-role-values :instance id
                       :extra (set/difference (set (keys roles)) buffer-values)})))
    (when-not (map? attributes)
      (throw (ex-info "program link instance attributes must be a map"
                      {:reason :program-link-instance-attributes
                       :instance id :attributes attributes}))))
  instance)

(defn program-instance
  "Construct an equation-first emitted program instance whose call buffers are LinkValue IDs.

   KernelGraph temporaries remain graph-private. `:roles` may refine residency policy for public
   program buffer values but never supplies or reconstructs a binding."
  [{:keys [id call roles attributes] :or {roles {} attributes {}}}]
  (validate-program-instance! (->ProgramLinkInstance id call roles attributes)))

(defn instance-arguments
  "Return the descriptor's ordered specialization arguments. Explicit arguments retain array
   values for descriptor shape closures; absent arguments preserve the hand-built-plan
   shorthand of nil array positions plus exact named scalars. Runtime pointers are never taken
   from this vector: resident binding uses certified LinkValue leaves exclusively."
  [instance]
  (let [{:keys [descriptor scalars arguments]} (validate-instance! instance)
        array-params (set (:array-params descriptor))]
    (or arguments
        (mapv (fn [parameter]
                (if (contains? array-params parameter) nil (get scalars parameter)))
              (:all-params descriptor)))))

(defn descriptor-pointer-symbols
  "Return the exact set of compiler symbols that the resident descriptor binds as pointers.

   This is part of the public descriptor-to-LinkPlan conversion boundary. Consumers must use it
   instead of independently reconstructing pointer membership from names or parameter roles."
  [descriptor]
  (into #{}
        (mapcat (fn [step]
                  (if (= :scatter (:convention step))
                    (:arrays step)
                    (keep (fn [{:keys [kind sym]}]
                            (when (not= :scalar kind) sym))
                          (:argument-specs step)))))
        (:steps descriptor)))

(defn descriptor-allocation-leaves
  "Return one descriptor allocation's ordered physical leaf specifications. Legacy flat
   allocations become a single `:value` leaf; composite allocations declare `:leaves` containing
   `:field`, `:dtype`, and `:size-fn`."
  ([allocation] (descriptor-allocation-leaves allocation nil))
  ([{:keys [sym dtype size-fn leaves] :as allocation} default-dtype]
   (let [leaves (if (contains? allocation :leaves)
                  leaves
                  [{:field :value :dtype (or dtype default-dtype) :size-fn size-fn}])]
     (when-not (and (vector? leaves) (seq leaves))
       (throw (ex-info "descriptor allocation requires ordered physical leaves"
                       {:reason :link-allocation-leaves :symbol sym :leaves leaves})))
     (doseq [{:keys [field dtype size-fn] :as leaf} leaves]
       (when (or (nil? field) (nil? dtype) (not (ifn? size-fn)))
         (throw (ex-info "descriptor allocation leaf requires :field, :dtype, and :size-fn"
                         {:reason :link-allocation-leaf :symbol sym :leaf leaf}))))
     (let [fields (mapv :field leaves)]
       (when-not (= (count fields) (count (distinct fields)))
         (throw (ex-info "descriptor allocation fields must be unique and ordered"
                         {:reason :link-allocation-fields :symbol sym :fields fields}))))
     leaves)))

(defn- step-interface [step]
  (or (:artifact step)
      (some-> (:dispatch step) kdispatch/default-alternative)
      (throw (ex-info "a linkable descriptor step has no executable interface"
                      {:reason :link-step-interface :phase (:phase step)
                       :convention (:convention step)}))))

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
  [nodes values instance step-index step]
  (let [{:keys [id descriptor bindings schedule]} instance
        args (instance-arguments instance)
        interface (kexec/validate! (step-interface step))
        pointer-specs (filterv #(not= :scalar (:kind %)) (:argument-specs step))
        slot-groups (kabi/logical-pointer-slot-groups (kexec/abi interface))
        logical-names (mapv :binding slot-groups)
        spec-syms (mapv :sym pointer-specs)]
    (when-not (= (count spec-syms) (count logical-names))
      (throw (ex-info "link descriptor step pointer plan differs from its executable ABI"
                      {:reason :link-step-abi :instance id :step step-index
                       :phase (:phase step) :argument-symbols spec-syms
                       :abi-bindings logical-names})))
    (let [binding-facts
          (mapv (fn [sym {:keys [slots]}]
                  (let [value-id (get bindings sym ::missing)
                        link-value (get values value-id)
                        leaves (:leaves link-value)]
                    (when (= ::missing value-id)
                      (throw (ex-info "link instance omits a descriptor pointer binding"
                                      {:reason :link-missing-binding :instance id :symbol sym
                                       :phase (:phase step)})))
                    (when-not link-value
                      (throw (ex-info "link instance binding names an absent logical value"
                                      {:reason :link-absent-value :instance id :symbol sym
                                       :value value-id :phase (:phase step)})))
                    (when-not (= (count slots) (count leaves))
                      (throw (ex-info "logical value leaf count differs from its physical ABI group"
                                      {:reason :link-value-abi-leaves :instance id :symbol sym
                                       :value value-id :slots slots :leaves leaves})))
                    (when (and (< 1 (count leaves)) (not (:logical-bindings? step)))
                      (throw (ex-info "composite link value requires logical descriptor binding"
                                      {:reason :link-composite-binding-mode :instance id
                                       :symbol sym :value value-id :phase (:phase step)})))
                    (let [leaf-facts
                          (mapv
                           (fn [slot {:keys [name node] :as leaf}]
                             (let [link-node (get nodes node)]
                               (when-not link-node
                                 (throw (ex-info "link value leaf names an absent physical node"
                                                 {:reason :link-value-leaf-node :instance id
                                                  :symbol sym :value value-id :leaf leaf})))
                               (when-not (bview/contiguous? (:view link-node))
                                 (throw (ex-info "linked kernel ABI bindings currently require contiguous views"
                                                 {:reason :link-noncontiguous-binding :instance id
                                                  :symbol sym :node node :phase (:phase step)
                                                  :shape (get-in link-node [:view :shape])
                                                  :strides (get-in link-node [:view :strides])})))
                               (when (and (:field slot) (not= (:field slot) name))
                                 (throw (ex-info "link value physical field order differs from its ABI"
                                                 {:reason :link-value-abi-field :instance id
                                                  :symbol sym :value value-id :node node
                                                  :expected (:field slot) :actual name})))
                               (when-not (= (dtype/canon (get-in link-node [:view :dtype]))
                                            (dtype/canon (:dtype slot)))
                                 (throw (ex-info "link node dtype differs from the executable ABI"
                                                 {:reason :link-node-dtype :instance id :symbol sym
                                                  :value value-id :node node
                                                  :expected (:dtype slot)
                                                  :actual (get-in link-node [:view :dtype])})))
                               {:symbol sym :value value-id :node node
                                :field name :access (kabi/slot-access slot)}))
                           slots leaves)]
                      {:symbol sym :value value-id :slots slots :leaves leaves
                       :facts leaf-facts})))
                spec-syms slot-groups)
          physical-pointers (vec (mapcat (comp (partial map :node) :leaves) binding-facts))
          scalar-arguments
          (mapv (fn [{:keys [type value-fn]}] {:type type :value (value-fn args)})
                (filterv #(= :scalar (:kind %)) (:argument-specs step)))
          runtime-arguments
          (loop [slots (kexec/abi interface)
                 pointers physical-pointers scalars scalar-arguments result []]
            (if-let [slot (first slots)]
              (if (= :scalar (:kind slot))
                (recur (next slots) pointers (next scalars) (conj result (first scalars)))
                (recur (next slots) (next pointers) scalars (conj result (first pointers))))
              (vec result)))
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
      {:instance id :step step-index :phase (:phase step)
       :facts (vec (mapcat :facts binding-facts))})))

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
  [nodes values instance step-index step]
  (if (= :scatter (:convention step))
    (scatter-step-facts nodes instance step-index step)
    (abi-step-facts nodes values instance step-index step)))

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
  (let [{:keys [id target nodes values instances outputs aliases attributes]} plan]
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
    (when-not (and (map? values) (seq values))
      (throw (ex-info "a link plan requires a non-empty logical value map"
                      {:reason :link-values :values values})))
    (let [claimed (volatile! #{})]
      (doseq [[value-id link-value] values]
        (value link-value)
        (when-not (= value-id (:id link-value))
          (throw (ex-info "link value map key differs from the value identity"
                          {:reason :link-value-map-key :key value-id
                           :value (:id link-value)})))
        (let [leaf-nodes (mapv :node (:leaves link-value))
              leaf-records (mapv nodes leaf-nodes)
              roles (set (map :role leaf-records))
              ownerships (set (map #(get-in % [:view :allocation :ownership]) leaf-records))
              memory-spaces (set (map #(get-in % [:view :allocation :memory-space]) leaf-records))
              devices (set (map #(get-in % [:view :allocation :device]) leaf-records))]
          (when-not (every? some? leaf-records)
            (throw (ex-info "link value leaf names an absent physical node"
                            {:reason :link-value-leaf-node :value value-id
                             :leaves leaf-nodes :nodes (set (keys nodes))})))
          (when-let [overlap (seq (set/intersection @claimed (set leaf-nodes)))]
            (throw (ex-info "one physical node cannot belong to multiple logical values"
                            {:reason :link-value-overlap :value value-id :nodes (set overlap)})))
          (vswap! claimed into leaf-nodes)
          (when-not (= 1 (count roles))
            (throw (ex-info "one logical value requires one dataflow role across its leaves"
                            {:reason :link-value-roles :value value-id :roles roles})))
          (when-not (= 1 (count ownerships))
            (throw (ex-info "one logical value requires atomic ownership across its leaves"
                            {:reason :link-value-ownership :value value-id
                             :ownerships ownerships})))
          (when-not (= 1 (count memory-spaces))
            (throw (ex-info "one logical value requires one memory space across its leaves"
                            {:reason :link-value-memory-spaces :value value-id
                             :memory-spaces memory-spaces})))
          (when-not (= 1 (count devices))
            (throw (ex-info "one logical value cannot span devices before sharding lowering"
                            {:reason :link-value-devices :value value-id :devices devices})))
          (let [abstract (:abstract link-value)
                abstract-device (get-in abstract [:placement :device])]
            (when (and (:ownership abstract)
                       (not= #{(:ownership abstract)} ownerships))
              (throw (ex-info "logical ownership differs from its physical realization"
                              {:reason :link-value-abstract-ownership :value value-id
                               :abstract (:ownership abstract) :physical ownerships})))
            (when (and (:memory-space abstract)
                       (not= #{(:memory-space abstract)} memory-spaces))
              (throw (ex-info "logical memory space differs from its physical realization"
                              {:reason :link-value-abstract-memory-space :value value-id
                               :abstract (:memory-space abstract) :physical memory-spaces})))
            (when (and abstract-device (not= #{abstract-device} devices))
              (throw (ex-info "logical placement differs from its physical realization"
                              {:reason :link-value-abstract-device :value value-id
                               :abstract abstract-device :physical devices}))))))
      (when-not (= @claimed (set (keys nodes)))
        (throw (ex-info "every physical LinkNode must realize exactly one logical value"
                        {:reason :link-value-coverage :claimed @claimed
                         :nodes (set (keys nodes))}))))
    (when-not (and (vector? instances) (seq instances))
      (throw (ex-info "a link plan requires an ordered non-empty instance vector"
                      {:reason :link-instances :instances instances})))
    (doseq [link-instance instances]
      (cond
        (link-instance? link-instance) (validate-instance! link-instance)
        (program-link-instance? link-instance) (validate-program-instance! link-instance)
        :else
        (throw (ex-info "link plan contains an unknown instance variant"
                        {:reason :link-instance-type :instance link-instance
                         :actual (type link-instance)}))))
    (when-not (= (count instances) (count (distinct (map :id instances))))
      (throw (ex-info "link instance identities must be unique"
                      {:reason :link-instance-ids :ids (mapv :id instances)})))
    (when-not (and (vector? outputs) (every? #(contains? nodes %) outputs))
      (throw (ex-info "link plan outputs must be an ordered vector of existing node identities"
                      {:reason :link-outputs :outputs outputs :nodes (set (keys nodes))})))
    (when-not (= (count outputs) (count (distinct outputs)))
      (throw (ex-info "link plan output identities must be unique"
                      {:reason :link-output-duplicates :outputs outputs})))
    (let [positions (zipmap outputs (range))]
      (doseq [[value-id link-value] values
              :let [leaf-nodes (mapv :node (:leaves link-value))
                    leaf-positions (vec (keep positions leaf-nodes))]
              :when (seq leaf-positions)]
        (when-not (and (= (count leaf-nodes) (count leaf-positions))
                       (= leaf-positions
                          (vec (range (first leaf-positions)
                                      (+ (first leaf-positions) (count leaf-positions))))))
          (throw (ex-info "public outputs must flatten whole logical values in field order"
                          {:reason :link-output-value :value value-id :leaves leaf-nodes
                           :positions leaf-positions :outputs outputs})))))
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

(defn- validate-instance-bindings! [nodes values instance]
  (let [{:keys [id descriptor bindings roles]} instance
        expected (descriptor-pointer-symbols descriptor)
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
    (doseq [[sym value-id] bindings]
      (when-not (contains? values value-id)
        (throw (ex-info "link instance binding names an absent logical value"
                        {:reason :link-absent-value :instance id :symbol sym :value value-id})))
      (when (and (= :constant (get roles sym))
                 (not= :constant (get-in nodes [(-> values (get value-id) :leaves first :node)
                                                :role])))
        (throw (ex-info "a constant instance binding requires a constant logical value"
                        {:reason :link-role-mismatch :instance id :symbol sym :value value-id
                         :value-role (get-in nodes [(-> values (get value-id) :leaves first :node)
                                                    :role])}))))
    (let [args (instance-arguments instance)]
      (doseq [{:keys [sym] :as allocation} (:allocs descriptor)]
        (let [value-id (get bindings sym)
              leaves (get-in values [value-id :leaves])
              specs (descriptor-allocation-leaves allocation (:dtype descriptor))]
          (when-not (= (count specs) (count leaves))
            (throw (ex-info "descriptor allocation leaf count differs from its logical value"
                            {:reason :link-allocation-leaf-count :instance id :symbol sym
                             :value value-id :specs specs :leaves leaves})))
          (doseq [[{:keys [field dtype size-fn]} {:keys [name node]}]
                  (map vector specs leaves)]
            (let [link-node (get nodes node)
                  expected-elements
                  (try (long (size-fn args))
                       (catch Exception e
                         (throw
                          (ex-info
                           "link descriptor allocation extent did not resolve from its specialization environment"
                           {:reason :link-allocation-shape :instance id :symbol sym :field field}
                           e))))
                  actual-elements (shape-elements (get-in link-node [:view :shape]))]
              (when-not (= field name)
                (throw (ex-info "descriptor allocation field differs from its logical leaf"
                                {:reason :link-allocation-field :instance id :symbol sym
                                 :expected field :actual name :node node})))
              (when-not (= (dtype/canon dtype) (dtype/canon (get-in link-node [:view :dtype])))
                (throw (ex-info "link internal node dtype differs from its descriptor allocation"
                                {:reason :link-allocation-dtype :instance id :symbol sym
                                 :field field :node node :expected dtype
                                 :actual (get-in link-node [:view :dtype])})))
              (when-not (= expected-elements actual-elements)
                (throw (ex-info "link internal node shape differs from its descriptor allocation"
                                {:reason :link-allocation-shape :instance id :symbol sym
                                 :field field :node node :expected-elements expected-elements
                                 :actual-elements actual-elements})))))))
      (mapv (fn [[step-index step]]
              (instance-step-facts nodes values instance step-index step))
            (map-indexed vector (:steps descriptor))))))

(defn- program-value-node!
  [nodes values instance-id compiler-value value-id]
  (let [link-value (get values value-id)]
    (when-not link-value
      (throw (ex-info "emitted program buffer names an absent logical LinkValue"
                      {:reason :program-link-absent-value :instance instance-id
                       :compiler-value compiler-value :value value-id})))
    (when-not (= 1 (count (:leaves link-value)))
      (throw (ex-info "one emitted program graph buffer requires one physical LinkValue leaf"
                      {:reason :program-link-leaf-count :instance instance-id
                       :compiler-value compiler-value :value value-id
                       :leaves (:leaves link-value)})))
    (get nodes (get-in link-value [:leaves 0 :node]))))

(defn- validate-program-buffer-contracts!
  [nodes values {:keys [id call roles]}]
  (let [program-values (get-in call [:program :values])]
    (doseq [[compiler-value value-id] (:buffers call)]
      (let [expected (get program-values compiler-value)
            link-value (get values value-id)
            node (program-value-node! nodes values id compiler-value value-id)]
        (when-not (and expected
                       (= (count (:shape expected)) (count (get-in node [:view :shape])))
                       (av/storage-contract-compatible? expected (:abstract link-value)))
          (throw (ex-info "emitted program buffer differs from its logical LinkValue contract"
                          {:reason :program-link-value-contract :instance id
                           :compiler-value compiler-value :value value-id
                           :expected expected :actual (:abstract link-value)
                           :physical-shape (get-in node [:view :shape])})))
        (when (and (= :constant (get roles compiler-value))
                   (not= :constant (:role node)))
          (throw (ex-info "a constant program binding requires a constant LinkValue"
                          {:reason :program-link-role-mismatch :instance id
                           :compiler-value compiler-value :value value-id
                           :value-role (:role node)})))))
    (doseq [[compiler-value value-id] (:loop-scratch call)]
      (program-value-node! nodes values id compiler-value value-id))))

(defn- program-graph-fact
  [nodes values instance-id step-id phase graph buffers]
  (let [graph (kgraph/validate! graph)
        external
        (vals
         (reduce (fn [by-id buffer]
                   (assoc by-id (:id buffer) buffer))
                 {} (concat (:inputs graph) (:outputs graph))))]
    {:instance instance-id :step step-id :phase phase
     :facts
     (mapv
      (fn [{:keys [id dtype role]}]
        (let [value-id (get buffers id ::missing)]
          (when (= ::missing value-id)
            (throw (ex-info "emitted graph buffer has no linked program binding"
                            {:reason :program-link-graph-buffer :instance instance-id
                             :step step-id :buffer id})))
          (let [node (program-value-node! nodes values instance-id id value-id)]
            (when-not (= (dtype/canon dtype) (dtype/canon (get-in node [:view :dtype])))
              (throw (ex-info "emitted graph buffer dtype differs from its linked node"
                              {:reason :program-link-graph-dtype :instance instance-id
                               :step step-id :buffer id :value value-id
                               :expected dtype :actual (get-in node [:view :dtype])})))
            {:symbol id :node (:id node)
             :access (case role
                       :input :read
                       :output :write
                       :inout :read-write)})))
      external)}))

(defn- validate-program-instance-bindings!
  [nodes values instance]
  (let [{:keys [id call]} (validate-program-instance! instance)]
    (validate-program-buffer-contracts! nodes values instance)
    (vec
     (mapcat
      (fn [[step-index step]]
        (cond
          (program-call/evaluated-host-equation? step) []
          (program-call/emitted-equation-call? step)
          [(program-graph-fact nodes values id step-index
                               (get-in step [:equation :id])
                               (:graph step) (:buffers step))]
          (loop-call/structured-loop-call? step)
          (mapv (fn [iteration]
                  (let [{:keys [buffers]} (loop-call/iteration-binding step iteration)]
                    (program-graph-fact nodes values id [step-index iteration]
                                        :structured-loop-iteration
                                        (:graph step) buffers)))
                (range (min 3 (:trip-count step))))))
      (map-indexed vector (:steps call))))))

(defn- validate-effects! [{:keys [target nodes values instances outputs aliases] :as plan}]
  (when (and (not (let [target-name (name target)]
                    (or (= "ze" target-name) (.startsWith target-name "ze:"))))
             (some #(= :scatter (:convention %))
                   (mapcat (fn [instance]
                             (when (link-instance? instance)
                               (get-in instance [:descriptor :steps])))
                           instances)))
    (throw (ex-info "linked scatter execution is not available on this target backend"
                    {:reason :link-target-convention :target target :convention :scatter})))
  (let [step-facts
        (mapcat #(if (link-instance? %)
                   (validate-instance-bindings! nodes values %)
                   (validate-program-instance-bindings! nodes values %))
                instances)
        initialized (volatile! (into #{}
                                     (keep (fn [[id {:keys [role source]}]]
                                             ;; Ownership answers who releases storage, not whether
                                             ;; the storage contains a value. In particular, an
                                             ;; imported :internal node still needs an ordered writer.
                                             (when (or source
                                                       (contains? #{:input :constant :state} role))
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

   `:nodes` may be a node-id map or an ordered collection of LinkNodes. Optional `:values` groups
   nodes into ordered logical LinkValues; every unclaimed node receives an implicit one-leaf value
   with the same identity. `:aliases` explicitly declares overlapping views. Instance order and
   each descriptor's step order define the current serial dependency schedule."
  [{:keys [id target nodes values instances outputs aliases attributes]
    :or {outputs [] aliases #{} attributes {}}}]
  (let [nodes (if (map? nodes) nodes (into {} (map (juxt :id identity)) nodes))
        values (normalize-values nodes values)
        aliases (into #{} (map canonical-alias-pair) aliases)]
    (validate! (->LinkPlan id target nodes values (vec instances) (vec outputs) aliases
                           attributes))))

(defn instance-roles
  "Resolve compiler-symbol roles for the runtime binder. Only `:constant` affects executable
   prologue hoisting today; the complete role map remains data for residency/lifetime policy."
  [plan instance]
  (when-not (link-plan? plan)
    (throw (ex-info "instance-roles requires a LinkPlan" {:actual (type plan)})))
  (let [nodes (:nodes plan)
        values (:values plan)]
    (cond
      (link-instance? instance)
      (let [instance (validate-instance! instance)]
        (merge
         (into {}
               (map (fn [[sym value-id]]
                      (let [node-id (get-in values [value-id :leaves 0 :node])]
                        [sym (case (get-in nodes [node-id :role])
                               :internal :scratch
                               (get-in nodes [node-id :role]))]))
                    (:bindings instance)))
         (:roles instance)))

      (program-link-instance? instance)
      (let [{:keys [call roles]} (validate-program-instance! instance)]
        (merge
         (into {}
               (map (fn [[compiler-value value-id]]
                      (let [node-id (get-in values [value-id :leaves 0 :node])]
                        [compiler-value
                         (case (get-in nodes [node-id :role])
                           :internal :scratch
                           (get-in nodes [node-id :role]))]))
                    (:buffers call)))
         roles))

      :else
      (throw (ex-info "instance-roles requires a recognized LinkPlan instance"
                      {:reason :link-instance-type :instance instance
                       :actual (type instance)})))))
