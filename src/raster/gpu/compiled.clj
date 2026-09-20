(ns raster.gpu.compiled
  "The `Compiled` artifact — a functional, inspectable GPU program value (S4 §2).

   `(r/compile #'train-step args opts)` returns a `Compiled` that implements `IFn`: calling it
   replays the resident program and returns device values, not host arrays. For composition,
   `(r/lower ...)` returns an allocation-free `Prepared`; independently lowered values compose
   through semantic boundaries and only the composite is instantiated. Resident descriptors and
   compositions are certified into the same LinkPlan/LinkedExecutable path.

   The three artifact primitives, honestly scoped to the whole-program MVP:
     • device value      — outputs are DeviceArrays over resident buffers; no host round-trip.
     • functional invoke  — `(step inputs)` → `{out-key → DeviceArray}`; mutation of resident
                            :state is invisible (the old input value is consumed/invalidated).
     • donation           — a donated in→out pair reuses the resident buffer; the input value
                            is marked consumed (reads throw), the output value is fresh.

   Roles are DERIVED (§4.2): `:donate` syms → :state (donated), `:constants` syms → :constant
   (captured once at bind, never per-call), the remainder default to the descriptor's derived
   read-only→:input / written→:output. Backend-neutral: `:target` selects the runtime; nothing
   here hardcodes `:ze`."
  (:refer-clojure :exclude [compile])
  (:require [clojure.set :as set]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.invocation-link :as invocation-link]
            [raster.compiler.ir.link-composition :as link-composition]
            [raster.compiler.ir.link-plan :as link-plan]
            [raster.compiler.ir.resident-plan :as resident-plan]
            [raster.compiler.pipeline :as pl]
            [raster.gpu.core :as gpu]
            [raster.gpu.link :as gpu-link]
            [raster.gpu.value :as v]))

(declare invoke-compiled)

;; ================================================================
;; The record
;; ================================================================

(defrecord Compiled
           [lowering     ;; certified resident or composition lowering into LinkPlan
            executable  ;; LinkedExecutable: the sole resident runtime representation
            in-tree      ;; ordered [{:key :sym :role :donate? :shape :dtype} …] — the arg spec
            out-tree     ;; ordered [{:key :sym :shape :dtype :from} …] — the result spec (multi-output)
            donated      ;; {in-key → out-key} — the alias plan (JAX input_output_aliases)
            schedule     ;; the S6 Schedule map (reserved; nil until S6 fills it)
            target       ;; device-id + (future) HardwareDescriptor
            descriptor   ;; raw descriptor or inspectable composite descriptor
            args         ;; captured specialization arguments (empty for a composite)
            live-outputs] ;; atom holding the DeviceArrays projected by the LAST invocation. They
                          ;; alias resident buffers the next replay overwrites, so they are
                          ;; invalidated (marked dead) at the start of the next invoke and at close!
                          ;; — otherwise a retained old output would silently observe a mutation.
  clojure.lang.IFn
  (invoke [this inputs] (invoke-compiled this inputs))
  (invoke [this] (invoke-compiled this {}))
  (applyTo [this argseq] (apply invoke-compiled this argseq)))

(defrecord Prepared
           [lowering in-tree out-tree donated schedule target descriptor args])

(defn compiled? [x] (instance? Compiled x))
(defn prepared? [x] (instance? Prepared x))

;; ================================================================
;; Role derivation (§4.2) and tree construction
;; ================================================================

(defn- derive-roles
  "Effective {sym → role} for certified LinkPlan lowering: donated → :state, constants →
   :constant, and the rest fall through to the descriptor's derived defaults."
  [descriptor donate constants explicit-roles]
  (let [donate-set   (set donate)
        constant-set (set constants)
        both         (set/intersection donate-set constant-set)]
    (when (seq both)
      (throw (ex-info (str "compile: syms are both :donate and :constants — a param is either "
                           "donated (mutable :state) or frozen (:constant), not both: " both)
                      {:conflict both})))
    (merge (into {} (map (fn [s] [s :state]) donate-set))
           (into {} (map (fn [s] [s :constant]) constant-set))
           explicit-roles)))

(defn- build-in-tree
  [lowering descriptor arguments donate]
  (let [donate-set (set donate)
        values (:values (:certificate lowering))
        argument-map (zipmap (:all-params descriptor) arguments)]
    (vec (for [p (:array-params descriptor)
               :let [{:keys [node role shape dtype]} (get values p)]]
           {:key     (keyword (name p))
            :sym     p
            :node    node
            :role    role
            :donate? (contains? donate-set p)
            :shape   shape
            :dtype   dtype
            :default (get argument-map p)}))))

(defn- build-out-tree
  "Out-tree = donated in→out nodes + any explicit :outputs + the functional :result-sym + taps.
   Each projects to a DeviceArray over a resident buffer (§3.4 multi-output)."
  [lowering donate outputs result-sym taps]
  (let [values (:values (:certificate lowering))
        output-node (fn [key sym from]
                      (let [{:keys [node shape dtype]} (get values sym)]
                        {:key key :sym sym :node node :shape shape :dtype dtype :from from}))
        donate-nodes  (for [s donate]
                        (output-node (keyword (str (name s) "'")) s :donated))
        output-nodes  (for [s outputs]
                        (output-node (keyword (name s)) s :output))
        result-node   (when result-sym
                        [(output-node (keyword (name result-sym)) result-sym :result)])
        tap-nodes     (for [s taps]
                        (output-node (keyword (name s)) s :tap))]
    (vec (concat donate-nodes output-nodes result-node tap-nodes))))

(defn- compilation-id
  [fn-var target dtype descriptor args]
  (let [m (meta fn-var)
        qualified (symbol (str (ns-name (:ns m))) (str (:name m)))
        arrays (set (:array-params descriptor))
        argmap (zipmap (:all-params descriptor) args)
        signature (mapv (fn [parameter]
                          (let [value (get argmap parameter)]
                            (if (contains? arrays parameter)
                              [:array (.getName (.getComponentType (class value)))
                               (java.lang.reflect.Array/getLength value)]
                              value)))
                        (:all-params descriptor))]
    [::compiled qualified target dtype signature (:schedule descriptor)]))

;; ================================================================
;; Pure lowering, composition, and runtime compilation
;; ================================================================

(defn- lower-resident-descriptor
  "Lower a deftm var into a certified, allocation-free `Prepared` artifact.

   args  — example args in the descriptor's :all-params order. Supplies BOTH the shapes
           compile-gpu-program derives AND the eventual resident initializers.
   opts  — {:target :ze:0            device-id (default :ze:0)
            :dtype  :float           element dtype (default :float)
            :donate  [sym …]         resident :state threaded as values (donation)
            :constants [sym …]       frozen, captured once at bind, never per-call
            :outputs [sym …]         additional written params to project as outputs
            :taps    [sym …]         internal nodes to additionally expose (§5.1)
            :roles   {sym → role}    explicit role override (last word)
            :gemm-precision :mixed-f16-f32|:f32-scalar
            :on-non-resident :nil|:throw
            :schedule <map>}         reserved S6 schedule (threaded into the cache key)"
  [fn-var args {:keys [target dtype donate constants outputs taps roles
                       gemm-precision on-non-resident schedule]
                :or {target :ze:0 dtype :float on-non-resident :nil}}]
  (let [prog (apply pl/compile-gpu-program fn-var target
                    (cond-> [:dtype dtype :on-non-resident on-non-resident]
                      gemm-precision (conj :gemm-precision gemm-precision)
                      ;; forward the S6 schedule so it is resolved + gated by compile-gpu-program;
                      ;; the RESOLVED schedule is read back off the descriptor below (never the raw
                      ;; input). Harmless where compile-gpu-program predates :schedule (ignored kwarg).
                      schedule (conj :schedule schedule)))
        _ (when-not prog
            (throw (ex-info "compile: compile-gpu-program returned nil — a step fell back to host (non-resident). Pass :on-non-resident :throw to see which."
                            {:fn fn-var :target target})))
        eff-roles (derive-roles prog donate constants roles)
        result-sym (:result-sym prog)
        public-symbols (vec (distinct (concat donate outputs
                                              (when result-sym [result-sym]) taps)))
        lowering (resident-plan/lower
                  {:id (compilation-id fn-var target dtype prog args)
                   :target target :descriptor prog :arguments args
                   :roles eff-roles :outputs public-symbols})
        in-tree  (build-in-tree lowering prog args donate)
        out-tree (build-out-tree lowering donate outputs result-sym taps)
        donated  (into {} (map (fn [s] [(keyword (name s)) (keyword (str (name s) "'"))]) donate))]
    (->Prepared lowering in-tree out-tree donated (:schedule prog) target prog args)))

(defn- equation-first-value
  [plan node role]
  (let [view (get-in plan [:nodes node :view])]
    {:node node :role role :shape (:shape view) :dtype (:dtype view)}))

(defn- equation-first-output-key
  [value index total]
  (cond
    (= 1 total) :result
    (instance? clojure.lang.Named value) (keyword (name value))
    :else (keyword (str "result-" index))))

(defn- lower-equation-first
  [fn-var args {:keys [target dtype donate constants outputs taps roles]
                :or {target :ze:0 dtype :float}
                :as opts}]
  (let [compilation-options (apply dissoc opts
                                   [:compiler :donate :constants :outputs :taps :roles
                                    :profile? :on-non-resident :gemm-precision])
        compilation (equation-first/compile
                     fn-var (assoc compilation-options :target target :dtype dtype))
        raw-plan (equation-first/lower compilation args)
        attributes (:attributes raw-plan)
        public-bindings (:public-buffer-bindings attributes)
        public-defaults (:public-buffer-roles attributes)
        public-symbols (set (keys public-bindings))
        requested-symbols (set (concat donate constants (keys roles)))
        unknown (set/difference requested-symbols public-symbols)
        _ (when (seq unknown)
            (throw (ex-info "equation-first roles name non-buffer public parameters"
                            {:reason :compiled-equation-first-role-symbols
                             :symbols unknown :available public-symbols})))
        effective-roles (merge public-defaults
                               (derive-roles nil donate constants roles))
        token-roles (into {} (map (fn [[symbol role]]
                                    [(get public-bindings symbol) role]))
                          effective-roles)
        compiler-bindings (:compiler-buffer-bindings attributes)
        compiler-roles (into {} (keep (fn [[compiler-value token]]
                                        (when-let [role (get token-roles token)]
                                          [compiler-value role])))
                             compiler-bindings)
        plan (-> (reduce-kv (fn [plan token role]
                              (assoc-in plan [:nodes token :role] role))
                            raw-plan token-roles)
                 (assoc-in [:instances 0 :roles] compiler-roles)
                 (assoc-in [:attributes :public-buffer-roles] effective-roles)
                 link-plan/validate!)
        parameters (get-in compilation [:semantic :attributes :invocation-plan :parameters])
        argument-map (zipmap (map :symbol parameters) args)
        donate-set (set donate)
        in-tree (vec
                 (keep (fn [{:keys [symbol]}]
                         (when-let [node (get public-bindings symbol)]
                           (merge {:key (keyword (name symbol))
                                   :sym symbol
                                   :donate? (contains? donate-set symbol)
                                   :default (get argument-map symbol)}
                                  (equation-first-value plan node
                                                        (get effective-roles symbol)))))
                       parameters))
        resolve-node (fn [value]
                       (or (get public-bindings value)
                           (get compiler-bindings value)
                           (throw (ex-info "equation-first output names no resident value"
                                           {:reason :compiled-equation-first-output
                                            :value value}))))
        output-entry (fn [key value from]
                       (merge {:key key :sym value :from from}
                              (equation-first-value plan (resolve-node value) nil)))
        donated-nodes (map-indexed
                       (fn [_ symbol]
                         (output-entry (keyword (str (name symbol) "'")) symbol :donated))
                       donate)
        explicit-nodes (map-indexed
                        (fn [_ value]
                          (output-entry (keyword (name value)) value :output))
                        outputs)
        semantic-outputs (vec (:semantic-outputs attributes))
        semantic-nodes (map-indexed
                        (fn [index [value _]]
                          (output-entry
                           (equation-first-output-key value index (count semantic-outputs))
                           value :result))
                        semantic-outputs)
        tap-nodes (map-indexed
                   (fn [_ value]
                     (output-entry (keyword (name value)) value :tap))
                   taps)
        out-tree (reduce (fn [entries entry]
                           ;; One physical result has one value wrapper/lifetime. An explicit,
                           ;; donated, or tap key takes precedence over the implicit result key.
                           (if (some #(= (:node %) (:node entry)) entries)
                             entries
                             (conj entries entry)))
                         [] (concat donated-nodes explicit-nodes semantic-nodes tap-nodes))
        duplicate-keys (->> out-tree (map :key) frequencies
                            (keep (fn [[key count]] (when (< 1 count) key))) set)
        _ (when (seq duplicate-keys)
            (throw (ex-info "equation-first outputs require unique semantic keys"
                            {:reason :compiled-equation-first-output-keys
                             :keys duplicate-keys})))
        lowering (invocation-link/certify plan)
        steps (mapv (fn [index kernel]
                      {:convention :kernel-body
                       :phase (keyword (str "kernel-" index))
                       :kernel-name (:kernel-name kernel)})
                    (range) (:kernels compilation))
        descriptor {:all-params (mapv :symbol parameters)
                    :array-params (mapv :symbol (filter #(contains? public-bindings (:symbol %))
                                                       parameters))
                    :scalar-params (mapv :symbol (remove #(contains? public-bindings (:symbol %))
                                                        parameters))
                    :steps steps :result-sym nil :equation-first? true}
        schedule {:compiler :equation-first :stats (:stats compilation)}
        donated (into {} (map (fn [symbol]
                                [(keyword (name symbol))
                                 (keyword (str (name symbol) "'"))]))
                      donate)]
    (->Prepared lowering in-tree out-tree donated schedule target descriptor args)))

(defn lower
  "Lower a deftm Var into an allocation-free `Prepared` artifact.

   `:compiler :equation-first` selects the typed SOAC→scheduled KernelBody→LinkPlan vertical.
   `:resident-descriptor` remains the default during migration. Selection is explicit: the
   equation-first compiler never falls back to a resident descriptor after a coverage failure."
  [fn-var args {:keys [compiler] :or {compiler :resident-descriptor} :as opts}]
  (case compiler
    :resident-descriptor (lower-resident-descriptor fn-var args opts)
    :equation-first (lower-equation-first fn-var args opts)
    (throw (ex-info "unknown compiled lowering vertical"
                    {:reason :compiled-compiler :compiler compiler
                     :allowed #{:resident-descriptor :equation-first}}))))

(defn instantiate!
  "Instantiate one pure Prepared artifact as a callable Compiled value. All component plans have
   already been composed, so this performs one allocation/binding/graph-recording operation."
  ([prepared] (instantiate! prepared {}))
  ([prepared opts]
   (when-not (prepared? prepared)
     (throw (ex-info "instantiate! requires an allocation-free Prepared artifact"
                     {:reason :compiled-prepared-type :actual (type prepared)})))
   (let [{:keys [lowering in-tree out-tree donated schedule target descriptor args]} prepared
         executable (gpu-link/instantiate! (:plan lowering) opts)]
     (->Compiled lowering executable in-tree out-tree donated schedule target descriptor args
                 (atom nil)))))

(defn- semantic-entry! [components side [component-id key :as reference]]
  (when-not (and (vector? reference) (= 2 (count reference)))
    (throw (ex-info "a semantic artifact reference is [component-id key]"
                    {:reason :compiled-composition-reference :reference reference})))
  (let [prepared (get components component-id)
        entries (case side :input (:in-tree prepared) :output (:out-tree prepared))
        matches (filterv #(= key (:key %)) entries)]
    (when-not (= 1 (count matches))
      (throw (ex-info "a semantic artifact reference must resolve exactly once"
                      {:reason :compiled-composition-reference :side side :reference reference
                       :available (mapv :key entries)})))
    (first matches)))

(defn compose
  "Compose independently lowered Prepared artifacts through semantic boundary keys.

   Request shape:
   {:id stable-id
    :components [{:id component-id :program prepared} ...]
    :connections [{:from [producer-id output-key] :to [consumer-id input-key]} ...]
    :shares [[[component-id input-or-constant-key] ...] ...]
    :outputs [{:key composite-output-key :from [component-id output-key]} ...]}

   Remaining component inputs are exposed under `[component-id input-key]`. Composition happens
   before allocation, so connected intermediates and shared constants have one physical
   allocation. Donation across independent artifacts is deferred until the ownership/view pass
   can certify cross-component state threading."
  [{:keys [id components connections shares outputs attributes]
    :or {connections [] shares [] attributes {}}}]
  (let [components (mapv (fn [component]
                           (when-not (and (map? component) (contains? component :id)
                                          (prepared? (:program component)))
                             (throw (ex-info
                                     "each compiled component requires :id and a Prepared :program"
                                     {:reason :compiled-composition-component
                                      :component component})))
                           component)
                         components)
        component-map (into {} (map (juxt :id :program)) components)
        _ (when-not (= (count components) (count component-map))
            (throw (ex-info "compiled component identities must be unique"
                            {:reason :compiled-composition-component-ids
                             :ids (mapv :id components)})))
        donated-components (filterv (comp seq :donated :program) components)
        _ (when (seq donated-components)
            (throw (ex-info "cross-component donation requires the composite ownership pass"
                            {:reason :compiled-composition-donation
                             :components (mapv :id donated-components)})))
        resolved-connections
        (mapv (fn [{:keys [from to]}]
                {:from from :to to
                 :from-entry (semantic-entry! component-map :output from)
                 :to-entry (semantic-entry! component-map :input to)})
              connections)
        resolved-shares
        (mapv (fn [group]
                (mapv (fn [reference]
                        {:reference reference
                         :entry (semantic-entry! component-map :input reference)})
                      group))
              shares)
        output-specs
        (mapv (fn [{:keys [key from] :as output}]
                (when-not (= #{:key :from} (set (keys output)))
                  (throw (ex-info "a composite output requires exactly :key and :from"
                                  {:reason :compiled-composition-output :output output})))
                {:key key :from from
                 :entry (semantic-entry! component-map :output from)})
              outputs)
        _ (when-not (= (count output-specs) (count (distinct (map :key output-specs))))
            (throw (ex-info "composite output keys must be unique and ordered"
                            {:reason :compiled-composition-output-keys
                             :keys (mapv :key output-specs)})))
        low-level
        (link-composition/compose
         {:id id
          :components (mapv (fn [{:keys [id program]}]
                              {:id id :lowering (:lowering program)})
                            components)
          :connections (mapv (fn [{:keys [from to from-entry to-entry]}]
                               {:from [(first from) (:node from-entry)]
                                :to [(first to) (:node to-entry)]})
                             resolved-connections)
          :shares (mapv (fn [group]
                          (mapv (fn [{:keys [reference entry]}]
                                  [(first reference) (:node entry)])
                                group))
                        resolved-shares)
          :outputs (mapv (fn [{:keys [from entry]}]
                           [(first from) (:node entry)])
                         output-specs)
          :attributes attributes})
        node-mapping (get-in low-level [:certificate :node-mapping])
        mapped-node (fn [component-id node-id]
                      (or (get node-mapping [component-id node-id])
                          (throw (ex-info
                                  "certified semantic node disappeared during composition"
                                  {:reason :compiled-composition-node
                                   :component component-id :node node-id}))))
        consumed-inputs (set (map :to resolved-connections))
        discarded-shares (set (mapcat (fn [group] (map :reference (rest group)))
                                      resolved-shares))
        in-tree
        (vec
         (for [{component-id :id program :program} components
               entry (:in-tree program)
               :let [reference [component-id (:key entry)]]
               :when (not (contains? consumed-inputs reference))
               :when (not (contains? discarded-shares reference))]
           (assoc entry :key reference :sym [component-id (:sym entry)]
                  :node (mapped-node component-id (:node entry)))))
        out-tree
        (mapv (fn [{:keys [key from entry]}]
                (assoc entry :key key :sym [(first from) (:sym entry)] :from :composed
                       :node (mapped-node (first from) (:node entry))))
              output-specs)
        descriptor {:all-params [] :array-params [] :scalar-params []
                    :steps (vec (mapcat (comp :steps :descriptor :program) components))
                    :result-sym nil :composite? true}
        schedules (mapv (comp :schedule :program) components)]
    (->Prepared low-level in-tree out-tree {} schedules (:target (:plan low-level))
                descriptor [])))

(defn compile
  "Lower and instantiate a deftm as one callable Compiled artifact. Use `lower`, `compose`, then
   `instantiate!` when independently compiled programs must share resident values."
  [fn-var args opts]
  (instantiate! (lower fn-var args opts) (select-keys opts [:profile?])))

;; ================================================================
;; Functional invocation (§2.3)
;; ================================================================

(defn- project-node
  "Wrap one linked output view as an external DeviceArray. LinkedExecutable retains allocation
   ownership; the wrapper owns only its value lifetime."
  [executable {:keys [node]} target]
  (let [resident (gpu-link/node-view executable node)
        buffer (gpu/buffer (:session executable) (:key resident))]
    (v/wrap-external-view buffer target (:view resident))))

(defn- invalidate-live-outputs!
  [^Compiled c]
  (when-let [live-outputs (:live-outputs c)]
    (doseq [device-array @live-outputs] (v/free! device-array))
    (reset! live-outputs nil))
  c)

(defn invoke-compiled
  "Replay the artifact and return device values. `inputs` : {in-key → DeviceArray|host-array}.
     1. write each dynamic input (host upload, exact-view no-op, or device-to-device copy);
     2. consume exact resident donated values (donation-invalidation, §1.3);
     3. replay the linked graph with no output download;
     4. project out-tree nodes as external DeviceArrays over stable LinkPlan views.
   Mutation of resident :state is invisible: the caller sees fresh output values and the old
   donated inputs invalidated — never a mutation."
  [^Compiled c inputs]
  (let [{:keys [executable in-tree out-tree donated target]} c
        in-nodes     (into {} (map (juxt :key identity)) in-tree)
        input-nodes  (filterv #(= :input (:role %)) in-tree)
        input-keys   (set (map :key input-nodes))
        donated-keys (set (keys donated))
        ;; 0. VALIDATE inputs: every passed key must be an :input-role param or a donated slot —
        ;;    never a :constant/:state key silently ignored (fail-loud, §7.7).
        _ (doseq [[k _v] inputs]
            (when-not (or (contains? input-keys k) (contains? donated-keys k))
              (throw (ex-info (str "invoke: unsupported input key " k " — only :input-role params "
                                   (vec input-keys) " or donated slots " (vec donated-keys)
                                   " may be passed; a :constant/:state slot is captured at bind")
                              {:key k :inputs (keys inputs)}))))
        ;; 1. Every dynamic input is refreshed on every invocation, preserving the resident-program
        ;;    contract. gpu-link/write! accepts host values and performs D2D for foreign device
        ;;    values; it never materializes a DeviceArray through v/->host.
        _ (doseq [{:keys [key node default]} input-nodes]
            (gpu-link/write! executable node (get inputs key default)))
        ;; 2. Donation remains stricter than an ordinary input: the passed value must already be
        ;;    this exact state view. Moving a foreign value and then calling it donation would hide
        ;;    a copy and misrepresent ownership.
        _ (doseq [[k _out] donated]
            (when-let [val (get inputs k)]
              (when-not (v/device-array? val)
                (throw (ex-info (str "invoke: donated input " k " must be a DeviceArray naming "
                                     "this artifact's exact resident view")
                                {:key k :actual (type val)})))
              (let [{node-id :node :as input-node} (get in-nodes k)
                    resident (gpu-link/node-view executable node-id)
                    resident-buffer (gpu/buffer (:session executable) (:key resident))]
                (when-not (and (identical? (:buffer val) resident-buffer)
                               (bview/same-range? (:view val) (:view resident)))
                  (throw (ex-info (str "invoke: donated input " k " is not this artifact's resident "
                                       "view — thread the exact donated output back")
                                  {:key k :node node-id})))
                (when-not (= [(:dtype input-node) (:shape input-node)]
                             [(:dtype val) (:shape val)])
                  (throw (ex-info (str "invoke: donated input " k " dtype/shape differs from its slot")
                                  {:key k :expected (select-keys input-node [:dtype :shape])
                                   :actual (select-keys val [:dtype :shape])})))
                (v/consume! val))))]
    ;; 3. invalidate the PREVIOUS batch of outputs — they alias resident buffers this replay
    ;;    overwrites, so a retained old wrapper would observe a silent mutation (§2.3). ::external
    ;;    free! only marks the wrapper dead; it never frees the session-owned buffer.
    (invalidate-live-outputs! c)
    ;; 4. replay, no download.
    (gpu-link/run! executable)
    ;; 5. project outputs as resident device values; record them for next-call invalidation.
    (let [out (into {} (map (fn [{:keys [key] :as node}]
                              [key (project-node executable node target)]))
                    out-tree)]
      (when-let [live-outputs (:live-outputs c)]
        (reset! live-outputs (vec (vals out))))
      out)))

;; ================================================================
;; Inspection (§2.1) + lifecycle
;; ================================================================

(defn explain
  "Print the artifact's shape: in-tree / out-tree / donation plan / target / schedule and the
   resident step-kind histogram. Works before or after instantiation and returns its argument."
  [c]
  (let [{:keys [in-tree out-tree donated target schedule descriptor]} c
        kinds (frequencies (map :convention (:steps descriptor)))]
    (println (if (prepared? c) "Prepared artifact" "Compiled artifact"))
    (println "  target   :" target)
    (println "  in-tree  :" (mapv (fn [n] [(:key n) (:role n) (when (:donate? n) :donate)]) in-tree))
    (println "  out-tree :" (mapv (fn [n] [(:key n) (:from n)]) out-tree))
    (println "  donated  :" donated)
    (println "  schedule :" (or schedule :none))
    (println "  steps    :" (count (:steps descriptor)) "resident, by kind:" kinds))
  c)

(defn execution-info
  "Report the instantiated artifact's admitted resident schedules, without running it.
   Unlike `ir`, this observes binding-time selection. It is not a replay or timing report."
  [c]
  (when-not (compiled? c)
    (throw (ex-info "execution info requires an instantiated compiled artifact"
                    {:reason :compiled-execution-info-unbound})))
  (gpu-link/execution-info (:executable c)))

(defn ir
  "Return the descriptor's resident steps (the artifact's lowered IR) for inspection."
  [c]
  (mapv #(select-keys % [:convention :phase :variant :kernel-name]) (:steps (:descriptor c))))

(defn plan
  "Return the validated LinkPlan that is the artifact's public executable representation."
  [c]
  (:plan (:lowering c)))

(defn certificate
  "Return the checkable resident or composition lowering certificate."
  [c]
  (:certificate (:lowering c)))

(defn- refresh-captured-inputs!
  [^Compiled c]
  (doseq [{:keys [node role default]} (:in-tree c) :when (= :input role)]
    (gpu-link/write! (:executable c) node default))
  c)

(defn profile
  "Device-event profile of one linked replay. The artifact must have been compiled with
   `{:profile? true}`. Returns the historical profile map, including downloaded semantic results."
  [^Compiled c]
  (refresh-captured-inputs! c)
  (invalidate-live-outputs! c)
  (let [profile-result (gpu-link/profile! (:executable c))
        result (into {} (map (fn [{:keys [key node]}]
                               [key (gpu-link/download (:executable c) node)]))
                     (:out-tree c))]
    (assoc profile-result :result result)))

(defn measure
  "Explicit offline device-event measurement of a Compiled artifact.

   The artifact must have been compiled with {:profile? true}. Returns a stable Measurement;
   options are those of gpu-link/measure!, including the required :before-sample! restore hook
   for stateful programs. This does not choose or cache a schedule by itself."
  [^Compiled c & {:as opts}]
  (refresh-captured-inputs! c)
  (invalidate-live-outputs! c)
  (apply gpu-link/measure! (:executable c) (mapcat identity opts)))

(defn cache-key
  "The serializable identity of the artifact minus closures (§2b C5): in/out trees, donation,
   schedule, target, and the descriptor's step kinds + concrete shapes. Excludes the non-
   serializable closures (:n-fn/:m-fn/…) and SPIR-V modules."
  [c]
  {:in-tree  (mapv #(select-keys % [:key :role :donate? :shape :dtype]) (:in-tree c))
   :out-tree (mapv #(select-keys % [:key :from :shape :dtype]) (:out-tree c))
   :donated  (:donated c)
   :schedule (:schedule c)
   :target   (:target c)
   :steps    (frequencies (map :convention (:steps (:descriptor c))))})

(defn close!
  "Release the artifact's linked executable (buffers + graph + kernels). Invalidates
   any still-live projected output values FIRST, so a `->host` on a value returned before close!
   fails loud (use-after-free) instead of copying from a zeMemFree'd segment (SIGSEGV/garbage)."
  [^Compiled c]
  (invalidate-live-outputs! c)
  (gpu-link/close! (:executable c))
  nil)
