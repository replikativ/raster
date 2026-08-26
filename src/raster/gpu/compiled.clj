(ns raster.gpu.compiled
  "The `Compiled` artifact — a functional, inspectable GPU program value (S4 §2).

   `(r/compile #'train-step args {:target :ze:0 :donate [adapters…] :constants [Wq …]})`
   returns a `Compiled` that implements `IFn`: calling it replays the resident program and
   returns device values (`raster.gpu.value/DeviceArray`), not host arrays. The resident
   descriptor is certified into the same LinkPlan used for composition and instantiated as a
   LinkedExecutable. There is one value/view/runtime path, not an artifact-only binder beside it.

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
            [raster.compiler.ir.buffer-view :as bview]
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
           [lowering     ;; CertifiedResidentPlan: checkable descriptor→LinkPlan witness
            executable  ;; LinkedExecutable: the sole resident runtime representation
            in-tree      ;; ordered [{:key :sym :role :donate? :shape :dtype} …] — the arg spec
            out-tree     ;; ordered [{:key :sym :shape :dtype :from} …] — the result spec (multi-output)
            donated      ;; {in-key → out-key} — the alias plan (JAX input_output_aliases)
            schedule     ;; the S6 Schedule map (reserved; nil until S6 fills it)
            target       ;; device-id + (future) HardwareDescriptor
            descriptor   ;; the raw compile-gpu-program descriptor — inspectable
            args         ;; captured example args (:all-params order): resident bind contents + shape source
            live-outputs] ;; atom holding the DeviceArrays projected by the LAST invocation. They
                          ;; alias resident buffers the next replay overwrites, so they are
                          ;; invalidated (marked dead) at the start of the next invoke and at close!
                          ;; — otherwise a retained old output would silently observe a mutation.
  clojure.lang.IFn
  (invoke [this inputs] (invoke-compiled this inputs))
  (invoke [this] (invoke-compiled this {}))
  (applyTo [this argseq] (apply invoke-compiled this argseq)))

(defn compiled? [x] (instance? Compiled x))

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
  [lowering descriptor donate]
  (let [donate-set (set donate)
        values (:values (:certificate lowering))]
    (vec (for [p (:array-params descriptor)
               :let [{:keys [node role shape dtype]} (get values p)]]
           {:key     (keyword (name p))
            :sym     p
            :node    node
            :role    role
            :donate? (contains? donate-set p)
            :shape   shape
            :dtype   dtype}))))

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
;; Compile (the public verb)
;; ================================================================

(defn compile
  "Compile a deftm var into a `Compiled` artifact bound on `target`.

   args  — example args in the descriptor's :all-params order. Supplies BOTH the shapes
           compile-gpu-program derives AND the initial resident buffer contents at bind.
   opts  — {:target :ze:0            device-id (default :ze:0)
            :dtype  :float           element dtype (default :float)
            :donate  [sym …]         resident :state threaded as values (donation)
            :constants [sym …]       frozen, captured once at bind, never per-call
            :outputs [sym …]         additional written params to project as outputs
            :taps    [sym …]         internal nodes to additionally expose (§5.1)
            :roles   {sym → role}    explicit role override (last word)
            :gemm-precision :f16-xmx|:f32-scalar
            :on-non-resident :nil|:throw
            :profile? bool           bind in profiling mode (for r/profile)
            :schedule <map>}         reserved S6 schedule (threaded into the cache key)"
  [fn-var args {:keys [target dtype donate constants outputs taps roles
                       gemm-precision on-non-resident profile? schedule]
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
        executable (gpu-link/instantiate! (:plan lowering) {:profile? profile?})
        in-tree  (build-in-tree lowering prog donate)
        out-tree (build-out-tree lowering donate outputs result-sym taps)
        donated  (into {} (map (fn [s] [(keyword (name s)) (keyword (str (name s) "'"))]) donate))]
    (->Compiled lowering executable in-tree out-tree donated (:schedule prog) target prog args
                (atom nil))))

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
  (let [{:keys [executable in-tree out-tree donated descriptor target args]} c
        in-nodes     (into {} (map (juxt :key identity)) in-tree)
        input-nodes  (filterv #(= :input (:role %)) in-tree)
        input-keys   (set (map :key input-nodes))
        donated-keys (set (keys donated))
        captured (zipmap (:all-params descriptor) args)
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
        _ (doseq [{:keys [key sym node]} input-nodes]
            (gpu-link/write! executable node (get inputs key (get captured sym))))
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
   resident step-kind histogram. Returns the Compiled unchanged."
  [^Compiled c]
  (let [{:keys [in-tree out-tree donated target schedule descriptor]} c
        kinds (frequencies (map :convention (:steps descriptor)))]
    (println "Compiled artifact")
    (println "  target   :" target)
    (println "  in-tree  :" (mapv (fn [n] [(:key n) (:role n) (when (:donate? n) :donate)]) in-tree))
    (println "  out-tree :" (mapv (fn [n] [(:key n) (:from n)]) out-tree))
    (println "  donated  :" donated)
    (println "  schedule :" (or schedule :none))
    (println "  steps    :" (count (:steps descriptor)) "resident, by kind:" kinds))
  c)

(defn ir
  "Return the descriptor's resident steps (the artifact's lowered IR) for inspection."
  [^Compiled c]
  (mapv #(select-keys % [:convention :phase :variant :kernel-name]) (:steps (:descriptor c))))

(defn plan
  "Return the validated LinkPlan that is the artifact's public executable representation."
  [^Compiled c]
  (:plan (:lowering c)))

(defn certificate
  "Return the checkable resident-descriptor→LinkPlan lowering certificate."
  [^Compiled c]
  (:certificate (:lowering c)))

(defn- refresh-captured-inputs!
  [^Compiled c]
  (let [captured (zipmap (get-in c [:descriptor :all-params]) (:args c))]
    (doseq [{:keys [sym node role]} (:in-tree c) :when (= :input role)]
      (gpu-link/write! (:executable c) node (get captured sym))))
  c)

(defn profile
  "Device-event profile of one linked replay. The artifact must have been compiled with
   `{:profile? true}`. Returns the historical profile map, including downloaded semantic results."
  [^Compiled c]
  (refresh-captured-inputs! c)
  (invalidate-live-outputs! c)
  (let [descriptor (:descriptor c)
        certificate (certificate c)
        roles (:roles certificate)
        bindings (:bindings certificate)
        result-sym (:result-sym descriptor)
        output-symbols (vec (concat (for [sym (:array-params descriptor)
                                          :when (= :output (get roles sym))]
                                      sym)
                                    (when (and result-sym
                                               (not (some #{result-sym}
                                                          (:array-params descriptor))))
                                      [result-sym])))
        profile-result (gpu-link/profile! (:executable c))
        result (into {} (map (fn [sym]
                               [sym (gpu-link/download (:executable c) (get bindings sym))]))
                     output-symbols)]
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
  [^Compiled c]
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
