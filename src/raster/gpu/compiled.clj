(ns raster.gpu.compiled
  "The `Compiled` artifact — a functional, inspectable GPU program value (S4 §2).

   `(r/compile #'train-step args {:target :ze:0 :donate [adapters…] :constants [Wq …]})`
   returns a `Compiled` that implements `IFn`: calling it replays the resident program and
   returns device values (`raster.gpu.value/DeviceArray`), not host arrays. It wraps today's
   `bind-program!`/`replay-program!` with ZERO change to the kernel/graph machinery — the
   executable still lives in the session; `Compiled` lifts the *invocation contract* to
   values-in / values-out with donation as a boundary write-back shim.

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
  (:require [raster.gpu.core :as gpu]
            [raster.gpu.value :as v]
            [raster.compiler.pipeline :as pl]))

(declare invoke-compiled)

;; ================================================================
;; The record
;; ================================================================

(defrecord Compiled
           [session      ;; the GPU session atom the executable lives in (MVP: not yet lifted out)
            handle       ;; the bind-program! handle {::gpu/program-key … :descriptor …}
            in-tree      ;; ordered [{:key :sym :role :donate? :shape :dtype} …] — the arg spec
            out-tree     ;; ordered [{:key :sym :shape :dtype :from} …] — the result spec (multi-output)
            donated      ;; {in-key → out-key} — the alias plan (JAX input_output_aliases)
            schedule     ;; the S6 Schedule map (reserved; nil until S6 fills it)
            target       ;; device-id + (future) HardwareDescriptor
            descriptor   ;; the raw compile-gpu-program descriptor — inspectable
            args]         ;; captured example args (:all-params order): resident bind contents + shape source
  clojure.lang.IFn
  (invoke [this inputs] (invoke-compiled this inputs))
  (invoke [this] (invoke-compiled this {}))
  (applyTo [this argseq] (apply invoke-compiled this argseq)))

(defn compiled? [x] (instance? Compiled x))

;; ================================================================
;; Role derivation (§4.2) and tree construction
;; ================================================================

(defn- derive-roles
  "Effective {sym → role} for bind-program!: donated → :state, constants → :constant, the rest
   fall through to the descriptor's derived defaults (read-only→:input, written→:output)."
  [descriptor donate constants explicit-roles]
  (let [donate-set   (set donate)
        constant-set (set constants)]
    (merge (into {} (map (fn [s] [s :state]) donate-set))
           (into {} (map (fn [s] [s :constant]) constant-set))
           explicit-roles)))

(defn- flat-n
  "Element count of a JVM array (the flat logical shape for the MVP)."
  [arr]
  (when arr (java.lang.reflect.Array/getLength arr)))

(defn- build-in-tree
  [descriptor argmap donate constants effective-roles]
  (let [donate-set (set donate)]
    (vec (for [p (:array-params descriptor)
               :let [arr (get argmap p)]]
           {:key     (keyword (name p))
            :sym     p
            :role    (get effective-roles p (get (:array-roles descriptor) p :input))
            :donate? (contains? donate-set p)
            :shape   (when arr [(flat-n arr)])
            :dtype   (:dtype descriptor)}))))

(defn- build-out-tree
  "Out-tree = donated in→out nodes + any explicit :outputs + the functional :result-sym + taps.
   Each projects to a DeviceArray over a resident buffer (§3.4 multi-output)."
  [descriptor argmap donate outputs result-sym taps dtype]
  (let [donate-nodes  (for [s donate]
                        {:key (keyword (str (name s) "'")) :sym s
                         :shape (when-let [a (get argmap s)] [(flat-n a)]) :dtype dtype
                         :from :donated})
        output-nodes  (for [s outputs]
                        {:key (keyword (name s)) :sym s
                         :shape (when-let [a (get argmap s)] [(flat-n a)]) :dtype dtype
                         :from :output})
        result-node   (when result-sym
                        [{:key (keyword (name result-sym)) :sym result-sym
                          :shape nil :dtype dtype :from :result}])
        tap-nodes     (for [s taps]
                        {:key (keyword (name s)) :sym s :shape nil :dtype dtype :from :tap})]
    (vec (concat donate-nodes output-nodes result-node tap-nodes))))

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
            :inputs  [sym …]         per-call uploads (default: descriptor-derived)
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
                      gemm-precision (conj :gemm-precision gemm-precision)))
        _ (when-not prog
            (throw (ex-info "compile: compile-gpu-program returned nil — a step fell back to host (non-resident). Pass :on-non-resident :throw to see which."
                            {:fn fn-var :target target})))
        argmap    (zipmap (:all-params prog) args)
        eff-roles (derive-roles prog donate constants roles)
        result-sym (:result-sym prog)
        sess    (gpu/make-session target)
        handle  (gpu/bind-program! sess prog args eff-roles
                                   (cond-> {} profile? (assoc :profile? true)))
        in-tree  (build-in-tree prog argmap donate constants eff-roles)
        out-tree (build-out-tree prog argmap donate outputs result-sym taps dtype)
        donated  (into {} (map (fn [s] [(keyword (name s)) (keyword (str (name s) "'"))]) donate))]
    (->Compiled sess handle in-tree out-tree donated schedule target prog args)))

;; ================================================================
;; Functional invocation (§2.3)
;; ================================================================

(defn- project-node
  "Wrap a resident output node's live session buffer as an ::external DeviceArray (the session
   owns the buffer's lifetime; the value never frees it). Returns nil if the node has no resident
   buffer (e.g. a Void map-void result)."
  [session handle {:keys [sym shape dtype]} target]
  (let [k   (gpu/resident-key session handle sym)
        buf (gpu/buffer session k)]
    (when buf
      (v/wrap-external buf target
                       (or dtype (:dtype buf))
                       (or shape [(:n-elements buf)])))))

(defn invoke-compiled
  "Replay the artifact and return device values. `inputs` : {in-key → DeviceArray|host-array}.
     1. consume any donated-slot DeviceArrays passed in (donation-invalidation, §1.3);
     2. assemble the args vector (captured resident contents; :input keys overridden by inputs);
     3. replay the ONE recorded graph with NO host download (replay-program!);
     4. project out-tree nodes as ::external DeviceArrays over the resident buffers.
   Mutation of resident :state is invisible: the caller sees fresh output values and the old
   donated inputs invalidated — never a mutation."
  [^Compiled c inputs]
  (let [{:keys [session handle in-tree out-tree donated descriptor target args]} c
        key->sym  (into {} (map (juxt :key :sym)) in-tree)
        ;; 1. donation-invalidation: consume passed DeviceArrays bound to a donated slot.
        _ (doseq [[k _out] donated]
            (let [val (get inputs k)]
              (when (v/device-array? val) (v/consume! val))))
        ;; 2. assemble args: start from captured, override :input-role keys the caller passed.
        input-syms (set (map :sym (filter #(= :input (:role %)) in-tree)))
        argmap (reduce (fn [m [k val]]
                         (let [s (get key->sym k)]
                           (if (and s (contains? input-syms s))
                             (assoc m s (if (v/device-array? val) (v/->host val) val))
                             m)))
                       (zipmap (:all-params descriptor) args)
                       inputs)
        call-args (mapv argmap (:all-params descriptor))]
    ;; 3. replay, no download.
    (gpu/replay-program! session handle call-args)
    ;; 4. project outputs as resident device values.
    (into {} (keep (fn [{:keys [key] :as node}]
                     (when-let [da (project-node session handle node target)]
                       [key da])))
          out-tree)))

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

(defn profile
  "Device-event profile of one replay (delegates to profile-program!). The artifact must have
   been compiled with {:profile? true}. Returns profile-program!'s map."
  [^Compiled c]
  (gpu/profile-program! (:session c) (:handle c) (:args c)))

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
  "Release the artifact's session (frees resident buffers + destroys graphs/kernels)."
  [^Compiled c]
  (gpu/close-session! (:session c))
  nil)
