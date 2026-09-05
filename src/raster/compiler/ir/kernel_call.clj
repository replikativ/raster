(ns raster.compiler.ir.kernel-call
  "A verified executable call of one emitted kernel.

   KernelArtifact owns target code, ABI and symbolic launch. KernelCall adds runtime values in the
   identical ABI order and a concrete 1-3D LaunchGeometry. Backends consume this value instead of
   reconstructing arguments or launch dimensions from a convention-specific marker."
  (:require [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-launch :as klaunch])
  (:import [java.lang.foreign MemorySegment]))

(defrecord KernelCall [artifact arguments geometry])

(defn kernel-call?
  "Recognize executable calls across Typed Clojure's child DynamicClassLoaders."
  [x]
  (and x (= "raster.compiler.ir.kernel_call.KernelCall"
            (.getName (class x)))))

(defn logical-argument-plan
  "Project an artifact's physical ABI into ordered logical caller arguments.

   Ordinary pointers and every scalar remain one entry. Contiguous physical pointer slots that
   share `:binding` become one logical pointer entry (an SoA, for example). The physical slots and
   compiler arguments are retained on every entry, so this is a checked view of the one ABI rather
   than a second signature."
  [artifact]
  (let [artifact (kart/validate! artifact)
        pairs (mapv vector (:abi artifact) (:arguments artifact))]
    (reduce (fn [plan [index [slot argument]]]
              (let [pointer? (not= :scalar (:kind slot))
                    binding (when pointer? (or (:binding slot) (:name slot)))
                    previous (peek plan)]
                (if (and pointer?
                         (:pointer? previous)
                         (= binding (:binding previous))
                         (:binding slot))
                  (-> plan
                      pop
                      (conj (-> previous
                                (update :slots conj slot)
                                (update :physical-arguments conj argument)
                                (update :physical-indexes conj index))))
                  (conj plan
                        {:pointer? pointer?
                         :kind (if pointer? :pointer :scalar)
                         :binding (when pointer? binding)
                         :value (if pointer? binding argument)
                         :slots [slot]
                         :physical-arguments [argument]
                         :physical-indexes [index]}))))
            []
            (map-indexed vector pairs))))

(defn logical-arguments
  "Compiler values in logical caller order. SoA field slots collapse to their shared binding."
  [artifact]
  (mapv :value (logical-argument-plan artifact)))

(defn expand-logical-arguments
  "Expand logical runtime values into the artifact's complete physical ABI order.

   `expand-pointer` receives `[plan-entry logical-value]` and must return a vector containing one
   resident value per physical pointer slot. Scalar values are already physical and pass through.
   Representation-specific expansion therefore happens explicitly before `make`, while the
   resulting KernelCall and all backend binders remain physical and backend-neutral."
  [artifact logical-values expand-pointer]
  (let [artifact (kart/validate! artifact)
        plan (logical-argument-plan artifact)]
    (when-not (vector? logical-values)
      (throw (ex-info "logical kernel arguments must be an ordered vector"
                      {:kernel-name (:kernel-name artifact) :arguments logical-values})))
    (when-not (= (count plan) (count logical-values))
      (throw (ex-info "logical kernel argument count mismatch"
                      {:kernel-name (:kernel-name artifact)
                       :expected (count plan) :actual (count logical-values)
                       :plan plan})))
    (vec
     (mapcat (fn [entry value]
               (if (:pointer? entry)
                 (let [expanded (expand-pointer entry value)]
                   (when-not (vector? expanded)
                     (throw (ex-info "logical pointer expansion must return an ordered vector"
                                     {:kernel-name (:kernel-name artifact)
                                      :binding (:binding entry) :expanded expanded})))
                   (when-not (= (count (:slots entry)) (count expanded))
                     (throw (ex-info "logical pointer expansion count differs from physical ABI"
                                     {:kernel-name (:kernel-name artifact)
                                      :binding (:binding entry)
                                      :expected (count (:slots entry))
                                      :actual (count expanded)
                                      :slots (:slots entry)})))
                   expanded)
                 [value]))
             plan logical-values))))

(defn- scalar-value!
  [slot value]
  (when-not (and (map? value) (contains? value :value) (contains? value :type))
    (throw (ex-info "kernel scalar argument must be a typed value"
                    {:slot slot :value value})))
  (when-not (= (:kernel-dtype slot) (:type value))
    (throw (ex-info "kernel scalar argument has the wrong kernel dtype"
                    {:slot slot :expected (:kernel-dtype slot) :actual (:type value)
                     :value value})))
  (when (and (= :int (:kernel-dtype slot))
             (not (and (integer? (:value value))
                       (<= Integer/MIN_VALUE (:value value) Integer/MAX_VALUE))))
    (throw (ex-info "kernel int scalar argument is outside its physical ABI range"
                    {:reason :kernel-scalar-range :slot slot :value value
                     :minimum Integer/MIN_VALUE :maximum Integer/MAX_VALUE})))
  value)

(defn- resident-view
  [value]
  (cond
    (bview/buffer-view? value) value
    (and (map? value) (bview/buffer-view? (:view value))) (:view value)
    :else nil))

(defn- resident-segment
  [value]
  (cond
    (instance? MemorySegment value) value
    (and (map? value) (instance? MemorySegment (:segment value))) (:segment value)
    :else nil))

(defn- segment-overlaps?
  [^MemorySegment left ^MemorySegment right]
  (try
    (let [left-start (.address left)
          right-start (.address right)
          left-end (+ left-start (.byteSize left))
          right-end (+ right-start (.byteSize right))]
      (and (< left-start right-end) (< right-start left-end)))
    (catch UnsupportedOperationException _
      (identical? left right))))

(defn- pointer-overlaps?
  [left right]
  (let [left-view (resident-view left)
        right-view (resident-view right)
        left-segment (resident-segment left)
        right-segment (resident-segment right)]
    (cond
      (identical? left right) true
      (and left-view right-view) (bview/overlaps? left-view right-view)
      (and left-segment right-segment) (segment-overlaps? left-segment right-segment)
      :else (= left right))))

(defn- pointer-aligned?
  [value required]
  (let [view (resident-view value)
        segment (resident-segment value)]
    (cond
      view (and (>= (get-in view [:allocation :alignment]) required)
                (zero? (mod (:byte-offset view) required)))
      (and (map? value) (integer? (:alignment value)))
      (>= (:alignment value) required)
      segment (try
                (zero? (mod (.address ^MemorySegment segment) required))
                (catch UnsupportedOperationException _ false))
      :else false)))

(defn validate!
  "Validate and return a KernelCall. This is driver-independent: a backend subsequently checks
   that pointer values are its resident buffer representation and match ABI storage dtypes."
  [call]
  (when-not (kernel-call? call)
    (throw (ex-info "kernel call must be a KernelCall value"
                    {:call call :actual (type call)})))
  (let [{:keys [artifact arguments geometry]} call
        artifact (kart/validate! artifact)
        abi (:abi artifact)
        arguments (kabi/validate-arguments! abi arguments)
        _ (kabi/validate-alias-contracts! abi arguments pointer-overlaps?)
        geometry (klaunch/validate-geometry! geometry)
        spec (:launch artifact)]
    (when-not (= (klaunch/dimensions spec) (klaunch/dimensions geometry))
      (throw (ex-info "kernel call launch dimensionality differs from its artifact"
                      {:kernel-name (:kernel-name artifact)
                       :spec (:launch artifact) :geometry geometry})))
    ;; Emitted kernels may bake workgroup-sized local arrays or subgroup assumptions into source.
    ;; A scheduler may choose the number of groups per call, but changing this workgroup requires
    ;; re-emission and therefore a different artifact.
    (when-let [static-wg (when (every? integer? (:workgroup-size spec))
                           (klaunch/static-workgroup-size spec))]
      (when-not (= static-wg (:workgroup-size geometry))
        (throw (ex-info "kernel call workgroup differs from its emitted artifact"
                        {:kernel-name (:kernel-name artifact)
                         :expected static-wg :actual (:workgroup-size geometry)}))))
    (when-not (= (:shared-memory-bytes spec) (:shared-memory-bytes geometry))
      (throw (ex-info "kernel call shared memory differs from its emitted artifact"
                      {:kernel-name (:kernel-name artifact)
                       :expected (:shared-memory-bytes spec)
                       :actual (:shared-memory-bytes geometry)})))
    (doseq [[slot value] (map vector abi arguments)]
      (if (= :scalar (:kind slot))
        (scalar-value! slot value)
        (do
          (when (nil? value)
            (throw (ex-info "kernel pointer argument cannot be nil"
                            {:kernel-name (:kernel-name artifact) :slot slot})))
          (when (and (:alignment slot)
                     (not (pointer-aligned? value (:alignment slot))))
            (throw (ex-info "kernel pointer argument violates its ABI alignment contract"
                            {:reason :kernel-abi-pointer-alignment
                             :kernel-name (:kernel-name artifact) :slot slot
                             :required-alignment (:alignment slot)}))))))
    call))

(defn- runtime-number
  [value]
  (if (and (map? value) (contains? value :value)) (:value value) value))

(defn- argument-resolver
  [artifact arguments]
  (let [compiler-arguments (:arguments artifact)]
    (fn [compiler-value]
      (let [indexes (keep-indexed (fn [i value] (when (= compiler-value value) i))
                                  compiler-arguments)]
        (when-not (seq indexes)
          (throw (ex-info "launch value is not present in the artifact argument order"
                          {:kernel-name (:kernel-name artifact)
                           :value compiler-value :arguments compiler-arguments})))
        (let [values (mapv #(runtime-number (nth arguments %)) indexes)]
          (when-not (apply = values)
            (throw (ex-info "repeated launch value resolved to conflicting runtime arguments"
                            {:kernel-name (:kernel-name artifact)
                             :value compiler-value :indexes (vec indexes) :values values})))
          (first values))))))

(defn resolve-value
  "Resolve one compiler value through a checked call's artifact/runtime argument relation. This
   is used for artifact attributes such as a contraction's logical output extent; backends still
   receive no convention-specific positional metadata."
  [call compiler-value]
  (let [{:keys [artifact arguments]} (validate! call)]
    ((argument-resolver artifact arguments) compiler-value)))

(defn make
  "Construct a checked call from an artifact and complete ABI-ordered runtime values.

   The default geometry realizes the artifact launch by resolving compiler values against the
   artifact/runtime argument vectors. `:group-count` is the scheduling seam: it may override the
   realized group vector without changing the emitted workgroup (resident SegRed uses `[1]` to
   keep its partial result entirely on device)."
  ([artifact arguments] (make artifact arguments {}))
  ([artifact arguments {:keys [group-count resolve-value]}]
   (let [artifact (kart/validate! artifact)
         arguments (kabi/validate-arguments! (:abi artifact) arguments)
         resolver (or resolve-value (argument-resolver artifact arguments))
         realized (klaunch/realize (:launch artifact) resolver)
         geometry (if group-count
                    (klaunch/geometry
                     {:workgroup-size (:workgroup-size realized)
                      :group-count group-count
                      :shared-memory-bytes (:shared-memory-bytes realized)})
                    realized)]
     (validate! (->KernelCall artifact arguments geometry)))))

(defn binding-plan
  "Backend-neutral ordered binding plan for a checked call."
  [call]
  (let [{:keys [artifact arguments geometry]} (validate! call)
        pairs (mapv vector (:abi artifact) arguments)]
    {:kernel-name (:kernel-name artifact)
     :target (:target artifact)
     :artifact artifact
     :abi (:abi artifact)
     :arguments arguments
     :pairs pairs
     :pointer-pairs (filterv (fn [[slot _]] (not= :scalar (:kind slot))) pairs)
     :scalar-pairs (filterv (fn [[slot _]] (= :scalar (:kind slot))) pairs)
     :workgroup-size (:workgroup-size geometry)
     :group-count (:group-count geometry)
     :shared-memory-bytes (:shared-memory-bytes geometry)}))

(defn validate-registered!
  "Prove that a runtime registry entry is the artifact named by this call. Runtime-only cached
   handles may extend the record, so compare compiler-owned fields rather than record equality."
  [call registered]
  (let [artifact (:artifact (validate! call))
        registered (kart/validate! registered)
        compiler-fields [:kernel-name :target :source :abi :arguments :launch :temporaries
                         :effects :provenance :attributes]]
    (when-not (= (select-keys artifact compiler-fields)
                 (select-keys registered compiler-fields))
      (throw (ex-info "kernel call artifact differs from the registered artifact"
                      {:kernel-name (:kernel-name artifact)})))
    registered))
