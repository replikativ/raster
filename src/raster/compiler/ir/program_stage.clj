(ns raster.compiler.ir.program-stage
  "Pure semantic staging for compiled resident program descriptors.

   A caller declares a stable stage identity plus the state and value effects that form its public
   boundary. Raster finds the unique minimal ordered step interval implementing those effects,
   derives its inputs and internal writes from checked executable ABIs, and rejects ambiguous or
   leaking boundaries. This is operation-neutral: attention, solvers, collectives and external I/O
   use the same effect contract."
  (:refer-clojure :exclude [partition])
  (:require [clojure.set :as set]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-executable :as kexec]))

(defrecord ProgramStage [id start end inputs outputs state internal writes steps attributes])

(defn program-stage?
  [value]
  (and value (= "raster.compiler.ir.program_stage.ProgramStage"
                (.getName (class value)))))

(defn- validate-descriptor!
  [descriptor]
  (when-not (and (map? descriptor) (vector? (:steps descriptor)) (seq (:steps descriptor))
                 (vector? (:all-params descriptor)) (vector? (:array-params descriptor))
                 (vector? (:scalar-params descriptor)))
    (throw (ex-info "program staging requires a compiled resident program descriptor"
                    {:reason :program-stage-descriptor :descriptor descriptor})))
  descriptor)

(defn- slot-access
  [{:keys [kind role]}]
  (cond
    (= :inout role) :read-write
    (= :input kind) :read
    (= :output kind) :write
    :else nil))

(defn- merge-access
  [left right]
  (cond
    (nil? left) right
    (nil? right) left
    (= left right) left
    :else :read-write))

(defn- reads?
  [access]
  (contains? #{:read :read-write} access))

(defn- writes?
  [access]
  (contains? #{:write :read-write} access))

(defn- step-interface
  [step]
  (or (:artifact step)
      (some-> (:dispatch step) kdispatch/default-alternative)
      (throw (ex-info "program stage step has no executable interface"
                      {:reason :program-stage-interface
                       :phase (:phase step) :convention (:convention step)}))))

(defn step-accesses
  "Return `{compiler-symbol :read|:write|:read-write}` for one descriptor step.

   Access comes from the executable ABI, not parameter spelling conventions. Scatter retains its
   explicit three-buffer contract until it is represented by an ordinary executable ABI."
  [step]
  (if (= :scatter (:convention step))
    (let [[output source index :as arrays] (:arrays step)]
      (when-not (and (= 3 (count arrays)) (every? symbol? arrays))
        (throw (ex-info "program stage scatter requires output, source and index symbols"
                        {:reason :program-stage-scatter :phase (:phase step) :arrays arrays})))
      {output :write source :read index :read})
    (let [interface (kexec/validate! (step-interface step))
          slot-groups (kabi/logical-pointer-slot-groups (kexec/abi interface))
          logical-names (mapv :binding slot-groups)
          pointer-specs (filterv #(not= :scalar (:kind %)) (:argument-specs step))
          symbols (mapv :sym pointer-specs)]
      (when-not (= (count logical-names) (count symbols))
        (throw (ex-info "program stage pointer plan differs from its executable ABI"
                        {:reason :program-stage-abi :phase (:phase step)
                         :symbols symbols :abi-bindings logical-names})))
      (when-not (every? symbol? symbols)
        (throw (ex-info "program stage pointer plan requires compiler symbols"
                        {:reason :program-stage-symbols :phase (:phase step)
                         :symbols symbols})))
      (reduce (fn [accesses [symbol {:keys [binding slots]}]]
                (let [access (reduce merge-access nil
                                     (map slot-access slots))]
                  (when-not access
                    (throw (ex-info "program stage ABI pointer has no readable or writable effect"
                                    {:reason :program-stage-access :phase (:phase step)
                                     :symbol symbol :binding binding})))
                  (update accesses symbol merge-access access)))
              {}
              (map vector symbols slot-groups)))))

(defn descriptor-accesses
  "Return the ordered per-step access maps of a resident descriptor."
  [descriptor]
  (mapv step-accesses (:steps (validate-descriptor! descriptor))))

(defn- symbols-read
  [accesses]
  (into #{} (keep (fn [[symbol access]] (when (reads? access) symbol))) accesses))

(defn- symbols-written
  [accesses]
  (into #{} (keep (fn [[symbol access]] (when (writes? access) symbol))) accesses))

(defn- stage-inputs
  [accesses]
  (:inputs
   (reduce (fn [{:keys [written] :as state} step]
             (let [reads (symbols-read step)
                   writes (symbols-written step)]
               (-> state
                   (update :inputs into (set/difference reads written))
                   (update :written into writes))))
           {:inputs #{} :written #{}}
           accesses)))

(defn select
  "Select and validate one replaceable semantic stage.

   Contract keys:
   - `:id` is the stable semantic identity.
   - `:state` names externally persistent values written by the stage.
   - `:outputs` names ordinary values exported to later steps or the program result.
   - `:internal` names uniquely-written values used only to constrain the interval.
   - `:anchors` may override the union of those three effect classes.
   - `:attributes` is inspectable caller metadata.

   Every anchor must have exactly one writer. The minimal span containing those writers must
   classify every live value escaping the span as state or output; otherwise selection fails
   rather than silently cutting through a dependency."
  [descriptor {:keys [id state outputs internal anchors attributes]
               :or {state #{} outputs #{} internal #{} attributes {}}}]
  (let [descriptor (validate-descriptor! descriptor)
        state (set state)
        outputs (set outputs)
        internal (set internal)
        classified (set/union state outputs internal)
        anchors (set (or anchors classified))
        accesses (descriptor-accesses descriptor)]
    (when (nil? id)
      (throw (ex-info "program stage requires a stable identity"
                      {:reason :program-stage-id})))
    (when-not (and (every? symbol? state) (every? symbol? outputs) (every? symbol? internal)
                   (every? symbol? anchors) (map? attributes))
      (throw (ex-info "program stage effects must be symbol sets and attributes must be a map"
                      {:reason :program-stage-contract :state state :outputs outputs
                       :internal internal :anchors anchors :attributes attributes})))
    (when-not (= (+ (count state) (count outputs) (count internal)) (count classified))
      (throw (ex-info "program stage state, outputs and internal anchors must be disjoint"
                      {:reason :program-stage-effect-overlap
                       :state state :outputs outputs :internal internal})))
    (when-not (and (seq anchors) (set/subset? classified anchors))
      (throw (ex-info "program stage anchors must include every classified effect"
                      {:reason :program-stage-anchors :anchors anchors
                       :required classified})))
    (when-let [unclassified (seq (set/difference anchors classified))]
      (throw (ex-info "program stage anchors must be classified as state, output or internal"
                      {:reason :program-stage-unclassified-anchor :anchors (set unclassified)})))
    (let [writers-by-symbol
          (into {}
                (map (fn [symbol]
                       [symbol
                        (into []
                              (keep-indexed (fn [index step]
                                              (when (writes? (get step symbol)) index)))
                              accesses)]))
                anchors)]
      (doseq [[symbol writers] writers-by-symbol]
        (when-not (= 1 (count writers))
          (throw (ex-info "program stage anchor must have exactly one ordered writer"
                          {:reason :program-stage-ambiguous-anchor :stage id
                           :symbol symbol :writers writers}))))
      (let [writer-indexes (mapv (comp first val) writers-by-symbol)
            start (apply min writer-indexes)
            end (inc (apply max writer-indexes))
            selected-accesses (subvec accesses start end)
            after-accesses (subvec accesses end)
            writes (into #{} (mapcat symbols-written) selected-accesses)
            reads-after (into #{} (mapcat symbols-read) after-accesses)
            result-write (let [result (:result-sym descriptor)]
                           (if (and (symbol? result) (contains? writes result)) #{result} #{}))
            boundary-writes (set/union (set/intersection writes reads-after)
                                       result-write)
            declared-boundary (set/union state outputs)
            missing-boundary (set/difference boundary-writes declared-boundary)
            unwritten-effects (set/difference classified writes)]
        ;; Declared state/outputs may intentionally escape the whole descriptor and therefore be
        ;; dead under descriptor-local liveness. They must be written; every locally live write,
        ;; however, must be classified.
        (when (or (seq missing-boundary) (seq unwritten-effects))
          (throw (ex-info "program stage boundary effects differ from its declared state and outputs"
                          {:reason :program-stage-boundary :stage id
                           :declared declared-boundary :live boundary-writes
                           :missing missing-boundary :unwritten unwritten-effects
                           :range [start end]})))
        (->ProgramStage id start end (stage-inputs selected-accesses) outputs state internal writes
                        (subvec (:steps descriptor) start end) attributes)))))

(defn- projected-descriptor
  [descriptor stage part [start end]]
  (when (< start end)
    (let [steps (subvec (:steps descriptor) start end)
          used-symbols (into #{} (mapcat (comp keys step-accesses)) steps)]
      (-> descriptor
          (assoc :steps steps
                 :allocs (filterv #(contains? used-symbols (:sym %)) (:allocs descriptor))
                 :array-roles (select-keys (:array-roles descriptor) used-symbols)
                 :stage {:id (:id stage) :part part :source-range [start end]
                         :inputs (:inputs stage) :outputs (:outputs stage)
                         :state (:state stage) :internal (:internal stage)
                         :attributes (:attributes stage)})))))

(defn partition
  "Project a descriptor into optional `:before`, non-empty `:selected`, and optional `:after`
   descriptors around a validated ProgramStage. Scalar/all-parameter order is preserved so the
   compiler-authored size and scalar closures remain valid; unused scratch allocation contracts
   are removed from each projection."
  [descriptor stage]
  (let [descriptor (validate-descriptor! descriptor)
        stage (if (program-stage? stage)
                stage
                (throw (ex-info "program stage partition requires a ProgramStage"
                                {:reason :program-stage-type :actual (type stage)})))
        n (count (:steps descriptor))
        start (:start stage)
        end (:end stage)]
    (when-not (and (<= 0 start) (< start end) (<= end n)
                   (= (:steps stage) (subvec (:steps descriptor) start end)))
      (throw (ex-info "program stage does not identify this descriptor interval"
                      {:reason :program-stage-mismatch :stage (:id stage)
                       :range [start end] :step-count n})))
    {:before (projected-descriptor descriptor stage :before [0 start])
     :selected (projected-descriptor descriptor stage :selected [start end])
     :after (projected-descriptor descriptor stage :after [end n])}))
