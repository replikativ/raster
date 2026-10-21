(ns raster.compiler.passes.parallel.checked-map-stage
  "A checked map schedule with conditional result initialization.

   The semantic conversion stays in the source equation. This refinement computes
   its Double operand once, guards Float materialization, and records private
   per-lane status/diagnostics. It does NOT grant execution or initialization:
   the enclosing stage owner must observe successful completion first."
  (:require [clojure.set :as set]
            [raster.compiler.backend.intrinsics :as intrinsics]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.core.scalar-conversion :as conversion]
            [raster.compiler.ir.extent-proof :as extent-proof]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-graph :as graph]
            [raster.compiler.ir.numerical-contract :as numerics]
            [raster.compiler.ir.scheduled-graph-refinement :as refinement]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled]
            [raster.compiler.ir.semantic-fingerprint :as semantic]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as soac]
            [raster.compiler.passes.parallel.map-read-requirements :as reads]
            [raster.compiler.passes.parallel.scheduled-equation-graph :as equation-graph]
            [raster.compiler.passes.parallel.segmap-body :as map-body]))

(def kind :checked-float-map)

(def ^:private source-conversion
  {:source-dtype :double :target-dtype :float
   :rounding :nearest-even :overflow :trap
   :source-op 'clojure.core/float})

(defn candidate?
  "A cheap syntactic trigger, never admission evidence. plan rechecks the
   complete source, storage, read and numerical conditions independently."
  [algorithm source-graph]
  (and (soac/program-form? algorithm)
       (= 1 (count (soac/equations algorithm)))
       (= 'map (soac/operation-kind (first (soac/equations algorithm))))
       (= 1 (count (:nodes source-graph)))
       (= source-conversion (get-in (conversion/canonical-parts
                         (get-in source-graph [:nodes 0 :operation :scalar-region :result]))
                        [:attributes]))))

(defn- decline! [message data]
  (throw (ex-info message (assoc data :reason :checked-map-stage-declined))))

(defn- record-kind? [kind value]
  (= (str "raster.compiler.ir.kernel_body." kind) (some-> value class .getName)))

(defn- fresh-symbols [source prefixes]
  (let [occupied (volatile! (set (filter symbol? (tree-seq coll? seq source))))]
    (mapv (fn [prefix]
            (let [id (first (remove @occupied (map #(symbol (str prefix %)) (range))))]
              (vswap! occupied conj id) id)) prefixes)))

(defn- total-expression! [expression]
  (doseq [expression (filter #(record-kind? "ScalarExpr" %)
                            (tree-seq coll? seq expression))]
    (let [descriptor (intrinsics/descriptor (:op expression))]
      (when-not
       (and (not= :trap (get-in expression [:options :overflow]))
            (or (= :cast (:op expression))
                (= :cmp (:kind descriptor))
                (and (contains? #{:float :double} (:result-type expression))
                     (true? (:primitive-fp-total descriptor)))))
        (decline! "checked map operand has another potentially failing operation"
                  {:expression expression})))))

(defn- literal-loop-bound [expression]
  (cond
    (integer? expression) expression
    (and (record-kind? "Literal" expression)
         (contains? #{:int :long} (:type expression))
         (integer? (:value expression))) (:value expression)
    (and (record-kind? "IndexCast" expression)
         (= :exact (:overflow expression))
         (= :long (:dtype expression))) (literal-loop-bound (:argument expression))))

(defn- integral-header-leaf? [expression]
  (or (integer? expression) (symbol? expression)
      (and (record-kind? "Literal" expression)
           (contains? #{:int :long} (:type expression)))
      (and (record-kind? "IndexCast" expression)
           (= :long (:dtype expression)) (= :exact (:overflow expression))
           (integral-header-leaf? (:argument expression)))))

(defn- proved-unit-headers [kernel]
  ;; Validation proves the actual loop's bounds and induction value have the
  ;; same Long representation. No parameter hint or producer-supplied interval
  ;; is accepted here. Membership remains tied to exact operation occurrences.
  (body/validate! kernel)
  (let [occurrences (java.util.IdentityHashMap.)]
    (letfn [(visit [operations]
              (doseq [operation operations]
                (when (record-kind? "ForLoop" operation)
                  (.put occurrences operation (inc (long (or (.get occurrences operation) 0)))))
                (cond
                  (or (record-kind? "Guard" operation) (record-kind? "ForLoop" operation))
                  (visit (:operations operation))
                  (record-kind? "IfRegion" operation)
                  (do (visit (:then-operations operation)) (visit (:else-operations operation))))))]
      (visit (:operations kernel)))
    (fn [operation]
      (and (= 1 (.get occurrences operation))
           (= :long (get-in operation [:index :type]))
           (= 1 (:step operation))
           (not= :inclusive (get-in operation [:attributes :upper-bound]))
           (integral-header-leaf? (:lower operation))
           (integral-header-leaf? (:upper operation))))))

(defn- otherwise-total!
  ([operations] (otherwise-total! operations (constantly false)))
  ([operations proved-header?]
  ;; This is a deliberately closed schedule grammar, not a purity claim. A
  ;; validated load also requires the independent source address proof below.
  (doseq [operation operations]
    (cond
      (record-kind? "ScalarLoad" operation) nil
      (record-kind? "Yield" operation) (doseq [value (:values operation)] (total-expression! value))
      (record-kind? "ScalarCompute" operation)
      (total-expression! (:expression operation))
      (record-kind? "IfRegion" operation)
      (do (total-expression! (:condition operation))
          (otherwise-total! (:then-operations operation) proved-header?)
          (otherwise-total! (:else-operations operation) proved-header?))
      (record-kind? "ForLoop" operation)
      (let [lower (literal-loop-bound (:lower operation))
            upper (literal-loop-bound (:upper operation))]
        ;; For actual validated Long leaves, lower>=upper is zero-trip; otherwise
        ;; each executed index is <= upper-1, so the unit increment fits Long.
        ;; This proves finiteness, NOT a practical runtime or array-read bound.
        ;; Unowned controls retain the earlier conservative static proof.
        (when-not (or (proved-header? operation)
                      (and (= :long (get-in operation [:index :type]))
                           (= 1 (:step operation))
                           (not= :inclusive (get-in operation [:attributes :upper-bound]))
                           (integer? lower) (integer? upper)
                           (<= 0 lower upper Long/MAX_VALUE)))
          (decline! "checked map fold needs a finite nonoverflowing counted domain"
                    {:operation operation}))
        (doseq [argument (:iter-args operation)] (total-expression! (:initial argument)))
        (otherwise-total! (:operations operation) proved-header?))
      :else (decline! "checked map operand is not a total scalar/load region"
                      {:operation operation})))))

(defn- private-result-storage? [facts equation output]
  (let [results (nth equation 2)
        storage (get-in facts [:attributes :result-storage])
        host-return (:host-return (first storage))]
    (and (= 1 (count results))
         (= #{:memory/write} (:effects facts))
         (= {(first results) output} (:aliases facts))
         ;; Returning the destination or only its effect token does not change
         ;; the one-store schedule. Neither form proves physical ownership:
         ;; the same managed observer still owes invocation-private output.
         (contains? #{:buffer :effect} host-return)
         (= [{:destination output :access :write :host-return host-return}] storage))))

(defn- same-output-domain?
  [algorithm source-graph scheduled-body bound output]
  (or (= bound (:elements output))
      (when scheduled-body
        ;; The body, not a certificate's attached definitions, owns the causal
        ;; scalar equations. Reconstruct its entire source graph before using
        ;; the original storage shape as a volume witness. Scalar execution and
        ;; its checked overflow remain in the program, not in this proof.
        (let [expected (equation-graph/make
                        algorithm scheduled-body
                        (select-keys source-graph [:provenance :attributes]))
              numerical (filterv #(not (true? (get-in % [:attributes :host-only])))
                                 (:equations scheduled-body))
              equation (first numerical)]
          (and (semantic/equivalent? expected source-graph)
               (= 1 (count numerical))
               (semantic/equivalent? algorithm (:algorithm equation))
               (extent-proof/same-volume?
                (extent-proof/parallel-environment-before scheduled-body equation)
                bound (get-in scheduled-body [:values (:id output) :shape])))))))

(defn plan
  "Derive one checked result obligation and its complete physical schedule.
   A single-result map, either functional or an exact write-only destination,
   retains an explicit invocation-private output requirement. The managed owner
   must prove that requirement before execution; roles/effects alone do not.
   Unsupported candidates retain their original semantic operation."
  ([algorithm source-graph options] (plan algorithm source-graph options nil))
  ([algorithm source-graph options scheduled-body]
  (let [algorithm (soac/validate! algorithm)
        source-graph (graph/validate! source-graph)
        equation (first (soac/equations algorithm))
        facts (get-in (soac/facts algorithm) [:equations (second equation)])
        node (first (:nodes source-graph))
        operation (:operation node)
        region (:scalar-region operation)
        {:keys [attributes operand]} (conversion/canonical-parts (:result region))
        output (:out-sym operation)
        {index :name bound :bound} (first (get-in operation [:space :dims]))
        unmet (cond
                (not (and (= 1 (count (soac/equations algorithm)))
                          (= 'map (soac/operation-kind equation)))) :single-map-equation
                (not (and (= 1 (count (:nodes source-graph)))
                          (segop/seg-map? operation))) :single-map-node
                (not= 1 (count (get-in operation [:space :dims]))) :one-dimensional-domain
                (seq (:temporaries source-graph)) :no-existing-temporaries
                (not (and (= 1 (count (:outputs source-graph)))
                          (= output (:id (first (:outputs source-graph))))
                          (= :output (:role (first (:outputs source-graph))))
                          (same-output-domain? algorithm source-graph scheduled-body bound
                                               (first (:outputs source-graph))))) :single-output-boundary
                (not (or (and (empty? (:effects facts)) (empty? (:aliases facts))
                              (empty? (get-in facts [:attributes :result-storage])))
                         (private-result-storage? facts equation output))) :private-result-storage
                (not= #{output} (:outputs operation)) :single-output-store
                (contains? (:inputs operation) output) :write-only-output
                (not (and (vector? (:locals region))
                          (not (contains? region :effects)))) :value-producing-scalar-region
                (not= source-conversion attributes) :checked-outer-result-conversion)
        _ (when unmet
            (decline! "checked map requires a single write-only outer Double-to-Float result"
                      {:operation (:id operation) :conversion attributes :requirement unmet}))
        projected (reads/validate-and-project-addresses operation node source-graph)
        requirements (get-in node [:read-capacity-certificate :requirements])
        _ (when-not (set/subset? (:inputs operation) (set (keys requirements)))
            (decline! "checked map requires a complete independent source read-domain proof"
                      {:inputs (:inputs operation) :proved (set (keys requirements))}))
        [status diagnostics below above converted]
        (fresh-symbols [algorithm source-graph]
                       ["checked_status_" "checked_operand_" "checked_below_"
                        "checked_above_" "checked_float_"])
        ;; This private schedule input exposes the original Double producer. No
        ;; unchecked source conversion is invented and no target emits this
        ;; intermediate value as the source result.
        producer (-> projected
                     (assoc :dtype :double :cast-fn nil)
                     (assoc-in [:scalar-region :result] operand))
        scalar-types (merge (:scalar-types options)
                            (into {} (map (juxt :id :dtype)) (:scalars source-graph)))
        shapes (into {} (map (fn [buffer] [(:id buffer) [(:elements buffer)]]))
                     (concat (:inputs source-graph) (:outputs source-graph)))
        lowered (map-body/lower producer
                                (merge options
                                       {:array-types (assoc (into {} (map (juxt :id :dtype))
                                                                  (:inputs source-graph)) output :double)
                                        :array-shapes shapes :scalar-types scalar-types}))
        original-body (:kernel-body lowered)
        guard (first (:operations original-body))
        operations (:operations guard)
        store (last operations)
        _ (when-not (and (= 1 (count (:operations original-body)))
                         (record-kind? "Guard" guard) (= :map-active (:mask guard))
                         (record-kind? "ScalarStore" store) (= output (:buffer store))
                         (= [index] (:coordinates store))
                         (= :map-active (:predicate store)))
            (decline! "checked map requires one lane-owned final result store" {}))
        _ (otherwise-total! (butlast operations) (proved-unit-headers original-body))
        extent ['_n_bound]
        failure [(body/->ScalarStore diagnostics [index] (:value store) :map-active)
                 (body/->ScalarStore status [index] (body/literal 1 :int) :map-active)
                 (body/->Yield [])]
        success [(body/->ScalarCompute (body/value converted :float)
                                       (body/cast-expression (:value store) :float :nearest-even :ieee))
                 (assoc store :value converted)
                 (body/->ScalarStore status [index] (body/literal 0 :int) :map-active)
                 (body/->Yield [])]
        checked-operations
        (into (vec (butlast operations))
              [(body/->ScalarCompute (body/value below :predicate)
                                     (body/scalar-expression :lt :predicate
                                       [(:value store) (body/literal (- (double Float/MAX_VALUE)) :double)]))
               (body/->ScalarCompute (body/value above :predicate)
                                     (body/scalar-expression :gt :predicate
                                       [(:value store) (body/literal (double Float/MAX_VALUE) :double)]))
               (body/->IfRegion below failure
                               [(body/->IfRegion above failure success []) (body/->Yield [])] [])])
        obligation {:kind kind :node (:id node) :extent bound :output output
                    :output-ownership :invocation-private
                    :status status :diagnostics diagnostics :source-conversion attributes
                    :failure-selection :first-linear-index
                    :initialization :conditional-on-success}
        kernel-body
        (body/make
         (-> original-body
             (update :parameters
                     (fn [parameters]
                       (into (mapv #(if (= output (:id %))
                                      (assoc % :dtype :float :layout (layout/row-major (:shape %) :float)) %) parameters)
                             [(body/->KernelParameter status :output :int extent :global
                                                      (layout/row-major extent :int) :status)
                              (body/->KernelParameter diagnostics :output :double extent :global
                                                      (layout/row-major extent :double) :diagnostics)])))
             (assoc :operations [(assoc guard :operations checked-operations)])
             (assoc-in [:attributes :checked-stage-result] obligation)))
        arguments (mapv #(if (= '_n_bound (:id %)) bound (:id %)) (:parameters kernel-body))
        uses (scheduled/derive-uses kernel-body arguments)
        certificate
        (scheduled/make
         {:source operation :body kernel-body :arguments arguments
          :scalar-bindings (scheduled/derive-scalar-bindings kernel-body arguments scalar-types)
          :effects {:kind :checked-elementwise-map :uses uses :write-conflict :unique}
          :legality {:kind kind} :numerics {:mode :exact :policy :same-scalar-evaluation-order
                                          :scalar-math (numerics/validate-scalar-math-policy!
                                                        (:scalar-math options))}
          :provenance {:source-dialect :typed-soac}
          :attributes {:checked-stage-result obligation}})
        checked-graph
        (graph/make
         (-> source-graph
             (assoc :temporaries [(graph/buffer status :int bound :device :temporary)
                                  (graph/buffer diagnostics :double bound :device :temporary)])
             (assoc-in [:nodes 0 :uses]
                       (mapv #(graph/->ValueUse (:value %) (:access %)) uses))
             (assoc-in [:attributes :checked-stage-results] [obligation])))
        witness (refinement/make {:source source-graph :graph checked-graph
                                  :schedule {:kind kind :workgroup-size (get options :workgroup-size 256)}
                                  :numerics (:numerics certificate)})]
    (scheduled/validate-against-node! certificate (first (:nodes checked-graph)) checked-graph)
    {:refinement witness :graph checked-graph :stage-bodies [certificate]
     :stage-results [obligation]})))

(defn validate-reconstruction!
  "Reconstruct from the original semantic operation and source graph. Caller
   metadata or an emitted ABI never substitutes for this complete comparison."
  ([algorithm source-graph witness options]
   (validate-reconstruction! algorithm source-graph witness options nil))
  ([algorithm source-graph witness options scheduled-body]
  (refinement/validate-against! witness source-graph)
  (let [expected (plan algorithm source-graph
                       (assoc options :workgroup-size (get-in witness [:schedule :workgroup-size]))
                       scheduled-body)]
    (when-not (semantic/equivalent? (:refinement expected) witness)
      (throw (ex-info "checked stage refinement changed its exact source-derived schedule"
                      {:reason :checked-map-stage-refinement})))
    expected)))
