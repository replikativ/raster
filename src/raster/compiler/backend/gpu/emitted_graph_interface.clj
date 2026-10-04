(ns raster.compiler.backend.gpu.emitted-graph-interface
  "Shared ordered public ABI finalization for artifact-valued GPU graphs."
  (:require [clojure.set]
            [raster.compiler.core.dtype :as dt]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.scheduled-kernel-body :as scheduled-body]))

(defn finalize!
  "Give an artifact-valued graph its one ordered external ABI.

   Node artifacts remain free to order their own physical parameters. The graph boundary groups
   equal scalar uses, proves their emitted dtypes agree, and exposes every external buffer exactly
   once. An explicit target-neutral GraphScalar interface is authoritative: target artifacts may
   choose a physical kernel dtype, but cannot invent, omit, reorder, or retype public scalars."
  [emitted target-dialect scalar-types]
  (let [external-buffers (vec (distinct (concat (:inputs emitted) (:outputs emitted))))
        declared-scalars (:scalars emitted)
        declared-scalar-ids (set (map :id declared-scalars))
        ;; Artifacts own physical parameter order and kernel representation; the scheduled graph
        ;; owns the logical scalar dtype. Project that logical fact into every direct use before
        ;; KernelExecutable checks node-to-boundary agreement. Derived expressions remain private
        ;; and are checked for closure below.
        declared-by-id (into {} (map (juxt :id identity)) declared-scalars)
        ;; Typed scheduled graphs carry GraphScalar declarations. Compatibility graphs that have
        ;; not migrated yet still receive compiler-owned scalar facts from the enclosing program.
        ;; In both cases project only direct public arguments; target-private expressions retain
        ;; their independently checked representation.
        logical-dtype-by-id (merge (into {} (map (fn [[id scalar-dtype]]
                                                   [id (dt/canon scalar-dtype)]))
                                          scalar-types)
                                   (into {} (map (fn [[id scalar]] [id (:dtype scalar)]))
                                         declared-by-id))
        emitted (if (seq logical-dtype-by-id)
                  (kgraph/map-operations
                   emitted
                   (fn [node]
                     (let [artifact (:operation node)
                           certificate (get-in artifact
                                               [:provenance :scheduled-operation])]
                       (if (scheduled-body/scheduled-kernel-body? certificate)
                         (do
                           (doseq [[slot argument] (map vector (:abi artifact)
                                                        (:arguments artifact))
                                   :let [logical-dtype
                                         (and (= :scalar (:kind slot))
                                              (get logical-dtype-by-id argument))]
                                   :when logical-dtype]
                             (when-not (= logical-dtype (:dtype slot))
                               (throw (ex-info
                                       "scheduled-body ABI disagrees with its GraphScalar"
                                       {:reason :kernel-graph-scalar-interface
                                        :node (:id node) :argument argument
                                        :expected logical-dtype :actual (:dtype slot)}))))
                           artifact)
                         (kart/validate!
                          (update artifact :abi
                                  (fn [slots]
                                    (mapv (fn [slot argument]
                                            (if-let [logical-dtype
                                                     (and (= :scalar (:kind slot))
                                                          (get logical-dtype-by-id argument))]
                                              (assoc slot :dtype logical-dtype)
                                              slot))
                                          slots (:arguments artifact)))))))))
                  emitted)
        scalar-pairs (->> (:nodes emitted)
                          (mapcat (fn [node]
                                    (map vector (get-in node [:operation :abi])
                                         (get-in node [:operation :arguments]))))
                          (filter (fn [[slot argument]]
                                    (and (= :scalar (:kind slot))
                                         (or (symbol? argument)
                                             (keyword? argument)
                                             (contains? declared-scalar-ids argument)))))
                          vec)
        scalar-groups (group-by second scalar-pairs)
        scalar-arguments (if (some? declared-scalars)
                           (mapv :id declared-scalars)
                           (vec (sort-by name (keys scalar-groups))))
        _ (when (and (some? declared-scalars)
                     (not (clojure.set/subset? (set (keys scalar-groups))
                                               declared-scalar-ids)))
            (throw (ex-info "emitted kernel graph invented a public scalar argument"
                            {:reason :kernel-graph-scalar-interface
                             :expected scalar-arguments
                             :actual (vec (sort-by pr-str (keys scalar-groups)))})))
        _ (doseq [[argument pairs] scalar-groups]
            (when-not (apply = (map (comp :dtype first) pairs))
              (throw (ex-info "kernel graph scalar has inconsistent emitted ABI dtypes"
                              {:reason :kernel-graph-scalar-dtype
                               :argument argument :slots (mapv first pairs)}))))
        ;; Public graph slots are not physical kernel parameters. Preserve established names
        ;; where possible, but vector SSA identities need generated names that cannot capture
        ;; a later caller's symbol (or collide with another graph slot).
        interface-ids (concat (map :id external-buffers) scalar-arguments)
        reserved-names (set (keep #(when (instance? clojure.lang.Named %) (name %)) interface-ids))
        interface-names
        (:names
         (reduce (fn [{:keys [used] :as state} [ordinal id]]
                   (let [named? (instance? clojure.lang.Named id)
                         chosen (loop [candidate (if named? (name id) (str "graph_value_" ordinal))]
                                  (if (or (contains? used candidate)
                                          (and (not named?) (contains? reserved-names candidate)))
                                    (recur (str candidate "_")) candidate))]
                     (-> state (update :used conj chosen) (assoc-in [:names id] chosen))))
                 {:used #{} :names {}} (map-indexed vector interface-ids)))
        pointer-abi (mapv (fn [{:keys [id dtype role]}]
                            (kabi/slot id (case role
                                            :input :input
                                            :output :output
                                            :inout :inout)
                                       dtype
                                       :c-name (get interface-names id)
                                       :role (case role
                                               :input :operand
                                               :output :result
                                               :inout :inout)))
                          external-buffers)
        scalar-abi (mapv (fn [argument]
                           (let [slots (mapv first (get scalar-groups argument))
                                 declared (some #(when (= argument (:id %)) %) declared-scalars)
                                 ;; A graph is not a physical kernel. Each node preserves its
                                 ;; own checked conversion (e.g. long parameter versus int bound).
                                 physical-types (distinct (map :kernel-dtype slots))
                                 ;; Integral graph carriers follow the declared public width,
                                 ;; not whichever stage first consumes them. Node int/long
                                 ;; specializations remain checked independently at preflight.
                                 kernel-dtype (or (when (contains? #{:int :long} (:dtype declared))
                                                   (:dtype declared))
                                                  (if (= 1 (count physical-types))
                                                    (first physical-types)
                                                    (:dtype (first slots))))
                                 logical-dtype (or (:dtype declared)
                                                   (get scalar-types argument)
                                                   kernel-dtype)
                                 supplied-dtype (get scalar-types argument)
                                 role (if (some #(= :bound (:role %)) slots)
                                        :bound :parameter)]
                             (when (and declared supplied-dtype
                                        (not= (:dtype declared) (dt/canon supplied-dtype)))
                               (throw (ex-info
                                       "target scalar facts differ from the scheduled interface"
                                       {:reason :kernel-graph-scalar-logical-dtype
                                        :argument argument :scheduled (:dtype declared)
                                        :target supplied-dtype})))
                             (kabi/slot argument :scalar logical-dtype
                                        :c-name (get interface-names argument)
                                        :kernel-dtype kernel-dtype :role role)))
                         scalar-arguments)
        explicit-scalars
        (if (some? declared-scalars)
          declared-scalars
          (mapv (fn [slot argument]
                  (kgraph/scalar argument (:dtype slot)))
                scalar-abi scalar-arguments))
        explicit-nodes
        (mapv (fn [node]
                (let [artifact (:operation node)
                      actual (kgraph/scalar-argument-uses
                              (:abi artifact) (:arguments artifact))
                      scheduled (:scalar-uses node)]
                  (when (and (some? scheduled) (not= scheduled actual))
                    (throw (ex-info "emitted kernel scalar dependencies differ from its schedule"
                                    {:reason :kernel-graph-artifact-scalar-uses
                                     :node (:id node)
                                     :expected scheduled :actual actual})))
                  (assoc node :scalar-uses actual)))
              (:nodes emitted))]
    (-> emitted
        (assoc :abi (vec (concat pointer-abi scalar-abi))
               :arguments (vec (concat (map :id external-buffers) scalar-arguments))
               :scalars explicit-scalars
               :nodes explicit-nodes)
        (assoc-in [:provenance :target-dialect] target-dialect)
        (assoc-in [:attributes :emitted?] true)
        kgraph/validate!)))
