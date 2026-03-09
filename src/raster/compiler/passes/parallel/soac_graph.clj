(ns raster.compiler.passes.parallel.soac-graph
  "Fusion dependency graph with typed edge classification.

  Following Futhark's GraphRep.hs approach: builds a directed graph
  over SOAC nodes with edges classified by fusibility. Applies
  vertical fusion (producer→consumer inlining), horizontal fusion
  (independent same-bound maps), and iterates to fixpoint.

  Edge types:
    :dep     — consumer reads producer's array output (fusible)
    :inf-dep — consumer reads producer's scalar output (infusible)
    :cons    — consumer mutates array producer reads (anti-dep)"
  (:require [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.fusion-support :as fusion-support]
            [raster.compiler.passes.parallel.schedule-support :as schedule-support]
            [clojure.set :as set]))

;; ================================================================
;; FusionGraph record
;; ================================================================

(defrecord FusionGraph
           [nodes        ;; {id -> node}
            edges        ;; #{[from-id to-id edge-type]}
            producer-of  ;; {sym -> node-id}
            adjacency    ;; {id -> #{neighbor-ids}}  (all directions)
            alias-map])  ;; {sym -> root-sym} alias chains resolved transitively

;; ================================================================
;; Graph construction
;; ================================================================

(defn- build-producer-map
  "Map each produced symbol to its node id."
  [nodes-map]
  (reduce-kv
   (fn [acc id node]
     (reduce (fn [m sym] (assoc m sym id))
             acc (soac/node-produced-syms node)))
   {} nodes-map))

(defn- build-alias-map
  "Build a map from alias symbols to their roots.
  A ScalarBinding `h = out__` where `out__` is a simple symbol reference
  creates the alias `h → out__`. A SoacMap with sym `result` and
  outputs `#{out__}` creates `result → out__` (par/map! returns the output).
  Transitively resolves chains: `a = b, b = c` → `{a c, b c}`."
  [nodes-map]
  (let [;; Direct aliases: ScalarBinding with symbol init + SoacMap sym→output
        direct (reduce-kv
                (fn [m id node]
                  (cond
                    ;; ScalarBinding with symbol init: h = out__
                    (and (instance? raster.compiler.ir.soac.ScalarBinding node)
                         (symbol? (:expr node)))
                    (assoc m (:sym node) (:expr node))
                    ;; SoacMap: sym aliases the output buffer (par/map! returns it)
                    ;; Pure par/map has no separate output buffer — skip aliasing
                    (and (instance? raster.compiler.ir.soac.SoacMap node)
                         (not (:pure? node))
                         (= 1 (count (:outputs node))))
                    (assoc m (:sym node) (first (:outputs node)))
                    :else m))
                {} nodes-map)]
    ;; Resolve transitive chains
    (reduce-kv
     (fn [m sym target]
       (loop [t target, depth 0]
         (if (and (< depth 10) (contains? direct t))
           (recur (get direct t) (inc depth))
           (assoc m sym t))))
     {} direct)))

(defn- classify-edge
  "Classify an edge from producer to consumer.
  Returns :dep if consumer reads producer's array output (through aliases),
  :inf-dep if consumer reads producer's scalar binding.

  For SoacMap, both `:outputs` AND `:sym` are array outputs (sym is the
  return value of par/map!, which is the output buffer itself).
  For SoacReduce, the output is always a scalar — edges are :inf-dep."
  [producer-node _consumer-node input-sym alias-map]
  (cond
    ;; Reduce produces a scalar — always infusible dependency
    (instance? raster.compiler.ir.soac.SoacReduce producer-node)
    :inf-dep

    ;; Other SOACs (Map, Scan, Stencil): check if input is an array output
    (soac/soac? producer-node)
    (let [prod-outputs (soac/soac-outputs producer-node)
          prod-sym (:sym producer-node)
          resolved (get alias-map input-sym input-sym)]
      (if (or (contains? prod-outputs input-sym)
              (contains? prod-outputs resolved)
              (= input-sym prod-sym)
              (= resolved prod-sym))
        :dep
        :inf-dep))

    ;; Producer is scalar binding
    :else :inf-dep))

(defn build-fusion-graph
  "Build a FusionGraph from a sequence of SOAC/Scalar nodes.

  For each node, looks up each of its free symbols' producers.
  Creates typed edges based on whether the dependency is through
  an array output (:dep) or scalar (:inf-dep).

  Resolves alias chains: if consumer reads `h` and `h = out__` (ScalarBinding),
  the edge to the SoacMap that produces `out__` is classified as :dep (fusible)."
  [soac-nodes]
  (let [nodes-map (into {} (map (fn [n] [(:id n) n]) soac-nodes))
        producer-of (build-producer-map nodes-map)
        alias-map (build-alias-map nodes-map)
        edges (atom #{})
        adjacency (atom (into {} (map (fn [n] [(:id n) #{}]) soac-nodes)))]
    ;; Build edges.
    ;; For each free symbol in a consumer node, find the producer.
    ;; If the direct producer is a ScalarBinding alias, follow the chain
    ;; to find the ultimate SOAC producer and classify against it.
    (doseq [node soac-nodes]
      (let [free (soac/node-all-free-syms node)]
        (doseq [sym free]
          (when-let [direct-prod-id (get producer-of sym)]
            (when (not= direct-prod-id (:id node))
              ;; Follow alias chain: if direct producer is a ScalarBinding alias,
              ;; resolve through to the SOAC producer for edge classification.
              (let [resolved (get alias-map sym sym)
                    soac-prod-id (when (not= resolved sym)
                                   (get producer-of resolved))
                    ;; Use the SOAC producer if it exists and is different
                    [effective-prod-id effective-sym]
                    (if (and soac-prod-id
                             (soac/soac? (get nodes-map soac-prod-id)))
                      [soac-prod-id resolved]
                      [direct-prod-id sym])
                    prod-node (get nodes-map effective-prod-id)
                    edge-type (classify-edge prod-node node effective-sym alias-map)]
                ;; Create edge to effective producer
                (swap! edges conj [effective-prod-id (:id node) edge-type])
                (swap! adjacency update effective-prod-id (fnil conj #{}) (:id node))
                (swap! adjacency update (:id node) (fnil conj #{}) effective-prod-id)
                ;; Also create edge to the direct producer (alias ScalarBinding)
                ;; if different, to maintain ordering dependencies
                (when (and soac-prod-id (not= effective-prod-id direct-prod-id))
                  (swap! edges conj [direct-prod-id (:id node) :inf-dep])
                  (swap! adjacency update direct-prod-id (fnil conj #{}) (:id node))
                  (swap! adjacency update (:id node) (fnil conj #{}) direct-prod-id))))))))
    (->FusionGraph nodes-map @edges producer-of @adjacency alias-map)))

;; ================================================================
;; Reachability
;; ================================================================

(defn- graph-forward-adjacency
  [graph]
  (schedule-support/forward-adjacency (:edges graph)))

(defn reachable?
  "Check if to-id is reachable from from-id via forward edges."
  [graph from-id to-id]
  (schedule-support/reachable? (graph-forward-adjacency graph) from-id to-id))

(defn reachable-excluding?
  "Check if to-id is reachable from from-id excluding a set of node ids."
  [graph from-id to-id excluded]
  (schedule-support/reachable-excluding?
   (graph-forward-adjacency graph) from-id to-id excluded))

;; ================================================================
;; Vertical fusion
;; ================================================================

(defn- dep-edges-between
  "Get :dep edges from producer to consumer."
  [graph prod-id cons-id]
  (filter (fn [[from to typ]]
            (and (= from prod-id) (= to cons-id) (= typ :dep)))
          (:edges graph)))

(defn- inf-dep-edges-between
  "Get :inf-dep edges from producer to consumer."
  [graph prod-id cons-id]
  (filter (fn [[from to typ]]
            (and (= from prod-id) (= to cons-id) (= typ :inf-dep)))
          (:edges graph)))

(defn- other-consumers
  "Find node ids that consume producer's output, excluding consumer-id."
  [graph prod-id consumer-id]
  (set (keep (fn [[from to typ]]
               (when (and (= from prod-id) (= typ :dep) (not= to consumer-id))
                 to))
             (:edges graph))))

(defn- normalize-bound-step
  "One round of alias resolution + size-equiv + scalar-bound substitution."
  [bound-expr alias-map size-equiv scalar-bound]
  (let [;; First resolve aliases in all symbols
        alias-resolved (if (nil? alias-map)
                         bound-expr
                         (clojure.walk/postwalk
                          (fn [form]
                            (if (and (symbol? form) (contains? alias-map form))
                              (get alias-map form)
                              form))
                          bound-expr))]
    ;; Then resolve size equivalences: (alength X) → (alength Y) or → scalar
    (if (and (seq? alias-resolved)
             (let [head (first alias-resolved)]
               (or (= head '.invk) (= head 'clojure.core/alength))))
      (let [arr-sym (if (= '.invk (first alias-resolved))
                      (nth alias-resolved 2)  ;; (.invk alength-impl arr)
                      (second alias-resolved))]  ;; (clojure.core/alength arr)
        (cond
          ;; Array size-equiv: (alength buf) → (alength equiv-array)
          (and size-equiv (get size-equiv arr-sym))
          (let [equiv (get size-equiv arr-sym)]
            (if (= '.invk (first alias-resolved))
              (let [parts (vec alias-resolved)]
                (apply list (assoc parts 2 equiv)))
              (list (first alias-resolved) equiv)))

          ;; Scalar bound: (alength buf) → scalar when buf = (double-array scalar)
          (and scalar-bound (get scalar-bound arr-sym))
          (get scalar-bound arr-sym)

          :else alias-resolved))
      alias-resolved)))

(defn- normalize-bound
  "Normalize a bound expression by iterating alias + size-equiv resolution
  to a fixed point. Buffer chains like p → buf_X → (alength out_Y) → buf_Y → ...
  require multiple rounds to fully resolve."
  [bound-expr alias-map size-equiv scalar-bound]
  (loop [expr bound-expr
         n 0]
    (let [next (normalize-bound-step expr alias-map size-equiv scalar-bound)]
      (if (or (= next expr) (> n 10))
        next
        (recur next (inc n))))))

(defn- unwrap-long-cast
  "Strip (long X) wrapper, return X. Returns nil if not a long cast."
  [expr]
  (when (and (seq? expr) (= 'long (first expr)) (= 2 (count expr)))
    (second expr)))

(defn- build-size-equiv
  "Build a size equivalence map from allocation expressions.
  If `buf_X = (double-array (alength Y))`, then buf_X has size equiv Y.
  Also builds a scalar-bound map: if `buf_X = (double-array N)` or
  `(double-array (long N))`, then scalar-bound[buf_X] = N.
  This handles par/map expansion where the bound is a scalar variable.
  For pure par/pmap SOACs, the result symbol has size = bound.
  Resolves through alias chains.
  Returns {:array-equiv {buf → arr}, :scalar-bound {buf → scalar}}."
  [nodes-map alias-map]
  (reduce-kv
   (fn [m id node]
     (cond
       ;; Pure SoacMap: result has known length = bound
       ;; (alength pmap-sym) == bound, so register scalar-bound
       (and (instance? raster.compiler.ir.soac.SoacMap node)
            (:pure? node)
            (symbol? (:bound node)))
       (update m :scalar-bound assoc (:sym node) (:bound node))

       ;; Allocation expressions in ScalarBindings
       (and (instance? raster.compiler.ir.soac.ScalarBinding node)
            (seq? (:expr node)))
       (let [expr (:expr node)
             head (first expr)]
         (if (and (#{'clojure.core/double-array 'clojure.core/float-array
                     'clojure.core/int-array 'clojure.core/long-array} head)
                  (= 2 (count expr)))
           (let [size-arg (second expr)]
             (cond
               ;; (double-array (alength Y)) or (double-array (.invk alength-impl Y))
               (and (seq? size-arg)
                    (or (and (= '.invk (first size-arg)) (= 3 (count size-arg)))
                        (and (= 'clojure.core/alength (first size-arg)) (= 2 (count size-arg)))))
               (let [arr-sym (if (= '.invk (first size-arg))
                               (nth size-arg 2)
                               (second size-arg))
                     resolved (get alias-map arr-sym arr-sym)]
                 (update m :array-equiv assoc (:sym node) resolved))

               ;; (double-array N) where N is a symbol
               (symbol? size-arg)
               (update m :scalar-bound assoc (:sym node) size-arg)

               ;; (double-array (long N)) where N is a symbol
               (and (seq? size-arg) (unwrap-long-cast size-arg)
                    (symbol? (unwrap-long-cast size-arg)))
               (update m :scalar-bound assoc (:sym node)
                       (unwrap-long-cast size-arg))

               :else m))
           m))

       :else m))
   {:array-equiv {} :scalar-bound {}} nodes-map))

(defn can-fuse-vertically?
  "Check if producer can be vertically fused into consumer.

  Conditions:
    1. Both are SOACs
    2. There is a :dep edge from producer to consumer
    3. No :inf-dep edges between pair
    4. Same iteration bound (after alias normalization)
    5. Producer is a SoacMap (can inline map body)
    6. Sole-consumer path: no other consumer reaches this consumer
       through non-producer paths (ordering safety for producer removal).
       Multi-consumer path: producer stays alive, so reachability from
       other consumers to this consumer through independent paths is safe.

  Multi-consumer: When the producer has other consumers, fusion is still
  legal — the producer stays alive, and its body is inlined into this
  consumer (avoiding one array load). This follows Futhark's approach."
  [graph producer-id consumer-id]
  (let [nodes (:nodes graph)
        producer (get nodes producer-id)
        consumer (get nodes consumer-id)
        alias-map (:alias-map graph)]
    (and
      ;; 1. Both are SOACs
     (soac/soac? producer)
     (soac/soac? consumer)
      ;; 2. :dep edge exists
     (seq (dep-edges-between graph producer-id consumer-id))
      ;; 3. No :inf-dep edges between pair
     (empty? (inf-dep-edges-between graph producer-id consumer-id))
      ;; 4. Same bound (after alias + size normalization)
     (let [{:keys [array-equiv scalar-bound]} (build-size-equiv nodes alias-map)]
       (= (normalize-bound (soac/soac-bound producer) alias-map array-equiv scalar-bound)
          (normalize-bound (soac/soac-bound consumer) alias-map array-equiv scalar-bound)))
      ;; 5. Producer is a Map or Scan (can inline its body)
      ;; Map→X: substitute aget with producer body (eliminates intermediate array)
      ;; Scan→Map: fold consumer body into scan loop as side-effect aset
     (or (instance? raster.compiler.ir.soac.SoacMap producer)
         (and (instance? raster.compiler.ir.soac.SoacScan producer)
              (instance? raster.compiler.ir.soac.SoacMap consumer)))
      ;; 6. Ordering safety
     ;; Multi-consumer: producer stays alive (not removed), always safe.
     ;; Sole consumer: producer will be removed. Safe because sole consumer
     ;; means no other node depends on producer's output, so removing it
     ;; cannot break any ordering.
     true)))

(defn- fuse-vertical-scan-map
  "Fuse Scan producer into Map consumer by folding the map's body into
  the scan's loop as a side-effect aset.

  Before:  scan_out[i] = f(acc, in[i])     — pass 1
           map_out[i]  = g(scan_out[i])     — pass 2

  After:   (do (aset map_out i (cast (g scan_val)))  — side effect
               scan_val)                             — acc update

  The scan stays, the map is absorbed. Returns updated FusionGraph."
  [graph producer-id consumer-id]
  (let [nodes (:nodes graph)
        producer (get nodes producer-id)  ;; SoacScan
        consumer (get nodes consumer-id)  ;; SoacMap
        scan-out (:out producer)
        map-out  (first (soac/soac-outputs consumer))
        ;; Substitute (aget scan-out map-idx) → scan body in consumer lambda
        ;; This gives us the consumer body with scan's value inlined
        consumer-body (fusion-support/substitute-aget
                       (:lambda consumer) scan-out
                       (:idx producer) (:idx consumer)
                       (:lambda producer))
        ;; Build the aset side effect for consumer's output
        map-aset (fusion-support/emit-side-effect-aset
                  map-out (:idx producer) (:idx producer)
                  (:cast-fn consumer) consumer-body)
        ;; Fused scan body: (do (aset map-out i (cast body)) scan-body)
        ;; scan body must be last (returns accumulator value)
        fused-body (list 'do map-aset (:lambda producer))
        ;; Merge outputs: scan outputs + map output
        new-outputs (set/union (:outputs producer) #{map-out})
        ;; Merge inputs: scan inputs + consumer inputs - scan-out
        new-inputs (set/union (:inputs producer)
                              (disj (:inputs consumer) scan-out))
        new-scalars (set/union (:scalars producer)
                               (disj (:scalars consumer) scan-out))
        ;; The fused node is the scan with the map folded in
        updated-producer (assoc producer
                                :lambda fused-body
                                :outputs new-outputs
                                :inputs new-inputs
                                :scalars new-scalars)
        ;; Replace producer with fused version, remove consumer.
        ;; Add a ScalarBinding alias: consumer-sym = map-out
        ;; (so downstream references to consumer-sym resolve correctly)
        alias-node (soac/->ScalarBinding consumer-id (:sym consumer) map-out)
        new-nodes (-> nodes
                      (assoc producer-id updated-producer)
                      (dissoc consumer-id)
                      (assoc consumer-id alias-node))
        ;; Update edges: redirect consumer's downstream edges to producer
        old-edges (:edges graph)
        new-edges (set (keep (fn [[from to typ :as edge]]
                               (cond
                                 ;; Remove edges between producer and consumer
                                 (and (= from producer-id) (= to consumer-id)) nil
                                 (and (= from consumer-id) (= to producer-id)) nil
                                 ;; Redirect consumer's outgoing edges to producer
                                 (= from consumer-id) [producer-id to typ]
                                 ;; Redirect incoming edges to consumer → to producer
                                 ;; (skip if already pointing to producer)
                                 (= to consumer-id) (when (not= from producer-id)
                                                      [from producer-id typ])
                                 :else edge))
                             old-edges))
        clean-edges (set (remove (fn [[from to _]] (= from to)) new-edges))
        ;; Update producer-of: consumer-sym now produced by alias, map-out by producer
        new-producer-of (-> (:producer-of graph)
                            (assoc map-out producer-id)
                            (assoc (:sym consumer) consumer-id))
        new-adjacency (reduce (fn [adj [from to _]]
                                (-> adj
                                    (update from (fnil conj #{}) to)
                                    (update to (fnil conj #{}) from)))
                              (into {} (map (fn [id] [id #{}]) (keys new-nodes)))
                              clean-edges)]
    (->FusionGraph new-nodes clean-edges new-producer-of new-adjacency (build-alias-map new-nodes))))

(defn fuse-vertical
  "Fuse producer into consumer by inlining producer's lambda.

  Map→X:    Substitutes (aget producer-output consumer-idx) in consumer's
            lambda with producer's body. Merges input arrays.
  Scan→Map: Folds map body into scan loop as side-effect aset.

  Multi-consumer: When the producer has other consumers, it stays alive
  in the graph (only the edge to this consumer is removed). When the
  producer has no other consumers, it is removed entirely.

  Returns updated FusionGraph."
  [graph producer-id consumer-id]
  (if (instance? raster.compiler.ir.soac.SoacScan
                 (get (:nodes graph) producer-id))
    (fuse-vertical-scan-map graph producer-id consumer-id)
    ;; Map→X fusion (original path)
    (let [nodes (:nodes graph)
          producer (get nodes producer-id)
          consumer (get nodes consumer-id)
          prod-out (first (soac/soac-outputs producer))
          prod-sym (:sym producer)
          others (other-consumers graph producer-id consumer-id)
          keep-producer? (seq others)
        ;; Inline producer body into consumer lambda.
        ;; Try both the output buffer sym and the producer's binding sym
        ;; (alias), since consumers may read through either name.
        ;; E.g., producer: h = (par/map! out__ ...), consumer reads (aget h ...)
          new-lambda (let [subst1 (fusion-support/substitute-aget
                                   (:lambda consumer) prod-out
                                   (:idx producer) (:idx consumer)
                                   (:lambda producer))]
                       (if (= subst1 (:lambda consumer))
                       ;; First substitution was a no-op — try the alias sym
                         (fusion-support/substitute-aget
                          (:lambda consumer) prod-sym
                          (:idx producer) (:idx consumer)
                          (:lambda producer))
                         subst1))
        ;; Merge inputs: consumer inputs + producer inputs - producer output
          new-inputs (set/union (disj (:inputs consumer) prod-out)
                                (:inputs producer))
          new-scalars (set/union (disj (:scalars consumer) prod-out)
                                 (:scalars producer))
        ;; When fusing a pure producer (no output buffer), the consumer's bound
        ;; may reference (alength prod-sym) which becomes dangling after removal.
        ;; Replace with the producer's bound.
          new-bound (if (and (:pure? producer) (not keep-producer?))
                      (let [{:keys [array-equiv scalar-bound]}
                            (build-size-equiv nodes (:alias-map graph))]
                        (normalize-bound (:bound consumer) (:alias-map graph)
                                         array-equiv scalar-bound))
                      (:bound consumer))
        ;; Create updated consumer
          updated-consumer (assoc consumer
                                  :lambda new-lambda
                                  :inputs new-inputs
                                  :scalars new-scalars
                                  :bound new-bound)
        ;; Update nodes
          new-nodes (if keep-producer?
                    ;; Multi-consumer: keep producer, update consumer
                      (assoc nodes consumer-id updated-consumer)
                    ;; Sole consumer: remove producer
                      (-> nodes
                          (dissoc producer-id)
                          (assoc consumer-id updated-consumer)))
        ;; Update edges
          old-edges (:edges graph)
          new-edges
          (if keep-producer?
          ;; Multi-consumer: remove only the edge from producer→consumer,
          ;; add edges from producer's inputs to consumer
            (let [without-direct (disj old-edges [producer-id consumer-id :dep])
                ;; Consumer now depends on producer's inputs
                  producer-inputs-ids
                  (set (keep (fn [[from to _]] (when (= to producer-id) from))
                             old-edges))]
              (set/union without-direct
                         (set (for [pid producer-inputs-ids
                                    :when (not= pid consumer-id)]
                                [pid consumer-id :dep]))))
          ;; Sole consumer: redirect all producer edges
            (let [redirected
                  (set (keep (fn [[from to typ :as edge]]
                               (cond
                                 (= from producer-id)
                                 (when (not= to consumer-id)
                                   [consumer-id to typ])
                                 (= to producer-id)
                                 [from consumer-id typ]
                                 :else edge))
                             old-edges))]
              (set (remove (fn [[from to _]] (= from to)) redirected))))
        ;; Remove self-loops
          clean-edges (set (remove (fn [[from to _]] (= from to)) new-edges))
        ;; Update producer-of
          new-producer-of (if keep-producer?
                            (:producer-of graph)  ;; producer still exists
                            (-> (:producer-of graph)
                                (dissoc prod-out)
                                (dissoc (:sym producer))))
        ;; Rebuild adjacency
          new-adjacency (reduce (fn [adj [from to _]]
                                  (-> adj
                                      (update from (fnil conj #{}) to)
                                      (update to (fnil conj #{}) from)))
                                (into {} (map (fn [id] [id #{}]) (keys new-nodes)))
                                clean-edges)]
      (->FusionGraph new-nodes clean-edges new-producer-of new-adjacency (build-alias-map new-nodes)))))

;; ================================================================
;; Horizontal fusion
;; ================================================================

(defn- secondary-deps-before-primary?
  "Check that all *external* dependencies of a secondary node are satisfiable
  before the primary's position.  For each free symbol of the secondary that is
  NOT self-produced, its producer must have an ID ≤ primary-id.

  Self-produced symbols (those in `node-produced-syms`) are excluded because
  they are defined by the node itself — when the fused node moves to the
  primary's position, the post-fusion topological sort correctly schedules any
  allocation ScalarBindings that provide the output buffer."
  [graph secondary-id primary-id]
  (let [nodes (:nodes graph)
        sec (get nodes secondary-id)
        free (soac/node-all-free-syms sec)
        produced (soac/node-produced-syms sec)]
    (every? (fn [sym]
              (or (contains? produced sym) ;; self-produced — not an external dep
                  (let [prod-id (get (:producer-of graph) sym)]
                    ;; If no producer, it's an external param — always available.
                    ;; If producer is at or before primary, it's safe.
                    (or (nil? prod-id) (<= prod-id primary-id)))))
            free)))

(defn find-horizontal-groups
  "Find groups of SoacMap nodes with the same bound that are independent
  (no reachability between them). Returns vector of [id ...] groups."
  [graph]
  (let [nodes (:nodes graph)
        fwd (graph-forward-adjacency graph)
        ;; Collect all SoacMap nodes grouped by bound
        map-ids (reduce-kv (fn [acc id node]
                             (if (instance? raster.compiler.ir.soac.SoacMap node)
                               (conj acc id)
                               acc))
                           []
                           nodes)
        maps-by-bound (schedule-support/group-by-bound
                       map-ids
                       #(soac/soac-bound (get nodes %)))
        ;; For each group, filter to independent subsets
        groups (keep (fn [[_bound ids]]
                       (when (> (count ids) 1)
                         (let [independent
                               (schedule-support/maximal-independent-subset
                                ids
                                (fn [left right]
                                  (and (not (schedule-support/reachable? fwd left right))
                                       (not (schedule-support/reachable? fwd right left)))))]
                           (when (> (count independent) 1)
                             ;; Further filter: for each group, check that all
                             ;; secondaries' dependencies are before the primary.
                             ;; This prevents the fused map from being pushed past
                             ;; consumers of the primary's output by the topological sort.
                             (let [primary-id (apply min independent)
                                   safe-secondaries
                                   (filterv (fn [id]
                                              (or (= id primary-id)
                                                  (:pure? (get nodes id))
                                                  (secondary-deps-before-primary? graph id primary-id)))
                                            independent)]
                               (when (> (count safe-secondaries) 1)
                                 safe-secondaries))))))
                     maps-by-bound)]
    (vec groups)))

(defn- materialize-secondary-output
  "For a pure secondary SoacMap being horizontally fused, create a fresh
  output buffer symbol. Returns [out-sym alloc-expr] or nil if already imperative."
  [sec-node]
  (when (:pure? sec-node)
    (let [elem-type (or (:elem-type sec-node)
                        (case (:cast-fn sec-node)
                          (float clojure.core/float) :float
                          (long clojure.core/long) :long
                          (int clojure.core/int) :int
                          (double clojure.core/double) :double
                          nil))
          _ (when-not elem-type
              (throw (ex-info (str "soac-graph: cannot determine element type for pure SOAC node `"
                                   (:sym sec-node) "`. Ensure walker stamps :elem-type.")
                              {:node sec-node})))
          alloc-fn (case elem-type
                     :float  'clojure.core/float-array
                     :long   'clojure.core/long-array
                     :int    'clojure.core/int-array
                     :double 'clojure.core/double-array)
          array-tag (case elem-type
                      :float  'floats
                      :long   'longs
                      :int    'ints
                      :double 'doubles)
          out-sym (with-meta (gensym "hfuse_out__")
                    {:tag array-tag
                     :raster.type/tag array-tag
                     :raster.buffer/hoistable true})]
      [out-sym (list alloc-fn (:bound sec-node))])))

(defn fuse-horizontal
  "Fuse a group of independent SoacMap nodes into a single multi-output map.

  The primary map (lowest id) keeps its body as the return value.
  Other maps' bodies become side-effect aset writes in a do block.
  Pure secondary pmaps get their output buffers materialized inline.

  Returns updated FusionGraph."
  [graph group-ids]
  (let [sorted-ids (sort group-ids)
        primary-id (first sorted-ids)
        secondary-ids (rest sorted-ids)
        nodes (:nodes graph)
        primary (get nodes primary-id)
        ;; Materialize output buffers for pure maps (both primary and secondaries).
        ;; Pure maps have no output buffer — create one for the fused imperative map.
        primary-mat (materialize-secondary-output primary)
        primary-out-sym (if primary-mat
                          (first primary-mat)
                          (first (:outputs primary)))
        ;; For each secondary, determine its output buffer.
        sec-out-info
        (mapv (fn [sec-id]
                (let [sec (get nodes sec-id)
                      materialized (materialize-secondary-output sec)]
                  (if materialized
                    {:id sec-id :node sec
                     :out-sym (first materialized) :alloc-expr (second materialized)
                     :materialized? true}
                    {:id sec-id :node sec
                     :out-sym (first (:outputs sec))
                     :materialized? false})))
              secondary-ids)
        ;; Build side-effect writes for secondaries
        idx (:idx primary)
        side-effects
        (mapv (fn [{:keys [node out-sym]}]
                (fusion-support/emit-side-effect-aset
                 out-sym idx (:idx node) (:cast-fn node) (:lambda node)))
              sec-out-info)
        ;; Fused body: do block with side effects, primary body last
        fused-body (list* 'do (concat side-effects [(:lambda primary)]))
        ;; Merge inputs and scalars
        all-inputs (apply set/union (map #(:inputs (get nodes %)) sorted-ids))
        all-outputs (into #{primary-out-sym}
                          (map :out-sym sec-out-info))
        all-scalars (apply set/union (map #(:scalars (get nodes %)) sorted-ids))
        ;; Updated primary — no longer pure since it has side-effect asets
        updated-primary (assoc primary
                               :lambda fused-body
                               :inputs all-inputs
                               :outputs all-outputs
                               :scalars all-scalars
                               :pure? false
                               :primary-out primary-out-sym)
        ;; Allocation bindings for materialized pure nodes (primary + secondaries).
        ;; Use negative IDs to ensure allocations sort before the primary map.
        alloc-bindings
        (cond-> (vec (keep (fn [{:keys [id out-sym alloc-expr materialized?]}]
                             (when materialized?
                               (soac/->ScalarBinding (- -1000 id) out-sym alloc-expr)))
                           sec-out-info))
          primary-mat
          (conj (soac/->ScalarBinding (- -1000 primary-id) (first primary-mat) (second primary-mat))))
        ;; Alias bindings: secondary-sym = output-buf
        secondary-aliases
        (mapv (fn [{:keys [id node out-sym]}]
                (soac/->ScalarBinding id (:sym node) out-sym))
              sec-out-info)
        new-nodes (reduce dissoc nodes secondary-ids)
        new-nodes (assoc new-nodes primary-id updated-primary)
        new-nodes (reduce (fn [m node] (assoc m (:id node) node))
                          new-nodes (concat alloc-bindings secondary-aliases))
        ;; Update edges: redirect secondary edges to primary
        sec-set (set secondary-ids)
        new-edges (set (keep (fn [[from to typ :as edge]]
                               (cond
                                 (and (contains? sec-set from) (contains? sec-set to)) nil
                                 (contains? sec-set from) [primary-id to typ]
                                 (contains? sec-set to) [from primary-id typ]
                                 :else edge))
                             (:edges graph)))
        ;; Remove self-loops
        clean-edges (set (remove (fn [[from to _]] (= from to)) new-edges))
        ;; Update producer-of
        new-producer-of (reduce (fn [pof sec-id]
                                  (let [sec (get nodes sec-id)]
                                    (reduce (fn [m sym] (assoc m sym primary-id))
                                            pof (soac/node-produced-syms sec))))
                                (:producer-of graph) secondary-ids)
        ;; Rebuild adjacency
        new-adjacency (reduce (fn [adj [from to _]]
                                (-> adj
                                    (update from (fnil conj #{}) to)
                                    (update to (fnil conj #{}) from)))
                              (into {} (map (fn [id] [id #{}]) (keys new-nodes)))
                              clean-edges)]
    (->FusionGraph new-nodes clean-edges new-producer-of new-adjacency (build-alias-map new-nodes))))

;; ================================================================
;; Fixpoint fusion
;; ================================================================

(defn- apply-vertical-fusions
  "Apply all legal vertical fusions in one pass. Returns [graph count]."
  [graph]
  (loop [g graph fused 0]
    ;; Find first legal vertical fusion
    (let [candidate
          (first
           (for [[from to typ] (:edges g)
                 :when (= typ :dep)
                 :when (can-fuse-vertically? g from to)]
             [from to]))]
      (if candidate
        (recur (fuse-vertical g (first candidate) (second candidate))
               (inc fused))
        [g fused]))))

(defn- apply-horizontal-fusions
  "Apply all horizontal fusions in one pass. Returns [graph count]."
  [graph]
  (let [groups (find-horizontal-groups graph)]
    (if (empty? groups)
      [graph 0]
      (let [fused-count (reduce + (map #(dec (count %)) groups))]
        [(reduce fuse-horizontal graph groups) fused-count]))))

(defn fusion-fixpoint
  "Alternate vertical and horizontal fusion passes until no more fusions.
  Returns [FusionGraph stats-map].
  stats-map: {:vertical N :horizontal N :iterations N}"
  [graph]
  (loop [g graph
         total-v 0
         total-h 0
         iters 0]
    (let [[g1 v-count] (apply-vertical-fusions g)
          [g2 h-count] (apply-horizontal-fusions g1)]
      (if (and (zero? v-count) (zero? h-count))
        [g2 {:vertical (+ total-v v-count)
             :horizontal (+ total-h h-count)
             :iterations (inc iters)}]
        (recur g2
               (+ total-v v-count)
               (+ total-h h-count)
               (inc iters))))))
