(ns raster.compiler.passes.parallel.schedule-support
  "Shared bound-grouping, dependency, and reachability helpers for parallel passes."
  (:require [clojure.set :as set]))

(defn same-bound?
  "Check whether two bound expressions are structurally equal."
  [left right]
  (= left right))

(defn group-by-bound
  "Group items by bound expression, preserving input order within each group."
  [items bound-of]
  (reduce (fn [groups item]
            (update groups (bound-of item) (fnil conj []) item))
          {}
          items))

(defn maximal-independent-subset
  "Greedily choose a maximal subset where `independent?` holds pairwise.
  `independent?` must be symmetric."
  [items independent?]
  (reduce (fn [group item]
            (if (every? #(independent? % item) group)
              (conj group item)
              group))
          []
          items))

(defn build-symbol-dependency-graph
  "Build {binding-idx -> #{dependency-idx}} from binding pairs and a free-symbol collector.
  Self references are ignored."
  [pairs free-syms]
  (let [sym-to-idx (into {} (map-indexed (fn [i [sym _]] [sym i]) pairs))]
    (into {}
          (map-indexed
           (fn [i [_ expr]]
             [i (into #{}
                      (keep (fn [sym]
                              (let [idx (get sym-to-idx sym)]
                                (when (and idx (not= idx i)) idx))))
                      (free-syms expr))])
           pairs))))

(defn topo-order
  "Topologically emit nodes in `emit-order`, pulling dependencies first. Cycles are ignored
  the same way as the previous pass-local implementation."
  [dep-graph emit-order]
  (let [emitted (volatile! #{})
        visiting (volatile! #{})
        order (volatile! [])]
    (letfn [(emit! [node-id]
              (when-not (or (contains? @emitted node-id)
                            (contains? @visiting node-id))
                (vswap! visiting conj node-id)
                (doseq [dep (get dep-graph node-id #{})]
                  (emit! dep))
                (when-not (contains? @emitted node-id)
                  (vswap! emitted conj node-id)
                  (vswap! order conj node-id))
                (vswap! visiting disj node-id)))]
      (doseq [node-id emit-order]
        (emit! node-id))
      @order)))

(defn forward-adjacency
  "Build forward-only adjacency from edge triples `[from to type]`."
  [edges]
  (reduce (fn [adj [from to _]]
            (update adj from (fnil conj #{}) to))
          {}
          edges))

(defn reachable?
  "Check if `to-id` is reachable from `from-id` in a forward adjacency map."
  [fwd-adj from-id to-id]
  (loop [frontier #{from-id}
         visited #{}]
    (if (empty? frontier)
      false
      (if (contains? frontier to-id)
        true
        (let [new-visited (set/union visited frontier)
              next-frontier (set/difference
                             (apply set/union #{}
                                    (map #(get fwd-adj % #{}) frontier))
                             new-visited)]
          (recur next-frontier new-visited))))))

(defn reachable-excluding?
  "Check reachability while treating `excluded` nodes as already visited."
  [fwd-adj from-id to-id excluded]
  (loop [frontier #{from-id}
         visited (set excluded)]
    (if (empty? frontier)
      false
      (if (contains? frontier to-id)
        true
        (let [new-visited (set/union visited frontier)
              next-frontier (set/difference
                             (apply set/union #{}
                                    (map #(get fwd-adj % #{}) frontier))
                             new-visited)]
          (recur next-frontier new-visited))))))