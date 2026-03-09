(ns hooks.raster-core
  "clj-kondo hooks for raster.core macros: deftm, ftm, defvalue, defval,
   defabstract, defcache, ns-raster."
  (:require [clj-kondo.hooks-api :as api]))

;; ================================================================
;; Shared helpers
;; ================================================================

(defn- strip-annotations
  "Remove :- Type pairs from a parameter/field vector.
  Input:  [a :- Double, b :- (Array double)]
  Output: [a b]"
  [param-nodes]
  (loop [nodes (seq param-nodes)
         acc   []]
    (if-not nodes
      acc
      (let [node (first nodes)]
        (if (and (api/keyword-node? node)
                 (= :- (api/sexpr node)))
          ;; Skip :- and the type that follows
          (recur (nnext nodes) acc)
          ;; Keep the param (skip commas too)
          (recur (next nodes) (conj acc node)))))))

(defn- qualified-symbol-node?
  "Check if a node represents a qualified symbol like raster.numeric/+."
  [node]
  (when (api/token-node? node)
    (let [v (api/sexpr node)]
      (and (symbol? v) (namespace v)))))

(defn- strip-implements
  "Remove :implements ClassName from macro arg lists."
  [nodes]
  (loop [ns (seq nodes)
         acc []]
    (if-not ns
      acc
      (let [node (first ns)]
        (if (and (api/keyword-node? node)
                 (= :implements (api/sexpr node)))
          (recur (nnext ns) acc)
          (recur (next ns) (conj acc node)))))))

(defn- skip-all-t
  "Skip (All [T] ...) wrapper, returning the inner nodes.
  If the first node is a list starting with All, return its contents after [T].
  Otherwise return the nodes unchanged."
  [nodes]
  (let [first-node (first nodes)]
    (if (and first-node
             (api/list-node? first-node)
             (let [children (:children first-node)]
               (and (seq children)
                    (api/token-node? (first children))
                    (= 'All (api/sexpr (first children))))))
      ;; (All [T] params-vec body...) → skip All and [T], return rest
      (let [inner (drop 2 (:children first-node))]
        (concat inner (rest nodes)))
      nodes)))

;; ================================================================
;; deftm → rewrite to defn
;; ================================================================

(defn deftm
  "Hook: (deftm name [params :- types ...] :- RetType body...)
   Also handles: (deftm name (All [T] [params :- types ...] :- RetType body...))
   Rewrites to (defn name [params...] body...) for analysis."
  [{:keys [node]}]
  (let [children (rest (:children node))
        name-node (first children)
        rest-nodes (rest children)
        ;; Skip optional docstring
        [_docstring rest-nodes]
        (if (api/string-node? (first rest-nodes))
          [(first rest-nodes) (rest rest-nodes)]
          [nil rest-nodes])
        ;; Handle (All [T] ...) wrapper
        rest-nodes (skip-all-t rest-nodes)
        param-vec-node (first rest-nodes)
        after-params (rest rest-nodes)
        ;; Strip :- RetType from after params
        body (if (and (seq after-params)
                      (api/keyword-node? (first after-params))
                      (= :- (api/sexpr (first after-params))))
               (nnext after-params)
               after-params)]
    (when (and param-vec-node (api/vector-node? param-vec-node))
      (let [clean-params (strip-annotations (:children param-vec-node))
            qualified? (qualified-symbol-node? name-node)]
        (if qualified?
          {:node (api/list-node
                  (list*
                   (api/token-node 'fn)
                   (api/vector-node clean-params)
                   body))}
          {:node (api/list-node
                  (list*
                   (api/token-node 'defn)
                   name-node
                   (api/vector-node clean-params)
                   body))})))))

;; ================================================================
;; ftm → rewrite to fn
;; ================================================================

(defn ftm
  "Hook: (ftm [params :- types ...] :- RetType body...)
   Rewrites to (fn [params...] body...) for analysis."
  [{:keys [node]}]
  (let [children (rest (:children node))
        param-vec-node (first children)
        after-params (rest children)
        body (if (and (seq after-params)
                      (api/keyword-node? (first after-params))
                      (= :- (api/sexpr (first after-params))))
               (nnext after-params)
               after-params)]
    (when (and param-vec-node (api/vector-node? param-vec-node))
      (let [clean-params (strip-annotations (:children param-vec-node))]
        {:node (api/list-node
                (list*
                 (api/token-node 'fn)
                 (api/vector-node clean-params)
                 body))}))))

;; ================================================================
;; defvalue → rewrite to defrecord
;; ================================================================

(defn defvalue
  "Hook: (defvalue Name [x :- Double, y :- Double] :implements Foo)
   Also handles: (defvalue Name (All [T]) [x :- (Array T)] :implements Foo)
   Rewrites to (defrecord Name [x y]) for analysis."
  [{:keys [node]}]
  (let [children (rest (:children node))
        name-node (first children)
        rest-nodes (rest children)
        ;; Skip (All [T]) if present
        rest-nodes (skip-all-t rest-nodes)
        ;; Find the fields vector
        fields-node (first (filter #(api/vector-node? %) rest-nodes))
        fields-node (or fields-node (api/vector-node []))
        ;; Strip :- annotations from field vector
        clean-fields (strip-annotations (:children fields-node))]
    {:node (api/list-node
            (list
             (api/token-node 'defrecord)
             name-node
             (api/vector-node clean-fields)))}))

;; ================================================================
;; defval → singleton value type
;; ================================================================

(defn defval
  "Hook: (defval Name :implements Foo)
   Rewrites to (defrecord Name []) for analysis."
  [{:keys [node]}]
  (let [children (rest (:children node))
        name-node (first children)]
    {:node (api/list-node
            (list
             (api/token-node 'defrecord)
             name-node
             (api/vector-node [])))}))

;; ================================================================
;; defabstract → def
;; ================================================================

(defn defabstract
  "Hook: (defabstract Name :extends Parent)
   Rewrites to (def Name nil) to define the symbol."
  [{:keys [node]}]
  (let [children (rest (:children node))
        name-node (first children)]
    {:node (api/list-node
            (list
             (api/token-node 'def)
             name-node
             (api/token-node nil)))}))

;; ================================================================
;; defcache → defrecord
;; ================================================================

(defn defcache
  "Hook: (defcache Name [k :- (Array double)] :implements Foo)
   Also handles: (defcache Name (All [T]) [k :- (Array T)] :implements Foo)
   Rewrites to (defrecord Name [k]) for analysis."
  [{:keys [node]}]
  (let [children (rest (:children node))
        name-node (first children)
        rest-nodes (rest children)
        ;; Skip (All [T]) if present
        rest-nodes (skip-all-t rest-nodes)
        fields-node (first (filter #(api/vector-node? %) rest-nodes))
        fields-node (or fields-node (api/vector-node []))
        clean-fields (strip-annotations (:children fields-node))]
    {:node (api/list-node
            (list
             (api/token-node 'defrecord)
             name-node
             (api/vector-node clean-fields)))}))

;; ================================================================
;; Parallel primitives
;; ================================================================

(defn par-map!
  "Hook: (par/map! out idx bound cast body)
   Rewrites to (let [idx 0] body) to make idx visible."
  [{:keys [node]}]
  (let [children (rest (:children node))
        idx-node (second children)
        body-node (nth children 4)]
    {:node (api/list-node
            (list (api/token-node 'let)
                  (api/vector-node [idx-node (api/token-node 0)])
                  body-node))}))

(defn par-reduce
  "Hook: (par/reduce acc init idx bound body)"
  [{:keys [node]}]
  (let [children (rest (:children node))
        acc-node (first children)
        init-node (second children)
        idx-node (nth children 2)
        body-node (nth children 4)]
    {:node (api/list-node
            (list (api/token-node 'let)
                  (api/vector-node [acc-node init-node
                                    idx-node (api/token-node 0)])
                  body-node))}))

(defn par-scan
  "Hook: (par/scan out acc init idx bound cast body)"
  [{:keys [node]}]
  (let [children (rest (:children node))
        acc-node (second children)
        init-node (nth children 2)
        idx-node (nth children 3)
        body-node (nth children 6)]
    {:node (api/list-node
            (list (api/token-node 'let)
                  (api/vector-node [acc-node init-node
                                    idx-node (api/token-node 0)])
                  body-node))}))

(defn par-map-void!
  "Hook: (par/map-void! idx bound body)"
  [{:keys [node]}]
  (let [children (rest (:children node))
        idx-node (first children)
        body-node (nth children 2)]
    {:node (api/list-node
            (list (api/token-node 'let)
                  (api/vector-node [idx-node (api/token-node 0)])
                  body-node))}))

(defn par-stencil!
  "Hook: (par/stencil! out [in-arrays] radius boundary cast idx bound body)"
  [{:keys [node]}]
  (let [children (rest (:children node))
        idx-node (nth children 5)
        body-node (nth children 7)]
    {:node (api/list-node
            (list (api/token-node 'let)
                  (api/vector-node [idx-node (api/token-node 0)])
                  body-node))}))

;; ================================================================
;; Core combinators
;; ================================================================

(defn core-reduce!
  "Hook: (reduce! [acc init] [arrays...] body)
   Rewrites to (let [acc init] body)."
  [{:keys [node]}]
  (let [children (rest (:children node))
        acc-vec (first children)
        body-nodes (nnext children)]
    (if (api/vector-node? acc-vec)
      (let [elems (:children acc-vec)
            binding-pair (if (> (count elems) 2)
                           (vec (take-last 2 elems))
                           (vec elems))]
        {:node (api/list-node
                (list* (api/token-node 'let)
                       (api/vector-node binding-pair)
                       body-nodes))})
      {:node (api/list-node (list* (api/token-node 'do) (rest children)))})))

(defn core-stencil!
  "Hook: (stencil! [syms...] :radius R :boundary B body)"
  [{:keys [node]}]
  (let [children (rest (:children node))
        body-node (last children)]
    {:node (api/list-node
            (list (api/token-node 'let)
                  (api/vector-node [(api/token-node 'i) (api/token-node 0)])
                  body-node))}))

(defn core-muladd
  "Hook: (muladd expr) — pass through."
  [{:keys [node]}]
  (let [children (rest (:children node))]
    {:node (if (= 1 (count children))
             (first children)
             (api/list-node (list* (api/token-node 'do) children)))}))

;; ================================================================
;; ns-raster → ns
;; ================================================================

(defn ns-raster
  "Hook: (ns-raster my.ns (:require ...))
   Rewrites to (ns my.ns (:require ...))."
  [{:keys [node]}]
  (let [children (rest (:children node))]
    {:node (api/list-node
            (list*
             (api/token-node 'ns)
             children))}))

;; ================================================================
;; import-vars → def each symbol
;; ================================================================

(defn import-vars
  "Hook: (import-vars some.ns sym1 sym2 ...)
   Rewrites to (do (def sym1 nil) (def sym2 nil) ...) so kondo
   sees the symbols as defined in the current namespace."
  [{:keys [node]}]
  (let [children (rest (:children node)) ;; drop 'import-vars
        ;; First child is the ns symbol, rest are var names
        var-syms (rest children)]
    {:node (api/list-node
            (list*
             (api/token-node 'do)
             (map (fn [sym-node]
                    (api/list-node
                     (list (api/token-node 'def)
                           sym-node
                           (api/token-node nil))))
                  var-syms)))}))
