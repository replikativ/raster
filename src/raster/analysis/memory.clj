(ns raster.analysis.memory
  "Memory analysis algorithms: alias analysis, last-use analysis, escape analysis.

  Based on Futhark's approach to uniqueness types and consumption tracking.
  Provides S-expression-based analysis for buffer fusion and reuse decisions.

  Migrated from typed.cljc.checker.memory-analysis (TypedClojure) — only the
  S-expression subset needed by raster's fusion pass."
  (:require [clojure.set :as set]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Alias Analysis (from Futhark)

(defn empty-alias-state
  "Create an empty alias state."
  []
  {:aliases {}
   :consumed #{}})

(defn transitive-closure
  "Compute transitive closure of aliases.
  Given a set of names and an alias map, return the set plus all transitively aliased names."
  [names alias-map]
  (loop [current names
         result names]
    (if (empty? current)
      result
      (let [new-aliases (set/union
                         (into #{} (mapcat #(get alias-map % #{}) current)))
            added (set/difference new-aliases result)]
        (if (empty? added)
          result
          (recur added (set/union result added)))))))

(defn- reverse-alias-map
  "Create reverse alias mapping: if x aliases {y, z}, create {y #{x}, z #{x}}"
  [binding-name aliases]
  (into {}
        (map (fn [alias-name]
               [alias-name #{binding-name}]))
        aliases))

(defn- merge-alias-maps
  "Merge two alias maps, taking union of alias sets."
  [m1 m2]
  (merge-with set/union m1 m2))

(defn track-aliases
  "Track aliases through a let binding.

  Based on Futhark's trackAliases function:
  1. Compute transitive closure of binding's aliases
  2. Create reverse alias map (bidirectional aliasing)
  3. Merge with existing alias map
  4. Track consumed variables

  Args:
    state: Current alias state {:aliases AliasMap, :consumed ConsumedSet}
    binding-name: Name being bound
    binding-aliases: Direct aliases of the binding
    consumed-in-expr: Variables consumed in the binding expression

  Returns:
    Updated alias state"
  [{:keys [aliases consumed]} binding-name binding-aliases consumed-in-expr]
  (let [transitive-aliases (transitive-closure binding-aliases aliases)
        reverse-map (reverse-alias-map binding-name transitive-aliases)
        forward-map {binding-name transitive-aliases}
        new-aliases (-> aliases
                        (merge-alias-maps reverse-map)
                        (merge-alias-maps forward-map))
        consumed-with-aliases (transitive-closure consumed-in-expr aliases)
        new-consumed (set/union consumed consumed-with-aliases)]
    {:aliases new-aliases
     :consumed new-consumed}))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; S-expression free variable analysis

(defn sexp-free-vars
  "Extract free variables from an S-expression (not tools.analyzer AST).
  Handles: let*, loop*, fn*, if, do, quote, .method calls, new, throw,
  try/catch, recur. Returns set of free symbols."
  [expr]
  (letfn [(walk [e bound]
            (cond
              (symbol? e) (if (contains? bound e) #{} #{e})
              (not (sequential? e)) #{}
              :else
              (let [head (first e)]
                (cond
                  ;; let*/loop*: each init sees previous bindings; body sees all
                  (#{'let* 'loop*} head)
                  (let [pairs (partition 2 (second e))
                        [init-fvs new-bound]
                        (reduce (fn [[fvs b] [sym init]]
                                  [(set/union fvs (walk init b)) (conj b sym)])
                                [#{} bound] pairs)]
                    (set/union init-fvs
                               (reduce set/union #{}
                                       (map #(walk % new-bound) (nnext e)))))

                  ;; fn*: params scope over body, captures are free
                  (#{'fn 'fn*} head)
                  (let [arities (if (vector? (second e)) (list (rest e)) (rest e))]
                    (reduce set/union #{}
                            (map (fn [arity]
                                   (let [params (first arity)
                                         param-set (if (vector? params) (set params) #{})]
                                     (reduce set/union #{}
                                             (map #(walk % (set/union bound param-set))
                                                  (rest arity)))))
                                 arities)))

                  (= head 'if) (let [[_ t th el] e]
                                 (set/union (walk t bound) (walk th bound)
                                            (if el (walk el bound) #{})))
                  (= head 'do) (reduce set/union #{} (map #(walk % bound) (rest e)))
                  (= head 'quote) #{}
                  (#{'new 'throw 'recur} head) (reduce set/union #{} (map #(walk % bound) (rest e)))
                  ;; .invk: method call — head is syntax, walk obj + args
                  (= head '.invk) (reduce set/union #{} (map #(walk % bound) (rest e)))
                  ;; . (dot): instance method/field — walk target + args, skip method name
                  (= head '.) (let [[_ target _method-or-field & args] e]
                                (set/union (walk target bound)
                                           (reduce set/union #{} (map #(walk % bound) args))))
                  (= head 'try)
                  (let [clauses (rest e)]
                    (reduce set/union #{}
                            (map (fn [c]
                                   (if (and (sequential? c) (#{'catch 'finally} (first c)))
                                     (if (= 'catch (first c))
                                       (let [[_ _cls sym & body] c]
                                         (reduce set/union #{} (map #(walk % (conj bound sym)) body)))
                                       (reduce set/union #{} (map #(walk % bound) (rest c))))
                                     (walk c bound)))
                                 clauses)))
                  ;; General: walk all sub-forms (covers .method, .invk, function calls)
                  :else (reduce set/union #{} (map #(walk % bound) e))))))]
    (walk expr #{})))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; S-expression escape analysis

(defn escapes-via-closure-sexp?
  "Check if variable escapes by being captured in a fn*/fn closure
  within an S-expression."
  [var-name expr]
  (letfn [(has-capturing-fn? [e]
            (cond
              (not (sequential? e)) false
              (and (#{'fn 'fn*} (first e))
                   (contains? (sexp-free-vars e) var-name)) true
              :else (some has-capturing-fn? (rest e))))]
    (boolean (has-capturing-fn? expr))))

(defn escapes-via-return?
  "Check if variable escapes by being returned.
  A variable escapes if it appears anywhere in the return expression."
  [var-name return-expr]
  (boolean
   (contains? (sexp-free-vars return-expr) var-name)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; S-expression let analysis

(defn analyze-sexp-let
  "Analyze a (let* [sym init ...] body...) S-expression for memory reuse.
  Returns {:bindings [MemoryAnalysisResult ...], :used-after {idx #{syms}},
           :alias-state AliasState}."
  [let-form]
  (let [[_ bindings-vec & body-exprs] let-form
        pairs (vec (partition 2 bindings-vec))
        body  (if (= 1 (count body-exprs)) (first body-exprs) (cons 'do body-exprs))
        n     (count pairs)

        ;; Forward pass: build alias state
        binding-infos (mapv (fn [[sym init]]
                              {:name sym
                               :aliases (if (symbol? init) #{init} #{})
                               :consumed (if (symbol? init) #{} (sexp-free-vars init))})
                            pairs)

        alias-state (reduce (fn [st {:keys [name aliases consumed]}]
                              (track-aliases st name aliases consumed))
                            (empty-alias-state) binding-infos)

        ;; Backward pass: compute used-after[i] = syms used in bindings[i+1..] + body
        body-fvs (sexp-free-vars body)
        used-after (loop [i (dec n), used body-fvs, result {}]
                     (if (neg? i)
                       result
                       (let [[_sym init] (nth pairs i)
                             init-fvs (sexp-free-vars init)]
                         (recur (dec i)
                                (-> used (disj (first (nth pairs i))) (set/union init-fvs))
                                (assoc result i used)))))

        ;; Per-binding analysis
        {:keys [aliases consumed]} alias-state
        results (mapv (fn [i {:keys [name] :as info}]
                        (let [used-names (get used-after i #{})
                              trans-als (transitive-closure #{name} aliases)
                              unique? (= trans-als #{name})
                              last-use? (empty? (set/intersection trans-als used-names))
                              ;; Build scope: all init-exprs after i + body
                              scope (cons 'do (concat (map second (drop (inc i) pairs))
                                                      [body]))
                              escapes? (or (escapes-via-return? name body)
                                           (escapes-via-closure-sexp? name scope))
                              consumed? (contains? consumed name)
                              can-reuse? (and unique?
                                              (or last-use? consumed?)
                                              (not escapes?))]
                          {:binding-name name
                           :unique? unique?
                           :last-use? last-use?
                           :escapes? escapes?
                           :aliases (:aliases info)
                           :consumed? consumed?
                           :can-use-transient? can-reuse?
                           :can-reuse-buffer? can-reuse?}))
                      (range n) binding-infos)]
    {:bindings results
     :used-after used-after
     :alias-state alias-state}))
