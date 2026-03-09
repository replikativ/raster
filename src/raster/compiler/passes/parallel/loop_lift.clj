(ns raster.compiler.passes.parallel.loop-lift
  "Lift plain sequential loop forms into explicit raster.par operations.

  This pass is the raw-syntax recovery stage for parallel structure. It scans
  flattened let* bindings, recognizes loop patterns expressed with dotimes or
  loop/recur, and rewrites them into explicit raster.par nodes for downstream
  SIMD and GPU planning.

  Patterns detected:
    1. Map pattern: (dotimes [i n] (aset out i (cast body))) => raster.par/map!
    2. Reduce pattern: (loop [i 0 acc init] (if (< i n) (recur (inc i) body) acc))
                       => raster.par/reduce
    3. Scan pattern: loop/recur accumulation with per-step aset => raster.par/scan

  Safety invariants:
    - Map body must be element-wise (no loop-carried deps)
    - Map output array must not be read across indices (no RAW hazard)
    - All aget calls must use the loop variable as index
    - Reduce/scan loops must have the expected single index/accumulator shape"
  (:require [clojure.walk]
            [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.passes.parallel.patterns :as patterns]))

;; ================================================================
;; Symbol collection
;; ================================================================

;; ================================================================
;; Pattern: dotimes + aset => raster.par/map!
;; ================================================================

(defn- body-element-wise?
  "Check if body is element-wise w.r.t. the loop index.
  Element-wise means:
    - No writes to arrays (no aset except at top level)
    - All aget calls use idx-sym as the index
    - No references to the accumulator (if any)
  Returns true if body is safe for parallel map."
  [body idx-sym acc-sym]
  (letfn [(check [e]
            (cond
              ;; aget: index must match idx-sym
              (and (seq? e)
                   (descriptor/aget-op? (first e))
                   (>= (count e) 3))
              (and (patterns/idx-matches? (nth e 2) idx-sym)
                   (every? #(check %) (rest e)))

              ;; aset inside body = not element-wise (side effects)
              (and (seq? e)
                   (descriptor/aset-op? (first e)))
              false

              ;; Accumulator reference = loop-carried dependency (not for reduces)
              (and acc-sym (symbol? e) (= e acc-sym))
              false

              ;; let bindings: check both bindings and body, adding bound syms as locals
              (and (seq? e) (contains? #{'let 'let*} (first e)))
              (let [[_ binds & body-forms] e]
                (and (every? check (map second (partition 2 binds)))
                     (every? check body-forms)))

              ;; Recurse into sub-expressions
              (seq? e)
              (every? check (rest e))

              (vector? e)
              (every? check e)

              :else true))]
    (check body)))

(defn- no-raw-hazard?
  "Check for RAW (read-after-write) hazard on the output array.
  A hazard exists when the output is read at a different index than the write.
  Reading at the same element index (same idx-sym) is safe for parallel map —
  each thread reads and writes only its own element (e.g. SGD: A[i] += lr * dA[i])."
  [out-sym body idx-sym]
  (letfn [(unsafe-read? [e sym]
            (cond
              ;; aget of out-sym: safe only if index is exactly idx-sym
              (and (seq? e)
                   (descriptor/aget-op? (first e))
                   (>= (count e) 3)
                   (symbol? (second e))
                   (= (name sym) (name (second e))))
              (not (patterns/idx-matches? (nth e 2) idx-sym))

              (seq? e) (some #(unsafe-read? % sym) (rest e))
              (vector? e) (some #(unsafe-read? % sym) e)
              :else false))]
    (not (unsafe-read? body out-sym))))

(defn- dotimes-map-pattern?
  "Check if a form matches the dotimes->map! pattern.
  Returns {:out sym :idx sym :bound expr :cast fn-or-nil :body expr} or nil.

  Matches:
    (dotimes [i n] (aset out i (cast body)))
    (dotimes [i n] (let [...] (aset out i body)))
  And nested let wrappers:
    (let [n_ (int bound)] (dotimes [i n_] (aset out i body)))"
  [form]
  (when-let [{:keys [out-sym index-sym bound-expr cast-fn value-expr] :as _info}
             (patterns/match-dotimes-map-loop form)]
    (when (and (body-element-wise? value-expr index-sym nil)
               (no-raw-hazard? out-sym value-expr index-sym))
      {:out out-sym
       :idx index-sym
       :bound bound-expr
       :cast cast-fn
       :body value-expr})))

;; ================================================================
;; Pattern: nested dotimes => raster.par/map! (flattened 1D)
;; ================================================================

(defn- nested-dotimes-map-pattern?
  "Detect nested dotimes pattern for 2D array operations:
    (dotimes [i R] (dotimes [j C] (aset buf (+ (* i C) j) body)))
    (dotimes [i R] (let [...] (dotimes [j C] (let [...] (aset buf (+ (* i C) j) body)))))
  Only matches when the aset index is row-major linearization (i*C + j),
  ensuring the flattened 1D iteration produces identical results.
  Returns {:out sym :idx sym :bound expr :cast fn-or-nil :body expr} or nil."
  [form]
  (when-let [{:keys [out-sym index-sym bound-expr cast-fn value-expr] :as _info}
             (patterns/match-nested-dotimes-row-major-map form)]
    {:out out-sym
     :idx index-sym
     :bound bound-expr
     :cast cast-fn
     :body value-expr}))

(defn- loop-reduce-pattern?
  "Check if a form matches the loop->reduce pattern.
  Returns {:acc sym :init expr :idx sym :bound expr :body expr} or nil."
  [form]
  (when (seq? form)
    (cond
      (= 'loop (first form))
      (when-let [{:keys [acc-sym acc-init index-sym then-branch else-expr update-expr bound-expr]} (patterns/match-reduce-loop form)]
        (let [full-body (if (and (seq? then-branch)
                                 (contains? #{'let 'let*} (first then-branch)))
                          then-branch
                          update-expr)
              ewise? (body-element-wise? full-body index-sym nil)]
          (when ewise?
            {:acc acc-sym
             :init acc-init
             :idx index-sym
             :bound bound-expr
             :body full-body
             :post-fn (when (not= else-expr acc-sym) else-expr)})))

      (contains? #{'let 'let*} (first form))
      (let [[_ bindings-vec & body-exprs] form]
        (when (and (= 2 (count bindings-vec)) (= 1 (count body-exprs)))
          (let [[n-sym bound-wrapper] (take 2 bindings-vec)
                inner (first body-exprs)
                actual-bound (if (and (seq? bound-wrapper)
                                      (= 'int (first bound-wrapper))
                                      (= 2 (count bound-wrapper)))
                               (second bound-wrapper)
                               bound-wrapper)]
            (when-let [info (loop-reduce-pattern? inner)]
              (if (= (:bound info) n-sym)
                (assoc info :bound actual-bound)
                info))))))))

(defn- loop-scan-pattern?
  "Check if a form matches the loop->scan pattern.
  Returns {:out sym :acc sym :init expr :idx sym :bound expr :cast fn-or-nil :body expr}
  or nil."
  [form]
  (when (seq? form)
    (cond
      (= 'loop (first form))
      (when-let [{:keys [out-sym else-expr acc-sym acc-init index-sym bound-expr cast-fn acc-next-expr]} (patterns/match-scan-loop form)]
        (when (= out-sym else-expr)
          {:out else-expr
           :acc acc-sym
           :init acc-init
           :idx index-sym
           :bound bound-expr
           :cast cast-fn
           :body acc-next-expr}))

      (contains? #{'let 'let*} (first form))
      (let [[_ bindings-vec & body-exprs] form]
        (when (and (= 2 (count bindings-vec)) (= 1 (count body-exprs)))
          (let [[n-sym bound-wrapper] (take 2 bindings-vec)
                inner (first body-exprs)
                actual-bound (if (and (seq? bound-wrapper)
                                      (= 'int (first bound-wrapper))
                                      (= 2 (count bound-wrapper)))
                               (second bound-wrapper)
                               bound-wrapper)]
            (when-let [info (loop-scan-pattern? inner)]
              (if (= (:bound info) n-sym)
                (assoc info :bound actual-bound)
                info)))))

      :else nil)))

;; ================================================================
;; Main pass: lift loop forms in let* bindings
;; ================================================================

(defn- empty-stats []
  {:maps-detected 0 :reduces-detected 0 :scans-detected 0})

(defn- add-stats
  [left right]
  (merge-with + left right))

(defn- cast->elem-type-kw [cast]
  (case cast
    float :float
    double :double
    long :long
    int :int
    (nil) nil
    nil))

(defn- out-sym->elem-type
  "Infer elem-type from output symbol's type tag metadata.
  Used when cast is nil (e.g., in-place mutation like SGD updates)."
  [out-sym]
  (when (symbol? out-sym)
    (case (or (:raster.type/tag (meta out-sym)) (:tag (meta out-sym)))
      floats :float
      doubles :double
      longs :long
      ints :int
      nil)))

(defn- map-lift-form
  [{:keys [out idx bound cast body]}]
  (let [form (list 'raster.par/map! out idx bound cast body)
        et (or (cast->elem-type-kw cast) (out-sym->elem-type out))]
    (if et (with-meta form {:raster.type/elem-type et}) form)))

(defn- scan-lift-form
  [{:keys [out acc init idx bound cast body]}]
  (let [form (list 'raster.par/scan out acc init idx bound cast body)
        et (or (cast->elem-type-kw cast) (out-sym->elem-type out))]
    (if et (with-meta form {:raster.type/elem-type et}) form)))

(defn- reduce-lift-form
  [{:keys [acc init idx bound body post-fn]}]
  (let [reduce-form (list 'raster.par/reduce acc init idx bound body)
        ;; Infer elem-type from init expression
        et (cond
             (and (seq? init) (= 'float (first init))) :float
             (and (seq? init) (= 'double (first init))) :double
             (number? init) (condp instance? init
                              Double :double
                              Float :float
                              Long :long
                              Integer :int
                              (throw (ex-info (str "loop-lift: cannot infer element type for reduce init `"
                                                   (pr-str init) "` of type " (type init))
                                              {:init init})))
             :else (throw (ex-info (str "loop-lift: cannot infer element type for reduce init `"
                                        (pr-str init) "`. Use a typed cast like (double 0).")
                                   {:init init})))
        reduce-form (with-meta reduce-form {:raster.type/elem-type et})]
    (if post-fn
      (clojure.walk/postwalk-replace {acc reduce-form} post-fn)
      reduce-form)))

(defn- let-form?
  [form]
  (and (seq? form) (contains? #{'let 'let*} (first form))))

(defn- classify-loop-form
  [expr]
  (cond
    (nested-dotimes-map-pattern? expr) :nested-map
    (dotimes-map-pattern? expr) :map
    (loop-scan-pattern? expr) :scan
    (loop-reduce-pattern? expr) :reduce
    (let-form? expr) :nested-let
    :else :plain))

(defmulti rewrite-binding-expr
  "Rewrite a binding expression according to the loop shape it encodes."
  (fn [_expr] (classify-loop-form _expr)))

(defmethod rewrite-binding-expr :nested-map
  [expr]
  {:expr (map-lift-form (nested-dotimes-map-pattern? expr))
   :stats {:maps-detected 1 :reduces-detected 0 :scans-detected 0}})

(defmethod rewrite-binding-expr :map
  [expr]
  {:expr (map-lift-form (dotimes-map-pattern? expr))
   :stats {:maps-detected 1 :reduces-detected 0 :scans-detected 0}})

(defmethod rewrite-binding-expr :scan
  [expr]
  {:expr (scan-lift-form (loop-scan-pattern? expr))
   :stats {:maps-detected 0 :reduces-detected 0 :scans-detected 1}})

(defmethod rewrite-binding-expr :reduce
  [expr]
  {:expr (reduce-lift-form (loop-reduce-pattern? expr))
   :stats {:maps-detected 0 :reduces-detected 1 :scans-detected 0}})

(declare lift-parallel-forms)

(defmethod rewrite-binding-expr :nested-let
  [expr]
  (let [inner (lift-parallel-forms expr)]
    {:expr (:form inner)
     :stats (:stats inner)}))

(defmethod rewrite-binding-expr :plain
  [expr]
  {:expr expr :stats (empty-stats)})

(defn lift-parallel-forms
  "Scan a flat let* form for loop patterns that can be lifted into raster.par nodes.

  Detects:
    - nested dotimes + row-major aset => raster.par/map! (flattened 1D)
    - dotimes + aset => raster.par/map! (including in-place updates like SGD)
    - loop/recur accumulation => raster.par/reduce
    - loop/recur with aset + accumulator => raster.par/scan

  Returns {:form new-form :stats {:maps-detected N :reduces-detected N :scans-detected N}}"
  [let-form]
  (if-not (let-form? let-form)
    {:form let-form :stats (empty-stats)}
    (let [[let-sym bindings-vec & body-exprs] let-form
          pairs (vec (partition 2 bindings-vec))
          {:keys [pairs stats]}
          (reduce (fn [{:keys [pairs stats]} [sym expr]]
                    (let [{new-expr :expr new-stats :stats} (rewrite-binding-expr expr)]
                      {:pairs (conj pairs [sym new-expr])
                       :stats (add-stats stats new-stats)}))
                  {:pairs [] :stats (empty-stats)}
                  pairs)
          {:keys [body stats]}
          (reduce (fn [{:keys [body stats]} expr]
                    (if (let-form? expr)
                      (let [inner (lift-parallel-forms expr)]
                        {:body (conj body (:form inner))
                         :stats (add-stats stats (:stats inner))})
                      {:body (conj body expr)
                       :stats stats}))
                  {:body [] :stats stats}
                  body-exprs)]
      {:form (list* let-sym (vec (mapcat identity pairs)) body)
       :stats stats})))
