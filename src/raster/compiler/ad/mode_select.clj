(ns raster.compiler.ad.mode-select
  "Automatic forward/reverse/mixed-mode AD selection.

   Analyzes Jacobian structure statically via symbolic tracing and
   selects the optimal AD mode based on dimensions, sparsity, and linearity.

   JAX doesn't do this (user chooses). Enzyme can't (post-compilation).
   Raster can because trace-symbolic + sym.analysis give us the Jacobian
   structure statically.

   Usage:
     (analyze-jacobian-structure #'my-fn '[x y z])
     ;=> {:shape [1 3] :recommended-mode :reverse ...}

     (select-mode my-analysis)
     ;=> :reverse"
  (:require [raster.sym.core :as sym]
            [raster.sym.analysis :as sa]
            [clojure.set]))

;; ================================================================
;; Symbolic tracing helpers
;; ================================================================

(defn- trace-var-symbolic
  "Trace a deftm var symbolically, returning the unwrapped expression."
  [f-var param-syms]
  (let [bindings (into {} (map (fn [s] [s (sym/sym s)]) param-syms))
        result (sym/trace-symbolic f-var bindings)]
    (sym/unwrap result)))

;; ================================================================
;; Internal mode selection heuristic
;; ================================================================

(defn- select-mode-heuristic
  "Internal mode selection heuristic."
  [^long m ^long n ^double density linearity]
  (cond
    ;; Scalar output: reverse mode (standard gradient)
    (= m 1) :reverse
    ;; Scalar input: forward mode (single tangent)
    (= n 1) :forward
    ;; Thin Jacobian (m > 3n): forward (fewer passes)
    (> m (* 3 n)) :forward
    ;; Fat Jacobian (n > 3m): reverse (fewer passes)
    (> n (* 3 m)) :reverse
    ;; Sparse: mixed-sparse (seed only non-zero columns)
    (< density 0.3) :mixed-sparse
    ;; Default: reverse if more inputs than outputs
    (> n m) :reverse
    :else :forward))

;; ================================================================
;; Jacobian structure analysis
;; ================================================================

(defn analyze-jacobian-structure
  "Analyze the Jacobian structure of a deftm var.

   f-var: a deftm var
   param-syms: ordered variable symbols

   Returns:
   {:shape [m n]              ;; Jacobian dimensions (m=outputs, n=inputs)
    :sparsity [[bool...]]     ;; from sym_analysis/jacobian-sparsity
    :linearity {var → :constant/:linear/:quadratic/:nonlinear}
    :density double            ;; fraction of nonzero entries
    :recommended-mode :forward|:reverse|:mixed-sparse}"
  [f-var param-syms]
  (let [expr (trace-var-symbolic f-var param-syms)
        output-exprs [expr]
        m (count output-exprs)
        n (count param-syms)
        sparsity (sa/jacobian-sparsity output-exprs param-syms)
        linearity (sa/linearity-analysis expr param-syms)
        nnz (reduce + (map (fn [row] (count (filter true? row))) sparsity))
        total (* m n)
        density (if (zero? total) 0.0 (/ (double nnz) (double total)))
        mode (select-mode-heuristic m n density linearity)]
    {:shape [m n]
     :sparsity sparsity
     :linearity linearity
     :density density
     :recommended-mode mode}))

(defn analyze-from-exprs
  "Analyze Jacobian structure from raw symbolic expressions (no tracing needed).

   output-exprs: vector of S-expressions (one per output dimension)
   param-syms: ordered variable symbols

   Returns same structure as analyze-jacobian-structure."
  [output-exprs param-syms]
  (let [m (count output-exprs)
        n (count param-syms)
        sparsity (sa/jacobian-sparsity output-exprs param-syms)
        ;; For multi-output, take max degree across all outputs per variable
        linearity (into {}
                        (map (fn [v]
                               (let [deg-ord {:constant 0 :linear 1 :quadratic 2 :nonlinear 3}
                                     ord-deg {0 :constant 1 :linear 2 :quadratic 3 :nonlinear}
                                     max-deg (reduce (fn [best expr]
                                                       (let [la (sa/linearity-analysis expr [v])
                                                             deg (get la v :constant)]
                                                         (if (> (deg-ord deg 3) (deg-ord best 3))
                                                           deg best)))
                                                     :constant output-exprs)]
                                 [v max-deg]))
                             param-syms))
        nnz (reduce + (map (fn [row] (count (filter true? row))) sparsity))
        total (* m n)
        density (if (zero? total) 0.0 (/ (double nnz) (double total)))
        mode (select-mode-heuristic m n density linearity)]
    {:shape [m n]
     :sparsity sparsity
     :linearity linearity
     :density density
     :recommended-mode mode}))

;; ================================================================
;; Hessian mode selection
;; ================================================================

(defn analyze-hessian-structure
  "Analyze Hessian structure for a scalar-valued deftm var.

   Returns:
   {:sparsity [[bool...]]     ;; symmetric boolean matrix
    :density double
    :diagonal? bool            ;; true if only diagonal entries nonzero
    :recommended-mode :forward-over-reverse|:diagonal-jet|:sparse-hessian}"
  [f-var param-syms]
  (let [expr (trace-var-symbolic f-var param-syms)
        sparsity (sa/hessian-sparsity expr param-syms)
        n (count param-syms)
        nnz (reduce + (map (fn [row] (count (filter true? row))) sparsity))
        total (* n n)
        density (if (zero? total) 0.0 (/ (double nnz) (double total)))
        diagonal? (every? true?
                          (for [i (range n) j (range n)
                                :when (not= i j)]
                            (not (get-in sparsity [i j]))))
        mode (cond
               diagonal? :diagonal-jet
               (< density 0.3) :sparse-hessian
               :else :forward-over-reverse)]
    {:sparsity sparsity
     :density density
     :diagonal? diagonal?
     :recommended-mode mode}))

;; ================================================================
;; Public mode selection API
;; ================================================================

(defn select-mode
  "Select AD mode from analysis result.
   Returns :forward, :reverse, or :mixed-sparse."
  [analysis]
  (:recommended-mode analysis))

(defn select-hessian-mode
  "Select Hessian computation mode from analysis result.
   Returns :forward-over-reverse, :diagonal-jet, or :sparse-hessian."
  [analysis]
  (:recommended-mode analysis))

;; ================================================================
;; Sparsity-based seeding
;; ================================================================

(defn sparse-seeds
  "Compute seed vectors for sparse Jacobian computation.
   Uses column coloring: independent columns (no shared nonzero rows)
   can share a seed vector, reducing the number of AD passes.

   Returns a vector of seed vectors, each a map {col-index → 1.0}.
   Number of seeds <= number of columns (potentially much less for sparse)."
  [sparsity]
  (let [n (count (first sparsity))
        m (count sparsity)
        col-rows (vec (for [j (range n)]
                        (set (for [i (range m) :when (get-in sparsity [i j])] i))))
        colors (loop [j 0 assignments {}]
                 (if (>= j n)
                   assignments
                   (let [forbidden (set (for [j2 (range j)
                                              :when (seq (clojure.set/intersection
                                                          (nth col-rows j)
                                                          (nth col-rows j2)))]
                                          (get assignments j2)))
                         color (loop [c 0]
                                 (if (contains? forbidden c) (recur (inc c)) c))]
                     (recur (inc j) (assoc assignments j color)))))
        n-colors (if (empty? colors) 0 (inc (apply max (vals colors))))]
    (vec (for [c (range n-colors)]
           (into {} (for [[col col-color] colors :when (= col-color c)]
                      [col 1.0]))))))
