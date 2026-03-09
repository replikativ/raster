(ns raster.compiler.passes.parallel.simd
  "SIMD pattern detection and Java Vector API codegen.

  Detects vectorizable element-wise loops in walked S-expressions and
  generates SIMD code using Java Vector API (jdk.incubator.vector).
  Uses JIT isolation: SIMD code in separate class to avoid degradation.

  Accepts loop*, loop, and dotimes forms. The walker preserves loop/dotimes
  without macroexpanding to loop*, so we must handle all three.

  Supported patterns:
    :map             -- element-wise unary transform (relu, abs, sqrt, exp, neg)
    :map-binary      -- element-wise binary op on two arrays
    :map-scalar      -- scalar-array op: (aset out i (op scalar (aget a i)))
    :map-compound    -- compound: (aset out i (+ (aget u i) (* s (aget k i))))
    :reduce          -- fold/accumulate (sum, product, min, max)

  Supported ops:
    relu (Math/max 0 x), Math/abs, Math/sqrt, Math/exp, Math/log,
    Math/sin, Math/cos, +, -, *, /, Math/max, Math/min"
  (:require [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.passes.parallel.patterns :as patterns]))

;; SIMD op sets are centralized in op-descriptor
(def ^:private simd-unary-ops
  (set (concat (keys descriptor/simd-unary-ops)
               (keys descriptor/simd-lanewise-unary-ops))))

(def ^:private simd-binary-ops
  (set (keys descriptor/simd-binary-ops)))

;; ================================================================
;; Pattern detection helpers
;; ================================================================

(defn match-reduce-pattern
  "Detect reduction loops. Handles both binding orderings:
  (loop [acc init i 0] ...) and (loop [i 0 acc init] ...)
  Also handles (double acc) wrappers and qualified aget.
  Accepts loop*, loop, and clojure.core/loop."
  [loop-form]
  (when-let [result (patterns/match-binary-reduce-loop loop-form #(contains? simd-binary-ops %))]
    (assoc result :pattern :reduce)))
;; ================================================================
;; Main entry point for pattern detection
;; ================================================================

(defn vectorizable?
  "Analyze a loop/loop*/dotimes S-expr for SIMD pattern.
  Accepts loop*, loop, clojure.core/loop, and dotimes.
  Returns a pattern map or nil if not vectorizable."
  [form]
  (or (patterns/match-map-loop form {:unary-ops simd-unary-ops
                                     :binary-ops simd-binary-ops
                                     :include-relu? true
                                     :include-compound? true})
      (match-reduce-pattern form)))

;; ================================================================
;; SIMD code generation (S-expression form)
;; ================================================================

(defn generate-simd-sexp
  "Generate an S-expression that uses Java Vector API for the detected pattern.
  Returns a replacement S-expression for the loop, or nil if unsupported.

  The generated code uses DoubleVector.SPECIES_PREFERRED for auto-width
  detection (4 lanes AVX2, 8 lanes AVX-512)."
  [pattern {:keys [bound-sym]}]
  (case (:pattern pattern)
    :map
    (let [{:keys [op in-array out-array]} pattern
          species-sym (gensym "species_")
          lanes-sym (gensym "lanes_")
          i-sym (gensym "i_")
          vec-sym (gensym "vec_")
          zero-sym (gensym "zero_")
          simd-op (condp = op
                    :relu     `(.max ~vec-sym ~zero-sym)
                    'Math/abs `(.abs ~vec-sym)
                    'Math/sqrt `(.lanewise ~vec-sym
                                           jdk.incubator.vector.VectorOperators/SQRT)
                    'Math/exp `(.lanewise ~vec-sym
                                          jdk.incubator.vector.VectorOperators/EXP)
                    'Math/log `(.lanewise ~vec-sym
                                          jdk.incubator.vector.VectorOperators/LOG)
                    'Math/sin `(.lanewise ~vec-sym
                                          jdk.incubator.vector.VectorOperators/SIN)
                    'Math/cos `(.lanewise ~vec-sym
                                          jdk.incubator.vector.VectorOperators/COS)
                    vec-sym)
          scalar-op (condp = op
                      :relu     `(Math/max 0.0 (aget ~in-array ~i-sym))
                      'Math/abs `(Math/abs (aget ~in-array ~i-sym))
                      'Math/sqrt `(Math/sqrt (aget ~in-array ~i-sym))
                      'Math/exp `(Math/exp (aget ~in-array ~i-sym))
                      'Math/log `(Math/log (aget ~in-array ~i-sym))
                      'Math/sin `(Math/sin (aget ~in-array ~i-sym))
                      'Math/cos `(Math/cos (aget ~in-array ~i-sym))
                      `(aget ~in-array ~i-sym))]
      `(let* [~species-sym (jdk.incubator.vector.DoubleVector/SPECIES_PREFERRED)
              ~lanes-sym (.length ~species-sym)
              ~@(when (= op :relu)
                  [zero-sym `(jdk.incubator.vector.DoubleVector/zero ~species-sym)])]
         ;; SIMD main loop
             (loop* [~i-sym (int 0)]
                    (if (<= (+ ~i-sym ~lanes-sym) ~bound-sym)
                      (let* [~vec-sym (jdk.incubator.vector.DoubleVector/fromArray
                                       ~species-sym ~in-array ~i-sym)
                             ~vec-sym ~simd-op]
                            (.intoArray ~vec-sym ~out-array ~i-sym)
                            (recur (+ ~i-sym ~lanes-sym)))
             ;; Scalar tail
                      (loop* [~i-sym ~i-sym]
                             (if (< ~i-sym ~bound-sym)
                               (do (aset ~out-array ~i-sym ~scalar-op)
                                   (recur (inc ~i-sym)))
                               nil))))))

    :map-binary
    (let [{:keys [op in-array-1 in-array-2 out-array]} pattern
          species-sym (gensym "species_")
          lanes-sym (gensym "lanes_")
          i-sym (gensym "i_")
          vec1-sym (gensym "v1_")
          vec2-sym (gensym "v2_")
          method (condp = op '+ '.add '- '.sub '* '.mul '/ '.div
                        'Math/max '.max 'Math/min '.min nil)]
      (when method
        `(let* [~species-sym (jdk.incubator.vector.DoubleVector/SPECIES_PREFERRED)
                ~lanes-sym (.length ~species-sym)]
               (loop* [~i-sym (int 0)]
                      (if (<= (+ ~i-sym ~lanes-sym) ~bound-sym)
                        (let* [~vec1-sym (jdk.incubator.vector.DoubleVector/fromArray
                                          ~species-sym ~in-array-1 ~i-sym)
                               ~vec2-sym (jdk.incubator.vector.DoubleVector/fromArray
                                          ~species-sym ~in-array-2 ~i-sym)
                               ~vec1-sym (~method ~vec1-sym ~vec2-sym)]
                              (.intoArray ~vec1-sym ~out-array ~i-sym)
                              (recur (+ ~i-sym ~lanes-sym)))
                        (loop* [~i-sym ~i-sym]
                               (if (< ~i-sym ~bound-sym)
                                 (do (aset ~out-array ~i-sym
                                           (double (~op (aget ~in-array-1 ~i-sym)
                                                        (aget ~in-array-2 ~i-sym))))
                                     (recur (inc ~i-sym)))
                                 nil)))))))

    :map-scalar
    (let [{:keys [op scalar scalar-pos in-array out-array]} pattern
          species-sym (gensym "species_")
          lanes-sym (gensym "lanes_")
          i-sym (gensym "i_")
          vec-sym (gensym "vec_")
          scalar-vec-sym (gensym "svec_")
          simd-op (if (contains? #{'+ '- '* '/} op)
                    (let [method (condp = op '+ '.add '- '.sub '* '.mul '/ '.div)]
                      (if (= scalar-pos :left)
                        `(~method ~scalar-vec-sym ~vec-sym)
                        `(~method ~vec-sym ~scalar-vec-sym)))
                    (let [method (condp = op 'Math/max '.max 'Math/min '.min)]
                      `(~method ~vec-sym ~scalar-vec-sym)))
          scalar-expr (if (= scalar-pos :left)
                        `(~op ~scalar (aget ~in-array ~i-sym))
                        `(~op (aget ~in-array ~i-sym) ~scalar))]
      `(let* [~species-sym (jdk.incubator.vector.DoubleVector/SPECIES_PREFERRED)
              ~lanes-sym (.length ~species-sym)
              ~scalar-vec-sym (jdk.incubator.vector.DoubleVector/broadcast
                               ~species-sym (double ~scalar))]
             (loop* [~i-sym (int 0)]
                    (if (<= (+ ~i-sym ~lanes-sym) ~bound-sym)
                      (let* [~vec-sym (jdk.incubator.vector.DoubleVector/fromArray
                                       ~species-sym ~in-array ~i-sym)
                             ~vec-sym ~simd-op]
                            (.intoArray ~vec-sym ~out-array ~i-sym)
                            (recur (+ ~i-sym ~lanes-sym)))
                      (loop* [~i-sym ~i-sym]
                             (if (< ~i-sym ~bound-sym)
                               (do (aset ~out-array ~i-sym (double ~scalar-expr))
                                   (recur (inc ~i-sym)))
                               nil))))))

    :map-compound
    (let [{:keys [terms out-array]} pattern
          species-sym (gensym "species_")
          lanes-sym (gensym "lanes_")
          i-sym (gensym "i_")
          ;; Each term gets a vector sym
          term-syms (mapv (fn [_] (gensym "tv_")) terms)
          coeff-vec-syms (mapv (fn [_] (gensym "cv_")) terms)
          acc-sym (gensym "acc_")
          coeff-bindings (mapcat (fn [term cv-sym]
                                   (when (not= 1.0 (:coeff term))
                                     [cv-sym `(jdk.incubator.vector.DoubleVector/broadcast
                                               ~species-sym (double ~(:coeff term)))]))
                                 terms coeff-vec-syms)
          simd-load-bindings
          (mapcat (fn [term tv-sym cv-sym]
                    (if (= 1.0 (:coeff term))
                      [tv-sym `(jdk.incubator.vector.DoubleVector/fromArray
                                ~species-sym ~(:array term) ~i-sym)]
                      [tv-sym `(.mul
                                (jdk.incubator.vector.DoubleVector/fromArray
                                 ~species-sym ~(:array term) ~i-sym)
                                ~cv-sym)]))
                  terms term-syms coeff-vec-syms)
          sum-expr (reduce (fn [a b] `(.add ~a ~b)) term-syms)
          scalar-terms (map (fn [term]
                              (if (= 1.0 (:coeff term))
                                `(aget ~(:array term) ~i-sym)
                                `(* (double ~(:coeff term))
                                    (aget ~(:array term) ~i-sym))))
                            terms)
          scalar-sum (reduce (fn [a b] `(+ ~a ~b)) scalar-terms)]
      `(let* [~species-sym (jdk.incubator.vector.DoubleVector/SPECIES_PREFERRED)
              ~lanes-sym (.length ~species-sym)
              ~@coeff-bindings]
             (loop* [~i-sym (int 0)]
                    (if (<= (+ ~i-sym ~lanes-sym) ~bound-sym)
                      (let* [~@simd-load-bindings
                             ~acc-sym ~sum-expr]
                            (.intoArray ~acc-sym ~out-array ~i-sym)
                            (recur (+ ~i-sym ~lanes-sym)))
                      (loop* [~i-sym ~i-sym]
                             (if (< ~i-sym ~bound-sym)
                               (do (aset ~out-array ~i-sym (double ~scalar-sum))
                                   (recur (inc ~i-sym)))
                               nil))))))

    :map2
    (let [{:keys [out-array-1 out-array-2 in-arrays op1 op2]} pattern
          species-sym (gensym "species_")
          lanes-sym (gensym "lanes_")
          i-sym (gensym "i_")]
      ;; map2 SIMD only supported when both bodies are simple binary ops
      (when (and op1 op2 (= 1 (count in-arrays)))
        (let [in-array (first in-arrays)
              vec-sym (gensym "vec_")
              res1-sym (gensym "r1_")
              res2-sym (gensym "r2_")
              method1 (condp = op1 '+ '.add '- '.sub '* '.mul '/ '.div nil)
              method2 (condp = op2 '+ '.add '- '.sub '* '.mul '/ '.div nil)]
          (when (and method1 method2)
            `(let* [~species-sym (jdk.incubator.vector.DoubleVector/SPECIES_PREFERRED)
                    ~lanes-sym (.length ~species-sym)]
                   (loop* [~i-sym (int 0)]
                          (if (<= (+ ~i-sym ~lanes-sym) ~bound-sym)
                            (let* [~vec-sym (jdk.incubator.vector.DoubleVector/fromArray
                                             ~species-sym ~in-array ~i-sym)
                                   ~res1-sym (~method1 ~vec-sym ~vec-sym)
                                   ~res2-sym (~method2 ~vec-sym ~vec-sym)]
                                  (.intoArray ~res1-sym ~out-array-1 ~i-sym)
                                  (.intoArray ~res2-sym ~out-array-2 ~i-sym)
                                  (recur (+ ~i-sym ~lanes-sym)))
                            (loop* [~i-sym ~i-sym]
                                   (if (< ~i-sym ~bound-sym)
                                     (do (aset ~out-array-1 ~i-sym
                                               (double (~op1 (aget ~in-array ~i-sym)
                                                             (aget ~in-array ~i-sym))))
                                         (aset ~out-array-2 ~i-sym
                                               (double (~op2 (aget ~in-array ~i-sym)
                                                             (aget ~in-array ~i-sym))))
                                         (recur (inc ~i-sym)))
                                     nil)))))))))

    :stencil
    (let [{:keys [loads out-array]} pattern
          species-sym (gensym "species_")
          lanes-sym (gensym "lanes_")
          j-sym (gensym "j_")
          res-sym (gensym "res_")
          ;; Generate a vector sym and load binding for each shifted load
          load-syms (mapv (fn [_] (gensym "sv_")) loads)
          simd-load-bindings
          (mapcat (fn [load-info v-sym]
                    [v-sym `(jdk.incubator.vector.DoubleVector/fromArray
                             ~species-sym ~(:array load-info) ~(:offset-expr load-info))])
                  loads load-syms)
          ;; Sum all loaded vectors: v0.add(v1).add(v2)...
          sum-expr (clojure.core/reduce (fn [a b] `(.add ~a ~b)) load-syms)
          ;; Scalar fallback: sum of all aget calls
          scalar-loads (map (fn [load-info]
                              `(aget ~(:array load-info) ~(:offset-expr load-info)))
                            loads)
          scalar-sum (clojure.core/reduce (fn [a b] `(+ ~a ~b)) scalar-loads)]
      (when out-array
        `(let* [~species-sym (jdk.incubator.vector.DoubleVector/SPECIES_PREFERRED)
                ~lanes-sym (.length ~species-sym)]
               (loop* [~j-sym (int 0)]
                      (if (<= (+ ~j-sym ~lanes-sym) ~bound-sym)
                        (let* [~@simd-load-bindings
                               ~res-sym ~sum-expr]
                              (.intoArray ~res-sym ~out-array ~j-sym)
                              (recur (+ ~j-sym ~lanes-sym)))
                        (loop* [~j-sym ~j-sym]
                               (if (< ~j-sym ~bound-sym)
                                 (do (aset ~out-array ~j-sym (double ~scalar-sum))
                                     (recur (inc ~j-sym)))
                                 nil)))))))

    :butterfly
    (let [{:keys [re im wr wi base]} pattern
          species-sym (gensym "species_")
          lanes-sym (gensym "lanes_")
          i-sym (gensym "i_")
          wr-v (gensym "wr_v_")
          wi-v (gensym "wi_v_")
          re-lo (gensym "re_lo_")
          re-hi (gensym "re_hi_")
          im-lo (gensym "im_lo_")
          im-hi (gensym "im_hi_")
          vr-v (gensym "vr_v_")
          vi-v (gensym "vi_v_")
          lo-sym (gensym "lo_")
          hi-sym (gensym "hi_")
          ;; Scalar tail syms
          ur-s (gensym "ur_s_")
          ui-s (gensym "ui_s_")
          rhi-s (gensym "rhi_s_")
          ihi-s (gensym "ihi_s_")
          wr-s (gensym "wr_s_")
          wi-s (gensym "wi_s_")
          vr-s (gensym "vr_s_")
          vi-s (gensym "vi_s_")]
      `(let* [~species-sym (jdk.incubator.vector.DoubleVector/SPECIES_PREFERRED)
              ~lanes-sym (.length ~species-sym)]
             ;; SIMD main loop
             (loop* [~i-sym (int 0)]
                    (if (<= (+ ~i-sym ~lanes-sym) ~bound-sym)
                      (let* [~lo-sym (+ ~base ~i-sym)
                             ~hi-sym (+ ~lo-sym ~bound-sym)
                             ~wr-v (jdk.incubator.vector.DoubleVector/fromArray
                                    ~species-sym ~wr ~i-sym)
                             ~wi-v (jdk.incubator.vector.DoubleVector/fromArray
                                    ~species-sym ~wi ~i-sym)
                             ~re-lo (jdk.incubator.vector.DoubleVector/fromArray
                                     ~species-sym ~re ~lo-sym)
                             ~re-hi (jdk.incubator.vector.DoubleVector/fromArray
                                     ~species-sym ~re ~hi-sym)
                             ~im-lo (jdk.incubator.vector.DoubleVector/fromArray
                                     ~species-sym ~im ~lo-sym)
                             ~im-hi (jdk.incubator.vector.DoubleVector/fromArray
                                     ~species-sym ~im ~hi-sym)
                             ;; vr = wr*re_hi - wi*im_hi
                             ~vr-v (.sub (.mul ~wr-v ~re-hi) (.mul ~wi-v ~im-hi))
                             ;; vi = wr*im_hi + wi*re_hi
                             ~vi-v (.add (.mul ~wr-v ~im-hi) (.mul ~wi-v ~re-hi))]
                            ;; re[lo] = re_lo + vr, re[hi] = re_lo - vr
                            (.intoArray (.add ~re-lo ~vr-v) ~re ~lo-sym)
                            (.intoArray (.sub ~re-lo ~vr-v) ~re ~hi-sym)
                            ;; im[lo] = im_lo + vi, im[hi] = im_lo - vi
                            (.intoArray (.add ~im-lo ~vi-v) ~im ~lo-sym)
                            (.intoArray (.sub ~im-lo ~vi-v) ~im ~hi-sym)
                            (recur (+ ~i-sym ~lanes-sym)))
                      ;; Scalar tail
                      (loop* [~i-sym ~i-sym]
                             (if (< ~i-sym ~bound-sym)
                               (let* [~lo-sym (+ ~base ~i-sym)
                                      ~hi-sym (+ ~lo-sym ~bound-sym)
                                      ~ur-s (aget ~re ~lo-sym)
                                      ~ui-s (aget ~im ~lo-sym)
                                      ~rhi-s (aget ~re ~hi-sym)
                                      ~ihi-s (aget ~im ~hi-sym)
                                      ~wr-s (aget ~wr ~i-sym)
                                      ~wi-s (aget ~wi ~i-sym)
                                      ~vr-s (- (* ~wr-s ~rhi-s) (* ~wi-s ~ihi-s))
                                      ~vi-s (+ (* ~wr-s ~ihi-s) (* ~wi-s ~rhi-s))]
                                     (aset ~re ~lo-sym (+ ~ur-s ~vr-s))
                                     (aset ~re ~hi-sym (- ~ur-s ~vr-s))
                                     (aset ~im ~lo-sym (+ ~ui-s ~vi-s))
                                     (aset ~im ~hi-sym (- ~ui-s ~vi-s))
                                     (recur (inc ~i-sym)))
                               nil))))))

    :reduce nil ;; SIMD reduction not yet implemented — falls back to scalar

    ;; Unsupported pattern
    nil))

(defn emit-simd-class
  "Generate metadata for a separate SIMD class.
  JIT isolation: separate class prevents JIT degradation.

  Returns {:class-name name :pattern pattern :simd-sexp sexp} or nil."
  [class-name pattern]
  (when pattern
    {:class-name class-name
     :pattern pattern
     :simd-sexp (generate-simd-sexp pattern {:bound-sym 'n})}))
