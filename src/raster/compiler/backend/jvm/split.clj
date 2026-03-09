(ns raster.compiler.backend.jvm.split
  "C2-aware bytecode splitting for JVM compilation.

  Splits large method bodies into ≤280-byte same-class invokestatic helpers
  so that C2's FreqInlineSize=325 threshold allows chain inlining. This
  enables the lazy JIT path to achieve AOT-level performance without
  IR-level inlining.

  The splitter works on walked IR (let* bodies) before bytecode emission.
  It estimates bytecode size per form, greedily partitions into chunks,
  and extracts each chunk as a same-class static helper method.

  Size model calibrated against actual bytecode measurements:
    return x          →  2 bytes
    x + y             →  4 bytes
    Math/fma(x,y,z)   →  8 bytes
    aget u 0           →  5 bytes
    f.invk(k,u,t)     → 21 bytes
    dotimes[3] aset   → 43 bytes
    f call + 1 stage  → 74 bytes"
  (:require [raster.compiler.core.util :as util]))

;; ================================================================
;; Bytecode size estimator
;; ================================================================

(def ^:const TARGET-SIZE
  "Target maximum bytecode size per method chunk.
  Must be under C2's FreqInlineSize (325 bytes) to ensure inlining.
  Leave margin for method prologue/epilogue + chunk call overhead.
  The orchestrator compute_static also includes let bindings (~15 bytes
  per binding) which can't be split, so the effective budget for body
  forms in the orchestrator is TARGET-SIZE minus binding overhead."
  280)

(defn- estimate-symbol-load
  "Estimate bytes for loading a symbol: dload/aload = 1-2 bytes."
  [_sym]
  2)

(defn- estimate-literal
  "Estimate bytes for a literal value."
  [x]
  (cond
    (nil? x) 1            ;; aconst_null
    (instance? Long x) (if (<= -1 (long x) 5) 1 3)  ;; iconst/bipush/ldc
    (instance? Double x) 3  ;; ldc2_w
    (instance? Boolean x) 1 ;; iconst_0/1
    (string? x) 3           ;; ldc
    :else 3))

(defn estimate-form-size
  "Estimate the bytecode size (in bytes) that a walked IR form will compile to.
  Conservative — overestimates rather than underestimates."
  [form]
  (cond
    ;; Literals
    (not (seq? form))
    (if (symbol? form) (estimate-symbol-load form) (estimate-literal form))

    ;; Special forms
    :else
    (let [head (first form)]
      (case (when (symbol? head) head)
        ;; Let bindings: each binding = init + store(2), plus body
        (let let*)
        (let [[_ bindings & body] form
              pairs (partition 2 bindings)
              bind-size (reduce + (map (fn [[_sym init]] (+ 2 (estimate-form-size init))) pairs))
              body-size (reduce + (map estimate-form-size body))]
          (+ bind-size body-size))

        ;; Do: sum of all forms
        do
        (reduce + (map estimate-form-size (rest form)))

        ;; If: test + branch(3) + then + goto(3) + else
        if
        (let [[_ test then else] form]
          (+ (estimate-form-size test) 3
             (estimate-form-size then) 3
             (estimate-form-size (or else 0))))

        ;; Loop/recur: same as let + goto(3) per recur
        (loop loop*)
        (let [[_ bindings & body] form
              pairs (partition 2 bindings)
              bind-size (reduce + (map (fn [[_sym init]] (+ 2 (estimate-form-size init))) pairs))
              body-size (reduce + (map estimate-form-size body))]
          (+ bind-size body-size 3)) ;; +3 for loop back-edge

        recur
        (+ (reduce + (map estimate-form-size (rest form))) 3)

        ;; Dotimes: loop setup(8) + body + back-edge(5)
        dotimes
        (let [[_ [_i bound] & body] form]
          (+ 8 (estimate-form-size bound) (reduce + (map estimate-form-size body)) 5))

        ;; Par forms (expanded to dotimes by pre_expand_par_forms)
        raster.par/map!
        (let [[_ _arr _idx bound _cast body] form]
          (+ 15 (estimate-form-size bound) (estimate-form-size body)))

        raster.par/reduce
        (let [[_ _acc init _idx bound body] form]
          (+ 15 (estimate-form-size init) (estimate-form-size bound) (estimate-form-size body)))

        ;; Method calls
        .  ;; (. obj method args...)
        (let [args (drop 2 form)
              method-args (if (seq? (second (rest form)))
                            (rest (second (rest form)))
                            (drop 2 (rest form)))]
          (+ 5 ;; aload receiver + invokeinterface/invokevirtual
             (reduce + (map estimate-form-size method-args))))

        ;; .invk sugar — emitted by walker for typed dispatch
        .invk
        (+ 21 ;; receiver + instanceof/checkcast + invokeinterface + args
           (reduce + (map estimate-form-size (drop 2 form))))

        ;; Java static calls
        (let [name-str (str head)]
          (cond
            ;; Math/fma, Math/pow, etc.
            (.startsWith name-str "Math/")
            (+ 5 (reduce + (map estimate-form-size (rest form))))

            ;; System/arraycopy
            (.startsWith name-str "System/")
            (+ 5 (reduce + (map estimate-form-size (rest form))))

            ;; Clojure core ops: +, -, *, /, aget, aset, aclone, alength
            (.startsWith name-str "clojure.core/")
            (let [op (subs name-str 13)]
              (case op
                ("+" "-" "*" "/") (+ 1 (reduce + (map estimate-form-size (rest form))))
                ("aget")          (+ 1 (reduce + (map estimate-form-size (rest form))))
                ("aset")          (+ 1 (reduce + (map estimate-form-size (rest form))))
                ("aclone")        (+ 5 (estimate-form-size (second form)))
                ("alength")       (+ 3 (estimate-form-size (second form)))
                (">=" "<=" ">" "<" "==" "=")
                (+ 5 (reduce + (map estimate-form-size (rest form))))
                ;; Default clojure.core call
                (+ 10 (reduce + (map estimate-form-size (rest form))))))

            ;; Qualified var call (invokedynamic or invokestatic)
            (namespace (symbol name-str))
            (+ 15 (reduce + (map estimate-form-size (rest form))))

            ;; Unqualified call
            :else
            (+ 10 (reduce + (map estimate-form-size (rest form))))))))))

;; ================================================================
;; Body splitter
;; ================================================================

(defn- free-syms-in
  "Find free symbols in form that aren't in bound-set."
  [form bound-set]
  (util/free-syms form bound-set))

(defn split-for-c2
  "Split a let-body into chunks targeting C2's FreqInlineSize.
  Each chunk is a group of consecutive body forms whose combined estimated
  bytecode size ≤ TARGET-SIZE bytes.

  Returns {:main-chunks [[form...] [form...] ...]
           :chunk-sizes [estimated-bytes ...]}

  The caller should extract each chunk as a same-class invokestatic helper."
  [body-forms]
  (if (<= (count body-forms) 1)
    {:main-chunks [body-forms] :chunk-sizes [(reduce + (map estimate-form-size body-forms))]}
    (let [sizes (mapv estimate-form-size body-forms)]
      (loop [i 0
             current-chunk []
             current-size 0
             chunks []
             chunk-sizes []]
        (if (>= i (count body-forms))
          ;; Flush last chunk
          (let [final-chunks (if (seq current-chunk)
                               (conj chunks current-chunk)
                               chunks)
                final-sizes (if (seq current-chunk)
                              (conj chunk-sizes current-size)
                              chunk-sizes)]
            {:main-chunks final-chunks :chunk-sizes final-sizes})
          (let [form-size (nth sizes i)
                new-size (+ current-size form-size)]
            (if (and (> new-size TARGET-SIZE) (seq current-chunk))
              ;; This form would exceed the target — start a new chunk
              (recur i [] 0 (conj chunks current-chunk) (conj chunk-sizes current-size))
              ;; Add to current chunk
              (recur (inc i) (conj current-chunk (nth body-forms i)) new-size chunks chunk-sizes))))))))

(defn needs-splitting?
  "Check if a walked body would benefit from C2-aware splitting.
  Returns true if the estimated bytecode size exceeds TARGET-SIZE."
  [walked-body]
  (let [total (reduce + (map estimate-form-size walked-body))]
    (> total TARGET-SIZE)))
