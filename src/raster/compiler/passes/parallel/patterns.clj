(ns raster.compiler.passes.parallel.patterns
  "Shared loop and index matching utilities for parallel pattern passes."
  (:require [raster.compiler.core.op-descriptor :as descriptor]
            [raster.compiler.ir.form :as form]))

(def ^:private loop-heads
  "All forms that represent loops."
  #{'loop* 'loop 'clojure.core/loop})

(def ^:private aget-syms
  "All symbols that represent aget."
  descriptor/aget-ops)

(def ^:private aset-syms
  "All symbols that represent aset."
  descriptor/aset-ops)

(defn idx-matches?
  "Check if an index expression matches idx-sym.
	Accepts: idx-sym, (long idx-sym), (int idx-sym)."
  [idx-expr idx-sym]
  (or (= idx-expr idx-sym)
      (and (seq? idx-expr)
           (contains? #{'long 'int 'clojure.core/long 'clojure.core/int} (first idx-expr))
           (= 2 (count idx-expr))
           (= idx-sym (second idx-expr)))))

(defn aget-form?
  "Check if form is (aget arr idx-sym). Accepts qualified variants and
	cast-wrapped indexes. Returns arr symbol or nil."
  [form idx-sym]
  (when (and (seq? form)
             (contains? aget-syms (first form))
             (= 3 (count form))
             (idx-matches? (nth form 2) idx-sym))
    (second form)))

(defn match-aset-write
  "Match a direct element write of the form (aset out idx expr) or
	(aset out idx (cast expr)). Returns structural information or nil."
  [form idx-sym]
  (when (and (seq? form)
             (contains? aset-syms (first form))
             (>= (count form) 4))
    (let [[_ out idx-expr val-expr] form]
      (when (and (symbol? out)
                 (idx-matches? idx-expr idx-sym))
        (if (and (seq? val-expr)
                 (= 2 (count val-expr))
                 (contains? #{'double 'float 'int 'long} (first val-expr)))
          {:out-sym out
           :index-sym idx-sym
           :cast-fn (first val-expr)
           :value-expr (second val-expr)}
          {:out-sym out
           :index-sym idx-sym
           :cast-fn nil
           :value-expr val-expr})))))

(defn find-aset-through-let
  "Find an aset pattern through single-body let wrappers.
	Returns {:aset-form form :let-bindings [[sym expr] ...]} or nil."
  [form]
  (cond
    (and (seq? form)
         (contains? aset-syms (first form)))
    {:aset-form form :let-bindings []}

    (form/binding-form? form)
    (let [[_ binds & body-exprs] form]
      (when (= 1 (count body-exprs))
        (when-let [inner (find-aset-through-let (first body-exprs))]
          (update inner :let-bindings
                  (fn [bs] (into (vec (partition 2 binds)) bs))))))

    :else nil))

(defn unwrap-to-dotimes
  "Unwrap single-body let wrappers to find an inner dotimes form.
	Returns {:dotimes-form form :let-bindings [[sym expr] ...]} or nil."
  [form]
  (cond
    (and (seq? form) (= 'dotimes (first form)))
    {:dotimes-form form :let-bindings []}

    (form/binding-form? form)
    (let [[_ binds & body-exprs] form]
      (when (= 1 (count body-exprs))
        (when-let [inner (unwrap-to-dotimes (first body-exprs))]
          (update inner :let-bindings
                  (fn [bs] (into (vec (partition 2 binds)) bs))))))

    :else nil))

(defn unwrap-do-result
  "Unwrap a do-wrapper of the form (do inner result) or (do \"note\" inner result).
	Returns [inner-expr result-expr] or nil."
  [form]
  (when (and (seq? form) (= 'do (first form)))
    (let [parts (rest form)]
      (cond
        (= 2 (count parts))
        [(first parts) (second parts)]

        (and (= 3 (count parts)) (string? (first parts)))
        [(second parts) (nth parts 2)]

        :else nil))))

(def ^:private unchecked-add-ops
  "Unchecked add variants for index arithmetic."
  #{'unchecked-add 'clojure.core/unchecked-add})

(def ^:private unchecked-multiply-ops
  "Unchecked multiply variants for index arithmetic."
  #{'unchecked-multiply 'clojure.core/unchecked-multiply})

(defn- index-add-op?
  "True if sym is an addition op including unchecked variants (for index arithmetic)."
  [sym]
  (or (descriptor/addition-op? sym) (contains? unchecked-add-ops sym)))

(defn- index-multiply-op?
  "True if sym is a multiplication op including unchecked variants (for index arithmetic)."
  [sym]
  (or (descriptor/multiplication-op? sym) (contains? unchecked-multiply-ops sym)))

(defn row-major-linear-index?
  "Check whether idx-expr is the row-major linearization (+ (* i cols) j).
	Accepts qualified and unchecked arithmetic variants."
  [idx-expr i-sym j-sym cols-expr]
  (and (seq? idx-expr)
       (index-add-op? (first idx-expr))
       (= 3 (count idx-expr))
       (let [mul-expr (second idx-expr)]
         (and (seq? mul-expr)
              (index-multiply-op? (first mul-expr))
              (= 3 (count mul-expr))
              (= i-sym (second mul-expr))
              (= cols-expr (nth mul-expr 2))))
       (= j-sym (nth idx-expr 2))))

(defn column-major-linear-index?
  "Check whether idx-expr is the column-major linearization (+ (* j rows) i).
				Accepts qualified and unchecked arithmetic variants."
  [idx-expr i-sym j-sym rows-expr]
  (and (seq? idx-expr)
       (index-add-op? (first idx-expr))
       (= 3 (count idx-expr))
       (let [mul-expr (second idx-expr)]
         (and (seq? mul-expr)
              (index-multiply-op? (first mul-expr))
              (= 3 (count mul-expr))
              (= j-sym (second mul-expr))
              (= rows-expr (nth mul-expr 2))))
       (= i-sym (nth idx-expr 2))))

(defn match-dotimes-map-loop
  "Match a dotimes loop that writes one element with aset.
	Handles direct dotimes, do wrappers, and outer let wrappers that only bind
	temporary bound expressions.
	Returns {:out-sym :index-sym :bound-expr :cast-fn :value-expr} or nil."
  [form]
  (when (seq? form)
    (cond
      (= 'dotimes (first form))
      (let [[_ bindings & body-forms] form]
        (when (and (vector? bindings) (= 2 (count bindings)))
          (let [[idx-sym bound-expr] bindings
                raw-body (if (= 1 (count body-forms))
                           (first body-forms)
                           (when (and (seq? (first body-forms))
                                      (= 'do (first (first body-forms))))
                             (last (rest (first body-forms)))))
                aset-result (when raw-body
                              (or (when-let [info (match-aset-write raw-body idx-sym)]
                                    {:info info})
                                  (when-let [{:keys [aset-form let-bindings]} (find-aset-through-let raw-body)]
                                    (when-let [info (match-aset-write aset-form idx-sym)]
                                      (let [body-with-lets (if (seq let-bindings)
                                                             (list 'let* (vec (mapcat identity let-bindings))
                                                                   (:value-expr info))
                                                             (:value-expr info))]
                                        {:info (assoc info :value-expr body-with-lets)})))))]
            (when-let [{:keys [info]} aset-result]
              (assoc info :bound-expr bound-expr :index-sym idx-sym)))))

      (= 'do (first form))
      (let [body-forms (rest form)
            non-string-forms (drop-while string? body-forms)]
        (cond
          (= 2 (count non-string-forms))
          (let [inner (first non-string-forms)
                result (second non-string-forms)]
            (when-let [info (match-dotimes-map-loop inner)]
              (when (= result (:out-sym info))
                info)))

          (= 1 (count non-string-forms))
          (match-dotimes-map-loop (first non-string-forms))))

      (form/binding-form? form)
      (let [[_ bindings-vec & body-exprs] form
            binding-pairs (partition 2 bindings-vec)]
        (when (<= 1 (count body-exprs) 2)
          (let [inner (first body-exprs)
                subst-map (into {}
                                (map (fn [[sym val-expr]]
                                       (if (and (seq? val-expr)
                                                (contains? #{'int 'long 'clojure.core/int 'clojure.core/long}
                                                           (first val-expr))
                                                (= 2 (count val-expr)))
                                         [sym (second val-expr)]
                                         [sym val-expr]))
                                     binding-pairs))]
            (when-let [info (match-dotimes-map-loop inner)]
              (let [bound (:bound-expr info)
                    resolved-bound (get subst-map bound bound)]
                (assoc info :bound-expr resolved-bound)))))))))

(defn match-nested-dotimes-row-major-map
  "Match a nested dotimes row-major write that can be flattened to 1D.
	Returns {:out-sym :index-sym :bound-expr :cast-fn :value-expr} or nil."
  [form]
  (when (and (seq? form) (= 'dotimes (first form)))
    (let [[_ outer-binds & outer-body] form]
      (when (and (vector? outer-binds)
                 (= 2 (count outer-binds))
                 (= 1 (count outer-body)))
        (let [[i-sym rows-expr] outer-binds
              inner-result (unwrap-to-dotimes (first outer-body))]
          (when inner-result
            (let [{:keys [dotimes-form let-bindings]} inner-result
                  outer-lets let-bindings
                  [_ inner-binds & inner-body] dotimes-form]
              (when (and (vector? inner-binds)
                         (= 2 (count inner-binds))
                         (= 1 (count inner-body)))
                (let [[j-sym cols-expr] inner-binds
                      candidate (first inner-body)
                      aset-result (or (find-aset-through-let candidate)
                                      (when (and (seq? candidate)
                                                 (contains? aset-syms (first candidate)))
                                        {:aset-form candidate :let-bindings []}))]
                  (when aset-result
                    (let [{:keys [aset-form let-bindings]} aset-result]
                      (when (and (seq? aset-form) (>= (count aset-form) 4))
                        (let [[_ out-sym idx-expr val-expr] aset-form]
                          (when (and (symbol? out-sym)
                                     (row-major-linear-index? idx-expr i-sym j-sym cols-expr))
                            (let [cast-fn (when (and (seq? val-expr)
                                                     (= 2 (count val-expr))
                                                     (contains? #{'double 'float 'int 'long} (first val-expr)))
                                            (first val-expr))
                                  body (if cast-fn (second val-expr) val-expr)
                                  all-lets (into (vec outer-lets) let-bindings)
                                  body-with-lets (if (seq all-lets)
                                                   (list 'let* (vec (mapcat identity all-lets)) body)
                                                   body)
                                  flat-idx (gensym "flat_idx__")
                                  flat-body (list 'let* [i-sym (list 'clojure.core/quot flat-idx cols-expr)
                                                         j-sym (list 'clojure.core/rem flat-idx cols-expr)]
                                                  body-with-lets)]
                              {:out-sym out-sym
                               :index-sym flat-idx
                               :bound-expr (list 'clojure.core/* rows-expr cols-expr)
                               :cast-fn cast-fn
                               :value-expr flat-body})))))))))))))))

(defn zero-expr?
  "Check if expr evaluates to 0.0. Handles literal 0.0 and (double 0.0)."
  [expr]
  (or (and (number? expr) (== 0.0 (double expr)))
      (and (seq? expr)
           (= 'double (first expr))
           (= 2 (count expr))
           (number? (second expr))
           (== 0.0 (double (second expr))))))

(declare normalize-loop)

(defn match-relu-map-body
  "Detect relu bodies of the form (aset out i (Math/max 0 (aget in i)))."
  [body-form index-sym]
  (when-let [{:keys [out-sym value-expr]} (match-aset-write body-form index-sym)]
    (when (and (seq? value-expr)
               (= 'Math/max (first value-expr))
               (= 3 (count value-expr)))
      (let [[_ zero-arg val-arg] value-expr]
        (or (when (zero-expr? zero-arg)
              (when-let [in-arr (aget-form? val-arg index-sym)]
                {:pattern :map
                 :op :relu
                 :in-array in-arr
                 :out-array out-sym
                 :in-place (= in-arr out-sym)}))
            (when (zero-expr? val-arg)
              (when-let [in-arr (aget-form? zero-arg index-sym)]
                {:pattern :map
                 :op :relu
                 :in-array in-arr
                 :out-array out-sym
                 :in-place (= in-arr out-sym)})))))))

(defn match-unary-map-body
  "Detect unary element-wise bodies of the form (aset out i (f (aget in i)))."
  [body-form index-sym valid-ops]
  (when-let [{:keys [out-sym value-expr]} (match-aset-write body-form index-sym)]
    (when (and (seq? value-expr)
               (= 2 (count value-expr))
               (symbol? (first value-expr)))
      (let [op (first value-expr)]
        (when-let [in-arr (aget-form? (second value-expr) index-sym)]
          (when (contains? valid-ops op)
            {:pattern :map
             :op op
             :in-array in-arr
             :out-array out-sym
             :in-place (= in-arr out-sym)}))))))

(defn match-binary-map-body
  "Detect binary element-wise bodies of the form (aset out i (op (aget a i) (aget b i)))."
  [body-form index-sym valid-ops]
  (when-let [{:keys [out-sym value-expr]} (match-aset-write body-form index-sym)]
    (when (and (seq? value-expr)
               (= 3 (count value-expr)))
      (let [[op arg1 arg2] value-expr]
        (when (contains? valid-ops op)
          (let [arr1 (aget-form? arg1 index-sym)
                arr2 (aget-form? arg2 index-sym)]
            (when (and arr1 arr2)
              {:pattern :map-binary
               :op op
               :in-array-1 arr1
               :in-array-2 arr2
               :out-array out-sym})))))))

(defn match-scalar-map-body
  "Detect scalar-array bodies of the form (aset out i (op scalar (aget a i)))
	or (aset out i (op (aget a i) scalar))."
  [body-form index-sym valid-ops]
  (when-let [{:keys [out-sym value-expr]} (match-aset-write body-form index-sym)]
    (when (and (seq? value-expr)
               (= 3 (count value-expr)))
      (let [[op arg1 arg2] value-expr]
        (when (contains? valid-ops op)
          (let [arr1 (aget-form? arg1 index-sym)
                arr2 (aget-form? arg2 index-sym)]
            (cond
              (and (nil? arr1) arr2
                   (or (number? arg1) (symbol? arg1)))
              {:pattern :map-scalar
               :op op
               :scalar arg1
               :scalar-pos :left
               :in-array arr2
               :out-array out-sym}

              (and arr1 (nil? arr2)
                   (or (number? arg2) (symbol? arg2)))
              {:pattern :map-scalar
               :op op
               :scalar arg2
               :scalar-pos :right
               :in-array arr1
               :out-array out-sym}

              :else nil)))))))

(defn- parse-linear-term
  [expr index-sym]
  (cond
    (aget-form? expr index-sym)
    {:coeff 1.0 :array (aget-form? expr index-sym)}

    (and (seq? expr)
         (descriptor/multiplication-op? (first expr))
         (= 3 (count expr)))
    (let [[_ left right] expr
          arr-left (aget-form? left index-sym)
          arr-right (aget-form? right index-sym)]
      (cond
        (and (nil? arr-left) arr-right (or (number? left) (symbol? left)))
        {:coeff left :array arr-right}

        (and arr-left (nil? arr-right) (or (number? right) (symbol? right)))
        {:coeff right :array arr-left}

        :else nil))

    :else nil))

(defn- parse-linear-sum
  [expr index-sym]
  (if (and (seq? expr) (descriptor/addition-op? (first expr)))
    (mapcat #(parse-linear-sum % index-sym) (rest expr))
    (when-let [term (parse-linear-term expr index-sym)]
      [term])))

(defn match-compound-map-body
  "Detect compound linear-combination array expressions.
	Returns {:pattern :map-compound :terms [...] :out-array sym} or nil."
  [body-form index-sym]
  (when-let [{:keys [out-sym value-expr]} (match-aset-write body-form index-sym)]
    (when-let [terms (seq (parse-linear-sum value-expr index-sym))]
      (when (>= (count terms) 2)
        {:pattern :map-compound
         :terms (vec terms)
         :out-array out-sym}))))

(defn match-map-loop
  "Match a map-style loop and classify the body into a higher-level descriptor.
	Options:
	  :unary-ops       set of allowed unary ops
	  :binary-ops      set of allowed binary ops
	  :include-relu?   whether to match relu-style max(0, x)
	  :include-compound? whether to match linear-combination bodies"
  [loop-form {:keys [unary-ops binary-ops include-relu? include-compound?]
              :or {include-relu? true include-compound? true}}]
  (when-let [normalized (normalize-loop loop-form)]
    (when (= :map-loop (:kind normalized))
      (let [{:keys [index-sym body-forms bound]} normalized]
        (when (and (= 1 (count body-forms)) bound)
          (let [body-form (first body-forms)]
            (or (when include-relu?
                  (match-relu-map-body body-form index-sym))
                (when unary-ops
                  (match-unary-map-body body-form index-sym unary-ops))
                (when binary-ops
                  (match-binary-map-body body-form index-sym binary-ops))
                (when binary-ops
                  (match-scalar-map-body body-form index-sym binary-ops))
                (when include-compound?
                  (match-compound-map-body body-form index-sym)))))))))

(defn match-do-wrapped-transpose
  "Detect a do-wrapped transpose pattern over nested dotimes loops.
	Matches writes of the form out[j * rows + i] = src[i * cols + j].
	Returns {:src sym :out sym :out-buf sym :rows expr :cols expr} or nil."
  [form]
  (when-let [[inner result] (unwrap-do-result form)]
    (when (and (seq? inner) (= 'dotimes (first inner)))
      (let [[_ outer-binds & outer-body] inner]
        (when (and (vector? outer-binds)
                   (= 2 (count outer-binds))
                   (= 1 (count outer-body)))
          (let [[i-sym rows-expr] outer-binds
                inner-result (unwrap-to-dotimes (first outer-body))]
            (when inner-result
              (let [{:keys [dotimes-form]} inner-result
                    [_ inner-binds & inner-body] dotimes-form]
                (when (and (vector? inner-binds)
                           (= 2 (count inner-binds))
                           (= 1 (count inner-body)))
                  (let [[j-sym cols-expr] inner-binds
                        candidate (first inner-body)
                        aset-result (or (find-aset-through-let candidate)
                                        (when (and (seq? candidate)
                                                   (contains? aset-syms (first candidate)))
                                          {:aset-form candidate :let-bindings []}))]
                    (when-let [{:keys [aset-form]} aset-result]
                      (when (and (seq? aset-form) (>= (count aset-form) 4))
                        (let [[_ out-sym write-idx read-expr] aset-form]
                          (when (and (symbol? out-sym)
                                     (column-major-linear-index? write-idx i-sym j-sym rows-expr)
                                     (seq? read-expr)
                                     (contains? aget-syms (first read-expr))
                                     (= 3 (count read-expr)))
                            (let [src-sym (second read-expr)
                                  read-idx (nth read-expr 2)]
                              (when (and (symbol? src-sym)
                                         (row-major-linear-index? read-idx i-sym j-sym cols-expr))
                                {:src src-sym
                                 :out out-sym
                                 :out-buf result
                                 :rows rows-expr
                                 :cols cols-expr}))))))))))))))))

(defn normalize-loop
  "Normalize loop*, loop, and dotimes into a unified representation.
	Returns {:kind :map-loop|:reduce-loop, :index-sym sym, :bound bound,
					 :body-forms [...], :acc-sym sym, :acc-init expr} or nil.

	dotimes: (let [n (long bound)] (loop [i 0] (when (< i n) body (recur (unchecked-inc i)))))
	loop (1-var): (loop [i 0] (if (< i n) (do body (recur (inc i))) nil))
	loop (2-var): (loop [acc init i 0] (if (< i n) (recur new-acc (inc i)) acc))"
  [form]
  (cond
    (and (seq? form)
         (= 'dotimes (first form)))
    (let [[_ [idx-sym bound-expr] & body] form]
      {:kind :map-loop
       :index-sym idx-sym
       :bound bound-expr
       :body-forms (vec body)})

    (and (seq? form)
         (contains? loop-heads (first form)))
    (let [[_ bindings & body] form
          pairs (partition 2 bindings)]
      (cond
        (= 1 (count pairs))
        (let [[[idx-sym idx-init]] pairs
              body-form (last body)]
          (when (and (number? idx-init)
                     (== 0 (long idx-init))
                     (seq? body-form))
            (let [head (first body-form)]
              (cond
                (= 'if head)
                (let [[_ test then-branch _else] body-form]
                  (when (seq? then-branch)
                    (let [work-forms (if (= 'do (first then-branch))
                                       (vec (butlast (rest then-branch)))
                                       [])
                          work-forms (if (empty? work-forms)
                                       (if (= 'do (first then-branch))
                                         (vec (butlast (rest then-branch)))
                                         [])
                                       work-forms)]
                      {:kind :map-loop
                       :index-sym idx-sym
                       :bound (when (and (seq? test) (= 3 (count test)))
                                (nth test 2))
                       :body-forms work-forms})))

                (= 'when head)
                (let [[_ _test & when-body] body-form
                      work-forms (vec (butlast when-body))]
                  {:kind :map-loop
                   :index-sym idx-sym
                   :bound nil
                   :body-forms work-forms})

                :else nil))))

        (= 2 (count pairs))
        (let [[[sym1 init1] [sym2 init2]] pairs
              int-zero? (fn [x] (and (integer? x) (zero? x)))
              [idx-sym acc-sym acc-init]
              (cond
                (and (int-zero? init1) (not (int-zero? init2))) [sym1 sym2 init2]
                (and (int-zero? init2) (not (int-zero? init1))) [sym2 sym1 init1]
                (int-zero? init2) [sym2 sym1 init1]
                :else nil)
              body-form (last body)]
          (when (and idx-sym
                     (seq? body-form)
                     (= 'if (first body-form)))
            {:kind :reduce-loop
             :index-sym idx-sym
             :acc-sym acc-sym
             :acc-init acc-init
             :bound (let [test (second body-form)]
                      (when (and (seq? test) (= 3 (count test)))
                        (nth test 2)))
             :body-form body-form}))

        :else nil))

    :else nil))

;; ================================================================
;; Stencil pattern detection — shifted array loads
;; ================================================================

(defn- extract-aget-with-offset
  "If form is (aget arr offset-expr), return {:array arr :offset-expr offset-expr}.
  Accepts qualified aget variants."
  [form]
  (when (and (seq? form)
             (contains? aget-syms (first form))
             (= 3 (count form)))
    (let [arr (second form)
          offset-expr (nth form 2)]
      (when (symbol? arr)
        {:array arr :offset-expr offset-expr}))))

(defn- collect-aget-loads
  "Recursively collect all aget calls with their offset expressions from a body.
  Returns a seq of {:array sym :offset-expr expr}."
  [form]
  (cond
    (not (seq? form)) []

    (and (contains? aget-syms (first form))
         (= 3 (count form)))
    (if-let [info (extract-aget-with-offset form)]
      [info]
      [])

    :else
    (mapcat collect-aget-loads (rest form))))

(defn- has-offset-index?
  "True if the aget index expression uses arithmetic with offsets
  (e.g., (+ base (dec j)), (+ base (inc j)), (+ base2 j)).
  Simple plain index-sym references are NOT stencil loads."
  [offset-expr idx-sym]
  (and (seq? offset-expr)
       (not (idx-matches? offset-expr idx-sym))))

(defn match-stencil-body
  "Detect stencil access patterns in a loop body — array loads at shifted indices.
  Matches bodies that read from arrays at index offsets like:
    (aget u (+ base (dec j))), (aget u (+ base (inc j))), (aget u (+ base2 j))

  Returns {:pattern :stencil :loads [{:array sym :offset-expr expr} ...]}
  or nil if fewer than 2 shifted loads are found."
  [body-form idx-sym]
  (let [all-loads (collect-aget-loads body-form)
        shifted-loads (filter #(has-offset-index? (:offset-expr %) idx-sym) all-loads)]
    (when (>= (count shifted-loads) 2)
      {:pattern :stencil
       :loads (vec shifted-loads)})))

(defn acc-ref?
  "Check if expr refers to acc-sym, possibly wrapped in (double acc)."
  [expr acc-sym]
  (or (= expr acc-sym)
      (and (seq? expr)
           (= 'double (first expr))
           (= 2 (count expr))
           (= (second expr) acc-sym))))

(defn find-recur-form
  "Find the recur form in a then-branch. Handles direct, do-wrapped, and
	let-wrapped recur forms."
  [form]
  (when (seq? form)
    (cond
      (= 'recur (first form)) form

      (= 'do (first form))
      (let [last-form (last form)]
        (when (and (seq? last-form) (= 'recur (first last-form)))
          last-form))

      (form/binding-form? form)
      (let [last-form (last (drop 2 form))]
        (when (and (seq? last-form) (= 'recur (first last-form)))
          last-form))

      :else nil)))

(defn match-reduce-loop
  "Generic matcher for reduction loops.
	Returns a structural descriptor of the loop/update shape or nil."
  [loop-form]
  (when-let [normalized (normalize-loop loop-form)]
    (when (= :reduce-loop (:kind normalized))
      (let [{:keys [acc-sym acc-init index-sym body-form]} normalized
            [_ test then-branch else-branch] body-form
            recur-form (find-recur-form then-branch)]
        (when recur-form
          (let [recur-args (vec (rest recur-form))
                idx-update? (fn [expr]
                              (and (seq? expr)
                                   (descriptor/increment-op? (first expr))
                                   (= (second expr) index-sym)))
                [update-expr idx-update-expr]
                (cond
                  (and (= 2 (count recur-args))
                       (idx-update? (second recur-args)))
                  [(first recur-args) (second recur-args)]

                  (and (= 2 (count recur-args))
                       (idx-update? (first recur-args)))
                  [(second recur-args) (first recur-args)]

                  :else [nil nil])]
            (when update-expr
              {:acc-sym acc-sym
               :acc-init acc-init
               :index-sym index-sym
               :bound-expr (when (and (seq? test) (= 3 (count test)))
                             (nth test 2))
               :test-expr test
               :then-branch then-branch
               :else-expr else-branch
               :recur-form recur-form
               :idx-update-expr idx-update-expr
               :update-expr update-expr})))))))

(defn match-binary-reduce-loop
  "Generic matcher for simple binary reduction loops.
	valid-op? is a predicate over the reduction operator symbol.

	Returns a map with reduction structure or nil."
  [loop-form valid-op?]
  (when-let [{:keys [acc-sym acc-init index-sym update-expr bound-expr]} (match-reduce-loop loop-form)]
    (when (and (seq? update-expr)
               (>= (count update-expr) 3))
      (let [op (first update-expr)
            [arg1 arg2] (rest update-expr)]
        (when (valid-op? op)
          (let [aget-expr (cond
                            (and (acc-ref? arg1 acc-sym)
                                 (seq? arg2)
                                 (aget-form? arg2 index-sym))
                            arg2

                            (and (acc-ref? arg2 acc-sym)
                                 (seq? arg1)
                                 (aget-form? arg1 index-sym))
                            arg1

                            :else nil)]
            (when aget-expr
              {:op op
               :acc-sym acc-sym
               :acc-init acc-init
               :index-sym index-sym
               :array-sym (second aget-expr)
               :bound-expr bound-expr
               :aget-expr aget-expr
               :update-expr update-expr})))))))

(defn match-scan-loop
  "Generic matcher for accumulation loops that also write each intermediate
	accumulator state into an output array. Returns a structural descriptor or nil."
  [loop-form]
  (when-let [{:keys [acc-sym acc-init index-sym then-branch else-expr bound-expr update-expr]} (match-reduce-loop loop-form)]
    (when (form/binding-form? then-branch)
      (let [[_ let-bindings & let-body] then-branch]
        (when (and (vector? let-bindings)
                   (= 2 (count let-bindings))
                   (= 2 (count let-body)))
          (let [acc-next-sym (first let-bindings)
                acc-next-expr (second let-bindings)
                aset-form (first let-body)
                recur-form (second let-body)]
            (when (and (= update-expr acc-next-sym)
                       (seq? recur-form)
                       (= 'recur (first recur-form))
                       (= 3 (count recur-form))
                       (= acc-next-sym (nth recur-form 2))
                       (symbol? else-expr))
              (when-let [{:keys [out-sym cast-fn value-expr]} (match-aset-write aset-form index-sym)]
                (when (= value-expr acc-next-sym)
                  {:out-sym out-sym
                   :acc-sym acc-sym
                   :acc-init acc-init
                   :index-sym index-sym
                   :bound-expr bound-expr
                   :cast-fn cast-fn
                   :acc-next-sym acc-next-sym
                   :acc-next-expr acc-next-expr
                   :else-expr else-expr
                   :then-branch then-branch
                   :aset-form aset-form
                   :recur-form recur-form})))))))))