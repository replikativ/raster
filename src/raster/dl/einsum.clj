(ns raster.dl.einsum
  "Einstein summation and tensor rearrangement for Raster DL.

  einsum — Expresses tensor contractions, products, transposes, traces, etc.
  using Einstein summation notation:
    (einsum \"ij,jk->ik\" A B)    ; matrix multiply
    (einsum \"ij->ji\" A)          ; transpose
    (einsum \"ii->\" A)            ; trace
    (einsum \"i,j->ij\" a b)      ; outer product
    (einsum \"bij,bjk->bik\" A B) ; batch matmul

  rearrange — Reshape, transpose, merge, and split tensor dimensions
  using a pattern language inspired by einops:
    (rearrange A \"b c h w -> b (c h) w\")         ; merge dims
    (rearrange A \"b (c h) w -> b c h w\" {:c 3})  ; split dim
    (rearrange A \"b c h w -> b h w c\")            ; transpose

  All functions operate on flat typed arrays (double[], float[], etc.)
  with shapes tracked in the raster.dl.tensor shape registry."
  (:refer-clojure :exclude [aget aset alength aclone])
  (:require [raster.dl.tensor :as t]
            [raster.arrays :refer [aget aset alength aclone] :as ra]
            [raster.ad.templates :as tmpl]
            [raster.par :as par]
            [clojure.string :as str]))

;; ================================================================
;; einsum: parsing
;; ================================================================

(defn- parse-subscript
  "Parse an einsum subscript string like 'ij,jk->ik'.
  Returns {:inputs [[:i :j] [:j :k]] :output [:i :k]}.
  If no '->' is given, output is implicit (sum over all repeated indices)."
  [subscript]
  (let [subscript (str/trim subscript)]
    (if (str/includes? subscript "->")
      (let [[inputs-str output-str] (str/split subscript #"->" 2)
            input-parts (str/split (str/trim inputs-str) #"\s*,\s*")
            inputs (mapv (fn [s]
                           (mapv (fn [c] (keyword (str c)))
                                 (str/trim s)))
                         input-parts)
            output (if (str/blank? output-str)
                     []
                     (mapv (fn [c] (keyword (str c)))
                           (str/trim output-str)))]
        {:inputs inputs :output output})
      ;; Implicit mode: find indices that appear exactly once → output
      (let [input-parts (str/split subscript #"\s*,\s*")
            inputs (mapv (fn [s]
                           (mapv (fn [c] (keyword (str c)))
                                 (str/trim s)))
                         input-parts)
            all-indices (mapcat identity inputs)
            freqs (frequencies all-indices)
            output (vec (distinct (filter #(= 1 (freqs %)) all-indices)))]
        {:inputs inputs :output output}))))

(defn- build-dim-map
  "Build a map from index label to dimension size.
  Validates consistency across all inputs."
  [parsed tensors]
  (let [{:keys [inputs]} parsed]
    (assert (= (count inputs) (count tensors))
            (str "Expected " (count inputs) " tensors, got " (count tensors)))
    (reduce
     (fn [dim-map [labels tensor]]
       (let [shape (t/tshape tensor)]
         (assert shape (str "Tensor has no registered shape. Use (tensor data shape) to register."))
         (assert (= (count labels) (count shape))
                 (str "Subscript " labels " has " (count labels)
                      " indices but tensor has shape " shape " (" (count shape) " dims)"))
         (reduce
          (fn [dm [label size]]
            (if-let [existing (dm label)]
              (do (assert (= existing size)
                          (str "Inconsistent dimension for " label ": " existing " vs " size))
                  dm)
              (assoc dm label size)))
          dim-map
          (map vector labels shape))))
     {}
     (map vector inputs tensors))))

;; ================================================================
;; einsum: generic implementation
;; ================================================================

(defn- compute-strides
  "Compute strides for a list of dimension labels given the dim-map."
  [labels dim-map]
  (let [n (count labels)]
    (loop [i (dec n) s 1 acc (transient (vec (repeat n 0)))]
      (if (>= i 0)
        (let [size (dim-map (nth labels i))]
          (recur (dec i) (* s (long size)) (assoc! acc i s)))
        (persistent! acc)))))

(defn- flat-index
  "Compute flat index for a tensor given index assignments and the tensor's labels."
  ^long [labels strides index-assignments]
  (long (reduce (fn [^long acc ^long i]
                  (+ acc (* (long (index-assignments (nth labels i)))
                            (long (nth strides i)))))
                0
                (range (count labels)))))

(defn einsum
  "Einstein summation on tensors.

  subscript: String like \"ij,jk->ik\" (explicit output) or \"ij,ji\" (implicit).
  tensors: The input tensors (flat double[] with shape registered via tensor).

  Examples:
    (einsum \"ij,jk->ik\" A B)     ; matmul
    (einsum \"ij->ji\" A)           ; transpose
    (einsum \"ii->\" A)             ; trace (returns scalar tensor)
    (einsum \"i,j->ij\" a b)       ; outer product
    (einsum \"bij,bjk->bik\" A B)  ; batch matmul
    (einsum \"ij,ij->\" A B)       ; Frobenius inner product
    (einsum \"ijk->j\" A)          ; sum over dims i,k"
  [subscript & tensors]
  (let [parsed (parse-subscript subscript)
        {:keys [inputs output]} parsed
        dim-map (build-dim-map parsed tensors)
        ;; All unique indices
        all-labels (vec (distinct (mapcat identity inputs)))
        ;; Contraction indices: in inputs but not in output
        output-set (set output)
        contract-labels (vec (remove output-set all-labels))
        ;; Output shape
        output-shape (mapv dim-map output)
        output-size (reduce * 1 output-shape)
        result (ra/alloc-like (first tensors) output-size)
        ;; Precompute strides for each input tensor
        input-strides (mapv #(compute-strides % dim-map) inputs)
        output-strides (compute-strides output dim-map)
        ;; Contraction dimensions and their sizes
        contract-sizes (mapv dim-map contract-labels)
        contract-total (reduce * 1 contract-sizes)
        ;; Output dimensions and their sizes
        output-sizes (mapv dim-map output)
        output-total output-size
        ;; Precompute: for each contraction flat index, what are the individual indices?
        contract-n (count contract-labels)
        output-n (count output)
        ;; Convert tensors to array for fast access
        tensor-arr (object-array tensors)]
    ;; Iterate over all output positions
    (dotimes [out-flat output-total]
      ;; Decompose out-flat into output index values
      (let [out-indices (loop [rem out-flat i 0 acc (transient {})]
                          (if (< i output-n)
                            (let [stride (long (nth output-strides i))
                                  idx (quot rem stride)
                                  rem (mod rem stride)]
                              (recur rem (inc i) (assoc! acc (nth output i) idx)))
                            (persistent! acc)))
            ;; Sum over all contraction index combinations
            sum (if (empty? contract-labels)
                  ;; No contraction — just product of input elements
                  (reduce
                   (fn [^double prod ^long ti]
                     (let [tensor (clojure.core/aget tensor-arr ti)
                           fi (flat-index (nth inputs ti) (nth input-strides ti) out-indices)]
                       (* prod (double (aget tensor fi)))))
                   1.0
                   (range (count inputs)))
                  ;; Contraction: iterate over all contraction combos
                  (loop [c-flat 0 sum 0.0]
                    (if (< c-flat contract-total)
                      ;; Decompose c-flat into contraction index values
                      (let [all-indices
                            (loop [rem c-flat ci 0 acc (transient out-indices)]
                              (if (< ci contract-n)
                                (let [size (long (nth contract-sizes ci))
                                      ;; compute stride for this contraction dim
                                      stride (loop [j (inc ci) s 1]
                                               (if (< j contract-n)
                                                 (recur (inc j) (* s (long (nth contract-sizes j))))
                                                 s))
                                      idx (quot rem stride)
                                      rem (mod rem stride)]
                                  (recur rem (inc ci) (assoc! acc (nth contract-labels ci) idx)))
                                (persistent! acc)))
                            ;; Product of input elements at these indices
                            prod (reduce
                                  (fn [^double p ^long ti]
                                    (let [tensor (clojure.core/aget tensor-arr ti)
                                          fi (flat-index (nth inputs ti) (nth input-strides ti) all-indices)]
                                      (* p (double (aget tensor fi)))))
                                  1.0
                                  (range (count inputs)))]
                        (recur (inc c-flat) (+ sum prod)))
                      sum)))]
        (aset result out-flat sum)))
    (t/register-tensor result output-shape)))

;; ================================================================
;; rearrange: parsing
;; ================================================================

(defn- parse-rearrange-side
  "Parse one side of a rearrange pattern (e.g., 'b c h w' or 'b (c h) w').
  Returns a list of tokens, where each token is either:
    :name — a simple axis name (keyword)
    [:group [:name1 :name2 ...]] — a grouped axis"
  [s]
  (let [s (str/trim s)]
    (loop [i 0 tokens []]
      (if (>= i (count s))
        tokens
        (let [c (nth s i)]
          (cond
            (Character/isWhitespace c)
            (recur (inc i) tokens)

            (= c \()
            ;; Parse group until ')'
            (let [close (.indexOf s ")" i)]
              (assert (>= close 0) (str "Unmatched '(' in pattern: " s))
              (let [inner (subs s (inc i) close)
                    names (mapv (comp keyword str/trim)
                                (str/split (str/trim inner) #"\s+"))]
                (recur (inc close) (conj tokens [:group names]))))

            (Character/isLetterOrDigit c)
            ;; Parse name
            (let [end (loop [j (inc i)]
                        (if (and (< j (count s))
                                 (let [ch (nth s j)]
                                   (or (Character/isLetterOrDigit ch) (= ch \_))))
                          (recur (inc j))
                          j))
                  name (keyword (subs s i end))]
              (recur end (conj tokens name)))

            :else
            (throw (ex-info (str "Unexpected character '" c "' in pattern: " s)
                            {:pattern s :pos i}))))))))

(defn- parse-rearrange-pattern
  "Parse a full rearrange pattern 'left -> right'.
  Returns {:left [...] :right [...]}."
  [pattern]
  (let [[left-str right-str] (str/split pattern #"\s*->\s*" 2)]
    (assert right-str (str "Rearrange pattern must contain '->': " pattern))
    {:left (parse-rearrange-side left-str)
     :right (parse-rearrange-side right-str)}))

(defn- extract-names
  "Extract all axis names from a parsed side."
  [side]
  (mapcat (fn [tok]
            (if (keyword? tok)
              [tok]
              (second tok)))  ;; [:group [...names...]]
          side))

(defn- resolve-axis-sizes
  "Given the left side tokens and the input shape, resolve each axis name to its size.
  For grouped axes like (c h), if axis_lengths provides some sizes, infer the rest.
  Returns a map of axis-name -> size."
  [left-tokens input-shape axis-lengths]
  (assert (= (count left-tokens) (count input-shape))
          (str "Pattern has " (count left-tokens) " axes but tensor has "
               (count input-shape) " dims. Pattern: " left-tokens " Shape: " input-shape))
  (reduce
   (fn [sizes [tok dim-size]]
     (if (keyword? tok)
       (do (when-let [provided (get axis-lengths tok)]
             (assert (= provided dim-size)
                     (str "Axis " tok " has size " dim-size " but axis_lengths specifies " provided)))
           (assoc sizes tok dim-size))
       ;; Group: (c h) with combined dim-size
       (let [[_ names] tok
             ;; Check which names have known sizes
             known (reduce (fn [m nm]
                             (if-let [s (get axis-lengths nm)]
                               (assoc m nm s)
                               m))
                           {} names)
             n-unknown (- (count names) (count known))]
         (assert (<= n-unknown 1)
                 (str "Cannot infer sizes for grouped axis " names
                      " — at most 1 unknown allowed. Known: " known
                      ", total size: " dim-size))
         (if (zero? n-unknown)
           ;; All known — verify they multiply to dim-size
           (let [product (reduce * 1 (vals known))]
             (assert (= product dim-size)
                     (str "Grouped axis " names " sizes " known
                          " multiply to " product " but dim is " dim-size))
             (merge sizes known))
           ;; One unknown — infer it
           (let [product-known (reduce * 1 (vals known))
                 _ (assert (zero? (mod dim-size product-known))
                           (str "Cannot split " dim-size " evenly with known " known))
                 inferred-size (quot dim-size product-known)
                 unknown-name (first (remove (set (keys known)) names))]
             (merge sizes known {unknown-name inferred-size}))))))
   {}
   (map vector left-tokens input-shape)))

;; ================================================================
;; rearrange: execution
;; ================================================================

(defn rearrange
  "Rearrange a tensor's dimensions using a pattern string.

  pattern: String like 'b c h w -> b (c h) w'
  tensor: flat double[] with shape registered
  axis-lengths: optional map of axis-name (keyword) -> size for splitting

  Operations supported:
    Transpose:  'b c h w -> b h w c'
    Merge:      'b c h w -> b (c h w)'
    Split:      'b (c h) w -> b c h w'   (needs {:c 3} or {:h N} in axis-lengths)
    Reorder:    'h w c -> c h w'

  Examples:
    (rearrange img \"b c h w -> b h w c\")
    (rearrange img \"b c (h p1) (w p2) -> b (h w) (p1 p2 c)\" {:p1 2 :p2 2})
    (rearrange x \"b (h n) -> b h n\" {:n 4})"
  ([tensor pattern]
   (rearrange tensor pattern {}))
  ([tensor pattern axis-lengths]
   (let [{:keys [left right]} (parse-rearrange-pattern pattern)
         input-shape (t/tshape tensor)
         _ (assert input-shape "Tensor has no registered shape")
         ;; Resolve all axis sizes
         sizes (resolve-axis-sizes left input-shape axis-lengths)
         ;; Compute output shape from right side
         output-shape (mapv (fn [tok]
                              (if (keyword? tok)
                                (do (assert (contains? sizes tok)
                                            (str "Unknown axis " tok " in output pattern"))
                                    (sizes tok))
                                ;; Group: multiply sizes
                                (let [[_ names] tok]
                                  (reduce (fn [^long acc nm]
                                            (assert (contains? sizes nm)
                                                    (str "Unknown axis " nm " in output pattern"))
                                            (* acc (long (sizes nm))))
                                          1 names))))
                            right)
         ;; Build ordered list of all axis names on left side (expanded)
         left-names (vec (extract-names left))
         right-names (vec (extract-names right))
         ;; Verify same set of names
         _ (assert (= (set left-names) (set right-names))
                   (str "Left and right patterns must use the same axis names. "
                        "Left: " left-names " Right: " right-names))
         ;; Compute: for each output flat index → input flat index
         ;; We need the permutation from right-names order to left-names order
         ;; Axis order in the flattened input: left-names order
         ;; Axis order in the flattened output: right-names order
         n-axes (count left-names)
         left-sizes (mapv sizes left-names)
         right-sizes (mapv sizes right-names)
         ;; Strides for input (left-names order)
         left-strides (t/strides left-sizes)
         ;; Strides for output (right-names order)
         right-strides (t/strides right-sizes)
         ;; Map from axis name to its left-stride
         left-stride-map (zipmap left-names left-strides)
         ;; For each right-axis position, what is the corresponding left-stride?
         ;; This allows us to decompose output flat index → input flat index efficiently
         total (reduce * 1 right-sizes)
         result (ra/alloc-like tensor total)]
     ;; Iterate over all output positions
     (dotimes [out-idx total]
       ;; Decompose out-idx into per-axis indices using right-strides
       (let [in-idx (loop [rem out-idx ri 0 in-flat 0]
                      (if (< ri n-axes)
                        (let [stride (long (nth right-strides ri))
                              axis-idx (quot rem stride)
                              rem (mod rem stride)
                              ;; This axis in the right is right-names[ri]
                              ;; Its stride in the left layout is left-stride-map[right-names[ri]]
                              left-stride (long (left-stride-map (nth right-names ri)))]
                          (recur rem (inc ri) (+ in-flat (* axis-idx left-stride))))
                        in-flat))]
         (aset result out-idx (aget tensor in-idx))))
     (t/register-tensor result output-shape))))

;; ================================================================
;; AD rrules
;; ================================================================

(defn- labels->str
  "Convert a vector of keyword labels to a subscript string, e.g. [:i :j] -> \"ij\"."
  [labels]
  (apply str (map name labels)))

(defn- build-grad-subscript
  "Build the einsum subscript for ∂L/∂T_p.
  Given original parsed einsum {inputs output}, the gradient w.r.t. input p is:
    einsum(output-labels, other-input-labels... -> input_p-labels, dy, other-tensors...)"
  [parsed p]
  (let [{:keys [inputs output]} parsed
        ;; Input subscripts: dy uses output labels, other tensors keep their labels
        grad-input-subs (into [(labels->str output)]
                              (keep-indexed (fn [i s] (when (not= i p) (labels->str s))))
                              inputs)
        grad-output-sub (labels->str (nth inputs p))]
    (str (str/join "," grad-input-subs) "->" grad-output-sub)))

(defn- has-repeated-labels?
  "True if a label vector has any repeated labels (e.g. [:i :i])."
  [labels]
  (not= (count labels) (count (set labels))))

(defn- einsum-grad-fallback
  "Compute the gradient of einsum w.r.t. input p using elementwise finite
  computation when the gradient subscript has repeated output indices.
  This handles trace-like operations where the standard einsum approach fails."
  [parsed p tensors dy dim-map]
  (let [{:keys [inputs output]} parsed
        input-labels (nth inputs p)
        input-shape (mapv dim-map input-labels)
        input-size (reduce * 1 input-shape)
        input-strides (compute-strides input-labels dim-map)
        ;; We compute dT_p by iterating over all output positions and
        ;; for each, computing the contribution to each input_p position.
        ;; For trace: output is scalar, so we iterate contractions only.
        ;; General formula: dT_p[idx_p] = sum_{other indices} dy[out_idx] * prod_{q!=p} T_q[idx_q]
        output-strides (compute-strides output dim-map)
        all-labels (vec (distinct (mapcat identity inputs)))
        output-set (set output)
        ;; The input_p indices + contraction indices + output indices are the iteration space
        ;; We need to iterate over all index combinations that include input_p's indices
        ;; and sum over the rest. That's: all indices except input_p's unique ones kept free.
        ;; Actually: iterate over ALL unique indices, for each combo compute contribution.
        all-sizes (mapv dim-map all-labels)
        all-total (reduce * 1 all-sizes)
        all-n (count all-labels)
        grad (ra/alloc-like (nth tensors 0) input-size)
        other-inputs (keep-indexed (fn [i _] (when (not= i p) i)) (range (count tensors)))
        other-strides (mapv #(compute-strides (nth inputs %) dim-map) (range (count tensors)))]
    (dotimes [flat all-total]
      (let [idx-map (loop [rem flat ai 0 acc (transient {})]
                      (if (< ai all-n)
                        (let [stride (loop [j (inc ai) s 1]
                                       (if (< j all-n)
                                         (recur (inc j) (* s (long (nth all-sizes j))))
                                         s))
                              idx (quot rem stride)
                              rem (mod rem stride)]
                          (recur rem (inc ai) (assoc! acc (nth all-labels ai) idx)))
                        (persistent! acc)))
            ;; Output index for dy
            out-flat (if (empty? output)
                       0
                       (long (reduce (fn [^long acc ^long oi]
                                       (+ acc (* (long (idx-map (nth output oi)))
                                                 (long (nth output-strides oi)))))
                                     0 (range (count output)))))
            ;; Input_p flat index
            in-flat (long (reduce (fn [^long acc ^long ii]
                                    (+ acc (* (long (idx-map (nth input-labels ii)))
                                              (long (nth input-strides ii)))))
                                  0 (range (count input-labels))))
            ;; Product of other inputs at this index combo
            prod (reduce (fn [^double p qi]
                           (let [q-labels (nth inputs qi)
                                 q-strides (nth other-strides qi)
                                 q-flat (long (reduce (fn [^long a ^long li]
                                                        (+ a (* (long (idx-map (nth q-labels li)))
                                                                (long (nth q-strides li)))))
                                                      0 (range (count q-labels))))]
                             (* p (double (aget (nth tensors qi) q-flat)))))
                         1.0
                         other-inputs)
            dy-val (double (aget dy out-flat))]
        (aset grad in-flat (+ (double (aget grad in-flat)) (* dy-val prod)))))
    (t/register-tensor grad input-shape)))

;; einsum rrule: ∂L/∂T_p = einsum(dy + other tensors → T_p's subscript)
;; Falls back to manual computation for repeated-index outputs (e.g. trace)
(tmpl/merge-into-template! 'raster.dl.einsum/einsum
                           {:pullback-factory (fn [_result & args]
                                                (let [subscript (first args)
                                                      tensors (rest args)
                                                      parsed (parse-subscript subscript)
                                                      n (count tensors)
                                                      dim-map (build-dim-map parsed tensors)]
                                                  (fn [dy]
                                                    (into [nil] ;; nil gradient for the subscript string
                                                          (map (fn [p]
                                                                 (let [input-labels (nth (:inputs parsed) p)]
                                                                   (if (has-repeated-labels? input-labels)
                                            ;; Repeated indices in grad output (e.g. trace) — use fallback
                                                                     (einsum-grad-fallback parsed p tensors dy dim-map)
                                            ;; Standard case: gradient is itself an einsum
                                                                     (let [grad-sub (build-grad-subscript parsed p)
                                                                           other-tensors (keep-indexed
                                                                                          (fn [i t] (when (not= i p) t))
                                                                                          tensors)
                                                                           grad-inputs (into [dy] other-tensors)]
                                                                       (apply einsum grad-sub grad-inputs))))))
                                                          (range n)))))})

;; rearrange rrule: backward = rearrange with swapped pattern
(defn- unparse-side
  "Convert parsed rearrange tokens back to a pattern string."
  [tokens]
  (str/join " " (map (fn [tok]
                       (if (keyword? tok)
                         (name tok)
                         (str "(" (str/join " " (map name (second tok))) ")")))
                     tokens)))

(tmpl/merge-into-template! 'raster.dl.einsum/rearrange
                           {:pullback-factory (fn [_result & args]
                                                (let [tensor (first args)
                                                      pattern (second args)
                                                      axis-lengths (nth args 2 {})
                                                      {:keys [left right]} (parse-rearrange-pattern pattern)
                                                      input-shape (t/tshape tensor)
                             ;; Resolve all axis sizes from the forward pass
                                                      sizes (resolve-axis-sizes left input-shape axis-lengths)
                             ;; Inverse pattern: swap left and right
                                                      inv-pattern (str (unparse-side right) " -> " (unparse-side left))]
                                                  (fn [dy]
                           ;; The inverse rearrange has left=original-right, right=original-left.
                           ;; Groups on the inverse LEFT are resolved from dy's shape (no problem).
                           ;; Groups on the inverse RIGHT (= original left) need axis-lengths
                           ;; so resolve-axis-sizes can split them.
                           ;; Provide all resolved sizes — extra entries are harmless.
                                                    [(rearrange dy inv-pattern sizes) nil nil])))})
