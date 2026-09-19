(ns raster.quant.ggml-kernels
  "GPU dot kernels with ggml's generic vec_dot arithmetic.

  Each work-item computes one `(row, output)` dot product the way
  ggml/src/ggml-cpu/quants.c's `ggml_vec_dot_*_generic` does for one row:
  integer sums are exact, and the float steps happen in the same order with the
  same operands, so the result is bit-identical to `raster.quant.ggml/vec-dot`
  without contraction. The inputs are `raster.quant.ggml/kernel-layout`
  decodings of ggml blocks: codes packed four int8 per int32 word in element
  order, FP16 scales as the floats they denote.

  The K-quant kernels keep ggml's eight float lanes: lane l accumulates element
  l of every 8-element chunk, and the lanes are summed in order at the end.
  They extract codes element by element; a faster lane-major packing must
  reproduce these kernels bit for bit."
  (:require [clojure.walk]
            [raster.core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.math :as math]
            [raster.numeric :as rn]
            [raster.par :as par]))

;; The K-quant kernels spell out ggml's eight lanes inline: a helper deftm taking
;; arrays leaves the typed route, whose emitter writes one operation per
;; statement and so leaves no expression for the OpenCL compiler to contract.
;; `lane-form` generates each lane's loop with plain, lane-unique locals, and
;; `k-quant-dot` splices them into a quoted kernel template.

(defn- s8-form
  "Form for signed byte `k` of int32 word `w`, bound to local `v`."
  [v w k]
  (list 'let [v (list 'bit-and (list 'bit-shift-right w (list '* 8 k)) 0xFF)]
        (list 'if (list '> v 127) (list '- v 256) v)))

(defn- u8-form
  "Form for unsigned byte `k` of int32 word `w`."
  [_ w k]
  (list 'bit-and (list 'bit-shift-right w (list '* 8 k)) 0xFF))

(defn- lane-form
  "Form for ggml's int32 lane `l` of one K-quant super-block: element l of each
  of the 32 eight-element chunks times the chunk's scale. `chunks-per-scale` is
  4 for q4_K (a 6-bit scale per 32 elements) and 2 for q6_K (an int8 scale per
  16). Reads the template's `wq`, `xq`, `wsc`, `ww`, `xw` and `scb`."
  [l weight-code chunks-per-scale]
  (let [n #(symbol (str % "-l" l))
        c (n "c") acc (n "acc") e (n "e") wv (n "wv") xv (n "xv") sc (n "sc")]
    (list 'loop [c 0 acc 0]
          (list 'if (list '< c 32)
                (list 'let [e (list '+ (list '* c 8) l)
                            wv (weight-code (n "wbyte") (list 'ra/aget 'wq (list '+ 'ww (list 'quot e 4)))
                                            (list 'rem e 4))
                            xv (s8-form (n "xbyte") (list 'ra/aget 'xq (list '+ 'xw (list 'quot e 4)))
                                        (list 'rem e 4))
                            sc (list 'ra/aget 'wsc (list '+ 'scb (list 'quot c chunks-per-scale)))]
                      (list 'recur (list 'inc c) (list '+ acc (list '* sc (list '* wv xv)))))
                acc))))

(defn- k-quant-dot
  "Splice lanes into `template`, replacing each `(lane l)` placeholder."
  [template weight-code chunks-per-scale]
  (clojure.walk/postwalk
   (fn [form]
     (if (and (seq? form) (= 'lane (first form)))
       (lane-form (second form) weight-code chunks-per-scale)
       form))
   template))

(deftm qdot-q8-0-rows!
  "q8_0 weights times q8_0 activations: ggml_vec_dot_q8_0_q8_0_generic,
  `sumf += sumi*(dw*dx)` per 32-element block."
  [xq :- (Array int), xd :- (Array float),
   wq :- (Array int), wd :- (Array float),
   y :- (Array float), in :- Long, out :- Long, nrows :- Long] :- Void
  (par/map-void! ro (* nrows out)
                 (let [row (quot ro out)
                       o (rem ro out)
                       nb (quot in 32)
                       acc (loop [b 0 sumf (float 0.0)]
                             (if (< b nb)
                               (let [wb (+ (* o nb) b)
                                     xb (+ (* row nb) b)
                                     sumi (loop [w 0 s 0]
                                            (if (< w 8)
                                              (recur (inc w) (par/dp4a (ra/aget wq (+ (* wb 8) w))
                                                                       (ra/aget xq (+ (* xb 8) w)) s))
                                              s))
                                     dd (* (ra/aget wd wb) (ra/aget xd xb))]
                                 (recur (inc b) (+ sumf (* (float sumi) dd))))
                               sumf))]
                   (ra/aset y ro acc))))

(deftm qdot-q5-0-rows!
  "q5_0 weights (codes q-16) times q8_0 activations:
  ggml_vec_dot_q5_0_q8_0_generic, `sumf += (dw*dx)*sumi` per block."
  [xq :- (Array int), xd :- (Array float),
   wq :- (Array int), wd :- (Array float),
   y :- (Array float), in :- Long, out :- Long, nrows :- Long] :- Void
  (par/map-void! ro (* nrows out)
                 (let [row (quot ro out)
                       o (rem ro out)
                       nb (quot in 32)
                       acc (loop [b 0 sumf (float 0.0)]
                             (if (< b nb)
                               (let [wb (+ (* o nb) b)
                                     xb (+ (* row nb) b)
                                     sumi (loop [w 0 s 0]
                                            (if (< w 8)
                                              (recur (inc w) (par/dp4a (ra/aget wq (+ (* wb 8) w))
                                                                       (ra/aget xq (+ (* xb 8) w)) s))
                                              s))
                                     dd (* (ra/aget wd wb) (ra/aget xd xb))]
                                 (recur (inc b) (+ sumf (* dd (float sumi)))))
                               sumf))]
                   (ra/aset y ro acc))))

(defmacro ^:private def-q4-K-dot []
  (k-quant-dot
   '(deftm qdot-q4-K-rows!
      "q4_K weights times q8_K activations: ggml_vec_dot_q4_K_q8_K_generic without
      contraction. Per super-block: eight exact int lanes, `sums[l] += (dw*dx)*lane[l]`,
      then `sumf -= (dmin*dx)*sumi` over the bsums and mins; finally the lanes are
      added to sumf in order."
      [xq :- (Array int), xd :- (Array float), xbs :- (Array int),
       wq :- (Array int), wd :- (Array float), wdmin :- (Array float),
       wsc :- (Array int), wm :- (Array int),
       y :- (Array float), in :- Long, out :- Long, nrows :- Long] :- Void
      (par/map-void! ro (* nrows out)
                     (let [row (quot ro out)
                           o (rem ro out)
                           nb (quot in 256)
                           acc
                           (loop [b 0 s0 (float 0.0) s1 (float 0.0) s2 (float 0.0) s3 (float 0.0)
                                  s4 (float 0.0) s5 (float 0.0) s6 (float 0.0) s7 (float 0.0)
                                  sumf (float 0.0)]
                             (if (< b nb)
                               (let [wb (+ (* o nb) b)
                                     xb (+ (* row nb) b)
                                     xw (* xb 64)
                                     ww (* wb 64)
                                     scb (* wb 8)
                                     d (* (ra/aget wd wb) (ra/aget xd xb))
                                     dmin (* (ra/aget wdmin wb) (ra/aget xd xb))
                                     sumi (loop [j 0 s 0]
                                            (if (< j 16)
                                              (recur (inc j) (+ s (* (ra/aget xbs (+ (* xb 16) j))
                                                                     (ra/aget wm (+ scb (quot j 2))))))
                                              s))
                                     l0 (lane 0) l1 (lane 1) l2 (lane 2) l3 (lane 3)
                                     l4 (lane 4) l5 (lane 5) l6 (lane 6) l7 (lane 7)
                                     ;; each product is its own statement, so no
                                     ;; expression is left to contract into an FMA
                                     p0 (* d (float l0)) p1 (* d (float l1))
                                     p2 (* d (float l2)) p3 (* d (float l3))
                                     p4 (* d (float l4)) p5 (* d (float l5))
                                     p6 (* d (float l6)) p7 (* d (float l7))
                                     pm (* dmin (float sumi))]
                                 (recur (inc b)
                                        (+ s0 p0) (+ s1 p1) (+ s2 p2) (+ s3 p3)
                                        (+ s4 p4) (+ s5 p5) (+ s6 p6) (+ s7 p7)
                                        (- sumf pm)))
                               (+ (+ (+ (+ (+ (+ (+ (+ sumf s0) s1) s2) s3) s4) s5) s6) s7)))]
                       (ra/aset y ro acc))))
   u8-form 4))

(def-q4-K-dot)

(defmacro ^:private def-q6-K-dot []
  (k-quant-dot
   '(deftm qdot-q6-K-rows!
      "q6_K weights (codes q-32) times q8_K activations: ggml_vec_dot_q6_K_q8_K_generic.
      Per super-block eight exact int lanes and `sums[l] += (dw*dx)*lane[l]`; the
      lanes are summed in order at the end."
      [xq :- (Array int), xd :- (Array float),
       wq :- (Array int), wd :- (Array float), wsc :- (Array int),
       y :- (Array float), in :- Long, out :- Long, nrows :- Long] :- Void
      (par/map-void! ro (* nrows out)
                     (let [row (quot ro out)
                           o (rem ro out)
                           nb (quot in 256)
                           acc
                           (loop [b 0 s0 (float 0.0) s1 (float 0.0) s2 (float 0.0) s3 (float 0.0)
                                  s4 (float 0.0) s5 (float 0.0) s6 (float 0.0) s7 (float 0.0)]
                             (if (< b nb)
                               (let [wb (+ (* o nb) b)
                                     xb (+ (* row nb) b)
                                     xw (* xb 64)
                                     ww (* wb 64)
                                     scb (* wb 16)
                                     d (* (ra/aget wd wb) (ra/aget xd xb))
                                     l0 (lane 0) l1 (lane 1) l2 (lane 2) l3 (lane 3)
                                     l4 (lane 4) l5 (lane 5) l6 (lane 6) l7 (lane 7)
                                     p0 (* d (float l0)) p1 (* d (float l1))
                                     p2 (* d (float l2)) p3 (* d (float l3))
                                     p4 (* d (float l4)) p5 (* d (float l5))
                                     p6 (* d (float l6)) p7 (* d (float l7))]
                                 (recur (inc b)
                                        (+ s0 p0) (+ s1 p1) (+ s2 p2) (+ s3 p3)
                                        (+ s4 p4) (+ s5 p5) (+ s6 p6) (+ s7 p7)))
                               (+ (+ (+ (+ (+ (+ (+ (+ (float 0.0) s0) s1) s2) s3) s4) s5) s6) s7)))]
                       (ra/aset y ro acc))))
   s8-form 2))

(def-q6-K-dot)

;; ---------------------------------------------------------------------------
;; Activation quantizers: ggml's quantize_row_q8_0_ref and quantize_row_q8_K_ref
;; ---------------------------------------------------------------------------
;;
;; Each scalar step is exact on the device. Division is correctly rounded
;; (ze-runtime/spirv-build-flags). `roundf` and ggml's `nearest_int` are built
;; from floor/ceil and the fraction they leave, which is exact in float. FP16
;; rounding of the q8_0 scale finds the value's binade by exact power-of-two
;; steps and rounds the value on FP16's grid there, ties to even.

(def ^:private exact-counter (atom 0))

(defn- local [prefix]
  (symbol (str prefix "-" (swap! exact-counter inc))))

(defn- rint-form
  "Round float `v` to the nearest integer, ties to even (ggml nearest_int),
  as a float; exact for |v| < 2^22."
  [v]
  (let [x (local "x") f (local "f") frac (local "frac")]
    (list 'let [x v
                f (list 'math/floor x)
                frac (list '- x f)]
          (list 'if (list '> frac (list 'float 0.5)) (list '+ f (list 'float 1.0))
                (list 'if (list '< frac (list 'float 0.5)) f
                      (list 'if (list '== 0 (list 'rem (list 'long f) 2)) f
                            (list '+ f (list 'float 1.0))))))))

(defn- roundf-form
  "C roundf of float `v`: nearest integer, halves away from zero, as a float."
  [v]
  (let [x (local "x") t (local "t") frac (local "frac")]
    (list 'let [x v
                t (list 'if (list '>= x (list 'float 0.0)) (list 'math/floor x) (list 'math/ceil x))
                frac (list '- x t)]
          (list 'if (list '>= frac (list 'float 0.5)) (list '+ t (list 'float 1.0))
                (list 'if (list '<= frac (list 'float -0.5)) (list '- t (list 'float 1.0)) t)))))

(defn- fp16-round-form
  "The float value of non-negative float `v` after GGML_FP32_TO_FP16 and back:
  round to FP16's 11 significant bits, ties to even, with FP16 subnormals
  (spacing 2^-24) and overflow to infinity from 65520 up. The binade of v is
  found by 160 exact doubling/halving steps toward 2^floor(log2 v), enough for
  every positive float; each loop has the typed route's one-carry shape."
  [v]
  (let [x (local "x") i (local "i") q (local "q") p (local "p") quantum (local "quantum")
        r (local "r")]
    (list 'let [x v]
          (list 'if (list '== x (list 'float 0.0))
                (list 'float 0.0)
                (list 'let [p (list 'loop [i 0 q (list 'float 1.0)]
                                    (list 'if (list '< i 160)
                                          (list 'recur (list 'inc i)
                                                (list 'if (list '<= (list '* q (list 'float 2.0)) x)
                                                      (list '* q (list 'float 2.0))
                                                      (list 'if (list '> q x)
                                                            (list '* q (list 'float 0.5))
                                                            q)))
                                          q))
                            ;; FP16 keeps 10 fraction bits; below 2^-14 the spacing is 2^-24
                            quantum (list 'if (list '< p (list 'float 6.1035156E-5))
                                          (list 'float 5.9604645E-8)
                                          (list '* p (list 'float 9.765625E-4)))
                            r (list '* (rint-form (list '/ x quantum)) quantum)]
                      (list 'if (list '>= r (list 'float 65520.0))
                            (list 'float Float/POSITIVE_INFINITY)
                            r))))))

(defn- word-form
  "Form storing the int32 word of four codes (element order) at `xq[index]`.
  `code` maps a float product to its integer code form."
  [index products code]
  (let [qs (mapv #(local (str "q" %)) (range 4))
        word (local "word")]
    (list 'let (vec (concat (mapcat (fn [q p] [q (list 'long (code p))]) qs products)
                            [word (list 'rn/bit-or
                                        (list 'rn/bit-or (list 'rn/bit-and (qs 0) 0xFF)
                                              (list 'rn/bit-shift-left (list 'rn/bit-and (qs 1) 0xFF) 8))
                                        (list 'rn/bit-or (list 'rn/bit-shift-left (list 'rn/bit-and (qs 2) 0xFF) 16)
                                              (list 'rn/bit-shift-left (list 'rn/bit-and (qs 3) 0xFF) 24)))]))
          ;; the word's bit pattern as a signed int32
          (list 'ra/aset 'xq index
                (list 'int (list 'if (list '>= word 2147483648) (list '- word 4294967296) word))))))

(defn- exact-forms
  "Replace `(rint v)`, `(roundf v)` and `(fp16-round v)` placeholders."
  [template]
  (clojure.walk/postwalk
   (fn [form]
     (if (seq? form)
       (case (first form)
         rint (rint-form (second form))
         roundf (roundf-form (second form))
         fp16-round (fp16-round-form (second form))
         form)
       form))
   template))

(defmacro ^:private def-q8-0-quantizer []
  (let [words (for [w (range 8)]
                (word-form (list '+ (list '* 'b 8) w)
                           (for [k (range 4)]
                             (list '* (list 'ra/aget 'x (list '+ 'base (+ (* 4 w) k))) 'id))
                           #(list 'roundf %)))]
    (exact-forms
     (list 'deftm 'quant-act-q8-0-rows!
           "quantize_row_q8_0_ref over rows of floats whose width is a multiple of 32: one
  work-item per 32-element block writes eight code words (element order) and the
  block's scale as the float its FP16 encoding denotes."
           '[x :- (Array float), xq :- (Array int), xd :- (Array float), nblocks :- Long]
           :- 'Void
           (list 'par/map-void! 'b 'nblocks
                 (concat
                  (list 'let '[base (* b 32)
                               amax (loop [j 0 m (float 0.0)]
                                      (if (< j 32)
                                        (let [v (ra/aget x (+ base j))
                                              a (if (< v (float 0.0)) (- (float 0.0) v) v)]
                                          (recur (inc j) (if (> m a) m a)))
                                        m))
                               d (/ amax (float 127.0))
                               id (if (== d (float 0.0)) (float 0.0) (/ (float 1.0) d))]
                        '(ra/aset xd b (fp16-round d)))
                  words))))))

(def-q8-0-quantizer)

(defmacro ^:private def-q8-K-quantizer []
  (let [words (for [w (range 4)]
                (word-form (list '+ (list '* 'b 64) (list '* 'g 4) w)
                           (for [k (range 4)]
                             (list '* 'iscale (list 'ra/aget 'x (list '+ 'gbase (+ (* 4 w) k)))))
                           #(list 'min 127 (list 'long (list 'rint %)))))]
    (exact-forms
     (list 'deftm 'quant-act-q8-K-rows!
           "quantize_row_q8_K_ref over rows of floats whose width is a multiple of 256: one
  work-item per 16-element group writes four code words (element order) and the
  group's code sum; group 0 of each block also writes the float scale 1/iscale,
  with iscale = -127/max and max the first element of largest magnitude."
           '[x :- (Array float), xq :- (Array int), xd :- (Array float), xbs :- (Array int),
             ngroups :- Long]
           :- 'Void
           (list 'par/map-void! 'bg 'ngroups
                 (concat
                  (list 'let '[b (quot bg 16)
                               g (rem bg 16)
                               base (* b 256)
                               gbase (+ base (* g 16))
                               amax (loop [j 0 m (float 0.0)]
                                      (if (< j 256)
                                        (let [v (ra/aget x (+ base j))
                                              a (if (< v (float 0.0)) (- (float 0.0) v) v)]
                                          (recur (inc j) (if (> m a) m a)))
                                        m))
                               ;; ggml keeps the first element of largest magnitude
                               first-max (loop [j 0 found -1]
                                           (if (< j 256)
                                             (let [v (ra/aget x (+ base j))
                                                   a (if (< v (float 0.0)) (- (float 0.0) v) v)]
                                               (recur (inc j) (if (< found 0) (if (== a amax) j found) found)))
                                             found))
                               mx (if (== amax (float 0.0)) (float 0.0) (ra/aget x (+ base first-max)))
                               iscale (if (== amax (float 0.0)) (float 0.0) (/ (float -127.0) mx))
                               sum (loop [j 0 s 0]
                                     (if (< j 16)
                                       (let [p (* iscale (ra/aget x (+ gbase j)))]
                                         (recur (inc j) (+ s (min 127 (long (rint p))))))
                                       s))]
                        '(ra/aset xbs bg (int sum))
                        '(when (== g 0)
                           (ra/aset xd b (if (== amax (float 0.0)) (float 0.0) (/ (float 1.0) iscale)))))
                  words))))))

(def-q8-K-quantizer)
