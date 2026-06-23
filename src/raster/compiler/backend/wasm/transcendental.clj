(ns raster.compiler.backend.wasm.transcendental
  "Polynomial / compositional lowerings for the elementary-math functions on the
   wasm backend (which has native opcodes only for sqrt/abs/min/max/floor/trunc/
   neg). Each `form` returns a Clojure IR form (f64, built from +,-,*,/,floor,
   sqrt,abs,min,max and value-position `if`) that the wasm emitter compiles INLINE
   via its normal emit-val path — reusing the whole emitter (incl. branching)
   instead of hand-writing bytecode. No bit operations are used anywhere.

   Bases (everything else composes from these):
     sin/cos — reduce to [-pi,pi] + degree-9 odd minimax.
     exp     — x/1024 + degree-6 Taylor + 10 squarings.
     log     — 12-sqrt reduction (m=x^(1/4096)) + atanh series.
     atan    — |x|<=1 minimax; |x|>1 via pi/2 - atan(1/x); odd in x.

   Compositions: tan=sin/cos; pow=exp(y·log x); log2/10=log·k; exp2/10=exp(x·k);
   expm1=exp-1; log1p=log(1+x); sinh/cosh/tanh from exp; asinh/acosh/atanh from
   log+sqrt; asin=atan(x/√(1-x²)); acos=π/2-asin; atan2=atan(y/x)+quadrant;
   cbrt=sign·exp(log|x|/3) (0 guarded); hypot=√(x²+y²); clamp=min(max…);
   ceil=-floor(-x); signum/copysign/flipsign via `if`.

   Accuracy (sampled vs java.lang.Math): sin/cos<5e-6, exp<1e-9 rel, log<1e-11,
   atan/asin/acos<3e-5, hyperbolics<1e-11. f32 args compute in f64 (promote in,
   demote out). Domain errors (log/pow/asin/acosh out of range) yield NaN/Inf as
   Math does. Not lowered (need ulp/bit ops or composite returns): round (int
   result), sincos (tuple), eps/nextfloat/prevfloat (ulp).")

;; sin minimax (x^1,3,5,7,9) fit on [-pi,pi]
(def ^:private S1 0.99998452052085140)
(def ^:private S3 -0.16663246572801355)
(def ^:private S5 0.0083123299197536420)
(def ^:private S7 -1.9315312848219816E-4)
(def ^:private S9 2.1727453521840316E-6)
;; atan minimax (x^1,3,5,7,9) fit on [-1,1]
(def ^:private A1 0.99988298437192560)
(def ^:private A3 -0.33059280040940203)
(def ^:private A5 0.18142102014086180)
(def ^:private A7 -0.087124349804662150)
(def ^:private A9 0.021841323968575240)

(def ^:private TAU 6.283185307179586)
(def ^:private INV-TAU 0.15915494309189535)
(def ^:private PI 3.141592653589793)
(def ^:private PI2 1.5707963267948966)
(def ^:private INV-1024 9.765625E-4)
(def ^:private INV-LN2 1.4426950408889634)
(def ^:private INV-LN10 0.4342944819032518)
(def ^:private LN2 0.6931471805599453)
(def ^:private LN10 2.302585092994046)
(def ^:private DEG2RAD 0.017453292519943295)
(def ^:private RAD2DEG 57.29577951308232)
(def ^:private THIRD (/ 1.0 3))

(defn- neg [x] (list '- 0.0 x))

(defn- horner [x coeffs]
  (reduce (fn [acc c] (list '+ c (list '* x acc)))
          (last coeffs) (reverse (butlast coeffs))))

(defn- sin-of [x]
  (list 'let* ['tr-sr  (list '- x (list '* (list 'floor (list '+ (list '* x INV-TAU) 0.5)) TAU))
               'tr-sr2 (list '* 'tr-sr 'tr-sr)]
        (list '* 'tr-sr (horner 'tr-sr2 [S1 S3 S5 S7 S9]))))

(defn- exp-of [arg]
  (let [taylor (horner 'tr-et [1.0 1.0 0.5 (/ 1.0 6) (/ 1.0 24) (/ 1.0 120) (/ 1.0 720)])
        nm     (mapv #(symbol (str "tr-ep" %)) (range 11))
        sqs    (mapcat (fn [i] [(nm i) (list '* (nm (dec i)) (nm (dec i)))]) (range 1 11))
        binds  (into ['tr-et (list '* arg INV-1024) (nm 0) taylor] sqs)]
    (list 'let* binds (nm 10))))

(defn- log-of [arg]
  (let [m  (mapv #(symbol (str "tr-lm" %)) (range 13))
        sq (mapcat (fn [i] [(m i) (list 'sqrt (m (dec i)))]) (range 1 13))
        binds (into [(m 0) arg]
                    (concat sq ['tr-lt  (list '/ (list '- (m 12) 1.0) (list '+ (m 12) 1.0))
                                'tr-lt2 (list '* 'tr-lt 'tr-lt)]))]
    (list 'let* binds
          (list '* (list '* 8192.0 'tr-lt)
                (list '+ 1.0 (list '* 'tr-lt2 (list '+ (/ 1.0 3) (list '* 'tr-lt2 (/ 1.0 5)))))))))

(defn- atan-of
  "atan of arg form: reduce |x|>1 via pi/2 - atan(1/x); odd in x; minimax on [0,1]."
  [arg]
  (list 'let* ['tr-ax  arg
               'tr-aa  (list 'abs 'tr-ax)
               'tr-az  (list 'if (list '> 'tr-aa 1.0) (list '/ 1.0 'tr-aa) 'tr-aa)
               'tr-az2 (list '* 'tr-az 'tr-az)
               'tr-ap  (list '* 'tr-az (horner 'tr-az2 [A1 A3 A5 A7 A9]))
               'tr-ar  (list 'if (list '> 'tr-aa 1.0) (list '- PI2 'tr-ap) 'tr-ap)]
        (list 'if (list '< 'tr-ax 0.0) (neg 'tr-ar) 'tr-ar)))

(defn form
  "IR form (f64) computing math op `k` applied to arg forms `args`, or nil if `k`
   has no wasm lowering. Each arg is bound once (no re-evaluation)."
  [k args]
  (let [a (first args) b (second args)]
    (case k
      :sin (list 'let* ['tr-x a] (sin-of 'tr-x))
      :cos (list 'let* ['tr-x a] (sin-of (list '+ 'tr-x PI2)))
      :tan (list 'let* ['tr-x a] (list '/ (sin-of 'tr-x) (sin-of (list '+ 'tr-x PI2))))
      :exp (exp-of a)
      :log (log-of a)
      :pow (list 'let* ['tr-pb a 'tr-py b] (exp-of (list '* 'tr-py (log-of 'tr-pb))))
      :fma (list '+ (list '* a b) (nth args 2))
      :atan (atan-of a)
      :asin (atan-of (list '/ a (list 'sqrt (list '- 1.0 (list '* a a)))))
      :acos (list '- PI2 (atan-of (list '/ a (list 'sqrt (list '- 1.0 (list '* a a))))))
      :atan2 (list 'let* ['tr-2y a 'tr-2x b 'tr-2r (atan-of (list '/ 'tr-2y 'tr-2x))]
                   (list 'if (list '> 'tr-2x 0.0) 'tr-2r
                         (list 'if (list '< 'tr-2x 0.0)
                               (list 'if (list '>= 'tr-2y 0.0) (list '+ 'tr-2r PI) (list '- 'tr-2r PI))
                               (list 'if (list '> 'tr-2y 0.0) PI2
                                     (list 'if (list '< 'tr-2y 0.0) (neg PI2) 0.0)))))
      :sinh (list 'let* ['tr-he (exp-of a)] (list '* 0.5 (list '- 'tr-he (list '/ 1.0 'tr-he))))
      :cosh (list 'let* ['tr-he (exp-of a)] (list '* 0.5 (list '+ 'tr-he (list '/ 1.0 'tr-he))))
      :tanh (list 'let* ['tr-he (exp-of (list '* 2.0 a))] (list '/ (list '- 'tr-he 1.0) (list '+ 'tr-he 1.0)))
      :asinh (list 'let* ['tr-hx a] (log-of (list '+ 'tr-hx (list 'sqrt (list '+ (list '* 'tr-hx 'tr-hx) 1.0)))))
      :acosh (list 'let* ['tr-hx a] (log-of (list '+ 'tr-hx (list 'sqrt (list '- (list '* 'tr-hx 'tr-hx) 1.0)))))
      :atanh (list 'let* ['tr-hx a] (list '* 0.5 (log-of (list '/ (list '+ 1.0 'tr-hx) (list '- 1.0 'tr-hx)))))
      :cbrt (list 'let* ['tr-cx a 'tr-ca (list 'abs 'tr-cx)]
                  (list 'if (list '> 'tr-ca 0.0)
                        (list 'let* ['tr-cr (exp-of (list '* THIRD (log-of 'tr-ca)))]
                              (list 'if (list '< 'tr-cx 0.0) (neg 'tr-cr) 'tr-cr))
                        0.0))
      :log2 (list '* (log-of a) INV-LN2)
      :log10 (list '* (log-of a) INV-LN10)
      :exp2 (exp-of (list '* a LN2))
      :exp10 (exp-of (list '* a LN10))
      :expm1 (list '- (exp-of a) 1.0)
      :log1p (log-of (list '+ 1.0 a))
      :hypot (list 'let* ['tr-hx a 'tr-hy b] (list 'sqrt (list '+ (list '* 'tr-hx 'tr-hx) (list '* 'tr-hy 'tr-hy))))
      :deg2rad (list '* a DEG2RAD)
      :rad2deg (list '* a RAD2DEG)
      :clamp (list 'min (list 'max a b) (nth args 2))
      :ceil (neg (list 'floor (neg a)))
      :signum (list 'let* ['tr-gx a]
                    (list 'if (list '> 'tr-gx 0.0) 1.0 (list 'if (list '< 'tr-gx 0.0) -1.0 0.0)))
      :copysign (list 'let* ['tr-cax (list 'abs a)] (list 'if (list '< b 0.0) (neg 'tr-cax) 'tr-cax))
      :flipsign (list 'let* ['tr-fx a] (list 'if (list '< b 0.0) (neg 'tr-fx) 'tr-fx))
      nil)))
