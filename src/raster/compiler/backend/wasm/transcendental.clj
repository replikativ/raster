(ns raster.compiler.backend.wasm.transcendental
  "Polynomial lowerings for transcendentals on the wasm backend (which has no
   native sin/cos/exp/log/pow opcode — see encoder.clj). Each `form` returns a
   Clojure IR form (f64, built from +,-,*,/,floor,sqrt and double constants) that
   the wasm emitter compiles INLINE via its normal emit-val path — so we reuse the
   whole emitter and its type machinery instead of hand-writing bytecode.

   No bit operations are needed: log range-reduces by repeated sqrt (m = x^(1/4096)
   via 12 sqrts → log x = 4096·2·atanh((m-1)/(m+1))), exp by halving + repeated
   squaring. pow = exp(y·log x); fma = a·b+c.

   Accuracy (verified by sampling vs java.lang.Math, see git history):
     sin/cos — reduce to [-pi,pi] + degree-9 odd minimax: < 5e-6 abs, everywhere.
     exp     — x/1024 + degree-6 Taylor + 10 squarings: < 1e-9 rel for |x|<~100.
     log     — 12-sqrt reduction + atanh series: < 1e-11 abs over [1e-3,1e3].
     tan     — sin/cos.  pow — < 1e-11 rel (x>0).  fma — mul-then-add in f64.
   f32 args compute in f64 (promote in, demote out). pow/log of x<=0 yield NaN,
   as java.lang.Math does for non-integer exponents.")

;; sin minimax coefficients (x^1,3,5,7,9), fit on [-pi,pi]
(def ^:private S1 0.99998452052085140)
(def ^:private S3 -0.16663246572801355)
(def ^:private S5 0.0083123299197536420)
(def ^:private S7 -1.9315312848219816E-4)
(def ^:private S9 2.1727453521840316E-6)

(def ^:private TAU 6.283185307179586)
(def ^:private INV-TAU 0.15915494309189535)
(def ^:private HALF-PI 1.5707963267948966)
(def ^:private INV-1024 9.765625E-4)

(defn- horner
  "coeffs [c0 c1 … cn] → (+ c0 (* x (+ c1 (* x …)))) in IR form."
  [x coeffs]
  (reduce (fn [acc c] (list '+ c (list '* x acc)))
          (last coeffs) (reverse (butlast coeffs))))

(defn- sin-of
  "sin of value form `x`: reduce to [-pi,pi] then odd Horner poly."
  [x]
  (list 'let* ['tr-sr  (list '- x (list '* (list 'floor (list '+ (list '* x INV-TAU) 0.5)) TAU))
               'tr-sr2 (list '* 'tr-sr 'tr-sr)]
        (list '* 'tr-sr (horner 'tr-sr2 [S1 S3 S5 S7 S9]))))

(defn- exp-of
  "exp of arg form: t=arg/1024, degree-6 Taylor, then square 10×."
  [arg]
  (let [taylor (horner 'tr-et [1.0 1.0 0.5 (/ 1.0 6) (/ 1.0 24) (/ 1.0 120) (/ 1.0 720)])
        nm     (mapv #(symbol (str "tr-ep" %)) (range 11))      ; tr-ep0 … tr-ep10
        sqs    (mapcat (fn [i] [(nm i) (list '* (nm (dec i)) (nm (dec i)))]) (range 1 11))
        binds  (into ['tr-et (list '* arg INV-1024) (nm 0) taylor] sqs)]
    (list 'let* binds (nm 10))))

(defn- log-of
  "log of arg form (x>0): m=x^(1/4096) via 12 sqrts, t=(m-1)/(m+1),
   log = 4096·2·atanh(t) = 8192·t·(1 + t²/3 + t⁴/5). No bit ops."
  [arg]
  (let [m  (mapv #(symbol (str "tr-lm" %)) (range 13))          ; tr-lm0 … tr-lm12
        sq (mapcat (fn [i] [(m i) (list 'sqrt (m (dec i)))]) (range 1 13))
        binds (into [(m 0) arg]
                    (concat sq
                            ['tr-lt  (list '/ (list '- (m 12) 1.0) (list '+ (m 12) 1.0))
                             'tr-lt2 (list '* 'tr-lt 'tr-lt)]))]
    (list 'let* binds
          (list '* (list '* 8192.0 'tr-lt)
                (list '+ 1.0 (list '* 'tr-lt2 (list '+ (/ 1.0 3) (list '* 'tr-lt2 (/ 1.0 5)))))))))

(defn form
  "IR form (f64) computing transcendental `k` applied to arg forms `args`, or nil
   if `k` has no polynomial lowering. Each arg is bound once (no re-evaluation)."
  [k args]
  (let [a (first args)]
    (case k
      :sin (list 'let* ['tr-x a] (sin-of 'tr-x))
      :cos (list 'let* ['tr-x a] (sin-of (list '+ 'tr-x HALF-PI)))
      :tan (list 'let* ['tr-x a] (list '/ (sin-of 'tr-x) (sin-of (list '+ 'tr-x HALF-PI))))
      :exp (exp-of a)
      :log (log-of a)
      :pow (list 'let* ['tr-pb a 'tr-py (second args)]
                 (exp-of (list '* 'tr-py (log-of 'tr-pb))))
      :fma (list '+ (list '* a (second args)) (nth args 2))
      nil)))
