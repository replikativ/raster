(ns raster.compiler.backend.wasm.transcendental
  "Polynomial lowerings for transcendentals on the wasm backend (which has no
   native sin/cos/exp opcode — see encoder.clj). Each entry returns a Clojure IR
   form (f64, built from +,-,*,/,floor and double constants) that the wasm emitter
   compiles INLINE via its normal emit-val path — so we reuse the whole emitter and
   its type machinery instead of hand-writing bytecode.

   Accuracy (verified by least-squares fit + sampling, see git history):
     sin/cos — floor-based argument reduction to [-pi,pi] + a degree-9 odd minimax
               fit on [-pi,pi]: < 2e-5 absolute, everywhere (reduction is periodic).
     exp     — x/1024 + degree-6 Taylor + 10 squarings (multiplies only, no bit
               ops): < 1e-9 relative for |x| <~ 100, degrading gracefully beyond.
     tan     — sin/cos.

   log/pow are intentionally NOT lowered here: correct range reduction needs
   exponent-bit extraction (i64 reinterpret + shift) which encoder.clj lacks. The
   registry leaves their :wasm facet absent, so they surface a clear error.")

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
  "sin of a value symbol `x`: reduce to [-pi,pi] then odd Horner poly."
  [x]
  (list 'let* ['tr-r  (list '- x (list '* (list 'floor (list '+ (list '* x INV-TAU) 0.5)) TAU))
               'tr-r2 (list '* 'tr-r 'tr-r)]
        (list '* 'tr-r (horner 'tr-r2 [S1 S3 S5 S7 S9]))))

(defn- cos-of [x] (sin-of (list '+ x HALF-PI)))

(defn- exp-form
  "exp of an arg form: t=arg/1024, degree-6 Taylor, then square 10×."
  [arg]
  (let [taylor (horner 'tr-t [1.0 1.0 0.5 (/ 1.0 6) (/ 1.0 24) (/ 1.0 120) (/ 1.0 720)])
        nm     (mapv #(symbol (str "tr-p" %)) (range 11))      ; tr-p0 … tr-p10
        sqs    (mapcat (fn [i] [(nm i) (list '* (nm (dec i)) (nm (dec i)))]) (range 1 11))
        binds  (into ['tr-t (list '* arg INV-1024) (nm 0) taylor] sqs)]
    (list 'let* binds (nm 10))))

(defn form
  "IR form (f64) computing transcendental `k` applied to the arg form, or nil if
   `k` has no polynomial lowering. The arg is bound once (no re-evaluation)."
  [k arg]
  (case k
    :sin (list 'let* ['tr-x arg] (sin-of 'tr-x))
    :cos (list 'let* ['tr-x arg] (cos-of 'tr-x))
    :tan (list 'let* ['tr-x arg] (list '/ (sin-of 'tr-x) (cos-of 'tr-x)))
    :exp (exp-form arg)
    nil))
