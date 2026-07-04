(ns raster.ad.laws-test
  "AD laws suite — the executable form of the commuting-diagram obligations in
  .internal/ad_formal_framework.md (§9 O1–O6, §12 O7/O8 + fan-out efficiency).

  Each deftest is one law family, property-style over a small corpus of deftm
  functions. FD (central differences) is the external referee for O1/O4/O7
  (ChainRulesTestUtils / JAX check_grads methodology); O7 needs no FD oracle
  for the pairing itself; O8 checks pullback linearity directly against the
  tangent-algebra axioms.

  Tolerances are per-dtype (double vs FD: 1e-8 scaled; float composites: 1e-3),
  and sample points are chosen away from kinks (no oracle exists at
  nondifferentiable points — §12)."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.math :as m]
            [raster.arrays :as ra]
            [raster.ad.reverse :as rev]
            [raster.ad.tangent :as tangent]
            [raster.ad.forward :as fwd]
            [raster.ad.jet :as jet]
            [raster.ad.templates :as tmpl]
            [raster.sym.core :as sym]
            [raster.sym.diff :as sdiff]
            [raster.sci.special :as special]
            [raster.dl.nn :as nn]
            [raster.dl.loss :as loss]))

;; ================================================================
;; Tolerances (per-dtype, scaled) and FD referee
;; ================================================================

(def ^:private tol-double 1e-8)
(def ^:private tol-float 1e-3)

(defn- close?
  "Scaled comparison: |a-b| <= tol * max(1, |a|, |b|). Relative for large
  magnitudes (Rosenbrock grads ~1e2), absolute near zero."
  [a b tol]
  (<= (Math/abs (- (double a) (double b)))
      (* tol (max 1.0 (Math/abs (double a)) (Math/abs (double b))))))

(defn- central-fd
  "Central finite difference of scalar f at x (double)."
  ([f x] (central-fd f x 1e-5))
  ([f x h]
   (/ (- (double (f (+ x h))) (double (f (- x h)))) (* 2.0 h))))

(defn- fa
  "Deterministic Gaussian float array."
  ^floats [n seed]
  (let [r (java.util.Random. (long seed)) a (float-array n)]
    (dotimes [i n] (aset a i (float (.nextGaussian r)))) a))

(defn- fdot
  "Double-precision dot product of two float arrays."
  ^double [^floats a ^floats b]
  (let [len (alength a)]
    (loop [i 0 s 0.0]
      (if (< i len)
        (recur (inc i) (+ s (* (double (aget a i)) (double (aget b i)))))
        s))))

(defn- faxpy
  "New float array a + h*v."
  ^floats [^floats a ^double h ^floats v]
  (let [len (alength a) out (float-array len)]
    (dotimes [i len] (aset out i (float (+ (aget a i) (* h (aget v i))))))
    out))

(defn- fadd ^floats [^floats a ^floats b]
  (let [len (alength a) out (float-array len)]
    (dotimes [i len] (aset out i (float (+ (aget a i) (aget b i))))) out))

(defn- fscale ^floats [^double c ^floats a]
  (let [len (alength a) out (float-array len)]
    (dotimes [i len] (aset out i (float (* c (aget a i))))) out))

(defn- farr-close? [^floats a ^floats b tol]
  (and (= (alength a) (alength b))
       (every? true? (map #(close? %1 %2 tol) a b))))

;; ================================================================
;; Corpus — scalar fns (unique laws-* names; qualified raster ops so
;; both the reverse transform AND the Dual carrier can consume them)
;; ================================================================

;; polynomial: 3x^3 - 2x^2 + x - 5
(deftm laws-f1 [x :- Double] :- Double
  (n/+ (n/- (n/* (n/* 3.0 x) (n/* x x)) (n/* 2.0 (n/* x x)))
       (n/- x 5.0)))

(defn- laws-f1' [x] (+ (- (* 9.0 x x) (* 4.0 x)) 1.0))

;; x*sin(x) + exp(x)
(deftm laws-f2 [x :- Double] :- Double
  (n/+ (n/* x (m/sin x)) (m/exp x)))

(defn- laws-f2' [x] (+ (Math/sin x) (* x (Math/cos x)) (Math/exp x)))

;; fan-out reuse: x used 4+ times (x^2 + x*sin(x) + x)
(deftm laws-f3 [x :- Double] :- Double
  (n/+ (n/+ (n/* x x) (n/* x (m/sin x))) x))

(defn- laws-f3' [x] (+ (+ (+ (* 2.0 x) (Math/sin x)) (* x (Math/cos x))) 1.0))

;; if-branch (kink at x = 1)
(deftm laws-f4 [x :- Double] :- Double
  (if (n/> x 1.0) (n/* x x) (n/* 2.0 x)))

(defn- laws-f4' [x] (if (> x 1.0) (* 2.0 x) 2.0))

;; special fn
(deftm laws-f5 [x :- Double] :- Double
  (special/erf x))

(defn- erf-analytic-deriv [x]
  (* (/ 2.0 (Math/sqrt Math/PI)) (Math/exp (- (* x x)))))

;; 2-param: Rosenbrock
(deftm laws-f6 [x :- Double, y :- Double] :- Double
  (n/+ (n/* (n/- 1.0 x) (n/- 1.0 x))
       (n/* 100.0 (n/* (n/- y (n/* x x)) (n/- y (n/* x x))))))

(defn- laws-f6-dx [x y] (+ (* -2.0 (- 1.0 x)) (* -400.0 x (- y (* x x)))))
(defn- laws-f6-dy [x y] (* 200.0 (- y (* x x))))

;; quadratic for O4 HVP: x^2 + x*y (Hessian [[2 1] [1 0]])
(deftm laws-quad [x :- Double, y :- Double] :- Double
  (n/+ (n/* x x) (n/* x y)))

;; pure fan-out: x + x + x + x — grad is exactly 4.0 in ANY summation order
(deftm laws-fan4 [x :- Double] :- Double
  (n/+ (n/+ x x) (n/+ x x)))

;; zero-contribution branch: else-branch depends only on y
(deftm laws-zbranch [x :- Double, y :- Double] :- Double
  (if (n/> x 0.0) (n/* x x) (n/* 3.0 y)))

;; Long parameter slot (⊥ / NoTangent behavior)
(deftm laws-longslot [x :- Double, k :- Long] :- Double
  (n/* x (double k)))

;; ================================================================
;; Corpus — array fns (float, the Phase-1-fixed composite path:
;; mse-loss over linear-nb inlines through lower-composites)
;; ================================================================

(deftm laws-lnb-mse [x :- (Array float) W :- (Array float) tgt :- (Array float)
                     batch :- Long in :- Long out :- Long] :- Double
  (loss/mse-loss (nn/linear-nb x W batch in out) tgt (clojure.core/* batch out)))

;; g(x) = <f(x), w> for O7 (w passed as a param array)
(deftm laws-lnb-dotw [x :- (Array float) W :- (Array float) w :- (Array float)
                      batch :- Long in :- Long out :- Long] :- Double
  (let [y (nn/linear-nb x W batch in out)
        n* (clojure.core/* batch out)]
    (loop [i 0 s 0.0]
      (if (clojure.core/< i n*)
        (recur (clojure.core/inc i) (+ s (* (ra/aget y i) (ra/aget w i))))
        s))))

;; ================================================================
;; Corpus — O5 boundary fns (defined here; AD over them must THROW)
;; ================================================================

(deftm laws-case-fn [x :- Double, k :- Long] :- Double
  (let [a (case k 1 (n/* x x) (n/* 2.0 x))] a))

;; Let-bound loop with a NON-canonical body (let-wrapped if): not liftable to
;; tail position, so it must hit the fail-loud :loop record. (Canonical
;; tail-liftable loops ARE differentiable — covered by rad_test loop tests.)
(deftm laws-loop-fn [x :- Double] :- Double
  (let [a (loop [i 0 acc 0.0]
            (let [nxt (n/+ acc x)]
              (if (< i 3) (recur (inc i) nxt) acc)))]
    (n/* a a)))

;; body IS a canonical tail-accumulation loop — the shape that used to yield a
;; silent broken IFn from value+grad (flatten nil, unguarded). Now fails loud.
(deftm laws-tail-loop-fn [x :- Double] :- Double
  (loop [i 0 acc 0.0]
    (if (< i 4)
      (recur (inc i) (n/+ acc (n/* x x)))
      acc)))

;; Deliberately-UNCOVERED op for the §11 admissibility law: Math/expm1 as a
;; raw JVM static interop call has a reverse template (Math/expm1) but can
;; NEVER have a Dual lift (primitive-typed static method), so forward mode
;; must be inadmissible while reverse works. (Every templated scalar deftm
;; op now has a Dual overload — interop is the remaining uncovered class.)
(deftm laws-uncovered-fn [x :- Double] :- Double
  (n/* x (Math/expm1 (double x))))

;; ================================================================
;; O1 — mode agreement: grad_rev = grad_fwd = FD on the shared domain
;; ================================================================

(deftest o1-mode-agreement-law
  (testing "reverse = forward = central FD at 3 points, per scalar corpus fn"
    (doseq [[fvar f analytic pts]
            [[#'laws-f1 laws-f1 laws-f1' [-1.7 0.6 2.3]]
             [#'laws-f2 laws-f2 laws-f2' [0.4 1.3 2.1]]
             [#'laws-f3 laws-f3 laws-f3' [0.7 1.3 2.4]]]]
      (let [vg-r (rev/value+grad fvar)
            vg-f (rev/value+grad fvar :mode :forward)]
        (doseq [x pts]
          (let [gr (nth (vg-r x) 1)
                gf (nth (vg-f x) 1)
                fd (central-fd f x)
                an (analytic x)]
            (is (close? gr gf tol-double) (str fvar " rev=fwd at " x))
            (is (close? gr fd tol-double) (str fvar " rev=FD at " x))
            (is (close? gr an tol-double) (str fvar " rev=analytic at " x)))))))

  (testing "2-param fn (Rosenbrock): both partials agree across modes + FD"
    (let [vg-r (rev/value+grad #'laws-f6)
          vg-f (rev/value+grad #'laws-f6 :mode :forward)]
      (doseq [[x y] [[0.8 1.2] [2.0 3.0] [-1.0 2.0]]]
        (let [[_ rx ry] (vg-r x y)
              [_ fx fy] (vg-f x y)
              fdx (central-fd #(laws-f6 % y) x)
              fdy (central-fd #(laws-f6 x %) y)]
          (is (close? rx fx tol-double) (str "d/dx rev=fwd at " [x y]))
          (is (close? ry fy tol-double) (str "d/dy rev=fwd at " [x y]))
          (is (close? rx fdx tol-double) (str "d/dx rev=FD at " [x y]))
          (is (close? ry fdy tol-double) (str "d/dy rev=FD at " [x y]))
          (is (close? rx (laws-f6-dx x y) tol-double))
          (is (close? ry (laws-f6-dy x y) tol-double))))))

  (testing "if-branch fn: reverse = forward = FD away from the kink"
    ;; Forward over branches works via Dual comparison overloads (compare .v —
    ;; branch selection on the primal, a.e.-correct; framework §8). Was a
    ;; KNOWN-GAP (no Dual `>`), fixed 2026-07-04 after the laws suite found it.
    (let [vg-r (rev/value+grad #'laws-f4)
          vg-f (rev/value+grad #'laws-f4 :mode :forward)]
      (doseq [x [0.4 2.5 3.1]]
        (let [gr (nth (vg-r x) 1)
              gf (nth (vg-f x) 1)]
          (is (close? gr (central-fd laws-f4 x) tol-double) (str "branch rev=FD at " x))
          (is (close? gr (laws-f4' x) tol-double) (str "branch rev=analytic at " x))
          (is (close? gf gr tol-double) (str "branch fwd=rev at " x))))))

  (testing "erf: reverse = forward = analytic; FD only loosely (approximation slope)"
    ;; erf got its Dual lift in the carrier-coverage completion (framework
    ;; §4a/§11): forward mode is available and must agree with reverse.
    ;; The erf rrule (and the Dual scale) is the ANALYTIC derivative
    ;; 2/sqrt(pi)*exp(-x^2); the primal is the Abramowitz-Stegun 7.1.26
    ;; approximation (max err 1.5e-7), so FD-of-the-primal converges ~1.4e-6
    ;; away from the rule — the 5e-6 tolerance absorbs the approximation's
    ;; slope error (measured, stable across h), not an AD error. A wrong
    ;; rule would be off by O(1).
    (let [vg-r (rev/value+grad #'laws-f5)
          vg-f (rev/value+grad #'laws-f5 :mode :forward)]
      (doseq [x [0.3 0.5 1.1]]
        (let [gr (nth (vg-r x) 1)
              gf (nth (vg-f x) 1)]
          (is (close? gr (erf-analytic-deriv x) 1e-10) (str "erf rev=analytic at " x))
          (is (close? gf (erf-analytic-deriv x) 1e-10) (str "erf fwd=analytic at " x))
          (is (close? gr (central-fd laws-f5 x 1e-4) 5e-6) (str "erf rev~FD at " x)))))))

;; ================================================================
;; O3 — tangent-algebra laws: ⊕ accumulation, 0̄, ⊥ slots
;; ================================================================

(deftest o3-tangent-algebra-law
  (testing "fan-out accumulation is EXACT (⊕ correctness)"
    ;; Reassociation-free case: every contribution is 1.0, so the ⊕-sum over
    ;; the 4 uses of x must be EXACTLY 4.0 — a dropped or doubled
    ;; contribution is off by a whole unit.
    (let [vg (rev/value+grad #'laws-fan4)]
      (doseq [x [1.3 0.7 2.4]]
        (is (== 4.0 (nth (vg x) 1)) (str "x+x+x+x grad bit-exact 4.0 at " x))))
    ;; Mixed fan-out (x used in x^2, x*sin x, and bare): agreement with the
    ;; analytic sum to 1 ulp-scale (1e-14 scaled) — the AD accumulation order
    ;; may legally reassociate the ⊕-sum vs the hand-written analytic form.
    ;; A missing contribution would be O(1), not 1 ulp.
    (let [vg (rev/value+grad #'laws-f3)]
      (doseq [x [1.3 0.7 2.4]]
        (is (close? (nth (vg x) 1) (laws-f3' x) 1e-14)
            (str "fan-out grad = analytic at " x)))))

  (testing "zero-contribution branch: untaken/constant path contributes 0̄"
    (let [vg (rev/value+grad #'laws-zbranch)]
      (let [[v dx dy] (vg 2.0 5.0)]
        (is (== 4.0 v))
        (is (== 4.0 dx) "taken branch grad")
        (is (== 0.0 dy) "y unused in taken branch → zero"))
      (let [[v dx dy] (vg -2.0 5.0)]
        (is (== 15.0 v))
        (is (== 0.0 dx) "x only in the untaken branch → zero")
        (is (== 3.0 dy)))))

  (testing "⊥ (NoTangent): Long param slot receives nil grad and never seeds"
    (let [[v dx dk] ((rev/value+grad #'laws-longslot) 2.0 3)]
      (is (== 6.0 v))
      (is (== 3.0 dx))
      (is (nil? dk) "Long slot is ⊥ → nil, not 0.0"))
    ;; Array composite: the three Long shape params are all ⊥ slots.
    (let [x (fa 8 1) W (fa 12 2) tgt (fa 6 3)
          res ((rev/value+grad #'laws-lnb-mse) x W tgt 2 4 3)]
      (is (every? nil? (drop 4 res)) "batch/in/out Long slots are nil"))))

;; ================================================================
;; O3b — Π (tangent-type protocol): per-param dtype preservation
;; (framework §5, task #44). Every cotangent lives in the tangent space
;; of ITS OWN primal — dtype derives from each param's type tag, and
;; the seed from the RESULT's type, never from a global binary switch.
;; ================================================================

;; Mixed scalars: Double × Float. d_x must stay Double, d_y must be Float.
(deftm laws-pi-scalar [x :- Double, y :- Float] :- Double
  (n/* x y))

;; THE mixed acceptance shape: float array params + a Double scalar param
;; + Long ⊥ slots, Double loss. loss = w * mse(linear-nb(x,W), tgt).
(deftm laws-pi-mixed-d [x :- (Array float) W :- (Array float) tgt :- (Array float)
                        w :- Double batch :- Long in :- Long out :- Long] :- Double
  (n/* w (loss/mse-loss (nn/linear-nb x W batch in out) tgt (clojure.core/* batch out))))

;; Same but w :- Float: d_w must come back Float (Π projects the Double
;; cotangent dy·mse into Float32 — the daxpy-diff!-class bug direction).
(deftm laws-pi-mixed-f [x :- (Array float) W :- (Array float) tgt :- (Array float)
                        w :- Float batch :- Long in :- Long out :- Long] :- Double
  (n/* w (loss/mse-loss (nn/linear-nb x W batch in out) tgt (clojure.core/* batch out))))

;; Double-array twin of laws-lnb-mse for the double[] regression direction.
(deftm laws-pi-darr [x :- (Array double) W :- (Array double) tgt :- (Array double)
                     batch :- Long in :- Long out :- Long] :- Double
  (loss/mse-loss (nn/linear-nb x W batch in out) tgt (clojure.core/* batch out)))

(defn- da
  "Deterministic Gaussian double array."
  ^doubles [n seed]
  (let [r (java.util.Random. (long seed)) a (double-array n)]
    (dotimes [i n] (aset a i (.nextGaussian r))) a))

(deftest o3b-tangent-type-protocol-law
  (testing "mixed scalar dtypes: each grad in its own primal's tangent space"
    (let [[v dx dy] ((rev/value+grad #'laws-pi-scalar) 2.0 (float 3.0))]
      (is (instance? Double v))
      (is (instance? Double dx) "d_x: Double primal → Double cotangent")
      (is (instance? Float dy) "d_y: Float primal → Float cotangent (Π projects)")
      (is (close? dx 3.0 tol-double))
      (is (close? dy 2.0 tol-float))))

  (testing "mixed array/scalar fn: per-param dtypes preserved in ONE fn"
    (let [x (fa 8 11) W (fa 12 12) tgt (fa 6 13)
          [v gx gW gtgt gw & longs] ((rev/value+grad #'laws-pi-mixed-d) x W tgt 2.0 2 4 3)
          mse (loss/mse-loss (nn/linear-nb x W 2 4 3) tgt 6)]
      (is (instance? Double v) "Double loss")
      (is (instance? (Class/forName "[F") gx) "grad-x is float[]")
      (is (instance? (Class/forName "[F") gW) "grad-W is float[]")
      (is (instance? (Class/forName "[F") gtgt) "grad-tgt is float[] (typed zero default)")
      (is (instance? Double gw) "grad-w is Double, not Float")
      (is (every? nil? longs) "Long slots stay ⊥/nil")
      (is (close? gw mse tol-double) "d(w·L)/dw = L")))

  (testing "Float scalar param in a Double-loss fn: grad projected to Float"
    (let [x (fa 8 21) W (fa 12 22) tgt (fa 6 23)
          [v gx _gW _gtgt gw] ((rev/value+grad #'laws-pi-mixed-f) x W tgt (float 2.0) 2 4 3)
          mse (loss/mse-loss (nn/linear-nb x W 2 4 3) tgt 6)]
      (is (instance? Double v))
      (is (instance? (Class/forName "[F") gx))
      (is (instance? Float gw) "d_w: Float primal → Float cotangent")
      (is (close? gw mse tol-float))))

  (testing "float-only fn still yields float[] grads (regression)"
    (let [x (fa 8 1) W (fa 12 2) tgt (fa 6 3)
          [v gx gW gtgt] ((rev/value+grad #'laws-lnb-mse) x W tgt 2 4 3)]
      (is (instance? Double v))
      (is (instance? (Class/forName "[F") gx))
      (is (instance? (Class/forName "[F") gW))
      (is (instance? (Class/forName "[F") gtgt))))

  (testing "double-only fn yields Double/double[] grads with no projection churn"
    (let [[v dx dy] ((rev/value+grad #'laws-f6) 0.8 1.2)]
      (is (instance? Double v))
      (is (instance? Double dx))
      (is (instance? Double dy)))
    (let [x (da 8 31) W (da 12 32) tgt (da 6 33)
          [v gx gW gtgt] ((rev/value+grad #'laws-pi-darr) x W tgt 2 4 3)]
      (is (instance? Double v))
      (is (instance? (Class/forName "[D") gx) "grad-x is double[]")
      (is (instance? (Class/forName "[D") gW) "grad-W is double[]")
      (is (instance? (Class/forName "[D") gtgt) "grad-tgt is double[]")))

  (testing "materialized zeros are typed or fail loud (unit: the protocol + helper)"
    ;; The protocol ns directly:
    (is (= {:kind :array :dtype :float} (tangent/tangent-kind 'floats)))
    (is (= {:kind :scalar :dtype :double} (tangent/tangent-kind 'double)))
    (is (= {:kind :none} (tangent/tangent-kind 'long)))
    (is (= {:kind :none} (tangent/tangent-kind nil)) "untagged = statically unknown = ⊥")
    (is (= (list 'float-array 'n) (tangent/zero-expr 'floats 'n)))
    (is (= 0.0 (tangent/zero-expr 'double 'n)))
    (is (thrown? clojure.lang.ExceptionInfo (tangent/zero-expr nil 'n))
        "unknown tangent space cannot materialize a zero")
    (is (thrown? clojure.lang.ExceptionInfo (tangent/zero-expr 'long 'n))
        "⊥ has no zero — NoTangent ≠ ZeroTangent")
    ;; nil-safe runtime projection (0̄ passes through):
    (is (nil? (tangent/project-double nil)))
    (is (instance? Float (tangent/project-float 1.5)))
    ;; zero-like: the dynamic fallback derives dtype+shape from the value.
    (is (instance? (Class/forName "[F") (tangent/zero-like (float-array 3))))
    (is (instance? Float (tangent/zero-like (float 2.0))))
    (is (nil? (tangent/zero-like nil)))
    ;; array-zero-of-shape (private helper): tag-reading path emits typed
    ;; ctors; untagged throws instead of the old silent scalar 0.0.
    (let [azos @#'rev/array-zero-of-shape]
      (is (= (list 'float-array (list 'raster.arrays/alength 'xs))
             (azos (with-meta 'xs {:raster.type/tag 'floats}))))
      (is (thrown? clojure.lang.ExceptionInfo (azos 'untagged))
          "untagged array slot fails loud, no scalar-0.0 mis-shape"))))

;; ================================================================
;; O4 — composition: HVP = FD(grad); Jet-k coeffs = k-th derivatives
;; ================================================================

(deftest o4-composition-law
  (testing "compile-hvp-fn on quadratic: exact Hessian columns"
    (let [hvp (rev/compile-hvp-fn #'laws-quad)]
      (is (= [2.0 1.0] (hvp [3.0 2.0] [1.0 0.0])))
      (is (= [1.0 0.0] (hvp [3.0 2.0] [0.0 1.0])))))

  (testing "compile-hvp-fn on Rosenbrock: HVP(v) = directional FD of grad"
    (let [hvp (rev/compile-hvp-fn #'laws-f6)
          grad-fn (fn [x y] (vec (rest ((rev/value+grad #'laws-f6) x y))))
          h 1e-5]
      (doseq [[[x y] [vx vy]] [[[2.0 3.0] [1.0 0.0]]
                               [[2.0 3.0] [0.3 -0.7]]
                               [[0.8 1.2] [1.0 1.0]]]]
        (let [hv (hvp [x y] [vx vy])
              g+ (grad-fn (+ x (* h vx)) (+ y (* h vy)))
              g- (grad-fn (- x (* h vx)) (- y (* h vy)))
              fd-hv (mapv #(/ (- %1 %2) (* 2.0 h)) g+ g-)]
          (is (close? (nth hv 0) (nth fd-hv 0) 1e-4) (str "HVP[0] at " [x y] " v=" [vx vy]))
          (is (close? (nth hv 1) (nth fd-hv 1) 1e-4) (str "HVP[1] at " [x y] " v=" [vx vy]))))))

  (testing "Jet: higher-derivatives of sin at pi = [sin cos -sin -cos] pattern"
    (let [d (vec (jet/higher-derivatives (fn [x] (m/sin x)) Math/PI 3))]
      (is (close? (nth d 0) 0.0 1e-12) "sin(pi)")
      (is (close? (nth d 1) -1.0 1e-12) "cos(pi)")
      (is (close? (nth d 2) 0.0 1e-12) "-sin(pi)")
      (is (close? (nth d 3) 1.0 1e-12) "-cos(pi)"))))

;; ================================================================
;; O5 — boundaries: unsupported forms throw (never silent), ⊥ never seeds,
;; :auto is constrained by carrier coverage (currently: loud trap)
;; ================================================================

(deftest o5-boundary-law
  (testing "`case` over an active value throws with a clear message"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo
         #"AD through `case\*?` is not supported"
         (rev/value+grad #'laws-case-fn))))

  (testing "raw (non-liftable) let-bound loop throws with a clear message"
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo
         #"Cannot differentiate through raw `loop`"
         (rev/value+grad #'laws-loop-fn))))

  (testing "tail-loop body: value+grad fails LOUDLY (never a broken IFn)"
    ;; Laws-suite finding 2026-07-04: a body that IS (or tail-lifts to) a raw
    ;; loop produced flatten nil → an unguarded broken IFn (wrong-arity at call
    ;; time). Now guarded: clear ex-info at construction time. The compiled /
    ;; inline path handles these loops; the runtime assembler does not yet.
    (is (thrown-with-msg?
         Exception
         #"cannot assemble a runtime gradient"
         (rev/value+grad #'laws-tail-loop-fn))))

  (testing ":auto is admissibility-constrained (§11): erf goes forward AND is correct"
    ;; FLIPPED 2026-07-04 (was the KNOWN-GAP ':auto trap'): erf now has a
    ;; Dual lift, so :auto (n=1 + admissible) legitimately selects forward.
    ;; The LAW is the gradient value, whatever admissible mode is selected.
    (is (rev/forward-admissible? #'laws-f5) "erf body is Dual-covered")
    (is (close? (nth ((rev/value+grad #'laws-f5 :mode :auto) 0.7) 1)
                (erf-analytic-deriv 0.7) 1e-10)
        ":auto on erf returns the analytic gradient")
    ;; ...and :reverse on the same fn agrees (coverage exists there too).
    (is (close? (nth ((rev/value+grad #'laws-f5 :mode :reverse) 0.5) 1)
                (erf-analytic-deriv 0.5) 1e-10)))

  (testing ":auto hard-filters inadmissible forward; :forward fails loud at construction"
    ;; laws-uncovered-fn calls Math/expm1 (JVM static interop): reverse-
    ;; templated, but no Dual lift can exist. n=1 would select forward by
    ;; the Griewank dimension test alone — admissibility must force reverse
    ;; (§11: select argmin-cost over ADMISSIBLE interpretations only).
    (is (not (rev/forward-admissible? #'laws-uncovered-fn)))
    (is (= '[Math/expm1] (:uncovered-ops (rev/forward-coverage #'laws-uncovered-fn)))
        "the coverage artifact names exactly the uncovered op")
    (let [x 0.7
          expected (+ (Math/expm1 x) (* x (Math/exp x)))] ;; d/dx x·expm1(x)
      (is (close? (nth ((rev/value+grad #'laws-uncovered-fn :mode :auto) x) 1)
                  expected tol-double)
          ":auto selects reverse and the gradient is correct"))
    ;; :mode :forward itself: a clear CONSTRUCTION-time ex-info naming the
    ;; uncovered op — never a No-matching-method at call time.
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"Math/expm1"
                          (rev/value+grad #'laws-uncovered-fn :mode :forward)))))

;; ================================================================
;; O6 — initiality / two-route symbolic agreement:
;; sdiff on the traced form  ≡  Dual{Sym} through the SAME ops.
;; This is the diff.clj ≡ Dual{Sym} law gating diff.clj retirement (§10.3).
;; ================================================================

;; Flow the corpus ops over an arbitrary carrier (Sym or Dual__Sym): the
;; UNCHANGED numeric/math ops, generic dispatch. Must stay in sync with
;; laws-f2 / the x^2*cos(x) form below.
;;
;; Corpus rule: SHARED-COVERAGE ops only — an op joins when it has BOTH a
;; Dual lift (raster.ad.forward) AND Sym support (sym/core overload + a
;; sdiff rule). tan/tanh/atan2 joined with the carrier-coverage completion.
;; erf is deliberately absent: it has a Dual lift but NO Sym overload/sdiff
;; rule, so there is no second route to agree with (its Dual route is
;; law-checked against the analytic derivative in O1/O5 instead).
(defn- o6-f2-ops [x] (n/+ (n/* x (m/sin x)) (m/exp x)))
(defn- o6-x2cos-ops [x] (n/* (n/* x x) (m/cos x)))
(defn- o6-xtan-ops [x] (n/* x (m/tan x)))
(defn- o6-tanh-ops [x] (m/tanh (n/* x x)))
(defn- o6-atan2-ops [x] (m/atan2 x 2.0))

(defn- num-eval
  "Numerically evaluate a symbolic derivative form at x. Direct structural
  interpreter — avoids eval/*ns* resolution issues, and handles the
  Sym-wrapped-constant hazard (sym-interpret may return a Sym) by operating
  on the unwrapped form only."
  [form x]
  (let [ev (fn ev [f]
             (cond
               (number? f) (double f)
               (= 'x f) (double x)
               (seq? f)
               (let [args (mapv ev (rest f))]
                 (case (first f)
                   + (reduce + args)
                   - (if (= 1 (count args)) (- (first args)) (reduce - args))
                   * (reduce * args)
                   / (reduce / args)
                   sin (Math/sin (first args))
                   cos (Math/cos (first args))
                   exp (Math/exp (first args))
                   tan (Math/tan (first args))
                   tanh (Math/tanh (first args))
                   atan2 (Math/atan2 (first args) (second args))
                   sqrt (Math/sqrt (first args))
                   pow (Math/pow (first args) (second args))
                   (throw (ex-info "num-eval: unknown op in symbolic derivative"
                                   {:op (first f) :form form}))))
               :else (throw (ex-info "num-eval: unknown atom" {:atom f :form form}))))]
    (ev (sym/unwrap form))))

(defn- dual-sym-derivative
  "Route (b): flow Dual{Sym} through ops-fn, extract the symbolic partial.

  Context-classloader note: raster.core/get-or-create-parametric-class!
  resolves the freshly eval'd Dual__Sym specialization via Class/forName on
  the THREAD CONTEXT classloader. Under a REPL that is Clojure's
  DynamicClassLoader and it works; under the plain test runner / `clojure -e`
  the context loader is the AppClassLoader and forName throws
  ClassNotFoundException (latent env-sensitivity in the parametric-class
  cache, not an AD defect). Pin a DynamicClassLoader for the duration."
  [ops-fn]
  (let [t (Thread/currentThread)
        old (.getContextClassLoader t)]
    (try
      (.setContextClassLoader t (clojure.lang.RT/makeClassLoader))
      (let [dx (fwd/make-dual (sym/sym 'x) (object-array [(sym/sym 1)]))
            res (ops-fn dx)
            partials ^objects (.partials res)]
        (sym/unwrap (aget partials 0)))
      (finally (.setContextClassLoader t old)))))

(deftest o6-initiality-law
  (doseq [[label ops-fn analytic]
          [["x*sin(x)+exp(x)" o6-f2-ops laws-f2']
           ["x^2*cos(x)" o6-x2cos-ops
            (fn [x] (- (* 2.0 x (Math/cos x)) (* x x (Math/sin x))))]
           ["x*tan(x)" o6-xtan-ops
            (fn [x] (let [t (Math/tan x)] (+ t (* x (+ 1.0 (* t t))))))]
           ["tanh(x^2)" o6-tanh-ops
            (fn [x] (let [t (Math/tanh (* x x))] (* (- 1.0 (* t t)) 2.0 x)))]
           ["atan2(x,2)" o6-atan2-ops
            (fn [x] (/ 2.0 (+ 4.0 (* x x))))]]]
    (testing (str "two-route symbolic derivative agreement: " label)
      (let [traced (sym/unwrap (ops-fn (sym/sym 'x)))     ;; Sym trace of f
            route-a (sdiff/differentiate traced 'x)        ;; diff.clj calculus
            route-b (dual-sym-derivative ops-fn)]          ;; Dual{Sym}, no table
        (doseq [x [0.9 1.7]]
          (let [va (num-eval route-a x)
                vb (num-eval route-b x)
                an (analytic x)]
            (is (close? va vb 1e-10) (str label ": routes agree at " x))
            (is (close? va an 1e-10) (str label ": sdiff = analytic at " x))
            (is (close? vb an 1e-10) (str label ": Dual{Sym} = analytic at " x))))))))

;; ================================================================
;; O7 — adjoint dot-product identity: <J·v, w> = <v, J^T·w>
;; frule and rrule test each other with NO FD oracle for the pairing
;; (JAX check_vjp). LHS via directional evaluation of the PRIMAL only
;; (linear-nb is linear in x, so directional FD is exact up to float
;; rounding); RHS via reverse-mode of g(x) = <f(x), w>.
;; ================================================================

(deftest o7-adjoint-identity-law
  (testing "array composite (linear-nb, float): <J·v, w> = <v, grad <f(x),w>>"
    (let [batch 2 in 4 out 3
          x (fa (* batch in) 11) W (fa (* out in) 12)
          v (fa (* batch in) 13)                     ;; input tangent
          w (fa (* batch out) 14)                    ;; output cotangent
          h 1e-2
          y+ (nn/linear-nb (faxpy x h v) W batch in out)
          y- (nn/linear-nb (faxpy x (- h) v) W batch in out)
          lhs (/ (- (fdot y+ w) (fdot y- w)) (* 2.0 h))
          res ((rev/value+grad #'laws-lnb-dotw) x W w batch in out)
          gx ^floats (nth res 1)
          rhs (fdot v gx)]
      (is (close? lhs rhs tol-float) "adjoint identity over x")
      ;; Sanity: the scalar pairing itself matches the primal value.
      (is (close? (first res) (fdot (nn/linear-nb x W batch in out) w) tol-float))))

  (testing "scalar anchor: <f'(x)v, w> = <v, d/dx (w*f(x))>"
    (let [x 1.3 v 0.37 w -2.1
          lhs (* w (central-fd laws-f2 x) v)
          gx (nth ((rev/value+grad #'laws-f2) x) 1)
          rhs (* v w gx)]
      (is (close? lhs rhs tol-double)))))

;; ================================================================
;; O8 — rule linearity: every pullback is a LINEAR map over the tangent
;; algebra: pb(a ⊕ b) = pb(a) ⊕ pb(b), pb(0̄) = 0̄, pb(c·a) = c·pb(a).
;; Via template-pullback (the eval'd consumption of the ONE rule source).
;; nil (⊥) slots must be nil on both sides.
;; ================================================================

(deftest o8-rule-linearity-law
  (testing "scalar pullbacks: raster.numeric/* and Math/sin"
    (doseq [[label pb-args]
            [["raster.numeric/*"
              (let [pb-fn (tmpl/template-pullback 'raster.numeric/*)]
                (pb-fn 6.0 2.0 3.0))]
             ["Math/sin"
              (let [pb-fn (tmpl/template-pullback 'Math/sin)]
                (pb-fn (Math/sin 0.7) 0.7))]]]
      (let [pb pb-args
            a 0.37 b -1.2 c 2.5
            ga (pb a) gb (pb b) gab (pb (+ a b)) g0 (pb 0.0) gca (pb (* c a))]
        (dotimes [i (count ga)]
          (is (close? (nth gab i) (+ (nth ga i) (nth gb i)) 1e-12)
              (str label " additivity slot " i))
          (is (close? (nth g0 i) 0.0 1e-12) (str label " pb(0)=0 slot " i))
          (is (close? (nth gca i) (* c (nth ga i)) 1e-12)
              (str label " homogeneity slot " i))))))

  (testing "array pullback: raster.dl.nn/linear-nb (float cotangents)"
    (let [pb-fn (tmpl/template-pullback 'raster.dl.nn/linear-nb)
          batch 2 in 4 out 3
          x (fa (* batch in) 21) W (fa (* out in) 22)
          y (nn/linear-nb x W batch in out)
          pb (pb-fn y x W batch in out)
          a (fa (* batch out) 23) b (fa (* batch out) 24) c 2.5
          ga (pb a) gb (pb b) gab (pb (fadd a b))
          g0 (pb (float-array (* batch out))) gca (pb (fscale c a))]
      (doseq [slot [0 1]]                              ;; dx, dW
        (is (farr-close? (nth gab slot) (fadd (nth ga slot) (nth gb slot)) tol-float)
            (str "linear-nb additivity slot " slot))
        (is (every? #(close? % 0.0 tol-float) (nth g0 slot))
            (str "linear-nb pb(0)=0 slot " slot))
        (is (farr-close? (nth gca slot) (fscale c (nth ga slot)) tol-float)
            (str "linear-nb homogeneity slot " slot)))
      (doseq [slot [2 3 4]]                            ;; Long shape params: ⊥
        (is (and (nil? (nth ga slot)) (nil? (nth gab slot)))
            (str "linear-nb ⊥ slot " slot " nil on both sides")))))

  (testing "array pullback: raster.dl.loss/mse-loss (scalar cotangent)"
    (let [pb-fn (tmpl/template-pullback 'raster.dl.loss/mse-loss)
          pred (fa 6 31) tgt (fa 6 32)
          l (loss/mse-loss pred tgt 6)
          pb (pb-fn l pred tgt 6)
          a 0.37 b -1.2 c 2.5
          ga (pb a) gb (pb b) gab (pb (+ a b)) g0 (pb 0.0) gca (pb (* c a))]
      (is (farr-close? (nth gab 0) (fadd (nth ga 0) (nth gb 0)) tol-float)
          "mse d-pred additivity")
      (is (every? #(close? % 0.0 tol-float) (nth g0 0)) "mse pb(0)=0")
      (is (farr-close? (nth gca 0) (fscale c (nth ga 0)) tol-float)
          "mse d-pred homogeneity")
      (doseq [slot [1 2]]
        (is (and (nil? (nth ga slot)) (nil? (nth gab slot)))
            (str "mse ⊥ slot " slot " nil on both sides"))))))

;; ================================================================
;; Fan-out scaling guard (§12, Smeding–Vákár): the reverse transform must be
;; call-at-most-once — naive closure-tape reverse is EXPONENTIAL under
;; fan-out. Sharing chain a_{i+1} = a_i + a_i at depth 24: grad must be
;; exactly 2^24 and one value+grad must be fast (linear). The 5s bound is
;; generous to avoid flaky CI; an exponential regression takes minutes.
;; ================================================================

(def ^:private fanout-depth 24)

;; Programmatically generate the sharing-chain deftm at load time.
(eval
 (let [a (fn [i] (symbol (str "lws_a" i)))
       bindings (vec (mapcat (fn [i] [(a (inc i))
                                      (list 'raster.numeric/+ (a i) (a i))])
                             (range fanout-depth)))
       body (list 'let* (into [(a 0) 'x] bindings) (a fanout-depth))]
   (list 'raster.core/deftm 'laws-chain24 '[x :- Double] :- 'Double body)))

(deftest fanout-scaling-guard
  (testing "depth-24 sharing chain: exact 2^24 gradient, linear-time reverse"
    (let [t0 (System/nanoTime)
          res ((rev/value+grad #'laws-chain24) 1.0)
          elapsed-ms (/ (- (System/nanoTime) t0) 1e6)]
      (is (== (Math/pow 2.0 fanout-depth) (nth res 0)) "primal = 2^24 * x")
      (is (== (Math/pow 2.0 fanout-depth) (nth res 1)) "grad exactly 2^24")
      (is (< elapsed-ms 5000.0)
          (str "value+grad under fan-out must be linear-time, took "
               elapsed-ms "ms")))))
