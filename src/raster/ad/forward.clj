(ns raster.ad.forward
  "Forward-mode automatic differentiation using multi-partial dual numbers.

   Each Dual carries N partial derivatives simultaneously, enabling
   chunk-based gradient computation: ceil(n/chunk) passes instead of n.

   Extends raster.numeric/+, -, *, / with Dual number dispatch,
   so any function written with polymorphic arithmetic automatically
   supports AD — no code changes needed.

   Usage:
     (require '[raster.ad.forward :refer [dual derivative gradient jacobian exp sin cos]])
     (derivative (fn [x] (* x x x)) 2.0) ;=> 12.0
     (gradient (fn [xs] (+ (* (nth xs 0) (nth xs 0))
                           (* (nth xs 1) (nth xs 1))))
               [3.0 4.0]) ;=> [6.0 8.0]"
  (:refer-clojure :exclude [+ - * / abs aget aset alength aclone])
  (:require [raster.core :refer [deftm defvalue defval broadcast]]
            [raster.arrays :refer [aget aset alength aclone]]
            [raster.par]
            [raster.numeric :as n]
            [raster.math :as m]
            [raster.sci.special :as special]
            [raster.types.promote :as promote]))

;; ================================================================
;; Dual number type — multi-partial
;; ================================================================

(defvalue Dual (All [T]) [v :- T, partials :- (Array T)])

(def ^:const DEFAULT_CHUNK 8)

(deftm dual
  "Wrap a scalar value as a Dual number with zero partials."
  [v :- Double] :- Dual
  (->Dual (double v) (double-array [0.0])))

(deftm dual [v :- Double, dv :- Double] :- Dual
  (->Dual (double v) (double-array [(double dv)])))

(defn make-dual
  "Create a Dual number for arbitrary element types.
   For numeric values, uses the standard double Dual.
   For non-numeric values (e.g., Sym), auto-generates the parametric
   specialization (Dual__Sym etc.) and creates an instance.

   (make-dual 3.0 (double-array [1.0 0.0]))          ;; standard Dual
   (make-dual (sym 'x) (object-array [(sym 1) (sym 0)])) ;; Dual__Sym"
  [v partials]
  (if (number? v)
    (->Dual (double v) (if (.isArray (class partials))
                         (let [ct (.getComponentType (class partials))]
                           (if (= ct Double/TYPE) partials
                               (double-array (map double partials))))
                         (double-array (map double partials))))
    (let [elem-type (class v)
          elem-name (.getSimpleName elem-type)
          dual-name (str "Dual__" elem-name)
          fqn (str "raster.ad.forward." dual-name)
          this-ns (the-ns 'raster.ad.forward)]
      ;; Import element type class into this namespace
      (.importClass ^clojure.lang.Namespace this-ns elem-type)
      ;; Generate Dual specialization via shared cache — ensures a single
      ;; Class object across make-dual and parametric-specialize!
      (let [cls-sym (symbol dual-name)
            cls (raster.core/get-or-create-parametric-class!
                 fqn
                 (fn []
                   (let [tmpl (get @raster.core/parametric-value-registry 'Dual)
                         elem-sym (symbol elem-name)
                         fields (clojure.walk/postwalk
                                 #(if (= % 'T) elem-sym %) (:fields tmpl))]
                     (binding [*ns* this-ns]
                       (eval (list* 'raster.core/defvalue cls-sym fields
                                    (apply concat (:opts tmpl))))))))
            ctor-sym (symbol (str "->" dual-name))]
        ;; Import and resolve the constructor
        (.importClass ^clojure.lang.Namespace this-ns cls)
        (let [ctor-var (or (ns-resolve this-ns ctor-sym)
                           (ns-resolve (the-ns 'user) ctor-sym))]
          (ctor-var v partials))))))

(defn- zero-partials
  "Create a zero partials array matching the type and size of an existing partials array."
  [ps]
  (java.lang.reflect.Array/newInstance
   (.getComponentType (class ps))
   (int (alength ps))))

;; ================================================================
;; Arithmetic: Dual × Dual (parametric over element type T)
;; ================================================================

(deftm raster.numeric/+ (All [T] [x :- Dual, y :- Dual] :- Dual
                             (let [px (.partials x) py (.partials y)
                                   len (alength px)
                                   result (aclone px)]
                               (dotimes [i len]
                                 (aset result i (n/+ (aget px i) (aget py i))))
                               (->Dual (n/+ (.v x) (.v y)) result))))

(deftm raster.numeric/- (All [T] [x :- Dual] :- Dual
                             (let [px (.partials x)
                                   len (alength px)
                                   result (aclone px)]
                               (dotimes [i len]
                                 (aset result i (n/- (aget px i))))
                               (->Dual (n/- (.v x)) result))))

(deftm raster.numeric/- (All [T] [x :- Dual, y :- Dual] :- Dual
                             (let [px (.partials x) py (.partials y)
                                   len (alength px)
                                   result (aclone px)]
                               (dotimes [i len]
                                 (aset result i (n/- (aget px i) (aget py i))))
                               (->Dual (n/- (.v x) (.v y)) result))))

(deftm raster.numeric/* (All [T] [x :- Dual, y :- Dual] :- Dual
                             (let [xv (.v x) yv (.v y)
                                   px (.partials x) py (.partials y)
                                   len (alength px)
                                   result (aclone px)]
                               (dotimes [i len]
                                 (aset result i (n/+ (n/* xv (aget py i)) (n/* (aget px i) yv))))
                               (->Dual (n/* xv yv) result))))

(deftm raster.numeric// (All [T] [x :- Dual, y :- Dual] :- Dual
                             (let [xv (.v x) yv (.v y) yv2 (n/* yv yv)
                                   px (.partials x) py (.partials y)
                                   len (alength px)
                                   result (aclone px)]
                               (dotimes [i len]
                                 (aset result i (n// (n/- (n/* (aget px i) yv) (n/* xv (aget py i)))
                                                     yv2)))
                               (->Dual (n// xv yv) result))))

;; ================================================================
;; Arithmetic: Dual × Double (concrete fast path — primitive ops)
;; ================================================================

(deftm raster.numeric/+ [x :- Dual, y :- Double] :- Dual
  (->Dual (clojure.core/+ (.v x) y) (aclone (.partials x))))

(deftm raster.numeric/+ [x :- Double, y :- Dual] :- Dual
  (->Dual (clojure.core/+ x (.v y)) (aclone (.partials y))))

(deftm raster.numeric/- [x :- Dual, y :- Double] :- Dual
  (->Dual (clojure.core/- (.v x) y) (aclone (.partials x))))

(deftm raster.numeric/- [x :- Double, y :- Dual] :- Dual
  (let [py (.partials y)]
    (->Dual (clojure.core/- x (.v y))
            (broadcast [py] (clojure.core/- py)))))

(deftm raster.numeric/* [x :- Dual, y :- Double] :- Dual
  (let [px (.partials x)]
    (->Dual (clojure.core/* (.v x) y)
            (broadcast [px] (clojure.core/* px y)))))

(deftm raster.numeric/* [x :- Double, y :- Dual] :- Dual
  (let [py (.partials y)]
    (->Dual (clojure.core/* x (.v y))
            (broadcast [py] (clojure.core/* x py)))))

(deftm raster.numeric// [x :- Dual, y :- Double] :- Dual
  (let [px (.partials x)]
    (->Dual (clojure.core// (.v x) y)
            (broadcast [px] (clojure.core// px y)))))

(deftm raster.numeric// [x :- Double, y :- Dual] :- Dual
  (let [yv (.v y)
        yv2 (clojure.core/* yv yv)
        py (.partials y)]
    (->Dual (clojure.core// x yv)
            (broadcast [py] (clojure.core/- (clojure.core// (clojure.core/* x py) yv2))))))

;; ================================================================
;; Arithmetic: Dual × Number (parametric fallback for non-double T)
;; ================================================================

(deftm raster.numeric/+ (All [T] [x :- Dual, y :- Number] :- Dual
                             (->Dual (n/+ (.v x) y) (aclone (.partials x)))))

(deftm raster.numeric/+ (All [T] [x :- Number, y :- Dual] :- Dual
                             (->Dual (n/+ x (.v y)) (aclone (.partials y)))))

(deftm raster.numeric/- (All [T] [x :- Dual, y :- Number] :- Dual
                             (->Dual (n/- (.v x) y) (aclone (.partials x)))))

(deftm raster.numeric/- (All [T] [x :- Number, y :- Dual] :- Dual
                             (let [py (.partials y)
                                   len (alength py)
                                   result (aclone py)]
                               (dotimes [i len]
                                 (aset result i (n/- (aget py i))))
                               (->Dual (n/- x (.v y)) result))))

(deftm raster.numeric/* (All [T] [x :- Dual, y :- Number] :- Dual
                             (let [px (.partials x)
                                   len (alength px)
                                   result (aclone px)]
                               (dotimes [i len]
                                 (aset result i (n/* (aget px i) y)))
                               (->Dual (n/* (.v x) y) result))))

(deftm raster.numeric/* (All [T] [x :- Number, y :- Dual] :- Dual
                             (let [py (.partials y)
                                   len (alength py)
                                   result (aclone py)]
                               (dotimes [i len]
                                 (aset result i (n/* x (aget py i))))
                               (->Dual (n/* x (.v y)) result))))

(deftm raster.numeric// (All [T] [x :- Dual, y :- Number] :- Dual
                             (let [px (.partials x)
                                   len (alength px)
                                   result (aclone px)]
                               (dotimes [i len]
                                 (aset result i (n// (aget px i) y)))
                               (->Dual (n// (.v x) y) result))))

(deftm raster.numeric// (All [T] [x :- Number, y :- Dual] :- Dual
                             (let [yv (.v y)
                                   yv2 (n/* yv yv)
                                   py (.partials y)
                                   len (alength py)
                                   result (aclone py)]
                               (dotimes [i len]
                                 (aset result i (n/- (n// (n/* x (aget py i)) yv2))))
                               (->Dual (n// x yv) result))))

;; ================================================================
;; Math functions (generic: Double + Dual)
;; ================================================================

;; Register Dual dispatch on raster.math functions (qualified names)
;; so that rrule pullbacks can call through raster.math/* and handle
;; Dual numbers for forward-over-reverse nested gradients.

(deftm raster.math/exp (All [T] [x :- Dual] :- Dual
                            (let [ev (m/exp (.v x))
                                  px (.partials x)
                                  len (alength px)
                                  result (aclone px)]
                              (dotimes [i len]
                                (aset result i (n/* (aget px i) ev)))
                              (->Dual ev result))))

(deftm raster.math/sin (All [T] [x :- Dual] :- Dual
                            (let [cv (m/cos (.v x))
                                  px (.partials x)
                                  len (alength px)
                                  result (aclone px)]
                              (dotimes [i len]
                                (aset result i (n/* (aget px i) cv)))
                              (->Dual (m/sin (.v x)) result))))

(deftm raster.math/cos (All [T] [x :- Dual] :- Dual
                            (let [neg-sv (n/- (m/sin (.v x)))
                                  px (.partials x)
                                  len (alength px)
                                  result (aclone px)]
                              (dotimes [i len]
                                (aset result i (n/* (aget px i) neg-sv)))
                              (->Dual (m/cos (.v x)) result))))

(deftm raster.numeric/sqrt (All [T] [x :- Dual] :- Dual
                                (let [sv (n/sqrt (.v x))
                                      inv2sv (n// 1.0 (n/* 2.0 sv))
                                      px (.partials x)
                                      len (alength px)
                                      result (aclone px)]
                                  (dotimes [i len]
                                    (aset result i (n/* (aget px i) inv2sv)))
                                  (->Dual sv result))))

(deftm raster.math/log (All [T] [x :- Dual] :- Dual
                            (let [inv-xv (n// 1.0 (.v x))
                                  px (.partials x)
                                  len (alength px)
                                  result (aclone px)]
                              (dotimes [i len]
                                (aset result i (n/* (aget px i) inv-xv)))
                              (->Dual (m/log (.v x)) result))))

(deftm raster.numeric/pow (All [T] [x :- Dual, n :- Double] :- Dual
                               (let [xv (.v x)
                                     scale (n/* n (n/pow xv (n/- n 1.0)))
                                     px (.partials x)
                                     len (alength px)
                                     result (aclone px)]
                                 (dotimes [i len]
                                   (aset result i (n/* (aget px i) scale)))
                                 (->Dual (n/pow xv n) result))))

;; pow with an active (Dual) exponent: d/dx x^n = n*x^(n-1), d/dn x^n = x^n ln(x)
(deftm raster.numeric/pow (All [T] [x :- Dual, y :- Dual] :- Dual
                               (let [xv (.v x) yv (.v y)
                                     pv (n/pow xv yv)
                                     dfdx (n/* yv (n/pow xv (n/- yv 1.0)))
                                     dfdy (n/* pv (m/log xv))
                                     px (.partials x) py (.partials y)
                                     len (alength px)
                                     result (aclone px)]
                                 (dotimes [i len]
                                   (aset result i (n/+ (n/* (aget px i) dfdx)
                                                       (n/* (aget py i) dfdy))))
                                 (->Dual pv result))))

(deftm raster.numeric/pow (All [T] [x :- Double, y :- Dual] :- Dual
                               (let [yv (.v y)
                                     pv (n/pow x yv)
                                     dfdy (n/* pv (m/log x))
                                     py (.partials y)
                                     len (alength py)
                                     result (aclone py)]
                                 (dotimes [i len]
                                   (aset result i (n/* (aget py i) dfdy)))
                                 (->Dual pv result))))

;; ================================================================
;; Math functions — carrier-coverage completion (framework §4a/§11):
;; every scalar op with a reverse template gets a Dual lift. Forward
;; mode is rule-free by construction — these are Σ-algebra overloads,
;; NOT template registrations. Same chunked-partials style as above;
;; parametric (All [T]) + n/ ops so Dual{Sym} flows (initiality, O6).
;; ================================================================

(deftm raster.math/tan (All [T] [x :- Dual] :- Dual
                            (let [tv (m/tan (.v x))
                                  scale (n/+ 1.0 (n/* tv tv))
                                  px (.partials x)
                                  len (alength px)
                                  result (aclone px)]
                              (dotimes [i len]
                                (aset result i (n/* (aget px i) scale)))
                              (->Dual tv result))))

(deftm raster.math/asin (All [T] [x :- Dual] :- Dual
                             (let [xv (.v x)
                                   scale (n// 1.0 (n/sqrt (n/- 1.0 (n/* xv xv))))
                                   px (.partials x)
                                   len (alength px)
                                   result (aclone px)]
                               (dotimes [i len]
                                 (aset result i (n/* (aget px i) scale)))
                               (->Dual (m/asin xv) result))))

(deftm raster.math/acos (All [T] [x :- Dual] :- Dual
                             (let [xv (.v x)
                                   scale (n/- (n// 1.0 (n/sqrt (n/- 1.0 (n/* xv xv)))))
                                   px (.partials x)
                                   len (alength px)
                                   result (aclone px)]
                               (dotimes [i len]
                                 (aset result i (n/* (aget px i) scale)))
                               (->Dual (m/acos xv) result))))

(deftm raster.math/atan (All [T] [x :- Dual] :- Dual
                             (let [xv (.v x)
                                   scale (n// 1.0 (n/+ 1.0 (n/* xv xv)))
                                   px (.partials x)
                                   len (alength px)
                                   result (aclone px)]
                               (dotimes [i len]
                                 (aset result i (n/* (aget px i) scale)))
                               (->Dual (m/atan xv) result))))

;; atan2(y, x): d/dy = x/(x²+y²), d/dx = -y/(x²+y²)
(deftm raster.math/atan2 (All [T] [y :- Dual, x :- Dual] :- Dual
                              (let [yv (.v y) xv (.v x)
                                    denom (n/+ (n/* xv xv) (n/* yv yv))
                                    py (.partials y) px (.partials x)
                                    len (alength py)
                                    result (aclone py)]
                                (dotimes [i len]
                                  (aset result i (n// (n/- (n/* (aget py i) xv)
                                                           (n/* (aget px i) yv))
                                                      denom)))
                                (->Dual (m/atan2 yv xv) result))))

(deftm raster.math/atan2 (All [T] [y :- Dual, x :- Number] :- Dual
                              (let [yv (.v y)
                                    denom (n/+ (n/* x x) (n/* yv yv))
                                    py (.partials y)
                                    len (alength py)
                                    result (aclone py)]
                                (dotimes [i len]
                                  (aset result i (n// (n/* (aget py i) x) denom)))
                                (->Dual (m/atan2 yv x) result))))

(deftm raster.math/atan2 (All [T] [y :- Number, x :- Dual] :- Dual
                              (let [xv (.v x)
                                    denom (n/+ (n/* xv xv) (n/* y y))
                                    px (.partials x)
                                    len (alength px)
                                    result (aclone px)]
                                (dotimes [i len]
                                  (aset result i (n/- (n// (n/* (aget px i) y) denom))))
                                (->Dual (m/atan2 y xv) result))))

(deftm raster.math/sinh (All [T] [x :- Dual] :- Dual
                             (let [xv (.v x)
                                   scale (m/cosh xv)
                                   px (.partials x)
                                   len (alength px)
                                   result (aclone px)]
                               (dotimes [i len]
                                 (aset result i (n/* (aget px i) scale)))
                               (->Dual (m/sinh xv) result))))

(deftm raster.math/cosh (All [T] [x :- Dual] :- Dual
                             (let [xv (.v x)
                                   scale (m/sinh xv)
                                   px (.partials x)
                                   len (alength px)
                                   result (aclone px)]
                               (dotimes [i len]
                                 (aset result i (n/* (aget px i) scale)))
                               (->Dual (m/cosh xv) result))))

(deftm raster.math/tanh (All [T] [x :- Dual] :- Dual
                             (let [tv (m/tanh (.v x))
                                   scale (n/- 1.0 (n/* tv tv))
                                   px (.partials x)
                                   len (alength px)
                                   result (aclone px)]
                               (dotimes [i len]
                                 (aset result i (n/* (aget px i) scale)))
                               (->Dual tv result))))

(deftm raster.math/log1p (All [T] [x :- Dual] :- Dual
                              (let [xv (.v x)
                                    scale (n// 1.0 (n/+ 1.0 xv))
                                    px (.partials x)
                                    len (alength px)
                                    result (aclone px)]
                                (dotimes [i len]
                                  (aset result i (n/* (aget px i) scale)))
                                (->Dual (m/log1p xv) result))))

(deftm raster.math/expm1 (All [T] [x :- Dual] :- Dual
                              (let [ev (m/expm1 (.v x))
                                    scale (n/+ ev 1.0)
                                    px (.partials x)
                                    len (alength px)
                                    result (aclone px)]
                                (dotimes [i len]
                                  (aset result i (n/* (aget px i) scale)))
                                (->Dual ev result))))

(deftm raster.math/cbrt (All [T] [x :- Dual] :- Dual
                             (let [cv (m/cbrt (.v x))
                                   scale (n// 1.0 (n/* 3.0 (n/* cv cv)))
                                   px (.partials x)
                                   len (alength px)
                                   result (aclone px)]
                               (dotimes [i len]
                                 (aset result i (n/* (aget px i) scale)))
                               (->Dual cv result))))

;; sqrt/pow on raster.math (issue #55) — same lifts as the raster.numeric ones.
(deftm raster.math/sqrt (All [T] [x :- Dual] :- Dual
                             (let [sv (n/sqrt (.v x))
                                   inv2sv (n// 1.0 (n/* 2.0 sv))
                                   px (.partials x)
                                   len (alength px)
                                   result (aclone px)]
                               (dotimes [i len]
                                 (aset result i (n/* (aget px i) inv2sv)))
                               (->Dual sv result))))

(deftm raster.math/pow (All [T] [x :- Dual, n :- Double] :- Dual
                            (let [xv (.v x)
                                  scale (n/* n (n/pow xv (n/- n 1.0)))
                                  px (.partials x)
                                  len (alength px)
                                  result (aclone px)]
                              (dotimes [i len]
                                (aset result i (n/* (aget px i) scale)))
                              (->Dual (n/pow xv n) result))))

(deftm raster.math/pow (All [T] [x :- Dual, y :- Dual] :- Dual
                            (let [xv (.v x) yv (.v y)
                                  pv (n/pow xv yv)
                                  dfdx (n/* yv (n/pow xv (n/- yv 1.0)))
                                  dfdy (n/* pv (m/log xv))
                                  px (.partials x) py (.partials y)
                                  len (alength px)
                                  result (aclone px)]
                              (dotimes [i len]
                                (aset result i (n/+ (n/* (aget px i) dfdx)
                                                    (n/* (aget py i) dfdy))))
                              (->Dual pv result))))

(deftm raster.math/pow (All [T] [x :- Double, y :- Dual] :- Dual
                            (let [yv (.v y)
                                  pv (n/pow x yv)
                                  dfdy (n/* pv (m/log x))
                                  py (.partials y)
                                  len (alength py)
                                  result (aclone py)]
                              (dotimes [i len]
                                (aset result i (n/* (aget py i) dfdy)))
                              (->Dual pv result))))

(deftm raster.math/log10 (All [T] [x :- Dual] :- Dual
                              (let [xv (.v x)
                                    scale (n// 1.0 (n/* xv 2.302585092994046))
                                    px (.partials x)
                                    len (alength px)
                                    result (aclone px)]
                                (dotimes [i len]
                                  (aset result i (n/* (aget px i) scale)))
                                (->Dual (m/log10 xv) result))))

;; hypot(x, y): d/dx = x/hypot, d/dy = y/hypot
(deftm raster.math/hypot (All [T] [x :- Dual, y :- Dual] :- Dual
                              (let [xv (.v x) yv (.v y)
                                    hv (m/hypot xv yv)
                                    px (.partials x) py (.partials y)
                                    len (alength px)
                                    result (aclone px)]
                                (dotimes [i len]
                                  (aset result i (n// (n/+ (n/* (aget px i) xv)
                                                           (n/* (aget py i) yv))
                                                      hv)))
                                (->Dual hv result))))

(deftm raster.math/hypot (All [T] [x :- Dual, y :- Number] :- Dual
                              (let [xv (.v x)
                                    hv (m/hypot xv y)
                                    px (.partials x)
                                    len (alength px)
                                    result (aclone px)]
                                (dotimes [i len]
                                  (aset result i (n// (n/* (aget px i) xv) hv)))
                                (->Dual hv result))))

(deftm raster.math/hypot (All [T] [x :- Number, y :- Dual] :- Dual
                              (let [yv (.v y)
                                    hv (m/hypot x yv)
                                    py (.partials y)
                                    len (alength py)
                                    result (aclone py)]
                                (dotimes [i len]
                                  (aset result i (n// (n/* (aget py i) yv) hv)))
                                (->Dual hv result))))

;; ================================================================
;; min/max — primal-select (branch on .v, like the comparisons above)
;; + partial routing: the winner's partials flow, the loser's are
;; dropped. Tie-breaking mirrors the reverse templates (min: x on
;; ties via <=; max: x on ties via >=) so both modes agree at kinks.
;; ================================================================

(deftm raster.numeric/min [x :- Dual, y :- Dual] :- Dual
  (if (clojure.core/<= (.v x) (.v y)) x y))

(deftm raster.numeric/min [x :- Dual, y :- Double] :- Dual
  (if (clojure.core/<= (.v x) y) x (->Dual y (zero-partials (.partials x)))))

(deftm raster.numeric/min [x :- Double, y :- Dual] :- Dual
  (if (clojure.core/<= x (.v y)) (->Dual x (zero-partials (.partials y))) y))

(deftm raster.numeric/max [x :- Dual, y :- Dual] :- Dual
  (if (clojure.core/>= (.v x) (.v y)) x y))

(deftm raster.numeric/max [x :- Dual, y :- Double] :- Dual
  (if (clojure.core/>= (.v x) y) x (->Dual y (zero-partials (.partials x)))))

(deftm raster.numeric/max [x :- Double, y :- Dual] :- Dual
  (if (clojure.core/>= x (.v y)) (->Dual x (zero-partials (.partials y))) y))

;; ================================================================
;; Special functions (raster.sci.special) — concrete Double carriers
;; (the primal calls the double-only special-function implementations)
;; ================================================================

;; erf: d/dx = 2/sqrt(pi) * exp(-x²)
(deftm raster.sci.special/erf [x :- Dual] :- Dual
  (let [xv (.v x)
        scale (n/* 1.1283791670955126 (m/exp (n/- (n/* xv xv))))
        px (.partials x)
        len (alength px)
        result (aclone px)]
    (dotimes [i len]
      (aset result i (n/* (aget px i) scale)))
    (->Dual (special/erf xv) result)))

;; erfc = 1 - erf: d/dx = -2/sqrt(pi) * exp(-x²)
(deftm raster.sci.special/erfc [x :- Dual] :- Dual
  (let [xv (.v x)
        scale (n/* -1.1283791670955126 (m/exp (n/- (n/* xv xv))))
        px (.partials x)
        len (alength px)
        result (aclone px)]
    (dotimes [i len]
      (aset result i (n/* (aget px i) scale)))
    (->Dual (special/erfc xv) result)))

;; erfinv: d/dy = sqrt(pi)/2 * exp(erfinv(y)²)
(deftm raster.sci.special/erfinv [y :- Dual] :- Dual
  (let [rv (special/erfinv (.v y))
        scale (n/* 0.8862269254527579 (m/exp (n/* rv rv)))
        py (.partials y)
        len (alength py)
        result (aclone py)]
    (dotimes [i len]
      (aset result i (n/* (aget py i) scale)))
    (->Dual rv result)))

;; lgamma: d/dx = digamma(x)
(deftm raster.sci.special/lgamma [x :- Dual] :- Dual
  (let [xv (.v x)
        scale (special/digamma xv)
        px (.partials x)
        len (alength px)
        result (aclone px)]
    (dotimes [i len]
      (aset result i (n/* (aget px i) scale)))
    (->Dual (special/lgamma xv) result)))

;; digamma: d/dx = trigamma(x)
(deftm raster.sci.special/digamma [x :- Dual] :- Dual
  (let [xv (.v x)
        scale (special/trigamma xv)
        px (.partials x)
        len (alength px)
        result (aclone px)]
    (dotimes [i len]
      (aset result i (n/* (aget px i) scale)))
    (->Dual (special/digamma xv) result)))

;; expit (sigmoid): d/dx = s(x)(1 - s(x))
(deftm raster.sci.special/expit [x :- Dual] :- Dual
  (let [sv (special/expit (.v x))
        scale (n/* sv (n/- 1.0 sv))
        px (.partials x)
        len (alength px)
        result (aclone px)]
    (dotimes [i len]
      (aset result i (n/* (aget px i) scale)))
    (->Dual sv result)))

;; logit: d/dx = 1/(x(1-x))
(deftm raster.sci.special/logit [x :- Dual] :- Dual
  (let [xv (.v x)
        scale (n// 1.0 (n/* xv (n/- 1.0 xv)))
        px (.partials x)
        len (alength px)
        result (aclone px)]
    (dotimes [i len]
      (aset result i (n/* (aget px i) scale)))
    (->Dual (special/logit xv) result)))

;; log1pexp (softplus): d/dx = expit(x)
(deftm raster.sci.special/log1pexp [x :- Dual] :- Dual
  (let [xv (.v x)
        scale (special/expit xv)
        px (.partials x)
        len (alength px)
        result (aclone px)]
    (dotimes [i len]
      (aset result i (n/* (aget px i) scale)))
    (->Dual (special/log1pexp xv) result)))

;; besselj0: d/dx J0(x) = -J1(x)
(deftm raster.sci.special/besselj0 [x :- Dual] :- Dual
  (let [xv (.v x)
        scale (n/- (special/besselj1 xv))
        px (.partials x)
        len (alength px)
        result (aclone px)]
    (dotimes [i len]
      (aset result i (n/* (aget px i) scale)))
    (->Dual (special/besselj0 xv) result)))

;; besselj1: d/dx J1(x) = J0(x) - J1(x)/x
(deftm raster.sci.special/besselj1 [x :- Dual] :- Dual
  (let [xv (.v x)
        scale (n/- (special/besselj0 xv) (n// (special/besselj1 xv) xv))
        px (.partials x)
        len (alength px)
        result (aclone px)]
    (dotimes [i len]
      (aset result i (n/* (aget px i) scale)))
    (->Dual (special/besselj1 xv) result)))

;; betainc: differentiable in x only (a, b parameters — same contract
;; as the reverse template): d/dx I_x(a,b) = x^(a-1)(1-x)^(b-1)/B(a,b)
(deftm raster.sci.special/betainc [a :- Double, b :- Double, x :- Dual] :- Dual
  (let [xv (.v x)
        scale (n// (n/* (n/pow xv (n/- a 1.0))
                        (n/pow (n/- 1.0 xv) (n/- b 1.0)))
                   (special/beta a b))
        px (.partials x)
        len (alength px)
        result (aclone px)]
    (dotimes [i len]
      (aset result i (n/* (aget px i) scale)))
    (->Dual (special/betainc a b xv) result)))

;; Re-export math functions for backward compatibility.
;; After loading raster.ad, these dispatch to raster.math/* or raster.numeric/*
;; which now handle both Double and Dual.
(def ^{:doc "sin dispatching to raster.math/sin (handles Double and Dual)."} sin raster.math/sin)
(def ^{:doc "cos dispatching to raster.math/cos (handles Double and Dual)."} cos raster.math/cos)
(def ^{:doc "exp dispatching to raster.math/exp (handles Double and Dual)."} exp raster.math/exp)
(def ^{:doc "log dispatching to raster.math/log (handles Double and Dual)."} log raster.math/log)
(def ^{:doc "sqrt dispatching to raster.numeric/sqrt (handles Double and Dual)."} sqrt raster.numeric/sqrt)
(def ^{:doc "pow dispatching to raster.numeric/pow (handles Double and Dual)."} pow raster.numeric/pow)

(deftm raster.numeric/abs [x :- Dual] :- Dual
  (let [xv (.v x)]
    (if (>= xv 0.0)
      x
      (let [px (.partials x)]
        (->Dual (clojure.core/- xv)
                (broadcast [px] (clojure.core/- px)))))))

(deftm raster.numeric/sign [x :- Dual] :- Dual
  (->Dual (m/signum (.v x)) (zero-partials (.partials x))))

(deftm raster.numeric/zero? [x :- Dual]
  (== 0.0 (.v x)))

(deftm raster.numeric/zero [x :- Dual] :- Dual
  (->Dual 0.0 (zero-partials (.partials x))))

(deftm raster.numeric/one [x :- Dual] :- Dual
  (->Dual 1.0 (zero-partials (.partials x))))

(deftm raster.numeric/real-value [x :- Dual] :- Double
  (.v ^Dual x))

(deftm raster.numeric/derivative-value [x :- Dual] :- Double
  (aget (.partials ^Dual x) 0))

;; ================================================================
;; Comparisons — branch selection on the PRIMAL value (standard
;; forward-AD semantics; ForwardDiff.jl does the same). The derivative
;; of the branch indicator is zero a.e., so comparing .v is exactly
;; the "differentiate the taken branch" rule (a.e.-correct at kinks —
;; see .internal/ad_formal_framework.md §8). Without these, forward
;; mode over ANY branching fn throws (laws O1 gap, found 2026-07-04).
;; ================================================================

(deftm raster.numeric/> [x :- Dual, y :- Dual] :- Boolean (clojure.core/> (.v x) (.v y)))
(deftm raster.numeric/> [x :- Dual, y :- Double] :- Boolean (clojure.core/> (.v x) y))
(deftm raster.numeric/> [x :- Double, y :- Dual] :- Boolean (clojure.core/> x (.v y)))

(deftm raster.numeric/< [x :- Dual, y :- Dual] :- Boolean (clojure.core/< (.v x) (.v y)))
(deftm raster.numeric/< [x :- Dual, y :- Double] :- Boolean (clojure.core/< (.v x) y))
(deftm raster.numeric/< [x :- Double, y :- Dual] :- Boolean (clojure.core/< x (.v y)))

(deftm raster.numeric/>= [x :- Dual, y :- Dual] :- Boolean (clojure.core/>= (.v x) (.v y)))
(deftm raster.numeric/>= [x :- Dual, y :- Double] :- Boolean (clojure.core/>= (.v x) y))
(deftm raster.numeric/>= [x :- Double, y :- Dual] :- Boolean (clojure.core/>= x (.v y)))

(deftm raster.numeric/<= [x :- Dual, y :- Dual] :- Boolean (clojure.core/<= (.v x) (.v y)))
(deftm raster.numeric/<= [x :- Dual, y :- Double] :- Boolean (clojure.core/<= (.v x) y))
(deftm raster.numeric/<= [x :- Double, y :- Dual] :- Boolean (clojure.core/<= x (.v y)))

;; ================================================================
;; Gradient control
;; ================================================================

(defn stop-gradient
  "Identity in the forward pass, but blocks gradient flow in both modes.
  Forward AD: strips Dual partials to zero. Reverse AD: template emits d_x = 0.
  (stop-gradient x) => x (forward), d/dx = 0 (backward)"
  [x]
  (if (instance? Dual x)
    (->Dual (.v ^Dual x) (zero-partials (.partials ^Dual x)))
    x))

(defn straight-through
  "Apply f to x in the forward pass, but pass gradients through as if
  f were the identity. Useful for non-differentiable ops like rounding.
  Forward AD: applies f to primal, keeps Dual partials.
  Reverse AD: template emits d_x = dy.
  (straight-through #(Math/floor %) x) => (Math/floor x) (forward), d/dx = 1 (backward)"
  [f x]
  (if (instance? Dual x)
    (->Dual (double (f (.v ^Dual x))) (.partials ^Dual x))
    (f x)))

;; ================================================================
;; Derivative computation (single variable — backward compatible)
;; ================================================================

(defn derivative
  "Compute f'(x) using forward-mode AD.
   f must use raster.numeric arithmetic and/or raster.ad math functions.

   (derivative (fn [x] (* x x x)) 2.0) ;=> 12.0"
  [f ^double x]
  (let [result (f (->Dual x (double-array [1.0])))]
    (if (instance? Dual result)
      (aget (.partials ^Dual result) 0)
      0.0)))

;; ================================================================
;; Type promotion registration
;; ================================================================

(defval TDual)
(promote/register-type-tag! Dual (->TDual))
(promote/register-promote-rule! Long Dual Dual)
(promote/register-promote-rule! Double Dual Dual)

(deftm raster.types.promote/convert [t :- TDual, x :- Dual] :- Dual x)
(deftm raster.types.promote/convert [t :- TDual, x :- Number] :- Dual
  (->Dual (.doubleValue ^Number x) (double-array [0.0])))

;; ================================================================
;; Chunk-based gradient
;; ================================================================

(defn gradient
  "Compute ∇f(x) for f: Rⁿ → R using forward-mode AD.
   x is a sequential collection of numbers.
   Returns a double-array of partial derivatives.

   Uses chunk-based forward AD: ceil(n/chunk-size) passes instead of n."
  ([f x] (gradient f x DEFAULT_CHUNK))
  ([f x ^long chunk-size]
   (let [n (count x)
         x-vals (double-array (map double x))
         grad (double-array n)]
     (loop [offset 0]
       (when (< offset n)
         (let [c (min chunk-size (clojure.core/- n offset))
               ;; Build seeded input: each element is a Dual with c partials
               ;; Only element [offset+j] gets seed 1.0 in position j
               x-dual (let [arr (object-array n)]
                        (dotimes [i n]
                          (let [ps (double-array c)]
                             ;; Set seed: if this element is in the current chunk
                            (let [j (clojure.core/- i offset)]
                              (when (and (>= j 0) (< j c))
                                (aset ps j 1.0)))
                            (aset arr i (->Dual (aget x-vals i) ps))))
                        (vec arr))
               result (f x-dual)]
           ;; Extract partials from result
           (when (instance? Dual result)
             (let [rp (.partials ^Dual result)]
               (dotimes [j c]
                 (aset grad (clojure.core/+ offset j) (aget rp j)))))
           (recur (clojure.core/+ offset c)))))
     grad)))

;; ================================================================
;; Jacobian
;; ================================================================

(defn jacobian
  "Compute J_f(x) for f: Rⁿ → Rᵐ. Returns double-array [m×n] row-major.
   f should return a vector/seq of Dual numbers."
  ([f x] (jacobian f x DEFAULT_CHUNK))
  ([f x ^long chunk-size]
   (let [n (count x)
         x-vals (double-array (map double x))
         ;; First pass to determine m
         result0 (f (vec (map #(->Dual % (double-array 1)) x-vals)))
         m (count result0)
         jac (double-array (clojure.core/* m n))]
     (loop [offset 0]
       (when (< offset n)
         (let [c (min chunk-size (clojure.core/- n offset))
               x-dual (let [arr (object-array n)]
                        (dotimes [i n]
                          (let [ps (double-array c)
                                j (clojure.core/- i offset)]
                            (when (and (>= j 0) (< j c))
                              (aset ps j 1.0))
                            (aset arr i (->Dual (aget x-vals i) ps))))
                        (vec arr))
               result (f x-dual)]
           (dotimes [row m]
             (let [elem (nth result row)]
               (when (instance? Dual elem)
                 (let [rp (.partials ^Dual elem)]
                   (dotimes [j c]
                     (aset jac (clojure.core/+ (clojure.core/* row n) offset j)
                           (aget rp j)))))))
           (recur (clojure.core/+ offset c)))))
     jac)))
