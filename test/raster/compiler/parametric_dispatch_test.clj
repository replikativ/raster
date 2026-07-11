(ns raster.compiler.parametric-dispatch-test
  "A parametric (All [T]) deftm specialized at float must wire its compiled impl
  into the typed dispatch object — not fall back to untyped dispatch.

  Regression for: parametric-specialize! registers the impl *var* as the typed
  impl; make-typed-dispatch set the typed field directly to that Var and threw
  ('can not set IFn__floats… field to clojure.lang.Var'), silently falling back
  to a plain dispatch fn. The float kernel then ran interpreted (~30x slower) —
  e.g. UMAP's cos-dist/init-leaves over float data. Fix derefs the Var so the
  field holds the compiled IFn__ instance."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.core :refer [deftm]]
            ;; deftm bodies use raster.numeric/raster.arrays ops — load them so a
            ;; single-ns test run can bytecode-compile the specializations.
            [raster.numeric]
            [raster.arrays]
            [raster.par]))

(deftm pdot
  "Parametric dot product — instantiated per element type."
  (All [T] [a :- (Array T) b :- (Array T) n :- Long] :- Double
       (raster.par/reduce acc 0.0 i n
                          (raster.numeric/+ acc (raster.numeric/* (raster.arrays/aget a i)
                                                                  (raster.arrays/aget b i))))))

(deftest parametric-float-wires-typed-dispatch
  (testing "float specialization wires into the typed dispatch object (not untyped fallback)"
    (let [af (float-array [1.0 2.0 3.0 4.0])
          bf (float-array [1.0 1.0 1.0 1.0])
          ad (double-array [1.0 2.0 3.0 4.0])
          bd (double-array [1.0 1.0 1.0 1.0])]
      ;; trigger both specializations
      (is (== 10.0 (pdot af bf 4)) "float dot correct")
      (is (== 10.0 (pdot ad bd 4)) "double dot correct")
      ;; After specialization the generic var must hold a typed dispatch instance
      ;; (raster.dispatch.D_*). The pre-fix bug left a plain dispatch fn because
      ;; make-typed-dispatch threw on the float (Var) field and fell back.
      (let [cls (.getName (class @#'pdot))]
        (is (re-find #"raster\.dispatch\.D_" cls)
            (str "pdot dispatch is " cls " — expected a typed dispatch class; a plain "
                 "fn means float wiring fell back to untyped (interpreted) dispatch")))
      ;; And the typed dispatch must implement the float IFn__ interface so a
      ;; compiled .invk caller reaches the fast path.
      (let [ifaces (set (map #(.getName %) (.getInterfaces (class @#'pdot))))]
        (is (some #(re-find #"IFn__floats" %) ifaces)
            (str "pdot dispatch interfaces " ifaces " — missing IFn__floats* (float .invk path)"))))))

;; 22 params — past Clojure's 20-positional-param limit. Regression for the
;; parametric-specialize! placeholder defn: it eval'd a POSITIONAL param vector,
;; so any (All [T]) deftm with >20 params (e.g. a 37-param transformer block)
;; died at specialization with "Can't specify more than 20 params". The
;; placeholder is variadic now (the real params live in ::deftm-params meta and
;; the bytecode impl carries the typed signature).
(deftm psum22
  "Sum 20 arrays element-wise over [0,n) — a >20-param parametric kernel."
  (All [T] [a1 :- (Array T) a2 :- (Array T) a3 :- (Array T) a4 :- (Array T)
            a5 :- (Array T) a6 :- (Array T) a7 :- (Array T) a8 :- (Array T)
            a9 :- (Array T) a10 :- (Array T) a11 :- (Array T) a12 :- (Array T)
            a13 :- (Array T) a14 :- (Array T) a15 :- (Array T) a16 :- (Array T)
            a17 :- (Array T) a18 :- (Array T) a19 :- (Array T) a20 :- (Array T)
            n :- Long scale :- T] :- Double
       (raster.par/reduce
        acc 0.0 i n
        (raster.numeric/+
         acc
         (raster.numeric/*
          scale
          (raster.numeric/+ (raster.arrays/aget a1 i) (raster.arrays/aget a2 i)
                            (raster.arrays/aget a3 i) (raster.arrays/aget a4 i)
                            (raster.arrays/aget a5 i) (raster.arrays/aget a6 i)
                            (raster.arrays/aget a7 i) (raster.arrays/aget a8 i)
                            (raster.arrays/aget a9 i) (raster.arrays/aget a10 i)
                            (raster.arrays/aget a11 i) (raster.arrays/aget a12 i)
                            (raster.arrays/aget a13 i) (raster.arrays/aget a14 i)
                            (raster.arrays/aget a15 i) (raster.arrays/aget a16 i)
                            (raster.arrays/aget a17 i) (raster.arrays/aget a18 i)
                            (raster.arrays/aget a19 i) (raster.arrays/aget a20 i)))))))

(deftest parametric-22-param-specializes-and-runs
  (testing "a 22-param (All [T]) deftm specializes + runs at float (no 20-param wall)"
    (let [n 3
          fas (mapv (fn [k] (float-array (repeat n (float (inc k))))) (range 20))
          ;; sum over 20 arrays of constant (k+1): per-element 1+2+...+20 = 210; n=3 → 630
          expect (double (* n 210))]
      (is (== expect (double (apply psum22 (into fas [n (float 1.0)]))))
          "float specialization of a 22-param parametric deftm runs")
      (let [das (mapv (fn [k] (double-array (repeat n (double (inc k))))) (range 20))]
        (is (== expect (double (apply psum22 (into das [n 1.0]))))
            "double specialization runs too")))))
