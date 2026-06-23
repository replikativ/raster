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
            [raster.core :refer [deftm]]))

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
