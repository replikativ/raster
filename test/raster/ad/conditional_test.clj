(ns raster.ad.conditional-test
  "Lexical conditional AD: primal evaluation, selected residuals, and derivative oracles."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.arrays :as ra]
            [raster.compiler.equation-first :as equation-first]
            [raster.ad.reverse :as rev]
            [raster.ad.jvp :as jvp]
            [raster.ad.reverse.normalize :as anf]))

(deftm conditional-root [x :- Double] :- Double
  (if (n/> x 0.0) (n/sqrt x) (n/sqrt (n/- x))))

(deftm conditional-core-name [count :- Double] :- Double
  (if (n/> count 0.0) (n/sqrt count) (n/sqrt (n/- count))))

(deftm conditional-root-local [x :- Double] :- Double
  (let [result (if (n/> x 0.0)
                 (let [argument x] (n/sqrt argument))
                 (let [argument (n/- x)] (n/sqrt argument)))]
    (n/* result result)))

(deftm conditional-read [x :- Double, observations :- (Array double)] :- Double
  (if (n/> x 0.0) (n/* x x) (aget observations 0)))

(deftm conditional-loaded-scale [x :- Double, values :- (Array double)] :- Double
  (if (n/> x 0.0) (n/* x (aget values 0)) (n/sqrt (n/- x))))

(deftm conditional-nested [x :- Double] :- Double
  (if (n/> x 0.0)
    (if (n/> x 2.0) (n/sqrt x) (n/* x x))
    (n/sqrt (n/- x))))

(deftm conditional-shadow [x :- Double, y :- Double] :- Double
  (if (n/> x 0.0) (let [y (n/* x x)] y) y))

(deftm conditional-shadow-constant [x :- Double, y :- Double] :- Double
  (if (n/> x 0.0) (let [y 7.0] y) y))

(deftm conditional-gradient-fill!
  [x :- Double, out :- (Array double), count :- Long] :- Void
  (let [vg ((rev/value+grad #'conditional-root) x)
        gradient (nth vg 1)]
    (par/map-void! i count (ra/aset out i gradient))))

(defn- close? [a b]
  (and (Double/isFinite (double a))
       (< (Math/abs (- (double a) (double b))) 1.0e-8)))

(deftest conditional-normalization-retains-arm-evaluation
  (let [source '(if decision (Math/sqrt x) (aget values 0))
        [bindings result] (anf/normalize-for-ad [] [source] gensym)]
    (is (= 2 (count bindings)))
    (is (= source (second bindings)))
    (is (= result (first bindings)))))

(deftest normalization-preserves-retained-call-identity
  (let [binding (with-meta 'predicate {:raster.op/original 'raster.numeric/>})
        normalized (anf/anf-normalize-bindings
                     [binding '(.invk comparison-impl x 0.0)] gensym)]
    (is (= 'raster.numeric/> (:raster.op/original (meta (second normalized)))))
    (is (= '(.invk comparison-impl x 0.0) (second normalized)))))

(deftest nested-conditional-residuals-remain-lexical
  (doseq [[x expected] [[4.0 0.25] [1.0 2.0] [-4.0 -0.25]]]
    (let [[value gradient] ((rev/value+grad #'conditional-nested) x)
          [primal tangent] ((jvp/jvp #'conditional-nested) x 1.0)]
      (is (= (conditional-nested x) value primal))
      (is (close? gradient expected))
      (is (close? tangent expected)))))

(deftest branch-local-binders-do-not-inherit-outer-cotangents
  (let [[value dx dy] ((rev/value+grad #'conditional-shadow) 2.0 9.0)
        [primal tangent] ((jvp/jvp #'conditional-shadow) 2.0 9.0 0.0 1.0)
        [constant cdx cdy] ((rev/value+grad #'conditional-shadow-constant) 2.0 9.0)
        [constant-primal constant-tangent]
        ((jvp/jvp #'conditional-shadow-constant) 2.0 9.0 0.0 1.0)]
    (is (= 4.0 value primal))
    (is (= 4.0 dx))
    (is (= 0.0 dy tangent))
    (is (= 7.0 constant constant-primal))
    (is (= 0.0 cdx cdy constant-tangent))))

(deftest inactive-partial-math-does-not-poison-gradients
  (doseq [[x expected] [[4.0 0.25] [-4.0 -0.25]]]
    (let [[value gradient] ((rev/value+grad #'conditional-root) x)
          [primal tangent] ((jvp/jvp #'conditional-root) x 1.0)
          [gradients hv] ((jvp/hvp #'conditional-root) x 1.0)
          h 1.0e-5
          fd (/ (- (conditional-root (+ x h)) (conditional-root (- x h))) (* 2 h))]
      (is (= (conditional-root x) value primal))
      (is (close? gradient expected))
      (is (close? gradient fd))
      (is (close? tangent expected))
      (is (close? (first gradients) expected))
      (is (close? (first hv) -0.03125)))))

(deftest declared-parameters-can-shadow-core-var-names
  (doseq [[x expected] [[4.0 0.25] [-4.0 -0.25]]]
    (let [[value gradient] ((rev/value+grad #'conditional-core-name) x)
          [primal tangent] ((jvp/jvp #'conditional-core-name) x 1.0)
          [gradients hv] ((jvp/hvp #'conditional-core-name) x 1.0)]
      (is (= 2.0 value primal))
      (is (close? gradient expected))
      (is (close? tangent expected))
      (is (close? (first gradients) expected))
      (is (close? (first hv) -0.03125)))))

(deftest conditional-let-and-composition-preserve-scopes
  (doseq [[x expected] [[4.0 1.0] [-4.0 -1.0]]]
    (let [[value gradient] ((rev/value+grad #'conditional-root-local) x)]
      (is (= (conditional-root-local x) value))
      (is (close? gradient expected)))))

(deftest inactive-checked-read-is-not-executed
  (let [data (double-array 0)
        [value gradient data-gradient]
        ((rev/value+grad #'conditional-read :wrt [0]) 2.0 data)]
    (is (= 4.0 (conditional-read 2.0 data) value))
    (is (= 4.0 gradient))
    (is (nil? data-gradient))))

(deftest selected-residual-does-not-replay-primal-reads
  (let [data (double-array [3.0])
        [value pullback] (rev/vjp #'conditional-loaded-scale 4.0 data)]
    (aset data 0 7.0)
    (let [[gradient data-gradient] (pullback 1.0)]
      (is (= 12.0 value))
      (is (= 3.0 gradient) "pullback consumes the saved value, not the modified array")
      (is (= [4.0] (vec data-gradient))))))

(deftest conditional-residual-cannot-be-guessed-into-a-gpu-scalar
  (let [output (double-array 2)]
    (conditional-gradient-fill! 4.0 output 2)
    (is (= [0.25 0.25] (vec output)))
    (let [decline (try (equation-first/compile #'conditional-gradient-fill!
                                             {:target :ocl:0 :dtype :double})
                       nil
                       (catch clojure.lang.ExceptionInfo e (ex-data e)))]
      (is (= :ad-conditional-residual-not-lowered (:reason decline)))
      (is (= 'if (first (:source-form decline)))))))
