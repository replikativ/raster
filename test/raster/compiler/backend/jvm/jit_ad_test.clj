(ns raster.compiler.backend.jvm.jit-ad-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm broadcast]]
            [raster.numeric :as numeric]
            [raster.arrays :as arrays]
            [raster.ad.reverse :as reverse]
            [raster.compiler.passes.scalar.inline :as inline]))

(deftm square-loss [x :- Double] :- Double (* x x))

(deftm overloaded-loss [x :- Float] :- Float (numeric/* x x))
(deftm overloaded-loss [x :- Double] :- Double (numeric/+ (numeric/* x x) x))

(deftest ad-overload-selection-retains-let-bound-source-type
  (let [source '(let* [local (clojure.core/float x)
                      vg ((raster.ad.reverse/value+grad
                           (var raster.compiler.backend.jvm.jit-ad-test/overloaded-loss)) local)
                      gradient (clojure.core/nth vg 1)] gradient)
        transformed (inline/inline-ad-applications source {'x 'double})
        f (eval (list 'fn '[x] transformed))]
    (is (= (float 6.0) (f 3.0)))
    (is (instance? Float (f 3.0)))))

(deftm direct-gradient [x :- Double] :- Double
  (let [vg ((raster.ad.reverse/value+grad
             (var raster.compiler.backend.jvm.jit-ad-test/square-loss)) x)
        gradient (nth vg 1)]
    gradient))

(deftm float-source-gradient [x :- Float] :- Double
  (let [vg ((raster.ad.reverse/value+grad
             (var raster.compiler.backend.jvm.jit-ad-test/square-loss)) (double x))
        gradient (nth vg 1)]
    gradient))

(deftm array-loss [x :- (Array float)] :- Float
  (numeric/* (arrays/aget x 0) (arrays/aget x 0)))

(deftm unannotated-array-gradient [x :- (Array float) scale :- Long] :- (Array float)
  (let [vg ((raster.ad.reverse/value+grad
             (var raster.compiler.backend.jvm.jit-ad-test/array-loss)) x)
        gradient (nth vg 1)]
    (broadcast [gradient] (numeric/* gradient (float scale)))))

(deftest lazy-jit-retains-array-gradient-through-unannotated-projection
  (with-redefs [reverse/value+grad
                (fn [& _] (throw (ex-info "runtime AD construction reached" {})))
                clojure.core/aget
                (fn [& _] (throw (ex-info "boxed array read reached" {})))]
    (is (= [12.0 0.0 0.0]
           (vec (unannotated-array-gradient (float-array [3.0 7.0 9.0]) 2))))))

(deftest lazy-jit-normalizes-direct-ad-without-runtime-construction
  (with-redefs [reverse/value+grad
                (fn [& _] (throw (ex-info "runtime AD construction reached" {})))]
    (doseq [x [0.0 -3.25 1.0000000000000002]]
      (is (= (Double/doubleToRawLongBits (* 2.0 x))
             (Double/doubleToRawLongBits (direct-gradient x)))))
    (doseq [x [(float 1.0000001) (float -0.125) (float 123.456)]]
      (is (= (Double/doubleToRawLongBits (* 2.0 (double x)))
             (Double/doubleToRawLongBits (float-source-gradient x)))))))

(deftest ad-only-normalization-preserves-unrelated-call-boundaries
  (doseq [source ['(let* [answer (ordinary-helper x)] answer)
                  '(let* [f (raster.core/ftm [x :- Double] :- Double (* x x))
                          answer (f x)] answer)
                  '(let* [answer (if take? ((raster.ad.reverse/value+grad
                                            (var raster.compiler.backend.jvm.jit-ad-test/square-loss))
                                           (draw)) :not-taken)] answer)
                  '(loop* [i 0] (if (< i 3) (recur (inc i)) i))]]
    (is (= source (inline/inline-ad-applications source {'x 'double})))))

(deftest ad-tuple-projection-does-not-cross-shadowing-or-quotation
  (let [nested '(let* [vg [:shadowed]] (clojure.core/nth vg 0))
        caught '(try (throw (ex-info "caught" {}))
                     (catch clojure.lang.ExceptionInfo vg (clojure.core/nth vg 0)))
        quoted '(quote (clojure.core/nth vg 0))
        source (list 'let* ['vg '((raster.ad.reverse/value+grad
                                  (var raster.compiler.backend.jvm.jit-ad-test/square-loss)) x)
                           'nested nested 'caught caught 'quoted quoted]
                     '[nested caught quoted])
        transformed (inline/inline-ad-applications source {'x 'double})
        bindings (into {} (map vec (partition 2 (second transformed))))]
    (is (= nested (get bindings 'nested)))
    (is (= caught (get bindings 'caught)))
    (is (= quoted (get bindings 'quoted)))))
