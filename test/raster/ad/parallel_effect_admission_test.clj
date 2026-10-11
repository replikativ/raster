(ns raster.ad.parallel-effect-admission-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.ad.reverse :as rev]
            [raster.ad.jvp :as jvp]
            [raster.ad.templates :as tmpl]
            [raster.arrays :as ra]
            [raster.numeric :as n]
            [raster.par :as par]))

(deftm active-effect-square [x :- Double size :- Long] :- Double
  (let [out (double-array size)]
    (par/map-void! i size (ra/aset out i (n/* x x)))
    (par/reduce total 0.0 i size (n/+ total (ra/aget out i)))))

(deftm core-spelled-active-effect-square [count :- Double size :- Long] :- Double
  (let [out (double-array size)]
    (par/map-void! i size (ra/aset out i (n/* count count)))
    (par/reduce total 0.0 i size (n/+ total (ra/aget out i)))))

(deftm inactive-effect-square [h :- Double size :- Long] :- Double
  (let [out (double-array size)]
    (par/map-void! i size (ra/aset out i 4.0))
    (n/+ h (par/reduce total 0.0 i size (n/+ total (ra/aget out i))))))

(deftm excluded-effect-square [h :- Double x :- Double size :- Long] :- Double
  (let [out (double-array size)]
    (par/map-void! i size (ra/aset out i (n/* x x)))
    (n/+ h (par/reduce total 0.0 i size (n/+ total (ra/aget out i))))))

(deftm supported-map-reduce-square [x :- Double size :- Long] :- Double
  (let [out (double-array size)
        mapped (par/map! out i size double (n/* x x))]
    (par/reduce total 0.0 i size (n/+ total (ra/aget mapped i)))))

(deftm supported-scan-sum [x :- Double size :- Long] :- Double
  (let [out (double-array size)
        scanned (par/scan out acc 0.0 i size double (n/+ acc x))]
    (ra/aget scanned (dec size))))

(deftm supported-template-scaled-dot
  [alpha :- Double a :- (Array double) b :- (Array double)] :- Double
  (par/dot-product (par/scale alpha a) b))

(deftest active-parallel-effects-refuse-instead-of-dropping-captured-derivatives
  ;; Independent polynomial/central-difference witness. No generated function
  ;; may return the historically wrong [12, 0] for this source computation.
  (doseq [source [#'active-effect-square #'core-spelled-active-effect-square]]
    (let [x 2.0 size 3 h 0.125
          expected (* size x x)
          derivative (* 2.0 size x)]
      (is (= expected (source x size)))
      (is (= derivative (/ (- (source (+ x h) size) (source (- x h) size))
                           (* 2.0 h))))
      (doseq [construct [rev/value+grad jvp/jvp]]
        (is (= :ad-unsupported-active-parallel-region
               (try (construct source) nil
                    (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))))

(deftest inactive-and-excluded-parallel-effects-remain-primal
  (is (= 14.0 (inactive-effect-square 2.0 3)))
  (is (= [14.0 1.0 nil] ((rev/value+grad #'inactive-effect-square :wrt [0]) 2.0 3)))
  (is (= [14.0 1.0] ((jvp/jvp #'inactive-effect-square) 2.0 3 1.0)))
  (is (= 17.0 (excluded-effect-square 5.0 2.0 3)))
  (is (= [17.0 1.0 nil nil]
         ((rev/value+grad #'excluded-effect-square :wrt [0]) 5.0 2.0 3))))

(deftest existing-scoped-map-reduce-and-scan-rules-remain-admitted
  (doseq [[source expected derivative] [[#'supported-map-reduce-square 12.0 12.0]
                                       [#'supported-scan-sum 6.0 3.0]]]
    (is (= expected (source 2.0 3)))
    (is (= [expected derivative nil] ((rev/value+grad source :wrt [0]) 2.0 3)))
    (is (= [expected derivative] ((jvp/jvp source) 2.0 3 1.0)))))

(deftest declared-parallel-call-templates-remain-differentiable
  (let [a (double-array [1.0 -2.0])
        b (double-array [3.0 4.0])
        seed (double-array [0.5 -1.0])
        dot -5.0
        gradient (rev/reified-grad #'supported-template-scaled-dot 'a)
        [value dalpha da db] ((rev/value+grad #'supported-template-scaled-dot) 2.0 a b)
        [tvalue tangent] ((jvp/jvp #'supported-template-scaled-dot)
                         2.0 a b 1.0 seed (double-array 2))
        [gvalue dc dga dgb dseed] ((rev/value+grad gradient) 2.0 a b seed)]
    ;; Independent coordinates, not an AD-generated comparison oracle.
    (is (= (* 2.0 dot) value tvalue))
    (is (= dot dalpha))
    (is (= [6.0 8.0] (vec da)))
    (is (= [2.0 -4.0] (vec db)))
    (is (= -10.0 tangent))
    ;; <gradient_a, seed> = alpha * sum_i b_i*seed_i.
    (is (= -5.0 gvalue))
    (is (= -2.5 dc))
    (is (or (nil? dga) (= [0.0 0.0] (vec dga))))
    (is (= [1.0 -2.0] (vec dgb)))
    (is (= [6.0 8.0] (vec dseed)))))

(deftest parallel-template-admission-respects-call-identity-and-mode
  (let [classify (ns-resolve 'raster.ad.reverse 'ad-form-kind)
        fold (ns-resolve 'raster.ad.jvp 'jvp-fold)
        expression '(raster.par/dot-product a b)]
    (is (= :call (classify expression true 'result)))
    (let [masked (with-meta expression {:raster.op/lexical-callee true})]
      (is (= :ad-unsupported-active-parallel-region
             (try (classify masked true 'result) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    ;; Direct calls derive identity from their actual head, not dispatch-only
    ;; advisory metadata. Withdrawal of the rule must restore the refusal.
    (with-redefs [tmpl/resolve-template (constantly nil)]
      (is (= :ad-unsupported-active-parallel-region
             (try (classify expression true 'result) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))
    ;; A reverse template alone must never authorize a forward zero tangent.
    (with-redefs [tmpl/op-jvp-fn (constantly nil)]
      (is (= :ad-unsupported-active-parallel-region
             (try (fold ['result expression] {'a 'da}) nil
                  (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))))
