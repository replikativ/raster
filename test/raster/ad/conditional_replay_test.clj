(ns raster.ad.conditional-replay-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as rev]))

(deftm overwritten [x :- Double, ys :- (Array double), flag :- Long] :- Double
  (let [value (if (= flag 1)
                (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i)))) x)
        ignored (aset ys 0 99.0)]
    value))

(deftm overwritten-alias [x :- Double, ys :- (Array double), flag :- Long] :- Double
  (let [value (if (= flag 1)
                (let [view ys]
                  (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget view i))))) x)
        ignored (aset ys 0 99.0)]
    value))

(deftm nested-choice [x :- Double, ys :- (Array double), flag :- Long] :- Double
  (let [value (if (= flag 1)
                (if (= flag 1)
                  (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i)))) x) x)
        ignored (aset ys 0 99.0)]
    value))

(deftm within-arm-overwrite [x :- Double, ys :- (Array double), flag :- Long] :- Double
  (if (= flag 1)
    (let [view ys
          value (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget view i))))
          ignored (aset ys 0 99.0)] value)
    x))

(deftm earlier-write [x :- Double, ys :- (Array double), flag :- Long] :- Double
  (let [ignored (aset ys 0 3.0)]
    (if (= flag 1)
      (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i)))) x)))

(deftm exclusive-arms [x :- Double, ys :- (Array double), flag :- Long] :- Double
  (if (= flag 1)
    (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))
    (let [view ys ignored (aset view 0 3.0)] x)))

(deftm earlier-replay-later-arm-write [x :- Double, ys :- (Array double), flag :- Long] :- Double
  (let [before (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))
        after (if (= flag 1) x (let [view ys ignored (aset view 0 3.0)] x))]
    (n/+ before after)))

(deftm predicate-write [x :- Double, ys :- (Array double), flag :- Long] :- Double
  (if (do (aset ys 0 3.0) (= flag 1))
    (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i)))) x))

(defn- refusal [v]
  (try (rev/value+grad v :wrt [0]) nil
       (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))

(deftest conditional-residual-replay-is-protected-across-scope-boundaries
  ;; Previously each overwrite variant returned [6,101,nil,nil] rather than
  ;; derivative 3: the conditional dropped the reduction's replay obligation.
  (doseq [v [#'overwritten #'overwritten-alias #'nested-choice #'within-arm-overwrite
             #'earlier-replay-later-arm-write]]
    (is (= :replayed-array-overwritten (refusal v)) (str v)))
  (let [ys (double-array [1.0 2.0])]
    (is (= 6.0 (overwritten 2.0 ys 1)))
    (is (= [99.0 2.0] (vec ys)))))

(deftest mutually-exclusive-arms-do-not-invalidate-each-other
  (doseq [v [#'exclusive-arms #'earlier-write #'predicate-write]]
    (let [gradient (rev/value+grad v :wrt [0])]
      (doseq [flag [1 0]]
        (let [ys (double-array [1.0 2.0])
              source-ys (aclone ys)
              [value dx dy dflag] (gradient 2.0 ys flag)
              coefficient (if (= flag 1) (if (= v #'exclusive-arms) 3.0 5.0) 1.0)]
          (is (= (* 2.0 coefficient) value (@v 2.0 source-ys flag)))
          (is (= coefficient dx))
          (is (= (vec source-ys) (vec ys)))
          (is (nil? dy))
          (is (nil? dflag)))))))

(deftest malformed-region-summaries-and-unclosed-predicates-fail-loudly
  (let [typed-write (with-meta '(.invk write-impl dst 0 1.0)
                     {:raster.op/original 'raster.arrays/aset})]
    (is (= '#{dst} (#'rev/written-syms typed-write)))
    (is (= '#{dst} (#'rev/written-syms '(clojure.core/aset dst 0 1.0)))))
  (is (= :ad-region-effects
         (try (#'rev/record-memory-effects {:effects {:kind :choice :regions [{}]}} nil) nil
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
  (is (= :ad-region-unclosed-test
         (try (#'rev/ad-record :if 'result '(if (predicate x) x x) {'x true}) nil
              (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
