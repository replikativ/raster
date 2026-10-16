(ns raster.ad.source-memory-admission-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as rev]))

(deftm store-read [ys :- (Array double)] :- Double
  (let [ignored (aset ys 0 7.0)] (aget ys 0)))

(deftm active-store-read [x :- Double ys :- (Array double)] :- Double
  (let [ignored (aset ys 0 (n/* x 3.0))] (aget ys 0)))

(deftm map-read [ys :- (Array double)] :- Double
  (let [ignored (par/map! ys i 2 double 7.0)] (aget ys 0)))

(deftm active-map-read [x :- Double ys :- (Array double)] :- Double
  (let [ignored (par/map! ys i 2 double (n/* x 3.0))] (aget ys 0)))

(deftm active-rewrite [x :- Double zs :- (Array double)] :- Double
  (let [ignored (par/map! zs i 2 double (n/* x 3.0))
        rewritten (par/map! zs i 2 double 7.0)] (aget zs 0)))

(deftm active-store-choice
  [x :- Double ys :- (Array double) flag :- Long] :- Double
  (let [ignored (if (= flag 1) (aset ys 0 (n/* x 3.0)) 0.0)] (aget ys 0)))

(deftm excluded-counter
  [x :- Double ys :- (Array double) counter :- (Array int)] :- Double
  (let [ignored (aset counter 0 (int 1))] (n/* x (aget ys 0))))

(deftm active-alias-write [ys :- (Array double) flag :- Long] :- Double
  (let [view ys ignored (if (= flag 1) (aset view 0 7.0) 0.0)]
    (aget ys 0)))

(deftm stale-view-write [x :- Double ys :- (Array double) flag :- Long] :- Double
  (let [view ys
        written (par/map! ys i 2 double (n/* x 3.0))
        ignored (if (= flag 1) (aset view 0 7.0) 0.0)]
    (aget ys 0)))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest writes-to-active-old-versions-refuse-before-effects
  (doseq [[v args wrt] [[#'store-read [(double-array [1.0 2.0])] [0]]
                        [#'map-read [(double-array [1.0 2.0])] [0]]
                        [#'active-map-read [2.0 (double-array [1.0 2.0])] [0 1]]]]
    (let [ys (last args)]
      (is (= :ad-active-storage-write
             (reason #(apply (rev/value+grad v :wrt wrt) args))))
      (is (= [1.0 2.0] (vec ys)))))
  (let [ys (double-array [1.0 2.0])]
    (is (= :ad-unversioned-store
           (reason #((rev/value+grad #'active-store-read :wrt [0]) 2.0 ys))))
    (is (= [1.0 2.0] (vec ys))))
  (is (= :ad-active-storage-write
         (reason #((rev/value+grad #'active-rewrite :wrt [0])
                   2.0 (double-array [1.0 2.0]))))))

(deftest enclosing-active-storage-remains-active-through-child-aliases
  (let [ys (double-array [1.0 2.0])]
    (is (= :ad-active-storage-write
           (reason #((rev/value+grad #'active-alias-write :wrt [0]) ys 1))))
    (is (= [1.0 2.0] (vec ys))))
  ;; The view was bound while inactive. A later SOAC activates its storage,
  ;; so a constant write through the old view must not bypass the version gate.
  (let [ys (double-array [1.0 2.0])]
    (is (= :ad-active-storage-write
           (reason #((rev/value+grad #'stale-view-write :wrt [0]) 2.0 ys 1))))
    (is (= [1.0 2.0] (vec ys)))))

(deftest disjoint-new-output-and-excluded-integer-storage-remain-valid
  (is (= [6.0 3.0 nil]
         ((rev/value+grad #'active-map-read :wrt [0]) 2.0 (double-array 2))))
  (let [counter (int-array 1)]
    (is (= [2.0 1.0 nil nil]
           ((rev/value+grad #'excluded-counter :wrt [0])
            2.0 (double-array [1.0 2.0]) counter)))
    (is (= [1] (vec counter)))))

(deftest active-scalar-stores-in-branches-refuse-before-effects
  (let [ys (double-array [1.0 2.0])]
    (is (= :ad-unversioned-store
           (reason #((rev/value+grad #'active-store-choice :wrt [0]) 2.0 ys 1))))
    (is (= [1.0 2.0] (vec ys)))))

(deftest independent-primal-oracles-distinguish-memory-versions
  (let [h 1.0e-5
        difference (fn [f x] (/ (- (f (+ x h)) (f (- x h))) (* 2.0 h)))]
    (doseq [f [store-read map-read]]
      (is (zero? (difference #(f (double-array [% 2.0])) 1.0))))
    (doseq [f [active-store-read active-map-read]]
      (is (< (Math/abs (- 3.0 (difference #(f % (double-array 2)) 2.0))) 1.0e-8)))))
