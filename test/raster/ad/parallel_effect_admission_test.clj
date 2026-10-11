(ns raster.ad.parallel-effect-admission-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.ad.reverse :as rev]
            [raster.ad.jvp :as jvp]
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
