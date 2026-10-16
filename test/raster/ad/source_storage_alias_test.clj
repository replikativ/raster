(ns raster.ad.source-storage-alias-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as rev]))

(deftm independent-ports [x :- Double ys :- (Array double) zs :- (Array double)] :- Double
  (let [ignored (par/map! zs i 2 double (n/* x 3.0))] (aget ys 0)))

(deftm prior-view [x :- Double zs :- (Array double)] :- Double
  (let [view zs ignored (par/map! zs i 2 double (n/* x 3.0))] (aget view 0)))

(deftm inactive-choice
  [x :- Double ys :- (Array double) zs :- (Array double) flag :- Long] :- Double
  (let [ignored (if (= flag 1) (par/map! zs i 2 double 7.0) nil)]
    (n/* x (aget ys 0))))

(deftm active-choice
  [x :- Double ys :- (Array double) zs :- (Array double) flag :- Long] :- Double
  (let [ignored (if (= flag 1) (par/map! zs i 2 double (n/* x 3.0)) nil)]
    (n/* x (aget ys 0))))

(deftm read-only-ports [x :- Double ys :- (Array double) zs :- (Array double)] :- Double
  (n/* x (n/+ (aget ys 0) (aget zs 0))))

(deftm single-output-version [x :- Double ys :- (Array double)] :- Double
  (let [ignored (par/map! ys i 2 double (n/* x (aget ys i)))]
    (par/reduce acc 0.0 i 2 (n/+ acc (aget ys i)))))

(deftm single-output-read [x :- Double ys :- (Array double)] :- Double
  (let [ignored (par/map! ys i 2 double (n/* x (aget ys i)))] (aget ys 0)))

(deftm cross-lane-output-read [x :- Double ys :- (Array double)] :- Double
  (let [ignored (par/map! ys i 2 double (n/* x (aget ys 0)))] (aget ys 1)))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest passive-write-activity-uses-positive-alias-and-exact-root-facts
  (let [aliases {'view '#{root} 'root '#{view}}
        facts {'root {:value 'root} 'view {:value 'view :alias 'root}}]
    (is (#'rev/active-source-storage? 'view {'root true 'view false} aliases facts))
    (is (#'rev/active-source-storage? 'view {'root true 'view false} {} facts))
    (is (not (#'rev/active-source-storage? 'view {'root false 'view false} aliases facts))))
  (let [ys (with-meta 'ys {:raster.type/tag 'doubles})
        zs (with-meta 'zs {:raster.type/tag 'doubles})
        scope {:values [ys zs] :value-facts {ys {:value ys} zs {:value zs}}
               :aliases {} :activity {ys true}}]
    (with-bindings {#'rev/*source-memory-context* scope #'rev/*gensym-counter* (atom 0)}
      (is (= 2 (count (#'rev/source-write-disjoint-bindings #{zs} #{} {}))))
      (is (empty? (#'rev/source-write-disjoint-bindings #{} #{} {}))))))

(deftest independent-write-ports-require-runtime-disjointness
  (let [gradient (rev/value+grad #'independent-ports :wrt [0])
        ys (double-array [1.0 2.0]) zs (double-array 2)]
    (is (= [1.0 0.0 nil nil] (gradient 2.0 ys zs)))
    (is (= [6.0 6.0] (vec zs)))
    (is (= :ad-unversioned-storage-alias (reason #(gradient 2.0 ys ys))))
    (is (= [1.0 2.0] (vec ys))))
  (let [call (fn [x] (let [ys (double-array [1.0 2.0])] (independent-ports x ys ys)))
        h 1.0e-5]
    (is (< (Math/abs (- 3.0 (/ (- (call (+ 2.0 h)) (call (- 2.0 h))) (* 2.0 h))))
           1.0e-8)))
  (let [ys (double-array [1.0 2.0])]
    (is (= :ad-unversioned-storage-alias
           (reason #((rev/value+grad #'prior-view :wrt [0]) 2.0 ys))))
    (is (= [1.0 2.0] (vec ys)))))

(deftest inactive-choice-keeps-lawful-passive-alias-writes
  (let [gradient (rev/value+grad #'inactive-choice :wrt [0])
        ys (double-array [1.0 2.0])]
    (is (= [2.0 1.0 nil nil nil] (gradient 2.0 ys ys 0)))
    (is (= [1.0 2.0] (vec ys)))
    (is (= [14.0 7.0 nil nil nil] (gradient 2.0 ys ys 1)))
    (is (= [7.0 7.0] (vec ys)))))

(deftest differentiated-output-guards-follow-the-selected-child
  (let [gradient (rev/value+grad #'active-choice :wrt [0])
        shared (double-array [1.0 2.0])]
    (is (= [2.0 1.0 nil nil nil] (gradient 2.0 shared shared 0)))
    (is (= [1.0 2.0] (vec shared)))
    (is (= :ad-unversioned-storage-alias (reason #(gradient 2.0 shared shared 1))))
    (is (= [1.0 2.0] (vec shared)))
    (let [out (double-array 2)]
      (is (= [2.0 1.0 nil nil nil] (gradient 2.0 shared out 1)))
      (is (= [6.0 6.0] (vec out))))))

(deftest read-only-aliases-remain-valid
  (let [ys (double-array [1.0 2.0])]
    (is (= [4.0 2.0 nil nil]
           ((rev/value+grad #'read-only-ports :wrt [0]) 2.0 ys ys)))))

(deftest own-input-output-storage-needs-version-or-stability-proof
  (doseq [source [#'single-output-read #'cross-lane-output-read]]
    (let [ys (double-array [1.0 2.0])]
      (is (= :ad-unversioned-storage-read-write
             (reason #((rev/value+grad source :wrt [0]) 2.0 ys))))
      (is (= [1.0 2.0] (vec ys)))))
  ;; The CPU map definition traverses in order: lane0 changes lane1's input.
  ;; Neither independent-lane AD nor an index-shaped exemption is a proof.
  (let [h 1.0e-5 call #(cross-lane-output-read % (double-array [1.0 2.0]))]
    (is (= 4.0 (call 2.0)))
    (is (< (Math/abs (- 4.0 (/ (- (call (+ 2.0 h)) (call (- 2.0 h))) (* 2.0 h))))
           1.0e-8))))

(deftest excluded-input-is-not-a-new-version-proof-for-reductions
  ;; The reviewed parent already refused at the downstream reduction because
  ;; its global excluded-array set cannot represent the version change. The
  ;; stability contract now refuses earlier, at the in-place map itself.
  (is (= 6.0 (single-output-version 2.0 (double-array [1.0 2.0]))))
  (is (= :ad-unversioned-storage-read-write
         (reason #((rev/value+grad #'single-output-version :wrt [0])
                   2.0 (double-array [1.0 2.0]))))))
