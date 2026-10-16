(ns raster.ad.visible-replay-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as rev]))

(deftm ^:no-inline opaque-buffer [ys :- (Array double)] :- (Array double) ys)
(deftm ^:no-inline clone-buffer [ys :- (Array double)] :- (Array double) (aclone ys))

(deftm visible-opaque-read [x :- Double ys :- (Array double)] :- Double
  (let [source (opaque-buffer ys)
        value (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget source i))))
        ignored (aset ys 0 99.0)] value))

(deftm visible-clone-read [x :- Double ys :- (Array double)] :- Double
  (let [source (clone-buffer ys)
        value (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget source i))))
        ignored (aset ys 0 99.0)] value))

(deftm child-alias-read
  [x :- Double ys :- (Array double) zs :- (Array double) flag :- Long] :- Double
  (let [value (if (= flag 1)
                (let [source ys]
                  (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget source i))))) x)
        ignored (aset zs 0 99.0)] value))

(deftm child-opaque-read
  [x :- Double ys :- (Array double) zs :- (Array double) flag :- Long] :- Double
  (let [value (if (= flag 1)
                (let [source (opaque-buffer ys)]
                  (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget source i))))) x)
        ignored (aset zs 0 99.0)] value))

(deftm child-clone-read
  [x :- Double ys :- (Array double) zs :- (Array double) flag :- Long] :- Double
  (let [value (if (= flag 1)
                (let [source (clone-buffer ys)]
                  (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget source i))))) x)
        ignored (aset zs 0 99.0)] value))

(deftm child-scalar-replay
  [x :- Double ys :- (Array double) zs :- (Array double) flag :- Long] :- Double
  (let [value (if (= flag 1)
                (let [factor (n/* x 3.0)]
                  (par/reduce acc 0.0 i 2 (n/+ acc (n/* factor (aget ys i))))) x)
        ignored (aset zs 0 99.0)] value))

(deftm child-opaque-read-counter
  [x :- Double ys :- (Array double) counts :- (Array int) flag :- Long] :- Double
  (let [value (if (= flag 1)
                (let [source (opaque-buffer ys)]
                  (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget source i))))) x)
        ignored (aset counts 0 (int 9))] value))

(deftm child-opaque-read-only
  [x :- Double ys :- (Array double) flag :- Long] :- Double
  (if (= flag 1)
    (let [source (opaque-buffer ys)]
      (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget source i))))) x))

(deftm plain-read-only [x :- Double ys :- (Array double)] :- Double
  (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i)))))

(deftm counted-opaque-read-only
  [x :- Double ys :- (Array double) cnt :- Long] :- Double
  (let [source (opaque-buffer ys)]
    (par/reduce acc 0.0 i cnt (n/+ acc (n/* x (aget source i))))))

(deftm private-replay-after-effect
  [x :- Double ys :- (Array double) zs :- (Array double) counts :- (Array int)] :- Double
  (let [before (aset counts 0 (int 7))
        value (if (> x 0.0)
                (let [source (opaque-buffer ys)]
                  (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget source i))))) x)
        after (aset zs 0 99.0)] value))

(deftest visible-replay-objects-use-runtime-identity-not-callee-spelling
  (let [ys (double-array [1.0 2.0])
        gradient (rev/value+grad #'visible-opaque-read :wrt [0])]
    (is (= :ad-replayed-array-alias
           (try (gradient 2.0 ys) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= [1.0 2.0] (vec ys))))
  (let [ys (double-array [1.0 2.0])]
    (is (= [6.0 3.0 nil] ((rev/value+grad #'visible-clone-read :wrt [0]) 2.0 ys)))
    (is (= [99.0 2.0] (vec ys)))))

(deftest exact-child-aliases-project-to-visible-objects
  (let [gradient (rev/value+grad #'child-alias-read :wrt [0])
        ys (double-array [1.0 2.0])]
    (is (= [6.0 3.0 nil nil nil] (gradient 2.0 ys (double-array 2) 1)))
    (is (= :ad-replayed-array-alias
           (try (gradient 2.0 ys ys 1) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= [1.0 2.0] (vec ys)))
    (is (= [2.0 1.0 nil nil nil] (gradient 2.0 ys ys 0)))
    (is (= [99.0 2.0] (vec ys)))))

(deftest unprojectable-private-array-replay-refuses-before-execution
  (doseq [v [#'child-opaque-read #'child-clone-read]]
    (let [ys (double-array [1.0 2.0]) zs (double-array [4.0 5.0])]
      (is (= :ad-unprojectable-replay-root
             (try ((rev/value+grad v :wrt [0]) 2.0 ys zs 1) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (= [1.0 2.0] (vec ys)))
      (is (= [4.0 5.0] (vec zs))))))

(deftest scalar-facts-and-incompatible-classes-do-not-create-alias-obligations
  (is (= [18.0 9.0 nil nil nil]
         ((rev/value+grad #'child-scalar-replay :wrt [0]) 2.0
          (double-array [1.0 2.0]) (double-array 2) 1)))
  (let [counts (int-array 1)]
    (is (= [6.0 3.0 nil nil nil]
           ((rev/value+grad #'child-opaque-read-counter :wrt [0]) 2.0
            (double-array [1.0 2.0]) counts 1)))
    (is (= [9] (vec counts)))))

(deftest private-replay-without-later-writes-remains-valid
  (doseq [[flag expected] [[0 [2.0 1.0 nil nil]] [1 [6.0 3.0 nil nil]]]]
    (is (= expected ((rev/value+grad #'child-opaque-read-only :wrt [0]) 2.0
                    (double-array [1.0 2.0]) flag))))
  ;; A public VJP differentiates all arrays too; an opaque helper intentionally
  ;; has no array derivative rule. Use an admitted read to qualify delayed seeds.
  (let [[value pb] (rev/vjp #'plain-read-only 2.0 (double-array [1.0 2.0]))]
    (is (= 6.0 value))
    (is (= 3.0 (first (pb 1.0))))
    (is (= 6.0 (first (pb 2.0))))))

(deftest zero-trip-and-pre-execution-refusal
  (is (= [0.0 0.0 nil nil]
         ((rev/value+grad #'counted-opaque-read-only :wrt [0]) 2.0 (double-array 0) 0)))
  (let [ys (double-array [1.0 2.0]) zs (double-array 2) counts (int-array 1)]
    (is (= :ad-unprojectable-replay-root
           (try ((rev/value+grad #'private-replay-after-effect :wrt [0]) 2.0 ys zs counts) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= [0] (vec counts)))
    (is (= [0.0 0.0] (vec zs)))))

(deftest exact-alias-projection-never-uses-may-alias-or-cycles
  (let [x (with-meta 'x {:raster.type/tag 'doubles})]
    (is (= x (#'rev/exact-replay-root 'local {'local {:value 'local :alias 'x}
                                            'x {:value x}})))
    (is (= 'local (#'rev/exact-replay-root 'local {'local {:value 'local}})))
    (is (= :ad-replay-alias-cycle
           (try (#'rev/exact-replay-root 'a {'a {:value 'a :alias 'b}
                                          'b {:value 'b :alias 'a}}) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
