(ns raster.ad.source-forward-scope-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as rev]))

(deftm prefix-replay-then-loop-write
  [x :- Double ys :- (Array double) cnt :- Long] :- Double
  (let [prior (par/reduce s 0.0 j 1 (n/+ s (n/* x (aget ys j))))]
    (loop [i 1 acc prior]
      (if (< i cnt)
        (let [ignored (aset ys 0 7.0)] (recur (inc i) (n/+ acc x))) acc))))

(deftm prefix-write-then-read-only-loop
  [x :- Double ys :- (Array double) cnt :- Long] :- Double
  (let [ignored (aset ys 0 3.0)]
    (loop [i 1 acc x]
      (if (< i cnt) (recur (inc i) (n/+ acc (n/* x (aget ys 0)))) acc))))

(deftm conditional-prefix-replay-then-loop-write
  [x :- Double ys :- (Array double) cnt :- Long flag :- Long] :- Double
  (let [prior (if (= flag 1) (par/reduce s 0.0 j 1 (n/+ s (n/* x (aget ys j)))) x)]
    (loop [i 1 acc prior]
      (if (< i cnt)
        (let [ignored (aset ys 0 7.0)] (recur (inc i) (n/+ acc x))) acc))))

(deftest prefix-replay-is-not-lost-at-loop-construction
  (let [h 1.0e-5 call #(prefix-replay-then-loop-write % (double-array [1.0]) 3)
        ys (double-array [1.0])]
    (is (= 6.0 (call 2.0)))
    (is (< (Math/abs (- 3.0 (/ (- (call (+ 2.0 h)) (call (- 2.0 h))) (* 2.0 h)))) 1.0e-8))
    ;; The previous accepted derivative was 9, not 3. A replay invalidation is
    ;; unsupported, not a valid derivative evaluated at the overwritten value.
    (let [gradient (rev/value+grad #'prefix-replay-then-loop-write :wrt [0])]
      (is (= [2.0 1.0 nil nil] (gradient 2.0 ys 1)) "zero trips execute no write/guard")
      (is (= :ad-replayed-array-alias
             (try (gradient 2.0 ys 3) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
    (is (= [1.0] (vec ys)))))

(deftest inherited-replay-condition-remains-selected-at-the-write
  (let [gradient (rev/value+grad #'conditional-prefix-replay-then-loop-write :wrt [0])
        ys (double-array [1.0])]
    (is (= [6.0 3.0 nil nil nil] (gradient 2.0 ys 3 0)))
    (is (= [7.0] (vec ys)))
    (let [fresh (double-array [1.0])]
      (is (= :ad-replayed-array-alias
             (try (gradient 2.0 fresh 3 1) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (= [1.0] (vec fresh))))))

(deftest lawful-earlier-write-remains-valid
  (let [ys (double-array [1.0])]
    (is (= [14.0 7.0 nil nil]
           ((rev/value+grad #'prefix-write-then-read-only-loop :wrt [0]) 2.0 ys 3)))
    (is (= [3.0] (vec ys)))))

(deftest scope-keeps-visibility-and-conditional-obligations-distinct
  (let [ambient {:values '[input] :value-facts {'input {:value 'input}}
                 :aliases {} :activity {'input true}
                 :replayed {'input 'selected} :unprojectable {'hidden 'selected}}
        state {:values '[input local] :value-facts {'input {:value 'input}
                                                  'local {:value 'local}
                                                  'private {:value 'private}}
               :aliases {'local '#{input}} :activity {'input false 'local true}
               :replayed '#{must-not-be-used-as-conditions}
               :input-replay-conditions {'other true} :unprojectable {'private true}}]
    (with-bindings {#'rev/*source-memory-context* ambient}
      (let [scope (#'rev/source-memory-scope state)]
        (is (= '[input local] (:values scope)))
        (is (contains? (:value-facts scope) 'private))
        (is (not (some #{'private} (:values scope))))
        (is (= {'input true 'local true} (:activity scope)))
        (is (= {'input 'selected 'other true} (:replayed scope)))
        (is (= {'hidden 'selected 'private true} (:unprojectable scope)))
        (is (= (:aliases state) (:aliases scope)))))
    (is (= :ad-forward-scope
           (try (#'rev/source-memory-scope (dissoc state :values)) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
