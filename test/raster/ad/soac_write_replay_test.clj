(ns raster.ad.soac-write-replay-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as rev]))

(deftm reduce-before-map [x :- Double ys :- (Array double) zs :- (Array double)] :- Double
  (let [value (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))
        written (par/map! zs i 2 double 7.0)] value))

(deftm reduce-before-active-map [x :- Double ys :- (Array double) zs :- (Array double)] :- Double
  (let [value (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))
        written (par/map! zs i 2 double (n/* x 3.0))] value))

(deftm reduce-before-gather
  [x :- Double ys :- (Array double) zs :- (Array double)
   data :- (Array double) index :- (Array int)] :- Double
  (let [value (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))
        written (par/gather zs data index 2)] value))

(deftm map-before-reduce [x :- Double ys :- (Array double) zs :- (Array double)] :- Double
  (let [written (par/map! zs i 2 double 7.0)]
    (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))))

(deftm conditional-map-write
  [x :- Double ys :- (Array double) zs :- (Array double) flag :- Long] :- Double
  (let [value (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))
        branch (if (= flag 1) x (do (par/map! zs i 2 double 7.0) x))]
    (n/+ value branch)))

(deftest declared-output-writes-protect-prior-replays
  (doseq [[v tail written] [[#'reduce-before-map [] [7.0 7.0]]
                            [#'reduce-before-active-map [] [6.0 6.0]]
                            [#'reduce-before-gather [(double-array [7.0 8.0])
                                                    (int-array [0 1])] [7.0 8.0]]]]
    (let [gradient (rev/value+grad v :wrt [0])
          ys (double-array [1.0 2.0]) zs (double-array 2)]
      (is (= 3.0 (second (apply gradient 2.0 ys zs tail))))
      (is (= [1.0 2.0] (vec ys)))
      (is (= written (vec zs)))
      (is (= :ad-replayed-array-alias
             (try (apply gradient 2.0 ys ys tail) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (= [1.0 2.0] (vec ys))))))

(deftest declared-output-guards-survive-compiled-jvm-projection
  ;; The active-map control above must exercise a nontrivial forward patch,
  ;; not merely the identity patch of a constant map. Observe existing facts;
  ;; do not add a production flag or infer tape structure from emitted strings.
  (let [original @#'rev/gen-reverse-par-map patched (atom false)]
    (with-redefs-fn {#'rev/gen-reverse-par-map
                    (fn [& args]
                      (let [record (apply original args)]
                        (when (:forward-code record) (reset! patched true))
                        record))}
      #(rev/prepare-value+grad #'reduce-before-active-map [0]))
    (is (true? @patched)))
  (let [gradient (rev/value+grad #'reduce-before-map :wrt [0] :compile? true)
        ys (double-array [1.0 2.0])]
    (is (true? (:raster.ad.reverse/compiled? (meta gradient))))
    (is (= :ad-replayed-array-alias
           (try (gradient 2.0 ys ys) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= [1.0 2.0] (vec ys)))))

(deftest earlier-writes-and-unselected-write-arms-remain-valid
  (let [ys (double-array [1.0 2.0])]
    (is (= [28.0 14.0 nil nil]
           ((rev/value+grad #'map-before-reduce :wrt [0]) 2.0 ys ys)))
    (is (= [7.0 7.0] (vec ys))))
  (let [gradient (rev/value+grad #'conditional-map-write :wrt [0])
        ys (double-array [1.0 2.0])]
    (is (= [8.0 4.0 nil nil nil] (gradient 2.0 ys ys 1)))
    (is (= [1.0 2.0] (vec ys)))
    (is (= :ad-replayed-array-alias
           (try (gradient 2.0 ys ys 0) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= [1.0 2.0] (vec ys)))))

(deftest guarded-write-objects-must-be-visible-not-merely-declared
  (let [ys (with-meta 'ys {:raster.type/tag 'doubles})
        hidden (with-meta 'hidden {:raster.type/tag 'doubles})
        scope {:values [ys] :value-facts {ys {:value ys} hidden {:value hidden}}}]
    (with-bindings {#'rev/*source-memory-context* scope}
      (is (= :ad-unprojectable-write-root
             (try (#'rev/replay-input-guard-bindings {ys true} #{'hidden}) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
  ;; No object comparison is needed if retained classes prove incompatibility.
  (let [ys (with-meta 'ys {:raster.type/tag 'doubles})
        hidden (with-meta 'hidden {:raster.type/tag 'ints})]
    (with-bindings {#'rev/*source-memory-context*
                    {:values [ys] :value-facts {ys {:value ys} hidden {:value hidden}}}}
      (is (= [] (#'rev/replay-input-guard-bindings {ys true} #{'hidden}))))))
