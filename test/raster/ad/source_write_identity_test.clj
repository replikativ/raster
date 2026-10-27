(ns raster.ad.source-write-identity-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.par :as par]
            [raster.ad.reverse :as rev]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.pipeline :as pipeline]))

(deftm ^:no-inline opaque-buffer [ys :- (Array double)] :- (Array double) ys)
;; A familiar allocator basename is deliberately NOT a freshness certificate.
(deftm ^:no-inline alloc-like [ys :- (Array double)] :- (Array double) ys)

(deftm opaque-write [x :- Double ys :- (Array double)] :- Double
  (let [target (opaque-buffer ys)
        value (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))
        ignored (aset target 0 99.0)]
    value))

(deftm named-like-allocator-write [x :- Double ys :- (Array double)] :- Double
  (let [target (alloc-like ys)
        value (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))
        ignored (aset target 0 99.0)]
    value))

(deftm opaque-choice-write [x :- Double ys :- (Array double) flag :- Long] :- Double
  (let [target (opaque-buffer ys)
        value (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))
        ignored (if (= flag 1) (aset target 0 99.0) 0.0)]
    value))

(deftm earlier-opaque-write [x :- Double ys :- (Array double)] :- Double
  (let [target (opaque-buffer ys)
        ignored (aset target 0 3.0)]
    (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))))

(deftm private-write [x :- Double ys :- (Array double)] :- Double
  (let [target (double-array 2)
        value (par/reduce acc 0.0 i 2 (n/+ acc (n/* x (aget ys i))))
        ignored (aset target 0 99.0)]
    value))

(deftm private-gradient [x :- Double ys :- (Array double)] :- Double
  (let [vg ((rev/value+grad #'private-write :wrt [0]) x ys)] (nth vg 1)))

(deftest opaque-write-identities-are-checked-before-mutation
  (doseq [[v compile?] [[#'opaque-write false] [#'opaque-write true]
                       [#'named-like-allocator-write false]]]
    (let [gradient (rev/value+grad v :wrt [0] :compile? compile?)
          ys (double-array [1.0 2.0])]
      (when compile? (is (true? (:raster.ad.reverse/compiled? (meta gradient)))))
      (is (= :ad-replayed-array-alias
             (try (gradient 2.0 ys) nil
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
      (is (= [1.0 2.0] (vec ys))))))

(deftest opaque-write-choice-follows-the-selected-path
  (let [gradient (rev/value+grad #'opaque-choice-write :wrt [0])
        ys (double-array [1.0 2.0])]
    (is (= [6.0 3.0 nil nil] (gradient 2.0 ys 0)))
    (is (= :ad-replayed-array-alias
           (try (gradient 2.0 ys 1) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (= [1.0 2.0] (vec ys)))))

(deftest earlier-writes-and-genuinely-private-destinations-remain-valid
  (doseq [[v expected after] [[#'earlier-opaque-write [10.0 5.0 nil] [3.0 2.0]]
                             [#'private-write [6.0 3.0 nil] [1.0 2.0]]]]
    (let [ys (double-array [1.0 2.0])]
      (is (= expected ((rev/value+grad v :wrt [0]) 2.0 ys)))
      (is (= after (vec ys))))))

(deftest unnormalized-write-destinations-refuse-without-evaluation
  (let [effects (atom 0)
        destination (list 'do (list 'swap! effects 'inc) 'ys)]
    (is (= :ad-unclosed-write-target
           (try (#'rev/replay-input-guard-bindings {'ys true} #{destination}) nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
    (is (zero? @effects))))

(deftest private-destination-native-guard-is-not-erased
  (let [data (try
               (let [compiled (pipeline/compile-aot #'private-gradient :target :c :dtype :double)]
                 (when (instance? java.lang.AutoCloseable compiled)
                   (.close ^java.lang.AutoCloseable compiled))
                 nil)
               (catch clojure.lang.ExceptionInfo e (ex-data e)))]
    ;; The completion gate now precedes the unsupported guard/tuple checks.
    ;; Neither the conversion nor the private-destination alias obligation may
    ;; be erased just to reach a later gate. No native acceptance claim.
    (is (= :unclosed-c-family-value-completion (:reason data)))
    (is (completion/contains-completion? (:expression data)))
    (is (some #{"Reverse AD replay input aliases a later write"}
              (tree-seq coll? seq (:expression data))))
    (is (some #{:ad-replayed-array-alias}
              (tree-seq coll? seq (:expression data))))))
