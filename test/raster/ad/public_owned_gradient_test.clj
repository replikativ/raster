(ns raster.ad.public-owned-gradient-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [raster.core :as core :refer [deftm]]
            [raster.numeric :as n]
            [raster.arrays :as a]
            [raster.par :as p]
            [raster.ad.reverse :as reverse]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.core.value-completion :as completion]
            [raster.compiler.passes.scalar.inline :as inline]))

(deftm response [amplitude :- Double kappa :- Double exposure :- Double] :- Double
  (n/* amplitude (n// exposure (n/+ kappa exposure))))
(deftm tail-factor [theta :- (Array double) exposure :- Double] :- Double
  (let [amplitude (a/aget theta 0) kappa (a/aget theta 1)]
    (n/* 2.0 (response amplitude kappa exposure))))
(deftm nested-tail-factor [theta :- (Array double) exposure :- Double] :- Double
  (let [amplitude (a/aget theta 0) kappa (a/aget theta 1)]
    (n/* 2.0 (n/- (response amplitude kappa exposure) 0.3))))
(deftm reduction-tail-factor [theta :- (Array double) data :- (Array double) cnt :- Long] :- Double
  (let [amplitude (a/aget theta 0) kappa (a/aget theta 1)]
    (n/* 2.0 (response amplitude kappa
                        (p/reduce acc 0.0 i cnt
                          (n/+ acc (response amplitude kappa (a/aget data i))))))))

(deftm reduction-tail-gradient [theta :- (Array double) data :- (Array double) cnt :- Long] :- (Array double)
  (let [result ((reverse/value+grad #'reduction-tail-factor :wrt [0]) theta data cnt)
        gradient (nth result 1)]
    (p/map [i 3] (double (if (zero? i) (nth result 0) (a/aget gradient (dec i)))))))

(deftm packed-tail-factor [theta :- (Array double) data :- (Array double) cnt :- Long] :- Double
  (let [amplitude (a/aget theta 0) kappa (a/aget theta 1)]
    (n/* 2.0 (response amplitude kappa
                        (p/reduce acc 0.0 i cnt
                          (n/+ acc (response amplitude kappa (a/aget data (+ (* (long i) 3) 1)))))))))
(deftm packed-tail-gradient [theta :- (Array double) data :- (Array double) cnt :- Long] :- (Array double)
  (let [result ((reverse/value+grad #'packed-tail-factor :wrt [0]) theta data cnt)
        gradient (nth result 1)]
    (p/map [i 3] (double (if (zero? i) (nth result 0) (a/aget gradient (dec i)))))))

(deftm tail-gradient [theta :- (Array double) exposure :- Double] :- (Array double)
  (let [result ((reverse/value+grad #'tail-factor :wrt [0]) theta exposure)
        gradient (nth result 1)]
    (p/map [i 3] (double (if (zero? i) (nth result 0) (a/aget gradient (dec i)))))))

(deftm argument-order [data :- (Array double) events :- (Array long)] :- Double
  (let [result ((reverse/value+grad #'tail-factor :wrt [0])
                nil
                (do (a/aset events 0 (long 1)) (a/aget data 0)))]
    (double (nth result 0))))

(use-fixtures :once (fn [run] (response 0.8 0.7 1.0) (run)))

(defn- expected [amplitude kappa exposure offset]
  (let [den (+ kappa exposure)]
    [(- (/ (* 2.0 amplitude exposure) den) offset)
     (/ (* 2.0 exposure) den)
     (- (/ (* 2.0 amplitude exposure) (* den den)))]))

(defn- close? [left right] (< (Math/abs (- left right)) 1e-12))

(deftest owned-reduction-composes-into-runtime-and-invoked-aot-helpers
  (doseq [[factor wrapper data] [[#'reduction-tail-factor #'reduction-tail-gradient (double-array [0.2 0.8 1.1])]
                                 [#'packed-tail-factor #'packed-tail-gradient (double-array [9.0 0.2 9.0 9.0 0.8 9.0 9.0 1.1 9.0])]]]
   (let [prepared (reverse/prepare-owned-bytecode-value+grad factor [0])
        direct (#'reverse/compile-prepared-value+grad prepared)
        aot (pipeline/compile-aot wrapper)]
    (doseq [[amplitude kappa] [[0.8 0.7] [1.2 0.4]]
            cnt [0 1 3]]
      (let [theta (double-array [amplitude kappa])
            [value gradient inactive-data inactive-count] (direct theta data cnt)
            actual (into [value] gradient)
            h 1e-6
            expected (into [(factor theta data cnt)]
                           (for [coordinate [0 1]]
                             (let [plus (aclone theta) minus (aclone theta)]
                               (aset plus coordinate (+ (aget theta coordinate) h))
                               (aset minus coordinate (- (aget theta coordinate) h))
                               (/ (- (factor plus data cnt)
                                     (factor minus data cnt)) (* 2.0 h)))))]
        (is (nil? inactive-data))
        (is (nil? inactive-count))
        (is (every? true? (map #(< (Math/abs (- %1 %2)) 1e-8) actual expected)))
        (is (every? true? (map #(< (Math/abs (- %1 %2)) 1e-8) (aot theta data cnt) expected)))))
    (is (thrown? ArithmeticException (direct (double-array [0.8 0.7]) (double-array 0) Long/MAX_VALUE))))))

(deftest public-runtime-and-explicit-compiled-construction-share-the-owner
  (doseq [[v offset] [[#'tail-factor 0.0] [#'nested-tail-factor 0.6]]
          compile? [false true]]
    (let [f (reverse/value+grad v :wrt [0] :compile? compile?)]
      (is (= :jvm-bytecode (get-in (meta f) [:raster.compiler/source-owner :backend])))
      (doseq [[amplitude kappa exposure] [[0.8 0.7 1.0] [1.2 0.4 0.2] [-0.2 1.1 2.0]]]
        (let [[value gradient inactive] (f (double-array [amplitude kappa]) exposure)]
          (is (nil? inactive))
          (is (every? true? (map close? (into [value] gradient)
                                      (expected amplitude kappa exposure offset)))))))))

(deftest public-invoked-jvm-aot-retains-the-owned-call
  (let [f (pipeline/compile-aot #'tail-gradient)]
    (doseq [[amplitude kappa exposure] [[0.8 0.7 1.0] [1.2 0.4 0.2]]]
      (is (every? true? (map close? (f (double-array [amplitude kappa]) exposure)
                                  (expected amplitude kappa exposure 0.0)))))))

(deftest invoked-aot-completes-public-arguments-before-entering-the-callee
  (let [f (pipeline/compile-aot #'argument-order)
        events (long-array [0])]
    (is (thrown? ArrayIndexOutOfBoundsException (f (double-array 0) events)))
    (is (= [1] (vec events)))))

(deftest invoked-aot-rejects-a-replaced-selected-helper
  (let [f (pipeline/compile-aot #'tail-gradient)
        theta (double-array [0.8 0.7])
        source (core/resolve-deftm-var #'response)
        receiver (ns-resolve (.ns ^clojure.lang.Var source)
                             (symbol (str (:name (meta source)) "-impl")))]
    (is (= 3 (count (f theta 1.0))))
    (with-redefs-fn {receiver nil}
      #(is (thrown? clojure.lang.ExceptionInfo (f theta 1.0))))
    (is (every? true? (map close? (f theta 1.0) (expected 0.8 0.7 1.0 0.0))))))

(deftest portable-source-inliner-does-not-drop-a-private-bytecode-owner
  (is (= :unprojected-jvm-source-owner
         (try (binding [inline/*owned-jvm-calls?* false]
                (inline/inline-value+grad-call
                 {:var-sym 'raster.ad.public-owned-gradient-test/tail-factor
                  :args ['theta 'exposure] :mode :value+grad :options [:wrt [0]]}
                 :param-env {'theta 'doubles 'exposure 'double}))
              (catch clojure.lang.ExceptionInfo error (:reason (ex-data error)))))))

(deftest dispatch-diagnostics-traverse-only-executable-completion-operands
  (let [call '(raster.example/unresolved x)
        completed (completion/complete-value completion/object-input-contract call)
        literal (list 'quote [call])
        malformed (list completion/completion-symbol {} call)]
    (doseq [expression [completed literal malformed]]
      (is (= (count (#'pipeline/collect-undevirtualized expression))
             (#'pipeline/count-undevirtualized expression))))
    (is (= [call] (vec (#'pipeline/collect-undevirtualized completed))))
    (is (empty? (#'pipeline/collect-undevirtualized literal)))
    (is (= 2 (#'pipeline/count-undevirtualized malformed)))
    (is (not (#'pipeline/has-undevirtualized-calls? literal)))
    (is (#'pipeline/has-undevirtualized-calls? completed))))
