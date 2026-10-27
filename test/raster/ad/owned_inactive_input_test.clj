(ns raster.ad.owned-inactive-input-test
  "Owned bytecode AD retains inactive input completion and source evaluation order."
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.compiler.core.method-entry :as entry]
            [raster.numeric]
            [raster.ad.reverse :as reverse]))

(deftm application-square [x :- Double] :- Double (clojure.core/* x x))
(deftm application-square-caller [x :- Double] :- Double (application-square x))
(deftm application-inactive-array-square [unused :- (Array double) x :- Double] :- Double
  (clojure.core/* x x))
(deftm application-inactive-array-caller [unused :- (Array double) x :- Double] :- Double
  (application-inactive-array-square unused x))
(deftm application-inactive-long-branch [selector :- Long x :- Double] :- Double
  (if (clojure.core/zero? selector)
    (clojure.core/* x x)
    (clojure.core/+ x x)))
(deftm application-inactive-long-caller [selector :- Long x :- Double] :- Double
  (application-inactive-long-branch selector x))
(deftm application-inactive-int-branch [selector :- Integer x :- Double] :- Double
  (if (clojure.core/zero? selector) (clojure.core/* x x) (clojure.core/+ x x)))
(deftm application-inactive-int-caller [selector :- Integer x :- Double] :- Double
  (application-inactive-int-branch selector x))
(deftm application-active-discrete-caller [selector :- Double x :- Double] :- Double
  (application-inactive-long-branch (long selector) x))
(deftm application-used-array [data :- (Array double) x :- Double] :- Double
  (clojure.core/+ (clojure.core/aget data 0) (clojure.core/* x x)))
(deftm application-used-array-caller [data :- (Array double) x :- Double] :- Double
  (application-used-array data x))

(defmacro define-object-array-ad-caller []
  (let [operation (symbol (str (ns-name *ns*)) "application-inactive-array-square")
        implementation (symbol (str (ns-name *ns*)) "application-inactive-array-square_m_doubles_double-impl")
        selected (entry/selected-definition operation implementation 2)
        receiver (with-meta implementation
                   {:raster.type/tag (get-in selected [:entry :typed-iface])
                    :raster.type/ret-tag 'double})]
    (list 'raster.core/deftm 'application-object-array-caller
          '[unused :- Object later :- (Array double) x :- Double] ':- 'Double
          (with-meta (list '.invk receiver 'unused '(raster.numeric/+ x (clojure.core/aget later 0)))
            {:raster.op/original operation}))))

(declare application-object-array-caller)
(define-object-array-ad-caller)

(defn- compiled-owned [source wrt]
  (#'reverse/compile-prepared-value+grad
   (reverse/prepare-owned-bytecode-value+grad source wrt)))

(deftest application-completion-does-not-become-a-numerical-ad-operator
  (let [gradient (compiled-owned #'application-square-caller [0])]
    (doseq [x [-2.0 0.0 3.0]]
      (is (= [(* x x) (* 2.0 x)] (gradient x))))))

(deftest inactive-nullable-array-keeps-its-primal-check-without-an-array-tangent
  (let [gradient (compiled-owned #'application-inactive-array-caller [1])
        ;; AD selects an entry at construction. Compare its original selected
        ;; entry, not the generic dispatcher (which rejects String earlier).
        original (deref (core/resolve-deftm-var #'application-inactive-array-caller))]
    (doseq [array [nil (double-array [9.0])]]
      (is (= 4.0 (application-inactive-array-caller array 2.0)))
      (is (= [4.0 nil 4.0] (gradient array 2.0))
          "the unused inactive array does not acquire a length probe or cotangent"))
    (is (thrown? ClassCastException (original "not an array" 2.0)))
    (is (thrown? ClassCastException (gradient "not an array" 2.0))
        "excluding an argument from AD must not remove its original check")))

(deftest inactive-long-input-keeps-its-primal-control-without-a-tangent
  (let [gradient (compiled-owned #'application-inactive-long-caller [1])]
    (doseq [[selector expected derivative]
            [[0 9.0 6.0] [1 6.0 2.0] [Long/MIN_VALUE 6.0 2.0]
             [Long/MAX_VALUE 6.0 2.0]]]
      (is (= expected (application-inactive-long-caller selector 3.0)))
      (is (= [expected nil derivative] (gradient selector 3.0))))))

(deftest inactive-int-input-keeps-its-primal-control-without-a-tangent
  (let [gradient (compiled-owned #'application-inactive-int-caller [1])]
    (doseq [selector [(int 0) (int 1) Integer/MIN_VALUE Integer/MAX_VALUE]]
      (let [expected (if (zero? selector) [9.0 nil 6.0] [6.0 nil 2.0])]
        (is (= (first expected) (application-inactive-int-caller selector 3.0)))
        (is (= expected (gradient selector 3.0)))))))

(deftest unsupported-active-discrete-and-used-array-regions-still-refuse
  (doseq [[source wrt] [[#'application-active-discrete-caller [0 1]]
                        [#'application-used-array-caller [1]]]]
    (is (= :ad-unowned-derived-pullback
           (try (reverse/prepare-owned-bytecode-value+grad source wrt) nil
                (catch clojure.lang.ExceptionInfo error (:reason (ex-data error))))))))

(deftest inner-array-check-precedes-the-later-numerical-operand-in-ad
  (let [gradient (compiled-owned #'application-object-array-caller [2])]
    (is (= 9.0 (application-object-array-caller nil (double-array [1.0]) 2.0)))
    (is (= [9.0 nil nil 6.0] (gradient nil (double-array [1.0]) 2.0)))
    (doseq [invoke [application-object-array-caller gradient]]
      (is (thrown? ClassCastException (invoke "bad first argument" nil 2.0))
          "the inner helper CHECKCAST precedes the later null-array read")
      (is (thrown? NullPointerException (invoke nil nil 2.0))
          "a successful nullable CHECKCAST does not suppress the later read"))))
