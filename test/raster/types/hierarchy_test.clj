(ns raster.types.hierarchy-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.core :refer [defabstract defvalue defval deftm
                                 ambiguities invoke prefer-method!
                                 remove-method!]]))

;; ================================================================
;; Abstract type hierarchy
;; ================================================================

(defabstract Algorithm)
(defabstract AdaptiveAlgorithm :extends Algorithm)
(defabstract FixedStepAlgorithm :extends Algorithm)

(defvalue Euler [] :implements FixedStepAlgorithm)
(defvalue RK4 [] :implements FixedStepAlgorithm)
(defvalue DP5 [atol :- Double, rtol :- Double] :implements AdaptiveAlgorithm)

(deftest abstract-type-hierarchy-test
  (testing "interface inheritance"
    (is (instance? Algorithm (->Euler)))
    (is (instance? FixedStepAlgorithm (->Euler)))
    (is (not (instance? AdaptiveAlgorithm (->Euler))))
    (is (instance? Algorithm (->DP5 1e-6 1e-3)))
    (is (instance? AdaptiveAlgorithm (->DP5 1e-6 1e-3)))
    (is (not (instance? FixedStepAlgorithm (->DP5 1e-6 1e-3))))))

;; ================================================================
;; Dispatch on abstract types
;; ================================================================

(deftm describe [alg :- Algorithm] "algorithm")
(deftm describe [alg :- AdaptiveAlgorithm] "adaptive")
(deftm describe [alg :- FixedStepAlgorithm] "fixed-step")
(deftm describe [alg :- Euler] "euler")

(deftest abstract-dispatch-test
  (testing "most specific concrete type wins"
    (is (= "euler" (describe (->Euler)))))
  (testing "abstract type catches subtypes without specific methods"
    (is (= "fixed-step" (describe (->RK4)))))
  (testing "adaptive abstract type"
    (is (= "adaptive" (describe (->DP5 1e-6 1e-3))))))

;; ================================================================
;; defval — singleton types for dispatch
;; ================================================================

(defval InPlace)
(defval OutOfPlace)

(deftm workspace-type [alg :- Algorithm, mode :- InPlace] :mutable)
(deftm workspace-type [alg :- Algorithm, mode :- OutOfPlace] :constant)

(deftest defval-dispatch-test
  (testing "Val-style dispatch"
    (is (= :mutable (workspace-type (->Euler) (->InPlace))))
    (is (= :constant (workspace-type (->DP5 1e-6 1e-3) (->OutOfPlace))))))

;; ================================================================
;; defvalue with typed fields (was defcache before unification)
;; ================================================================

(defvalue TestCache [k1 :- (Array double), k2 :- (Array double), value :- Double])

(deftest defvalue-cache-test
  (testing "creates record with typed fields"
    (let [c (->TestCache (double-array [1 2]) (double-array [3 4]) 5.0)]
      (is (= 5.0 (:value c)))
      (is (= [1.0 2.0] (vec (:k1 c)))))))

;; ================================================================
;; Ambiguity detection
;; ================================================================

(defabstract Renderable)
(defabstract Storable)

(deftm ambig-fn [x :- Renderable] :- Long 1)
(deftm ambig-fn [x :- Storable] :- Long 2)

(deftm no-ambig-fn [x :- Algorithm] :- Long 0)
(deftm no-ambig-fn [x :- AdaptiveAlgorithm] :- Long 1)

(deftest ambiguity-detection-test
  (testing "detects ambiguous unrelated interfaces"
    (is (seq (ambiguities #'ambig-fn))))
  (testing "no ambiguity for parent/child hierarchy"
    (is (empty? (ambiguities #'no-ambig-fn))))
  (testing "sibling interfaces are ambiguous"
    (is (= 1 (count (ambiguities #'describe))))))

;; ================================================================
;; Method disambiguation
;; ================================================================

(deftest invoke-test
  (testing "explicit method selection by tags"
    (is (= 1 (invoke #'ambig-fn '[Renderable] (reify Renderable))))
    (is (= 2 (invoke #'ambig-fn '[Storable] (reify Storable)))))
  (testing "invoke on non-ambiguous method"
    (is (= "euler" (invoke #'describe '[Euler] (->Euler))))))

(defabstract Pref1)
(defabstract Pref2)

(deftm pref-test [x :- Pref1] :- Long 10)
(deftm pref-test [x :- Pref2] :- Long 20)

(deftest prefer-method-test
  (testing "prefer-method! resolves ambiguity"
    (prefer-method! #'pref-test '[Pref1] '[Pref2])
    (let [both (reify Pref1 Pref2)]
      (is (= 10 (pref-test both))))))

(deftm removable [x :- Long] :- Long 1)
(deftm removable [x :- Double] :- Double 2.0)

(deftest remove-method-test
  (testing "remove-method! removes a specific method"
    (is (= 1 (removable 42)))
    (remove-method! #'removable '[long])
    (is (thrown? clojure.lang.ExceptionInfo (removable 42)))
    ;; Double method still works
    (is (= 2.0 (removable 3.14)))))
