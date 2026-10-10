(ns raster.compiler.core.call-signature-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.inference :as inf]
            [raster.numeric :as n]
            [raster.ad.forward])
  (:import [raster.ad.forward Dual]))

(def executions (atom 0))

(deftm asymmetric [x :- Double y :- Double] :- Double
  (do (swap! executions inc) (n/+ x y)))
(deftm asymmetric [x :- Dual y :- Double] :- Dual
  (do (swap! executions inc) (n/+ x y)))

(deftest registered-signature-is-position-aware-and-nonexecuting
  (let [before @executions
        op 'raster.compiler.core.call-signature-test/asymmetric
        table @(:raster.core/dispatch-table (meta #'asymmetric))]
    (testing "a supported family is not evidence for another argument tuple"
      (is (some? (inf/registered-call-signature op '[Dual double])))
      (is (nil? (inf/registered-call-signature op '[double Dual])))
      (is (nil? (inf/registered-call-signature op '[Dual Dual])))
      (is (nil? (inf/registered-call-signature op '[nil double]))))
    (testing "projection agrees with the compiler's existing call resolver"
      (is (= (inf/registered-call-signature op '[Dual double])
             (inf/try-resolve-call op '[x y] {'x {:tag 'Dual} 'y {:tag 'double}}))))
    (is (= before @executions) "queries never execute the user helper")
    (is (identical? table @(:raster.core/dispatch-table (meta #'asymmetric)))
        "queries do not register or specialize methods")))

(deftest selected-result-is-not-seed-dependence
  (let [signature (inf/registered-call-signature 'raster.numeric/real-value '[Dual])]
    (is (some? signature))
    (is (= 'double (:return-tag signature)))
    (is (symbol? (:mangled-sym signature)))))

(deftest registered-signature-retains-promotion-evidence
  (let [op 'raster.numeric/+
        signature (inf/registered-call-signature op '[float double])]
    (is (= '[double double] (:tags signature)))
    (is (= '[double nil] (:promotion-casts signature)))
    (is (= 'double (:return-tag signature)))
    (is (= signature (inf/try-resolve-call op '[x y]
                                         {'x {:tag 'float} 'y {:tag 'double}})))))

(deftest parametric-applicability-is-not-a-result-type
  (let [op 'raster.compiler.core.call-signature-test/signature-only]
    (dispatch/register-parametric! op '[T] '[T T] nil '[x y] nil *ns*)
    (try
      (let [before @dispatch/parametric-registry]
        (is (= {:bindings {'T 'double} :return-tag nil}
               (dispatch/parametric-call-signature op '[double double])))
        (is (nil? (dispatch/parametric-call-signature op '[double long])))
        (is (nil? (dispatch/parametric-call-signature op '[nil double])))
        (is (nil? (dispatch/signature-result-tag op '[double double])))
        (is (identical? before @dispatch/parametric-registry)))
      (finally (swap! dispatch/parametric-registry dissoc op)))))

(deftest result-query-keeps-existing-declared-result-fallback
  (let [op 'raster.compiler.core.call-signature-test/signature-result-fallback]
    (dispatch/register-parametric! op '[T] '[T T] nil '[x y] nil *ns*)
    (dispatch/register-parametric! op '[T U] '[T U] 'U '[x y] nil *ns*)
    (try
      (is (nil? (:return-tag (dispatch/parametric-call-signature op '[double double]))))
      (is (= 'double (dispatch/signature-result-tag op '[double double])))
      (finally (swap! dispatch/parametric-registry dissoc op)))))
