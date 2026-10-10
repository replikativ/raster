(ns raster.compiler.specialization-return-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm]]
            [raster.ad.reverse :as reverse]
            [raster.ad.forward]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.compiler.core.inference :as inf]
            [raster.compiler.core.types :as types])
  (:import [raster.ad.forward Dual]))

(deftm invalid-result (All [T] [x :- T] :- T (double x)))
(deftm invalid-body [x :- Double] :- Double (invalid-result x))
(deftm invalid-branch (All [T] [x :- T c :- Long] :- T
                       (let [value (if (== c 0) x (double x))] value)))
(deftm invalid-branch-body [x :- Double c :- Long] :- Double
  (invalid-branch x c))
(def helper-executions (atom 0))
(defn ^{:tag 'double} effectful-result-declared [x] (swap! helper-executions inc) (double x))
(deftm invalid-effect-declared (All [T] [x :- T] :- T (effectful-result-declared x)))
(deftm invalid-effect-body [x :- Double] :- Double (invalid-effect-declared x))
(deftm return-identity (All [T] [x :- T] :- T x))
(deftm identity-body [x :- Double] :- Double (return-identity x))
(deftm numeric-return (All [T] [x :- T] :- Double x))
(deftm emission-probe (All [T] [x :- T] :- T x))

(defn- thrown [f]
  (try (f) nil (catch Throwable error error)))

(deftest result-alternatives-retain-lexical-joins-and-unknowns
  (let [env {'x 'raster.ad.forward.Dual 'c 'long}]
    (doseq [expression ['(if c x (double x))
                        '(if c (double x) x)
                        '(let* [v (if c x (double x))] v)]]
      (is (= #{'raster.ad.forward.Dual 'double}
             (set (inf/infer-result-tags expression env *ns*)))))
    (is (= [nil] (inf/infer-result-tags '(loop* [v x] v) env *ns*)))
    (is (= [nil] (inf/infer-result-tags '(clojure.core/loop [v 1.0] v) env *ns*)))
    (is (= ['raster.ad.forward.Dual]
           (inf/infer-result-tags '(if true x (double x)) env *ns*)))
    (is (= ['raster.ad.forward.Dual]
           (inf/infer-result-tags '(if false (double x) x) env *ns*)))
    (let [use (with-meta 'x {:tag 'double :raster.type/tag 'double})]
      (is (= [nil] (inf/infer-result-tags
                   (list 'clojure.core/let ['x '(unknown-result)] (list '+ use)) env *ns*))))
    (is (= [nil] (inf/infer-result-tags '(let* [x (unknown-result)] x) env *ns*))
        "an unknown local shadows rather than inherits an outer tag")))

(deftest boxing-conflict-does-not-reject-legal-numeric-or-supertype-returns
  (is (types/boxed-return-incompatible? 'double 'raster.ad.forward.Dual))
  (doseq [declared '[double float long Object Number java.lang.Number]]
    (is (not (types/boxed-return-incompatible? 'double declared))))
  (is (not (types/boxed-return-incompatible? nil 'raster.ad.forward.Dual))))

(deftest reference-target-is-resolved-in-the-template-namespace
  (let [source (create-ns (gensym "raster.test.return_source_"))
        caller (create-ns (gensym "raster.test.return_caller_"))
        emission (AssertionError. "reached legal emission")]
    (try
      (binding [*ns* source] (refer 'clojure.core) (intern source 'Target Number))
      (binding [*ns* caller] (refer 'clojure.core) (intern caller 'Target String))
      (with-redefs [core/do-bytecode-upgrade! (fn [& _] (throw emission))]
        (binding [*ns* caller]
          (is (identical? emission
                          (thrown #(#'core/register-parametric-specialization!
                                    (symbol (str (ns-name source)) "boxed-result")
                                    {:annotations '[T] :ret-annotation 'Target :type-vars '[T]
                                     :params '[x] :body '(double x) :source-ns source}
                                    {'T 'double} [])))
              "Number accepts boxed Double even though caller's Target is String")))
      (finally (remove-ns (ns-name source)) (remove-ns (ns-name caller))))))

(deftest invalid-specialization-refuses-before-emission-or-publication
  (let [original core/do-bytecode-upgrade! emissions (atom 0)
        executions @helper-executions]
    (with-redefs [core/do-bytecode-upgrade!
                  (fn [& args]
                    (if (.contains (str (second args)) "invalid-")
                      (do (swap! emissions inc) (throw (AssertionError. "invalid emission reached")))
                      (apply original args)))]
      (doseq [operation ['raster.compiler.specialization-return-test/invalid-result
                         'raster.compiler.specialization-return-test/invalid-branch
                         'raster.compiler.specialization-return-test/invalid-effect-declared]
              _ (range 2)]
        (let [tags (if (= "invalid-branch" (name operation))
                     '[raster.ad.forward.Dual long] '[raster.ad.forward.Dual])
              error (thrown #(dispatch/register-parametric-call-tags! operation tags))]
          (is (instance? clojure.lang.ExceptionInfo error))
          (is (= :specialization-return-incompatible (:reason (ex-data error))))
          (is (= :parametric (:selection (dispatch/selected-call-signature
                                         operation tags
                                         (inf/registered-call-selection operation tags))))))))
    (is (zero? @emissions))
    (is (= executions @helper-executions))))

(deftest public-forward-admission-is-cold-and-warm-negative
  (doseq [v [#'invalid-body #'invalid-branch-body #'invalid-effect-body]
          _ (range 2)]
    (is (false? (:admissible? (reverse/forward-coverage v))))
    (is (thrown? clojure.lang.ExceptionInfo (reverse/value+grad v :mode :forward))))
  (is (= [2.0 1.0] ((reverse/value+grad #'identity-body :mode :forward) 2.0)))
  (is (= 3.0 (numeric-return 3))))

(deftest verifier-errors-still-propagate-unchanged
  (let [failure (VerifyError. "injected emitter failure")]
    (with-redefs [core/do-bytecode-upgrade! (fn [& _] (throw failure))]
      (is (identical? failure
                      (thrown #(dispatch/register-parametric-call-tags!
                                'raster.compiler.specialization-return-test/emission-probe
                                '[raster.ad.forward.Dual])))))))
