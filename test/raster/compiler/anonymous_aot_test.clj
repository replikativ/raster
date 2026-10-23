(ns raster.compiler.anonymous-aot-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core :refer [deftm ftm]]
            [raster.numeric :as n]
            [raster.ad.reverse :as reverse]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.backend.jvm.source-owner :as source-owner]
            [raster.compiler.core.dispatch :as dispatch]))

(deftm square [x :- Double] :- Double (n/* x x))
(def anonymous-square (ftm [x :- Double] :- Double (n/* x x)))

(defmacro capture-hidden [] 'hidden)
(defmacro source-policy-value [] (if *unchecked-math* 2.0 1.0))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest anonymous-entry-retains-declarations-and-does-not-register-functions
  (let [before-interns (ns-interns 'raster.compiler.anonymous-aot-test)
        before-dispatch (set (keys @dispatch/dispatch-tables))
        before-revision (dispatch/compiler-definition-revision)
        f (ftm [x :- Double] :- Double (n/* x x))
        compiled (pipeline/compile-aot f :simd? false)]
    (is (= '[Double] (:raster.core/deftm-annotations (meta f))))
    (is (= 'Object (:raster.core/return-tag (meta f))))
    (is (= 'Double (:raster.core/deftm-return-annotation (meta f))))
    (is (= #{} (:raster.core/lexical-captures (meta f))))
    (is (= [4.0 9.0 0.25] (mapv compiled [2.0 3.0 -0.5])))
    (is (= before-revision (dispatch/compiler-definition-revision)))
    (is (= before-interns (ns-interns 'raster.compiler.anonymous-aot-test)))
    (is (= before-dispatch (set (keys @dispatch/dispatch-tables))))
    (is (map? (:raster.compiler.pipeline/compiled-diagnostics (meta compiled))))))

(deftest anonymous-entry-uses-its-defining-namespace-not-the-compilation-caller
  (let [f (ftm [x :- Double] :- Double (n/+ (square x) 1.0))
        compiled (binding [*ns* (the-ns 'clojure.core)]
                   (pipeline/compile-aot f :simd? false))]
    (is (= [5.0 10.0] (mapv compiled [2.0 3.0])))))

(deftest var-held-anonymous-ad-resolves-the-functions-source-context
  (let [gradient (binding [*ns* (the-ns 'clojure.core)]
                   (reverse/value+grad #'anonymous-square))]
    (is (= [4.0 4.0] (gradient 2.0)))
    (is (= [9.0 6.0] (gradient 3.0)))))

(deftest anonymous-source-policy-is-retained-and-required
  (let [f (binding [*unchecked-math* false]
            (eval '(raster.core/ftm [x :- Long] :- Long (clojure.core/inc x))))
        compiled (binding [*unchecked-math* true] (pipeline/compile-aot f :simd? false))]
    (is (= {:unchecked-math false} (:raster.core/source-policy (meta f))))
    (is (= 42 (compiled 41)))
    (is (thrown? ArithmeticException (compiled Long/MAX_VALUE)))
    (is (= :unclosed-anonymous-source-policy
           (reason #(pipeline/compile-aot (vary-meta f dissoc :raster.core/source-policy))))))
  (let [f (binding [*unchecked-math* false]
            (eval '(raster.core/ftm [] :- Double
                     (raster.compiler.anonymous-aot-test/source-policy-value))))
        compiled (binding [*unchecked-math* true] (pipeline/compile-aot f :simd? false))]
    (is (= 1.0 (f) (compiled))))
  (doseq [policy [true :warn-on-boxed]]
    (let [f (binding [*unchecked-math* policy]
              (eval '(raster.core/ftm [x :- Long] :- Long (clojure.core/inc x))))]
      (is (= Long/MIN_VALUE (f Long/MAX_VALUE)))
      (is (= :unsupported-anonymous-source-policy (reason #(pipeline/compile-aot f))))
      (is (= :unsupported-anonymous-source-policy (reason #(reverse/value+grad f)))))))

(deftest capture-denials-include-shadowing-core-names-and-macro-introduced-captures
  (let [hidden (Object.)
        + (fn [x] x)
        ordinary (ftm [] hidden)
        expanded (ftm [] (capture-hidden))
        shadowing (ftm [x :- Double] (+ x))]
    (doseq [f [ordinary expanded shadowing]]
      (is (seq (:raster.core/lexical-captures (meta f))))
      (is (= :unprojected-jvm-source-owner (reason #(pipeline/compile-aot f))))
      (is (= :unprojected-jvm-source-owner (reason #(reverse/value+grad f)))))
    (is (identical? hidden (ordinary)))
    (is (identical? hidden (expanded)))))

(deftest unsupported-anonymous-targets-refuse-before-emission
  (let [f (ftm [x :- Double] :- Double (n/* x x))]
    (is (= :unsupported-anonymous-aot-target (reason #(pipeline/compile-aot f :target :c))))
    (is (= :unsupported-anonymous-aot-target
           (reason #(pipeline/compile-aot f :target-device :ocl:not-a-device))))))

(deftest anonymous-return-declarations-do-not-invent-primitive-completions
  (doseq [f [(ftm [] :- Double nil) (ftm [] :- Double (float 1.25))
             (ftm [] :- Object "object")]]
    (let [compiled (pipeline/compile-aot f :simd? false)]
      (is (= (f) (compiled)))
      (is (= (class (f)) (class (compiled)))))))

(deftest replaced-namespace-is-not-the-retained-source-owner
  (let [name (gensym "raster.anonymous.owner")
        original (create-ns name)]
    (try
      (let [f (binding [*ns* original]
                (eval '(do (clojure.core/defn helper [] 1.0)
                           (raster.core/ftm [] :- Double (helper)))))]
        (remove-ns name)
        (let [replacement (create-ns name)]
          (binding [*ns* replacement]
            (eval '(clojure.core/defn helper [] 2.0))))
        (is (= 1.0 (f)))
        (is (= :stale-anonymous-source-namespace (reason #(pipeline/compile-aot f))))
        (is (= :stale-anonymous-source-namespace (reason #(reverse/value+grad f)))))
      (finally (remove-ns name)))))

(deftest namespace-replacement-during-compilation-refuses-publication
  (let [name (gensym "raster.anonymous.publication")
        original (create-ns name)]
    (try
      (let [f (binding [*ns* original] (eval '(raster.core/ftm [] 1.0)))]
        (is (= :stale-anonymous-source-namespace
               (reason #(source-owner/call-with-source-context
                         (meta f) (fn [] (remove-ns name) (create-ns name) :unpublished))))))
      (finally (remove-ns name)))))

(deftest compiling-a-gradient-does-not-publish-a-wrapper-or-stale-retained-gradients
  (let [retained (reverse/value+grad #'square)
        _ (is (= [4.0 4.0] (retained 2.0)))
        before-interns (ns-interns 'raster.compiler.anonymous-aot-test)
        before-dispatch (set (keys @dispatch/dispatch-tables))
        before-revision (dispatch/compiler-definition-revision)
        compiled (reverse/value+grad #'square :compile? true)
        wrapper (#'reverse/compiled-value+grad-fn #'square nil)]
    (is (true? (:raster.ad.reverse/compiled? (meta compiled)))
        (str (:raster.ad.reverse/compile-failure (meta compiled))))
    (is (= [9.0 6.0] (compiled 3.0)))
    (is (ifn? (:fn wrapper)) (str (:failure wrapper)))
    (is (= [9.0 6.0] ((:fn wrapper) 3.0)))
    (is (= [9.0 6.0] (retained 3.0)))
    (is (= before-revision (dispatch/compiler-definition-revision)))
    (is (= before-interns (ns-interns 'raster.compiler.anonymous-aot-test)))
    (is (= before-dispatch (set (keys @dispatch/dispatch-tables))))))
