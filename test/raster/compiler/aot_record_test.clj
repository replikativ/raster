(ns raster.compiler.aot-record-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.numeric :as n]
            [raster.sci.distributions :as dist]
            [raster.compiler.pipeline :as pipeline]))

;; Deliberately do not import the record classes into this caller namespace.
(deftm normal-lp [x :- Double s :- Double] :- Double
  (dist/logpdf (dist/->Normal 0.0 s) x))

(deftm two-record-lp [x :- Double s :- Double] :- Double
  (n/+ (dist/logpdf (dist/->Normal 0.0 s) x)
       (dist/logpdf (dist/->Exponential s) x)))

(deftest inlined-record-field-class-authority
  (doseq [[v exponential?] [[#'normal-lp false] [#'two-record-lp true]]]
    (testing (str (:name (meta v)))
      (let [compiled (pipeline/compile-aot v :dtype :double)]
        (doseq [[x s] [[0.5 2.0] [0.75 0.4] [1.25 3.0]]]
          (let [normal (* -0.5 (+ (Math/log (* 2.0 Math/PI))
                                     (* 2.0 (Math/log s))
                                     (* (/ x s) (/ x s))))
                ;; Normal logpdf is -1/2 (...); exponential uses rate s.
                expected (+ normal (if exponential? (- (Math/log s) (* s x)) 0.0))]
            (is (< (Math/abs (- expected (compiled x s))) 1.0e-12))))))))

(deftest loader-selection-verifies-retained-class-identity
  (let [cls (class (dist/->Normal 0.0 1.0))
        wrong (proxy [clojure.lang.DynamicClassLoader] []
                (loadClass [name]
                  (if (= name (.getName cls)) String (proxy-super loadClass name))))
        selected (#'pipeline/retained-compilation-loader
                  wrong *ns* (with-meta 'record-local {:tag cls}) [])]
    (is (not (identical? wrong selected)))
    (is (identical? cls (Class/forName (.getName cls) false selected)))))

(deftest incompatible-same-name-classes-are-refused
  (let [name "raster.test.AotDuplicateIdentity"
        bytes (.build (java.lang.classfile.ClassFile/of)
                      (java.lang.constant.ClassDesc/of name)
                      (reify java.util.function.Consumer
                        (accept [_ builder] (.withFlags builder (int 1)))))
        a (.defineClass (clojure.lang.DynamicClassLoader.) name bytes nil)
        b (.defineClass (clojure.lang.DynamicClassLoader.) name bytes nil)]
    (is (= (.getName a) (.getName b)))
    (is (not (identical? a b)))
    (is (= :aot-class-identity-conflict
           (try (#'pipeline/retained-compilation-loader
                 (clojure.lang.DynamicClassLoader.) *ns* nil [a b])
                nil
                (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))))
