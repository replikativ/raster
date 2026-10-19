(ns raster.compiler.backend.cpu.native-proof-owner-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm]]
            [raster.compiler.backend.cpu.aot :as aot]
            [raster.compiler.backend.cpu.codegen :as codegen]
            [raster.compiler.core.method-entry :as method-entry]
            [raster.compiler.pipeline :as pipeline]))

(def source-definition (Object.))
(deftm primitive-square [x :- Double] :- Double (clojure.core/* x x))

(deftest native-owner-starts-before-definition-dependent-frontend-analysis
  (let [reason (with-redefs [pipeline/deftm-frontend-facts
                             (fn [& _]
                               (is (some? method-entry/*emission-inputs*))
                               (throw (ex-info "stop before compilation" {:reason :frontend-owner-probe})))]
                 (try (aot/compile-aot-c #'primitive-square :double)
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))]
    (is (= :frontend-owner-probe reason))))

(deftest native-publication-refuses-historical-proof-replacement
  (doseq [nested? [false true]]
    (let [attempts (atom 0) invocations (atom 0)
          inherited (when nested? (atom {:static [] :values []}))
          failure
          (binding [method-entry/*emission-inputs* inherited]
            (with-redefs [source-definition (Object.)
                          pipeline/deftm-frontend-facts (fn [& _] {:return-tag 'double})
                          aot/fused-scalar-form
                          (fn [& _]
                            (is (some? method-entry/*emission-inputs*))
                            (when nested? (is (identical? inherited method-entry/*emission-inputs*)))
                            (method-entry/observe-definition-input!
                              #'source-definition (:ns (meta #'source-definition))
                              'raster.compiler.backend.cpu.native-proof-owner-test/source-definition)
                            {:form '(let* [] (clojure.core/* x x)) :params ['x] :param-env {'x 'double}})
                          codegen/compile-source!
                          (fn [_]
                            (swap! attempts inc)
                            (alter-var-root #'source-definition (constantly (Object.)))
                            "mocked-native-library")
                          codegen/load-kernel (fn [& _] (fn [& _] (swap! invocations inc)))]
              (try (aot/compile-aot-c #'primitive-square :double) nil
                   (catch clojure.lang.ExceptionInfo e e))))]
      (is (= :stale-definition-proof-input (:reason (ex-data failure))))
      (is (= (if nested? 1 3) @attempts) "only the outer owner retries the whole compilation")
      (is (zero? @invocations) "compilation must not execute the numerical kernel"))))

(deftest native-owner-preserves-the-actual-primitive-scalar-call
  (let [compiled (aot/compile-aot-c #'primitive-square :double)]
    (doseq [x [-3.5 -0.0 0.0 Math/PI]]
      (is (= (Double/doubleToRawLongBits (* x x))
             (Double/doubleToRawLongBits (double (compiled x))))))))
