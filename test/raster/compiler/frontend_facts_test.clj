(ns raster.compiler.frontend-facts-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :refer [deftm defvalue]]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.core.dispatch :as dispatch]
            [raster.arrays :as arrays]
            [raster.numeric :as n]
            [raster.par :as par]))

(deftm mixed-storage [input :- (Array float) scale :- Double count :- Long] :- Double
  (n/* scale (double (arrays/aget input count))))

(deftm generic-scale
  (All [T] [input :- (Array T) count :- Long] :- (Array T)
    (let [output (arrays/alloc-like input count)]
      (par/map! output i count nil (n/* (arrays/aget input i) (arrays/aget input i)))
      output)))

(defvalue FrontendBundle [values :- (Array float) labels :- (Array int)])
(deftm aggregate-read [bundle :- FrontendBundle count :- Long] :- Float
  (arrays/aget (.values bundle) count))

(deftest shared-specialization-and-storage-facts
  (is (= :float (:effective-dtype (pipeline/deftm-frontend-facts #'mixed-storage nil)))
      "implicit dtype retains the existing array-preferred dispatch rule")
  (let [facts (pipeline/deftm-frontend-facts #'mixed-storage :double)
        options (#'equation-first/compiler-options #'mixed-storage :ocl:frontend-test :double
                 {:preserve-declared-array-storage? true})
        derived (opencl-pass/derive-param-types (:parameters facts)
                                                (mapv :tag (:param-specs facts))
                                                (:effective-dtype facts)
                                                {:preserve-declared-array-storage? true})]
    (is (= :double (:effective-dtype facts)))
    (is (= '[input scale count] (:parameters facts)))
    (is (= '{input floats scale double count long} (:param-env facts)))
    (is (= (:array-types derived) (:array-types options)))
    (is (= (:scalar-types derived) (:scalar-types options)))
    (is (= (:source-ns facts) (:source-ns options)))
    (is (= (:return-tag facts) (:return-tag options)))
    (is (= :double (:dtype options)))))

(deftest frontend-facts-do-not-walk-or-emit
  (with-redefs [pipeline/get-walked-body (fn [& _] (throw (AssertionError. "frontend facts walked source")))]
    (is (= '[input scale count]
           (:parameters (pipeline/deftm-frontend-facts #'mixed-storage :double))))))

(deftest resident-entry-consumes-shared-facts-before-passes
  (let [facts (pipeline/deftm-frontend-facts #'mixed-storage :double)
        derived (opencl-pass/derive-param-types (:parameters facts)
                                                (mapv :tag (:param-specs facts)) :double
                                                {:preserve-declared-array-storage? true})
        observed (atom nil)
        sentinel (ex-info "observed resident pre-pass boundary" {})]
    (with-redefs [hardware/descriptor-for (constantly nil)
                  pipeline/get-walked-body (fn [& _] '[(double (raster.arrays/aget input count))])
                  pipeline/run-passes (fn [_ _ opts & _]
                                        (reset! observed opts) (throw sentinel))]
      (is (identical? sentinel
                      (try (pipeline/compile-gpu-program #'mixed-storage :ocl:frontend-test
                                                        :dtype :double
                                                        :preserve-declared-array-storage? true)
                           (catch clojure.lang.ExceptionInfo e e)))))
    (is (= :double (:dtype @observed)))
    (is (= (:parameters facts) (:active-params @observed)))
    (is (= (:param-env facts) (:param-env @observed)))
    (is (= (:array-types derived) (:array-types @observed)))
    (is (= (:scalar-types derived) (:scalar-types @observed)))
    (is (= (:source-ns facts) (:source-ns @observed)))))

(deftest resident-host-only-admission-precedes-fact-extraction
  (let [resolved (:resolved-var (pipeline/deftm-frontend-facts #'mixed-storage :double))]
    (with-redefs [hardware/descriptor-for (constantly nil)
                  dispatch/host-only? #(identical? resolved %)
                  pipeline/deftm-frontend-facts (fn [& _] (throw (AssertionError. "host-only facts extracted")))]
      (is (= :gpu-compiler-host-only
             (try (pipeline/compile-gpu-program #'mixed-storage :ocl:frontend-test :dtype :double)
                  (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))))

(deftest parametric-facts-use-the-resolved-specialization
  (doseq [dtype [:float :double]]
    (let [facts (pipeline/deftm-frontend-facts #'generic-scale dtype)
          options (#'equation-first/compiler-options #'generic-scale :ocl:frontend-test dtype {})]
      (is (= dtype (:effective-dtype facts) (:dtype options)))
      (is (= (:parameters facts) (:active-params options)))
      (is (= (:param-env facts) (:param-env options)))
      (is (= (pipeline/build-param-env (:resolved-var facts) dtype) (:param-env facts)))
      (is (= (mapv :tag (:param-specs facts))
             (mapv (:param-env facts) (:parameters facts)))))))

(deftest aggregate-representation-consumes-shared-facts
  (let [facts (pipeline/deftm-frontend-facts #'aggregate-read :float)
        representation (equation-first/parameter-representation #'aggregate-read :float)]
    (is (= (:param-specs facts) (:params representation)))
    (is (seq (:soa-env representation)))
    (is (seq (get-in representation [:projection :physical-parameters])))))
