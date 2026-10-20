(ns raster.compiler.retained-frontend-source-test
  (:require [clojure.test :refer [deftest is use-fixtures]]
            [raster.core :as core]
            [raster.par :as par]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.passes.scalar.inline :as inline]
            [raster.hardware-fixture :as hardware-fixture]
            [raster.runtime.hardware :as hardware]))

(def target :ocl:retained-frontend-source-test)
(use-fixtures :each hardware-fixture/isolated
  (fn [f]
    (hardware/register-target-device!
     target {:name "Synthetic retained frontend target"
             :capabilities {:total-eus 32 :subgroup-sizes [16]
                            :max-workgroup-size 1024 :shared-local-memory 65536}})
    (f)))

(defmacro scale-factor [] 2.0)
(core/deftm scaled [xs :- (Array double)] :- (Array double)
  (par/map [i (alength xs)] (* (aget xs i) (scale-factor))))
(core/deftm scaled [xs :- (Array float)] :- (Array float)
  (par/map [i (alength xs)] (float (* (aget xs i) (float (scale-factor))))))
(core/deftm late-helper [x :- Double] :- Double
  (let [result (+ x (scale-factor))] result))
(core/deftm mixed-scalar [factor :- Double xs :- (Array float)] :- (Array float)
  (par/scale factor xs))

(defn- original-source [dtype]
  (let [resolved (core/resolve-deftm-var #'scaled {:dtype dtype})]
    (entry/call-with-definition-inputs
     #(pipeline/retained-frontend-source #'scaled resolved dtype))))

(defn- reason [f]
  (try (f) nil (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))

(deftest materialized-macro-expansion-is-the-source-of-later-compilation
  (is (= [2.0 4.0] (vec (scaled (double-array [1.0 2.0])))))
  (let [before (original-source :double)
        original @#'scale-factor]
    (is (vector? (:body before)))
    (try
      (alter-var-root #'scale-factor (constantly (fn [& _] 7.0)))
      (is (= [2.0 4.0] (vec (scaled (double-array [1.0 2.0])))))
      (let [after (original-source :double)
            compiled (equation-first/compile #'scaled {:target target :dtype :double})
            backed (core/resolve-deftm-var #'scaled {:dtype :double})
            direct (equation-first/compile backed {:target target :dtype :double})]
        (is (= (:body before) (:body after)))
        (is (true? (boolean (some #{2.0} (tree-seq coll? seq (:semantic compiled))))))
        (is (false? (boolean (some #{7.0} (tree-seq coll? seq (:semantic compiled))))))
        (is (false? (boolean (some #{7.0} (tree-seq coll? seq (:semantic direct)))))))
      (finally (alter-var-root #'scale-factor (constantly original))))))

(deftest genuine-specializations-keep-their-original-typing
  (is (= [2.0] (vec (scaled (double-array [1.0])))))
  (is (= [(float 2.0)] (vec (scaled (float-array [1.0])))))
  (let [double-source (original-source :double)
        float-source (original-source :float)]
    (is (= :double (get-in double-source [:frontend-context :parameter-dtype])))
    (is (= :float (get-in float-source [:frontend-context :parameter-dtype])))
    (is (not (identical? (get-in double-source [:source-entry :source-var])
                         (get-in float-source [:source-entry :source-var]))))
    (is (not= (:body double-source) (:body float-source)))
    (let [resolved (core/resolve-deftm-var #'scaled {:dtype :double})]
      (is (= :retained-frontend-dtype-mismatch
             (reason #(entry/call-with-definition-inputs
                       (fn [] (pipeline/retained-frontend-source #'scaled resolved :float)))))))))

(deftest frontend-map-edits-cannot-rewrite-original-producer-evidence
  (scaled (double-array [1.0]))
  (let [frontend (original-source :double)
        source (get-in frontend [:source-entry :source-var])
        original (meta source)]
    (is (= :unowned-checked-result-frontend
           (reason #(entry/assert-frontend-source! (assoc frontend :body '[42.0]) source))))
    (is (= :unowned-checked-result-frontend
           (reason #(entry/assert-frontend-source!
                     (assoc frontend :source-frontend (Object.)) source))))
    (try
      (alter-meta! source update :raster.core/clojure-source-entry assoc :clojure-body '[42.0])
      (is (= :retained-frontend-owner-mismatch (reason #(original-source :double))))
      (finally (reset-meta! source original)))))

(deftest helper-import-uses-the-actual-implementation-not-an-earlier-walk-cache
  (let [source (core/resolve-deftm-var #'late-helper {:dtype :double})
        cached (core/ensure-walked-body! source)
        original @#'scale-factor]
    (is (true? (boolean (some #{2.0} (tree-seq coll? seq cached)))))
    (try
      (alter-var-root #'scale-factor (constantly (fn [& _] 7.0)))
      (is (= 8.0 (late-helper 1.0)))
      (let [imported (entry/call-with-definition-inputs
                      #(binding [inline/*inline-scalar-bodies?* true]
                         ((deref (ns-resolve 'raster.compiler.passes.scalar.inline 'try-resolve-walked-body))
                          source)))]
        (is (true? (boolean (some #{7.0} (tree-seq coll? seq (:walked-body imported))))))
        (is (false? (boolean (some #{2.0} (tree-seq coll? seq (:walked-body imported)))))))
      (finally (alter-var-root #'scale-factor (constantly original))))))

(deftest preserved-storage-keeps-source-typing-separate-from-target-carrier
  (is (= [(float 2.0)] (vec (mixed-scalar 2.0 (float-array [1.0])))))
  (let [compiled (equation-first/compile
                  #'mixed-scalar {:target target :dtype :double
                                  :preserve-declared-array-storage? true})]
    (is (some? (:ordered-source-owner compiled)))
    (is (= :float (get-in compiled [:options :array-types 'xs])))
    (is (= :double (get-in compiled [:options :scalar-types 'factor])))))
