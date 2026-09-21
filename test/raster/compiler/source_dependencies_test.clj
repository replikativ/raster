(ns raster.compiler.source-dependencies-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.source-dependencies :as dependencies]
            [raster.core :refer [deftm]]))

(deftm dependency-leaf
  [x :- Double] :- Double
  (raster.numeric/* x x))

(deftm dependency-root
  [x :- Double] :- Double
  (raster.numeric/+ (dependency-leaf x) 1.0))

(defn opaque-helper ^double [^double x] x)

(deftm opaque-root
  [x :- Double] :- Double
  (opaque-helper x))

(defn- dependency-symbols [manifest]
  (set (map :symbol (:dependencies manifest))))

(def ^:private raster-build-namespaces
  '#{raster.arrays raster.dl.nn raster.numeric raster.par})

(deftest retained-deftm-dependencies-are-transitive-and-stable
  (let [options {:build-owned-namespaces raster-build-namespaces}
        first (dependencies/manifest #'dependency-root :double options)
        second (dependencies/manifest #'dependency-root :double options)]
    (is (:complete? first))
    (is (empty? (:blockers first)))
    (is (= (:fingerprint first) (:fingerprint second)))
    (is (contains? (dependency-symbols first)
                   'raster.compiler.source-dependencies-test/dependency-root_m_double))
    (is (contains? (dependency-symbols first)
                   'raster.compiler.source-dependencies-test/dependency-leaf_m_double))
    (is (= #{'raster.compiler.source-dependencies-test/dependency-root_m_double
             'raster.compiler.source-dependencies-test/dependency-leaf_m_double}
           (dependency-symbols first)))
    (is (contains? (set (:covered-build-calls first)) 'raster.numeric/*))
    (is (contains? (set (:covered-build-calls first)) 'raster.numeric/+))))

(deftest production-reduction-dependency-closure-is-complete
  (require 'raster.dl.nn)
  (let [manifest (dependencies/manifest
                  (ns-resolve 'raster.dl.nn 'rms-norm!) :float
                  {:build-owned-namespaces raster-build-namespaces})]
    (is (:complete? manifest) (pr-str (:blockers manifest)))
    (is (= 1 (count (:dependencies manifest))))
    (is (seq (:covered-build-calls manifest)))))

(deftest opaque-application-vars-block-persistence
  (let [manifest (dependencies/manifest
                  #'opaque-root :double
                  {:build-owned-namespaces raster-build-namespaces})]
    (is (false? (:complete? manifest)))
    (is (= [{:kind :opaque-application-var
             :owner 'raster.compiler.source-dependencies-test/opaque-root_m_double
             :call 'opaque-helper
             :resolved 'raster.compiler.source-dependencies-test/opaque-helper}]
           (:blockers manifest)))))
