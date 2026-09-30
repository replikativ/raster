(ns raster.compiler.source-dependencies-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.source-dependencies :as dependencies]
            [raster.compiler.fixtures.scalar-helpers :as helpers]
            [raster.gpu.compiled :as compiled]
            [raster.runtime.hardware :as hardware]
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

(deftest transitive-helper-replacement-invalidates-real-public-templates
  (let [target :cuda:source-reload-oracle
        arguments [(float-array [0.25 -2.5 8.0]) 3]
        options {:compiler :equation-first :dtype :float :target target}
        evidence #(dependencies/manifest #'helpers/map-reload-helper :float
                                        {:build-owned-namespaces raster-build-namespaces})
        root-fingerprint (fn [manifest]
                           (some #(when (= 'raster.compiler.fixtures.scalar-helpers/map-reload-helper_m_floats_long
                                           (:symbol %))
                                    (:source-fingerprint %))
                                 (:dependencies manifest)))]
    (hardware/register-target-device!
     target {:type :cuda :name "Source-only helper reload oracle"
             :capabilities {:compute-capability [8 0] :warp-size 32
                            :subgroup-sizes [32] :max-workgroup-size 1024
                            :shared-local-memory 65536 :total-eus 108}})
    (compiled/clear-compilation-cache!)
    (helpers/redefine-reload-tail! 1.0)
    (try
      (let [before-evidence (evidence)
            _ (compiled/lower #'helpers/map-reload-helper arguments options)
            repeated (compiled/lower #'helpers/map-reload-helper arguments options)]
        (is (:complete? before-evidence))
        (is (some? (root-fingerprint before-evidence)))
        ;; The source manifest conservatively includes registered backing methods; it is
        ;; not a typed call-site selector and need not materialize Float before compilation.
        (is (contains? (dependency-symbols before-evidence)
                       'raster.compiler.fixtures.scalar-helpers/reload-tail_m_double))
        (is (true? (get-in (compiled/preparation-report repeated) [:template :cache-hit?])))
        (helpers/redefine-reload-tail! 2.0)
        (let [after-evidence (evidence)
              after (compiled/lower #'helpers/map-reload-helper arguments options)
              after-repeated (compiled/lower #'helpers/map-reload-helper arguments options)]
          (is (:complete? after-evidence))
          (is (not= (:fingerprint before-evidence) (:fingerprint after-evidence)))
          (is (false? (get-in (compiled/preparation-report after) [:template :cache-hit?])))
          (is (true? (get-in (compiled/preparation-report after-repeated) [:template :cache-hit?])))
          (is (= (root-fingerprint before-evidence) (root-fingerprint after-evidence))
              "only the transitive callee changed; the public caller source is unchanged")
          (is (= [0.5 -5.0 16.0] (vec (apply helpers/map-reload-helper arguments))))))
      (finally
        (helpers/redefine-reload-tail! 1.0)
        (compiled/clear-compilation-cache!)))))
