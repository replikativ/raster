(ns raster.compiler.owned-helper-source-readiness-test
  (:require [clojure.test :refer [deftest is]]
            [raster.core :as core]
            [raster.compiler.core.inline-call-plan :as call-plan]
            [raster.compiler.core.method-entry :as entry]
            [raster.compiler.equation-first :as equation-first]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.scalar.source-import :as source-import]))

(defn- with-cold-array-helper [f]
  (let [scope (create-ns (gensym "raster.compiler.cold-array-helper-"))]
    (try
      (binding [*ns* scope]
        (refer 'clojure.core)
        (require '[raster.core :as r] '[raster.par :as par] '[raster.arrays :as arrays])
        (eval '(r/deftm helper [alpha :- Double values :- (Array float)] :- (Array float)
                 (par/pmap i (arrays/alength values) float (* alpha (aget values i)))))
        (eval '(r/deftm owner [alpha :- Double values :- (Array float)] :- (Array float)
                 (helper alpha values))))
      (f (ns-resolve scope 'owner) (ns-resolve scope 'helper))
      (finally
        (doseq [name '[owner helper] :let [v (ns-resolve scope name)] :when v]
          (core/unregister! v))
        (remove-ns (ns-name scope))))))

(defn- original-emission [helper]
  (entry/call-with-definition-inputs
   #(some-> (core/resolve-deftm-var helper)
            entry/original-source-frontend entry/observed-frontend-emission)))

(defn- compile-cold [owner]
  (let [target :ocl:cold-owned-helper]
    (equation-first/compile
     owner {:target target :dtype :double :preserve-declared-array-storage? true}
     {:device-id target :device-type :gpu :backend :ocl :fp64? true
      :subgroup-dialect :intel-opencl :max-workgroup-size 256})))

(deftest optional-reference-application-query-does-not-demand-preparation
  (with-cold-array-helper
    (fn [owner helper]
      (let [source-var (core/resolve-deftm-var owner)
            facts (pipeline/deftm-frontend-facts source-var :double)
            source (first (pipeline/get-walked-body source-var :double))
            options {:source-var source-var :source-ns (:source-ns facts)
                     :target-device :ocl:cold-owned-helper
                     :active-params (:parameters facts) :public-parameters (:parameters facts)}
            call (if (contains? '#{let let*} (first source))
                   (second (second source)) source)]
        (is (nil? (original-emission helper)))
        (is (= 1 (entry/call-with-definition-inputs
                  #(.size (call-plan/call-plans (call-plan/source-context options) source)))))
        (with-redefs [core/prepare-owned-source-frontend!
                      (fn [& _] (throw (AssertionError. "optional query prepared a helper")))]
          (is (nil? (entry/call-with-definition-inputs
                     #(source-import/instantiate-ready-reference-helper source call options)))))
        (is (nil? (original-emission helper)))))))

(deftest explicit-target-construction-prepares-only-its-cold-selected-helper
  (with-cold-array-helper
    (fn [owner helper]
      (let [prepare! core/prepare-owned-source-frontend!
            preparations (atom [])
            helper-source (core/resolve-deftm-var helper)]
        (is (nil? (original-emission helper)))
        (let [compilation
              (with-redefs [core/prepare-owned-source-frontend!
                            (fn [source]
                              (swap! preparations conj source)
                              (prepare! source))]
                (compile-cold owner))]
          (is (= :typed-parallel (get-in compilation [:semantic :dialect])))
          (is (= 1 (count (:kernels compilation))))
          (is (= 1 (count (filter #(identical? helper-source %) @preparations))))
          (is (some? (original-emission helper))))))))

(deftest owned-but-unimportable-helper-is-not-prepared-repeatedly
  (with-cold-array-helper
    (fn [owner helper]
      (let [prepare! core/prepare-owned-source-frontend!
            helper-source (core/resolve-deftm-var helper)
            preparations (atom [])]
        ;; Simulate consumer abstention, not absent source ownership. After the
        ;; one authentic preparation the same request must fail coverage, not
        ;; repeatedly prepare/restamp the unsupported module.
        (with-redefs [source-import/instantiate-ready-reference-helper (constantly nil)
                      core/prepare-owned-source-frontend!
                      (fn [source] (swap! preparations conj source) (prepare! source))]
          (is (= :equation-first-coverage
                 (try (compile-cold owner) nil
                      (catch clojure.lang.ExceptionInfo e (:reason (ex-data e))))))
          (is (= 1 (count (filter #(identical? helper-source %) @preparations)))))))))

(deftest prerequisite-preparation-failure-is-not-retried-or-replaced
  (with-cold-array-helper
    (fn [owner _helper]
      (let [failure (ex-info "original preparation failed" {:reason ::preparation-failure})
            preparations (atom 0)]
        (with-redefs [core/prepare-owned-source-frontend!
                      (fn [_] (swap! preparations inc) (throw failure))]
          (is (identical? failure (try (compile-cold owner) nil (catch Exception e e))))
          (is (= 1 @preparations)))))))
