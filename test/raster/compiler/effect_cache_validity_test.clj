(ns raster.compiler.effect-cache-validity-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.java.io :as io]
            [beichte.core :as beichte]
            [raster.compiler.passes.scalar.effects :as effects]))

(defn- with-definitions [f]
  ;; load-file supplies real file/line source evidence to Beichte, rather than
  ;; mocking its classification or assuming an eval has retrievable source.
  (let [fixture-ns 'raster.compiler.fixtures.effect-cache-definition]
    (try
      (doseq [name ["effect_cache_pure"]]
        (let [resource (io/resource (str "raster/compiler/fixtures/" name ".clj"))]
          (load-file (.getPath resource))))
      (f fixture-ns)
      (finally (remove-ns fixture-ns)))))

(defn- mutate! []
  (load-file (.getPath (io/resource "raster/compiler/fixtures/effect_cache_mutating.clj"))))

(deftest ordinary-helper-redefinition-invalidates-purity
  (with-definitions
    (fn [_]
      (let [expression '(raster.compiler.fixtures.effect-cache-definition/child 1)]
        (is (effects/removable-expr? expression))
        (mutate!)
        (is (= :mutation (:effect (beichte/analyze-full expression))))
        (is (= :mutation (:effect (effects/descriptor expression))))
        (is (not (effects/removable-expr? expression)))))))

(deftest transitive-helper-redefinition-does-not-reuse-parent-purity
  (with-definitions
    (fn [namespace]
      (let [parent (ns-resolve namespace 'parent)
            root @parent
            expression '(raster.compiler.fixtures.effect-cache-definition/parent 1)]
        (is (effects/removable-expr? expression))
        (mutate!)
        (is (identical? root @parent))
        (is (= :mutation (:effect (effects/descriptor expression))))))))

(deftest var-effect-analysis-is-fresh-as-well
  (with-definitions
    (fn [namespace]
      (let [child (ns-resolve namespace 'child)
            parent (ns-resolve namespace 'parent)]
        (is (= :pure (effects/analyze-var-effect child)))
        (is (= :pure (effects/analyze-var-effect parent)))
        (mutate!)
        (is (= :mutation (effects/analyze-var-effect child)))
        (is (= :mutation (effects/analyze-var-effect parent)))))))

(deftest missing-helper-publication-does-not-retain-an-old-refusal
  (let [fixture-ns 'raster.compiler.fixtures.effect-cache-definition
        expression '(raster.compiler.fixtures.effect-cache-definition/child 1)]
    (when (find-ns fixture-ns) (remove-ns fixture-ns))
    (is (not (effects/removable-expr? expression)))
    (with-definitions
      (fn [_] (is (effects/removable-expr? expression))))))

(deftest exact-declared-operation-identity-protects-alias-retargeting
  (let [namespace (create-ns (gensym "raster.effect-alias-test."))]
    (try
      (binding [*ns* namespace]
        (refer 'clojure.core)
        (intern namespace 'operation @#'clojure.core/+)
        ;; Explicit refer mappings exercise resolved Var identity without a
        ;; source helper: both target operations have declared effect contracts.
        (ns-unmap namespace 'operation)
        (refer 'clojure.core :only '[+] :rename '{+ operation})
        (is (effects/removable-expr? '(operation x y)))
        (ns-unmap namespace 'operation)
        (refer 'clojure.core :only '[swap!] :rename '{swap! operation})
        (is (not (effects/removable-expr? '(operation x y)))))
      (finally (remove-ns (ns-name namespace))))))

(deftest only-closed-contracts-retain-cross-call-memoization
  (let [analyzer (ns-resolve 'raster.compiler.passes.scalar.effects 'analyze-descriptor)
        original @analyzer
        calls (atom 0)
        unique (gensym "effect-cache-local-")
        closed (list 'clojure.core/+ unique 7)
        open (list 'if unique 7 9)]
    (with-redefs-fn
      {analyzer (fn [expression] (swap! calls inc) (original expression))}
      #(do
         (is (effects/removable-expr? closed))
         (is (effects/removable-expr? closed))
         (is (= 1 @calls))
         (is (effects/removable-expr? open))
         (is (effects/removable-expr? open))
         (is (= 3 @calls))))))
