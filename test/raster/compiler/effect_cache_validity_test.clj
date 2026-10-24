(ns raster.compiler.effect-cache-validity-test
  (:require [clojure.test :refer [deftest is]]
            [clojure.java.io :as io]
            [raster.compiler.passes.scalar.effects :as effects]))

(defn- with-definitions [f]
  ;; Exercise real executable redefinitions. File/line text is deliberately
  ;; NOT compiler evidence that a current helper's implementation is pure.
  (let [fixture-ns 'raster.compiler.fixtures.effect-cache-definition]
    (try
      (doseq [name ["effect_cache_pure"]]
        (let [resource (io/resource (str "raster/compiler/fixtures/" name ".source"))]
          (load-file (.getPath resource))))
      (f fixture-ns)
      (finally (remove-ns fixture-ns)))))

(defn- mutate! []
  (load-file (.getPath (io/resource "raster/compiler/fixtures/effect_cache_mutating.source"))))

(deftest unowned-helper-redefinitions-never-acquire-or-cache-purity
  (with-definitions
    (fn [namespace]
      (let [child (ns-resolve namespace 'child)
            parent (ns-resolve namespace 'parent)
            state @(ns-resolve namespace 'state)
            child-root @child parent-root @parent
            location (select-keys (meta child) [:file :line])
            analyzer (ns-resolve 'raster.compiler.passes.scalar.effects 'analyze-descriptor)
            analyze! @analyzer calls (atom 0)
            check! (fn []
                     (let [before @calls effects-before @state]
                       (doseq [operation ['child 'parent]
                               :let [v (ns-resolve namespace operation)
                                     expression (list (symbol (str namespace) (name operation)) 1)]]
                         (is (= :io (:effect (effects/descriptor expression))))
                         (is (not (effects/removable-expr? expression)))
                         (is (= :io (effects/analyze-var-effect v))))
                       (is (= (+ before 4) @calls)
                           "each unowned descriptor query delegates afresh, including repeated refusals")
                       (is (= effects-before @state) "analysis executes no helper")))]
        (is (= 2 (child 1) (parent 1)))
        (with-redefs-fn
          {analyzer (fn [expression] (swap! calls inc) (analyze! expression))}
          (fn []
            (check!)
            ;; The file still describes the pure definition, but the actual
            ;; child and the unchanged parent's runtime behavior have changed.
            (with-redefs-fn
              {child (fn [value] (swap! state inc) value)}
              (fn []
                (is (= location (select-keys (meta child) [:file :line])))
                (is (identical? parent-root @parent))
                (check!)
                (let [before @state]
                  (is (= 1 (parent 1)))
                  (is (= (inc before) @state)))))
            (is (identical? child-root @child))
            (mutate!)
            (is (not (identical? child-root @child)))
            (is (identical? parent-root @parent))
            (check!)
            (let [before @state]
              (is (= 1 (child 1) (parent 1)))
              (is (= (+ before 2) @state)))))))))

(deftest exact-declared-operation-identity-protects-alias-retargeting
  (let [namespace (create-ns (gensym "raster.effect-alias-test."))]
    (try
      (binding [*ns* namespace]
        (refer 'clojure.core)
        (let [expression (list (symbol (str (ns-name namespace)) "operation") 'x 'y)]
          (is (= :io (:effect (effects/descriptor expression))))
          ;; Publish actual refer mappings, not an interned copy or a new
          ;; registry entry. An earlier refusal must not hide this pure Var.
          (refer 'clojure.core :only '[+] :rename '{+ operation})
          (is (identical? #'clojure.core/+ (ns-resolve namespace 'operation)))
          (is (effects/removable-expr? expression))
          (is (effects/removable-expr? expression))
          (ns-unmap namespace 'operation)
          (refer 'clojure.core :only '[swap!] :rename '{swap! operation})
          (is (identical? #'clojure.core/swap! (ns-resolve namespace 'operation)))
          (is (not (effects/removable-expr? expression)))))
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
