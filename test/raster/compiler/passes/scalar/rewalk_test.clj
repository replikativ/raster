(ns raster.compiler.passes.scalar.rewalk-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.passes.scalar.rewalk :as rewalk]))

(deftest passthrough-when-disabled-test
  (testing "pe-rewalk is a no-op when simplify is disabled"
    (let [form '(let* [a (raster.numeric/+ x y)] a)]
      (is (= form (rewalk/pe-rewalk form {:simplify? false}))))))

(deftest rewalk-devirtualizes-and-preserves-effect-bindings-test
  (testing "scalar bindings are re-walked while effect bindings are preserved"
    (let [form '(let* [a (raster.numeric/+ x y)
                       _effect (println a)]
                      a)
          {:keys [form stats]} (rewalk/pe-rewalk form {:simplify? true
                                                       :active-params '[x y]
                                                       :dtype :double})]
      (is (= 2 (:pe-rewalk-bindings-before stats)))
      (is (= 2 (:pe-rewalk-bindings-after stats)))
      (is (= 'let* (first form)))
      (is (some #{'_effect} (take-nth 2 (second form))))
      (is (some seq? (tree-seq coll? seq form)))
      (is (some #(and (seq? %) (= '.invk (first %)))
                (tree-seq coll? seq form))))))