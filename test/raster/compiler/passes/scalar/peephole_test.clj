(ns raster.compiler.passes.scalar.peephole-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.passes.scalar.peephole :as peep]
            [raster.compiler.ir.form :as form]))

(deftest copy-propagation-requires-one-alias-on-every-normal-return-path
  (doseq [source ['(let* [r (if flag x y)] (clojure.core/* r r))
                  '(let* [r (if flag x (clojure.core/+ y 1.0))] (identity r))
                  '(let* [r (if flag x 7.0)] (identity r))
                  '(let* [r (if flag x nil)] (identity r))
                  '(let* [r (if flag (let* [x 1.0] x) x)] (identity r))
                  '(let* [r (let* [x (clojure.core/+ x 1.0)] x)] (identity r))]]
    (let [lowered (peep/copy-propagate-let source)
          original (eval (list 'fn* '[flag x y] source))
          compiled (eval (list 'fn* '[flag x y] lowered))]
      (doseq [flag [true false]]
        (is (= (original flag 2.0 7.0) (compiled flag 2.0 7.0))
            (str source " -> " lowered " flag=" flag))))))

(deftest exact-tail-aliases-distinguish-unknown-values-from-nonreturning-paths
  (doseq [expression ['(if test x y)
                      '(if test x (unknown y))
                      '(if test x nil)
                      '(if test x 7.0)
                      '(if test x)
                      '(if test x (do))
                      '(do)
                      '(if test (let* [x 1.0] x) x)
                      '(if test (loop* [x 1.0] x) x)
                      '(let* [x (compute x)] x)
                      '(if test (recur x) (recur y))]]
    (is (nil? (form/tail-symbol expression)) (str expression)))
  (doseq [expression ['(if test x x)
                      '(if test (recur x) x)
                      '(if test x (recur x))
                      '(if test (throw problem) x)
                      '(loop* [i 0] (if test (let* [j 1] (recur j)) x))
                      '(let* [ignored (effect)] (do (effect) x))]]
    (is (= 'x (form/tail-symbol expression)) (str expression))))

(deftest exact-alias-copy-propagation-preserves-selected-effects
  (doseq [same? [true false]]
    (let [source (if same?
                   '(let* [r (if flag (do (aset seen 0 1) x)
                                      (do (aset seen 0 2) x))]
                      (clojure.core/* r r))
                   '(let* [r (if flag (do (aset seen 0 1) x)
                                      (do (aset seen 0 2) y))]
                      (clojure.core/* r r)))
          lowered (peep/copy-propagate-let source)
          original (eval (list 'fn* '[flag x y seen] source))
          compiled (eval (list 'fn* '[flag x y seen] lowered))]
      (doseq [flag [true false]]
        (let [before (int-array 1) after (int-array 1)]
          (is (= (original flag 2.0 7.0 before) (compiled flag 2.0 7.0 after)))
          (is (= (vec before) (vec after))))))))

(deftest alias-substitution-preserves-results-quoted-data-and-lexical-binding
  (doseq [source ['(let* [alias x] alias)
                  '(let* [alias x] (quote alias))
                  '(let* [alias x] (let* [alias y] (clojure.core/+ alias x)))
                  '(let* [alias x] (let* [x y] (clojure.core/+ alias x)))
                  '(let* [alias x] ((fn* [alias] (clojure.core/+ alias x)) y))]]
    (let [lowered (peep/copy-propagate-let source)
          original (eval (list 'fn* '[x y] source))
          compiled (eval (list 'fn* '[x y] lowered))]
      (doseq [[x y] [[2.0 7.0] [-3.0 0.5]]]
        (is (= (original x y) (compiled x y)) (str source " -> " lowered))))))

;; ================================================================
;; Non-let forms pass through
;; ================================================================

(deftest passthrough-test
  (testing "non-let form passes through unchanged"
    (let [form '(+ x y)
          {:keys [form stats]} (peep/apply-rules form)]
      (is (= '(+ x y) form))
      (is (= 0 (:rk-fused stats)))
      (is (= 0 (:copy-fused stats))))))

;; ================================================================
;; Alias inlining infrastructure
;; ================================================================

(deftest inline-aliases-test
  (testing "alias chains are resolved through apply-rules"
    ;; When a let* has alias bindings (sym = sym), the inline-aliases pass
    ;; should substitute the alias target throughout subsequent bindings.
    (let [form '(let* [a (some-op x)
                       b a
                       c (other-op b)]
                      c)
          {:keys [form]} (peep/apply-rules form)]
      ;; After alias inlining, 'b' (alias of 'a') should be resolved:
      ;; 'c' should reference 'a' directly, not 'b'
      (let [[_ bindings & _] form
            pairs (partition 2 bindings)
            ;; Filter out nil'd-out alias bindings
            live-pairs (remove (fn [[_ e]] (nil? e)) pairs)
            exprs (map second live-pairs)]
        ;; The (other-op b) should become (other-op a)
        (is (some #(and (seq? %) (= 'a (second %))) exprs)
            "Alias 'b' should be resolved to 'a' in downstream expressions"))))

  (testing "transitive alias chains are resolved"
    (let [form '(let* [a (compute x)
                       b a
                       c b
                       d (use c)]
                      d)
          {:keys [form]} (peep/apply-rules form)]
      (let [[_ bindings & _] form
            pairs (partition 2 bindings)
            live-pairs (remove (fn [[_ e]] (nil? e)) pairs)
            exprs (map second live-pairs)]
        ;; (use c) should become (use a) through transitive resolution
        (is (some #(and (seq? %) (= 'a (second %))) exprs)
            "Transitive alias chain b->a, c->b should resolve to 'a'")))))
