(ns raster.compiler.passes.scalar.peephole-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.passes.scalar.peephole :as peep]))

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
