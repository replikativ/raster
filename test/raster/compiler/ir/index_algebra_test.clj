(ns raster.compiler.ir.index-algebra-test
  "Injectivity facts over mixed-radix index forms: what the frontend may certify as `:unique`."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.index-algebra :as ia]))

(def ^:private rope-locals
  '[{:id hdim2 :init (quot head-dim 2)}
    {:id per-row :init (* heads hdim2)}
    {:id t :init (quot idx per-row)}
    {:id r0 :init (rem idx per-row)}
    {:id h :init (quot r0 hdim2)}
    {:id i :init (rem r0 hdim2)}])

(deftest monomials-compare-as-products-of-non-negative-factors
  (is (= {:const 6 :factors '[a b]} (ia/monomial '(* 2 (* a (* 3 b))))))
  (is (nil? (ia/monomial '(+ a 1))))
  (is (ia/dominates? (ia/monomial '(* a b)) (ia/monomial 'a)))
  (is (not (ia/dominates? (ia/monomial 'a) (ia/monomial '(* a b)))))
  (testing "a quot fact closes the half-dimension comparison"
    (is (ia/dominates? (ia/monomial 'head-dim) (ia/monomial 'hdim2)
                       [{:factor 'hdim2 :times {:const 2 :factors []}
                         :le {:const 1 :factors ['head-dim]}}]))))

(deftest row-major-forms-are-injective-and-dropped-digits-are-not
  (let [row (ia/index-form '(+ (* r feat) j) 'r 'rows [] '{j feat})
        overlap (ia/index-form '(+ (* r 3) j) 'r 'rows [] '{j 4})
        rope (ia/index-form '(+ (+ (* t (* heads head-dim)) (* h head-dim)) i)
                            'idx 'n rope-locals {})
        dropped (ia/index-form '(+ (* t (* heads head-dim)) i) 'idx 'n rope-locals {})
        whole (ia/index-form '(+ idx i) 'idx 'n rope-locals {})]
    (is (ia/injective? row))
    (is (not (ia/injective? overlap)) "row stride 3 under a 4-wide inner digit overlaps")
    (is (ia/injective? rope) "the RoPE layout is row-major over (t h i)")
    (is (not (ia/injective? dropped)) "dropping the head digit makes heads collide")
    (is (nil? whole) "a decomposed index may not also appear whole")))

(deftest constant-offsets-are-one-more-digit
  (let [rope (ia/index-form '(+ (+ (* t (* heads head-dim)) (* h head-dim)) i)
                            'idx 'n rope-locals {})
        row (ia/index-form '(+ (* r feat) j) 'r 'rows [] '{j feat})]
    (is (ia/disjoint-offsets? rope '[0 hdim2])
        "the two RoPE halves fit under head-dim because 2·hdim2 ≤ head-dim")
    (is (not (ia/disjoint-offsets? rope '[0 head-dim]))
        "an offset of a whole head lands on the next head's row")
    (is (not (ia/disjoint-offsets? row [0 2]))
        "two literal offsets inside a symbolic-width row overlap")))
