(ns raster.compiler.ir.index-algebra-test
  "Injectivity facts over mixed-radix index forms: what the frontend may certify as `:unique`."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.index-algebra :as ia]))

(def ^:private rope-locals
  ;; the map extent `batch·seq-len·heads·hdim2` is an exact multiple of every divisor below
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
                            'idx '(* (* batch seq-len) (* heads hdim2)) rope-locals {})
        dropped (ia/index-form '(+ (* t (* heads head-dim)) i) 'idx '(* (* batch seq-len) (* heads hdim2)) rope-locals {})
        whole (ia/index-form '(+ idx i) 'idx '(* (* batch seq-len) (* heads hdim2)) rope-locals {})]
    (is (ia/injective? row))
    (is (not (ia/injective? overlap)) "row stride 3 under a 4-wide inner digit overlaps")
    (is (ia/injective? rope) "the RoPE layout is row-major over (t h i)")
    (is (not (ia/injective? dropped)) "dropping the head digit makes heads collide")
    (is (nil? whole) "a decomposed index may not also appear whole")))

(deftest constant-offsets-are-one-more-digit
  (let [rope (ia/index-form '(+ (+ (* t (* heads head-dim)) (* h head-dim)) i)
                            'idx '(* (* batch seq-len) (* heads hdim2)) rope-locals {})
        row (ia/index-form '(+ (* r feat) j) 'r 'rows [] '{j feat})]
    (is (ia/disjoint-offsets? rope '[0 hdim2])
        "the two RoPE halves fit under head-dim because 2·hdim2 ≤ head-dim")
    (is (not (ia/disjoint-offsets? rope '[0 head-dim]))
        "an offset of a whole head lands on the next head's row")
    (is (not (ia/disjoint-offsets? row [0 2]))
        "two literal offsets inside a symbolic-width row overlap")))

(deftest lossy-projections-and-unresolved-locals-are-not-injective
  (testing "a lone quot or rem is a projection, not a decomposition"
    (is (not (ia/injective? (ia/index-form 'q 'idx 4 '[{:id q :init (quot idx 2)}] {}))))
    (is (not (ia/injective? (ia/index-form 'r 'idx 4 '[{:id r :init (rem idx 2)}] {})))))
  (testing "a matched pair whose divisor divides the extent decomposes"
    (is (ia/injective? (ia/index-form '(+ (* q 2) r) 'idx '(* 2 m)
                                      '[{:id q :init (quot idx 2)} {:id r :init (rem idx 2)}] {}))))
  (testing "a divisor that does not divide the extent is undecidable"
    (is (nil? (ia/index-form '(+ (* q 3) r) 'idx 'n
                             '[{:id q :init (quot idx 3)} {:id r :init (rem idx 3)}] {}))))
  (testing "a local the algebra cannot normalize makes the form undecidable"
    (is (nil? (ia/index-form '(+ i k) 'i 'n '[{:id k :init (- 0 i)}] {})))
    (is (nil? (ia/index-form '(+ i k) 'i 'n '[{:id k :init (long (aget slots i))}] {})))))

(deftest affine-sums-add-coefficients-and-keep-offset-multiplicity
  (let [doubled (ia/index-form '(+ (* i 4) (* i 4) j (* j 7)) 'i 2 [] '{j 2})]
    (is (= 8 (get-in doubled [:terms 'i :coefficient :const])))
    (is (= 8 (get-in doubled [:terms 'j :coefficient :const])))
    (is (not (ia/injective? doubled)) "8i + 8j collides at (1,0) and (0,1)"))
  (let [a (ia/index-form '(* i (* 2 d)) 'i 4 [] {})
        b (ia/index-form '(+ (* i (* 2 d)) d d) 'i 4 [] {})]
    (is (= {:const 2 :factors ['d]} (:offset b)) "d + d is 2d, not d")
    (is (not (ia/disjoint-offsets? a [(:offset a) (:offset b)]))
        "an offset of a whole stride lands on the next item")))

(deftest coefficients-need-a-lower-radix-to-vouch-for-their-factors
  (is (not (ia/injective? (ia/index-form '(* i stride) 'i 'n [] {})))
      "a captured stride may be zero and collapse every item")
  (is (ia/injective? (ia/index-form '(+ (* r feat) j) 'r 'rows [] '{j feat}))
      "feat is the inner digit's radix: a zero feat empties the loop"))

(deftest an-extent-may-not-depend-on-an-unresolved-local
  ;; `q = (quot (- 8 i) 2)` varies with the item: it is neither a digit nor invariant, so it may
  ;; not become an extent factor. Items (1,0) and (0,3) both address 3 in `i·q + j`.
  (is (nil? (ia/index-form '(+ (* i q) j) 'i 8
                           '[{:id y :init (- 8 i)} {:id q :init (quot y 2)}] '{j q}))))

(deftest loop-extents-must-be-invariant-radices
  (testing "a triangular loop index is not a digit, so the form is undecidable"
    (is (nil? (ia/index-form '(+ (* r 4) j) 'r 'rows [] '{j (+ r 1)})))
    (is (nil? (ia/index-form '(+ (* r 4) j) 'r 'rows [] '{j (quot n 2)}))))
  (testing "a coefficient may not mention a digit"
    (is (nil? (ia/index-form '(+ (+ (* t (* 2 (* h H))) (* h (* 2 h))) j) 'idx '(* T H)
                             '[{:id t :init (quot idx H)} {:id h :init (rem idx H)}]
                             '{j (* 2 h)})))))
