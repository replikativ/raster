(ns raster.compiler.ir.index-algebra-test
  "Injectivity facts over mixed-radix index forms: what the frontend may certify as `:unique`."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.index-algebra :as ia]
            [raster.compiler.ir.kernel-launch :as launch]))

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

(deftest launch-storage-products-share-the-index-monomial-algebra
  (let [extent (launch/product (launch/runtime-value 'rows)
                               (launch/runtime-value 'columns))
        locals '[{:id row :init (quot index columns)}
                 {:id column :init (rem index columns)}]
        form (ia/index-form '(+ (* row columns) column) 'index extent locals {})]
    (is (= {:const 1 :factors '[columns rows]} (ia/monomial extent)))
    (is (ia/injective? form))))

(deftest empty-nonpositive-rectangular-domains-retain-their-product
  (is (= {:const 1 :factors '[columns rows]}
         (ia/monomial '(if (< rows 1) 0
                           (if (< columns 1) 0 (* rows columns))))))
  (is (nil? (ia/monomial '(if (< unrelated 1) 0 (* rows columns))))
      "an unrelated guard is not evidence about a rectangular product")
  (is (nil? (ia/monomial '(if (< rows 1) 7 (* rows columns))))
      "a nonempty alternative is not an empty-domain guard"))

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

(deftest dropped-digit-is-injective-only-on-a-proven-fixed-domain
  (let [locals '[{:id q :init (quot idx 4)} {:id r :init (rem idx 4)}]
        form (ia/index-form 'q 'idx 20 locals {})]
    (is (not (ia/injective? form)))
    (doseq [r (range 4)]
      (let [restricted (ia/restrict-fixed-leaves form {'r r})
            indices (filter #(= r (rem % 4)) (range 20))
            addresses (map #(quot % 4) indices)]
        (is (ia/injective? restricted))
        (is (= {'r r} (:fixed-leaves restricted)))
        (is (nil? (ia/restrict-fixed-leaves restricted {})) "cannot erase an existing domain witness")
        (is (= (count addresses) (count (distinct addresses))))))
    (doseq [fixed [{'q 0} {'idx 0} {'unknown 0} {'r 0.5}]]
      (is (nil? (ia/restrict-fixed-leaves form fixed))))
    (is (not (ia/injective? (ia/restrict-fixed-leaves form {}))))))

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

(deftest dense-forms-distinguish-complete-images-from-injective-padding
  (let [dense (ia/index-form '(+ (* r 4) j) 'r 'rows [] '{j 4})
        padded (ia/index-form '(+ (* r 5) j) 'r 'rows [] '{j 4})
        translated (ia/index-form '(+ base (* r width) j) 'r 'rows [] '{j width})]
    (is (ia/dense? dense))
    (is (ia/injective? padded))
    (is (not (ia/dense? padded)) "an unconstrained row stride may leave holes")
    (is (ia/injective? translated))
    (is (not (ia/dense? translated)) "a translated tile does not cover an allocation from zero")))

(deftest represented-digits-have-an-exact-span-even-when-a-broadcast-digit-is-dropped
  (let [locals '[{:id q :init (quot idx width)}
                 {:id column :init (rem idx width)}
                 {:id row :init (quot q heads)}
                 {:id head :init (rem q heads)}]
        broadcast (ia/index-form '(+ (* row width) column) 'idx '(* rows (* heads width))
                                 locals {})
        padded (ia/index-form '(+ (* row stride) column) 'idx '(* rows (* heads width))
                              locals {})
        translated (ia/index-form '(+ base (* row width) column) 'idx
                                  '(* rows (* heads width)) locals {})]
    (is (not (ia/dense? broadcast)) "different heads repeat the same input rows")
    (is (= {:const 1 :factors '[rows width]}
           (ia/zero-based-dense-span broadcast))
        "the represented row/column digits still enumerate one exact input interval")
    (is (nil? (ia/zero-based-dense-span padded)) "an unrelated stride may leave gaps")
    (is (nil? (ia/zero-based-dense-span translated))
        "an invariant translation needs an independent base/capacity proof")))

(deftest consecutive-translated-slabs-form-one-dense-image
  (let [low (ia/index-form '(+ (* row (* 2 half)) i) 'row 'rows [] '{i half})
        high (ia/index-form '(+ (* row (* 2 half)) i half) 'row 'rows [] '{i half})
        gap (ia/index-form '(+ (* row (* 3 half)) i (* 2 half)) 'row 'rows [] '{i half})]
    (is (not (ia/dense? low)) "one half-row is injective but incomplete")
    (is (ia/dense-translated-forms? [low high]))
    (is (not (ia/dense-translated-forms? [low gap]))
        "disjoint translated stores with a missing middle slab are not complete")))

(deftest common-symbolic-translations-cancel-relationally
  (let [low (ia/index-form '(+ base i) 'i 'half [] {})
        high (ia/index-form '(+ base i half) 'i 'half [] {})
        unrelated (ia/index-form '(+ base i stride) 'i 'half [] {})]
    (is (ia/injective? low))
    (is (ia/injective? high))
    (is (= [{:const 1 :factors '[base]} {:const 1 :factors '[half]}]
           (:offset-terms high)))
    (is (ia/disjoint-translated-forms? [low high])
        "translation by base does not alter the two adjacent half-width slabs")
    (is (not (ia/disjoint-translated-forms? [low unrelated]))
        "an unrelated symbolic displacement carries no separation proof")))

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

(deftest a-product-of-a-sum-by-an-invariant-distributes
  ;; attention prefill: sc[(i·n-q + hq)·nrows + j] over idx with digits i, hq, j
  (let [extent '(clojure.core/* nrows (clojure.core/* n-q nrows))
        locals '[{:id per-i :init (clojure.core/* n-q nrows)}
                 {:id i :init (clojure.core/quot idx per-i)}
                 {:id rest0 :init (clojure.core/rem idx per-i)}
                 {:id hq :init (clojure.core/quot rest0 nrows)}
                 {:id j :init (clojure.core/rem rest0 nrows)}
                 {:id row :init (clojure.core/+ (clojure.core/* i n-q) hq)}]
        form (ia/index-form '(clojure.core/+ (clojure.core/* row nrows) j) 'idx extent locals {})]
    (is (some? form))
    (is (ia/injective? form))
    (is (= '{i {:const 1 :factors [n-q nrows]} hq {:const 1 :factors [nrows]} j {:const 1 :factors []}}
           (into {} (map (fn [[digit {:keys [coefficient]}]] [digit coefficient])) (:terms form)))))
  (testing "a scaled constant becomes a symbolic offset"
    (let [form (ia/index-form '(clojure.core/* (clojure.core/+ i 1) n) 'i 'm [] {})]
      (is (= {:const 1 :factors '[n]} (:offset form)))))
  (testing "a product of two digit-carrying factors is not affine"
    (is (nil? (ia/index-form '(clojure.core/* (clojure.core/+ i 1) (clojure.core/+ j 1)) 'idx
                             '(clojure.core/* m n)
                             '[{:id i :init (clojure.core/quot idx n)} {:id j :init (clojure.core/rem idx n)}]
                             {})))))
