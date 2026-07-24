(ns raster.compiler.ir.axis-map-test
  "Phase 1/2: the indexing-map algebra. Device-free — this is pure index data.
   Covers the cases that used to be ad-hoc special cases (:nn/:nt, transposed output) plus the
   ones that were not expressible at all (merge/split, broadcast, diagonal, batch)."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.axis-map :as am]))

(deftest plain-row-major-index
  (testing "A[i,l] over [M K] → i*K + l"
    (is (= '(clojure.core/+ (clojure.core/* i 4) l)
           (am/index-expr (am/of-axes '[[i 3] [l 4]])))))
  (testing "innermost axis contributes bare (no ×1)"
    (is (= 'l (am/index-expr (am/of-axes '[[l 4]])))))
  (testing "shape / n-elements"
    (is (= [3 4] (am/shape (am/of-axes '[[i 3] [l 4]]))))
    (is (= 12 (am/n-elements (am/of-axes '[[i 3] [l 4]]))))))

(deftest orientation-is-data-not-a-special-case
  (let [A  (am/of-axes '[[i 3] [l 4]])
        Bnn (am/of-axes '[[l 4] [j 2]])     ; B[K,N]  (the old ":nn")
        Bnt (am/of-axes '[[j 2] [l 4]])]    ; B[N,K]  (the old ":nt")
    (testing "the two orientations are just different maps, with different indices"
      (is (= '(clojure.core/+ (clojure.core/* l 2) j) (am/index-expr Bnn)))
      (is (= '(clojure.core/+ (clojure.core/* j 4) l) (am/index-expr Bnt))))
    (testing "canonical? replaces the (+ (* outer stride) inner) pattern match"
      (is (am/canonical? A '[i l]))
      (is (am/canonical? Bnt '[j l]))
      (is (not (am/canonical? Bnn '[j l]))))
    (testing "the mismatch is a PERMUTATION — the general form of the :nn→:nt rewrite"
      (is (= [1 0] (am/permutation Bnn Bnt)))
      (is (am/transposed-2d? Bnn Bnt))
      (is (not (am/transposed-2d? A A))))))

(deftest merge-and-split
  (testing "einops \"b (c h) w\": c,h merged into ONE physical dim"
    (let [m (am/of-groups '[[[b 2]] [[c 3] [h 4]] [[w 5]]])]
      (is (= [2 12 5] (am/shape m)))                       ; group extent = 3*4
      (is (= 120 (am/n-elements m)))
      ;; index = b*(12*5) + (c*4 + h)*5 + w
      (is (= '(clojure.core/+ (clojure.core/* b 60)
                              (clojure.core/* (clojure.core/+ (clojure.core/* c 4) h) 5)
                              w)
             (am/index-expr m)))))
  (testing "split is the same structure read the other way; a regroup is NOT a permutation"
    (let [merged (am/of-groups '[[[b 2]] [[c 3] [h 4]]])
          split  (am/of-axes   '[[b 2] [c 3] [h 4]])]
      (is (am/same-atomic-order? merged split))            ; ⇒ pure reshape, NO data movement
      (is (nil? (am/permutation merged split)))            ; not a reordering
      (is (= (am/n-elements merged) (am/n-elements split)))
      ;; and crucially the flat index is IDENTICAL — a reinterpretation, not a copy
      (is (= (am/index-expr merged) (am/index-expr split))))))

(deftest broadcast-diagonal-batch
  (testing "broadcast: the operand simply omits an iteration axis"
    (let [bias (am/of-axes '[[j 5]])]
      (is (= 'j (am/index-expr bias)))
      (is (= '[j] (am/axes bias)))))
  (testing "diagonal: an axis repeats"
    (is (= '(clojure.core/+ (clojure.core/* i 3) i)
           (am/index-expr (am/of-axes '[[i 3] [i 3]])))))
  (testing "batch: an extra leading group, no new concept"
    (let [m (am/of-axes '[[b 2] [i 3] [l 4]])]
      (is (= [2 3 4] (am/shape m)))
      (is (= '[b i l] (am/axes m))))))

(deftest symbolic-extents
  (testing "symbolic extents stay symbolic; literals still fold"
    (let [m (am/of-axes '[[i m] [l k]])]
      (is (= '(clojure.core/+ (clojure.core/* i k) l) (am/index-expr m)))
      (is (= '[m k] (am/shape m)))
      (is (= '(clojure.core/* m k) (am/n-elements m))))))
