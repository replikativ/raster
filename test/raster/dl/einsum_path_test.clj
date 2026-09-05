(ns raster.dl.einsum-path-test
  "B2: n-operand einsum → an ORDERED PATH of pairwise contractions.

   Generality by decomposition: each step is a 2-operand `par/contract` the gate can route to a
   peak leaf, so an n-operand einsum needs no monolithic general kernel. Device-free — this
   validates the index algebra, the greedy order, and that executing the chain equals a direct
   reference computation."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.einsum :as es]
            [raster.par]))

;; ── the per-step set algebra (opt_einsum's find_contraction) ─────────────────────────
(deftest pair-contraction-index-algebra
  (testing "keep = needed downstream (output ∪ other operands); sum = only in this pair"
    (let [r (es/pair-contraction '[[:i :j] [:j :k] [:k :l]] 1 2 '[:i :l])]
      ;; contracting (jk, kl): j survives (operand 0 has it), l survives (output), k dies
      (is (= '[:j :l] (:keep r)))
      (is (= '[:k] (:sum r)))
      (is (= '[[:i :j] [:j :l]] (:remaining r)))))
  (testing "a label in the output is always kept, even if only one operand has it"
    (let [r (es/pair-contraction '[[:i :j] [:j :k]] 0 1 '[:i :k])]
      (is (= '[:i :k] (:keep r)))
      (is (= '[:j] (:sum r)))))
  (testing "a label in NEITHER output nor any other operand is summed"
    (let [r (es/pair-contraction '[[:i :j] [:i :j]] 0 1 '[])]
      (is (= [] (:keep r)))
      (is (= '[:i :j] (:sum r))))))

;; ── the greedy order ────────────────────────────────────────────────────────────────
(deftest contraction-path-is-greedy-and-deterministic
  (testing "2 operands → the single pair"
    (is (= [[0 1]] (es/contraction-path '[[:i :j] [:j :k]] '[:i :k] {:i 2 :j 3 :k 4}))))
  (testing "3 operands: picks the pair with the most memory removed, then the rest"
    ;; ij,jk,kl->il : (jk,kl)→jl removes more than (ij,jk)→ik, so it goes first
    (is (= [[1 2] [0 1]]
           (es/contraction-path '[[:i :j] [:j :k] [:k :l]] '[:i :l] {:i 2 :j 3 :k 4 :l 5}))))
  (testing "deterministic: same inputs → same path"
    (let [args ['[[:a :b] [:b :c] [:c :d] [:d :e]] '[:a :e] {:a 2 :b 3 :c 4 :d 5 :e 6}]]
      (is (= (apply es/contraction-path args) (apply es/contraction-path args)))))
  (testing "operands sharing no label still get paired (outer-product fallback, no hang)"
    (is (= 1 (count (es/contraction-path '[[:i] [:j]] '[:i :j] {:i 2 :j 3}))))))

;; ── the emitted step chain ──────────────────────────────────────────────────────────
(deftest steps-are-pairwise-contractions-landing-on-the-output
  (let [{:keys [steps result path]}
        (es/einsum->contract-steps "ij,jk,kl->il" {:i 2 :j 3 :k 4 :l 5} 'OUT '[A B C])]
    (testing "one step per path entry; each is a par/contract"
      (is (= (count path) (count steps)))
      (is (every? #(= 'raster.par/contract (first (:form %))) steps)))
    (testing "each step contracts exactly the labels that die there"
      (is (= '[[k 4]] (nth (:form (first steps)) 3)))
      (is (= '[[j 3]] (nth (:form (second steps)) 3))))
    (testing "the LAST step lands on the requested output layout (no trailing transpose)"
      (is (= '[:i :l] (:labels (peek steps))))
      (is (= '[[i 2] [l 5]] (nth (:form (peek steps)) 2)))
      (is (= 'OUT result)))
    (testing "intermediates are fresh and threaded, not aliased to inputs"
      (is (not-any? #{'A 'B 'C} (map :out steps))))))

;; ── executing the chain == a direct reference computation ────────────────────────────
(defn- run-steps
  "Execute a step chain on CPU: eval each par/contract form with its operands bound."
  [{:keys [steps result]} env]
  (let [env (reduce (fn [env {:keys [out form shape]}]
                      (let [syms (vec (keys env))
                            buf (double-array (reduce * 1 shape))
                            f (eval (list 'fn (conj syms out) form))]
                        (apply f (conj (mapv env syms) buf))
                        (assoc env out buf)))
                    env steps)]
    (vec (get env result))))

(deftest chained-steps-equal-the-reference
  (testing "ij,jk,kl->il via a 2-step path == direct triple loop"
    (let [I 2 J 3 K 4 L 5
          A (double-array (map #(* 0.1 (inc %)) (range (* I J))))
          B (double-array (map #(* 0.2 (inc %)) (range (* J K))))
          C (double-array (map #(* 0.3 (inc %)) (range (* K L))))
          plan (es/einsum->contract-steps "ij,jk,kl->il" {:i I :j J :k K :l L} 'OUT '[A B C])
          got (run-steps plan {'A A 'B B 'C C})
          ref (vec (for [i (range I) l (range L)]
                     (reduce + (for [j (range J) k (range K)]
                                 (* (aget A (+ (* i J) j))
                                    (aget B (+ (* j K) k))
                                    (aget C (+ (* k L) l)))))))]
      (is (= (count ref) (count got)))
      (is (every? true? (map #(< (Math/abs (- %1 %2)) 1e-9) got ref)))))
  (testing "4 operands, ab,bc,cd,de->ae"
    (let [dm {:a 2 :b 3 :c 2 :d 3 :e 2}
          mk (fn [n seed] (double-array (map #(* seed (inc %)) (range n))))
          A (mk 6 0.1) B (mk 6 0.2) C (mk 6 0.3) D (mk 6 0.4)
          plan (es/einsum->contract-steps "ab,bc,cd,de->ae" dm 'OUT '[A B C D])
          got (run-steps plan {'A A 'B B 'C C 'D D})
          ref (vec (for [a (range 2) e (range 2)]
                     (reduce + (for [b (range 3) c (range 2) d (range 3)]
                                 (* (aget A (+ (* a 3) b)) (aget B (+ (* b 2) c))
                                    (aget C (+ (* c 3) d)) (aget D (+ (* d 2) e)))))))]
      (is (= 3 (count (:steps plan))))
      (is (every? true? (map #(< (Math/abs (- %1 %2)) 1e-9) got ref)))))
  (testing "the 2-operand case still reduces to a single step == einsum->contract-form"
    (let [plan (es/einsum->contract-steps "ij,jk->ik" {:i 2 :j 3 :k 4} 'OUT '[A B])]
      (is (= 1 (count (:steps plan))))
      (is (= (es/einsum->contract-form "ij,jk->ik" {:i 2 :j 3 :k 4} 'OUT '[A B])
             (:form (first (:steps plan))))))))

;; ── G1: UNARY einsums (transpose / axis-reduction / diagonal) ────────────────────────
(deftest unary-einsums-lower-and-compute
  (testing "ij->ji is a 0-contract transpose; body reads the INPUT map's index"
    (let [f (es/einsum->contract-form "ij->ji" {:i 2 :j 3} 'O '[A])]
      (is (= '[[j 3] [i 2]] (nth f 2)))          ; free axes in OUTPUT order
      (is (= [] (nth f 3)))
      (is (= '(aget A (clojure.core/+ (clojure.core/* i 3) j)) (nth f 4)))))
  (testing "ij->i reduces the summed axis"
    (let [f (es/einsum->contract-form "ij->i" {:i 2 :j 3} 'O '[A])]
      (is (= '[[i 2]] (nth f 2)))
      (is (= '[[j 3]] (nth f 3)))))
  (testing "ii->i is the DIAGONAL — a repeated label makes the axis-map emit i·N + i"
    (let [f (es/einsum->contract-form "ii->i" {:i 3} 'O '[A])]
      (is (= '[[i 3]] (nth f 2)))
      (is (= [] (nth f 3)))
      (is (= '(aget A (clojure.core/+ (clojure.core/* i 3) i)) (nth f 4)))))
  (testing "a unary einsum yields exactly ONE step (the greedy path is empty)"
    (let [plan (es/einsum->contract-steps "ij->ji" {:i 2 :j 3} 'O '[A])]
      (is (= 1 (count (:steps plan))))
      (is (= [] (:path plan)))
      (is (= 'O (:result plan)))))
  (testing "executing the unary steps == reference"
    (let [A (double-array [1 2 3 4 5 6])            ; 2×3
          B (double-array [1 2 3 4 5 6 7 8 9])]     ; 3×3
      (is (= (run-steps (es/einsum->contract-steps "ij->ji" {:i 2 :j 3} 'O '[A]) {'A A})
             (vec (for [j (range 3) i (range 2)] (aget A (+ (* i 3) j))))))
      (is (= (run-steps (es/einsum->contract-steps "ij->i" {:i 2 :j 3} 'O '[A]) {'A A})
             (vec (for [i (range 2)] (reduce + (for [j (range 3)] (aget A (+ (* i 3) j))))))))
      (is (= (run-steps (es/einsum->contract-steps "ii->i" {:i 3} 'O '[B]) {'B B})
             (vec (for [i (range 3)] (aget B (+ (* i 3) i)))))))))

;; ── coverage ledger: what the contract path expresses, asserted EXPLICITLY ───────────
;; "Properly covered" has to be checkable, not claimed. This pins the current frontier so it
;; cannot silently regress, and names the remaining gap (G2: scalar output / 0 free axes).
(deftest coverage-ledger-of-the-contract-path
  (let [expressible? (fn [sub dm syms]
                       (try (es/einsum->contract-form sub dm 'O syms) true
                            (catch Throwable _ false)))]
    (testing "EXPRESSIBLE: binary contractions, broadcasts, and all unary forms"
      (doseq [[sub dm syms] [["ij,jk->ik"    {:i 2 :j 3 :k 4}       '[A B]]
                             ["bij,bjk->bik" {:b 2 :i 2 :j 3 :k 4}  '[A B]]
                             ["ij,ij->ij"    {:i 2 :j 3}            '[A B]]
                             ["i,j->ij"      {:i 2 :j 3}            '[A B]]
                             ["ij->ji"       {:i 2 :j 3}            '[A]]
                             ["ij->i"        {:i 2 :j 3}            '[A]]
                             ["ii->i"        {:i 3}                 '[A]]]]
        (is (expressible? sub dm syms) (str sub " should be expressible"))))
    (testing "EXPRESSIBLE: scalar output too — the (0 free, n contract) reduction cell (G2)"
      (doseq [[sub dm syms] [["ij->"  {:i 2 :j 3} '[A]]
                             ["ii->"  {:i 3}      '[A]]
                             ["i,i->" {:i 3}      '[A B]]]]
        (is (expressible? sub dm syms) (str sub " should be expressible"))))
    (testing "COVERAGE IS COMPLETE: every subscript in einsum's own test suite lowers"
      (doseq [[sub dm syms] [["ij,jk->ik" {:i 2 :j 3 :k 4} '[A B]]
                             ["bij,bjk->bik" {:b 2 :i 2 :j 3 :k 4} '[A B]]
                             ["ij,ij->ij" {:i 2 :j 3} '[A B]] ["i,j->ij" {:i 2 :j 3} '[A B]]
                             ["i,i->" {:i 3} '[A B]] ["ij->ji" {:i 2 :j 3} '[A]]
                             ["ij->i" {:i 2 :j 3} '[A]] ["ij->" {:i 2 :j 3} '[A]]
                             ["ii->" {:i 3} '[A]] ["ii->i" {:i 3} '[A]]]]
        (is (expressible? sub dm syms) (str sub " — all 10 einsum-test subscripts must lower"))))))

;; ── G2: SCALAR output — the (0 free, n contract) cell of contract's algebra ───────────
;; contract's algebra has four cells: (n free, 0 contract) = map, (n, n) = contraction,
;; (0, n) = FULL REDUCTION, and 0 free axes is a legal result in that algebra — so the
;; primitive supports it. It needs no new IR or emitter: the SegSpace degenerates to the 1-D
;; shape (segop/seg-space-1d?) that the existing two-phase tree reduction already consumes.
(deftest scalar-output-is-the-reduction-cell
  (testing "the macro: 0 free axes = the empty product, one iteration, writes out[0]"
    (let [A (double-array [1 2 3 4]) B (double-array [1 2 3 4]) O (double-array 1)]
      (raster.par/contract O [] [[i 4]]
                           (clojure.core/* (clojure.core/aget A i) (clojure.core/aget B i)))
      (is (= 30.0 (aget O 0)))))                          ; dot product
  (testing "sum over all axes (2 contract axes, flattened)"
    (let [A (double-array [1 2 3 4 5 6]) O (double-array 1)]
      (raster.par/contract O [] [[i 6]] (clojure.core/aget A i))
      (is (= 21.0 (aget O 0)))))
  (testing "trace: a repeated label gives the diagonal, reduced to a scalar"
    (let [M (double-array [1 2 3 4 5 6 7 8 9]) O (double-array 1)]
      (raster.par/contract O [] [[i 3]]
                           (clojure.core/aget M (clojure.core/+ (clojure.core/* i 3) i)))
      (is (= 15.0 (aget O 0)))))
  (testing "einsum lowers all three scalar-output subscripts"
    (doseq [[sub dm syms] [["i,i->" {:i 3} '[A B]] ["ij->" {:i 2 :j 3} '[A]] ["ii->" {:i 3} '[A]]]]
      (let [f (es/einsum->contract-form sub dm 'O syms)]
        (is (= [] (nth f 2)) (str sub " has 0 free axes"))
        (is (seq (nth f 3)) (str sub " contracts at least one axis")))))
  (testing "the router sends it to the two-phase reduction, with its own invoke protocol"
    (let [r ((requiring-resolve 'raster.compiler.passes.parallel.contract-route/route-contraction)
             '(raster.par/contract O [] [[i 8]] (* (aget A i) (aget B i))) :dtype :double)]
      (is (= :full-reduce (:strategy r)))
      (is (= :reduction (:invoke r)))
      (is (nil? (:n-phases r))
          "one scheduled artifact does not claim ownership of a hidden second phase")
      (is (= 1 (:out-elems r))))))
