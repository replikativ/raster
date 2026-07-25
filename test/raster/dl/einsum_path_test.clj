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
