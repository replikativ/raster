(ns raster.compiler.passes.parallel.gemm-recognize-test
  "#38 Stage 1 — the tileable-redomap (GEMM) matcher. Device-free. The matcher is
   the riskiest logic in #38 (a false positive → a tiled XMX kernel emitted for a
   non-GEMM redomap = silent miscompile), so this corpus pins both the POSITIVE
   extractions (all four transpose variants, α/β) and — load-bearing — the
   NEGATIVE cases that MUST reject."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.passes.parallel.gemm-recognize :as gr]))

;; Canonical row-major GEMM redomap: C[m×n] = A[m×k]·B[k×n].
;;   outer dotimes i∈[0,m), j∈[0,n); C at row-major (i,j,n)
;;   inner reduce p∈[0,k); acc += A[i,p]·B[p,j]
(defn- gemm
  "Build a nested-dotimes GEMM with the given operand index exprs + optional
   accumulating (β=1) store."
  [idx-a idx-b & {:keys [beta] :or {beta 0}}]
  (let [dot (list 'loop ['acc 0.0 'p 0]
                  (list 'if '(< p k)
                        (list 'recur (list '+ 'acc (list '* (list 'aget 'A idx-a) (list 'aget 'B idx-b)))
                              '(inc p))
                        'acc))
        stored (if (= beta 1) (list '+ '(aget C (+ (* i n) j)) dot) dot)]
    (list 'dotimes '[i m]
          (list 'dotimes '[j n]
                (list 'aset 'C '(+ (* i n) j) stored)))))

(deftest matches-all-transpose-variants
  (testing ":nn — A[i,p] B[p,j]"
    (let [r (gr/match-gemm-loop-nest (gemm '(+ (* i k) p) '(+ (* p n) j)))]
      (is (= {:variant :nn :A 'A :B 'B :C 'C :m 'm :n 'n :k 'k :alpha 1.0 :beta 0.0} r))))
  (testing ":nt — B transposed, B[j,p]"
    (is (= :nt (:variant (gr/match-gemm-loop-nest (gemm '(+ (* i k) p) '(+ (* j k) p)))))))
  (testing ":tn — A transposed, A[p,i]"
    (is (= :tn (:variant (gr/match-gemm-loop-nest (gemm '(+ (* p m) i) '(+ (* p n) j)))))))
  (testing ":tt — both transposed"
    (is (= :tt (:variant (gr/match-gemm-loop-nest (gemm '(+ (* p m) i) '(+ (* j k) p))))))))

(deftest extraction-contract-matches-blas-shape
  (testing "emits the same keys as gpu-plan/match-gemm-call"
    (let [r (gr/match-gemm-loop-nest (gemm '(+ (* i k) p) '(+ (* p n) j)))]
      (is (= #{:variant :A :B :C :m :k :n :alpha :beta} (set (keys r)))))))

(deftest beta-and-commutativity
  (testing "accumulating store (C += A·B) is β=1"
    (is (= 1.0 (:beta (gr/match-gemm-loop-nest (gemm '(+ (* i k) p) '(+ (* p n) j) :beta 1))))))
  (testing "fresh store is β=0"
    (is (= 0.0 (:beta (gr/match-gemm-loop-nest (gemm '(+ (* i k) p) '(+ (* p n) j)))))))
  (testing "product is commutative — B·A written still resolves A=A, B=B by index shape"
    (let [comm (list 'dotimes '[i m]
                     (list 'dotimes '[j n]
                           (list 'aset 'C '(+ (* i n) j)
                                 (list 'loop ['acc 0.0 'p 0]
                                       (list 'if '(< p k)
                                             (list 'recur '(+ acc (* (aget B (+ (* p n) j)) (aget A (+ (* i k) p)))) '(inc p))
                                             'acc)))))
          r (gr/match-gemm-loop-nest comm)]
      (is (= 'A (:A r)))
      (is (= 'B (:B r)))
      (is (= :nn (:variant r))))))

(deftest widening-cast-store
  (testing "a cast on the stored value (double …) is stripped, still matches"
    (let [casted (list 'dotimes '[i m]
                       (list 'dotimes '[j n]
                             (list 'aset 'C '(+ (* i n) j)
                                   (list 'double
                                         (list 'loop ['acc 0.0 'p 0]
                                               (list 'if '(< p k)
                                                     (list 'recur '(+ acc (* (aget A (+ (* i k) p)) (aget B (+ (* p n) j)))) '(inc p))
                                                     'acc))))))]
      (is (= :nn (:variant (gr/match-gemm-loop-nest casted)))))))

(deftest rejects-non-gemm-redomaps
  (testing "both operands indexed by the SAME output position (no reduce variance) → nil"
    (is (nil? (gr/match-gemm-loop-nest (gemm '(+ (* i n) j) '(+ (* i n) j))))))
  (testing "reduce of a single array (not a product) → nil"
    (let [sum1 (list 'dotimes '[i m]
                     (list 'dotimes '[j n]
                           (list 'aset 'C '(+ (* i n) j)
                                 (list 'loop ['acc 0.0 'p 0]
                                       (list 'if '(< p k)
                                             (list 'recur '(+ acc (aget A (+ (* i k) p))) '(inc p))
                                             'acc)))))]
      (is (nil? (gr/match-gemm-loop-nest sum1)))))
  (testing "accumulator not initialized to 0 → nil (not a fresh dot product)"
    (let [nz (list 'dotimes '[i m]
                   (list 'dotimes '[j n]
                         (list 'aset 'C '(+ (* i n) j)
                               (list 'loop ['acc 1.0 'p 0]
                                     (list 'if '(< p k)
                                           (list 'recur '(+ acc (* (aget A (+ (* i k) p)) (aget B (+ (* p n) j)))) '(inc p))
                                           'acc)))))]
      (is (nil? (gr/match-gemm-loop-nest nz)))))
  (testing "C written column-major (+ (* j n) i) → nil (index shape mismatch)"
    (let [badc (list 'dotimes '[i m]
                     (list 'dotimes '[j n]
                           (list 'aset 'C '(+ (* j n) i)
                                 (list 'loop ['acc 0.0 'p 0]
                                       (list 'if '(< p k)
                                             (list 'recur '(+ acc (* (aget A (+ (* i k) p)) (aget B (+ (* p n) j)))) '(inc p))
                                             'acc)))))]
      (is (nil? (gr/match-gemm-loop-nest badc)))))
  (testing "not a nested dotimes at all → nil"
    (is (nil? (gr/match-gemm-loop-nest '(dotimes [i m] (aset C i (aget A i))))))
    (is (nil? (gr/match-gemm-loop-nest '(+ 1 2))))))

;; ── binding-level entry point (do-unwrap + :result-sym for rewrite-gemm) ──────────
(deftest binding-level-match-gemm-redomap
  (let [nest (gemm '(+ (* i k) p) '(+ (* p n) j))]
    (testing "a bare nested-dotimes binding value → result-sym defaults to the output C"
      (is (= 'C (:result-sym (gr/match-gemm-redomap nest)))))
    (testing "a (do (dotimes …) Cout) wrapper → result-sym is the do's tail buffer"
      (let [r (gr/match-gemm-redomap (list 'do nest 'Cout))]
        (is (= :nn (:variant r)))
        (is (= 'C (:C r)))
        (is (= 'Cout (:result-sym r)))))
    (testing "the descriptor still carries the full rewrite-gemm contract"
      (is (= #{:variant :A :B :C :m :k :n :alpha :beta :result-sym}
             (set (keys (gr/match-gemm-redomap nest))))))
    (testing "a non-GEMM do body → nil"
      (is (nil? (gr/match-gemm-redomap '(do (dotimes [i n] (aset C i (aget A i))) C)))))
    (testing "a plain non-loop expr → nil"
      (is (nil? (gr/match-gemm-redomap '(+ 1 2)))))))
