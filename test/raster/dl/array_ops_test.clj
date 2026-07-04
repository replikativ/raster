(ns raster.dl.array-ops-test
  (:require [clojure.test :refer [deftest testing is use-fixtures]]
            [raster.dl.array-ops :as ops]
            [raster.dl.gsdm :as gsdm]
            [raster.dl.nn :as nn]
            [raster.ad.templates :as tmpl]
            [raster.linalg.blas :as blas]))

(use-fixtures :once
  (fn [f] (if (blas/available?) (f) (println "[SKIP] No BLAS library"))))

(defn- approx=
  ([a b] (approx= a b 1e-6))
  ([a b eps] (< (Math/abs (- (double a) (double b))) (double eps))))

(defn- arr-finite?
  [^doubles arr]
  (every? #(Double/isFinite %) arr))

(defn- arr-approx=
  "Check two arrays are approximately equal."
  ([^doubles a ^doubles b] (arr-approx= a b 1e-6))
  ([^doubles a ^doubles b eps]
   (and (= (alength a) (alength b))
        (every? identity
                (map (fn [i] (approx= (aget a (int i)) (aget b (int i)) eps))
                     (range (alength a)))))))

;; ================================================================
;; Forward op tests
;; ================================================================

(deftest array-add-test
  (testing "elementwise addition"
    (let [a (double-array [1.0 2.0 3.0])
          b (double-array [4.0 5.0 6.0])
          out (ops/array-add a b 3)]
      (is (= 3 (alength out)))
      (is (approx= 5.0 (aget out 0)))
      (is (approx= 7.0 (aget out 1)))
      (is (approx= 9.0 (aget out 2))))))

(deftest broadcast-add-test
  (testing "broadcast addition: h[b*dim+d] + t[d]"
    (let [h (double-array [1 2 3 4 5 6])  ;; 2 batches, dim=3
          t (double-array [10 20 30])
          out (ops/broadcast-add h t 2 3)]
      (is (= 6 (alength out)))
      (is (approx= 11.0 (aget out 0)))
      (is (approx= 22.0 (aget out 1)))
      (is (approx= 33.0 (aget out 2)))
      (is (approx= 14.0 (aget out 3)))
      (is (approx= 25.0 (aget out 4)))
      (is (approx= 36.0 (aget out 5))))))

;; ================================================================
;; Generic primitive forward tests
;; ================================================================

(deftest scatter-add-test
  (testing "scatter-add accumulates correctly"
    (let [vals (double-array [1.0 2.0 3.0 4.0])  ;; 2 pairs, stride=2
          indices (long-array [0 0])  ;; both go to dest 0
          out (ops/scatter-add vals indices 2 2 2)]
      (is (= 4 (alength out)))  ;; 2 destinations * stride 2
      ;; Dest 0: sum of pairs 0,1 for each stride element
      (is (approx= 4.0 (aget out 0)))  ;; 1+3
      (is (approx= 6.0 (aget out 1)))  ;; 2+4
      ;; Dest 1: no pairs → 0
      (is (approx= 0.0 (aget out 2)))
      (is (approx= 0.0 (aget out 3)))))

  (testing "scatter-add with different destinations"
    (let [vals (double-array [1.0 2.0 3.0 4.0])  ;; 2 pairs, stride=2
          indices (long-array [0 1])  ;; pair 0→dest 0, pair 1→dest 1
          out (ops/scatter-add vals indices 2 2 2)]
      (is (approx= 1.0 (aget out 0)))
      (is (approx= 2.0 (aget out 1)))
      (is (approx= 3.0 (aget out 2)))
      (is (approx= 4.0 (aget out 3))))))

(deftest gather-test
  (testing "gather selects correctly"
    (let [src (double-array [10.0 20.0 30.0 40.0])  ;; 2 entities, stride=2
          indices (long-array [1 0 1])  ;; 3 pairs gathering from entities
          out (ops/gather src indices 2 3 2)]
      (is (= 6 (alength out)))  ;; 3 pairs * stride 2
      ;; Pair 0: entity 1 → [30, 40]
      (is (approx= 30.0 (aget out 0)))
      (is (approx= 40.0 (aget out 1)))
      ;; Pair 1: entity 0 → [10, 20]
      (is (approx= 10.0 (aget out 2)))
      (is (approx= 20.0 (aget out 3)))
      ;; Pair 2: entity 1 → [30, 40]
      (is (approx= 30.0 (aget out 4)))
      (is (approx= 40.0 (aget out 5))))))

(deftest scatter-add-gather-adjoint-test
  (testing "scatter-add and gather are adjoints"
    ;; For any x, y: <scatter-add(x), y> = <x, gather(y)>
    (let [n-pairs 3 n-dst 2 stride 2
          x (double-array [1 2 3 4 5 6])
          y (double-array [10 20 30 40])
          indices (long-array [0 1 0])
          sx (ops/scatter-add x indices n-dst n-pairs stride)
          gy (ops/gather y indices n-dst n-pairs stride)
          ;; <scatter-add(x), y>
          lhs (loop [i 0 s 0.0]
                (if (< i (alength sx))
                  (recur (inc i) (+ s (* (aget sx i) (aget y i))))
                  s))
          ;; <x, gather(y)>
          rhs (loop [i 0 s 0.0]
                (if (< i (alength x))
                  (recur (inc i) (+ s (* (aget x i) (aget gy i))))
                  s))]
      (is (approx= lhs rhs 1e-10)))))

(deftest indexed-dot-test
  (testing "indexed-dot on simple multi-head case"
    (let [n-nodes 3 n-edges 2 emb-dim 4 n-heads 2
          dk (quot emb-dim n-heads)
          ;; A: [n-nodes * emb-dim]
          A (double-array [1 2 3 4  5 6 7 8  9 10 11 12])
          B (double-array [1 0 1 0  0 1 0 1  1 1 1 1])
          idx-a (long-array [0 1])  ;; edge 0: node 0, edge 1: node 1
          idx-b (long-array [1 2])  ;; edge 0: node 1, edge 1: node 2
          out (ops/indexed-dot A B idx-a idx-b n-nodes n-nodes
                               n-edges dk emb-dim n-heads)]
      (is (= (* n-edges n-heads) (alength out)))
      ;; Edge 0, head 0: A[0*4+0*2+0..1]·B[1*4+0*2+0..1] = [1,2]·[0,1] = 2
      (is (approx= 2.0 (aget out 0)))
      ;; Edge 0, head 1: A[0*4+1*2+0..1]·B[1*4+1*2+0..1] = [3,4]·[0,1] = 4
      (is (approx= 4.0 (aget out 1)))
      ;; Edge 1, head 0: A[1*4+0..1]·B[2*4+0..1] = [5,6]·[1,1] = 11
      (is (approx= 11.0 (aget out 2)))
      ;; Edge 1, head 1: A[1*4+2..3]·B[2*4+2..3] = [7,8]·[1,1] = 15
      (is (approx= 15.0 (aget out 3))))))

(deftest scatter-mul-add-test
  (testing "scatter-mul-add shape and values"
    (let [n-dst 2 n-src 3 n-pairs 3 n-heads 2 emb-dim 4 dk 2
          coeffs (double-array [1 1  2 2  3 3])  ;; 3 pairs * 2 heads
          src (double-array [1 1 1 1  2 2 2 2  3 3 3 3])  ;; 3 entities * 4 dim
          dst-indices (long-array [0 0 1])
          src-indices (long-array [0 1 2])
          out (ops/scatter-mul-add coeffs src dst-indices src-indices
                                   n-dst n-src n-pairs dk emb-dim n-heads)]
      (is (= (* n-dst emb-dim) (alength out)))
      (is (arr-finite? out))
      ;; Dest 0 gets contributions from pairs 0,1
      ;; head 0, d0: 1*1 + 2*2 = 5
      (is (approx= 5.0 (aget out 0))))))

(deftest scale-clamp-exp-test
  (testing "scale-clamp-exp basic behavior"
    (let [x (double-array [0.0 1.0 -1.0 100.0])
          out (ops/scale-clamp-exp x 1.0 5.0 4)]
      (is (= 4 (alength out)))
      (is (approx= 1.0 (aget out 0)))  ;; exp(0)
      (is (approx= (Math/exp 1.0) (aget out 1)))
      (is (approx= (Math/exp -1.0) (aget out 2)))
      (is (approx= (Math/exp 5.0) (aget out 3)))))  ;; clamped to 5

  (testing "scale-clamp-exp with scaling"
    (let [x (double-array [2.0])
          out (ops/scale-clamp-exp x 0.5 5.0 1)]
      (is (approx= (Math/exp 1.0) (aget out 0))))))

(deftest reduce-axis-test
  (testing "reduce-axis sums along first axis"
    (let [src (double-array [1 2 3  4 5 6  7 8 9])  ;; 3 rows, 3 cols
          out (ops/reduce-axis src 3 3)]
      (is (= 3 (alength out)))
      (is (approx= 12.0 (aget out 0)))  ;; 1+4+7
      (is (approx= 15.0 (aget out 1)))  ;; 2+5+8
      (is (approx= 18.0 (aget out 2))))))  ;; 3+6+9

(deftest segment-div-test
  (testing "segment-div divides correctly"
    (let [wV (double-array [10 20 30 40])  ;; 2 nodes, dim=2, 1 head
          Z (double-array [2.0 5.0])        ;; 2 nodes, 1 head
          out (ops/segment-div wV Z 2 2 1 0.0)]
      (is (approx= 5.0 (aget out 0)))
      (is (approx= 10.0 (aget out 1)))
      (is (approx= 6.0 (aget out 2)))
      (is (approx= 8.0 (aget out 3))))))

(deftest flat-embed-op-test
  (testing "flat embedding shape and values with state + position"
    (let [n-vars 3 emb-dim 4 n-spaces 2 n-states 4
          values (double-array [1.0 2.0 3.0])
          spaces (long-array [0 1 0])
          states (long-array [0 1 0])  ;; LATENT, OBSERVED, LATENT
          We (double-array [0.5 0.5 0.5 0.5])
          be (double-array [0.1 0.2 0.3 0.4])
          space-emb (double-array [1 2 3 4  5 6 7 8])       ;; 2 spaces * 4 dim
          state-emb (double-array (repeat (* n-states emb-dim) 0.0))  ;; zeros for predictability
          pos-emb (double-array (repeat (* n-vars emb-dim) 0.0))      ;; zeros for predictability
          out (ops/flat-embed-op values space-emb spaces state-emb states pos-emb
                                 We be n-vars emb-dim n-spaces n-states)]
      (is (= (* n-vars emb-dim) (alength out)))
      (is (arr-finite? out))
      ;; var 0: val=1, space=0, state-emb=0, pos-emb=0
      ;; → We[0]*1 + be[0] + space0[0] + 0 + 0 = 0.5+0.1+1 = 1.6
      (is (approx= 1.6 (aget out 0))))))

(deftest dot-rows-test
  (testing "per-row dot product"
    (let [h (double-array [1 2 3  4 5 6])  ;; 2 rows, dim=3
          W (double-array [1 0 1])
          bias (double-array [0.5])
          out (ops/dot-rows h W bias 2 3)]
      (is (= 2 (alength out)))
      ;; row 0: 1*1 + 2*0 + 3*1 + 0.5 = 4.5
      (is (approx= 4.5 (aget out 0)))
      ;; row 1: 4*1 + 5*0 + 6*1 + 0.5 = 10.5
      (is (approx= 10.5 (aget out 1))))))

(deftest masked-mse-loss-test
  (testing "MSE on non-observed variables"
    (let [pred (double-array [1.0 2.0 3.0 4.0])
          target (double-array [1.5 2.0 2.5 3.5])
          states (long-array [0 1 0 0])  ;; index 1 is observed
          loss (ops/masked-mse-loss pred target states 4)]
      (is (Double/isFinite loss))
      ;; non-observed: indices 0,2,3
      ;; diffs: -0.5, 0.5, 0.5 → squares: 0.25, 0.25, 0.25 → mean = 0.25
      (is (approx= 0.25 loss)))))

(deftest sinusoidal-embed-backward-test
  (testing "sinusoidal backward produces finite result"
    (let [dim 8
          dy (double-array (repeat dim 1.0))
          dt (gsdm/sinusoidal-embed-backward dy 50.0 dim)]
      (is (Double/isFinite dt)))))

;; ================================================================
;; Numerical gradient checks
;; ================================================================

(defn- numerical-grad
  "Compute numerical gradient of f(x) w.r.t. x[idx] using central differences."
  [f ^doubles x idx & {:keys [eps] :or {eps 1e-5}}]
  (let [x+ (double-array (alength x))
        x- (double-array (alength x))]
    (System/arraycopy x 0 x+ 0 (alength x))
    (System/arraycopy x 0 x- 0 (alength x))
    (aset x+ (int idx) (+ (aget x (int idx)) eps))
    (aset x- (int idx) (- (aget x (int idx)) eps))
    (/ (- (double (f x+)) (double (f x-))) (* 2.0 eps))))

(deftest array-add-gradient-test
  (testing "array-add gradient via finite differences"
    (let [n 4
          a (double-array [1.0 2.0 3.0 4.0])
          b (double-array [0.5 1.5 2.5 3.5])
          ;; Sum of output as scalar loss
          f-a (fn [^doubles a'] (let [out (ops/array-add a' b n)]
                                  (loop [i 0 s 0.0]
                                    (if (< i n) (recur (inc i) (+ s (aget out i))) s))))
          ;; Analytical gradient via rrule
          result (ops/array-add a b n)
          pullback-fn (tmpl/template-pullback 'raster.dl.array-ops/array-add)
          pullback ((pullback-fn result a b n) (double-array (repeat n 1.0)))
          da (first pullback)]
      (dotimes [i n]
        (let [ng (numerical-grad f-a a i)]
          (is (approx= ng (aget ^doubles da i) 1e-4)
              (str "array-add grad mismatch at " i)))))))

(deftest broadcast-add-gradient-test
  (testing "broadcast-add gradient for h and t"
    (let [batch 2 dim 3
          h (double-array [1 2 3 4 5 6])
          t (double-array [10 20 30])
          n (* batch dim)
          ;; Loss = sum of output
          f-h (fn [^doubles h'] (let [out (ops/broadcast-add h' t batch dim)]
                                  (loop [i 0 s 0.0]
                                    (if (< i n) (recur (inc i) (+ s (aget out i))) s))))
          f-t (fn [^doubles t'] (let [out (ops/broadcast-add h t' batch dim)]
                                  (loop [i 0 s 0.0]
                                    (if (< i n) (recur (inc i) (+ s (aget out i))) s))))
          result (ops/broadcast-add h t batch dim)
          pullback-fn (tmpl/template-pullback 'raster.dl.array-ops/broadcast-add)
          grads ((pullback-fn result h t batch dim) (double-array (repeat n 1.0)))
          dh (nth grads 0)
          dt (nth grads 1)]
      ;; Check h gradients
      (dotimes [i n]
        (is (approx= (numerical-grad f-h h i) (aget ^doubles dh i) 1e-4)
            (str "broadcast-add dh mismatch at " i)))
      ;; Check t gradients
      (dotimes [i dim]
        (is (approx= (numerical-grad f-t t i) (aget ^doubles dt i) 1e-4)
            (str "broadcast-add dt mismatch at " i))))))

;; ================================================================
;; Generic primitive gradient checks
;; ================================================================

(deftest scatter-add-gradient-test
  (testing "scatter-add gradient via finite differences"
    (let [n-pairs 3 n-dst 2 stride 2
          vals (double-array [1 2 3 4 5 6])
          indices (long-array [0 1 0])
          f-vals (fn [^doubles v']
                   (let [out (ops/scatter-add v' indices n-dst n-pairs stride)]
                     (loop [i 0 s 0.0]
                       (if (< i (alength out)) (recur (inc i) (+ s (aget out i))) s))))
          result (ops/scatter-add vals indices n-dst n-pairs stride)
          pullback-fn (tmpl/template-pullback 'raster.dl.array-ops/scatter-add)
          grads ((pullback-fn result vals indices n-dst n-pairs stride)
                 (double-array (repeat (* n-dst stride) 1.0)))
          d-vals (first grads)]
      (dotimes [i (* n-pairs stride)]
        (is (approx= (numerical-grad f-vals vals i) (aget ^doubles d-vals i) 1e-4)
            (str "scatter-add d-vals mismatch at " i))))))

(deftest gather-gradient-test
  (testing "gather gradient via finite differences"
    (let [n-src 3 n-pairs 4 stride 2
          src (double-array [1 2 3 4 5 6])
          indices (long-array [0 2 1 0])
          f-src (fn [^doubles s']
                  (let [out (ops/gather s' indices n-src n-pairs stride)]
                    (loop [i 0 s 0.0]
                      (if (< i (alength out)) (recur (inc i) (+ s (aget out i))) s))))
          result (ops/gather src indices n-src n-pairs stride)
          pullback-fn (tmpl/template-pullback 'raster.dl.array-ops/gather)
          grads ((pullback-fn result src indices n-src n-pairs stride)
                 (double-array (repeat (* n-pairs stride) 1.0)))
          d-src (first grads)]
      (dotimes [i (* n-src stride)]
        (is (approx= (numerical-grad f-src src i) (aget ^doubles d-src i) 1e-4)
            (str "gather d-src mismatch at " i))))))

(deftest indexed-dot-gradient-test
  (testing "indexed-dot gradient via finite differences"
    (let [n-nodes 3 n-edges 2 emb-dim 4 n-heads 2 dk 2
          A (double-array (repeatedly (* n-nodes emb-dim) #(- (rand) 0.5)))
          B (double-array (repeatedly (* n-nodes emb-dim) #(- (rand) 0.5)))
          idx-a (long-array [0 1])
          idx-b (long-array [1 2])
          n-out (* n-edges n-heads)
          f-A (fn [^doubles A']
                (let [out (ops/indexed-dot A' B idx-a idx-b n-nodes n-nodes
                                           n-edges dk emb-dim n-heads)]
                  (loop [i 0 s 0.0]
                    (if (< i n-out) (recur (inc i) (+ s (aget out i))) s))))
          f-B (fn [^doubles B']
                (let [out (ops/indexed-dot A B' idx-a idx-b n-nodes n-nodes
                                           n-edges dk emb-dim n-heads)]
                  (loop [i 0 s 0.0]
                    (if (< i n-out) (recur (inc i) (+ s (aget out i))) s))))
          result (ops/indexed-dot A B idx-a idx-b n-nodes n-nodes
                                  n-edges dk emb-dim n-heads)
          pullback-fn (tmpl/template-pullback 'raster.dl.array-ops/indexed-dot)
          grads ((pullback-fn result A B idx-a idx-b n-nodes n-nodes
                              n-edges dk emb-dim n-heads)
                 (double-array (repeat n-out 1.0)))
          dA (nth grads 0)
          dB (nth grads 1)]
      (dotimes [i (* n-nodes emb-dim)]
        (is (approx= (numerical-grad f-A A i) (aget ^doubles dA i) 1e-4)
            (str "indexed-dot dA mismatch at " i)))
      (dotimes [i (* n-nodes emb-dim)]
        (is (approx= (numerical-grad f-B B i) (aget ^doubles dB i) 1e-4)
            (str "indexed-dot dB mismatch at " i))))))

(deftest scatter-mul-add-gradient-test
  (testing "scatter-mul-add gradient via finite differences"
    (let [n-dst 2 n-src 3 n-pairs 3 n-heads 2 emb-dim 4 dk 2
          coeffs (double-array (repeatedly (* n-pairs n-heads) #(- (rand) 0.5)))
          src (double-array (repeatedly (* n-src emb-dim) #(- (rand) 0.5)))
          dst-indices (long-array [0 0 1])
          src-indices (long-array [0 1 2])
          n-out (* n-dst emb-dim)
          f-coeffs (fn [^doubles c']
                     (let [out (ops/scatter-mul-add c' src dst-indices src-indices
                                                    n-dst n-src n-pairs dk emb-dim n-heads)]
                       (loop [i 0 s 0.0]
                         (if (< i n-out) (recur (inc i) (+ s (aget out i))) s))))
          f-src (fn [^doubles s']
                  (let [out (ops/scatter-mul-add coeffs s' dst-indices src-indices
                                                 n-dst n-src n-pairs dk emb-dim n-heads)]
                    (loop [i 0 s 0.0]
                      (if (< i n-out) (recur (inc i) (+ s (aget out i))) s))))
          result (ops/scatter-mul-add coeffs src dst-indices src-indices
                                      n-dst n-src n-pairs dk emb-dim n-heads)
          pullback-fn (tmpl/template-pullback 'raster.dl.array-ops/scatter-mul-add)
          grads ((pullback-fn result coeffs src dst-indices src-indices
                              n-dst n-src n-pairs dk emb-dim n-heads)
                 (double-array (repeat n-out 1.0)))
          d-coeffs (nth grads 0)
          d-src (nth grads 1)]
      (dotimes [i (* n-pairs n-heads)]
        (is (approx= (numerical-grad f-coeffs coeffs i) (aget ^doubles d-coeffs i) 1e-4)
            (str "scatter-mul-add d-coeffs mismatch at " i)))
      (dotimes [i (* n-src emb-dim)]
        (is (approx= (numerical-grad f-src src i) (aget ^doubles d-src i) 1e-4)
            (str "scatter-mul-add d-src mismatch at " i))))))

(deftest scale-clamp-exp-gradient-test
  (testing "scale-clamp-exp gradient via finite differences"
    (let [n 4
          x (double-array [0.5 -0.3 2.0 -1.0])
          scale 0.7
          bound 5.0
          f-x (fn [^doubles x']
                (let [out (ops/scale-clamp-exp x' scale bound n)]
                  (loop [i 0 s 0.0]
                    (if (< i n) (recur (inc i) (+ s (aget out i))) s))))
          result (ops/scale-clamp-exp x scale bound n)
          pullback-fn (tmpl/template-pullback 'raster.dl.array-ops/scale-clamp-exp)
          grads ((pullback-fn result x scale bound n)
                 (double-array (repeat n 1.0)))
          dx (first grads)]
      (dotimes [i n]
        (is (approx= (numerical-grad f-x x i) (aget ^doubles dx i) 1e-4)
            (str "scale-clamp-exp dx mismatch at " i))))))

(deftest reduce-axis-gradient-test
  (testing "reduce-axis gradient via finite differences"
    (let [n-rows 3 n-cols 2
          src (double-array [1 2 3 4 5 6])
          f-src (fn [^doubles s']
                  (let [out (ops/reduce-axis s' n-rows n-cols)]
                    (loop [i 0 s 0.0]
                      (if (< i n-cols) (recur (inc i) (+ s (aget out i))) s))))
          result (ops/reduce-axis src n-rows n-cols)
          pullback-fn (tmpl/template-pullback 'raster.dl.array-ops/reduce-axis)
          grads ((pullback-fn result src n-rows n-cols)
                 (double-array (repeat n-cols 1.0)))
          d-src (first grads)]
      (dotimes [i (* n-rows n-cols)]
        (is (approx= (numerical-grad f-src src i) (aget ^doubles d-src i) 1e-4)
            (str "reduce-axis d-src mismatch at " i))))))

(deftest segment-div-gradient-test
  (testing "segment-div gradient for wV and Z"
    (let [n-nodes 2 emb-dim 4 n-heads 2
          wV (double-array (repeatedly (* n-nodes emb-dim) #(- (rand) 0.5)))
          Z (double-array (repeatedly (* n-nodes n-heads) #(+ 1.0 (rand))))
          eps 1e-6
          n (* n-nodes emb-dim)
          f-wV (fn [^doubles w']
                 (let [out (ops/segment-div w' Z n-nodes emb-dim n-heads eps)]
                   (loop [i 0 s 0.0]
                     (if (< i n) (recur (inc i) (+ s (aget out i))) s))))
          f-Z (fn [^doubles z']
                (let [out (ops/segment-div wV z' n-nodes emb-dim n-heads eps)]
                  (loop [i 0 s 0.0]
                    (if (< i n) (recur (inc i) (+ s (aget out i))) s))))
          result (ops/segment-div wV Z n-nodes emb-dim n-heads eps)
          pullback-fn (tmpl/template-pullback 'raster.dl.array-ops/segment-div)
          grads ((pullback-fn result wV Z n-nodes emb-dim n-heads eps)
                 (double-array (repeat n 1.0)))
          dwV (nth grads 0)
          dZ (nth grads 1)]
      (dotimes [i n]
        (is (approx= (numerical-grad f-wV wV i) (aget ^doubles dwV i) 1e-4)
            (str "segment-div dwV mismatch at " i)))
      (dotimes [i (* n-nodes n-heads)]
        (is (approx= (numerical-grad f-Z Z i) (aget ^doubles dZ i) 1e-3)
            (str "segment-div dZ mismatch at " i))))))

;; ================================================================
;; Existing gradient checks
;; ================================================================

(deftest dot-rows-gradient-test
  (testing "dot-rows gradient"
    (let [n-rows 2 dim 3
          h (double-array [1 2 3 4 5 6])
          W (double-array [1 0 1])
          bias (double-array [0.5])
          ;; Loss = sum of output
          f-h (fn [^doubles h'] (let [out (ops/dot-rows h' W bias n-rows dim)]
                                  (loop [i 0 s 0.0]
                                    (if (< i n-rows) (recur (inc i) (+ s (aget out i))) s))))
          f-W (fn [^doubles W'] (let [out (ops/dot-rows h W' bias n-rows dim)]
                                  (loop [i 0 s 0.0]
                                    (if (< i n-rows) (recur (inc i) (+ s (aget out i))) s))))
          result (ops/dot-rows h W bias n-rows dim)
          pullback-fn (tmpl/template-pullback 'raster.dl.array-ops/dot-rows)
          grads ((pullback-fn result h W bias n-rows dim) (double-array (repeat n-rows 1.0)))
          dh (nth grads 0)
          dW (nth grads 1)
          dbias (nth grads 2)]
      (dotimes [i (* n-rows dim)]
        (is (approx= (numerical-grad f-h h i) (aget ^doubles dh i) 1e-4)
            (str "dot-rows dh mismatch at " i)))
      (dotimes [i dim]
        (is (approx= (numerical-grad f-W W i) (aget ^doubles dW i) 1e-4)
            (str "dot-rows dW mismatch at " i)))
      ;; bias grad
      (is (approx= 2.0 (aget ^doubles dbias 0) 1e-4)  ;; sum of dy = 2
          "dot-rows dbias"))))

(deftest flat-embed-gradient-test
  (testing "flat-embed-op gradient for values and We"
    (let [n-vars 3 emb-dim 4 n-spaces 2 n-states 4
          values (double-array [1.0 2.0 3.0])
          spaces (long-array [0 1 0])
          states (long-array [0 1 0])
          We (double-array [0.5 0.5 0.5 0.5])
          be (double-array [0.1 0.2 0.3 0.4])
          space-emb (double-array [1 2 3 4 5 6 7 8])
          state-emb (double-array (repeat (* n-states emb-dim) 0.1))
          pos-emb (double-array (repeat (* n-vars emb-dim) 0.05))
          n (* n-vars emb-dim)
          f-values (fn [^doubles v']
                     (let [out (ops/flat-embed-op v' space-emb spaces state-emb states pos-emb
                                                  We be n-vars emb-dim n-spaces n-states)]
                       (loop [i 0 s 0.0]
                         (if (< i n) (recur (inc i) (+ s (aget out i))) s))))
          f-We (fn [^doubles We']
                 (let [out (ops/flat-embed-op values space-emb spaces state-emb states pos-emb
                                              We' be n-vars emb-dim n-spaces n-states)]
                   (loop [i 0 s 0.0]
                     (if (< i n) (recur (inc i) (+ s (aget out i))) s))))
          result (ops/flat-embed-op values space-emb spaces state-emb states pos-emb
                                    We be n-vars emb-dim n-spaces n-states)
          pullback-fn (tmpl/template-pullback 'raster.dl.array-ops/flat-embed-op)
          grads ((pullback-fn result values space-emb spaces state-emb states pos-emb
                              We be n-vars emb-dim n-spaces n-states)
                 (double-array (repeat n 1.0)))
          d-values (nth grads 0)
          d-We (nth grads 6)]
      (dotimes [i n-vars]
        (is (approx= (numerical-grad f-values values i) (aget ^doubles d-values i) 1e-4)
            (str "flat-embed d-values mismatch at " i)))
      (dotimes [i emb-dim]
        (is (approx= (numerical-grad f-We We i) (aget ^doubles d-We i) 1e-4)
            (str "flat-embed dWe mismatch at " i))))))

(deftest masked-mse-gradient-test
  (testing "masked-mse-loss gradient"
    (let [n 4
          pred (double-array [1.0 2.0 3.0 4.0])
          target (double-array [1.5 2.0 2.5 3.5])
          states (long-array [0 1 0 0])
          f-pred (fn [^doubles p'] (double (ops/masked-mse-loss p' target states n)))
          pullback-fn (tmpl/template-pullback 'raster.dl.array-ops/masked-mse-loss)
          grads ((pullback-fn nil pred target states n) 1.0)
          d-pred (first grads)]
      (dotimes [i n]
        (is (approx= (numerical-grad f-pred pred i) (aget ^doubles d-pred i) 1e-4)
            (str "masked-mse d-pred mismatch at " i))))))

(deftest embed-timestep-gradient-test
  (testing "embed-timestep gradient via sinusoidal-embed-backward"
    (let [dim 8
          t 50.0
          eps 1e-5
          ref (double-array 1)
          fwd+ (gsdm/embed-timestep ref (+ t eps) dim)
          fwd- (gsdm/embed-timestep ref (- t eps) dim)
          ;; Loss = sum of embedding
          loss+ (loop [i 0 s 0.0] (if (< i dim) (recur (inc i) (+ s (aget fwd+ i))) s))
          loss- (loop [i 0 s 0.0] (if (< i dim) (recur (inc i) (+ s (aget fwd- i))) s))
          numerical-dt (/ (- loss+ loss-) (* 2.0 eps))
          ;; Analytical via backward
          dy (double-array (repeat dim 1.0))
          analytical-dt (gsdm/sinusoidal-embed-backward dy t dim)]
      (is (approx= numerical-dt analytical-dt 1e-3)
          (str "embed-timestep dt: numerical=" numerical-dt " analytical=" analytical-dt)))))

;; ================================================================
;; Forward equivalence with original GSDM ops
;; ================================================================

(deftest resnet-block-equivalence-test
  (testing "refactored resnet-block produces same output shape"
    (let [n-nodes 5 d 8
          rng (java.util.Random. 42)
          rand-arr (fn [n] (let [a (double-array n)]
                             (dotimes [i n] (aset a i (* 0.1 (.nextGaussian rng))))
                             a))
          ones-arr (fn [n] (let [a (double-array n)] (java.util.Arrays/fill a 1.0) a))
          x (rand-arr (* n-nodes d))
          temb (rand-arr d)
          W1 (rand-arr (* d d)) b1 (double-array d)
          W2 (rand-arr (* d d)) b2 (double-array d)
          Wt (rand-arr (* d d)) bt (double-array d)
          g1 (ones-arr d) bn1 (double-array d)
          g2 (ones-arr d) bn2 (double-array d)
          out (gsdm/resnet-block x temb W1 b1 W2 b2 Wt bt g1 bn1 g2 bn2 n-nodes d)]
      (is (= (* n-nodes d) (alength out)))
      (is (arr-finite? out)))))

(deftest graph-attention-equivalence-test
  (testing "graph-attention with generic primitives produces finite results"
    (let [n-nodes 4 n-edges 6 d 8 n-heads 2
          rng (java.util.Random. 42)
          rand-arr (fn [n] (let [a (double-array n)]
                             (dotimes [i n] (aset a i (* 0.05 (.nextGaussian rng))))
                             a))
          ones-arr (fn [n] (let [a (double-array n)] (java.util.Arrays/fill a 1.0) a))
          h (rand-arr (* n-nodes d))
          src-edges (long-array [0 0 0 1 2 3])
          dst-edges (long-array [1 2 3 0 0 0])
          out (gsdm/graph-attention-multihead h
                                              (nn/xavier-init d d) (rand-arr d)  ;; Wq, bq
                                              (nn/xavier-init d d) (rand-arr d)  ;; Wk, bk
                                              (nn/xavier-init d d) (rand-arr d)  ;; Wv, bv
                                              (ones-arr d) (double-array d)      ;; gamma, beta
                                              src-edges dst-edges
                                              n-nodes n-edges d n-heads)]
      (is (= (* n-nodes d) (alength out)))
      (is (arr-finite? out)))))

(deftest flat-embed-equivalence-test
  (testing "refactored flat-embed via array-ops"
    (let [n-vars 5 d 8 n-spaces 3 n-states 4
          rng (java.util.Random. 42)
          values (double-array [0.1 0.2 0.3 0.4 0.5])
          spaces (long-array [0 1 2 0 1])
          states (long-array [0 1 0 0 1])
          We (double-array (repeatedly d #(.nextGaussian rng)))
          be (double-array d)
          space-emb (double-array (repeatedly (* n-spaces d) #(* 0.1 (.nextGaussian rng))))
          state-emb (double-array (repeatedly (* n-states d) #(* 0.1 (.nextGaussian rng))))
          pos-emb (double-array (repeatedly (* n-vars d) #(* 0.05 (.nextGaussian rng))))
          out (gsdm/flat-embed values space-emb spaces state-emb states pos-emb
                               We be n-vars d n-spaces n-states)]
      (is (= (* n-vars d) (alength out)))
      (is (arr-finite? out)))))

(deftest flat-unembed-equivalence-test
  (testing "refactored flat-unembed via dot-rows"
    (let [n-vars 5 d 8
          rng (java.util.Random. 42)
          emb (double-array (repeatedly (* n-vars d) #(* 0.1 (.nextGaussian rng))))
          Wu1 (nn/xavier-init d d) bu1 (double-array d)
          Wu2 (double-array (repeatedly d #(* 0.1 (.nextGaussian rng))))
          bu2 (double-array [0.0])
          out (gsdm/flat-unembed emb Wu1 bu1 Wu2 bu2 n-vars d)]
      (is (= n-vars (alength out)))
      (is (arr-finite? out)))))
