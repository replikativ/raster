(ns raster.dl.einsum-test
  (:require [clojure.test :refer [deftest testing is]]
            [raster.dl.einsum :as ein]
            [raster.dl.tensor :as t]
            [raster.ad.templates :as tmpl]))

(defn- approx=
  ([a b] (approx= a b 1e-6))
  ([a b eps] (< (Math/abs (- (double a) (double b))) (double eps))))

(defn- tensor-approx=
  "Check if two tensors are element-wise approximately equal."
  [a b & [eps]]
  (let [eps (or eps 1e-6)
        n (alength ^doubles a)]
    (and (= n (alength ^doubles b))
         (every? identity
                 (map (fn [i] (approx= (aget ^doubles a i) (aget ^doubles b i) eps))
                      (range n))))))

;; ================================================================
;; einsum tests
;; ================================================================

(deftest einsum-matmul-test
  (testing "matrix multiplication via einsum"
    ;; A: [2,3], B: [3,2]
    (let [A (t/tensor [1 2 3 4 5 6] [2 3])
          B (t/tensor [7 8 9 10 11 12] [3 2])
          C (ein/einsum "ij,jk->ik" A B)]
      ;; C = [[1*7+2*9+3*11, 1*8+2*10+3*12],
      ;;      [4*7+5*9+6*11, 4*8+5*10+6*12]]
      ;; = [[58, 64], [139, 154]]
      (is (= [2 2] (t/tshape C)))
      (is (approx= 58.0 (aget ^doubles C 0)))
      (is (approx= 64.0 (aget ^doubles C 1)))
      (is (approx= 139.0 (aget ^doubles C 2)))
      (is (approx= 154.0 (aget ^doubles C 3))))))

(deftest einsum-transpose-test
  (testing "transpose via einsum"
    (let [A (t/tensor [1 2 3 4 5 6] [2 3])
          At (ein/einsum "ij->ji" A)]
      (is (= [3 2] (t/tshape At)))
      ;; [[1,4],[2,5],[3,6]]
      (is (approx= 1.0 (aget ^doubles At 0)))
      (is (approx= 4.0 (aget ^doubles At 1)))
      (is (approx= 2.0 (aget ^doubles At 2)))
      (is (approx= 5.0 (aget ^doubles At 3)))
      (is (approx= 3.0 (aget ^doubles At 4)))
      (is (approx= 6.0 (aget ^doubles At 5))))))

(deftest einsum-trace-test
  (testing "trace via einsum"
    ;; A = [[1,2],[3,4]], trace = 1+4 = 5
    (let [A (t/tensor [1 2 3 4] [2 2])
          tr (ein/einsum "ii->" A)]
      (is (= [] (t/tshape tr)))
      (is (approx= 5.0 (aget ^doubles tr 0))))))

(deftest einsum-outer-product-test
  (testing "outer product via einsum"
    (let [a (t/tensor [1 2 3] [3])
          b (t/tensor [4 5] [2])
          C (ein/einsum "i,j->ij" a b)]
      ;; [[4,5],[8,10],[12,15]]
      (is (= [3 2] (t/tshape C)))
      (is (approx= 4.0 (aget ^doubles C 0)))
      (is (approx= 5.0 (aget ^doubles C 1)))
      (is (approx= 8.0 (aget ^doubles C 2)))
      (is (approx= 10.0 (aget ^doubles C 3)))
      (is (approx= 12.0 (aget ^doubles C 4)))
      (is (approx= 15.0 (aget ^doubles C 5))))))

(deftest einsum-dot-product-test
  (testing "dot product via einsum"
    (let [a (t/tensor [1 2 3] [3])
          b (t/tensor [4 5 6] [3])
          c (ein/einsum "i,i->" a b)]
      ;; 1*4 + 2*5 + 3*6 = 32
      (is (= [] (t/tshape c)))
      (is (approx= 32.0 (aget ^doubles c 0))))))

(deftest einsum-batch-matmul-test
  (testing "batch matrix multiplication via einsum"
    ;; A: [2, 2, 3], B: [2, 3, 2]
    ;; batch 0: [[1,2,3],[4,5,6]] @ [[1,2],[3,4],[5,6]] = [[22,28],[49,64]]
    ;; batch 1: [[7,8,9],[10,11,12]] @ [[7,8],[9,10],[11,12]] = [[226,256],[313,356]]
    (let [A (t/tensor [1 2 3 4 5 6 7 8 9 10 11 12] [2 2 3])
          B (t/tensor [1 2 3 4 5 6 7 8 9 10 11 12] [2 3 2])
          C (ein/einsum "bij,bjk->bik" A B)]
      (is (= [2 2 2] (t/tshape C)))
      ;; batch 0
      (is (approx= 22.0 (aget ^doubles C 0)))
      (is (approx= 28.0 (aget ^doubles C 1)))
      (is (approx= 49.0 (aget ^doubles C 2)))
      (is (approx= 64.0 (aget ^doubles C 3)))
      ;; batch 1: [[7,8,9],[10,11,12]] @ [[7,8],[9,10],[11,12]]
      ;; = [[49+72+99, 56+80+108], [70+99+132, 80+110+144]]
      (is (approx= 220.0 (aget ^doubles C 4)))
      (is (approx= 244.0 (aget ^doubles C 5)))
      (is (approx= 301.0 (aget ^doubles C 6)))
      (is (approx= 334.0 (aget ^doubles C 7))))))

(deftest einsum-sum-axis-test
  (testing "sum over one axis via einsum"
    ;; A: [2,3], sum over j → [2]
    (let [A (t/tensor [1 2 3 4 5 6] [2 3])
          s (ein/einsum "ij->i" A)]
      (is (= [2] (t/tshape s)))
      (is (approx= 6.0 (aget ^doubles s 0)))   ;; 1+2+3
      (is (approx= 15.0 (aget ^doubles s 1))))) ;; 4+5+6

  (testing "sum over all axes"
    (let [A (t/tensor [1 2 3 4 5 6] [2 3])
          s (ein/einsum "ij->" A)]
      (is (= [] (t/tshape s)))
      (is (approx= 21.0 (aget ^doubles s 0))))))

(deftest einsum-diag-test
  (testing "extract diagonal via einsum"
    ;; A = [[1,2,3],[4,5,6],[7,8,9]], diag → [1,5,9]
    (let [A (t/tensor [1 2 3 4 5 6 7 8 9] [3 3])
          d (ein/einsum "ii->i" A)]
      (is (= [3] (t/tshape d)))
      (is (approx= 1.0 (aget ^doubles d 0)))
      (is (approx= 5.0 (aget ^doubles d 1)))
      (is (approx= 9.0 (aget ^doubles d 2))))))

(deftest einsum-element-wise-test
  (testing "element-wise (Hadamard) product via einsum"
    (let [a (t/tensor [1 2 3 4] [2 2])
          b (t/tensor [5 6 7 8] [2 2])
          c (ein/einsum "ij,ij->ij" a b)]
      (is (= [2 2] (t/tshape c)))
      (is (approx= 5.0 (aget ^doubles c 0)))
      (is (approx= 12.0 (aget ^doubles c 1)))
      (is (approx= 21.0 (aget ^doubles c 2)))
      (is (approx= 32.0 (aget ^doubles c 3))))))

;; ================================================================
;; rearrange tests
;; ================================================================

(deftest rearrange-transpose-test
  (testing "simple transpose via rearrange"
    ;; [2,3] -> [3,2]
    (let [A (t/tensor [1 2 3 4 5 6] [2 3])
          At (ein/rearrange A "i j -> j i")]
      (is (= [3 2] (t/tshape At)))
      ;; [[1,4],[2,5],[3,6]]
      (is (approx= 1.0 (aget ^doubles At 0)))
      (is (approx= 4.0 (aget ^doubles At 1)))
      (is (approx= 2.0 (aget ^doubles At 2)))
      (is (approx= 5.0 (aget ^doubles At 3)))
      (is (approx= 3.0 (aget ^doubles At 4)))
      (is (approx= 6.0 (aget ^doubles At 5))))))

(deftest rearrange-merge-test
  (testing "merge dimensions via rearrange"
    ;; [2, 3, 4] -> [2, 12]
    (let [A (t/tensor (range 24) [2 3 4])
          B (ein/rearrange A "b c w -> b (c w)")]
      (is (= [2 12] (t/tshape B)))
      ;; Since c,w are already contiguous and in order, values should be unchanged
      (dotimes [i 24]
        (is (approx= (double i) (aget ^doubles B i)))))))

(deftest rearrange-split-test
  (testing "split dimension via rearrange"
    ;; [2, 12] -> [2, 3, 4]
    (let [A (t/tensor (range 24) [2 12])
          B (ein/rearrange A "b (c w) -> b c w" {:c 3})]
      (is (= [2 3 4] (t/tshape B)))
      ;; Values should be unchanged since we're just splitting a contiguous dim
      (dotimes [i 24]
        (is (approx= (double i) (aget ^doubles B i)))))))

(deftest rearrange-4d-transpose-test
  (testing "NCHW -> NHWC transpose via rearrange"
    ;; [1, 2, 3, 4] -> [1, 3, 4, 2]
    (let [A (t/tensor (range 24) [1 2 3 4])
          B (ein/rearrange A "b c h w -> b h w c")]
      (is (= [1 3 4 2] (t/tshape B)))
      ;; Check a few elements
      ;; A[0,0,0,0] = 0 -> B[0,0,0,0] = 0
      (is (approx= 0.0 (aget ^doubles B (t/tidx [1 3 4 2] 0 0 0 0))))
      ;; A[0,1,0,0] = 12 -> B[0,0,0,1] = 12
      (is (approx= 12.0 (aget ^doubles B (t/tidx [1 3 4 2] 0 0 0 1))))
      ;; A[0,0,1,2] = 6 -> B[0,1,2,0] = 6
      (is (approx= 6.0 (aget ^doubles B (t/tidx [1 3 4 2] 0 1 2 0))))
      ;; A[0,1,2,3] = 23 -> B[0,2,3,1] = 23
      (is (approx= 23.0 (aget ^doubles B (t/tidx [1 3 4 2] 0 2 3 1)))))))

(deftest rearrange-merge-and-reorder-test
  (testing "merge + reorder via rearrange"
    ;; [2, 3, 4] -> [4, 6]  (reorder: w first, then merge b,c)
    (let [A (t/tensor (range 24) [2 3 4])
          B (ein/rearrange A "b c w -> w (b c)")]
      (is (= [4 6] (t/tshape B)))
      ;; B[w, b*3+c] = A[b, c, w]
      ;; B[0, 0] = A[0,0,0] = 0
      (is (approx= 0.0 (aget ^doubles B (t/tidx [4 6] 0 0))))
      ;; B[1, 0] = A[0,0,1] = 1
      (is (approx= 1.0 (aget ^doubles B (t/tidx [4 6] 1 0))))
      ;; B[0, 1] = A[0,1,0] = 4
      (is (approx= 4.0 (aget ^doubles B (t/tidx [4 6] 0 1))))
      ;; B[0, 3] = A[1,0,0] = 12
      (is (approx= 12.0 (aget ^doubles B (t/tidx [4 6] 0 3)))))))

(deftest rearrange-split-and-reorder-test
  (testing "split + reorder via rearrange"
    ;; [4, 6] -> [2, 3, 4]  (split second dim into b,c; reorder to b,c,w)
    (let [;; Build A in w-major order: A[w, b*3+c] layout
          A (t/tensor [0 4 8 12 16 20    ;; w=0
                       1 5 9 13 17 21    ;; w=1
                       2 6 10 14 18 22   ;; w=2
                       3 7 11 15 19 23]  ;; w=3
                      [4 6])
          B (ein/rearrange A "w (b c) -> b c w" {:b 2})]
      (is (= [2 3 4] (t/tshape B)))
      ;; B[b,c,w] = A[w, b*3+c]
      ;; B[0,0,0] = A[0,0] = 0
      (is (approx= 0.0 (aget ^doubles B (t/tidx [2 3 4] 0 0 0))))
      ;; B[0,0,1] = A[1,0] = 1
      (is (approx= 1.0 (aget ^doubles B (t/tidx [2 3 4] 0 0 1))))
      ;; B[0,1,0] = A[0,1] = 4
      (is (approx= 4.0 (aget ^doubles B (t/tidx [2 3 4] 0 1 0))))
      ;; B[1,0,0] = A[0,3] = 12
      (is (approx= 12.0 (aget ^doubles B (t/tidx [2 3 4] 1 0 0)))))))

(deftest rearrange-identity-test
  (testing "identity rearrange"
    (let [A (t/tensor (range 12) [3 4])
          B (ein/rearrange A "i j -> i j")]
      (is (= [3 4] (t/tshape B)))
      (dotimes [i 12]
        (is (approx= (double i) (aget ^doubles B i)))))))

(deftest rearrange-flatten-test
  (testing "flatten via rearrange"
    (let [A (t/tensor (range 24) [2 3 4])
          B (ein/rearrange A "a b c -> (a b c)")]
      (is (= [24] (t/tshape B)))
      (dotimes [i 24]
        (is (approx= (double i) (aget ^doubles B i)))))))

(deftest rearrange-patchify-test
  (testing "patchify-like rearrange (split spatial into patches)"
    ;; Simulate extracting 2x2 patches from a 4x4 image
    ;; [1, 4, 4] -> [1, 2, 2, 2, 2]  (b, h//p1, w//p2, p1, p2)
    (let [img (t/tensor (range 16) [1 4 4])
          ;; Split h into (h1 p1), w into (w1 p2) where p1=p2=2
          patches (ein/rearrange img "b (h1 p1) (w1 p2) -> b h1 w1 p1 p2"
                                 {:p1 2 :p2 2})]
      (is (= [1 2 2 2 2] (t/tshape patches)))
      ;; Patch [0,0] = top-left 2x2: [0,1,4,5]
      (is (approx= 0.0 (aget ^doubles patches (t/tidx [1 2 2 2 2] 0 0 0 0 0))))
      (is (approx= 1.0 (aget ^doubles patches (t/tidx [1 2 2 2 2] 0 0 0 0 1))))
      (is (approx= 4.0 (aget ^doubles patches (t/tidx [1 2 2 2 2] 0 0 0 1 0))))
      (is (approx= 5.0 (aget ^doubles patches (t/tidx [1 2 2 2 2] 0 0 0 1 1))))
      ;; Patch [0,1] = top-right 2x2: [2,3,6,7]
      (is (approx= 2.0 (aget ^doubles patches (t/tidx [1 2 2 2 2] 0 0 1 0 0))))
      (is (approx= 3.0 (aget ^doubles patches (t/tidx [1 2 2 2 2] 0 0 1 0 1))))
      ;; Patch [1,0] = bottom-left 2x2: [8,9,12,13]
      (is (approx= 8.0 (aget ^doubles patches (t/tidx [1 2 2 2 2] 0 1 0 0 0))))
      (is (approx= 9.0 (aget ^doubles patches (t/tidx [1 2 2 2 2] 0 1 0 0 1)))))))

;; ================================================================
;; AD rrule tests
;; ================================================================

(defn- numerical-grad-array
  "Finite difference gradient of scalar function f w.r.t. array a."
  [f ^doubles a eps]
  (let [n (alength a)
        grad (double-array n)
        f0 (double (f))]
    (dotimes [i n]
      (let [orig (aget a i)]
        (aset a i (+ orig eps))
        (let [f+ (double (f))]
          (aset a i orig)
          (aset grad i (/ (- f+ f0) eps)))))
    grad))

(defn- arr-approx=
  ([a b] (arr-approx= a b 1e-4))
  ([^doubles a ^doubles b eps]
   (and (= (alength a) (alength b))
        (every? true?
                (for [i (range (alength a))]
                  (< (Math/abs (- (aget a i) (aget b i))) (double eps)))))))

(defn- sum-array ^double [^doubles a]
  (loop [i 0 s 0.0]
    (if (< i (alength a))
      (recur (inc i) (+ s (aget a i)))
      s)))

(deftest einsum-matmul-gradient-test
  (testing "einsum matmul gradients vs finite diff"
    (let [A (t/tensor [0.1 0.2 0.3 0.4 0.5 0.6] [2 3])
          B (t/tensor [0.7 0.8 0.9 1.0 1.1 1.2] [3 2])
          rrfn (tmpl/get-pullback-factory 'raster.dl.einsum/einsum)
          C (ein/einsum "ij,jk->ik" A B)
          pb (rrfn C "ij,jk->ik" A B)
          ;; dy = ones (loss = sum of output)
          dy (t/tensor (double-array (repeat 4 1.0)) [2 2])
          [_ dA dB] (pb dy)
          ;; Numerical gradients
          num-dA (numerical-grad-array
                  #(sum-array (ein/einsum "ij,jk->ik" A B)) A 1e-5)
          num-dB (numerical-grad-array
                  #(sum-array (ein/einsum "ij,jk->ik" A B)) B 1e-5)]
      (is (arr-approx= ^doubles dA num-dA))
      (is (arr-approx= ^doubles dB num-dB)))))

(deftest einsum-outer-product-gradient-test
  (testing "einsum outer product gradients vs finite diff"
    (let [a (t/tensor [1.0 2.0 3.0] [3])
          b (t/tensor [4.0 5.0] [2])
          rrfn (tmpl/get-pullback-factory 'raster.dl.einsum/einsum)
          C (ein/einsum "i,j->ij" a b)
          pb (rrfn C "i,j->ij" a b)
          dy (t/tensor (double-array (repeat 6 1.0)) [3 2])
          [_ da db] (pb dy)
          num-da (numerical-grad-array
                  #(sum-array (ein/einsum "i,j->ij" a b)) a 1e-5)
          num-db (numerical-grad-array
                  #(sum-array (ein/einsum "i,j->ij" a b)) b 1e-5)]
      (is (arr-approx= ^doubles da num-da))
      (is (arr-approx= ^doubles db num-db)))))

(deftest einsum-trace-gradient-test
  (testing "einsum trace gradient vs finite diff"
    (let [A (t/tensor [1.0 2.0 3.0 4.0] [2 2])
          rrfn (tmpl/get-pullback-factory 'raster.dl.einsum/einsum)
          tr (ein/einsum "ii->" A)
          pb (rrfn tr "ii->" A)
          dy (t/tensor [1.0] [])
          [_ dA] (pb dy)
          num-dA (numerical-grad-array
                  #(aget ^doubles (ein/einsum "ii->" A) 0) A 1e-5)]
      (is (arr-approx= ^doubles dA num-dA)))))

(deftest einsum-single-input-gradient-test
  (testing "einsum transpose gradient (single input)"
    (let [A (t/tensor [1.0 2.0 3.0 4.0 5.0 6.0] [2 3])
          rrfn (tmpl/get-pullback-factory 'raster.dl.einsum/einsum)
          At (ein/einsum "ij->ji" A)
          pb (rrfn At "ij->ji" A)
          dy (t/tensor (double-array (repeat 6 1.0)) [3 2])
          [_ dA] (pb dy)
          num-dA (numerical-grad-array
                  #(sum-array (ein/einsum "ij->ji" A)) A 1e-5)]
      (is (arr-approx= ^doubles dA num-dA)))))

(deftest rearrange-transpose-gradient-test
  (testing "rearrange transpose gradient"
    (let [A (t/tensor [1.0 2.0 3.0 4.0 5.0 6.0] [2 3])
          rrfn (tmpl/get-pullback-factory 'raster.dl.einsum/rearrange)
          At (ein/rearrange A "i j -> j i")
          pb (rrfn At A "i j -> j i")
          dy (t/tensor (double-array (repeat 6 1.0)) [3 2])
          [dA _ _] (pb dy)
          num-dA (numerical-grad-array
                  #(sum-array (ein/rearrange A "i j -> j i")) A 1e-5)]
      (is (arr-approx= ^doubles dA num-dA)))))

(deftest rearrange-merge-gradient-test
  (testing "rearrange merge gradient"
    (let [A (t/tensor (double-array (map double (range 24))) [2 3 4])
          rrfn (tmpl/get-pullback-factory 'raster.dl.einsum/rearrange)
          B (ein/rearrange A "b c w -> b (c w)")
          pb (rrfn B A "b c w -> b (c w)")
          dy (t/tensor (double-array (repeat 24 1.0)) [2 12])
          [dA _ _] (pb dy)
          num-dA (numerical-grad-array
                  #(sum-array (ein/rearrange A "b c w -> b (c w)")) A 1e-5)]
      (is (arr-approx= ^doubles dA num-dA)))))

(deftest rearrange-4d-transpose-gradient-test
  (testing "rearrange NCHW->NHWC gradient"
    (let [A (t/tensor (double-array (map double (range 24))) [1 2 3 4])
          rrfn (tmpl/get-pullback-factory 'raster.dl.einsum/rearrange)
          B (ein/rearrange A "b c h w -> b h w c")
          pb (rrfn B A "b c h w -> b h w c")
          dy (t/tensor (double-array (repeat 24 1.0)) [1 3 4 2])
          [dA _ _] (pb dy)
          num-dA (numerical-grad-array
                  #(sum-array (ein/rearrange A "b c h w -> b h w c")) A 1e-5)]
      (is (arr-approx= ^doubles dA num-dA)))))
