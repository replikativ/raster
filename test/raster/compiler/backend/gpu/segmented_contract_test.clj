(ns raster.compiler.backend.gpu.segmented-contract-test
  "W1 step 2b: naive segmented-reduction emitter (contraction → OpenCL), device-free
   string-structure checks. Composes contract-lower → generate-segmented-reduce-kernel."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [raster.compiler.passes.parallel.contract-lower :as cl]
            [raster.compiler.backend.gpu.segop-opencl :as sco]))

(defn- emit [form]
  (let [sr (cl/contract-form->segred form)]
    (:source (sco/generate-segmented-reduce-kernel sr 'C))))

(deftest segmented-matmul-nn-structure
  (testing "matmul :nn emits one-thread-per-segment + k-loop + combine + store"
    (let [src (emit '(raster.par/contract C [[i m] [j n]] [[l k]]
                       (* (aget A (+ (* i k) l)) (aget B (+ (* l n) j)))))]
      (testing "segment id + trailing count guard (num-segments passed as _nseg)"
        (is (str/includes? src "int seg = get_global_id(0)"))
        (is (str/includes? src "seg >= _nseg"))
        (is (str/includes? src "int _nseg")))
      (testing "row-major decompose of the segment id into free indices i,j"
        ;; i = (seg / n) % m ; j = seg % n
        (is (re-find #"int i = \(seg / \(n\)\) % m;" src))
        (is (re-find #"int j = seg % n;" src)))
      (testing "sequential fold over the reduced (contracted) axis l < k"
        (is (str/includes? src "for (int idx = 0; idx < k; idx++)")))
      (testing "combine accumulates the product; result stored per segment"
        (is (str/includes? src "acc = (acc + "))
        (is (str/includes? src "out[seg] = acc"))
        (is (str/includes? src "A[")) (is (str/includes? src "B[")))
      (testing "signature: const input arrays, output, int dim params"
        (is (str/includes? src "__global const double* restrict A"))
        (is (str/includes? src "__global const double* restrict B"))
        (is (str/includes? src "__global double* restrict out"))
        (is (str/includes? src "int k"))))))

(deftest segmented-matvec-single-segment-dim
  (testing "1 free axis (matvec): decompose is just seg % m, num-segments = m"
    (let [src (emit '(raster.par/contract y [[i m]] [[l k]]
                       (* (aget A (+ (* i k) l)) (aget x l))))]
      (is (str/includes? src "seg >= _nseg"))
      (is (re-find #"int i = seg % m;" src))
      (is (str/includes? src "for (int idx = 0; idx < k; idx++)")))))

(deftest segmented-float-dtype
  (testing "dtype :float threads through param + acc types"
    (let [sr  (cl/contract-form->segred
               '(raster.par/contract C [[i m] [j n]] [[l k]]
                  (* (aget A (+ (* i k) l)) (aget B (+ (* l n) j))))
               :dtype :float)
          src (:source (sco/generate-segmented-reduce-kernel sr 'C :dtype :float))]
      (is (str/includes? src "__global const float* restrict A"))
      (is (str/includes? src "float acc = ")))))
