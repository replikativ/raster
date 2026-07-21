(ns raster.dl.einsum-contract-test
  "B1: the general einsum surface lowers to the SOAC contraction path.
   einsum->contract-form parses a subscript into a par/contract form (free = output labels,
   contract = summed labels); routing it sends it to the peak leaves / portable fallback. This
   is the bridge that unifies einsum with the peak GEMM path — no interpreted loop."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.dl.einsum :as es]
            [raster.compiler.passes.parallel.contract-route :as route]))

(def ^:private gpu?
  (delay (try (require 'raster.gpu.ze-runtime)
              (boolean (seq ((resolve 'raster.gpu.ze-runtime/query-devices))))
              (catch Throwable _ false))))

(deftest einsum-lowers-to-contract-form
  (testing "matmul ij,jk->ik → canonical contract form"
    (is (= '(raster.par/contract C [[i 3] [k 2]] [[j 4]]
              (* (aget A (+ (* i 4) j)) (aget B (+ (* j 2) k))))
           (es/einsum->contract-form "ij,jk->ik" {:i 3 :j 4 :k 2} 'C '[A B]))))
  (testing "transposed output ij,jk->ki → free axes in output order [k i]"
    (is (= '[[k 2] [i 3]] (nth (es/einsum->contract-form "ij,jk->ki" {:i 3 :j 4 :k 2} 'C '[A B]) 2))))
  (testing "batch bij,bjk->bik → 3 free axes + 1 contract"
    (let [form (es/einsum->contract-form "bij,bjk->bik" {:b 2 :i 3 :j 4 :k 2} 'C '[A B])]
      (is (= 3 (count (nth form 2)))) (is (= 1 (count (nth form 3)))))))

(defn- go [subscript dim-map bufs out-size]
  (let [ze (find-ns 'raster.gpu.ze-runtime)
        r (route/route-contraction (es/einsum->contract-form subscript dim-map 'C '[A B]) :dtype :double)
        o ((ns-resolve ze 'make-buffer) out-size :double)]
    ((ns-resolve ze 'register-kernel!) (:kernel-name r) {:source (:source r) :dtype :double})
    (let [{:keys [kernel-handle]} ((ns-resolve ze 'ensure-kernel-loaded!) (:kernel-name r))
          args (into (mapv #(:segment (get bufs %)) (:array-params r))
                     (into [(:segment o)] (:scalar-args r)))]
      ((ns-resolve ze 'launch-2d!) kernel-handle (:wg r) (:grid r) args)
      [(:strategy r) (vec ((ns-resolve ze 'buffer->double-array) o))])))

(deftest einsum-contract-matches-reference-on-device
  (if-not @gpu?
    (println "[skip] einsum-contract-device: no GPU")
    (let [ze (find-ns 'raster.gpu.ze-runtime)
          mk (fn [xs] ((ns-resolve ze 'array->buffer!) ((ns-resolve ze 'make-buffer) (count xs) :double)
                       (double-array (map double xs))))
          Ad (double-array (map #(* 0.1 (double %)) (range 12)))    ; 3×4
          Bd (double-array (map #(* 0.2 (double %)) (range 8)))     ; 4×2
          bufs {'A (mk (vec Ad)) 'B (mk (vec Bd))}
          close? (fn [xs ys] (every? true? (map #(< (Math/abs (- (double %1) (double %2))) 1.0e-9) xs ys)))]
      (testing "ij,jk->ik matmul"
        (let [[s r] (go "ij,jk->ik" {:i 3 :j 4 :k 2} bufs 6)
              ref (for [i (range 3) k (range 2)] (reduce + (for [j (range 4)] (* (aget Ad (+ (* i 4) j)) (aget Bd (+ (* j 2) k))))))]
          (is (close? r ref) (str "strategy " s))))
      (testing "ij,jk->ki transposed output"
        (let [[_ r] (go "ij,jk->ki" {:i 3 :j 4 :k 2} bufs 6)
              ref (for [k (range 2) i (range 3)] (reduce + (for [j (range 4)] (* (aget Ad (+ (* i 4) j)) (aget Bd (+ (* j 2) k))))))]
          (is (close? r ref))))
      (testing "bij,bjk->bik batch matmul → naive segmented reduce"
        (let [B 2 M 3 K 4 N 2
              Abd (double-array (map #(* 0.1 (double %)) (range (* B M K))))
              Bbd (double-array (map #(* 0.2 (double %)) (range (* B K N))))
              bufs2 {'A (mk (vec Abd)) 'B (mk (vec Bbd))}
              [s r] (go "bij,bjk->bik" {:b B :i M :j K :k N} bufs2 (* B M N))
              ref (for [b (range B) i (range M) k (range N)]
                    (reduce + (for [j (range K)] (* (aget Abd (+ (* (+ (* b M) i) K) j))
                                                    (aget Bbd (+ (* (+ (* b K) j) N) k))))))]
          (is (= :naive-segred s))
          (is (close? r ref)))))))
