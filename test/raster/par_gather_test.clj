(ns raster.par-gather-test
  "Tests for the par/gather combinator: scalar expansion + SIMD vgather emit.
   gather: out[i] = src[index[i]]."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.par :as par]
            [raster.core :refer [deftm]]
            [raster.compiler.ir.par :as ir-par]
            [raster.compiler.backend.jvm.par-simd :as psimd]))

(defn- reference-gather [^doubles src ^ints idx n]
  (let [out (double-array n)]
    (dotimes [i n] (aset out i (aget src (aget idx i))))
    out))

;; ---- scalar (macro) path ----------------------------------------------------

(deftest scalar-gather-flat
  (testing "(par/gather out src index n) = out[i]=src[index[i]]"
    (let [src (double-array [10.0 20.0 30.0 40.0 50.0])
          idx (int-array [4 0 2 0 3 1])
          out (double-array 6)]
      (par/gather out src idx 6)
      (is (= [50.0 10.0 30.0 10.0 40.0 20.0] (vec out))))))

(deftest scalar-gather-strided
  (testing "strided gather copies stride-wide rows"
    (let [src (double-array [1.0 2.0 3.0 4.0 5.0 6.0])  ; 3 rows x 2
          idx (int-array [2 0])
          out (double-array 4)]
      (par/gather out src idx 2 2)
      (is (= [5.0 6.0 1.0 2.0] (vec out))))))

;; ---- SIMD emit --------------------------------------------------------------

(deftest simd-pass-emits-vgather
  (testing "simd-pass lowers flat par/gather to a hardware vgather (fromArray indexMap + intoArray)"
    (let [simd (:form (psimd/simd-pass '(raster.par/gather out src idx n)))
          s    (pr-str simd)]
      (is (clojure.string/includes? s "fromArray"))
      (is (clojure.string/includes? s "intoArray")))))

;; ---- end-to-end through the JIT (walker → simd-pass → bytecode) --------------

(deftm gather-kernel [out :- (Array double), src :- (Array double),
                      idx :- (Array int), n :- Long] :- (Array double)
  (raster.par/gather out src idx n))

(deftest simd-gather-end-to-end
  (testing "JIT-compiled par/gather with SIMD matches the reference, incl. scalar tail"
    (doseq [n [40 37 8 7 1 0]]   ; multiples of lanes and not, plus tiny/empty
      (let [src (double-array (map #(* 1.5 (inc %)) (range (max 1 n))))
            idx (int-array (shuffle (range (max 1 n))))
            out (double-array (max 1 n))
            expect (reference-gather src idx n)]
        (binding [raster.core/*jit-simd?* true]
          (gather-kernel out src idx n))
        (is (java.util.Arrays/equals ^doubles (double-array (take n out))
                                     ^doubles expect)
            (str "n=" n))))))
