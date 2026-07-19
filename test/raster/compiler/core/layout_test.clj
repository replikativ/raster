(ns raster.compiler.core.layout-test
  "The descriptor-driven Q4 stream layout must reproduce the hand-written
   repack-stream byte-for-byte at NC=8 (the AVX2 kernel's required layout), proving
   layout-as-schedule slots under the existing kernel without drift."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.core.layout :as layout]
            [raster.compiler.backend.cpu.quant :as quant]))

(def avx2 {:vector-bits 256 :has-native-dot-reduce true :num-vector-registers 16
           :llc-bytes 16777216 :balance 40})

(defn- rand-weights [out in seed]
  (let [r (java.util.Random. seed)
        wq (byte-array (quot (* out in) 2))
        ws (float-array (* out (quot in 32)))]
    (.nextBytes r wq)
    (dotimes [i (alength ws)] (aset ws i (float (- (.nextDouble r) 0.5))))
    [wq ws]))

(deftest descriptor-derives-avx2-tile
  (testing "NC for AVX2 = 8, the hand kernel's hard-coded column tile"
    (let [l (layout/quant-stream-layout avx2)]
      (is (= :q4-col-interleaved (:kind l)))
      (is (= 8 (:tile l)))
      (is (= 4 (:half l)))
      (is (= 8 (:igroups l)))            ; block 32 / k-group 4
      (is (= 16 (:bytes-per-igroup l)))))) ; (8/2)*4 = AVX2's 16 bytes/igroup

(deftest byte-identical-to-hand-repack
  (testing "descriptor-driven repack == hand repack-stream byte-for-byte (NC=8)"
    (doseq [[out in] [[1024 640] [640 2048] [16 64] [6912 1152]]]
      (let [[wq ws] (rand-weights out in (+ out in))
            l (layout/quant-stream-layout avx2)
            a (layout/repack l wq ws out in)
            b (quant/repack-stream wq ws out in)]
        (is (java.util.Arrays/equals ^bytes (:wqi a) ^bytes (:wqi b))
            (str "wqi mismatch for [" out " " in "]"))
        (is (java.util.Arrays/equals ^floats (:wsi a) ^floats (:wsi b))
            (str "wsi mismatch for [" out " " in "]"))))))

(deftest generalizes-to-avx512-tile
  (testing "NC=16 descriptor yields a self-consistent larger interleave"
    (let [l (layout/quant-stream-layout {:vector-bits 512})]
      (is (= 16 (:tile l)))
      (is (= 8 (:half l)))
      (is (= 32 (:bytes-per-igroup l)))   ; (16/2)*4
      ;; physical sizes scale with the tile, not the convention
      (is (= (* (quot 1024 16) (quot 640 32) 8 32) (layout/wqi-len l 1024 640))))))

;; ===========================================================================
;; Stage-0 microbench (design §5d) — the cheap abstraction-validating gate for
;; the GENERAL layout facet, run BEFORE any emitter rewrite.
;; ===========================================================================

(deftest row-major-offset-matches-the-natural-expression
  (testing "row-major layout->offset is the byte-identical linear index the emitter already builds"
    (is (= '(+ (* i n) j) (layout/layout->offset (layout/row-major '[m n]) '[i j])))
    (is (= '(+ (+ (* i (* k n)) (* j k)) p)
           (layout/layout->offset (layout/row-major '[m n k]) '[i j p])) "3-D row-major nests"))
  (testing "numeric strides fold"
    (is (= '(+ (* i 128) j) (layout/layout->offset (layout/row-major [64 128]) '[i j])))))

(deftest transpose-is-just-a-perm-flip
  (testing "col-major / transpose swaps the strides — axis0 becomes unit-stride"
    (is (= '(+ i (* j m)) (layout/layout->offset (layout/col-major '[m n] :half) '[i j])))
    (is (= [1 0] (:perm (layout/transpose-layout (layout/row-major '[m n])))))))

(deftest round-trip-identity-dense-layouts
  (testing "layout->offset ∘ offset->coords = identity for every dense strided layout"
    (doseq [lay [(layout/row-major [4 6]) (layout/col-major [4 6] :float)
                 (layout/transpose-layout (layout/row-major [8 3])) (layout/row-major [2 3 5])]]
      (let [{:keys [shape]} lay
            three? (= 3 (count shape))]
        (doseq [coords (for [i (range (nth shape 0)) j (range (nth shape 1))
                             p (range (if three? (nth shape 2) 1))]
                         (if three? [i j p] [i j]))]
          (is (= coords (layout/offset->coords lay (layout/layout->offset lay coords)))
              (str "round-trip failed for " (:kind lay) " " coords)))))))

(deftest offsets-are-a-bijection
  (testing "a dense layout covers [0, prod(shape)) exactly once — no gaps, no collisions"
    (doseq [lay [(layout/row-major [4 6]) (layout/col-major [4 6] :float)
                 (layout/transpose-layout (layout/row-major [5 7]))]]
      (let [[a b] (:shape lay)
            offs (for [i (range a) j (range b)] (layout/layout->offset lay [i j]))]
        (is (= (set (range (* a b))) (set offs)))))))

(deftest cost-model-trichotomy-sanity
  (testing "identity convert is FREE (register shuffle)"
    (let [rm (layout/row-major [64 128] :half)]
      (is (= :register-shuffle (:class (layout/convert-cost rm rm {}))))
      (is (= 0 (:bytes (layout/convert-cost rm rm {}))))))
  (testing "a transpose across the coalescing axis needs a shared-memory round-trip, priced in bytes"
    (let [rm (layout/row-major [64 128] :half) tr (layout/transpose-layout rm)]
      (is (= :shared-roundtrip (:class (layout/convert-cost rm tr {}))))
      (is (= (* 64 128 2) (:bytes (layout/convert-cost rm tr {})))))))

(deftest derive-layout-picks-the-right-kind
  (let [desc {:matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16} :subgroup-size 16}]
    (is (= :mma-frag    (:kind (layout/derive-layout :mma-acc :half desc))))
    (is (= :dot-operand (:kind (layout/derive-layout :dot-operand :half desc :op-idx 1))))
    (is (= 1 (:op-idx (layout/derive-layout :dot-operand :half desc :op-idx 1))))
    (is (= 2 (:k-width (layout/derive-layout :dot-operand :half desc))) "kWidth = 32/16 = 2 for f16")
    (is (= :blocked   (:kind (layout/derive-layout :coalesced-load :half desc :shape [64 128]))))
    (is (= :row-major (:kind (layout/derive-layout :elementwise :float desc :shape [64 128]))))
    (is (= [1 0]      (:perm (layout/derive-layout :transpose :float desc :shape [64 128]))))))

(deftest quant-layout-lifts-into-the-general-facet
  (testing "the CPU quant layout becomes one datatype with the general facet"
    (let [q (layout/quant-stream-layout {:vector-bits 256})
          f (layout/quant-layout->facet q)]
      (is (= :quant (:general f)))
      (is (= (:tile q) (:tile f)) "preserves the quant fields")
      (is (layout/row-major-unit? (layout/row-major '[m n])) "row-major default is recognized trivial")
      (is (not (layout/row-major-unit? (layout/col-major '[m n] :float)))))))
