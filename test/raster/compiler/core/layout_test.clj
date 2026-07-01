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
