(ns raster.quant.tile-test
  "The 8-column tile (reading the interleaved repack layout) must produce the same raw
   int32 column dots as the already-validated single-column wi8-dot-q4 (reading the
   row-major weight). This pins the tile's layout access to the layout descriptor —
   the dpbusd C-helper is then a faithful drop-in for this reference."
  (:require [clojure.test :refer [deftest is]]
            [raster.quant.kernels :as qc]
            [raster.compiler.backend.cpu.quant :as quant]))

(defn- rand-bytes [n seed]
  (let [a (byte-array n)] (.nextBytes (java.util.Random. seed) a) a))

(deftest tile-matches-single-column
  (doseq [[out in] [[16 64] [1024 640] [8 32] [24 128]]]
    (let [half (quot in 2)
          nb (quot in 32)
          wq (rand-bytes (quot (* out in) 2) (+ out in))
          ws (let [a (float-array (* out nb))] (dotimes [i (alength a)] (aset a i (float 1.0))) a)
          xq (rand-bytes in (* 7 (+ out in)))
          {:keys [wqi]} (quant/repack-stream wq ws out in)
          ng (quot out 8)
          out8 (int-array 8)]
      (doseq [gi (range ng) b (range nb)]
        ;; woff = group weight base + block: gi*nb*128 + b*128
        (qc/wi8-dot-q4-x8 wqi (+ (* gi nb 128) (* b 128)) xq (* b 32) out8)
        (dotimes [L 8]
          (let [col (+ (* gi 8) L)
                ref (qc/wi8-dot-q4 wq (+ (* col half) (* b 16)) xq (* b 32))]
            (is (= (long (aget out8 L)) ref)
                (str "tile col " col " block " b " [" out "x" in "]"))))))))
