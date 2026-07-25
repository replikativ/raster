(ns raster.gpu.byte-transpose-test
  "B3-insert prerequisite: the int8 (:byte) 2D transpose — the operand-prep primitive that lets
   an :nn int8 operand reach the dp4a peak leaf (transpose B[K,N]→[N,K] at BYTE granularity, then
   reinterpret as packed int). Transposing at int32 granularity would scramble dp4a's K-packing,
   so the transpose must be byte-typed. Gated on a real GPU."
  (:require [clojure.test :refer [deftest is testing]]))

(def ^:private gpu?
  (delay (try (require 'raster.gpu.ze-runtime)
              (boolean (seq ((resolve 'raster.gpu.ze-runtime/query-devices))))
              (catch Throwable _ false))))

(deftest byte-transpose-round-trips-on-device
  (if-not @gpu?
    (println "[skip] byte-transpose: no GPU")
    (testing "int8 [rows,cols] → [cols,rows] transpose is element-exact"
      (let [ze (find-ns 'raster.gpu.ze-runtime)
            make-buffer (ns-resolve ze 'make-buffer)
            arr->buf! (ns-resolve ze 'array->buffer!)
            bind-t! (ns-resolve ze 'bind-registered-transpose!)
            record! (ns-resolve ze 'record-graph!)
            replay! (ns-resolve ze 'replay-graph!)
            buf->d (ns-resolve ze 'buffer->double-array)
            rows 3 cols 5
            in (byte-array (map #(byte (- (mod (* 7 %) 251) 125)) (range (* rows cols))))
            ibuf (arr->buf! (make-buffer (* rows cols) :byte) in)
            obuf (make-buffer (* rows cols) :byte)
            g (record! [{:bound (bind-t! ibuf obuf rows cols :byte) :kernel-name "t"}])]
        (replay! g)
        (let [out (mapv int (buf->d obuf))
              ref (vec (for [j (range cols) i (range rows)] (int (aget in (+ (* i cols) j)))))]
          (is (= out ref)))))))
