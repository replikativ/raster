(ns raster.gpu.byte-transpose-test
  "B3-insert prerequisite: the int8 (:byte) 2D transpose — the operand-prep primitive that lets
   an :nn int8 operand reach the dp4a peak leaf (transpose B[K,N]→[N,K] at BYTE granularity, then
   reinterpret as packed int). Transposing at int32 granularity would scramble dp4a's K-packing,
   so the transpose must be byte-typed. Gated on a real GPU."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.gpu.core :as gpu]
            [raster.gpu.layout-transform-device-support :as layout]))

(def ^:private gpu?
  (delay (try (require 'raster.gpu.ze-runtime)
              (boolean (seq ((resolve 'raster.gpu.ze-runtime/query-devices))))
              (catch Throwable _ false))))

(deftest byte-transpose-round-trips-on-device
  (if-not @gpu?
    (println "[skip] byte-transpose: no GPU")
    (testing "int8 [rows,cols] → [cols,rows] transpose is element-exact"
      (let [rows 3 cols 5
            in (byte-array (map #(byte (- (mod (* 7 %) 251) 125)) (range (* rows cols))))]
        (gpu/with-gpu-session [session :ze:0]
          (gpu/alloc! session {:input [:byte (* rows cols) in]
                               :output [:byte (* rows cols) nil]})
          (let [handle (gpu/bind-kernel-call!
                        session :transpose (layout/transpose-artifact :byte)
                        [:input :output {:type :int :value rows} {:type :int :value cols}])]
            (try
              (gpu/run-kernel-graph! session handle)
              (let [out (mapv int (gpu/download session :output))
                    ref (vec (for [j (range cols) i (range rows)]
                               (int (aget in (+ (* i cols) j)))))]
                (is (= out ref)))
              (finally (gpu/release-kernel-graph! session handle)))))))))
