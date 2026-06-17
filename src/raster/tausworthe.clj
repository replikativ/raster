(ns raster.tausworthe
  "Tausworthe RNG — replicates umap.utils.tau_rand_int exactly.

  A shared low-level primitive: the RP-tree forest (raster.spatial.*) and the
  UMAP/EVoC SGD layout both draw from it. numba int64 arithmetic (wraparound,
  arithmetic >>) == JVM long, so mirroring the same expressions reproduces the
  negative-sample stream bit-for-bit. state is a flat long[] of n_vertices*3;
  `base` = vertex*3. Mutates in place, returns the draw as a *signed* int32
  widened to long (matching numpy's i4 return)."
  (:refer-clojure :exclude [aget aset alength + - * / < > <= >= ==
                            mod bit-and bit-or bit-xor
                            bit-shift-left bit-shift-right unsigned-bit-shift-right])
  (:require [raster.core :refer [deftm]]
            [raster.arrays :refer [aget aset alength]]
            [raster.numeric :refer [+ - * / < > <= >= == mod
                                    bit-and bit-or bit-xor
                                    bit-shift-left bit-shift-right
                                    unsigned-bit-shift-right]]))

(deftm tau-rand-int! [state :- (Array long) base :- Long] :- Long
  (let [s0 (aget state base)
        s1 (aget state (+ base 1))
        s2 (aget state (+ base 2))
        n0 (bit-xor (bit-and (bit-shift-left (bit-and s0 4294967294) 12) 4294967295)
                    (bit-shift-right (bit-xor (bit-and (bit-shift-left s0 13) 4294967295) s0) 19))
        n1 (bit-xor (bit-and (bit-shift-left (bit-and s1 4294967288) 4) 4294967295)
                    (bit-shift-right (bit-xor (bit-and (bit-shift-left s1 2) 4294967295) s1) 25))
        n2 (bit-xor (bit-and (bit-shift-left (bit-and s2 4294967280) 17) 4294967295)
                    (bit-shift-right (bit-xor (bit-and (bit-shift-left s2 3) 4294967295) s2) 11))]
    (aset state base n0)
    (aset state (+ base 1) n1)
    (aset state (+ base 2) n2)
    (let [m (bit-and (bit-xor (bit-xor n0 n1) n2) 4294967295)]
      ;; reinterpret low 32 bits as signed int32
      (if (>= m 2147483648) (- m 4294967296) m))))
