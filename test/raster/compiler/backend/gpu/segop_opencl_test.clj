(ns raster.compiler.backend.gpu.segop-opencl-test
  "SegRed OpenCL kernel generation — pins #55: aget lowering must NOT be
   ns-sensitive. A parametric-array kernel's reads arrive DEVIRTUALIZED
   ((.invk aget-impl arr i) with :raster.op/original), a typed-array kernel's
   as bare clojure.core/aget — both must lower to the subscript `arr[i]`.
   Before the fix SegRed skipped normalize-array-prims (SegMap didn't) and its
   let*-rewrap dropped the metadata, so the devirtualized shape fell through
   to a broken gpufn_aget helper call (qlinear-k) while the typed shape
   emitted x[i] (decoder-gpu)."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.passes.parallel.soac-lower :as lower]
            [raster.compiler.backend.gpu.segop-opencl :as sg]))

(defn- segred-source [body-expr]
  (let [form (list 'raster.par/reduce 'acc 0.0 'j 'n body-expr)
        s (soac/par-form->soac 'result form 0)
        segops (lower/lower-reduce s nil)]
    (:source (sg/generate-segred-kernel (first segops) 'out :dtype :float))))

(deftest segred-devirtualized-aget-lowers-to-subscript
  (testing "the parametric (.invk aget-impl …) shape — the qlinear-k side of #55"
    (let [aget-invk (with-meta (list '.invk 'raster.arrays/aget_m_floats_long-impl 'a 'j)
                               {:raster.op/original 'raster.arrays/aget
                                :raster.type/tag 'float})
          plus-invk (with-meta (list '.invk 'raster.numeric/_plus__m_double_double-impl
                                     'acc (list 'double aget-invk))
                               {:raster.op/original 'raster.numeric/+
                                :raster.type/tag 'double})
          src (segred-source plus-invk)]
      (is (not (re-find #"gpufn_aget" src))
          "devirtualized aget must normalize to a subscript, not a helper call")
      (is (re-find #"a\[" src)))))

(deftest segred-bare-aget-lowers-to-subscript
  (testing "the typed clojure.core/aget shape — the decoder-gpu side of #55"
    (let [src (segred-source '(+ acc (clojure.core/aget a j)))]
      (is (not (re-find #"gpufn_aget" src)))
      (is (re-find #"a\[" src)))))

;; ================================================================
;; Horizontally-fused multi-output SegMap: the SECONDARY output (written
;; only via a side-effect aset in the fused lambda) must be a NON-const
;; __global array param — never a scalar. Before the fix it was declared
;; `float hfuse_out__N` (a scalar) while the body indexed it as an array:
;; a broken kernel, and the extraction layer then eval'd the bare buffer
;; symbol on the host (`Unable to resolve symbol: hfuse_out__N`).
;; ================================================================

(deftest segmap-fused-secondary-output-is-array-param
  (let [form '(raster.par/map! hout1 i n float
                               (do (raster.arrays/aset hout2 i (float (* (clojure.core/aget d i)
                                                                         (clojure.core/aget a i))))
                                   (* (clojure.core/aget d i) (clojure.core/aget b i))))
        s (soac/par-form->soac 'da form 0)
        segmap (first (lower/lower-map s nil :dtype :float))
        k (sg/generate-segmap-kernel segmap 'hout1 :dtype :float)]
    (testing "secondary output is an array param, not a scalar param"
      (is (some #{'hout2} (:array-params k)))
      (is (not (some #{'hout2} (:scalar-params k))))
      (is (some #{'hout2} (:written-arrays k))))
    (testing "declared __global and NON-const in the C signature"
      (is (re-find #"__global float\* restrict hout2" (:source k)))
      (is (not (re-find #"const float\* restrict hout2" (:source k)))))
    (testing "written via subscript in the body"
      (is (re-find #"hout2\[" (:source k))))))
