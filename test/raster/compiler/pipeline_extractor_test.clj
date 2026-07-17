(ns raster.compiler.pipeline-extractor-test
  "Resident GPU-program extractor hardening (silently-ignored-information family).

   extract-gpu-program walks a straight-line fused GPU IR into flat kernel steps.
   parse-gpu-step destructures a FIXED-length invoke prefix per convention. With no
   arity check an extra operand was silently DROPPED and a short form bound nil into a
   size slot — the same shape as the resident-GEMM alpha/beta drop (#65). Each arm now
   returns nil on an unmodeled arity so the extractor rejects it BY NAME
   (:unparseable-kernel-invoke) via its existing ::non-resident mechanism, instead of
   emitting a miscompiled kernel step. These pin the reject; before the fix each malformed
   form extracted to a (wrong) step with no ::non-resident."
  (:require [clojure.test :refer [deftest testing is]]
            [raster.compiler.pipeline :as pipeline]))

(def ^:private nr-key :raster.compiler.pipeline/non-resident)

(defn- why [form] (get-in (pipeline/extract-gpu-program form) [nr-key :why]))

(deftest map-void-arity
  (testing "a correct 5-wide map-void invoke extracts to one step, no rejection"
    (let [r (pipeline/extract-gpu-program
             '(let* [out (raster.gpu.ze-runtime/invoke-registered-map-void-kernel
                          "k" [a b] [s] 10)]
                    out))]
      (is (nil? (nr-key r)))
      (is (= 1 (count (:steps r))))
      (is (= :map-void (:convention (first (:steps r)))))))
  (testing "an EXTRA operand is rejected by name, not silently dropped"
    (is (= :unparseable-kernel-invoke
           (why '(let* [out (raster.gpu.ze-runtime/invoke-registered-map-void-kernel
                             "k" [a b] [s] 10 extra)]
                       out))))))

(deftest map-arity
  (testing "a correct 6-wide map invoke extracts"
    (let [r (pipeline/extract-gpu-program
             '(let* [out (raster.gpu.ze-runtime/invoke-registered-kernel
                          "k" [a b] out0 [s] 10)]
                    out))]
      (is (nil? (nr-key r)))
      (is (= :map (:convention (first (:steps r)))))))
  (testing "a short 5-wide map invoke (missing a slot) is rejected"
    (is (= :unparseable-kernel-invoke
           (why '(let* [out (raster.gpu.ze-runtime/invoke-registered-kernel
                             "k" [a b] out0 10)]
                       out))))))

(deftest reduce-arity
  (testing "a 5-wide resident reduction and a 4-wide legacy reduction both extract"
    (is (= :reduce (:convention (first (:steps (pipeline/extract-gpu-program
                                                '(let* [out (raster.gpu.ze-runtime/invoke-reduction-kernel
                                                             "k" [a] obuf 10)]
                                                       out)))))))
    (is (= :reduce (:convention (first (:steps (pipeline/extract-gpu-program
                                                '(let* [out (raster.gpu.ze-runtime/invoke-reduction-kernel
                                                             "k" [a] 10)]
                                                       out))))))))
  (testing "a 6-wide reduction (unmodeled) is REJECTED, not silently read as legacy 3-arg"
    ;; Before the fix this fell to the else-branch and destructured [_ kname inputs n],
    ;; binding n = the out-buf symbol — a silent miscompile.
    (is (= :unparseable-kernel-invoke
           (why '(let* [out (raster.gpu.ze-runtime/invoke-reduction-kernel
                             "k" [a] obuf 10 extra)]
                       out))))))
