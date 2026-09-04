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
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.ir.kernel-abi :as kabi]))

(def ^:private nr-key :raster.compiler.pipeline/non-resident)

(defn- why [form] (get-in (pipeline/extract-gpu-program form) [nr-key :why]))

(deftest map-void-arity
  (doseq [head '[raster.gpu.ze-runtime/invoke-registered-map-void-kernel
                 raster.gpu.ocl-runtime/invoke-registered-map-void-kernel]]
    (testing (str "a correct target-specific 5-wide map-void invoke extracts: " head)
      (let [r (pipeline/extract-gpu-program
               (list 'let* ['out (list head "k" '[a b] '[s] 10)] 'out))]
        (is (nil? (nr-key r)))
        (is (= 1 (count (:steps r))))
        (is (= :map-void (:convention (first (:steps r))))))))
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

(deftest contraction-arity-and-artifact-owned-geometry
  (testing "a correct ordered-ABI contraction marker extracts without flattening its arguments"
    (let [r (pipeline/extract-gpu-program
             '(let* [result (raster.gpu.ze-runtime/invoke-registered-contraction!
                             "contract_k" [A scale C M N K])]
                    result))
          step (first (:steps r))]
      (is (nil? (nr-key r)))
      (is (= :contract (:convention step)))
      (is (= '[A scale C M N K] (:arguments step)))
      (is (not (contains? step :wg-exprs)))
      (is (not (contains? step :grid-exprs)))))
  (testing "extra operands and a non-vector ABI value list are rejected by marker name"
    (is (= :unparseable-kernel-invoke
           (why '(let* [result (raster.gpu.ze-runtime/invoke-registered-contraction!
                                "contract_k" [A C] extra)]
                       result))))
    (is (= :unparseable-kernel-invoke
           (why '(let* [result (raster.gpu.ze-runtime/invoke-registered-contraction!
                                "contract_k" (list A C))]
                       result))))))

(deftest reduce-arity
  (testing "an ordered resident reduction preserves the complete argument vector"
    (let [step (first (:steps (pipeline/extract-gpu-program
                               '(let* [out (raster.gpu.ze-runtime/invoke-registered-reduction-kernel
                                            "k" [a obuf scale n])]
                                      out))))]
      (is (= :reduce (:convention step)))
      (is (= '[a obuf scale n] (:arguments step)))))
  (testing "the compatibility reduction marker is no longer a production kernel convention"
    (let [heads (var-get (requiring-resolve 'raster.compiler.pipeline/gpu-invoke-heads))]
      (is (not (contains? heads 'raster.gpu.ze-runtime/invoke-reduction-kernel))))
    (is (= :no-kernel-steps
           (why '(let* [out (raster.gpu.ze-runtime/invoke-reduction-kernel "k" [a] 10)]
                       out)))))
  (testing "ordered reduction requires exactly one vector operand"
    (is (= :unparseable-kernel-invoke
           (why '(let* [out (raster.gpu.ze-runtime/invoke-registered-reduction-kernel
                             "k" [a obuf n] extra)]
                       out))))
    (is (= :unparseable-kernel-invoke
           (why '(let* [out (raster.gpu.ze-runtime/invoke-registered-reduction-kernel
                             "k" (list a obuf n))]
                       out))))))

(deftest ordered-reduction-residency-is-decided-by-result-role
  (let [abi [(kabi/slot 'a :input :float :role :operand)
             (kabi/slot 'out :output :float :role :result)
             (kabi/slot 'scale :scalar :float :role :parameter)
             (kabi/slot '_n_bound :scalar :int :role :bound)]
        info (constantly {:abi abi})]
    (testing "nil at the ABI result role is an explicit host-scalar staging fallback"
      (let [r (pipeline/extract-gpu-program
               '(let* [out (raster.gpu.ze-runtime/invoke-registered-reduction-kernel
                            "k" [a nil scale n])]
                      out)
               info)]
        (is (= :host-scalar-reduction (get-in r [nr-key :why])))))
    (testing "a concrete result buffer remains resident and aliases the marker binding"
      (let [r (pipeline/extract-gpu-program
               '(let* [out (raster.gpu.ze-runtime/invoke-registered-reduction-kernel
                            "k" [a obuf scale n])]
                      out)
               info)]
        (is (nil? (nr-key r)))
        (is (= 'obuf (:result r)))))))

(deftest a-host-loop-after-a-kernel-step-is-not-straight-line
  ;; A `dotimes` binding is untagged and reads no intermediate buffer, so it used to pass as a
  ;; size-let closure: the loop's effect (here a reduction into `out`) silently vanished from
  ;; the resident program. It is host control flow and is rejected by name.
  (is (= :host-control-flow
         (why '(let* [rows (clojure.core/alength b)
                      cols (clojure.core/alength x)
                      fill (raster.gpu.ze-runtime/invoke-registered-kernel "k" [b] out [] rows)
                      _eff (dotimes [i rows]
                             (dotimes [j cols]
                               (clojure.core/aset out i (clojure.core/+ (clojure.core/aget out i)
                                                                         (clojure.core/aget x j)))))]
                     out)))))

(deftest a-mutating-host-call-is-not-a-size-let
  ;; a copy the region-copy pass retains (same array: memmove) is still an effect
  (is (= :host-control-flow
         (why '(let* [fill (raster.gpu.ze-runtime/invoke-registered-kernel "k" [b] out [] n)
                      _eff (java.lang.System/arraycopy out (int 0) out (int 1) n)]
                     out)))))
