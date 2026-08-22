(ns raster.compiler.intrinsic-opacity-test
  "A HARDWARE INTRINSIC'S BODY IS THE JVM REFERENCE, NEVER A TEMPLATE TO INLINE.

   `par/dp4a` was a plain `defn`. The walker had no return type for it, so every let-bound
   `(par/dp4a …)` intermediate reached the GPU fixpoint edge untagged and the typedness census
   (correctly) refused gemma-3-270m's `bind-decode!` — 14 bindings, all from the d1/d2 pairs in
   the Q4_K kernel. Found by the pretrained-rstr continuation anchor on the 0.2.287→0.2.320 bump.

   The naive fix — make it a `deftm` — was WORSE and silent: the inliner expanded the deftm's body
   into the kernel, including its inner `(fn [x sh] …)` sign-extension helper, which has no C
   lowering and was emitted as raw Clojure text (`fn_star_(([x_α sh_α] (let* …`). Invalid OpenCL,
   no error, and `rstr_dp4a` vanished from the kernel. Nothing caught it: the existing dp4a device
   test routes through the staged contraction emitter, never through a deftm BODY.

   The fix is the existing int8-MAC-seam convention (`quant/kernels.clj`): `^:no-inline`. The
   backend then lowers the devirtualized call through the intrinsics registry. This test pins all
   three properties, on a deftm body, which is the path that was unguarded."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.arrays :as ra]
            [raster.par :as par]
            [raster.compiler.pipeline :as pl]
            [raster.compiler.core.dispatch :as dispatch]))

;; a deftm that uses dp4a in exactly the shape that failed: a let-bound intermediate chain
(deftm dp4a-chain! [wp :- (Array int), xp :- (Array int), out :- (Array int), n :- Long] :- Void
  (par/map-void! i n
    (let [wi (ra/aget wp i)
          lo (bit-and wi 0x0F0F0F0F)
          hi (bit-and (bit-shift-right wi 4) 0x0F0F0F0F)
          d1 (par/dp4a lo (ra/aget xp i) 0)
          d2 (par/dp4a hi (ra/aget xp i) d1)]
      (ra/aset out i d2))))

(deftest an-intrinsic-is-opaque-to-the-inliner
  (testing "the single predicate every pass consults says so"
    (is (dispatch/no-inline? #'par/dp4a))))

(deftest an-intrinsic-intermediate-is-tagged
  (testing "every walked dp4a call carries its Long result type — this is what the census needs"
    (let [wb ((requiring-resolve 'raster.core/ensure-walked-body!) #'dp4a-chain!)
          calls (filter #(= 'raster.par/dp4a (:raster.op/original (meta %))) (tree-seq coll? seq wb))]
      (is (= 2 (count calls)))
      (is (every? #(= 'long (:raster.type/tag (meta %))) calls)
          (str "tags: " (mapv #(:raster.type/tag (meta %)) calls))))))

(deftest an-intrinsic-lowers-through-the-registry-not-its-body
  (testing "the GPU kernel calls rstr_dp4a and contains NO inlined Clojure — the silent failure
            mode was raw `fn_star_(([x sh] (let* …` text in the OpenCL"
    (let [src (str (:source (first (:kernels (pl/show-pipeline #'dp4a-chain!
                                                               :target-device :ze:0 :dtype :float)))))]
      (is (= 2 (count (re-seq #"rstr_dp4a\((?!int a)" src))) "both chained CALLS lowered to the intrinsic (the definition excluded)")
      (is (re-find #"inline int rstr_dp4a\(int a, int b, int acc\)" src)
          "…and the intrinsic is DEFINED, from the registry's c-helper-src. The call was right and
           the definition was missing: `use of undeclared identifier 'rstr_dp4a'` at OpenCL
           compile. A test that checks only the call cannot see this.")
      (is (not (re-find #"fn_star_|let\*|clojure\.core|unchecked_int" src))
          "no fragment of the JVM reference body leaked into the C"))))

(deftest the-kernel-actually-compiles-for-the-device
  (testing "the only assertion that catches BOTH a leaked body and a missing helper definition:
            hand the emitted source to the OpenCL compiler"
    (let [src (str (:source (first (:kernels (pl/show-pipeline #'dp4a-chain!
                                                               :target-device :ze:0 :dtype :float)))))
          hex (get-in ((requiring-resolve 'raster.runtime.hardware/device) :ze:0) [:capabilities :device-id-hex])]
      (is (some? ((requiring-resolve 'raster.compiler.support.spirv-cache/compile-opencl-to-spirv) src :device hex))))))

(deftest the-jvm-reference-is-unchanged
  (testing "deftm + ^:no-inline changed NOTHING about the reference semantics (sign-heavy lanes)"
    (is (= -16007 (par/dp4a 0x7F80FF01 0x80FF017F -5)))
    (is (= 4 (par/dp4a 0xFFFFFFFF 0xFFFFFFFF 0)) "(-1)^2 x 4 lanes")
    (is (= 65536 (par/dp4a 0x80808080 0x80808080 0)) "(-128)^2 x 4 lanes")))
