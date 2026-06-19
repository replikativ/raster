(ns raster.compiler.backend.wasm.emit-test
  "Track A — differential + validity tests for the JVM-side wasm backend.
   Emits .wasm from a real deftm's post-pass IR, then uses Chicory (a pure-JVM
   WebAssembly runtime) to (a) PARSE it — which validates the module, malformed
   bytes throw — and (b) EXECUTE it, asserting results match a reference.
   No node / native deps; runs in `clojure -M:test`."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.core :refer [deftm]]
            [raster.numeric]
            [raster.arrays]
            [raster.compiler.pipeline :as pl]
            [raster.compiler.backend.wasm.emit :as wemit])
  (:import [com.dylibso.chicory.wasm Parser]
           [com.dylibso.chicory.runtime Instance]))

(deftm saxpy-k! [a :- Double, x :- (Array double), y :- (Array double), n :- Long] :- Long
  (loop [i 0]
    (if (clojure.core/< (long i) (long n))
      (do (raster.arrays/aset y (long i)
            (raster.numeric/+ (raster.numeric/* a (raster.arrays/aget x i))
                              (raster.arrays/aget y i)))
          (recur (clojure.core/inc (long i))))
      n)))

(defn- post-pass-ir
  "The exact S-expr compile-aot hands the bytecode backend (pipeline.clj)."
  [v]
  (let [g (resolve 'raster.compiler.pipeline/get-walked-body)
        b (resolve 'raster.compiler.pipeline/build-param-env)
        c (resolve 'raster.compiler.pipeline/clean-params)
        p (resolve 'raster.compiler.pipeline/get-params)]
    (pl/run-passes (first (g v :double)) pl/forward-passes
                   {:inline? true :simd? false :dtype :double
                    :active-params (c (p v :double)) :param-env (b v :double)
                    :source-ns (.ns ^clojure.lang.Var v)})))

(defn- instantiate [^bytes wasm]
  (-> (Instance/builder (Parser/parse wasm)) (.build)))

(deftest saxpy-emits-valid-and-correct-wasm
  (let [m (wemit/compile-kernel
           {:name "saxpy"
            :params [{:sym 'a :tag 'double} {:sym 'x :tag 'doubles}
                     {:sym 'y :tag 'doubles} {:sym 'n :tag 'long}]
            :ir (post-pass-ir #'saxpy-k!)})
        inst (instantiate (:bytes m))      ; parse = validate; throws on malformed
        mem  (.memory inst)
        saxpy (.export inst "saxpy")
        n 1000, a 3.0]
    (testing "module shape"
      (is (= [:f64 :i32 :i32 :i32] (:param-types m)))
      (is (pos? (count (:bytes m)))))
    (testing "executes saxpy correctly vs reference (Chicory)"
      (dotimes [i n]
        (.writeF64 mem (* 8 i) (double i))
        (.writeF64 mem (* 8 (+ n i)) (* 0.5 (double i))))
      (.apply saxpy (long-array [(Double/doubleToRawLongBits a) 0 (* 8 n) n]))
      (let [max-err (reduce (fn [mx i]
                              (max mx (Math/abs (- (.readDouble mem (* 8 (+ n i)))
                                                   (+ (* a i) (* 0.5 i))))))
                            0.0 (range n))]
        (is (< max-err 1e-9) (str "saxpy max-err=" max-err))))))
