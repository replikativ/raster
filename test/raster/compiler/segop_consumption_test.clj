(ns raster.compiler.segop-consumption-test
  "THE SEGOP BOUNDARY IS REAL: GPU and JVM SIMD backends CONSUME what segop-lower computed.
   Device-free.

   `segop-lower` lowered every par form with the real target device and attached the SegOp to
   the binder symbol as `::segops`. Nothing read it (north-star §2.1: 'the pass arrows are nominal
   at the most important boundary'). `opencl-pass` re-lowered each form from scratch with `:ze:0`
   HARDCODED — ignoring both the stored result and its own `device-id`. Two lowerings of one form,
   one of them on the wrong device, and no way to tell which a kernel came from.

   Now the stored SegOp is consumed; re-lowering is the fallback, with the real device-id, and
   both paths are COUNTED (`:segop-reused` / `:segop-relowered`) so a form that bypasses the
   pass's output shows up in the stats instead of silently taking a second path."
  (:require [clojure.test :refer [deftest is testing]]
            [clojure.string :as str]
            [raster.compiler.passes.parallel.segop-lower-pass :as slp]
            [raster.compiler.ir.soac :as soac]
            [raster.compiler.backend.gpu.opencl-pass :as op]
            [raster.compiler.backend.jvm.par-simd :as par-simd]))

(def ^:private map-form
  '(let* [o (raster.par/map! O i 8192 nil (clojure.core/* (clojure.core/aget X i) 2.0))] o))
(def ^:private reduce-form
  '(let* [r (raster.par/reduce acc 0.0 i 8192 (clojure.core/+ acc (clojure.core/aget X i)))] r))

(defn- lowered [form] (:form (slp/segop-lower-pass form {:target-device :ze:0 :dtype :double})))
(defn- run [form] (op/opencl-pass form :device-id :ze:0 :dtype :double :min-elements 0))
(defn- lowered-cpu [form dtype]
  (:form (slp/segop-lower-pass form {:target-device :cpu:0 :dtype dtype})))
(defn- run-simd [form] (par-simd/simd-pass form :min-elements 0))
(defn- norm [src] (str/replace (str src) #"_\d{3,}" "_N"))

(deftest a-lowered-binding-is-consumed-not-relowered
  (doseq [[label form key] [["par/map!" map-form :ze-maps] ["par/reduce" reduce-form :ze-reduces]]]
    (testing label
      (let [st (:stats (run (lowered form)))]
        (is (= 1 (get st key)) "one kernel")
        (is (= 1 (:segop-reused st)) "…from the SegOp segop-lower attached")
        (is (nil? (:segop-relowered st)) "…and NOT re-lowered")))))

(deftest an-unlowered-form-is-relowered-and-says-so
  (testing "door C hands opencl-pass a walked body with no segop-lower pass run; the fallback
            must still work — and must be VISIBLE, not a silent second path"
    (doseq [[label form] [["par/map!" map-form] ["par/reduce" reduce-form]]]
      (testing label
        (let [st (:stats (run form))]
          (is (= 1 (:segop-relowered st)))
          (is (nil? (:segop-reused st))))))))

(deftest consuming-is-a-refactor-the-kernel-is-identical
  (testing "same kernel source whether the SegOp was consumed or re-lowered — this is the
            assertion that makes the switch safe"
    (doseq [form [map-form reduce-form]]
      (is (= (norm (:source (first (:kernels (run (lowered form))))))
             (norm (:source (first (:kernels (run form))))))))))

(deftest the-fallback-uses-the-real-device-id
  (testing "re-lowering used `:ze:0` hardcoded. A registered non-:ze:0 target must reach
            lower-soac — observable because compute-launch-params asks the descriptor for THAT id"
    (let [calls (atom [])
          orig raster.compiler.passes.parallel.soac-lower/lower-soac]
      (with-redefs [raster.compiler.passes.parallel.soac-lower/lower-soac
                    (fn [soac device-id & opts] (swap! calls conj device-id) (apply orig soac :ze:0 opts))]
        (op/opencl-pass map-form :device-id :ze:1 :dtype :double :min-elements 0))
      (is (= [:ze:1] @calls) (str "lower-soac was called with " @calls ", not the pass's device-id")))))

(deftest a-body-position-form-is-consumed-too
  (testing "segop-lower wraps a body-position par form in (do …) carrying ::body-segops; that
            must be consumed exactly like a binding's ::segops"
    (let [form '(let* [n 8192] (raster.par/map! O i n nil (clojure.core/* (clojure.core/aget X i) 2.0)))
          st (:stats (run (lowered form)))]
      (is (= 1 (:ze-maps st)))
      (is (= 1 (:segop-reused st)) "consumed from ::body-segops")
      (is (nil? (:segop-relowered st))))))

(deftest jvm-simd-consumes-the-boundary-too
  (testing "matching map/reduce SegOps are reused rather than independently re-derived"
    (doseq [[label form stat-key]
            [["par/map!" map-form :simd-maps]
             ["par/reduce" reduce-form :simd-reduces]]]
      (testing label
        (let [st (:stats (run-simd (lowered-cpu form :double)))]
          (is (= 1 (get st stat-key)))
          (is (= 1 (:segop-reused st)))
          (is (nil? (:segop-relowered st))))))))

(deftest jvm-simd-refuses-a-bound-segop-with-the-wrong-dtype
  (testing "the backend re-lowers when its locally derived dtype disagrees with the boundary;
            consuming the float SegOp as a double SIMD operation would silently change semantics"
    (let [st (:stats (run-simd (lowered-cpu map-form :float)))]
      (is (= 1 (:simd-maps st)))
      (is (= 1 (:segop-relowered st)))
      (is (nil? (:segop-reused st))))))

(deftest jvm-simd-does-not-turn-an-implementation-bug-into-scalar-fallback
  (testing "an unstructured re-lowering exception is a compiler bug, not an unsupported SIMD form"
    (with-redefs [soac/par-form->soac
                  (fn [& _] (throw (NullPointerException. "simulated SIMD lowering bug")))]
      (is (thrown-with-msg? NullPointerException #"simulated SIMD lowering bug"
                            (run-simd map-form))))))
