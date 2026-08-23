(ns raster.compiler.segop-consumption-test
  "THE SEGOP BOUNDARY IS REAL: opencl-pass CONSUMES what segop-lower computed. Device-free.

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
            [raster.compiler.backend.gpu.opencl-pass :as op]))

(def ^:private map-form
  '(let* [o (raster.par/map! O i 8192 nil (clojure.core/* (clojure.core/aget X i) 2.0))] o))
(def ^:private reduce-form
  '(let* [r (raster.par/reduce acc 0.0 i 8192 (clojure.core/+ acc (clojure.core/aget X i)))] r))

(defn- lowered [form] (:form (slp/segop-lower-pass form {:target-device :ze:0 :dtype :double})))
(defn- run [form] (op/opencl-pass form :device-id :ze:0 :dtype :double :min-elements 0))
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
