(ns raster.compiler.passes.parallel.typed-soac-route-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-route :as route]))

(def ^:private map-map
  '(let* [y (raster.par/pmap i n float
                             (* (clojure.core/aget x i) (clojure.core/aget x i)))
          z (raster.par/pmap j n float (+ (clojure.core/aget y j) 1.0))]
         z))

(def ^:private map-reduce
  '(let* [y (raster.par/pmap i n float
                             (* (clojure.core/aget x i) (clojure.core/aget x i)))
          total (raster.par/reduce acc 0.0 j n (+ acc (clojure.core/aget y j)))]
         total))

(deftest pure-vertical-fusion-becomes-a-first-class-typed-program
  (let [{:keys [program stats]} (route/attempt map-map nil :float)
        equation (first (:equations program))]
    (is (= :typed-soac (:dialect program)))
    (is (= :typed-soac (:route stats)))
    (is (:shadow-certified stats))
    (is (= 1 (:vertical stats)))
    (is (= 1 (count (:equations program))))
    (is (= (:algorithm equation) (dialect/validate! (:algorithm equation))))
    (is (not-any? #{'y} (flatten (:algorithm equation))))
    (is (not-any? #{'y} (flatten (:source program))))
    (is (some #{'clojure.core/float-array} (flatten (:source program))))))

(deftest host-visible-intermediate-declines-the-single-consumer-subset
  (let [source '(let* [y (raster.par/pmap i n float (* (clojure.core/aget x i) 2.0))
                       z (raster.par/pmap j n float (+ (clojure.core/aget y j) 1.0))]
                      [y z])]
    (is (= :typed-soac-unknown-value
           (get-in (route/attempt source nil :float) [:declined :reason]))
        "the typed route must not erase a value also consumed by scalar host control")))

(deftest effectful-host-binding-keeps-the-compatibility-route
  (let [source '(let* [y (raster.par/pmap i n float (* (clojure.core/aget x i) 2.0))
                       side (println y)
                       z (raster.par/pmap j n float (+ (clojure.core/aget y j) 1.0))]
                      z)]
    (is (nil? (route/attempt source nil :float)))))

(deftest semantic-alength-bounds-normalize-to-the-producing-extent
  (let [source '(let* [n (clojure.core/alength x)
                       y (raster.par/pmap i n float (* (clojure.core/aget x i) 2.0))
                       z (raster.par/pmap j (clojure.core/alength y) float
                                          (+ (clojure.core/aget y j) 1.0))]
                      z)
        program (:program (route/attempt source nil :float {'x :double}))
        algorithm (get-in program [:equations 0 :algorithm])]
    (is (= :typed-soac (:dialect program)))
    (is (= 'n (dialect/operation-extent (first (dialect/equations algorithm)))))
    (is (= :double (get-in (dialect/facts algorithm) [:values 'x :dtype])))
    (is (= :float (get-in (dialect/facts algorithm) [:values 'z :dtype])))))

(deftest typed-map-and-reduction-schedule-without-source-relowering
  (doseq [[label source operation-class stat]
          [["map" map-map raster.compiler.ir.segop.SegMap :simd-maps]
           ["reduce" map-reduce raster.compiler.ir.segop.SegRed :simd-reduces]]]
    (testing label
      (let [typed (:program (route/attempt source nil :float))
            {:keys [form stats]} (segop-lower/segop-lower-pass typed {:dtype :float})
            simd (par-simd/simd-pass form :min-elements 1)]
        (is (= :segop (:dialect form)))
        (is (= 1 (:typed-soac-reused stats)))
        (is (every? #(.isInstance ^Class operation-class %)
                    (:operations (first (:equations form)))))
        (is (= 1 (get-in simd [:stats stat])))
        (is (= 1 (get-in simd [:stats :segop-reused])))
        (is (nil? (get-in simd [:stats :segop-relowered])))
        (parallel-program/validate! form segop/segop-node?)))))

(deftest gpu-emission-consumes-the-same-certified-segmap
  (let [typed (:program (route/attempt map-map nil :float))
        scheduled (:form (segop-lower/segop-lower-pass
                          typed {:dtype :float :target-device :ocl:0}))
        emitted (opencl-pass/opencl-pass scheduled :device-id :ocl:0 :dtype :float)]
    (is (= 1 (get-in emitted [:stats :ze-maps])))
    (is (= 1 (get-in emitted [:stats :segop-reused])))
    (is (nil? (get-in emitted [:stats :segop-relowered])))
    (is (= 1 (count (:kernels emitted))))))

(deftest production-pass-chain-preserves-the-typed-envelope
  (let [scheduled (pipeline/run-passes
                   map-map [:soac-fuse :materialize :compound-detect :segop-lower]
                   {:dtype :float} :write-read-fused)
        simd (par-simd/simd-pass scheduled :min-elements 1)]
    (is (= :segop (:dialect scheduled)))
    (is (= :typed-soac (get-in scheduled [:provenance :source-dialect])))
    (is (= :typed-soac (get-in scheduled [:equations 0 :attributes :algorithm-dialect])))
    (is (= 1 (get-in simd [:stats :segop-reused])))
    (is (nil? (get-in simd [:stats :segop-relowered])))))
