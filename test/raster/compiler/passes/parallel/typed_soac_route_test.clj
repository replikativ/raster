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

(deftest scalar-equations-require-retained-source-type-facts
  (let [source '(let* [n (clojure.core/alength x)
                       y (raster.par/pmap i n float (* (clojure.core/aget x i) 2.0))
                       z (raster.par/pmap j n float (+ (clojure.core/aget y j) 1.0))]
                      z)]
    (is (= :unsupported-scalar-binding
           (get-in (route/attempt source nil :float {'x :float}) [:declined :reason]))
        "the compatibility adapter must not reconstruct a missing TypedClojure result type")))

(deftest semantic-alength-bounds-normalize-to-the-producing-extent
  (let [source '(let* [^long n (clojure.core/alength x)
                       y (raster.par/pmap i n float (* (clojure.core/aget x i) 2.0))
                       z (raster.par/pmap j (clojure.core/alength y) float
                                          (+ (clojure.core/aget y j) 1.0))]
                      z)
        program (:program (route/attempt source nil :float {'x :double}))
        scalar-equation (some #(when (= 'scalar
                                        (dialect/operation-kind
                                         (first (dialect/equations (:algorithm %))))) %)
                              (:equations program))
        map-equation (some #(when (= 'map
                                     (dialect/operation-kind
                                      (first (dialect/equations (:algorithm %))))) %)
                           (:equations program))
        algorithm (:algorithm map-equation)]
    (is (= :typed-soac (:dialect program)))
    (is (= [:binding 'n] (:site scalar-equation)))
    (is (= '(clojure.core/alength x)
           (:source scalar-equation)))
    (is (= 'n (dialect/operation-extent (first (dialect/equations algorithm)))))
    (is (= :double (get-in (dialect/facts algorithm) [:values 'x :dtype])))
    (is (= :float (get-in (dialect/facts algorithm) [:values 'z :dtype])))))

(deftest scalar-shapes-and-stable-tensor-captures-route-a-nested-reduction
  (let [source
        '(let* [^long rows (clojure.core/alength b1)
                ^long cols (clojure.core/alength x)
                h (raster.par/pmap i rows double
                                   (+ (clojure.core/aget b1 i)
                                      (raster.par/reduce
                                       acc 0.0 j cols
                                       (+ acc (* (clojure.core/aget W1 (+ (* i cols) j))
                                                 (clojure.core/aget x j))))))
                a (raster.par/pmap k (clojure.core/alength h) double
                                   (max 0.0 (clojure.core/aget h k)))
                ^long out-rows (clojure.core/alength b2)
                ^long reused-cols (clojure.core/alength a)
                out (raster.par/pmap o out-rows double
                                     (+ (clojure.core/aget b2 o)
                                        (raster.par/reduce
                                         acc 0.0 j reused-cols
                                         (+ acc (* (clojure.core/aget W2
                                                                      (+ (* o reused-cols) j))
                                                   (clojure.core/aget a j))))))]
               out)
        {:keys [program stats]} (route/attempt
                                 source nil :double
                                 {'W1 :double 'W2 :double 'b1 :double 'b2 :double 'x :double})
        scheduled (:form (segop-lower/segop-lower-pass
                          program {:dtype :double :target-device :ocl:0}))
        emitted (opencl-pass/opencl-pass scheduled :device-id :ocl:0
                                         :dtype :double :min-elements 1)
        equation-kinds (mapv #(dialect/operation-kind
                               (first (dialect/equations (:algorithm %))))
                             (:equations program))
        scheduled-lambdas (->> (:equations scheduled)
                               (mapcat :operations)
                               (filter #(instance? raster.compiler.ir.segop.SegMap %))
                               (map :lambda))
        argument-kinds (mapv (fn [kernel]
                               (into {} (map (juxt :name :kind)) (:abi kernel)))
                             (:kernels emitted))
        kernel-sources (mapv :source (:kernels emitted))]
    (is (= :typed-soac (:route stats)))
    (is (:shadow-certified stats))
    (is (= 1 (:vertical stats)))
    (is (= ['scalar 'scalar 'map 'scalar 'scalar 'map] equation-kinds))
    (is (= 'rows (get-in program [:equations 4 :source]))
        "alength of the produced activation is value-numbered to its certified extent")
    (is (not-any? #{'raster.par/reduce} (mapcat flatten scheduled-lambdas))
        "nested functional reductions become scalar-region control in the scheduled IR")
    (is (some #{'loop*} (mapcat flatten scheduled-lambdas)))
    (is (every? #(not (re-find #"unchecked_inc_int|raster\.par/reduce" %)) kernel-sources))
    (is (every? #(re-find #"int (cols|reused_cols)" %) kernel-sources)
        "typed scalar shape facts, rather than dtype/name heuristics, determine the kernel ABI")
    (is (= 2 (get-in emitted [:stats :segop-reused])))
    (is (= :input (get (first argument-kinds) 'W1)))
    (is (= :input (get (first argument-kinds) 'x)))
    (is (= :input (get (second argument-kinds) 'W2)))
    (is (= :input (get (second argument-kinds) 'a)))
    (is (= :scalar (get (second argument-kinds) 'reused-cols)))))

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
