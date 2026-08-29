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

(def ^:private horizontal-maps
  '(let* [u (raster.par/pmap i n float (* (clojure.core/aget a i) 2.0))
          v (raster.par/pmap j n float (+ (clojure.core/aget b j) 1.0))]
         [u v]))

(deftest pure-vertical-fusion-becomes-a-first-class-typed-program
  (let [{:keys [program stats]} (route/attempt map-map :float)
        equation (first (:equations program))]
    (is (= :typed-soac (:dialect program)))
    (is (= :typed-soac (:route stats)))
    (is (= :analyzed-source (:front-end stats)))
    (is (:typed-validated stats))
    (is (= 1 (:vertical stats)))
    (is (= 1 (count (:equations program))))
    (is (= (:algorithm equation) (dialect/validate! (:algorithm equation))))
    (is (not-any? #{'y} (flatten (:algorithm equation))))
    (is (not-any? #{'y} (flatten (:source program))))
    (is (some #{'clojure.core/float-array} (flatten (:source program))))))

(deftest host-visible-intermediate-remains-materialized-on-the-typed-route
  (let [source '(let* [y (raster.par/pmap i n float (* (clojure.core/aget x i) 2.0))
                       z (raster.par/pmap j n float (+ (clojure.core/aget y j) 1.0))]
                      [y z])
        {:keys [program stats]} (route/attempt source :float {'x :float})]
    (is (= :typed-soac (:dialect program)))
    (is (:typed-validated stats))
    (is (zero? (:vertical stats)))
    (is (= 2 (count (:equations program))))
    (is (= '[y z] (:outputs program))
        "a host-visible producer is retained, not erased by vertical fusion")))

(deftest effectful-host-binding-keeps-the-compatibility-route
  (let [source '(let* [y (raster.par/pmap i n float (* (clojure.core/aget x i) 2.0))
                       side (println y)
                       z (raster.par/pmap j n float (+ (clojure.core/aget y j) 1.0))]
                      z)]
    (is (nil? (route/attempt source :float)))))

(deftest scalar-equations-require-retained-source-type-facts
  (let [source '(let* [n (clojure.core/alength x)
                       y (raster.par/pmap i n float (* (clojure.core/aget x i) 2.0))
                       z (raster.par/pmap j n float (+ (clojure.core/aget y j) 1.0))]
                      z)]
    (is (= :unsupported-scalar-binding
           (get-in (route/attempt source :float {'x :float}) [:declined :reason]))
        "the compatibility adapter must not reconstruct a missing TypedClojure result type")))

(deftest semantic-alength-bounds-normalize-to-the-producing-extent
  (let [source '(let* [^long n (clojure.core/alength x)
                       y (raster.par/pmap i n float (* (clojure.core/aget x i) 2.0))
                       z (raster.par/pmap j (clojure.core/alength y) float
                                          (+ (clojure.core/aget y j) 1.0))]
                      z)
        program (:program (route/attempt source :float {'x :double}))
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
                                 source :double
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
    (is (:typed-validated stats))
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
      (let [typed (:program (route/attempt source :float))
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

(deftest typed-horizontal-fusion-lowers-to-one-explicit-multi-output-segmap
  (let [{:keys [program stats]} (route/attempt
                                 horizontal-maps :float {'a :float 'b :float})
        scheduled (:form (segop-lower/segop-lower-pass
                          program {:dtype :float :target-device :ocl:0}))
        operation (first (:operations (first (:equations scheduled))))
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        emitted (opencl-pass/opencl-pass scheduled :device-id :ocl:0
                                         :dtype :float :min-elements 1)
        kernel (first (:kernels emitted))]
    (is (= :typed-soac (:dialect program)))
    (is (:typed-validated stats))
    (is (= 1 (:horizontal stats)))
    (is (= '[u v] (dialect/outputs (get-in program [:equations 0 :algorithm]))))
    (is (= #{'u 'v} (:outputs operation)))
    (is (= 'u (:out-sym operation)))
    (is (some #{'clojure.core/aset} (flatten (:lambda operation))))
    (is (nil? (get-in jvm [:stats :segop-relowered])))
    (is (= [:input :input :output :output :scalar]
           (mapv :kind (:abi kernel))))
    (is (re-find #"__global float\* restrict [uv]" (:source kernel)))))

(deftest unfused-map-also-uses-the-typed-production-route
  (let [source '(let* [y (raster.par/pmap i n float
                                          (* (clojure.core/aget x i) 2.0))]
                      y)
        {:keys [program stats]} (route/attempt source :float {'x :float})]
    (is (= :typed-soac (:dialect program)))
    (is (:typed-validated stats))
    (is (zero? (:vertical stats)))
    (is (zero? (:horizontal stats)))
    (is (= 'map (dialect/operation-kind
                 (first (dialect/equations
                         (get-in program [:equations 0 :algorithm]))))))))

(deftest gpu-emission-consumes-the-same-certified-segmap
  (let [typed (:program (route/attempt map-map :float))
        scheduled (:form (segop-lower/segop-lower-pass
                          typed {:dtype :float :target-device :ocl:0}))
        emitted (opencl-pass/opencl-pass scheduled :device-id :ocl:0 :dtype :float)]
    (is (= 1 (get-in emitted [:stats :ze-maps])))
    (is (= 1 (get-in emitted [:stats :segop-reused])))
    (is (nil? (get-in emitted [:stats :segop-relowered])))
    (is (= 1 (count (:kernels emitted))))))

(deftest resident-reduction-realization-stays-on-the-typed-spine
  (let [source
        '(let* [total (raster.par/reduce acc 0.0 i n
                                         (+ acc (* ^double scale
                                                   (clojure.core/aget a i))))
                ^float scaled (* total ^float gain)
                result (raster.par/map-void!
                        j (long n)
                        (clojure.core/aset out j
                                           (float (* (clojure.core/aget a j)
                                                     scaled))))]
               result)
        {:keys [program stats]} (route/attempt
                                 source :float
                                 {'a :float 'out :float}
                                 {:resident-reductions? true})
        scheduled (:form (segop-lower/segop-lower-pass
                          program {:dtype :float :target-device :ze:0}))
        reductions (->> (:equations scheduled)
                        (mapcat :operations)
                        (filter #(instance? raster.compiler.ir.segop.SegRed %))
                        vec)
        phase-one (some #(when (= :block-local (:phase %)) %) reductions)
        phase-two (some #(when (= :cross-block (:phase %)) %) reductions)
        partial (first (:outputs phase-one))]
    (is (= :typed-soac (:dialect program)))
    (is (= :typed-soac (:route stats)))
    (is (:typed-validated stats))
    (is (= 1 (:resident-reductions stats)))
    (is (= 1 (:inlined-scalars stats)))
    (is (= [] (get-in program [:values 'total :shape])))
    (is (= :resident-scalar-buffer
           (get-in program [:values 'total :representation :kind])))
    (is (some #{'raster.par/reduce-into} (flatten (:source program))))
    (is (not-any? #{'scaled} (flatten (:source program))))
    (is (= #{:memory/write}
           (get-in (dialect/facts (get-in program [:equations 1 :algorithm]))
                   [:equations 2 :effects])))
    (is (= 'out
           (get-in (dialect/facts (get-in program [:equations 1 :algorithm]))
                   [:equations 2 :attributes :destination])))
    (is (= 2 (count reductions)))
    (is (= #{partial} (:inputs phase-two)))
    (is (= :double (get-in scheduled [:values partial :dtype])))))

(deftest host-visible-reduction-is-not-represented-as-resident-storage
  (let [source
        '(let* [y (raster.par/pmap i n float
                                   (* (clojure.core/aget x i) 2.0))
                total (raster.par/reduce acc 0.0 j n
                                         (+ acc (clojure.core/aget y j)))]
               total)
        {:keys [program stats]} (route/attempt
                                 source :float {'x :float}
                                 {:resident-reductions? true})]
    (is (= :typed-soac (:dialect program)))
    (is (= 1 (:vertical stats)))
    (is (zero? (:resident-reductions stats)))
    (is (= :plain (get-in program [:values 'total :representation :kind])))
    (is (some #{'raster.par/reduce} (flatten (:source program))))
    (is (not-any? #{'raster.par/reduce-into} (flatten (:source program))))))

(deftest host-visible-dependent-scalar-blocks-its-resident-root
  (let [source
        '(let* [total (raster.par/reduce acc 0.0 i n
                                         (+ acc (clojure.core/aget a i)))
                ^double scaled (* total ^double gain)
                result (raster.par/map-void!
                        j n (clojure.core/aset out j
                                               (float (* (clojure.core/aget a j)
                                                         scaled))))]
               [scaled result])
        {:keys [program stats]} (route/attempt
                                 source :float {'a :float 'out :float}
                                 {:resident-reductions? true})]
    (is (= :typed-soac (:dialect program)))
    (is (zero? (:resident-reductions stats)))
    (is (= :plain (get-in program [:values 'total :representation :kind])))
    (is (= [:binding 'scaled] (get-in program [:equations 1 :site])))
    (is (some #{'raster.par/reduce} (flatten (:source program))))
    (is (not-any? #{'raster.par/reduce-into} (flatten (:source program))))))

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
