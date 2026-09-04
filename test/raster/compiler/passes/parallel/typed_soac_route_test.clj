(ns raster.compiler.passes.parallel.typed-soac-route-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.backend.gpu.segop-opencl :as segop-opencl]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.kernel-body :as kernel-body]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-graph :as kernel-graph]
            [raster.compiler.ir.kernel-graph-call :as kernel-graph-call]
            [raster.compiler.ir.kernel-launch :as kernel-launch]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.contraction-facts :as contraction-facts]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.parallel.contract-lower :as contract-lower]
            [raster.compiler.passes.parallel.contract-route :as contract-route]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.passes.parallel.typed-soac-frontend :as frontend]
            [raster.compiler.passes.parallel.typed-soac-fusion :as fusion]
            [raster.compiler.passes.parallel.typed-soac-route :as route]
            [raster.gpu.dispatch-tuning :as dispatch-tuning]
            [raster.gpu.program-tuning :as program-tuning]))

(defn- scalar-stores
  [artifact]
  (filterv #(= "raster.compiler.ir.kernel_body.ScalarStore"
               (.getName (class %)))
           (get-in artifact [:attributes :kernel-body :operations])))

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

(def ^:private expensive-fanout
  '(let* [shared (raster.par/pmap i n float
                                  (Math/exp (Math/exp (Math/exp
                                                       (clojure.core/aget x i)))))
          mapped (raster.par/pmap j n float (* (clojure.core/aget shared j) 2.0))
          reduced (raster.par/reduce acc 0.0 k n
                                     (+ acc (clojure.core/aget shared k)))]
         [mapped reduced]))

(deftest rng-fill-is-an-ordinary-typed-map
  (let [source '(let* [result (raster.par/rng-fill! seeds n base-seed)] result)
        {:keys [program stats]}
        (route/attempt source :long {'seeds :long}
                       {:scalar-types {'n :int 'base-seed :long}})
        algorithm (get-in program [:equations 0 :algorithm])
        equation (first (dialect/equations algorithm))
        operation (dialect/operation-parts equation)
        locals (:locals (dialect/lambda-parts (:lambda operation)))
        scheduled (:form (segop-lower/segop-lower-pass
                          program {:dtype :long :target-device :ze:0}))
        segmap (first (get-in scheduled [:equations 0 :operations]))
        emitted (segop-opencl/generate-scheduled-segmap-kernel
                 segmap :dtype :long
                 :array-types {'seeds :long}
                 :scalar-types {'n :int 'base-seed :long})]
    (is (= :typed-soac (:route stats)))
    (is (= 'map (:kind operation))
        "RNG is scalar algebra in the existing functional map vocabulary")
    (is (= 6 (count locals)))
    (is (= [:long :long :long :long :long :long] (mapv :dtype locals)))
    (is (= #{:wrap}
           (into #{}
                 (keep #(when (= "ScalarCompute" (some-> % class .getSimpleName))
                          (get-in % [:expression :options :overflow])))
                 (get-in emitted [:attributes :kernel-body :operations])))
        "unchecked SplitMix arithmetic remains explicit after scheduling")
    (is (re-find #"\(ulong\).* \* \(ulong\)" (:source emitted)))
    (is (not (re-find #"par_rng_fill" (:source emitted)))))
  (testing "the direct backend API enters the same complete typed mini-program"
    (let [{:keys [kernels form]}
          (opencl-pass/opencl-pass
           '(raster.par/rng-fill! seeds n base-seed)
           :dtype :long :min-elements 1
           :array-types {'seeds :long}
           :scalar-types {'n :int 'base-seed :long})]
      (is (= 1 (count kernels)))
      (is (re-find #"\(ulong\).* \* \(ulong\)" (:source (first kernels))))
      (is (some #{'raster.gpu.ze-runtime/invoke-registered-kernel} (flatten form)))
      (is (not-any? #{'raster.gpu.ze-runtime/invoke-registered-rng-fill-kernel}
                    (flatten form)))))
  (testing "the primitive signature casts emitted by macro expansion are canonicalized"
    (let [{:keys [program]}
          (route/attempt
           '(let* [result (raster.par/rng-fill! seeds (int n) (long base-seed))]
                  result)
           :long {'seeds :long})
          algorithm (get-in program [:equations 0 :algorithm])
          operation (dialect/operation-parts (first (dialect/equations algorithm)))]
      (is (= ['base-seed] (:captures operation)))
      (is (= 'n (get-in operation [:attributes :extent]))))))

(deftest production-route-retains-hardware-costed-placement-witnesses
  (let [poor (route/attempt expensive-fanout :float {'x :float}
                            {:abstract-machine {:ridge {:float 2.0}}})
        rich (route/attempt expensive-fanout :float {'x :float}
                            {:abstract-machine {:ridge {:float 100.0}}})
        poor-witness (-> poor :stats :placements first)
        rich-witness (-> rich :stats :placements first)]
    (is (= :materialize (:decision poor-witness)))
    (is (= 3 (count (get-in poor [:program :equations]))))
    (is (= (:placements (:stats poor))
           (get-in poor [:program :attributes :fusion/placements])))
    (is (= :recompute (:decision rich-witness)))
    (is (= 2 (count (get-in rich [:program :equations]))))
    (is (= :typed-soac (get-in rich [:stats :route])))))

(def ^:private inclusive-scan
  '(let* [result (raster.par/scan out acc 0.0 i n float
                                  (+ acc (clojure.core/aget x i)))]
         result))

(def ^:private exclusive-scan
  '(let* [result (raster.par/scan-exclusive out acc 0.0 i n float
                                            (+ acc (clojure.core/aget x i)))]
         result))

(def ^:private contraction-result-map
  '(let* [contract-step
          (raster.par/contract
           C [[i 4] [j 8]] [[l 16]]
           (* (clojure.core/aget A (+ (* i 16) l))
              (clojure.core/aget B (+ (* l 8) j))))
          map-step
          (raster.par/map! D t 32 nil
                           (* (+ (clojure.core/aget C t)
                                 (clojure.core/aget bias (mod t 8)))
                              scale))]
         map-step))

(deftest contraction-result-map-fuses-on-the-production-typed-route
  (let [{:keys [form stats]}
        (pipeline/schedule-parallel-form
         contraction-result-map
         {:target-device :ocl:0 :dtype :float
          :array-types {'A :float 'B :float 'C :float 'D :float 'bias :float}
          :scalar-types {'scale :float}})
        equation (first (:equations form))
        algorithm (:algorithm equation)
        typed-equation (first (dialect/equations algorithm))
        transform (get-in (dialect/operation-parts typed-equation)
                          [:attributes :result-transform])]
    (is (= :typed-soac (:source-dialect stats)))
    (is (= 1 (get-in stats [:typed-soac :vertical])))
    (is (= 1 (count (:equations form))))
    (is (= '[A B bias scale] (:operands equation)))
    (is (= '[map-step] (:results equation)))
    (is (= 'D (get-in equation [:attributes :result-storage 0 :destination])))
    (is (= '[bias] (mapv :value (:operands transform))))
    (is (= '[scale] (mapv :value (:scalars transform))))
    (is (not-any? #{'C [:effect-map 0 0]}
                  (keys (:values (dialect/facts algorithm)))))))

(deftest gpu-session-scheduling-enters-the-shared-typed-boundary
  (let [source '(raster.par/map! target i n float
                                 (+ (clojure.core/aget x i) 1.0))
        {:keys [form stats]}
        (pipeline/schedule-parallel-form
         source {:target-device :ocl:0 :dtype :float
                 :array-types {'x :float 'target :float}})
        equation (first (:equations form))
        emitted (opencl-pass/opencl-pass form :device-id :ocl:0
                                         :dtype :float :min-elements 0)
        realized-expression (-> form :source second second)]
    (testing "the top-level session expression has a typed algorithm and explicit destination"
      (is (= :segop (:dialect form)))
      (is (= :typed-soac (get-in form [:provenance :source-dialect])))
      (is (= :typed-soac (:source-dialect stats)))
      (is (= :typed-soac (get-in equation [:attributes :algorithm-dialect])))
      (is (= [{:destination 'target :access :write :host-return :buffer}]
             (get-in equation [:attributes :result-storage])))
      (is (= 'raster.par/map! (first realized-expression))
          "materialization preserves map!'s destination-returning semantics"))
    (testing "the emitter consumes the scheduled SegMap instead of reconstructing source"
      (is (= 1 (get-in emitted [:stats :segop-reused])))
      (is (nil? (get-in emitted [:stats :segop-relowered])))
      (is (= 1 (get-in emitted [:stats :ze-maps]))))))

(deftest destination-read-lowers-to-one-inout-result-slot
  (let [source '(raster.par/map! target i n float
                                 (+ (clojure.core/aget target i) 1.0))
        {:keys [form stats]}
        (pipeline/schedule-parallel-form
         source {:target-device :ocl:0 :dtype :float
                 :array-types {'target :float}})
        equation (first (:equations form))
        emitted (opencl-pass/opencl-pass form :device-id :ocl:0
                                         :dtype :float :min-elements 0)
        artifact (first (:kernels emitted))
        pointer-slots (filterv #(not= :scalar (:kind %)) (:abi artifact))]
    (is (= :typed-soac (:source-dialect stats)))
    (is (= :read-write (get-in equation [:attributes :result-storage 0 :access])))
    (is (= ['target] (mapv :name pointer-slots)))
    (is (= [:inout] (mapv :kind pointer-slots)))
    (is (= [:result] (mapv :role pointer-slots)))
    (is (= '[target n] (:arguments artifact)))
    (is (re-find #"rstr_map_load_\d+ = .*target\[" (:source artifact))
        "the scalar-region read is projected through the sole typed result parameter")
    (is (re-find #"target\[.*\] = rstr_map_value_\d+" (:source artifact)))
    (is (some #{'raster.gpu.ocl-runtime/invoke-registered-kernel}
              (tree-seq coll? seq (:form emitted))))
    (is (not (some #{'raster.gpu.ze-runtime/invoke-registered-kernel}
                   (tree-seq coll? seq (:form emitted))))
        "an OpenCL program must not leak a Level Zero staging call")))

(deftest offset-map-routes-as-typed-unique-scatter
  (let [source '(let* [result
                       (raster.par/map! out i n :offset base float
                                        (clojure.core/aget x i))]
                      result)
        {:keys [form stats]}
        (pipeline/schedule-parallel-form
         source {:target-device :ocl:0 :dtype :float
                 :array-types {'x :float 'out :float}
                 :scalar-types {'n :long 'base :long}})
        equation (first (:equations form))
        operation (first (:operations equation))
        emitted (opencl-pass/opencl-pass form :device-id :ocl:0
                                         :dtype :float :min-elements 0)
        artifact (first (:kernels emitted))
        jvm (par-simd/simd-pass form :min-elements 1)
        execute (eval (list 'fn '[out x n base] (:form jvm)))
        out (float-array [99.0 99.0 99.0 99.0 99.0 99.0])
        result (execute out (float-array [10.0 11.0 12.0]) 3 2)]
    (is (= :typed-soac (:source-dialect stats)))
    (is (instance? raster.compiler.ir.segop.SegMap operation))
    (is (= :unique (:write-conflict operation)))
    (is (= :read-write (get-in equation [:attributes :result-storage 0 :access])))
    (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
    (is (re-find #"out_\[.*base.*rstr_i" (:source artifact))
        "the scheduled GPU store retains destination[base+i] through typed index lowering")
    (is (some #{'base} (:arguments artifact)))
    (is (= 1 (get-in emitted [:stats :segop-reused])))
    (is (nil? (get-in emitted [:stats :segop-relowered])))
    (is (identical? out result))
    (is (= [99.0 99.0 10.0 11.0 12.0 99.0] (mapv double result)))))

(deftest tuple-map-deduplicates-a-read-write-physical-result
  (let [source '(raster.par/map-void!
                 i n
                 (do (clojure.core/aset a i (clojure.core/aget x i))
                     (clojure.core/aset b i
                                        (+ (clojure.core/aget b i) 2.0))))
        {:keys [form stats]}
        (pipeline/schedule-parallel-form
         source {:target-device :ocl:0 :dtype :float
                 :array-types {'x :float 'a :float 'b :float}})
        equation (first (:equations form))
        emitted (opencl-pass/opencl-pass form :device-id :ocl:0
                                         :dtype :float :min-elements 0)
        artifact (first (:kernels emitted))
        pointer-slots (filterv #(not= :scalar (:kind %)) (:abi artifact))]
    (is (= :typed-soac (:source-dialect stats)))
    (is (= [:write :read-write]
           (mapv :access (get-in equation [:attributes :result-storage]))))
    (is (= ['x 'b 'a] (mapv :name pointer-slots)))
    (is (= [:input :inout :output] (mapv :kind pointer-slots)))
    (is (= 1 (count (filter #(= 'b (:name %)) pointer-slots)))
        "a read/write destination is one physical ABI value, not aliased input and output slots")
    (is (= #{'a 'b} (:outputs (first (:operations equation)))))
    (is (re-find #"b\[.*\] = rstr_map_value_\d+" (:source artifact)))
    (is (re-find #"a\[.*\] = rstr_map_load_\d+" (:source artifact)))))

(deftest typed-tuple-map-preserves-effect-semantics-on-the-jvm
  (let [source '(let* [effect
                       (raster.par/map-void!
                        i n
                        (do (clojure.core/aset a i
                                               (float (+ (clojure.core/aget x i) 1.0)))
                            (clojure.core/aset b i
                                               (float (+ (clojure.core/aget b i) 2.0)))))]
                      effect)
        typed (:program (route/attempt source :float
                                       {'x :float 'a :float 'b :float}))
        scheduled (:form (segop-lower/segop-lower-pass typed {:dtype :float}))
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[x a b n] (:form jvm)))
        x (float-array [1.0 2.0 3.0 4.0])
        a (float-array 4)
        b (float-array [10.0 20.0 30.0 40.0])]
    (is (nil? (execute x a b 4))
        "the source effect binder remains nil after TypedSOAC materialization")
    (is (= [2.0 3.0 4.0 5.0] (mapv double a)))
    (is (= [12.0 22.0 32.0 42.0] (mapv double b)))))

(deftest shared-tuple-work-stays-single-copy-through-jvm-and-gpu-lowering
  (let [source '(let* [effect
                       (raster.par/map-void!
                        i n
                        (let* [^float shifted (+ (clojure.core/aget x i) 1.0)
                               ^float squared (* shifted shifted)]
                              (clojure.core/aset a i (float shifted))
                              (clojure.core/aset b i (float squared))))]
                      effect)
        typed (:program (route/attempt source :float {'x :float 'a :float 'b :float}))
        scheduled (:form (segop-lower/segop-lower-pass
                          typed {:dtype :float :target-device :ocl:0}))
        emitted (opencl-pass/opencl-pass scheduled :device-id :ocl:0
                                         :dtype :float :min-elements 0)
        kernel-source (:source (first (:kernels emitted)))
        operation (first (:operations (first (:equations scheduled))))
        portable (segop-opencl/generate-segmap-kernel-body
                  operation
                  :array-types {'x :float 'a :float 'b :float})
        portable-source (:source portable)
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[x a b n] (:form jvm)))
        x (float-array [1.0 2.0 3.0 4.0])
        a (float-array 4)
        b (float-array 4)]
    (is (= 1 (count (re-seq #"x\[" kernel-source)))
        "the shared producer is emitted once, not projected into both results")
    (is (= :kernel-body (get-in (first (:kernels emitted))
                                [:attributes :emission-route])))
    (is (= [:float :float] (mapv :dtype (get-in operation [:scalar-region :locals])))
        "SegMap scheduling retains TypedSOAC local dtypes as data, not symbol metadata")
    (is (= 1 (count (re-seq #"x\[" portable-source)))
        "KernelBody retains the same typed local instead of duplicating its load")
    (is (= 1 (count (filter #(and (= "ScalarCompute" (some-> % class .getSimpleName))
                                  (= :* (get-in % [:expression :op])))
                            (get-in portable [:attributes :kernel-body :operations]))))
        "the shared square is one scalar SSA definition")
    (is (nil? (execute x a b 4)))
    (is (= [2.0 3.0 4.0 5.0] (mapv double a)))
    (is (= [4.0 9.0 16.0 25.0] (mapv double b)))))

(deftest ordered-map-loop-retains-its-lexical-update-region
  (let [source '(let* [result
                       (raster.par/map! out i n float
                                        (loop* [j 0 acc (float 0.0)]
                                               (if (< (long j) (long width))
                                                 (let* [value
                                                        (clojure.core/aget
                                                         x (+ (* (long i) (long width))
                                                              (long j)))]
                                                       (recur (inc (long j))
                                                              (+ (float acc) (float value))))
                                                 acc)))]
                      result)
        typed (:program
               (route/attempt source :float {'x :float 'out :float}
                              {:scalar-types {'n :long 'width :long}}))
        scheduled (:form
                   (segop-lower/segop-lower-pass
                    typed {:dtype :float :target-device :ocl:0
                           :array-types {'x :float 'out :float}
                           :scalar-types {'n :long 'width :long}}))
        operation (first (:operations (first (:equations scheduled))))
        artifact (segop-opencl/generate-scheduled-segmap-kernel
                  operation :array-types {'x :float 'out :float}
                  :scalar-types {'n :long 'width :long})
        loop-operation
        (some #(when (= "ForLoop" (some-> % class .getSimpleName)) %)
              (get-in artifact [:attributes :kernel-body :operations]))]
    (is (= :kernel-body (get-in artifact [:attributes :emission-route])))
    (is (nil? (get-in artifact [:attributes :kernel-body-decline])))
    (is loop-operation)
    (is (some #(= "ScalarLoad" (some-> % class .getSimpleName))
              (:operations loop-operation))
        "the scoped let-bound value remains a load inside the ordered loop body")))

(deftest typed-inout-preserves-sequential-jvm-semantics
  (let [source '(let* [step (raster.par/map! target i n float
                                             (* (clojure.core/aget target i) 2.0))]
                      step)
        typed (:program (route/attempt source :float {'target :float}))
        scheduled (:form (segop-lower/segop-lower-pass typed {:dtype :float}))
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[target n] (:form jvm)))
        target (float-array [1.0 2.0 3.0 4.0])
        result (execute target 4)]
    (is (identical? target result))
    (is (= [2.0 4.0 6.0 8.0] (mapv double result)))
    (is (= 1 (get-in jvm [:stats :segop-reused])))
    (is (nil? (get-in jvm [:stats :segop-relowered])))))

(deftest gpu-session-scheduling-exposes-each-top-level-do-step
  (let [{:keys [form stats]}
        (pipeline/schedule-parallel-form
         '(do (raster.par/map! first-out i n float
                               (+ (clojure.core/aget x i) 1.0))
              (raster.par/map! second-out j n float
                               (* (clojure.core/aget first-out j) 2.0)))
         {:target-device :ocl:0 :dtype :float
          :array-types {'x :float 'first-out :float 'second-out :float}})]
    (is (= :typed-soac (:source-dialect stats)))
    (is (= :typed-soac (get-in form [:provenance :source-dialect])))
    (is (every? #(= :typed-soac (get-in % [:attributes :algorithm-dialect]))
                (:equations form)))
    (is (not (some #(and (seq? %) (= 'do (first %)))
                   (tree-seq coll? seq (:source form))))
        "top-level do is an ordered binding spine before semantic analysis")))

(deftest local-destination-allocation-survives-typed-materialization
  (let [source '(let* [target (clojure.core/float-array n)
                       step (raster.par/map! target i n float
                                             (+ (clojure.core/aget x i) 1.0))]
                      step)
        {:keys [program stats]} (route/attempt source :float {'x :float 'target :float})
        materialized-bindings (apply hash-map (second (:source program)))]
    (is (= :typed-soac (:route stats)))
    (is (= '(clojure.core/float-array n) (get materialized-bindings 'target))
        "the typed algorithm owns the write boundary, not host-side buffer allocation")
    (is (= 'raster.par/map! (first (get materialized-bindings 'step))))))

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
        {:keys [program stats]} (route/attempt source :float {'x :float})
        hardware-guided (route/attempt source :float {'x :float}
                                       {:abstract-machine {:ridge {:float 100.0}}})]
    (is (= :typed-soac (:dialect program)))
    (is (:typed-validated stats))
    (is (zero? (:vertical stats)))
    (is (= 2 (count (:equations program))))
    (is (= '[y z] (:outputs program))
        "a host-visible producer is retained, not erased by vertical fusion")
    (is (= 1 (get-in hardware-guided [:stats :vertical])))
    (is (= 1 (count (get-in hardware-guided [:program :equations])))
        "removing the read lets horizontal fusion produce both observable values in one map")
    (is (= '[y z] (-> hardware-guided :program :equations first :results)))
    (is (= '[y z] (get-in hardware-guided [:program :outputs])))))

(deftest effectful-host-binding-delimits-certified-typed-islands
  (let [source '(let* [y (raster.par/pmap i n float (* (clojure.core/aget x i) 2.0))
                       side (println y)
                       z (raster.par/pmap j n float (+ (clojure.core/aget y j) 1.0))]
                      z)
        {:keys [program stats]} (route/attempt source :float {'x :float})
        source-pairs (partition 2 (second (:source program)))
        lowered (segop-lower/segop-lower-pass program {:dtype :float})]
    (is (= :typed-soac (:dialect program)))
    (is (= '[y z] (:outputs program))
        "y crosses the typed/host boundary and therefore remains materialized")
    (is (nil? (get-in program [:values 'side]))
        "opaque host values do not pretend to be functional TypedSOAC values")
    (is (= [1] (get-in program [:attributes :host-binding-ids])))
    (is (= 2 (count (:equations program))))
    (is (= 2 (get-in lowered [:stats :typed-soac-reused]))
        "both sides of the host barrier stay on the shared typed schedule vertical")
    (is (zero? (:vertical stats))
        "the dependent maps cannot fuse across an opaque host effect")
    (is (= '(println y) (some #(when (= 'side (first %)) (second %)) source-pairs)))))

(deftest opaque-host-binding-also-blocks-horizontal-code-motion
  (let [source '(let* [left (raster.par/pmap i n float
                                             (+ (clojure.core/aget a i) 1.0))
                       side (println :between-kernels)
                       right (raster.par/pmap j n float
                                              (* (clojure.core/aget b j) 2.0))]
                      [left right])
        {:keys [program stats]}
        (route/attempt source :float {'a :float 'b :float})]
    (is (= 2 (count (:equations program))))
    (is (zero? (:horizontal stats)))
    (is (= [1] (get-in program [:attributes :host-binding-ids])))
    (is (some #{'println} (flatten (:source program))))))

(deftest scalar-only-host-control-does-not-construct-an-empty-typed-program
  (is (nil? (route/attempt '(let* [x 1 y (println x)] y) :float))))

(deftest nested-compatibility-parallel-work-is-an-opaque-host-barrier
  (let [source '(let* [mapped (raster.par/pmap i n double
                                               (* (clojure.core/aget x i) 2.0))
                       mean (let* [total (raster.par/reduce
                                          acc 0.0 j n
                                          (+ acc (clojure.core/aget mapped j)))]
                                  (/ total (double n)))]
                      mean)
        {:keys [program stats]} (route/attempt source :double {'x :double})]
    (is (= :typed-soac (:dialect program)))
    (is (= 1 (count (:equations program))))
    (is (= '[mapped] (:outputs program))
        "the typed result consumed by compatibility host work stays materialized")
    (is (= [1] (get-in program [:attributes :host-binding-ids])))
    (is (some #{'raster.par/reduce} (flatten (:source program))))
    (is (:typed-validated stats))))

(deftest scalar-equations-require-retained-source-type-facts
  (let [source '(let* [n (clojure.core/alength x)
                       y (raster.par/pmap i n float (* (clojure.core/aget x i) 2.0))
                       z (raster.par/pmap j n float (+ (clojure.core/aget y j) 1.0))]
                      z)]
    (is (= :unsupported-scalar-binding
           (get-in (route/attempt source :float {'x :float}) [:declined :reason]))
        "the compatibility adapter must not reconstruct a missing TypedClojure result type")))

(deftest route-distinguishes-source-coverage-from-compiler-contradictions
  (testing "an uncertified recurrence is an explicit source coverage decline"
    (let [source '(let* [result (raster.par/scan target h 0.0 i n float
                                                 (Math/tanh
                                                  (+ h (clojure.core/aget x i))))]
                        result)]
      (is (= :scan-not-associative
             (get-in (route/attempt source :float {'x :float 'target :float})
                     [:declined :reason])))))
  (testing "a pre-certificate shape contradiction is an honest admission decline"
    (with-redefs [frontend/form->program
                  (fn [& _]
                    (throw (ex-info "simulated value contradiction"
                                    {:reason :source-value-conflict})))]
      (is (= :source-value-conflict
             (get-in (route/attempt '(let* [x 1] x) :float) [:declined :reason])))))
  (testing "a contradiction after TypedSOAC construction is never compatibility fallback"
    (let [source '(let* [result (raster.par/pmap i n float
                                                 (clojure.core/aget x i))]
                        result)
          typed (frontend/form->program source {:dtype :float :array-types {'x :float}})]
      (with-redefs [frontend/form->program (fn [& _] typed)
                    fusion/fusion-fixpoint
                    (fn [& _]
                      (throw (ex-info "simulated fusion contradiction"
                                      {:reason :typed-soac-fusion-contradiction})))]
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"simulated fusion contradiction"
                              (route/attempt source :float {'x :float})))))))

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
        "nested source reductions are represented by typed scalar Fold terms")
    (is (some #{'fold} (mapcat flatten scheduled-lambdas)))
    (is (every? #(not (re-find #"unchecked_inc_int|raster\.par/reduce" %)) kernel-sources))
    (is (every? #(re-find #"for \(int j = 0;" %) kernel-sources)
        "the portable C leaf emits each typed Fold as explicit scalar control")
    (is (every? #(re-find #"long (cols|reused_cols)" %) kernel-sources)
        "retained long shape facts, rather than dtype/name heuristics, determine the kernel ABI")
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

(deftest typed-inclusive-scan-owns-one-certified-scheduled-graph
  (let [{:keys [program stats]} (route/attempt inclusive-scan :float
                                               {'x :float 'out :float})
        lowered (segop-lower/segop-lower-pass
                 program {:dtype :float :target-device :ocl:0
                          :array-types {'x :float 'out :float}})
        scheduled (:form lowered)
        equation (first (:equations scheduled))
        graph (get-in equation [:attributes :kernel-graph])]
    (is (= :typed-soac (:route stats)))
    (is (= 'scan (dialect/operation-kind
                  (first (dialect/equations (:algorithm equation))))))
    (is (= #{:memory/write} (:effects equation)))
    (is (= 1 (get-in lowered [:stats :typed-soac-reused])))
    (is (= 1 (get-in lowered [:stats :kernel-graphs-lowered])))
    (is (= [:intra-block :block-scan nil]
           (mapv :phase (:operations equation))))
    (is (= 3 (count (:nodes graph))))
    (is (= :typed-soac (get-in (first (:operations equation)) [:algorithm-dialect])))
    (is (= #{'out} (set (map :id (:outputs graph)))))
    (is (= 1 (count (:temporaries graph))))
    (parallel-program/validate! scheduled segop/segop-node?)))

(deftest typed-map-scan-fuses-before-the-shared-scan-schedule
  (let [source '(let* [mapped (raster.par/pmap i n float
                                               (* (clojure.core/aget x i) 2.0))
                       result (raster.par/scan out acc 0.0 j n float
                                               (+ acc (clojure.core/aget mapped j)))]
                      result)
        {:keys [program stats]} (route/attempt source :float {'x :float 'out :float})
        lowered (segop-lower/segop-lower-pass
                 program {:dtype :float :target-device :ocl:0
                          :array-types {'x :float 'out :float}})
        equation (first (:equations (:form lowered)))]
    (is (= 1 (:vertical stats)))
    (is (= 1 (count (:equations program))))
    (is (not-any? #{'mapped} (flatten (:source program))))
    (is (= 1 (get-in lowered [:stats :kernel-graphs-lowered])))
    (is (= [:intra-block :block-scan nil]
           (mapv :phase (:operations equation))))
    (parallel-program/validate! (:form lowered) segop/segop-node?)))

(deftest typed-inclusive-scan-target-lowers-to-one-executable-dispatch
  (let [typed (:program (route/attempt inclusive-scan :float
                                       {'x :float 'out :float}))
        scheduled (:form (segop-lower/segop-lower-pass
                          typed {:dtype :float :target-device :ocl:0
                                 :array-types {'x :float 'out :float}}))
        emitted (opencl-pass/opencl-pass scheduled :device-id :ocl:0 :dtype :float)
        dispatch (first (:dispatches emitted))
        executable (kdispatch/default-alternative dispatch)
        binding-expr (nth (second (:form emitted)) 1)]
    (is (= 1 (get-in emitted [:stats :kernel-graphs])))
    (is (not (contains? (:stats emitted) :graph-staging-fallbacks)))
    (is (= 3 (count (:kernels emitted))))
    (is (= 1 (count (:dispatches emitted))))
    (is (kernel-graph/kernel-graph? executable))
    (is (= executable (kdispatch/select-alternative dispatch [] :scheduled-graph)))
    (is (= 'raster.compiler.pipeline/invoke-scheduled-executable!
           (first binding-expr)))
    (is (= :ocl:0 (nth binding-expr 1)))
    (is (= (:id dispatch) (nth binding-expr 2)))
    (is (= (:arguments executable) (nth binding-expr 3)))
    (is (= 4 (count binding-expr))
        "the emitted form carries no duplicate sequential source implementation")))

(deftest typed-inclusive-scan-preserves-exact-sequential-jvm-semantics
  (let [typed (:program (route/attempt inclusive-scan :float
                                       {'x :float 'out :float}))
        scheduled (:form (segop-lower/segop-lower-pass typed {:dtype :float}))
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[out x n] (:form jvm)))
        out (float-array 5)
        result (execute out (float-array [1.0 2.0 3.0 4.0 5.0]) 5)]
    (is (identical? out result))
    (is (= [1.0 3.0 6.0 10.0 15.0] (mapv double result)))
    (is (nil? (get-in jvm [:stats :segop-relowered]))
        "JVM keeps the exact inclusive recurrence until a certified vector scan schedule lands")))

(deftest typed-exclusive-scan-keeps-one-algorithm-and-specializes-its-result-layout
  (let [{:keys [program stats]} (route/attempt exclusive-scan :float
                                               {'x :float 'out :float})
        scheduled (:form (segop-lower/segop-lower-pass
                          program {:dtype :float :target-device :ocl:0
                                   :array-types {'x :float 'out :float}}))
        equation (first (:equations scheduled))
        graph (get-in equation [:attributes :kernel-graph])
        emitted (opencl-pass/opencl-pass scheduled :device-id :ocl:0 :dtype :float)
        [intra _ carry] (:kernels emitted)
        result-index [(kernel-body/expression :add 'scan-index 1)]]
    (is (= :typed-soac (:route stats)))
    (is (= :exclusive
           (get-in (dialect/operation-parts
                    (first (dialect/equations (:algorithm equation))))
                   [:attributes :mode])))
    (is (= :exclusive (get-in graph [:attributes :scan-mode])))
    (is (= (kernel-launch/sum 'n 1) (:elements (first (:outputs graph)))))
    (is (= 3 (count (:nodes graph))))
    (is (some #(and (= 'out (:buffer %)) (= result-index (:coordinates %)))
              (scalar-stores intra)))
    (is (some #(and (= 'out (:buffer %)) (= [0] (:coordinates %))
                    (= :scan-first-lane (:predicate %)))
              (scalar-stores intra)))
    (is (some #(and (= 'out (:buffer %)) (= result-index (:coordinates %)))
              (scalar-stores carry)))
    (is (= 1 (get-in emitted [:stats :kernel-graphs])))
    (is (nil? (get-in emitted [:stats :segop-relowered])))))

(deftest typed-exclusive-scan-preserves-exact-sequential-jvm-semantics
  (let [typed (:program (route/attempt exclusive-scan :float
                                       {'x :float 'out :float}))
        scheduled (:form (segop-lower/segop-lower-pass typed {:dtype :float}))
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[out x n] (:form jvm)))
        out (float-array 6)
        result (execute out (float-array [1.0 2.0 3.0 4.0 5.0]) 5)]
    (is (identical? out result))
    (is (= [0.0 1.0 3.0 6.0 10.0 15.0] (mapv double result)))
    (is (nil? (get-in jvm [:stats :segop-relowered]))
        "JVM retains the exact exclusive recurrence without rebuilding its algorithm")))

(deftest raw-exclusive-scan-cannot-enter-the-gpu-backend-around-typed-scheduling
  (try
    (opencl-pass/opencl-pass
     '(raster.par/scan-exclusive out acc 0.0 i n float
                                 (+ acc (clojure.core/aget x i)))
     :device-id :ocl:0 :dtype :float)
    (is false "raw source must not select the obsolete backend-local scan generator")
    (catch clojure.lang.ExceptionInfo exception
      (is (= :exclusive-scan-requires-typed-schedule
             (:reason (ex-data exception)))))))

(deftest zero-length-exclusive-scan-still-materializes-the-identity
  (let [source '(let* [result (raster.par/scan-exclusive out acc 0.0 i 0 float
                                                         (+ acc 1.0))]
                      result)
        typed (:program (route/attempt source :float {'out :float}))
        scheduled (:form (segop-lower/segop-lower-pass
                          typed {:dtype :float :target-device :ocl:0
                                 :array-types {'out :float}}))
        graph (get-in scheduled [:equations 0 :attributes :kernel-graph])
        emitted (opencl-pass/opencl-pass scheduled :device-id :ocl:0 :dtype :float)
        artifact (first (:kernels emitted))
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[out] (:form jvm)))
        out (float-array 1)]
    (is (= 1 (count (:nodes graph))))
    (is (= (kernel-launch/sum 0 1) (:elements (first (:outputs graph)))))
    (is (= [1] (get-in artifact [:launch :group-count])))
    (is (some #(and (= 'out (:buffer %)) (= [0] (:coordinates %))
                    (= :scan-first-lane (:predicate %)))
              (scalar-stores artifact)))
    (is (identical? out (execute out)))
    (is (= 0.0 (double (aget out 0))))))

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
    (is (= :kernel-body (get-in kernel [:attributes :emission-route])))
    (is (re-find #"__global float\* [uv]" (:source kernel))
        "the scheduled multi-result body, not the single-result compatibility emitter, owns storage")))

(deftest typed-horizontal-fusion-combines-distinct-materialized-write-boundaries
  (let [source '(let* [u (raster.par/map! u-out i n float
                                          (* (clojure.core/aget a i) 2.0))
                       v (raster.par/map! v-out j n float
                                          (+ (clojure.core/aget b j) 1.0))]
                      [u v])
        {:keys [program stats]}
        (route/attempt source :float
                       {'a :float 'b :float 'u-out :float 'v-out :float}
                       {:scalar-types {'n :int}})
        projected (:source program)
        scheduled (:form
                   (segop-lower/segop-lower-pass
                    program {:dtype :float :target-device :ocl:0
                             :array-types {'a :float 'b :float
                                           'u-out :float 'v-out :float}
                             :scalar-types {'n :int}}))
        emitted (opencl-pass/opencl-pass scheduled :device-id :ocl:0
                                         :dtype :float :min-elements 1)
        parallel-forms (filter #(and (seq? %)
                                     (contains? #{'raster.par/map! 'raster.par/map-void!}
                                                (first %)))
                               (tree-seq coll? seq projected))]
    (is (= 1 (:horizontal stats)))
    (is (= 1 (count parallel-forms))
        "host projection must not resurrect either original producer")
    (is (= 1 (count (:equations program))))
    (is (= 1 (count (:kernels emitted))))
    (is (= 1 (get-in emitted [:stats :segop-reused])))
    (is (nil? (get-in emitted [:stats :segop-relowered])))
    (is (= #{'u-out 'v-out}
           (set (map :destination
                     (get-in program [:equations 0 :attributes :result-storage])))))))

(deftest effect-only-tuple-map-lowers-to-one-explicit-multi-output-segmap
  (let [source
        '(let* [effect
                (raster.par/map-void!
                 i n
                 (do (clojure.core/aset a i
                                        (float (+ (clojure.core/aget x i) 1.0)))
                     (clojure.core/aset b i
                                        (float (* (clojure.core/aget y i) 2.0)))))]
               [a b])
        {:keys [program stats]}
        (route/attempt source :float {'x :float 'y :float 'a :float 'b :float})
        equation (first (:equations program))
        algorithm (:algorithm equation)
        scheduled (:form (segop-lower/segop-lower-pass
                          program {:dtype :float :target-device :ocl:0}))
        operation (first (:operations (first (:equations scheduled))))
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        emitted (opencl-pass/opencl-pass scheduled :device-id :ocl:0
                                         :dtype :float :min-elements 1)
        kernel (first (:kernels emitted))]
    (is (= :typed-soac (:dialect program)))
    (is (:typed-validated stats))
    (is (= [] (:outputs program))
        "the effect binder keeps its host nil semantics")
    (is (= [[:effect-map 0 0] [:effect-map 0 1]]
           (dialect/outputs algorithm)))
    (is (= ['a 'b]
           (dialect/physical-results algorithm
                                     (first (dialect/equations algorithm)))))
    (is (= #{'a 'b} (:outputs operation)))
    (is (= 'a (:out-sym operation)))
    (is (some #{'clojure.core/aset} (flatten (:lambda operation))))
    (is (nil? (get-in jvm [:stats :segop-relowered])))
    (is (= [:input :input :output :output :scalar]
           (mapv :kind (:abi kernel))))
    (is (= ['x 'y 'b 'a 'n] (:arguments kernel)))
    (is (= 1 (get-in emitted [:stats :ze-maps])))
    (is (= 1 (get-in emitted [:stats :segop-reused])))
    (is (nil? (get-in emitted [:stats :segop-relowered])))))

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

(deftest typed-contraction-schedule-view-retains-body-scalars
  (let [source
        '(let* [step (raster.par/contract C [[i m] [j n]] [[l k]]
                                          (* alpha
                                             (clojure.core/aget A (+ (* i k) l))
                                             (clojure.core/aget B (+ (* l n) j))))]
               step)
        {:keys [form]}
        (pipeline/schedule-parallel-form
         source {:target-device :ocl:0 :dtype :float
                 :array-types {'A :float 'B :float 'C :float}
                 :scalar-types {'m :long 'n :long 'k :long 'alpha :float}})
        operation (-> form :equations first :operations first)]
    (is (= '#{m n k alpha} (segop/operation-scalars operation)))
    (is (= '#{A B} (segop/operation-inputs operation)))
    (is (= '#{C} (segop/operation-outputs operation)))))

(deftest gpu-emission-consumes-the-scheduled-typed-contraction
  (let [source
        '(let* [step (raster.par/contract C [[i m] [j n]] [[l k]]
                                          (* (clojure.core/aget A (+ (* i k) l))
                                             (clojure.core/aget B (+ (* l n) j))))]
               step)
        {:keys [form stats]}
        (pipeline/schedule-parallel-form
         source {:target-device :ocl:0 :dtype :float
                 :array-types {'A :float 'B :float 'C :float}
                 :scalar-types {'m :long 'n :long 'k :long}})
        operation (-> form :equations first :operations first)
        algorithm (-> form :equations first :algorithm)
        directly-routed
        (with-redefs [contraction-facts/contraction-facts
                      (fn [& _]
                        (throw (ex-info "typed routing reparsed source" {})))
                      contraction-facts/surface-form
                      (fn [& _]
                        (throw (ex-info "typed routing manufactured source" {})))
                      contract-lower/contract-form->segred
                      (fn [& _]
                        (throw (ex-info "typed routing rebuilt its scheduled SegRed" {})))]
          (contract-route/route-typed-contraction
           algorithm operation
           :dtype :float :desc {}))
        candidate-routes
        (with-redefs [contraction-facts/contraction-facts
                      (fn [& _]
                        (throw (ex-info "candidate routing reparsed source" {})))
                      contraction-facts/surface-form
                      (fn [& _]
                        (throw (ex-info "candidate routing manufactured source" {})))
                      contract-lower/contract-form->segred
                      (fn [& _]
                        (throw (ex-info "candidate routing rebuilt its scheduled SegRed" {})))]
          (contract-route/route-typed-contraction-candidates!
           algorithm operation
           :dtype :float :desc {}))
        emitted
        (with-redefs [contraction-facts/contraction-facts
                      (fn [& _]
                        (throw (ex-info "typed emission reparsed source" {})))
                      contraction-facts/surface-form
                      (fn [& _]
                        (throw (ex-info "typed emission manufactured source" {})))
                      contract-lower/contract-form->segred
                      (fn [& _]
                        (throw (ex-info "typed emission rebuilt its scheduled SegRed" {})))]
          (opencl-pass/opencl-pass form :device-id :ocl:0
                                   :dtype :float :min-elements 0))]
    (is (= :typed-soac (:source-dialect stats)))
    (is (instance? raster.compiler.ir.segop.SegRed operation))
    (is (= :contraction (:phase operation)))
    (is (= :hardware-contraction-candidates (get-in operation [:schedule :strategy])))
    (is (some? (:artifact directly-routed)))
    (is (= [:portable] (mapv :family (:candidates candidate-routes))))
    (is (= #{:matrix :register-tiled}
           (set (map :candidate-family (:declines candidate-routes)))))
    (is (not-any? #(= :schedule-family-disabled (:reason %))
                  (:declines candidate-routes)))
    (let [dispatch (contract-route/route-typed-contraction-dispatch
                    algorithm operation :dtype :float :desc {})
          alternative (first (:alternatives dispatch))]
      (is (= '[A B C m n k] (:arguments alternative)))
      (is (= [:extent :extent :extent]
             (mapv :role (drop 3 (:abi alternative)))))
      (is (kernel-graph-call/kernel-graph-call?
           (kernel-graph-call/make alternative {'A :a 'B :b 'C :c}
                                   {'m {:type :int :value 7}
                                    'n {:type :int :value 5}
                                    'k {:type :int :value 3}}))))
    (try
      (contract-route/route-typed-contraction
       algorithm operation :dtype :double :desc {})
      (is false "a route dtype that disagrees with the typed equation must fail")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :typed-contraction-dtype (:reason (ex-data exception))))))
    (try
      (contract-route/route-typed-contraction
       algorithm (assoc operation :phase :segmented) :dtype :float :desc {})
      (is false "a SegRed from another scheduled phase must fail the typed seam")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :typed-contraction-phase (:reason (ex-data exception))))))
    (try
      (contract-route/route-typed-contraction
       algorithm (assoc operation :schedule (assoc (:schedule operation) :strategy :workgroup-tree))
       :dtype :float :desc {})
      (is false "a non-contraction reduction schedule must fail")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :typed-contraction-schedule (:reason (ex-data exception))))))
    (try
      (contract-route/route-typed-contraction
       algorithm (assoc operation :schedule
                        (assoc-in (:schedule operation) [:tuning-space :families] [:matrix]))
       :dtype :float :desc {})
      (is false "a pinned family that cannot lower the equation must fail")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :no-legal-contraction-family (:reason (ex-data exception))))))
    (try
      (contract-route/route-typed-contraction-candidates!
       algorithm (assoc operation :schedule
                        (assoc-in (:schedule operation) [:tuning-space :families] [:matrix]))
       :dtype :float :desc {})
      (is false "candidate enumeration must fail when every enabled family declines")
      (catch clojure.lang.ExceptionInfo exception
        (let [{:keys [reason route]} (ex-data exception)]
          (is (= :typed-contraction-no-candidates reason))
          (is (empty? (:candidates route)))
          (is (= #{:matrix} (set (map :candidate-family (:declines route))))))))
    (is (= :raster.par/contract
           (get-in (dialect/operation-parts (first (dialect/equations algorithm)))
                   [:attributes :attributes :source-operation])))
    (is (= 1 (get-in emitted [:stats :ze-contracts])))
    (is (= 1 (get-in emitted [:stats :segop-reused])))
    (is (nil? (get-in emitted [:stats :segop-relowered])))
    (is (zero? (get-in emitted [:stats :typed-contraction-dispatch-declines
                                :typed-contraction-dispatch-dynamic-scalar] 0)))
    (is (= 1 (count (:dispatches emitted))))
    (is (= 1 (count (:kernels emitted))))))

(deftest compatibility-contraction-without-source-or-schedule-fails-loud
  (let [source '(raster.par/contract C [[i 8]] [[l 8]]
                                       (* (clojure.core/aget A (+ (* i 8) l))
                                          (clojure.core/aget B l)))
        facts (dissoc (contraction-facts/contraction-facts source :dtype :float) :form)]
    (try
      (contract-route/route-contraction nil :dtype :float :facts facts)
      (is false "a compatibility fallback must not reconstruct syntax implicitly")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :contraction-compatibility-form-required
               (:reason (ex-data exception))))
        (is (= :portable-segred (:leaf (ex-data exception))))
        (is (= :none (:fallback (ex-data exception))))))))

(deftest dynamic-f32-contraction-owns-its-dpas-graph-alternatives
  (let [source
        '(let* [step (raster.par/contract C [[i m] [j n]] [[l k]]
                                          (* (clojure.core/aget A (+ (* i k) l))
                                             (clojure.core/aget B (+ (* l n) j))))]
               step)
        {:keys [form]}
        (pipeline/schedule-parallel-form
         source {:target-device :ze:0 :dtype :float
                 :array-types {'A :float 'B :float 'C :float}
                 :scalar-types {'m :int 'n :int 'k :int}})
        operation (-> form :equations first :operations first)
        algorithm (-> form :equations first :algorithm)
        descriptor {:backend :ze
                    :matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}
                    :execution {:subgroup-sizes #{16 32} :max-workgroup-size 1024}
                    :subgroup-size 16 :max-workgroup-size 1024
                    :grf-bytes-per-lane 256 :machine-lanes 8192
                    :shared-local-memory 131072}
        dispatch (contract-route/route-typed-contraction-dispatch
                  algorithm operation :dtype :float :desc descriptor
                  :precision :mixed-f16-f32)
        strategies (mapv kdispatch/alternative-strategy (:alternatives dispatch))
        select (fn [m n k]
                 (kdispatch/alternative-strategy
                  (kdispatch/select-alternative dispatch
                                                [:a :b :c m n k])))]
    (is (= [:portable-segred :xmx-direct :xmx-split-k] strategies))
    (is (apply = (map :abi (:alternatives dispatch))))
    (is (apply = (map :arguments (:alternatives dispatch))))
    (is (= '[A B C m n k] (:arguments (first (:alternatives dispatch)))))
    (is (= :xmx-direct (select 512 512 512)))
    (is (= :portable-segred (select 512 510 512))
        "a misaligned half row pitch cannot enter the DPAS graph")
    (is (= :portable-segred (select 512 512 510))
        "a partial matrix K fragment cannot enter the DPAS graph")
    (is (= :mixed-f16-f32
           (get-in dispatch [:attributes :candidate-schedules :xmx-direct :precision])))
    (is (nil? (get-in dispatch [:attributes :matrix-graph-decline])))
    (testing "a non-DPAS target keeps the same semantic contraction on portable schedules"
      (let [portable (contract-route/route-typed-contraction-dispatch
                      algorithm operation :dtype :float
                      :desc (assoc descriptor :backend :cuda
                                   :matrix {:family :mma :m 16 :n 16 :k 16 :subgroup 32})
                      :precision :mixed-f16-f32)]
        (is (= [:portable-segred]
               (mapv kdispatch/alternative-strategy (:alternatives portable))))
        (is (= :mixed-dpas-target-capability
               (get-in portable [:attributes :matrix-graph-decline :reason])))))))

(deftest typed-contraction-schedule-families-control-leaf-selection
  (let [source
        '(let* [step (raster.par/contract C [[i 128] [j 128]] [[l 128]]
                                          (* (clojure.core/aget A (+ (* i 128) l))
                                             (clojure.core/aget B (+ (* l 128) j))))]
               step)
        {:keys [form]}
        (pipeline/schedule-parallel-form
         source {:target-device :ocl:0 :dtype :half
                 :array-types {'A :half 'B :half 'C :half}})
        operation (-> form :equations first :operations first)
        algorithm (-> form :equations first :algorithm)
        descriptor {:matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}
                    :grf-bytes-per-lane 256 :subgroup-size 16
                    :max-workgroup-size 1024 :shared-local-memory 131072}
        without-surface
        (fn [thunk]
          (with-redefs [contraction-facts/surface-form
                        (fn [& _]
                          (throw (ex-info "ordinary typed family manufactured source" {})))]
            (thunk)))
        select-family
        (fn [family]
          (without-surface
           #(contract-route/route-typed-contraction
             algorithm (assoc operation :schedule
                              (assoc-in (:schedule operation) [:tuning-space :families] [family]))
             :dtype :half :desc descriptor)))
        candidates
        (without-surface
         #(contract-route/route-typed-contraction-candidates!
           algorithm operation
           :dtype :half :desc descriptor))
        dispatch
        (without-surface
         #(contract-route/route-typed-contraction-dispatch
           algorithm operation
           :dtype :half :desc descriptor))
        alternatives (:alternatives dispatch)
        emitted (without-surface
                 #(with-redefs [hardware/descriptor-for (constantly descriptor)]
                    (opencl-pass/opencl-pass form :device-id :ocl:0
                                             :dtype :half :min-elements 0)))
        emitted-dispatch (first (:dispatches emitted))
        measured-selector {:kind :fixed-strategy :strategy :portable-segred}
        measured-emitted
        (without-surface
         #(with-redefs [hardware/descriptor-for (constantly descriptor)]
            (opencl-pass/opencl-pass
             form :device-id :ocl:0 :dtype :half :min-elements 0
             :schedule {:typed-contraction
                        {:measured-selectors {(:id emitted-dispatch) measured-selector}}})))
        measured-dispatch (first (:dispatches measured-emitted))]
    (is (= :dpas (:strategy (select-family :matrix))))
    (is (= :regtiled (:strategy (select-family :register-tiled))))
    (is (= :portable-segred (:strategy (select-family :portable))))
    (is (= [:matrix :register-tiled :portable]
           (mapv :family (:candidates candidates))))
    (is (= [:dpas :regtiled :portable-segred]
           (mapv :strategy (:candidates candidates))))
    (is (= [[:matrix] [:register-tiled] [:portable]]
           (mapv #(get-in % [:candidate-schedule :tuning-space :families])
                 (:candidates candidates))))
    (is (every? (comp some? :artifact) (:candidates candidates)))
    (is (empty? (:declines candidates)))
    (is (= :dpas (:default-strategy dispatch)))
    (is (re-matches #"raster_typed_contraction_dispatch_[0-9a-f]{8}" (:id dispatch)))
    (is (= [:typed-contraction :measured-selectors]
           (get-in dispatch [:attributes :tuning :schedule-path])))
    (is (= (:id dispatch) (get-in dispatch [:attributes :tuning :schedule-key])))
    (is (map? (get-in dispatch [:attributes :tuning :numerical-mode])))
    (is (= :mixed-f16-f32
           (get-in dispatch [:attributes :tuning :numerical-mode :precision])))
    (is (map? (get-in dispatch [:attributes :tuning :layout])))
    (is (= {:path [:typed-contraction :measured-selectors]
            :key (:id dispatch)}
           (get-in (program-tuning/dispatch-signature dispatch) [:schedule-target])))
    (is (= [:dpas :regtiled :portable-segred]
           (mapv #(get-in % [:attributes :strategy]) alternatives)))
    (is (every? kernel-graph/kernel-graph? alternatives))
    (is (apply = (map :abi alternatives)))
    (is (= '[A B C] (:arguments (first alternatives))))
    (is (= [6 3 4]
           (mapv #(count (get-in % [:nodes 0 :operation :abi])) alternatives))
        "leaf-only shape scalars stay private to each candidate graph")
    (doseq [alternative alternatives]
      (is (kernel-graph-call/kernel-graph-call?
           (kernel-graph-call/make alternative {'A :a 'B :b 'C :c} {}))))
    (is (= 1 (count (:dispatches emitted))))
    (is (= [:dpas :regtiled :portable-segred]
           (mapv #(get-in % [:attributes :strategy])
                 (:alternatives emitted-dispatch))))
    (is (some #(and (seq? %)
                    (= 'raster.compiler.pipeline/invoke-scheduled-executable! (first %)))
              (tree-seq coll? seq (:form emitted))))
    (is (= 3 (count (:kernels emitted))))
    (is (= 1 (get-in emitted [:stats :ze-contracts])))
    (is (= 1 (get-in emitted [:stats :kernel-graphs])))
    (is (= (:id emitted-dispatch) (:id measured-dispatch)))
    (is (= (mapv dispatch-tuning/executable-signature (:alternatives emitted-dispatch))
           (mapv dispatch-tuning/executable-signature (:alternatives measured-dispatch)))
        "counter-based emitter names must not invalidate the tuning cache on recompilation")
    (is (= measured-selector (:selector measured-dispatch)))
    (is (= :measured-fixed (get-in measured-dispatch [:attributes :selection])))
    (with-redefs [contract-route/route-contraction (fn [& _] {:strategy :full-reduce})]
      (try
        (contract-route/route-typed-contraction
         algorithm (assoc operation :schedule
                          (assoc-in (:schedule operation) [:tuning-space :families] [:matrix]))
         :dtype :half :desc descriptor)
        (is false "a routed leaf outside the pinned schedule family must be rejected")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :no-legal-contraction-family (:reason (ex-data exception))))
          (is (= :strategy-outside-schedule-families
                 (get-in (ex-data exception) [:declines 0 :reason]))))))
    (doseq [families [[] [:unknown]]]
      (try
        (contract-route/route-typed-contraction
         algorithm (assoc operation :schedule
                          (assoc-in (:schedule operation) [:tuning-space :families] families))
         :dtype :half :desc descriptor)
        (is false "an empty or unknown candidate family must fail")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :typed-contraction-families (:reason (ex-data exception)))))))))

(deftest typed-contraction-result-transform-reaches-the-matrix-store
  (let [transform {:acc 'acc
                   :expr '(raster.numeric/*
                           (raster.numeric/+ acc (clojure.core/aget bias j)) scale)
                   :operands [{:sym 'bias :dtype :float
                               :map {:groups [[['j 128]]]}}]
                   :scalars [{:sym 'scale :dtype :float}]
                   :dtype :float}
        contract (apply list
                        (concat
                         '(raster.par/contract C [[i 128] [j 128]] [[l 128]]
                                               (* (clojure.core/aget A (+ (* i 128) l))
                                                  (clojure.core/aget B (+ (* l 128) j))))
                         [:epilogue transform]))
        source (list 'let* ['step contract] 'step)
        {:keys [form stats]}
        (pipeline/schedule-parallel-form
         source {:target-device :ocl:0 :dtype :half
                 :array-types {'A :half 'B :half 'C :half 'bias :float}
                 :scalar-types {'scale :float}})
        operation (-> form :equations first :operations first)
        algorithm (-> form :equations first :algorithm)
        descriptor {:matrix {:family :dpas :m 8 :n 16 :k 16 :subgroup 16}
                    :grf-bytes-per-lane 256 :subgroup-size 16
                    :max-workgroup-size 1024 :shared-local-memory 131072}
        routed (contract-route/route-typed-contraction
                algorithm (assoc operation :schedule
                                 (assoc-in (:schedule operation)
                                           [:tuning-space :families] [:matrix]))
                :dtype :half :desc descriptor)
        emitted (with-redefs [hardware/descriptor-for (constantly descriptor)]
                  (opencl-pass/opencl-pass form :device-id :ocl:0
                                           :dtype :half :min-elements 0))
        emitted-dispatch (first (:dispatches emitted))
        alternative (first (:alternatives emitted-dispatch))
        call (kernel-graph-call/make
              alternative
              {'A (Object.) 'B (Object.) 'C (Object.) 'bias (Object.)}
              {'scale {:type :float :value 0.5}})]
    (is (= :typed-soac (:source-dialect stats)))
    (is (= :float
           (get-in (dialect/operation-parts
                    (first (dialect/equations algorithm)))
                   [:attributes :result-transform :result-dtype])))
    (is (= '#{A B bias} (segop/operation-inputs operation)))
    (is (= '#{scale} (segop/operation-scalars operation)))
    (is (= :dpas (:strategy routed)))
    (is (true? (:fused-epilogue routed)))
    (is (= '[bias] (:epilogue-operands routed)))
    (is (= '[A B C M N K bias scale] (mapv :name (:abi routed))))
    (is (re-find #"bias\[.*col" (:source routed)))
    (is (re-find #"\* scale" (:source routed)))
    (is (= 1 (get-in emitted [:stats :ze-contracts])))
    (is (some #(= '[A B C M N K bias scale] (mapv :name (:abi %)))
              (:kernels emitted)))
    (is (= [:dpas :regtiled :portable-segred]
           (mapv #(get-in % [:attributes :strategy])
                 (:alternatives emitted-dispatch))))
    (is (empty? (get-in emitted-dispatch [:attributes :declines])))
    (is (kernel-graph-call/kernel-graph-call? call))
    (is (= {:type :float :value 0.5}
           (last (-> call :nodes first :call :arguments))))
    (is (nil? (get-in emitted [:stats :segop-relowered])))))

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
                   [:equations 2 :attributes :result-storage 0 :destination])))
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
    (is (= 1 (:vertical stats)))
    (is (zero? (:resident-reductions stats)))
    (is (nil? (get-in program [:values 'total]))
        "the unobservable pre-epilogue scalar is no longer materialized")
    (is (= :plain (get-in program [:values 'scaled :representation :kind])))
    (is (= [:binding 'scaled] (get-in program [:equations 0 :site])))
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

(deftest host-controlled-bindings-keep-the-scalars-they-read
  ;; `cols` is read only by the host reduction loop the typed program leaves under host
  ;; control. The realized source must still bind it: the host closure is a root of the
  ;; dependency closure, not only the body and the physical destinations.
  (let [routed (route/attempt
                '(let* [^long rows (clojure.core/alength b)
                        ^long cols (clojure.core/alength x)
                        fill (raster.par/map! out i rows nil (clojure.core/aget b i))
                        _eff (dotimes [i rows]
                               (dotimes [j cols]
                                 (clojure.core/aset out i (clojure.core/+ (clojure.core/aget out i)
                                                                           (clojure.core/aget x j)))))]
                       out)
                :float {'b :float 'x :float 'out :float} {})
        source (:source (:program routed))
        binders (take-nth 2 (second source))]
    (is (nil? (:declined routed)))
    (is (some #{'cols} binders) (pr-str source))
    (is (some #{'_eff} binders) "the host loop itself is retained as written")))

(deftest a-producer-the-host-reads-is-not-fused-away
  ;; `y` has one typed consumer (`z`) and one host reader (the loop). Counting only typed uses
  ;; would inline `y` into `z` and then resurrect its source binding for the host: the producer
  ;; would run twice. Host reads are uses, so `y` stays its own equation.
  (let [routed (route/attempt
                '(let* [y (raster.par/pmap i n float (clojure.core/* 2.0 (clojure.core/aget x i)))
                        z (raster.par/pmap j n float (clojure.core/+ 1.0 (clojure.core/aget y j)))
                        _eff (dotimes [k n]
                               (clojure.core/aset acc 0 (clojure.core/+ (clojure.core/aget acc 0)
                                                                         (clojure.core/aget y k))))]
                       z)
                :float {'x :float 'acc :float} {:scalar-types {'n :long}})
        program (:program routed)
        source (:source program)
        binders (vec (take-nth 2 (second source)))]
    (is (nil? (:declined routed)))
    (is (= 2 (count (:equations program))) "y and z remain separate equations")
    (is (= 1 (count (filter #{'y} binders))) "y is materialized exactly once")))
