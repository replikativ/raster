(ns raster.compiler.passes.parallel.typed-soac-route-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.backend.jvm.par-simd :as par-simd]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-graph :as kernel-graph]
            [raster.compiler.ir.kernel-graph-call :as kernel-graph-call]
            [raster.compiler.ir.kernel-launch :as kernel-launch]
            [raster.compiler.ir.parallel-program :as parallel-program]
            [raster.compiler.ir.contraction-facts :as contraction-facts]
            [raster.compiler.ir.segop :as segop]
            [raster.compiler.ir.soac-dialect :as dialect]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.passes.parallel.contract-route :as contract-route]
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

(def ^:private inclusive-scan
  '(let* [result (raster.par/scan out acc 0.0 i n float
                                  (+ acc (clojure.core/aget x i)))]
         result))

(def ^:private exclusive-scan
  '(let* [result (raster.par/scan-exclusive out acc 0.0 i n float
                                            (+ acc (clojure.core/aget x i)))]
         result))

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
    (is (re-find #"inout_result\[idx\]" (:source artifact))
        "the scalar-region read is projected through the sole result parameter")
    (is (some #{'raster.gpu.ocl-runtime/invoke-registered-kernel}
              (tree-seq coll? seq (:form emitted))))
    (is (not (some #{'raster.gpu.ze-runtime/invoke-registered-kernel}
                   (tree-seq coll? seq (:form emitted))))
        "an OpenCL program must not leak a Level Zero staging call")))

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
    (is (= ['b 'x 'a] (mapv :name pointer-slots)))
    (is (= [:inout :input :output] (mapv :kind pointer-slots)))
    (is (= 1 (count (filter #(= 'b (:name %)) pointer-slots)))
        "a read/write destination is one physical ABI value, not aliased input and output slots")
    (is (= #{'a 'b} (:outputs (first (:operations equation)))))
    (is (re-find #"b\[idx\] = \(float\)" (:source artifact)))
    (is (re-find #"out\[idx\]" (:source artifact)))))

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
        jvm (par-simd/simd-pass scheduled :min-elements 1)
        execute (eval (list 'fn '[x a b n] (:form jvm)))
        x (float-array [1.0 2.0 3.0 4.0])
        a (float-array 4)
        b (float-array 4)]
    (is (= 1 (count (re-seq #"x\[idx\]" kernel-source)))
        "the shared producer is emitted once, not projected into both results")
    (is (re-find #"float rstr_local_0" kernel-source))
    (is (nil? (execute x a b 4)))
    (is (= [2.0 3.0 4.0 5.0] (mapv double a)))
    (is (= [4.0 9.0 16.0 25.0] (mapv double b)))))

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
        [intra _ carry] (:kernels emitted)]
    (is (= :typed-soac (:route stats)))
    (is (= :exclusive
           (get-in (dialect/operation-parts
                    (first (dialect/equations (:algorithm equation))))
                   [:attributes :mode])))
    (is (= :exclusive (get-in graph [:attributes :scan-mode])))
    (is (= (kernel-launch/sum 'n 1) (:elements (first (:outputs graph)))))
    (is (= 3 (count (:nodes graph))))
    (is (re-find #"\[0\] = 0.0f" (:source intra)))
    (is (re-find #"\[idx \+ 1\] = sdata\[tid\]" (:source intra)))
    (is (re-find #"\[idx \+ 1\]" (:source carry)))
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
    (is (re-find #"\[0\] = 0.0f" (:source artifact)))
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
    (is (re-find #"__global float\* restrict [uv]" (:source kernel)))))

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
                        (throw (ex-info "typed routing reparsed source" {})))]
          (contract-route/route-typed-contraction
           algorithm (:id operation) (:schedule operation)
           :dtype :float :desc {}))
        candidate-routes
        (with-redefs [contraction-facts/contraction-facts
                      (fn [& _]
                        (throw (ex-info "candidate routing reparsed source" {})))]
          (contract-route/route-typed-contraction-candidates!
           algorithm (:id operation) (:schedule operation)
           :dtype :float :desc {}))
        emitted
        (with-redefs [contraction-facts/contraction-facts
                      (fn [& _]
                        (throw (ex-info "typed emission reparsed source" {})))]
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
    (try
      (contract-route/route-static-typed-contraction-dispatch
       algorithm (:id operation) (:schedule operation)
       :dtype :float :desc {})
      (is false "a static dispatch must not bake runtime-dependent contraction dimensions")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :typed-contraction-dispatch-dynamic-scalar
               (:reason (ex-data exception))))))
    (try
      (contract-route/route-typed-contraction
       algorithm (:id operation) (:schedule operation) :dtype :double :desc {})
      (is false "a route dtype that disagrees with the typed equation must fail")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :typed-contraction-dtype (:reason (ex-data exception))))))
    (try
      (contract-route/route-typed-contraction
       algorithm (:id operation) (assoc (:schedule operation) :strategy :workgroup-tree)
       :dtype :float :desc {})
      (is false "a non-contraction reduction schedule must fail")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :typed-contraction-schedule (:reason (ex-data exception))))))
    (try
      (contract-route/route-typed-contraction
       algorithm (:id operation)
       (assoc-in (:schedule operation) [:tuning-space :families] [:matrix])
       :dtype :float :desc {})
      (is false "a pinned family that cannot lower the equation must fail")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :no-legal-contraction-family (:reason (ex-data exception))))))
    (try
      (contract-route/route-typed-contraction-candidates!
       algorithm (:id operation)
       (assoc-in (:schedule operation) [:tuning-space :families] [:matrix])
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
    (is (= 1 (get-in emitted [:stats :typed-contraction-dispatch-declines
                              :typed-contraction-dispatch-dynamic-scalar])))
    (is (empty? (:dispatches emitted)))
    (is (= 1 (count (:kernels emitted))))))

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
        select-family
        (fn [family]
          (contract-route/route-typed-contraction
           algorithm (:id operation)
           (assoc-in (:schedule operation) [:tuning-space :families] [family])
           :dtype :half :desc descriptor))
        candidates
        (contract-route/route-typed-contraction-candidates!
         algorithm (:id operation) (:schedule operation)
         :dtype :half :desc descriptor)
        dispatch
        (contract-route/route-static-typed-contraction-dispatch
         algorithm (:id operation) (:schedule operation)
         :dtype :half :desc descriptor)
        alternatives (:alternatives dispatch)
        emitted (with-redefs [hardware/descriptor-for (constantly descriptor)]
                  (opencl-pass/opencl-pass form :device-id :ocl:0
                                           :dtype :half :min-elements 0))
        emitted-dispatch (first (:dispatches emitted))]
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
    (with-redefs [contract-route/route-contraction (fn [& _] {:strategy :full-reduce})]
      (try
        (contract-route/route-typed-contraction
         algorithm (:id operation)
         (assoc-in (:schedule operation) [:tuning-space :families] [:matrix])
         :dtype :half :desc descriptor)
        (is false "a routed leaf outside the pinned schedule family must be rejected")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :no-legal-contraction-family (:reason (ex-data exception))))
          (is (= :strategy-outside-schedule-families
                 (get-in (ex-data exception) [:declines 0 :reason]))))))
    (doseq [families [[] [:unknown]]]
      (try
        (contract-route/route-typed-contraction
         algorithm (:id operation)
         (assoc-in (:schedule operation) [:tuning-space :families] families)
         :dtype :half :desc descriptor)
        (is false "an empty or unknown candidate family must fail")
        (catch clojure.lang.ExceptionInfo exception
          (is (= :typed-contraction-families (:reason (ex-data exception)))))))))

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
