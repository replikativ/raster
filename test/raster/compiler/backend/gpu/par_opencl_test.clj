(ns raster.compiler.backend.gpu.par-opencl-test
  (:require [clojure.test :refer [deftest is testing use-fixtures]]
            [clojure.walk :as walk]
            [raster.compiler.backend.gpu.par-opencl :as par-opencl]
            [raster.compiler.backend.gpu.opencl-pass :as opencl-pass]
            [raster.compiler.reference.indexed-transfer-opencl :as indexed-reference]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.passes.parallel.device :as device]
            [raster.compiler.passes.parallel.segop-lower-pass :as segop-lower]
            [raster.compiler.support.spirv-cache :as spirv-cache]
            [raster.runtime.hardware :as hw]
            [raster.hardware-fixture :as hardware-fixture]
            [clojure.string :as str]))

;; Set up target device for tests (no actual GPU required)
(use-fixtures :each
  hardware-fixture/isolated
  (fn [f]
    (hw/init!)
    (hw/register-target-device! :ze:0
                                {:name "Intel(R) Arc(TM) Graphics"
                                 :capabilities {:total-eus 64
                                                :threads-per-eu 8
                                                :simd-width 16
                                                :subgroup-sizes [16 32]
                                                :max-workgroup-size 1024
                                                :shared-local-memory 131072}})
    (f)))

;; ================================================================
;; Kernel generation: par/map!
;; ================================================================

(defn- emitted-map
  [form & opts]
  (first (:kernels (apply opencl-pass/opencl-pass form
                          (concat [:min-elements 0] opts)))))

(deftest declared-gpu-parameter-types-preserve-the-scalar-array-partition
  (is (= {:scalar-types {'in :long 'scale :float}
          :array-types {'packed :int 'metadata :byte 'values :float 'cache :half}}
         (opencl-pass/derive-param-types
          '[packed metadata values cache in scale]
          '[ints bytes floats shorts long float]
          :float))))

(deftest generate-segmap-kernel-artifact-simple-test
  (testing "Simple element-wise add with OpenCL syntax"
    (let [form '(raster.par/map! out i n double (+ (aget a i) (aget b i)))
          kernel (emitted-map form)]
      (is (kart/kernel-artifact? kernel))
      (is (string? (:kernel-name kernel)))
      (is (string? (:source kernel)))
      ;; OpenCL-specific syntax
      (is (str/includes? (:source kernel) "__kernel void"))
      (is (str/includes? (:source kernel) "get_group_id(0)"))
      (is (str/includes? (:source kernel) "get_local_id(0)"))
      (is (str/includes? (:source kernel) "__global const double* restrict"))
      (is (str/includes? (:source kernel) "__global double* out"))
      ;; Must NOT have CUDA syntax
      (is (not (str/includes? (:source kernel) "blockIdx")))
      (is (not (str/includes? (:source kernel) "__global__")))
      (is (not (str/includes? (:source kernel) "extern \"C\"")))
      ;; Check parameter lists
      (is (= 3 (count (kart/attribute kernel :array-params))))
      (is (= '[a b out _n_bound] (mapv :name (:abi kernel))))
      (is (= [:input :input :output :scalar] (mapv :kind (:abi kernel))))
      (is (= '[a b out n] (:arguments kernel)))
      (is (= :double (kart/attribute kernel :dtype)))
      (is (= :kernel-body (get-in kernel [:provenance :dialect])))
      (is (= :kernel-body (kart/attribute kernel :emission-route)))
      (is (= [256] (get-in kernel [:launch :workgroup-size]))))))

(deftest generate-segmap-kernel-scalar-test
  (testing "Map with scalar parameter"
    (let [form '(raster.par/map! out i n double (* alpha (aget a i)))
          kernel (emitted-map form :scalar-types {'alpha :double 'n :int})]
      (is (kart/kernel-artifact? kernel))
      (is (str/includes? (:source kernel) "alpha"))
      (is (= 2 (count (kart/attribute kernel :array-params))))
      (is (= '[a out alpha _n_bound] (mapv :name (:abi kernel))))
      (is (= [:double :double :double :int] (mapv :dtype (:abi kernel)))))))

(deftest generate-segmap-kernel-math-test
  (testing "Map with math operations"
    (let [form '(raster.par/map! out i n double (Math/sin (aget a i)))
          kernel (emitted-map form)]
      (is (some? kernel))
      (is (str/includes? (:source kernel) "sin(")))))

(deftest generate-segmap-kernel-float-test
  (testing "Float dtype kernel — no fp64 pragma"
    (let [form '(raster.par/map! out i n float (+ (aget a i) (aget b i)))
          kernel (emitted-map form :dtype :float)]
      (is (some? kernel))
      (is (str/includes? (:source kernel) "float"))
      (is (not (str/includes? (:source kernel) "cl_khr_fp64")))
      (is (= :float (kart/attribute kernel :dtype))))))

;; ================================================================
;; Kernel generation: par/map-void!
;; ================================================================

(deftest generate-par-map-void-ordered-abi-test
  (testing "multi-write/inout map-void ABI follows the emitted signature and preserves dtypes"
    (let [form '(raster.par/map-void! i n
                                      (do (aset y i (* scale (aget x i)))
                                          (aset state i (+ (aget state i) limit))))
          k (par-opencl/generate-par-map-void-kernel
             form :dtype :float
             :array-types {'state :int 'x :float 'y :float}
             :scalar-types {'limit :int 'scale :float})]
      (is (kart/kernel-artifact? k))
      (is (= '[state x y limit scale _n_bound] (mapv :name (:abi k))))
      (is (= [:output :input :output :scalar :scalar :scalar] (mapv :kind (:abi k))))
      (is (= [:int :float :float :int :float :int] (mapv :dtype (:abi k))))
      (is (= [:inout :operand :effect :parameter :parameter :bound]
             (mapv :role (:abi k))))
      (is (= '[state x y] (kart/attribute k :array-params)))
      (is (= '[state x y limit scale n] (:arguments k))))))

(deftest multi-store-lexical-snapshots-do-not-vectorize-past-their-writes
  (let [form '(raster.par/map-void!
               i n
               (let* [^float next-state (+ (clojure.core/aget state i)
                                           (clojure.core/aget grad i))]
                     (clojure.core/aset state i (float next-state))
                     (clojure.core/aset param i
                                        (float (- (clojure.core/aget param i)
                                                  next-state)))))
        source (:source
                (par-opencl/generate-par-map-void-kernel
                 form :dtype :float
                 :array-types {'state :float 'grad :float 'param :float}))]
    (is (str/includes? source "float next_state ="))
    (is (not (str/includes? source "vstore4"))
        "inlining a pre-store snapshot into sequential vector stores changes semantics")
    (is (< (str/index-of source "float next_state =")
           (str/index-of source "state[idx] =")))))

(deftest generate-gather-kernel-is-a-side-effect-artifact
  (let [k (indexed-reference/generate-par-gather-kernel
           '(raster.par/gather out src index n stride) :dtype :float)]
    (is (kart/kernel-artifact? k))
    (is (= '[out src index stride _n_bound] (mapv :name (:abi k))))
    (is (= '[out src index stride n] (:arguments k)))
    (is (= '[out src index stride n] (kcall/logical-arguments k)))
    (is (= :side-effect-map (get-in k [:effects :kind])))
    (is (= [256] (get-in k [:launch :workgroup-size])))))

(deftest strided-transfers-use-typed-lowering-inside-host-control
  (with-redefs [indexed-reference/generate-par-gather-kernel
                (fn [& _] (throw (ex-info "source oracle reached" {})))
                indexed-reference/generate-par-scatter-kernel
                (fn [& _] (throw (ex-info "source oracle reached" {})))]
    (doseq [op '[raster.par/gather raster.par/scatter!]
            dtype [:int :long]
            wrap [identity #(list 'let* ['result %] 'result)
                  #(list 'let* [] %)
                  #(list 'do %)
                  #(list 'if 'enabled % nil)]]
      (let [source (wrap (list op 'out 'src 'indices 'n 'stride))
            emitted (opencl-pass/opencl-pass
                     source :min-elements 0 :dtype :float
                     :array-types {'out :float 'src :float 'indices :int}
                     :scalar-types {'n dtype 'stride dtype 'enabled :boolean})
            artifact (first (:kernels emitted))]
        (is (= [:kernel-body] (mapv #(get-in % [:attributes :emission-route])
                                   (:kernels emitted))) (pr-str source))
        (is (zero? (get-in emitted [:stats :fallback])))
        (is (= dtype (:dtype (first (filter #(= 'stride (:name %)) (:abi artifact))))))
        (is (= :long (:dtype (first (filter #(= :bound (:role %)) (:abi artifact)))))
            "the hoisted product retains its analyzed Long dtype")
        (when (= 'if (first source))
          (is (= '(if enabled) (take 2 (:form emitted))))
          (is (nil? (last (:form emitted))))
          (is (= 'let* (first (nth (:form emitted) 2)))
              "the extent binding stays inside the chosen host branch"))
        (when (= 'raster.par/scatter! op)
          (is (= :reducing-scatter (get-in artifact [:effects :kind])))
          (is (= :inout (:kind (first (filter #(= 'out (:name %)) (:abi artifact)))))))))))

(deftest checked-extent-survives-horizontal-fusion-before-submission
  (let [source '(let* [a (raster.par/map! left i (int n) float (+ (aget x i) 1.0))
                       b (raster.par/map! right j (int n) float (* (aget x j) 2.0))]
                      [a b])
        emitted (opencl-pass/opencl-pass
                 source :min-elements 0 :dtype :float
                 :array-types {'x :float 'left :float 'right :float}
                 :scalar-types {'n :long})
        ;; Clojure's evaluator rejects an int hint on an already primitive initializer;
        ;; Raster's bytecode emitter accepts it. Remove only that evaluator hint, retaining
        ;; every executable cast and the compiler's own type facts.
        host-form (walk/postwalk
                   #(if (symbol? %) (vary-meta % dissoc :tag) %)
                   (walk/postwalk-replace
                    {'raster.gpu.ze-runtime/invoke-registered-map-void-kernel 'submit!}
                    (:form emitted)))
        execute (eval (list 'fn '[submit! n x left right] host-form))
        submissions (atom 0)
        submit! (fn [& _] (swap! submissions inc))]
    (is (= 1 (get-in emitted [:stats :direct-scheduling :horizontal])))
    (is (= 1 (get-in emitted [:stats :direct-scheduling :typed-scalar-equations])))
    (is (= 1 (count (:kernels emitted))))
    (execute submit! 3 nil nil nil)
    (is (= 1 @submissions))
    (reset! submissions 0)
    (doseq [n [(inc (long Integer/MAX_VALUE)) (dec (long Integer/MIN_VALUE))]]
      (is (thrown? ArithmeticException (execute submit! n nil nil nil))))
    (is (zero? @submissions) "the retained checked conversion runs before any kernel call")))

(deftest inactive-host-branch-does-not-evaluate-its-checked-count
  (let [emitted (opencl-pass/opencl-pass
                 '(if enabled
                    (let* [^int count (int n)
                           result (raster.par/map! out i count float (aget x i))]
                          result)
                    nil)
                 :min-elements 0 :dtype :float
                 :array-types {'x :float 'out :float}
                 :scalar-types {'n :long 'enabled :boolean})
        host-form (walk/postwalk
                   #(if (symbol? %) (vary-meta % dissoc :tag) %)
                   (walk/postwalk-replace
                    {'raster.gpu.ze-runtime/invoke-registered-kernel 'submit!}
                    (:form emitted)))
        execute (eval (list 'fn '[submit! enabled n x out] host-form))
        submit! (fn [& _] (throw (AssertionError. "unexpected submission")))]
    (is (nil? (execute submit! false nil nil nil)))
    (is (thrown? ArithmeticException
                 (execute submit! true (inc (long Integer/MAX_VALUE)) nil nil)))))

(deftest strided-transfer-retained-programs-do-not-reenter-source-lowering
  (doseq [op '[raster.par/gather raster.par/scatter!]]
    (let [source (list op 'out 'src 'indices 'n 'stride)
          program (:program (segop-lower/schedule-single-program
                             'result source
                             {:target-device :ze:0 :dtype :float
                              :array-types {'out :float 'src :float 'indices :int}
                              :scalar-types {'n :long 'stride :long}}))
          emit #(opencl-pass/opencl-pass % :min-elements 0 :dtype :float)]
      (with-redefs [segop-lower/schedule-single-program
                    (fn [& _] (throw (ex-info "typed program reparsed" {})))
                    segop-lower/schedule-source-program
                    (fn [& _] (throw (ex-info "typed program reparsed" {})))]
        (is (= [:kernel-body] (mapv #(get-in % [:attributes :emission-route])
                                   (:kernels (emit program)))))
        (is (= :unscheduled-indexed-transfer
               (try (emit (assoc program :source (list 'do source))) nil
                    (catch clojure.lang.ExceptionInfo e (:reason (ex-data e)))))))
      (let [emitted (emit (assoc-in program [:provenance :source-dialect] :soac))]
        (is (= [:kernel-body] (mapv #(get-in % [:attributes :emission-route])
                                   (:kernels emitted))))
        (is (= [:long :long]
               (mapv :dtype (filter #(= :scalar (:kind %))
                                   (:abi (first (:kernels emitted)))))))))))

;; ================================================================
;; Kernel generation: par/reduce
;; ================================================================

(deftest generate-segred-kernel-artifact-test
  (testing "Sum reduction reaches one verified SegRed kernel artifact"
    (let [form '(raster.par/reduce acc 0.0 i n (+ acc (aget a i)))
          kernel (first (:kernels (opencl-pass/opencl-pass form :min-elements 0)))]
      (is (kart/kernel-artifact? kernel))
      (is (string? (:source kernel)))
      ;; OpenCL-specific syntax
      (is (str/includes? (:source kernel) "__kernel void"))
      (is (str/includes? (:source kernel) "__local double* rstr_workgroup_reduction_scratch"))
      (is (str/includes? (:source kernel) "barrier(CLK_LOCAL_MEM_FENCE)"))
      (is (str/includes? (:source kernel) "get_group_id(0)"))
      ;; The portable workgroup tree does not depend on a vendor subgroup dialect.
      (is (not (str/includes? (:source kernel) "cl_intel_subgroups")))
      ;; Must NOT have CUDA
      (is (not (str/includes? (:source kernel) "__syncthreads()")))
      (is (= :kernel-body (get-in kernel [:provenance :dialect])))
      (is (= :segred (get-in kernel [:provenance :source-dialect])))
      (is (= :kernel-body (get-in kernel [:attributes :emission-route])))
      (let [workgroup-size (first (get-in kernel [:launch :workgroup-size]))]
        (is (pos-int? workgroup-size))
        (is (= (* workgroup-size 8)
               (get-in kernel [:launch :shared-memory-bytes])))))))

;; ================================================================
;; Pipeline pass: opencl-pass
;; ================================================================

(deftest opencl-pass-map-test
  (testing "par/map! gets replaced with ze invoke-kernel marker"
    (let [form '(let* [out (double-array n)]
                      (raster.par/map! out i n double (+ (aget a i) (aget b i)))
                      out)
          result (opencl-pass/opencl-pass form :device-id :ze:0)]
      (is (map? result))
      (is (some? (:form result)))
      (is (map? (:stats result)))
      (is (= 1 (:ze-maps (:stats result))))
      (is (= 1 (count (:kernels result))))
      (is (kart/kernel-artifact? (first (:kernels result))))
      ;; Check the marker form
      (let [transformed (:form result)
            body (nth transformed 2)] ;; body of let*
        (is (and (seq? body)
                 (= 'raster.gpu.ze-runtime/invoke-registered-kernel (first body))))))))

(deftest opencl-pass-map-void-marker-follows-abi-test
  (testing "the compatibility marker is projected from the ordered ABI"
    (doseq [[device expected-head]
            [[:ze:0 'raster.gpu.ze-runtime/invoke-registered-map-void-kernel]
             [:ocl:0 'raster.gpu.ocl-runtime/invoke-registered-map-void-kernel]]]
      (let [form '(raster.par/map-void! i n
                                        (do (aset y i (* scale (aget x i)))
                                            (aset state i (+ (aget state i) limit))))
            result (opencl-pass/opencl-pass
                    form :device-id device :dtype :float
                    :array-types {'state :int 'x :float 'y :float}
                    :scalar-types {'limit :int 'scale :float})
            marker (:form result)
            abi (:abi (first (:kernels result)))]
        (is (= expected-head (first marker)))
        (is (= (kabi/pointer-binding-names abi) (nth marker 2)))
        (is (= '[limit scale] (nth marker 3)))
        (is (= 'n (nth marker 4)))))))

(deftest opencl-pass-fallback-test
  (testing "Small arrays fall back to scalar expansion"
    (let [form '(raster.par/map! out i 100 double (+ (aget a i) (aget b i)))
          result (opencl-pass/opencl-pass form :device-id :ze:0 :min-elements 4096)]
      (is (= 1 (:fallback (:stats result))))
      (is (= 0 (:ze-maps (:stats result)))))))

(deftest opencl-pass-reduce-test
  (testing "par/reduce executes its complete scheduled graph and materializes one host scalar"
    (let [product (with-meta '(* scale (aget a i)) {:raster.type/tag 'float})
          form (with-meta
                 (list 'raster.par/reduce 'acc 0.0 'i 'n (list '+ 'acc product))
                 {:raster.type/elem-type :float})
          result (opencl-pass/opencl-pass form :device-id :ze:0 :dtype :float
                                          :scalar-types {'scale :float 'n :int})]
      (is (= 1 (:ze-reduces (:stats result))))
      (is (= 2 (count (:kernels result))))
      (let [invocation (some #(when (and (seq? %)
                                         (= 'raster.compiler.pipeline/invoke-scheduled-executable!
                                            (first %))) %)
                             (tree-seq coll? seq (:form result)))
            dispatch (first (:dispatches result))
            graph (first (:alternatives dispatch))
            scalar-read (some #(when (and (seq? %)
                                          (= 'clojure.core/aget (first %))) %)
                              (tree-seq coll? seq (:form result)))]
        (is invocation)
        (is (= '[a (float-array 1) scale n] (nth invocation 3)))
        (is (= [:block-local :cross-block]
               (mapv #(get-in % [:operation :attributes :phase]) (:nodes graph))))
        (is (= 1 (count (:temporaries graph))))
        (is (= 'clojure.core/aget (first scalar-read)))))
  (testing "reduce-into supplies its resident result at the same ordered ABI slot"
    (let [product (with-meta '(* scale (aget a i)) {:raster.type/tag 'float})
          form (with-meta
                 (list 'raster.par/reduce-into 'obuf 'acc 0.0 'i 'n
                       (list '+ 'acc product))
                 {:raster.type/elem-type :float})
          result (opencl-pass/opencl-pass form :device-id :ze:0 :dtype :float
                                          :scalar-types {'scale :float 'n :int})]
      (let [invocation (some #(when (and (seq? %)
                                        (= 'raster.compiler.pipeline/invoke-scheduled-executable!
                                           (first %))) %)
                              (tree-seq coll? seq (:form result)))]
        (is invocation)
        (is (= '[a obuf scale n] (nth invocation 3))))
      (is (= 1 (count (:dispatches result))))
      (is (= :scheduled-graph
             (:default-strategy (first (:dispatches result)))))
      (is (= [:block-local :cross-block]
             (mapv #(get-in % [:attributes :phase]) (:kernels result))))
      (is (= 'obuf (second (:arguments (last (:kernels result))))))
      (is (= [1] (get-in result [:kernels 1 :launch :group-count])))
      (is (zero? (get-in result [:stats :segop-relowered] 0)))))))

(deftest opencl-pass-full-contraction-reduction-uses-a-complete-executable
  (let [product (with-meta '(* (aget A i) (aget B i)) {:raster.type/tag 'float})
        form (list 'raster.par/contract 'O [] '[[i 8]] product)
        result (opencl-pass/opencl-pass form :device-id :ze:0 :dtype :float)
        kernel (first (:kernels result))]
    (is (= 'raster.compiler.pipeline/invoke-scheduled-executable!
           (first (second (second (:form result))))))
    (is (= '[A B O] (nth (second (second (:form result))) 3)))
    (is (= '[A B O _n_bound] (mapv :name (:abi kernel))))
    (is (= [:operand :operand :result :bound] (mapv :role (:abi kernel))))))

(deftest opencl-pass-nested-let-test
  (testing "Nested let* forms are traversed"
    (let [form '(let* [tmp (double-array n)
                       _ (raster.par/map! tmp i n double (Math/sin (aget a i)))
                       out (double-array n)
                       _ (raster.par/map! out i n double (* 2.0 (aget tmp i)))]
                      out)
          result (opencl-pass/opencl-pass form :device-id :ze:0)]
      (is (= 2 (:ze-maps (:stats result))))
      (is (= 2 (count (:kernels result)))))))

;; ================================================================
;; SPIR-V cache
;; ================================================================

(deftest spirv-cache-test
  (testing "Cache create and stats"
    (let [cache (spirv-cache/make-cache
                 :dir (str (System/getProperty "java.io.tmpdir")
                           "/raster-test-spirv-" (System/nanoTime)))]
      (is (= {:hits 0 :misses 0 :compiles 0}
             (spirv-cache/cache-stats cache)))
      ;; Put and get
      (let [src "__kernel void test(int n) {}"
            spv (byte-array [0x07 0x23 0x02 0x03])] ;; fake SPIR-V magic
        (spirv-cache/put-cache! cache src spv)
        (is (= 1 (:compiles (spirv-cache/cache-stats cache))))
        (let [cached (spirv-cache/get-cached cache src)]
          (is (some? cached))
          (is (= (seq spv) (seq cached)))
          (is (= 1 (:hits (spirv-cache/cache-stats cache))))))
      ;; Cleanup
      (spirv-cache/clear-cache! cache))))

;; ================================================================
;; Compound kernel codegen: aget index handling
;; ================================================================

(deftest compound-kernel-stencil-indexing-test
  (testing "Stencil aget emits full index expressions, not just idx"
    (let [metadata {:execution {:kind :compound :strategy :local :parallel-bound 64 :phase-count 1 :phase-kinds [:fused]}
                    :trip-count-sym '_step
                    :trip-count-bound 'nsteps
                    :inputs []
                    :outputs ['u]
                    :scratch ['k1]
                    :scalars ['alpha 'inv-dx2]
                    :phases [{:type :stencil :out 'k1
                              :inputs ['u] :idx 'i :bound 64
                              :body '(* alpha (* inv-dx2
                                                 (+ (clojure.core/aget u (clojure.core/- i 1))
                                                    (* -2.0 (clojure.core/aget u i))
                                                    (clojure.core/aget u (clojure.core/+ i 1)))))}
                             {:type :map :out 'u
                              :inputs ['u 'k1] :idx 'i :bound 64
                              :body '(+ (clojure.core/aget u i) (clojure.core/aget k1 i))}]}
          kernel (par-opencl/generate-compound-local-kernel metadata
                                                            :dtype :double)]
      ;; Stencil must use offset indexing
      (is (str/includes? (:source kernel) "u[(i - 1)]"))
      (is (str/includes? (:source kernel) "u[(i + 1)]"))
      ;; Not just u[i] for all accesses
      (is (> (count (re-seq #"u\[\(i [+-]" (:source kernel))) 0)))))

(deftest compound-kernel-local-array-size-test
  (testing "__local arrays use fixed max size, not runtime n"
    (let [metadata {:execution {:kind :compound :strategy :local
                                :parallel-bound '(clojure.core/alength k1)
                                :phase-count 1 :phase-kinds [:fused]}
                    :trip-count-sym '_step
                    :trip-count-bound 'nsteps
                    :inputs []
                    :outputs ['u]
                    :scratch ['k1]
                    :scalars []
                    :phases [{:type :map :out 'k1
                              :inputs ['u] :idx 'i :bound 'n
                              :body '(clojure.core/aget u i)}
                             {:type :map :out 'u
                              :inputs ['u 'k1] :idx 'i :bound 'n
                              :body '(+ (clojure.core/aget u i) (clojure.core/aget k1 i))}]}
          kernel (par-opencl/generate-compound-local-kernel metadata
                                                            :dtype :double)]
      ;; Must use a numeric constant, not "(clojure.core/alength k1)"
      (is (not (str/includes? (:source kernel) "clojure")))
      (is (not (str/includes? (:source kernel) "alength")))
      ;; Should have a numeric size like [1024]
      (is (re-find #"__local double \w+\[\d+\]" (:source kernel))))))

(deftest compound-kernel-fp64-pragma-test
  (testing "Double dtype includes fp64 pragma, float does not"
    (let [metadata {:execution {:kind :compound :strategy :local :parallel-bound 32 :phase-count 1 :phase-kinds [:fused]}
                    :trip-count-sym '_s :trip-count-bound 'ns
                    :inputs [] :outputs ['u] :scratch ['k]
                    :scalars []
                    :phases [{:type :map :out 'k :inputs ['u] :idx 'i :bound 32
                              :body '(clojure.core/aget u i)}
                             {:type :map :out 'u :inputs ['k] :idx 'i :bound 32
                              :body '(clojure.core/aget k i)}]}
          k-dbl (par-opencl/generate-compound-local-kernel metadata :dtype :double)
          k-flt (par-opencl/generate-compound-local-kernel metadata :dtype :float)]
      (is (str/includes? (:source k-dbl) "cl_khr_fp64"))
      (is (not (str/includes? (:source k-flt) "cl_khr_fp64")))
      (is (str/includes? (:source k-dbl) "__local double"))
      (is (str/includes? (:source k-flt) "__local float")))))

(deftest compound-kernel-output-copy-in-test
  (testing "Output arrays (read+write) are copied from __global to __local"
    (let [metadata {:execution {:kind :compound :strategy :local :parallel-bound 32 :phase-count 1 :phase-kinds [:fused]}
                    :trip-count-sym '_s :trip-count-bound 'ns
                    :inputs [] :outputs ['u] :scratch ['k]
                    :scalars []
                    :phases [{:type :map :out 'k :inputs ['u] :idx 'i :bound 32
                              :body '(clojure.core/aget u i)}
                             {:type :map :out 'u :inputs ['k] :idx 'i :bound 32
                              :body '(clojure.core/aget k i)}]}
          kernel (par-opencl/generate-compound-local-kernel metadata :dtype :double)]
      ;; u must be copied in (read+write)
      (is (str/includes? (:source kernel) "u[i] = u_global[i]"))
      ;; u must be copied out
      (is (str/includes? (:source kernel) "u_global[i] = u[i]")))))

;; ================================================================
;; Device integration
;; ================================================================

(deftest device-type-test
  (testing "Level Zero device type detection"
    (is (= :ze (device/device-type :ze:0)))
    (is (= :ze (device/device-type :ze:1)))))

(deftest select-backend-test
  (testing "Level Zero device selects :opencl backend"
    (is (= :opencl (device/select-backend :ze:0 nil)))
    (is (= :opencl (device/select-backend :ze:0 100000)))))
