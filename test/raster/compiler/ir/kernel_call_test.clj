(ns raster.compiler.ir.kernel-call-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-body :as body]
            [raster.compiler.ir.kernel-executable :as kexec]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.core.layout :as layout]
            [raster.gpu.dispatch-tuning :as tuning]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.resource-cleanup :as cleanup]
            [raster.gpu.resident-value :as resident-value]
            [raster.gpu.ze-runtime :as ze]))

(def ^:private artifact
  (kart/make
   {:kernel-name "kernel_call_contract_test"
    :source "__kernel void kernel_call_contract_test(__global const float* x, __global float* out, float scale, int n) {}"
    :abi [(kabi/slot 'x :input :float :role :operand)
          (kabi/slot 'out :output :float :role :result)
          (kabi/slot 'scale :scalar :float :role :parameter)
          (kabi/slot 'n :scalar :int :role :bound)]
    :arguments '[x out scale n]
    :launch (klaunch/spec {:workgroup-size [256]
                           :group-count [(klaunch/ceil-div 'n 256)]})}))

(def ^:private args
  [:resident-x :resident-out {:type :float :value 2.0} {:type :int :value 513}])

(defn- retained-capacity-artifact
  ([shape] (retained-capacity-artifact shape (layout/row-major shape :float)))
  ([shape storage-layout]
   (let [launch (klaunch/spec {:workgroup-size [4] :group-count [1]})
         kernel (body/make
                 {:id :retained-capacity
                  :parameters [(body/->KernelParameter 'x :input :float shape :global storage-layout :operand)
                               (body/->KernelParameter 'out :output :float [1] :global
                                                       (layout/row-major [1] :float) :result)
                               (body/->KernelParameter 'n :scalar :int [] nil nil :bound)]
                  :launch launch})]
     (kart/make {:kernel-name "retained_capacity_test"
                 :source "__kernel void retained_capacity_test(__global const float* x, __global float* out, int n) {}"
                 :abi [(kabi/slot 'x :input :float :role :operand)
                       (kabi/slot 'out :output :float :role :result)
                       (kabi/slot 'n :scalar :int :role :bound)]
                 :arguments '[x out n] :launch launch :attributes {:kernel-body kernel}}))))

(defn- host-capacity [value _slot]
  (java.lang.reflect.Array/getLength value))

(deftest retained-body-capacity-uses-storage-extents-and-exact-projection
  (doseq [[required supplied] [[5 4] [2 1]]]
    (let [kernel (retained-capacity-artifact [required])
          arguments [(float-array supplied) (float-array 1) {:type :int :value 1}]]
      (try
        (kcall/validate-retained-input-capacities! kernel arguments host-capacity)
        (is false "an undersized retained input must decline")
        (catch clojure.lang.ExceptionInfo e
          (is (= :kernel-body-buffer-capacity (:reason (ex-data e))))
          (is (= required (:required-elements (ex-data e))))
          (is (= supplied (:buffer-elements (ex-data e)))))))
    (let [kernel (retained-capacity-artifact [required])]
      (is (identical? kernel
                      (kcall/validate-retained-input-capacities!
                       kernel [(float-array required) (float-array 1) {:type :int :value 1}]
                       host-capacity)))))
  (let [kernel (retained-capacity-artifact ['n])
        arguments [(float-array 4) (float-array 1) {:type :int :value 5}]]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"smaller than"
                         (kcall/validate-retained-input-capacities! kernel arguments host-capacity)))
    (is (identical? kernel (kcall/validate-retained-input-capacities! kernel arguments (constantly nil))))
    (doseq [bad [(assoc-in kernel [:attributes :kernel-body :parameters 0 :id] 'other)
                 (update-in kernel [:attributes :kernel-body :parameters] pop)]]
      (try
        (kcall/validate-retained-input-capacities! bad arguments host-capacity)
        (is false "a stale projection cannot supply storage authority")
        (catch clojure.lang.ExceptionInfo e
          (is (= :kernel-body-capacity-projection (:reason (ex-data e))))))))
  (let [kernel (retained-capacity-artifact [Long/MAX_VALUE 2])]
    (is (thrown? ArithmeticException
                 (kcall/validate-retained-input-capacities!
                  kernel [(float-array 1) (float-array 1) {:type :int :value 1}] host-capacity))))
  (let [kernel (retained-capacity-artifact [2 3]
                                          (assoc (layout/row-major [2 3] :float) :strides [5 1]))]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"smaller than"
                         (kcall/validate-retained-input-capacities!
                          kernel [(float-array 7) (float-array 1) {:type :int :value 1}] host-capacity)))
    (is (identical? kernel
                    (kcall/validate-retained-input-capacities!
                     kernel [(float-array 8) (float-array 1) {:type :int :value 1}] host-capacity))))
  (let [plain {:kernel-name "legacy" :workgroup-size 4}
        capacity-calls (atom 0)]
    (is (identical? plain (kcall/validate-retained-input-capacities!
                          plain [] (fn [& _] (swap! capacity-calls inc)))))
    (is (zero? @capacity-calls))))

(deftest retained-input-capacity-declines-public-direct-routes-before-native-contact
  (doseq [required [5 2]
          [registry invoke] [[ocl/kernel-registry ocl/invoke-registered-map-void-kernel]
                             [ze/kernel-registry ze/invoke-registered-map-void-kernel]
                             [ocl/kernel-registry (fn [name arrays _ n]
                                                    (ocl/invoke-registered-kernel name [(first arrays)] (second arrays) [] n))]
                             [ze/kernel-registry (fn [name arrays _ n]
                                                   (ze/invoke-registered-kernel name [(first arrays)] (second arrays) [] n))]
                             [ze/kernel-registry (fn [name arrays _ n]
                                                   (ze/invoke-registered-contraction! name (conj arrays n)))]]]
    (let [kernel (assoc-in (retained-capacity-artifact [required]) [:attributes :out-elems] 1)
          native-calls (atom 0)]
      (with-redefs [ocl/ensure-kernel-loaded! (fn [& _] (swap! native-calls inc) (throw (ex-info "unexpected native loading" {})))
                    ze/ensure-kernel-loaded! (fn [& _] (swap! native-calls inc) (throw (ex-info "unexpected native loading" {})))]
        (let [prior @registry]
          (try
            (reset! registry (assoc prior (:kernel-name kernel) kernel))
            (try
              (invoke (:kernel-name kernel) [(float-array (dec required)) (float-array 1)] [] 1)
              (is false "known undersized input must fail before loading or allocation")
              (catch clojure.lang.ExceptionInfo e
                (is (= :kernel-body-buffer-capacity (:reason (ex-data e))))
                (is (= required (:required-elements (ex-data e))))
                (is (= (dec required) (:buffer-elements (ex-data e))))))
            (is (zero? @native-calls))
            (finally (reset! registry prior))))))))

(deftest retained-input-capacity-declines-both-resident-binders-before-loading
  (doseq [required [5 2]
          [registry bind make-buffer context-key runtime-state]
          [[ocl/kernel-registry ocl/bind-kernel-call
            (fn [segment count] (ocl/->OclBuffer segment segment count (* count 4) :float 64))
            :raster.gpu.ocl-runtime/allocation-context (var-get (ns-resolve 'raster.gpu.ocl-runtime 'state))]
           [ze/kernel-registry ze/bind-kernel-call
            (fn [segment count] (ze/->DeviceBuffer segment count (* count 4) :float))
            :raster.gpu.ze-runtime/allocation-context (var-get (ns-resolve 'raster.gpu.ze-runtime 'state))]]]
    (let [kernel (retained-capacity-artifact [required])
          owned (fn [count]
                  (assoc (make-buffer (java.lang.foreign.MemorySegment/ofArray (float-array count)) count)
                         ::cleanup/owner (cleanup/owner []) context-key (:context @runtime-state)))
          call (kcall/make kernel [(owned (dec required)) (owned 1) {:type :int :value 1}])
          native-calls (atom 0)
          prior @registry]
      (with-redefs [ocl/ensure-kernel-loaded! (fn [& _] (swap! native-calls inc) (throw (ex-info "unexpected native loading" {})))
                    ze/ensure-kernel-loaded! (fn [& _] (swap! native-calls inc) (throw (ex-info "unexpected native loading" {})))]
        (try
          (reset! registry (assoc prior (:kernel-name kernel) kernel))
          (try
            (bind call)
            (is false "the resident binder must check retained input storage before native loading")
            (catch clojure.lang.ExceptionInfo e
              (is (= :kernel-body-buffer-capacity (:reason (ex-data e))))
              (is (= required (:required-elements (ex-data e))))
              (is (= (dec required) (:buffer-elements (ex-data e))))))
          (is (zero? @native-calls))
          (finally (reset! registry prior)))))))

(deftest retained-pitched-storage-preserves-zero-and-checked-stride-arithmetic
  (let [make-kernel #(retained-capacity-artifact %1 (assoc (layout/row-major %1 :float) :strides %2))
        arguments [(float-array 0) (float-array 1) {:type :int :value 0}]
        empty-kernel (make-kernel ['n 3] [5 1])]
    (is (identical? empty-kernel
                    (kcall/validate-retained-input-capacities! empty-kernel arguments host-capacity)))
    (is (thrown? ArithmeticException
                 (kcall/validate-retained-input-capacities!
                  (make-kernel [3 3] [Long/MAX_VALUE 1]) arguments host-capacity)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"strides must be non-negative"
                         (kcall/validate-retained-input-capacities!
                          (make-kernel [2 3] [-1 1]) arguments host-capacity)))))

(deftest retained-storage-capacity-preserves-packed-byte-units-and-known-ze-ranges
  (let [kernel (-> (retained-capacity-artifact [5])
                   (assoc :source "__kernel void retained_capacity_test(__global const int* x, __global float* out, int n) {}")
                   (assoc-in [:abi 0] (kabi/slot 'x :input :byte :kernel-dtype :int :role :operand))
                   (assoc-in [:attributes :kernel-body :parameters 0 :dtype] :byte)
                   (assoc-in [:attributes :kernel-body :parameters 0 :layout] (layout/row-major [5] :byte)))
        arguments [(byte-array 5) (float-array 1) {:type :int :value 1}]]
    (is (identical? kernel (kcall/validate-retained-input-capacities! kernel arguments host-capacity)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"smaller than"
                         (kcall/validate-retained-input-capacities!
                          kernel (assoc arguments 0 (byte-array 4)) host-capacity))))
  (let [kernel (retained-capacity-artifact [5])
        call (kcall/make kernel [(java.lang.foreign.MemorySegment/ofArray (float-array 4))
                                (java.lang.foreign.MemorySegment/ofArray (float-array 1))
                                {:type :int :value 1}])
        prior @ze/kernel-registry
        native-calls (atom 0)]
    (with-redefs [ze/ensure-kernel-loaded! (fn [& _] (swap! native-calls inc) (throw (ex-info "unexpected native loading" {})))]
      (try
        (reset! ze/kernel-registry (assoc prior (:kernel-name kernel) kernel))
        (try
          (ze/bind-kernel-call call)
          (is false "a known ZE memory range must retain its physical byte-derived capacity")
          (catch clojure.lang.ExceptionInfo e
            (is (= :kernel-body-buffer-capacity (:reason (ex-data e))))
            (is (= 5 (:required-elements (ex-data e))))
            (is (= 4 (:buffer-elements (ex-data e))))))
        (is (zero? @native-calls))
        (finally (reset! ze/kernel-registry prior))))))

(deftest plain-direct-map-admission-does-not-acquire-retained-body-authority
  (doseq [[registry invoke] [[ocl/kernel-registry ocl/invoke-registered-map-void-kernel]
                             [ze/kernel-registry ze/invoke-registered-map-void-kernel]]]
    (let [name "plain_capacity_boundary" prior @registry
          load-failure (ex-info "loader boundary reached" {})
          loads (atom 0)
          loader (fn [& _] (swap! loads inc) (throw load-failure))]
      (with-redefs [ocl/ensure-kernel-loaded! loader ze/ensure-kernel-loaded! loader]
        (try
          (reset! registry (assoc prior name {:kernel-name name :workgroup-size 4 :dtype :float}))
          (try (invoke name [(float-array 1)] [] 1)
               (is false "the existing plain positive route must reach its normal loader")
               (catch Throwable e (is (identical? load-failure e))))
          (is (= 1 @loads))
          (doseq [bound [-1 1.5]]
            (is (thrown? clojure.lang.ExceptionInfo (invoke name [(float-array 1)] [] bound))))
          (is (= 1 @loads) "existing plain bound validation still precedes driver contact")
          (finally (reset! registry prior)))))))

(deftest retained-input-check-preserves-realized-output-authority-and-inout-reads
  (let [launch (klaunch/spec {:workgroup-size [4] :group-count [8]})
        kernel (-> (retained-capacity-artifact [5])
                   (assoc :launch launch :effects {:kind :scalar-reduction-phase})
                   (assoc-in [:attributes :kernel-body :launch] launch)
                   (assoc-in [:attributes :kernel-body :parameters 1 :shape] [8])
                   (assoc-in [:attributes :kernel-body :parameters 1 :layout] (layout/row-major [8] :float)))
        arguments [(float-array 5) (float-array 1) {:type :int :value 1}]
        call (kcall/make kernel arguments {:group-count [1]})
        plan (kcall/binding-plan call)]
    (is (= [1] (:group-count plan)))
    (is (identical? plan (kcall/validate-resident-output-capacities! call plan kernel host-capacity))
        "the actual one-group phase requires one output, not eight default-group outputs")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"smaller than its retained"
                         (kcall/validate-resident-output-capacities!
                          call (assoc plan :arguments (assoc arguments 0 (float-array 4))) kernel host-capacity)))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"scheduled group count"
                         (kcall/validate-resident-output-capacities!
                          call (assoc plan :pointer-pairs [[(second (:abi kernel)) (float-array 0)]])
                          kernel host-capacity))))
  (let [kernel (-> (retained-capacity-artifact [1])
                   (assoc :source "__kernel void retained_capacity_test(__global const float* x, __global float* out, int n) {}")
                   (assoc-in [:abi 1] (kabi/slot 'out :inout :float :role :result))
                   (assoc-in [:attributes :kernel-body :parameters 1 :kind] :inout)
                   (assoc-in [:attributes :kernel-body :parameters 1 :shape] [5])
                   (assoc-in [:attributes :kernel-body :parameters 1 :layout] (layout/row-major [5] :float)))]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"smaller than its retained"
                         (kcall/validate-retained-input-capacities!
                          kernel [(float-array 1) (float-array 4) {:type :int :value 1}] host-capacity)))))

(deftest raw-ze-contraction-preserves-shared-output-capacity-for-both-effect-tags
  (doseq [kind [:pure-contraction :tensor-contraction]]
    (let [kernel (-> (retained-capacity-artifact [2])
                     (assoc :effects {:kind kind})
                     (assoc-in [:attributes :out-elems] 2))
          owned (fn [count]
                  (assoc (ze/->DeviceBuffer (java.lang.foreign.MemorySegment/ofArray (float-array count))
                                            count (* count 4) :float)
                         ::cleanup/owner (cleanup/owner [])
                         :raster.gpu.ze-runtime/allocation-context
                         (:context @(var-get (ns-resolve 'raster.gpu.ze-runtime 'state)))))
          input (owned 2) prior @ze/kernel-registry
          loads (atom 0) loader-failure (ex-info "native loading boundary" {})]
      (with-redefs [ze/ensure-kernel-loaded! (fn [& _] (swap! loads inc) (throw loader-failure))]
        (try
          (reset! ze/kernel-registry (assoc prior (:kernel-name kernel) kernel))
          (try
            (ze/invoke-registered-contraction! (:kernel-name kernel) [input (owned 1) 1])
            (is false "both emitted contraction tags must reject known undersized outputs")
            (catch clojure.lang.ExceptionInfo e
              (is (= 2 (:out-elems (ex-data e))))
              (is (= 1 (:buffer-elements (ex-data e))))))
          (is (zero? @loads))
          (try
            (ze/invoke-registered-contraction! (:kernel-name kernel) [input (owned 2) 1])
            (is false "sufficient known input/output storage must reach the existing loader")
            (catch Throwable e (is (identical? loader-failure e))))
          (is (= 1 @loads))
          (finally (reset! ze/kernel-registry prior)))))))

(deftest checked-preconditions-do-not-repeat-synchronous-artifact-validation
  (let [call (kcall/make artifact args)
        validations (atom 0)
        scalars (atom 0)
        validate-artifact kart/validate!
        validate-scalar kcall/validate-scalar-value!]
    (with-redefs [kart/validate! (fn [value] (swap! validations inc) (validate-artifact value))
                  kcall/validate-scalar-value! (fn [slot value]
                                               (swap! scalars inc)
                                               (validate-scalar slot value))]
      (doseq [check [#(kcall/validate! call)
                     #(kcall/realize-launch artifact args)
                     #(kcall/validate-preconditions! artifact args)
                     #(kcall/validate! call)]]
        (reset! validations 0)
        (reset! scalars 0)
        (check)
        (is (= 1 @validations) "each public entry starts a fresh artifact check")
        (is (= 2 @scalars) "each physical scalar is still checked once"))
      (reset! validations 0)
      (reset! scalars 0)
      (kcall/make artifact args)
      (is (= 2 @validations) "construction retains independent final call validation")
      (is (= 4 @scalars))
      (doseq [bad [(pop args) (seq args)]
              check [#(kcall/realize-launch artifact bad)
                     #(kcall/validate-preconditions! artifact bad)
                     #(kcall/validate! (assoc call :arguments bad))]]
        (is (thrown? clojure.lang.ExceptionInfo (check))))
      (is (thrown? clojure.lang.ExceptionInfo
                   (kcall/validate-preconditions! (assoc artifact :abi []) args)))
      (is (thrown? clojure.lang.ExceptionInfo
                   (kcall/validate! (assoc call :arguments (assoc args 3 {:type :int :value 1.5}))))))))

(deftest scalar-preconditions-cannot-be-bypassed-by-direct-calls-or-launch-overrides
  (let [constrained (assoc artifact :preconditions [{:expression 'n :op :>= :value 64}
                                                   {:expression 'n :op :<= :value 1024}])
        bad (assoc args 3 {:type :int :value 32})
        raw (assoc (kcall/make artifact bad) :artifact constrained)]
    (is (kcall/kernel-call? (kcall/make constrained args)))
    (is (true? (kcall/validate-preconditions! constrained (assoc args 0 nil 1 nil)))
        "scalar preflight does not need allocated pointers")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"differs from the registered artifact"
                          (kcall/validate-registered! (kcall/make constrained args) artifact)))
    (doseq [call [#(kcall/make constrained bad)
                  #(kcall/make constrained bad {:group-count [1]})
                  #(kcall/realize-launch constrained bad)
                  #(kcall/validate! raw)]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"scalar precondition failed" (call))))
    (is (not= (:source-hash (tuning/executable-signature artifact))
              (:source-hash (tuning/executable-signature constrained)))
        "constraints participate in tuning identity")))

(deftest preconditions-use-only-integral-scalar-abi-values
  (doseq [expression ['missing 'x 'scale]]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"outside the integral scalar ABI"
                          (kart/validate! (assoc artifact :preconditions
                                                 [{:expression expression :op :>= :value 1}])))))
  (is (thrown-with-msg? clojure.lang.ExceptionInfo #"checked integer comparison"
                        (kart/validate! (assoc artifact :preconditions
                                               [{:expression 'n :op :eval :value 1}])))))

(deftest preconditions-preserve-checked-arithmetic-and-order
  (let [overflow (klaunch/product Long/MAX_VALUE 2)
        guarded (assoc artifact :preconditions [{:expression 'n :op :>= :value 64}
                                                {:expression overflow :op :>= :value 0}])]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"scalar precondition failed"
                          (kcall/make guarded (assoc args 3 {:type :int :value 32}))))
    (is (thrown? ArithmeticException (kcall/make guarded args)))))

(deftest preconditions-cannot-hide-invalid-long-values-or-literal-specializations
  (let [wide (kart/make {:kernel-name "wide_guard" :source "__kernel void wide_guard(__global float* out, long n) {}"
                         :abi [(kabi/slot 'out :output :float) (kabi/slot 'n :scalar :long)]
                         :arguments ['out 'n]
                         :launch (klaunch/spec {:workgroup-size [1] :group-count [1]})
                         :preconditions [{:expression 'n :op :>= :value 0}]})]
    (doseq [value [1.5 (inc' Long/MAX_VALUE) (dec' Long/MIN_VALUE)]]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"physical ABI range"
                            (kcall/make wide [:out {:type :long :value value}]))))
    (let [literal (assoc wide :arguments ['out 64] :preconditions [{:expression 64 :op :>= :value 64}])]
      (is (kcall/kernel-call? (kcall/make literal [:out {:type :long :value 64}])))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"literal specialization"
                            (kcall/make literal [:out {:type :long :value 32}]))))))

(deftest scheduled-call-overrides-cannot-overflow-hardware-indices
  (let [kernel (body/make
                {:id :range-check
                 :parameters [(body/->KernelParameter 'n :scalar :int [] nil nil :bound)]
                 :indices [(body/->IndexBinding 'group :group 0)]
                 :launch (:launch artifact)})
        scheduled-artifact (assoc-in artifact [:attributes :kernel-body] kernel)]
    (is (kcall/kernel-call? (kcall/make scheduled-artifact args)))
    (is (kcall/kernel-call?
         (kcall/make scheduled-artifact args {:group-count [(inc (long Integer/MAX_VALUE))]})))
    (try
      (kcall/make scheduled-artifact args {:group-count [(+ 2 (long Integer/MAX_VALUE))]})
      (is false "a scheduling override must not bypass the body's physical index width")
      (catch clojure.lang.ExceptionInfo e
        (is (= :kernel-launch-index-range (:reason (ex-data e))))))
    (let [large-launch (klaunch/spec {:workgroup-size [256]
                                      :group-count [(klaunch/product 'n 10000000)]})
          large-artifact (-> scheduled-artifact
                             (assoc :launch large-launch)
                             (assoc-in [:attributes :kernel-body :launch] large-launch))]
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"hardware index representation"
                           (kcall/realize-launch large-artifact args))
          "compatibility staging checks before allocating partial storage")
      (is (kcall/kernel-call? (kcall/make large-artifact args {:group-count [1]}))
          "a legal explicit override is checked instead of the unused default geometry"))))

(def ^:private stable-artifact
  (kart/make (assoc-in artifact [:abi 0 :aliasing] :no-write-alias)))

(deftest calls-enforce-stable-input-ranges-without-forbidding-in-place-by-default
  (let [allocation (bview/allocation {:id :call-alias :byte-size 64 :memory-space :device
                                      :ownership :borrowed})
        left (bview/view allocation {:id :left :dtype :float :shape [8]})
        overlap (bview/view allocation {:id :overlap :byte-offset 16
                                        :dtype :float :shape [8]})
        right (bview/view allocation {:id :right :byte-offset 32
                                      :dtype :float :shape [8]})
        scalars [{:type :float :value 1.0} {:type :int :value 8}]]
    (is (kcall/kernel-call? (kcall/make stable-artifact (into [left right] scalars))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"stable input overlaps"
                          (kcall/make stable-artifact (into [left overlap] scalars))))
    (is (kcall/kernel-call? (kcall/make artifact (into [left left] scalars)))))
  (testing "raw resident segments are checked by byte range"
    (let [segment (java.lang.foreign.MemorySegment/ofArray (float-array 16))
          left (.asSlice segment 0 32)
          overlap (.asSlice segment 16 32)
          right (.asSlice segment 32 32)
          scalars [{:type :float :value 1.0} {:type :int :value 8}]]
      (is (kcall/kernel-call? (kcall/make stable-artifact (into [left right] scalars))))
      (is (thrown-with-msg? clojure.lang.ExceptionInfo #"stable input overlaps"
                            (kcall/make stable-artifact
                                        (into [left overlap] scalars)))))))

(deftest calls-enforce-pointer-alignment-carried-by-the-abi
  (let [aligned-artifact (kart/make (assoc-in artifact [:abi 0 :alignment] 16))
        allocation (bview/allocation {:id :call-alignment :byte-size 64
                                      :memory-space :device :alignment 16
                                      :ownership :borrowed})
        aligned (bview/view allocation {:id :aligned :byte-offset 0
                                        :dtype :float :shape [4]})
        misaligned (bview/view allocation {:id :misaligned :byte-offset 4
                                           :dtype :float :shape [4]})
        suffix [:resident-out {:type :float :value 1.0} {:type :int :value 4}]]
    (is (kcall/kernel-call? (kcall/make aligned-artifact (into [aligned] suffix))))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"violates its ABI alignment contract"
         (kcall/make aligned-artifact (into [misaligned] suffix))))))

(deftest call-realizes-artifact-launch-from-ordered-values
  (let [call (kcall/make artifact args)
        plan (kcall/binding-plan call)]
    (is (kcall/kernel-call? call))
    (is (= [256] (:workgroup-size plan)))
    (is (= [3] (:group-count plan)))
    (is (= [:resident-x :resident-out] (mapv second (:pointer-pairs plan))))
    (is (= [:float :int] (mapv (comp :type second) (:scalar-pairs plan))))))

(deftest resident-output-capacity-rules-are-shared-before-driver-binding
  (let [call (kcall/make artifact args)
        plan (kcall/binding-plan call)
        capacity (fn [elements]
                   (fn [value _] (when (= :resident-out value) elements)))
        reduction (assoc artifact :effects {:kind :scalar-reduction-phase})
        contraction (-> artifact
                        (assoc :effects {:kind :tensor-contraction})
                        (assoc-in [:attributes :out-elems] 4))]
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"reduction result buffer"
                          (kcall/validate-resident-output-capacities!
                           call plan reduction (capacity 2))))
    (is (= plan (kcall/validate-resident-output-capacities!
                 call plan reduction (capacity 3))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"contraction output buffer"
                          (kcall/validate-resident-output-capacities!
                           call plan contraction (capacity 3))))
    (is (= plan (kcall/validate-resident-output-capacities!
                 call plan contraction (capacity 4))))
    (is (= plan (kcall/validate-resident-output-capacities!
                 call plan contraction (capacity nil)))
        "opaque OpenCL handles do not claim a physical allocation capacity")
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"output extent must be non-negative"
                          (kcall/validate-resident-output-capacities!
                           call plan (assoc-in contraction [:attributes :out-elems] -1)
                           (capacity 100))))))

(deftest extent-bounds-reject-negative-values-before-launch
  (let [arguments [:resident-x :resident-out
                   {:type :float :value 2.0} {:type :int :value -1}]]
    (try
      (kcall/make artifact arguments)
      (is false "a negative semantic extent must not reach launch realization")
      (catch clojure.lang.ExceptionInfo exception
        (is (= :kernel-bound-range (:reason (ex-data exception))))))))

(deftest staged-scalar-normalization-cannot-truncate-an-extent
  (doseq [[value reason] [[-1 :kernel-bound-range]
                          [-0.5 :kernel-executable-integral-scalar]
                          [3.5 :kernel-executable-integral-scalar]]]
    (try
      (kexec/typed-runtime-arguments
       artifact [:resident-x :resident-out 2.0 value])
      (is false "an extent must remain an exact non-negative integer")
      (catch clojure.lang.ExceptionInfo exception
        (is (= reason (:reason (ex-data exception))))))))

(deftest scheduling-may-change-groups-but-not-emitted-workgroup
  (let [call (kcall/make artifact args {:group-count [1]})]
    (is (= [1] (get-in call [:geometry :group-count]))))
  (is (thrown-with-msg?
       clojure.lang.ExceptionInfo #"workgroup differs"
       (kcall/validate!
        (kcall/->KernelCall artifact args
                            (klaunch/geometry {:workgroup-size [128] :group-count [5]}))))))

(deftest calls-preserve-one-to-three-dimensional-geometry
  (doseq [dimensions (range 1 4)]
    (let [workgroup (vec (repeat dimensions 4))
          groups (vec (repeat dimensions 2))
          nd-artifact (kart/make (assoc artifact :launch
                                        (klaunch/spec {:workgroup-size workgroup
                                                       :group-count groups})))
          call (kcall/make nd-artifact args)]
      (is (= dimensions (klaunch/dimensions (:geometry call))))
      (is (= workgroup (get-in call [:geometry :workgroup-size])))
      (is (= groups (get-in call [:geometry :group-count]))))))

(deftest launch-realization-preserves-the-artifact-occupancy-cap-at-max-int
  (let [capped
        (kart/make
         {:kernel-name "capped_reduction_launch_test"
          :source "__kernel void capped_reduction_launch_test(__global float* out, int n) {}"
          :abi [(kabi/slot 'out :output :float :role :result)
                (kabi/slot 'n :scalar :int :role :bound)]
          :arguments '[out n]
          :launch (klaunch/spec
                   {:workgroup-size [256]
                    :group-count [(klaunch/minimum
                                   17 (klaunch/ceil-div 'n 256))]})})
        geometry (kcall/realize-launch
                  capped [:resident-out {:type :int :value Integer/MAX_VALUE}])]
    (is (= [256] (:workgroup-size geometry)))
    (is (= [17] (:group-count geometry))
        "staging must not reconstruct an uncapped ceil-div launch")))

(deftest opencl-raw-handles-do-not-claim-device-allocation-capacity
  (let [capacity (ns-resolve 'raster.gpu.ocl-runtime 'known-buffer-capacity)
        handle (java.lang.foreign.MemorySegment/ofArray (long-array 1))
        owned (ocl/->OclBuffer handle handle 1024 4096 :float 64)]
    (is (nil? (capacity handle))
        "a MemorySegment accepted as cl_mem describes the handle, not the allocation")
    (is (= 1024 (capacity owned)))))

(deftest call-rejects-untyped-or-mistyped-scalars
  (testing "runtime scalar typing is part of the call, not a backend guess"
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"typed value"
                          (kcall/make artifact
                                      [:x :out 2.0 {:type :int :value 513}])))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"wrong kernel dtype"
                          (kcall/make artifact
                                      [:x :out {:type :double :value 2.0}
                                       {:type :int :value 513}])))))

(deftest registered-artifact-check-ignores-runtime-cache-fields
  (let [call (kcall/make artifact args)]
    (is (= :native-handle
           (:module (kcall/validate-registered! call
                                                (assoc artifact :module :native-handle)))))
    (is (thrown-with-msg? clojure.lang.ExceptionInfo #"differs from"
                          (kcall/validate-registered!
                           call (assoc artifact :source
                                       "__kernel void kernel_call_contract_test(__global const float* x, __global float* out, float scale, int n) { out[0] = 0; }"))))))

(deftest logical-soa-plan-expands-once-before-physical-call
  (let [soa-artifact
        (kart/make
         {:kernel-name "logical_soa_contract_test"
          :source (str "__kernel void logical_soa_contract_test("
                       "__global const float* particles_x, "
                       "__global const int* particles_id, __global float* out, int n) {}")
          :abi [(kabi/slot 'particles_x :input :float :binding 'particles :field :x
                           :role :operand)
                (kabi/slot 'particles_id :input :int :binding 'particles :field :id
                           :role :operand)
                (kabi/slot 'out :output :float :role :effect)
                (kabi/slot 'n :scalar :int :role :bound)]
          :arguments '[particles_x particles_id out n]
          :launch (klaunch/spec {:workgroup-size [64]
                                 :group-count [(klaunch/ceil-div 'n 64)]})})
        plan (kcall/logical-argument-plan soa-artifact)
        physical (kcall/expand-logical-arguments
                  soa-artifact
                  [:resident-particles :resident-out {:type :int :value 65}]
                  (fn [{:keys [binding]} value]
                    (if (= binding 'particles)
                      [(keyword (str (name value) "-x"))
                       (keyword (str (name value) "-id"))]
                      [value])))
        call (kcall/make soa-artifact physical)]
    (is (= '[particles out n] (kcall/logical-arguments soa-artifact)))
    (is (= [2 1 1] (mapv (comp count :slots) plan)))
    (is (= [:resident-particles-x :resident-particles-id :resident-out
            {:type :int :value 65}]
           physical))
    (is (= [2] (get-in call [:geometry :group-count])))
    (let [resident (resident-value/composite
                    :particles
                    [{:name :x :value {:dtype :float :resident :x}}
                     {:name :id :value {:dtype :int :resident :id}}])]
      (is (= [{:dtype :float :resident :x} {:dtype :int :resident :id}]
             (ze/expand-pointer-binding (first plan) resident)))
      (is (= [{:dtype :float :resident :x} {:dtype :int :resident :id}]
             (ocl/expand-pointer-binding (first plan) resident))))
    (let [reordered (resident-value/composite
                     :particles
                     [{:name :id :value {:dtype :int :resident :id}}
                      {:name :x :value {:dtype :float :resident :x}}])]
      (doseq [expand [ze/expand-pointer-binding ocl/expand-pointer-binding]]
        (is (= [{:dtype :float :resident :x} {:dtype :int :resident :id}]
               (expand (first plan) reordered))
            "explicit fields project storage order independently of physical ABI order")))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"differs from physical ABI"
         (kcall/expand-logical-arguments soa-artifact
                                         [:particles :out {:type :int :value 1}]
                                         (fn [_ value] [value]))))))

(deftest logical-composite-may-project-a-field-subset-per-kernel
  (let [slot (kabi/slot 'particles_x :input :float
                        :binding 'particles :field :x :role :operand)
        group {:binding 'particles :slots [slot]}
        resident (resident-value/composite
                  :particles
                  [{:name :x :value {:dtype :float :resident :x}}
                   {:name :id :value {:dtype :int :resident :id}}])]
    (is (= [{:dtype :float :resident :x}]
           (ze/expand-pointer-binding group resident)))
    (is (= [{:dtype :float :resident :x}]
           (ocl/expand-pointer-binding group resident)))
    (doseq [expand [ze/expand-pointer-binding ocl/expand-pointer-binding]]
      (is (= [{:dtype :float :resident :x}]
             (expand group (update resident :fields #(vec (reverse %)))))
          "subset projection is independent of composite field storage order"))))

(deftest both-resident-backends-consume-the-same-call-contract
  (let [call (kcall/make artifact args)]
    (doseq [[register! bind!] [[ze/register-kernel! ze/bind-kernel-call]
                               [ocl/register-kernel! ocl/bind-kernel-call]]]
      (register! "kernel_call_contract_test" artifact)
      (try
        ;; Keywords are valid opaque values to neutral KernelCall validation, but not backend
        ;; resident buffers. Both binders must reject at the identical post-call boundary before
        ;; either native driver is initialized.
        (is (thrown-with-msg? clojure.lang.ExceptionInfo #"KernelCall requires"
                              (bind! call)))
        (finally
          (register! "kernel_call_contract_test" {:test-tombstone? true}))))))
