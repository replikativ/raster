(ns raster.compiler.ir.kernel-call-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.buffer-view :as bview]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.gpu.ocl-runtime :as ocl]
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
          :source "__kernel void capped_reduction_launch_test(int n) {}"
          :abi [(kabi/slot 'n :scalar :int :role :bound)]
          :arguments '[n]
          :launch (klaunch/spec
                   {:workgroup-size [256]
                    :group-count [(klaunch/minimum
                                   17 (klaunch/ceil-div 'n 256))]})})
        geometry (kcall/realize-launch
                  capped [{:type :int :value Integer/MAX_VALUE}])]
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
    (is (= [:seg-x :seg-id]
           (ze/expand-pointer-binding
            (first plan)
            (ze/->GpuSoA 'Particle 'ParticleSoA 65
                         [{:name "x" :dtype :float :seg :seg-x}
                          {:name "id" :dtype :int :seg :seg-id}]))))
    (let [resident (resident-value/composite
                    :particles
                    [{:name :x :value {:dtype :float :resident :x}}
                     {:name :id :value {:dtype :int :resident :id}}])]
      (is (= [{:dtype :float :resident :x} {:dtype :int :resident :id}]
             (ze/expand-pointer-binding (first plan) resident)))
      (is (= [{:dtype :float :resident :x} {:dtype :int :resident :id}]
             (ocl/expand-pointer-binding (first plan) resident))))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"field order/name"
         (ze/expand-pointer-binding
          (first plan)
          (ze/->GpuSoA 'Particle 'ParticleSoA 65
                       [{:name "id" :dtype :float :seg :seg-id}
                        {:name "x" :dtype :int :seg :seg-x}]))))
    (is (thrown-with-msg?
         clojure.lang.ExceptionInfo #"differs from physical ABI"
         (kcall/expand-logical-arguments soa-artifact
                                         [:particles :out {:type :int :value 1}]
                                         (fn [_ value] [value]))))))

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
