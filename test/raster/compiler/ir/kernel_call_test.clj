(ns raster.compiler.ir.kernel-call-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.gpu.ocl-runtime :as ocl]
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
          :abi [(kabi/slot 'particles_x :input :float :binding 'particles :role :operand)
                (kabi/slot 'particles_id :input :int :binding 'particles :role :operand)
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
