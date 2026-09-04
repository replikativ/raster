(ns raster.compiler.ir.resident-plan-test
  (:require [clojure.test :refer [deftest is testing]]
            [raster.arrays :as arrays]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.ir.kernel-abi :as kabi]
            [raster.compiler.ir.kernel-artifact :as artifact]
            [raster.compiler.ir.kernel-launch :as launch]
            [raster.compiler.ir.link-plan :as link]
            [raster.compiler.ir.resident-plan :as resident]
            [raster.core :refer [deftm]]
            [raster.dl.nn :as nn]))

(deftm production-map
  [x :- (Array float) out :- (Array float) scale :- Float n :- Long] :- (Array float)
  (raster.par/map! out i n float (* scale (arrays/aget x i))))

(def ^:private kernel
  (artifact/make
   {:kernel-name "resident_axpy"
    :source "__kernel void resident_axpy(const float* x, const float* w, float* y, long n) {}"
    :abi [(kabi/slot 'x :input :float)
          (kabi/slot 'w :input :float)
          (kabi/slot 'y :output :float)
          (kabi/slot 'n :scalar :long)]
    :arguments '[x w y n]
    :launch (launch/spec {:workgroup-size [64]
                          :group-count [(launch/ceil-div 'n 64)]})
    :effects {:kind :map :reads '[x w] :writes '[y]}}))

(defn- descriptor []
  {:dtype :float
   :all-params '[x w n]
   :array-params '[x w]
   :scalar-params '[n]
   :array-roles {'x :input 'w :input}
   :allocs [{:sym 'y :dtype :float :size-fn (fn [args] (long (nth args 2)))}]
   :steps [{:phase :map :convention :map :artifact kernel
            :argument-specs [{:kind :input :sym 'x}
                             {:kind :input :sym 'w}
                             {:kind :output :sym 'y}
                             {:kind :scalar :type :long
                              :value-fn (fn [args] (long (nth args 2)))}]}]
   :result-sym 'y})

(defn- reason-of [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo error
      (:reason (ex-data error)))))

(deftest resident-descriptor-lowers-once-to-a-certified-plan
  (let [x (float-array 32)
        w (float-array 32)
        lowering (resident/lower
                  {:id :axpy :target :ze:0 :descriptor (descriptor) :arguments [x w 32]
                   :roles {'w :constant} :shapes {'x [4 8] 'w [4 8] 'y [4 8]}
                   :node-ids {'x :x 'w :weight 'y :result}})
        plan (:plan lowering)
        certificate (:certificate lowering)]
    (is (resident/certified-plan? lowering))
    (is (identical? lowering (resident/verify! lowering)))
    (is (link/link-plan? plan))
    (is (= '[x w n] (:parameter-order certificate)) "the source ABI order is witnessed")
    (is (= :ze:0 (:target certificate)))
    (is (= {'x :x 'w :weight 'y :result} (:bindings certificate)))
    (is (= {'n 32} (:scalars certificate)))
    (is (= {'x :input 'w :constant 'y :output} (:roles certificate)))
    (is (= [:result] (:outputs plan)))
    (is (= [4 8] (get-in plan [:nodes :x :view :shape])))
    (is (identical? x (get-in plan [:nodes :x :source])))
    (is (= :allocation (get-in certificate [:values 'y :origin])))))

(deftest external-ownership-is-not-mistaken-for-host-initialization
  (let [lowering (resident/lower
                  {:id :external :target :ze:0 :descriptor (descriptor)
                   :arguments [(float-array 8) (float-array 8) 8]
                   :ownership {'x :external}})
        x-node (get-in lowering [:plan :nodes [:external 'x]])]
    (is (= :external (get-in x-node [:view :allocation :ownership])))
    (is (nil? (:source x-node)))
    (is (= :input (:role x-node)))
    (is (resident/certified-plan? lowering))))

(deftest production-compiler-descriptor-crosses-the-same-certified-boundary
  (let [n 64
        descriptor (pipeline/compile-gpu-program #'production-map :ze:0
                                                 :dtype :float :on-non-resident :throw)
        lowering (resident/lower
                  {:id :production-map :target :ze:0 :descriptor descriptor
                   :arguments [(float-array n) (float-array n) (float 2.0) n]})]
    (is (resident/certified-plan? lowering))
    (is (= (:all-params descriptor)
           (get-in lowering [:certificate :parameter-order])))
    (is (= (set (link/descriptor-pointer-symbols descriptor))
           (set (keys (get-in lowering [:certificate :bindings])))))
    (is (= [(get-in lowering [:certificate :bindings (:result-sym descriptor)])]
           (get-in lowering [:plan :outputs])))))

(deftest array-dependent-specialization-closures-survive-link-lowering
  (let [x (float-array 12)
        w (float-array 12)
        descriptor (assoc-in (descriptor) [:allocs 0 :size-fn]
                             (fn [arguments]
                               (java.lang.reflect.Array/getLength (nth arguments 0))))
        lowering (resident/lower
                  {:id :array-shaped :target :ze:0 :descriptor descriptor
                   :arguments [x w 12]})
        instance (first (get-in lowering [:plan :instances]))]
    (is (identical? x (first (link/instance-arguments instance))))
    (is (= [12] (get-in lowering [:plan :nodes [:array-shaped 'y] :view :shape])))
    (is (resident/certified-plan? lowering))))

(deftest generated-gemm-descriptor-crosses-the-same-certified-boundary
  (let [rows 4
        width 16
        descriptor (pipeline/compile-gpu-program #'nn/linear-nb :ze:0
                                                 :dtype :float :on-non-resident :throw)
        lowering (resident/lower
                  {:id :production-gemm :target :ze:0 :descriptor descriptor
                   :arguments [(float-array (* rows width))
                               (float-array (* width width)) rows width width]
                   :shapes {'x [rows width] 'W [width width]
                            (:result-sym descriptor) [rows width]}})]
    ;; linear-nb's BLAS GEMM is a typed contraction dispatch, not a resident :gemm binding.
    ;; The executable container keeps its semantic extents public for runtime schedule choice.
    (is (= [:executable] (mapv :convention (:steps descriptor))))
    (is (resident/certified-plan? lowering))
    (is (= [rows width]
           (get-in lowering [:certificate :values (:result-sym descriptor) :shape])))
    (is (= (:schedule descriptor) (get-in lowering [:certificate :schedule])))))

(deftest the-certificate-detects-lowering-drift
  (let [lowering (resident/lower
                  {:id :axpy :target :ze:0 :descriptor (descriptor)
                   :arguments [(float-array 8) (float-array 8) 8]})]
    (testing "a modified witness is rejected"
      (is (= :resident-plan-certificate
             (reason-of #(resident/verify!
                          (assoc-in lowering [:certificate :scalars 'n] 9))))))
    (testing "a modified target still passes its own structural checks but not the source witness"
      (is (= :resident-plan-certificate
             (reason-of #(resident/verify!
                          (assoc-in lowering [:plan :id] :changed-plan))))))))

(deftest conversion-fails-closed-before-runtime-contact
  (testing "arguments must exactly follow the descriptor ABI"
    (is (= :resident-plan-arguments
           (reason-of #(resident/lower
                        {:id :bad :target :ze:0 :descriptor (descriptor)
                         :arguments [(float-array 8) (float-array 8)]})))))
  (testing "logical shapes must preserve the realized storage extent"
    (is (= :resident-plan-shape
           (reason-of #(resident/lower
                        {:id :bad :target :ze:0 :descriptor (descriptor)
                         :arguments [(float-array 8) (float-array 8) 8]
                         :shapes {'x [3 3]}})))))
  (testing "every executable pointer requires exactly one parameter/allocation contract"
    (let [bad (update (descriptor) :allocs empty)]
      (is (= :resident-plan-pointer-contract
             (reason-of #(resident/lower
                          {:id :bad :target :ze:0 :descriptor bad
                           :arguments [(float-array 8) (float-array 8) 8]}))))))
  (testing "stable identities cannot accidentally alias two semantic values"
    (is (= :resident-plan-node-ids
           (reason-of #(resident/lower
                        {:id :bad :target :ze:0 :descriptor (descriptor)
                         :arguments [(float-array 8) (float-array 8) 8]
                         :node-ids {'x :same 'w :same}}))))))
