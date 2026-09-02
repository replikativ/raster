(ns raster.compiler.contraction-marker-abi-test
  "The contraction emitter, route, marker, registry and runtime share one ordered typed ABI."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.arrays :as ra]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.backend.gpu.opencl-pass :as op]
            [raster.compiler.core.hardware :as hardware]
            [raster.compiler.ir.axis-map :as axis-map]
            [raster.compiler.ir.kernel-artifact :as kart]
            [raster.compiler.ir.kernel-call :as kcall]
            [raster.compiler.ir.kernel-dispatch :as kdispatch]
            [raster.compiler.ir.kernel-executable :as kexec]
            [raster.compiler.ir.kernel-graph :as kgraph]
            [raster.compiler.ir.kernel-graph-call :as kgcall]
            [raster.compiler.ir.kernel-launch :as klaunch]
            [raster.compiler.passes.parallel.contract-route :as route]
            [raster.core :refer [deftm]]
            [raster.gpu.ocl-runtime :as ocl]
            [raster.gpu.ze-runtime :as ze]))

(deftm resident-contract-descriptor-probe
  [A :- (Array float) B :- (Array float)] :- (Array float)
  (let [C (ra/alloc-like A 64)]
    (raster.par/contract C [[i 8] [j 8]] [[l 8]]
                         (clojure.core/*
                          (ra/aget A (clojure.core/+ (clojure.core/* i 8) l))
                          (ra/aget B (clojure.core/+ (clojure.core/* l 8) j))))
    C))

(defn- mm [m n k]
  (list 'raster.par/contract 'C [['i m] ['j n]] [['l k]]
        (list '* (list 'aget 'A (list '+ (list '* 'i k) 'l))
              (list 'aget 'B (list '+ (list '* 'l n) 'j)))))

(defn- bias-epilogue-contract []
  (let [contract (assoc (vec (mm 128 256 64)) 1 'out)]
    (apply list
           (concat contract
                   [:epilogue
                    {:acc 'acc
                     :expr '(raster.numeric/+ acc (clojure.core/aget bias j))
                     :operands [{:sym 'bias :dtype :float
                                 :map (axis-map/of-axes '[[j 256]])}]
                     :dtype :float}]))))

(defn- compile-form
  ([contract] (compile-form contract :half))
  ([contract dtype]
   (with-redefs [hardware/descriptor-for (constantly nil)]
     (op/opencl-pass (list 'let* ['result contract] 'result)
                     :device-id :ze:0 :dtype dtype :min-elements 0))))

(defn- thrown-data [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo e
      (ex-data e))))

(defn- probe-artifact
  [kernel-name source abi arguments]
  (kart/make {:kernel-name kernel-name
              :source source
              :abi abi
              :arguments arguments
              :launch (klaunch/spec {:workgroup-size [1 1] :group-count [1 1]})
              :effects {:kind :tensor-contraction}
              :attributes {:out-elems 1 :dtype :float :out-dtype :float}}))

(deftest epilogue-operands-are-carried-in-the-ordered-marker
  (let [contract (bias-epilogue-contract)
        compiled (compile-form contract)
        kernel (first (:kernels compiled))
        marker (-> compiled :form second second)]
    (is (= :dpas (kart/attribute kernel :strategy)))
    (is (= '[A B out M N K bias] (mapv :name (:abi kernel))))
    (is (= [:input :input :output :scalar :scalar :scalar :input]
           (mapv :kind (:abi kernel))))
    (testing "values appear in exactly the same order as ABI slots"
      (is (= '[A B out 128 256 64 bias] (nth marker 2))))))

(deftest codegen-only-routing-facts-remain-legal-and-observable
  (let [dp4a '(raster.par/contract C [[i 4] [j 4]] [[l 8]]
                                   (* (aget A (+ (* i 8) l))
                                      (aget B (+ (* j 8) l))))
        compiled (compile-form dp4a :byte)
        kernel (first (:kernels compiled))]
    (testing "tensorization and packing are already baked into source and need no extra launch slot"
      (is (= :dp4a (kart/attribute kernel :strategy)))
      (is (true? (kart/attribute kernel :tensorized)))
      (is (= :int8x4 (kart/attribute kernel :packed)))
      (is (= [:byte :byte] (mapv :dtype (take 2 (:abi kernel)))))
      (is (= [:int :int] (mapv :kernel-dtype (take 2 (:abi kernel))))))
    (testing "the ordinary two-input descriptor still emits the production marker"
      (is (= 'raster.gpu.ze-runtime/invoke-registered-contraction!
             (-> compiled :form second second first))))))

(deftest epilogue-scalars-occupy-their-real-signature-position
  (let [contract (concat (mm 128 256 64)
                         [:epilogue {:acc 'acc
                                     :expr '(raster.numeric/* acc alpha)
                                     :scalars [{:sym 'alpha :dtype :float}]}])
        compiled (compile-form contract)
        kernel (first (:kernels compiled))
        marker (-> compiled :form second second)]
    (is (= '[A B C M N K alpha] (mapv :name (:abi kernel))))
    (is (= '[A B C 128 256 64 alpha] (nth marker 2)))))

(deftest non-argument-pre-steps-remain-fail-loud
  (let [base {:strategy :probe
              :kernel-name "probe_contract"
              :source "__kernel void probe_contract(__global double* A, __global double* C) {}"
              :array-params '[A]
              :dtype :double :out-dtype :double :out-elems 1
              :wg [1 1] :grid [1 1] :scalar-args []}
        contract (mm 1 1 1)]
    (let [value [{:op :transpose}]
          data (with-redefs [route/route-contraction
                             (fn [& _] (assoc base :pre-steps value))]
                 (thrown-data #(compile-form contract)))]
      (is (= :raster/fatal (:reason data)))
      (is (= [:pre-steps] (:unsupported-fields data)))
      (is (= {:pre-steps value} (:unsupported-values data))))))

(deftest runtime-rejects-an-abi-value-count-mismatch-before-touching-the-driver
  (let [kernel-name (str "abi_count_probe_" (gensym))
        abi [{:name 'A :c-name "A" :kind :input :dtype :float :kernel-dtype :float}
             {:name 'C :c-name "C" :kind :output :dtype :float :kernel-dtype :float
              :role :result}]
        artifact (probe-artifact
                  kernel-name
                  (str "__kernel void " kernel-name
                       "(__global float* A, __global float* C) {}")
                  abi '[A C])
        _ (ze/register-kernel! kernel-name artifact)
        data (thrown-data #(ze/invoke-registered-contraction!
                            kernel-name [(float-array 1)]))]
    (is (= 2 (:expected data)))
    (is (= 1 (:actual data)))))

(deftest staged-artifact-launch-preserves-all-three-geometry-axes
  (let [seen (atom [])]
    (with-redefs [ze/bind-kernel! (fn [kernel workgroup arguments]
                                    (swap! seen conj [:bind kernel workgroup arguments])
                                    :bound)
                  ze/launch-bound! (fn [bound groups]
                                     (swap! seen conj [:launch bound groups]))]
      (ze/launch-geometry! :kernel [64 1 1] [2 3 4] [:a :b]))
    (is (= [[:bind :kernel [64 1 1] [:a :b]]
            [:launch :bound [2 3 4]]]
           @seen))))

(deftest staged-dispatch-selects-after-abi-scalars-are-concrete
  (let [abi [{:name 'A :c-name "A" :kind :input :dtype :float :kernel-dtype :float}
             {:name 'C :c-name "C" :kind :output :dtype :float :kernel-dtype :float
              :role :result}
             {:name 'width :c-name "width" :kind :scalar :dtype :long :kernel-dtype :long}]
        make-artifact
        (fn [kernel-name strategy]
          (kart/make
           {:kernel-name kernel-name
            :source (str "__kernel void " kernel-name
                         "(__global float* A, __global float* C, long width) {}")
            :abi abi
            :arguments '[A C width]
            :launch (klaunch/spec {:workgroup-size [1] :group-count [1]})
            :effects {:kind :probe}
            :attributes {:strategy strategy}}))
        reference (make-artifact "staged_dispatch_reference" :reference)
        subgroup (make-artifact "staged_dispatch_subgroup" :subgroup)
        dispatch (kdispatch/make
                  {:id "staged_dispatch_probe"
                   :alternatives [reference subgroup]
                   :default-strategy :reference
                   :selector {:kind :runtime-scalar-threshold
                              :argument 'width :threshold 256
                              :at-least :subgroup :otherwise :reference}})
        seen (atom [])]
    (doseq [artifact [reference subgroup]]
      (ze/register-kernel! (:kernel-name artifact) artifact))
    (ze/register-kernel-dispatch! dispatch)
    (with-redefs [ze/invoke-registered-contraction!
                  (fn [kernel-name arguments]
                    (swap! seen conj [kernel-name arguments]))]
      (ze/invoke-registered-contraction-dispatch!
       (:id dispatch) (:kernel-name reference) [:a :c 128])
      (ze/invoke-registered-contraction-dispatch!
       (:id dispatch) (:kernel-name reference) [:a :c 256]))
    (is (= [["staged_dispatch_reference" [:a :c 128]]
            ["staged_dispatch_subgroup" [:a :c 256]]]
           @seen))))

(deftest resident-binders-validate-ordered-values-before-touching-a-driver
  (let [abi [{:name 'A :c-name "A" :kind :input :dtype :float :kernel-dtype :float}
             {:name 'alpha :c-name "alpha" :kind :scalar :dtype :float :kernel-dtype :float}
             {:name 'C :c-name "C" :kind :output :dtype :float :kernel-dtype :float :role :result}]
        raw-seg (java.lang.foreign.MemorySegment/ofArray (float-array 1))]
    (doseq [[backend register! bind!]
            [[:ze ze/register-kernel! ze/bind-kernel-call]
             [:ocl ocl/register-kernel! ocl/bind-kernel-call]]]
      (let [kernel-name (str "resident_abi_probe_" (name backend) "_" (gensym))
            artifact (probe-artifact
                      kernel-name
                      (str "__kernel void " kernel-name
                           "(__global float* A, float alpha, __global float* C) {}")
                      abi '[A alpha C])
            _ (register! kernel-name artifact)
            pointer-call (kcall/make artifact
                                     [(Object.) {:type :float :value 1.0} (Object.)])
            pointer-data (thrown-data #(bind! pointer-call))
            scalar-data (thrown-data #(kcall/make
                                       artifact
                                       [raw-seg {:type :int :value 1} raw-seg]))]
        (is (= kernel-name (:kernel-name pointer-data)))
        (is (= :input (get-in pointer-data [:slot :kind])))
        (is (= Object (:value-type pointer-data)))
        (is (= 'alpha (get-in scalar-data [:slot :name])))
        (is (= :float (:expected scalar-data)))
        (is (= :int (:actual scalar-data)))))))

(deftest compile-gpu-program-extracts-a-real-contraction-as-an-ordered-resident-executable
  (let [descriptor (pipeline/compile-gpu-program #'resident-contract-descriptor-probe
                                                 :ze:0 :dtype :float)
        step (first (:steps descriptor))
        executable (:artifact step)
        args [(float-array 64) (float-array 64)]
        call-arguments (mapv (fn [{:keys [kind type value-fn]}]
                               (if (= :scalar kind)
                                 {:type type :value (value-fn args)}
                                 (Object.)))
                             (:argument-specs step))
        {:keys [buffers scalar-values]} (kexec/graph-bindings executable call-arguments)
        call (kgcall/make executable buffers scalar-values)
        kernel-call (get-in call [:nodes 0 :call])]
    (is (= :executable (:convention step)))
    (is (kgraph/kernel-graph? executable)
        "typed contraction candidates retain their common graph boundary")
    (is (= '[A B C] (mapv (comp :name :slot) (:argument-specs step))))
    (is (= [:input :input :output] (mapv :kind (:argument-specs step))))
    (is (= 'C (:result-sym descriptor)) "the invoke result aliases its ABI :result buffer")
    (is (= {'A :input 'B :input} (:array-roles descriptor)))
    (is (= :float (:dtype (first (:allocs descriptor))))
        "scratch allocation dtype comes from the contraction output ABI")
    (is (= 64 (kart/attribute (first (kexec/artifacts executable)) :out-elems)))
    (is (= 2 (count (get-in kernel-call [:geometry :workgroup-size]))))
    (is (= 2 (count (get-in kernel-call [:geometry :group-count]))))
    (is (every? pos? (get-in kernel-call [:geometry :workgroup-size])))
    (is (every? pos? (get-in kernel-call [:geometry :group-count])))))
