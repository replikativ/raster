(ns raster.compiler.contraction-marker-abi-test
  "The contraction emitter, route, marker, registry and runtime share one ordered typed ABI."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.arrays :as ra]
            [raster.compiler.pipeline :as pipeline]
            [raster.compiler.backend.gpu.opencl-pass :as op]
            [raster.compiler.passes.parallel.contract-route :as route]
            [raster.compiler.passes.parallel.par-fusion :as fusion]
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

(defn- fuse-epilogue [body]
  (second (:bindings
           (fusion/fuse-contract-map
            ['C (mm 128 256 64)
             'out (list 'raster.par/map! 'out 't (* 128 256) nil body)]
            []))))

(defn- compile-form
  ([contract] (compile-form contract :half))
  ([contract dtype]
   (op/opencl-pass (list 'let* ['result contract] 'result)
                   :device-id :ze:0 :dtype dtype :min-elements 0)))

(defn- thrown-data [thunk]
  (try
    (thunk)
    nil
    (catch clojure.lang.ExceptionInfo e
      (ex-data e))))

(deftest fused-epilogue-operands-are-carried-in-the-ordered-marker
  (let [contract (fuse-epilogue
                  (list 'raster.numeric/+ (list 'aget 'C 't)
                        (list 'aget 'bias (list 'mod 't 256))))
        compiled (compile-form contract)
        kernel (first (:kernels compiled))
        marker (-> compiled :form second second)]
    (is (= :dpas (:strategy kernel)))
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
      (is (= :dp4a (:strategy kernel)))
      (is (true? (:tensorized kernel)))
      (is (= :int8x4 (:packed kernel)))
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
             {:name 'C :c-name "C" :kind :output :dtype :float :kernel-dtype :float}]
        _ (ze/register-kernel! kernel-name {:abi abi})
        data (thrown-data #(ze/invoke-registered-contraction!
                            kernel-name [(float-array 1)] 1 [1 1] [1 1]))]
    (is (= 2 (:expected data)))
    (is (= 1 (:actual data)))))

(deftest resident-binders-validate-ordered-values-before-touching-a-driver
  (let [abi [{:name 'A :c-name "A" :kind :input :dtype :float :kernel-dtype :float}
             {:name 'alpha :c-name "alpha" :kind :scalar :dtype :float :kernel-dtype :float}
             {:name 'C :c-name "C" :kind :output :dtype :float :kernel-dtype :float :role :result}]
        raw-seg (java.lang.foreign.MemorySegment/ofArray (float-array 1))]
    (doseq [[backend register! bind!]
            [[:ze ze/register-kernel! ze/bind-registered-contraction!]
             [:ocl ocl/register-kernel! ocl/bind-registered-contraction!]]]
      (let [kernel-name (str "resident_abi_probe_" (name backend) "_" (gensym))
            _ (register! kernel-name {:abi abi})
            pointer-data (thrown-data #(bind! kernel-name
                                              [(Object.) {:type :float :value 1.0} (Object.)]
                                              1 [1 1] [1 1]))
            scalar-data (thrown-data #(bind! kernel-name
                                             [raw-seg {:type :int :value 1} raw-seg]
                                             1 [1 1] [1 1]))]
        (is (= kernel-name (:kernel-name pointer-data)))
        (is (= :input (get-in pointer-data [:slot :kind])))
        (is (= Object (:value-type pointer-data)))
        (is (= 'alpha (get-in scalar-data [:slot :name])))
        (is (= :float (:expected scalar-data)))
        (is (= :int (:actual scalar-data)))))))

(deftest compile-gpu-program-extracts-a-real-contraction-as-an-ordered-resident-step
  (let [descriptor (pipeline/compile-gpu-program #'resident-contract-descriptor-probe
                                                  :ze:0 :dtype :float)
        step (first (:steps descriptor))
        args [(float-array 64) (float-array 64)]]
    (is (= :contract (:convention step)))
    (is (= '[A B C] (mapv (comp :name :slot) (:argument-specs step))))
    (is (= [:input :input :output] (mapv :kind (:argument-specs step))))
    (is (= 'C (:result-sym descriptor)) "the invoke result aliases its ABI :result buffer")
    (is (= {'A :input 'B :input} (:array-roles descriptor)))
    (is (= :float (:dtype (first (:allocs descriptor))))
        "scratch allocation dtype comes from the contraction output ABI")
    (is (= 64 (long ((:out-elems-fn step) args))))
    (is (every? pos? (map #(% args) (:wg-fns step))))
    (is (every? pos? (map #(% args) (:grid-fns step))))))
