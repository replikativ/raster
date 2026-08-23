(ns raster.compiler.contraction-marker-abi-test
  "The contraction route can describe capabilities the current invocation marker cannot bind.

   Until contraction descriptors and runtime launches share one ordered typed ABI, the compiler
   must reject those descriptors rather than emit a valid kernel with an under-bound invocation."
  (:require [clojure.test :refer [deftest is testing]]
            [raster.compiler.backend.gpu.opencl-pass :as op]
            [raster.compiler.passes.parallel.contract-route :as route]
            [raster.compiler.passes.parallel.par-fusion :as fusion]))

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

(deftest fused-epilogue-operands-are-refused-before-an-under-bound-marker-is-emitted
  (let [contract (fuse-epilogue
                  (list 'raster.numeric/+ (list 'aget 'C 't)
                        (list 'aget 'bias (list 'mod 't 256))))
        data (thrown-data #(compile-form contract))]
    (testing "this is a real routed DPAS descriptor, not a synthetic malformed map"
      (is (= :dpas (:strategy data))))
    (testing "the marker would bind [A B] while the kernel signature also requires bias"
      (is (= :raster/fatal (:reason data)))
      (is (= :ordered-contraction-kernel-abi (:missing-rule data)))
      (is (= [:epilogue-operands] (:unsupported-fields data)))
      (is (= {:epilogue-operands '[bias]} (:unsupported-values data))))))

(deftest codegen-only-routing-facts-remain-legal-and-observable
  (let [dp4a '(raster.par/contract C [[i 4] [j 4]] [[l 8]]
                                   (* (aget A (+ (* i 8) l))
                                      (aget B (+ (* j 8) l))))
        compiled (compile-form dp4a :byte)
        kernel (first (:kernels compiled))]
    (testing "tensorization and packing are already baked into source and need no extra launch slot"
      (is (= :dp4a (:strategy kernel)))
      (is (true? (:tensorized kernel)))
      (is (= :int8x4 (:packed kernel))))
    (testing "the ordinary two-input descriptor still emits the production marker"
      (is (= 'raster.gpu.ze-runtime/invoke-registered-contraction!
             (-> compiled :form second second first))))))

(deftest every-unexpressed-marker-field-is-fail-loud
  (let [base {:strategy :probe
              :kernel-name "probe_contract"
              :source "__kernel void probe_contract(__global double* A, __global double* C) {}"
              :array-params '[A]
              :dtype :double :out-dtype :double :out-elems 1
              :wg [1 1] :grid [1 1] :scalar-args []}
        contract (mm 1 1 1)]
    (doseq [[field value] [[:pre-steps [{:op :transpose}]]
                           [:lift-operands '[scale]]
                           [:epilogue-operands '[bias]]
                           [:epilogue-scalars '[alpha]]]]
      (testing (name field)
        (let [data (with-redefs [route/route-contraction
                                 (fn [& _] (assoc base field value))]
                     (thrown-data #(compile-form contract)))]
          (is (= :raster/fatal (:reason data)))
          (is (= [field] (:unsupported-fields data)))
          (is (= {field value} (:unsupported-values data))))))))
