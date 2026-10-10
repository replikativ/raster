(ns raster.compiler.integral-reduction-device-test
  (:require [clojure.test :refer [deftest is]]
            [raster.compiler.integral-reduction-test :as fixture]
            [raster.compiler.core.dtype :as dtype]
            [raster.compiler.pipeline :as pipeline]
            [raster.gpu.runtime-backend :as backend]
            [raster.gpu.device-probe :as opencl]
            [raster.dl.gpu-grad-parity :as ze]))

(defn check-device! [device]
  (let [namespace (backend/runtime-namespace device)
        resolve-runtime #(requiring-resolve (symbol (str namespace) %))
        arena ((resolve-runtime "make-kernel-arena!"))]
    (try
      (with-bindings {(resolve-runtime "*current-arena*") arena}
        (doseq [storage [:int :long] operator [:sum :product :min :max]]
          (let [emitted (fixture/emit storage operator :double device)
                _ (#'pipeline/register-gpu-dispatches! (:dispatches emitted) device)
                compiled (eval (list 'fn ['a 'n] (:form emitted)))
                original (eval (list 'fn ['a 'n] (fixture/source storage operator)))
                {:keys [min max]} (:limits (dtype/info storage))
                neutral (case operator :sum 0 :product 1 :min max :max min)]
            (is (= 2 (count (:kernels emitted))))
            (doseq [n [0 1 4097]]
              (let [input ((resolve (dtype/jvm-array-constructor storage)) (repeat n neutral))]
                (when (pos? n) (clojure.lang.RT/aset input 0 max))
                (when (> n 1)
                  (clojure.lang.RT/aset input (dec n) (case operator :sum 1 :product 2 min)))
                (let [expected (fixture/modular-reference storage operator (vec input))]
                  (is (= expected (original input n) (compiled input n))
                      (str {:device device :storage storage :operator operator :n n}))))))))
      (finally ((resolve-runtime "close-kernel-arena!") arena)))))

(deftest integral-reductions-match-jvm-and-independent-modular-oracles
  (if @opencl/opencl-available?
    (check-device! :ocl:0)
    (opencl/opencl-skip! "integral workgroup reductions"))
  (if @ze/gpu-available?
    (check-device! :ze:0)
    (ze/gpu-skip! "integral workgroup reductions")))
